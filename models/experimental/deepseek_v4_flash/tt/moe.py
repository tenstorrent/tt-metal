from typing import NamedTuple, Optional

import ttnn
import torch

from .common import DeepSeekV4Module, _profile, _region
from .decode_prefetch import check_decode_layout, decode_prefetch_page_bytes, make_decode_prefetch_buffers
from .layers import Linear, LinearDecode
from .l1_weights import packed_weight_spec
from .system_config import active_system_config
from .weight_cache import WeightCache, _as_cache, _load_weight, _materialize, _memo

# ---------------------------------------------------------------------------- #
# DeepSeek-V4-Flash Mixture-of-Experts (prefill)
#
# ttnn port of ``DeepseekV4SparseMoeBlock`` (and its ``DeepseekV4TopKRouter`` /
# ``DeepseekV4Experts`` / ``DeepseekV4MLP`` shared expert) from
# ``modular_deepseek_v4.py``. Scope is the standard top-k routed MoE block (the
# ``mlp_layer_types == "moe"`` path); the static ``hash_moe`` router is out of
# scope here (it only swaps the *which-experts* selection for a frozen
# ``tid2eid[input_ids]`` lookup, leaving the expert / shared-expert compute
# identical).
#
# Layout conventions, matching the reference:
#   B = batch, S = seq length, T = B*S flattened tokens, H = hidden_size,
#   E = num routed experts, I = moe_intermediate_size, k = num_experts_per_tok.
#
# The reference dispatches each token to its top-k experts and loops over the
# *hit* experts. We instead run a *dense* batched compute: every expert is
# evaluated for every token, then masked by the per-token routing weight (0 for
# unselected experts) and summed across the expert axis. This is the standard
# small-mesh ttnn MoE shape (cf. ``models/demos/gpt_oss``); it is mathematically
# identical to the gather/scatter reference because unselected experts get a
# routing weight of exactly 0.
# ---------------------------------------------------------------------------- #


class SparseRouting(NamedTuple):
    """A routing decision in the form ``fused_experts`` consumes it.

    The op takes the selected expert ids and the score row they index and does the
    normalize/scale itself, so the router hands over ``ttnn.topk``'s output untouched.
    Widening this into a dense ``[1,1,T,E]`` weight row (scatter a one-hot mask, mask the
    scores, sum, divide, scale, relayout -- nine device ops) only to have the op's first
    kernel scan those E columns straight back down to k values is pure round-tripping.

    ``scores``: ``[1,1,T,E]`` TILE bf16, the *unbiased* per-expert scores -- the values
    that become the weights. ``indices``: ``[1,1,T,k]`` TILE, which experts won: uint16
    from the learned router's topk (possibly ranked on a bias-corrected copy of
    ``scores``), bf16 from the hash router's table lookup, since ``ttnn.embedding``
    gathers only bf16. The op reads either.
    """

    scores: ttnn.Tensor
    indices: ttnn.Tensor


# Guards the per-token renormalize against an all-zero score row. Configured by
# ``moe.routing_eps`` in the system profile.


class DeepSeekV4MLP(DeepSeekV4Module):
    """Dense SwiGLU MLP (matches ``DeepseekV4MLP`` / ``LlamaMLP``).

    Used as the always-on *shared expert*: ``down(silu(gate(x)) * up(x))`` with
    no clamp (the routed experts clamp; the shared expert does not).

    ``use_prefetcher=True`` runs the three projections as :class:`LinearDecode` on
    DRISC-prefetched weights, streamed through the same GCB the attention block uses (see
    ``decode_prefetch``) instead of reading them from DRAM on every call. That path is decode
    shaped: it width-shards the tokens over one tile-row, which caps it at the 32 rows a tile
    holds, so a prefill-width input has to use the default ``ttnn.linear`` path. It also needs
    ``config``, to check the fixed weight layouts against the shapes this model wants.
    """

    def __init__(
        self,
        weights: dict,
        prefix: str,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        config=None,
        use_prefetcher: bool = False,
        prefetch_buffers: Optional[dict] = None,
        packed_weights=None,
        tp_size: int = 1,
    ):
        cache = _as_cache(cache)
        self.device = device
        self.use_prefetcher = use_prefetcher
        self.tp_size = tp_size
        # After transpose, gate/up are ``[H, I]`` (column-parallel: shard N) and down
        # is ``[I, H]`` (row-parallel: shard K).
        gate_up_mapper = ttnn.ShardTensorToMesh(device, dim=1) if tp_size > 1 else None
        down_mapper = ttnn.ShardTensorToMesh(device, dim=0) if tp_size > 1 else None
        tp_tag = f".tp{tp_size}" if tp_size > 1 else ""
        if packed_weights is not None:
            tensor, layout, slot = packed_weights

            def packed_projection(name, weight_key, K, N):
                spec = packed_weight_spec(layout, slot, name)
                return LinearDecode(
                    weights[weight_key],
                    device,
                    cache.file(weight_key.removesuffix(".weight")),
                    dtype=ttnn.bfloat4_b,
                    K=K,
                    N=N,
                    partial_width_sharded=spec.k_blocks > 1,
                    k_blocks=spec.k_blocks,
                    n_blocks=spec.n_blocks,
                    packed_weight_tensor=tensor,
                    packed_weight_spec=spec,
                )

            hidden, inter = config.hidden_size, config.moe_intermediate_size
            self.gate_proj = packed_projection("shared_gate_proj", f"{prefix}.gate_proj.weight", hidden, inter)
            self.up_proj = packed_projection("shared_up_proj", f"{prefix}.up_proj.weight", hidden, inter)
            self.down_proj = packed_projection("shared_down_proj", f"{prefix}.down_proj.weight", inter, hidden)
            return
        if not use_prefetcher:
            self.gate_proj = Linear(
                weights[f"{prefix}.gate_proj.weight"],
                device,
                cache.file(f"{prefix}.gate_proj{tp_tag}"),
                dtype=weight_dtype,
                mesh_mapper=gate_up_mapper,
            )
            self.up_proj = Linear(
                weights[f"{prefix}.up_proj.weight"],
                device,
                cache.file(f"{prefix}.up_proj{tp_tag}"),
                dtype=weight_dtype,
                mesh_mapper=gate_up_mapper,
            )
            self.down_proj = Linear(
                weights[f"{prefix}.down_proj.weight"],
                device,
                cache.file(f"{prefix}.down_proj{tp_tag}"),
                dtype=weight_dtype,
                mesh_mapper=down_mapper,
            )
            return

        hidden, inter = config.hidden_size, config.moe_intermediate_size
        if prefetch_buffers is None:
            prefetch_buffers = make_decode_prefetch_buffers(device, weight_dtype)
        prefetch = {"use_prefetcher": True, "global_cb_page_bytes": decode_prefetch_page_bytes(weight_dtype)}
        self.gate_proj = LinearDecode(
            weights[f"{prefix}.gate_proj.weight"],
            device,
            cache.file(f"{prefix}.gate_proj{tp_tag}"),
            dtype=weight_dtype,
            **check_decode_layout("shared_gate_proj", hidden, inter),
            global_cb=prefetch_buffers["shared_gate_proj"],
            **prefetch,
        )
        self.up_proj = LinearDecode(
            weights[f"{prefix}.up_proj.weight"],
            device,
            cache.file(f"{prefix}.up_proj{tp_tag}"),
            dtype=weight_dtype,
            **check_decode_layout("shared_up_proj", hidden, inter),
            global_cb=prefetch_buffers["shared_up_proj"],
            **prefetch,
        )
        self.down_proj = LinearDecode(
            weights[f"{prefix}.down_proj.weight"],
            device,
            cache.file(f"{prefix}.down_proj{tp_tag}"),
            dtype=weight_dtype,
            **check_decode_layout("shared_down_proj", inter, hidden),
            global_cb=prefetch_buffers["shared_down_proj"],
            **prefetch,
        )

    def prefetch_weights(self):
        """Stage the three projection weights ahead of the :meth:`forward` that uses them.

        Queued gate, up, down: the order :meth:`forward` runs them, which is the order they
        must come off the shared GCB's single FIFO. The attention block's weights precede
        them, since it runs first in the decoder layer.
        """
        if not self.use_prefetcher:
            return
        self.gate_proj.fetch_weights()
        self.up_proj.fetch_weights()
        self.down_proj.fetch_weights()

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """``x`` ``[1, 1, T, H]`` (tokens packed onto the row axis) -> ``[1, 1, T, H]``.

        The packed form is what ``matmul_decode`` needs on the prefetched path -- it reads the
        row axis as the tokens and a leading rank-4 batch as separate matmuls, so per-user rows
        would decode as T one-token matmuls -- and it is what the caller already built for the
        router, so nothing is reshaped here.
        """
        # Prefetched, gate and up leave their results width-sharded over the 32 cores holding
        # 64 columns each, which is exactly how down_proj wants its activation sharded along
        # K, so the product feeds it where it already sits and nothing reshards in between.
        out = self.down_proj(ttnn.multiply(ttnn.silu(self.gate_proj(x)), self.up_proj(x)))
        return out


class DeepSeekV4TopKRouter(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4TopKRouter``.

    ``sqrtsoftplus`` of the gate logits gives per-expert scores, and the top-k experts are
    selected by ``ttnn.topk`` on ``scores + e_score_correction_bias``. That pair -- the
    unbiased scores and the winning ids -- is the whole routing decision, and it is what
    :class:`DeepSeekV4PreloadedExperts` is handed (see :class:`SparseRouting`).

    The renormalize-and-scale tail the reference applies to the selected scores happens
    inside ``fused_experts``, on the k values per token it already has to read, rather
    than here across a dense E-wide row.
    """

    def __init__(
        self, config, weights: dict, device: ttnn.MeshDevice, cache: Optional[WeightCache] = None, packed_weights=None
    ):
        self.device = device
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        cache = _as_cache(cache)
        if packed_weights is None:
            self.gate = Linear(weights["gate.weight"], device, cache.file("gate"))
        else:
            tensor, layout, slot = packed_weights
            spec = packed_weight_spec(layout, slot, "router_gate")
            self.gate = LinearDecode(
                weights["gate.weight"],
                device,
                cache.file("gate"),
                dtype=ttnn.bfloat4_b,
                K=spec.K,
                N=spec.N,
                n_blocks=spec.n_blocks,
                packed_weight_tensor=tensor,
                packed_weight_spec=spec,
            )
        bias = _materialize(
            weights["gate.e_score_correction_bias"], cache.file("gate.e_score_correction_bias"), ttnn.bfloat16
        )
        self.e_score_correction_bias = _load_weight(
            bias.reshape(1, 1, 1, self.num_experts) if bias is not None else None,
            device,
            cache_file_name=cache.file("gate.e_score_correction_bias"),
        )

    def _scores(self, x_flat: ttnn.Tensor) -> ttnn.Tensor:
        """Per-expert ``sqrtsoftplus`` gate scores ``[1,1,T,E]``."""
        return ttnn.sqrt(ttnn.softplus(self.gate(x_flat)))

    def forward(self, x_flat: ttnn.Tensor) -> SparseRouting:
        """``x_flat`` is ``[1, 1, T, H]``; returns the selected experts and their scores.

        Trace-safe as it stands, so prefill and the captured decode share this one path:
        every op here allocates its own output and nothing is host-initialised.
        """
        scores = self._scores(x_flat)  # [1, 1, T, E]
        # Ranked on the bias-corrected scores, weighted by the uncorrected ones -- which is
        # why both halves of the pair travel to the expert op instead of just the winners'
        # values. topk's ids are returned exactly as produced: TILE uint16, [1,1,T,k].
        biased = ttnn.add(scores, self.e_score_correction_bias)
        _profile(self.device)
        biased = ttnn.to_memory_config(biased, ttnn.DRAM_MEMORY_CONFIG)
        _, top_idx = ttnn.topk(biased, self.top_k, dim=-1)
        scores = ttnn.to_memory_config(scores, ttnn.DRAM_MEMORY_CONFIG)
        return SparseRouting(scores=scores, indices=top_idx)


class DeepSeekV4HashRouter(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4HashRouter`` (the first ``num_hash_layers`` MoE
    layers, paper §2.1).

    Expert *selection* is a frozen ``tid2eid[input_ids]`` lookup — a fixed
    token-id -> expert-id table — rather than a learned top-k argmax. The learned
    gate still produces the per-expert ``sqrtsoftplus`` scores that weight the
    selected experts; only the *which-experts* decision is static. The output is the
    same :class:`SparseRouting` pair the learned router emits, so both feed the expert
    compute through one contract.

    The selection is gathered *fully on device* by embedding the token id in the frozen
    ``tid2eid`` table, which is already the ``[vocab, k]`` list of expert ids the sparse
    contract wants — no host-side scatter and no per-step host->device copy. It is held as
    bfloat16 because that is the only dtype :func:`ttnn.embedding` gathers from; every
    expert id is exact there (E <= 256), and ``fused_experts`` reads bf16-encoded ids
    directly. That leaves the table at ``[vocab, k]`` instead of the ``[vocab, E]`` one-hot
    it used to expand to — for a 128k vocab, 1.5 MB instead of 64 MB.
    """

    def __init__(
        self, config, weights: dict, device: ttnn.MeshDevice, cache: Optional[WeightCache] = None, packed_weights=None
    ):
        self.device = device
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        cache = _as_cache(cache)
        if packed_weights is None:
            self.gate = Linear(weights["gate.weight"], device, cache.file("gate"))
        else:
            tensor, layout, slot = packed_weights
            spec = packed_weight_spec(layout, slot, "router_gate")
            self.gate = LinearDecode(
                weights["gate.weight"],
                device,
                cache.file("gate"),
                dtype=ttnn.bfloat4_b,
                K=spec.K,
                N=spec.N,
                n_blocks=spec.n_blocks,
                packed_weight_tensor=tensor,
                packed_weight_spec=spec,
            )
        # tid2eid [vocab, top_k]: frozen token-id -> expert-id table (host-side,
        # no tile cache) -- always materialise.
        tid = weights["gate.tid2eid"]
        tid = tid() if callable(tid) else tid
        self.tid2eid = tid.long()
        assert self.num_experts <= 256, (
            f"hash routing embeds expert ids as bf16, which is only exact below 256; "
            f"num_local_experts is {self.num_experts}"
        )
        self.eid_table = ttnn.from_torch(
            self.tid2eid.float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )

    def _scores(self, x_flat: ttnn.Tensor) -> ttnn.Tensor:
        """Per-expert ``sqrtsoftplus`` gate scores (shared head of both routing paths)."""
        return ttnn.sqrt(ttnn.softplus(self.gate(x_flat)))

    def _select(self, token_in: ttnn.Tensor, t: int) -> ttnn.Tensor:
        """Gather the ``[1,1,T,k]`` selected expert ids for on-device token ids ``[1,T]``."""
        ids = ttnn.embedding(token_in, self.eid_table, layout=ttnn.TILE_LAYOUT)  # [1, T, k] bf16
        return ttnn.reshape(ids, [1, 1, t, self.top_k])

    def forward(self, x_flat: ttnn.Tensor, input_ids: torch.Tensor) -> SparseRouting:
        """``x_flat`` ``[1,1,T,H]`` and ``input_ids`` torch ``[..]`` (T tokens).

        The prefill entry point: it uploads the host ``input_ids`` and then routes exactly
        as :meth:`forward_static` does from the persistent on-device ids.
        """
        scores = self._scores(x_flat)  # [1, 1, T, E]
        t = x_flat.shape[2]
        _profile(self.device)
        ids_tt = ttnn.from_torch(
            input_ids.reshape(1, t).long().to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=(ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None),
        )
        return SparseRouting(scores=scores, indices=self._select(ids_tt, t))

    def forward_static(self, x_flat: ttnn.Tensor, token_in: ttnn.Tensor) -> SparseRouting:
        """Trace-safe, fully on-device hash routing: ``token_in`` ``[1,T]`` are the
        (persistent, on-device) decode token ids, one per user of a batched step."""
        ids = self._select(token_in, x_flat.shape[2])
        return SparseRouting(scores=self._scores(x_flat), indices=ids)


# --------------------------------------------------------------------------- #
# fused_experts (single-op decode path)
#
# ``ttnn.experimental.deepseek.moe.fused_experts`` runs the whole routed-expert
# FFN (gate_up + SwiGLU + down + routing-weighted accumulation) for one token in
# a single device op. Weights are DRAM ND-sharded 64 ways (one shard per original
# 8x8 core); H must be exactly ``_FUSED_HIDDEN`` (64 * 2 * 32). With 6 selected
# experts the op runs on a 12x8 (96-core) grid, 16 cores per expert. It is
# decode-native (``T == 1``). The weights must be DRAM ND-sharded with one shard
# per original compute core (see below), a layout distinct from the plain matmul
# weights used by the prefill loop, so the decode path keeps its own copy.
# --------------------------------------------------------------------------- #
# The tile is a hardware invariant, unlike the core count and bank count, which are
# ``moe.fused_num_cores`` / ``moe.fused_dram_banks`` in the system profile.
_FUSED_TILE = 32


def _fused_hidden(num_cores: int) -> int:
    """The only hidden size the op accepts: each of ``num_cores`` cores owns exactly
    2 output tiles of the H row (4096 on a 64-core grid)."""
    return num_cores * 2 * _FUSED_TILE


def _swiglu_cols_per_core(intermediate: int, num_cores: Optional[int] = None) -> int:
    """SwiGLU output columns in each DRAM shard, i.e. the I dim spread over shards.

    One 32-column I-tile per shard at I == 2048, and still one tile when TP slices I
    below 2048, so gate_up stays DRAM-busy on every NoC port. Mirrors
    ``swiglu_tiles_per_shard_for`` in the program factory. On the 6-expert path those
    shards are spread across the 16 cores of each expert group (4 shards/core at
    I=2048, 1 at I=512).
    """
    if num_cores is None:
        num_cores = active_system_config().moe.fused_num_cores
    i_tiles = intermediate // _FUSED_TILE
    return _FUSED_TILE * max(1, i_tiles // num_cores)


def _tp_cluster_axis(device: ttnn.MeshDevice) -> int:
    """Mesh axis that holds the TP group. A 1xN submesh shards along axis 1."""
    shape = tuple(device.shape)
    if len(shape) == 2 and shape[1] > 1:
        return 1
    return 0


def _pack_gate_up_for_tp(gate_up: torch.Tensor, tp_size: int) -> torch.Tensor:
    """Reorder ``[2I, H]`` so rank ``r`` owns ``cat(gate[r], up[r])`` along dim 0.

    Naive sharding of ``[gate | up]`` would split mid-gate on later ranks. After this
    pack, ``ShardTensorToMesh(dim=0)`` gives each chip ``[2 * I/tp, H]``.
    """
    two_i, hidden = gate_up.shape
    intermediate = two_i // 2
    if intermediate % tp_size:
        raise ValueError(f"moe_intermediate_size {intermediate} is not divisible by tp_size {tp_size}")
    i_local = intermediate // tp_size
    parts = []
    for rank in range(tp_size):
        lo, hi = rank * i_local, (rank + 1) * i_local
        parts.append(torch.cat([gate_up[lo:hi], gate_up[intermediate + lo : intermediate + hi]], dim=0))
    return torch.cat(parts, dim=0)


def _interleave_gate_up_tp(gate_up: torch.Tensor, tp_size: int, swiglu_cols: int) -> torch.Tensor:
    """Per-rank ``fused_experts`` interleave of a packed ``[2I, H]`` gate_up.

    Each rank's ``[H, 2*I_local]`` is interleaved independently (the op only sees
    ``I_local``), then concatenated on the column axis so ``ShardTensorToMesh(dim=1)``
    yields one rank's interleaved weight.
    """
    packed = _pack_gate_up_for_tp(gate_up, tp_size)
    two_i, _ = packed.shape
    i_local_two = two_i // tp_size
    parts = []
    for rank in range(tp_size):
        local = packed[rank * i_local_two : (rank + 1) * i_local_two]
        parts.append(_interleave_gate_up(local.t().contiguous(), swiglu_cols))
    return torch.cat(parts, dim=1)


def _tp_all_reduce(tensor: ttnn.Tensor, device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Sum TP partials (row-parallel down-proj) back to a replicated ``[... H]``."""
    return ttnn.all_reduce(
        tensor,
        cluster_axis=_tp_cluster_axis(device),
        num_links=1,
        topology=ttnn.Topology.Linear,
    )


def _interleave_gate_up(w: torch.Tensor, block: int) -> torch.Tensor:
    """Permute a ``[K, 2I]`` gate_up weight into per-core ``[gate_block | up_block]``
    order so each ``[K, 2*block]`` DRAM shard holds a core's gate columns followed
    by its paired up columns (what ``fused_experts`` reads in a single NoC read).

    ``gate = w[:, :I]``, ``up = w[:, I:]``; output column ``c*2*block + h*block + t``
    maps to ``w[:, h*I + c*block + t]``.
    """
    k, two_i = w.shape
    intermediate = two_i // 2
    blocks = intermediate // block
    return w.reshape(k, 2, blocks, block).permute(0, 2, 1, 3).reshape(k, two_i).contiguous()


def _fused_nd_dram_config(rows: int, cols: int, shard_width: int, dram_banks: int) -> ttnn.MemoryConfig:
    """DRAM ND-shard config: ``rows x shard_width`` shards round-robined over the
    DRAM banks (one shard per compute core), as ``fused_experts`` expects."""
    assert cols % shard_width == 0, f"last dim {cols} must divide into shards of {shard_width}"
    dram_core_range_set = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(bank, 0), ttnn.CoreCoord(bank, 0)) for bank in range(dram_banks)]
    )
    return ttnn.MemoryConfig(
        ttnn.BufferType.DRAM,
        ttnn.NdShardSpec(
            shard_shape=[rows, shard_width],
            grid=dram_core_range_set,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )


def _load_fused_weight(
    tensor: Optional[torch.Tensor],
    device: ttnn.MeshDevice,
    nd_config: ttnn.MemoryConfig,
    *,
    cache_file_name: Optional[str] = None,
    dtype: ttnn.DataType = ttnn.bfloat4_b,
    mesh_mapper=None,
) -> ttnn.Tensor:
    """Load a ``fused_experts`` weight as a DRAM ND-sharded tensor.

    The tile cache cannot round-trip an ND-shard memory config (a cache *hit*
    reloads the tensor with its plain serialized spec), so the (interleaved)
    weight is cached in standard interleaved DRAM under its own cache entry and
    then resharded to the ND-shard layout on device.
    """
    sharded = ttnn.as_tensor(
        tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=nd_config,
        cache_file_name=cache_file_name,
        mesh_mapper=mesh_mapper,
    )
    return sharded


class DeepSeekV4PreloadedExperts(DeepSeekV4Module):
    """Routed-experts compute via the single-op ``fused_experts`` kernel.

    The whole routed-expert FFN for one token (gate_up + SwiGLU + down +
    routing-weighted accumulation) runs in a single ``fused_experts`` device op.
    The op is hard-wired to the real V4-Flash sizes -- down weights 64-way
    sharded along H (each shard 2 output tiles), so ``H`` must be exactly
    ``_FUSED_HIDDEN`` (``64 * 2 * 32 == 4096``). Gate_up is one I-tile per
    DRAM shard, so a TP-sliced ``I`` (e.g. 512) yields fewer, still-full
    16-core groups rather than idle cores. With 6 selected experts compute
    uses a 12x8 grid (16 cores per expert).
    Both prefill and decode go through the op: it is natively single-token
    (``T == 1``), so **prefill is computed by decode** -- each of the ``T`` tokens
    runs as its own op and the per-token outputs are concatenated.

    Every expert is kept resident on device as DRAM ND-sharded weights (one shard
    per compute core), in low precision (``BFloat4_b`` by default; ~3.5 GB for the
    256 experts, a natural match for the MXFP4 checkpoint). At init it pulls each
    expert's dequantized weights from the host ``provider`` once, permutes the
    gate_up into the op's interleaved per-core layout, and uploads the ND-sharded
    tensors; ``forward`` then runs purely on device with no per-step host
    transfers beyond reading the (tiny) routing weights to pick the hit experts.

    ``provider(expert_idx) -> (gate_up [2I, H], down [H, I])`` returns host
    torch tensors (the HF packed layout: ``gate_up`` is ``cat([w_gate, w_up])``).
    Experts with zero total routing weight are skipped (matching the reference's
    ``hit`` set), so only the experts some token actually selected are computed.
    """

    def __init__(
        self,
        config,
        provider,
        device: ttnn.MeshDevice,
        dtype: Optional[ttnn.DataType] = None,
        cache: Optional[WeightCache] = None,
        system_config=None,
        tp_size: int = 1,
    ):
        # The system profile supplies the fused-op geometry, the L1 expert block size
        # and the default weight precision; an explicit ``dtype`` still wins.
        sys_cfg = system_config or active_system_config()
        self.system_config = sys_cfg
        self.experts_block_size = sys_cfg.moe.experts_block_size
        self.routing_eps = sys_cfg.moe.routing_eps
        num_cores = sys_cfg.moe.fused_num_cores
        dram_banks = sys_cfg.moe.fused_dram_banks
        dtype = dtype if dtype is not None else sys_cfg.decode.ttnn_weight_dtype

        self.device = device
        self.tp_size = tp_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        intermediate_full = config.moe_intermediate_size
        if intermediate_full % tp_size:
            raise ValueError(f"moe_intermediate_size {intermediate_full} is not divisible by tp_size {tp_size}")
        # Each TP rank runs fused_experts on its I-slice; the full width is recovered
        # by all-reducing the down-proj partials in :class:`DeepSeekV4SparseMoeBlock`.
        self.intermediate = intermediate_full // tp_size
        self.hidden = config.hidden_size
        self.limit = config.swiglu_limit
        # Applied inside the op on the sparse path, where the weights are derived here
        # rather than handed over pre-normalized.
        self.routed_scaling_factor = config.routed_scaling_factor
        cache = _as_cache(cache)

        # ``fused_experts`` is hard-wired to the real V4-Flash sizes: ``H == 4096``
        # on the 64-core grid and ``I`` a multiple of the per-core SwiGLU column slice.
        # There is no fallback path -- this class is for that config only.
        fused_hidden = _fused_hidden(num_cores)
        swiglu_cols = _swiglu_cols_per_core(self.intermediate, num_cores)
        if self.hidden != fused_hidden or self.intermediate % swiglu_cols != 0:
            raise ValueError(
                f"DeepSeekV4PreloadedExperts requires the fused_experts layout "
                f"(H == {fused_hidden} for moe.fused_num_cores={num_cores}, I % {swiglu_cols} == 0); "
                f"got H={self.hidden}, I={self.intermediate}"
            )
        gate_up_nd = _fused_nd_dram_config(self.hidden, 2 * self.intermediate, 2 * swiglu_cols, dram_banks)
        down_nd = _fused_nd_dram_config(self.intermediate, self.hidden, self.hidden // num_cores, dram_banks)

        # Upload every expert once as the op's DRAM ND-sharded weights (gate_up
        # interleaved per core, down ND-sharded), stored in low precision. With
        # caching enabled and a hit, the provider (and its expensive dequant) is
        # skipped entirely; the ND-shard layout can't round-trip the tile cache,
        # so the interleaved weight is cached in standard DRAM and resharded on
        # device (see :func:`_load_fused_weight`).
        gate_up_mapper = ttnn.ShardTensorToMesh(device, dim=1) if self.tp_size > 1 else None
        down_mapper = ttnn.ShardTensorToMesh(device, dim=0) if self.tp_size > 1 else None
        tp_tag = f".tp{self.tp_size}" if self.tp_size > 1 else ""

        self._gate_up_fused: list[ttnn.Tensor] = []
        self._down_fused: list[ttnn.Tensor] = []
        for e in range(self.num_experts):
            gu_f_name, dn_f_name = f"experts.{e}.gate_up_fused{tp_tag}", f"experts.{e}.down_fused{tp_tag}"
            need_torch = not (cache.hit(gu_f_name, dtype) and cache.hit(dn_f_name, dtype))
            if cache.require_cache and need_torch:
                raise RuntimeError(f"weight cache miss for routed expert {e} (gate_up/down) with require_cache=True")
            gate_up_w, down_w = provider(e) if need_torch else (None, None)
            # Provider gives gate_up [2I, H] / down [H, I]; transpose to matmul-ready
            # [H, 2I] / [I, H] (memoized so each is materialized at most once).
            if self.tp_size > 1:
                gu_il = _materialize(
                    (
                        (lambda gw=gate_up_w: _interleave_gate_up_tp(gw, self.tp_size, swiglu_cols))
                        if gate_up_w is not None
                        else (lambda: None)
                    ),
                    cache.file(gu_f_name),
                    dtype,
                )
                down_t = _memo((lambda dw=down_w: dw.t().contiguous()) if down_w is not None else (lambda: None))
            else:
                gate_up_t = _memo(
                    (lambda gw=gate_up_w: gw.t().contiguous()) if gate_up_w is not None else (lambda: None)
                )
                down_t = _memo((lambda dw=down_w: dw.t().contiguous()) if down_w is not None else (lambda: None))
                gu_il = _materialize(
                    lambda: _interleave_gate_up(gate_up_t(), swiglu_cols), cache.file(gu_f_name), dtype
                )
            self._gate_up_fused.append(
                _load_fused_weight(
                    gu_il,
                    device,
                    gate_up_nd,
                    cache_file_name=cache.file(gu_f_name),
                    dtype=dtype,
                    mesh_mapper=gate_up_mapper,
                )
            )
            self._down_fused.append(
                _load_fused_weight(
                    down_t(),
                    device,
                    down_nd,
                    cache_file_name=cache.file(dn_f_name),
                    dtype=dtype,
                    mesh_mapper=down_mapper,
                )
            )

    def _run_fused(self, x_tok: ttnn.Tensor, routing: SparseRouting) -> ttnn.Tensor:
        """Run ``fused_experts`` for one token. ``x_tok`` is ``[1,1,1,H]`` (TILE) and
        ``routing`` that token's slice of the router's output; returns ``[1,1,1,H]``.

        ``num_experts`` is always ``top_k``: one token selects at most that many distinct
        experts, so the op's program -- and any trace holding it -- is the same every step.
        Both tensors go in exactly as the router produced them: the op reads the ids and
        the score row out of their tiles and applies the normalize-and-scale tail itself.

        ``experts_block_size`` (``moe.experts_block_size``) is how many experts' SwiGLU
        activations are resident at once. It sizes the op's dominant per-core CB, so it is
        the knob to turn when the op's static CBs collide with the L1 buffers live at the
        call; the cost is one extra chip-wide gather/broadcast barrier per block.
        """
        indices = ttnn.to_memory_config(routing.indices, ttnn.DRAM_MEMORY_CONFIG)
        scores = ttnn.to_memory_config(routing.scores, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.experimental.deepseek.moe.fused_experts(
            x_tok,
            routing_indices=indices,
            routing_scores=scores,
            gate_up_weights=self._gate_up_fused,
            down_weights=self._down_fused,
            num_experts=self.top_k,
            intermediate_size=self.intermediate,
            swiglu_limit=self.limit,
            top_k=self.top_k,
            routed_scaling_factor=self.routed_scaling_factor,
            routing_eps=self.routing_eps,
            experts_block_size=self.experts_block_size,
        )  # [1, 1, H]
        return ttnn.reshape(out, [1, 1, 1, self.hidden])

    def _token_routing(self, routing: SparseRouting, i: int) -> SparseRouting:
        """Token ``i``'s slice of a ``T``-token routing decision."""
        return SparseRouting(
            scores=ttnn.slice(routing.scores, [0, 0, i, 0], [1, 1, i + 1, self.num_experts]),
            indices=ttnn.slice(routing.indices, [0, 0, i, 0], [1, 1, i + 1, self.top_k]),
        )

    def forward(self, x_flat: ttnn.Tensor, routing: SparseRouting) -> ttnn.Tensor:
        """``x_flat`` ``[1,1,T,H]`` plus the router's decision; returns ``[1,1,T,H]``.

        Trace-safe: nothing is read back to host. ``T`` tokens run as ``T`` single-token
        ops rather than one wider one -- they route to different experts, so there is no
        weight to share between them -- and ``T`` is fixed at capture time, so a captured
        trace stays a flat op sequence. Prefill therefore goes through the same path.
        """
        t = x_flat.shape[2]
        _profile(self.device)

        if t == 1:
            out = self._run_fused(x_flat, routing)
            _profile(self.device)
            return out

        # One row is not a whole tile, so it cannot be cut out of a width-sharded tensor,
        # and a batched step arrives sharded from the post-attention norm.
        if x_flat.is_sharded():
            x_flat = ttnn.to_memory_config(x_flat, ttnn.DRAM_MEMORY_CONFIG)
        h = x_flat.shape[3]
        per_token = []
        for i in range(t):
            per_token.append(
                self._run_fused(ttnn.slice(x_flat, [0, 0, i, 0], [1, 1, i + 1, h]), self._token_routing(routing, i))
            )
            _profile(self.device)
        return ttnn.concat(per_token, dim=2)

    # The routed FFN has no host readback and no host-initialised operands, so the traced
    # decode path is just :meth:`forward`.
    decode_static = forward


class DeepSeekV4SparseMoeBlock(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4SparseMoeBlock`` (standard ``moe`` layer).

    ``routed = experts(router(x)) ; return routed + shared_experts(x)``.
    """

    def __init__(
        self,
        config,
        weights: dict,
        device: ttnn.MeshDevice,
        experts,
        gate=None,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        use_prefetcher: bool = False,
        prefetch_buffers: Optional[dict] = None,
        packed_weights=None,
        tp_size: int = 1,
    ):
        self.device = device
        self.hidden = config.hidden_size
        self.packed_weights = packed_weights
        self.tp_size = tp_size
        cache = _as_cache(cache)
        # ``gate`` may be injected (e.g. a :class:`DeepSeekV4HashRouter` for the
        # first ``num_hash_layers`` layers); otherwise the learned top-k router.
        self.gate = (
            gate
            if gate is not None
            else DeepSeekV4TopKRouter(config, weights, device, cache=cache, packed_weights=packed_weights)
        )
        self.is_hash = isinstance(self.gate, DeepSeekV4HashRouter)
        # The routed-expert compute (a :class:`DeepSeekV4PreloadedExperts` keeping
        # all 256 experts resident on device in BFloat4_b) is always injected.
        self.experts = experts
        # Only the shared expert is prefetched: the routed experts are already resident.
        self.shared_experts = DeepSeekV4MLP(
            weights,
            "shared_experts",
            device,
            cache=cache,
            weight_dtype=weight_dtype,
            config=config,
            use_prefetcher=use_prefetcher,
            prefetch_buffers=prefetch_buffers,
            packed_weights=packed_weights,
            tp_size=tp_size,
        )

    def prefetch_weights(self):
        """Stage the shared expert's weights ahead of the decode that uses them."""
        self.shared_experts.prefetch_weights()

    def forward(self, hidden: ttnn.Tensor, input_ids: Optional[torch.Tensor] = None) -> ttnn.Tensor:
        """``hidden`` ``[B, S, 1, H]`` -> ``[B, S, 1, H]``. ``input_ids`` is required
        only for hash-routed layers (frozen ``tid2eid`` selection)."""
        b, s, _, h = hidden.shape
        x_flat = ttnn.reshape(hidden, [1, 1, b * s, h])
        _profile(self.device)

        with _region("MOE_ROUTER"):
            # Either router hands over the same (scores, selected ids) pair; they differ only
            # in how the ids are chosen (learned topk vs frozen table lookup).
            routing = self.gate(x_flat, input_ids) if self.is_hash else self.gate(x_flat)
        _profile(self.device)

        with _region("MOE_EXPERTS"):
            routed = self.experts(x_flat, routing)  # [1, 1, T, H]
        _profile(self.device)

        with _region("MOE_SHARED"):
            shared = self.shared_experts(x_flat)  # [1, 1, T, H]
            if self.packed_weights is not None:
                shared = ttnn.to_memory_config(shared, routed.memory_config())

        _profile(self.device)

        # Both halves are I-sliced under TP, so their H-partials add first and one
        # all-reduce recovers the full residual (Megatron MLP TP).
        combined = ttnn.add(routed, shared)
        if self.tp_size > 1:
            combined = _tp_all_reduce(combined, self.device)
        return ttnn.reshape(combined, [b, s, 1, h])

    def decode_static(self, hidden: ttnn.Tensor, hash_token: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """Trace-safe single-token-per-user MoE. ``hidden`` ``[B, 1, 1, H]`` -> same.

        Routing stays entirely on device: the learned top-k router is already
        host-sync-free, and hash layers gather their selected expert ids on device from
        the persistent ``hash_token`` ``[1,B]`` device token ids (see
        :meth:`DeepSeekV4HashRouter.forward_static`). The routed FFN runs through the
        no-host-readback fused-experts decode path.

        The batch's users are flattened onto the token axis, which is the layout the
        router and the expert compute already work in -- so the shared expert and the
        gate see one wider matmul while the routed experts, which each user sends
        somewhere different, stay one op per user.
        """
        b, h = hidden.shape[0], hidden.shape[-1]
        x_flat = ttnn.reshape(hidden, [1, 1, b, h])
        # The learned router has no separate trace-safe variant: it allocates every operand
        # it uses, so :meth:`DeepSeekV4TopKRouter.forward` is already capture-safe.
        if self.is_hash:
            routing = self.gate.forward_static(x_flat, hash_token)
        else:
            routing = self.gate(x_flat)
        routed = self.experts.decode_static(x_flat, routing)  # [1, 1, B, H]
        shared = self.shared_experts(x_flat)  # [1, 1, B, H]
        if self.packed_weights is not None:
            shared = ttnn.to_memory_config(shared, routed.memory_config())
        combined = ttnn.add(routed, shared)
        if self.tp_size > 1:
            combined = _tp_all_reduce(combined, self.device)
        return ttnn.reshape(combined, [b, 1, 1, h])
