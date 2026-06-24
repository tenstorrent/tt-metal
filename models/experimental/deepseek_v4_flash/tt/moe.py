from typing import Optional

import ttnn
import torch

from .common import DeepSeekV4Module, _profile, _region
from .layers import Linear
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


class DeepSeekV4MLP(DeepSeekV4Module):
    """Dense SwiGLU MLP (matches ``DeepseekV4MLP`` / ``LlamaMLP``).

    Used as the always-on *shared expert*: ``down(silu(gate(x)) * up(x))`` with
    no clamp (the routed experts clamp; the shared expert does not).
    """

    def __init__(
        self,
        weights: dict,
        prefix: str,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        cache = _as_cache(cache)
        self.gate_proj = Linear(
            weights[f"{prefix}.gate_proj.weight"], device, cache.file(f"{prefix}.gate_proj"), dtype=weight_dtype
        )
        self.up_proj = Linear(
            weights[f"{prefix}.up_proj.weight"], device, cache.file(f"{prefix}.up_proj"), dtype=weight_dtype
        )
        self.down_proj = Linear(
            weights[f"{prefix}.down_proj.weight"], device, cache.file(f"{prefix}.down_proj"), dtype=weight_dtype
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gate = ttnn.silu(self.gate_proj(x))
        return self.down_proj(ttnn.multiply(gate, self.up_proj(x)))


class DeepSeekV4TopKRouter(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4TopKRouter``.

    Produces a *dense* ``[1, 1, T, E]`` routing-weight tensor: ``sqrtsoftplus``
    of the gate logits gives per-expert scores; the top-k experts (by
    ``scores + e_score_correction_bias``) are selected via ``ttnn.topk`` +
    ``ttnn.scatter`` into a one-hot mask; the masked scores are renormalised to
    sum to 1 per token and scaled by ``routed_scaling_factor``. Unselected
    experts carry weight 0, which lets the dense expert compute drop them.
    """

    def __init__(self, config, weights: dict, device: ttnn.MeshDevice, cache: Optional[WeightCache] = None):
        self.device = device
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        cache = _as_cache(cache)
        self.gate = Linear(weights["gate.weight"], device, cache.file("gate"))
        bias = _materialize(
            weights["gate.e_score_correction_bias"], cache.file("gate.e_score_correction_bias"), ttnn.bfloat16
        )
        self.e_score_correction_bias = _load_weight(
            bias.reshape(1, 1, 1, self.num_experts) if bias is not None else None,
            device,
            cache_file_name=cache.file("gate.e_score_correction_bias"),
        )
        # Persistent scatter operands for the trace-safe decode path (T == 1).
        # ``ttnn.zeros`` / ``ttnn.ones`` host-init their buffers (a host->device
        # write that is illegal mid-capture), so the static router reuses these
        # pre-built constants instead of allocating + writing them each call.
        self._scatter_zeros = ttnn.zeros([1, 1, 1, self.num_experts], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)
        self._scatter_ones = ttnn.ones([1, 1, 1, self.top_k], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)

    def forward(self, x_flat: ttnn.Tensor) -> ttnn.Tensor:
        """``x_flat`` is ``[1, 1, T, H]``; returns routing weights ``[1, 1, T, E]``."""
        logits = self.gate(x_flat)  # [1, 1, T, E]
        scores = ttnn.sqrt(ttnn.softplus(logits))
        biased = ttnn.add(scores, self.e_score_correction_bias)
        _profile(self.device)

        # Top-k selection -> one-hot mask. Scatter (rather than a >= threshold
        # compare) selects exactly k experts even if two scores collide under
        # bf16 rounding. Scatter wants ROW_MAJOR + a matching-rank index tensor.
        _, top_idx = ttnn.topk(biased, self.top_k, dim=-1)  # [1, 1, T, k]
        t = x_flat.shape[2]
        top_idx = ttnn.to_layout(top_idx, ttnn.ROW_MAJOR_LAYOUT)
        mask = ttnn.zeros([1, 1, t, self.num_experts], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self.device)
        src = ttnn.ones([1, 1, t, self.top_k], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self.device)
        mask = ttnn.scatter(mask, -1, top_idx, src)
        mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)

        # Weights are the *unbiased* scores gathered at the selected experts,
        # normalised per token, then scaled. Masking before the sum makes the
        # dense [1,1,T,E] tensor equal the reference's gathered/normalised one.
        selected = ttnn.multiply(scores, mask)
        denom = ttnn.add(ttnn.sum(selected, dim=-1, keepdim=True), 1.0e-20)
        return ttnn.multiply(ttnn.div(selected, denom), self.routed_scaling_factor)

    def forward_static(self, x_flat: ttnn.Tensor) -> ttnn.Tensor:
        """Trace-safe single-token (``T == 1``) top-k routing -> ``[1,1,1,E]``.

        Identical math to :meth:`forward`, but the scatter's zeros / ones operands
        are the persistent constants built at init rather than freshly
        ``ttnn.zeros`` / ``ttnn.ones`` tensors (whose host-init write is rejected
        during trace capture). Scatter allocates its own output, which is allowed.
        """
        logits = self.gate(x_flat)  # [1, 1, 1, E]
        scores = ttnn.sqrt(ttnn.softplus(logits))
        biased = ttnn.add(scores, self.e_score_correction_bias)

        _, top_idx = ttnn.topk(biased, self.top_k, dim=-1)  # [1, 1, 1, k]
        top_idx = ttnn.to_layout(top_idx, ttnn.ROW_MAJOR_LAYOUT)
        mask = ttnn.scatter(self._scatter_zeros, -1, top_idx, self._scatter_ones)
        mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)

        selected = ttnn.multiply(scores, mask)
        denom = ttnn.add(ttnn.sum(selected, dim=-1, keepdim=True), 1.0e-20)
        return ttnn.multiply(ttnn.div(selected, denom), self.routed_scaling_factor)


class DeepSeekV4HashRouter(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4HashRouter`` (the first ``num_hash_layers`` MoE
    layers, paper §2.1).

    Expert *selection* is a frozen ``tid2eid[input_ids]`` lookup — a fixed
    token-id -> expert-id table — rather than a learned top-k argmax. The learned
    gate still produces the per-expert ``sqrtsoftplus`` scores that weight the
    selected experts; only the *which-experts* decision is static. As with
    :class:`DeepSeekV4TopKRouter` we emit a dense ``[1,1,T,E]`` weight tensor
    (selected experts carry their renormalised score, the rest are 0) so the same
    dense / preloaded expert compute consumes it.

    For the traced decode path the selection is done *fully on device*: the frozen
    ``tid2eid`` table is materialised once into a dense one-hot expert-mask table
    ``[vocab, E]`` resident on device, and the per-token selection mask is gathered
    with :func:`ttnn.embedding` straight from the (on-device) token id — no
    host-side scatter and no per-step host->device mask copy. The host prefill
    ``forward`` keeps the simple host scatter.
    """

    def __init__(self, config, weights: dict, device: ttnn.MeshDevice, cache: Optional[WeightCache] = None):
        self.device = device
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        cache = _as_cache(cache)
        self.gate = Linear(weights["gate.weight"], device, cache.file("gate"))
        # tid2eid [vocab, top_k]: frozen token-id -> expert-id table (host-side,
        # no tile cache) -- always materialise.
        tid = weights["gate.tid2eid"]
        tid = tid() if callable(tid) else tid
        self.tid2eid = tid.long()
        # Dense one-hot expert-selection table [vocab, E] resident on device: row
        # ``t`` is the selection mask for token id ``t``. ``ttnn.embedding`` gathers
        # the per-token mask on device from the (on-device) token id, so the static
        # decode router needs no host-side scatter / per-step mask copy.
        vocab = self.tid2eid.shape[0]
        mask_table = torch.zeros(vocab, self.num_experts, dtype=torch.float32)
        mask_table.scatter_(1, self.tid2eid, 1.0)
        self.mask_table = ttnn.from_torch(mask_table, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    def forward(self, x_flat: ttnn.Tensor, input_ids: torch.Tensor) -> ttnn.Tensor:
        """``x_flat`` ``[1,1,T,H]`` and ``input_ids`` torch ``[..]`` (T tokens);
        returns dense routing weights ``[1,1,T,E]``."""
        logits = self.gate(x_flat)  # [1, 1, T, E]
        scores = ttnn.sqrt(ttnn.softplus(logits))
        t = x_flat.shape[2]
        _profile(self.device)

        # Per-token expert selection gathered on device from the one-hot mask table
        # via ``ttnn.embedding`` (token ids -> dense one-hot mask [1,1,T,E]).
        ids_tt = ttnn.from_torch(
            input_ids.reshape(1, t).long().to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        mask_tt = ttnn.embedding(ids_tt, self.mask_table, layout=ttnn.TILE_LAYOUT)  # [1, T, E]
        mask_tt = ttnn.reshape(mask_tt, [1, 1, t, self.num_experts])

        selected = ttnn.multiply(scores, mask_tt)
        denom = ttnn.add(ttnn.sum(selected, dim=-1, keepdim=True), 1.0e-20)
        return ttnn.multiply(ttnn.div(selected, denom), self.routed_scaling_factor)

    def forward_static(self, x_flat: ttnn.Tensor, token_in: ttnn.Tensor) -> ttnn.Tensor:
        """Trace-safe, fully on-device hash routing: ``token_in`` ``[1,1]`` is the
        (persistent, on-device) decode token id. The per-token expert-selection
        mask is gathered from the on-device ``mask_table`` with :func:`ttnn.embedding`
        and the gate score path stays on device. Returns dense routing weights
        ``[1,1,1,E]``."""
        mask_tt = ttnn.embedding(token_in, self.mask_table, layout=ttnn.TILE_LAYOUT)  # [1, 1, E]
        mask_tt = ttnn.reshape(mask_tt, [1, 1, 1, self.num_experts])
        logits = self.gate(x_flat)
        scores = ttnn.sqrt(ttnn.softplus(logits))
        selected = ttnn.multiply(scores, mask_tt)
        denom = ttnn.add(ttnn.sum(selected, dim=-1, keepdim=True), 1.0e-20)
        return ttnn.multiply(ttnn.div(selected, denom), self.routed_scaling_factor)


# --------------------------------------------------------------------------- #
# fused_experts (single-op decode path)
#
# ``ttnn.experimental.deepseek.moe.fused_experts`` runs the whole routed-expert
# FFN (gate_up + SwiGLU + down + routing-weighted accumulation) for one token in
# a single device op. It is hard-wired to the real V4-Flash sizes -- an 8x8 (64)
# compute grid where each core owns 2 output tiles, so the hidden size must be
# exactly ``_FUSED_HIDDEN`` (64 * 2 * 32) -- and is decode-only (``T == 1``). The
# weights must be DRAM ND-sharded with one shard per core (see below), a layout
# distinct from the plain matmul weights used by the prefill loop, so the decode
# path keeps its own copy.
# --------------------------------------------------------------------------- #
_FUSED_HIDDEN = 4096  # op requires H == 64 cores * 2 tiles * 32 = 4096
_FUSED_COLS_PER_CORE = 64  # SwiGLU output columns per core (2 tiles)
_FUSED_NUM_CORES = 64  # 8x8 compute grid
_FUSED_DRAM_BANKS = 8  # Blackhole DRAM banks (round-robin shard target)


def _interleave_gate_up(w: torch.Tensor, block: int = _FUSED_COLS_PER_CORE) -> torch.Tensor:
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


def _fused_nd_dram_config(rows: int, cols: int, shard_width: int) -> ttnn.MemoryConfig:
    """DRAM ND-shard config: ``rows x shard_width`` shards round-robined over the
    DRAM banks (one shard per compute core), as ``fused_experts`` expects."""
    assert cols % shard_width == 0, f"last dim {cols} must divide into shards of {shard_width}"
    dram_core_range_set = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(bank, 0), ttnn.CoreCoord(bank, 0)) for bank in range(_FUSED_DRAM_BANKS)]
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
    )
    return sharded


class DeepSeekV4PreloadedExperts(DeepSeekV4Module):
    """Routed-experts compute via the single-op ``fused_experts`` kernel.

    The whole routed-expert FFN for one token (gate_up + SwiGLU + down +
    routing-weighted accumulation) runs in a single ``fused_experts`` device op.
    The op is hard-wired to the real V4-Flash sizes -- an 8x8 (64) compute grid
    where each core owns 2 output tiles, so ``H`` must be exactly ``_FUSED_HIDDEN``
    (``64 * 2 * 32 == 4096``) and ``I`` a multiple of the 64-column per-core
    slice. Both prefill and decode go through the op: it is natively single-token
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
        dtype: ttnn.DataType = ttnn.bfloat4_b,
        cache: Optional[WeightCache] = None,
    ):
        self.device = device
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.intermediate = config.moe_intermediate_size
        self.hidden = config.hidden_size
        self.limit = config.swiglu_limit
        cache = _as_cache(cache)

        # ``fused_experts`` is hard-wired to the real V4-Flash sizes: ``H == 4096``
        # on the 64-core grid and ``I`` a multiple of the 64-column per-core slice.
        # There is no fallback path -- this class is for that config only.
        if self.hidden != _FUSED_HIDDEN or self.intermediate % _FUSED_COLS_PER_CORE != 0:
            raise ValueError(
                f"DeepSeekV4PreloadedExperts requires the fused_experts layout "
                f"(H == {_FUSED_HIDDEN}, I % {_FUSED_COLS_PER_CORE} == 0); "
                f"got H={self.hidden}, I={self.intermediate}"
            )
        gate_up_nd = _fused_nd_dram_config(self.hidden, 2 * self.intermediate, 2 * _FUSED_COLS_PER_CORE)
        down_nd = _fused_nd_dram_config(self.intermediate, self.hidden, self.hidden // _FUSED_NUM_CORES)

        # Upload every expert once as the op's DRAM ND-sharded weights (gate_up
        # interleaved per core, down ND-sharded), stored in low precision. With
        # caching enabled and a hit, the provider (and its expensive dequant) is
        # skipped entirely; the ND-shard layout can't round-trip the tile cache,
        # so the interleaved weight is cached in standard DRAM and resharded on
        # device (see :func:`_load_fused_weight`).
        self._gate_up_fused: list[ttnn.Tensor] = []
        self._down_fused: list[ttnn.Tensor] = []
        for e in range(self.num_experts):
            gu_f_name, dn_f_name = f"experts.{e}.gate_up_fused", f"experts.{e}.down_fused"
            need_torch = not (cache.hit(gu_f_name, dtype) and cache.hit(dn_f_name, dtype))
            if cache.require_cache and need_torch:
                raise RuntimeError(f"weight cache miss for routed expert {e} (gate_up/down) with require_cache=True")
            gate_up_w, down_w = provider(e) if need_torch else (None, None)
            # Provider gives gate_up [2I, H] / down [H, I]; transpose to matmul-ready
            # [H, 2I] / [I, H] (memoized so each is materialized at most once).
            gate_up_t = _memo((lambda gw=gate_up_w: gw.t().contiguous()) if gate_up_w is not None else (lambda: None))
            down_t = _memo((lambda dw=down_w: dw.t().contiguous()) if down_w is not None else (lambda: None))
            gu_il = _materialize(lambda: _interleave_gate_up(gate_up_t()), cache.file(gu_f_name), dtype)
            self._gate_up_fused.append(
                _load_fused_weight(gu_il, device, gate_up_nd, cache_file_name=cache.file(gu_f_name), dtype=dtype)
            )
            self._down_fused.append(
                _load_fused_weight(down_t(), device, down_nd, cache_file_name=cache.file(dn_f_name), dtype=dtype)
            )

    def _run_fused(self, x_tok: ttnn.Tensor, routing_row: ttnn.Tensor, num_experts: int) -> ttnn.Tensor:
        """Run ``fused_experts`` for one token. ``x_tok`` ``[1,1,1,H]`` (TILE) and
        ``routing_row`` a ROW_MAJOR routing slice; returns ``[1,1,1,H]``."""
        routing_row = ttnn.reshape(routing_row, [1, 1, 1, self.num_experts])
        out = ttnn.experimental.deepseek.moe.fused_experts(
            x_tok,
            routing_weights=routing_row,
            gate_up_weights=self._gate_up_fused,
            down_weights=self._down_fused,
            num_experts=num_experts,
            intermediate_size=self.intermediate,
            swiglu_limit=self.limit,
        )  # [1, 1, H]
        return ttnn.reshape(out, [1, 1, 1, self.hidden])

    def _decode_token(self, x_tok: ttnn.Tensor, rw_tok: ttnn.Tensor) -> ttnn.Tensor:
        """Run one token's routed FFN through ``fused_experts``.

        ``x_tok`` ``[1,1,1,H]`` and ``rw_tok`` the host routing-weight row ``[E]``;
        returns ``[1,1,1,H]``. The op finds the active (non-zero) experts from the
        routing row itself, so we only pass ``num_experts`` = the hit count.
        """
        routing_row = ttnn.to_layout(rw_tok, ttnn.ROW_MAJOR_LAYOUT)
        out = self._run_fused(x_tok, routing_row, 6)
        _profile(self.device)
        return out

    def forward(self, x_flat: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        """``x_flat`` ``[1,1,T,H]`` and ``routing_weights`` ``[1,1,T,E]``; returns ``[1,1,T,H]``.

        Every token runs as its own single-token ``fused_experts`` op (the op is
        natively ``T == 1``), so prefill is computed by decode: the ``T`` per-token
        outputs are concatenated back into ``[1,1,T,H]``.
        """
        t = x_flat.shape[2]
        _profile(self.device)

        if t == 1:
            return self._decode_token(x_flat, routing_weights)

        # Prefill loops the single-token op over ``T`` tokens. Slicing a single,
        # non-tile-aligned row out of a TILE tensor forces an untilize/unpad +
        # re-tilize per token (and the routing row needs a per-token untilize for
        # the ROW_MAJOR op input). Hoist both layout conversions out of the loop:
        # untilize ``x_flat`` / ``routing_weights`` once, slice rows cheaply in
        # ROW_MAJOR, and tilize only the small ``[1,1,1,H]`` activation row the op
        # actually consumes.
        x_rm = ttnn.to_layout(x_flat, ttnn.ROW_MAJOR_LAYOUT)  # [1, 1, T, H]
        rw_rm = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)  # [1, 1, T, E]
        outs = []
        for ti in range(t):
            x_tok_rm = ttnn.slice(x_rm, [0, 0, ti, 0], [1, 1, ti + 1, self.hidden])
            x_tok = ttnn.to_layout(x_tok_rm, ttnn.TILE_LAYOUT)
            routing_row = ttnn.slice(rw_rm, [0, 0, ti, 0], [1, 1, ti + 1, self.num_experts])
            outs.append(self._run_fused(x_tok, routing_row, 6))
        return ttnn.concat(outs, dim=2)  # [1, 1, T, H]

    def decode_static(self, x_tok: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        """Trace-safe single-token routed FFN. ``x_tok`` ``[1,1,1,H]`` and
        ``routing_weights`` ``[1,1,1,E]`` (a device tensor); returns ``[1,1,1,H]``.

        Unlike :meth:`forward`, the active experts are *not* read back to host:
        ``fused_experts`` finds the non-zero experts from the routing row on device
        and ``num_experts`` is fixed to ``num_experts_per_tok`` (the router always
        selects exactly ``top_k``), so the op's program — and hence the trace — is
        invariant across steps.
        """
        routing_row = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)
        routing_row = ttnn.reshape(routing_row, [1, 1, 1, self.num_experts])
        out = ttnn.experimental.deepseek.moe.fused_experts(
            x_tok,
            routing_weights=routing_row,
            gate_up_weights=self._gate_up_fused,
            down_weights=self._down_fused,
            num_experts=self.top_k,
            intermediate_size=self.intermediate,
            swiglu_limit=self.limit,
        )  # [1, 1, H]
        return ttnn.reshape(out, [1, 1, 1, self.hidden])


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
    ):
        self.device = device
        self.hidden = config.hidden_size
        cache = _as_cache(cache)
        # ``gate`` may be injected (e.g. a :class:`DeepSeekV4HashRouter` for the
        # first ``num_hash_layers`` layers); otherwise the learned top-k router.
        self.gate = gate if gate is not None else DeepSeekV4TopKRouter(config, weights, device, cache=cache)
        self.is_hash = isinstance(self.gate, DeepSeekV4HashRouter)
        # The routed-expert compute (a :class:`DeepSeekV4PreloadedExperts` keeping
        # all 256 experts resident on device in BFloat4_b) is always injected.
        self.experts = experts
        self.shared_experts = DeepSeekV4MLP(weights, "shared_experts", device, cache=cache, weight_dtype=weight_dtype)

    def forward(self, hidden: ttnn.Tensor, input_ids: Optional[torch.Tensor] = None) -> ttnn.Tensor:
        """``hidden`` ``[B, S, 1, H]`` -> ``[B, S, 1, H]``. ``input_ids`` is required
        only for hash-routed layers (frozen ``tid2eid`` selection)."""
        b, s, _, h = hidden.shape
        x_flat = ttnn.reshape(hidden, [1, 1, b * s, h])
        _profile(self.device)

        with _region("MOE_ROUTER"):
            if self.is_hash:
                routing_weights = self.gate(x_flat, input_ids)  # [1, 1, T, E]
            else:
                routing_weights = self.gate(x_flat)  # [1, 1, T, E]
        _profile(self.device)

        with _region("MOE_EXPERTS"):
            routed = self.experts(x_flat, routing_weights)  # [1, 1, T, H]
            routed = ttnn.reshape(routed, [b, s, 1, h])
        _profile(self.device)

        with _region("MOE_SHARED"):
            shared = self.shared_experts(hidden)  # [B, S, 1, H]

        _profile(self.device)

        return ttnn.add(routed, shared)

    def decode_static(self, hidden: ttnn.Tensor, hash_token: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """Trace-safe single-token MoE. ``hidden`` ``[1, 1, 1, H]`` -> ``[1, 1, 1, H]``.

        Routing stays entirely on device: the learned top-k router is already
        host-sync-free, and hash layers gather their expert-selection mask on device
        from the persistent ``hash_token`` ``[1,1]`` device token id (see
        :meth:`DeepSeekV4HashRouter.forward_static`). The routed FFN runs through the
        no-host-readback fused-experts decode path.
        """
        h = hidden.shape[-1]
        x_flat = ttnn.reshape(hidden, [1, 1, 1, h])
        if self.is_hash:
            routing_weights = self.gate.forward_static(x_flat, hash_token)
        else:
            routing_weights = self.gate.forward_static(x_flat)
        routed = self.experts.decode_static(x_flat, routing_weights)  # [1, 1, 1, H]
        routed = ttnn.reshape(routed, [1, 1, 1, h])
        shared = self.shared_experts(hidden)
        return ttnn.add(routed, shared)
