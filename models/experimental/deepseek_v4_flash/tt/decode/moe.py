"""DeepSeek-V4-Flash MoE: the routers, the shared expert and the routed experts.

ttnn port of ``DeepseekV4SparseMoeBlock`` and the pieces it is built from -- the learned
``DeepseekV4TopKRouter``, the frozen ``DeepseekV4HashRouter`` and the ``DeepseekV4MLP``
shared expert -- from ``modular_deepseek_v4.py``. The routed FFN is not a per-expert host
loop: it is one ``fused_experts`` device op that picks the hit experts on device.

Letters (the package convention; activations here are rank 4 with ``S == 1``, weights rank 2):
``B`` users decoded per step; ``T`` the token rows an activation's *shard* holds -- its
HEIGHT_SHARDED shard height, 1 on the single-token decode path, and *not* the reference's
``T`` (``B*S`` flattened tokens); ``D`` hidden_size; ``E`` num_local_experts; ``I``
moe_intermediate_size; ``k`` num_experts_per_tok; ``K`` / ``N`` matmul in/out features.
"""

import os
from typing import NamedTuple, Optional

import ttnn
import torch

from ..common import CCL_TOPOLOGY, DeepSeekV4Module, _profile, width_sharded_l1_config
from .decode_prefetch import (
    DECODE_LAYOUTS,
    check_decode_layout,
    decode_prefetch_rings,
    make_decode_prefetch_buffers,
    ring_prefetch,
    tp_gate_up_layout,
)
from ..layers import Linear, LinearDecode, _core_grid_contains
from ..system_config import active_system_config
from ..weight_cache import WeightCache, _as_cache, _load_weight, _materialize, _memo


class SparseRouting(NamedTuple):
    """A routing decision in the form ``fused_experts`` consumes it.

    Exactly one of ``ranking`` / ``indices`` says *which* experts were selected, and both are
    handed to the op untouched -- it does the normalize/scale itself. Widening this into a dense
    ``[1,1,T,E]`` weight row (scatter a one-hot mask, mask the scores, sum, divide, scale,
    relayout -- nine device ops) would only have the op's first kernel scan those E columns
    straight back down to k values.

    ``scores``: ``[1,1,T,E]`` bf16, the *unbiased* per-expert scores -- the values that become
    the weights. TILE, or (``T == 1``) ROW_MAJOR, the stick a hub-mode gate emits.

    ``ranking``: ``[1,1,T,E]`` bf16, the row to rank on -- normally
    ``scores + e_score_correction_bias``. Filled by the learned router: the expert op top-k's it
    on device, on the leader core that already reads the row for the weights, so the router needs
    neither a ``ttnn.topk`` launch nor a DRAM round-trip of its id output.

    ``indices``: ``[1,1,T,k]`` TILE, the already-chosen ids (bf16, the only dtype
    ``ttnn.embedding`` gathers), which the op reads instead of ranking. Filled by the hash router,
    whose selection is a frozen table lookup rather than a top-k.
    """

    scores: ttnn.Tensor
    indices: Optional[ttnn.Tensor] = None
    ranking: Optional[ttnn.Tensor] = None


class DeepSeekV4MLP(DeepSeekV4Module):
    """Dense SwiGLU MLP (matches ``DeepseekV4MLP`` / ``LlamaMLP``).

    Used as the always-on *shared expert*: ``down(silu(gate(x)) * up(x))``, with no clamp
    unless ``clamp_swiglu`` is set (V4's routed experts clamp; its shared expert does not).

    ``use_prefetcher=True`` runs the three projections as :class:`LinearDecode`. Gate and up are
    full-width (hub mode) on q_a's 32-core ring and consume a ROW_MAJOR HEIGHT_SHARDED replica of
    the tokens (the decode all-gather); they emit ROW_MAJOR WIDTH_SHARDED ``[T, I]`` (64 columns
    per core), which is the K-sharding down wants of its activation. Down stays on the shared
    64-core decode GCB.

    Under TP, gate and up cannot join that 32-core ring (``N = I/TP`` is only 8 cores) and a
    private GCB on those cores collides with ``fused_hyperconnection`` static CBs, so they take a
    transient DRAM->L1 copy instead -- that cut is live at TP4. The copy path is decode shaped: it
    width-shards the token rows over one tile row, which caps it at the 32 rows a tile holds, so a
    wider input cannot use it. ``config`` is needed either way, to check the fixed weight layouts
    against the shapes this model wants.

    Subclasses for other checkpoints override ``decode_layouts`` (the gate/up/down cuts),
    ``prefetch_rings`` (``tp_size -> {name: PrefetchRing}``, the GCB each streams through) and
    ``clamp_swiglu`` (whether the activation takes the routed experts' ``swiglu_limit`` clamp).
    """

    decode_layouts = DECODE_LAYOUTS
    prefetch_rings = staticmethod(decode_prefetch_rings)
    clamp_swiglu = False

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
        tp_size: int = 1,
        matmul_decode: Optional[bool] = None,
    ):
        """Build the shared expert's three projections.

        ``weights`` holds the checkpoint's ``<prefix>.gate_proj`` / ``up_proj`` / ``down_proj``
        ``.weight`` torch tensors ``[I, D]`` / ``[I, D]`` / ``[D, I]`` (``[I/TP, D]`` and
        ``[D, I/TP]`` per rank under TP). ``matmul_decode`` (default: ``use_prefetcher``) selects
        :class:`LinearDecode` over plain :class:`Linear` weights. The ``LinearDecode`` weights
        stream through ``prefetch_buffers`` when ``use_prefetcher`` is set -- the mapping shared by
        every layer on the device, since a fresh dict per layer would build a GCB per layer and
        overflow the DRISC senders' state zone -- and are copied DRAM -> L1 per call otherwise.
        """
        cache = _as_cache(cache)
        self.device = device
        self.use_prefetcher = use_prefetcher
        self.tp_size = tp_size
        self.swiglu_limit = config.swiglu_limit if self.clamp_swiglu else None
        # After the transpose the weights are matmul-shaped: gate/up ``[D, I]``
        # (column-parallel: shard N) and down ``[I, D]`` (row-parallel: shard K).
        gate_up_mapper = ttnn.ShardTensorToMesh(device, dim=1) if tp_size > 1 else None
        down_mapper = ttnn.ShardTensorToMesh(device, dim=0) if tp_size > 1 else None
        tp_tag = f".tp{tp_size}" if tp_size > 1 else ""
        if not (use_prefetcher if matmul_decode is None else matmul_decode):
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
        local_inter = inter // tp_size if tp_size > 1 else inter
        gate_up_layout = dict(check_decode_layout("shared_gate_proj", hidden, inter, layouts=self.decode_layouts))
        down_layout = dict(check_decode_layout("shared_down_proj", inter, hidden, layouts=self.decode_layouts))
        gate_up_prefetch = {"use_prefetcher": False}
        down_prefetch = {"use_prefetcher": False}
        if use_prefetcher:
            if prefetch_buffers is None:
                prefetch_buffers = make_decode_prefetch_buffers(device, weight_dtype)
            # On V4, gate/up share q_a's 32-receiver FIFO (and CSA and the hyper-connections'
            # fn), queued after those in :meth:`DeepSeekV4SparseMoeBlock.prefetch_weights`.
            rings = self.prefetch_rings(tp_size)
            gate_up_prefetch = ring_prefetch(rings, "shared_gate_proj", prefetch_buffers, device, weight_dtype)
            down_prefetch = ring_prefetch(rings, "shared_down_proj", prefetch_buffers, device, weight_dtype)
        decode_gate_up_mapper = ttnn.ShardTensorToMesh(device, dim=-1) if tp_size > 1 else None
        decode_down_mapper = ttnn.ShardTensorToMesh(device, dim=-2) if tp_size > 1 else None
        if tp_size > 1:
            # Per-rank gate/up is N=I/TP (8 cores at I=2048, TP=4) and cannot join
            # q_a's 32-receiver ring. A private GCB on those cores also collides with
            # fused_hyperconnection static CBs on (0,0), so they stay on DRAM->L1.
            # Down only cuts K and stays on 64 cores.
            gate_up_layout = tp_gate_up_layout(tp_size, hidden, local_inter, layouts=self.decode_layouts)
            down_layout = {"K": local_inter, "N": hidden}
            gate_up_prefetch = {"use_prefetcher": False}
        self.gate_proj = LinearDecode(
            weights[f"{prefix}.gate_proj.weight"],
            device,
            cache.file(f"{prefix}.gate_proj{tp_tag}.decode" if tp_size > 1 else f"{prefix}.gate_proj{tp_tag}.full"),
            dtype=weight_dtype,
            mesh_mapper=decode_gate_up_mapper,
            rectangle_b_grid=True,
            **gate_up_layout,
            **gate_up_prefetch,
            use_rm_hs=True,
        )
        self.up_proj = LinearDecode(
            weights[f"{prefix}.up_proj.weight"],
            device,
            cache.file(f"{prefix}.up_proj{tp_tag}.decode" if tp_size > 1 else f"{prefix}.up_proj{tp_tag}.full"),
            dtype=weight_dtype,
            mesh_mapper=decode_gate_up_mapper,
            rectangle_b_grid=True,
            **gate_up_layout,
            **gate_up_prefetch,
            use_rm_hs=True,
        )
        self.down_proj = LinearDecode(
            weights[f"{prefix}.down_proj.weight"],
            device,
            cache.file(f"{prefix}.down_proj{tp_tag}"),
            dtype=weight_dtype,
            mesh_mapper=decode_down_mapper,
            num_inputA_cores=max(1, local_inter // 64),
            **down_prefetch,
            rectangle_b_grid=True,
            **down_layout,
            use_rm_hs=False,
        )
        assert self.gate_proj._can_matmul_decode_rm_hs() and self.up_proj._can_matmul_decode_rm_hs(), (
            "shared expert gate/up must use ROW_MAJOR HEIGHT_SHARDED matmul_decode, "
            f"but gate partial={self.gate_proj.partial_width_sharded} up partial={self.up_proj.partial_width_sharded}"
        )
        assert self.down_proj._can_matmul_decode_rm_hs(), (
            "shared expert down must use ROW_MAJOR HEIGHT_SHARDED matmul_decode, "
            f"but down partial={self.down_proj.partial_width_sharded}"
        )

    def prefetch_weights(self):
        """Stage the three ``[D, I]`` / ``[D, I]`` / ``[I, D]`` projection weights ahead of the
        :meth:`forward` that uses them.

        Queued gate, up, then down: gate/up stream through q_a's 32-receiver FIFO (queued after
        q_a, CSA and the FFN hyper-connection's ``fn``), down through the shared 64-core GCB. Under TP, gate/up are DRAM->L1 copies.
        Queue order on each ring has to match consume order, here and against the attention block
        (whose weights precede down on the shared buffer, because attention runs first in the
        decoder layer).
        """
        if not self.use_prefetcher:
            return
        self.gate_proj.fetch_weights()
        self.up_proj.fetch_weights()
        self.down_proj.fetch_weights()

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """``x`` ``[1, 1, T, D]`` -> ``[1, 1, T, D]`` in the same layout.

        ``T`` is the packed-token count the matmul sees -- the shard height of the replica, one row
        per user of the step -- which is the form ``matmul_decode`` needs on the prefetched path
        and the form the caller already built for the router, so nothing is reshaped here. A
        per-user leading rank-4 batch would instead decode as one matmul per user.
        """
        if isinstance(self.gate_proj, LinearDecode) and self.gate_proj.use_rm_hs:
            if not self.gate_proj._is_replicated_rm_hs(x):
                x = self.gate_proj.to_replicated_rm_hs_activation(x)
        # Full-width gate/up consume the ROW_MAJOR HEIGHT_SHARDED replica and leave
        # ROW_MAJOR WIDTH_SHARDED results over the 32 cores holding 64 columns each,
        # which is how down_proj wants its activation sharded along K, so the product
        # feeds it where it already sits.
        gate, up = self.gate_proj(x), self.up_proj(x)
        if self.swiglu_limit is not None:
            # Not ``ttnn.clamped_silu_glu``: on these ROW_MAJOR WIDTH_SHARDED operands its result is wrong.
            gate = ttnn.clamp(gate, max=self.swiglu_limit)
            up = ttnn.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
        gated = ttnn.multiply(ttnn.silu(gate), up)
        out = self.down_proj(gated)
        return out


def _router_gate_activation(gate, x_flat: ttnn.Tensor) -> ttnn.Tensor:
    """Match ``x_flat`` ``[1,1,T,D]`` to a ``LinearDecode`` gate's ``use_rm_hs`` contract.

    Returned as-is when it is already a ROW_MAJOR HEIGHT_SHARDED replica covering the gate's B
    cores, otherwise replicated onto them (:meth:`LinearDecode.to_replicated_rm_hs_activation`);
    a ``use_rm_hs=False`` gate gets tiled WIDTH_SHARDED A instead, and a plain :class:`Linear`
    gate is handed ``x_flat`` untouched.
    """
    if not isinstance(gate, LinearDecode):
        return x_flat
    if gate.use_rm_hs:
        if gate._is_replicated_rm_hs(x_flat):
            a_grid = x_flat.memory_config().shard_spec.grid
            b_grid = gate.b_core_grid()
            if a_grid == b_grid or _core_grid_contains(a_grid, b_grid):
                return x_flat
        return gate.to_replicated_rm_hs_activation(x_flat)
    return gate.to_width_sharded_activation(x_flat)


def _make_router_gate(
    weights: dict,
    device: ttnn.MeshDevice,
    cache: WeightCache,
    config,
    use_prefetcher: bool = False,
    prefetch_buffers: Optional[dict] = None,
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
    matmul_decode: Optional[bool] = None,
    layouts: dict = DECODE_LAYOUTS,
    rings: Optional[dict] = None,
):
    """The learned ``[D, E]`` router projection, as :class:`Linear` or :class:`LinearDecode`.

    ``matmul_decode`` (default: ``use_prefetcher``) picks :class:`LinearDecode`; without it
    (prefill) the gate keeps ``ttnn.linear``. A ``LinearDecode`` gate without the prefetcher
    copies its weight DRAM -> L1 per call. Decode runs
    ``matmul_decode`` in hub mode: the gate is full-width on ``n_blocks`` cores (8 for V4, one
    output tile per core), so it consumes the decode all-gather replica (ROW_MAJOR HEIGHT_SHARDED
    A) where it sits instead of unreplicating it through DRAM and re-sharding it -- that round
    trip was four extra device ops per step. The 8-receiver cut cannot join the shared 64-receiver
    ring or q_a's 32-receiver one, so it streams through :data:`ROUTER_GATE_GCB`. ``layouts`` is
    the table holding the ``router_gate`` cut and ``rings`` (default
    :func:`~.decode_prefetch.decode_prefetch_rings`) the GCB it streams through, if any.
    """
    if not (use_prefetcher if matmul_decode is None else matmul_decode):
        return Linear(weights["gate.weight"], device, cache.file("gate"))
    layout = dict(check_decode_layout("router_gate", config.hidden_size, config.num_local_experts, layouts=layouts))
    prefetch = {"use_prefetcher": False}
    if use_prefetcher:
        prefetch = ring_prefetch(
            decode_prefetch_rings() if rings is None else rings,
            "router_gate",
            {} if prefetch_buffers is None else prefetch_buffers,
            device,
            weight_dtype,
        )
    gate = LinearDecode(
        weights["gate.weight"],
        device,
        cache.file("gate.decode"),
        dtype=weight_dtype,
        **layout,
        **prefetch,
        rectangle_b_grid=True,
        use_rm_hs=True,
    )
    assert gate.num_inputB_cores == layout["n_blocks"], (
        f"router gate must be full-width on n_blocks={layout['n_blocks']} cores, "
        f"got {gate.num_inputB_cores} B cores"
    )
    assert gate._can_matmul_decode_rm_hs(), (
        "the router gate must run hub-mode matmul_decode so it reads the decode all-gather replica in "
        f"place, but its layout is partial={gate.partial_width_sharded}"
    )
    return gate


class DeepSeekV4TopKRouter(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4TopKRouter``.

    ``sqrtsoftplus`` of the gate logits gives per-expert scores; the bias-corrected row
    ``scores + e_score_correction_bias`` *is* the routing decision this module emits, and the
    top-k is taken inside ``fused_experts`` (see :class:`SparseRouting`). That removes a
    ``ttnn.topk`` -- and the DRAM round-trip of its id output -- from every step: the op ranks on
    the row it has to read anyway for the weights, on the core that broadcasts the winners.

    The reference's renormalize-and-scale tail likewise runs inside ``fused_experts``, on the k
    values per token it already reads, rather than here across a dense E-wide row.

    ``decode_layouts`` is the table holding the gate's ``router_gate`` cut and
    ``prefetch_rings`` the ring table naming the GCB it streams through.
    """

    decode_layouts = DECODE_LAYOUTS
    prefetch_rings = staticmethod(decode_prefetch_rings)

    def __init__(
        self,
        config,
        weights: dict,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        use_prefetcher: bool = False,
        prefetch_buffers: Optional[dict] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        matmul_decode: Optional[bool] = None,
    ):
        """``weights["gate.weight"]`` is the checkpoint's ``[E, D]`` projection and
        ``weights["gate.e_score_correction_bias"]`` the ``[E]`` bias, loaded as a
        ``[1,1,1,E]`` ROW_MAJOR row -- the layout the ROW_MAJOR score row it is added to uses.
        ``use_prefetcher`` / ``prefetch_buffers`` / ``matmul_decode`` select and feed the
        :class:`LinearDecode` gate of :func:`_make_router_gate`.
        """
        self.device = device
        self.num_experts = config.num_local_experts
        # The top-k is applied inside ``fused_experts`` now, on the row this module emits; the
        # attribute stays because it is the router's k (and the hash router exposes the same one).
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        cache = _as_cache(cache)
        self.gate = _make_router_gate(
            weights,
            device,
            cache,
            config,
            use_prefetcher=use_prefetcher,
            prefetch_buffers=prefetch_buffers,
            weight_dtype=weight_dtype,
            matmul_decode=matmul_decode,
            layouts=self.decode_layouts,
            rings=self.prefetch_rings(),
        )
        bias = _materialize(
            weights["gate.e_score_correction_bias"], cache.file("gate.e_score_correction_bias"), ttnn.bfloat16
        )
        self.e_score_correction_bias = _load_weight(
            bias.reshape(1, 1, 1, self.num_experts) if bias is not None else None,
            device,
            cache_file_name=cache.file("gate.e_score_correction_bias"),
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def prefetch_weights(self):
        """Stage the ``[D, E]`` gate weight ahead of the decode that reads it.

        The gate streams through :data:`ROUTER_GATE_GCB` (hub mode needs a full-width cut on 8
        receivers), so this queues that ring rather than copying DRAM->L1.
        """
        if isinstance(self.gate, LinearDecode):
            self.gate.fetch_weights()

    def _scores(self, x_flat: ttnn.Tensor) -> ttnn.Tensor:
        """Per-expert ``sqrtsoftplus`` gate scores ``[1,1,T,E]``.

        ROW_MAJOR and one row per core on a hub-mode (``use_rm_hs``) gate -- the stick the fused
        expert op reads directly, so nothing tilizes it -- and TILE on the ``ttnn.linear`` path.
        """
        x = _router_gate_activation(self.gate, x_flat)
        return ttnn.sqrt(ttnn.softplus(self.gate(x)))

    def ranking_bias(self, image_mask: Optional[ttnn.Tensor] = None, layout=ttnn.ROW_MAJOR_LAYOUT) -> ttnn.Tensor:
        """The bias added to the scores for ranking: ``e_score_correction_bias`` ``[1,1,1,E]``. A checkpoint
        that routes image tokens with their own bias overrides this to pick it per token from ``image_mask``
        (``[1, T]`` uint32 ROW_MAJOR, 1 for an image token); V4 has none. ``layout`` is that per-token row's."""
        if image_mask is not None:
            raise ValueError("this router has no image-token bias")
        return self.e_score_correction_bias

    def forward(self, x_flat: ttnn.Tensor, image_mask: Optional[ttnn.Tensor] = None) -> SparseRouting:
        """``x_flat`` ``[1,1,T,D]`` ROW_MAJOR -> the ``[1,1,T,E]`` score and ranking rows.

        Trace-safe as it stands, so prefill and the captured decode share this one path: every op
        allocates its own output and nothing is host-initialised. ``image_mask`` see :meth:`ranking_bias`.
        """
        assert x_flat.layout == ttnn.ROW_MAJOR_LAYOUT, "x_flat must be in row-major layout"
        scores = self._scores(x_flat)  # [1, 1, T, E]

        # Ranked on the bias-corrected scores, weighted by the uncorrected ones -- which is why
        # both rows travel to the expert op instead of just the winners' values. The op top-k's
        # `biased` on device and gathers `scores` at the winners, so there is no ttnn.topk here
        # and no id tensor to round-trip through DRAM.
        bias = self.ranking_bias(image_mask)
        biased = ttnn.add(scores, bias)
        if bias is not self.e_score_correction_bias:
            ttnn.deallocate(bias)
        _profile(self.device)
        return SparseRouting(scores=scores, ranking=biased)


class DeepSeekV4HashRouter(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4HashRouter`` (the first ``num_hash_layers`` MoE layers,
    paper §2.1).

    Expert *selection* is a frozen ``tid2eid[input_ids]`` lookup -- a fixed token-id -> expert-id
    table -- rather than a learned top-k argmax; the learned gate still produces the per-expert
    ``sqrtsoftplus`` scores that weight the selected experts. The output is the same
    :class:`SparseRouting` contract the learned router emits -- ``indices`` from the table lookup
    where that one fills ``ranking`` -- so both feed the expert compute through one contract.

    The selection is gathered *fully on device* by embedding the token id in the frozen table,
    which is already the ``[V, k]`` list of expert ids the sparse contract wants: no host-side
    scatter and no per-step host->device copy. The table is held as bfloat16 because that is the
    only dtype :func:`ttnn.embedding` gathers from; every expert id is exact there (E <= 256 is
    asserted) and ``fused_experts`` reads bf16-encoded ids directly. It is ``[V, k]`` (1.5 MB at a
    128k vocab), not a ``[V, E]`` one-hot (64 MB).
    """

    def __init__(
        self,
        config,
        weights: dict,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        use_prefetcher: bool = False,
        prefetch_buffers: Optional[dict] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        matmul_decode: Optional[bool] = None,
    ):
        """``weights["gate.tid2eid"]`` is the frozen ``[V, k]`` int64 token-id -> expert-id table,
        uploaded as the bf16 ROW_MAJOR device tensor ``eid_table`` (host-side only, so it is
        always read, cache or not). The learned gate and the ``[E]`` correction bias are built
        exactly as in :class:`DeepSeekV4TopKRouter`.
        """
        self.device = device
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        cache = _as_cache(cache)
        self.gate = _make_router_gate(
            weights,
            device,
            cache,
            config,
            use_prefetcher=use_prefetcher,
            prefetch_buffers=prefetch_buffers,
            weight_dtype=weight_dtype,
            matmul_decode=matmul_decode,
        )
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

    def prefetch_weights(self):
        """Stage the ``[D, E]`` gate weight ahead of the decode that reads it.

        The gate streams through :data:`ROUTER_GATE_GCB` (hub mode needs a full-width cut on 8
        receivers), so this queues that ring rather than copying DRAM->L1.
        """
        if isinstance(self.gate, LinearDecode):
            self.gate.fetch_weights()

    def _scores(self, x_flat: ttnn.Tensor) -> ttnn.Tensor:
        """Per-expert ``sqrtsoftplus`` gate scores ``[1,1,T,E]`` (shared head of both paths).

        ROW_MAJOR and one row per core, the stick a hub-mode gate emits; the fused-expert op
        reads that form directly (see ``fused_experts_device_operation.cpp``), so nothing tilizes
        it on this path.
        """
        return ttnn.sqrt(ttnn.softplus(self.gate(_router_gate_activation(self.gate, x_flat))))

    def _select(self, token_in: ttnn.Tensor, t: int) -> ttnn.Tensor:
        """Gather the selected expert ids for on-device token ids.

        ``token_in`` ``[1,B]`` -> ``[1,1,t,k]`` bf16 TILE, ``t`` being the row count
        :meth:`forward_static` derived.
        """
        ids = ttnn.embedding(token_in, self.eid_table, layout=ttnn.TILE_LAYOUT)  # [1, T, k] bf16
        return ttnn.reshape(ids, [1, 1, t, self.top_k])

    def forward_static(self, x_flat: ttnn.Tensor, token_in: ttnn.Tensor) -> SparseRouting:
        """Trace-safe, fully on-device hash routing.

        ``token_in`` ``[1,B]`` are the persistent on-device token ids, one per user of the step,
        and ``x_flat`` ``[1,1,T,D]`` the routed activation; the returned :class:`SparseRouting`
        carries the ``[1,1,T,E]`` scores and the ``[1,1,T,k]`` table-looked-up ids. ``T`` is 1 for
        a ROW_MAJOR HEIGHT_SHARDED (replica) activation -- a replica holds one token row per core
        -- and ``x_flat.shape[2]`` otherwise.

        The only entry point. A caller holding host ids uploads them itself (see
        :meth:`DeepSeekV4DecoderLayer.decode`), which keeps the host copy out of anything a trace
        captures.
        """
        if (
            x_flat.layout == ttnn.ROW_MAJOR_LAYOUT
            and x_flat.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        ):
            t = 1
        else:
            t = x_flat.shape[2]
        ids = self._select(token_in, t)
        return SparseRouting(scores=self._scores(x_flat), indices=ids)


# --------------------------------------------------------------------------- #
# fused_experts (the routed-expert path)
#
# ``ttnn.experimental.deepseek.moe.fused_experts`` runs the whole routed-expert FFN
# (gate_up + SwiGLU + down + routing-weighted accumulation) in a single device op,
# selecting the hit experts on device from the router's ranking row or its ids. The
# op is decode-native -- one token row, ``T == 1`` -- and prefill replays that per
# token. Its weights are DRAM ND-sharded, one shard per original 8x8 compute core,
# and its hidden size is exactly :func:`_fused_hidden` (64 * 2 * 32 == 4096 on the
# default 64-core grid). With 6 selected experts the op runs on a 12x8 (96-core)
# grid, 16 cores per expert.
# --------------------------------------------------------------------------- #
# The tile is a hardware invariant, unlike the core count and bank count, which are
# ``moe.fused_num_cores`` / ``moe.fused_dram_banks`` in the system profile.
_FUSED_TILE = 32
# Compute-grid geometry from ``fused_experts_program_factory``: 6 selected experts run on 12x8
# (16 cores each) when the per-rank ``I / 32`` divides by 16, else on 12x9 (18 gate_up cores each,
# down still on the bottom 2x8); anything else stays on the serial ``num_cores / 8`` x 8 grid (8x8
# at the 64-core default), including when the device cannot host the parallel grid.
_FUSED_PARALLEL_EXPERTS = 6
_FUSED_GRID_Y = 8
_FUSED_PARALLEL_GRID_X = _FUSED_PARALLEL_EXPERTS * 2
_FUSED_DOWN_CORES_PER_EXPERT = 2 * _FUSED_GRID_Y
# Run the shared expert inside ``fused_experts`` wherever the op takes its 6-expert path (the only
# path that fuses it). ``DEEPSEEK_V4_FUSE_SHARED_EXPERT=0`` keeps it as its own projections.
FUSE_SHARED_EXPERT = os.environ.get("DEEPSEEK_V4_FUSE_SHARED_EXPERT", "1") not in ("0", "", "false", "False")


def _fused_parallel_grid_y(intermediate: int) -> int:
    """Rows of the 6-expert grid for a per-rank ``intermediate``: 8, or 9 when its tiles do not
    split evenly over 16 cores. Mirrors ``parallel_grid_y`` in the program factory."""
    return _FUSED_GRID_Y if (intermediate // _FUSED_TILE) % _FUSED_DOWN_CORES_PER_EXPERT == 0 else _FUSED_GRID_Y + 1


def _fused_parallel(top_k: int, device, intermediate: int) -> bool:
    """Whether ``fused_experts`` takes its 6-expert path for ``top_k`` selected experts and a per-rank
    ``intermediate``: 6 experts, and a device grid that holds the 12-wide grid. Mirrors
    ``parallel_experts`` in the program factory."""
    grid = device.compute_with_storage_grid_size()
    return (
        top_k == _FUSED_PARALLEL_EXPERTS
        and grid.x >= _FUSED_PARALLEL_GRID_X
        and grid.y >= _fused_parallel_grid_y(intermediate)
    )


def _fused_hidden(num_cores: int) -> int:
    """The only hidden size (``D``) the op accepts: each of ``num_cores`` cores owns
    exactly 2 output tiles of the hidden row (4096 on the 64-core grid)."""
    return num_cores * 2 * _FUSED_TILE


def _fused_compute_core_range_set(top_k: int, device, num_cores: int, intermediate: int) -> ttnn.CoreRangeSet:
    """Cores ``fused_experts`` occupies for ``top_k`` selected experts and a per-rank
    ``intermediate``, as a ``CoreRangeSet``.

    On the 6-expert path the ``[(0,0) .. (11,7)]`` rectangle (one expert per 2x8 column pair), or
    ``[(0,0) .. (11,8)]`` when ``intermediate`` needs 18 gate_up cores per expert; the
    ``num_cores``-core serial grid ``[(0,0) .. (num_cores/8 - 1, 7)]`` otherwise. The replica
    ``all_gather_for_matmul`` multicasts has to cover every one of these: the op aliases
    ``cb_input`` over each core's shard and will not broadcast a row that is already supposed to
    be local.
    """
    parallel = _fused_parallel(top_k, device, intermediate)
    gy = _fused_parallel_grid_y(intermediate) if parallel else _FUSED_GRID_Y
    gx = _FUSED_PARALLEL_GRID_X if parallel else num_cores // _FUSED_GRID_Y
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})


def _swiglu_cols_per_core(intermediate: int, num_cores: Optional[int] = None) -> int:
    """SwiGLU output columns in each DRAM shard, i.e. the ``I`` dim spread over shards.

    One 32-column I-tile per shard at I == 2048, and still one tile when TP slices I below 2048,
    so gate_up stays DRAM-busy on every NoC port (rather than a few cores covering all of I while
    the rest idle). Mirrors ``swiglu_tiles_per_shard_for`` in the program factory. On the 6-expert
    path those shards are spread across the 16 cores of each expert group (4 shards/core at
    I=2048, 1 at I=512).
    """
    if num_cores is None:
        num_cores = active_system_config().moe.fused_num_cores
    i_tiles = intermediate // _FUSED_TILE
    return _FUSED_TILE * max(1, i_tiles // num_cores)


def _tp_cluster_axis(device: ttnn.MeshDevice) -> int:
    """Mesh axis that holds the TP group: 1 for a 1xN submesh, else 0."""
    shape = tuple(device.shape)
    if len(shape) == 2 and shape[1] > 1:
        return 1
    return 0


def _pack_gate_up_for_tp(gate_up: torch.Tensor, tp_size: int) -> torch.Tensor:
    """Reorder a host ``[2I, D]`` gate_up so rank ``r`` owns ``cat(gate[r], up[r])`` along dim 0.

    Naive sharding of ``[gate | up]`` would split mid-gate on later ranks. After this pack,
    ``ShardTensorToMesh(dim=0)`` gives each chip ``[2 * I/TP, D]``.
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
    """``[2I, D]`` host gate_up -> the block-interleaved, per-rank-major ``[D, 2I]`` weight.

    Each rank's ``[D, 2*I_local]`` is interleaved independently (the op only ever sees
    ``I_local`` -- see :func:`_interleave_gate_up`), then the ranks are concatenated on the column
    axis so ``ShardTensorToMesh(dim=1)`` yields exactly one rank's interleaved weight.
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
    """Sum the TP partials (row-parallel down-proj) back to a replicated ``[..., D]``.

    ``tensor`` is the DRAM-interleaved ``[1,1,T,D]`` partial from the fused expert op; the result
    is the same shape, replicated across the TP axis (``CCL_TOPOLOGY`` all-reduce, both links).
    """
    return ttnn.all_reduce(
        tensor,
        cluster_axis=_tp_cluster_axis(device),
        num_links=2,
        topology=CCL_TOPOLOGY,
    )


def _interleave_gate_up(w: torch.Tensor, block: int) -> torch.Tensor:
    """Permute a host ``[D, 2I]`` gate_up weight into per-core ``[gate_block | up_block]`` order.

    Each ``[D, 2*block]`` DRAM shard then holds a core's gate columns followed by its paired up
    columns, which is what ``fused_experts`` reads in a single NoC read. ``gate = w[:, :I]``,
    ``up = w[:, I:]``; output column ``c*2*block + h*block + t`` maps to ``w[:, h*I + c*block + t]``.
    """
    k, two_i = w.shape
    intermediate = two_i // 2
    blocks = intermediate // block
    return w.reshape(k, 2, blocks, block).permute(0, 2, 1, 3).reshape(k, two_i).contiguous()


def _fused_nd_dram_config(rows: int, cols: int, shard_width: int, dram_banks: int) -> ttnn.MemoryConfig:
    """DRAM ND-shard config with one ``[rows, shard_width]`` shard per compute core.

    Used for the ``fused_experts`` weights: gate_up ``[D, 2*swiglu_cols]`` per core, down
    ``[I, D/num_cores]``. The shards are distributed ``ROUND_ROBIN_1D`` over ``dram_banks``
    (shard ``p`` sits on bank ``p % dram_banks``), and the op fetches each core's slice by its
    row-major index, so the shard order has to be the core order -- which is what
    ``_interleave_gate_up`` records into the host weight.
    """
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
    """Load a ``fused_experts`` weight as a DRAM ND-sharded TILE tensor.

    ``tensor`` is the host (interleaved or transposed) weight -- ``[D, 2I]`` gate_up, ``[I, D]``
    down -- or ``None`` on a verified cache hit, and ``nd_config`` the ND-shard spec from
    :func:`_fused_nd_dram_config`. The tile cache cannot round-trip an ND-shard memory config (a
    cache *hit* reloads the tensor with its plain serialized spec), so the weight is cached in
    standard interleaved DRAM under its own cache entry and resharded to the ND-shard layout on
    device.
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


class FusedSharedExpert:
    """The shared expert as ``fused_experts`` runs it, next to the routed experts (6-expert path).

    Same math as :class:`DeepSeekV4MLP`, ``down(silu(gate(x)) * up(x))`` clamped by ``swiglu_limit``
    when ``clamp_swiglu`` is set (``inf`` otherwise), but the three projections become two DRAM
    ND-sharded weights the op reads in place: ``gate_up`` ``[D, 2I]`` with gate/up interleaved per
    I-tile (one ``[D, 64]`` shard per I-tile) and ``down`` ``[I, D]`` cut into ``[32, 64]`` shards.
    Under TP each rank holds an ``I/TP`` slice, like the routed experts, and the block's all-reduce
    sums the down partials.
    """

    def __init__(
        self,
        weights: dict,
        prefix: str,
        device: ttnn.MeshDevice,
        config,
        dram_banks: int,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        clamp_swiglu: bool = False,
        tp_size: int = 1,
    ):
        """``weights`` holds ``<prefix>.gate_proj`` / ``up_proj`` / ``down_proj`` ``.weight`` as
        ``[I, D]`` / ``[I, D]`` / ``[D, I]`` torch tensors or thunks (read only on a cache miss)."""
        cache = _as_cache(cache)
        hidden, inter = config.hidden_size, config.moe_intermediate_size
        if inter % (_FUSED_TILE * tp_size):
            raise ValueError(f"shared expert I={inter} does not split into whole tiles over TP{tp_size}")
        local_inter = inter // tp_size
        self.swiglu_limit = config.swiglu_limit if clamp_swiglu else float("inf")
        tp_tag = f".tp{tp_size}" if tp_size > 1 else ""
        gate = _memo(weights[f"{prefix}.gate_proj.weight"])
        up = _memo(weights[f"{prefix}.up_proj.weight"])
        down = _memo(weights[f"{prefix}.down_proj.weight"])
        gate_up_name, down_name = f"{prefix}.gate_up_fused{tp_tag}", f"{prefix}.down_fused{tp_tag}"
        gate_up_w = _materialize(
            lambda: _interleave_gate_up_tp(torch.cat([gate(), up()]), tp_size, _FUSED_TILE),
            cache.file(gate_up_name),
            weight_dtype,
        )
        down_w = _materialize(lambda: down().t().contiguous(), cache.file(down_name), weight_dtype)
        self.gate_up = _load_fused_weight(
            gate_up_w,
            device,
            _fused_nd_dram_config(hidden, 2 * local_inter, 2 * _FUSED_TILE, dram_banks),
            cache_file_name=cache.file(gate_up_name),
            dtype=weight_dtype,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1) if tp_size > 1 else None,
        )
        self.down = _load_fused_weight(
            down_w,
            device,
            _fused_nd_dram_config(_FUSED_TILE, hidden, 2 * _FUSED_TILE, dram_banks),
            cache_file_name=cache.file(down_name),
            dtype=weight_dtype,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0) if tp_size > 1 else None,
        )


class DeepSeekV4PreloadedExperts(DeepSeekV4Module):
    """Routed-experts compute via the single-op ``fused_experts`` kernel.

    The whole routed-expert FFN for one token (gate_up + SwiGLU + down + routing-weighted
    accumulation) runs in a single ``fused_experts`` device op, which picks the hit experts on
    device from the router's row or ids. The op is hard-wired to the real V4-Flash sizes: down
    weights are sharded 64 ways along the hidden dim (2 output tiles each), so ``D`` must be
    exactly :func:`_fused_hidden` (``64 * 2 * 32 == 4096``), and gate_up is one I-tile per DRAM
    shard, so a TP-sliced ``I`` (e.g. 512) yields fewer, still-full 16-core groups rather than
    idle cores. With 6 selected experts compute uses a 12x8 grid (16 cores per expert).

    The op is natively single-token (``T == 1``), so **prefill is computed by decode**: the model
    replays this path once per token, and :meth:`forward` asserts the shard height is 1.

    Every expert is kept resident on device as DRAM ND-sharded weights (one shard per compute
    core) in low precision (``BFloat4_b`` by default; ~3.5 GB for the 256 experts, a natural match
    for the MXFP4 checkpoint). :meth:`__init__` pulls each expert's dequantized weights from the
    host ``provider`` once, permutes the gate_up into the op's interleaved per-core layout and
    uploads the ND-sharded tensors; :meth:`forward` then runs purely on device, with no per-step
    host transfer at all.

    ``provider(expert_idx) -> (gate_up [2I, D], down [D, I])`` returns host torch tensors (the HF
    packed layout: ``gate_up`` is ``cat([w_gate, w_up])``). Experts with zero total routing weight
    are skipped by the op (matching the reference's ``hit`` set), so only the experts some token
    actually selected are fetched and computed.
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
        """Upload every expert once, as the op's DRAM ND-sharded weights.

        ``provider(e)`` is called only where the cache misses; a hit skips it (and its dequant)
        entirely. Gate_up lands as ``[D, 2*swiglu_cols]``-per-core interleaved shards, down as
        ``[I, D/num_cores]`` per core. Under TP the gate_up shards N (so the op runs the rank's
        ``I/TP`` slice) and down shards K, whose partials
        :class:`DeepSeekV4SparseMoeBlock` all-reduces. ``system_config`` supplies the op geometry
        (``moe.fused_num_cores`` / ``fused_dram_banks``), the L1 expert block size and
        ``routing_eps``; ``dtype`` defaults to the profile's weight dtype, and an explicit one
        wins. Raises if the config is not the fused layout this op is hard-wired to.
        """
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
        self._core_grid = _fused_compute_core_range_set(self.top_k, device, num_cores, self.intermediate)
        # The 6-expert path is the only one that can fuse the shared expert into the op.
        self.parallel = _fused_parallel(self.top_k, device, self.intermediate)
        self.hidden = config.hidden_size
        self.limit = config.swiglu_limit
        # Applied inside the op on the sparse path, where the weights are derived here
        # rather than handed over pre-normalized.
        self.routed_scaling_factor = config.routed_scaling_factor
        cache = _as_cache(cache)

        # ``fused_experts`` is hard-wired to the real V4-Flash sizes: ``D == 4096``
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

        # Upload every expert once as the op's DRAM ND-sharded weights (gate_up interleaved
        # per core, down ND-sharded) in low precision. With caching enabled and a hit, the
        # provider (and its expensive dequant) is skipped entirely; the ND-shard layout can't
        # round-trip the tile cache, so the interleaved weight is cached in standard DRAM and
        # resharded on device (see :func:`_load_fused_weight`).
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
            # The provider gives gate_up [2I, D] / down [D, I]; transpose to the matmul-ready
            # [D, 2I] / [I, D] (memoized, so each source is materialized at most once).
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

    def core_grid(self) -> ttnn.CoreRangeSet:
        """Compute cores the fused_experts op occupies for this layer's ``top_k``, as the
        ``[(0,0) .. (11,7)]`` rectangle at ``top_k == 6`` (``(11,8)`` when the per-rank ``I`` needs
        18 gate_up cores per expert; else ``[(0,0) .. (7,7)]``).

        ``all_gather_for_matmul`` has to multicast onto (at least) this set so every
        compute core already holds the activation replica.
        """
        return self._core_grid

    @property
    def gate_up_weights(self) -> list[ttnn.Tensor]:
        """The per-expert ND-sharded gate_up weights (read in place by the prefill routed-expert op)."""
        return self._gate_up_fused

    @property
    def down_weights(self) -> list[ttnn.Tensor]:
        """The per-expert ND-sharded down weights (read in place by the prefill routed-expert op)."""
        return self._down_fused

    def _run_fused(
        self, x_tok: ttnn.Tensor, routing: SparseRouting, shared: Optional[FusedSharedExpert] = None
    ) -> ttnn.Tensor:
        """Run ``fused_experts`` for one token row; returns ``[1,1,1,D]``.

        ``shared`` (6-expert path only) is the always-active shared expert, which the op then adds
        to the output itself.

        ``x_tok`` is the single-token activation the op accepts -- ROW_MAJOR interleaved
        ``[1,1,1,D]``, or the ROW_MAJOR HEIGHT_SHARDED L1 replica (``[1,1,cores,D]`` with shard
        ``[1,D]``) that ``all_gather_for_matmul`` produces and ``matmul_decode`` consumes in place
        -- and ``routing`` is that token's decision, with ``scores`` ``[1,1,1,E]`` and either
        ``ranking`` ``[1,1,1,E]`` or ``indices`` ``[1,1,1,k]``. Each routing tensor is then moved
        to DRAM interleaved, leaving its layout untouched.

        ``num_experts`` is always ``top_k``: one token row selects at most that many distinct
        experts, so the op's program -- and any trace holding it -- is the same every step.

        Both routing tensors go in exactly as the router produced them. A ``ranking`` row (the
        learned router) makes the op find the top-k itself; ``indices`` (the hash router's table
        lookup) makes it read those instead. Either way the op applies the normalize-and-scale tail
        itself, and ``routing_eps`` (``moe.routing_eps``) is the epsilon added to each token's
        score sum before dividing, which guards the renormalize against an all-zero score row.

        ``experts_block_size`` (``moe.experts_block_size``) is how many experts' SwiGLU
        activations are resident at once. It sizes the op's dominant per-core CB, so it is the knob
        to turn when the op's static CBs collide with the L1 buffers live at the call; the cost is
        one extra chip-wide gather/broadcast barrier per block.
        """
        scores = ttnn.to_memory_config(routing.scores, ttnn.DRAM_MEMORY_CONFIG)
        indices = None
        ranking = None
        if routing.ranking is not None:
            ranking = ttnn.to_memory_config(routing.ranking, ttnn.DRAM_MEMORY_CONFIG)
        else:
            indices = ttnn.to_memory_config(routing.indices, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.experimental.deepseek.moe.fused_experts(
            x_tok,
            routing_scores=scores,
            routing_indices=indices,
            ranking_scores=ranking,
            gate_up_weights=self._gate_up_fused,
            down_weights=self._down_fused,
            num_experts=self.top_k,
            intermediate_size=self.intermediate,
            swiglu_limit=self.limit,
            top_k=self.top_k,
            routed_scaling_factor=self.routed_scaling_factor,
            routing_eps=self.routing_eps,
            experts_block_size=self.experts_block_size,
            shared_gate_up_weights=None if shared is None else shared.gate_up,
            shared_down_weights=None if shared is None else shared.down,
            shared_swiglu_limit=None if shared is None else shared.swiglu_limit,
        )  # [1, 1, D]
        return ttnn.reshape(out, [1, 1, 1, self.hidden])

    def forward(
        self, x_flat: ttnn.Tensor, routing: SparseRouting, shared: Optional[FusedSharedExpert] = None
    ) -> ttnn.Tensor:
        """``x_flat`` ``[1,1,T,D]`` plus the token's routing decision -> ``[1,1,1,D]``, plus the
        ``shared`` expert when given.

        Trace-safe: nothing is read back to host. ``T`` is the activation's *shard height*, not the
        token axis of the gathered tensor: the decode all-gather replicates the token row onto
        every core of the destination grid, so ``T`` is 1 while ``x_flat.shape[2]`` is that grid's
        core count. ``T`` is fixed at capture time, so a captured trace stays a flat op sequence
        and prefill goes through the same path. (The assert is the op's own requirement: its
        ROW_MAJOR input is one token row, so a batched MoE is not implemented.)
        """
        mem = x_flat.memory_config()
        if (
            x_flat.is_sharded()
            and mem.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            and mem.shard_spec is not None
        ):
            t = mem.shard_spec.shape[0]
        else:
            t = x_flat.shape[2]
        _profile(self.device)
        assert t == 1, "t must be 1"
        out = self._run_fused(x_flat, routing, shared)
        return out

    # The routed FFN has no host readback and no host-initialised operands, so the traced
    # decode path is just :meth:`forward`.
    decode_static = forward


class DeepSeekV4SparseMoeBlock(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4SparseMoeBlock`` (standard ``moe`` layer).

    ``routed = experts(router(x)) ; return routed + shared_experts(x)``.

    ``router_cls`` / ``shared_expert_cls`` are the learned router and shared expert this block
    builds; subclasses for other checkpoints swap them.
    """

    router_cls = DeepSeekV4TopKRouter
    shared_expert_cls = DeepSeekV4MLP

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
        tp_size: int = 1,
        matmul_decode: Optional[bool] = None,
    ):
        """``gate`` may be injected (a :class:`DeepSeekV4HashRouter` for the first
        ``num_hash_layers`` MoE layers); otherwise this builds the learned
        :class:`DeepSeekV4TopKRouter` from ``weights``. ``experts`` is always injected -- the
        routed compute is a :class:`DeepSeekV4PreloadedExperts` holding every expert (256 here)
        resident on device in BFloat4_b, so it has nothing of its own to prefetch. The shared
        expert is built from ``weights["shared_experts.*"]``. ``matmul_decode`` is passed to the
        router and the shared expert (see :class:`DeepSeekV4MLP`).
        """
        self.device = device
        self.hidden = config.hidden_size
        self.tp_size = tp_size
        cache = _as_cache(cache)
        self.gate = (
            gate
            if gate is not None
            else self.router_cls(
                config,
                weights,
                device,
                cache=cache,
                use_prefetcher=use_prefetcher,
                prefetch_buffers=prefetch_buffers,
                weight_dtype=weight_dtype,
                matmul_decode=matmul_decode,
            )
        )
        self.is_hash = isinstance(self.gate, DeepSeekV4HashRouter)
        self.experts = experts
        # On the 6-expert path ``fused_experts`` also runs the shared expert (unless
        # DEEPSEEK_V4_FUSE_SHARED_EXPERT=0), so there is no separate shared MLP to build.
        self.fused_shared = None
        self.shared_experts = None
        if FUSE_SHARED_EXPERT and experts.parallel:
            self.fused_shared = FusedSharedExpert(
                weights,
                "shared_experts",
                device,
                config,
                dram_banks=experts.system_config.moe.fused_dram_banks,
                cache=cache,
                weight_dtype=weight_dtype,
                clamp_swiglu=self.shared_expert_cls.clamp_swiglu,
                tp_size=tp_size,
            )
        else:
            self.shared_experts = self.shared_expert_cls(
                weights,
                "shared_experts",
                device,
                cache=cache,
                weight_dtype=weight_dtype,
                config=config,
                use_prefetcher=use_prefetcher,
                prefetch_buffers=prefetch_buffers,
                tp_size=tp_size,
                matmul_decode=matmul_decode,
            )

    def prefetch_weights(self):
        """Stage the ``[D, E]`` router gate, then the shared expert's ``[D, I]`` gate/up and
        ``[I, D]`` down, ahead of the decode that uses them.

        The gate goes first: it queues on its private 8-receiver ring (:data:`ROUTER_GATE_GCB`).
        Shared-expert gate/up then queue on q_a's 32-receiver FIFO (after q_a / CSA) and down on
        the shared decode GCB after o_b. Queue order on each ring has to match consume order.
        """
        self.gate.prefetch_weights()
        if self.shared_experts is not None:
            self.shared_experts.prefetch_weights()

    def decode_static(
        self, hidden: ttnn.Tensor, hash_token: ttnn.Tensor | None = None, image_mask: ttnn.Tensor | None = None
    ) -> ttnn.Tensor:
        """Trace-safe MoE. ``hidden`` ``[B, 1, 1, D]`` -> the same shape.

        The one entry point for this block (decode *and* prefill): the model prefill replays the
        decoder layer's decode path one token at a time, so ``B`` is the step's batch and the
        routed path gets a single token row (``T == 1``; see
        :meth:`DeepSeekV4PreloadedExperts.forward`).

        ``hidden`` is flattened to ``[1, 1, B, D]`` -- the token-per-row layout the router and the
        shared expert work in -- then the activation is multicast onto the fused-expert compute
        cores (12x8 at ``top_k == 6``) as a ROW_MAJOR HEIGHT_SHARDED replica, which
        ``fused_experts`` and the shared expert's :class:`LinearDecode` projections both consume in
        place (its B grid is inside that rectangle). The shared expert therefore sees one wider
        matmul while the routed experts run one op on the token row.

        Routing stays entirely on device: the learned top-k router is already host-sync-free, and
        hash layers gather their selected expert ids on device from the persistent ``hash_token``
        ``[1,B]`` device token ids (see :meth:`DeepSeekV4HashRouter.forward_static`). The routed
        FFN runs through the no-host-readback fused-experts path, and under TP the row-parallel
        down partials are all-reduced back to a replicated ``[1,1,B,D]`` before the reshape back.

        ``image_mask`` (``[1, B]`` uint32, 1 for an image token) goes to the learned router's
        :meth:`DeepSeekV4TopKRouter.ranking_bias`.
        """
        b, h = hidden.shape[0], hidden.shape[-1]
        x_flat = ttnn.reshape(hidden, [1, 1, b, h])
        dest = self.experts.core_grid()
        mem = x_flat.memory_config()
        already_replica = (
            x_flat.is_sharded()
            and mem.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            and mem.shard_spec is not None
            and _core_grid_contains(mem.shard_spec.grid, dest)
        )
        if not already_replica:
            # ``all_gather_for_matmul`` only accepts L1 WIDTH/HEIGHT_SHARDED. Decode RMSNorm
            # already leaves that; an interleaved tensor is width-sharded over one tile row
            # first, and h must be tile-aligned for that.
            if not x_flat.is_sharded():
                x_flat = ttnn.to_memory_config(x_flat, width_sharded_l1_config(1, h, self.device))
                x_flat = ttnn.experimental.deepseek.all_gather_for_matmul(x_flat, dest)
            else:
                x_flat = ttnn.experimental.deepseek.all_gather_for_matmul(x_flat, dest)
        # The learned router has no separate trace-safe variant: it allocates every operand
        # it uses, so :meth:`DeepSeekV4TopKRouter.forward` is already capture-safe.
        if self.is_hash:
            routing = self.gate.forward_static(x_flat, hash_token)
        else:
            routing = self.gate(x_flat, image_mask)
        if self.fused_shared is not None:
            combined = self.experts.decode_static(x_flat, routing, self.fused_shared)  # [1, 1, 1, D]
        else:
            routed = self.experts.decode_static(x_flat, routing)  # [1, 1, 1, D]
            shared = self.shared_experts(x_flat)  # [1, 1, B, D]
            combined = ttnn.add(routed, shared)
        combined = ttnn.to_memory_config(combined, ttnn.DRAM_MEMORY_CONFIG)
        if self.tp_size > 1:
            combined = _tp_all_reduce(combined, self.device)
        return ttnn.reshape(combined, [b, 1, 1, h])
