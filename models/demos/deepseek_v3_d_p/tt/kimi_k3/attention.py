# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MLA and KDA behind one interface, so a Kimi-K3 block never asks which it has.

The hybrid schedule is the whole reason this file exists. 24 of K3's 93 layers run full attention
and write a KV slab; the other 69 run a linear recurrence and write none. A block that branched on
that would branch in five places — building the module, calling it, the KV pad-zero, the migration
ack, and the layer-index it reports to taps. Instead the block holds one collaborator that answers
`writes_kv` and `layer_idx`, and the branch happens once, in `build_attention`.

The KDA side owns one layout bridge, and it is worth being precise about its cost because it looks
like an addition and is not. `ttKDA._validate_forward` requires `[B, T_local, 7168]` — three
dimensions, and the **full** hidden dim on every chip, because `input_projection` is column-parallel
(`tt/kda/weights.py`, `shard_dim=-1`). So the block's TP-sharded `[1, 1, T, 7168/tp]` is
all-gathered on the TP axis on the way in, and `ttKDA._project_output` closes the pair with
`reduce_scatter_minimal_async` on the same axis on the way out. That is exactly the collective pair
`TtPrefillBlock._dense_ffn_path` already issues for every dense layer — same axis, same `num_links`,
same 640 x 7168 bf16 = 9.2 MB per chip per 5120-token chunk. KDA reuses the model's existing
collective rather than adding one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import ttnn


@dataclass(frozen=True)
class K3AttnContext:
    """Everything an attention module needs from the caller for one chunk.

    One object rather than eleven keyword arguments, because MLA reads most of it and KDA reads
    none of it — a KDA layer carries its state on the module, not in the call.
    """

    rope_tensors: Optional[dict] = None
    kvpe_cache: object = None
    # RANK-LOCAL MLA slot, from `KimiK3LayerSchedule.kv_slot`. `None` on a KDA layer, which has no
    # slot at all; see layer_schedule.py for why this is not the global one.
    cache_layer_idx: Optional[int] = None
    cache_user_id: int = 0
    actual_start: Optional[int] = None
    # Absolute KV position past this chunk's last real token. MLA clamps its cache write to it
    # (#54744); the traced path takes the same value off `metadata[2]` on device instead.
    actual_end: Optional[int] = None
    # The traced path's `(slot_id, actual_start, actual_end)` triple of 1-element uint32 tensors.
    metadata: Optional[tuple] = None


class K3Attention(Protocol):
    """What a Kimi-K3 block needs from its attention, and nothing more."""

    layer_idx: int
    """GLOBAL model layer index — what names weights and what taps report."""

    writes_kv: bool
    """Whether this layer fills a KV slab, and so whether the block owes a pad-zero and an ack."""

    def forward(self, normed: ttnn.Tensor, ctx: K3AttnContext) -> ttnn.Tensor:
        """`[1, 1, T_local, emb/tp]` in, the same out. Ownership of the return passes to the caller."""
        ...


class TtK3MlaAttention:
    """`ttMLA` under the protocol.

    A thin adapter and deliberately nothing more: K3's MLA differences — NoPE, the output gate, 96
    heads — are all read off the `config` inside `ttMLA.__init__`, so there is no K3-specific
    attention code to write. What K3 does change is on the *caller's* side: `layer_num` is the
    rank's MLA count rather than its layer count, and `cache_layer_idx` is a rank-local slot from
    the schedule rather than an enumerate index.
    """

    writes_kv = True

    def __init__(self, mla):
        self._mla = mla

    @property
    def layer_idx(self) -> int:
        return self._mla.layer_idx

    @property
    def sp_factor(self) -> int:
        return self._mla.sp_factor

    @property
    def sp_axis(self) -> int:
        return self._mla.sp_axis

    @property
    def mla(self):
        """The wrapped module, for the KV pad-zero and ack the block still owes on this layer."""
        return self._mla

    def forward(self, normed: ttnn.Tensor, ctx: K3AttnContext) -> ttnn.Tensor:
        """K3 is dense, so the indexer arguments are absent and the return is the bare tensor.

        `resolve_has_indexer` reports False for K3's config (no `has_indexer`, no `index_*` fields),
        which is also what lets the trace guard in `TtPrefillTransformer.set_trace_controller` pass.
        """
        return self._mla.forward(
            normed,
            ctx.rope_tensors,
            ctx.kvpe_cache,
            cache_layer_idx=ctx.cache_layer_idx,
            actual_start=ctx.actual_start,
            # #54744 clamps the KV write to the real tokens. The shared block threads this through;
            # without it here Kimi-K3 silently writes past actual_end into the pad window.
            actual_end=ctx.actual_end,
            cache_user_id=ctx.cache_user_id,
            metadata=ctx.metadata,
        )


class TtK3KdaAttention:
    """`ttKDA` under the protocol, with its carry.

    The carry lives in a `KdaStateCache` keyed by global layer index, not on the layer: `ttKDA` is
    purely functional and retains nothing, and the cache is what keeps one address per carry so a
    captured trace can advance the recurrence with no host in the loop. `commit` is a device copy
    and belongs inside the captured region, which is why it happens here in `forward` rather than
    being left to the caller.
    """

    writes_kv = False

    def __init__(self, kda, layer_idx: int, tp_axis: int, num_links: int, tp_topology, state_cache=None):
        self.kda = kda
        self.layer_idx = layer_idx
        self._states = state_cache
        self._tp_axis = tp_axis
        self._num_links = num_links
        self._tp_topology = tp_topology

    def bind_state_cache(self, state_cache) -> None:
        """Give this layer its carry.

        Deferred on purpose: `KdaStateCache` allocates from the `ttKDA` instances, so the cache
        cannot exist until the layers do, and the layers must not each allocate their own — the
        whole point of the cache is one address per carry for the life of the run. So construction
        is two-phase, and this is the seam rather than an attribute someone reaches into.
        """
        self._states = state_cache

    def forward(self, normed: ttnn.Tensor, ctx: K3AttnContext) -> ttnn.Tensor:
        gathered = ttnn.all_gather(
            normed,
            dim=-1,
            cluster_axis=self._tp_axis,
            num_links=self._num_links,
            topology=self._tp_topology,
        )
        # `[1, 1, T, d]` -> `[1, T, d]`: ttKDA takes three dimensions, and the leading one is batch.
        hidden = ttnn.squeeze(gathered, dim=0)
        ttnn.deallocate(gathered)

        if self._states is None:
            raise ValueError(
                f"KDA layer {self.layer_idx} has no state cache; call bind_state_cache() before the first forward"
            )
        output, new_state = self.kda.forward(hidden, self._states.read(self.layer_idx, ctx.cache_user_id))
        ttnn.deallocate(hidden)
        self._states.commit(self.layer_idx, new_state, ctx.cache_user_id)

        # `_project_output` reduce-scatters on the TP axis and returns `[B, T, d/tp]`, so the only
        # thing left is the leading axis the block's residual expects.
        return ttnn.unsqueeze(output, dim=0)


def build_attention(
    mesh_device,
    config,
    model_cfg: type,
    state_dict: dict,
    layer_idx: int,
    schedule,
    *,
    seq_len: int,
    state_cache=None,
    sp_axis: int = 0,
    tp_axis: int = 1,
    num_links: int = 1,
    topology=None,
    weight_cache_path=None,
    max_seq_len: Optional[int] = None,
    is_chunked: bool = False,
    slot_num: int = 1,
    kv_only: bool = False,
    is_balanced: bool = False,
    first_layer_idx: int = 0,
) -> K3Attention:
    """The one place the hybrid schedule turns into a module.

    Everything below this call is polymorphic; everything above it knows the schedule. `topology` is
    the per-axis pair the model opened — `per_axis_topology()` — and is passed to MLA whole (its SDPA
    runs on the SP axis) but to KDA as its TP element only, since KDA's own CCL is TP-side.
    """
    from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
    from models.demos.deepseek_v3_d_p.tt.kda.config import kimi_k3_program_config
    from models.demos.deepseek_v3_d_p.tt.kda.kda import ttKDA
    from models.demos.deepseek_v3_d_p.tt.mla.mla import ttMLA
    from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl, per_axis_topology

    topology = topology if topology is not None else per_axis_topology()
    tp_topology = topology[tp_axis] if isinstance(topology, (tuple, list)) else topology

    if schedule.is_mla(layer_idx):
        return TtK3MlaAttention(
            ttMLA(
                config,
                state_dict.get("mla_weights", {}),
                mesh_device,
                layer_idx=layer_idx,
                seq_len=max_seq_len if max_seq_len is not None else seq_len,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                topology=topology,
                is_balanced=is_balanced,
                weight_cache_path=weight_cache_path,
                is_chunked=is_chunked,
                active_seq_len=seq_len,
                slot_num=slot_num,
                # The rank's MLA count, not its layer count: the KV cache holds one slot per
                # full-attention layer. See layer_schedule.py.
                layer_num=schedule.num_mla_layers,
                kv_only=kv_only,
                first_layer_idx=first_layer_idx,
            )
        )

    # `state_cache` may be None here: the cache allocates from the ttKDA instances, so it is built
    # once every layer exists and bound with `bind_state_cache`. `forward` raises if that never
    # happened, rather than silently starting each chunk from a fresh zero state.
    return TtK3KdaAttention(
        ttKDA(
            mesh_device,
            kimi_k3_kda_config(),
            state_dict.get("kda_weights"),
            layer_idx=layer_idx,
            weight_cache_path=weight_cache_path,
            tt_ccl=get_tt_ccl(mesh_device),
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            program_config=kimi_k3_program_config(tp_ccl_topology=tp_topology),
        ),
        layer_idx=layer_idx,
        tp_axis=tp_axis,
        num_links=num_links,
        tp_topology=tp_topology,
        state_cache=state_cache,
    )
