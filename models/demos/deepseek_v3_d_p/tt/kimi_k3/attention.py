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
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaStates, deallocate_state, device_scalar


def validate_kda_bounds(actual_start: int | None, actual_end: int | None) -> None:
    """Validate host bounds; device-only trace metadata has the same caller contract.

    The start is 32-token aligned; the end is a prompt's real length and need not be.
    """
    start = 0 if actual_start is None else actual_start
    if start < 0 or start % ttnn.TILE_SIZE != 0:
        raise ValueError(f"K3 KDA actual_start must be nonnegative and 32-token aligned, got {start}")
    if actual_end is not None and actual_end <= start:
        raise ValueError("K3 KDA requires actual_end > actual_start")


def build_kda_scalars(
    device, metadata: Optional[tuple], cache_user_id: int, actual_start: Optional[int], actual_end: Optional[int]
) -> tuple[tuple, tuple]:
    """The `(slot, actual_start, actual_end)` device scalars every KDA layer of one chunk reads, and the ones to free.

    A traced chunk carries them in `metadata`, borrowed as they are: replay rewrites those tensors in place
    and nothing is freed. An eager chunk uploads its host ints once, after validating them, and the second
    element lists what the caller frees after its last KDA layer. An absent end means full capacity and
    uploads nothing. Built once per chunk, not per layer, so a rank's KDA layers share the uploads.
    """
    if metadata is not None:
        return tuple(metadata[0:3]), ()
    validate_kda_bounds(actual_start, actual_end)
    slot = device_scalar(device, cache_user_id)
    start = device_scalar(device, 0 if actual_start is None else actual_start)
    end = None if actual_end is None else device_scalar(device, actual_end)
    owned = (slot, start) if end is None else (slot, start, end)
    return (slot, start, end), owned


@dataclass(frozen=True)
class K3AttnContext:
    """Everything an attention module needs from the caller for one chunk.

    MLA consumes the KV and rotary inputs, the host ints when eager and `metadata` when traced.
    KDA reads only `kda_scalars` and `kda_states`: the slot selects its carry in the slabs on
    device, the bounds order its sequence-parallel segments and stop it at the last real token.
    """

    rope_tensors: Optional[dict] = None
    kvpe_cache: object = None
    # RANK-LOCAL MLA slot, from `KimiK3LayerSchedule.kv_slot`. `None` on a KDA layer, which has no
    # slot at all; see layer_schedule.py for why this is not the global one.
    cache_layer_idx: Optional[int] = None
    cache_user_id: int = 0
    actual_start: Optional[int] = None
    # Absolute position past this chunk's last real token. MLA clamps its cache write to it
    # (#54744), and KDA stops its carries there; trace reads `metadata[2]` on device.
    actual_end: Optional[int] = None
    # The traced path's `(slot_id, actual_start, actual_end)` triple of 1-element uint32 tensors.
    # KDA requires a 32-token aligned start on every replay; the end may be unaligned. Device-only
    # values are caller preconditions; do not round real lengths or read them back.
    metadata: Optional[tuple] = None
    # What the KDA layers read: the `(slot, actual_start, actual_end)` device scalars from
    # `build_kda_scalars`, the same objects for every KDA layer of the chunk. `actual_end` may be
    # `None` (full capacity). The caller owns them; a KDA layer never uploads or frees a scalar.
    kda_scalars: Optional[tuple] = None
    # The engine-owned `KdaStates` slabs, the one store of every KDA carry: a KDA layer loads its
    # slot's carry from them and stores the replacement back, inside the captured region. The
    # runtime hands them to the model at `compile()`; a test that builds the model allocates its own.
    kda_states: Optional[KdaStates] = None


class K3Attention(Protocol):
    """What a Kimi-K3 block needs from its attention, and nothing more."""

    layer_idx: int
    """GLOBAL model layer index — what names weights and what taps report."""

    writes_kv: bool
    """Whether this layer fills a KV slab, and so whether the block owes a pad-zero."""

    def forward(self, normed: ttnn.Tensor, ctx: K3AttnContext) -> ttnn.Tensor:
        """`[1, 1, T_local, emb/tp]` in, the same out. Ownership of the return passes to the caller."""
        ...


class TtK3MlaAttention(LightweightModule):
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

    def set_tp_shard_kv(self, tp_shard_kv: Optional[bool]):
        self._mla.set_tp_shard_kv(tp_shard_kv)

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


class TtK3KdaAttention(LightweightModule):
    """`ttKDA` under the protocol, reading and writing its carry in the engine's `KdaStates` slabs.

    `ttKDA` is purely functional and retains nothing, so the carry has to live somewhere with one
    address for the life of the run, or a captured trace could not advance the recurrence with no
    host in the loop. That somewhere is the migration slab itself, the way MLA's KV cache is its own
    migration format: per chunk the layer loads its slot's carry into the model's one scratch carry,
    runs, and stores the replacement back, every copy a device program that reads the slot on
    device and belongs inside the captured region. That is why load and store happen here in
    `forward` rather than being left to the caller.
    """

    writes_kv = False

    def __init__(self, kda, layer_idx: int, tp_axis: int, num_links: int, tp_topology, scratch: KdaState):
        self.kda = kda
        self.layer_idx = layer_idx
        # BORROWED from the model and shared by its KDA layers: the only state `ttKDA.forward` reads.
        self._scratch = scratch
        self._tp_axis = tp_axis
        self._num_links = num_links
        self._tp_topology = tp_topology

    def forward(self, normed: ttnn.Tensor, ctx: K3AttnContext) -> ttnn.Tensor:
        if ctx.kda_scalars is None:
            raise ValueError(
                f"KDA layer {self.layer_idx} has no kda_scalars; the caller builds them once per chunk "
                "with build_kda_scalars"
            )
        if ctx.kda_states is None:
            raise ValueError(
                f"KDA layer {self.layer_idx} has no KDA state slabs; the runtime hands kv_caches.kda_states "
                "to the model at compile()"
            )
        # The slot is a device tensor on every path, so the carry a capture touches is chosen when it
        # replays, not when it is recorded; the bounds likewise change per replay without a host read.
        slot, actual_start, actual_end = ctx.kda_scalars
        gathered = ttnn.all_gather(
            normed,
            dim=-1,
            cluster_axis=self._tp_axis,
            num_links=self._num_links,
            topology=self._tp_topology,
        )
        # `[1, 1, T, d]` -> `[1, T, d]`: ttKDA takes three dimensions, and the leading one is batch.
        # The squeeze is a view over `gathered`'s buffer, so `gathered` must stay allocated until the
        # forward is done; the `ttnn.deallocate(hidden)` below frees it.
        hidden = ttnn.squeeze(gathered, dim=0)
        ctx.kda_states.import_layer(self._scratch, slot, self.layer_idx)
        output, new_state = self.kda.forward(hidden, self._scratch, actual_start=actual_start, actual_end=actual_end)
        ttnn.deallocate(hidden)
        ctx.kda_states.export_layer(new_state, slot, self.layer_idx)
        deallocate_state(new_state)

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
    kda_scratch: Optional[KdaState] = None,
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
    tp_shard_kv: Optional[bool] = None,
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
                tp_shard_kv=tp_shard_kv,
            )
        )

    if kda_scratch is None:
        raise ValueError(f"layer {layer_idx} is KDA and needs the model's scratch carry (kda_scratch)")
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
            program_config=kimi_k3_program_config(
                active_seq_len_local=seq_len // tuple(mesh_device.shape)[sp_axis], tp_ccl_topology=tp_topology
            ),
            active_seq_len=seq_len,
        ),
        layer_idx=layer_idx,
        tp_axis=tp_axis,
        num_links=num_links,
        tp_topology=tp_topology,
        scratch=kda_scratch,
    )
