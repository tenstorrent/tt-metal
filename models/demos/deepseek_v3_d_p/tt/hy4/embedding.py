# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview prefill token embedding, vocab-sharded over all 32 chips (#65892), on decode's table.

The table and the lookup are decode's ``TtHy4Embedding`` (models/experimental/hy4_preview/tt/embedding.py): chip
``c = 4 * row + col`` (row-major over the 8x4 mesh, ``VocabShards``) holds token ids ``[3776 c, 3776 (c + 1))`` as a
``[3778, 6144]`` BF16 table with a zero row on each side, and a token id ``t`` reads row
``clamp(t - 3776 c + 1, 0, 3777)``: its own row on the chip that owns it, a zero row on the other 31. The device
tensor is the same shape, dtype, layout and chip placement as decode's, so one copy can serve both engines
(``TtHy4PrefillEmbedding.sharing``); prefill only changes how the rows reach the residual layout.

Decode wants ``[U, 6144 / 4]`` on every chip and all-reduces. Prefill wants the block residual, ``[S / 8, 6144 / 4]``
per chip: the sequence over the SP axis (axis 0, 8 chips) and the hidden over the TP axis (axis 1, 4 chips). Per
chunk, on chip (r, c), with ``L = S / 8`` ids arriving per chip (SP-sharded, the same on the 4 chips of a row):

    ids        [1, 1, 1, L]        uint32, this row's slice of the chunk
    all-gather over SP (ids only)  -> [1, 1, 1, S]  the whole chunk, the same on all 32 chips (S x 4 B)
    table rows + lookup            -> [1, 1, S, 6144] BF16: the rows of the tokens chip c owns, zeros elsewhere
    reduce-scatter over SP, seq    -> [1, 1, L, 6144]  row r's slice of the sequence, summed over the 8 chips
                                                        of column c
    reduce-scatter over TP, hidden -> [1, 1, L, 1536]  col c's slice of the hidden, summed over the 4 columns

Every sum adds one real row to 31 zero rows (each token has exactly one owner among the 32 chips: 8 summed in the
first scatter, 4 in the second), so the result is the checkpoint's BF16 row bit for bit. Gathering the ids and
scattering the rows over the same SP axis in the same chip order hands every chip back exactly the positions it
sent, whatever order the runner laid the chunk out in (balanced / rotated chunks included).

Cost per 4096-token chunk, per chip: a 16 KB id gather, a 50 MB lookup result, and two reduce-scatters that send
7/8 of 50 MB and 3/4 of 6.3 MB (~49 MB). A table split 8 ways by vocab over SP and 4 ways by hidden over TP would hold
the same 1/32 per chip and need only the SP reduce-scatter of ``[S, 1536]`` (7/8 of 12.6 MB, 4x less), but its shards
are not decode's: the engines would each hold a copy (2 x 46 MB per chip). The collectives run once per chunk against
the ~10 per layer of 78 layers, so sharing the table wins; measured in tests/test_hy4_prefill_embedding.py.

Token ids outside ``[0, vocab)`` (``MTP_PAD_TOKEN_ID``) embed as zeros: the clamp lands them on a zero row on every
chip. ``TtParallelEmbedding`` clamps them to the last vocab row instead; either way they are padding.

Collectives run on the prefill's persistent ``TT_CCL`` semaphores (``all_gather_async``,
``reduce_scatter_minimal_async``), so they take nothing from L1_SMALL (``HY4_L1_SMALL_SIZE``).

Weight cache: one tensorbin, ``embed_vocab{chips}_{rows}x{cols}``, with a ``LayoutCacheGuard`` stamp
(``utils/sharded_cache.py``): written only after real weights were converted, required before a load without them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.utils.sharded_cache import LayoutCacheGuard, as_tensor_file, layout_descriptor
from models.demos.deepseek_v3_d_p.utils.weight_cache_dump import weight_cache_dump_mode
from models.experimental.hy4_preview.reference.lm_head import LMHeadConfig, VocabShards
from models.experimental.hy4_preview.tt.common import MeshPlan
from models.experimental.hy4_preview.tt.embedding import TtHy4Embedding, per_chip_mapper, per_chip_shape

CACHE_FORMAT = 1  # bump when the cached table changes without the mesh or the vocab split changing
TABLE_DTYPE = ttnn.bfloat16


# ----------------------------------------------------------------------------------------------- model config
def vocab_sharded(model_cfg) -> bool:
    """Whether a model config embeds with the vocab-sharded table (``EMBED_VOCAB_SHARDED``; default off)."""
    return bool(getattr(model_cfg, "EMBED_VOCAB_SHARDED", False))


def embedding_mesh(model_cfg) -> tuple[tuple[int, int], int]:
    """(mesh shape, SP axis) an ``EMBED_VOCAB_SHARDED`` config runs on, which its cache name and stamp carry: the
    tp32 mesh of the 2D FFN (``FFN_2D_MESH``; the hidden on ``FFN_2D_HIDDEN_AXIS``, the sequence on the other)."""
    if not vocab_sharded(model_cfg):
        raise ValueError(f"{getattr(model_cfg, '__name__', model_cfg)} does not set EMBED_VOCAB_SHARDED")
    return tuple(int(n) for n in model_cfg.FFN_2D_MESH), 1 - int(model_cfg.FFN_2D_HIDDEN_AXIS)


# ----------------------------------------------------------------------------------------------- sharding math
def num_chips(mesh_shape: Sequence[int]) -> int:
    rows, cols = (int(n) for n in mesh_shape)
    return rows * cols


def chip_of(mesh_shape: Sequence[int], coord: Sequence[int]) -> int:
    """Linear chip index of mesh coordinate ``(row, col)``: row-major, the order ``VocabShards`` counts chips in."""
    return int(coord[0]) * int(mesh_shape[1]) + int(coord[1])


def host_table(weight: torch.Tensor, mesh_shape: Sequence[int]) -> torch.Tensor:
    """``[vocab, hidden]`` -> ``[rows, cols, rows_per_chip + 2, hidden]``: entry (r, c) is the table chip (r, c) holds
    (decode's ``VocabShards.split_table`` laid out over the mesh)."""
    shards = VocabShards(weight.shape[0], num_chips(mesh_shape))
    return shards.split_table(weight).reshape(*mesh_shape, shards.table_rows, weight.shape[-1])


def reference_output(
    weight: torch.Tensor, token_ids: torch.Tensor, mesh_shape: Sequence[int], sp_axis: int
) -> dict[tuple[int, int], torch.Tensor]:
    """What chip (r, c) holds after ``forward`` of a chunk ``token_ids`` [S], computed the way the device does it:
    per-chip lookup of padded table rows, the SP reduce-scatter over the sequence, the TP one over the hidden.
    ``{(r, c): [S / sp, hidden / tp]}``. Exact against ``weight[token_ids]`` sliced (one non-zero term per sum)."""
    mesh_shape = tuple(int(n) for n in mesh_shape)
    tp_axis = 1 - sp_axis
    sp, tp = mesh_shape[sp_axis], mesh_shape[tp_axis]
    S, H = token_ids.numel(), weight.shape[-1]
    shards = VocabShards(weight.shape[0], num_chips(mesh_shape))
    tables = host_table(weight, mesh_shape)
    rows = shards.table_rows_for(token_ids.flatten().long())  # [chips, S]
    looked = {}
    for r in range(mesh_shape[0]):
        for c in range(mesh_shape[1]):
            looked[(r, c)] = tables[r, c][rows[chip_of(mesh_shape, (r, c))]]  # [S, H]
    out = {}
    for r in range(mesh_shape[0]):
        for c in range(mesh_shape[1]):
            i_sp = (r, c)[sp_axis]
            i_tp = (r, c)[tp_axis]
            peers = [(r2, c2) for (r2, c2) in looked if (r2, c2)[tp_axis] == i_tp]  # the SP group of (r, c)
            seq = sum(looked[p][i_sp * (S // sp) : (i_sp + 1) * (S // sp)] for p in peers)  # [S / sp, H]
            out[(r, c)] = seq
    # TP reduce-scatter over the hidden: sum the SP-scattered rows over the TP group, keep this chip's hidden slice.
    final = {}
    for (r, c), seq in out.items():
        i_sp, i_tp = (r, c)[sp_axis], (r, c)[tp_axis]
        group = [k for k in out if k[sp_axis] == i_sp]
        total = sum(out[k] for k in group)
        final[(r, c)] = total[:, i_tp * (H // tp) : (i_tp + 1) * (H // tp)]
    return final


def collective_bytes(seq_len: int, hidden: int, mesh_shape: Sequence[int], sp_axis: int, bytes_per: int = 2) -> dict:
    """Per-chip bytes each step of ``forward`` moves for one chunk of ``seq_len`` tokens (ring collectives: a
    reduce-scatter over n chips sends (n - 1) / n of its input)."""
    sp, tp = int(mesh_shape[sp_axis]), int(mesh_shape[1 - sp_axis])
    lookup = seq_len * hidden * bytes_per
    return {
        "id_gather": (sp - 1) * (seq_len // sp) * 4,
        "lookup_out": lookup,
        "rs_sp": lookup * (sp - 1) // sp,
        "rs_tp": (lookup // sp) * (tp - 1) // tp,
    }


# ----------------------------------------------------------------------------------------------- weight cache
def cache_name(mesh_shape: Sequence[int]) -> str:
    rows, cols = (int(n) for n in mesh_shape)
    return f"embed_vocab{rows * cols}_{rows}x{cols}"


def cache_guard(cache_path: Optional[Path], mesh_shape: Sequence[int], sp_axis: int, vocab_size: int):
    """The layout stamp of the vocab-sharded table (``utils/sharded_cache.py``)."""
    shards = VocabShards(vocab_size, num_chips(mesh_shape))
    descriptor = layout_descriptor(
        "TtHy4PrefillEmbedding",
        CACHE_FORMAT,
        mesh_shape,
        {"seq": sp_axis, "hidden": 1 - sp_axis},
        {"table": TABLE_DTYPE},
        extra={"vocab_split": f"row-major {shards.num_chips} x {shards.rows_per_chip} ids, zero row each side"},
    )
    name = cache_name(mesh_shape)
    return LayoutCacheGuard(cache_path, name, f"{name}_dtype_*.tensorbin", descriptor)


def cache_file(cache_path: Path, mesh_shape: Sequence[int]) -> Path:
    return as_tensor_file(Path(cache_path) / cache_name(mesh_shape), TABLE_DTYPE, ttnn.ROW_MAJOR_LAYOUT)


# ----------------------------------------------------------------------------------------------- module
class TtHy4PrefillEmbedding(TtHy4Embedding):
    """SP-sharded token ids ``[1, 1, S / sp]`` -> ``[1, 1, S / sp, hidden / tp]`` BF16 (module docstring).

    The drop-in for ``TtParallelEmbedding`` on an ``EMBED_VOCAB_SHARDED`` model: same input and output layout, a
    32th of the table per chip instead of a quarter."""

    @staticmethod
    def check_cache_complete(cache_path: Path, mesh_shape: Sequence[int], sp_axis: int, vocab_size: int) -> bool:
        """The table's file under a matching layout stamp (an unstamped or other-layout file does not count)."""
        if cache_path is None or not cache_file(cache_path, mesh_shape).is_file():
            logger.debug(f"TTNN cache missing: {cache_name(mesh_shape)}")
            return False
        return cache_guard(cache_path, mesh_shape, sp_axis, vocab_size).stamped()

    def __init__(
        self,
        mesh_device,
        vocab_size: int,
        emb_dim: int,
        torch_weight: Optional[torch.Tensor] = None,
        sp_axis: int = 0,
        tp_axis: int = 1,
        weight_cache_path: Optional[Path] = None,
        num_links: int = 1,
        topology=None,
        shared_table: Optional[TtHy4Embedding] = None,
    ):
        LightweightModule.__init__(self)
        assert sp_axis in (0, 1) and tp_axis == 1 - sp_axis, f"sp_axis {sp_axis} / tp_axis {tp_axis}"
        self.mesh_device = mesh_device
        self.mesh_shape = tuple(int(n) for n in mesh_device.shape)
        self.sp_axis, self.tp_axis = sp_axis, tp_axis
        self.sp, self.tp = self.mesh_shape[sp_axis], self.mesh_shape[tp_axis]
        self.num_links = num_links
        # Per mesh axis, as the prefill block takes it.
        self.topology = topology if isinstance(topology, (tuple, list)) else (topology, topology)
        # Decode's state: the plan places the table (row-major chip order), cfg / shards are the vocab contract.
        self.plan = MeshPlan(mesh_device, hidden_axis=tp_axis, num_links=num_links)
        self.cfg = LMHeadConfig(hidden_size=emb_dim, vocab_size=vocab_size)
        self.out_dtype = ttnn.bfloat16
        self.chip_shape = per_chip_shape(self.plan)
        self.shards = VocabShards(vocab_size, self.chip_shape[0] * self.chip_shape[1])
        assert emb_dim % (self.tp * ttnn.TILE_SIZE) == 0, f"hidden {emb_dim} over TP {self.tp}"

        from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl

        self.tt_ccl = get_tt_ccl(mesh_device)

        if shared_table is not None:
            # One copy for both engines: decode's device tensors, as they are.
            assert (shared_table.cfg.vocab_size, shared_table.cfg.hidden_size) == (vocab_size, emb_dim)
            assert tuple(shared_table.chip_shape) == tuple(self.chip_shape), (shared_table.chip_shape, self.chip_shape)
            self.table, self.row_origin = shared_table.table, shared_table.row_origin
            return
        self.table = self._load_table(torch_weight, weight_cache_path)
        self.row_origin = ttnn.from_torch(
            (self.shards.offsets() - 1).to(torch.int32).reshape(*self.chip_shape, 1, 1),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=per_chip_mapper(self.plan),
        )

    @classmethod
    def sharing(cls, decode_embedding: TtHy4Embedding, *, sp_axis=0, tp_axis=1, num_links=1, topology=None):
        """A prefill embedding reading the table a decode ``TtHy4Embedding`` already holds (no second copy)."""
        plan = decode_embedding.plan
        return cls(
            plan.mesh_device,
            decode_embedding.cfg.vocab_size,
            decode_embedding.cfg.hidden_size,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            num_links=num_links,
            topology=topology,
            shared_table=decode_embedding,
        )

    def _load_table(self, torch_weight, cache_path):
        vocab, hidden = self.cfg.vocab_size, self.cfg.hidden_size
        guard = cache_guard(cache_path, self.mesh_shape, self.sp_axis, vocab)
        guard.check()  # a table cut for another mesh / split, or unstamped, refuses to load
        if torch_weight is not None:
            assert tuple(torch_weight.shape) == (vocab, hidden), tuple(torch_weight.shape)
            table = host_table(torch_weight.float(), self.chip_shape)
        else:
            if cache_path is None:
                raise ValueError("TtHy4PrefillEmbedding needs the table: pass torch_weight or a weight_cache_path")
            # A load without weights: as_tensor would dump this placeholder on a miss, so the file must be there.
            guard.require_cached([cache_file(cache_path, self.mesh_shape)])
            table = torch.empty(*self.chip_shape, self.shards.table_rows, hidden)
        tt = ttnn.as_tensor(
            table,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            dtype=TABLE_DTYPE,
            mesh_mapper=per_chip_mapper(self.plan),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=None if cache_path is None else str(Path(cache_path) / cache_name(self.mesh_shape)),
            cache_dump_mode=weight_cache_dump_mode(),
        )
        if torch_weight is not None:  # only real weights are stamped
            guard.write()
        return tt

    @staticmethod
    def build_ttnn_cache(torch_weight, mesh_device, cache_path: Path, sp_axis: int = 0):
        """Write the table's cache file and stamp from the checkpoint table (device built, then freed)."""
        emb = TtHy4PrefillEmbedding(
            mesh_device,
            torch_weight.shape[0],
            torch_weight.shape[1],
            torch_weight=torch_weight,
            sp_axis=sp_axis,
            tp_axis=1 - sp_axis,
            weight_cache_path=cache_path,
        )
        emb.deallocate()

    def deallocate(self):
        ttnn.deallocate(self.table)
        ttnn.deallocate(self.row_origin)

    def table_bytes_per_chip(self) -> int:
        return self.shards.table_rows * self.cfg.hidden_size * 2

    # ------------------------------------------------------------------------------------------
    def _gather_ids(self, ids):
        """[1, 1, 1, L] uint32 TILE, this SP rank's ids -> [1, 1, 1, sp * L], the chunk, on every chip."""
        if self.sp == 1:
            return ids
        return ttnn.experimental.all_gather_async(
            ids,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.sp_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.sp_axis),
            num_links=self.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.topology[self.sp_axis] or ttnn.Topology.Linear,
            cluster_axis=self.sp_axis,
        )

    def _reduce_scatter(self, x, dim: int, axis: int):
        if self.mesh_shape[axis] == 1:
            return x
        out = ttnn.experimental.reduce_scatter_minimal_async(
            x,
            persistent_output_buffers=None,
            dim=dim,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=axis),
            num_links=self.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.topology[axis] or ttnn.Topology.Linear,
            cluster_axis=axis,
        )
        ttnn.deallocate(x)
        return out

    def spread(self, x):
        """[1, 1, S, hidden], each token non-zero on its owner only -> [1, 1, S / sp, hidden / tp]."""
        x = self._reduce_scatter(x, dim=2, axis=self.sp_axis)
        return self._reduce_scatter(x, dim=3, axis=self.tp_axis)

    def forward(self, token_ids):
        """``token_ids`` uint32 ``[1, 1, L]`` (or ``[1, L]``) per chip, SP-sharded, the same over TP; ``L`` a multiple
        of 32. Not consumed. -> ``[1, 1, L, hidden / tp]`` BF16 TILE, the block residual layout."""
        L = int(token_ids.shape[-1])
        assert L % ttnn.TILE_SIZE == 0, f"ids per chip ({L}) must be a multiple of {ttnn.TILE_SIZE}; pad the chunk"
        ids = ttnn.to_layout(ttnn.reshape(token_ids, [1, 1, 1, L]), ttnn.TILE_LAYOUT)
        gathered = self._gather_ids(ids)
        if gathered is not ids:
            ttnn.deallocate(ids)
        rows = self.table_rows(gathered)  # [1, sp * L] uint32: per chip, its table row for every id
        ttnn.deallocate(gathered)
        x = self.lookup(rows)  # [1, 1, sp * L, hidden] BF16
        ttnn.deallocate(rows)
        return self.spread(x)
