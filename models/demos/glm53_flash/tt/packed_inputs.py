# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Packed decode inputs (``GLM53_DECODE_PACKED_INPUTS=1``): the host-only planner of the ONE int32 tensor a decode
step uploads instead of the per-field refreshes.

The direct path refreshes every per-step input of the traced decode step with its own
``ttnn.copy_host_to_device_tensor``: the token ids (``GLM53MeshModel.load_ids``), the three position tensors, the
dense bound and the sparse-decode metadata (``FunctionalDecoder.update_decode_positions`` ->
``SparseDecodeMeta.update``: a per-user list of ``valid_len`` tensors, ``tail_pool``, ``valid_len_rows``,
``cur_pos``), i.e. ``7 + users_per_chip`` (+1 with ``dense_positions``) host->device copies, each written to every
chip of the mesh. The packed path uploads one ``[1, 1, segments * H, W]`` int32 ROW_MAJOR host tensor sharded so that
every chip receives its own ``[1, 1, H, W]`` segment, and the decode step starts with a device-side unpack
(``tt/mesh_model.py`` ``PackedDecodeInputs.unpack``: slice + reinterpret + copy into the existing persistent field
tensors), so the rest of the graph is unchanged.

Per-chip segment (``PackedDecodeLayout.fields``; ``B`` = users per row, ``Bc`` = users per chip = ``B`` or
``B / cols`` with the user-partitioned cache, ``GLM53_PARTITION_USERS``):

    row 0                  ids              uint32  [1,1,1,B]   the row's B token ids (the same on every column)
    row 1                  positions        int32   [Bc]        the chip's users' positions
    row 2                  pool_ids         int32   [Bc]        position // index_kpool
    row 3                  ring_slots       int32   [Bc]        position % index_kpool
    row 4                  dense_positions  int32   [Bc]        min(position, dense bound) (only when the model carries it)
    row 4 / 5              cur_pos          int32   [Bc]        SparseDecodeMeta.cur_pos
    rows R0..R0+Bc, col 0  tail_pool        uint32  [Bc,1,1,1]  SparseDecodeMeta.tail_pool
    rows R0..R0+Bc, col 16 valid_len_rows   uint32  [Bc,1]      SparseDecodeMeta.valid_len_rows
    row R0+u, col 16       valid_len[u]     uint32  [1] / [1,1] the same column, one element per user (``per_user_valid_len``)

``R0`` = the number of row fields; ``H = R0 + Bc``; ``W = max(B, 32)``. Row fields are row slices ``[1,1,1,n]``,
column fields column slices ``[1,1,n,1]`` (a ROW_MAJOR slice keeps the field's last dim, so the reshape to the
field's shape is a free view); the column fields start on 64-byte boundaries (``COL_STRIDE``). Every value is
non-negative and below 2**31, so the int32 upload reinterpreted as uint32 (``ttnn.bitcast``) is exact for the
uint32 fields. The host values are the direct path's: ``position_fields`` (``decode_position_tensors`` /
``_dense_bound``) and ``sparse_host_values`` (``SparseDecodeMeta.host_values`` builds its tensors from it).

Segments are ordered like the direct path's placement (``dsa._meta_upload``): one per mesh row (replicated over the
columns) without the partition, one per chip in row-major chip order with it (chip ``(r, c)`` = segment
``r * cols + c``: ``ShardTensorToMesh`` on dim 2). This module imports torch only (host tests run without ttnn).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

INT32 = "int32"
UINT32 = "uint32"
COL_STRIDE = 16  # column fields start at 64-byte boundaries (the Blackhole DRAM alignment): aligned slice begins
INT32_MAX = 2**31 - 1
_SPARSE_FIELDS = ("cur_pos", "tail_pool", "valid_len_rows")
# GLM53_RESIDENT_DECODE (tt/resident_decode.py, doc/mesh/resident_decode.md): the traced decode step feeds its sampled
# ids back and advances the live positions on device; the served adapter then declares supports_async_decode
RESIDENT_ENV = "GLM53_RESIDENT_DECODE"


def resident_decode_enabled(env=None) -> bool:
    """``GLM53_RESIDENT_DECODE=1``: the mesh model's decode step carries the device-side feedback and the served
    adapter runs the plugin's reload-plan chain (``reload_inputs=False`` steps). Default off until the served A/B."""
    import os

    env = os.environ if env is None else env
    return str(env.get(RESIDENT_ENV, "0")).strip().lower() in ("1", "true", "yes", "on")


def position_fields(positions: Sequence[int], kpool: int, dense_max_pos: int) -> dict[str, list[int]]:
    """The per-user position-derived host values of the direct path: ``positions``, ``pool_ids`` (position //
    kpool), ``ring_slots`` (position % kpool) and ``dense_positions`` (min(position, dense_max_pos))."""
    pos = [int(p) for p in positions]
    return {
        "positions": pos,
        "pool_ids": [p // kpool for p in pos],
        "ring_slots": [p % kpool for p in pos],
        "dense_positions": [min(p, dense_max_pos) for p in pos],
    }


def sparse_host_values(
    positions: Sequence[int], kpool: int, select_pools: int
) -> tuple[list[int], list[int], list[int]]:
    """``(valid, tail, cur)`` per user, the ``SparseDecodeMeta`` host values (``tt/dsa.py`` documents them): with
    ``P`` = kpool, ``K`` = select_pools, ``n = (a + 1) // P`` complete pools and ``t = (a + 1) % P`` tail tokens of a
    user at position ``a``: ``valid = max(n, K)``, ``tail = n``, ``cur = a`` for a dense user (``n < K``) else
    ``P * K - 1 + t``."""
    P, K = int(kpool), int(select_pools)
    valid, tail, cur = [], [], []
    for a in positions:
        a = int(a)
        n, t = (a + 1) // P, (a + 1) % P
        dense = n < K
        valid.append(max(n, K))
        tail.append(n)
        cur.append(a if dense else P * K - 1 + t)
    return valid, tail, cur


@dataclass(frozen=True)
class PackedField:
    """One field of the per-chip segment: the 4-D slice ``[start, end)`` of the ``[1, 1, H, W]`` segment, its device
    dtype and whether it is a column slice (``[1,1,n,1]``) or a row slice (``[1,1,1,n]``)."""

    name: str
    dtype: str
    start: tuple[int, int, int, int]
    end: tuple[int, int, int, int]
    column: bool

    @property
    def n(self) -> int:
        return (self.end[2] - self.start[2]) * (self.end[3] - self.start[3])

    @property
    def row(self) -> int:
        return self.start[2]

    @property
    def col(self) -> int:
        return self.start[3]


@dataclass(frozen=True)
class PackedDecodeLayout:
    """The packed layout of a mesh (``rows`` x ``cols``) decoding ``users_per_row`` user slots per row.
    ``partition``: the user-partitioned DSA cache (the row's users split over the columns; the ids stay replicated).
    ``dense``: the model's ``DecodePositions`` carries ``dense_positions``. ``per_user_valid_len``: the unpack also
    refreshes the per-user ``valid_len`` list (consumed by the unbatched top-k only, ``GLM53_TOPK_BATCHED=0``)."""

    rows: int
    cols: int
    users_per_row: int
    index_kpool: int
    index_select_pools: int
    dense_max_pos: int
    partition: bool = False
    dense: bool = False
    per_user_valid_len: bool = False

    def __post_init__(self):
        if self.rows < 1 or self.cols < 1:
            raise ValueError(f"mesh ({self.rows}, {self.cols}) must have at least one row and one column")
        if self.users_per_row < 1:
            raise ValueError(f"users_per_row {self.users_per_row} must be >= 1")
        if self.index_kpool < 1 or self.index_select_pools < 1 or self.dense_max_pos < 0:
            raise ValueError("index_kpool / index_select_pools must be >= 1 and dense_max_pos >= 0")
        if self.partition and self.cols > 1 and self.users_per_row % self.cols:
            raise ValueError(
                f"the user-partitioned cache splits {self.users_per_row} users per row over {self.cols} columns: "
                "users_per_row must be a multiple of cols"
            )

    # ----------------------------------------------------------------------------------------- geometry
    @property
    def segment_cols(self) -> int:
        """Segments per mesh row: one (replicated over the columns) or ``cols`` with the partition."""
        return self.cols if self.partition and self.cols > 1 else 1

    @property
    def users_per_chip(self) -> int:
        return self.users_per_row // self.segment_cols

    @property
    def segments(self) -> int:
        return self.rows * self.segment_cols

    @property
    def total_users(self) -> int:
        return self.rows * self.users_per_row

    @property
    def row_fields(self) -> tuple[str, ...]:
        return (
            ("ids", "positions", "pool_ids", "ring_slots") + (("dense_positions",) if self.dense else ()) + ("cur_pos",)
        )

    @property
    def column_block(self) -> int:
        """``R0``: the first row of the column fields."""
        return len(self.row_fields)

    @property
    def height(self) -> int:
        return self.column_block + self.users_per_chip

    @property
    def width(self) -> int:
        return max(self.users_per_row, 2 * COL_STRIDE)

    @property
    def segment_shape(self) -> tuple[int, int, int, int]:
        """The per-chip device tensor shape."""
        return (1, 1, self.height, self.width)

    @property
    def host_shape(self) -> tuple[int, int, int, int]:
        """The packed host tensor (every segment stacked on dim 2)."""
        return (1, 1, self.segments * self.height, self.width)

    @property
    def fields(self) -> dict[str, PackedField]:
        B, Bc, R0 = self.users_per_row, self.users_per_chip, self.column_block
        out: dict[str, PackedField] = {}
        for r, name in enumerate(self.row_fields):
            n = B if name == "ids" else Bc
            dtype = UINT32 if name == "ids" else INT32
            out[name] = PackedField(name, dtype, (0, 0, r, 0), (1, 1, r + 1, n), column=False)
        out["tail_pool"] = PackedField("tail_pool", UINT32, (0, 0, R0, 0), (1, 1, R0 + Bc, 1), column=True)
        c = COL_STRIDE
        out["valid_len_rows"] = PackedField(
            "valid_len_rows", UINT32, (0, 0, R0, c), (1, 1, R0 + Bc, c + 1), column=True
        )
        if self.per_user_valid_len:
            for u in range(Bc):
                out[f"valid_len[{u}]"] = PackedField(
                    f"valid_len[{u}]", UINT32, (0, 0, R0 + u, c), (1, 1, R0 + u + 1, c + 1), column=True
                )
        return out

    def segment(self, row: int, col: int = 0) -> int:
        """The segment chip ``(row, col)`` reads."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"chip ({row}, {col}) outside the ({self.rows}, {self.cols}) mesh")
        return row * self.segment_cols + (col if self.segment_cols > 1 else 0)

    def chip_users(self, row: int, col: int = 0) -> tuple[int, int]:
        """``[first, end)`` of the mesh-wide user list held by chip ``(row, col)``'s per-chip fields."""
        first = self.segment(row, col) * self.users_per_chip
        return first, first + self.users_per_chip

    # --------------------------------------------------------------------------------------------- values
    def host_fields(self, ids: torch.Tensor, positions: Sequence[int]) -> dict[str, list[int]]:
        """The mesh-wide host values of every field (the direct path's tensors, flattened): ``ids`` ``[rows * B]``,
        the position fields and the sparse metadata, each a list over the mesh-wide user list."""
        ids_l = [int(v) for v in ids.reshape(-1).tolist()]
        if len(ids_l) != self.total_users:
            raise ValueError(f"{len(ids_l)} ids for {self.total_users} user slots")
        if len(positions) != self.total_users:
            raise ValueError(f"{len(positions)} positions for {self.total_users} user slots")
        pos = position_fields(positions, self.index_kpool, self.dense_max_pos)
        valid, tail, cur = sparse_host_values(positions, self.index_kpool, self.index_select_pools)
        out = {"ids": ids_l, **pos, "cur_pos": cur, "tail_pool": tail, "valid_len_rows": valid}
        if not self.dense:
            del out["dense_positions"]
        for name, vals in out.items():
            lo, hi = min(vals), max(vals)
            if lo < 0 or hi > INT32_MAX:
                raise ValueError(f"{name}: values must be in [0, 2**31) for the int32 upload (got {lo}..{hi})")
        return out

    def pack(self, ids: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
        """``ids`` (any shape, ``rows * B`` padded token ids, row r's users first) + the ``rows * B`` positions ->
        the packed host tensor ``[1, 1, segments * H, W]`` int32 (the direct path's ``copy_host_to_device_tensor``
        host values, laid out per ``fields``)."""
        vals = self.host_fields(ids, positions)
        B, Bc, R0 = self.users_per_row, self.users_per_chip, self.column_block
        fields = self.fields
        out = torch.zeros(self.segments, self.height, self.width, dtype=torch.int32)
        ids2 = torch.tensor(vals["ids"], dtype=torch.int32).reshape(self.rows, B)
        for seg in range(self.segments):
            r = seg // self.segment_cols
            u0 = seg * Bc
            out[seg, fields["ids"].row, :B] = ids2[r]
            for name in self.row_fields[1:]:
                out[seg, fields[name].row, :Bc] = torch.tensor(vals[name][u0 : u0 + Bc], dtype=torch.int32)
            for name in ("tail_pool", "valid_len_rows"):
                out[seg, R0 : R0 + Bc, fields[name].col] = torch.tensor(vals[name][u0 : u0 + Bc], dtype=torch.int32)
        return out.reshape(self.host_shape)

    def segment_of(self, packed: torch.Tensor, segment: int) -> torch.Tensor:
        """Segment ``segment`` of a packed host tensor, ``[1, 1, H, W]`` (what chip ``segment`` receives)."""
        if tuple(packed.shape) != self.host_shape:
            raise ValueError(f"packed tensor {tuple(packed.shape)} is not {self.host_shape}")
        H = self.height
        return packed[:, :, segment * H : (segment + 1) * H, :]

    def unpack_host(self, packed: torch.Tensor, segment: int) -> dict[str, list[int]]:
        """The per-field values the device unpack produces on the chip reading ``segment`` (flattened, in the
        field's element order): the round-trip check of ``pack``."""
        seg = self.segment_of(packed, segment)
        return {
            name: [int(v) for v in seg[0, 0, f.start[2] : f.end[2], f.start[3] : f.end[3]].reshape(-1).tolist()]
            for name, f in self.fields.items()
        }

    def expected_chip_fields(
        self, ids: torch.Tensor, positions: Sequence[int], row: int, col: int = 0
    ) -> dict[str, list[int]]:
        """The direct path's per-chip values of chip ``(row, col)`` (``ids``: the row's B ids; the others: the
        chip's ``users_per_chip`` users), keyed like ``fields``."""
        vals = self.host_fields(ids, positions)
        first, end = self.chip_users(row, col)
        out = {"ids": vals["ids"][row * self.users_per_row : (row + 1) * self.users_per_row]}
        for name in list(self.row_fields[1:]) + ["tail_pool", "valid_len_rows"]:
            out[name] = vals[name][first:end]
        if self.per_user_valid_len:
            for u in range(self.users_per_chip):
                out[f"valid_len[{u}]"] = [vals["valid_len_rows"][first + u]]
        return out
