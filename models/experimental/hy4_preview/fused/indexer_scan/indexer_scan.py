# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Index-key scan and chip-local top-2048 with positions, for up to 8 users, as one ``ttnn.generic_op`` program
per chip (piece 4 of the decode megakernel, fused/README.md).

What a chip does on a full-indexer layer, for every user u:

    score[t] = sum_h w[u, h] * relu(q[u, h] . key[u, t])     over the chip's stripe of the user's index keys,
                                                             positions the query may not see masked with NEG
    top      = the 2048 best (score, position) pairs of the stripe

Storage read in place (the prefill stack's index-key cache):
one DRAM tensor per chip, ``[slots, 1, stripe rows, 128]`` BFP8 tiles, ND-sharded with one shard = one 32-row block
(4 tiles, 4352 bytes), shards round-robin over the 8 DRAM banks, slot-major. Bank b holds blocks b, b + 8, ... of a
slot back to back, so one core per (user, bank) reads its share with large reads from a single bank.

Cores: 64 "scan cores", one per (user, bank). Each scores its 1/8 of the user's stripe, keeps its own top-2048
and takes part in a binary tree over the user's 8 banks (bank 1 -> 0, 3 -> 2, ...; 2 -> 0, 6 -> 4; 4 -> 0). The
user's bank-0 core ends with the chip's top-2048 for that user and writes it to DRAM.

Arithmetic (kernels/scan_compute.cpp): keys are srcB and the query srcA of the FPU, so per 32-position block
``relu(K @ Qt)`` is one [32 positions x 32 heads] tile; the head sum is a second matmul against a selector tile
that places the block's 32 scores in one column of the score tile. FP32 accumulation in DEST, BF16 scores.
The scores go through the topk_xl LLK in its fused form: a 32-bit word = [bf16 score | u16 stamp], so the order
is the BF16 score's and equal scores are ordered by stamp (position). Precision is stated in ``Scan.precision``.

The query a core takes is prepared by ``device_query``: by default the head weight's magnitude is folded into the
query head (w relu(q . k) = sign(w) relu(|w| q . k)), which leaves the head sum with +-1 multipliers. Per-step
scalars (context length, score centre, the 32 multipliers) come from a device tensor, one page per user, so a
captured trace follows them. CONTRACT.md has the interfaces to the neighbouring pieces.

Output per user (``Scan.read``): 2048 fused words in rank order, i.e. BF16 scores best first with their stripe
rows; rows the query does not see carry NEG (or -inf padding) and are reported as invalid.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
KV_STORE_KERNELS = str(Path(__file__).parent.parent / "kv_store" / "kernels")
CB_FRAMES = 7  # the frame tables under the key reader (free in kernels/scan_common.hpp)
FRAMES_DEFAULT = os.environ.get("HY4_SCAN_FRAMES") == "1"  # tests: every Scan reads its keys through frame tables
KERNELS = "models/experimental/hy4_preview/fused/indexer_scan/kernels"

TILE = 32
BANKS = 8
KEY_DIM = 128
HEADS = 32
TOPK = 2048
KEY_TILE_BYTES = 1088
KEY_BLOCK_BYTES = 4 * KEY_TILE_BYTES
BF16_TILE_BYTES = 2048
WORD_TILE_BYTES = 4096
NEG = -(2.0**100)

# circular buffers (kernels/scan_common.hpp)
CB_K, CB_QT, CB_SEL, CB_MASK, CB_META, CB_LANDING, CB_CENTRE = 0, 1, 2, 3, 4, 5, 6
CB_P, CB_SCORES, CB_SEND, CB_OUT = 16, 17, 18, 19
SEM_CREDIT, SEM_DATA = 0, 1

FIDELITY = {
    "lofi": ttnn.MathFidelity.LoFi,
    "hifi2": ttnn.MathFidelity.HiFi2,
    "hifi3": ttnn.MathFidelity.HiFi3,
    "hifi4": ttnn.MathFidelity.HiFi4,
}


def is_mock() -> bool:
    return bool(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"))


# ---------------------------------------------------------------------------------------------
# Layout arithmetic shared with the kernels
# ---------------------------------------------------------------------------------------------
def tile_word(row, col):
    """Word offset of element (row, col) of a 32x32 tile inside its page (4 faces of 16x16)."""
    return ((((row >> 4) << 1) | (col >> 4)) << 8) | ((row & 15) << 4) | (col & 15)


def raw_words(tiles: torch.Tensor) -> torch.Tensor:
    """[..., 32, 32] tiles as ttnn.to_torch returns them -> [..., 1024] in the order the words lie in the page."""
    r = torch.arange(TILE)[:, None].expand(TILE, TILE)
    c = torch.arange(TILE)[None, :].expand(TILE, TILE)
    order = torch.empty(TILE * TILE, dtype=torch.long)
    order[tile_word(r, c).reshape(-1)] = (r * TILE + c).reshape(-1)
    return tiles.reshape(*tiles.shape[:-2], TILE * TILE)[..., order]


def rank_word(rank):
    """Word offset, within a survivor's 2 raw tiles, of the entry of rank ``rank`` (0 = best): the sorted
    sequence runs down the 16 columns of the [128 rows x 16] arrangement of the two tiles' face rows."""
    return ((rank & 127) << 4) | (rank >> 7)


def visible_rows(context: int, chip: int, num_chips: int, block: int, stripe_rows: int) -> int:
    """Rows of chip ``chip``'s stripe that hold one of the positions [0, context): a prefix of the stripe."""
    cycle = num_chips * block
    in_block = min(max(context % cycle - chip * block, 0), block)
    return min(context // cycle * block + in_block, stripe_rows)


def row_position(row, chip: int, num_chips: int, block: int):
    """Sequence position of stripe row ``row`` of chip ``chip`` (tt/kv_layout.py BlockCyclicLayout.position)."""
    return (row // block * num_chips + chip) * block + row % block


@dataclass
class ScanConfig:
    users: int = 8
    stripe_rows: int = 32768  # positions per user on a chip
    num_chips: int = 32  # chips a context is striped over (block-cyclic)
    chip_block: int = 128  # tokens a chip holds per turn
    mode: str = "tree"  # "scores": store the scores | "local": every core's own top-2048 | "tree": the chip's
    fidelity: str = "hifi4"
    blocks_per_pass: int = 8  # key blocks per DEST pass (FP32 full-sync DEST holds 8 tiles)
    # The key stream. stream_depth N > 0: NOC packets of blocks_per_read blocks (at most 3 = 13,056 bytes), N in
    # flight on their own transaction ids. 0: one awaited read of blocks_per_read blocks at a time.
    stream_depth: int = 4
    blocks_per_read: int | None = None  # default: 3 per packet, or 16 per awaited read
    k_depth_reads: int = 2  # awaited reads the key CB holds (stream_depth 0)
    reader_noc: int = 0  # NoC the key stream arrives on
    # "grid": core (user, bank), one rectangle of cores: the cheapest launch (3.9 us for 64 cores against 11.2 us
    # for the 9 ranges of "rows") and the best at the AgentX shape. "rows": a bank's cores on the worker row of
    # its NOC0 DRAM endpoint: the fastest stream, 10 us better at 8 users x 32,768 positions. "bank_rows": an
    # earlier row heuristic.
    placement: str = "grid"
    selectors: str = "l1"  # head-sum selector tiles: "l1" resident, rewritten in place | "dram": read from a tensor
    # the query's 4 tiles as [32 heads x 32 dims] (a head a row), transposed by the unpacker, instead of
    # [32 dims x 32 heads]: what a query assembled on the chips from per-head pieces is (layer_attn/PLAN.md section
    # 21). Default (None): off, unless HY4_SCAN_QT_ROWS=1 (tests: every case of the piece in this form).
    query_rows: bool | None = None
    # measurement only (results are garbage): 1 = no key reads from DRAM, 2 = no scoring arithmetic, 4 = every
    # kernel returns at once (the program's launch floor), 8 = with 4: the program carries no semaphores
    ablate: int = 0

    def __post_init__(self):
        assert self.mode in ("scores", "local", "tree"), self.mode
        assert self.selectors in ("dram", "l1"), self.selectors
        if self.query_rows is None:
            self.query_rows = os.environ.get("HY4_SCAN_QT_ROWS") == "1"
        assert 1 <= self.users <= 8
        assert self.stripe_rows % (TILE * BANKS) == 0, "every slot must start on DRAM bank 0"
        assert self.num_chips & (self.num_chips - 1) == 0 and self.chip_block & (self.chip_block - 1) == 0
        assert self.blocks_per_pass in (1, 2, 4, 8) and 0 <= self.stream_depth <= 8
        if self.blocks_per_read is None:
            self.blocks_per_read = 3 if self.stream_depth else 16
        if self.stream_depth:
            assert self.blocks_per_read * KEY_BLOCK_BYTES <= 16384, "a packet is one NOC burst (16,384 bytes)"
        else:
            assert self.blocks_per_read % self.blocks_per_pass == 0
        assert BANKS * self.chunks_per_bank <= 32, "the fused stamp holds 32 chunk ids per user"

    @property
    def k_blocks(self) -> int:
        """Key blocks the key CB holds. A pass of the compute kernel and a packet of the reader must each lie
        whole inside it, so it is a multiple of both."""
        if not self.stream_depth:
            return self.k_depth_reads * self.blocks_per_read
        import math

        unit = math.lcm(self.blocks_per_read, self.blocks_per_pass)
        need = self.blocks_per_pass + (self.stream_depth + 1) * self.blocks_per_read
        return -(-need // unit) * unit

    @property
    def bank_rows(self) -> int:
        return self.stripe_rows // BANKS

    @property
    def bank_blocks(self) -> int:
        return self.bank_rows // TILE

    @property
    def score_tiles(self) -> int:
        """Score tiles of one bank's whole stream (1024 positions each)."""
        return -(-self.bank_rows // 1024)

    @property
    def chunks_per_bank(self) -> int:
        return -(-self.score_tiles // 2)


def scan_core(user: int, bank: int):
    """Logical core of (user, bank): one row of the worker grid per bank, one column per user."""
    return (user, bank)


def bank_row_cores(mesh_device, users: int, noc, coord=(0, 0)):
    """core_of for placement "bank_rows": bank b's cores fill the worker row of its DRAM endpoint on ``noc``
    (and the next row), on the endpoint's side of the grid, nearest column first."""
    grid = mesh_device.compute_with_storage_grid_size()
    best = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(noc, ttnn.MeshCoordinate(*coord))
    taken, table = set(), {}
    for bank in range(BANKS):
        ax, ay = best[bank].x, best[bank].y
        columns = list(range(ax, grid.x)) if ax else list(range(0, min(7, grid.x)))
        candidates = [(x, (ay + dy) % grid.y) for dy in range(grid.y) for x in columns]
        picked = []
        for core in candidates:
            if core not in taken:
                picked.append(core)
                taken.add(core)
            if len(picked) == users:
                break
        assert len(picked) == users, f"bank {bank}: no room for {users} cores"
        for user, core in enumerate(picked):
            table[(user, bank)] = core
    return lambda user, bank: table[(user, bank)]


def row_cores(mesh_device, users: int, coord=(0, 0)):
    """core_of for placement "rows": models/demos/gpt_oss/tt/fused_decode/stream.py row_placement, the placement
    that reaches the chip's DRAM peak there: a bank's readers on the worker row of its NOC0 endpoint, nearest
    first, a row shared by two banks split at the right bank's worker."""
    from models.demos.gpt_oss.tt.fused_decode import stream

    placed = stream.row_placement(mesh_device, users)
    return lambda user, bank: (placed[bank][user].x, placed[bank][user].y)


def _core_set(cores):
    """The cores as few rectangles as merge() finds: dispatch writes binaries and arguments per core range."""
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores[:1]])
    for c in cores[1:]:
        ranges = ranges.merge(ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c))]))
    return ranges


def key_memory_config(mesh_device):
    """The prefill stack's index-key memory config (models/demos/deepseek_v3_d_p/utils/kv_cache_utils.py
    init_kvpe_cache): DRAM, one ND shard per 32-row block, round-robin over the DRAM banks."""
    banks = mesh_device.dram_grid_size().x
    assert banks == BANKS, f"{banks} DRAM banks: the scan is laid out for {BANKS}"
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(banks)])
    spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, TILE, KEY_DIM],
        grid=grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=spec)


def keys_to_device(mesh_device, keys: torch.Tensor):
    """Host keys [slots, stripe rows, 128] -> the cache tensor [slots, 1, rows, 128], BFP8, laid out as prefill's."""
    slots, rows, dim = keys.shape
    assert dim == KEY_DIM and rows % (TILE * BANKS) == 0
    return ttnn.from_torch(
        keys.float().reshape(slots, 1, rows, dim).contiguous(),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=key_memory_config(mesh_device),
    )


def device_query(q: torch.Tensor, w: torch.Tensor, fold: bool = None):
    """Queries [U, 32 heads, 128] (the cache's frame: Hadamard basis) and head weights [U, 32] -> what the scan
    cores take: (qt [U, 128, 32] the query the FPU sees, in fp32 holding BF16 values; s [U, 32] the head-sum
    multipliers). ``fold``: |w| goes into the query head and s = sign(w); otherwise the query is q and s = w.
    Default: folded, unless HY4_SCAN_FOLD=0 (tests: the unfolded form everywhere the caller does not say; it is
    what a query assembled on the chips can have, layer_attn/PLAN.md section 21)."""
    if fold is None:
        fold = os.environ.get("HY4_SCAN_FOLD", "1") != "0"
    q, w = q.float(), w.float()
    if fold:
        qd = (q * w.abs()[:, :, None]).bfloat16().float()
        s = torch.where(w < 0, -1.0, 1.0)
    else:
        qd = q.bfloat16().float()
        s = w.bfloat16().float()
    return qd.transpose(1, 2).contiguous(), s


def golden_scores(qt: torch.Tensor, s: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Exact arithmetic on the device's own operands: qt [128, 32], s [32], keys [T, 128] -> [T] float64."""
    per_head = torch.relu(keys.double() @ qt.double())
    return per_head @ s.double()


class Scan:
    """The program, its input tensors and its output on ``mesh_device``.

    ``keys``: the index-key cache (keys_to_device, or the prefill stack's own tensor); ``slots[u]``: user u's slot
    in it (prefill: ``user * 21 + full_rank``). ``coords``: mesh coordinates that run the program (default all);
    ``chip_index(coord)``: the chip's index in the block-cyclic layout (default row-major).
    """

    precision = (
        "keys BFP8 (as stored) in srcB, query BF16 in srcA, FP32 DEST accumulation; relu(K @ Qt) rounded to BF16 "
        "by the packer before the head sum; scores rounded to BF16 by the packer; top-k orders BF16 scores, "
        "equal scores by stamp"
    )

    def __init__(
        self, mesh_device, cfg: ScanConfig, keys, slots, coords=None, chip_index=None, core_of=None, frames=None
    ):
        self.mesh_device, self.cfg, self.keys = mesh_device, cfg, keys
        self.reader_noc = ttnn.NOC.NOC_1 if cfg.reader_noc else ttnn.NOC.NOC_0
        if core_of is None:
            core_of = {
                "grid": lambda: scan_core,
                "bank_rows": lambda: bank_row_cores(mesh_device, cfg.users, self.reader_noc),
                "rows": lambda: row_cores(mesh_device, cfg.users),
            }[cfg.placement]()
        self.slots = list(slots)
        assert len(self.slots) == cfg.users
        shape = tuple(keys.shape)
        assert shape[1:] == (1, cfg.stripe_rows, KEY_DIM) and max(self.slots) < shape[0], (shape, self.slots)
        rows, cols = tuple(mesh_device.shape)
        self.coords = [(r, c) for r in range(rows) for c in range(cols)] if coords is None else list(coords)
        self.chip_index = chip_index or (lambda coord: coord[0] * cols + coord[1])
        self.core_of = core_of
        grid = mesh_device.compute_with_storage_grid_size()
        self.cores = [core_of(u, b) for u in range(cfg.users) for b in range(BANKS)]
        assert len(set(self.cores)) == len(self.cores) and all(
            x < grid.x and y < grid.y for x, y in self.cores
        ), f"scan cores do not fit the {grid.x}x{grid.y} worker grid"
        # frames: the key reader takes its block addresses from fused/kv_store's frame tables (kernels/scan_reader.cpp
        # HY4_SCAN_FRAMES). (layers, rank): today's slot-major cache with ``layers`` slots per user, this scan on slot
        # user * layers + rank, identity tables (the same bytes, the same packets). (Geometry, rank): that layout
        # (a frame-major pool), tables from set_frames.
        if frames is None and FRAMES_DEFAULT and cfg.stream_depth > 0 and cfg.stripe_rows % 256 == 0:
            frames = (shape[0] // cfg.users, self.slots[0])  # (the awaited-read form and odd stripes stay as they are)
        self.geometry = self.frame_tables = None
        if frames is not None:
            from models.experimental.hy4_preview.fused.kv_store import device as kv_device
            from models.experimental.hy4_preview.fused.kv_store import frames as kv_frames

            geometry, self.rank = frames
            assert cfg.stream_depth > 0, "the paged key stream is the pipelined one"
            if isinstance(geometry, int):
                layers = geometry
                assert self.slots == [u * layers + self.rank for u in range(cfg.users)], (self.slots, layers, self.rank)
                assert cfg.stripe_rows % kv_frames.FRAME_ROWS == 0, "a frame is 256 stripe rows"
                kind = kv_frames.Kind("index", layers, KEY_DIM, KEY_BLOCK_BYTES, 0)
                geometry = kv_frames.Geometry.contiguous(kind, cfg.stripe_rows // kv_frames.FRAME_ROWS)
            self.geometry = geometry
            self.frame_tables = kv_device.FrameTables(mesh_device, self.cores, rows=cfg.users)
            if geometry.frame_stride == 1:
                self.set_frames([geometry.identity_table(u) for u in range(cfg.users)])

        U = cfg.users
        dram = ttnn.DRAM_MEMORY_CONFIG
        self.qt = ttnn.from_torch(
            torch.zeros((U, 1, HEADS, KEY_DIM) if cfg.query_rows else (U, 1, KEY_DIM, HEADS)),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=dram,
        )
        self.sel = ttnn.from_torch(
            torch.zeros(U, HEADS, TILE, TILE),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=dram,
        )
        # The step tensor: one 128-byte page of uint32 per user (kernels/scan_common.hpp step_*): word 4 the
        # context length, word 5 the score centre (BF16 bits), words 16..31 the 32 head-sum multipliers (BF16).
        self._step = torch.zeros(8, 32, dtype=torch.int32)
        self.step = ttnn.from_torch(
            self._step.reshape(1, 1, 8, 32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=dram,
        )
        self.sel_l1 = None
        if cfg.selectors == "l1":
            # 32 selector tiles per scan core, in L1 behind cb_sel; all zero but the multipliers the reader writes
            self.sel_l1 = ttnn.from_torch(
                torch.zeros(1, 1, len(self.cores) * HEADS * TILE, TILE),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(_core_set(self.cores), (HEADS * TILE, TILE), ttnn.ShardOrientation.ROW_MAJOR),
                ),
            )
        if cfg.mode == "scores":
            self.out = ttnn.from_torch(
                torch.zeros(U * BANKS, cfg.score_tiles, TILE, TILE),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=dram,
            )
        else:
            self.out = ttnn.from_torch(
                torch.zeros(U * BANKS, 2, TILE, TILE, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=dram,
            )
        self.centres = torch.zeros(U)
        self._descriptor = None

    # inputs -------------------------------------------------------------------------------------
    def set_frames(self, tables):
        """tables: one table of 128 frame bases per user (kv_store's KvStore.table). In place."""
        self.frame_tables.write([v for t in tables for v in t])

    def _write(self, tensor, value: torch.Tensor, dtype, layout):
        host = ttnn.from_torch(value.contiguous(), dtype=dtype, layout=layout)
        ttnn.copy_host_to_device_tensor(host, tensor)

    def _write_step(self):
        self._write(self.step, self._step.reshape(1, 1, 8, 32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    def set_query(self, qt: torch.Tensor, s: torch.Tensor):
        """qt [U, 128, 32], s [U, 32] from device_query. Rewrites the device tensors in place (a trace follows)."""
        U = self.cfg.users
        assert tuple(qt.shape) == (U, KEY_DIM, HEADS) and tuple(s.shape) == (U, HEADS)
        if self.cfg.query_rows:  # (tile t: the heads' dims 32 t .., a head a row)
            self._write(
                self.qt,
                qt.transpose(1, 2).reshape(U, 1, HEADS, KEY_DIM).float().contiguous(),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
            )
        else:
            self._write(self.qt, qt.reshape(U, 1, KEY_DIM, HEADS).float(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        bits = s.bfloat16().view(torch.int16).to(torch.int32) & 0xFFFF
        packed = bits[:, 0::2] | (bits[:, 1::2] << 16)
        self._step[:U, 16:32] = torch.where(packed >= 1 << 31, packed - (1 << 32), packed).to(torch.int32)
        self._write_step()
        if self.cfg.selectors == "dram":
            sel = torch.zeros(U, HEADS, TILE, TILE)
            for block in range(TILE):
                sel[:, block, :, block] = s
            self._write(self.sel, sel, ttnn.bfloat16, ttnn.TILE_LAYOUT)

    def set_contexts(self, contexts, centres=None):
        """contexts[u]: user u sees positions [0, contexts[u]). centres[u] (optional, BF16 values): the scores of
        user u leave as BF16(score - centre); give the same centre to every chip, e.g. the user's global cut of
        the previous step. In place."""
        users = self.cfg.users
        self.centres = torch.zeros(users) if centres is None else torch.as_tensor(centres, dtype=torch.float32)
        assert bool((self.centres.bfloat16().float() == self.centres).all()), "centres must be BF16 values"
        self._step[:users, 4] = torch.as_tensor(contexts, dtype=torch.int32)
        self._step[:users, 5] = self.centres.bfloat16().view(torch.int16).to(torch.int32) & 0xFFFF
        self._write_step()

    # program ------------------------------------------------------------------------------------
    def vc_of(self, user: int, bank: int) -> int:
        """Response virtual channel of a core's pipelined key stream: spread over the 4 unicast channels so that
        the streams sharing a row of the grid do not all queue behind one another."""
        return (bank + user) & 3

    def _tree(self, bank: int):
        """(levels at which a survivor lands on ``bank``'s core, the bank it sends its own to or None)."""
        if self.cfg.mode != "tree":
            return 0, None
        if bank == 0:
            return 3, None
        low = (bank & -bank).bit_length() - 1
        return low, bank - (1 << low)

    def append_to(self, desc, coord, semaphore_base: int = 0, slots=None, rank=None):
        """Append this piece to ``desc``, the ProgramDescriptor of chip ``coord`` (a layer program that also holds
        other pieces): 3 kernels on the 64 scan cores, their circular buffers and 2 program semaphores with ids
        ``semaphore_base`` and ``semaphore_base + 1``. The scan cores must not carry other kernels. The caller
        passes ``io_tensors()`` to generic_op along with its own.
        slots (default: the object's): the users' slots in the key cache for THIS descriptor; rank (with frame
        tables): the layer's rank. One Scan (one query tensor, one step tensor, one survivor tensor) then serves
        every full-indexer layer of a step, a descriptor per layer."""
        cfg, mesh = self.cfg, self.mesh_device
        slots = self.slots if slots is None else list(slots)
        assert len(slots) == cfg.users and max(slots) < tuple(self.keys.shape)[0], slots
        scores_only = 1 if cfg.mode == "scores" else 0
        cores = _core_set(self.cores)
        noc = {core: mesh.worker_core_from_logical_core(ttnn.CoreCoord(*core)) for core in self.cores}

        def cb(index, dtype, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
            )

        d = [(r, c) for r in range(mesh.shape[0]) for c in range(mesh.shape[1])].index(tuple(coord))
        on_chip = lambda t: ttnn.get_device_tensors(t)[d]
        keys, qt, sel, step, out = (on_chip(t) for t in (self.keys, self.qt, self.sel, self.step, self.out))

        k_pages = cfg.k_blocks * 4
        cbs = [
            cb(CB_K, ttnn.bfloat8_b, KEY_TILE_BYTES, k_pages),
            cb(CB_QT, ttnn.bfloat16, BF16_TILE_BYTES, 4),
            cb(CB_MASK, ttnn.bfloat16, BF16_TILE_BYTES, 2),
            cb(CB_META, ttnn.uint32, 128, 1),
            cb(CB_LANDING, ttnn.uint32, WORD_TILE_BYTES, 2),
            cb(CB_CENTRE, ttnn.bfloat16, BF16_TILE_BYTES, 2),
            cb(CB_P, ttnn.bfloat16, BF16_TILE_BYTES, 32),
            cb(CB_SCORES, ttnn.bfloat16, BF16_TILE_BYTES, 2 * cfg.chunks_per_bank),
            cb(CB_SEND, ttnn.uint32, WORD_TILE_BYTES, 2),
            cb(CB_OUT, ttnn.uint32, WORD_TILE_BYTES, 2),
        ]
        if self.sel_l1 is not None:
            cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_SEL, on_chip(self.sel_l1)))
        else:
            cbs.append(cb(CB_SEL, ttnn.bfloat16, BF16_TILE_BYTES, 32))
        if self.geometry is not None:
            cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_FRAMES, on_chip(self.frame_tables.tensor)))
        semaphores = [
            ttnn.SemaphoreDescriptor(id=semaphore_base + i, core_ranges=cores, initial_value=0) for i in range(2)
        ]
        log2 = lambda v: v.bit_length() - 1
        reader_ct = [
            cfg.blocks_per_pass,
            cfg.blocks_per_read,
            log2(cfg.chip_block),
            log2(cfg.num_chips),
            cfg.stripe_rows,
            BF16_TILE_BYTES,
            cfg.ablate,
            1 if cfg.selectors == "l1" else 0,
            cfg.stream_depth,
            cfg.k_blocks,
        ]
        compute_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=FIDELITY[cfg.fidelity], math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        )

        reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        pages_per_core = cfg.score_tiles if scores_only else 2
        for user in range(cfg.users):
            slot_offset = slots[user] * (cfg.bank_blocks * KEY_BLOCK_BYTES)
            if self.geometry is not None:  # the layer's slot step; the user's part is in its frame table
                slot_offset = self.geometry.slot_bytes(self.rank if rank is None else rank)
            for bank in range(BANKS):
                x, y = self.core_of(user, bank)
                reader_rt[x][y] = [
                    keys.buffer_address(),
                    bank,
                    slot_offset,
                    self.chip_index(coord),
                    user,
                    step.buffer_address(),
                    qt.buffer_address(),
                    sel.buffer_address(),
                    self.vc_of(user, bank),
                ]
                receives, target = self._tree(bank)
                compute_rt[x][y] = [bank * cfg.chunks_per_bank, receives, 0 if target is None else 1]
                partners = []
                for level in range(3):
                    p = noc[self.core_of(user, bank + (1 << level))] if level < receives else None
                    partners += [p.x, p.y] if p is not None else [0, 0]
                t = noc[self.core_of(user, target)] if target is not None else None
                writer_rt[x][y] = (
                    [receives, 0 if target is None else 1]
                    + partners
                    + ([t.x, t.y] if t is not None else [0, 0])
                    + [out.buffer_address(), (user * BANKS + bank) * pages_per_core]
                )
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=f"{KERNELS}/scan_reader.cpp",
                core_ranges=cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                defines=[("HY4_SCAN_FRAMES", "1")] if self.geometry is not None else [],
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=self.reader_noc
                ),
                compiler_include_paths=[str(KERNEL_DIR)] + ([KV_STORE_KERNELS] if self.geometry is not None else []),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{KERNELS}/scan_writer.cpp",
                core_ranges=cores,
                compile_time_args=[scores_only, cfg.score_tiles, semaphore_base, cfg.ablate],
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
                compiler_include_paths=[str(KERNEL_DIR)],
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{KERNELS}/scan_compute.cpp",
                core_ranges=cores,
                compile_time_args=[cfg.blocks_per_pass, scores_only, cfg.score_tiles, cfg.ablate],
                runtime_args=compute_rt,
                defines=[("HY4_SCAN_QT_ROWS", "1")] if cfg.query_rows else [],
                config=compute_config,
                compiler_include_paths=[str(KERNEL_DIR)],
            ),
        ]
        desc.kernels = list(desc.kernels) + kernels
        desc.cbs = list(desc.cbs) + cbs
        desc.semaphores = list(desc.semaphores) + ([] if cfg.ablate & 8 else semaphores)
        return desc

    def descriptor(self, slots=None, rank=None):
        """The MeshProgramDescriptor of the piece on its own: one program per chip of ``coords`` (slots, rank: see
        append_to)."""
        mesh_desc = ttnn.MeshProgramDescriptor()
        for coord in self.coords:
            desc = self.append_to(
                ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=[]), coord, slots=slots, rank=rank
            )
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))] = desc
        return mesh_desc

    def io_tensors(self):
        """The tensors the program reads and writes (the output last)."""
        return self._io()

    def _io(self):
        extra = [self.sel_l1] if self.sel_l1 is not None else []
        extra += [self.frame_tables.tensor] if self.geometry is not None else []
        return [self.keys, self.qt, self.sel, self.step] + extra + [self.out]

    def compile(self):
        """Compile and size-check the program without running it (what the mock Galaxy can do)."""
        ttnn.experimental.prepare_generic_op(self._io(), self.descriptor())

    def __call__(self):
        """Run; returns the output tensor (rewritten by every run)."""
        if self._descriptor is None:
            self._descriptor = self.descriptor()
        return ttnn.generic_op(self._io(), self._descriptor)

    # host views ---------------------------------------------------------------------------------
    def stamp_row(self, stamp: torch.Tensor) -> torch.Tensor:
        """Stripe row of a fused stamp: [chunk 15:11 | block-in-tile 10:6 | tile-in-chunk 5 | position 4:0]."""
        cfg = self.cfg
        chunk, col, tile, row = stamp >> 11, (stamp >> 6) & 31, (stamp >> 5) & 1, stamp & 31
        bank, local = chunk // cfg.chunks_per_bank, chunk % cfg.chunks_per_bank
        bank_block = (local * 2 + tile) * TILE + col
        return (bank_block * BANKS + bank) * TILE + row

    def row_stamp(self, row: torch.Tensor) -> torch.Tensor:
        """The inverse of stamp_row."""
        cfg = self.cfg
        block, pos = row // TILE, row % TILE
        bank, bank_block = block % BANKS, block // BANKS
        tile, col = bank_block // TILE, bank_block % TILE
        chunk = bank * cfg.chunks_per_bank + tile // 2
        return (chunk << 11) | (col << 6) | ((tile & 1) << 5) | pos

    def decode_scores(self, t: torch.Tensor) -> torch.Tensor:
        """The "scores" output as to_torch returns it -> [U, stripe rows] in stripe-row order."""
        cfg = self.cfg
        t = t.float().reshape(cfg.users, BANKS, cfg.score_tiles, TILE, TILE)  # [u, bank, tile, pos, block]
        t = t.permute(0, 2, 4, 1, 3)  # [u, tile, block, bank, pos]
        return t.reshape(cfg.users, cfg.score_tiles * TILE * BANKS * TILE)[:, : cfg.stripe_rows]

    def read_scores(self, device: int = 0) -> torch.Tensor:
        """mode "scores": [U, stripe rows] BF16 scores in stripe-row order (NEG where masked; rows of score
        tiles past the context hold whatever the pages held)."""
        assert self.cfg.mode == "scores"
        return self.decode_scores(ttnn.to_torch(ttnn.get_device_tensors(self.out)[device]))

    def decode(self, t: torch.Tensor) -> dict:
        """The "local" / "tree" output as to_torch returns it (int, [U * 8, 2, 32, 32]) -> see read()."""
        cfg = self.cfg
        t = t.to(torch.int64) & 0xFFFFFFFF
        t = raw_words(t.reshape(cfg.users, BANKS, 2, TILE, TILE)).reshape(cfg.users, BANKS, 2 * TILE * TILE)
        words = t[..., rank_word(torch.arange(TOPK))]
        if cfg.mode == "tree":
            words = words[:, 0]
        score = (words >> 16).to(torch.int32).to(torch.int16).view(torch.bfloat16).float()
        return {"score": score, "row": self.stamp_row(words & 0xFFFF), "valid": score > 0.5 * NEG, "word": words}

    def read(self, device: int = 0) -> dict:
        """mode "tree": per user the chip's top-2048, best first. mode "local": the same per (user, bank).
        Returns {"score": float [..., 2048], "row": int64 [..., 2048] stripe rows, "valid": bool, "word": the
        fused words as int64}."""
        assert self.cfg.mode in ("local", "tree")
        return self.decode(ttnn.to_torch(ttnn.get_device_tensors(self.out)[device]))


# ---------------------------------------------------------------------------------------------
# The chip-local half of the cross-chip merge: the global cut -> this chip's winners
# ---------------------------------------------------------------------------------------------
CUT_KEEP_ALL_TIES = 0x7FFF


def cut_by_count(keep: int) -> int:
    """Cut word: the chip keeps its ``keep`` best entries."""
    return (1 << 31) | int(keep)


def cut_by_threshold(threshold: float, ties: int = CUT_KEEP_ALL_TIES) -> int:
    """Cut word: the chip keeps every entry with score > threshold and its first ``ties`` entries (rank order)
    with score == threshold (a BF16 value). With all ties kept the count answers "how many scores >= t"."""
    bits = int(torch.tensor(threshold).bfloat16().view(torch.int16)) & 0xFFFF
    assert float(torch.tensor(bits).to(torch.int16).view(torch.bfloat16)) == float(threshold), "not a BF16 value"
    return (int(ties) << 16) | bits


class Cut:
    """Given every user's cut, write the table the sparse row reader takes (fused/sparse_mla/layout.py
    selection_table): uint32 [1, 1, 1 + users, 2048] ROW_MAJOR, row 0 = counts, row 1 + u = user u's stripe rows,
    best first. Runs on the users' root scan cores of a ``Scan`` in mode "tree" and reads its output."""

    def __init__(self, scan: Scan):
        assert scan.cfg.mode == "tree"
        self.scan, self.mesh_device, self.cfg = scan, scan.mesh_device, scan.cfg
        U = self.cfg.users
        self.cuts = ttnn.from_torch(
            torch.zeros(1, 1, 1, 16, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.table = ttnn.from_torch(
            torch.zeros(1, 1, 1 + U, TOPK, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._descriptor = None

    def set_cuts(self, words):
        """words[u]: cut_by_count(...) or cut_by_threshold(...). In place (a trace follows)."""
        value = torch.zeros(16, dtype=torch.int64)
        value[: self.cfg.users] = torch.as_tensor(words, dtype=torch.int64)
        value = torch.where(value >= 1 << 31, value - (1 << 32), value).to(torch.int32).reshape(1, 1, 1, 16)
        host = ttnn.from_torch(value, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, self.cuts)

    def descriptor(self):
        scan, cfg, mesh = self.scan, self.cfg, self.mesh_device
        user_cores = [scan.core_of(u, 0) for u in range(cfg.users)]
        cores = _core_set(user_cores)
        collector = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*user_cores[0]))

        def cb(index, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.uint32, page_size=page)
                ],
            )

        device_index = {
            coord: i for i, coord in enumerate((r, c) for r in range(mesh.shape[0]) for c in range(mesh.shape[1]))
        }
        survivors, cuts, table = (ttnn.get_device_tensors(t) for t in (scan.out, self.cuts, self.table))
        mesh_desc = ttnn.MeshProgramDescriptor()
        for coord in scan.coords:
            d = device_index[coord]
            rt = ttnn.RuntimeArgs()
            for user, (x, y) in enumerate(user_cores):
                rt[x][y] = [
                    user,
                    survivors[d].buffer_address(),
                    user * BANKS * 2,
                    cuts[d].buffer_address(),
                    table[d].buffer_address(),
                    collector.x,
                    collector.y,
                ]
            desc = ttnn.ProgramDescriptor(
                kernels=[
                    ttnn.KernelDescriptor(
                        kernel_source=f"{KERNELS}/cut.cpp",
                        core_ranges=cores,
                        compile_time_args=[cfg.users, cfg.chunks_per_bank.bit_length() - 1],
                        runtime_args=rt,
                        config=ttnn.WriterConfigDescriptor(),
                        compiler_include_paths=[str(KERNEL_DIR)],
                    )
                ],
                semaphores=[
                    ttnn.SemaphoreDescriptor(id=u, core_ranges=cores, initial_value=0) for u in range(cfg.users)
                ],
                cbs=[cb(0, WORD_TILE_BYTES, 2), cb(1, TOPK * 4, 1), cb(2, 64, 1)],
            )
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))] = desc
        return mesh_desc

    def _io(self):
        return [self.scan.out, self.cuts, self.table]

    def compile(self):
        ttnn.experimental.prepare_generic_op(self._io(), self.descriptor())

    def __call__(self):
        if self._descriptor is None:
            self._descriptor = self.descriptor()
        return ttnn.generic_op(self._io(), self._descriptor)

    def read(self, device: int = 0):
        """(counts int64 [U], rows int64 [U, 2048]; only the first counts[u] entries of a row mean anything)."""
        t = ttnn.to_torch(ttnn.get_device_tensors(self.table)[device]).to(torch.int64) & 0xFFFFFFFF
        t = t.reshape(1 + self.cfg.users, TOPK)
        return t[0, : self.cfg.users], t[1:]
