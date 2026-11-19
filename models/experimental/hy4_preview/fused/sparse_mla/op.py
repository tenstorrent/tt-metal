# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program descriptors of the sparse-MLA piece, launched with ``ttnn.generic_op`` on one chip.

    read_rows   the row readers and converters alone: selected packed rows -> BF16 K tiles in a tensor
    attend      rows by id from DRAM -> partial (O, max, sum) per user and head (layout.py states the contract)
    append      one new packed row per user at a given position

Cores (layout.sblocks): 8 S-blocks of 2 reader / converter cores and 8 attention cores (8 heads each). Chunk c
(128 list entries) of user u is served by S-block (u + c) % 8, so 8 users with at most 128 rows each run side by
side; a user's S-blocks merge their partials along a tree and the user's partial ends on S-block u.

Interfaces to the other pieces of the decode layer
---------------------------------------------------
kv    (from prefill, and from ``append``) the prefill stack's packed KV storage of this chip,
      [slots, 1, R, 656] fp8_e4m3 ROW_MAJOR, ND-sharded [1, 1, 32, 656] round-robin over the DRAM banks;
      slot = user * num_layers + layer, ``layer`` is a common runtime argument of the readers.
sel   (from the selection: pieces 4 and 6) uint32 [1, 1, 1 + users, topk] ROW_MAJOR, any buffer a TensorAccessor
      reads: row 0 = how many of each user's picks this chip holds, row 1 + u = their rows in this chip's slab
      of the user's slot (row(p) = (p // 4096) * 128 + p % 128), valid entries first, any order, no holes. It
      must be complete when the program starts.
q     (from the query all-to-all: piece 7) page (u, g) = the absorbed queries of heads 8g .. 8g + 7 of user u:
      reference/attention_decode.py ``latent_queries`` (q_nope W_k | q_rope), [8, 576] BF16 as 18 tiles of
      [8, 32], in the cache's frame (tt/frames.py tail_to_device) and NOT scaled (the softmax scale 1/16 is
      applied in the exponent). Read once by attention core (b, g) for every user S-block b has a chunk of.
out   (to the cross-chip merge: piece 6) page (u, g) = the partial of user u, heads 8g .. 8g + 7: O as 16 tiles
      of [8, 32] BF16 (unnormalised, latent space), then one [8, 32] tile with max in column 0 and sum in column
      1 (max of the UNSCALED logits), then a status page. Written by attention core (u, g), which also holds it
      in its cb_final_o / cb_final_ms until it is written: a fused merge can take it from there. The sink is not
      in it. A user without rows on the chip gets O = 0, sum = 0, max = layout.EMPTY_MAX.
append: ``rows`` are packed rows (656 bytes) as the prefill stack packs them (per_token_cast_to_fp8 with
      power-of-two scales, pack_scaled_fp8_kv_cache); ``positions`` come from a device tensor.
"""

from __future__ import annotations

import struct
from pathlib import Path

import torch

import ttnn
import ttnn._ttnn.program_descriptor as _pd
from models.experimental.hy4_preview.fused.sparse_mla.layout import (
    EMPTY_MAX_BF16,
    TILE,
    X0_LEFT,
    X0_RIGHT,
    SparseMlaConfig,
    pack_tiny,
    sblocks,
    unpack_tiny,
)

KERNEL_DIR = "models/experimental/hy4_preview/fused/sparse_mla/kernels/"
assert (Path(__file__).parent / "kernels").is_dir()

# CB ids of a reader core. R_K is the attention cores' K buffer: declared on the readers too so that it has one L1
# address everywhere; a reader sends its converted slabs (R_KOUT) into it.
R_ROWS, R_K, R_SCALE, R_LAT, R_ROPE_RM, R_CTRL, R_IDX, R_HDR, R_KOUT = range(9)
# CB ids of an attention core (A_Q must be 0: the compute kernel's hardware startup is seeded from CB 0).
(A_Q, A_K, A_MASK, A_MS_IN, A_OUT_IN, A_SEND_O, A_SEND_MS, A_FINAL_O, A_FINAL_MS) = range(9)
(A_PART_O, A_PART_MS, A_TMP_O, A_TMP_MS, A_CTRL, A_HDR_IO, A_HDR_K) = range(9, 16)
assert A_K == R_K
SEM_READY, SEM_DATA, SEM_MERGE, SEM_ERR = 0, 1, 2, 5  # SEM_MERGE .. SEM_MERGE + 2: one per merge step
NUM_SEMAPHORES = 6
TINY_ROWS = 8
# Columns of the [8, 32] max/sum tile the flash-attention LLKs pack (row = head).
MS_MAX_COL, MS_SUM_COL = 0, 1
STATUS_MAGIC = float(struct.unpack("<f", struct.pack("<I", 0x4D4B << 16))[0])  # the bf16 0x4D4B a root core writes


def core_set(cores) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def _dm(processor, noc):
    return ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc)


def _cb(cores, index, dtype, page, pages, extra=()):
    formats = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)]
    formats += [ttnn.CBFormatDescriptor(buffer_index=i, data_format=d, page_size=page) for i, d in extra]
    return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cores, format_descriptors=formats)


def _tiny_cb(cores, index, pages):
    """A CB of [8, 32] BF16 tiles (512 bytes): the tile shape of the 8-head flash-attention LLKs."""
    tile = ttnn.Tile((TINY_ROWS, TILE))
    page = tile.get_tile_size(ttnn.bfloat16)
    fmt = ttnn.CBFormatDescriptor(
        buffer_index=index, data_format=ttnn.bfloat16, page_size=page, tile=ttnn.TileDescriptor(tile)
    )
    return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cores, format_descriptors=[fmt])


class SparseMla:
    def __init__(self, cfg: SparseMlaConfig, device):
        self.cfg = cfg
        self.device = device
        self.blocks = sblocks(cfg)
        grid = device.compute_with_storage_grid_size()
        noc = lambda c: device.worker_core_from_logical_core(ttnn.CoreCoord(*c))
        self.coords = []  # per S-block: reader 0 and attention group 0, NoC coordinates (kernels/sm_common.hpp)
        for blk in self.blocks:
            for x, y in (*blk.readers, *blk.attention):
                assert x < grid.x and y < grid.y, f"core ({x}, {y}) is outside the {grid.x}x{grid.y} worker grid"
            r0, a0 = noc(blk.readers[0]), noc(blk.attention[0])
            for v, core in enumerate(blk.readers):
                assert (noc(core).x, noc(core).y) == (r0.x, r0.y + v), "reader cores are not stacked in NoC space"
            for g, core in enumerate(blk.attention):
                assert (noc(core).x, noc(core).y) == (a0.x + g % 4, a0.y + g // 4), "attention cores: not 4x2"
            self.coords += [r0.x, r0.y, a0.x, a0.y]
        # both reader columns whole, also the cores that hold no reader: rectangles launch faster than single cores
        self.readers = core_set([(x0, y) for x0 in (X0_LEFT, X0_RIGHT) for y in range(cfg.groups)])
        self.attention = core_set([c for blk in self.blocks for c in blk.attention])
        self.both = core_set(
            [(x0, y) for x0 in (X0_LEFT, X0_RIGHT) for y in range(cfg.groups)]
            + [c for blk in self.blocks for c in blk.attention]
        )
        self.banks = device.dram_grid_size().x
        assert self.banks & (self.banks - 1) == 0
        # Global semaphores (persistent L1 words on every core of the piece), not program semaphores: the
        # dispatcher initialises a program semaphore on every core at every launch, about 4 us each on these
        # cores (tests/test_latency.py test_launch_shapes). The kernels leave them at zero.
        self.semaphores = [ttnn.create_global_semaphore(device, self.both, 0) for _ in range(NUM_SEMAPHORES)]
        self.sem = [ttnn.get_global_semaphore_address(s) for s in self.semaphores]

    def reset_semaphores(self):
        """After a launch that did not end cleanly (a wait expired): counters may be left non-zero."""
        for s in self.semaphores:
            ttnn.reset_global_semaphore_value(s, 0)

    # ---- shared pieces -------------------------------------------------------------------------------------
    def check_kv(self, kv):
        cfg = self.cfg
        assert kv.dtype == ttnn.fp8_e4m3 and kv.layout == ttnn.ROW_MAJOR_LAYOUT, (kv.dtype, kv.layout)
        assert tuple(kv.shape)[1:] == (1, cfg.rows_per_slot, cfg.row_bytes), tuple(kv.shape)
        spec = kv.memory_config().nd_shard_spec
        assert spec is not None and list(spec.shard_shape) == [1, 1, cfg.shard_rows, cfg.row_bytes], spec
        assert cfg.rows_per_slot % cfg.shard_rows == 0

    @property
    def idx_scratch_bytes(self) -> int:
        """The reader's scratch page: ids of the current chunk | ids of an early-fetched chunk | header."""
        return 2 * self.cfg.chunk_rows * 4 + 64

    def _reader_named(self):
        cfg = self.cfg
        return [
            ("cb_rows", R_ROWS),
            ("cb_k", R_K),
            ("cb_kout", R_KOUT),
            ("cb_scale", R_SCALE),
            ("cb_lat", R_LAT),
            ("cb_rope_rm", R_ROPE_RM),
            ("cb_ctrl", R_CTRL),
            ("cb_idx", R_IDX),
            ("cb_hdr", R_HDR),
            ("users", cfg.users),
            ("groups", cfg.groups),
            ("converters", cfg.converters),
            ("head_groups", cfg.head_groups),
            ("chunk_rows", cfg.chunk_rows),
            ("topk", cfg.topk),
            ("row_stride", cfg.row_stride),
            ("latent", cfg.latent),
            ("rope", cfg.rope),
            ("dht", cfg.dht),
            ("scale_blocks", cfg.scale_blocks),
            ("rows_per_slot", cfg.rows_per_slot),
            ("slot_stride", cfg.num_layers),
            ("banks", self.banks),
            ("shard_rows", cfg.shard_rows),
            ("x0_left", X0_LEFT),
            ("x0_right", X0_RIGHT),
            ("sem_ready", self.sem[SEM_READY]),
            ("sem_data", self.sem[SEM_DATA]),
            ("sem_err", self.sem[SEM_ERR]),
            ("tbl_mcast", 1),  # index of the coordinate table in the sender's common runtime arguments
        ]

    def _reader_cbs(self, readers):
        cfg = self.cfg
        bf16, fp32 = ttnn.bfloat16, ttnn.float32
        own = -(-cfg.slabs // cfg.converters)  # slabs of a chunk one converter makes
        return [
            # two 32-row slabs of packed rows; the BF16 id is a second view of the same memory for the RoPE field
            _cb(readers, R_ROWS, ttnn.fp8_e4m3, cfg.row_stride, 2 * TILE, extra=[(R_ROPE_RM, bf16)]),
            _cb(readers, R_KOUT, bf16, 2048, 2 * own * cfg.dht),
            _cb(readers, R_SCALE, fp32, 4096, 2 * cfg.scale_blocks),
            _cb(readers, R_LAT, bf16, 2048, cfg.vdht),
            _cb(readers, R_CTRL, bf16, 16, 4),
            _cb(readers, R_IDX, bf16, self.idx_scratch_bytes, 1),
            _cb(readers, R_HDR, bf16, 64, 1),
        ]

    def _reader_kernels(self, kv, sel, layer, fidelity=ttnn.MathFidelity.LoFi, defines=()):
        """The row reader and the converter on every reader core. ``fidelity`` is the converter's scale multiply:
        a power-of-two scale times a 4-bit significand is exact at any fidelity (tests/test_read_rows.py)."""
        named = self._reader_named()
        # an FP8 CB on the core requires a 32-bit DEST (program.cpp: "Fp8_e4m3 / Lf8 require fp32_dest_acc_en")
        compute = ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True, math_approx_mode=False)
        modes = [_pd.UnpackToDestMode.Default] * 64
        modes[R_ROWS] = _pd.UnpackToDestMode.UnpackToDestFp32  # FP8 unpacks into a 32-bit DEST
        compute.unpack_to_dest_mode = _pd.VectorUnpackToDestMode(modes)
        return [
            ttnn.KernelDescriptor(
                kernel_source=KERNEL_DIR + "sm_reader.cpp",
                core_ranges=self.readers,
                compile_time_args=list(ttnn.TensorAccessorArgs(sel).get_compile_time_args()),
                named_compile_time_args=named,
                defines=list(defines),
                common_runtime_args=[kv.buffer_address(), sel.buffer_address(), layer],
                config=_dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KERNEL_DIR + "sm_convert.cpp",
                core_ranges=self.readers,
                named_compile_time_args=named,
                defines=list(defines),
                config=compute,
            ),
        ]

    # ---- the reader alone ------------------------------------------------------------------------------------
    def read_rows_program(self, kv, sel, out, layer: int, drop: bool = False, fidelity=ttnn.MathFidelity.LoFi):
        """out: [1, 1, users * topk, width] BF16 TILE in DRAM; row ``u * topk + i`` receives list entry i of
        user u (only slabs that hold a valid entry are written). ``drop``: read and convert, write nothing
        (stage timing)."""
        self.check_kv(kv)
        kernels = self._reader_kernels(kv, sel, layer, fidelity=fidelity)
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=KERNEL_DIR + "sm_dump.cpp",
                core_ranges=self.readers,
                defines=[("SM_DROP", "1")] if drop else [],
                compile_time_args=list(ttnn.TensorAccessorArgs(sel).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
                named_compile_time_args=self._reader_named(),
                common_runtime_args=[sel.buffer_address(), out.buffer_address()],
                config=_dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
            )
        )
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=self._reader_cbs(self.readers))

    def read_only_program(self, kv, sel, layer: int, cores):
        """The row reader with nothing behind it, on any ``cores``: every chunk's rows are read from DRAM into L1
        and dropped. Chunk c of user u goes to core (u + c) % len(cores). Measures the row read rate."""
        cfg = self.cfg
        self.check_kv(kv)
        readers = core_set(cores)
        override = {"groups": len(cores), "converters": 1}
        named = [(k, override.get(k, v)) for k, v in self._reader_named()]
        rt = ttnn.RuntimeArgs()
        for b, (x, y) in enumerate(cores):
            rt[x][y] = [b]
        kernel = ttnn.KernelDescriptor(
            kernel_source=KERNEL_DIR + "sm_reader.cpp",
            core_ranges=readers,
            compile_time_args=list(ttnn.TensorAccessorArgs(sel).get_compile_time_args()),
            named_compile_time_args=named,
            defines=[("SM_READ_ONLY", "1"), ("SM_ROLE_FROM_ARGS", "1")],
            runtime_args=rt,
            common_runtime_args=[kv.buffer_address(), sel.buffer_address(), layer],
            config=_dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
        )
        cbs = [
            _cb(readers, R_ROWS, ttnn.bfloat16, cfg.row_stride, 2 * TILE),
            _cb(readers, R_SCALE, ttnn.float32, 4096, cfg.scale_blocks),
            _cb(readers, R_CTRL, ttnn.bfloat16, 16, 4),
            _cb(readers, R_IDX, ttnn.bfloat16, self.idx_scratch_bytes, 1),
        ]
        return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)

    def read_rows(self, kv, sel, out, layer: int):
        return ttnn.generic_op([kv, sel, out], self.read_rows_program(kv, sel, out, layer))

    # ---- attention -------------------------------------------------------------------------------------------
    @property
    def q_page_values(self) -> int:
        return self.cfg.dht * TINY_ROWS * TILE

    @property
    def out_page_values(self) -> int:
        """O tiles | max/sum tile | 64 bytes of status, in BF16 values."""
        return (self.cfg.vdht + 1) * TINY_ROWS * TILE + 32

    def attend_program(
        self,
        kv,
        sel,
        q,
        out,
        layer: int,
        fidelity=ttnn.MathFidelity.LoFi,
        convert_fidelity=ttnn.MathFidelity.LoFi,
        nop: bool = False,
    ):
        """q:   [1, 1, users * head_groups, q_page_values] BF16 ROW_MAJOR; page (u, g) = the absorbed queries of
                head group g of user u as DHt [8, 32] tiles (queries_to_pages).
        out: [1, 1, users * head_groups, out_page_values] BF16 ROW_MAJOR; page (u, g) = the partial of user u,
                heads 8g .. 8g + 7 (pages_to_partials).
        ``fidelity`` reaches the merges of partials only: the two matmuls of the flash-attention LLKs run their
        MVMUL at LoFi whatever the kernel's fidelity. ``nop``: every kernel returns at once (the launch alone)."""
        cfg = self.cfg
        self.check_kv(kv)
        assert cfg.users == cfg.groups, "user u's partial ends on S-block u"
        assert tuple(q.shape) == (1, 1, cfg.users * cfg.head_groups, self.q_page_values), tuple(q.shape)
        assert tuple(out.shape) == (1, 1, cfg.users * cfg.head_groups, self.out_page_values), tuple(out.shape)
        att, both = self.attention, self.both
        defines = [("SM_NOP", "1")] if nop else []

        named_a = [
            ("cb_q", A_Q),
            ("cb_k", A_K),
            ("cb_mask", A_MASK),
            ("cb_ms_in", A_MS_IN),
            ("cb_out_in", A_OUT_IN),
            ("cb_send_o", A_SEND_O),
            ("cb_send_ms", A_SEND_MS),
            ("cb_final_o", A_FINAL_O),
            ("cb_final_ms", A_FINAL_MS),
            ("cb_part_o", A_PART_O),
            ("cb_part_ms", A_PART_MS),
            ("cb_tmp_o", A_TMP_O),
            ("cb_tmp_ms", A_TMP_MS),
            ("cb_ctrl_a", A_CTRL),
            ("cb_hdr_io", A_HDR_IO),
            ("cb_hdr_k", A_HDR_K),
            ("users", cfg.users),
            ("groups", cfg.groups),
            ("converters", cfg.converters),
            ("head_groups", cfg.head_groups),
            ("chunk_rows", cfg.chunk_rows),
            ("dht", cfg.dht),
            ("vdht", cfg.vdht),
            ("x0_left", X0_LEFT),
            ("x0_right", X0_RIGHT),
            ("sem_ready", self.sem[SEM_READY]),
            ("sem_data", self.sem[SEM_DATA]),
            ("sem_err", self.sem[SEM_ERR]),
            ("sem_merge0", self.sem[SEM_MERGE]),
            ("sem_merge1", self.sem[SEM_MERGE + 1]),
            ("sem_merge2", self.sem[SEM_MERGE + 2]),
            ("tbl_krecv", 1),  # index of the coordinate table in the kernel's common runtime arguments
            ("tbl_io", 3),
            ("scale_fp32", int.from_bytes(struct.pack("<f", cfg.scale), "little")),
            ("dst_size", 8),
            ("empty_max_bf16", EMPTY_MAX_BF16),
        ]
        named_r = self._reader_named()

        users, steps = cfg.users, 3
        cbs = self._reader_cbs(self.readers) + [
            # two chunks of K tiles on every attention core; on the readers it only fixes the address
            _cb(both, A_K, ttnn.bfloat16, 2048, 2 * cfg.slabs * cfg.dht),
            _tiny_cb(att, A_Q, users * cfg.dht),
            _tiny_cb(att, A_MASK, users),
            _tiny_cb(att, A_MS_IN, users * steps),
            _tiny_cb(att, A_OUT_IN, users * steps * cfg.vdht),
            _tiny_cb(att, A_SEND_O, users * cfg.vdht),
            _tiny_cb(att, A_SEND_MS, users),
            _tiny_cb(att, A_FINAL_O, cfg.vdht),
            _tiny_cb(att, A_FINAL_MS, 1),
            _tiny_cb(att, A_PART_O, users * cfg.vdht),
            _tiny_cb(att, A_PART_MS, users),
            _tiny_cb(att, A_TMP_O, cfg.vdht),
            _tiny_cb(att, A_TMP_MS, 1),
            _cb(att, A_CTRL, ttnn.bfloat16, 64, 1),
            _cb(att, A_HDR_IO, ttnn.bfloat16, 64, 1),
            _cb(att, A_HDR_K, ttnn.bfloat16, 64, 1),
        ]

        sel_ct = list(ttnn.TensorAccessorArgs(sel).get_compile_time_args())
        io_ct = sel_ct + list(ttnn.TensorAccessorArgs(q).get_compile_time_args())
        io_ct += list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
        kernels = self._reader_kernels(kv, sel, layer, fidelity=convert_fidelity, defines=defines) + [
            ttnn.KernelDescriptor(
                kernel_source=KERNEL_DIR + "sm_mcast.cpp",
                core_ranges=self.readers,
                compile_time_args=sel_ct,
                named_compile_time_args=named_r,
                defines=defines,
                common_runtime_args=[sel.buffer_address()] + self.coords,
                config=_dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KERNEL_DIR + "sm_krecv.cpp",
                core_ranges=att,
                compile_time_args=sel_ct,
                named_compile_time_args=named_a,
                defines=defines,
                common_runtime_args=[sel.buffer_address()] + self.coords,
                config=_dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KERNEL_DIR + "sm_attn_io.cpp",
                core_ranges=att,
                compile_time_args=io_ct,
                named_compile_time_args=named_a,
                defines=defines,
                common_runtime_args=[sel.buffer_address(), q.buffer_address(), out.buffer_address()] + self.coords,
                config=_dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KERNEL_DIR + "sm_attn.cpp",
                core_ranges=att,
                named_compile_time_args=named_a,
                defines=defines,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=fidelity, fp32_dest_acc_en=False, math_approx_mode=False
                ),
            ),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    def attend(self, kv, sel, q, out, layer: int, **kwargs):
        return ttnn.generic_op([kv, sel, q, out], self.attend_program(kv, sel, q, out, layer, **kwargs))

    # ---- host side of the attention tensors -----------------------------------------------------------------------
    def queries_to_pages(self, q_lat: torch.Tensor) -> torch.Tensor:
        """Absorbed queries [users, heads, width] (reference latent_queries, one row per head) -> the q tensor's
        pages [1, 1, users * head_groups, q_page_values] bf16."""
        cfg = self.cfg
        assert tuple(q_lat.shape) == (cfg.users, cfg.heads, cfg.width), tuple(q_lat.shape)
        groups = q_lat.to(torch.bfloat16).reshape(cfg.users * cfg.head_groups, cfg.heads_per_core, cfg.width)
        return torch.stack([pack_tiny(g) for g in groups])[None, None]

    def pages_to_partials(self, pages: torch.Tensor):
        """The out tensor [1, 1, users * head_groups, out_page_values] -> (O [users, heads, latent],
        m [users, heads], s [users, heads], status [users, head_groups, 4]: STATUS_MAGIC, wait flags (0 = no wait
        expired on the core), receive slots published, partials sent). ``self.last_ms`` keeps the raw max/sum
        tiles [users, head_groups, 8, 32]."""
        cfg = self.cfg
        pages = pages.reshape(cfg.users * cfg.head_groups, self.out_page_values).to(torch.bfloat16)
        tile = TINY_ROWS * TILE
        o, m, s, status, raw = [], [], [], [], []
        for page in pages:
            o.append(unpack_tiny(page[: cfg.vdht * tile], TINY_ROWS))
            ms = unpack_tiny(page[cfg.vdht * tile : (cfg.vdht + 1) * tile], TINY_ROWS)
            raw.append(ms.float())
            m.append(ms[:, MS_MAX_COL])
            s.append(ms[:, MS_SUM_COL])
            status.append(page[(cfg.vdht + 1) * tile :][:4].float())
        shape = (cfg.users, cfg.heads)
        self.last_ms = torch.stack(raw).reshape(cfg.users, cfg.head_groups, TINY_ROWS, TILE)
        return (
            torch.stack(o).reshape(*shape, cfg.latent).float(),
            torch.stack(m).reshape(shape).float(),
            torch.stack(s).reshape(shape).float(),
            torch.stack(status).reshape(cfg.users, cfg.head_groups, 4),
        )

    # ---- single-row append -------------------------------------------------------------------------------------
    def append_program(self, kv, rows, positions, layer: int, chip: int, core=(5, 0), stripe_block=128, chips=32):
        """rows: [users, 1, 1, 656] fp8_e4m3 ROW_MAJOR, the packed row of each user's new token (the prefill
        stack's pack); positions: uint32 [1, 1, 1, users]; ``chip`` is this chip's linear index in the stripe
        (4 * mesh row + mesh column). The row of user u is written when this chip owns positions[u]."""
        cfg = self.cfg
        self.check_kv(kv)
        assert rows.dtype == ttnn.fp8_e4m3 and rows.layout == ttnn.ROW_MAJOR_LAYOUT
        assert tuple(rows.shape) == (cfg.users, 1, 1, cfg.row_bytes), tuple(rows.shape)
        assert tuple(positions.shape) == (1, 1, 1, cfg.users) and positions.dtype == ttnn.uint32
        cores = core_set([core])
        named = [
            ("cb_row", 0),
            ("cb_pos", 1),
            ("users", cfg.users),
            ("row_bytes", cfg.row_bytes),
            ("row_stride", cfg.row_stride),
            ("rows_per_slot", cfg.rows_per_slot),
            ("slot_stride", cfg.num_layers),
            ("banks", self.banks),
            ("shard_rows", cfg.shard_rows),
            ("stripe_block", stripe_block),
            ("stripe_chips", chips),
        ]
        rt = ttnn.RuntimeArgs()
        rt[core[0]][core[1]] = [chip]
        kernel = ttnn.KernelDescriptor(
            kernel_source=KERNEL_DIR + "sm_append.cpp",
            core_ranges=cores,
            compile_time_args=list(ttnn.TensorAccessorArgs(rows).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(positions).get_compile_time_args()),
            named_compile_time_args=named,
            runtime_args=rt,
            common_runtime_args=[kv.buffer_address(), rows.buffer_address(), positions.buffer_address(), layer],
            config=_dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
        )
        cbs = [
            _cb(cores, 0, ttnn.bfloat16, cfg.row_stride, cfg.users),
            _cb(cores, 1, ttnn.bfloat16, 64, 1),
        ]
        return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)

    def append(self, kv, rows, positions, layer: int, chip: int, **kwargs):
        return ttnn.generic_op([rows, positions, kv], self.append_program(kv, rows, positions, layer, chip, **kwargs))

    # ---- launch cost ---------------------------------------------------------------------------------------------
    def nop_program(self, cores, dm: int = 1, compute: bool = False, cbs: int = 0, semaphores: int = 0):
        """Kernels that return at once on ``cores`` (a CoreRangeSet): ``dm`` data-movement kernels (1 or 2), a
        compute kernel, ``cbs`` circular buffers of one [8, 32] tile and ``semaphores`` semaphores. Its traced
        time is what a program of that shape costs before any kernel does anything."""
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=KERNEL_DIR + "sm_nop.cpp",
                core_ranges=cores,
                config=_dm(processor, noc),
            )
            for processor, noc in [
                (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
                (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
            ][:dm]
        ]
        if compute:
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=KERNEL_DIR + "sm_nop.cpp",
                    core_ranges=cores,
                    config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi),
                )
            )
        return ttnn.ProgramDescriptor(
            kernels=kernels,
            semaphores=[ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, cores, 0) for i in range(semaphores)],
            cbs=[_tiny_cb(cores, i, 1) for i in range(cbs)],
        )
