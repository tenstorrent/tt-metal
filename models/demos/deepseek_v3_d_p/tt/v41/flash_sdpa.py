# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The decode step's SDPA as a flash-decode program over the tile's live rows (``V41_DECODE_SDPA=flash``).

``sparse_sdpa`` gives each of the tile's 32 query rows one core, which walks the row's selected keys in chunks
of
128 serially: on the one-row (c1) step the live row's core takes ~83 us a layer while 129 cores idle.
:func:`flash_sdpa` takes the same operands (the ``q_heads`` output ``[1, H, S, D]`` row-major
bf16 with the chip's heads padded to a tile, the layer's KV tensor in either served format, the index rows
``[1, 1, S, W]`` uint32 with a contiguous sentinel tail per row, the per-head sinks) and the same contract,
but
splits every live row's keys into units of ``kpu`` keys spread over the grid (``kernels/fsdpa_*.cpp``):

* phase 0: every core reads the live page (``[1, 1, 1, S]`` uint32: row ``r`` is live unless word ``r`` is
  ``LIVE_SENTINEL``; the c1 step's ``c1put`` page, ``FlashSdpa.all_live`` for every row) and agrees on the
  live rows
  and so on the same unit and merge assignment;
* a unit ``(row, j)`` on core ``(row rank * UMAX + j) % N`` reads the row's query heads and the keys
  ``[j * kpu, (j + 1) * kpu)`` of its index row (zero rows and ``-inf`` scores past the row's valid count, as
  the
  kernel it replaces), computes the unnormalized partial ``(max, sum tile, P @ V)`` of those keys in one pass
  (``sparse_sdpa``'s chunk arithmetic: the scaled-FP8 decode, the latent and RoPE products, the running-max
  softmax at the op's scale), writes its max, sum and column-``v`` acc tiles straight into the L1 partial CBs
  of
  each of the row's ``VT`` merge cores (``sdpa_decode``'s reducer exchange: no DRAM round trip, the write
  barrier before the semaphore increment orders the data), then signals the row's mergers;
* a merge ``(row, v)`` for each 32-column tile ``v`` of the output on core ``N - 1 - (row rank * VT + v) % N``
  (its ``k``-th merge in slot ``k = (row rank * VT + v) // N`` of its partial CBs, ``k < nsem``) waits for the
  row's partials, combines them by the log-sum-exp rule with the sink in the denominator (``exp((m_j - M) *
  scale)``
  weights, as ``sparse_sdpa``'s SALAD correction and sink fold compute them) and writes the 32 heads' columns
  to
  the row-major output. The rows that are not live get zero output rows.

``max_live`` bounds the live rows a call may carry (``nsem = ceil(max_live * VT / N)`` partial slots a core:
120 KB of L1 a slot); the decoder's c1 capture needs ``1 + K`` and sizes one slot. ``debug_scratch`` makes the
writer also copy every partial to a DRAM scratch (``FlashSdpa.scratch``) for the test's per-unit diagnostic.

A row's ``1 + K`` verify rows (the speculative branch's design) are ``1 + K`` live rows of the tile, each with
its
own index row, so they need nothing beyond this. The output is not bit-identical to ``sparse_sdpa``'s: the
keys'
accumulation order differs (``tests/v41/test_v41_flash_sdpa.py`` measures the difference).
"""

import os

import ttnn

_KERNELS = "models/demos/deepseek_v3_d_p/tt/v41/kernels"
_TILE = 32
_BF16_TILE = 2048
_FP32_TILE = 4096
_BFP8_TILE = 1088
SENTINEL = 0xFFFFFFFF
LIVE_SENTINEL = 32  # the c1 one-hot pages' sentinel (moe_c1._SENTINEL): a row whose live-page word is this is not live
SCALE_BLOCK_WIDTH = 128  # sparse_sdpa_common.hpp
DRAM_ALIGN = 64  # Blackhole DRAM page alignment: the packed KV row's physical stride

# CB ids, the single source for the three kernels (kernels/fsdpa_common.hpp mirrors it)
CB = dict(
    q_rm=0,
    q_in=1,
    k_rm=2,
    k_in=3,
    neginf=4,
    mask_part=5,
    scale=6,
    qk_im=7,
    max=8,
    sum=9,
    out=10,
    ctrl_c=11,
    ctrl_w=12,
    col_identity=13,
    k_rope_rm=14,
    k_scale_bcast=15,
    k_latent_tile=16,
    k_rope_tile=17,
    sink=18,
    pm=19,
    ps=20,
    pa=21,
    mM=22,
    mw=23,
    mws=24,
    mS=25,
    mrecip=26,
    mA=27,
    max_w=28,
    o_rm=29,
    rscratch=30,
    wscratch=31,
)
KPU_DEFAULT = int(os.environ.get("V41_FLASH_KPU", "32"))


def _align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


class FlashSdpa:
    """The program's per-device constants: the grid's NoC coordinate table (a uint32 tensor the writer reads
    to
    address the mergers' partial CBs and semaphores), the all-live page and, under ``debug_scratch``, the DRAM
    partial scratch. Build one before any capture (a persistent tensor allocated after a
    capture may be overwritten by its replay). A call with more live rows than ``max_live`` computes the first
    ``max_live`` and zeroes the rest (the reader clamps the plan)."""

    def __init__(
        self,
        device,
        heads: int = 32,
        rows: int = _TILE,
        width: int = 640,
        v_dim: int = 512,
        kpu=None,
        real_heads=None,
        max_live=None,
        debug_scratch=None,
    ):
        """``heads``: the query tile's heads (32); ``real_heads``: the leading ones that are not zero padding
        (the
        chip's 16; the others are not read); ``max_live``: the live rows a call may carry (default every row);
        ``debug_scratch``: the DRAM copy of the partials (default ``V41_FLASH_DEBUG_SCRATCH``)."""
        import torch

        self.device = device
        grid = device.compute_with_storage_grid_size()
        self.gx, self.gy = grid.x, grid.y
        self.n_cores = self.gx * self.gy
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.gx - 1, self.gy - 1))])
        noc = [
            device.worker_core_from_logical_core(ttnn.CoreCoord(i % self.gx, i // self.gx)) for i in range(self.n_cores)
        ]
        table = torch.tensor([[c.x, c.y] for c in noc], dtype=torch.int32).reshape(1, 1, 1, -1)
        self.noc_table = ttnn.from_torch(
            table,
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        self.kpu = KPU_DEFAULT if kpu is None else kpu
        assert self.kpu % _TILE == 0 and self.kpu <= 160, self.kpu
        self.heads, self.rows, self.v_dim = heads, rows, v_dim
        self.real_heads = heads if real_heads is None else real_heads
        assert 0 < self.real_heads <= heads, (self.real_heads, heads)
        self.vt = v_dim // _TILE
        self.umax = -(-width // self.kpu)
        self.max_live = rows if max_live is None else max_live
        assert 0 < self.max_live <= rows, (self.max_live, rows)
        self.nsem = -(-self.max_live * self.vt // self.n_cores)
        # the debug scratch: slot (row rank, unit) holds the max tile, the sum tile and the VT acc tiles
        self.slot_tiles = 2 + self.vt
        self.debug_scratch = (
            os.environ.get("V41_FLASH_DEBUG_SCRATCH", "0") == "1" if debug_scratch is None else bool(debug_scratch)
        )
        self.scratch = (
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, rows * self.umax * self.slot_tiles * _TILE, _TILE]),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            if self.debug_scratch
            else None
        )
        # the live page of a call without one: every row live
        self.all_live = ttnn.from_torch(
            torch.zeros(1, 1, 1, rows, dtype=torch.int32),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    def live_page(self, live_rows):
        """A live page for the tile rows ``live_rows`` (the same on every chip; the tests')."""
        import torch

        page = torch.full((1, 1, 1, self.rows), LIVE_SENTINEL, dtype=torch.int32)
        page[0, 0, 0, list(live_rows)] = 0
        return ttnn.from_torch(
            page,
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )

    def __call__(self, q, kv, rows, scaled: bool, scale: float, sink, live=None):
        return flash_sdpa(self, q, kv, rows, scaled, scale, sink, live)


def flash_sdpa(fs: FlashSdpa, q, kv, rows, scaled: bool, scale: float, sink, live=None):
    """``sparse_sdpa(q, kv, rows, v_dim, kv_format, scale, attention_sink=sink)`` as the flash-decode program:
    ``q`` ``[1, H, S, D]`` row-major bf16 (``H`` one tile of heads), ``kv`` the KV tensor (``[B, 1, T, D]``
    bf16 rows
    or the packed scaled-FP8 rows, ``scaled``), ``rows`` ``[1, 1, S, W]`` uint32 row-major, ``sink`` ``[1, 1,
    1, H]``
    bf16 row-major, ``live`` ``[1, 1, 1, S]`` uint32 row-major (row ``r`` live unless word ``r`` is
    ``LIVE_SENTINEL``; default every row) -> ``[1, H, S, V]`` row-major bf16: the live rows' attention, the
    other
    rows zero."""
    import struct

    H, S, D = q.shape[1], q.shape[2], q.shape[3]
    W = rows.shape[3]
    V = fs.v_dim
    assert q.dtype == ttnn.bfloat16 and q.layout == ttnn.ROW_MAJOR_LAYOUT and q.shape[0] == 1, (q.dtype, q.shape)
    assert H == _TILE and S == fs.rows and W <= fs.umax * fs.kpu, (H, S, W, fs.umax, fs.kpu)
    assert rows.dtype == ttnn.uint32 and rows.layout == ttnn.ROW_MAJOR_LAYOUT, (rows.dtype, rows.layout)
    assert sink.dtype == ttnn.bfloat16 and tuple(sink.shape) == (1, 1, 1, H), (sink.dtype, tuple(sink.shape))
    assert kv.layout == ttnn.ROW_MAJOR_LAYOUT and D % _TILE == 0 and V % _TILE == 0 and V <= D
    kv_is_fp8 = kv.dtype == ttnn.fp8_e4m3
    assert scaled == kv_is_fp8 or not scaled, (scaled, kv.dtype)  # the served formats: bf16 rows or scaled fp8
    DHt, vDHt, rope_DHt = D // _TILE, V // _TILE, (D - V) // _TILE
    KPU = fs.kpu
    Skt = KPU // _TILE
    UMAX = fs.umax
    N = fs.n_cores
    kv_elem = kv.element_size()
    q_row_bytes = D * 2
    k_row_bytes = kv.shape[3] * kv_elem
    scale_blocks = V // SCALE_BLOCK_WIDTH if scaled else 0
    packed_row_bytes = _align(k_row_bytes, DRAM_ALIGN) if scaled else 0
    if scaled:
        assert kv.shape[3] == V + scale_blocks * 4 + (D - V) * 2, (tuple(kv.shape), V, D)
    scale_bits = struct.unpack("<I", struct.pack("<f", scale))[0]
    fp32_acc = kv_is_fp8
    k_in_fmt = ttnn.bfloat8_b if kv_is_fp8 else ttnn.bfloat16
    k_in_tile = _BFP8_TILE if kv_is_fp8 else _BF16_TILE
    k_in_width = vDHt if scaled else DHt
    device = q.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, H, S, V]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    common = [H, S, W, D, V, KPU, UMAX, N, int(scaled), packed_row_bytes, 2, kv_elem, scale_bits, 1]
    common += [1, fs.nsem, fs.vt, scale_blocks, int(kv_is_fp8), fs.slot_tiles, k_row_bytes, LIVE_SENTINEL]
    common += [fs.real_heads, int(fs.debug_scratch)]
    scratch = fs.scratch if fs.debug_scratch else out  # the writer's scratch accessor: unused without the flag
    live = fs.all_live if live is None else live
    assert live.dtype == ttnn.uint32 and live.layout == ttnn.ROW_MAJOR_LAYOUT and tuple(live.shape) == (1, 1, 1, S)
    acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()
    rt = ttnn.RuntimeArgs()
    for i in range(N):
        rt[i % fs.gx][i // fs.gx] = [i]
    compute = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32_acc, math_approx_mode=True
    )
    if kv_is_fp8:
        modes = [ttnn.UnpackToDestMode.Default] * 64  # one entry per hardware CB slot (NUM_CIRCULAR_BUFFERS)
        modes[CB["k_rm"]] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute.unpack_to_dest_mode = modes
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNELS}/fsdpa_reader.cpp",
            core_ranges=fs.cores,
            compile_time_args=common + acc(q) + acc(kv) + acc(rows) + acc(live),
            runtime_args=rt,
            common_runtime_args=[
                q.buffer_address(),
                kv.buffer_address(),
                rows.buffer_address(),
                live.buffer_address(),
            ],
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNELS}/fsdpa_writer.cpp",
            core_ranges=fs.cores,
            compile_time_args=common + acc(scratch) + acc(out) + acc(sink) + acc(fs.noc_table),
            runtime_args=rt,
            common_runtime_args=[
                scratch.buffer_address(),
                out.buffer_address(),
                sink.buffer_address(),
                fs.noc_table.buffer_address(),
            ],
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNELS}/fsdpa_compute.cpp",
            core_ranges=fs.cores,
            compile_time_args=common,
            runtime_args=rt,
            config=compute,
        ),
    ]

    def cb(index, fmt, pages, page, alias=None):
        fds = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)]
        if alias is not None:
            fds.append(ttnn.CBFormatDescriptor(buffer_index=alias[0], data_format=alias[1], page_size=page))
        return ttnn.CBDescriptor(total_size=pages * page, core_ranges=fs.cores, format_descriptors=fds)

    bf = ttnn.bfloat16
    ctrl_page = 64 + S * 4  # the plan page {n_live, live[S]}; the unit / merge pages use the first word
    cbs = [
        cb(CB["q_rm"], bf, H, q_row_bytes),
        cb(CB["q_in"], bf, DHt, _BF16_TILE),
        cb(CB["k_in"], k_in_fmt, Skt * k_in_width, k_in_tile),
        cb(CB["neginf"], bf, 1, _BF16_TILE),
        cb(CB["mask_part"], bf, 1, _BF16_TILE),
        cb(CB["scale"], bf, 1, _BF16_TILE),
        cb(CB["qk_im"], bf, Skt, _BF16_TILE),
        cb(CB["max"], bf, 1, _BF16_TILE),
        cb(CB["sum"], bf, 1, _BF16_TILE),
        cb(CB["out"], bf, vDHt, _BF16_TILE),
        cb(CB["ctrl_c"], bf, 4, ctrl_page),
        cb(CB["ctrl_w"], bf, 4, ctrl_page),
        cb(CB["col_identity"], bf, 1, _BF16_TILE),
        cb(CB["sink"], bf, 1, _BF16_TILE),
        # the partial CBs: nsem slots of UMAX pages, written by the units' writers on other cores
        cb(CB["pm"], bf, fs.nsem * UMAX, _BF16_TILE),
        cb(CB["ps"], bf, fs.nsem * UMAX, _BF16_TILE),
        cb(CB["pa"], bf, fs.nsem * UMAX, _BF16_TILE),
        cb(CB["mM"], bf, 2, _BF16_TILE),  # the running max ping-pong, then the final tile
        cb(CB["mw"], bf, UMAX, _BF16_TILE),  # taken whole per merge
        cb(CB["mws"], bf, 1, _BF16_TILE),
        cb(CB["mS"], bf, 1, _BF16_TILE),
        cb(CB["mrecip"], bf, 1, _BF16_TILE),
        cb(CB["mA"], bf, 1, _BF16_TILE),
        cb(CB["max_w"], bf, 1, _BF16_TILE),
        cb(CB["o_rm"], bf, 1, _BF16_TILE),
        # reader scratch: the live page, one index row (64-aligned each), the live list
        cb(CB["rscratch"], ttnn.uint32, 1, _align(S * 4, 64) + _align(W * 4, 64) + 256),
        # writer scratch: the NoC table, the sink row, a zero row-major output row (1 KB)
        cb(CB["wscratch"], ttnn.uint32, 1, _align(N * 8, 64) + 64 + V * 2 + 256),
    ]
    if scaled:
        # the packed rows' FIFO; with a RoPE tail (D > V) a format-only bf16 alias of the same pages
        # (sparse_sdpa)
        rope_alias = (CB["k_rope_rm"], bf) if rope_DHt > 0 else None
        cbs += [
            cb(CB["k_rm"], ttnn.fp8_e4m3, _TILE * 2, packed_row_bytes, alias=rope_alias),
            cb(CB["k_scale_bcast"], ttnn.float32, scale_blocks * 2, _FP32_TILE),
            cb(CB["k_latent_tile"], ttnn.bfloat8_b, vDHt, _BFP8_TILE),
        ]
        if rope_DHt > 0:
            cbs.append(cb(CB["k_rope_tile"], bf, Skt * rope_DHt, _BF16_TILE))
    else:
        assert kv.dtype == ttnn.bfloat16, kv.dtype  # the served plain format is bf16 rows
        cbs += [cb(CB["k_rm"], bf, KPU, k_row_bytes)]
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=fs.cores, initial_value=0) for i in range(fs.nsem)]
    ttnn.generic_op(
        [q, kv, rows, sink, live, fs.noc_table] + ([fs.scratch] if fs.debug_scratch else []) + [out],
        ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs),
    )
    return out
