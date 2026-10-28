# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 decode tile dense projections at the DRAM's rate (``V41_DECODE_DENSE=matvec``).

On the decode tile (one tile row of M) ``ttnn.linear`` reads the attention projections' bfp8 weights at 140 to
250
GB/s (``wo_b`` 44.5 us for 11.1 MB, ``wo_a`` 42 for 8.9, ``wq_a`` 12.5 for 1.7): the multicast
matmul's few in1 readers and its block pipeline are sized for many tile rows. :class:`DecodeDense` is the
one-row
form of the routed expert reader: every core of the grid streams the column tiles it owns over K (two
blocks in flight), the row block is read once and multicast from core 0, the dot product accumulates in fp32
in
DST over the whole K at the caller's fidelity (HiFi2 for bf16 x bfp8, the served ``DENSE_COMPUTE_CONFIG``'s),
the
output tile goes straight to its page. The kernels are ``kernels/dense_row_{reader,compute,writer}.cpp``.
Batched
weights (``wo_a``'s groups, ``[1, g, K, N]``) are g independent products on g row blocks of ``x``.
"""

import os

import ttnn

DECODE_DENSE = os.environ.get("V41_DECODE_DENSE", "linear")
assert DECODE_DENSE in ("linear", "matvec"), DECODE_DENSE
_KERNELS = "models/demos/deepseek_v3_d_p/tt/v41/kernels"
_TILE = 32
X_TILE, W_TILE, OUT_TILE = 2048, 1088, 2048  # bf16, bfp8, bf16 tiles
_FIDELITY = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2, "hifi4": ttnn.MathFidelity.HiFi4}


def column_memory_config(device, k: int):
    """A DRAM memory config whose shards are whole tile columns (``[k, 32]``, round-robin over the DRAM
    banks):
    every column's K tiles contiguous, so the reader fetches a block of them with one request."""
    dram = device.dram_grid_size()
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=ttnn.Shape([k, _TILE]),
            grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))]),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def is_column_sharded(w) -> bool:
    spec = w.memory_config().nd_shard_spec
    return spec is not None and tuple(int(d) for d in spec.shard_shape) == (int(w.padded_shape[-2]), _TILE)


class DecodeDense:
    """``y[1, B, 32, N] = x[1, B, 32, K] @ w[1, B, K, N]`` for a bf16 TILE DRAM-interleaved ``x`` of one tile
    row
    per batch and a bfp8 TILE ``w`` in DRAM (interleaved, or column-sharded by :func:`column_memory_config`),
    as
    ``ttnn.linear`` computes it on the decode tile, with every core doing the same number of tile matmuls: the
    work is cut into units of one column and one K segment (``S`` segments of ``L`` tiles a column), unit
    ``u = j * S + s`` on core ``u % n_cores``, the segments' fp32 partials summed on the column's owner (the
    kernels' headers have the dataflow)."""

    RING = 3  # weight blocks in the ring: two in flight beside the one the compute reads
    PART_TILE = 4096  # an fp32 tile

    def __init__(self, w, device, fidelity: str | None = None, segments: int | None = None):
        assert w.dtype == ttnn.bfloat8_b and w.layout == ttnn.TILE_LAYOUT, (w.dtype, w.layout)
        assert w.memory_config().buffer_type == ttnn.BufferType.DRAM, w.memory_config()
        self.col_contig = is_column_sharded(w)
        assert self.col_contig or not w.memory_config().is_sharded(), "DecodeDense: interleaved or column-sharded"
        shape = tuple(int(d) for d in w.padded_shape)
        assert len(shape) == 4 and shape[0] == 1, shape
        self.w, self.B, K, N = w, shape[1], shape[2], shape[3]
        assert K % _TILE == 0 and N % _TILE == 0, shape
        self.KT, self.NT = K // _TILE, N // _TILE
        self.K, self.N = K, N
        grid = device.compute_with_storage_grid_size()
        self.gx, self.gy = grid.x, grid.y
        self.n_cores = self.gx * self.gy
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.gx - 1, self.gy - 1))])
        noc = lambda x, y: device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        self.noc = [noc(i % self.gx, i // self.gx) for i in range(self.n_cores)]
        c00, c11 = self.noc[0], self.noc[-1]
        self.rect = [c00.x, c00.y, c11.x, c11.y]
        self.device = device
        self.XT = self.B * self.KT
        self.X_SENDERS = min(self.gy, self.XT)
        self.cols = self.B * self.NT
        # the segments a column: the S (dividing KT) that minimizes the busiest core's tile matmuls plus the
        # owner's S - 1 adds (V41_DECODE_DENSE_SEGMENTS overrides)
        env = os.environ.get("V41_DECODE_DENSE_SEGMENTS", "")
        if segments is None and env and env != "auto":
            segments = int(env)
        if segments is None:
            # the balanced split (the S minimizing the busiest core's tile matmuls plus the owner's adds) was
            # measured slower than one column a core (the cost is the bytes, not the
            # compute); V41_DECODE_DENSE_SEGMENTS=auto picks it, the default is 1
            cost = lambda S: -(-self.cols * S // self.n_cores) * (self.KT // S) + 2 * (S - 1)
            segments = min((S for S in range(1, 17) if self.KT % S == 0), key=cost) if env == "auto" else 1
        assert self.KT % segments == 0, (self.KT, segments)
        self.S = segments
        self.L = self.KT // self.S
        self.BLK = max(b for b in (32, 20, 16, 10, 8, 5, 4, 2, 1) if self.L % b == 0)
        self.fidelity = fidelity or os.environ.get("V41_DECODE_DENSE_FIDELITY", "hifi2")
        self.probe = {"": 0, "noread": 1, "nocompute": 2, "none": 3}[os.environ.get("V41_DECODE_DENSE_PROBE", "")]
        self.compute = ttnn.ComputeConfigDescriptor(
            math_fidelity=_FIDELITY[self.fidelity], fp32_dest_acc_en=True, math_approx_mode=False
        )
        # per core: its units (j, s) in order, the owned ones' indices, and for the others the owner's NoC
        # coordinates and the receive slot (owned index on the owner x (S - 1) + s - 1); the x stripe it reads
        units = [[] for _ in range(self.n_cores)]
        owned_index = {}  # column -> its owned index on the owner core
        for j in range(self.cols):
            for s in range(self.S):
                core = (j * self.S + s) % self.n_cores
                if s == 0:
                    owned_index[j] = sum(1 for (jj, ss) in units[core] if ss == 0)
                units[core].append((j, s))
        self.per_core = []
        stripe, extra = divmod(self.XT, self.X_SENDERS)
        first = 0
        self.n_owned_max = 0
        for i in range(self.n_cores):
            entries = []
            for j, s in units[i]:
                owner = (j * self.S) % self.n_cores
                oc = self.noc[owner]
                slot = owned_index[j] * (self.S - 1) + (s - 1) if s > 0 else 0
                entries.append({"j": j, "s": s, "owned": int(s == 0), "onx": oc.x, "ony": oc.y, "slot": slot})
            n_owned = sum(e["owned"] for e in entries)
            self.n_owned_max = max(self.n_owned_max, n_owned)
            sender = i < self.X_SENDERS
            count = (stripe + (i < extra)) if sender else 0
            self.per_core.append({"units": entries, "n_owned": n_owned, "x_first": first, "x_count": count})
            first += count
        assert first == self.XT, (first, self.XT)
        self.units_max = max(len(c["units"]) for c in self.per_core)
        self.waddr = w.buffer_address()

    def __call__(self, x):
        assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT, (x.dtype, x.layout)
        assert not x.memory_config().is_sharded(), "DecodeDense: a DRAM-interleaved row"
        xs = tuple(int(d) for d in x.padded_shape)
        assert xs == (1, self.B, _TILE, self.K), (xs, self.B, self.K)
        y = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, self.B, _TILE, self.N]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        cb_x, cb_w, cb_part, cb_own, cb_recv, cb_out = 0, 1, 2, 3, 4, 16
        rd, wr, cm = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        xaddr, yaddr = x.buffer_address(), y.buffer_address()
        S = self.S
        for i, c in enumerate(self.per_core):
            cx, cy = i % self.gx, i // self.gx
            n_units, n_owned = len(c["units"]), c["n_owned"]
            rd[cx][cy] = (
                [xaddr, self.waddr, int(i == 0), self.rect[0], self.rect[1]]
                + self.rect
                + [0, 1, 2, int(c["x_count"] > 0), c["x_first"], c["x_count"], n_units, n_owned]
                + [v for e in c["units"] for v in (e["j"], e["s"])]
            )
            wr[cx][cy] = [yaddr, 2, n_units, n_owned] + [
                v for e in c["units"] for v in (e["j"], e["owned"], e["onx"], e["ony"], e["slot"])
            ]
            cm[cx][cy] = [n_units, n_owned] + [v for e in c["units"] for v in (e["j"], e["s"], e["owned"])]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=f"{_KERNELS}/dense_row_reader.cpp",
                core_ranges=self.cores,
                compile_time_args=[cb_x, cb_w, cb_recv, self.KT, self.NT, self.B, self.L, self.BLK, self.RING]
                + [self.n_cores, self.X_SENDERS, X_TILE, W_TILE, int(self.col_contig), self.probe, S]
                + ttnn.TensorAccessorArgs(x).get_compile_time_args()
                + ttnn.TensorAccessorArgs(self.w).get_compile_time_args(),
                runtime_args=rd,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{_KERNELS}/dense_row_writer.cpp",
                core_ranges=self.cores,
                compile_time_args=[cb_part, cb_recv, cb_out, self.PART_TILE, OUT_TILE, S]
                + ttnn.TensorAccessorArgs(y).get_compile_time_args(),
                runtime_args=wr,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{_KERNELS}/dense_row_compute.cpp",
                core_ranges=self.cores,
                compile_time_args=[cb_x, cb_w, cb_part, cb_own, cb_recv, cb_out, self.KT, self.NT, self.B]
                + [self.L, self.BLK, self.probe, S],
                runtime_args=cm,
                config=self.compute,
            ),
        ]

        def cb(index, fmt, pages, page):
            f = ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)
            return ttnn.CBDescriptor(total_size=pages * page, core_ranges=self.cores, format_descriptors=[f])

        recv_pages = max(1, self.n_owned_max * (S - 1))
        cbs = [
            cb(cb_x, ttnn.bfloat16, self.XT, X_TILE),
            cb(cb_w, ttnn.bfloat8_b, self.RING * self.BLK, W_TILE),
            cb(cb_part, ttnn.float32, 2, self.PART_TILE),
            cb(cb_own, ttnn.float32, max(1, self.n_owned_max), self.PART_TILE),
            cb(cb_recv, ttnn.float32, recv_pages, self.PART_TILE),
            cb(cb_out, ttnn.bfloat16, 2, OUT_TILE),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.cores, initial_value=0) for i in range(3)]
        ttnn.generic_op([x, self.w, y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
        return y
