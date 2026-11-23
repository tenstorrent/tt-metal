# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Shared plumbing of the Inkling-2-Mini fused residual kernels (ttnn.generic_op, JIT kernels, one Tensix core).

Dense form: a C-vector stored contiguously, padded with zeros to whole 1024-element pages. On device it is the raw
page content of a [1, 1, 32, 32 * P] TILE tensor (page p = tile p); the kernels view each page as one 32x32 tile.
Host helpers below permute a torch vector so that tilizing it yields exactly that raw page content.
"""

import os

import torch

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")  # not cwd-relative
PAGE = 1024  # elements per dense page
CORE = ttnn.CoreCoord(0, 0)
CORES = ttnn.CoreRangeSet([ttnn.CoreRange(CORE, CORE)])
ES = {ttnn.float32: 4, ttnn.bfloat16: 2}


def pages(c):
    return (c + PAGE - 1) // PAGE


def dense_matrix(vecs, n_pages=None):
    """[N, C] torch -> [1, 1, 32, 32 * N * P] torch whose TILE layout holds vector n in pages n*P .. n*P + P - 1."""
    vecs = vecs.reshape(-1, vecs.shape[-1]).float()
    n, c = vecs.shape
    p = n_pages or pages(c)
    v = torch.nn.functional.pad(vecs, (0, p * PAGE - c)).reshape(n * p, 2, 2, 16, 16)  # [page, fr, fc, r, c]
    m = v.permute(0, 1, 3, 2, 4).reshape(n * p, 32, 32)  # [page, row, col]
    return m.permute(1, 0, 2).reshape(1, 1, 32, n * p * 32)


def from_dense_matrix(m, c, n=1):
    """Inverse of dense_matrix: [.., 32, 32 * N * P] -> [N, C]."""
    m = m.reshape(32, -1, 32).permute(1, 0, 2)  # [page, row, col]
    v = m.reshape(-1, 2, 16, 2, 16).permute(0, 1, 3, 2, 4).reshape(n, -1)
    return v[:, :c]


def dense_tensor(device, vecs, dtype=ttnn.float32):
    return ttnn.from_torch(
        dense_matrix(vecs), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def zeros_row_form(device, rows, c, dtype):
    """Zeroed [1, 1, rows, C] TILE DRAM tensor (the kernels write only the logical rows)."""
    return ttnn.from_torch(
        torch.zeros(1, 1, rows, c),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def check_row_input(t, l1_ok=False):
    """l1_ok: also accept an L1-interleaved row tensor (only kernels that handle input kind 2, see input_args)."""
    assert t.layout == ttnn.TILE_LAYOUT and t.dtype in ES, (t.layout, t.dtype)
    kinds = (ttnn.BufferType.DRAM, ttnn.BufferType.L1) if l1_ok else (ttnn.BufferType.DRAM,)
    assert t.memory_config().buffer_type in kinds and not t.is_sharded(), t.memory_config()


def tile_bytes(dtype):
    return 1024 * ES[dtype]


def cb(index, dtype, n_pages, page_bytes=None, cores=None):
    page_bytes = page_bytes or tile_bytes(dtype)
    if _REC is not None:
        return _CSpec(index, dtype, n_pages, page_bytes, cores)
    fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)
    return ttnn.CBDescriptor(total_size=n_pages * page_bytes, core_ranges=cores or CORES, format_descriptors=[fmt])


def stage_cb(index, ntc, cores=None):
    """L1 scratch for 64B-aligned DRAM reads (rowio::gather_row): 128 B per tile + alignment slack."""
    return cb(index, ttnn.bfloat16, 1, ntc * 128 + 64, cores)


def compute_config(fp32_cbs, full_sync=False):
    """fp32_cbs: CBs read by copy_tile that must reach DST unrounded (UnpackToDestFp32; FPU ops read via SrcA/B).
    full_sync: 8 fp32 DST tiles instead of 4."""
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=full_sync
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for i in fp32_cbs:
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    return cfg


def kernel(name, config, named, rt=None, cores=None, defines=None):
    """rt: runtime args of the single CORE, or {CoreCoord: args} with cores = their CoreRangeSet."""
    if _REC is not None:
        assert not defines, "per_row records no defines"
        return _KSpec(name, config, named, rt, cores)
    args = ttnn.RuntimeArgs()
    if rt is not None:
        for core, vals in rt.items() if isinstance(rt, dict) else [(CORE, rt)]:
            args[core.x][core.y] = [int(v) for v in vals]
    return ttnn.KernelDescriptor(
        kernel_source=os.path.join(KDIR, name),
        core_ranges=cores or CORES,
        named_compile_time_args=list(named.items()),
        runtime_args=args,
        config=config,
        defines=list(defines or []),
    )


def fixed_rt(named, *rts):
    """Pad every core's runtime args of each {core: args} dict to one length and add those lengths to the named
    compile-time args. generic_op's program cache key ignores runtime-arg counts, so a module whose per-core
    counts depend on the row count (the multi-row prefill kernels) would otherwise reuse a cached program built
    for another count and fail its arg-count check. The kernels ignore the zero padding."""
    named = dict(named)
    for i, rt in enumerate(rts):
        n = max(len(v) for v in rt.values())
        for v in rt.values():
            v.extend([0] * (n - len(v)))
        named[f"rt_len{i}"] = n
    return named


def core_list(device, n, first_row=0):
    """n worker cores, row-major over the compute grid starting at first_row: (list, CoreRangeSet)."""
    gx = device.compute_with_storage_grid_size().x
    cores = [ttnn.CoreCoord(i % gx, first_row + i // gx) for i in range(n)]
    return cores, ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


# One-shot kernels for the next bank-streamed linear program (BankLinear / StagedBankLinear): a callable
# busy -> (kernels, io tensors), busy = the {(x, y)} logical cores the linear uses (the kernels must avoid them). The
# attention prep's input fetch rides in the w_all program this way (fused/attn.py PREP_PRE).
RIDE = None


def take_ride(cores):
    global RIDE
    ride, RIDE = RIDE, None
    return ride({tuple(c) for c in cores}) if ride is not None else ([], [])


def run(io_tensors, kernels, cbs):
    """io_tensors: every tensor the kernels touch (keeps them alive / tracked); the last one is returned.
    Inside per_row(): recorded for the merged R-row program instead."""
    if _REC is not None:
        _REC[_ROW].append((io_tensors, kernels, cbs))
        return io_tensors[-1]
    if _JOIN is not None:
        _JOIN.append((io_tensors, kernels, cbs))
        return io_tensors[-1]
    return ttnn.generic_op(io_tensors, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))


# Dense twins: every row-form output of a fused kernel also gets a dense copy (a few page writes). A fused
# kernel given that same tensor object as input reads the twin's pages instead of gathering the row (2 x 72 small
# NOC reads, ~6 us). Keyed by object identity, so a tensor that passed through any other op has no twin.
_TWIN = {}


def set_twin(row, dense, nvec):
    _TWIN[id(row)] = (row, dense, nvec)


def twin(t):
    """(dense tensor, number of vectors) of a fused-kernel output, else None."""
    e = _TWIN.get(id(t))
    return (e[1], e[2]) if e is not None and e[0] is t else None


def input_args(t, row=0):
    """Reader runtime args (addr, page bytes, elem bytes, row / first page, kind) of a row-form input t: its dense
    twin when it has one (kind 1), else its rows in DRAM (0) or, interleaved, in L1 (2: conv's x only)."""
    tw = twin(t)
    if tw is not None:
        dense, nvec = tw
        return [
            dense.buffer_address(),
            tile_bytes(dense.dtype),
            ES[dense.dtype],
            min(row, nvec - 1) * pages(t.shape[-1]),
            1,
        ]
    l1 = t.memory_config().buffer_type == ttnn.BufferType.L1
    return [t.buffer_address(), tile_bytes(t.dtype), ES[t.dtype], row, 2 if l1 else 0]


def input_dtype(t):
    tw = twin(t)
    return tw[0].dtype if tw is not None else t.dtype


class OutCache:
    """Per-module zeroed output buffers: an output stays valid until the same module is called again with the
    same key. First use must be eager (it writes zeros from host), as the model and tests do before capture."""

    def __init__(self, device):
        self.device = device
        self.bufs = {}

    def get(self, key, rows, c, dtype):
        key = row_key(key)
        if key not in self.bufs:
            self.bufs[key] = zeros_row_form(self.device, rows, c, dtype)
        return self.bufs[key]

    def shared_twin(self, key, rows, c, dtype, nvec):
        """One row-form output + dense twin for ALL verify rows of an F.per_row call (not keyed by row): each row's
        program writes its vectors at its own offset."""
        k = ("shared", key, n_rows())
        if k not in self.bufs:
            self.bufs[k] = zeros_row_form(self.device, rows, c, dtype)
            self.bufs[("twin", k)] = dense_tensor(self.device, torch.zeros(nvec, c), dtype)
            set_twin(self.bufs[k], self.bufs[("twin", k)], nvec)
        return self.bufs[k], self.bufs[("twin", k)]

    def get_twin(self, key, rows, c, dtype, nvec=None):
        """Row-form output buffer plus its dense twin (nvec vectors, default rows)."""
        out = self.get(key, rows, c, dtype)
        tkey = ("twin", row_key(key))
        if tkey not in self.bufs:
            nvec = nvec or rows
            self.bufs[tkey] = dense_tensor(self.device, torch.zeros(nvec, c), dtype)
            set_twin(out, self.bufs[tkey], nvec)
        return out, self.bufs[tkey]


_RMS_DTYPE = None  # does this build's rms_norm take an output dtype?


def rms_norm_f32(x, **kw):
    """ttnn.rms_norm with an fp32 result for a bf16 or fp32 input. A bf16 input goes in as it is when the build's
    rms_norm takes `dtype` (bit-equal to typecasting it to fp32 first: the norm's math is fp32 and bf16 is exact in
    fp32), else through a typecast."""
    global _RMS_DTYPE
    if x.dtype != ttnn.float32 and _RMS_DTYPE is not False:
        try:
            y = ttnn.rms_norm(x, dtype=ttnn.float32, **kw)
            _RMS_DTYPE = True
            return y
        except TypeError:
            _RMS_DTYPE = False
    return ttnn.rms_norm(x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32), **kw)


# ---------------------------------------------------------------- joined programs (decode glue)
# joined(device, [(core, fn), ...]) runs each fn with the single-core default CORE moved to its core and the F.run
# calls recorded, then launches ONE program holding all of them (their kernels, CBs and io tensors) plus semaphore
# JOIN_SEM on those cores. The parts hand data over through DRAM as separate programs do; a part that reads an
# earlier part's output waits on JOIN_SEM, which that part's last writer increments after its write barrier
# (conv_writer `sig`, attnres_reader ATTNRES_WAIT_SEM). Same kernels and inputs, so the same bits as separate programs.
_JOIN = None
JOIN_SEM = 0


def joined(device, parts):
    global _JOIN, CORE, CORES
    assert _REC is None and _JOIN is None, "joined does not nest or run inside per_row"
    rec, outs, saved = [], [], (CORE, CORES)
    _JOIN = rec
    try:
        for core, fn in parts:
            CORE, CORES = core, ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
            outs.append(fn())
    finally:
        _JOIN = None
        CORE, CORES = saved
    io, seen = [], set()
    for t in (t for p in rec for t in p[0]):
        if id(t) not in seen:
            seen.add(id(t))
            io.append(t)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _ in parts])
    sems = [ttnn.SemaphoreDescriptor(id=JOIN_SEM, core_ranges=crs, initial_value=0)]
    kernels = [k for p in rec for k in p[1]]
    cbs = [c for p in rec for c in p[2]]
    ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems))
    return outs


def noc_xy(device, core):
    c = device.worker_core_from_logical_core(core)
    return c.x, c.y


# ---------------------------------------------------------------- R rows (speculative verify)
# per_row(R, fn) runs fn(r) for every verify row r with the F.kernel / F.cb / F.run calls recorded, then launches
# each recorded program ONCE for all rows: row r's copy of the 1-row program on its own cores (same kernels, same
# compile-time args, its own buffers). Every row computes exactly what the 1-row call computes on its inputs.
# Buffers: OutCache keys and FusedRowOps stacks are per row (row 0 shares the 1-row buffers).
_ROW = None  # row being recorded
_REC = None  # per row: [(io, kernels, cbs)] in call order


def n_rows():
    """Number of verify rows of the per_row call being recorded (1 outside)."""
    return len(_REC) if _REC is not None else 1


def row_index():
    """Verify row being recorded inside per_row (0 outside)."""
    return _ROW or 0


def row_key(key):
    return key if not _ROW else (key, "row", _ROW)


class Rows(list):
    """An R-row value: one tensor per verify row."""


class _KSpec:
    def __init__(self, name, config, named, rt, cores):
        self.name, self.config, self.named = name, config, dict(named)
        if rt is None:
            self.rt = None
        elif isinstance(rt, dict):
            self.rt = {(c.x, c.y): [int(v) for v in a] for c, a in rt.items()}
        else:
            self.rt = {(CORE.x, CORE.y): [int(v) for v in rt]}
        self.cores = _xy(cores or CORES)
        self.common_runtime_args = None


class _CSpec:
    def __init__(self, index, dtype, n_pages, page_bytes, cores):
        self.key = (index, dtype, n_pages, page_bytes)
        self.cores = _xy(cores or CORES)


def _xy(crs):
    return sorted(
        {(x, y) for r in crs.ranges() for y in range(r.start.y, r.end.y + 1) for x in range(r.start.x, r.end.x + 1)},
        key=lambda c: (c[1], c[0]),
    )


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores])


ROW_POOL_ROWS = 6  # logical rows 0..5 hold the rows' cores: off the w_all staging block (rows 6..9, mk/overlap.py)


def _pool(device):
    gx = device.compute_with_storage_grid_size().x
    return [(x, y) for y in range(ROW_POOL_ROWS) for x in range(gx)]


def per_row(device, n_rows, fn):
    """Rows(fn(r) for r < n_rows), each recorded program launched once for all rows (see above)."""
    global _ROW, _REC
    assert _REC is None, "per_row does not nest"
    rec = [[] for _ in range(n_rows)]
    outs = Rows()
    _REC = rec
    try:
        for r in range(n_rows):
            _ROW = r
            outs.append(fn(r))
    finally:
        _ROW, _REC = None, None
    n = len(rec[0])
    assert all(len(p) == n for p in rec), [len(p) for p in rec]
    for i in range(n):
        _launch(device, [p[i] for p in rec])
    return outs


def _launch(device, progs):
    """One program from the rows' copies of one recorded program."""

    def cores_of(prog):
        _, ks, cs = prog
        return sorted({c for k in ks for c in k.cores} | {c for b in cs for c in b.cores}, key=lambda c: (c[1], c[0]))

    base = cores_of(progs[0])
    pool = _pool(device)
    assert len(base) * len(progs) <= len(pool), (len(base), len(progs))
    maps = []
    for r, prog in enumerate(progs):
        assert cores_of(prog) == base, "rows must use the same cores"
        maps.append({c: pool[r * len(base) + i] for i, c in enumerate(base)})
    kernels, cbs = [], []
    for j in range(len(progs[0][1])):
        groups = {}  # (source, named) -> [(row, spec)]
        for r, (_, ks, _) in enumerate(progs):
            k = ks[j]
            groups.setdefault((k.name, tuple(sorted(k.named.items()))), []).append((r, k))
        for (name, _), members in groups.items():
            k0 = members[0][1]
            cores = [maps[r][c] for r, k in members for c in k.cores]
            args = ttnn.RuntimeArgs()
            for r, k in members:
                assert k.common_runtime_args == k0.common_runtime_args
                for c, a in (k.rt or {}).items():
                    m = maps[r][c]
                    args[m[0]][m[1]] = a
            kd = ttnn.KernelDescriptor(
                kernel_source=os.path.join(KDIR, name),
                core_ranges=_crs(cores),
                named_compile_time_args=list(k0.named.items()),
                runtime_args=args,
                config=k0.config,
            )
            if k0.common_runtime_args is not None:
                kd.common_runtime_args = k0.common_runtime_args
            kernels.append(kd)
    groups = {}  # CB (index, format, size) -> cores; rows may differ (e.g. conv's own-lag CB on rows 1..)
    for r, (_, _, cs) in enumerate(progs):
        for b in cs:
            groups.setdefault(b.key, []).extend(maps[r][c] for c in b.cores)
    for (index, dtype, n_pages, page_bytes), cores in groups.items():
        fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)
        cbs.append(
            ttnn.CBDescriptor(total_size=n_pages * page_bytes, core_ranges=_crs(cores), format_descriptors=[fmt])
        )
    io, seen = [], set()
    for prog in progs:
        for t in prog[0]:
            if id(t) not in seen:
                seen.add(id(t))
                io.append(t)
    return ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))


_RMS_BF16 = None  # does this build's rms_norm(dtype=bfloat16) round as ttnn.typecast? (probe_rms_bf16)


def probe_rms_bf16(device, ckc):
    """Set whether rms_norm(fp32 x, dtype=bfloat16) equals typecast(rms_norm(x)) on this build: builds with the dtype
    keyword but without the rounding in DEST pack 0.04% of values otherwise (not exact). Host reads: call it at
    construction, before any trace capture."""
    global _RMS_BF16
    x = ttnn.from_torch(torch.randn(1, 1, 64, 2304, generator=torch.Generator().manual_seed(0)) * 3,
                        dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)  # fmt: skip
    try:
        a = ttnn.rms_norm(x, epsilon=1e-6, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
    except TypeError:
        _RMS_BF16 = False
        return False
    b = ttnn.typecast(ttnn.rms_norm(x, epsilon=1e-6, compute_kernel_config=ckc), ttnn.bfloat16)
    _RMS_BF16 = a.dtype == ttnn.bfloat16 and torch.equal(ttnn.to_torch(a), ttnn.to_torch(b))
    return _RMS_BF16


def rms_norm_bf16(x, **kw):
    """ttnn.rms_norm of an fp32 input to a bf16 result: in the op when probe_rms_bf16 found it bit-equal to
    ttnn.typecast of the fp32 result (rounding in DEST), else through a typecast (also when never probed)."""
    if x.dtype == ttnn.float32 and _RMS_BF16:
        return ttnn.rms_norm(x, dtype=ttnn.bfloat16, **kw)
    y = ttnn.rms_norm(x, **kw)
    return y if y.dtype == ttnn.bfloat16 else ttnn.typecast(y, ttnn.bfloat16)
