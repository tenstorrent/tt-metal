# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused all-gather along M over the mesh rows + matmul for the Qwen context-parallel prefill (layout B).

Replaces ``_row_ag`` (all_gather_async of the norm output [1, 1, S/32, K] bf8 along dim 2 over cluster axis 0,
FABRIC_1D_RING) followed by ``ttnn.experimental.minimal_matmul`` on the gathered [1, 1, S/4, K] buffer, as one
``ttnn.generic_op`` program per device:

- Matmul on grid columns 0..5 x rows 0..9: the stock minimal_matmul program for the transposed core grid (M over x,
  N over y, in0 relayed down each column, in1 along each row, the in1 kernel writes the output), with the model's
  K block / N block / subblock and one N block per core. The kernels (kernels/mm_*.cpp) are the stock legacy kernels
  reduced to this case, with the M blocks taken from an explicit per-column list. A tile's K order depends only on
  K_block and the N-block parity, never on the M split, so the output bits equal the stock call's.
- All-gather on column 6: cores (6, 0..3) are the fabric senders of links 0..3 (BRISC fwd, NCRISC bwd), core (6, 4)
  the hub. Each device multicasts its slot to every other device's gathered buffer in waves of rows; each
  wave-half ends with an atomic inc of that (slot, wave)'s arrival word on the hub of every receiver. The hub
  releases each matmul column's blocks in list order once their wave has landed (own-slot blocks at once).
- Block order (``plan``): blocks are the waves (8 tile rows) of every slot, own slot first, then by arrival.
  Resident plans (the core's whole K x N_block weight fits in L1, QKV): in1 is relayed only for each column's first
  block and then re-pushed from L1, so the columns run independently and take the remote blocks by list scheduling.
  Lockstep plans (FF1): in1 is relayed along the grid rows for every block, so the columns move in steps of one
  block each and a step costs the slowest of its blocks.

``matmul_port`` runs only the matmul part on a full in0 buffer (no fabric): the check that the port and the block
lists keep the stock bits.

``qwen_ag_matmul_v2``: the same op with the matmul on the full 7x10 grid; the eight fabric senders run inside in0
receiver cores (kernels/v2_in0.cpp: a non-blocking sender pump called from every wait of the in0 relay loop), the
injectors wait on per-wave words (every remote slot's wave j landed) and there is no hub. Column x takes a piece of the
own slot and then every wave of remote slot (r + 1 + x) % 8, so each wave is one lockstep step. Used for FF1.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import ttnn
from models.demos.llama3_70b_galaxy.tt.custom_ops.cache_key import tensor_key

KERNEL_ROOT = "models/demos/llama3_70b_galaxy/tt/custom_ops/qwen_ag_matmul/kernels"
KERNELS = {k: f"{KERNEL_ROOT}/{k}.cpp" for k in ("mm_in0", "mm_in1_out", "mm_compute", "ag_sender", "ag_hub")}

TILE = 32
TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.float32: 4096}
ROWS_RING = 8  # mesh rows (cluster axis 0, the ring)
LINKS = 4
TPP = 4  # tiles per fabric packet
BATCH = 4  # packets per sender batch (one header each; the pool would hold 8)
HDR = 64  # packet header slot stride
MAX_WAVES = 8
GX, GY = 6, 10  # matmul grid
SND_X = 6  # sender / hub column
HUB = (6, 4)

# semaphore ids (program-local)
SEM_IN0_SENDER, SEM_IN0_RECEIVER, SEM_IN0_VALID = 0, 1, 2
SEM_IN1_SENDER, SEM_IN1_RECEIVER, SEM_IN1_VALID = 3, 4, 5
SEM_READY, SEM_GO_F, SEM_GO_B = 6, 7, 8
# CB ids
CB_IN0, CB_IN1, CB_OUT, CB_INTERM = 0, 1, 2, 3
CB_STAGE_F, CB_STAGE_B, CB_HDR_F, CB_HDR_B = 8, 9, 10, 11

_DESCRIPTOR_CACHE: dict = {}
# One key per (op, bucket, layer): the key holds the weight address, so every layer is its own entry. 4 prefill
# buckets (16k / 24k / 32k / 40k) x 64 layers x 2 ops (QKV v1, FF1 v2) = 512, x 2 for a second set of activation
# addresses (b1 and b32 allocate differently). Hitting the cap clears the cache (host rebuild cost only).
_DESCRIPTOR_CACHE_MAX = 4 * 64 * 2 * 2
_SEMS: dict = {}


@dataclass(frozen=True)
class MMConfig:
    M_block: int
    K_block: int
    N_block: int
    subblock_h: int
    subblock_w: int

    @staticmethod
    def of(cfg) -> "MMConfig":
        if isinstance(cfg, MMConfig):
            return cfg
        return MMConfig(
            int(cfg.M_block_size),
            int(cfg.K_block_size),
            int(cfg.N_block_size),
            int(cfg.subblock_h),
            int(cfg.subblock_w),
        )


# ---------------------------------------------------------------------------------------------------------------
# Plan: waves and per-column block lists
# ---------------------------------------------------------------------------------------------------------------
def _schedule(waves, nslots, ncols, my_slot):
    """Blocks (slot, wave, rows) in step order: by expected arrival (own slot at t=0, wave j of the others at the
    cumulative fraction of the slot sent), larger first; padded to whole steps by splitting own-slot blocks, else by
    zero-row blocks in the first step. Returns the flat list (step-major) and the arrival fraction of each."""
    T = sum(waves)
    cum, arr = 0, []
    for w in waves:
        cum += w
        arr.append(cum / T)
    local = [(my_slot, j, w, 0.0) for j, w in enumerate(waves)]
    remote = [(s, j, w, arr[j]) for s in range(nslots) if s != my_slot for j, w in enumerate(waves)]
    blocks = local + remote
    # split own-slot blocks until the count fills whole steps
    while len(blocks) % ncols:
        cands = [b for b in blocks if b[3] == 0.0 and b[2] >= 4 and b[2] % 4 == 0]
        if not cands:
            break
        big = max(cands, key=lambda b: b[2])
        i = blocks.index(big)
        blocks[i : i + 1] = [(big[0], big[1], big[2] // 2, 0.0), (big[0], big[1], big[2] // 2, 0.0)]
    blocks = [b + (k,) for k, b in enumerate(blocks)]  # stable tag
    blocks.sort(key=lambda b: (b[3], -b[2], b[4]))
    pad = (-len(blocks)) % ncols
    blocks = [(my_slot, -1, 0, 0.0, -1)] * pad + blocks
    return blocks


def default_waves(T: int, wave: int = 8) -> tuple:
    """Waves of `wave` tile rows, the remainder last. Measured at S=40960 (job363-366): 8-row waves are the best of
    the sizes tried for QKV (resident 1.58-1.59 ms; 16,8,8,8 1.77; 4,8,8,8,8,4 1.60; 8,8,8,8,4,4 1.62), and the
    wave count costs the all-gather nothing (1.12-1.15 ms alone)."""
    q, rem = divmod(T, wave)
    return (wave,) * q + ((rem,) if rem else ())


@dataclass(frozen=True)
class Plan:
    T: int  # tile rows per slot
    waves: tuple  # rows per wave
    starts: tuple  # first row (within the slot) of each wave
    cols: tuple  # per device slot r: per column: tuple of (m_start_tile, rows, slot, wave)
    resident: bool = False  # independent columns (in1 resident in L1 after the first block)


def _pieces(rows, piece):
    out = []
    while rows > 0:
        out.append(min(piece, rows))
        rows -= out[-1]
    return out


def _independent(waves, nslots, ncols, my_slot, split, per_row=0.0158, ovh=0.03):
    """Per-column lists when the columns run independently: own-slot blocks first (at least one per column, the
    first block of every column relays in1 in lockstep), then every remote block in arrival order to the column that
    frees up first (list scheduling with a linear block cost); the last wave's blocks are cut into `split`-row
    pieces so the tail spreads over the columns."""
    T = sum(waves)
    starts = [sum(waves[:j]) for j in range(len(waves))]
    local = [(starts[j], w, my_slot, j) for j, w in enumerate(waves)]
    while len(local) < ncols:
        i = max(range(len(local)), key=lambda k: local[k][1])
        r0, w, s, j = local[i]
        if w < 4:
            raise ValueError("own slot too small for the grid")
        h = w // 2
        local[i : i + 1] = [(r0, h, s, j), (r0 + h, w - h, s, j)]
    blocks = [(0.0, r0, w, s, j) for r0, w, s, j in local]
    cum = 0
    for j, w in enumerate(waves):
        cum += w
        for s in range(nslots):
            if s == my_slot:
                continue
            pieces = _pieces(w, split) if (split and j == len(waves) - 1) else [w]
            off = 0
            for pr in pieces:
                blocks.append((cum / T, starts[j] + off, pr, s, j))
                off += pr
    free = [0.0] * ncols
    cols = [[] for _ in range(ncols)]
    for k, (arr, r0, w, s, j) in enumerate(blocks):
        c = k if k < ncols else min(range(ncols), key=lambda i: (free[i], i))
        free[c] = max(free[c], arr) + ovh + per_row * w
        cols[c].append((s * T + r0, w, s, j))
    return cols


@functools.lru_cache(maxsize=64)
def plan(
    T: int,
    M_block: int,
    N: int,
    waves: tuple | None = None,
    ncols: int = GX,
    nslots: int = ROWS_RING,
    resident: bool = False,
    split: int = 0,
) -> Plan:
    if waves is None:
        waves = default_waves(T)
    if sum(waves) != T or max(waves) > M_block or len(waves) > MAX_WAVES:
        raise ValueError(f"bad waves {waves} for slot rows {T} (M block {M_block})")
    starts = tuple(sum(waves[:j]) for j in range(len(waves)))
    per_dev = []
    for r in range(nslots):
        if resident:
            per_dev.append(tuple(tuple(c) for c in _independent(waves, nslots, ncols, r, split)))
            continue
        blocks = _schedule(waves, nslots, ncols, r)
        cols = [[] for _ in range(ncols)]
        # own-slot splits: recompute the row offsets of split halves (same slot, wave; consecutive)
        seen = {}
        for i, (s, j, rows, _, _) in enumerate(blocks):
            if j < 0:
                m0 = 0
            else:
                off = seen.get((s, j), 0)
                seen[(s, j)] = off + rows
                m0 = s * T + starts[j] + off
            cols[i % ncols].append((m0, rows, s, j))
        per_dev.append(tuple(tuple(c) for c in cols))
    return Plan(T, tuple(waves), starts, tuple(per_dev), resident)


def check_plan(p: Plan, nslots: int = ROWS_RING):
    """Every device's lists cover each gathered row exactly once; equal list lengths over the columns (lockstep
    plans); every column's first block is from its own slot (resident plans)."""
    for r, cols in enumerate(p.cols):
        if p.resident:
            assert all(len(c) > 0 and c[0][2] == r for c in cols), (r, cols)
        else:
            assert len({len(c) for c in cols}) == 1, (r, [len(c) for c in cols])
        cover = [0] * (nslots * p.T)
        for c in cols:
            for m0, rows, s, j in c:
                for m in range(m0, m0 + rows):
                    cover[m] += 1
        assert all(v == 1 for v in cover), (r, cover)


# ---------------------------------------------------------------------------------------------------------------
# Program pieces
# ---------------------------------------------------------------------------------------------------------------
def _crs(x0, y0, x1, y1):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


def _accessor(t) -> list[int]:
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _cb(idx, n_tiles, page, dtype, cores):
    return ttnn.CBDescriptor(
        total_size=n_tiles * page,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page)],
    )


def _phys(mesh, x, y):
    p = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
    return int(p.x), int(p.y)


def _matmul_kernels(
    mesh, a, w, out, xl, cfg: MMConfig, ckc, col_lists, my_slot, T, fused, probe=None, resident=False, gx=GX
):
    """KernelDescriptors, CBs and semaphores of the matmul part (grid gx x GY). The CBs hold blocks of the plan's
    largest block (the M block size does not change the bits); resident: the in1 CB holds the whole K x N block."""
    M_eff = max(rows for c in col_lists for _, rows, _, _ in c)
    M_tiles = int(a.shape[-2]) // TILE
    K = int(a.shape[-1])
    N = int(w.shape[-1])
    K_tiles, N_tiles = K // TILE, N // TILE
    if K_tiles % cfg.K_block:
        raise ValueError(f"K tiles {K_tiles} must be a multiple of K block {cfg.K_block}")
    if N_tiles % GY or N_tiles // GY != cfg.N_block:
        raise ValueError(f"N tiles {N_tiles} must give one N block of {cfg.N_block} per grid row ({GY})")
    padded_K = K_tiles
    K_blocks = padded_K // cfg.K_block
    Ntpc = N_tiles // GY
    k_blocks_per_core = -(-K_blocks // GY)
    in0_ts, in1_ts, out_ts = TILE_BYTES[a.dtype], TILE_BYTES[w.dtype], TILE_BYTES[out.dtype]
    interm_dtype = ttnn.float32 if ckc.fp32_dest_acc_en else ttnn.bfloat16

    mm = _crs(0, 0, gx - 1, GY - 1)
    cbs = [
        _cb(CB_IN0, 2 * M_eff * cfg.K_block, in0_ts, a.dtype, mm),
        _cb(CB_IN1, (K_blocks if resident else 2) * cfg.K_block * cfg.N_block, in1_ts, w.dtype, mm),
        _cb(CB_OUT, 2 * M_eff * cfg.N_block, out_ts, out.dtype, mm),
        _cb(CB_INTERM, M_eff * cfg.N_block, TILE_BYTES[interm_dtype], interm_dtype, mm),
    ]
    sems = [
        ttnn.SemaphoreDescriptor(id=i, core_ranges=mm, initial_value=v)
        for i, v in (
            (SEM_IN0_SENDER, 0),
            (SEM_IN0_RECEIVER, 0),
            (SEM_IN0_VALID, 1),
            (SEM_IN1_SENDER, 0),
            (SEM_IN1_RECEIVER, 0),
            (SEM_IN1_VALID, 1),
            (SEM_READY, 0),
        )
    ]
    defines = [("FUSED", "1")] if fused else []
    if resident:
        defines.append(("IN1_RESIDENT", "1"))
    if probe == "nowait":  # timing probe: the injectors do not wait for the arrivals (wrong output)
        defines.append(("PROBE_NOWAIT", "1"))
    elif probe == "nomm":  # timing probe: the matmul kernels return at once (all-gather only)
        defines.append(("PROBE_NOMM", "1"))
    in0_acc = _accessor(a) + _accessor(xl if xl is not None else a)
    w_acc = _accessor(w) + _accessor(out)
    xl_addr = xl.buffer_address() if xl is not None else a.buffer_address()

    # Every matmul kernel takes only common runtime args; a core finds its neighbours, N range and block list
    # from its logical coordinates. The grid's NOC coordinates are separable (x from the column, y from the row).
    px = [_phys(mesh, x, 0)[0] for x in range(gx)]
    py = [_phys(mesh, 0, y)[1] for y in range(GY)]
    for x in range(gx):
        for y in range(GY):
            assert _phys(mesh, x, y) == (px[x], py[y]), "non-separable worker NOC coordinates"

    def lists(base):
        """gx absolute offsets, then per column: num_blocks, (m_start_tile, rows) pairs."""
        offs, body = [], []
        at = base + gx
        for x in range(gx):
            offs.append(at + len(body))
            body.append(len(col_lists[x]))
            for m0, rows, _, _ in col_lists[x]:
                body += [m0, rows]
        return offs + body

    head0 = [a.buffer_address(), xl_addr] + px + py
    in0_common = head0 + lists(len(head0))
    head1 = [w.buffer_address(), out.buffer_address()] + px + py
    in1_common = head1 + lists(len(head1))
    comp_common = lists(0)

    def in0_ct(inj):
        ct = [K_tiles, padded_K, M_eff, cfg.K_block, in0_ts, SEM_IN0_SENDER, SEM_IN0_RECEIVER, SEM_IN0_VALID]
        return ct + [int(inj), SEM_READY, T, my_slot, M_tiles, gx, GY]

    def in1_ct(inj):
        ct = [K_tiles, padded_K, N_tiles, M_eff, cfg.K_block, cfg.N_block, in1_ts, out_ts]
        return ct + [
            SEM_IN1_SENDER,
            SEM_IN1_RECEIVER,
            SEM_IN1_VALID,
            int(inj),
            M_tiles,
            gx,
            GY,
            Ntpc,
            k_blocks_per_core,
        ]

    dm0 = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)
    dm1 = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)
    kernels = []
    for inj, cores in ((True, _crs(0, 0, gx - 1, 0)), (False, _crs(0, 1, gx - 1, GY - 1))):
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=KERNELS["mm_in0"],
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores,
                compile_time_args=in0_ct(inj) + in0_acc,
                common_runtime_args=in0_common,
                defines=defines,
                config=dm0,
            )
        )
    for inj, cores in ((True, _crs(0, 0, 0, GY - 1)), (False, _crs(1, 0, gx - 1, GY - 1))):
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=KERNELS["mm_in1_out"],
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores,
                compile_time_args=in1_ct(inj) + w_acc,
                common_runtime_args=in1_common,
                defines=defines,
                config=dm1,
            )
        )
    kernels.append(
        ttnn.KernelDescriptor(
            kernel_source=KERNELS["mm_compute"],
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=mm,
            compile_time_args=[K_blocks, M_eff, cfg.K_block, cfg.N_block, cfg.subblock_h, cfg.subblock_w, Ntpc, 0],
            common_runtime_args=comp_common,
            defines=defines,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ckc.math_fidelity,
                fp32_dest_acc_en=bool(ckc.fp32_dest_acc_en),
                dst_full_sync_en=False,  # the stock factory double-buffers DEST (double_buffer_dest)
                math_approx_mode=bool(ckc.math_approx_mode),
            ),
        )
    )
    return kernels, cbs, sems


class _GlobalSems:
    """Barrier and arrival words on the hub core, at the same L1 address on every device, incremented by peers over
    the fabric: one zero-filled L1 buffer sharded on the hub core (one allocation and one write, instead of 65
    global semaphores, whose blocking creation cost ~35 ms per prefill, job369), words at a 16-byte stride."""

    STRIDE = 16

    def __init__(self, mesh):
        import torch

        n = 1 + ROWS_RING * MAX_WAVES
        width = n * self.STRIDE // 4  # uint32 elements
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(_crs(*HUB, *HUB), [1, width], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.buf = ttnn.from_torch(
            torch.zeros(1, 1, 1, width, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=mem,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        # The zeros must land on every device before any device runs the op: a fast peer's barrier inc would be
        # wiped by a slower device's queued zero-fill (a hang). Global semaphores get this from their blocking
        # initial write; here it is an explicit sync, once per prefill.
        ttnn.synchronize_device(mesh)
        base = self.buf.buffer_address()
        self.barrier = base
        self.arr = [[base + self.STRIDE * (1 + s * MAX_WAVES + j) for j in range(MAX_WAVES)] for s in range(ROWS_RING)]
        self.key = (base,)


def _sems(mesh):
    key = id(mesh)
    if key not in _SEMS:
        _SEMS[key] = _GlobalSems(mesh)
    return _SEMS[key]


def release_semaphores(mesh):
    """Free the barrier / arrival words (L1 on the hub core). Call when a prefill is done: L1 left allocated across
    the prefill -> decode switch moves the decode global CB and the decode trace then misses its program cache
    (job368). The next call re-creates them (zeroed). Every word is back at 0 after each call."""
    gs = _SEMS.pop(id(mesh), None)
    if gs is not None:
        ttnn.deallocate(gs.buf)


def check_k_split(K_tiles: int):
    """Each link sends K_tiles / LINKS columns as two halves of whole TPP-tile packets (ag_sender.cpp and
    v2_in0.cpp). The kernels floor both divisions, so any other K silently leaves columns out of the gathered buffer."""
    if K_tiles % (LINKS * 2 * TPP):
        raise ValueError(f"K tiles {K_tiles} must split into {LINKS} links x 2 halves of whole {TPP}-tile packets")


def _fabric_kernels(mesh, xl, g, p: Plan, r, K_tiles, copy_local, gs: _GlobalSems, probe=None):
    """Sender (cores (6, 0..3), BRISC fwd / NCRISC bwd) and hub (6, 4) kernels of device row r."""
    page = TILE_BYTES[xl.dtype]
    check_k_split(K_tiles)
    link_cols = K_tiles // LINKS
    hub_phys = _phys(mesh, *HUB)
    snd = [(SND_X, l) for l in range(LINKS)]
    snd_crs = _crs(SND_X, 0, SND_X, LINKS - 1)
    col6 = _crs(SND_X, 0, SND_X, LINKS)
    acc = _accessor(xl) + _accessor(g)
    waves_rt = []
    for j, (r0, rows) in enumerate(zip(p.starts, p.waves)):
        waves_rt += [r0, rows, gs.arr[r][j]]
    kernels = []
    for fwd in (True, False):
        ra, rb = (4, 3) if fwd else (3, 4)
        for l, (x, y) in enumerate(snd):
            ct = [page, ra, rb, int(fwd), int(l == 0), 4 if fwd else 3, BATCH, HDR, K_tiles, l * link_cols, link_cols]
            ct += [int(copy_local), len(p.waves), SEM_GO_F if fwd else SEM_GO_B]
            ct += [CB_STAGE_F if fwd else CB_STAGE_B, CB_HDR_F if fwd else CB_HDR_B] + acc
            rt = [xl.buffer_address(), g.buffer_address(), r * p.T, hub_phys[0], hub_phys[1], gs.barrier] + waves_rt
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS["ag_sender"],
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=_crs(x, y, x, y),
                    compile_time_args=ct,
                    runtime_args=[((x, y), rt)],
                    config=ttnn.WriterConfigDescriptor() if fwd else ttnn.ReaderConfigDescriptor(),
                )
            )
    hub_rt = [gs.barrier]
    for x, y in snd:
        hub_rt += list(_phys(mesh, x, y))
    for x in range(GX):
        hub_rt += list(_phys(mesh, x, 0))
    words = set()
    for col in p.cols[r]:
        hub_rt.append(len(col))
        for _, rows, s, j in col:
            if s == r or j < 0:
                hub_rt.append(0)
            else:
                hub_rt.append(gs.arr[s][j])
                words.add(gs.arr[s][j])
    hub_rt += [len(words)] + sorted(words)
    kernels.append(
        ttnn.KernelDescriptor(
            kernel_source=KERNELS["ag_hub"],
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=_crs(*HUB, *HUB),
            compile_time_args=[ROWS_RING - 1, 2 * LINKS, GX, SEM_GO_F, SEM_GO_B, SEM_READY, LINKS],
            runtime_args=[(HUB, hub_rt)],
            defines=[("PROBE_NORELAY", "1")] if probe == "agonly" else [],
            config=ttnn.WriterConfigDescriptor(),
        )
    )
    stage = 2 * BATCH * TPP
    cbs = [
        _cb(CB_STAGE_F, stage, page, xl.dtype, snd_crs),
        _cb(CB_STAGE_B, stage, page, xl.dtype, snd_crs),
        ttnn.CBDescriptor(
            total_size=(BATCH + 1) * HDR,
            core_ranges=snd_crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_HDR_F, data_format=ttnn.uint32, page_size=HDR)],
        ),
        ttnn.CBDescriptor(
            total_size=(BATCH + 1) * HDR,
            core_ranges=snd_crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_HDR_B, data_format=ttnn.uint32, page_size=HDR)],
        ),
    ]
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=col6, initial_value=0) for i in (SEM_GO_F, SEM_GO_B)]
    return kernels, cbs, sems, snd


def check_ring(mesh):
    """The ops gather over the 8 mesh rows as one ring (neighbours (r +- 1) % 8, wrap 7 -> 0 included) and build
    1D fabric packet headers (HDR = 64 B stride): they need 8 mesh rows and FABRIC_1D_RING."""
    rows = int(list(mesh.shape)[0])
    if rows != ROWS_RING:
        raise ValueError(f"mesh rows {rows} != {ROWS_RING}")
    fabric = ttnn.get_fabric_config()
    if fabric != ttnn.FabricConfig.FABRIC_1D_RING:
        raise ValueError(f"fabric config {fabric}, need FABRIC_1D_RING")


def _check_inputs(name, t, dtypes=(ttnn.bfloat8_b, ttnn.bfloat16)):
    if t.layout != ttnn.TILE_LAYOUT or t.memory_config() != ttnn.DRAM_MEMORY_CONFIG or t.dtype not in dtypes:
        raise ValueError(f"{name} must be TILE DRAM interleaved {dtypes}, got {t.layout} {t.memory_config()} {t.dtype}")


RESIDENT_IN1_MAX_BYTES = 720 * 1024


def resident_fits(K: int, N: int, config, in1_tile_bytes: int = TILE_BYTES[ttnn.bfloat8_b]) -> bool:
    """A core's whole K x N_block weight fits the resident in1 CB budget (QKV: 160 x 4 bf8 tiles = 680 KiB)."""
    cfg = MMConfig.of(config)
    return (K // TILE) * cfg.N_block * in1_tile_bytes <= RESIDENT_IN1_MAX_BYTES


# ---------------------------------------------------------------------------------------------------------------
# Public ops
# ---------------------------------------------------------------------------------------------------------------
def matmul_port(
    a,
    w,
    *,
    config,
    compute_kernel_config,
    dtype=ttnn.bfloat8_b,
    slot_rows=None,
    waves=None,
    out=None,
    resident=False,
    split=0,
    gx=GX,
):
    """minimal_matmul(a, w) on the gx x 10 grid with the planned block lists, no fabric (all of a read from DRAM).
    Same bits as the stock call with the same K block / N block (any grid, any M split)."""
    _check_inputs("a", a)
    _check_inputs("w", w)
    cfg = MMConfig.of(config)
    mesh = a.device()
    rows, cols = list(mesh.shape)
    M_tiles = int(a.shape[-2]) // TILE
    T = slot_rows or M_tiles // ROWS_RING
    N = int(w.shape[-1])
    p = plan(T, cfg.M_block, N, tuple(waves) if waves else None, ncols=gx, resident=resident, split=split)
    if out is None:
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, int(a.shape[-2]), N]), dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
    key = ("port", gx, cfg, p, str(compute_kernel_config.math_fidelity), bool(compute_kernel_config.fp32_dest_acc_en))
    key += (bool(compute_kernel_config.math_approx_mode), tuple(mesh.shape))
    key += tuple(tensor_key(t) for t in (a, w, out))
    mp = _DESCRIPTOR_CACHE.get(key)
    if mp is None:
        k, c, s = _matmul_kernels(
            mesh, a, w, out, None, cfg, compute_kernel_config, p.cols[0], 0, T, fused=False, resident=resident, gx=gx
        )
        prog = ttnn.ProgramDescriptor(kernels=k, semaphores=s, cbs=c)
        mp = ttnn.MeshProgramDescriptor()
        mp[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(rows - 1, cols - 1))] = prog
        if len(_DESCRIPTOR_CACHE) >= _DESCRIPTOR_CACHE_MAX:
            _DESCRIPTOR_CACHE.clear()
        _DESCRIPTOR_CACHE[key] = mp
    ttnn.generic_op([a, w, out], mp)
    return out


def qwen_ag_matmul(
    x_local,
    w,
    *,
    config,
    compute_kernel_config,
    dtype=ttnn.bfloat8_b,
    copy_local=False,
    waves=None,
    gathered=None,
    out=None,
    probe=None,
    resident=None,
    split=0,
):
    """Fused ``_row_ag(x_local)`` + ``minimal_matmul(gathered, w)``.

    x_local: [1, 1, S/32, K] bf8 per device (TILE, DRAM interleaved), the row-sharded norm output.
    w: [1, 1, K, N] per device (row r's weight), N / 32 = 10 * N_block.
    Returns (out [1, 1, S/4, N], gathered [1, 1, S/4, K]). gathered holds every other row's slot; this device's own
    slot only with copy_local (FF1, so that FF3 can read the whole buffer), else it is left unwritten.
    """
    _check_inputs("x_local", x_local, (ttnn.bfloat8_b,))
    _check_inputs("w", w)
    cfg = MMConfig.of(config)
    mesh = x_local.device()
    check_ring(mesh)
    rows, cols = list(mesh.shape)
    S32, K = int(x_local.shape[-2]), int(x_local.shape[-1])
    if S32 % TILE or K % TILE or K != int(w.shape[-2]):
        raise ValueError(f"x_local {tuple(x_local.shape)} vs w {tuple(w.shape)}")
    check_k_split(K // TILE)
    T = S32 // TILE
    N = int(w.shape[-1])
    if resident is None:
        resident = resident_fits(K, N, cfg, TILE_BYTES[w.dtype])
    p = plan(T, cfg.M_block, N, tuple(waves) if waves else None, resident=resident, split=split)
    if gathered is None:
        gathered = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, ROWS_RING * S32, K]), x_local.dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
    if out is None:
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, ROWS_RING * S32, N]), dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
    gs = _sems(mesh)
    ckc = compute_kernel_config
    key = ("fused", probe, cfg, p, bool(copy_local), str(ckc.math_fidelity), bool(ckc.fp32_dest_acc_en))
    key += (bool(ckc.math_approx_mode), tuple(tensor_key(t) for t in (x_local, w, gathered, out)), gs.key)
    mp = _DESCRIPTOR_CACHE.get(key)
    if mp is None:
        mp = ttnn.MeshProgramDescriptor()
        K_tiles = K // TILE
        for r in range(rows):
            for c in range(cols):
                coord = ttnn.MeshCoordinate(r, c)
                if probe == "agonly":  # timing probe: the all-gather program alone
                    k0, c0, s0 = [], [], []
                else:
                    k0, c0, s0 = _matmul_kernels(
                        mesh,
                        gathered,
                        w,
                        out,
                        x_local,
                        cfg,
                        ckc,
                        p.cols[r],
                        r,
                        T,
                        fused=True,
                        probe=probe,
                        resident=resident,
                    )
                k1, c1, s1, snd = _fabric_kernels(mesh, x_local, gathered, p, r, K_tiles, copy_local, gs, probe)
                kernels = k0 + k1
                prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=s0 + s1, cbs=c0 + c1)
                me = mesh.get_fabric_node_id(coord)
                fwd_node = mesh.get_fabric_node_id(ttnn.MeshCoordinate((r + 1) % rows, c))
                bwd_node = mesh.get_fabric_node_id(ttnn.MeshCoordinate((r - 1) % rows, c))
                base = len(k0)
                for d, fwd in enumerate((True, False)):
                    for l, (x, y) in enumerate(snd):
                        kidx = base + d * len(snd) + l
                        core = ttnn.CoreCoord(x, y)
                        args = ttnn.setup_fabric_connection(me, fwd_node if fwd else bwd_node, l, prog, core)
                        prog.kernels[kidx].runtime_args[x][y].extend(args)
                mp[ttnn.MeshCoordinateRange(coord, coord)] = prog
        if len(_DESCRIPTOR_CACHE) >= _DESCRIPTOR_CACHE_MAX:
            _DESCRIPTOR_CACHE.clear()
        _DESCRIPTOR_CACHE[key] = mp
    ttnn.generic_op([x_local, w, gathered, out], mp)
    return out, gathered


# ---------------------------------------------------------------------------------------------------------------
# v2: the matmul on the full 7x10 grid, the fabric senders folded into eight in0 receiver cores (kernels/v2_in0.cpp)
# ---------------------------------------------------------------------------------------------------------------
GX2 = 7
OWN = 0xFFFF
# (x, y) of the sender of (link, fwd): in0 sinks of row 9, plus the row-8 relay of column 6
V2_SENDERS = {(0, True): (0, 9), (1, True): (1, 9), (2, True): (2, 9), (3, True): (3, 9)}
V2_SENDERS.update({(0, False): (4, 9), (1, False): (5, 9), (2, False): (6, 9), (3, False): (6, 8)})


def v2_waves(T: int, resident: bool) -> tuple:
    """Lockstep (non-resident, FF1) plans: a short first wave (the matmul starts on the own slot meanwhile), then
    16-row waves (fewer, larger steps re-stream W1 less), the rest last. job383 at S=40960, FF1 ms (stock pair
    2.79-2.90): 12,16,12 2.649; 8,16,16 2.718; 8,8,8,16 2.801; 8x5 2.861; 8,8,16,8 2.943. Resident plans (QKV):
    8-row waves (8x5 1.594; 8,8,8,16 1.662; 8,16,16 1.734)."""
    if resident:
        return default_waves(T)
    first = 12 if T >= 36 else 8
    out, rem = [first], T - first
    while rem > 16 + 8:
        out.append(16)
        rem -= 16
    if rem > 16:
        out += [rem - rem // 2, rem // 2]
    elif rem:
        out.append(rem)
    return tuple(out)


@functools.lru_cache(maxsize=64)
def plan_v2(T: int, my_slot: int, ncols: int = GX2, nslots: int = ROWS_RING, waves: tuple | None = None):
    """Column x: one piece of the own slot (the slot split over the columns), then every wave of remote slot
    (my_slot + 1 + x) % nslots. With 7 columns and 7 remote slots every wave is one step of one block per column."""
    assert ncols == nslots - 1
    waves = tuple(waves) if waves else default_waves(T)
    assert sum(waves) == T and len(waves) <= MAX_WAVES and max(waves) <= 16, waves
    starts = [sum(waves[:j]) for j in range(len(waves))]
    q, rem = divmod(T, ncols)
    pieces = [q + (1 if x < rem else 0) for x in range(ncols)]
    cols, off = [], 0
    for x in range(ncols):
        col = [(my_slot * T + off, pieces[x], OWN)] if pieces[x] else []
        off += pieces[x]
        s = (my_slot + 1 + x) % nslots
        col += [(s * T + starts[j], w, j) for j, w in enumerate(waves)]
        cols.append(tuple(col))
    return waves, tuple(starts), tuple(cols)


class _V2Words:
    """Barrier word (sender cores) and wave words (injectors) at one L1 address on every device and core: a
    zero-filled L1 buffer sharded over the injector and sender cores, created with a device sync."""

    def __init__(self, mesh, cores):
        import torch

        width = (1 + MAX_WAVES) * _GlobalSems.STRIDE // 4
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, [1, width], ttnn.ShardOrientation.ROW_MAJOR),
        )
        n = cores.num_cores()
        self.buf = ttnn.from_torch(
            torch.zeros(1, 1, n, width, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=mem,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        ttnn.synchronize_device(mesh)
        self.addr = self.buf.buffer_address()


_V2: dict = {}


def _v2_word_cores():
    cs = [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX2 - 1, 0))]
    cs += [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in sorted(set(V2_SENDERS.values()))]
    return ttnn.CoreRangeSet(cs)


def release_semaphores_v2(mesh):
    """As release_semaphores, for qwen_ag_matmul_v2's word buffer."""
    w = _V2.pop(id(mesh), None)
    if w is not None:
        ttnn.deallocate(w.buf)


def _v2_program(mesh, xl, w, g, out, cfg, ckc, r, c, T, copy_local, resident, words, probe=None, waves=None):
    waves, starts, cols = plan_v2(T, r, waves=waves)
    pair_lists = [[(m0, rows, m0 // T, j) for m0, rows, j in col] for col in cols]
    # matmul in1 / compute kernels from v1 (pairs), with the grid width GX2
    k1, cbs, sems = _matmul_kernels(
        mesh, g, w, out, xl, cfg, ckc, pair_lists, r, T, fused=False, resident=resident, gx=GX2
    )
    k1 = [k for k in k1 if KERNELS["mm_in0"] not in str(k.kernel_source)]
    M_eff = max(rows for col in cols for _, rows, _ in col)
    K_tiles = int(xl.shape[-1]) // TILE
    in0_ts = TILE_BYTES[g.dtype]
    px = [_phys(mesh, x, 0)[0] for x in range(GX2)]
    py = [_phys(mesh, 0, y)[1] for y in range(GY)]
    head = [g.buffer_address(), xl.buffer_address(), words.addr] + px + py
    offs, body = [], []
    at = len(head) + GX2
    for col in cols:
        offs.append(at + len(body))
        body.append(len(col))
        for m0, rows, j in col:
            body += [m0, rows, j]
    common = head + offs + body
    for r0, rows in zip(starts, waves):
        common += [r0, rows]
    snd_order = [V2_SENDERS[(l, f)] for f in (True, False) for l in range(LINKS)]
    for x, y in snd_order:
        common += list(_phys(mesh, x, y))
    acc = _accessor(g) + _accessor(xl)
    base_ct = [K_tiles, K_tiles, M_eff, cfg.K_block, in0_ts, SEM_IN0_SENDER, SEM_IN0_RECEIVER, SEM_IN0_VALID]
    arrivals = 2 * LINKS * (ROWS_RING - 1)
    dm0 = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)
    M_tiles = int(g.shape[-2]) // TILE

    def ct(inj):
        return base_ct + [int(inj), arrivals, T, r, M_tiles, GX2, GY, len(waves)] + acc

    snd_set = set(snd_order)
    recv = [
        ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
        for x in range(GX2)
        for y in range(1, GY)
        if (x, y) not in snd_set
    ]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_ROOT}/v2_in0.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct(inj),
            common_runtime_args=common,
            defines=[("PROBE_NOWAIT", "1")] if probe == "nowait" else [],
            config=dm0,
        )
        for inj, cores in ((True, _crs(0, 0, GX2 - 1, 0)), (False, ttnn.CoreRangeSet(recv)))
    ]
    page = TILE_BYTES[xl.dtype]
    link_cols = K_tiles // LINKS
    snd_k = []
    for fwd in (True, False):
        ra, rb = (4, 3) if fwd else (3, 4)
        for l in range(LINKS):
            x, y = V2_SENDERS[(l, fwd)]
            sct = [page, ra, rb, int(fwd), int(l == 0), 4 if fwd else 3, BATCH, HDR, l * link_cols, link_cols]
            sct += [int(copy_local), CB_STAGE_F, CB_HDR_F, len(snd_order)]
            snd_k.append((len(kernels), l, fwd, x, y))
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=f"{KERNEL_ROOT}/v2_in0.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=_crs(x, y, x, y),
                    compile_time_args=ct(False) + sct,
                    common_runtime_args=common,
                    runtime_args=[((x, y), [])],
                    defines=[("SENDER", "1")],
                    config=dm0,
                )
            )
    snd_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in snd_order])
    cbs = cbs + [
        _cb(CB_STAGE_F, 2 * BATCH * TPP, page, xl.dtype, snd_crs),
        ttnn.CBDescriptor(
            total_size=(BATCH + GX2) * HDR,
            core_ranges=snd_crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_HDR_F, data_format=ttnn.uint32, page_size=HDR)],
        ),
    ]
    prog = ttnn.ProgramDescriptor(kernels=kernels + k1, semaphores=sems, cbs=cbs)
    coord = ttnn.MeshCoordinate(r, c)
    me = mesh.get_fabric_node_id(coord)
    fwd_node = mesh.get_fabric_node_id(ttnn.MeshCoordinate((r + 1) % ROWS_RING, c))
    bwd_node = mesh.get_fabric_node_id(ttnn.MeshCoordinate((r - 1) % ROWS_RING, c))
    for kidx, l, fwd, x, y in snd_k:
        args = ttnn.setup_fabric_connection(me, fwd_node if fwd else bwd_node, l, prog, ttnn.CoreCoord(x, y))
        prog.kernels[kidx].runtime_args[x][y].extend(args)
    return prog


def qwen_ag_matmul_v2(
    x_local,
    w,
    *,
    config,
    compute_kernel_config,
    dtype=ttnn.bfloat8_b,
    copy_local=False,
    resident=None,
    gathered=None,
    out=None,
    probe=None,
    waves=None,
):
    """qwen_ag_matmul on the full 7x10 grid (no separate sender column): same contract and bits."""
    _check_inputs("x_local", x_local, (ttnn.bfloat8_b,))
    _check_inputs("w", w)
    cfg = MMConfig.of(config)
    mesh = x_local.device()
    check_ring(mesh)
    rows, cols = list(mesh.shape)
    S32, K = int(x_local.shape[-2]), int(x_local.shape[-1])
    if S32 % TILE or K % TILE or K != int(w.shape[-2]):
        raise ValueError(f"x_local {tuple(x_local.shape)} vs w {tuple(w.shape)}")
    check_k_split(K // TILE)
    T = S32 // TILE
    N = int(w.shape[-1])
    if resident is None:
        resident = resident_fits(K, N, cfg, TILE_BYTES[w.dtype])
    if waves is None:
        waves = v2_waves(T, resident)
    if gathered is None:
        gathered = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, ROWS_RING * S32, K]), x_local.dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
    if out is None:
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, ROWS_RING * S32, N]), dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
    if id(mesh) not in _V2:
        _V2[id(mesh)] = _V2Words(mesh, _v2_word_cores())
    words = _V2[id(mesh)]
    ckc = compute_kernel_config
    key = (
        "v2",
        probe,
        tuple(waves) if waves else None,
        cfg,
        T,
        bool(copy_local),
        bool(resident),
        str(ckc.math_fidelity),
        bool(ckc.fp32_dest_acc_en),
    )
    key += (bool(ckc.math_approx_mode), tuple(tensor_key(t) for t in (x_local, w, gathered, out)), words.addr)
    mp = _DESCRIPTOR_CACHE.get(key)
    if mp is None:
        mp = ttnn.MeshProgramDescriptor()
        for r in range(rows):
            for c in range(cols):
                prog = _v2_program(
                    mesh, x_local, w, gathered, out, cfg, ckc, r, c, T, copy_local, resident, words, probe, waves
                )
                mp[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = prog
        if len(_DESCRIPTOR_CACHE) >= _DESCRIPTOR_CACHE_MAX:
            _DESCRIPTOR_CACHE.clear()
        _DESCRIPTOR_CACHE[key] = mp
    ttnn.generic_op([x_local, w, gathered, out], mp)
    return out, gathered
