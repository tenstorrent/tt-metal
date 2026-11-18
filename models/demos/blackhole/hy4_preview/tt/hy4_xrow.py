# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""B45x (HY4_XROW=1, default OFF): device-to-device cross-row COPY of a retained prefix chain of the Hy4 pool (mesh row A -> mesh row B), chain granular on the identity page table.

Per-user state of Hy4 serving is the pool pages only (78 SCALED_FP8 kv pools, 21 indexer key pools, the MTP kv / key pools); page p = tokens [512 p, 512 p + 512) striped over the 8 mesh columns (64-token slices),
one independent pool per mesh row. The column axis is untouched (SPMD); only the row axis moves. Per pool tensor and chunk of CP pool pages:
  1. stage-in : pool pages [p0, p0 + n) -> persistent stage tensor (every device; uint32 RM, page = the pool's page bytes, so the move is dtype blind),
  2. all_gather over the ROW axis (cluster_axis 0) of the stage -> [rows, ...] (every row now holds the donor row's block; a fresh DRAM tensor, freed behind the stage-out),
  3. stage-out: on the receiving row only (control page: skip on the other rows) the donor block -> pool pages [p0, p0 + n).
NEVER ttnn.point_to_point (its program cache parks a GlobalSemaphore of L1 on every worker core per entry: clashes with the SDPA CBs, known boot trap, DSV4 xrow_d2d2 RESULT). ttnn.all_gather (non async) holds no persistent L1.
Source: pattern of DSV4 ``deepseek_v41_flash/tt/xrow_d2d.py`` rows mode (branch ssinghal/dsv4p1-xrow-d2d2 075ffbcedeb); kernel ``xrow_kernels/xrow_copy.cpp`` (JIT, no C++ build).
Any exception = the caller recomputes cold (never a wrong hit): the destination chain is cleared BEFORE the copy.
"""
import os
import time

PAGE_TOK = 512  # tokens per pool page
KV_PAGE_B, KV_UNITS = 704, 64  # SCALED_FP8 row (656 B at the 64 B aligned stride), rows per pool page
KEY_UNITS = 8  # tile pages per pool page of a [P,1,64,128] key pool
KEY_PAGE_B = {"bfloat8_b": 1088, "bfloat16": 2048}
CTL_WORDS = 16
KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xrow_kernels")


def enabled():
    """HY4_XROW=1 and (live file HY4_XROW_FILE, if set, holding 1)."""
    if os.environ.get("HY4_XROW", "0") != "1":
        return False
    f = os.environ.get("HY4_XROW_FILE")
    if f:
        try:
            with open(f) as fh:
                return fh.read().strip() == "1"
        except (OSError, ValueError):
            pass
    return True


def xrow_max():
    return int(os.environ.get("HY4_XROW_MAX", "1048576"))


def min_hit():
    return int(os.environ.get("HY4_XROW_MIN", "1024"))


# ------------------------------------------------------------------ CPU model (pure numpy, tested without a device)
def blocks_of(h):
    """number of pool pages of a copy of the first h tokens; h must be a positive multiple of 512 (a partial last page is never copied: the retained chains are full 512-blocks)."""
    if h <= 0 or h % PAGE_TOK:
        raise ValueError(f"xrow copy length {h} is not a positive multiple of {PAGE_TOK}")
    return h // PAGE_TOK


def chunk_plan(npool, cp):
    """[(p0, n)] chunks of at most cp pool pages covering pages 0..npool-1."""
    return [(p0, min(cp, npool - p0)) for p0 in range(0, npool, cp)]


def page_ranges(p0, n, units, cp, src_row):
    """(stage-in src_page0, stage-out src_page0 in the gathered tensor, dst_page0, npages) of a chunk: pages (units per pool page) of the pool and of the gathered block of ``src_row``."""
    return p0 * units, src_row * cp * units, p0 * units, n * units


def ref_copy(pools, src, dst, h, units):
    """CPU reference of the copy. pools[row][col] = ndarray [pages*units, ...] (one pool tensor of one device); returns a new nested list where rows dst of every column holds pages 0..h/512 of row src."""
    n = blocks_of(h) * units
    out = [[x.copy() for x in row] for row in pools]
    if src == dst:
        raise ValueError("xrow copy: src == dst")
    for c in range(len(pools[0])):
        out[dst][c][:n] = pools[src][c][:n]
    return out


def copy_cost_s(h, per_token_s=0.55 / 65536):
    """predicted copy time (kv-design s6: 0.55 / 1.1 / 2.4 s at 64k / 131k / 291k)."""
    return h * per_token_s


def cold_cost_s(h, per_token_s=5.2e-3):
    return h * per_token_s


def plan_copies(rets, live, taken, prompts, cur_rows, pf_rows=512, own_hit=None, min_hit_tokens=1024, cap=1 << 20):
    """CPU planner of the row choice policy (engine.assign_rows calls it). rets[r] = retained token array or None, live[r] = row decodes, taken = rows already chosen for requests of this call (updated).
    prompts = token arrays. Returns (rows, copies) with copies {request index: (src row, h tokens)}. Policy: among free rows prefer the row with the longest own match; if no free row
    holds the longest match in the whole table by >= min_hit_tokens (and the copy is cheaper than the cold prefill), serve on the free row the own policy picks and copy from the row with the longest match (any row: a
    decoding row's retained pages 0..h stay untouched by its decode). Never copy when no row is free; a source row is never also a destination of this call (its chain must still be there when the copy runs) and
    a destination is never a source."""
    nrows = len(rets)

    def match(r, a):
        t = rets[r]
        if t is None or len(t) == 0:
            return 0
        m = min(len(t), len(a))
        ne = (t[:m] != a[:m]).nonzero()[0]
        return int(ne[0]) if len(ne) else m

    def hit(r, a):
        return (min(match(r, a), len(a) - 1) // pf_rows) * pf_rows

    taken = set(taken)
    out, copies = [None] * len(prompts), {}
    srcs, dsts = set(), set()
    order = []
    for i, a in enumerate(prompts):
        free = [r for r in range(nrows) if not live[r] and r not in taken]
        order.append((max((hit(r, a) for r in free), default=0), i))
    for _, i in sorted(order, reverse=True):
        a = prompts[i]
        c = [r for r in range(nrows) if not live[r] and r not in taken]
        if not c:
            out[i] = cur_rows[i]
            continue
        h, r = max(((hit(q, a), q) for q in c), key=lambda x: x[0])
        if h <= 0:
            r = min(
                c,
                key=lambda q: (
                    rets[q] is not None and len(rets[q]) > 0,
                    0.0 if own_hit is None else own_hit[q],
                    q != cur_rows[i],
                ),
            )
        # copy candidate: the row (other than r) with the longest match
        cand = [(hit(q, a), q) for q in range(nrows) if q != r and q not in dsts]
        hs, s = max(cand, key=lambda x: x[0], default=(0, None))
        hs = min(hs, cap)
        if (
            s is not None
            and hs >= min_hit_tokens
            and hs - (h if h > 0 else 0) >= min_hit_tokens
            and copy_cost_s(hs) < cold_cost_s(hs - h)
        ):
            if r not in srcs and s not in dsts:
                copies[i] = (s, hs)
                srcs.add(s)
                dsts.add(r)
        out[i] = r
        taken.add(r)
    return out, copies


# ------------------------------------------------------------------ device
class Hy4Xrow:
    """persistent stage tensors (built in the engine warm-up, BEFORE the first trace capture) + the copy.

    tensors: list of (name, ttnn tensor [1,1,P*64,656] fp8 kv pool or [P,1,64,128] key pool)."""

    def __init__(self, md, tensors, key_dtype="bfloat8_b"):
        import ttnn

        self.md, self.tensors = md, list(tensors)
        self.rows, self.cols = int(md.shape[0]), int(md.shape[1])
        self.cp_kv = int(os.environ.get("HY4_XROW_CP_KV", "128"))
        self.cp_key = int(os.environ.get("HY4_XROW_CP_KEY", "256"))
        self.batch = int(os.environ.get("HY4_XROW_B", "16"))
        self.ncores = int(os.environ.get("HY4_XROW_CORES", "32"))
        self.topo = ttnn.Topology.Ring if self.rows >= 4 else ttnn.Topology.Linear
        self.shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape))
        self.stage, self.kind = {}, {}
        self.tensors = [(x[0], x[1]) for x in self.tensors]
        for x in tensors:
            name, t = x[0], x[1]
            kind = (
                x[2] if len(x) > 2 else ("kv" if int(t.shape[-1]) == 656 else "key")
            )  # (kind override: dtype-blind probes)
            self.kind[name] = kind
            pb, u, cp = (
                (KV_PAGE_B, KV_UNITS, self.cp_kv) if kind == "kv" else (KEY_PAGE_B[key_dtype], KEY_UNITS, self.cp_key)
            )
            if (kind, pb) not in self.stage:
                self.stage[(kind, pb)] = ttnn.zeros(
                    [1, 1, cp * u, pb // 4],
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=md,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
        self.key_page = KEY_PAGE_B[key_dtype]
        self.ctl_in = self._ctl(0, None)
        self.stats = dict(copies=0, failed=0, tokens=0, seconds=0.0)
        self.warm_s = 0.0

    def _ctl(self, off, dst_row):
        """uint32 RM [rows,1,1,16] sharded over the rows: word 0 = page offset of the donor block, word 1 = skip (all rows but ``dst_row``; dst_row None = skip nowhere)."""
        import torch

        import ttnn

        c = torch.zeros(self.rows, 1, 1, CTL_WORDS, dtype=torch.int32)
        c[:, 0, 0, 0] = off
        if dst_row is not None:  # dst_row -1: every row skips
            c[:, 0, 0, 1] = 1
            if dst_row >= 0:
                c[dst_row, 0, 0, 1] = 0
        return ttnn.from_torch(
            c,
            device=self.md,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.shard,
        )

    def _prog(self, src, dst, ctl, npages, src_page0, dst_page0, page):
        import ttnn
        from models.demos.blackhole.hy4_preview.tt.hy4_pool import _acc, _core_set, _cores, _hash

        grid = self.md.compute_with_storage_grid_size()
        nc = self.ncores  # fixed core count: the program hash must not depend on the chunk size
        cores = _cores(nc, grid.x)
        cs = _core_set(cores)
        rt = ttnn.RuntimeArgs()
        for i, (x, y) in enumerate(cores):
            rt[x][y] = [i]
        B = self.batch
        cb = ttnn.CBDescriptor(
            total_size=B * page + 64,
            core_ranges=cs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.int32, page_size=B * page + 64)
            ],
        )
        ct = [page, nc, 0, B] + _acc(src) + _acc(dst) + _acc(ctl)
        k = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/xrow_copy.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cs,
            compile_time_args=ct,
            runtime_args=rt,
            common_runtime_args=[
                src.buffer_address(),
                dst.buffer_address(),
                ctl.buffer_address(),
                npages,
                src_page0,
                dst_page0,
            ],
            config=ttnn.ReaderConfigDescriptor(),
        )
        prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])
        prog.custom_program_hash = _hash(0x9C1, page, nc, B, tuple(ct))
        ttnn.generic_op([src, dst, ctl], prog)

    def _one(self, name, t, ctl_out, p0, n):
        import ttnn

        kind = self.kind[name]
        pb, u, cp = (KV_PAGE_B, KV_UNITS, self.cp_kv) if kind == "kv" else (self.key_page, KEY_UNITS, self.cp_key)
        stage = self.stage[(kind, pb)]
        self._prog(t, stage, self.ctl_in, n * u, p0 * u, 0, pb)
        full = ttnn.all_gather(stage, dim=0, cluster_axis=0, num_links=1, topology=self.topo)  # [rows, 1, cp*u, pb/4]
        self._prog(full, t, ctl_out, n * u, 0, p0 * u, pb)
        return full

    def copy(self, src, dst, h, only=None, allow_same=False):
        """pool[dst][pages 0..h/512) = pool[src][...] for every pool tensor (in place, synchronous). Raises on any error (the caller recomputes cold). ``only``: restrict to the first N tensors (probes)."""
        import ttnn

        if not (0 <= src < self.rows and 0 <= dst < self.rows) or (src == dst and not allow_same):
            raise ValueError(f"xrow copy rows {src} -> {dst}")
        npool = blocks_of(h)
        t0 = time.perf_counter()
        keep = []
        try:
            for name, t in self.tensors[:only]:
                kind = self.kind[name]
                cp, u = (self.cp_kv, KV_UNITS) if kind == "kv" else (self.cp_key, KEY_UNITS)
                ctl_out = self._ctl(src * cp * u, dst)
                keep.append(ctl_out)
                for p0, n in chunk_plan(npool, cp):
                    keep.append(self._one(name, t, ctl_out, p0, n))
            ttnn.synchronize_device(self.md)
        except Exception:
            self.stats["failed"] += 1
            raise
        finally:
            for x in keep:
                ttnn.deallocate(x)
        dt = time.perf_counter() - t0
        self.stats["copies"] += 1
        self.stats["tokens"] += h
        self.stats["seconds"] += dt
        return dt

    def warm(self):
        """compile every program with a copy that moves nothing (all rows skip): stage-in programs run (harmless), the stage-out programs skip everywhere."""
        import ttnn

        t0 = time.perf_counter()
        keep = []
        ctl_skip = self._ctl(0, -1)  # every row skips
        for name, t in self.tensors:
            kind = self.kind[name]
            cp, u = (self.cp_kv, KV_UNITS) if kind == "kv" else (self.cp_key, KEY_UNITS)
            keep.append(self._one(name, t, ctl_skip, 0, cp))
        ttnn.synchronize_device(self.md)
        for x in keep + [ctl_skip]:
            ttnn.deallocate(x)
        self.warm_s = time.perf_counter() - t0
        return self.warm_s
