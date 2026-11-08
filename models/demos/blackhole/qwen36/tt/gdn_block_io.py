# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_MAMBA_SPEC: the GDN state moves between the B=1 prefill scratch and vLLM's state blocks as ONE generic_op
(kernels/gdn_block_io.cpp) each, instead of the per-layer ttnn op chains of SpecGdnBlockState (_write_rec /
_write_carry / _read_rec / _read_carry and the paged_fill_cache zeroing).

Why. A prefill of the new path writes its final state into the request's block (about 550 small ops), the state at
vLLM's last checkpoint into the checkpoint block (about 400 ops) and zeroes the attention blocks a GDN group held
(32 paged_fill_cache calls of 8 blocks). Each op moves a few kilobytes and costs a few microseconds of launch, so these
took 4.5, 3.3 and 2.0 ms of device time before the first token. One program per move reads and writes the same pages
from every core of the grid.

Same pages, same bytes: the rec heads are the 16 fp32 tile pages SpecBlockLayout.head_pages names, the conv history is
the bf16 tile pages of conv_page holding the (kc - 1) x C carry in row-major order, zero padded, as _write_carry leaves
it. A restore writes conv_carry's tile rows past kc - 1 as zeros. tests/test_spec_block_state_device.py checks each
mode against the op chains.

The descriptor is rebuilt per call: block pages and buffer addresses are common runtime args, which do not enter the
program hash, so each mode compiles once per core grid (WINDOW and CHAIN grids grow with the entry count) and result
accessor in SpecGdnBlockState.warmup, and every later call reuses it. Nothing is allocated per call.
"""

from pathlib import Path

import ttnn

KERNEL = str(Path(__file__).resolve().parent / "kernels" / "gdn_block_io.cpp")
WRITE, WRITE_SNAP, RESTORE, ZERO, WINDOW, CHAIN = 0, 1, 2, 3, 4, 5
CB_L1 = 0
FP32_PAGE, BF16_PAGE = 4096, 2048
BATCH = 4  # fp32 pages per read / write round of one core (kernels/gdn_block_io.cpp)
ZERO_MAX_BLOCKS = 32  # block ids per ZERO call: a fixed runtime-arg count keeps one program
WIN_MAX = 16  # entries per WINDOW call, for the same reason
RES_COUNTS = 1  # spec_chain.RES_COUNTS: the result buffer row of accepted counts by engine row
NO_T = 0xFFFFFFFF
MAX_RUNTIME_ARGS = 341


def _al64(x):
    return (int(x) + 63) // 64 * 64


def _accessor(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _interleaved(t, layout, dtype):
    if t.layout != layout:
        return f"{tuple(t.shape)} layout {t.layout}"
    if t.dtype != dtype:
        return f"{tuple(t.shape)} dtype {t.dtype}"
    if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return f"{tuple(t.shape)} not interleaved"
    return None


class GdnBlockIO:
    """The block moves of one SpecGdnBlockState ``state`` (its layers, layout, groups and mesh) as generic_ops.
    ``kv_caches``: the model's [(K, V)] attention caches, for zero()."""

    def __init__(self, state, kv_caches=()):
        self.state = state
        self.kv = list(kv_caches)
        grid = state.mesh.compute_with_storage_grid_size()
        self.gx, self.gy = int(grid.x), int(grid.y)

    # ------------------------------------------------------------------ contract
    def reason(self):
        """None if every layer's tensors fit the kernel's contract, else why not (the op chains then run)."""
        st, lay = self.state, self.state.layout
        if not st.layers:
            return "no GDN layers"
        n_args = 1 + st.G * (lay.nv + 1) + 5 * len(st.layers)
        if n_args > MAX_RUNTIME_ARGS:
            return f"{n_args} runtime args"
        v32 = v16 = None
        for _, dn, _ in st.layers:
            pv = dn._spec_paged
            if dn.Dk != 128 or dn.Dv != 128:
                return f"head dims {dn.Dk} x {dn.Dv}"
            if dn.qkv_dim_tp % 32 or dn.K - 1 > 16:
                return f"conv carry {dn.K - 1} x {dn.qkv_dim_tp}"
            if (dn.K - 1) * (dn.qkv_dim_tp // 32) > lay.conv_pages * 32:
                return "conv history pages"
            for view in (pv.rec1, pv.rec2):
                acc = _accessor(view)
                if v32 is None:
                    v32 = acc
                elif acc != v32:
                    return "fp32 views differ in layout"
            for view in (pv.conv, pv.conv_k):
                acc = _accessor(view)
                if v16 is None:
                    v16 = acc
                elif acc != v16:
                    return "bf16 views differ in layout"
            if pv.rec1.buffer_address() != pv.conv_k.buffer_address() or (
                pv.rec2.buffer_address() != pv.conv.buffer_address()
            ):
                return "the fp32 and bf16 views of a buffer start at different addresses"
            if (
                int(pv.rec1.buffer_aligned_page_size()) != FP32_PAGE
                or int(pv.conv.buffer_aligned_page_size()) != BF16_PAGE
            ):
                return "view page sizes"
        return None

    def zero_reason(self):
        if not self.kv:
            return "no KV caches"
        acc, tpb = None, None
        for pair in self.kv:
            for t in pair:
                why = _interleaved(t, ttnn.TILE_LAYOUT, t.dtype)
                if why is not None:
                    return why
                a = _accessor(t)
                if acc is None:
                    acc = a
                elif a != acc:
                    return "K / V caches differ in layout"
                shape = [int(x) for x in t.shape]
                n = shape[1] * (shape[2] // 32) * (shape[3] // 32)
                if tpb is None:
                    tpb = n
                elif n != tpb:
                    return "K / V caches differ in block size"
                if int(t.buffer_aligned_page_size()) > FP32_PAGE:
                    return f"page of {int(t.buffer_aligned_page_size())} B"
        if 2 * len(self.kv) + 1 + ZERO_MAX_BLOCKS > MAX_RUNTIME_ARGS:
            return "runtime args"
        return None

    def _scratch_reason(self, mode, snap):
        for _, dn, _ in self.state.layers:
            if mode == WRITE_SNAP:
                rec, conv = getattr(dn, "_snap_rec", None), getattr(dn, "_snap_conv_rm", None)
                if rec is None or conv is None:
                    return "no in-replay snapshot buffers"
                if not 0 <= int(snap) < int(rec.shape[0]):
                    return f"snapshot slot {snap}"
                checks = ((rec, ttnn.TILE_LAYOUT, ttnn.float32), (conv, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16))
            else:
                rec, conv = dn.rec_state, dn.conv_carry
                checks = ((rec, ttnn.TILE_LAYOUT, ttnn.float32), (conv, ttnn.TILE_LAYOUT, ttnn.bfloat16))
            for t, layout, dtype in checks:
                why = _interleaved(t, layout, dtype)
                if why is not None:
                    return why
            if tuple(int(x) for x in list(rec.shape)[1:]) != (dn.Nv, 128, 128) or int(conv.shape[-1]) != dn.qkv_dim_tp:
                return f"scratch shapes {tuple(rec.shape)} {tuple(conv.shape)}"
        return None

    # ------------------------------------------------------------------ program
    def _cores(self, items):
        n = max(1, min(int(items), self.gx * self.gy))
        nx = min(n, self.gx)
        ny = -(-n // nx)
        return n, nx, ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nx - 1, ny - 1))])

    def _run(self, mode, cta_extra, accessors, common, io, items, l1):
        n, nx, cores = self._cores(items)
        cb_bytes = _al64(l1) + 64 + 64  # + the counts line of CHAIN, + the base alignment
        cta = [mode] + cta_extra[:8] + [nx, n, CB_L1, cb_bytes] + cta_extra[8:]
        for t in accessors:
            cta += _accessor(t)
        kernel = ttnn.KernelDescriptor(
            kernel_source=KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=cta,
            common_runtime_args=[int(x) for x in common],
            config=ttnn.ReaderConfigDescriptor(),
        )
        cb = ttnn.CBDescriptor(
            total_size=cb_bytes,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_L1, data_format=ttnn.uint32, page_size=cb_bytes)
            ],
        )
        return ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb]))

    def _move(self, mode, blocks, snap=0):
        """WRITE / WRITE_SNAP / RESTORE between the scratch (or snapshot slot ``snap``) and ``blocks`` ([G] ids)."""
        st, lay = self.state, self.state.layout
        why = self._scratch_reason(mode, snap)
        if why is not None:
            raise ValueError(f"GDN block I/O: {why}")
        _, dn0, _ = st.layers[0]
        kc1, C = dn0.K - 1, dn0.qkv_dim_tp
        common = [int(snap)]
        for g in range(st.G):
            b = int(blocks[g])
            common += [int(p) for p in lay.head_pages([b])[0].tolist()] + [int(lay.conv_page(b))]
        for _, dn, g in st.layers:
            pv = dn._spec_paged
            if mode == WRITE_SNAP:
                rec, conv = dn._snap_rec, dn._snap_conv_rm
            else:
                rec, conv = dn.rec_state, dn.conv_carry
            common += [
                g,
                rec.buffer_address(),
                conv.buffer_address(),
                pv.rec1.buffer_address(),
                pv.rec2.buffer_address(),
            ]
        if mode == WRITE_SNAP:
            rec0, conv0 = dn0._snap_rec, dn0._snap_conv_rm
        else:
            rec0, conv0 = dn0.rec_state, dn0.conv_carry
        rm_row = int(conv0.buffer_aligned_page_size()) if mode == WRITE_SNAP else 0
        ct = C // 32
        stg = {
            WRITE: ct * 2 * _al64(kc1 * 32 + 32),
            WRITE_SNAP: kc1 * _al64(rm_row),
            RESTORE: lay.conv_pages * BF16_PAGE,
        }[mode]
        asm = (BATCH if mode == RESTORE else lay.conv_pages) * BF16_PAGE
        l1 = BATCH * FP32_PAGE + _al64(stg) + asm
        pv0 = dn0._spec_paged
        kv0 = self.kv[0][0] if self.kv else pv0.k
        cta = [len(st.layers), lay.nv, st.G, lay.pool1_pages, lay.conv_pages_total, lay.conv_pages, kc1, ct]
        # KV_BUFS, KV_TPB, ZERO_CHUNK, ZERO_MAX_BLOCKS (ZERO only), RM_ROW, WIN_MAX, CH_MAX
        cta += [0, 1, 1, 0, rm_row, WIN_MAX, self._ch_max()]
        self._run(
            mode,
            cta,
            (pv0.rec1, pv0.conv, rec0, conv0, kv0),
            common,
            [rec0, pv0.rec1],
            len(st.layers) * (lay.nv + 1),
            l1,
        )

    def write(self, blocks):
        """The bound scratch (rec_state, conv_carry of every GDN layer) -> ``blocks`` ([G] ids)."""
        self._move(WRITE, blocks)

    def write_snapshot(self, blocks, k):
        """In-replay snapshot slot ``k`` (_snap_rec, _snap_conv_rm) -> ``blocks``."""
        self._move(WRITE_SNAP, blocks, snap=k)

    def restore(self, blocks):
        """``blocks`` -> the bound scratch."""
        self._move(RESTORE, blocks)

    def zero(self, block_ids):
        """Zero ``block_ids`` in every K and V buffer, ZERO_MAX_BLOCKS ids per program."""
        ids = [int(b) for b in block_ids]
        if not ids:
            return
        bufs = [t for pair in self.kv for t in pair]
        k0 = bufs[0]
        shape = [int(x) for x in k0.shape]
        tpb = shape[1] * (shape[2] // 32) * (shape[3] // 32)
        chunk = next(c for c in (32, 16, 8, 4, 2, 1) if tpb % c == 0)
        for i in range(0, len(ids), ZERO_MAX_BLOCKS):
            part = ids[i : i + ZERO_MAX_BLOCKS]
            common = [t.buffer_address() for t in bufs] + [len(part)] + part + [0] * (ZERO_MAX_BLOCKS - len(part))
            # unused move arguments: nonzero where the kernel divides by them
            cta = [0, 1, 0, 0, 0, 1, 1, 1, len(bufs), tpb, chunk, ZERO_MAX_BLOCKS, 0, WIN_MAX, 1]
            items = len(part) * len(bufs) * (tpb // chunk)
            self._run(ZERO, cta, (k0, k0, k0, k0, k0), common, [k0, bufs[-1]], items, FP32_PAGE)

    def window_reason(self):
        """None if every layer's verify window fits WINDOW, else why not."""
        for _, dn, _ in self.state.layers:
            w = dn.verify_win_cur()
            why = _interleaved(w, ttnn.TILE_LAYOUT, ttnn.bfloat16)
            if why is not None:
                return why
            if int(w.shape[-2]) > 16 or int(w.shape[-1]) != dn.qkv_dim_tp:
                return f"window {tuple(w.shape)}"
        return None

    def window(self, entries):
        """The boundary conv writes: entry (u, t, g, b) writes the conv rows after token t of row u's verify window
        (verify_win_cur, rows t + 1 .. t + kc - 1) into block b's conv history in every GDN layer of group g."""
        st, lay = self.state, self.state.layout
        entries = [tuple(int(x) for x in e) for e in entries]
        if not entries:
            return
        _, dn0, _ = st.layers[0]
        kc1, C = dn0.K - 1, dn0.qkv_dim_tp
        L = int(dn0.verify_win_cur().shape[-2])
        if any(not 0 <= t <= L - 1 - kc1 for _, t, _, _ in entries):
            raise ValueError(f"GDN block I/O: window column outside rows 1 .. {L - 1}: {entries}")
        layers = []
        for _, dn, g in st.layers:
            pv = dn._spec_paged
            layers += [g, 0, dn.verify_win_cur().buffer_address(), pv.rec1.buffer_address(), pv.rec2.buffer_address()]
        ct = C // 32
        l1 = BATCH * FP32_PAGE + ct * 2 * _al64(kc1 * 32 + 32) + lay.conv_pages * BF16_PAGE
        w0 = dn0.verify_win_cur()
        cta = [len(st.layers), lay.nv, st.G, lay.pool1_pages, lay.conv_pages_total, lay.conv_pages, kc1, ct]
        cta += [0, 1, 1, 0, 0, WIN_MAX, self._ch_max()]
        pv0 = dn0._spec_paged
        for i in range(0, len(entries), WIN_MAX):
            part = entries[i : i + WIN_MAX]
            common = [len(part)]
            for u, t, g, b in part:
                common += [u, t, g, int(lay.conv_page(b))]
            common += [0] * (4 * (WIN_MAX - len(part))) + layers
            accessors = (pv0.rec1, pv0.conv, w0, w0, pv0.k)
            self._run(WINDOW, cta, accessors, common, [w0, pv0.conv], len(part) * len(st.layers), l1)

    def _entry_counts(self, per_entry, most):
        """Entry counts 1 .. ``most``, one per core grid _cores gives a call of that many entries (``per_entry`` work
        items each). The grid is in the program hash, so a call with another entry count can need another program."""
        counts, grids = [], set()
        for m in range(1, int(most) + 1):
            n = self._cores(m * per_entry)[0]
            if n not in grids:
                grids.add(n)
                counts.append(m)
        return counts

    def warmup(self, results=()):
        """Compile every WINDOW and CHAIN program a step can build, writing only vLLM's null block 0: one call per core
        grid, and CHAIN without a result buffer and with each of ``results`` (the result accessor is a compile-time
        argument). Call before the decode traces are captured: a program compiled later misses the program cache."""
        st = self.state
        nl = len(st.layers)
        for m in self._entry_counts(nl, WIN_MAX):
            self.window([(0, 0, e % st.G, 0) for e in range(m)])
        for res in (None, *results):
            for m in self._entry_counts(nl * (st.layout.nv + 1), self._ch_max()):
                self.chain([(0, e % st.G, 0, 0, None, (1,)) for e in range(m)], res)

    def _ch_max(self):
        """CHAIN entries per program: what the runtime args leave after the header and the layers."""
        nv = self.state.layout.nv if self.state.layout is not None else 1
        return max(1, (MAX_RUNTIME_ARGS - 3 - 5 * len(self.state.layers)) // (4 + 2 * (nv + 1)))

    def chain(self, entries, result=None):
        """The chained copies: entry (u, g, src, dst, t, counts) copies block src's state into block dst in every GDN
        layer of group g, with the conv history of src (t None) or the conv rows after token t of row u's verify
        window. ``result`` (the pending fused step's int32 ROW_MAJOR result buffer): an entry runs only where row u's
        count (row RES_COUNTS) is in ``counts``; None: every entry runs (the host already selected them)."""
        st, lay = self.state, self.state.layout
        entries = list(entries)
        if not entries:
            return
        _, dn0, _ = st.layers[0]
        kc1, C = dn0.K - 1, dn0.qkv_dim_tp
        L = int(dn0.verify_win_cur().shape[-2])
        if any(t is not None and not 0 <= int(t) <= L - 1 - kc1 for _, _, _, _, t, _ in entries):
            raise ValueError(f"GDN block I/O: chained copy window column outside rows 1 .. {L - 1}")
        if result is not None:
            why = _interleaved(result, ttnn.ROW_MAJOR_LAYOUT, ttnn.int32)
            u_max = max(int(e[0]) for e in entries)
            if why is None and int(result.buffer_aligned_page_size()) < (4 * u_max) // 64 * 64 + 64:
                why = f"result rows of {int(result.buffer_aligned_page_size())} B"
            if why is not None:
                raise ValueError(f"GDN block I/O: result buffer {why}")
        pages = lambda b: [int(p) for p in lay.head_pages([int(b)])[0].tolist()] + [int(lay.conv_page(int(b)))]
        layers = []
        for _, dn, g in st.layers:
            pv = dn._spec_paged
            layers += [g, 0, dn.verify_win_cur().buffer_address(), pv.rec1.buffer_address(), pv.rec2.buffer_address()]
        ct, ch_max = C // 32, self._ch_max()
        words = 4 + 2 * (lay.nv + 1)
        stg = max(ct * 2 * _al64(kc1 * 32 + 32), lay.conv_pages * BF16_PAGE)
        l1 = BATCH * FP32_PAGE + stg + lay.conv_pages * BF16_PAGE
        w0 = dn0.verify_win_cur()
        pv0 = dn0._spec_paged
        cta = [len(st.layers), lay.nv, st.G, lay.pool1_pages, lay.conv_pages_total, lay.conv_pages, kc1, ct]
        cta += [0, 1, 1, 0, 0, WIN_MAX, ch_max]
        res = result if result is not None else w0
        for i in range(0, len(entries), ch_max):
            part = entries[i : i + ch_max]
            common = [len(part), res.buffer_address(), int(result is not None)]
            for u, g, src, dst, t, counts in part:
                mask = sum(1 << (int(c) - 1) for c in counts)
                common += [int(u), int(g), NO_T if t is None else int(t), mask] + pages(src) + pages(dst)
            common += [0] * (words * (ch_max - len(part))) + layers
            accessors = (pv0.rec1, pv0.conv, w0, w0, res)
            items = len(part) * len(st.layers) * (lay.nv + 1)
            self._run(CHAIN, cta, accessors, common, [w0, pv0.rec1], items, l1)
