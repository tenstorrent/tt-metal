#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""l1_march.py: run.sh's L1 march check, right after G3/G3b in the bus-bar mix.

The stage-2 coverage review: run.sh had no SRAM/L1 check. One pass writes one pattern to an L1 tensor
HEIGHT_SHARDED over the compute grid (one shard per Tensix core, sized so the source and its copy fill the allocatable
L1), copies it on the device L1 to L1 onto the grid shifted by one column (shard i moves to the next core), reads the
copy back per chip and compares it word for word with the written pattern. Only on a mismatch is the source read back
too, so the bad core is named exactly: a bad source word is the source core's, a good source with a bad copy is the
copy's core. Each pattern runs twice in a row, one pass each way (columns 0..X-2 -> 1..X-1, then back; schedule()), so
every core is written directly and read directly and each edge column holds the pattern at both buffer addresses.
Patterns (16-bit words, both halves of each uint32): 0x5555 and 0xAAAA first (every cell at 0 and 1 at both addresses
in four passes), then address-in-data (shard index and word index), 0x0000 and 0xFFFF; full mode adds the inverses.
Passes loop until the wall cap (always at least one pattern both ways: a pattern's second pass always runs). Plain
leaves MARGIN free per core for the copy op; full starts at no margin and steps up (FULL_MARGINS) only while the
allocator or the copy runs out of L1, so the march covers the allocatable L1 down to the allocator's minimum. The two
buffers' L1 addresses are logged per direction (the result's addrs), so the L1 range covered is measured.

Vote across chips: a mismatch that more than half of the chips show identically is the test's own (layout, op), so it
is a WARN that names nothing; a chip that differs alone is its L1 FAIL (fleet/chunk_sched.l1_lines), one line per core
and pattern: 'L1 FAIL <BDF> tray N core x,y pattern P words k'. It covers only the allocatable L1: firmware and kernel
config regions are N3's.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))  # the sibling programs, also when run or imported through a link
import re
import time

W = 256  # uint32 words per row: one row is 1 KB of L1
MARGIN = 128 << 10  # L1 bytes per core left free for the copy op's own buffers
FULL_MARGINS = [0, 16 << 10, 32 << 10, 64 << 10, MARGIN]  # full: tried in turn while the L1 runs out
OOM = re.compile(r"out of memory|not enough space|clash with l1|beyond max l1|circular buffer", re.I)  # L1 ran out
FALLBACK_KB = 256  # per-core shard size when the free L1 cannot be read
BASE = [("5555", 0x55555555), ("aaaa", 0xAAAAAAAA), ("addr", None), ("0000", 0x00000000), ("ffff", 0xFFFFFFFF)]


def patterns(full=False):
    """[(name, word or None for address-in-data, invert)]: the base five, plus their inverses in full mode."""
    out = [(n, v, False) for n, v in BASE]
    if full:
        out += [("~" + n, v, True) for n, v in BASE]
    return out


def schedule(k, pats, full=False):
    """Pass k's (pattern, src, dst) columns: src 0 copies columns 0..X-2 -> 1..X-1, src 1 copies back. Every pattern runs
    twice in a row, even pass then odd pass, in both modes (5555, aaaa first): an edge column is a source only one way and
    a copy only the other, so only both passes put the pattern at both buffer addresses there. A plain alternation over
    five patterns gave most patterns one direction only within plain's wall cap. full is kept for the callers."""
    return pats[(k // 2) % len(pats)], k % 2, 1 - k % 2


def pattern_data(torch, pat, n, rows):
    """Pattern pat's (n * rows, W) int32 words: shard s holds rows s*rows..(s+1)*rows-1. Address-in-data puts the
    shard index in the top 12 bits and the word index within the shard in the low 20, so every word is unique."""
    name, word, inv = pat
    if word is None:
        idx = torch.arange(rows * W, dtype=torch.int64).view(1, rows, W)
        sh = torch.arange(n, dtype=torch.int64).view(n, 1, 1)
        g = ((sh & 0xFFF) << 20) | (idx & 0xFFFFF)
    else:
        g = torch.full((n, rows, W), word, dtype=torch.int64)
    if inv:
        g = g ^ 0xFFFFFFFF
    g = torch.where(g >= 2**31, g - 2**32, g)  # uint32 bits in int32
    return g.to(torch.int32).reshape(n * rows, W)


def l1_free(ttnn, mesh):
    """(bytes free per L1 bank, source): the largest free L1 block per core, else half the core's L1, else None."""
    dev = mesh
    try:
        dev = mesh.get_devices()[0]
    except Exception:  # noqa: BLE001
        pass
    try:
        mv = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
        return int(mv.largest_contiguous_bytes_free_per_bank), "memory_view"
    except Exception:  # noqa: BLE001
        pass
    try:
        return int(dev.l1_size_per_core()) // 2, "l1_size_per_core/2"
    except Exception:  # noqa: BLE001
        return None, "fallback"


def hexa(a):
    return "?" if a is None else f"0x{a:x}"


def grid(mesh):
    g = mesh.compute_with_storage_grid_size()
    return int(g.x), int(g.y)


def mem(ttnn, x0, gx, gy, rows):
    """HEIGHT_SHARDED L1 config over columns x0..x0+gx-1, rows 0..gy-1: shard i on core (x0 + i % gx, i // gx)."""
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, 0), ttnn.CoreCoord(x0 + gx - 1, gy - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs, [rows, W], ttnn.ShardOrientation.ROW_MAJOR),
    )


def baddr(ttnn, t):
    """A device tensor's L1 buffer address (the mesh tensor's, else its first chip's), None when ttnn cannot say."""
    for f in (lambda: t.buffer_address(), lambda: ttnn.get_device_tensors(t)[0].buffer_address()):
        try:
            return int(f())
        except Exception:  # noqa: BLE001
            pass
    return None


class Retry(Exception):
    """The copy ran out of L1: the pass reruns with the next margin."""


def inject(got, chips, k, src, at, cx, gy, rows, copied):
    """With TRAY_INJECT naming N1 and a chip: one bit flipped in that chip's readback, in the first word of the shard on
    the spec's core (default: the read's first core, at,0). A core on the copy side is the copy's word; a core on the
    source side only is the source's word, carried into the copy (then inject_src() flips the source readback too).
    Returns [(chip, word)] whose source readback must carry the flip."""

    import tray_inject

    out = []
    for i in range(len(got)):
        s = tray_inject.want("N1", chips.get(i, {}).get("pci"), k)
        if s is None:
            continue
        x, y = s["core"] or (at, 0)
        if at <= x < at + cx and 0 <= y < gy:
            x0 = at
        elif copied and src <= x < src + cx and 0 <= y < gy:
            x0 = src
        else:  # off the grid: the read's first core
            x0, x, y = at, at, 0
        j = ((y * cx) + (x - x0)) * rows * W
        tray_inject.applied(s, core=(x, y))
        got[i] = tray_inject.flip_tensor(s, got[i], j)
        if x0 != at:
            out.append((i, j))
    return out


def inject_src(sg, g, j):
    """TRAY_INJECT: the source readback sg with word j as the injected copy g has it (a bad source word)."""
    sg = sg.clone()
    sg.view(-1)[j] = g.reshape(-1)[j]
    return sg


def l1_check(ttnn, torch, mesh, chips, wall_s, full=False, kb=0, beat=None, say=None):
    """N1: write, copy L1 to the next core, read back and compare until wall_s s. chips: mesh index -> dict(pci, tray,
    dev). kb: per-core shard KB (0 = sized to the allocatable L1). Returns dict(passes, patterns, done, copy, kb_per_core,
    margin_kb (L1 left free per core; None when kb or no free size set the shard), addrs={even|odd: dict(src, copy) L1
    buffer addresses}, cores, l1_free, wall_s, test_side, chips={mesh index: dict(chip, passes, bad_passes, bad (words),
    first_pass, first_t, cores={"x,y": {pattern: words}})}). copy is 'reshard' while ttnn.to_memory_config runs, else
    'host' (a host write and readback of the source only) with copy_error. In full mode an out-of-L1 error at the source
    write or the copy reruns the pass with the next FULL_MARGINS margin, keeping the margin that worked."""
    gx, gy = grid(mesh)
    if gx < 2 or gy < 1:
        raise RuntimeError(f"compute grid {gx}x{gy}: no column to copy to")
    cx = gx - 1  # each side of the copy is cx columns wide
    n = cx * gy
    free, fsrc = l1_free(ttnn, mesh)
    sized = bool(free) and not kb
    margins = FULL_MARGINS if (full and sized) else [MARGIN]
    mi = 0
    if kb:
        rows = int(kb)
    elif free:
        rows = max(1, (free - margins[0]) // 2 // (W * 4))
    else:
        rows = FALLBACK_KB
    pats = patterns(full)
    new = lambda: dict(passes=0, bad_passes=0, bad=0, first_pass=None, first_t=None, cores={})  # noqa: E731
    st = {i: new() for i in chips}
    shard_of = torch.arange(n * rows).view(n * rows, 1).expand(n * rows, W) // rows
    t0 = time.monotonic()
    k, copy, cerr, last, test_side, done, addrs = 0, "reshard", None, 0.0, [], set(), {}

    def grow(e):
        """An out-of-L1 error e with a larger margin left: size the shards for it and say so (the pass reruns)."""
        nonlocal mi, rows, shard_of
        if not (OOM.search(str(e)) and mi + 1 < len(margins)):
            return False
        mi += 1
        rows = max(1, (free - margins[mi]) // 2 // (W * 4))
        shard_of = torch.arange(n * rows).view(n * rows, 1).expand(n * rows, W) // rows
        if say:
            say(
                f"L1 margin {margins[mi - 1] >> 10} KB ran out ({type(e).__name__}: {str(e)[:120]}): "
                f"{margins[mi] >> 10} KB, {rows} KB per core"
            )
        return True

    def core(x0, s):
        return f"{x0 + s % cx},{s // cx}"

    def read(t):
        return [
            ttnn.to_torch(d).contiguous().view(torch.int32).reshape(n * rows, W)
            for d in ttnn.get_device_tensors(t.cpu())
        ]

    while k == 0 or k % 2 == 1 or time.monotonic() - t0 + last < wall_s:  # a pattern's odd pass always runs
        p0 = time.monotonic()
        pat, src, dst = schedule(k, pats, full)
        gold = pattern_data(torch, pat, n, rows)
        ma, mb = mem(ttnn, src, cx, gy, rows), mem(ttnn, dst, cx, gy, rows)
        try:
            x = ttnn.from_torch(
                gold,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ma,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
        except Exception as e:  # noqa: BLE001
            if grow(e):
                continue
            raise
        y, at = None, src
        try:
            if copy == "reshard":
                try:
                    y, at = ttnn.to_memory_config(x, mb), dst
                except Exception as e:  # noqa: BLE001  the source round trip still checks every word, say so
                    if grow(e):
                        raise Retry() from e
                    copy, cerr = "host", f"{type(e).__name__}: {str(e)[:200]}"
                    if say:
                        say(f"ttnn.to_memory_config L1 to L1 failed ({cerr}): host write and readback only")
            ttnn.synchronize_device(mesh)
            way = "odd" if src else "even"
            first = way not in addrs
            addrs[way] = dict(src=baddr(ttnn, x), copy=baddr(ttnn, y) if y is not None else None)
            if first and say:
                say(
                    f"L1 buffers {way} pass: source at {hexa(addrs[way]['src'])}, copy at {hexa(addrs[way]['copy'])}, "
                    f"{rows} KB per core"
                )
            got = read(y if y is not None else x)
            inj = inject(got, chips, k, src, at, cx, gy, rows, y is not None)  # TRAY_INJECT: test-only, before compare
            sgot = None
            if y is not None and any(bool((g != gold).any()) for g in got):
                sgot = read(x)  # name the core: the source's own word, or the copy's
                for i, j in inj:
                    sgot[i] = inject_src(sgot[i], got[i], j)
        except Retry:
            continue
        finally:
            ttnn.deallocate(x)
            if y is not None:
                ttnn.deallocate(y)
        done.add(pat[0])
        diffs = {}
        for i, g in enumerate(got):
            s = st.setdefault(i, new())
            s["passes"] += 1
            d = g != gold
            if bool(d.any()):
                diffs[i] = d
        # the vote: a diff most chips share word for word is the test's own, never a chip's
        sig = {}
        for i, d in diffs.items():
            key = (int(d.sum()), int(torch.nonzero(d.reshape(-1))[0]), int(got[i].reshape(-1)[d.reshape(-1)][0]))
            sig.setdefault(key, []).append(i)
        common = [ii for ii in sig.values() if len(ii) * 2 > len(got)]
        if common:
            test_side.append(dict(pass_=k, pattern=pat[0], chips=len(common[0]), words=int(diffs[common[0][0]].sum())))
            if say:
                say(
                    f"pattern {pat[0]} pass {k}: {len(common[0])} of {len(got)} chips mismatched the same way: test-side, "
                    "not judged"
                )
            for i in common[0]:
                diffs.pop(i)
        for i, d in diffs.items():
            s = st[i]
            s["bad_passes"] += 1
            nb = int(d.sum())
            s["bad"] += nb
            if s["first_pass"] is None:
                s["first_pass"], s["first_t"] = k, round(time.time(), 1)
            sbad = (sgot[i] != gold) if sgot is not None else None
            for sh, v in enumerate(torch.bincount(shard_of[d], minlength=n).tolist()):
                if not v:
                    continue
                own = sbad is not None and bool(sbad[sh * rows : (sh + 1) * rows].any())
                key = core(src if (own or y is None) else at, sh)
                c = s["cores"].setdefault(key, {})
                c[pat[0]] = c.get(pat[0], 0) + v
            if say:
                ch = chips.get(i, {})
                say(f"BAD mesh{i} {ch.get('pci')} tray {ch.get('tray') or '?'}: {nb} words, pattern {pat[0]}, pass {k}")
        k += 1
        last = time.monotonic() - p0
        if beat:
            beat(f"l1 pass {k}")
    el = time.monotonic() - t0
    return dict(
        passes=k,
        patterns=len(pats),
        done=sorted(done, key=[p[0] for p in pats].index),
        copy=copy,
        **({"copy_error": cerr} if cerr else {}),
        kb_per_core=rows * W * 4 >> 10,
        margin_kb=(margins[mi] >> 10) if sized else None,
        addrs=addrs,
        cores=gx * gy,
        l1_free=dict(bytes=free, source=fsrc),
        wall_s=round(el, 1),
        test_side=test_side,
        chips={str(i): dict(chips.get(i, {}), **s) for i, s in st.items()},
    )
