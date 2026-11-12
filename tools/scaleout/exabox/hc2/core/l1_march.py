#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""l1_march.py: run.sh's L1 march check, right after G3/G3b in the bus-bar mix.

notes/goal54-stage2-coverage.md: run.sh had no SRAM/L1 check. One pass writes one pattern to an L1 tensor
HEIGHT_SHARDED over the compute grid (one shard per Tensix core, sized so the source and its copy fill the allocatable
L1), copies it on the device L1 to L1 onto the grid shifted by one column (shard i moves to the next core), reads the
copy back per chip and compares it word for word with the written pattern. Only on a mismatch is the source read back
too, so the bad core is named exactly: a bad source word is the source core's, a good source with a bad copy is the
copy's core. Passes alternate the direction (columns 0..X-2 -> 1..X-1, then back), so every core is written directly
and read directly. Patterns (16-bit words, both halves of each uint32): 0x5555 and 0xAAAA first (every cell at 0 and 1
in two passes), then address-in-data (shard index and word index), 0x0000 and 0xFFFF; full mode adds the inverses.
Passes loop until the wall cap (always at least one).

Vote across chips: a mismatch that more than half of the chips show identically is the test's own (layout, op), so it
is a WARN that names nothing; a chip that differs alone is its L1 FAIL (fleet/chunk_sched.l1_lines), one line per core
and pattern: 'L1 FAIL <BDF> tray N core x,y pattern P words k'. It covers only the allocatable L1: firmware and kernel
config regions are N3's.
"""
import time

W = 256                 # uint32 words per row: one row is 1 KB of L1
MARGIN = 128 << 10      # L1 bytes per core left free for the copy op's own buffers
FALLBACK_KB = 256       # per-core shard size when the free L1 cannot be read
BASE = [("5555", 0x55555555), ("aaaa", 0xAAAAAAAA), ("addr", None), ("0000", 0x00000000), ("ffff", 0xFFFFFFFF)]


def patterns(full=False):
    """[(name, word or None for address-in-data, invert)]: the base five, plus their inverses in full mode."""
    out = [(n, v, False) for n, v in BASE]
    if full:
        out += [("~" + n, v, True) for n, v in BASE]
    return out


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
    g = torch.where(g >= 2**31, g - 2**32, g)          # uint32 bits in int32
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


def grid(mesh):
    g = mesh.compute_with_storage_grid_size()
    return int(g.x), int(g.y)


def mem(ttnn, x0, gx, gy, rows):
    """HEIGHT_SHARDED L1 config over columns x0..x0+gx-1, rows 0..gy-1: shard i on core (x0 + i % gx, i // gx)."""
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, 0), ttnn.CoreCoord(x0 + gx - 1, gy - 1))})
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                             ttnn.ShardSpec(crs, [rows, W], ttnn.ShardOrientation.ROW_MAJOR))


def l1_check(ttnn, torch, mesh, chips, wall_s, full=False, kb=0, beat=None, say=None):
    """N1: write, copy L1 to the next core, read back and compare until wall_s s. chips: mesh index -> dict(pci, tray,
    dev). kb: per-core shard KB (0 = sized to the allocatable L1). Returns dict(passes, patterns, done, copy, kb_per_core,
    cores, l1_free, wall_s, test_side, chips={mesh index: dict(chip, passes, bad_passes, bad (words), first_pass,
    first_t, cores={"x,y": {pattern: words}})}). copy is 'reshard' while ttnn.to_memory_config runs, else 'host' (a
    host write and readback of the source only) with copy_error."""
    gx, gy = grid(mesh)
    if gx < 2 or gy < 1:
        raise RuntimeError(f"compute grid {gx}x{gy}: no column to copy to")
    cx = gx - 1                                        # each side of the copy is cx columns wide
    n = cx * gy
    free, fsrc = l1_free(ttnn, mesh)
    if kb:
        rows = int(kb)
    elif free:
        rows = max(1, (free - MARGIN) // 2 // (W * 4))
    else:
        rows = FALLBACK_KB
    pats = patterns(full)
    new = lambda: dict(passes=0, bad_passes=0, bad=0, first_pass=None, first_t=None, cores={})  # noqa: E731
    st = {i: new() for i in chips}
    shard_of = torch.arange(n * rows).view(n * rows, 1).expand(n * rows, W) // rows
    t0 = time.monotonic()
    k, copy, cerr, last, test_side, done = 0, "reshard", None, 0.0, [], set()

    def core(x0, s):
        return f"{x0 + s % cx},{s // cx}"

    def read(t):
        return [ttnn.to_torch(d).contiguous().view(torch.int32).reshape(n * rows, W)
                for d in ttnn.get_device_tensors(t.cpu())]

    while time.monotonic() - t0 + last < wall_s or k == 0:
        p0 = time.monotonic()
        pat = pats[k % len(pats)]
        src, dst = (0, 1) if k % 2 == 0 else (1, 0)    # columns 0..X-2 -> 1..X-1, then back
        gold = pattern_data(torch, pat, n, rows)
        ma, mb = mem(ttnn, src, cx, gy, rows), mem(ttnn, dst, cx, gy, rows)
        x = ttnn.from_torch(gold, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ma,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        y, at = None, src
        try:
            if copy == "reshard":
                try:
                    y, at = ttnn.to_memory_config(x, mb), dst
                except Exception as e:  # noqa: BLE001  the source round trip still checks every word, say so
                    copy, cerr = "host", f"{type(e).__name__}: {str(e)[:200]}"
                    if say:
                        say(f"ttnn.to_memory_config L1 to L1 failed ({cerr}): host write and readback only")
            ttnn.synchronize_device(mesh)
            got = read(y if y is not None else x)
            sgot = None
            if y is not None and any(bool((g != gold).any()) for g in got):
                sgot = read(x)                         # name the core: the source's own word, or the copy's
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
                say(f"pattern {pat[0]} pass {k}: {len(common[0])} of {len(got)} chips mismatched the same way: test-side, "
                    "not judged")
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
                own = sbad is not None and bool(sbad[sh * rows:(sh + 1) * rows].any())
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
    return dict(passes=k, patterns=len(pats), done=sorted(done, key=[p[0] for p in pats].index), copy=copy,
                **({"copy_error": cerr} if cerr else {}), kb_per_core=rows * W * 4 >> 10, cores=gx * gy,
                l1_free=dict(bytes=free, source=fsrc), wall_s=round(el, 1), test_side=test_side,
                chips={str(i): dict(chips.get(i, {}), **s) for i, s in st.items()})
