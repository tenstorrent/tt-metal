#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""dram_dbi.py: G3b, run.sh's DRAM byte/DBI pattern check, right after G3.

A factory DDR stress test failed one chip in GDDR ctrl D (bank 3),
channel B, byte 1, READ and READ_DBI, 3/6 loops, with training passing; run.sh had no DRAM byte/DBI pattern stress.
GDDR6 DBI-DC sends a byte inverted when it has more than 4 zero bits, so the patterns that flip DBI from beat to beat
are byte pairs on either side of that line (0x00/0xFF, 0x01/0xFE, 0xE0/0x1F), alternated every 1, 2, 4 and 8 bytes,
plus 0x55/0xAA (most DQ toggling, no inversion), walking ones and zeros, and random bytes.

One pass: a uint32 ROW_MAJOR tensor of ROWS rows of ROW_B bytes (one row = one DRAM page, page r in bank r % banks)
is written to every chip's DRAM (replicated), copied DRAM to DRAM on the device HOPS times (ttnn.clone; each hop reads
every page and writes it to the same bank of a new buffer), then read back per chip and compared bit for bit with the
host's own copy. Rows are grouped by bank count, and each group carries one pattern, so every bank gets every pattern
each pass; each pass rolls the patterns by one more byte so they land on every byte lane in turn. Passes loop until the
wall cap. A chip with any bad word is its NUMERIC FAIL (fleet/chunk_sched.dbi_lines), named by BDF and tray 1-4 with
its DRAM bank (GDDR controller A-H), the byte offset in the 32-byte burst, the XOR of flipped bits and the patterns.

--mode=full adds, after the passes, a rolling 1 GB window per chip (window_check, R6): window w = run count mod 32 (the
count in a per-host file outside the run dir, FLEET_DRAM_WINDOW_FILE, advanced once per full run), so 32 full runs walk
32 GB. ttnn has no placement call, so the window is pushed by allocating w GB of filler (ttnn.empty, never written)
before the 1 GB test buffer and freeing it after; the filler shrinks by halves when the DRAM has no room for it, and by
an eighth of a window at a time when the test buffer then has no room (window 31: 31 GB of filler leave no 1 GB), and
the page pattern is seeded by w either way. The address the allocator gave is logged per bank (start, end: an interleaved
buffer takes the same offsets in every bank), so the DRAM range covered is measured, not assumed.
"""
import json
import os
import sys

# the sibling programs, also when run or imported through a link
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import re
import socket
import time

ROWS, ROW_B = 4096, 2048  # 8 MB per chip; a row is one 2 KB DRAM page
WIN_ROWS, WIN_N = (1 << 30) // ROW_B, 32  # full: a 1 GB window, one of 32 (window = run count mod 32)
WIN_S = 60  # full: the window's share of G3b's wall cap (write, read back and compare 1 GB per chip)
CMP_ROWS = 1 << 15  # full: window rows compared at a time (64 MB), so no compare builds an index of every bad byte
NO_ROOM = re.compile(r"out of memory|not enough space|allocat", re.I)  # an allocator error: shrink the filler and retry
HOPS = 8  # DRAM-to-DRAM copies per pass on the device (reads and writes every page each hop)
PAIRS = [(0x00, 0xFF), (0x01, 0xFE), (0xE0, 0x1F), (0xFE, 0x01), (0x55, 0xAA)]
PERIODS = [1, 2, 4, 8]


def patterns(torch):
    """(names, table): table[p] is pattern p's ROW_B bytes (uint8)."""
    j = torch.arange(ROW_B)
    names, rows = [], []
    for a, b in PAIRS:
        for p in PERIODS:
            names.append(f"{a:02x}/{b:02x}x{p}")
            rows.append(torch.where((j // p) % 2 == 0, torch.tensor(a), torch.tensor(b)))
    names.append("walk1")
    rows.append(torch.bitwise_left_shift(torch.ones_like(j), j % 8))
    names.append("walk0")
    rows.append(255 - torch.bitwise_left_shift(torch.ones_like(j), j % 8))
    names.append("random")
    rows.append(torch.randint(0, 256, (ROW_B,), generator=torch.Generator().manual_seed(5975)))
    return names, torch.stack(rows).to(torch.uint8)


def pass_data(torch, table, nb, k):
    """Pass k's ROWS x ROW_B bytes and each row's pattern index: rows r..r+nb-1 (one per bank) share a pattern, the
    pattern rotates per row group and per pass, and the bytes roll by k so each pass shifts every byte lane."""
    npat = table.shape[0]
    pat = ((torch.arange(ROWS) // nb) + k) % npat
    return torch.roll(table, shifts=k, dims=1)[pat].contiguous(), pat


def inject(got, chips, k):
    """got, or with TRAY_INJECT naming G3b and a chip a copy with one bit flipped in that chip's byte 0 (row 0: DRAM
    bank 0, burst byte 0)."""

    import tray_inject

    got = list(got)
    for i in range(len(got)):
        s = tray_inject.want("G3b", chips.get(i, {}).get("pci"), k)
        if s is not None:
            tray_inject.applied(s, where="bank=0")
            got[i] = tray_inject.flip_tensor(s, got[i], 0)
    return got


def window_file():
    """The per-host window run counter, outside the run dir (like N3's code readback cursor)."""
    d = f"/var/tmp/health-check-dram-window.{os.getuid()}.{socket.gethostname().split('.')[0]}.json"
    return os.environ.get("FLEET_DRAM_WINDOW_FILE") or d


def window_next(path):
    """(this run's window index, note): runs % WIN_N from the counter file, which then counts this run. An unreadable
    file starts at window 0; an unwritable one keeps the window and says so."""
    try:
        with open(path) as f:
            runs = int(json.load(f).get("runs", 0))
    except Exception:  # noqa: BLE001
        runs = 0
    note = ""
    try:
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(dict(runs=runs + 1, t=round(time.time())), f)
        os.replace(tmp, path)
    except Exception as e:  # noqa: BLE001
        note = f"window counter not advanced ({type(e).__name__}: {str(e)[:80]})"
    return runs % WIN_N, note


def baddr(ttnn, t):
    """A device tensor's DRAM buffer address (per-bank offset), None when ttnn cannot say."""
    for f in (lambda: t.buffer_address(), lambda: ttnn.get_device_tensors(t)[0].buffer_address()):
        try:
            return int(f())
        except Exception:  # noqa: BLE001
            pass
    return None


def window_check(ttnn, torch, mesh, chips, w, nb=8, rows=WIN_ROWS, n=0, say=None):
    """Full mode's 1 GB window w: filler of w windows, then `rows` pages seeded by w written to every chip's DRAM, read
    back one chip at a time and compared byte for byte. n: the TRAY_INJECT compare count (G3b's passes before it).
    Returns dict(window, mb, filler_mb, addr, end (per-bank offsets), chips={mesh index: dict(chip, bad (bytes),
    per_bank, first_bank, first_addr, xor)})."""
    D = ttnn.DRAM_MEMORY_CONFIG
    names, table = patterns(torch)
    gold, _ = pass_data(torch, table, nb, w) if rows == ROWS else (None, None)
    if gold is None:
        npat = table.shape[0]
        gold = torch.roll(table, shifts=w, dims=1)[((torch.arange(rows) // nb) + w) % npat].contiguous()
    fill, fr, x = None, w * rows, None
    while x is None:
        while fr and fill is None:  # push the test buffer by w windows; less when the DRAM has no room
            try:
                fill = ttnn.empty([fr, ROW_B // 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, D)
            except Exception:  # noqa: BLE001
                fr //= 2
        try:
            x = ttnn.from_torch(
                gold.view(torch.int32).view(rows, ROW_B // 4),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=D,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
        except Exception as e:  # noqa: BLE001
            if not fr or not NO_ROOM.search(str(e)):
                raise
            fr = max(fr - max(rows // 8, 1), 0)  # the filler left no room for the test buffer: an eighth less, again
        finally:
            if fill is not None:
                ttnn.deallocate(fill)
                fill = None
    a = baddr(ttnn, x)
    end = a + -(-rows // nb) * ROW_B if a is not None else None
    if say:
        say(
            f"window {w} of {WIN_N}: {rows * ROW_B >> 20} MB per chip at DRAM bank offsets "
            f"{'?' if a is None else hex(a)}-{'?' if end is None else hex(end)} in each of {nb} banks, "
            f"filler {fr * ROW_B >> 20} MB"
        )
    row_bank = torch.arange(rows) % nb
    st = {}
    try:
        ttnn.synchronize_device(mesh)
        for i, d in enumerate(ttnn.get_device_tensors(x)):  # one chip at a time: 1 GB of host memory, not 1 GB per chip
            g = ttnn.to_torch(d).contiguous().view(torch.uint8).reshape(rows, ROW_B)
            s = window_inject(chips, i, n, w, a)
            if s is not None:
                g = s(g)
            c = st[str(i)] = dict(chips.get(i, {}), **window_compare(torch, g, gold, row_bank, nb, a))
            if not c["bad"]:
                continue
            if say:
                ch = chips.get(i, {})
                say(f"BAD mesh{i} {ch.get('pci')} tray {ch.get('tray') or '?'}: {c['bad']} bytes in window {w}")
    finally:
        ttnn.deallocate(x)
    return dict(window=w, mb=rows * ROW_B >> 20, filler_mb=fr * ROW_B >> 20, addr=a, end=end, chips=st)


def window_compare(torch, g, gold, row_bank, nb, a):
    """A window readback g against gold, CMP_ROWS rows at a time, so memory stays one slice even when every byte is
    wrong (a whole-window index list of a broken window ran out of memory and read as not checked): dict(bad (bytes),
    per_bank, first_bank, first_addr, xor)."""
    bad, per_bank, first, xor = 0, torch.zeros(nb, dtype=torch.float64), None, 0
    for r0 in range(0, g.shape[0], CMP_ROWS):
        x = torch.bitwise_xor(g[r0 : r0 + CMP_ROWS], gold[r0 : r0 + CMP_ROWS])
        rc = (x != 0).sum(1)
        cnt = int(rc.sum())
        if not cnt:
            continue
        bad += cnt
        per_bank += torch.bincount(row_bank[r0 : r0 + CMP_ROWS], weights=rc.to(torch.float64), minlength=nb)
        if first is None:
            r = int(torch.nonzero(rc)[0])
            first = (r0 + r, int(torch.nonzero(x[r])[0]))
        if xor != 0xFF:
            xor |= sum(1 << b for b in range(8) if bool(torch.bitwise_and(x, 1 << b).any()))
    return dict(
        bad=bad,
        per_bank=[int(v) for v in per_bank.tolist()],
        first_bank=None if first is None else first[0] % nb,
        first_addr=None if first is None else (a or 0) + (first[0] // nb) * ROW_B + first[1],
        xor=xor,
    )


def window_inject(chips, i, n, w, a):
    """TRAY_INJECT naming G3b and chip i: a function flipping one bit of the window readback's byte 0 (bank 0, the
    window's first address), applied on the host copy before the compare; else None."""

    import tray_inject

    s = tray_inject.want("G3b", chips.get(i, {}).get("pci"), n)
    if s is None:
        return None
    tray_inject._SEEN.discard((s["check"], s["bdf"]))  # the window's own applied line, even after the passes' one
    tray_inject.applied(s, where=f"where=window window={w} bank=0 addr={'?' if a is None else hex(a)}")
    return lambda g: tray_inject.flip_tensor(s, g, 0)


def dbi_check(ttnn, torch, mesh, chips, wall_s, nb=8, hops=HOPS, beat=None, say=None, window=None):
    """G3b: write, copy on device, read back and compare DBI patterns on every chip until wall_s s. chips: mesh index ->
    dict(pci, tray, dev). Returns dict(passes, hops, copy, mb, patterns, wall_s, banks, chips={mesh index: dict(chip,
    passes, bad_passes, bad (words), first_pass, first_t, per_bank, per_byte (32-byte burst offset), xor, pats)}).
    copy is 'clone' while ttnn.clone runs, else 'host' (a host write and readback only) with copy_error. window: full
    mode's window index (None: no window); the passes then stop WIN_S s early and the result carries window=dict(...
    window_check's, or error)."""
    D = ttnn.DRAM_MEMORY_CONFIG
    names, table = patterns(torch)
    row_bank = torch.arange(ROWS) % nb
    byte_off = torch.arange(ROW_B) % 32
    new = lambda: dict(
        passes=0,
        bad_passes=0,
        bad=0,
        first_pass=None,
        first_t=None,
        per_bank=[0] * nb,  # noqa: E731
        per_byte=[0] * 32,
        xor=0,
        pats={},
    )
    st = {i: new() for i in chips}
    t0 = time.monotonic()
    k, copy, cerr, last = 0, "clone", None, 0.0
    loop_s = wall_s - WIN_S if window is not None else wall_s
    while time.monotonic() - t0 + last < loop_s or k == 0:
        p0 = time.monotonic()
        gold, pat = pass_data(torch, table, nb, k)
        x = ttnn.from_torch(
            gold.view(torch.int32).view(ROWS, ROW_B // 4),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=D,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        try:
            for _ in range(hops if copy == "clone" else 0):
                try:
                    y = ttnn.clone(x, memory_config=D)
                except Exception as e:  # noqa: BLE001  the host round trip still checks every page, say so
                    copy, cerr = "host", f"{type(e).__name__}: {str(e)[:200]}"
                    if say:
                        say(f"ttnn.clone failed ({cerr}): host write and readback only")
                    break
                ttnn.deallocate(x)
                x = y
            ttnn.synchronize_device(mesh)
            got = [
                ttnn.to_torch(d).contiguous().view(torch.uint8).reshape(ROWS, ROW_B)
                for d in ttnn.get_device_tensors(x.cpu())
            ]
        finally:
            ttnn.deallocate(x)
        got = inject(got, chips, k)  # TRAY_INJECT: test-only flip of one chip's host copy, before the compare
        for i, g in enumerate(got):
            s = st.setdefault(i, new())
            s["passes"] += 1
            d = g != gold
            if not bool(d.any()):
                continue
            r, c = torch.nonzero(d, as_tuple=True)
            s["bad_passes"] += 1
            s["bad"] += int(r.numel())
            if s["first_pass"] is None:
                s["first_pass"], s["first_t"] = k, round(time.time(), 1)
            s["per_bank"] = [u + v for u, v in zip(s["per_bank"], torch.bincount(row_bank[r], minlength=nb).tolist())]
            s["per_byte"] = [u + v for u, v in zip(s["per_byte"], torch.bincount(byte_off[c], minlength=32).tolist())]
            xb = torch.bitwise_xor(g[r, c], gold[r, c])
            for bit in range(8):
                if bool((torch.bitwise_and(xb, 1 << bit) != 0).any()):
                    s["xor"] |= 1 << bit
            for pi, n in zip(*torch.unique(pat[r], return_counts=True)):
                s["pats"][names[int(pi)]] = s["pats"].get(names[int(pi)], 0) + int(n)
            if say:
                ch = chips.get(i, {})
                say(f"BAD mesh{i} {ch.get('pci')} tray {ch.get('tray') or '?'}: {int(r.numel())} bytes in pass {k}")
        k += 1
        last = time.monotonic() - p0
        if beat:
            beat(f"dbi pass {k}")
    win = None
    if window is not None:
        if beat:
            beat("dbi window")
        try:
            win = window_check(ttnn, torch, mesh, chips, window, nb, n=k, say=say)
        except Exception as e:  # noqa: BLE001  a window that could not run is a 'not checked' WARN, never a crash
            win = dict(window=window, error=f"{type(e).__name__}: {str(e)[:200]}")
            if say:
                say(f"window {window} not checked: {win['error']}")
    el = time.monotonic() - t0
    return dict(
        **({"window": win} if win is not None else {}),
        passes=k,
        hops=(hops if copy == "clone" else 0),
        copy=copy,
        **({"copy_error": cerr} if cerr else {}),
        mb=round(ROWS * ROW_B / 2**20, 1),
        patterns=len(names),
        wall_s=round(el, 1),
        banks=nb,
        chips={str(i): dict(chips.get(i, {}), **s) for i, s in st.items()},
    )
