#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""dram_dbi.py: G3b, run.sh's DRAM byte/DBI pattern check, right after G3.

R03 (notes/run-sh-coverage.md): BHFT ft_ddr_stress failed one chip in GDDR ctrl D (bank 3),
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
"""
import time

ROWS, ROW_B = 4096, 2048        # 8 MB per chip; a row is one 2 KB DRAM page
HOPS = 8                        # DRAM-to-DRAM copies per pass on the device (reads and writes every page each hop)
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


def dbi_check(ttnn, torch, mesh, chips, wall_s, nb=8, hops=HOPS, beat=None, say=None):
    """G3b: write, copy on device, read back and compare DBI patterns on every chip until wall_s s. chips: mesh index ->
    dict(pci, tray, dev). Returns dict(passes, hops, copy, mb, patterns, wall_s, banks, chips={mesh index: dict(chip,
    passes, bad_passes, bad (words), first_pass, first_t, per_bank, per_byte (32-byte burst offset), xor, pats)}).
    copy is 'clone' while ttnn.clone runs, else 'host' (a host write and readback only) with copy_error."""
    D = ttnn.DRAM_MEMORY_CONFIG
    names, table = patterns(torch)
    row_bank = torch.arange(ROWS) % nb
    byte_off = torch.arange(ROW_B) % 32
    new = lambda: dict(passes=0, bad_passes=0, bad=0, first_pass=None, first_t=None, per_bank=[0] * nb,  # noqa: E731
                       per_byte=[0] * 32, xor=0, pats={})
    st = {i: new() for i in chips}
    t0 = time.monotonic()
    k, copy, cerr, last = 0, "clone", None, 0.0
    while time.monotonic() - t0 + last < wall_s or k == 0:
        p0 = time.monotonic()
        gold, pat = pass_data(torch, table, nb, k)
        x = ttnn.from_torch(gold.view(torch.int32).view(ROWS, ROW_B // 4), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=mesh, memory_config=D, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
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
            got = [ttnn.to_torch(d).contiguous().view(torch.uint8).reshape(ROWS, ROW_B)
                   for d in ttnn.get_device_tensors(x.cpu())]
        finally:
            ttnn.deallocate(x)
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
    el = time.monotonic() - t0
    return dict(passes=k, hops=(hops if copy == "clone" else 0), copy=copy, **({"copy_error": cerr} if cerr else {}),
                mb=round(ROWS * ROW_B / 2**20, 1), patterns=len(names), wall_s=round(el, 1), banks=nb,
                chips={str(i): dict(chips.get(i, {}), **s) for i, s in st.items()})
