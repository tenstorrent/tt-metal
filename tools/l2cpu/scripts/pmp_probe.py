#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""PMP hardware probe of the x280 (hart 0 of one tile), through the firmware's guarded mailbox CSR read / write.
Fresh chip reset required (run through scripts/l2cpu_run.sh). The firmware starts WITHOUT the PMP policy; the probe
then finds:

  - entry count: write all-ones to pmpaddr0..63, read back (unimplemented entries read 0 or trap);
  - writable pmpcfg bits: write 0x1f (R W X, A = NAPOT, no L) to every byte of pmpcfg0..15 (odd numbers trap on RV64);
  - granularity G: pmpaddr0 = all-ones with entry 0 OFF, G = trailing zeros of the read-back (NAPOT minimum 2^(G+3));
    implemented address bits;
  - reset values: the first read of pmpcfg0/2 and pmpaddr0..7 (what a chip reset leaves);
  - Smepmp: is mseccfg (0x747) readable, and its RLB / MMWP / MML bits;
  - lock behaviour: a LOCKED NAPOT entry without permissions over the region scratch page must make an M-mode
    load fault (mcause 5) and store fault (mcause 7), ignore later writes to its pmpcfg byte and pmpaddr, and stay
    after a WARM restart; a LOCKED TOR entry must also lock the pmpaddr below it.

Every access stays inside the firmware's own region or the hart's CSRs: no address outside the region is touched.

    tools/l2cpu/scripts/l2cpu_run.sh "pmp probe" $PY tools/l2cpu/scripts/pmp_probe.py --json probe.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
L2CPU_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(L2CPU_DIR, "host"))

from l2cpu import L2cpuCtl, L2cpuHw, TtnnClusterBackend, layout as L  # noqa: E402
from l2cpu.bringup import region_base_pa  # noqa: E402
from l2cpu.hw import REGION_ALIGN  # noqa: E402

DEFAULT_IMAGE = os.path.join(L2CPU_DIR, "fw", "build", "bh-irq", "fw.bin")
ONES = (1 << 64) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    import ttnn

    out, ok = {}, True
    dev = ttnn.open_device(device_id=0)
    try:
        nbytes = L.L2CPU_REGION_MIN_SIZE + REGION_ALIGN
        buf = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 8, nbytes // 4]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
        )
        base = region_base_pa(buf.buffer_address())
        ctl = L2cpuCtl(L2cpuHw(TtnnClusterBackend(0), guard=True, log=None), base, log=None)
        ctl.start(open(a.image, "rb").read(), pmp=False)
        print(f"READY region 0x{base:x}", flush=True)

        def rd(csr):
            st, rep = ctl.mb(L.L2CPU_MB_CSR_READ, csr)
            return (rep[0], 0) if st == L.L2CPU_MB_OK else (None, rep[1] if st == L.L2CPU_MB_ERR_FAULT else -st)

        def wr(csr, v):
            st, rep = ctl.mb(L.L2CPU_MB_CSR_WRITE, csr, v & ONES)
            return (rep[0], 0) if st == L.L2CPU_MB_OK else (None, rep[1] if st == L.L2CPU_MB_ERR_FAULT else -st)

        def peek(pa):
            st, rep = ctl.mb(L.L2CPU_MB_PEEK32, pa)
            return st, rep[1]

        def poke(pa, v):
            st, rep = ctl.mb(L.L2CPU_MB_POKE32, pa, v)
            return st, rep[1]

        ids = {n: rd(c)[0] for n, c in (("misa", 0x301), ("marchid", 0xF12), ("mimpid", 0xF13), ("mvendorid", 0xF11))}
        out["ids"] = {k: hex(v) if v is not None else None for k, v in ids.items()}
        out["reset_pmpcfg"] = {
            hex(c): (hex(v) if v is not None else f"fault {e}") for c in (0x3A0, 0x3A2) for v, e in [rd(c)]
        }
        out["reset_pmpaddr"] = [hex(rd(0x3B0 + i)[0] or 0) for i in range(16)]
        v, e = rd(0x747)
        out["mseccfg"] = dict(readable=v is not None, value=hex(v) if v is not None else None, mcause=e)
        if v is not None:
            out["mseccfg"].update(MML=v & 1, MMWP=(v >> 1) & 1, RLB=(v >> 2) & 1)
        print("ids", out["ids"], "mseccfg", out["mseccfg"], flush=True)

        # entry count: pmpaddrN all-ones (entry OFF: every cfg is 0 here), read back, restore 0
        addr_rb = []
        for i in range(64):
            v, e = wr(0x3B0 + i, ONES)
            addr_rb.append((i, hex(v) if v is not None else None, e))
            wr(0x3B0 + i, 0)
        impl = [i for i, v, e in addr_rb if v not in (None, "0x0")]
        out["pmpaddr_allones"] = addr_rb
        out["entries"] = len(impl)
        out["entries_contiguous"] = impl == list(range(len(impl)))
        # pmpcfg writable bits (no L bit)
        cfg_rb = {}
        for c in range(0x3A0, 0x3B0):
            v, e = wr(c, 0x1F1F1F1F1F1F1F1F)
            cfg_rb[hex(c)] = (hex(v) if v is not None else None, e)
            if v is not None:
                wr(c, 0)
        out["pmpcfg_write_1f"] = cfg_rb
        # granularity: entry 0 OFF, pmpaddr0 = all-ones
        v, _ = wr(0x3B0, ONES)
        g = (v & -v).bit_length() - 1 if v else None
        out["pmpaddr0_allones_off"] = hex(v)
        out["G"] = g
        out["granule_bytes"] = 1 << (g + 2) if g is not None else None
        out["addr_bits"] = v.bit_length() + 2 if v else None  # physical address bits covered by pmpaddr
        wr(0x3A0, 0x18)  # entry 0 NAPOT, no permissions, unlocked: no effect on M-mode
        v2, _ = wr(0x3B0, 0)
        out["pmpaddr0_zero_napot"] = hex(v2)  # G >= 2: bits G-2..0 read as ones
        wr(0x3B0, ONES)  # NAPOT read of all-ones: the deny-all value (bit G-1 is stored, reads 0 in OFF / TOR)
        out["pmpaddr0_allones_napot"] = hex(rd(0x3B0)[0])
        wr(0x3A0, 0)
        wr(0x3B0, 0)
        print(
            f"entries {out['entries']} G {g} granule {out['granule_bytes']} B, pmpaddr bits {out['addr_bits']}",
            flush=True,
        )

        # lock behaviour, entry 0: NAPOT over the region scratch page (8 KiB), L, no R/W/X
        scr = base + L.L2CPU_OFF_SCRATCH
        lk = {}
        lk["peek_before"] = peek(scr)
        lk["poke_before"] = poke(scr, 0x1234)
        na = (scr >> 2) | ((L.L2CPU_SCRATCH_SIZE >> 3) - 1)
        lk["pmpaddr0_set"] = hex(wr(0x3B0, na)[0])
        lk["pmpaddr0_napot_value"] = hex(na)
        lk["pmpcfg0_lock"] = hex(wr(0x3A0, 0x98)[0])
        lk["peek_locked"] = peek(scr)
        lk["peek_locked_last_word"] = peek(scr + L.L2CPU_SCRATCH_SIZE - 4)
        lk["poke_locked"] = poke(scr, 0x5678)
        lk["peek_outside"] = peek(scr + L.L2CPU_SCRATCH_SIZE)
        lk["pmpcfg0_rewrite_0"] = hex(wr(0x3A0, 0)[0])
        lk["pmpaddr0_rewrite_0"] = hex(wr(0x3B0, 0)[0])
        lk["peek_after_rewrite"] = peek(scr)
        # entry 2 TOR [app_low, app_low + 4 KiB), L, no permissions: must also lock pmpaddr1
        lo, hi = base + L.L2CPU_OFF_APP_LOW, base + L.L2CPU_OFF_APP_LOW + 0x1000
        wr(0x3B1, lo >> 2)
        wr(0x3B2, hi >> 2)
        lk["pmpcfg0_tor"] = hex(wr(0x3A0, 0x98 | (0x88 << 16))[0])
        lk["tor_peek_lo"] = peek(lo)
        lk["tor_peek_hi_minus_4"] = peek(hi - 4)
        lk["tor_peek_hi"] = peek(hi)
        lk["pmpaddr1_rewrite_0"] = hex(wr(0x3B1, 0)[0])
        lk["pmpaddr1_expect"] = hex(lo >> 2)
        # the unlocked entry 1 (cfg byte 0, OFF) is still writable
        lk["pmpcfg0_byte1_write"] = hex(wr(0x3A0, 0x98 | (0x18 << 8) | (0x88 << 16))[0])
        wr(0x3A0, 0x98 | (0x88 << 16))
        # survives a WARM restart (software restart, same chip epoch)
        r = ctl.restart(None, warm=True)
        lk["restart_ms"] = round(r["t_total"] * 1e3, 2)
        lk["pmpcfg0_after_restart"] = hex(rd(0x3A0)[0])
        lk["peek_locked_after_restart"] = peek(scr)
        lk["tor_peek_after_restart"] = peek(lo)
        out["lock"] = lk
        for k, v in lk.items():
            print(f"  {k:28s} {v}", flush=True)

        exp = [
            ("peek_before", (0, 0)),
            ("peek_locked", (L.L2CPU_MB_ERR_FAULT, 5)),
            ("peek_locked_last_word", (L.L2CPU_MB_ERR_FAULT, 5)),
            ("poke_locked", (L.L2CPU_MB_ERR_FAULT, 7)),
            ("peek_outside", (0, 0)),
            ("peek_after_rewrite", (L.L2CPU_MB_ERR_FAULT, 5)),
            ("tor_peek_lo", (L.L2CPU_MB_ERR_FAULT, 5)),
            ("tor_peek_hi_minus_4", (L.L2CPU_MB_ERR_FAULT, 5)),
            ("tor_peek_hi", (0, 0)),
            ("peek_locked_after_restart", (L.L2CPU_MB_ERR_FAULT, 5)),
            ("tor_peek_after_restart", (L.L2CPU_MB_ERR_FAULT, 5)),
        ]
        checks = {k: tuple(lk[k]) == want for k, want in exp}
        checks["cfg_lock_ignores_writes"] = int(lk["pmpcfg0_rewrite_0"], 16) & 0xFF == 0x98
        checks["addr_lock_ignores_writes"] = lk["pmpaddr0_rewrite_0"] == lk["pmpaddr0_napot_value"]
        checks["tor_locks_addr_below"] = lk["pmpaddr1_rewrite_0"] == lk["pmpaddr1_expect"]
        out["checks"] = checks
        ok = all(checks.values())
        st, _ = ctl.mb(L.L2CPU_MB_PING)
        out["final_ping"] = st
        ok &= st == 0
        print("CHECKS", checks, flush=True)
    finally:
        ttnn.close_device(dev)
    print("PROBE", "PASS" if ok else "FAIL", json.dumps(out), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
