#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""PMP policy on the chip (fresh chip reset required; run through scripts/l2cpu_run.sh). The default image starts
with the policy (README "PMP policy"); then:

  - every hart reports APPLIED, pmpcfg0 / pmpaddr0..7 of hart 0 read back equal to the table;
  - mailbox PEEK / POKE outside the policy -> DENIED without an access (unmapped TLB window, the DMA controller,
    the RNMI handler registers, the resident code); allowed reads still work; locked CSRs ignore writes;
  - unguarded wild accesses (inject): hart 1 load through an unprogrammed TLB window (mcause 5), hart 2 store into the
    resident page (7), hart 3 jump into the uncached alias (1), hart 0 store into the RNMI handler registers (7):
    each traps and parks in the resident error park, nothing is written, the other harts keep their heartbeat;
  - restart re-uses the locked set (REUSED); RNMI COUNT and PARK still work under the locked entries;
  - the restart rule: a restart that asks for another policy (region size, policy off) is refused by the host;
  - N restarts alternating L1 / L2, PING + echo after each.

    tools/l2cpu/scripts/l2cpu_run.sh "pmp faults" $PY tools/l2cpu/scripts/pmp_faults.py --restarts 50
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "host"))
sys.path.insert(0, HERE)

from bringup_smoke import DEFAULT_IMAGE, REGION_BYTES, Smoke, finish  # noqa: E402
from l2cpu import L2cpuCtl, L2cpuCtlError, L2cpuHw, TtnnClusterBackend, layout as L, pmp as P  # noqa: E402
from l2cpu.bringup import region_base_pa  # noqa: E402

WILD_WINDOW = P.TLB2M_UNCACHED + 100 * P.TLB2M_SIZE  # small TLB window 100: never programmed by the firmware


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--restarts", type=int, default=50)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    import ttnn

    dev = ttnn.open_device(device_id=0)
    try:
        buf = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 8, REGION_BYTES // 4]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
        )
        base = region_base_pa(buf.buffer_address())
        ctl = L2cpuCtl(L2cpuHw(TtnnClusterBackend(0), guard=True, log=None), base, log=None)
        sm = Smoke(ctl)
        res = base + L.L2CPU_OFF_RESIDENT
        image = open(a.image, "rb").read()

        def bring_up():
            ctl.start(image, pmp=True)
            sm.req = ctl.r32(base + 0x1000)
            print(ctl.policy.describe(), flush=True)
            got = [ctl.mb(L.L2CPU_MB_CSR_READ, 0x3B0 + i)[1][0] for i in range(8)]
            cfg = ctl.mb(L.L2CPU_MB_CSR_READ, 0x3A0)[1][0]
            want = [e.addr for e in ctl.policy.entries]
            wcfg = sum(e.cfg << (8 * i) for i, e in enumerate(ctl.policy.entries))
            if got != want or cfg != wcfg:
                raise RuntimeError(
                    f"pmpaddr {[hex(v) for v in got]} cfg 0x{cfg:x}, table {[hex(v) for v in want]} 0x{wcfg:x}"
                )
            return f"region 0x{base:x}, harts {[r['pmp_name'] for r in ctl.records()]}, pmpcfg0 0x{cfg:016x} = table"

        def mailbox():
            out = []
            for name, cmd, args, want in (
                ("peek TLB window 100", L.L2CPU_MB_PEEK32, (WILD_WINDOW,), L.L2CPU_MB_ERR_DENIED),
                ("peek DMA controller", L.L2CPU_MB_PEEK32, (0x2008_0000,), L.L2CPU_MB_ERR_DENIED),
                ("poke RNMI handler", L.L2CPU_MB_POKE32, (L.L2CPU_RNMI_HANDLER, 0), L.L2CPU_MB_ERR_DENIED),
                ("poke resident code", L.L2CPU_MB_POKE32, (res + 0x10, 0), L.L2CPU_MB_ERR_DENIED),
                ("fill resident code", L.L2CPU_MB_FILL32, (res, 0, 64), L.L2CPU_MB_ERR_DENIED),
                ("peek hart status", L.L2CPU_MB_PEEK32, (L.L2CPU_HART_STATUS,), L.L2CPU_MB_OK),
                ("peek resident code", L.L2CPU_MB_PEEK32, (res,), L.L2CPU_MB_OK),
                ("poke region scratch", L.L2CPU_MB_POKE32, (base + L.L2CPU_OFF_SCRATCH, 7), L.L2CPU_MB_OK),
                (
                    "peek uncached alias",
                    L.L2CPU_MB_PEEK32,
                    (base - P.UNCACHED_DELTA + L.L2CPU_OFF_SCRATCH,),
                    L.L2CPU_MB_OK,
                ),
                (
                    "poke uncached alias",
                    L.L2CPU_MB_POKE32,
                    (base - P.UNCACHED_DELTA + L.L2CPU_OFF_SCRATCH, 0),
                    L.L2CPU_MB_ERR_DENIED,
                ),
            ):
                st, rep = ctl.mb(cmd, *args)
                if st != want:
                    raise RuntimeError(f"{name}: status {st} (want {want}), reply {rep[:2]}")
                out.append(f"{name}={'DENIED' if st == L.L2CPU_MB_ERR_DENIED else 'ok'}")
            cfg = ctl.mb(L.L2CPU_MB_CSR_READ, 0x3A0)[1][0]
            st, rep = ctl.mb(L.L2CPU_MB_CSR_WRITE, 0x3A0, 0)
            st2, rep2 = ctl.mb(L.L2CPU_MB_CSR_WRITE, 0x3B4, 0)
            if st or rep[0] != cfg or st2 or rep2[0] != ctl.policy.entries[4].addr:
                raise RuntimeError(f"locked CSR rewrite: {st} 0x{rep[0]:x} {st2} 0x{rep2[0]:x}")
            return ", ".join(out) + "; pmpcfg0 / pmpaddr4 rewrites ignored"

        def alive_except(dead):
            x = ctl.heartbeats()
            time.sleep(0.05)
            y = ctl.heartbeats()
            return [y[h] > x[h] for h in range(4)] == [h not in dead for h in range(4)]

        def wait_trap(h, mcause, mtval):
            t0 = time.time()
            while ctl.record(h)["state"] != L.L2CPU_STATE_PARKED or ctl.hart_state(h)["status"] != L.L2CPU_HART_PARKED:
                if time.time() - t0 > 1:
                    raise RuntimeError(f"hart {h} not parked: {ctl.record(h)} {ctl.hart_state(h)}")
            hs, rec = ctl.hart_state(h), ctl.record(h)
            if (hs["error"], hs["mcause"], hs["mtval"], rec["kind"]) != (
                L.L2CPU_ERR_TRAP,
                mcause,
                mtval,
                L.L2CPU_KIND_ERROR,
            ):
                raise RuntimeError(f"hart {h}: {hs} {rec}")
            return f"hart {h} mcause {mcause} mtval 0x{mtval:x} mepc 0x{hs['mepc']:x}"

        def inject(h, kind, addr):
            st, rep = ctl.mb(L.L2CPU_MB_INJECT, h, kind, addr)
            if st != L.L2CPU_MB_OK:
                raise RuntimeError(f"inject hart {h} kind {kind}: status {st}")

        def faults():
            out = []
            handler0 = ctl.handler_addrs()[0]
            code = ctl.hw.pa_read(res, 0x100)
            t0 = time.perf_counter()
            inject(1, L.L2CPU_INJECT_LOAD, WILD_WINDOW)
            out.append(wait_trap(1, 5, WILD_WINDOW))
            out.append(f"{(time.perf_counter() - t0) * 1e3:.1f} ms")
            inject(2, L.L2CPU_INJECT_STORE, res + 0x10)
            out.append(wait_trap(2, 7, res + 0x10))
            uc = base - P.UNCACHED_DELTA + L.L2CPU_OFF_SCRATCH
            inject(3, L.L2CPU_INJECT_JUMP, uc)
            out.append(wait_trap(3, 1, uc))
            if not alive_except({1, 2, 3}):
                raise RuntimeError("hart 0 heartbeat")
            inject(0, L.L2CPU_INJECT_STORE, L.L2CPU_RNMI_HANDLER)
            out.append(wait_trap(0, 7, L.L2CPU_RNMI_HANDLER))
            if ctl.handler_addrs()[0] != handler0 or ctl.hw.pa_read(res, 0x100) != code:
                raise RuntimeError("a store landed")
            err = ctl.error()
            return "; ".join(out) + f"; first error {err}; RNMI handler and resident code unchanged"

        def revive():
            r = ctl.restart(None, warm=True)
            sm.echo(5000)
            return f"WARM restart {r['t_total'] * 1e3:.2f} ms, levels {r['levels']}, harts {[x['pmp_name'] for x in ctl.records()]}"

        def rnmi():
            n = ctl.rnmi(0xF, L.L2CPU_RNMI_COUNT, timeout=0.2)
            if n != [1, 1, 1, 1] or not alive_except(set()):
                raise RuntimeError(f"COUNT {n}")
            ctl.inject(2, L.L2CPU_INJECT_SPIN)
            time.sleep(0.01)
            r = ctl.restart(None, warm=True)
            sm.echo(6000)
            if not any(lv.startswith("L2") for lv in r["levels"]) or r["stop_records"][2]["kind"] != L.L2CPU_KIND_RNMI:
                raise RuntimeError(f"levels {r['levels']} rec {r['stop_records'][2]}")
            return f"RNMI COUNT {n} (all resume); spinning hart 2 -> levels {r['levels']} kind RNMI, restart REUSED"

        def rule():
            out = []
            for name, pol in (
                ("region size x2", P.build(base, 2 * L.L2CPU_REGION_MIN_SIZE)),
                ("policy off", False),
                ("other windows", P.build(base, L.L2CPU_REGION_MIN_SIZE, windows=((32, 32, False),))),
            ):
                try:
                    ctl.restart(None, warm=True, pmp=pol)
                    raise RuntimeError(f"{name}: restart accepted")
                except L2cpuCtlError as e:
                    out.append(f"{name}: refused")
                    msg = str(e)
            if not ctl.is_alive():
                raise RuntimeError("firmware not alive after the refusals")
            ctl.restart(None, warm=True, pmp=True)
            sm.echo(7000)
            return "; ".join(out) + f" ('{msg[:60]}...'); same policy accepted"

        times = {"L1": [], "L2": []}

        def restarts():
            for i in range(a.restarts):
                lvl = "L2" if i % 2 else "L1"
                if lvl == "L2":
                    ctl.rnmi(0xF, L.L2CPU_RNMI_PARK, timeout=0.5)
                out = ctl.restart(None, warm=True)
                times[lvl].append(out["t_total"])
                st, _ = ctl.mb(L.L2CPU_MB_PING)
                if st != 0:
                    raise RuntimeError(f"PING after restart {i}: {st}")
                sm.echo(10000 + i)
            med = lambda v: sorted(v)[len(v) // 2] * 1e3 if v else 0  # noqa: E731
            return f"{a.restarts} restarts (REUSED each), median L1 {med(times['L1']):.2f} ms, L2 {med(times['L2']):.2f} ms"

        if not sm.step("bring-up with the policy", bring_up):
            return finish(sm, a)
        for name, fn in (
            ("mailbox outside the policy", mailbox),
            ("wild accesses trap and park", faults),
            ("restart re-uses the locked set", revive),
            ("RNMI under the locked entries", rnmi),
            ("restart rule (host refuses)", rule),
            (f"restarts x{a.restarts} (L1/L2)", restarts),
        ):
            if not sm.step(name, fn):
                break
        sm.step("final error word / alive", lambda: f"error {ctl.error()}, alive {ctl.is_alive()}")
        return finish(sm, a)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    sys.exit(main())
