#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""L2CPU smoke test on the chip (fresh chip reset required; run through scripts/l2cpu_run.sh):

  bring-up to READY -> heartbeats of all 4 harts -> mailbox PING -> echo work items (doorbell + 4 harts)
  -> trap test (hart 3 illegal instruction, resident error park, other harts alive) -> WARM restart revives it
  -> N restarts alternating L1 (mailbox PARK) and L2 (RNMI park), PING + echo after each -> report.

The region is the bank-5 slice of an interleaved ttnn DRAM buffer allocated in this process (bank 5 = the DRAM
local to the L2CPU tile of CPUs 0-3), so tt-metal never hands it to anything else while the process lives.
Prints one line per step with its wall clock, a JSON summary, and exits 0 only if every step passed.

    source tools/l2cpu/scripts/l2cpu_env.sh
    tools/l2cpu/scripts/l2cpu_run.sh "smoke" $PY tools/l2cpu/scripts/bringup_smoke.py --restarts 200
"""
import argparse
import json
import os
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
L2CPU_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(L2CPU_DIR, "host"))

from l2cpu import L2cpuCtl, L2cpuHw, TtnnClusterBackend, layout as L  # noqa: E402
from l2cpu.bringup import region_base_pa  # noqa: E402
from l2cpu.hw import REGION_ALIGN  # noqa: E402

DEFAULT_IMAGE = os.path.join(L2CPU_DIR, "fw", "build", "bh-irq", "fw.bin")
ECHO_REQ, ECHO_DONE, ECHO_VALUE, ECHO_RESULT = (L.L2CPU_OFF_APP_CTRL + o for o in (0x000, 0x040, 0x080, 0x100))
REGION_BYTES = L.L2CPU_REGION_MIN_SIZE + REGION_ALIGN  # per bank: the base is rounded up inside the buffer


class Smoke:
    def __init__(self, ctl):
        self.ctl, self.steps, self.ok = ctl, [], True
        self.req = ctl.r32(ctl.base + ECHO_REQ)

    def step(self, name, fn):
        t0 = time.perf_counter()
        try:
            detail = fn()
            ok = True
        except Exception as e:  # noqa: BLE001
            detail, ok = f"{type(e).__name__}: {e}", False
        dt = time.perf_counter() - t0
        self.steps.append(dict(step=name, ok=ok, wall_s=round(dt, 4), detail=detail))
        self.ok &= ok
        print(f"{'PASS' if ok else 'FAIL'} {name:34s} {dt * 1e3:10.2f} ms  {detail}", flush=True)
        if not ok:
            print(self.ctl.log_text()[0][-4000:], flush=True)
        return ok

    def echo(self, value, timeout=1.0):
        c, b = self.ctl, self.ctl.base
        c.w32(b + ECHO_VALUE, value)
        self.req += 1
        c.w32(b + ECHO_REQ, self.req)
        c.hw.doorbell(self.req | 0x80000000)  # never 0
        t0 = time.time()
        while c.r32(b + ECHO_DONE) != self.req:
            if time.time() - t0 > timeout:
                raise RuntimeError(f"echo {self.req} not done; error {c.error()}")
        d = c.hw.pa_read(b + ECHO_RESULT, 256)
        got = [struct.unpack_from("<I", d, 64 * h)[0] for h in range(4)]
        if got != [value + h for h in range(4)]:
            raise RuntimeError(f"echo results {got}")
        return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--restarts", type=int, default=200)
    ap.add_argument("--json", default=None, help="write the summary here too")
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
        image = open(a.image, "rb").read()

        def bring_up():
            info = ctl.start(image)
            sm.req = ctl.r32(base + ECHO_REQ)
            p = info["probe"]
            return f"region 0x{base:x}, RNMI handlers reset value {p['handlers_reset_value'][0]}, written ok {p['handlers_ok']}"

        if not sm.step("bring-up to READY", bring_up):
            return finish(sm, a)

        def heartbeats():
            x = ctl.heartbeats()
            time.sleep(0.1)
            y = ctl.heartbeats()
            d = [q - p for p, q in zip(x, y)]
            if not all(v > 50 for v in d):
                raise RuntimeError(f"heartbeat deltas {d}")
            return f"deltas over 0.1 s {d}, clock {ctl.ensure_clock():.0f} MHz"

        def ping():
            st, rep = ctl.mb(L.L2CPU_MB_PING)
            if st != 0:
                raise RuntimeError(f"status {st}")
            return f"fw 0x{rep[0]:x} layout {rep[1]} app {rep[4]}"

        def echoes():
            for i in range(100):
                sm.echo(1000 + i)
            return "100 work items over 4 harts"

        def trap():
            ctl.inject(3)
            t0 = time.time()
            while ctl.record(3)["state"] != L.L2CPU_STATE_PARKED:
                if time.time() - t0 > 1:
                    raise RuntimeError("hart 3 not parked")
            hs, rec = ctl.hart_state(3), ctl.record(3)
            x = ctl.heartbeats()
            time.sleep(0.05)
            y = ctl.heartbeats()
            alive = [y[h] > x[h] for h in range(4)]
            if not (
                hs["error"] == L.L2CPU_ERR_TRAP
                and hs["mcause"] == 2
                and rec["kind"] == L.L2CPU_KIND_ERROR
                and alive == [True, True, True, False]
            ):
                raise RuntimeError(f"hart 3 {hs} record {rec} alive {alive}")
            return f"hart 3 TRAP mcause 2 mepc 0x{hs['mepc']:x}, resident error park, harts 0-2 alive"

        def revive():
            out = ctl.restart(None, warm=True)
            sm.echo(5000)
            return f"WARM restart {out['t_total'] * 1e3:.2f} ms, levels {out['levels']}, all 4 harts serve"

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
            return (
                f"{a.restarts} restarts, PING + echo after each; restart median L1 {med(times['L1']):.2f} ms, "
                f"L2 {med(times['L2']):.2f} ms (+ RNMI park), restart_count {ctl.r32(base + L.L2CPU_OFF_RESTART_COUNT)}"
            )

        for name, fn in (
            ("heartbeats (4 harts)", heartbeats),
            ("mailbox ping", ping),
            ("echo work items", echoes),
            ("trap test (hart 3)", trap),
            ("restart revives hart 3", revive),
            (f"restarts x{a.restarts} (L1/L2)", restarts),
        ):
            if not sm.step(name, fn):
                break
        sm.step("final error word / alive", lambda: f"error {ctl.error()}, alive {ctl.is_alive()}")
        return finish(sm, a)
    finally:
        ttnn.close_device(dev)


def finish(sm, a):
    summary = dict(ok=sm.ok, steps=sm.steps)
    print("SUMMARY", json.dumps(summary), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(summary, f, indent=1)
    print("SMOKE", "PASS" if sm.ok else "FAIL", flush=True)
    return 0 if sm.ok else 1


if __name__ == "__main__":
    sys.exit(main())
