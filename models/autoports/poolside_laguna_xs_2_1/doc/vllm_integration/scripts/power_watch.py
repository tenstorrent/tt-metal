#!/usr/bin/env python3
"""Sample per-ASIC power/clock/temp + host-side server phase during a bench sweep."""
import json
import subprocess
import sys
import time

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/power_watch.log"
PERIOD = 2.0
TDP = 125.0


def snap():
    out = subprocess.run(["tt-smi", "-s"], capture_output=True, text=True).stdout
    d = json.loads(out)
    devs = []
    for dev in d["device_info"]:
        t = dev["telemetry"]
        s = dev["smbus_telem"]
        devs.append(
            dict(
                p=float(t["power"]),
                clk=int(t["aiclk"]),
                temp=float(t["asic_temperature"]),
                v=float(t["voltage"]),
                i=float(t["current"]),
                gddr_t=int(s["MAX_GDDR_TEMP"], 16),
            )
        )
    return d["time"][11:19], devs


def cur_bench():
    try:
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
        for line in ps.splitlines():
            if "vllm bench serve" in line and "timeout" not in line:
                il = ol = mc = "?"
                p = line.split()
                for k, name in (
                    ("--random-input-len", "il"),
                    ("--random-output-len", "ol"),
                    ("--max-concurrency", "mc"),
                ):
                    if k in p:
                        v = p[p.index(k) + 1]
                        if name == "il":
                            il = v
                        elif name == "ol":
                            ol = v
                        else:
                            mc = v
                return f"ISL={il} OSL={ol} C={mc}"
    except Exception:
        pass
    return "idle"


hdr = "time     bench                     " + " ".join(f"{'dev'+str(i):>22}" for i in range(4)) + "   sum_W  %TDP"
with open(LOG, "a", buffering=1) as f:
    f.write("\n=== power_watch start ===\n" + hdr + "\n")
    while True:
        try:
            ts, devs = snap()
            b = cur_bench()
            cells = " ".join(f"{d['p']:5.0f}W {d['clk']:4d}MHz {d['temp']:4.1f}C" for d in devs)
            tot = sum(d["p"] for d in devs)
            f.write(f"{ts} {b:<25} {cells}   {tot:5.0f}  {100*tot/(4*TDP):4.0f}%\n")
        except Exception as e:
            f.write(f"ERR {e}\n")
        time.sleep(PERIOD)
