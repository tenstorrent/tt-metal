"""1 ms Blackhole ARC telemetry sampler (tt_umd, read-only BAR reads of the telemetry table, no ARC messages after
open). Every chip's AICLK / TDP / TDC / VCORE / ASIC_TEMPERATURE / AICLK_ARB_MAX each ~1 ms; a record is written only
when a value changes (the firmware refreshes the table every UPDATE_TELEM_SPEED = 100 ms). Timestamps are
time.monotonic_ns() - the host clock the prefill producer's layer acks use - so the two align.

usage: telem_sampler.py OUT.csv [period_ms]   (stop with SIGTERM / SIGINT; flushes every second)
CSV: t_mono_ns,chip,tag,value
"""

import signal
import sys
import time

import tt_umd as u

out, period = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
T = u.TelemetryTag
TAGS = [("aiclk", T.AICLK), ("tdp", T.TDP), ("tdc", T.TDC), ("vcore", T.VCORE), ("temp", T.ASIC_TEMPERATURE),
        ("arb_max", T.AICLK_ARB_MAX)]
ids = u.PCIDevice.enumerate_devices()
devs = [u.TTDevice.create(i) for i in ids]
for d in devs:
    d.init_tt_device()
rd = [d.get_arc_telemetry_reader() for d in devs]
stop = False


def _stop(*_):
    global stop
    stop = True


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)
last = {}
f = open(out, "w")
f.write("t_mono_ns,chip,tag,value\n")
f.write(f"# pci ids {ids}; period {period} ms; start wall {time.time():.3f} mono_ns {time.monotonic_ns()}\n")
# the limits the firmware enforces and the versions, per chip, once (fields 9 and 4 of every measurement)
for c, r in enumerate(rd):
    lim = []
    for name in ("TDP_LIMIT_MAX", "TDC_LIMIT_MAX", "AICLK_LIMIT_MAX", "THM_LIMIT_THROTTLE", "BOARD_POWER_LIMIT",
                 "UPDATE_TELEM_SPEED", "FLASH_BUNDLE_VERSION", "CM_FW_VERSION", "ETH_FW_VERSION", "ASIC_LOCATION"):
        tag = getattr(T, name, None)
        try:
            lim.append(f"{name}={r.read_entry(tag) if tag is not None and r.is_entry_available(tag) else 'na'}")
        except Exception as e:  # noqa: BLE001
            lim.append(f"{name}=err")
    f.write(f"# limits chip {ids[c]} " + " ".join(lim) + "\n")
nxt = time.monotonic(); flushed = nxt; sweeps = 0; late = 0
while not stop:
    t = time.monotonic_ns()
    for c, r in enumerate(rd):
        for name, tag in TAGS:
            v = r.read_entry(tag)
            k = (c, name)
            if last.get(k) != v:
                last[k] = v
                f.write(f"{t},{ids[c]},{name},{v}\n")
    sweeps += 1
    nxt += period / 1000.0
    now = time.monotonic()
    if now < nxt:
        time.sleep(nxt - now)
    else:
        late += 1
        nxt = now
    if now - flushed > 1.0:
        f.flush(); flushed = now
f.write(f"# end mono_ns {time.monotonic_ns()} sweeps {sweeps} late {late}\n")
f.close()
