"""Iteration 51 scoping: per-chip ARC telemetry at 20 Hz through pyluwen (33 us per read) while a training run goes on
in another process: AICLK, TDP (W), TDC (A), VCORE (mV), ASIC temperature, the throttler bit mask. The tt-smi sampler
of the chains reads once per ~1.3 s, too coarse for a 3 s step phase. Usage: sampler.py <out.csv> [seconds] [hz]."""
import sys
import time
import pyluwen

out = open(sys.argv[1], "w")
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0
hz = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
chips = pyluwen.detect_chips()
out.write("t,chip,aiclk,tdp,tdc,vcore,temp,throttler\n")
t_end = time.time() + seconds
period = 1.0 / hz
while time.time() < t_end:
    t = time.time()
    for i, c in enumerate(chips):
        try:
            tel = c.get_telemetry()
        except Exception:  # noqa: BLE001  (the training process holds the device; reads can fail transiently)
            continue
        temp = tel.asic_temperature
        temp = (temp >> 16) + (temp & 0xFFFF) / 65536.0 if temp > 1000 else float(temp)  # 16.16 fixed point
        out.write(f"{t:.3f},{i},{tel.aiclk},{tel.tdp},{tel.tdc},{tel.vcore},{temp:.1f},{tel.throttler}\n")
    out.flush()
    time.sleep(max(0.0, period - (time.time() - t)))
