#!/usr/bin/env python3
# Task 3b: decompose the in0 ring on the primary 256x2048x1024 using the DIAG_ZONES per-step ring sub-zones
# (Z_R_INJECT = own-shard DRAM read, Z_R_RECVWAIT = wait for prev's forwarded shard, Z_R_FWD = forward
# write+signal) alongside the compute-side waits (Z_C_IN0WAIT progressive in0 stall, Z_C_IN1WAIT in1 stall).
# The ring sub-zones are in phase-1, identical across chain/tree/reduce-scatter, so this attributes WHERE the
# ~4.9us compute in0-wait originates (inject vs ring-hop recv-wait vs forward). Runs mask 16 (DIAG_ZONES) at
# the primary config under the device profiler; parses the zone CSV.
import os, subprocess, sys

sys.path.insert(0, os.path.dirname(__file__))
import oracle_ablate as oa
import zone_parse as zp

M, K, N = 256, 2048, 1024
Ns, Pk, Sm, kb, nsb = 1, 4, 2, 2, 4
MASK = 16  # DIAG_ZONES
FREQ = 1.35e9
BIN = f"{oa.ROOT}/build_Release/test/ttnn/unit_tests_ttnn"

try:
    os.remove(oa.BIN_CSV)
except OSError:
    pass
env = dict(os.environ)
env.update(
    TT_METAL_DEVICE_PROFILER="1",
    TT_METAL_HOME=oa.ROOT,
    ARCH_NAME="blackhole",
    RA_M=str(M),
    RA_K=str(K),
    RA_N=str(N),
    RA_NS=str(Ns),
    RA_PK=str(Pk),
    RA_SM=str(Sm),
    RA_KB=str(kb),
    RA_NSB=str(nsb),
    RA_MASK=str(MASK),
    RA_ITERS="8",
)
r = subprocess.run(
    [BIN, "--gtest_filter=RegimeADiagFixture.Run"], env=env, cwd=oa.ROOT, capture_output=True, text=True, timeout=400
)
if "[  PASSED  ]" not in r.stdout:
    print("RUN FAILED", r.stdout[-500:], r.stderr[-300:])
    sys.exit(1)
summ = zp.summarize_per_iter(oa.BIN_CSV, freq_hz=FREQ)  # per-ITERATION summed cost (corrected aggregation)
order = [
    "Z_RING",
    "Z_R_INJECT",
    "Z_R_RECVWAIT",
    "Z_R_FWD",
    "Z_C_IN0WAIT",
    "Z_C_IN1WAIT",
    "Z_PHASE2",
    "Z_P2_RECVWAIT",
    "Z_P2_OUTWAIT",
    "Z_P2_OUTWRITE",
]
print(
    f"\n=== in0-ring + compute-wait zone decomposition (primary {M}x{K}x{N} cfg({Ns},{Pk},{Sm},{kb},{nsb}), mask 16) ==="
)
print(f"{'zone':16s} {'ncores':>7s} {'min_us':>8s} {'med_us':>8s} {'max_us':>8s} {'spread%':>8s}")
for z in order:
    if z in summ:
        s = summ[z]
        print(
            f"{z:16s} {s['ncores']:>7d} {s['min_us']:>8.2f} {s['med_us']:>8.2f} {s['max_us']:>8.2f} {str(s['spread_pct']):>8s}"
        )
# any other zones seen
for z in sorted(summ):
    if z not in order:
        s = summ[z]
        print(
            f"{z:16s} {s['ncores']:>7d} {s['min_us']:>8.2f} {s['med_us']:>8.2f} {s['max_us']:>8.2f} {str(s['spread_pct']):>8s}"
        )
