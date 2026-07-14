#!/usr/bin/env python3
# Step 6: device-perf PARITY test for ttnn.experimental.regime_a_matmul vs the frozen C++ oracle
# (GOLDEN_PARITY_SUITE.json). Runs the SAME golden manual configs through the ttnn op, measures
# device-profiler kernel time with the SAME methodology as the oracle (max across (core,RISC) zones per
# run, min over runs 1..N), and reports the op's us + effective/delivered BW next to the oracle's.
#
# Parity criterion: op kernel time within 5% of the C++ prototype. One config per subprocess (fresh CSV),
# mirroring the oracle's one-process-per-config C++ sweep.
#
# Usage (env: source bh_env.sh && source python_env/bin/activate):
#   driver:  python3 tools/mm_sweep/regime_a_perf_parity.py
#   worker:  python3 tools/mm_sweep/regime_a_perf_parity.py --run M K N Ns Pk Sm kb nsb   (internal)
import csv, os, sys, json, subprocess
from collections import defaultdict

ROOT = "/localdev/cglagovich/tt-metal"
CSV = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
ORACLE = f"{ROOT}/tools/mm_sweep/golden_parity_suite.json"
FREQ = 1.35e9
PEAK = 512e9
TB = 2048
NRUN = 6  # best-of over runs 1.. (run 0 is warmup / program-cache compile)


def cdiv(a, b):
    return (a + b - 1) // b


def rup(x, y):
    return ((x + y - 1) // y) * y


def geo_bytes(M, K, N, Ns, Pk, Sm, kb, nsb):
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    Ktl = rup(cdiv(Kt, Pk), kb * 8)
    Kts = Pk * Ktl
    Mblk = cdiv(Mt, Sm)
    Mts = Sm * Mblk
    Nband = cdiv(Nt, 8)
    Nown = cdiv(Nband, Ns)
    Nbpc = cdiv(Nown, nsb)
    Nts = 8 * Ns * (Nbpc * nsb)
    real = (Mt * Kt + Kt * Nt + Mt * Nt) * TB
    phys = (Mts * Kts + Kts * Nts + Mts * Nts) * TB
    return real, phys


def parse_profiler():
    try:
        rows = list(csv.reader(open(CSV)))
    except Exception:
        return None
    ev = defaultdict(list)
    for row in rows[2:]:
        if len(row) < 12:
            continue
        if not row[10].strip().endswith("-KERNEL"):
            continue
        ev[(row[1], row[2], row[3], row[10].strip())].append((row[11].strip(), int(row[5])))
    dur = {}
    for k, l in ev.items():
        ds, st = [], None
        for t, c in l:
            if t == "ZONE_START":
                st = c
            elif t == "ZONE_END" and st is not None:
                ds.append(c - st)
                st = None
        dur[k] = ds
    if not dur:
        return None
    nruns = min((len(v) for v in dur.values()), default=0)
    if nruns < 2:
        return None
    return min(max(v[i] for v in dur.values()) for i in range(1, nruns))


def worker(M, K, N, Ns, Pk, Sm, kb, nsb):
    import torch
    import ttnn

    # fresh CSV so only this config's zones are parsed
    try:
        os.remove(CSV)
    except OSError:
        pass
    dev = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        t_in0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
        t_in1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
        in0 = ttnn.from_torch(t_in0, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        wcfg = ttnn.create_regime_a_weight_memory_config(list(t_in1.shape), ttnn.bfloat16, dev)
        in1 = ttnn.from_torch(t_in1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=wcfg)
        cfg = ttnn.RegimeAMatmulConfig(k_slices=Pk, n_slices=Ns, m_slices=Sm, k_block_tiles=kb, n_subblock_tiles=nsb)
        for _ in range(NRUN):
            out = ttnn.experimental.regime_a_matmul(in0, in1, config=cfg)
            ttnn.synchronize_device(dev)
        ttnn.ReadDeviceProfiler(dev)
    finally:
        ttnn.close_device(dev)
    cyc = parse_profiler()
    print(f"CYC {cyc if cyc is not None else -1}")


def driver():
    oracle = {tuple(e["shape"]): e["golden"] for e in json.load(open(ORACLE))}
    print(
        f"{'shape':22} {'cfg(Ns,Pk,Sm,kb,nsb)':22} {'oracle us':>9} {'op us':>8} {'delta':>7}  {'op eff%':>7} {'op deliv%':>9}"
    )
    worst = 0.0
    for shape, g in oracle.items():
        M, K, N = shape
        Ns, Pk, Sm, kb, nsb = g["Ns"], g["Pk"], g["Sm"], g["kb"], g["nsb"]
        env = dict(os.environ)
        env["TT_METAL_DEVICE_PROFILER"] = "1"
        a = [sys.executable, __file__, "--run", str(M), str(K), str(N), str(Ns), str(Pk), str(Sm), str(kb), str(nsb)]
        r = subprocess.run(a, env=env, capture_output=True, text=True, cwd=ROOT)
        cyc = None
        for line in r.stdout.splitlines():
            if line.startswith("CYC "):
                cyc = int(line.split()[1])
        if cyc is None or cyc < 0:
            print(
                f"{M}x{K}x{N:<12} {'('+str(Ns)+','+str(Pk)+','+str(Sm)+','+str(kb)+','+str(nsb)+')':22} MEASURE FAILED"
            )
            print(r.stdout[-500:], r.stderr[-800:])
            continue
        us = cyc / FREQ * 1e6
        real, phys = geo_bytes(M, K, N, Ns, Pk, Sm, kb, nsb)
        eff = real / (cyc / FREQ) / PEAK * 100
        deliv = phys / (cyc / FREQ) / PEAK * 100
        oracle_us = g["us"]
        delta = (us - oracle_us) / oracle_us * 100
        worst = max(worst, abs(delta))
        cfgs = f"({Ns},{Pk},{Sm},{kb},{nsb})"
        print(f"{M}x{K}x{N:<12} {cfgs:22} {oracle_us:9.1f} {us:8.1f} {delta:+6.1f}%  {eff:6.1f}% {deliv:8.1f}%")
    print(f"\nworst |delta| = {worst:.1f}%  (parity bar: <=5%)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        worker(*[int(x) for x in sys.argv[2:10]])
    else:
        driver()
