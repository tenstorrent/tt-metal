# Why matmul `L1_TO_L1` measurements are bistable

Some matmul measurements land on one of two discrete values, 2-6% apart, at a low
probability per execution. The main manual (`../README.md`) establishes *that*.
This is the hunt for *why*, one finding per section: what we found, and how to
reproduce it.

Every command assumes Wormhole, `source tests/.venv/bin/activate`, and
`export RUNNER_TEMP=$HOME/llk-wh-build`. Baselines referenced below live in
`~/wh_l1_x10` (10 runs, L1_TO_L1) and `~/wh_noise_isolates` (5 runs, isolates).

---

## 1. It is not any single thread

Of 42 flagged measurements, the same configurations measured one thread at a
time move by **0 of 42** (`MATH_ISOLATE`), **3 of 42** (`PACK_ISOLATE`) and
**9 of 42** (`UNPACK_ISOLATE`). A median of **0.2%** of each pipeline jump is
explained by the largest single-thread movement, against jumps with a median of
4,086 cycles and a maximum of 13,965.

```bash
python3 - <<'PY'
import pandas as pd
def load(p, rt=None):
    d = pd.read_csv(p, low_memory=False); d = d.loc[:, ~d.columns.duplicated()]
    d = d[d['marker'] == 'TILE_LOOP']
    return d[d['run_type'] == rt] if rt else d
l1  = load('~/wh_l1_x10/noise_report.points.csv'.replace('~','/home/nstojic'), 'L1_TO_L1')
iso = load('/home/nstojic/wh_noise_isolates/noise_report.points.csv')
cfg = sorted(set(c for c in l1.columns if c.startswith('cfg_')) & set(c for c in iso.columns if c.startswith('cfg_')))
for fr in (l1, iso): fr['k'] = fr['test'].astype(str) + '|' + fr[cfg].astype(str).agg('|'.join, axis=1)
m = l1.set_index('k').join(iso.pivot_table(index='k', columns='run_type', values='abs_spread', aggfunc='first'))
f = m[(m['spread'] > 0.02) & (m['abs_spread'] > 30)].copy()
f['explained_%'] = f[['MATH_ISOLATE','PACK_ISOLATE','UNPACK_ISOLATE']].max(axis=1) / f['abs_spread'] * 100
for c in ['MATH_ISOLATE','PACK_ISOLATE','UNPACK_ISOLATE']:
    print(f"{c:<16} moves >30 cycles: {int((f[c] > 30).sum())} of {len(f)}")
print(f"\nshare of jump explained by the largest thread -- median {f['explained_%'].median():.1f}%")
PY
```

**Caveat.** Isolate kernels are different binaries: the other threads do only
synchronisation, not real work. So this says no thread is unstable *under those
conditions*, not that it is stable inside the real pipeline.

---

## 2. Hardware counters cannot observe it

Across 19,920 variants with 25 flagged, the timing build separates flagged from
clean by a factor of 500 (median std **2,034.6** against **3.8**); the counter
build does not separate them at all (**2.2** against **1.8**), while measuring
the same work. The counter build also runs **1,289 cycles faster** on flagged
configurations and shows no offset on clean ones, which places it in the fast
state — the arithmetic matches a slow-state probability near 0.3.

```bash
cd ~/tt-metal/tt_metal/tt-llk/tests/python_tests
sed -i 's/^    configuration\.run(perf_report)$/    configuration.run(perf_report, run_count=5)/' perf_math_matmul.py
for MODE in "" "--enable-perf-counters --dump-csv-counters"; do
  rm -rf ../../perf_data
  CHIP_ARCH=wormhole pytest -q --compile-producer -n 10 -m perf --perf-run-types L1_TO_L1 -k perf_math_matmul $MODE .
  CHIP_ARCH=wormhole pytest -q --compile-consumer -n 15 -m perf --perf-run-types L1_TO_L1 -k perf_math_matmul $MODE .
  cp -r ../../perf_data ~/cnt_$([ -z "$MODE" ] && echo timing || echo counters)
done
git checkout -- perf_math_matmul.py
```

Then compare `std(L1_TO_L1)` in the timing snapshot against
`L1_TO_L1_std(<any counter>.cycles)` in the counter one, joined on the `cfg_`
columns.

**Trap.** The `.cycles` suffix is `OUT_L`, each bank's **total elapsed zone
time** — identical for every counter. It is not cycles attributable to that
counter. Five unrelated counters returning the same figure is the symptom.

---

## 3. Per-thread zones are too coarse to localise

Patching `_stats_l1_to_l1` to emit each thread's own zone duration gives, for all
29 flagged variants, four standard deviations equal to within a few cycles
(pipeline 4,573; unpack 4,569; math 4,580; pack 4,571). That looks decisive until
you check the means: each thread's zone spans **99.9-100.1%** of the pipeline
window, because a thread's `TILE_LOOP` zone includes the time it waits on the
others.

```bash
cd ~/tt-metal/tt_metal/tt-llk/tests/python_tests
python3 - <<'PY'
import pathlib
p = pathlib.Path('helpers/profiler.py'); t = p.read_text()
old = "    return _stats_timings(pd.concat(timings, ignore_index=True))"
new = """    result = _stats_timings(pd.concat(timings, ignore_index=True))
    for _n, _s in (("L1_TO_L1[UNPACK]", data.unpack()), ("L1_TO_L1[MATH]", data.math()),
                   ("L1_TO_L1[PACK]", data.pack())):
        _r = _stats_thread(_n, _s.raw())
        if not _r.empty:
            result = pd.merge(result, _r, on=MARKER, how="outer")
    return result"""
assert t.count(old) == 1
p.write_text(t.replace(old, new, 1)); print("patched")
PY
sed -i 's/^    configuration\.run(perf_report)$/    configuration.run(perf_report, run_count=5)/' perf_math_matmul.py
rm -rf ../../perf_data
CHIP_ARCH=wormhole pytest -q --compile-producer -n 10 -m perf --perf-run-types L1_TO_L1 -k perf_math_matmul .
CHIP_ARCH=wormhole pytest -q --compile-consumer -n 15 -m perf --perf-run-types L1_TO_L1 -k perf_math_matmul .
cp -r ../../perf_data ~/perthread
git checkout -- helpers/profiler.py perf_math_matmul.py
```

**What it does establish:** the effect survives an analysis-only instrumentation
change, so the counter build's immunity in section 2 is a property of that build.

---

## 4. Repetition cannot suppress it

Measured as **expected false failures per gate run** across 67,314 measurements —
the number that matters, because one firing measurement blocks the PR —
median-of-1 gives **22.6**, median-of-3 **7.5**, median-of-5 **5.2**, and
min-of-5 **22.1**. Median-of-5 costs five times the runtime and still fails every
PR five times over.

```bash
python3 - <<'PY'
import numpy as np, pandas as pd, random
random.seed(0)
d = pd.read_csv('/home/nstojic/wh_l1_x10/noise_report.points.csv', low_memory=False)
d = d.loc[:, ~d.columns.duplicated()]
d = d[(d['run_type']=='L1_TO_L1') & (d['marker'].isin(['TILE_LOOP','KERNEL']))]
runs = sorted([c for c in d.columns if c.startswith('run_') and c[4:].isdigit()], key=lambda c:int(c[4:]))
V = d[runs].to_numpy(float); V = V[~np.isnan(V).any(axis=1)]
for name, fn in (('median', np.median), ('min', np.min)):
    for k in (1, 3, 5):
        splits = [(lambda p: (p[:k], p[k:]))(random.sample(range(10), 2*k)) for _ in range(200)]
        fires = np.zeros(len(V))
        for a, b in splits:
            ma, mb = fn(V[:, a], axis=1), fn(V[:, b], axis=1); dd = ma - mb
            fires += ((np.abs(dd) > 30) & (np.abs(dd)/np.minimum(ma, mb) > 0.02)).astype(float)
        print(f"{name}-of-{k}: expected false failures/run {(fires/len(splits)).sum():.2f}")
PY
```

**Why the median stops helping:** the survivors are near-even splits, where the
median is itself a coin flip. **Why the minimum is worse:** it is driven by the
extreme, so whenever one side captures a rare fast run and the other does not,
the two minima differ by the full gap.

---

## 5. How the cost scales — unresolved

The **gap** between the two clusters, divided by `loop_factor x tile_cnt`, is
**1.5 to 3.5 cycles** across all 42 flagged measurements — far more consistent
than dividing by iterations alone (2.7 to 13.5), which favours a per-tile cost.
But at a fixed loop factor of 16 the **std** is flat across tile counts 1 to 16
(9.2, 8.8, 8.6, 8.9 … 5.9), which contradicts it.

Both cannot be right, and `std` is the unreliable one: for a bimodal measurement
`std = gap x sqrt(p(1-p))`, so it mixes the size of the jump with how often the
alternate state occurs. **Settling this needs gaps, not standard deviations**,
which means separate runs rather than `run_count`.

```bash
cd ~/tt-metal/tt_metal/tt-llk
sed -i 's/^            LOOP_FACTOR(1024),$/            LOOP_FACTOR(16),/' tests/python_tests/perf_math_matmul.py
BUILD_ROOT=$HOME/llk-wh-build SKIP_MAIN_CHECK=1 ALLOW_DIRTY=1 PERF_RUN_TYPES=L1_TO_L1 \
  OUT_DIR=$HOME/lf16_x10 .claude/scripts/run_perf_noise_baseline.sh wormhole 10
git checkout -- tests/python_tests/perf_math_matmul.py
```

**What is solid:** the effect persists down to a **4,945-cycle kernel**
(`loop_factor=16`), with a maximum std of 66.6 cycles — 1.3% of the measurement.
That is a 60-fold shrink from where it was found, and it is what makes RTL
simulation feasible.

```bash
.claude/scripts/perf_loop_factor_sweep.sh    # FACTORS="1024 256 64 16" RUNS=20
```

---

## 6. A quarantine list is not viable

Ranking the 83 flagged measurements by their per-run firing probability,
quarantining the worst 40 removes only **63.6%** of the risk and leaves **7.74**
expected false failures per gate run; you need essentially all 83 to reach zero.
Capture-recapture across the 5-run and 10-run baselines (53 and 83 caught, 30 in
both) estimates the true population at roughly **147**, so the list would be
incomplete the day it was written.

```bash
python3 - <<'PY'
import numpy as np, pandas as pd
d = pd.read_csv('/home/nstojic/wh_l1_x10/noise_report.points.csv', low_memory=False)
d = d.loc[:, ~d.columns.duplicated()]
d = d[(d['run_type']=='L1_TO_L1') & (d['marker'].isin(['TILE_LOOP','KERNEL']))]
runs = sorted([c for c in d.columns if c.startswith('run_') and c[4:].isdigit()], key=lambda c:int(c[4:]))
V = d[runs].to_numpy(float); V = V[~np.isnan(V).any(axis=1)]
pr = []
for v in V:
    mid = (v.min()+v.max())/2; lo, hi = v[v<mid], v[v>=mid]
    if len(lo)==0 or len(hi)==0: pr.append(0.0); continue
    gap = hi.mean()-lo.mean(); p = min(len(lo),len(hi))/len(v)
    pr.append(2*p*(1-p) if (gap > 30 and gap/lo.mean() > 0.02) else 0.0)
s = np.sort(np.array(pr))[::-1]; s = s[s>0]
print(f"can fire: {len(s)}   expected false failures/run: {s.sum():.2f}")
for n in (10,20,40,60,80,len(s)):
    print(f"  quarantine top {n:>3}: remaining {s.sum()-s[:n].sum():.2f}")
PY
```

---

## 7. Padding the pack thread does nothing — and that is a clue

Inserting 1 to 4 `TTI_NOP` before the pack thread's `TILE_LOOP` left the median
kernel at **16,716 / 16,718 / 16,718 / 16,715 / 16,718 cycles** and the bistable
share at **4.96 / 4.10 / 4.64 / 5.56 / 5.20 percent** — no effect on either.

Four NOPs costing **zero cycles** means the pack thread is **already waiting** at
loop entry, so the test never shifted anything and says nothing about the phase
model. It does confirm the packer is idle at the point where the two rhythms
diverge. The same experiment on the math thread (insert at the second
`START_PERF_MEASURE("TILE_LOOP")` rather than the third) is the untried version.

```bash
# see ~/nop_sweep.sh -- inserts before hits[2] (pack); use hits[1] for math, hits[0] for unpack
# always check median_cycles rises by ~N before reading the bistable column
```

---

## Where it stands

**Established:** the effect is real, discrete, matmul-only, and not caused by the
configuration, a fixed test set, per-run state, core placement, execution order,
concurrency, or the build state (`../README.md` sections 8.5-8.8). It is not any
single thread's own work, counters cannot see it, repetition cannot suppress it,
and no quarantine list can contain it.

**Not established:** the mechanism. The best available model is that the relative
phase of the three threads at loop entry decides which of two rhythms the
pipeline settles into, and the decision is made once. Nothing contradicts it;
nothing confirms it.

**The open lead:** a 4,945-cycle reproducer now exists, which is minutes of RTL
simulation rather than days. On a deterministic simulator you cannot observe a
coin flip, but you can *control* the initial state — perturb thread phase, DEST
bank parity, or the semaphore seed, and see which one flips the outcome. Each
perturbation isolates one handshake.

**For the gate:** none of this blocks the recommendation. 2% on `TILE_LOOP` and
`KERNEL` has zero false failures on 71,152 Blackhole and 7,890 non-matmul
Wormhole measurements. Matmul is excluded as a named hardware bug.
