# Ring Joint SDPA — chunked-prefill math-util: how to run & how to pick q/k

Practical guide for the chunked-prefill **compute-only** math-utilization sweep
(`test_kimi_chunked_perf_sweep.py`) and a heuristic for choosing `q_chunk` /
`k_chunk` / chunk size **without running the full sweep**.

Companion artifacts: `sdpa_factor_analysis.html` (interactive factor breakdown),
`make_kimi_sweep_viz.py` (turns the results `.md` into an HTML viz).

---

## 1. What the sweep measures

One profiled SDPA call per config: the **target chunk** (`per_device` Q rows /
device) attended against a K/V cache pre-grown to the full prefix — i.e. the
`50k+5k`-style point, not a replay of every chunk. It reports **pure-compute math
utilization** (CCL cores stripped).

> **Compute-only:** the kernels on this branch comment out NoC payload movement
> (`[COMPUTE-ONLY EXPERIMENT]` in `chain_link.hpp`, `ring_joint_reader.cpp`,
> `ring_joint_writer.cpp`) while keeping CB reserve/push + chain semaphore sync.
> So the numbers are the **compute ceiling** (no data-movement stalls). They are
> NOT end-to-end util and are not comparable across head-count or arch.

Enablers (vs main): `run_ring_joint_sdpa_chunked(do_check=, only_chunk=)`, the
`RJSDPA_SUBBLOCK` `log_info` in `ring_joint_sdpa_program_factory.cpp`.

---

## 2. How to run

### Build (required once per checkout)
The `RJSDPA_SUBBLOCK` print is host C++, so it needs a build. Kernel (data-movement)
edits compile at runtime — no rebuild for those.
```bash
./build_metal.sh
source python_env/bin/activate
```

### Full sweep (hours — ~1000+ configs)
```bash
scripts/run_safe_pytest.sh \
  tests/nightly/blackhole/sdpa/test_kimi_chunked_perf_sweep.py::test_kimi_chunked_perf_sweep
```
Results stream live to `kimi_50k_chunked_perf_sweep.md` (rewritten after every config).

### Targeted / minimal sweep (env overrides — recommended)
| env var | effect |
|---|---|
| `KIMI_SWEEP_PER_DEVICES="864,640,832"` | only these seq_len/device (else 512..1024 step 32) |
| `KIMI_SWEEP_QS="32,64"` | only these q_chunk (else {32,64,96,128}) |
| `KIMI_SWEEP_KCOUNT="2"` | cap k-steps per (pd,q); `0` = until OOM/shard |
| `KIMI_SWEEP_TARGET_PREFIX_SP8="51200"` | prefix target in sp=8 tokens (50k+5k) |
| `KIMI_SWEEP_LOGFILE="/tmp/sweep.log"` | file the orchestrator greps for `RJSDPA_SUBBLOCK` (point it at the run's stdout to fill the subblock columns) |

Example smoke test (2 configs, ~1 min):
```bash
KIMI_SWEEP_PER_DEVICES="640" KIMI_SWEEP_QS="32" KIMI_SWEEP_KCOUNT="2" \
KIMI_SWEEP_LOGFILE="/tmp/sweep.log" \
  scripts/run_safe_pytest.sh \
  tests/nightly/blackhole/sdpa/test_kimi_chunked_perf_sweep.py::test_kimi_chunked_perf_sweep \
  > /tmp/sweep.log 2>&1
```

### Notes
- The orchestrator holds the device lock and profiles each config in an inner
  subprocess (`test_chunked_single_shot`, env-driven). Don't run the inner test directly.
- `sp_size` and SDPA core count are **auto-detected** (QuietBox sp=4 → 10×10 = 100
  SDPA cores; Galaxy sp=8 → 11×10 = 110). Sizes are quoted in **sp=8** units
  (`chunk_sp8 = per_device × 8`); on an sp=4 box the on-device chunk is half that.
- Config is `kimi50k` (`CHUNKED_PREFILL_MODEL_CONFIGS`): nhq=16 per ring, nhk=1 (MLA),
  d_q=576, d_v=128. Watch the bash output for hangs (compute runs on uninitialized
  CBs by design; sync is preserved, so it shouldn't hang — but watch anyway).

---

## 3. The model — why util moves

Math util is FLOP-normalized, so it's a pure efficiency score that factors as:

```
util  ≈  E(q, k, per_device)  ×  occ̃(per_device, q)
         └ per-core efficiency ┘  └ APPROXIMATE multiplicative grid-fill ┘
```

- **`occ̃`** = fraction of the core grid kept busy. Dominant lever, hard ceiling.
  It's a ceil-based model (corr ≈ 0.83 with measured util, a bit too pessimistic
  when the last wave is <60% full).
- **`E`** = how efficient each busy core is — set by Q-reuse (q), DST packing (k),
  and overhead amortization (per_device). NOT a clean (q,k)-only constant.

### The 5 factors, ranked by how much they swing util
| factor | moves | swing | rule |
|---|---|---:|---|
| **Occupancy** (parallelism) | `occ` | **±26 pt** | make work just *below* a multiple of #cores |
| **Q-chunk** (K/V reuse) | `E` ↔ `occ` | ±12 pt | largest q the occupancy allows |
| **Subblock / DST** | `E` | ±11 pt | `k/32` divisible by 8 |
| **K-chunk** amortization | `E` | +4 pt | `k = 512` sweet spot, saturates |
| **Chunk size** (per_device) | `E` | ~few pt | bigger amortizes (at fixed occ) |

---

## 4. The heuristic — pick q_chunk and k_chunk (no full sweep)

Inputs: `per_device` (Q rows/device in the chunk), `H·B` heads (per device),
`C` SDPA cores. **On Galaxy this is H·B = 16, C = 110.**

### Step 1 — q_chunk: maximize core occupancy (pure arithmetic, 0 device runs)
For each candidate `q ∈ {32, 64, 96, 128}`:
```
work  = H·B · ceil(per_device / q)      # independent q-chunk work units
waves = ceil(work / C)                  # critical path = slowest core
occ   = work / (C · waves)              # 1.0 = last wave perfectly full
```
- Pick the q with the **highest occ**. Goal: `work` lands just *below* a multiple
  of `C` (a full last wave), never just above.
- **Tie-break** (occ within ~0.03): take the **largest** q — higher per-core
  efficiency (more Q rows reuse each K/V load; softmax/stat overhead amortized).

### Step 2 — k_chunk: full DST, then amortize
- Restrict to **`k` with `k/32` divisible by 8** → `{256, 512, 768}` (guarantees a
  width-8 QK subblock = full DST register; a prime `k/32` collapses width to 1 and
  silently costs ~10 pt).
- **Default `k = 512`** (amortizes per-k-chunk overhead, avoids L1/OOM).
- Cap at the per-device K shard `(n_prefix+1)·per_device`; larger k does nothing.

### Step 3 — verify with a micro-sweep
Profile only `q* × {256, 512, 768}` = 3 configs; keep the best. Optionally add the
width-7 k's `{448, 672}` (`k/32 = 7·n`), which occasionally edge out 512 by <0.3 pt.

---

## 5. Minimal sweep plan (find the best *triple*, ~40 runs not ~1000)

To find the best `(seq_len, q, k)` across a range:

1. **Compute occupancy for all (seq_len, q)** — free. Keep only the **frontier**:
   seq_len whose best-q `occ ≥ ~0.88`. Everything below can't win (amortization,
   ~7 pt, can't close an occ gap like 0.87→0.98).
2. **Anchor (12 runs):** at the predicted winner seq_len, run the *full* 4 q ×
   {256,512,768}. Confirms on real hardware that occupancy ranks q correctly and
   the k-rule holds. **If it doesn't, stop and widen** — `E` behaves differently
   than assumed.
3. **Frontier sweep (~24 runs):** each frontier seq_len × q* (+ runner-up q only on
   near-ties) × {256,512,768}.
4. **Refine (~6 runs):** top 2-3 triples + k ∈ {448,672} and ±1 k-step.

Galaxy (16h / 110c) frontier example: `{608, 640, 800, 832, 864, 992, 1024}`,
with q* = 32/64 depending on seq_len. Launch:
```bash
KIMI_SWEEP_PER_DEVICES="864,640,800,832,1024,608,992" KIMI_SWEEP_QS="32,64" \
  scripts/run_safe_pytest.sh \
  tests/nightly/blackhole/sdpa/test_kimi_chunked_perf_sweep.py::test_kimi_chunked_perf_sweep
```

### Why trust it — validated on the QuietBox sweep (14h/100c, 1174 configs)
- The highest-occupancy q **always** won math util (never overturned by a
  lower-occ q). Efficiency only ever broke ties → "max occ, then largest q".
- Heuristic regret vs the full-grid optimum: **0.22 pt** mean (k=512 default),
  **0.01 pt** mean (micro-sweep {256,512,768}) — exact optimum in 12/16 (k=512)
  or 15/16 (micro-sweep) seq_len. ~26× fewer configs.

---

## 6. Caveats
- **Head count = per device.** `work = H·B · ceil(pd/q)` uses heads *per device*.
  In `kimi50k`, `nhq=16` is per ring = per device (heads shard across tp_axis).
  The util math uses `nh_per_dev = model.nhq`. Get this wrong (e.g. `// tp_size`
  on Galaxy) and util is off by `tp_size`.
- **`k=512` is dims/L1-dependent.** The subblock rule (`k/32 ÷ 8`) is pure geometry
  and always holds; the *sweet-spot value* was validated for d_q=576/d_v=128 — on
  other dims confirm with the step-3 micro-sweep.
- **q-rule is general** (needs only heads, cores, per_device) and transfers across
  boxes; the irregular q* pattern on 16h/110c is just `16·ceil(pd/q)` vs 110.
- **Numbers are compute-only** and per-arch; don't compare across head-count/arch.
