# SDPA optimization protocol — how to approach this methodically

Read `SDPA_PERF_HANDOFF.md` first (objective, rules, levers, parallelization). This
doc is the *method*: the disciplined loop, how to profile, and how to keep a journal
so progress is auditable and reversible. It expands on
`tt_metal/tools/profiler/PROFILING_GUIDE.md` for this specific op.

## The loop (do this every iteration)

1. **Baseline.** Run the gate, record `device_kernel_ns` + `util` for all 4 cases.
2. **Profile — find the bottleneck before changing anything.** Don't guess. See
   "Profiling" below. Decide *which phase* dominates (QKᵀ, softmax/rescale, P·V, or
   KV reload / data movement).
3. **One hypothesis, one change.** Form a single falsifiable hypothesis ("c_kv=1 makes
   the softmax fire 512× for d=512; raising c_kv to 4 should cut device_kernel_ns on
   case 4"). Change *one* thing.
4. **Measure.** Re-run the gate. Compare `device_kernel_ns` to the previous best.
5. **Journal it** (see below) — hypothesis, change, numbers, keep/revert decision.
6. **Keep or revert.** Keep only if `device_kernel_ns` dropped *and* the gate still
   passes (PCC ≥ 0.99 on all 4, `cores == 64`). Otherwise revert and journal why.
7. Repeat from 2.

**Hard rules**
- **Change one thing at a time.** Two simultaneous changes = you can't attribute the result.
- **The gate is law:** PCC ≥ 0.99 on every case, and every case must still fill the grid
  (`cores == 64`). Breaking precision *below* 0.99 or dropping a case off the grid is a
  regression, not a win — even if it's faster.
- **`device_kernel_ns` is the score.** Drive it down. `util` is the same thing rescaled.
- **Never delete a faster config without journaling the exact change** — you must be
  able to reconstruct the best result.

## Measure (the gate + score)

```bash
scripts/run_safe_pytest.sh --run-all -s \
  tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_sdpa_perf_juice.py
```

Each case prints + appends to `generated/sdpa_juice_perf.txt`:
`cores=64/64 device_kernel_ns=<N> achieved=<X>TFLOPs util=<Y>% pcc=<P>`.

Baseline (current kernel, measured): case1 `h8/s4096/d64` = **12.23 ms, 1.09%**;
case2 `h8/s8192/d128` = **54.95 ms, 1.94%**. (cases 3–4 to be filled on first run.)

## Profiling — finding *why* it's slow

`device_kernel_ns` tells you *how* slow; to find *where* the time goes, profile.

```bash
scripts/run_safe_pytest.sh --tracy --run-all \
  tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_sdpa_perf_juice.py
# → per-op CSV under generated/profiler/run_safe_pytest_<ts>/reports/<ts>/ops_perf_results_<ts>.csv
```

What the CSV tells you, and the **gotcha for this op**:

- `DEVICE KERNEL DURATION [ns]` — the headline span. Confirmed `CORE COUNT = 64`,
  `MATH FIDELITY = HiFi3` (← the op defaults to HiFi3, *not* LoFi — switching the
  matmuls to LoFi/HiFi2 where PCC allows is a direct lever).
- ⚠️ **`DEVICE COMPUTE CB WAIT FRONT` / `CB RESERVE BACK` / `PER CORE MIN/MAX/AVG`
  come out EMPTY for this op.** Those columns are populated from the *framework's*
  built-in CB-wait zones, which a hand-written **generic-op** compute kernel does not
  emit. So you **cannot** use the automatic bottleneck columns here — you must
  instrument the kernel yourself.

### Instrument the kernel (the real profiling path for this op)

Drop `DeviceZoneScopedN` markers around the phases in
`kernels/scaled_dot_product_attention_compute.cpp` (no `#include` needed — the build
injects it when the profiler is on; the macros compile to nothing when off):

```cpp
{ DeviceZoneScopedN("SDPA-QKT");      /* Q·Kᵀ matmul */ }
{ DeviceZoneScopedN("SDPA-SOFTMAX");  /* max / exp / sum / online rescale */ }
{ DeviceZoneScopedN("SDPA-PV");       /* P·V matmul */ }
{ DeviceZoneScopedN("SDPA-KV-LOAD");  /* reader/compute KV chunk handling */ }
```

Kernels are JIT-compiled at runtime — **no full rebuild needed**, just re-run with
`--tracy`. Then read the per-zone durations from the raw device log
(`generated/profiler/.../profile_log_device.csv`): for each `(core, RISC, zone)`,
`duration = END_cycle − START_cycle`. The zone with the largest active delta between
CB-sync points is where the time actually goes.

**Interpretation rule (PROFILING_GUIDE §4):** per-RISC kernel durations are *occupancy*,
not *work* — a RISC's "duration" includes time parked in `cb_wait_front`/`cb_reserve_back`
waiting on the bottleneck stage. BRISC ≈ NCRISC ≈ TRISC durations is the *expected*
signature of any pipeline; it does **not** mean balanced. Use your explicit
`DeviceZoneScopedN` deltas (between CB sync points) to find the real active-time hog.

## The journal (breadcrumb-style)

Keep an append-only journal at `agent_logs/perf_journal.jsonl` — one JSON line per
attempt. This mirrors the implementer breadcrumbs: it makes your reasoning auditable,
stops you re-trying dead ends, and lets anyone (including you) reconstruct the best
config. Use the helper:

```bash
python3 tests/ttnn/unit_tests/operations/scaled_dot_product_attention/log_perf_attempt.py \
  --hypothesis "d=512 c_kv=1 → softmax fires per KV tile; raise c_kv to 4" \
  --change "program_descriptor: decouple c_kv from 16//Dt clamp, set c_kv=4" \
  --decision keep \
  --note "case4 device_kernel_ns 9.9e8 -> 4.1e8, pcc 0.997"
```

It timestamps the entry and auto-attaches the latest measured numbers from
`generated/sdpa_juice_perf.txt`. Schema per line:

```json
{"ts":"…","attempt":N,"hypothesis":"…","change":"…","files":["…"],
 "results":{"case":{"ns":…,"util":…,"pcc":…}}, "decision":"keep|revert","note":"…"}
```

**Journal discipline:** one entry per change, *before* you move on. Record the decision
(keep/revert) and the numbers that justified it. If you revert, say what the change was
and why it didn't help — that's the most valuable kind of entry.

## Suggested first moves (highest leverage — see handoff for the full list)

1. **`c_kv` for d=512.** `head_dim=512 ⇒ c=1 ⇒ c_kv=1`: the online-softmax loop runs once
   per KV *tile*. Decouple `c_kv` from the `16/Dt` clamp; bigger KV chunks = more matmul
   work per softmax step. Almost certainly the biggest single win on case 4.
2. **Math fidelity.** Op runs HiFi3 by default; LoFi/HiFi2 matmuls are faster — spend the
   PCC-0.99 slack here.
3. **Overlap / shrink the online-softmax bookkeeping** between the two matmuls.
4. **Bigger `c_q`** to amortize per-chunk softmax + KV reload across more matmul rows.

Re-profile after each — the bottleneck moves as you fix the current one.
