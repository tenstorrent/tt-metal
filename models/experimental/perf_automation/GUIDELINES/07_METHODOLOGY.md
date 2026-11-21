# 07 · Methodology — How to Sweep, Gate, and Not Fool Yourself

The fastest way to waste a week is to trust a sweep result that doesn't reproduce in the
model, or to ship an LN change that passes single-layer PCC and breaks the full model.
This file is the protocol that produced the three campaigns' results.

---

## 1. The incremental loop (per kept change)

1. **Isolated baseline** with Tracy — replicate the op under `signpost("start"/"stop")`,
   read `DEVICE KERNEL DURATION [ns]` from the ops-perf CSV.
2. **PCC gate** — candidate vs reference op on identical inputs; record cosine + `max|diff|`.
3. **Isolated candidate** Tracy.
4. **Wire into the model** behind an opt-in flag during exploration.
5. **End-to-end wall** — `perf.py` median of 3+, not best.
6. **End-to-end Tracy** — per-op CSV, signpost-bounded.
7. **Full-model PCC** must pass before flag-default → always-on.

Skipping step 4 or 7 was the source of every "regressed weeks later" rollback.

---

## 2. One variable at a time

Change exactly one thing per experiment: one matmul family's fidelity, OR one op's program
config, OR one memory boundary. When two changes are batched and the result regresses, you
cannot attribute it. Several BGE-M3 attempts had to be redone because two changes were
accidentally bundled.

---

## 3. The noise floor — median, and a real threshold

`perf.py` median-of-3 noise on Blackhole is ~10–20 µs. Use the **median**, not best-of-N
(best is biased toward favorable noise). A −0.5 µs/call standalone win × 24 calls = −12 µs
wall is **inside noise** — don't keep it. Targets for a confident decision: **≥50 µs wall
at small batch, ≥200 µs at large batch.** The Swin-L campaign (a confidence-scored
autoresearch loop) treated improvements under ~1× the run-to-run noise as "re-run to
confirm before keeping."

---

## 4. Two harness bugs that invalidate everything

Both *must* match production or your sweep lies:

- **`packer_l1_acc=False` in the harness** → matmul times ~3.5× too slow → bogus winners.
  Always `packer_l1_acc=True`.
- **Hard-coded 8×8 grid on Blackhole** → harness tunes against 64 cores, production uses
  110 → "winners" regress in-model. Always
  `device.compute_with_storage_grid_size()`.

**Signal something is wrong:** your standalone "prod baseline" doesn't match the in-model
Tracy number for the same op within noise. When that happens, **fix the harness, not the
production config.**

---

## 5. Single-layer PCC ≠ full-model PCC (for reductions)

| Change | 1 layer | full model (24 layers) |
|---|---:|---:|
| LoFi LayerNorm | 1.0 | 0.91 FAIL |
| one-pass LN reduction | 0.999 | 0.89 FAIL |
| bf8b sharded LN | 0.9999 | 0.50 FAIL |
| fp32_dest=False on LN | 0.999 | 0.89 FAIL |

Reduction ops (LN, softmax, GroupNorm) compound error over depth. **Always promote any
reduction-op change through full-model PCC.** Matmul/SDPA compound mildly — single-layer is
usually a fine predictor there. Depth matters: a 12-layer ViT tolerates approximations a
24-layer BERT does not (ViT runs `math_approx_mode=True` LN; BGE-M3 cannot).

---

## 5b. The accuracy ladder (bottom-up) and target PCCs

Test sub-modules up to full token generation. Target PCCs from the LLM campaigns:

| Level | Target PCC | Test |
|---|---|---|
| Sub-module (MLP, attention) | ~0.999 | per-module unit test vs HF reference |
| Single decoder layer | ~0.998 | one-layer test |
| Full model (all layers) | ~0.99 | model test, multi-iteration |
| Dataset eval | within a couple % of reference | perplexity / top-1,5 / task benchmark |

Metrics beyond PCC: **top-1/top-5 accuracy**, **perplexity**, **task benchmarks** (MMLU,
STSBenchmark, etc.), and **human ocular eval**. PCC alone can pass while generation quality
fails — if unit tests pass but the dataset/ocular eval fails, you have insufficient
consecutive-token testing or PCC targets set too low.

**Accuracy-debug ladder when a model is wrong:** find the smallest failing test → compare
op-by-op against the reference → for the suspect op try (a) higher fidelity/dtype, (b) force
DRAM-interleaved in/out, (c) drop the custom program config for the default, (d) verify
producer/consumer shard specs match, (e) for multi-device verify the CCL reduction dim /
cluster_axis, (f) regenerate cached weights from torch (a stale cached memcfg corrupts).

---

## 6. Signpost-bound the Tracy report

> For the full capture -> report -> op-bucketing workflow (commands, the CSV analysis
> script, ranking by device time and op count, drilling by shape), see
> **09_PROFILING_AND_OP_ANALYSIS.md**. This section covers only the signpost gotcha.

Default Tracy captures compile + warmup + every replay → doubled counts, inflated totals.
Bound the measured forward:

```python
ttnn.signpost("start"); out = model.forward(...); ttnn.synchronize_device(device); ttnn.signpost("stop")
```

Sanity-check op counts: for an N-layer encoder expect `4N` matmuls, `N` SDPA, `2N+1` norms.
If you see double, you're including warmup — fix the signpost range.

Filter the matmul rows by `(K, N)` to separate QKV / attn-out / FF1 / FF2 — they all show
as `MatmulDeviceOperation` and aggregate reports mix them.

---

## 6b. The five performance components

TT-NN wall time decomposes into: (1) main Python thread, (2) host API (C++ dispatch),
(3) host-device comms (PCIe + tilize/untilize), (4) device dispatch (op-to-op gap),
(5) device op execution. You control 1, 3, and (indirectly) 4-5.

- **Op-to-op gap** = host time + dispatch time. Tracing collapses host time; dispatch time
  (driven by runtime-arg count) is reduced by converting runtime args to compile-time args
  or fusing ops (LayerNorm, SDPA-decode are examples that were worth fusing). Traced
  op-to-op gap is typically < 6 us.
- **Profile the Python thread** with viztracer (only relevant when NOT tracing). Generate
  shard-spec / compute-kernel-config objects **once in the constructor**, never in the
  forward pass — torch module overhead on every call is real.
- A report showing more time in the op-to-op gap than in the ops themselves means you are
  host-bound → enable tracing (see 01 section 8 regime, and the host-bound playbook).

---

## 7. Diagnose with device time, ship wall time

`perf.py` wall = device kernel time + trace turnaround + host Python + driver polling.
Optimize against **wall** (the deliverable) but diagnose with **Tracy device** (low-noise).
A change with −50 µs device but +0 µs wall means host overhead absorbed it — that's a
signal to look at trace/2-CQ (09-style), not a win to ship by itself.

---

## 8. Watch downstream guards that silently undo your change

When Tracy shows the op didn't actually change, some downstream code reverted it:
- a default `score_dtype` that re-casts Q/K/V to bf16,
- an SDPA mask DRAM hard-assert that ships your L1 mask back,
- a `*_memory_config(seq, batch)` policy that returns DRAM regardless of your local intent.

**Trace the actual device op, not the Python intent**, then walk backwards to the guard.

---

## 9. Sweep scope discipline (lesson from the minimal_matmul sweep)

A naive block-config sweep can explode to thousands of candidates (Swin-L QKV: 6,510). Keep
it tractable:
- **Order configs best-first**: largest grid first, `M_block ≈ ceil(M_t/grid_y)` and
  `N_block ≈ ceil(N_t/grid_x)` first — don't iterate `M_block=1` (10% core utilization) before
  the plausible region.
- **Two-pass**: fast triage (few iters) over all candidates, then re-time the top-10
  accurately. Avoids paying 20-iter timing on obviously-bad configs.
- **Cap the grid set** to the device's real grids (`8×8` / `8×7` on WH; `11×10` / `10×12`
  on BH) — not every (gx, gy) permutation.
- A standalone winner that's still slower than the tuned baseline after a full sweep is a
  real negative result — `minimal_matmul` lost to tuned `ttnn.linear` for encoder-shaped
  QKV in two campaigns. Record it and move on.

---

## 10. Device hygiene during long sweeps

- Run sweeps detached (`setsid ... </dev/null &`) so an SSH drop doesn't kill them; log
  line-buffered so `tail -f` shows live progress.
- A `kill -9` on a process mid-allocation can wedge the PCIe/ARC link
  (`Read 0xffffffff ... board should be reset`). Recover with `tt-smi -r <id>`; if that
  fails it needs a driver reload or host reboot (often sudo). Prefer graceful kills.
- If a device is held by another user's process (lock `CHIP_IN_USE_*`), switch to a free
  device (`TT_VISIBLE_DEVICES`) rather than killing their job.

---

## 11. Required tooling

| Tool | Purpose |
|---|---|
| `perf.py` (or model perf test) | wall-time, median-of-N |
| Tracy harness with signposts | single-forward device time |
| `tt-perf-report` | Tracy CSV → signpost-bounded report |
| per-op CSV filter by (K,N) | separate matmul families |
| full-model PCC test | reduction-op gating |
| single-block PCC test | fast iteration |
| accuracy eval (task metric) | end-to-end correctness (catches what PCC misses) |

---

## 12. Quick reference

| Rule | Why |
|---|---|
| One variable per experiment | attribution |
| Median of 3+, ≥50/200 µs threshold | noise floor 10–20 µs |
| `packer_l1_acc=True` in sweeps | else 3.5× wrong |
| Real device grid in sweeps | else tunes wrong core count |
| Full-model PCC for reductions | depth-compounding error |
| Signpost-bound Tracy | else double-counts |
| Filter matmul by (K,N) | family ambiguity |
| Diagnose device, ship wall | they diverge when host-bound |
| Trace the kernel, not the intent | downstream guards |
| Order sweeps best-first, two-pass | thousands of configs otherwise |
