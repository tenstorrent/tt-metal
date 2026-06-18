# DeepSeek V3.2 test-suite speed report

Scope: `models/demos/deepseek_v32/tests/`. Goal: where the time goes, what is
unavoidable (downloads / cold compute), and concrete changes to make the dev loop
and future CI faster so we can afford broader coverage.

Measured on this box: **8-chip Blackhole (LoudBox)**, weights already cached (13 GB
HF shard for `DeepSeek-V3.2-Exp`), reference streams present at the default
`bit_sculpt` sibling path.

---

## 1. What is collected (exact, on an 8-chip box)

`pytest --collect-only` → **66 tests** (was 84 before rec. A landed). Marker split:
**dev=34, gate=33, nightly=0** (seq256 in `test_mla` carries both `dev` and `gate`).

| File | Tests | Tier | Per-test device cost driver |
|------|------:|------|------------------------------|
| `test_ops_shapes.py` | 15 | dev | tiny shapes; mesh open + first-run compile |
| `test_ops_numerics.py` | 15 | dev | small shapes; mesh open + compile |
| `test_indexer_chunked.py` | 1 | dev | TP-only (1 shape), 2k seq |
| `test_mla.py` | 15 | gate | full MLA forward, seq 256/2k/4k |
| `test_vs_gpu_ref.py` | 18 | gate | **5120-token forward ×3 layers ×3 shapes** (was 36) |
| `test_mla_perf.py` | 2 | (skipped on CI) | tracy harness, runs locally |

The counts are driven by **parametrization fan-out**, not by the number of test
functions (there are only ~11 functions). Two multipliers dominate:

- **Mesh-shape fan-out.** `parametrize_mesh_device()` runs *every* device test on
  *all* shapes the box supports. On a LoudBox that is **3 shapes**: `(8,1)`, `(4,2)`,
  `(2,4)`. (QuietBox=3, Galaxy=1.)
- **Reference-layer fan-out.** `test_vs_gpu_ref.py` sweeps **3 layers** (0, 30, 60).

`test_vs_gpu_ref.py` is now 9 host + **9 device** invocations (1 merged device test ×
3 layers × 3 shapes). Before rec. A it was 9 host + **27 device** (3 separate device
tests, two of which ran a full forward each).

---

## 2. Measured timing anchors

| Bucket | Measured | Notes |
|--------|---------:|-------|
| Host reference setup (CPU), per layer | **~17 s** | `_host` module fixture: cached-weight load + one 5120 CPU forward + indexer + ref-stream load |
| Host reference asserts (3 tests / layer) | **~3 s** | shared fixture, asserts are ~0–2 s each |
| Device per-invocation setup | **~2.8 s warm / ~10.6 s cold** | mesh open + `build_cpu_reference` (cached weights) + ref load |
| **Merged device ref test (warm), per (layer, shape)** | **~15 s** | ~2.8 s setup + ~12 s call = ONE 5120-token forward covering indexer + KV + output |
| Device first-run kernel JIT | **large** (~1–2 min) | one-time per kernel; persisted to `~/.cache/tt-metal-cache` |
| HF shard download (cold) | one-time | 13 GB, already cached here |

Measured (warm cache, 8-chip BH, L0/sp4×tp2): the **merged** `test_mla_device_vs_reference`
runs in **~15 s** and checks all four dimensions off one forward. `test_vs_gpu_ref.py` device
portion is now **9 invocations** → roughly **~2–3 min warm** (was 27 invocations / ~25–35 min
cold-estimate before rec. A). `test_mla.py` device (15, with skips, seq up to 4k) remains the
other gate cost. Gate is the long pole; dev (34 small-shape tests) is minutes.

---

## 3. Two findings that blocked / misled measurement (both now resolved)

> **Think-like-a-scientist flags — verify before trusting prior numbers.**

1. **`test_vs_gpu_ref.py` runtime docstring was stale — FIXED.** It claimed the device
   KV/MLA tests run a *"sparse_mla host fallback, ~24 GB RAM"*. That was no longer true:
   `tt/ops.py` calls real C++ device ops — `ttnn.experimental.deepseek.indexer_score`,
   `ttnn.experimental.topk_large_indices`, `ttnn.transformer.sparse_sdpa` (git:
   `05c173f`, `6a8e274`, `7c1a4f2`, `23c8407`). The docstring was rewritten as part of
   rec. A to describe the single merged forward and the device ops (no host fallback).

2. **`indexer_score` looked absent from the installed `ttnn` — now present.** An early
   device run failed with `AttributeError: module 'ttnn.experimental.deepseek' has no
   attribute 'indexer_score'` after ~25 s of kernel compile. On re-run (after the build
   finished installing) the op resolved and `test_mla_device_vs_reference` passes green,
   so no rebuild is needed in this checkout. **Remains a CI gotcha:** the gate job needs a
   build that *installs* the merged op (`./build_metal.sh`, the install step — see memory
   `ttnn-build-install-so`), or it fails late (after compile) rather than at collection.

---

## 4. Unavoidable costs (make them one-time, not per-run)

These are real and should be **cached/persisted**, not optimized away:

- **HF weight shards** (13 GB; layers 0/30/60 live in different shards). Persist
  `HF_HOME` across CI runs, or pre-seed via `variant.shared_path` / the variant env var
  (`get_or_download_model` already checks these first).
- **Official reference streams** (`bit_sculpt/results/deepseek-v32`). Persist and point
  `DEEPSEEK_V32_REF_DIR` at a shared mount.
- **On-disk kernel cache** (`~/.cache/tt-metal-cache`). Persisting it across CI runs
  removes the ~1–2 min first-run JIT recompile. Highest-leverage cache for CI.
- **CPU reference cache** (`DEEPSEEK_V32_MLA_REF_CACHE`, default `/tmp/...`). `test_mla`
  already disk-caches CPU truth here; persist it (currently absent → cold on this box).

---

## 5. Parameter & coverage map — what each axis tests

Each parametrization axis catches a *different* class of failure. The suite today multiplies
them (layer × shape × seq × …), but most axes are **independent** — a bug on one axis shows up
regardless of where you sit on the others. That independence is the lever for avoiding the
full cartesian product.

### Axes and what they exercise

| Axis (param) | Failure mode it catches | Result-invariant to other axes? | Swept today |
|---|---|---|---|
| **mesh shape** `(SP,TP)` | distribution: sharding, CCL all-gather / reduce-scatter, `mesh_partition`, L1 fit | **NO** — a bug hit TP=4 but not TP=2 | every device test, ×3 |
| **layer / weights** (0/30/60; `--ds-layer`) | accuracy on real weight distributions; PCC margin | port logic is layer-independent | `test_vs_gpu_ref`, ×3 |
| **seq_len** (256/2k/4k) | regime: dense (≤ index_topk=2048) vs sparse DSA (>2048); rope tables, cache sizing | independent of shape/layer | `test_mla`, ×3 |
| **chunk** (single-shot vs 1k) | chunked prefill: ring buffers, block-cyclic cache, `start_pos` causal offset | independent | `test_mla`, `test_indexer_chunked` |
| **op shape** (`sq,skv,k`) + **start_pos** | op contract & numerics: k64 (small) vs k2048 (prod), offset 0 vs chunked | independent | `test_ops_*`, ×2 |
| **n_runs** | run-to-run determinism of the DSA path (CCL order, topk ties) | leans distribution-dependent | `test_mla` (internal ×3) |
| `device_params`, `variant` | fabric/L1 config; model variant | fixed (1 value each) | not swept |

### Test groups (by what they prove)

| Group | Tests | Truth source | Weights | Tier |
|---|---|---|---|---|
| **Op contract** | `test_ops_shapes` | shape asserts | — | dev |
| **Op numerics** | `test_ops_numerics` | IndexerCPU / `torch.topk` / dense einsum | random | dev |
| **Indexer consistency** | `test_indexer_chunked` | chunked == single-shot (self) | random | dev |
| **E2E vs in-repo CPU** | `test_mla::*vs_cpu_reference` (+ chunked) | MLACPU | random | gate |
| **Determinism** | `test_mla::*determinism` | run0 == runN | random | gate |
| **vs OFFICIAL capture — host ceiling** | `test_vs_gpu_ref::*host*` | vLLM capture | pretrained | gate |
| **vs OFFICIAL capture — device port** | `test_vs_gpu_ref::test_mla_device*` | vLLM capture | pretrained | gate |
| **Perf** | `test_mla_perf` | none (tracy) | random | local |

### Avoiding the cartesian product: cover each axis once, not the grid

Because the axes are independent, you don't need layer×shape×seq — you need each axis swept
**at a fixed cheap point on the others** (a covering design, ≈ N+M+K runs instead of N·M·K):

- **Distribution suite** — *vary mesh shape; fix layer 0 + one active-regime seq.* The only
  group that must sweep all 3 shapes (rec. B). e.g. `test_vs_gpu_ref` device → layer 0 × 3 shapes.
- **Accuracy suite** — *vary layer/weights; fix one shape.* Cheapest on the **host** ceiling
  (no device, ~17 s/layer): keep all 3 layers there; one layer is enough on device (rec. C).
- **Regime suite** — *vary seq_len / chunk; fix one shape + layer.* `test_mla` only needs its
  seq sweep on a single representative shape; covering the shape axis is the distribution suite's job.
- **Op suite** — *vary op shape / start_pos; fix one shape for the inner loop* (full shape sweep
  stays in gate, rec. F).

Worked example — `test_vs_gpu_ref` (today 3 host + 9 device after the merge):
- **host: 3 layers × 1** (accuracy ceiling, cheap, no device),
- **device: layer 0 × 3 shapes = 3** (distribution + one device accuracy point),
- the shared point **(layer 0, `(4,2)`)** cross-checks host == device.
- → device **9 → 3**, and *every axis is still swept somewhere* — no coverage dropped.

> **Safety net for the one thing a covering design misses:** rare *interaction* bugs (a
> specific layer × specific shape) can't be caught by per-axis cover. Keep a **periodic
> full-grid run** as the backstop — the `nightly` marker is already registered (and currently
> empty), which is the natural home for it. Per-axis cover = the PR/CI gate; full grid = nightly.

---

## 6. Actionable speedups (ranked by impact / effort)

### A. Share one device forward across the three reference asserts — *biggest win* ✅ DONE
`test_kv_cache_device_vs_reference` and `test_mla_output_device_vs_reference` each called
`_run_device_forward()` independently → **two full 5120-token forwards where one would
do**, and `test_indexer_device_vs_reference` ran the stems a third time. The **host** side
already shares this work via the module-scoped `_host` fixture; the **device** side did not.

`mesh_device` is **function-scoped** (root conftest) and parametrized per-test over
`device_params` + 3 mesh shapes, so a cross-function shared fixture would ScopeMismatch /
fight the per-test device config. The faithful realization is therefore a **single merged
test** `test_mla_device_vs_reference` parametrized over `(layer, shape)`: one mesh open, one
forward, three assert blocks. The indexer logits/top-k are captured from that same forward
(SEQ_LEN > index_topk → DSA path calls `_indexer_topk` once) via monkeypatch, so the
indexer check needs no extra device run.

**Result: 27 device invocations → 9** (the file's device count went 27→9, suite 84→66,
gate 51→33). Verified on 8-chip Blackhole (L0, sp4×tp2): all four dimensions pass off one
~12 s forward — indexer logits 0.959, top-k mean 0.985, KV latent 0.998 (k_pe L2 0.0055),
output 0.99976.

### B. KEEP the full mesh-shape sweep — it catches real distribution bugs
~~Pin the reference tests to one shape~~ — **wrong**, do not do this. SP×TP is *not*
result-invariant: a real bug manifested at **TP=4 but not TP=2**. The sharding /
all-gather / `mesh_partition` paths differ per shape, so sweeping `(8,1)/(4,2)/(2,4)` is
genuine coverage, not redundancy. The savings on this file come from the **layer** axis
instead (see C), not the shape axis.

### C. Reduce the *layer* sweep on the device reference tests — *primary remaining win*
The device **port** logic is layer-independent (same ops, only the weights differ);
per-layer weight sensitivity is already covered cheaply on the **host** ceiling (3 layers
× ~17 s CPU). So fix the **device** reference test to **one layer** (e.g. layer 0) while
keeping the full mesh-shape sweep (per B) and the full 3-layer sweep on the host.
**Device invocations: 9 → 3** (1 layer × 3 shapes) on top of A.

### D. Disk-cache the `_host` fixture in `test_vs_gpu_ref.py`
It recomputes the 5120 CPU forward + indexer **every session** (~17 s/layer ×3 ≈ 50 s).
Mirror `run_cpu_reference`'s pattern: cache `(logits, topk, kvpe, out)` to
`DEEPSEEK_V32_MLA_REF_CACHE` keyed by `(layer, src_tag)`.

### E. Run the suite in a single pytest process
ttnn import (~seconds) and the in-process program cache are paid once per process.
Invoke `pytest models/demos/deepseek_v32/tests/ -m gate` as **one** command in CI (and
locally) rather than per-file, so import + warm program cache amortize across all tests.

### F. Give `dev` a single-shape fast path (full sweep stays in gate)
`dev` = 34 tests, with `test_ops_shapes`/`test_ops_numerics` fanning out over all 3 box
shapes. The inner loop doesn't need full distribution coverage on *every* edit — `gate`
still sweeps all shapes (and B keeps the sharding coverage that caught the TP=4-only bug).
So default `dev` to **one representative shape** for fast iteration, with an opt-in
`--ds-all-shapes`. **~3× faster inner loop**, without dropping any coverage from gate.

### G. Persist caches in CI (see §4)
HF shards, reference streams, kernel cache, CPU ref cache. Converts the "unavoidable"
download/compile into one-time setup amortized across the whole pipeline.

### H. Fix the stale docstring + rebuild dependency (see §3) — *docstring DONE*
The `test_vs_gpu_ref.py` docstring was re-measured and rewritten as part of rec. A.
Remaining: ensure the gate **CI** job builds/installs the merged `indexer_score` op
(`./build_metal.sh`), or it fails late after compile rather than at collection.

---

## 7. Projected effect

- **A (landed):** `test_vs_gpu_ref.py` device forwards **27 → 9** (one shared forward per
  (layer, shape) instead of indexer-rerun + two forwards). Measured ~15 s/test warm →
  device portion ~2–3 min warm. Suite 84 → 66, gate 51 → 33.
- **+ C (still open):** fixing the **device** reference test to one layer (while keeping
  the full mesh-shape sweep — B says do NOT drop shapes) takes device forwards **9 → 3**
  (1 layer × 3 shapes) — ~45 s warm.
- **Inner loop (dev):** F alone ≈ 3× faster; sub-minute per-op iteration once the
  kernel cache is warm.
- **D** removes ~50 s of host recompute; **E+G** remove repeated import and JIT/download.
- **Coverage headroom:** the freed budget lets us *intentionally layer* coverage —
  keep the full SP×TP sweep on the device reference path (it's load-bearing — caught a
  TP=4-only bug), but at a **single layer**, and keep the full **layer** sweep on the
  cheap host ceiling — instead of paying the full layer×shape product on device.

> A is measured (warm, L0/sp4×tp2) and its multipliers are exact from collection counts.
> B–G estimates are extrapolated from the measured setup/host/forward anchors in §2.
