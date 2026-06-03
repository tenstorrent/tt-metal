# pi0.5 Option B vs Option C — architecture & e2e perf comparison

**Date**: 2026-06-03
**Host**: g11blx01 (32-chip Blackhole Galaxy, `ClusterType.BLACKHOLE_GALAXY`)
**Checkpoint**: `/home/tt-admin/pi05_cache/pi05_libero_upstream`
**Workload (matched, both options)**:
- `LANG_SEQ_LEN = 256`, image tokens = 256, **prefix length = 512**
- `ACTION_HORIZON = 10` → padded to 32
- `NUM_DENOISE_STEPS = 10`
- **Shrunk layout: `vlm_depth=2`, `expert_depth=1`** (replicated upload doesn't fit full 18 layers per-chip in L1 yet on either option)
- warmup = 1 iter, measured = 3 iters

## TL;DR

| Stage                          | Option B (p50, ms) | Option C (p50, ms) | Δ (B−C) |
|--------------------------------|--------------------|--------------------|---------|
| Vision (SigLIP + projector)    | 152.90             | 148.40             | +4.5    |
| Transport → VLM                | 3.93               | 3.65               | +0.3    |
| VLM prefill (whole)            | 9.74¹              | 7.66               | +2.1    |
| KV migration → expert          | 10.22              | 10.02              | +0.2    |
| Denoise (10 Euler steps)       | 41.27              | 38.98              | +2.3    |
| Per-Euler-step                 | 4.07               | 3.84               | +0.23   |
| **TOTAL e2e**                  | **220.46**         | **209.44**         | **+11.0** |

¹ Option B VLM = stage_1 (3.27) + transport_1_to_2 (3.49) + stage_2 (2.98) ms.

**Option C is ~5% faster end-to-end at the shrunk depth on the current
replicated-weights implementation.** Vision dominates wall-clock in both
(≈70% of total). The 11 ms total gap comes mostly from Option B's
mid-VLM transport hop (3.49 ms) and its slower per-Euler step
(+0.23 ms × 10 = +2.3 ms).

> **Caveat that matters:** both bench paths currently run with
> **replicated** weights (no TP sharding). Option B's `expert_slice.py:17`
> and `vision_slice.py` explicitly note "TP=8 sharding is a follow-up."
> Until TP lands in B and layer-paired-L1 lands in C, this benchmark
> measures **orchestration overhead**, *not* the
> tensor-parallel-with-all-reduce vs no-collective parallelism tradeoff
> that motivated picking one option over the other. See
> `PI0_5_GALAXY_DEPLOYMENT_PLAN.md` §3 for the analytical model that
> argued Option C wins more once those land.

---

## 1. Architecture

### Option B — 4 uniform 4×2 submeshes (Stages × TP)

```
   col→  0 1 2 3
row↓  0  0 0 1 1   stage 0 (vision_embed)     8 chips, 4×2
      1  0 0 1 1   stage 1 (vlm_first_half)   8 chips, 4×2
      2  2 2 3 3   stage 2 (vlm_second_half)  8 chips, 4×2
      3  2 2 3 3   stage 3 (expert + denoise) 8 chips, 4×2
      4  ...                                  same column-tiled layout
      5
      6
      7
   ── Galaxy 8×4 ──── 32 chips ── all used ──
```

| Stage | Chips | Layers (full / shrunk)          | Inside-stage parallelism (target) | Today |
|-------|-------|---------------------------------|-----------------------------------|-------|
| 0 vision_embed       | 8 (4×2) | SigLIP-27 + mm_proj + embed   | TP=8 across mm_proj + ViT | replicated |
| 1 vlm_first_half     | 8 (4×2) | VLM layers 0–8  /  layers 0:1 | TP=8, all_reduce after o_proj / down_proj | replicated |
| 2 vlm_second_half    | 8 (4×2) | VLM layers 9–17 /  layers 1:2 | TP=8, all_reduce after o_proj / down_proj | replicated |
| 3 expert_denoise     | 8 (4×2) | expert 0–17 + suffix MLP / expert 0:1 | TP=8, all_reduce after o_proj / down_proj | replicated |

Inter-stage transport: **DRAM-bounce through host** (`transport.py`). KV
migration: full prefix KV from stage 2 → stage 3 via the same bounce
(see `kv_migration.py`).

### Option C — 3 heterogeneous submeshes (one per pipeline stage, no TP within)

```
   col→  0 1 2 3
row↓  0  V V _ _   V = vision  4 chips,  2×2 offset (0,0)
      1  V V _ _   _ = spare   4 chips,  2×2 offset (0,2)
      2  P P P D   P = prefill 18 chips, 6×3 offset (2,0)
      3  P P P D   D = denoise  6 chips, 6×1 offset (2,3)
      4  P P P D
      5  P P P D   Total used: 28 / 32 (4 spare reserved
      6  P P P D    for denoise replica / batching)
      7  P P P D
   ── Galaxy 8×4 ──── 32 chips ── 28 active, 4 spare ──
```

| Stage   | Chips | Layers per chip (full / shrunk)        | Parallelism mode | Today |
|---------|-------|----------------------------------------|------------------|-------|
| vision  | 4 (2×2) | 9 SigLIP layers × 3 chips + 1 mm_proj/embed chip / shrunk same | no TP, splits-across-chips via device_siglip_split | replicated |
| prefill | 18 (6×3) | 1 VLM transformer layer per chip / shrunk = 2 chips active | layer-paired (one layer = one chip) | replicated upload (smoke uses paired-L1, bench doesn't) |
| denoise | 6 (6×1) | 3 expert layers per chip + suffix MLP replicated / shrunk = 1 chip active | layer-paired stripe + replicated suffix | replicated |
| spare   | 4 (2×2) | unused — reserved for denoise replica / batching | — | — |

Inter-stage transport: **host-bounce** (`transport.py`, same fallback as
B). KV migration: per-VLM-layer pairing — prefill chip _i_ ships
(K, V) to denoise chip `i // 3`. The KV migration logic lives in
`kv_migration.py` and is layer-aware.

### Architectural deltas at a glance

| Dimension                          | Option B                              | Option C                                       |
|------------------------------------|---------------------------------------|------------------------------------------------|
| Pipeline stages                    | 4                                     | 3                                              |
| Submesh count                      | 4× uniform (4×2)                      | 3× heterogeneous + 1 spare                     |
| Chips used                         | 32 / 32                               | 28 / 32                                        |
| In-stage parallelism (intended)    | TP=8 with `all_reduce`                | none — one work-unit per chip                  |
| Collectives per VLM layer (intended) | 2× all_reduce (~625 µs each @ S=64)  | 0                                              |
| Weight residence                   | DRAM (today) → L1 once TP lands       | **L1-resident everywhere** by design           |
| VLM split point                    | Mid-stack (layer 9)                   | None — full prefill is one stage               |
| Mid-stack transport hop            | Yes (3.49 ms measured)                | None                                           |
| KV migration                       | Whole-cache bounce, stage 2 → stage 3 | Per-layer pairing, prefill chip i → denoise chip i//3 |
| Spare chips                        | 0                                     | 4 (for denoise replica / batching)             |

---

## 2. Measured e2e performance

### Option C — `test_oc_bench_e2e_staged_breakdown`

```
== Option C e2e staged breakdown (replicated, shrunk vlm_depth=2 expert_depth=1) ==
stage_0_vision_ms            p5= 142.97 p50= 148.40 p95= 150.00 ms  (n=3)
transport_0_to_1_ms          p5=   3.20 p50=   3.65 p95=   4.82 ms  (n=3)
stage_1_prefill_ms           p5=   7.63 p50=   7.66 p95=   8.11 ms  (n=3)
kv_migration_ms              p5=   9.20 p50=  10.02 p95=  10.93 ms  (n=3)
stage_2_denoise_ms           p5=  38.50 p50=  38.98 p95=  39.42 ms  (n=3)
transport_2_to_host_ms       p5=   0.00 p50=   0.00 p95=   0.00 ms  (n=3)
total_ms                     p5= 204.33 p50= 209.44 p95= 213.36 ms  (n=3)
denoise_step (per Euler)     p5=   3.74 p50=   3.84 p95=   4.41 ms  (n=30)
```

### Option B — `test_ob_bench_e2e_staged_breakdown`

```
== Option B e2e staged breakdown (replicated, shrunk vlm_depth=2 expert_depth=1,
   prefix=512, denoise_steps=10, horizon=10→pad32) ==
stage_0_vision_ms                p5= 149.04 p50= 152.90 p95= 155.14 ms  (n=3)
transport_0_to_1_ms              p5=   3.79 p50=   3.93 p95=   4.25 ms  (n=3)
stage_1_vlm_first_half_ms        p5=   3.02 p50=   3.27 p95=   3.48 ms  (n=3)
transport_1_to_2_ms              p5=   3.39 p50=   3.49 p95=   3.62 ms  (n=3)
stage_2_vlm_second_half_ms       p5=   2.82 p50=   2.98 p95=   3.04 ms  (n=3)
kv_migration_ms                  p5=   9.68 p50=  10.22 p95=  10.23 ms  (n=3)
stage_3_denoise_ms               p5=  40.92 p50=  41.27 p95=  41.91 ms  (n=3)
total_ms                         p5= 215.72 p50= 220.46 p95= 224.55 ms  (n=3)
denoise_step (per Euler)         p5=   4.00 p50=   4.07 p95=   4.74 ms  (n=30)
```

### Where the 11 ms total delta comes from

```
B − C breakdown of the 11 ms total gap (p50)
─────────────────────────────────────────────────
  +4.5 ms   vision           (152.90 vs 148.40)  — variance / mesh shape diff
  +0.3 ms   transport 0→1    (3.93  vs 3.65)     — same host-bounce, noise
  +2.1 ms   VLM phase total  (9.74  vs 7.66)     — B's mid-stack transport hop
  +0.2 ms   KV migration     (10.22 vs 10.02)    — payload-floored, identical
  +2.3 ms   denoise          (41.27 vs 38.98)    — 10 × +0.23 ms per Euler step
  ───────
 +11.0 ms total
```

The vision delta (+4.5 ms) probably isn't real — n=3 and the p5/p95 bands
overlap (B: 149.04–155.14 vs C: 142.97–150.00). Re-run with `ITERS=20+`
to confirm.

The other deltas track architecture cleanly:
- B's VLM phase pays one extra **inter-stage transport hop** (stages 1↔2)
  that C doesn't have. The hop is 3.49 ms at this prefix length.
- B's per-Euler step is +0.23 ms slower. With replicated weights this is
  just the orchestration cost difference (8-chip submesh vs 6-chip
  submesh + different dispatch patterns); once TP=8 lands in B it will
  also gain ~1.25 ms of all_reduce overhead per VLM layer per the §3.1
  analytical model — which would widen the gap further, in C's favor.

---

## 3. What this benchmark does NOT measure

1. **Real-depth wall-clock.** Both runs are at shrunk depth
   (vlm_depth=2 vs full 18; expert_depth=1 vs full 18). The dispatch-floor
   and per-layer-compute terms scale very differently between options at
   full depth — see §3.1 of `PI0_5_GALAXY_DEPLOYMENT_PLAN.md` for the
   analytical model.
2. **The TP-vs-DP tradeoff that picked C over B.** Both options bench in
   replicated mode today. Option B's design intent is TP=8 with two
   `all_reduce` calls per VLM layer (≈1.25 ms / layer at S=64 per
   `test_option_b_benchmark.py`); Option C's design intent is layer-paired
   L1 with no collectives. Neither is active in the e2e bench yet.
3. **Multi-batch throughput.** Both runs are single-batch. C reserves a
   2×2 spare submesh specifically for batch-2 denoise replica; B uses all
   32 chips for a single stream.
4. **Inter-stage transport efficiency at scale.** Both use the same
   host-bounce path in `transport.py`. When `ttnn.all_gather`-via-parent
   or direct D2D submesh→submesh sockets land, both options would
   benefit, but the per-call savings would be larger for B (it has two
   inter-stage hops in the VLM phase, C has one).

---

## 4. Test methodology / repro

### Environment setup

```bash
source python_env/bin/activate
# Health check the cluster — should print ClusterType.BLACKHOLE_GALAXY:
python -c "import ttnn._ttnn.cluster as c; print(c.get_cluster_type())"
```

If `get_cluster_type()` throws `IndexError: unordered_map::at`, the
cluster is in a bad state — try `tt-smi -r`, then `tt-smi -glx_reset_auto`
if that doesn't clear it. See `OPTION_C_BENCH_AND_PCC_SESSION.md`
"Next steps" for the full recovery sequence.

### Run the two benchmarks (un-chained — do NOT use `&&`)

```bash
# Option C (run first — known-good)
PI0_OC_BENCHMARK=1 PI0_OC_BENCH_WARMUP=1 PI0_OC_BENCH_ITERS=3 \
  python -m pytest -xvs \
  models/experimental/pi0_5/tests/test_option_c_benchmark.py::test_oc_bench_e2e_staged_breakdown

# Option B (separate process — DO NOT chain with &&; a hang in B would
# leak TLB resources into the OS and require a reboot)
PI0_OB_E2E_BENCHMARK=1 PI0_OB_E2E_WARMUP=1 PI0_OB_E2E_ITERS=3 \
  python -m pytest -xvs \
  models/experimental/pi0_5/tests/test_option_b_benchmark_e2e.py::test_ob_bench_e2e_staged_breakdown
```

### Env-var knobs

| Variable                     | Default | Description                              |
|------------------------------|---------|------------------------------------------|
| `PI0_OC_BENCHMARK`           | unset   | `1` to opt into Option C benchmarks      |
| `PI0_OC_BENCH_WARMUP`        | 2       | warmup iters                             |
| `PI0_OC_BENCH_ITERS`         | 5       | measured iters                           |
| `PI0_OC_DENOISE_STEPS`       | 10      | Euler steps                              |
| `PI0_OC_CHECKPOINT`          | `/home/tt-admin/pi05_cache/pi05_libero_upstream` | real checkpoint |
| `PI0_OB_E2E_BENCHMARK`       | unset   | `1` to opt into Option B e2e benchmarks  |
| `PI0_OB_E2E_WARMUP`          | 2       | warmup iters                             |
| `PI0_OB_E2E_ITERS`           | 5       | measured iters                           |
| `PI0_OB_DENOISE_STEPS`       | 10      | Euler steps                              |
| `PI0_OB_CHECKPOINT`          | `/home/tt-admin/pi05_cache/pi05_libero_upstream` | real checkpoint |

---

## 5. Bugs found while running these benchmarks (now fixed)

### 5.1 Parent-mesh close ordering

`ttnn.close_mesh_device(parent)` does **not** cascade to carved
submeshes. The original `open_galaxy_mesh()` finally block closed only
the parent, leaving submeshes alive — tt-metal threw `MeshDevice cq ID 0
is in use by parent mesh ID N during close of mesh ID M` at process exit
and **wedged Device 10's firmware state** for the next process. Next
mesh-open hung 10s in firmware init with the signature
`Device 10: Timeout (10000 ms) waiting for physical cores to finish:
4-2, 11-2, 14-2, ...`.

**Fix**: explicitly `close_mesh_device(sm)` each submesh in reverse
order before closing the parent, in both
`tt/option_b/mesh_setup.py:open_galaxy_mesh()` and
`tt/option_c/mesh_setup.py:open_galaxy_mesh()`. Best-effort try/except
so a bad submesh doesn't block the parent close. See
`project_pi05_mesh_close_ordering.md` (auto-memory) for diagnostic
details.

### 5.2 Attention mask in L1 violates SDPA precondition

`Pi0_5PipelineC._upload_replicated` defaulted to
`ttnn.L1_MEMORY_CONFIG`, and `_build_or_upload_prefix_mask` /
`_build_or_upload_joint_mask` used that default. SDPA throws:

```
TT_FATAL @ sdpa_device_operation.cpp:80:
  mask.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
```

Smoke never hit this because `test_full_pipeline_object_dry_run_c` only
runs `initialize()`, not `run_inference()`. The e2e bench is the first
caller that actually executes SDPA on a real workload.

**Fix**: both mask builders in `tt/option_c/pipeline.py` now pass
`memory_config=ttnn.DRAM_MEMORY_CONFIG` explicitly when uploading the
mask. Activations stay L1-resident.

---

## 6. Open questions / next benches to run

1. **Confirm the +4.5 ms vision variance is noise** — re-run both
   benches with `ITERS=20+` and tighter warmup (e.g. WARMUP=5).
2. **Full-depth replicated upload.** Both currently shrunk because the
   replicated path doesn't fit full 18 layers in per-chip L1. Option C
   has layer-paired-L1 wired and smoke-tested (smoke tests #8–#10);
   driving that through `Pi0_5PipelineC` with a `layer_paired_l1=True`
   flag would let us bench C at full depth. Option B needs TP=8 weight
   sharding before it can match.
3. **TP=8 for Option B.** Once TP lands, B's per-VLM-layer cost
   incorporates the all_reduce overhead documented in
   `test_option_b_benchmark.py` (≈1.25 ms / layer at S=64). At
   `vlm_depth=2` this would add ~2.5 ms to B's VLM phase total. The
   analytical 8.90 ms full-depth target in §3.1 of the deployment plan
   becomes directly comparable then.
4. **Inter-stage transport replacement.** Both use host-bounce today.
   D2D socket or all_gather-via-parent would shave ~3–4 ms per hop
   (cuts B's VLM phase to ~6 ms; C unaffected since it has no mid-stack
   hop).
5. **Option B PCC vs torch reference.** `test_pcc_option_c_vs_torch.py`
   exists for C; mirror it for B (`Pi0_5PipelineB`) so the pipelines are
   validated head-to-head at the same shrunk depth.
6. **Multi-batch denoise on C's spare submesh.** C reserves a 2×2 spare
   that's currently unused. With batch-2 denoise replica wired up, C
   could amortize the vision cost over two action samples.

---

## 7. Pointers

- Bench files:
  - `models/experimental/pi0_5/tests/test_option_c_benchmark.py`
  - `models/experimental/pi0_5/tests/test_option_b_benchmark_e2e.py`
- Architecture & layout:
  - `models/experimental/pi0_5/tt/option_b/README.md`
  - `models/experimental/pi0_5/tt/option_c/README.md`
- Deployment plan / analytical model:
  - `models/experimental/pi0_5/docs/PI0_5_GALAXY_DEPLOYMENT_PLAN.md`
- Prior session log:
  - `models/experimental/pi0_5/docs/OPTION_C_BENCH_AND_PCC_SESSION.md`
- Raw run logs (this session):
  - `_bench_runs/option_c_e2e_*.log`
  - `_bench_runs/option_b_e2e_*.log`
