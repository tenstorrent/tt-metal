# L1 footprint probe — tt-blaze pi0.5 design

A design (not yet implementation) for an L1 footprint probe analogous
to the Option B/C probes
(`models/experimental/pi0_5/tests/test_option_c_l1_footprint_probe.py`
and equivalent), targeting the tt-blaze pipeline.

The goal is the same as the existing probes: **measure per-chip L1
occupancy at three lifecycle phases (pre-init, post-init, post-warmup)
and assert it stays under the L1 cap.**

The implementation is deferred — we don't have the tt-blaze FusedOp
bodies filled in yet. But the design here pins down what the probe
needs to measure so a contributor implementing it doesn't have to
re-derive the budget math.

---

## 1. What to measure

Per chip, at each of three phases:

| Phase | When | What we expect |
|---|---|---|
| `pre_init`  | After `mesh_device` open, before any FusedOp construction | L1 used = 0 (modulo l1_small_size reservation) |
| `post_init` | After all FusedOps are constructed and weights uploaded | L1 used = weights + persistent metadata; no kernels yet |
| `post_warmup` | After one full inference (vision + prefill + denoise×10) | L1 used = weights + transient activation peak + kernel scratch |

Three phases lets us isolate:
- Weight upload bugs (`post_init` > expected).
- Transient activation explosions (`post_warmup` >> `post_init`).
- Leaks (`post_warmup` after multiple inferences > single inference).

Per chip per phase, capture:
- `l1_used_mb` — total L1 allocated, including CBs.
- `l1_free_mb` — remaining headroom.
- `dram_used_mb` — for cross-check; expect ~0 if L1 placement is
  working.

Report per-loudbox aggregate (max / mean across the 8 chips in the
loudbox) since each loudbox has uniform workload at TP=8.

---

## 2. Expected numbers (analytical, from OPTION_B_L1_ASSESSMENT.md +
  PI0_5_GALAXY_DEPLOYMENT_PLAN.md §6.1)

Per-chip L1 budget at TP=8 on Blackhole (180 MB cap):

| Stage | Loudbox | Layers / chip | Weights / chip | Activations + scratch | Total / chip | Headroom |
|---|---|---|---|---|---|---|
| Vision (SigLIP + proj) | A | 27 chained | ~73 MB | ~6 MB | ~79 MB | ~100 MB |
| VLM Prefill A | B | 9 (layers 0..8) | ~124 MB | ~10 MB | ~134 MB | ~45 MB |
| VLM Prefill B | C | 9 (layers 9..17) | ~124 MB | ~10 MB | ~134 MB | ~45 MB |
| Denoise (Expert + Suffix) | D | 18 expert + suffix | ~50 MB (incl. adaRMS DRAM bounce) + 4.5 MB VLM KV (post-migration) | ~5 MB | ~60 MB | ~115 MB |

Stages B and C are the tight ones — same as today's Option B (which is
the closest in-ttnn analog at TP=8). They should fit with similar
margin.

---

## 3. Assertions the probe enforces

For each chip, at each phase:

| Assertion | Threshold | Why |
|---|---|---|
| `post_init.l1_used <= 175 MB` | Hard cap | Per-chip L1 budget on Blackhole |
| `post_warmup.l1_used <= 175 MB` | Hard cap | Same |
| `post_warmup.l1_used <= post_init.l1_used + 20 MB` | Soft | Bounds transient peak |
| `dram_used == ~0 MB` (within 5 MB tolerance for kernel binaries) | Soft | Sanity that L1 placement is actually happening |

Per-loudbox additional:
- All 8 chips in a loudbox should have similar L1 use (within
  ±5 MB). Wide skew indicates broken TP sharding or asymmetric
  weight placement.

End-to-end:
- After 10 consecutive inferences, `post_warmup.l1_used` should be
  flat (no drift). Drift = leaked CB or socket buffer.

---

## 4. Env knobs

Mirroring the Option C probe (`PI0_OC_L1_PROBE_*` env vars), the
tt-blaze probe accepts:

| Env var | Default | Effect |
|---|---|---|
| `PI0_BLAZE_PROBE` | `1` (when running the probe test) | Enables the probe at all |
| `PI0_BLAZE_PROBE_NUM_INFERENCES` | `1` | Number of warmup inferences before `post_warmup` measurement. Use 10 for leak detection. |
| `PI0_BLAZE_PROBE_TP_SIZE` | `8` | TP factor inside each loudbox. Could test TP=4 / TP=2 / TP=1 to see CB-shrink behavior. |
| `PI0_BLAZE_PROBE_DENOISE_STEPS` | `10` | Override Euler step count (1 = single-step smoke; 10 = full). |
| `PI0_BLAZE_PROBE_DEVICE_LOOP` | `0` | `1` = device-driven denoise loop (LoopingConfig path); `0` = host-driven. |
| `PI0_BLAZE_PROBE_HOST_EMBED` | `1` | `1` = text embeddings on host (default per §7 of mapping_notes.md); `0` = on chip (won't fit, expect failure). |
| `PI0_BLAZE_PROBE_KV_MIGRATION_MODE` | `"layer_paired"` | `"layer_paired"` (default; 18 sockets) or `"broadcast"` (single socket, every denoise chip gets full VLM KV per §4 of mapping_notes.md). |

The probe writes results to
`models/experimental/pi0_5/tt_blaze/probe_results.json` (gitignored)
and prints a per-loudbox summary table to stdout. Same format as the
Option C probe so the two can be diffed.

---

## 5. Open questions the probe should resolve

These are the analytical claims in PI0_5_GALAXY_DEPLOYMENT_PLAN.md and
OPTION_B_L1_ASSESSMENT.md that need empirical confirmation on the
tt-blaze stack:

1. **Does the per-chip CB region actually shrink ~7× at TP=8 inside the
   FusedOp compiler?** The Option B analysis says yes (`out_block_w 43
   → 6 → ~0.10 MB / bank`), but the tt-blaze codegen may schedule
   things differently. Probe the post_init L1 for the VLM stages and
   compare to the analytical 124 MB.

2. **Does the 27-layer chained SigLIP FusedOp fit on loudbox A
   without per-layer CB bloat?** Each chained sub-emit allocates its
   own CBs; if those don't all get reused / scratchreset, we'd see
   linear growth with layer count. Expected: post_init L1 should be
   ~73 MB / chip. If it lands at >120 MB, the chained pattern needs
   `CbScratchReset` between layers (see fused_layer_contract.md §9 +
   `blaze/ops/cb_scratch_reset/`).

3. **Does KV migration completion actually free anything on the VLM
   stages?** Once VLM K/V is migrated to denoise, the VLM stage's
   own copy could in principle be deallocated. Probe should compare
   `post_init.l1_used` on stage 1/2 vs `post_warmup.l1_used` — if
   warmup is lower, dealloc worked; if same, the VLM still holds the
   KV (which is fine, it's only ~4.5 MB total across all 18 layers,
   so not worth chasing).

4. **Does the device-driven denoise loop have an iteration-count
   memory leak?** Test with `PI0_BLAZE_PROBE_NUM_INFERENCES=10` and
   `PI0_BLAZE_PROBE_DEVICE_LOOP=1`. Iteration 1 and iteration 10
   should report the same per-chip L1 on stage 3.

---

## 6. Implementation sketch

Mirror `tests/test_option_c_l1_footprint_probe.py`:

```python
@pytest.mark.skipif(
    os.environ.get("PI0_BLAZE_PROBE") != "1",
    reason="opt-in via PI0_BLAZE_PROBE=1",
)
def test_pi0_5_blaze_l1_footprint_probe():
    mesh = open_galaxy_mesh(...)

    # PHASE 0: pre_init
    pre_init = capture_l1_state(mesh)

    # PHASE 1: build the pipeline (uploads weights)
    pipe = Pi0_5BlazePipeline.build(mesh, load_hf_weights())
    pipe.start()
    post_init = capture_l1_state(mesh)

    # PHASE 2: warmup forward
    for _ in range(int(os.environ.get("PI0_BLAZE_PROBE_NUM_INFERENCES", "1"))):
        pixel_values, lang_tokens, noisy_actions = make_dummy_inputs()
        _ = pipe.run_inference(pixel_values, lang_tokens, noisy_actions)
    post_warmup = capture_l1_state(mesh)

    # PHASE 3: assert
    for chip in mesh.chips:
        assert post_init[chip].l1_used_mb <= 175
        assert post_warmup[chip].l1_used_mb <= 175
        assert post_warmup[chip].l1_used_mb - post_init[chip].l1_used_mb <= 20

    report = build_per_loudbox_summary(pre_init, post_init, post_warmup)
    print(report)
    write_json(report, "models/experimental/pi0_5/tt_blaze/probe_results.json")

    pipe.stop()
```

`capture_l1_state(mesh)` walks every chip and reads
`ttnn.l1_size_used(chip)` + `ttnn.dram_size_used(chip)` — same API the
Option C probe uses.

`build_per_loudbox_summary` aggregates max / mean / stddev across the
8 chips in each loudbox and renders a table matching the Option C
probe output format.

---

## 7. What the probe doesn't measure

- **Compute correctness.** PCC vs torch is a separate test (mirror
  `tests/pcc/test_pcc_option_c_vs_torch.py`).
- **End-to-end latency.** Wall-clock timing is a separate perf test
  (mirror `tests/perf/test_perf_ttnn_full_e2e_with_reports.py`).
- **Power / thermal.** Out of scope.
- **Cross-Galaxy multi-replica throughput.** Single-Galaxy only.

---

## 8. Pointers

- Option B/C probe references:
  - `models/experimental/pi0_5/tests/test_option_c_l1_footprint_probe.py`
- L1 budget math references:
  - `models/experimental/pi0_5/docs/L1_PLACEMENT_FINDINGS.md`
  - `models/experimental/pi0_5/docs/OPTION_B_L1_ASSESSMENT.md`
- tt-blaze probe / instrumentation references (to mirror):
  - `tt-blaze/tests/blaze/backed/*` — backed tests already use the
    DRAM-resident persistent tensor pattern; their per-chip memory
    measurement scaffolding is the closest existing analog.
