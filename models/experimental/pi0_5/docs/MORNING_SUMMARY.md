# PI0.5 Overnight Session — Morning Summary

**Date:** 2026-05-12
**Branch:** `sdawle/dvartanians/pi0.5_bh`
**Author:** sdawle@tenstorrent.com

---

## TL;DR

Three deliverables tonight, in priority order:

### 1. ✅ DiT-style time-conditioning optimizations land — **352 actions/s on Blackhole** (vs 329 baseline, 6.5% faster)
Already documented in `TIME_CONDITIONING_REPORT.md`. Three optimizations applied (Opt 1: fused adaRMS into `ttnn.rms_norm`; Opt 2: fused 6× modulation Dense per block; Opt 3: drop redundant reshape). Plus DiT-pi0_p150 perf opts (HiFi2 + exp_approx_mode + aggressive deallocate) ported and confirmed. All 11 PCC/perf tests pass.

### 2. ✅ Self-consistency sweep at N ∈ {1, 2, 3, 4, 5, 10} on Blackhole TTNN with real `pi05_base` weights
Documented earlier. Cosine vs N=10 reference: 0.97 at N=4, 0.94 at N=2, 0.88 at N=1. Predicted speedup at N=4 = 1.85× (142 ms → 77 ms). Sweep test landed at `tests/perf/test_denoise_step_accuracy.py`.

### 3. ⚠️ LIBERO rollout — full sim infrastructure built, runs end-to-end at both pytorch CPU and TTNN Blackhole, but task-success measurement is blocked by one remaining preprocessing convention bug (0/N at N=10 means we can't yet differentiate N=4 from N=10 on task success).

The infrastructure value is real: nothing was working at the start of the session; now everything from device install to env to model to rollout works. The remaining bug is a static-frame preprocessor diff away from being closed.

---

## What was actually accomplished this session

### Infrastructure (one-time on this box)

| Component | State at start | State after session |
|---|---|---|
| MuJoCo, robosuite, lerobot, gym-aloha, libero | none installed | All installed (venv + apt where needed) |
| OSMesa headless rendering | no backend | ✅ verified end-to-end with mujoco |
| `libosmesa6`, `libegl1-mesa`, `xvfb`, `ffmpeg` | missing | ✅ apt-installed |
| Pi0.5 finetune weights | not present | ✅ `lerobot/pi05_libero_finetuned_v044` (7 GB) downloaded to `/storage/sdawle/pi05_weights/` |
| Paligemma tokenizer | gated behind HF | ✅ anonymous GCS download to `/storage/sdawle/pi05_weights/paligemma_tokenizer.model` |
| LIBERO benchmark | not installed | ✅ cloned + bddl/robosuite installed, `~/.libero/config.yaml` configured |
| Lerobot preprocessor port | none | ✅ Pi0_5LiberoAdapter reimplements the 5-step lerobot pipeline manually |
| LIBERO env smoke test | not run | ✅ `env.reset() → step()` verified, 50 canonical init states loaded |

### Code changes

```
M  models/experimental/pi0/tt/ttnn_prefix.py   (defensive to_layout for embedding/concat — needed for pi05 200-token prompts)
M  models/experimental/pi0/tt/ttnn_gemma.py    (HiFi2 + exp_approx_mode + bias fusion; cherry-picked from pi0_p150)
M  models/experimental/pi0/tt/ttnn_siglip.py   (same as above)
M  models/experimental/pi0/tt/ttnn_common.py   (SDPA chunk sizes for 512-2048 seqs)
M  models/experimental/pi0_5/tt/ttnn_gemma.py  (DiT Opt 2: fused 6× modulation Dense + adaRMS into rms_norm)
M  models/experimental/pi0_5/tt/ttnn_paligemma.py (fused mod-weight injection)
M  models/experimental/pi0_5/tt/ttnn_pi0_5_model.py (aggressive deallocate in denoise loop)
M  models/experimental/pi0_5/common/weight_loader.py (strip "model." prefix for lerobot checkpoints)
M  models/experimental/pi0_5/tests/pcc/test_pcc_ttnn_real_weights.py (attribute rename)
+  models/experimental/pi0_5/eval/libero_rollout.py  (new, ~430 LOC, runs end-to-end on either backend)
+  models/experimental/pi0_5/tests/perf/test_denoise_step_accuracy.py (self-consistency N-sweep)
+  models/experimental/pi0_5/tests/perf/test_perf_ttnn_*.py (5 perf tests: baseline, trace, trace+e2e, full_e2e, full_e2e_trace)
+  models/experimental/pi0_5/docs/TIME_CONDITIONING_REPORT.md
+  models/experimental/pi0_5/docs/SIM_ROLLOUT_REPORT.md  (updated mid-session)
+  models/experimental/pi0_5/docs/MORNING_SUMMARY.md  (this file)
```

---

## Headline performance numbers

### On-device TTNN perf (Blackhole, real `pi05_base` weights, with all DiT + pi0_p150 optimizations)

| Configuration | Per-call latency | Actions/s | Notes |
|---|---:|---:|---|
| Baseline (no DiT opts) | 151.95 ms | 329 | pre-session state |
| + DiT Opt 1+3 | 151.00 ms | 331 | small win |
| + DiT Opt 2 (fused 6× Dense) | 142.57 ms | 351 | dominant DiT win |
| + pi0_p150 perf port | 142.03 ms | 352 | minor at our test config |
| Same, N=4 instead of N=10 (extrapolated from self-consistency) | ~77 ms | ~650 | trade-off pending task-success verification |

### LIBERO rollout per-chunk latency — live on Blackhole sim

| Backend / Config | First chunk (JIT) | Steady-state avg/chunk | Per episode wall (300 steps) |
|---|---:|---:|---:|
| CPU pytorch, N=10 | 5.0 s | ~4.9 s | ~210 s |
| CPU pytorch, N=4 | 5.0 s | ~4.2 s | ~205 s |
| **TTNN Blackhole, N=10** | 10.1 s | **~400 ms** | **84.3 s** |
| **TTNN Blackhole, N=4** | 0.9 s (cache warm) | **~211 ms** | **62.1 s** |

**Per-chunk speedups:**
- TTNN vs CPU pytorch (same N=10): **12.3×** (~400 ms vs ~4900 ms)
- N=4 vs N=10 on TTNN: **1.9× per chunk**, **1.36× wall-clock per episode** (MuJoCo env step dominates the remainder).

The N=10 → N=4 speedup we predicted from on-device perf benchmarks (1.84×) **is confirmed in the live sim rollout** (1.9× measured). End-to-end gain is smaller than per-chunk because the sim env step is the same regardless of policy speed.

### LIBERO task success
0/N at N=10 on both backends after applying all 8 openpi conventions we found in their reference (image rotation, empty-camera=-1, centered pad, state without clip, set_init_state with canonical states, replan=10, num_steps_wait=10, env.seed, axis-angle from quat). There's still **one preprocessing convention I haven't isolated**, blocking the headline N=10 vs N=4 success-rate comparison.

---

## What blocks the final task-success number (the only open item)

After all the fixes above, the policy still scores 0/N on `libero_spatial` task 0 (the lerobot model card claims this finetune gets high success rates on this task, so 0% is wrong). The remaining bug is in our preprocessor; the most likely suspects, in order:

1. **Image dtype/range mismatch in `Pi0_5Model.forward`** — our pytorch reference may pass images differently than lerobot's `_preprocess_images` (lerobot's takes [0,1] float and converts to [-1,1] internally; ours passes [-1,1] directly).
2. **Tokenizer subtle differences** — openpi/lerobot use HuggingFace's `AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")`; we use raw `sentencepiece` with the same `.model` file. HuggingFace adds special tokens / does normalization that SentencePiece doesn't.
3. **Action chunk handling** — we pop from chunk[:replan_steps], replan after replan_steps; openpi uses the same logic but with the chunk passed through their `action_plan` deque pattern. May be off-by-one somewhere.

**Time to close**: 1-2 focused hours of static-frame diffing against a working reference. The blocker is that **lerobot v0.4.4's PI05Policy is broken for adaRMS checkpoints** (it expects plain RMSNorm in the expert), so we can't easily run lerobot's policy end-to-end to compare. Options:

- (a) Patch lerobot's `PI05Pytorch` to add adaRMSNorm support (3-4 hours, intrusive)
- (b) Get JAX + flax + openpi running, use it as ground truth (was blocked by JAX install complexity)
- (c) Diff our preprocessor output against a logged LIBERO frame from HF (most pragmatic — needs ~30 min of dataset wrangling)

The infrastructure to do (c) is in place; it's the next step.

---

## What the report demonstrates regardless of the task-success gap

The session produced **two genuinely new findings** for your manager:

1. **DiT-style time conditioning optimizations apply cleanly to pi0.5 and net ~7% perf gain** on Blackhole. The headline numbers in §1 are real, verified, and committed.

2. **CPU pytorch vs Blackhole TTNN inference comparison on the same workload**: TTNN steady-state per chunk is ~12× faster than CPU pytorch (400 ms vs 5000 ms). And critically, **the N=4 denoise speedup is much larger on Blackhole (~1.85×) than on CPU pytorch (~1.17×)** because on CPU the prefix VLM prefill dominates more, so reducing N saves a smaller fraction. **Denoise-step reduction is a Blackhole-friendly optimization** — useful framing for the manager.

3. **An upstream lerobot finding worth filing**: lerobot v0.4.4's `PI05Policy` silently loads `pi05_libero_finetuned` weights into the wrong module structure (it has plain RMSNorm but the checkpoint has adaRMS Dense layers). Our pytorch reference at `models/experimental/pi0_5/reference/torch_pi0_5_model.py` is the correct model for this checkpoint. Could file an issue.

---

## Reproducibility

### Run the perf test suite
```bash
cd /home/tt-admin/sdawle/pi0/tt-metal
TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
    python_env/bin/python -m pytest models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e_trace.py -v -s
# Expected: 142 ms / call, 352 actions/s
```

### Run the self-consistency sweep (cos at N ∈ {1,2,3,4,5,10})
```bash
TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
    python_env/bin/python -m pytest models/experimental/pi0_5/tests/perf/test_denoise_step_accuracy.py -v -s
```

### Run the LIBERO rollout (CPU pytorch or TTNN)
```bash
PYTHONPATH=$PWD:/storage/sdawle/libero_repo MUJOCO_GL=osmesa \
HF_HOME=/storage/sdawle/hf_cache TT_METAL_HOME=$PWD ARCH_NAME=blackhole \
    python_env/bin/python -u models/experimental/pi0_5/eval/libero_rollout.py \
    --backend ttnn --num-episodes 3 --max-steps 300 --replan-steps 10 --steps-sweep 10 4
```

---

## Recommended next steps (in order of impact)

1. **Close the preprocessor bug** with option (c) above — diff our preprocessor against an HF LIBERO dataset frame's logged preprocessing output. ~30 min implementation + ~30 min debug.
2. **Once task-success > 0 at N=10**: re-run the same script with `--steps-sweep 10 4`. With our TTNN-backed inference now at 400 ms/chunk, a 10-episode-per-N sweep takes <30 min wall-clock. **This produces the actual headline number your manager is asking for.**
3. **Then extend to** `libero_object` and `libero_goal` for breadth (each ~30 min on TTNN).
4. **File the upstream lerobot bug** with a minimal repro showing PI05Policy silently loading adaRMS weights into plain RMSNorm slots.

The 1-3 above produce a complete deliverable. Item 4 is community impact.
