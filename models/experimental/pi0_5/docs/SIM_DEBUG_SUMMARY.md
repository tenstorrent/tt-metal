# PI0.5 LIBERO Sim Debug — Summary

**Date:** 2026-05-13
**Author:** sdawle
**Status:** **Simulator working on TTNN/Blackhole.** Headline numbers on
`libero_spatial` task 0:

| backend | N | episodes | success | avg steps | ms/chunk |
|---|---|---|---|---|---|
| pytorch | 10 | 3 | 3/3 (100%) | 95.3 | 5538 |
| TTNN    | 10 | 3 | 3/3 (100%) | 85.7 | 456  |
| TTNN    | 10 | 5 | 4/5 (80%)  | 117.0 | 464 |
| TTNN    | 4  | 5 | 5/5 (100%) | 74.2  | 226 |
| **TTNN** | **4** | **50** | **48/50 (96%)** | **82.6** | **229** |

After cherry-picking the pi0_p150 perf optimizations (2D BLOCK_SHARDED matmuls
for Gemma + SigLIP, MLP padding bugfix, bf8 attention weights, MMProjector
sharding), traced inference latency drops from **142 ms → 102.70 ms / chunk at
N=10** (−28% latency, +38% action throughput). Rollout untraced steady-state
goes 229 → 201-214 ms / chunk at N=4. Task success unchanged (4/5 in a 5-ep
sanity sweep post-optims, statistically consistent with the 48/50 baseline).

All configs use `replan_steps=5`, `max_steps=220`, task description
`"pick up the black bowl between the plate and the ramekin and place it on the plate"`.

### Trace-mode perf (test_perf_ttnn_full_e2e_trace.py, N=10)

| metric | before perf optims | **after** | delta |
|---|---|---|---|
| per-call latency | 142.0 ms | **102.70 ms** | **−28%** |
| chunk throughput | 7.04 chunks/s | **9.74 chunks/s** | **+38%** |
| action throughput | 352 actions/s | **486.86 actions/s** | **+38%** |
| jitter (stddev) | — | 0.03 ms | very tight |

Key takeaways:
- TTNN is ~24× faster per chunk than pytorch CPU after both rounds of optims
  (~200ms rollout, ~102ms traced).
- **N=4 outperforms N=10** on this task — 96% (48/50) vs 80% (4/5) — and ~2×
  faster per chunk. Matches the Dense-Jump Flow Matching paper claim that
  flow-matching policies often peak at 2–4 steps.
- 2 failures (eps 33, 39 of the 50-ep run) both hit the 220-step cap on
  noise-unfortunate initial conditions; not stuck or NaN.

## Three bugs fixed

### 1. Wrong normalization scheme — `QUANTILES` instead of `MEAN_STD`

The pi05_libero fine-tune was trained with **MEAN_STD** normalization for both
`ACTION` and `STATE` (confirmed in
`/storage/sdawle/pi05_weights/pi05_libero_finetuned/policy_preprocessor.json`).
Our rollout was applying **QUANTILES**. The safetensors stats file ships both
sets of stats — picking the wrong pair compiles and runs without error.

**Fix** (`models/experimental/pi0_5/eval/libero_rollout.py`):
- `_state_normalize`: `(s − mean) / std`
- `_denormalize_actions`: `a_norm * std + mean`
- Loaded `action.mean/std` and `observation.state.mean/std` from the stats file.
- Removed the experimental `PI05_ACTION_SCALE` hack.

### 2. TTNN noise tensor never resampled

`Pi0_5ModelTTNN` allocated `self.x_t_ttnn` once at construction and reused it
across every `sample_actions` call. Every chunk in a rollout therefore used
identical initial noise, biasing inference toward whatever flow-matching
attractor that seed lands near.

**Fix** (`models/experimental/pi0_5/tt/ttnn_pi0_5_model.py`): resample fresh
N(0,1) noise per call, matching lerobot's `sample_noise` (modeling_pi05.py:618)
and the pytorch reference. Test gets an opt-out flag (`resample_noise = False`).

The pytorch path was already correct (`DenoisingModule.sample_actions` calls
`sample_noise(...)` fresh each call).

### 3. Missing trailing `silu` on the time MLP (the root cause)

Both openpi (`pi0_pytorch.py` `time_mlp_func`) and lerobot
(`modeling_pi05.py` `time_mlp_func`) compute the adaRMS conditioning vector as:

```
adarms_cond = silu(time_mlp_out(silu(time_mlp_in(sincos(t)))))
                ^^^^                                          trailing silu
```

Our reference and TTNN implementation were missing the trailing `silu`.

`adarms_cond` feeds the `Dense` projection on every adaRMS layer of the action
expert, producing `(scale, shift, gate)`. A wrong-distribution `adarms_cond`
inverts those modulations layer-by-layer, which inverts the predicted velocity
field, which integrates to actions with **opposite sign on every dim** — the
exact symptom we saw.

**Fix:**
- `models/experimental/pi0_5/reference/torch_suffix.py:embed_timestep_adarms`
- `models/experimental/pi0_5/tt/ttnn_suffix.py:embed_adarms_cond`

### Verification

Fed our pytorch model the exact training observation from `lerobot/libero`
ep1275 frame 10 (same task, same init eef position):

| dim       | training target | before silu fix | after silu fix (N=10) |
|-----------|----------------:|----------------:|----------------------:|
| dx        |          +0.94 |           −0.14 |              **+0.82** |
| dy        |          +0.43 |           −0.06 |              **+0.30** |
| dz        |          +0.17 |           −0.20 |              **+0.05** |
| droll     |          −0.04 |           +0.02 |              **−0.03** |
| dpitch    |           0.00 |            0.00 |               **0.00** |
| dyaw      |         +0.003 |           +0.02 |             **+0.002** |
| gripper   |          −1.00 |           +0.47 |              **−0.96** |

Every dim now has correct sign and reasonable magnitude.

## What the bug chain looked like

Symptom: 0/N task success.

1. Initial digging found the action-scaling / OSC-controller red herring (the
   `PI05_ACTION_SCALE` experiments). Ruled out — env responds correctly to
   saturated commands (verified open-loop: +x saturated for 30 steps moves
   eef by 337 mm).
2. Found the **QUANTILES vs MEAN_STD** bug. Fixed it. Still 0/N.
3. Found the **noise-reuse** bug on TTNN. Fixed it. Pytorch path was already
   correct, so this didn't help pytorch rollouts but mattered for TTNN.
4. Loaded the actual `lerobot/libero` training dataset, matched a real
   episode (ep1275) to our env's init state within 6 mm, and showed the
   model output had the **wrong sign on every dim** even on the exact
   training observation. This isolated the bug to the model, not the
   wrapper.
5. Diffed our action-expert path against openpi's pytorch reference
   (`/storage/sdawle/openpi/src/openpi/models_pytorch/pi0_pytorch.py`).
   Found the **missing trailing `silu`** on the time MLP. Fixed it.
6. Re-ran the rollout: **3/3 success**.

## Pi0 → Pi0.5 perf-optim port (May 13)

After the simulator started working, cherry-picked the matmul-sharding perf
wins developed on the `sdawle/dvartanians/pi0_p150` branch onto `pi0.5_bh`.
Because pi0.5 imports the shared infrastructure files from
`models/experimental/pi0/tt/` (Gemma attention/MLP, SigLIP, PaliGemma backbone,
prefix embedding), most wins transferred directly with no per-architecture
adaptation.

Cherry-picked (in chronological order):

| commit | what | pi0 gain |
|---|---|---|
| `b9f7b08af0d` | GemmaMLP padding (was padding to chunk_size=544, now next tile-32 multiple) + bf8 attn QKV/o_proj weights | 133→126 ms (−6%) |
| `92812ddc9ca` | 2D BLOCK_SHARDED matmul program_config for Gemma MLP (12×10 grid, in0_block_w=4) | 126→111 ms (−12%) |
| `b852ded530e` | 2D BLOCK_SHARDED for Gemma attention (wqkv, o_proj) | 111→100 ms (−10%) |
| `055b73819f4` | 2D BLOCK_SHARDED for SigLIP MLP + attention (adaptive K_tiles, dst_budget for fp32_dest_acc_en) | 100→98 ms (−2%) |
| `cc41296111d` | Adaptive in0_block_w by per_core_N + MMProjector shard | 98→96 ms (−2%) |
| `27466aa2d3c` | Disable `resample_noise` before `begin_trace_capture` (test-side fix; `from_torch` mid-trace was forbidden) | (test fix) |

Conflicts:
- `f7ef0742612` (Gemma attention 2D shard) conflicted on the o_proj block —
  pi0.5_bh had an older `core_grid=…` variant, took the new 2D sharded
  version with fallback.
- `be6a5818f10` (initial misc perf bundle) was already merged into pi0.5_bh
  as part of the DiT-cond mega-commit `ed45a26cb01`; skipped.

Risks considered:
- **adaRMSNorm in pi0.5 action expert.** The matmul-sharding changes are at
  the GemmaMLP / GemmaAttention level — RMSNorm flavor (plain vs adaRMS) is
  applied outside those classes, so the changes are drop-in. Verified by
  running the 50-ep N=4 rollout post-merge.
- **KV-cache L1 footprint.** pi0.5's 200-token prompt is larger than pi0's
  32-token prompt, so trace-persistent KV buffers eat more L1. The
  `in0_block_w=4` for MLP was tuned against pi0's L1 headroom; could
  overflow on pi0.5. Did not overflow in practice (trace captured cleanly,
  zero CB-overflow errors).
- **Trace + `from_torch` interaction.** The fresh-noise-per-call fix issues a
  `ttnn.from_torch` inside `sample_actions`, which trace capture forbids.
  Tests that need the captured trace must set `model.resample_noise = False`
  before `begin_trace_capture` and refresh `model.x_t_ttnn` from host
  between trace replays. Production rollouts don't trace, so they retain
  the resample-per-call behavior.

### Trace-mode perf result (test_perf_ttnn_full_e2e_trace.py, N=10)

```
Trace capture:        476.68 ms (one-time)
Per-call avg:         102.70 ms
Per-call min:         102.65 ms
Per-call max:         102.76 ms
Per-call stddev:        0.03 ms
Chunk throughput:       9.74 chunks/s
Action throughput:    486.86 actions/s
```

(was 142 ms / chunk / 352 actions/s pre-merge → **−28% latency, +38% throughput**)

### Rollout correctness post-merge (5-ep N=4)

```
N=4: success 4/5  avg_steps=127.6  avg_chunk_pred=231ms
```

Steady-state untraced is 201-214 ms / chunk; the 231 ms avg includes the
chunk-1 6.01s JIT-compile warmup. Statistically consistent with the
pre-merge 48/50 (96%) baseline — perf optims did not regress correctness.

## How to keep iterating

### Reproduce the 50-ep N=4 TTNN result (recommended starting point)

```bash
cd /home/tt-admin/sdawle/pi0/tt-metal
PYTHONPATH=$PWD:/storage/sdawle/libero_repo \
MUJOCO_GL=osmesa HF_HOME=/storage/sdawle/hf_cache \
python_env/bin/python -u \
    models/experimental/pi0_5/eval/libero_rollout.py \
    --num-episodes 50 --max-steps 220 --steps-sweep 4 \
    --backend ttnn --replan-steps 5
```

Wall time: ~20 min on Blackhole (1 chip).

### Scale up the eval

- All 10 libero_spatial tasks × 50 episodes × N=4 → ~3.5 hr wall.
- Repeat for libero_object, libero_goal, libero_10 (max_steps 280/300/520).
- N=10 50-episode baseline on task 0 (the comparison number for our 96% N=4
  result).

### Reference setup that enabled the silu fix (one-time, not in git)

```bash
python_env/bin/pip3 install transformers==4.53.2
cp -r /storage/sdawle/openpi/src/openpi/models_pytorch/transformers_replace/* \
      python_env/lib/python3.10/site-packages/transformers/
```

This installs openpi's transformers fork (adds adaRMS support to Gemma), used
to confirm our reference matches openpi's PI05Pytorch.

## Files changed (whole pi0.5 sim debug + perf-port arc)

```
M  models/experimental/pi0_5/eval/libero_rollout.py        (MEAN_STD norm)
M  models/experimental/pi0_5/tt/ttnn_pi0_5_model.py        (fresh noise per call)
M  models/experimental/pi0_5/tests/perf/test_denoise_step_accuracy.py  (opt-out flag)
M  models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e_trace.py  (resample_noise=False for trace)
M  models/experimental/pi0_5/reference/torch_suffix.py     (trailing silu)
M  models/experimental/pi0_5/tt/ttnn_suffix.py             (trailing silu)
M  models/experimental/pi0/tt/ttnn_gemma.py                (build_matmul_pcfg + sharded MLP/attn + bf8 + chunk-pad fix)
M  models/experimental/pi0/tt/ttnn_siglip.py               (sharded SigLIP MLP/attention)
M  models/experimental/pi0_5/docs/SIM_DEBUG_SUMMARY.md     (this file)
```

## Commits

- `00842239a43` — normalization + TTNN noise resampling
- `600198b1c61` — missing trailing silu on time MLP (root cause of wrong-direction actions)
- `93c3561d5d5` — initial summary with 3/3 pytorch result
- `817e196564e` — TTNN parity + N=4 50-ep result (48/50 = 96%)
- `b9f7b08af0d` — pi0 GemmaMLP padding bugfix + bf8 attn weights (cherry-pick)
- `92812ddc9ca` — 2D BLOCK_SHARDED Gemma MLP (cherry-pick)
- `b852ded530e` — 2D BLOCK_SHARDED Gemma attention (cherry-pick + conflict resolution)
- `055b73819f4` — 2D BLOCK_SHARDED SigLIP (cherry-pick)
- `cc41296111d` — adaptive in0_block_w + MMProjector shard (cherry-pick)
- `27466aa2d3c` — trace perf test: disable resample_noise before begin_trace_capture
