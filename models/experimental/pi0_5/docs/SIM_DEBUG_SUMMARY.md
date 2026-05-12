# PI0.5 LIBERO Sim Debug — Summary

**Date:** 2026-05-12
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

All configs use `replan_steps=5`, `max_steps=220`, task description
`"pick up the black bowl between the plate and the ramekin and place it on the plate"`.

Key takeaways:
- TTNN is ~12× faster per chunk than pytorch CPU (229ms vs 5538ms at N=4/N=10).
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

## Files changed this session

```
M  models/experimental/pi0_5/eval/libero_rollout.py        (MEAN_STD norm)
M  models/experimental/pi0_5/tt/ttnn_pi0_5_model.py        (fresh noise per call)
M  models/experimental/pi0_5/tests/perf/test_denoise_step_accuracy.py  (opt-out flag)
M  models/experimental/pi0_5/reference/torch_suffix.py     (trailing silu)
M  models/experimental/pi0_5/tt/ttnn_suffix.py             (trailing silu)
M  models/experimental/pi0_5/docs/SIM_DEBUG_SUMMARY.md     (this file)
```

## Commits

- `00842239a43` — normalization + TTNN noise resampling
- `600198b1c61` — missing trailing silu on time MLP (root cause)
- `93c3561d5d5` — initial summary with 3/3 pytorch result
