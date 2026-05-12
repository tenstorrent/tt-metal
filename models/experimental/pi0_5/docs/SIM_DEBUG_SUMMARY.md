# PI0.5 LIBERO Sim Debug — Summary

**Date:** 2026-05-12
**Author:** sdawle
**Goal:** make `libero_rollout.py` succeed on LIBERO tasks (had been 0/N).

## Bugs found and fixed

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
- Removed the experimental `PI05_ACTION_SCALE` hack (no longer needed).

### 2. TTNN noise tensor never resampled

`Pi0_5ModelTTNN.__init__` allocated one fixed `self.x_t_ttnn` tensor at
construction. `sample_actions` then assigned `x_t_ttnn = self.x_t_ttnn` once
and integrated forward — but `self.x_t_ttnn` was never replaced with fresh
noise on subsequent calls. So every chunk in a rollout used the *same*
initial noise, biasing inference toward whatever flow-matching attractor
that seed lands near.

**Fix** (`models/experimental/pi0_5/tt/ttnn_pi0_5_model.py`):
- `sample_actions` now samples a fresh `torch.randn(1, action_horizon, action_dim)`
  and uploads it to a fresh TTNN buffer on every call, matching lerobot's
  `sample_noise` (modeling_pi05.py:618) and the pytorch reference.
- Opt-out flag `resample_noise = False` for the deterministic-noise test
  (`tests/perf/test_denoise_step_accuracy.py`).

The pytorch path was already correct (the shared `DenoisingModule.sample_actions`
calls `sample_noise(...)` fresh each call).

## Effect on action signature

Same observation, single chunk on `libero_spatial` task 0, after the
normalization fix:

| | act[0] (chunk 1) |
|---|---|
| Before (QUANTILES) | `[+0.051, +0.092, +0.006, +0.031, +0.010, −0.009, +0.597]` |
| After (MEAN_STD)   | `[−0.088, +0.100, −0.082, +0.016, +0.001, −0.013, +0.446]` |

Z-axis sign flipped from "stationary" to "descending" — directionally correct.

## What's still broken: model produces wrong-direction actions

After both fixes, **task success is still 0/N** across multiple seeds and
denoise step counts:

| Config | Result |
|---|---|
| seed=0, N=10, replan=5, max_steps=220 | 0/1 |
| seed=1, N=10, replan=5, max_steps=220 | 0/1 |
| seed=2, N=10, replan=5, max_steps=220 | 0/1 |

To isolate the cause, I loaded `lerobot/libero` from HuggingFace, found
**ep 1275** (matches our task description, init state within 6mm of our env),
and fed our model **the exact training-time observation** (image + state +
prompt) from frame 10 of that episode. Training data at that frame says the
correct action is:

```
[+0.9375, +0.4339, +0.1741, −0.0396, +0.0000, +0.0032, −1.0000]
                                                       ^ gripper open
```

i.e., saturated +x, +y, +z to descend toward the bowl with gripper open.

Our model on the **exact same observation** produces (3 seeds, N=10 and N=50):

```
seed 0 N=10: [−0.138, −0.057, −0.202, +0.018, −0.001, +0.020, +0.471]
seed 1 N=10: [−0.158, −0.001, −0.139, +0.000, +0.003, −0.002, +0.605]
seed 2 N=10: [−0.110, +0.104, −0.226, +0.010, −0.028, +0.040, +0.255]
seed 0 N=50: [−0.115, −0.016, −0.164, +0.015, +0.004, +0.021, +0.609]
seed 1 N=50: [−0.116, +0.027, −0.161, +0.001, +0.002, −0.003, +0.553]
seed 2 N=50: [−0.112, +0.121, −0.196, +0.008, −0.022, +0.035, +0.530]
```

Notice on x, z, and the gripper the **sign is consistently opposite** to the
training target, and the magnitudes are too small for the position dims.
Increasing N (10 → 50 steps) doesn't change this — output is converged but
wrong.

This is not a randomness issue and not a controller issue. It's a model-output
correctness issue.

## What was investigated and ruled out

| Hypothesis | Outcome |
|---|---|
| Wrong controller (JOINT_VELOCITY) | Ruled out — `OSC_POSE` is in use; verified an open-loop +x command for 30 steps moves eef by 337 mm. |
| Action saturation | Ruled out — even `PI05_ACTION_SCALE=10×` (saturated commands) moves the robot far less than expected. |
| Image rotation / preprocessing | Visually compared our env's rotated `agentview_image` vs the training dataset image for ep1275 — identical scene, orientation, colors. Same for wrist. |
| Empty-camera padding value | Already −1 (black in [−1,1]), matches lerobot. |
| Quaternion → axis-angle convention | Same formula as lerobot's `_quat2axisangle`. |
| State assembly order | `[eef_pos(3), axis_angle(3), gripper_qpos(2)]` — matches lerobot `LiberoProcessorStep` exactly. |
| State discretization | Bin output for our env state matches what training-time discretization would produce for the same physical state. |
| Prompt template | `"Task: {desc}, State: {bins};\nAction: "` matches `pi05_prepare_state_tokenizer_processor_step`. |
| SentencePiece tokenization | BOS + tokens, no EOS issues; first 30 token IDs look correct for the PaliGemma tokenizer. |
| Replan window 5 vs 10 vs full chunk | None work. |
| Initial noise reuse (TTNN) | Fixed; pytorch path was already correct. |
| QUANTILES vs MEAN_STD normalization | Fixed (confirmed bug). |
| Flow-matching dt sign / velocity sign | Tested by negating `dt` in the denoise integrator — output goes out of bounds, not closer to target. So the model's velocity output is *not* simply sign-flipped; it's the velocity *vector* that is wrong. |
| `lerobot.PI05Policy` as a reference | Cannot load — `ValueError: An incorrect transformer version is used` (lerobot pi05 requires a custom transformers fork they call `transformers_replace`). |

## Likely remaining root causes (untested)

1. **Custom adaRMSNorm implementation in our `torch_gemma.py` differs subtly
   from openpi's `transformers_replace`.** Our existing PCC tests pass against
   our own pytorch reference, not against the lerobot/openpi model. The
   chunk-order `(scale, shift, gate)` matches openpi's source, but there
   could be a difference in how the gate is applied to the residual, how the
   final norm is wired, or how `adarms_cond` flows through the layer stack.

2. **PaliGemma backbone attention masking with the longer pi05 prompt.**
   pi0 used 32 lang tokens; pi0.5 uses 200. The defensive `to_layout(TILE)`
   fixes we added during the rollout work may indicate the prefix path was
   not exercised at this prompt length during pi05 PCC testing.

3. **The pi05_libero fine-tune may itself be weak.** Without a working
   reference (openpi or lerobot), we can't tell whether the model would be
   right if we fed it through the right code path.

## Concrete next steps for whoever picks this up

1. **Run openpi's pi0_libero (not pi05) end-to-end** in this env. openpi has
   its own server + LIBERO eval (`/storage/sdawle/openpi/examples/libero/main.py`).
   If openpi's model gives the right actions on the same obs, the env and
   conventions are confirmed correct and the bug is in our model. If openpi
   also fails, the env wrapper has an issue.

2. **Numerical PCC of our adaRMSNorm output vs openpi's
   `transformers_replace/.../GemmaRMSNorm.forward`** on a single layer with
   loaded weights. Match input tensors, compare per-position outputs.

3. **Test our pi0.5 pytorch reference on the pi05_base checkpoint with a
   short prompt** — if base works but the lerobot finetune doesn't, the
   weight loader still has a mismatch we missed.

## Files changed

```
M  models/experimental/pi0_5/eval/libero_rollout.py
M  models/experimental/pi0_5/tt/ttnn_pi0_5_model.py
M  models/experimental/pi0_5/tests/perf/test_denoise_step_accuracy.py
+  models/experimental/pi0_5/docs/SIM_DEBUG_SUMMARY.md  (this file)
```

## Reproduce

```bash
cd /home/tt-admin/sdawle/pi0/tt-metal
PYTHONPATH=$PWD:/storage/sdawle/libero_repo \
MUJOCO_GL=osmesa HF_HOME=/storage/sdawle/hf_cache \
python_env/bin/python -u \
    models/experimental/pi0_5/eval/libero_rollout.py \
    --num-episodes 1 --max-steps 220 --steps-sweep 10 \
    --backend pytorch --replan-steps 5
```

Compare against the training trajectory inspection at:
```bash
HF_HOME=/storage/sdawle/hf_cache python_env/bin/python /tmp/inspect_ep.py
```
