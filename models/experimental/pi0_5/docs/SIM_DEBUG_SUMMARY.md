# PI0.5 LIBERO Sim Debug — Summary

**Date:** 2026-05-12
**Author:** sdawle
**Status:** **Simulator working.** `libero_spatial` task 0: **3/3 success** at
N=10 denoise steps, replan_steps=5, max_steps=220, pytorch backend.

```
LIBERO ROLLOUT SUMMARY — backend=pytorch, libero_spatial task 0
task: 'pick up the black bowl between the plate and the ramekin and place it on the plate'
N=10:  success 3/3  avg_steps=95.3  avg_chunk_pred=5538ms
```

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

### Reproduce the working rollout

```bash
cd /home/tt-admin/sdawle/pi0/tt-metal
PYTHONPATH=$PWD:/storage/sdawle/libero_repo \
MUJOCO_GL=osmesa HF_HOME=/storage/sdawle/hf_cache \
python_env/bin/python -u \
    models/experimental/pi0_5/eval/libero_rollout.py \
    --num-episodes 3 --max-steps 220 --steps-sweep 10 \
    --backend pytorch --replan-steps 5
```

### Scale up the eval

- 50 episodes × all 10 libero_spatial tasks → real success-rate number.
- Repeat for libero_object, libero_goal, libero_10 (max_steps 280/300/520).
- N=4 vs N=10 task-success sweep — now that actions are correct, the cosine
  sweep numbers (cos≈0.97 at N=4 vs N=10) actually mean something. Worth
  running both to see if N=4 holds up empirically.

### TTNN rollout

The same fix applies to the TTNN path. Rerun
`--backend ttnn` to confirm parity on Blackhole (~140 ms/chunk vs ~5.5 s for
pytorch CPU — should be much faster than wall-clock above).

### Reference setup that enabled the fix (one-time, not in git)

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
