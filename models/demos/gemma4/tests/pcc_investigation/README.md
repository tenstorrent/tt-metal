# Gemma-4-26B-A4B PCC investigation

Why Gemma-4-26B-A4B on QB2 (1x4) can't reach 0.99 PCC vs Hugging Face, and why other MoEs can.
Results: [FINDINGS.md](FINDINGS.md). Experiment scripts, run by hand from the repo root with `python_env`.

## Accuracy check: the accuracy gate

```
HF_HUB_OFFLINE=1 HF_MODEL=~/benchmark-data/gemma-4-26B-A4B-it HF_MODEL_ID=google/gemma-4-26B-A4B-it MESH_DEVICE=P150x4 \
  python_env/bin/python -m pytest models/demos/gemma4/tests/test_optimizer_gemma4_pcc.py::test_optimizer_gemma4_pcc -x -q -s
```

128 prefill + 128 decode tokens vs HF bf16; prints `ACCURACY ... top1_pct top5_pct mean_corr`.
The scripts below use other text and positions, so their numbers aren't comparable to the gate's.

Expected results (QB2, 2026-10-05):

| Code | Gate top-1 | Gate top-5 | Gate mean PCC | Decode ms/token |
|---|---|---|---|---|
| this branch (optimized + main 2026-10-05), switches off | 80.62% | 94.57% | 0.9637 | 23.70 |
| optimized, before merging main | 81.40% | 95.35% | 0.9655 | 24.15 |
| optimized + fp32 switches on | 83.72% | 97.67% | 0.9768 | 39.15 |
| unmodified port | 80.62% | 93.02% | 0.9610 | 54.44 |
| unmodified + fp32 switches on | 86.82% | 97.67% | 0.9770 | 70.37 |

`chip/tt_token_accuracy.py` (book text, 500 decode positions, vs HF bf16): this branch top-1 69.6%, top-5 88.6%,
mean PCC 0.9396; optimized before the merge 66.6% / 88.2% / 0.9376.

## Paths

| | Default | Env var |
|---|---|---|
| Data, logits, logs (7.9 GB) | `~/benchmark-data/gemma4-pcc/` | `GEMMA4_PCC_DATA` |
| Checkpoints | `~/benchmark-data/` | `GEMMA4_PCC_MODELS` |
| Repo root | from script location | `GEMMA4_PCC_REPO` |

`results/` has copies of the JSON outputs.

## Code versions

| Commit | |
|---|---|
| `b809f097c06` | unmodified port |
| `36b3922cc29` | optimized (Gio's tool, 21 keeps) |
| `87505f06886` | unmodified + fp32 switches |
| `ed77315d4c0` | optimized + fp32 switches (this branch's base) |

fp32 switches (off by default): `GEMMA4_FP32_ACTIVATIONS=1 GEMMA4_ROUTER_TOPK_ON_SCORES=1 GEMMA4_FP32_ATTENTION=1`.

## Scripts

**`reference/`**: Hugging Face on CPU
- `generate_reference_hf_with_logits.py`: book reference for `tt_token_accuracy.py` (bf16 `.refpt` + logits)
- `hf_fp32.py`, `hf_fp32_eager.py`: HF in fp32
- `hf_eager_save.py`: HF bf16, eager attention
- `hf_per_layer_ref.py`: per-layer fp32 reference for the per-layer PCC scripts
- `hf_gate_fp32_ref.py`: fp32 reference on the accuracy gate's tokens
- `cross_version.py`: transformers 5.18 vs 5.12.1 (`~/venvs/hf-5.18`)
- `chat_ids.py`, `chat_hf.py`, `hf_sanity.py`, `gemma_top_guesses.py`: sanity checks

**`chip/`**: QB2
- `run_merge_validation.sh <label>`: gate, token accuracy, per-layer PCC, perf
- `attn_sweep.py <label> [settings]`: attention precision/order sweep, model built once (`SWEEP_ISO=128` adds isolated per-layer error)
- `prefill_fp32_off.py <script|pytest> ...`: run with main's prefill fp32 norms switched off
- `tt_token_accuracy.py`: tt-metal standard protocol (512 prefill, 500 decode, book text)
- `tt_decode_only_accuracy.py`: same, decode only
- `tt_per_layer_pcc.py`, `tt_per_layer_pcc_decode.py`: per-layer / per-op PCC (prefill path / decode path)
- `run_per_layer_variants.sh`: per-layer PCC on all four code versions
- `fp32_op_check*.py`, `fp32_attn_ops_check.py`, `sdpa_decode_check.py`: single-op checks
- `fp32_layer_bisect.py`, `fp32_pos_bisect.py`: find where fp32 and bf16 runs diverge
- `tt_weight_dtypes.py`: on-device weight dtypes
- `test_full_model_qb2*.py`: unit full-model tests runnable on 1x4 (pytest)

**`hf_analysis/`**: why Gemma 4 is sensitive (CPU)
- `hf_self_agreement*.py`, `hf_expert_flips*.py`: HF sdpa vs eager, logits and expert choices
- `router_gaps.py`: 8th vs 9th expert score gap
- `gemma_precision_ablation.py`: bf16 rounding at one place at a time
- `compare_vs_fp32.py`: every saved run vs HF fp32
- `hf_bf16_per_layer.py`: HF bf16 per-layer baseline

**`cross_model/`**: Gemma 4 vs Qwen3-30B-A3B vs OLMoE (args: `<model_dir> <gemma4|qwen3_moe|olmoe>`)
- `cross_model_perturb*.py`: nudges, expert swaps, frozen routing
- `freeze_components.py`: nudge with routing / attention frozen
- `probe_layer_gain.py`: error growth inside one layer
- `massive_channels.py`, `attention_spread.py`, `moe_internals.py`: residual and attention statistics

**`google/`**: Google's JAX Gemma 4 (`~/venvs/gemma-jax`)
- `google_gemma4_*.py`: reference runs
- `google_ckpt_to_bf16_npy.py`: checkpoint conversion
- `google_noise_test*.py`, `google_sanity.py`, `google_load_probe.py`, `chat_google.py`: checks
