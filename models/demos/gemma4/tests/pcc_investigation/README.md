# Gemma-4-26B-A4B PCC investigation

Why the Tenstorrent Gemma-4-26B-A4B (QuietBox 2, 4× Blackhole p150, 1x4 mesh) can't reach 0.99 PCC against
Hugging Face, and why this happens with Gemma 4 but not with other MoE models (Qwen3-30B-A3B, OLMoE-1B-7B).
The results so far are in [FINDINGS.md](FINDINGS.md); every number there comes from a script in this folder.

These are experiment scripts, not CI tests. Each one is run by hand: `python <script> [args]` from the repo root,
with the repo's `python_env` unless noted.

## Where things live

| What | Where | Override |
|---|---|---|
| Scripts and small JSON results | this folder (`results/` holds copies of the JSON outputs) | |
| Saved logits, per-layer reference tensors, `.refpt` files (7.9 GB), logs | `~/benchmark-data/gemma4-pcc/` and `~/benchmark-data/gemma4-pcc/logs/` | `GEMMA4_PCC_DATA` |
| Model checkpoints: `gemma-4-26B-A4B-it`, `Qwen3-30B-A3B`, `OLMoE-1B-7B-0924`, Google's `gemma4-26b-a4b-it-google` (orbax) and its bf16 `.npy` copy | `~/benchmark-data/` | `GEMMA4_PCC_MODELS` |
| Repo root (for the book text `models/tt_transformers/tests/tale-of-two-cities.txt.bz2`) | found from the script's location | `GEMMA4_PCC_REPO` |

Every script reads its inputs from and writes its outputs to the data folder.

## Which code the chip scripts run

The chip scripts import `models.demos.gemma4` from whatever is checked out. The versions compared:

| Commit | What it is |
|---|---|
| `b809f097c06` | the unmodified port (the code before Gio's optimizer) |
| `36b3922cc29` | the optimized code (`jerrywangTT/qb2-gemma4-26b-scratch`, 21 optimizer keeps) |
| `87505f06886` | unmodified port + fp32 switches (`jerrywangTT/gemma4-fp32-activations`) |
| `ed77315d4c0` | optimized code + fp32 switches (`jerrywangTT/gemma4-fp32-activations-optimized`); this branch starts here |

The fp32 switches (`models/demos/gemma4/tt/fp32_mode.py`) are off by default. With them off, the code behaves exactly
as without them. Turn them on with `GEMMA4_FP32_ACTIVATIONS=1 GEMMA4_ROUTER_TOPK_ON_SCORES=1 GEMMA4_FP32_ATTENTION=1`.
They keep every intermediate value of the decode path in fp32; the weights stay bf16.

## Scripts

### `reference/`: Hugging Face reference runs (CPU)
| Script | What it does |
|---|---|
| `generate_reference_hf_with_logits.py` | tt-metal's standard `generate_reference_hf.py` plus saving the full logits. Writes `gemma-4-26B-A4B-it.refpt` (book tokens with BOS, HF top-5) and `.refpt.logits.pt`. The model loads in **bf16**: this is the reference the accuracy gate uses. |
| `hf_fp32.py`, `hf_fp32_eager.py` | Hugging Face entirely in fp32 (sdpa / eager attention) on the same 1023 tokens: the closest stand-in for exact arithmetic. |
| `hf_eager_save.py` | Hugging Face bf16 with eager attention instead of sdpa (same math, different rounding order). |
| `hf_per_layer_ref.py` | Hugging Face with fp32 arithmetic on 512 tokens. Saves, for every layer: its input, attention output, shared-MLP output, experts output, the router's 8 chosen experts and the layer output (`hf_per_layer_ref_512.pt`, used by the per-layer PCC scripts). |
| `cross_version.py` | Transformers 5.18 vs 5.12.1 on the same tokens (run with `~/venvs/hf-5.18`). |
| `chat_ids.py`, `chat_hf.py` | A chat-formatted prompt and answer, and Hugging Face's logits on it. |
| `hf_sanity.py`, `gemma_top_guesses.py` | Sanity checks: next-word guesses on a famous sentence; the most common top guesses on the book vs shuffled words. |

### `chip/`: runs on the QB2
| Script | What it does |
|---|---|
| `tt_token_accuracy.py` | tt-metal's standard token-accuracy protocol (simple_text_demo TokenAccuracy): prefill 512 book tokens, then 500 teacher-forced decode steps; top-1/top-5 vs HF and per-position PCC. |
| `tt_decode_only_accuracy.py` | The same, but every token goes through decode from position 0 (the path the fp32 switches cover). |
| `tt_per_layer_pcc.py` | Per-layer and per-operation PCC vs `hf_per_layer_ref_512.pt` through the prefill path: accumulated (normal run), isolated (each layer fed HF's exact input), and decode of positions 480-511. |
| `tt_per_layer_pcc_decode.py` | The same per-layer measurement through decode only (positions 0-511 one at a time, paged KV cache), accumulated and isolated. |
| `run_per_layer_variants.sh` | Runs both per-layer scripts on the four versions above, then switches back to the starting branch. |
| `fp32_op_check.py`, `fp32_op_check2.py`, `fp32_attn_ops_check.py`, `sdpa_decode_check.py` | Single-op checks of the ttnn ops on the decode path, bf16 vs fp32 activations, against torch fp32. `sdpa_decode_check.py` reproduces the `sdpa_decode` error with `fp32_dest_acc_en` once the position is ≥ 64. |
| `fp32_layer_bisect.py`, `fp32_pos_bisect.py` | Build the model with N layers and decode 16 tokens to find where fp32 and bf16 runs part. |
| `tt_weight_dtypes.py` | Prints the on-device dtype of key weights. |
| `test_full_model_qb2.py`, `test_full_model_qb2_bos.py` | `tests/unit/test_model.py::test_full_model{,_decode}` with the MoE `tp < 8` skip removed so they run on the 1x4 QB2; the `_bos` copy also prepends the start token. Run with pytest. |

### `hf_analysis/`: why Gemma 4 is sensitive, measured in Hugging Face (CPU)
| Script | What it does |
|---|---|
| `hf_self_agreement.py`, `hf_self_agreement_any.py` | Hugging Face bf16 vs itself with only the attention implementation changed (sdpa vs eager), for Gemma 4 or any model. |
| `hf_expert_flips.py`, `hf_expert_flips_any.py` | Whether the router's expert choices change between those two runs, by layer. |
| `router_gaps.py` | How close the 8th- and 9th-best experts are. |
| `gemma_precision_ablation.py` | fp32 Gemma 4 with bf16 rounding added at one place at a time (queries, scores, norms, router, ...). |
| `compare_vs_fp32.py` | Every saved run (HF bf16 sdpa/eager, HF 5.18, Google JAX, chip unmodified/optimized) against Hugging Face fp32. |
| `hf_bf16_per_layer.py` | Baseline for the chip per-layer numbers: Hugging Face's own bf16 run, per layer, accumulated and isolated. |

### `cross_model/`: the same experiments on Gemma 4, Qwen3-30B-A3B and OLMoE-1B-7B (CPU)
All take `<model_dir> <gemma4|qwen3_moe|olmoe>`.
| Script | What it does |
|---|---|
| `cross_model_perturb.py` | bf16 rounding of the embedding output or the router steps, everything else fp32: PCC, expert-flip rates by depth, router near-ties. |
| `cross_model_perturb2.py` … `cross_model_perturb6.py` | The same harness with other configs. 2: a 2⁻⁹ random nudge on the embedding output, plus a forced 8th↔9th expert swap in the first and in the middle layer. 3: nudge + swap in the first layer. 4: nudge, and nudge with routing frozen to the clean run. 5: nudge + swap in the middle layer. 6: nudge, nudge with frozen routing and middle-layer swap, plus the residual drift at every layer (used for the per-layer growth model). |
| `moe_internals.py` | Residual stream size per layer, update-to-residual ratio, attention score sizes. |
| `probe_layer_gain.py` | Runs one layer at a time on the clean input and on the input plus a 2⁻⁹ nudge (random direction, and the real direction the difference takes inside the model), routing frozen and free. Reports how much the relative difference grows at every point inside the layer. |
| `massive_channels.py` | How much of the residual stream's size sits in its 4 largest channels, and how the next norm weights them. |
| `freeze_components.py` | The 2⁻⁹ embedding nudge on the full model with the routing, the attention probabilities, or both frozen to the clean run (`FREEZE_SUBSETS=1`: attention frozen in only some layers). |
| `attention_spread.py` | Spread of the attention scores under the attention distribution, entropy, largest probability, per layer. |

### `google/`: Google DeepMind's own JAX Gemma 4 (run with `~/venvs/gemma-jax`)
| Script | What it does |
|---|---|
| `google_gemma4_reference.py`, `google_gemma4_highest.py`, `google_gemma4_attnfp32.py` | Google's implementation on the same 1023 tokens: default precision, `HIGHEST` matmul precision, fp32 attention. |
| `google_sanity.py`, `chat_google.py` | Tokenizer and weight checks against the HF checkpoint; the chat prompt. |
| `google_ckpt_to_bf16_npy.py` | Converts the 90 GB fp32 orbax checkpoint to bf16 `.npy` files a few entries at a time (a full load peaks near 100 GB). |
| `google_load_probe.py` | Measures load memory. |
| `google_noise_test.py`, `google_noise_test2.py` | The 2⁻⁹ embedding nudge on Google's implementation (test2 loads the `.npy` copy). |
