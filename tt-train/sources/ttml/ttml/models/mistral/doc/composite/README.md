# Composite bring-up: Mistral-Small-24B-Base-2501 LoRA on one Blackhole chip

Phase 2 of the tt-train bring-up (`../plan.md` is the contract). Everything here was produced on 2026-10-29 on chip 0 of the QB2 box (`e0118d75da98`), branch `agobeljicTT/mistral-bringup`; commands and artefact paths are in `work_log.md`, raw evidence under `evidence/`.

## What was built

- `ttml/models/llama`: `head_dim` decoupled from `hidden_size // num_heads` (Mistral has 32 x 128 = 4096 ≠ 5120); the safetensors loader takes the model's root name and follows `model.safetensors.index.json`; `Llama._embed` hook.
- `ttml/models/mistral`: `MistralConfig.from_hf` (model_type `mistral`, `sliding_window` must be null), `Mistral` (the Llama stack; the frozen embedding table is untilized once at first use instead of every step), `load_from_safetensors`, `useful_flops_per_step` (the plan's MFU numerator).
- `ops/linear_op.cpp`, `ops/binary_ops.cpp`: backward skips the gradient of an operand with `requires_grad=false` (port of `umales/qwen38`), so no frozen base weight, LM head or embedding gets a dW matmul or an untilize; RMSNorm already skipped `dgamma`. The freeze map of the plan is therefore honoured: the only gradients produced are the 8 adapter tensors per block.
- `sources/examples/mistral_train/train.py` (LoRA steps at a chosen depth, loss/data logs, `--mem-probe`), `torch_reference.py` (fp64 CPU replay of the same windows and LoRA init), `analyze_tracy.py` (depth fit, projection, ranked table).
- `tests/python/test_mistral.py`: the PCC harness.

Layout contract: activations cross module boundaries as `[B, 1, T, W]` bf16 TILE with heads merged on the last dim. Known layout cost: attention puts the head axis on dim 1 (`[B, H, T, D]`, no 32-row padding) via `nlp_create_qkv_heads` / `nlp_concat_heads` and their backward (transposes + concat): 0.29 ms per layer, 0.7 % of the device time (`../perf_analysis.md`). No reshape/permute puts a head axis on dim -2. LoRA rank 16 is padded to a 32-wide tile by the matmuls (the `[16]` in the Tracy shapes).

## PCC harness (real shapes, torch fp64 on bf16-rounded inputs)

`pytest tests/python/test_mistral.py` (`evidence/pcc_tables.txt` from the first runs; rerun at HEAD `c077a127726` after the loader fix: `evidence/pcc_harness_head.log`, 11 passed, same tables; `tests/python/test_llama.py` + loader tests `evidence/test_llama_head.log`, 131 passed; bars: forward PCC ≥ 0.999 and rel. L2 error ≤ 2e-2, gradients ≥ 0.99 / ≤ 5e-2, gradients through the 1024x1024 softmax ≥ 0.98 / ≤ 8e-2). Shapes: hidden 5120, 32 q / 8 kv heads x 128, intermediate 32768, vocab 131072, seq 1024, batch 1. All 11 tests pass.

| module | tensor | PCC | rel err |
|---|---|---|---|
| RMSNorm | out / dx / dgamma | 0.999998 / 0.999996 / 0.999994 | 2.4e-3 / 4.4e-3 / 4.7e-3 |
| RMSNorm, frozen gamma | dgamma | not produced (checked) | |
| attention (fused qkv, heads, RoPE, causal GQA SDPA, concat, out-proj) | out / dx / d w_qkv / d w_out | 0.999989 / 0.999984 / 0.999984 / 0.999989 | 4.8e-3 / 6.0e-3 / 6.0e-3 / 4.8e-3 |
| MLP (gate_up, swiglu_packed, down) | out / dx / d w_gate_up / d w2 | 0.999992 / 0.999990 / 0.999990 / 0.999992 | 4.0e-3 / 4.5e-3 / 4.5e-3 / 4.0e-3 |
| LoRA linear at the qkv shape, frozen base | out / dx / dA / dB; dW | 0.999997 x4; not produced | 2.4e-3 x4 |
| decoder block under LoRA (4 adapters) | out / dx | 0.999991 / 0.999981 | 4.6e-3 / 7.1e-3 |
| | dA, dB of qkv / out / gate_up / w2 | 0.99995 / 0.99996 / 0.99998 / 0.99998 (each pair) | 1.5e-2 / 1.2e-2 / 8.5e-3 / 8.1e-3 |
| | every base weight and gamma | no gradient (checked) | |
| frozen embedding (no-untilize path) | out vs ttml op, vs table rows | 1.0 | 0 |
| LM head (frozen) + cross-entropy | logits / loss / dx; dW | 0.999999 / - / 0.999999; not produced | 1.7e-3 / 8.6e-4 / 4.2e-3 |
| real checkpoint, 2 of 40 layers, vs `MistralForCausalLM` fp64 on Shakespeare tokens | logits | 0.999976 | 5.8e-3 |

No tensor passes PCC with a biased relative error: every relative error is at the bf16 level (3-7e-3) except the adapter gradients through the attention block (1.2-1.5e-2, still 3-5x inside the bar). The torch RoPE in the harness uses ttml's interleaved-pair convention so random weights are shared without the HF row permutation; the permutation and the fused-weight assembly are what the checkpoint row verifies.

## Loss trajectory vs torch (8 of 40 layers, real weights)

`train.py --layers 8 --steps 10` on Shakespeare windows (inputs `ids[0:1024]`, targets `ids[1:1025]`, mean over 1024 positions), then `torch_reference.py` on the same windows from the same LoRA init with the same AdamW (fp64, CPU). `evidence/loss_comparison.txt`:

| step | ttml | torch fp64 | diff |
|---|---|---|---|
| 1 | 12.1250 | 12.1058 | +0.019 |
| 2 | 11.9375 | 11.8370 | +0.101 |
| 3 | 11.3125 | 11.2683 | +0.044 |
| 4 | 10.4375 | 10.3571 | +0.080 |
| 5 | 9.7500 | 9.6544 | +0.096 |
| 6 | 9.8750 | 10.1088 | -0.234 |
| 7 | 9.2500 | 9.2275 | +0.023 |
| 8 | 8.6875 | 8.6786 | +0.009 |
| 9 | 8.9375 | 8.9936 | -0.056 |
| 10 | 8.5625 | 8.6285 | -0.066 |

First comparison (review round 1 found it insufficient): the device loss is a bf16 value (the CE kernel returns the logits' dtype, 1 ulp = 0.0625 here), the diffs above are +0.3, +1.6, +0.7, +1.3, +1.5, **-3.7**, +0.4, +0.1, -0.9, -1.1 ulps, and a tolerance of 0.25 set after the fact just contained the step-6 outlier. Resolved as follows.

**Tolerance, stated before the rerun.** Two sources separate: (1) the forward: bf16 matmuls with fp32 accumulation against fp64; measured at step 1 (identical parameters on both sides) as 0.019 at loss 12.1, 0.16 %; (2) the optimiser: ttml stores the adapters and the Adam moments in bf16 without stochastic rounding, and a step-1 Adam update is ~lr = 1e-4 per element against adapter ulps of 3e-5-6e-5 (|A| in 0.004-0.016), so up to a third of every update is lost to rounding, which no fp64 reference with fp64 parameters reproduces. The control is `torch_reference.py --bf16-params`: the same fp64 math with A, B and both moments rounded to bf16 after every step, exactly as ttml holds them. The ttml value compared is the **host fp32 loss** (`train.py --host-loss`: the device logits read back and the mean cross-entropy taken on the host in fp32), so the comparison is not quantised. Bar: **|ttml host loss - torch bf16-params loss| ≤ 0.10 at every step** (five times the measured forward-only deviation at step 1, 1 % of the loss), no sign bias (both signs present), and the two ttml runs identical. The unrounded-fp64 trajectory is reported next to it, not bounded. Script: `sources/examples/mistral_train/compare_loss.py` (asserts the bar; `evidence/loss_comparison_round2.txt`).

**Result** (`evidence/loss_comparison_round2.txt`; ttml rerun `evidence/train_8layers_hostloss.log`, device losses bitwise identical to both earlier runs):

| step | ttml device (bf16) | ttml host fp32 | torch fp64 | torch fp64, bf16 params | ttml host - bf16-params |
|---|---|---|---|---|---|
| 1 | 12.1250 | 12.1094 | 12.1058 | 12.1058 | +0.0036 |
| 2 | 11.9375 | 11.9118 | 11.8370 | 11.8364 | +0.0754 |
| 3 | 11.3125 | 11.3439 | 11.2683 | 11.2890 | +0.0549 |
| 4 | 10.4375 | 10.4290 | 10.3571 | 10.4068 | +0.0222 |
| 5 | 9.7500 | 9.7261 | 9.6544 | 9.7131 | +0.0130 |
| 6 | 9.8750 | 9.8665 | 10.1088 | 9.8938 | -0.0272 |
| 7 | 9.2500 | 9.2296 | 9.2275 | 9.2255 | +0.0041 |
| 8 | 8.6875 | 8.6797 | 8.6786 | 8.6779 | +0.0018 |
| 9 | 8.9375 | 8.9637 | 8.9936 | 8.9663 | -0.0026 |
| 10 | 8.5625 | 8.5903 | 8.6285 | 8.5896 | +0.0007 |

Max |diff| **0.075** (bar 0.10), mean +0.015, signs +8 / -2; PASS. The bf16-parameter replay reproduces the step-6 value (9.894 vs ttml 9.867, where the unrounded reference gives 10.109) and the +1-1.6 ulp drift of steps 2-5: the whole earlier discrepancy was bf16 storage of the adapters and moments (replay vs unrounded torch: max 0.215, mean -0.016), as the plan's §4 lever predicted; the forward itself matches to 0.004 at step 1. Remaining mean offset +0.015 (0.15 % of the loss) with 8 of 10 steps positive: bf16 compute in the forward/backward (the fp64 replay rounds only the parameters), at the level of the single-block harness relative errors; recorded, inside the bar. Anomaly ledger entry (step 6): **fixed / explained with numbers**. Loop candidates from this: `stochastic_rounding=True` or `AdamWFullPrecision` for the adapters (correctness lever, not perf), and an fp32 loss output.


Determinism: the run repeated with the same seed gives all 10 losses bitwise identical (`evidence/loss_8layers_run2.json`).

## Memory, watcher, perf

- Allocator (`--mem-probe`, `evidence/mem_probe_*layers.log`): weights + LoRA state 5.14 / 7.27 / 11.54 GiB and allocated after the forward 6.43 / 10.08 / 17.39 GiB at 2 / 4 / 8 layers; retained activations 778 MiB per layer (plan: 786). These are point reads after the forward, not the peak: the first backward transients (logits gradient 256 MiB, fp32 CE intermediates ~512 MiB, gate_up output gradient 128 MiB) land while every activation is still alive, so the 8-layer peak is ~18.5 GiB of 31.83; 16 layers would need 32.2 GiB before those transients, so 8 is the deepest single-chip depth (plan §4 calibration).
- Watcher-clean run: `TT_METAL_WATCHER=1`, 1 layer, 2 steps: completes, 0 errors in the 113211-line watcher log (`evidence/watcher_1layer_watcher_excerpt.txt`).
- Baseline: 434.0 ms/step, 33.32 % MFU at 8 of 40 layers (no profiler); Tracy at 1/2/4/8 layers fits `48.8 + 40.2 L` ms device with residuals ≤ 0.6 % and projects to 1659 ms device / 1947 ms wall at 40 layers (33.3 % MFU). Full tables, ranked costs and the gap to the plan target in `../perf_analysis.md`.

## Limitations

- The full 40-layer model cannot run on one chip (43.9 GiB of bf16 weights); every full-depth number is a projection from 1/2/4/8 layers, and the loss trajectory is at 8 of 40 layers against an 8-layer reference. The plan records a TP=4 run across the box as a user decision; it was not done.
- The loss value is bf16 (0.0625 quantum); the comparison above is bounded by that.
- Chip clock: the chip delivers 1.13-1.19 GHz under this load, not the nominal 1350 MHz. Raw Tracy cycle counters over host time give 1187 / 1171 / 1146 / 1130 MHz at 1 / 2 / 4 / 8 layers and 1349-1350 MHz for the idle-dominated first step (`evidence/clock_analysis.txt`); `tt-smi -s` during a 60-step 8-layer run (`evidence/clock_trace_8layers_tt-smi.txt`; each `tt-smi -s` call takes 1-3 s, so 75 samples over the run): every sample taken while the chip draws load power (109-166 W, 76-78 C) reads **1125-1162 MHz**, every sample at 1350 MHz is at < 100 W (device open, compiling or between runs), and 800 MHz is the idle clock after close; the run itself averaged 443.8 ms/step (warm chip, `evidence/clock_trace_8layers.log`), and the earlier 10 s-cadence probe (`evidence/aiclk_probe_4layers_tt-smi.txt`) had its 1350 MHz samples at 71 W, i.e. device open but idle, and 1162 MHz at 168 W under load. Tracy converts cycles at 1350 MHz, so every Tracy ms in `../perf_analysis.md` is cycle-time; wall = cycle-time x 1.16-1.19. The 148.5 TFLOP/s MFU denominator is kept as the team convention; `../plan.md` §3 calibration states the target at the delivered clock.
- Device time is read from Tracy at two steady steps per depth; the non-profiled wall-clock has 1-2 % run-to-run noise (434 vs 453 ms warm).
