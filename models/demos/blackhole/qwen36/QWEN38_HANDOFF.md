# Qwen3.6 / Qwen3.8-27B on QB2: handoff (2026-10-09)

Branch `atupe/qwen38-optimizations`. Model code: `models/demos/blackhole/qwen36` (the "qwen36" implementation, TP=4 on a 1×4 Blackhole mesh, P300x2).

The work imports optimizations from the faster reference implementation `models/demos/qwen38_27b_qb2` into `qwen36`.

| Document | What it holds |
|---|---|
| `QWEN38_OPTIMIZATION_PLAN.md` | The plan |
| `QWEN38_OPTIMIZATION_LOG.md` | Every change, measurement and decision, in full detail |
| This file | Status, what remains, and how to continue |
| `qwen38_handoff/` | Tools, notes and an untested patch |

| Commit | Content |
|---|---|
| `eac0c9b30aa` | Tier 1: single-user decode/prefill optimizations 1.1-1.4 and 1.6 |
| `755a0d6461d` | Section B: Tier 1 extended to batch > 1 and to vLLM serving (tt-inference-server) |
| this commit | Tier 2 precision flags (all opt-in), Qwen3.8 accuracy investigation and knobs, the Qwen3.8 reference file, accuracy_1536, handoff tools |

---

## 1. Status in one paragraph

Tier 1 and section B are done, measured and committed. They are default-on and accuracy-neutral on Qwen3.6:
- single-user decode 27.7 → 32.8 t/s/u;
- b8 22.2 → 24.4 t/s/u;
- serving TPOT −12 to −19%, TTFT −35 to −66%, and output tok/s +20 to +65% vs main.

Tier 2 (selective BFP4 weights) is implemented behind flags, all OFF by default in this commit. Tier 2 is "accuracy-neutral" on Qwen3.6, but on Qwen3.8 the precision picture changed. Investigation showed that the pre-existing BFP4 MLP gate/up weights (with LoFi MLP math) in the base implementation cost Qwen3.8 about 7 points of teacher-forced top-1 (92% vs 99%). That is not a bug in our code; Qwen3.8 is about 7x more precision-sensitive than 3.6.

**Open decision:** which precision config to ship, decided by task-level evals (GPQA-Diamond, MMLU-Pro, AIME26) against Qwen3.8's published GPU numbers. Those evals were not run on this machine. They are the next step; see §5.

---

## 2. What was done

### Tier 1 (committed in eac0c9b30aa, all default on; flag = 0 restores the old path)

| # | Change | Flag | Effect (Qwen3.6, b=1) |
|---|---|---|---|
| 1.1 | Replicated residual + `all_reduce_async` (2 CCL/layer instead of 4) | `QWEN36_DECODE_ALLREDUCE` | +3.7% decode |
| 1.3 | Lean attention decode (fused K+V update, RoPE without transposes, fused sigmoid gate) | `QWEN36_ATTN_DECODE_LEAN` | +2.1% |
| 1.2 | GDN fused decode (KDA conv + chunk_gated_delta_rule + sigmoid_gated_rms_norm) | `QWEN36_GDN_FUSED_DECODE` | +10.2% (largest) |
| 1.4 | Traced short-prompt prefill + on-device first-token argmax | `QWEN36_PREFILL_BUCKET_TRACE` | TTFT 0.13 → 0.08 s |
| 1.6 | Device-resident greedy decode loop | `QWEN36_DECODE_DEVICE_LOOP` (0 host, 1 device + fallback, 2 strict) | +1.3% |
| 1.5 | DRAM-sharded QKV at BFP8 | — | REVERTED: no end-to-end gain |

- Decode: 27.74 → 32.85 t/s/u (+18.4%).
- Accuracy_512: 98.44 → 98.63 top-1.

### Section B (committed in 755a0d6461d): make the wins visible for b > 1 and in serving

| # | Change | Flag |
|---|---|---|
| 1.2B v3 | Batched fused GDN decode for widths ≤ 9 (`QWEN36_GDN_FUSED_SCAN_MAXB`); wider widths use the original path, with an in-place conv-format sync | `QWEN36_GDN_FUSED_DECODE` |
| 1.4B | Traced prefill for any prompt length < 2048 (persistent GDN masks, trash block, device slot write); used by the demo and by vLLM single and batched prefill | `QWEN36_PREFILL_BUCKET_TRACE` |
| 1.6S | Device-resident serving decode (`decode_input_update_contract=1`, in-place token feedback, RoPE lookup and position +1 in-trace); declares `supports_async_decode` | `QWEN36_SERVE_DEVICE_DECODE` |
| 1.6S-r | Fast GDN slot remap (run-based gather, valid conv format only): about 290 → 25 ms per vLLM batch condense, bit-identical | none |
| 1.6B | Batched demo device decode loop | `QWEN36_DECODE_DEVICE_LOOP` |

Results:
- Demo b8: 23.00 → 24.41 t/s/user.
- Demo b32: 13.39 → 13.67.
- Arbitrary-length TTFT (ISL 300 / 1000): −29% / −25%.
- Serving: see §6.3.
- vLLM async scheduling is ON by default (user decision). To trade a little TPOT for flat TTFT at concurrency, launch tt-inference-server with `--vllm-override-args '{"no-async-scheduling": true}'`.

### Tier 2 (this commit): selective BFP4 and precision knobs, ALL OPT-IN

| # | Change | Flag (default) | Measured on Qwen3.6 (b=1, cumulative on 2.1) | On Qwen3.8 |
|---|---|---|---|---|
| 2.1 | MLP down weights BFP4 (dense only; MoE shared expert unchanged) | `QWEN36_BFP4_MLP_DOWN` (0) | 32.84 → 34.29 t/s/u (+4.4%), acc 98.63/100 (=) | costs accuracy (see §3) |
| 2.2 | GDN in-proj BFP4 + LoFi decode | `QWEN36_BFP4_GDN_IN` (0) | +3.6% (35.54), acc 97.66/99.80 (−0.97); B32 GDN PCC 0.935 | costs accuracy |
| 2.3 | Attention QKV and wo BFP4 + LoFi decode | `QWEN36_BFP4_ATTN` (0) | +2.0% (34.98), acc 98.44/99.80 (−1 token) | costs accuracy |
| — | MLP gate/up BFP8 (base is BFP4) | `QWEN36_MLP_GATEUP_BFP8` (0) | — | fixes most of the 3.8 gap, −10% decode |
| — | MLP math fidelity (lofi \| hifi2 \| hifi3 \| hifi4) | `QWEN36_MLP_FIDELITY` (lofi) | no effect with BFP4 weights | hifi2 needed with BFP8 gate/up (+1 pt, free) |
| — | LM head compute config | `QWEN36_LMHEAD_CFG` (default \| hifi2_fp32 \| hifi4_fp32) | — | no measurable effect |

- 2.4, DRAM-sharded multi-reader decode matmuls for the BFP4 groups (expected +1-2%), is implemented but UNTESTED and not applied: `qwen38_handoff/patches/2_4_dram_sharded_UNTESTED.patch`, with its README.
- The acceptance rule set by the user for Tier 2: a lower precision is OK where task-eval accuracy is at least the GPU implementation's numbers, within run-to-run noise.

---

## 3. Qwen3.8 accuracy findings (read this before changing precision)

Weights live in `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, which is also used by `qwen38_27b_qb2`. Compared with Qwen3.6:
- Same architecture and tensor names. config.json differs only in transformers_version.
- The chat template adds `reasoning_effort`, default xhigh.
- The weights are heavily retrained (30-80% relative difference) with similar value ranges.

**Quick check.** `accuracy_512` and `accuracy_1536` are teacher-forced top-1/top-5 against an HF CPU BF16 reference.
- The new reference is `models/tt_transformers/tests/reference_outputs/Qwen3.8-27B.refpt`: 2048 tokens, one forward pass.
- The tool that made it reproduces the committed 3.6 refpt 100%.
- `accuracy_1536` is new: prefill 512, then score 1536.
- `model_targets.yaml` maps Qwen3.8 to the 3.6 thresholds (97 / 99).

| Qwen3.8 config | decode t/s/u | acc512 top-1/top-5 | acc1536 top-1/top-5 |
|---|---|---|---|
| Base (gate/up BFP4, rest BFP8, MLP LoFi) = **this commit's defaults** | 32.84 | 91.99 / 99.41 | 85.42 / 98.11 |
| A: base + 2.1 + 2.3 (fastest) | 34.98 | 88.48 / 97.85 | 81.18 / 95.96 |
| C: gate/up BFP8 + 2.1 + 2.3 + MLP HiFi2 | 31.49 | 93.75 / 99.61 | 88.02 / 98.83 |
| D: gate/up BFP8 + MLP HiFi2 (most accurate) | 29.64 | 97.66 / 99.80 | 95.31 / 99.93 |

For comparison, Qwen3.6 on the same code scores 98.63 / 100 on acc512.

**Root cause (proved, see LOG "Qwen3.8 accuracy gap: root cause"):**
- **The device math is correct.** A bit-exact torch emulation of TT BFP8/BFP4 weight storage was validated against ttnn (max diff 0) and applied to the HF model. It reproduces the device's 3.8 score (91.8% vs 92.0%, with the same error positions) and its 3.6 score (98.8% vs 98.6%).
- **The gap comes from BFP4 gate/up weights.** With gate/up in BFP8, the emulated model gets 99.2%.
- **On device, BFP8 gate/up also needs HiFi2 MLP math.** LoFi multiplies only 5 bits of the weight operand.
- **The remaining ~1.5 pt between device D and the emulation is op-level rounding.** Qwen3.8 amplifies it: HF-3.8 itself moves 4 tokens between fp32 and bf16. It was not pursued further.
- **Ruled out:** our optimization flags (all-off gives 89.8%), trace, the decode path (prefill-only shows the same errors), cache contamination, activation outliers, the LM head accumulation, and reference noise.

**CPU emulation results (Qwen3.8):**
- Top-1 vs the HF bf16 reference, with gate/up BFP8 unless noted. Weight bytes are per token per device.

| Config | top-1 | bytes/token/device |
|---|---|---|
| all BFP8 | 99.22 | 6.81 GB |
| + lm_head BFP4 | 98.63 | |
| + GDN out BFP4 | 96.88 | |
| + MLP down BFP4 | 95.90 | |
| + attention BFP4 | 95.70 | |
| + GDN in BFP4 | 95.12 | |
| + down + attention BFP4 | 94.14 | 5.89 GB |
| gate-only BFP4 | 94.14 | |
| TT base (gate/up BFP4) | 91.80 | 5.38 GB |
| TT base + Tier 2 (2.1 + 2.3) | 89.26 | 4.46 GB |
| TT base + Tier 2 + 2.2 | 87.70 | |

- Use the emulator (`qwen38_handoff/emulation/`) as a device-free screen before any weight-precision change.

**What the quick check cannot tell you:** whether 88% vs 98% token agreement changes task scores. Only the task evals in §5 decide that.

---

## 4. Remaining optimizations (prioritized)

1. **Decide the precision config with task evals** (§5). Order agreed with the user: evaluate A (fastest) first; if it passes against the GPU numbers within noise, adopt it, otherwise evaluate D (then C).
   - **Cheap add-ons the emulator flags as nearly free:** LM head BFP4 (−0.6 pt emulated, saves 0.14 GB/token/device). It needs a dtype flag; not implemented.
   - **Tier-3 candidate:** GDN out-proj BFP4 (−2.3 pt emulated).
2. **2.4 DRAM-sharded multi-reader decode matmuls** for whichever groups end up BFP4. The patch is ready; test plan is in its README.
   - Risks: L1 clashes for the down and wo shapes, an extra ~6.7 GB of cache.
   - Expected gain is about 0.3-0.6 ms/token, and only at BFP4.
3. **Faster wide-batch decode:** the fused GDN path covers only decode widths ≤ 9 (B=32 runs at 13.67 t/s/user on the original path). It needs a kernel that scales past 110 cores.
4. **Faster batched prefill:** about 3.6 s for 31 users in serving R4.
5. **MTP speculative decoding** with the model's draft head: the largest potential (1.5-2x decode) and the largest effort.
6. **Serving plugin (outside tt-metal, vllm-tt-plugin):**
   - Publish a prefill step's tokens before submitting the next step (vLLM core.py `step_with_batch_queue` ~674). This removes the remaining async TTFT penalty (+16% at c8).
   - Add a separate `supports_resident_decode` capability, so resident decode can run under sync scheduling without declaring async.
7. **Optional:** the remaining ~1.5 pt device-vs-emulation gap on 3.8 (op numerics: GDN kernels, SDPA, norms).

---

## 5. Tests and evals to run

### 5.1 Environment (every device run)
- `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0`. Without it, the warm weight-cache load hangs (tt-metal#55957).
- `HF_MODEL=<dir>/Qwen3.8-27B`. The basename must be exactly `Qwen3.8-27B`, because the refpt and accuracy targets are found by basename.
- `TT_CACHE_PATH` must be a dedicated dir per checkpoint. The cache path is not model-specific, so pointing 3.8 at a 3.6 cache silently loads 3.6 weights.
- `MESH_DEVICE=P150x4` is what the demo scripts use for the 1×4 mesh.
- Example env and measurement scripts: `qwen38_handoff/scripts/env38.sh` and `measure38.sh`. Adjust the absolute paths.
- Cache sizes for Qwen3.8 (BFP8 + BFP4 variants + gate/up BFP8): about 57 GB. The first load takes about 150 s.

### 5.2 Quick checks after every change (minutes each)

| What | Command (from the repo root) |
|---|---|
| Speed and text | `pytest models/demos/blackhole/qwen36/demo/text_demo.py -k "traced_128 and not traced_128k"` (×2), `-k traced_4k`, `-k batched_128_b8` |
| Accuracy (teacher-forced) | `-k accuracy_512` and `-k accuracy_1536` (Qwen3.8 refpt) |
| Unit tests | `tests/test_mlp_tp.py`, `tests/test_attention_tp.py`, `tests/test_gdn_tp.py`, `tests/test_model_tp.py -k "contract or decode_batched"`, `tests/test_prefill_trace_any_len.py`, `tests/test_decode_bucketing.py` (needs an `HF_HUB_CACHE` whose Qwen snapshot links the local weights; 6 tests load by hub id), `tests/test_serve_decode_host.py` |
| Serving contract | `tests/test_vllm_decode_contract.py` (reference dump with `QWEN36_SERVE_DEVICE_DECODE=0 QWEN36_CONTRACT_DUMP=ref.json -k host_reload`, then `QWEN36_CONTRACT_REF=ref.json -k "not async_tpot"`, then `-k async_tpot`) |
| Weight-precision screen (no device) | `qwen38_handoff/emulation/emul_run.py` + `analyze_emul.py` (bit-exact BFP emulation on the HF model) |

Known pre-existing failures:
- `tests/test_prefill.py`: `test_masked_bucket_matches_reference[len137_b256, len256_b256, len256_b512]` and `test_masked_bucket_after_trace_capture`, all an L1 circular-buffer clash. They fail identically before these changes.
- On Qwen3.8 the demo accuracy tests assert ≥ 96.5 top-1 and "fail" on configs below that. The numbers are still printed.

### 5.3 Task-level evals (the deciding step, about 8 h per config; NOT run yet)

**Datasets:** run `python qwen38_handoff/eval/prepare_datasets.py --out <data>` for MMLU-Pro and AIME26 (ungated). Add `--gpqa` with an `HF_TOKEN` that has accepted the GPQA terms. Never commit GPQA data. The MMLU-Pro 500-question subset is `qwen38_handoff/eval/mmlu_pro_500_ids.txt` (seeded stratified by category).

**Serve Qwen3.8 with our implementation through tt-inference-server.** The full recipe is in `qwen38_handoff/notes/q38_switch.md` §Q4 and `scripts/serve/c5_plan.md`.
- Use `--custom-weights Qwen/Qwen3.8-27B` plus `--vllm-override-args '{"model": "<dir>/Qwen3.8-27B"}'`, so the tokenizer and chat template come from 3.8.
- Give it a separate persistent volume whose `cache_*/P300x2` points at the 3.8 TT cache.
- Set the config's env flags in the launch env:

| Config | Env flags |
|---|---|
| A | `QWEN36_BFP4_MLP_DOWN=1 QWEN36_BFP4_ATTN=1` |
| C | A + `QWEN36_MLP_GATEUP_BFP8=1 QWEN36_MLP_FIDELITY=hifi2` |
| D | `QWEN36_MLP_GATEUP_BFP8=1 QWEN36_MLP_FIDELITY=hifi2` |

- Verify the server log for: checkpoint dir, cache dir, and the precision INFO lines ("MLP down weights", "MLP gate/up weights", "Attention QKV/wo weights", "GDN in-proj weights", "MLP fidelity", "LM head compute config").

**Run.** The client is `qwen38_handoff/eval/task_eval.py`, an OpenAI-compatible client with concurrency, resume and JSONL output. Its defaults follow the card's GPU protocol: temperature 1.0, top_p 0.95, top_k 20, thinking on, template-default reasoning_effort xhigh.
```
python task_eval.py --task gpqa     --csv  <data>/gpqa_diamond.csv  --repeats 2 --concurrency 16 --max-tokens 32768 --out runs/A/gpqa.jsonl
python task_eval.py --task mmlupro  --data <data>/mmlu_pro_test.jsonl --ids mmlu_pro_500_ids.txt --concurrency 16 --out runs/A/mmlupro.jsonl
python task_eval.py --task aime     --data <data>/aime_2026.jsonl   --repeats 4 --concurrency 8 --max-tokens 65536 --out runs/A/aime.jsonl
python score.py runs/A/gpqa.r*.jsonl --ref 89.2     # mean, per-repeat, question-clustered bootstrap CI, truncation rate
```
- The client was validated offline against a fake server (extraction, resume, scoring). It has not been run against the real server.
- Watch the truncation rate (`finish_reason=length`). Mean thinking length on GPU is GPQA ~12.8k, MMLU-Pro ~3.7k and AIME26 ~15.7k tokens, so 32k truncates some AIME answers. Use 65k there and keep concurrency within KV capacity (`QWEN36_MAX_TOKENS_ALL_USERS` in the model spec, 525,312 tokens).

**GPU reference numbers to compare against** (sources in `qwen38_handoff/notes/gpu_reference.md`):

| Benchmark | Qwen3.8-27B official | Qwen3.8 independent GPU | Qwen3.6-27B official |
|---|---|---|---|
| GPQA-Diamond | 89.2 | 89.93 ± 0.70 (ThinkingCap card, base model, H200 vLLM, 4-32 seeds); 88.89 (Vals AI) | 87.8 (Unsloth BF16 88.13, FP8 86.87, NVFP4 86.3-86.9) |
| MMLU-Pro | — | 85.54 ± 0.63 (full 12,032); 84.34 (Vals AI) | 86.2 |
| AIME 2026 | — | 98.13 ± 0.74 | 94.1 |
| LiveCodeBench v6 | 90.3 | 91.14 ± 1.11 | 83.9 |
| IFBench | 79.5 | 79.75 ± 0.63 | 69.1 |
| HMMT Feb 26 | — | 95.83 ± 1.16 | 84.3 |

- The independent GPU protocol is temp 1.0, top_p 0.95, top_k 20, min_p 0, reasoning_effort xhigh, with very long caps (65k at Vals, about 254k at ThinkingCap).
- **Pass rule (agreed: "within noise"):** compute the mean over repeats with a question-clustered bootstrap 95% CI. PASS if the CI contains the GPU number, or the mean is above it. A real regression is the CI's upper bound below the GPU number.
- One GPQA run has about 2.3 pt SE. GPU FP8/NVFP4 builds themselves land 1.3-1.8 pt below BF16.

### 5.4 Serving benchmarks (after choosing a config)
- Scripts are in `qwen38_handoff/scripts/serve/`. bench.sh runs R1-R4, bench5.sh runs R5, correct.sh checks answers, and sum.py / sum5.py summarize.
- R1 128/128 c1 n8, R2 1024/128 c1 n8, R3 128/128 c8 n32, R4 128/128 c32 n64, R5 300/128 c1 n8.
- These were measured on Qwen3.6 (see §6.3). Re-baseline on Qwen3.8 with the chosen config.

---

## 6. Results so far

### 6.1 Task-level evals
None have been run. The GPQA/MMLU-Pro/AIME runs were cancelled on this machine (shared box, device contention). The tooling is ready (§5.3).

### 6.2 Quick-check (teacher-forced) accuracy
- **Qwen3.6:** Tier-1 final 98.63 / 100.00. With 2.1: 98.63 / 100. 2.1 + 2.2: 97.66 / 99.80. 2.1 + 2.3: 98.44 / 99.80.
- **Qwen3.8:** see §3. Additional rows:

| Config | acc512 top-1 |
|---|---|
| G1 gate/up BFP8 with MLP LoFi | 96.68 |
| H2 gate/up BFP8 with HiFi4 | 97.66 (same as HiFi2) |
| L0 base + LM head fp32-acc | 92.19 |
| L1 D + LM head fp32-acc HiFi2 | 97.46 |
| L2 D + LM head fp32-acc HiFi4 | 97.85 |
| F0 all optimization flags off (≈ main) | 89.84 |

### 6.3 Serving (Qwen3.6, tt-inference-server, max_num_seqs 32; TTFT ms / TPOT ms / output tok/s)

| Bench | C0 (≈ main) | C1 (Tier-1 commit) | C7 (section-B final, async on) | C7sync (async off) |
|---|---|---|---|---|
| R1 | 213.0 / 39.65 / 24.38 | 220.7 / 38.41 / 25.10 | 115.1 / 33.48 / 29.31 | 111.7 / 34.12 / 28.79 |
| R2 | 286.2 / 40.01 / 23.85 | 288.8 / 37.96 / 25.05 | 186.6 / 33.63 / 28.71 | 183.5 / 34.30 / 28.19 |
| R3 | 2186.8 / 55.74 / 110.52 | 1565.7 / 52.67 / 124.05 | 958.9 / 45.39 / 152.30 | 827.4 / 47.11 / 150.34 |
| R4 | 11144.8 / 85.56 / 186.08 | 7456.8 / 83.41 / 226.91 | 3783.2 / 75.03 / 307.65 | 3607.6 / 77.03 / 305.86 |

Answers are byte-identical from C0 to C7, and no request failed. Async TTFT analysis: `qwen38_handoff/notes/async_ttft_analysis.md`.

### 6.4 Demo (Qwen3.6, all defaults of this commit)
- traced_128 decode is about 32.8 t/s/u with TTFT 0.08 s.
- traced_4k: 32.6.
- b8: 24.4 t/s/user.
- b32: 13.7 t/s/user.
- accuracy_512: 98.63 / 100.

---

## 7. Gotchas learned the hard way
- **Shared machines:** check that nobody holds `/dev/tenstorrent/*` before device runs. A hung run can wedge the board (`tt-smi -r` needed). Never reset while someone else's job runs.
- **Trace warmup order:** traced prefill must not be captured before the eager decode warmup has allocated the decode trace inputs. That order hung `test_vllm_decode_contract.py`. Use the plugin's order: eager prefill, eager decode, then traced prefill and traced decode.
- **ttnn view aliasing:** `pad`, single-input `concat`, same-config `to_memory_config`, full-range `slice` and logical `reshape` can return the SAME buffer. Compare addresses before freeing.
- **Unit-test PCC thresholds:** they (0.92-0.97) are too loose to catch precision changes. Use accuracy_512/1536 and the emulator.
- **Model-level contract tests:** `test_model_tp_contract` compares two TT paths with the same weights, so it cannot see quantization error.
- **Async scheduling:** vLLM releases a prefill's first token only after the next step is submitted (TT prefill is synchronous), which is why async raises TTFT at concurrency.

---

## 8. File index (`qwen38_handoff/`)

| Path | Content |
|---|---|
| `eval/task_eval.py`, `gpqa_eval.py`, `score.py`, `compare.py`, `validate_*offline.py`, `prepare_datasets.py`, `mmlu_pro_500_ids.txt` | Task-eval client, scoring (bootstrap CI, McNemar), offline validation, dataset download |
| `emulation/bfp.py`, `val.py`, `emul_run.py`, `analyze_emul.py`, `an2.py`, `bytes.py`, `dtype_map.md`, `results*.txt`, `bytes.json` | Bit-exact TT BFP8/BFP4 weight emulation on the HF model; results |
| `reference/gen_ref_single_chunk.py`, `gen_ref_logp.py`, `analyze.py`, `analysis.txt`, `hf_fp32.py`, `act_stats.py`, `prefill_all_test.py`, `wdiff.py` | Reference refpt generation (single chunk), full-logit dumps, KL analysis, fp32 reference, activation stats, prefill-all-positions check, weight diff |
| `scripts/env38.sh`, `measure38.sh`, `gate2.sh`, `serve/*` | Run environment, demo measurement driver, device-free gate, serving bench scripts |
| `notes/recon.md`, `q38_switch.md`, `gpu_reference.md`, `eval_plan.md`, `async_ttft_analysis.md` | Tier-2 recon, switching to 3.8, GPU reference numbers and protocols, eval plan, async TTFT root cause |
| `patches/2_4_dram_sharded_UNTESTED.patch` + README | Change 2.4, ready to test |

The scripts contain absolute paths from the original machine (`/home/ttuser/atupe/...`). Adjust them, or pass `--data`, `--csv` and the env overrides.
