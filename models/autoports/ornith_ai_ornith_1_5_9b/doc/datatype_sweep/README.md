# Ornith-1.5-9B datatype sweep

Selected: **`bfp8_all_lofi`** — BFP8 projection weights, LoFi for every projection group, BFP8 KV cache and decode/large-prefill CCL, BF16 residuals and decode projection inputs. The complete [selected policy](selected_precision_config.json) is consumed by the normal model and generator constructors. No BFP4 group survives the full-model accuracy gate.

The selected candidate measures **114.540 traced teacher-forcing tokens/s/user**, with **106.419 ms TTFT** on AIME24 (161 prompt + 100 generated tokens). The default constructor reproduces **114.414 tokens/s/user** with identical accuracy ([default construction](selected_default.json)). AIME24 prefill and decode top-1/top-5/top-100 are **97% / 100% / 100%**. Natural-text decode is **91.0156% / 99.8047% / 100%**; natural-text prefill is **90.2344% / 100% / 100%** over 512 positions.

**Post-selection token-out headline: 116.141 tokens/s/user without per-token readback, and 88.885 ms TTFT**, using the original optimized-full-model benchmark through default construction ([selected token-out](token_out_selected.json)). This serial TP4 workload has batch 1, context 2,048, 128 input and 128 output tokens, three repetitions, device greedy sampling and warmed traces. Resident decode times 126 steps after initialization; TTFT comes from the same harness's warmed prefill/first-token path. The separate path with per-token readback measures 115.756 tokens/s/user. **Later reports and serving comparisons should use the resident token-out number, not teacher-forcing throughput.** No vLLM integration was performed.

Acceptance thresholds are unchanged: **top-1 ≥90%, top-5 ≥98%, top-100 =100% for both prefill and traced decode on each reference**. P1 passes over 10,518 answer positions: ΔNLL **+0.00174 nats/token**, KL×1000 **2.64908**, versus bounds 0.004 and 5. Its traced-decode comparison covers 1,024 positions and passes. The highest-precision fitting reference uses BF16 weights and HiFi4 projections; its direct HF KL×1000 is **0.54**, below 4. [P1 evidence](p1_initial.json), [runner gate artifact](../eval_readiness/precision_reasoning.json).

## Measurement and selection

Hardware is four Blackhole chips, IDs 0–3, mesh 1×4, TP4, FABRIC_1D. Every hardware job was serialized and bounded. Candidate runs use explicit `precision_config` build kwargs; the selected default run uses `{}`. All ranking samples assert exactly `generated_tokens - 1` model trace replays and zero captures. Each reference gets one warmup and two timed requests. Setup/capture costs are excluded from ranking.

Three serial BFP8 baseline measurements are 113.574848, 113.622205 and 113.146600 tokens/s/user: median **113.574848**, spread **0.475605**. The selected candidate exceeds the median by **0.965001**, exceeding that spread; the default reproduction also exceeds it. Differences inside the spread are ties, favoring the simpler established policy. The baseline point in the plots/table is the median of its three runs; raw per-run values and TTFT remain in the result ledger. The other candidates use their two-request warmed medians.

The same token-out benchmark independently confirms the direction: the baseline control gives **114.810 resident tokens/s/user** versus **116.141** selected, about **1.16% faster** ([baseline token-out](token_out_baseline.json)). Its shared TTFT measurement is 94.009 ms. These are separate from the teacher-forcing selection measurements and the prior optimized-full-model artifact.

The initial standard readiness refresh uses the pinned AIME24 chat-template reference with exactly 100 generated tokens. Baseline prefill is 99/100 top-1 and baseline decode is 98/100; top-5/top-100 are 100%. The two initial setup-inclusive readiness runs measure 101.13 and 100.75 traced tokens/s/user, with TTFT 6,620.12 and 6,527.13 ms. Those first-request numbers are not mixed with warmed ranking. See [baseline provenance](baseline_provenance.json) and `baseline_prefill.log`, `baseline_decode_1.log`, `baseline_decode_2.log`.

The second reference contains actual contiguous Tale of Two Cities tokens, not an HF-greedy continuation: prefix 128, continuation 512, tokenizer offset 4,096, original HF BF16 scores. [Natural reference provenance](natural_text_provenance.json). It is numerical continuation stress, not a chat-quality verdict. Binomial 95% half-widths are recorded per score: the selected AIME top-1 is ±3.34 percentage points; natural prefill/decode are ±2.57/±2.48 points. One-token differences are not treated as evidence of a quality improvement.

## Evaluated configurations

All rows below ran the complete 32-layer model on both references. Exact dtype groups, per-layer exceptions, compute fidelities, phase scores, TTFT, traced decode, commands, hardware, mesh, provenance and statuses are in [sweep_results.json](sweep_results.json) and [sweep_results.csv](sweep_results.csv). The performance column uses the AIME24 workload. Full-model gates also apply to the prefill scores, which are separate from chart coordinates.

| Configuration | Natural prefill top-1 | Natural decode top-1 | Decode top-5 minimum | Traced decode t/s/u | Result |
|---|---:|---:|---:|---:|---|
| [bfp8_mlp_lofi](configs/bfp8_mlp_lofi.json) | 91.41% | 91.60% | 100.00% | 113.575 | P1 passes; slower |
| [bfp8_all_hifi2](configs/bfp8_all_hifi2.json) | 91.41% | 92.19% | 100.00% | 113.089 | P1 passes; slower |
| [bfp8_all_lofi](configs/bfp8_all_lofi.json) | 90.23% | 91.02% | 99.80% | 114.540 | Selected |
| [bfp8_attention_lofi](configs/bfp8_attention_lofi.json) | 90.62% | 91.60% | 100.00% | 113.713 | P1 passes; slower |
| [bfp8_head_lofi](configs/bfp8_head_lofi.json) | 91.02% | 90.04% | 100.00% | 113.693 | Slower; P1 not run |
| [bfp8_kv_bf16](configs/bfp8_kv_bf16.json) | 90.62% | 91.60% | 100.00% | 113.238 | Slower; P1 not run |
| [bfp8_ccl_bf16](configs/bfp8_ccl_bf16.json) | 91.80% | 92.19% | 100.00% | 111.030 | Slower; P1 not run |
| [bfp8_decode_act_bfp8](configs/bfp8_decode_act_bfp8.json) | 91.41% | 90.82% | 100.00% | 108.285 | Slower; P1 not run |
| [bfp8_prefill_act_bf16](configs/bfp8_prefill_act_bf16.json) | 90.62% | 91.60% | 100.00% | 113.119 | Slower; P1 not run |
| [bfp4_mlp_lofi](configs/bfp4_mlp_lofi.json) | 80.27% | 81.05% | 98.05% | 122.310 | Fails full-model top-1 |
| [bfp4_mlp_hifi2](configs/bfp4_mlp_hifi2.json) | 80.47% | 81.05% | 98.44% | 120.062 | Fails full-model top-1 |
| [bfp4_gate_up_lofi](configs/bfp4_gate_up_lofi.json) | 81.45% | 82.62% | 99.22% | 120.278 | Fails full-model top-1 |
| [bfp4_gate_up_hifi2](configs/bfp4_gate_up_hifi2.json) | 82.42% | 83.40% | 99.22% | 118.220 | Fails full-model top-1 |
| [bfp4_down_lofi](configs/bfp4_down_lofi.json) | 85.16% | 86.91% | 99.61% | 115.148 | Fails full-model top-1 |
| [bfp4_attention_lofi](configs/bfp4_attention_lofi.json) | 80.27% | 80.08% | 98.83% | 118.724 | Fails full-model top-1 |
| [bfp4_head_lofi](configs/bfp4_head_lofi.json) | 87.11% | 88.67% | 100.00% | 117.099 | Fails full-model top-1 |
| [bf16_hifi4](configs/bf16_hifi4.json) | 95.90% | 93.95% | 100.00% | 78.272 | P1 passes; slower |
| [bfp4_attention_output_lofi](configs/bfp4_attention_output_lofi.json) | 88.67% | 89.06% | 99.80% | 114.408 | Fails full-model top-1 |
| [bfp4_attention_input_lofi](configs/bfp4_attention_input_lofi.json) | 80.66% | 82.23% | 98.83% | 117.156 | Fails full-model top-1 |

Every material BFP4 group has a LoFi trial. Inner-layer weight trials exclude layers 0 and 31; the LM head trial applies to the head. MLP and gate/up also have HiFi2 comparisons. Both attention groups were split after the combined candidate failed. Every BFP4 trial fails natural-text top-1 despite passing AIME24, so none proceeds as an eligible selected policy or uses a required-to-fit waiver.

| Material group | Runtime mapping | Evaluated dtype/fidelity policies | Decision |
|---|---|---|---|
| Attention input | Packed QKV and architecture-specific gates | BFP8 HiFi2/LoFi; BFP4 LoFi; BF16 HiFi4 | BFP8/LoFi selected; BFP4 split trial fails natural top-1 |
| Attention output | Full-attention output or linear-attention output | BFP8 HiFi2/LoFi; BFP4 LoFi; BF16 HiFi4 | BFP8/LoFi selected; BFP4 split trial fails natural top-1 |
| Gate/up | Packed SwiGLU prefill; separate gate/up decode | BFP8 LoFi/HiFi2; BFP4 LoFi/HiFi2; BF16 HiFi4 | BFP8/LoFi selected; both BFP4 fidelities fail |
| Down | Dense MLP down projection | BFP8 LoFi/HiFi2; BFP4 LoFi, plus combined MLP HiFi2; BF16 HiFi4 | BFP8/LoFi selected; BFP4 fails |
| LM head | Vocab-sharded prefill and DRAM-sharded decode head | BFP8 HiFi2/LoFi; BFP4 LoFi; BF16 HiFi4 | BFP8/LoFi selected; BFP4 fails |

The complete per-group performance and accuracy comparisons are the linked config rows above. BF16 KV/CCL controls and BFP8 decode-input activation were evaluated rather than inferred. BFP8 decode inputs are slower; BF16 CCL is slower. Four BFP8 projection-fidelity policies pass P1; the remaining full-model passing controls are all slower than the selected policy and do not need P1 to establish this selection.

## Pareto interpretation

![Top-1 Pareto frontier](top1_perf_pareto.png)

![Top-5 Pareto frontier](top5_perf_pareto.png)

Both pyplot charts plot every completed full-model configuration, connect the non-dominated evaluated frontier, mark the selected point red and draw the required vertical dotted threshold. The x coordinate is minimum teacher-forcing agreement across the two references; y is warmed traced AIME24 decode throughput. Gray crosses fail full-model accuracy, gray circles are slower controls without P1 scores, and green points pass the available P1/reference checks. The top-5 frontier alone cannot select a policy: BFP4 is fast and often retains top-5 agreement, but fails the simultaneous top-1 requirement. The chosen point is the fastest that satisfies both full-model gates and P1.

## Consumed precision contract

`build_generator` forwards a precision override to `OrnithModel`. With no override, `tt/precision.py` loads `doc/datatype_sweep/selected_precision_config.json`. The measured `selected_default.json` has empty build kwargs and records actual loaded tensor dtypes and compute objects. `token_out_selected.json` independently records the same policy through the normal builder. [Construction proof](precision_propagation.json) compares the shipped artifact with the measured winning and default paths.

| Field | Selected value and consumption |
|---|---|
| Weight groups / layer exceptions | All five groups BFP8; no layer exceptions. Loader quantizes each packed/separate projection and head accordingly. |
| Projection fidelities | All five groups LoFi, passed to per-group compute objects used by decode, ordinary/special prefill and packed SwiGLU. |
| Activation | Decode projection inputs BF16; normalized prefill projection inputs BFP8 from 512 physical rows, BF16 for short profiles. |
| Residual / logits / sampling | BF16 fixed kernel contracts; incompatible policy values are rejected. LM-head output dtype is read from the policy. |
| CCL | BFP8 persistent decode communication buffers and large-prefill transfer payloads; short prefill uses fixed BF16 communication. |
| KV cache | BFP8 allocation, fill/update conversion contracts, actual K/V storage validation and host byte accounting. |
| Other fixed assumptions | BF16 embeddings/norm weights, FP32 recurrent state and destination accumulation; decoder norm HiFi4, terminal norm and decode SDPA HiFi2. Unsupported changes are rejected, not silently accepted. |
| Context | Policy/checkpoint context 262,144; normal generator construction retains it. Smaller benchmark contexts are explicit workload settings. |

No field claims arbitrary configurability where the underlying kernel contract is fixed. The full-model builder and cache helpers provide the common construction contract for later serving work; there is no vLLM adapter in this stage and no serving validation claim. To return to the baseline, pass `kwargs/bfp8_mlp_lofi.json` through `--build-kwargs-file`.

## Capability, repair and validation

[Context contract](../context_contract.json) is recomputed for BFP8 KV, BF16 KV and the all-BF16 reference. Both KV policies and the reference pass full-model 262,144-token prefill, non-aligned 262,143-token prefill and terminal decode at position 262,143, with allocation tracking, live traces and no new programs. The selected policy has its own default-construction capacity artifact, `capacity_bfp8_all_lofi.json`. No advertised capability was reduced. Prepared free DRAM is 18.826613 GiB per device for both baseline and selected policy (delta 0 bytes); [runtime/capacity proof](precision_propagation.json). Batch 32 is validated at context 512; this does not claim 32 simultaneous maximum-context requests fit.

Short-prefix controls use 32 cached tokens and 256 real continuation tokens. BFP8 KV gives 92.58% top-1 at batch 1 and 92.19% for every batch-32 user; BF16 KV gives 92.97% at both batches. All top-5/top-100 scores are 100%; all 32 users agree for each same-input run. See `short_b1_*` and `short_b32_*` JSONs.

BF16 controls required two verified L1 repairs, documented in [AUTODEBUG.md](AUTODEBUG.md) and [AUTOFIX.md](AUTOFIX.md): large BF16 tensors stay in DRAM, and BF16 weights use K8 in the special 512–1,024-row matmul path. BFP8 geometry remains unchanged. Both repaired policies pass full-model accuracy; tracked non-aligned smokes cover 14 lengths through 32,769 with frozen program/capture counts. Initial failures remain preserved. A BF16 CCL same-dtype reshard was also corrected by using `to_memory_config`; its focused and full-model controls pass.

The selected resumed-prefill check uses the original optimized-full-model 513-token workload at the unchanged default thresholds, including both negative controls: [prefill_resume_check.json](prefill_resume_check.json). Selected prompt-based evidence and explicitly reused matched HF controls are under `qualitative/` and `qualitative_entropy_control/`; the verdict is [qualitative_verdict.json](qualitative_verdict.json). The P1 set uses four reasoning answers with the 6,144-token cap; it is not a broad downstream benchmark.

## Reproduction and artifact index

```bash
source python_env/bin/activate
source /home/ttuser/ornith-9b/lane-env.sh
```

Every bounded command, start/end timestamp, exit code and serial chip regime is recorded in [commands.jsonl](commands.jsonl). Candidate inputs are [candidate_matrix.json](candidate_matrix.json), `configs/*.json` and `kwargs/*.json`. Queue manifests retain the original matrix, repair, cache, P1 and selected-validation commands. The queue refuses to overwrite a log and observes a stop file between jobs.

- Full-model scores: each `<config_id>.json` and matching `.log`; default replay: `selected_default.json`.
- Selection and runtime proof: `selection.json`, `selected_precision_config.json`, `precision_propagation.json`.
- Final serving-comparable performance: `token_out_selected.json`; same-run baseline: `token_out_baseline.json`.
- P1: `candidates.json`, `reasoning_texts.jsonl`, `p1_initial.json`, `p1_final.json`, and `../eval_readiness/precision_reasoning.json`. Rebuildable score tensors remain local with a hash manifest.
- Context and trace safety: `capacity_*.json`, `placement_smoke_*.json`, `short_b*.json`, `prefill_resume_check.json`.
- Readiness and review: `stage_gate.log`, `stage_review_round1.md`; exact final verdict and local commit SHAs are in [work_log.md](work_log.md).
- All 68 raw logs are preserved locally and as lossless `.log.gz` archives because repository rules ignore `.log`; `log_archives.json` lists paths and SHA-256 hashes. The large AutoDebug log also appears in `large_artifacts.json`.

Limitations: accuracy is measured on the stated finite references and four reasoning texts, not a downstream benchmark guarantee. The natural-text prefill score is close to the 90% bar, and its sampling uncertainty is recorded above. Shared-suite generation has a 256-token cap, so unfinished coherent reasoning is not presented as a completed answer. Any controlled model behavior is documented with the matched HF evidence; no accuracy waiver is used for precision selection. No reset, remote push or vLLM integration occurred.

Qualitative JSON token IDs/text and `.txt.gz` files preserve exact generated content. Plain `.txt` views have repository whitespace/newline formatting. The blind scores use the original JSON content.
