# Falcon3-10B-Base full-model report

## Result

The full 40-layer HF autoregressive path passes the Metal readiness, correctness, capacity, trace, sampling, qualitative, Watcher, and reduced-profile gates on four Blackhole p300c devices. `tt/model.py` implements embeddings, the optimized TP4 decoder stack, final norm, untied TP LM head, paged cache ownership, and on-device sampling. `tt/generator.py` implements the standard readiness contract plus explicit serving-oriented low-level prefill/decode state. vLLM integration is intentionally absent.

Exact model provenance: `tiiuae/Falcon3-10B-Base` at revision `34bb99a889fe0426412da3dd2b46e6f64c8fd003`.

## Measured readiness

| Gate | Result |
|---|---|
| Prefill top-k, fresh AIME24 reference | top-1 91%, top-5 99%, top-100 100% |
| Traced teacher-forcing top-k | top-1 91%, top-5 99%, top-100 100% |
| Standard workload | batch 1, prompt 128, output 128, all 40 layers |
| Full-model TTFT | 2444.08 ms cold; 34.99 ms warm |
| Trace device-only decode | 52.10 t/s/u |
| Trace teacher-forcing decode | 52.10 t/s/u |
| Trace caller-visible token-out | 51.98 t/s/u |
| Canonical readiness-runner decode | 44.56 t/s/u, untraced runner boundary included |
| Model trace | 18.3984 ms/token |
| Sampling trace | 0.7939 ms/token, 4.14% of model-plus-sampling time |

The optimized measured path is the caller-visible token-out number. Its caller reads only the final sampled token for returned output; that read is not token feedback. The next model replay consumes the sampler's device `tt_out_tok` buffer directly.

## Preserved optimized policy

- Fixed TP4 mesh: 1x4 `FABRIC_1D_RING`, two links, no topology fallback.
- BFP4_B/LoFi decoder and untied TP LM-head weights.
- BFP8_B paged KV cache, one local KV head per rank; BF16 residual, activation, norm, and CCL state.
- Persistent BF16 all-reduce buffers shared through all 40 layers.
- Native L1 width-sharded inter-layer residual; no inter-layer host or DRAM materialization.
- Selected decoder core policies remain qkv=4, o=2, gate/up=24, down=8.
- Fixed 32-slot decode supports active/inactive rows and mixed prompt lengths.

The full-model policy is additive around the inherited optimized decoder. Its rejected alternatives are recorded in `rejection_ledger.md`; no rejected decoder strategy was silently restored.

## Context and cache

The advertised context remains the HF limit of 32768 tokens. A public 32767-token non-aligned prompt plus two token-out steps executed all 40 layers. The generator padded to 32768, issued 16 prefill chunks, populated all 1024 logical pages, and verified all 320 final-page K/V tensors (40 layers x 4 ranks x K/V) were nonzero. Cache and rotary positions both reached 32768.

Physical capacity was recomputed with the actual model and BFP8_B cache:

| Per-device DRAM | Bytes |
|---|---:|
| Allocator capacity | 30,082,731,008 |
| Full-model weights and static generator state | 3,143,638,528 |
| Batch-32 x 32768 paged KV cache | 22,817,013,760 |
| Combined | 25,960,652,288 |
| Remaining | 4,122,078,720 |

The fixed-slot batch-32 execution gate used prompt lengths 33 through 64. It passed exact repeatability, slot permutation, physical-page remap, active-16/inactive-row preservation, representative batch-1 controls, and split-greedy equivalence for all 32 rows. Full-context execution was exercised at batch 1; full-context batch 32 was physically allocated because a 32-user, 32768-token functional run is not a practical stage validation workload.

## Trace and sampler contracts

The generator captures separate model and sampling traces. The model trace advances device-resident cache and rotary positions with `plus_one`; the sampling trace writes the selected token into the persistent next-token buffer. Across 128 paired replays, the measured steady-state deltas were zero for host token, position, rotary-position, page-table, and sampling-parameter copies. Position and rotary position both reached 256, and the feedback buffer matched the previous sampled output.

Unchanged page tables caused no host copy. A changed table caused exactly one copy per device, updated the persistent trace input, and was then stable on replay; restoring the table restored exact logits and the sampled token. Reset releases installed traces before cache mutation and preserves the persistent input pool for safe recapture.

Both common samplers were tested with semantically greedy settings. Canonical Sampling1D split greedy, Sampling1D force-argmax, and TTSampling returned token 2107. Sampling1D split greedy measured 0.8035 ms/call versus 1.0054 ms/call for force-argmax and 1.0091 ms/call for TTSampling. The selected sampler gathers only rank candidates; TTSampling's full-vocabulary gather and broader mutable request state were rejected. The Watcher sampler test also proved exact packet gathering and all 32 synthetic winners.

## Qualitative verdict

The canonical 100-token HF and TT story completions share the first five tokens and then diverge, as expected from the measured non-100% top-1 agreement. Both remain coherent English narratives with no repetition loop, prompt/control-token leakage, wrong-language drift, or malformed early collapse.

A six-prompt shared suite covered a haiku, supervised versus unsupervised learning, a story, thermodynamics, English-to-French translation, and Fibonacci code. The automated degeneracy checker reported no finding. The TT answers were relevant and coherent; the French response stayed in the requested language and the Fibonacci response produced plausible code. A focused 100-token haiku control proved host eager greedy, traced split greedy, safe trace recapture, and traced teacher-forcing were token-exact. The detailed human review is in `qualitative_verdict.md`.

## Debug and profile evidence

The full 40-layer evidence workload passed with TENSIX Watcher enabled, including all samplers, page-table mutations, direct token feedback, positions, and 128 trace pairs. A separate sampler Watcher run passed packet, Ring hidden-state gather, split-greedy, and force-argmax checks. Ethernet Watcher was intentionally disabled because its instrumented firmware image exceeds the watcher buffer; the TENSIX run reported no NoC corruption, assertion, fatal, or stall.

Tracy was collected separately from Watcher on a reduced real-weight graph containing embedding, layer 0, final norm, LM head, and sampler. `tt-perf-report` classified the prefill path at 10.7% modeled DRAM roofline and 55 GB/s and decode at 11.9% and 61 GB/s; some ops remain unclassified by the report tool. The reduced profile is diagnostic only; the official performance numbers above come from the full 40-layer trace workload.

## Layer-stack attribution

The inherited batch-1 sequence-17 layer microbenchmark is 0.318242 ms, so a naive 40x projection is 12.7297 ms. That projection is not like-for-like with the full-model positions 128 through 255 and initially underpredicted the 18.3984 ms official model trace by 5.6687 ms.

The review remediation repeated the real full-model wrapper at depths 1, 2, 4, 8, 16, and 40 with the official batch-1 128+128 workload and a full 32768-token cache at every depth. Model-trace latency fits:

```text
model trace ms/token = 0.276425 + 0.472178 * decoder layers
R² = 0.9999998875
```

The 40-layer point is 19.16196 ms; the fit predicts 19.16353 ms, a -0.00157 ms residual. Thus the apparent gap is a longer-position effective decoder slope, not stack integration or terminal growth. Fixed embedding, final norm, four-shard LM head, and model-trace overhead together account for the 0.2764 ms intercept. The reduced report observes each of the four local `3072 x 8192` BFP4/LoFi LM-head matmuls at about 53.9 us; embedding/final norm and their layout operations occupy the rest of the intercept. Sampling is separate at 0.79377 ms, queued model-plus-sampler orchestration adds about 0.0020 ms, and the caller token observation adds 0.0456 ms in the sweep. No inter-layer gather, reshard, or host boundary appears in code or the linear scaling.

The official 18.3984 ms/51.98 t/s/u measurement remains the reported stage result because it is the frozen standard evidence run; the independent sweep's 19.1620 ms/49.99 t/s/u point records normal repeat variance and is retained rather than substituted. Structured attribution is in `results/full_model_depth_sweep.json` and `results/perf_summary.json`.

## Limitations

- Hardware and topology are intentionally fixed to four Blackhole p300c devices in a 1x4 ring.
- The base checkpoint has no tokenizer chat template. The AIME24 reference therefore uses its exact native completion format; `reference_metadata.json` records the failed chat-template probe instead of inventing a template.
- Full-context execution is proven at batch 1 and physical batch-32 full-context capacity is proven by allocation.
- Interpreter shutdown prints nanobind reference-leak diagnostics after successful readiness runs and clean device closure; this is a binding-lifetime diagnostic, not a model fallback or device failure.
- This stage does not implement or test vLLM.

## Artifacts

- `results/aime24_plain_100.refpt`: fresh exact-revision reference.
- `results/full_model_evidence.json`: all-layer performance, sampling, memory, and trace evidence.
- `results/full_model_contract_coverage.json`: mixed/non-aligned/chunk-boundary, inactive-row, reset, stochastic, and host-mode contracts.
- `results/full_context_coverage.json`: 32767-token public prompt through all 40 layers.
- `results/full_model_capacity.json`: physical batch-32 full-context allocation.
- `results/full_model_batch32.json`: fixed-slot functional and remap evidence.
- `results/autoregressive/` and `results/qualitative_suite/`: generated HF/TT outputs and control evidence.
- `results/full_model_watcher.json` and `results/full_model_sampler_watcher.json`: Watcher gates.
- `results/full_model_profile.json` and `profile/reduced/`: reduced Tracy evidence.
- `results/full_model_depth_sweep.json`: same-workload layer slope, fixed terminal intercept, and inherited lower-bound reconciliation.
- `sampler_contract_audit.md`, `runtime_fallback_audit.md`, `rejection_ledger.md`, and `work_log.md`: audits and reproduction log.
