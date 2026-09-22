# Selected terminal profile analysis

Saved-artifact analysis only; no device commands or implementation changes. Source hashes, all four devices, exact row boundaries and advice dispositions are in [profile_analysis_terminal.json](profile_analysis_terminal.json).

The complete selected model measures **25.259 ms/token (39.589 tokens/s)** with deferred delivery in `after_terminal_full.json`. The representative decoder kernels account for 22.921–22.970 ms when expanded to48 linear and16 full-attention layers. This makes the decoder the main decode cost; the clearest untried full-model opportunity in this profile is eager prefill dispatch.

## Measurement and boundaries

`before_prefill_trace` profiles real layers0/3 atB1/S128 plus embedding/RoPE, final norm, selected head, common split sampling and token history. It is not a64-layer profile. The full benchmark hasS128/G128 and127 timed autoregressive steps. Recorded model, generator, decoder, sampler and native-library hashes match between the profile and full benchmark.

Use each `device*_token_out_perf_report.csv` independently. Add `Device Time` and `Op-to-Op Gap` in microseconds, assigning each incoming gap to its receiving segment. The first report gap is blank. Row intervals below are zero-based and half-open; all four devices have the same144-operation sequence.

| Boundary | Rows | Device0 IDs | Kernel+gap range across devices, ms |
| --- | --- | --- | ---: |
| entry_and_rope | [0,9) | 1195–1203 | 0.065–0.066 |
| linear_layer | [9,57) | 1204–1251 | 0.426–0.427 |
| full_attention_layer | [57,99) | 1252–1293 | 0.312–0.312 |
| terminal_model | [99,113) | 1294–1307 | 0.968–0.974 |
| local_topk | [113,116) | 1308–1310 | 0.320–0.321 |
| candidate_gathers | [116,118) | 1311–1312 | 0.016–0.029 |
| candidate_selection_seed_feedback | [118,141) | 1313–1335 | 0.160–0.161 |
| history_append | [141,144) | 1336–1338 | 0.007–0.007 |

Layer starts are40-core hidden5120 RMSNorm rows9 and57. Row46 is linear MLP norm; row88 is full-attention MLP norm. Rows65/66 are Q/K head norms, so splitting at every norm would miscount layers. Row99 is the terminal RMSNorm followed by eight-core head reshard. GDN prep/scan at35/36 and paged SDPA at80 identify the two layer kinds. Layer-end residual adds are immediately followed by the next norm; no inter-layer gather or reshard is inserted.

## Full-stack estimate and its limits

| Quantity | ms/token |
| --- | ---: |
| Weighted decoder kernel sum | 22.921–22.970 |
| Weighted decoder kernels + profiled gaps | 25.460–25.503 |
| Decoder + once-per-token surrounding kernels | 24.351–24.402 |
| Decoder + surroundings, including all extrapolated gaps | 27.005–27.051 |
| Complete model, queued token out | 25.254 |
| Complete model, deferred all-token delivery | 25.259 |
| Complete model, immediate delivery | 25.288 |

The formula is48×linear +16×full; embedding, terminal, sampling and history are added once. The kernel sum is an empirical accounting floor from representative layers, not a theoretical DRAM roofline. No chip times are summed. Reported ranges preserve chip variation and are not confidence intervals.

The gap-inclusive expansion overshoots actual full decode by about1.77 ms. It therefore cannot be used to claim a measured full-stack device latency or a negative host overhead. Kernel-only expansion leaves0.857–0.908 ms for all dispatch/gaps and cross-workload effects combined, not specifically Python. The standalone Stage5 primary48×0.421990 +16×0.306304 =25.156384 ms includes isolated replay/synchronization overhead; its apparent~0.103 ms difference to full decode is not the terminal cost.

The actual reduced token-out window spans2.283–2.289 ms/device versus2.703 ms between host signposts including synchronization. These are same-profile measurements; the full-model estimate is a separate extrapolation.

## Terminal observations

- Four head matmuls execute BF16×BFP8/HiFi2 with FP32 accumulation, DRAM sharding, K block5 and2 readers/bank: three N16384 chunks and one N12928 tail. All decoder projection rows remain BF16×BFP4/LoFi with role-specific K blocks16/6/4/17. The native attributes verify16/24 compute workers; the tool’s8-core denominator and >100% FLOPs utilization are not valid saturation evidence.
- Local TopK is the110-core `TopkLargeIndicesDeviceOperation`, surrounded by route prep/finish. Its entire route is~0.320 ms, about1.27% of full decode; there is no generic one-core TopK or ArgMax fallback. The two gathers move32 candidates/device to128, not62080 logits/device. Existing power-of-two padding and force-argmax trials lost.
- The final logits row pad costs~24 us; it pads inactive users from1 to32 for the sampler, not vocabulary IDs. Candidate index formation, tie adjustment, seed update and sampling are separately indexed in JSON. Tie handling protects correctness and should not be removed merely to reduce op count.
- `SamplingDeviceOperation` has an incoming35–36 us gap even inside the confirmed sampling trace. The report’s generic “enable tracing” advice is already satisfied. A focused dispatch/seed-boundary experiment could investigate it, but deleting this gap would improve full decode by only~0.14%. It is not evidence of host sampling or host RNG.
- History append (`indexed_fill`, `copy`, cursor increment) costs~7.2 us with capacity1 here. Full capacity127 deferred minus queued timing is5.510 us/token; immediate minus deferred is28.318 us/token. These are whole-loop differences, not isolated kernel timings. Do not claim capacity-independent copy cost from this reduced profile.

## Advice disposition

**Act on full-model prefill dispatch.** The134-op eager prefill window contains3.017–3.062 ms kernels and3.518–3.568 ms gaps, totaling6.564–6.585 ms. There are81–83 gaps above6 us/device. The first embedding alone follows~373 us idle time. The signpost wraps request reset, prefill/head/first sampling and synchronization, so not every gap is decoder host dispatch. Trial bounded prefill tracing with stable request/cache/table storage and restoration, retain nonaligned length and batch/state correctness, then measure full TTFT. Stage5 already supplied validated caller-owned prefill trace evidence; full-model integration still needs its own result.

**Retain measured decoder choices.** [Stage5 advice closure](../optimized_multichip_decoder/advice_closure.md) and [collective contracts](../optimized_multichip_decoder/collective_contracts.md) cover packed projections, precision-locked geometry,40-core residual/norm, shared buffers and carried-sharded-residual families. Selected native AR costs~0.706 ms/two-kind stack; adapted RS/AG~0.744–0.747, fused distributed-norm sharded residual~0.775, rowAGMM~1.09 and MMRS~1.01 all lost. Rank-adapted monolithic GDN~0.551 ms/linear decode also lost. Current rows preserve these measured contracts, so generic CCL/fusion/precision advice alone does not justify repeating those sweeps.

**Retain selected terminal changes and test only specific remaining questions.** Head geometry949.427 us beat the999.725 us baseline; compatible reader1/3 lost and larger block10 hit L1. Deferred history delivers all tokens with small overhead. If the decode target requires more margin, investigate the remaining sampler dispatch gap or active-row/32-slot interface while preserving mixed-batch seed/tie behavior. Neither the sampler nor history explains a multi-millisecond unexplained full-model host gap.

This artifact does not declare Stage7 complete or replace full correctness, qualitative, watcher, context or final default validation. No new profiler capture was performed.

## Reproducible summary and read accounting

The standard-library-only [summary script](../../tests/summarize_full_model_perf.py) regenerates all four device partitions, weighted decoder and terminal estimates, runtime dtype/fidelity/collective/sampling audits, and same-profile host signpost accounting. It accepts `--profile-dir`, `--benchmark-json`, and `--output`. [interim_perf_summary.json](interim_perf_summary.json) was generated with the exact command in [interim_perf_summary.command.txt](interim_perf_summary.command.txt); its JSON also records the command, input hashes and saved source-manifest comparison. Rerun it on the final reduced profile and matching full benchmark when those artifacts are available.

Boundaries are inferred from the five hidden-width norms among seven total norms, checked against GDN and SDPA identities, then separated at the model/sampling trace transition (or first recognizable TopK/ArgMax route if raw trace IDs are unavailable). Head chunk count is derived from the terminal matmul rows. History is optional and recognized by its indexed-fill/copy/cursor suffix. Raw `.csv` and `.csv.gz` inputs are supported. Missing raw evidence leaves table-based timing available while recording the unavailable host/geometry evidence; ambiguous layer boundaries stop the report.

The inherited Stage5 storage plan and accounting agree on **56,770,560 bytes per linear layer and 53,821,440 per full-attention layer per device**, totaling **3,586,129,920 decoder projection bytes per token**. The current four head chunks add **339,804,160 bytes per device**: 160 K tiles, eight DRAM banks, bank widths of 64/64/64/52 tiles, and 1,088 bytes per BFP8 tile. The script derives bank width from N and runtime reader count, checks the selected head's output-width equality, and compares decoder shapes/dtypes/derived storage against the inherited plan. It does not interpret decoder `per_core_N` as an input weight-shard width. Unsupported geometry or a changed decoder source suppresses this estimate pending an audit.

Each full-attention layer needs at least 544 BF8 KV bytes per active context position per device, rounded to 32-position tiles. Across the full S128/G128 window, active contexts 129–255 add **1,392,640–2,228,224 bytes per token across the 16 local full-attention layers**. Decoder projections, head, and this minimum KV traffic divided by the inherited nominal **512 GB/s per device** give **7.670560–7.672192 ms/token**, mean **7.671370 ms/token**. This is a read-only lower bound; recurrent state, other activations, embeddings, constants, writes, extra KV rereads, CCL and dispatch are excluded. It is not measured bandwidth or an achievable target. No profiler FLOP/core-utilization denominator enters this calculation.

For the actual reduced profile, the same storage method gives a **0.879850 ms** lower bound using at least the first post-prefill decode context. Its observed device span is **2.283163–2.288699 ms**, and its synchronized host signpost spans **2.702587 ms**. These three numbers describe the reduced workload; the full-model device scalar remains null because the 48/16 expansion is an estimate. Runtime-source mismatches are recorded and suppress the headline benchmark read-bound scalar.

Host-only verification in [perf_summary_host_validation.json](perf_summary_host_validation.json) covers exact reproduction of the previous four-device timing estimates, the actual older eight-chunk/no-history profile with compressed raw CSV, missing raw inputs, rejected ambiguous norms/raw alignment, and suppressed read accounting for incompatible reader1 tail geometry. Black and Python syntax checks passed. No build or hardware run was needed; isort is unavailable in this environment.
