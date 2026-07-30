# google/gemma-4-12B Optimized Full Model

Batch-1 T3K result: TTFT 121.93 ms for the 149-token AIME24 prompt; fully traced on-device decode replay is 23.08 tokens/s/user, 43.32 ms/token. The host-masked text generation compatibility path is 4.26 tokens/s/user over 31 timed decode tokens.

This stage optimizes the complete repo-local TTNN full-model path on `ttnn.MeshShape(1, 8)` with `FABRIC_1D_RING`. It keeps all 48 decoder layers on the TP8 multichip path and extends optimization across token embeddings, final norm, tied LM head, logits movement, paged KV cache, generator boundaries, trace replay, and on-device greedy sampling. No vLLM work was started.

## Accepted Path

| Component | Final choice |
| --- | --- |
| Hardware | Wormhole T3K, 8 devices |
| Mesh/fabric | `ttnn.MeshShape(1, 8)`, `ttnn.FabricConfig.FABRIC_1D_RING`, Ring CCL |
| Embedding | Hidden-dimension TP shard, per-device `262144 x 480`, BF16 DRAM, ring all-gather to the residual stream |
| Decoder stack | All 48 layers use `MultichipDecoder`, TP8, local SDPA heads, row-parallel O/down projections, ring reductions |
| Residual layout | Replicated full hidden stream at full-model boundaries; local decode tensors width-sharded in L1 inside decoder/final head sections |
| KV cache | Paged per-device local KV heads; page table/current positions replicated |
| Final norm | `OptimizedRMSNorm` with decode-sharded output feeding the decode LM head |
| LM head | Tied embedding weight, vocab TP shard `3840 x 32768`; full-sequence prefill BF16 DRAM, decode/last-token prefill BFP4 LoFi DRAM-sharded matmul |
| Sampling | `models.common.sampling.SamplingGenerator`, forced greedy argmax, captured inside full-wrapper trace replay |
| MoE | Not applicable; this checkpoint is dense |

The decode LM head uses a 40-core width-sharded input/output contract and `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w=1)`. Larger `in0_block_w` and alternate output configs were tried; the failures were either a required sharded output-config assertion or L1 allocation overflows. The accepted BFP4 LoFi head keeps top-5/top-100 accuracy and improves traced decode versus BF16.

## Accuracy

Main reference: `../../readiness_aime24_plain.refpt`, generated from DeepSeek AIME24 prompt 0 with 32 HF greedy continuation tokens. The HF tokenizer for base `google/gemma-4-12B` has `chat_template = None`, so the requested chat-template rendering is not available for this checkpoint; plain AIME24 tokenization is the main gate.

| Check | Artifact | Result |
| --- | --- | --- |
| Prefill | `artifacts/run_prefill_check_aime24_plain_optimized.log` | top1 31/32, top5 32/32, top100 32/32 |
| Teacher forcing decode | `artifacts/run_teacher_forcing_aime24_plain_optimized.log` | top1 30/32, top5 32/32, top100 32/32 |
| Autoregressive readiness | `artifacts/run_autoregressive_aime24_plain_masked_optimized.log` | HF and TT both produced coherent 32-token AIME24 continuations |
| Full-wrapper traced sampling smoke | `artifacts/smoke_num_layers1_traced_decode_retry5.log` | capture token 499, replay token 499, match true |

The TT autoregressive continuation is not token-identical to HF after optimization, but it remains on-task and does not emit the multimodal special-token loop when the high-level host mask is enabled. The top-k acceptance gate remains stronger evidence for this base checkpoint: both prefill and teacher-forcing decode meet top5 100% and top100 100%.

## Performance

All timings are warmed and exclude model construction/weight loading.

| Path | TTFT ms | Decode metric | Artifact |
| --- | ---: | ---: | --- |
| Full-model baseline host-masked generation | 117.76 | 4.49 tokens/s/user | `../full_model/artifacts/perf_batch1_aime24_plain_masked_host_argmax.log` |
| Optimized host-masked generation | 121.93 | 4.26 tokens/s/user | `artifacts/perf_batch1_aime24_plain_masked_host_argmax_optimized.log` |
| Traced full-wrapper decode, BF16 LM head | n/a | 22.66 tokens/s/user, 44.12 ms/token | `artifacts/perf_batch1_aime24_plain_traced_on_device_sampling_lm_head_bf16.log` |
| Traced full-wrapper decode, final BFP4 LoFi LM head | n/a | 23.08 tokens/s/user, 43.32 ms/token | `artifacts/perf_batch1_aime24_plain_traced_on_device_sampling_optimized.log` |

The optimized production metric is the traced on-device decode path. The host-masked path remains a quality/compatibility path because Gemma's tokenizer contains non-EOS special IDs that the raw on-device sampler does not suppress.

## tt-perf-report

Profiler artifact root: `tracy/full_model_2_layers/`. Tracy captured a 2-layer full-wrapper path with a minimal internal layer count so the profiler could complete while still covering embedding, decoder, final norm, LM head, cache update, collectives, and sampling.

| Window | Device ops | Host ops | Device time | Op gap | Main costs |
| --- | ---: | ---: | ---: | ---: | --- |
| Prefill | 72 | 0 | 5,498 us | 3,766 us | final logits all-gather 923 us, BFP4 LM head 601 us, embedding/all-broadcast gaps, decoder matmuls/reductions |
| Traced decode | 111 | 0 | 7,997 us | 1,375 us | on-device argmax 4,067 us, async all-gather 1,447 us, BFP4 LM head 601 us |

Human-readable reports:

- `tracy/full_model_2_layers/prefill_perf_report.txt`
- `tracy/full_model_2_layers/decode_perf_report.txt`
- `tracy/full_model_2_layers/ops.csv`
- `tracy/full_model_2_layers/{prefill,decode}_perf_report.csv`

Actionable profiler advice was tried where it applied. Decode LM-head `in0_block_w >= 2` could not fit with legal output sharding on this shape; prefill L1-input advice was inherited from the decoder stage and rejected there due long-context PCC loss; broad CCL/fused-CCL opportunities were inherited from the optimized multichip decoder pass and rejected or ruled out with evidence.

## Optimization Checklist

| Item | Status |
| --- | --- |
| Decoder path fully traced with no host logits/argmax fallback | Complete for the measured `return_ttnn=True` trace replay path; scalar token/position refresh uses preallocated device input tensors |
| Decode activations width-sharded in L1 | Decoder, final norm, and decode LM-head inputs/outputs use local sharded contracts; documented reshard/DRAM transitions remain at TTNN op boundaries |
| Prefill activations DRAM interleaved and large matmuls explicitly configured | Complete, inherited from optimized multichip decoder and full LM head |
| SDPA/optimized composite ops used | Complete for prefill/decode attention paths |
| Explicit memory/program/compute configs | Complete for decoder matmuls, norms, CCLs, LM head, sampling, and KV-cache paths |
| Clean tile/core grids | Complete for TP8 hidden, intermediate, QKV, and vocab shards; decode LM head pads output shard to tile alignment |
| DRAM-sharded decode matmuls | Complete for decoder projections and decode LM head |
| Fused matmul-CCL opportunities | Profiled/ruled out in optimized multichip decoder; no accepted full-model-only fused API matches this replicated-residual row-parallel reduction contract |
| MoE active-expert path | Not applicable, dense model |
| Reduced precision/fidelity trials | Complete: decoder BFP4 trials rejected by PCC; LM-head BF16/BFP8/BFP4/BFP4-LoFi trials accepted and final BFP4 LoFi selected |
| Watcher | Clean with ETH watcher active and dispatch watcher disabled: `watcher_eth_no_dispatch/generated/watcher/watcher.log` |

## Runtime Audit

Clean measured trace path: no full-logits host readback and no host argmax in the signposted decode replay. `decode_forward(..., enable_trace=True, sample_on_device=True, return_ttnn=True)` captures the full TTNN decode forward, LM head, and `SamplingGenerator` greedy argmax, returning device tokens.

Named host boundaries:

- Weights, RoPE caches, page table, KV-cache allocation, and initial token/position tensors are built with `ttnn.from_torch` or `ttnn.as_tensor` outside measured steady-state decode.
- Trace replay refreshes scalar token and position inputs with `ttnn.copy_host_to_device_tensor` into preallocated device tensors before `ttnn.execute_trace`.
- Readiness checks and qualitative generation convert logits/tokens to torch for comparison.
- High-level `generate()` intentionally uses CPU logits plus special-token masking for text quality; this is not the measured optimized decode path.

## Limitations

- Base `google/gemma-4-12B` has no tokenizer chat template, so the main AIME24 readiness prompt is plain-tokenized rather than chat-rendered.
- The fully traced sampler is raw greedy argmax and does not apply host-side non-EOS special-token suppression.
- Trace capture uses the captured token position as a static decode index. The AIME24 prompt positions are below the sliding-window rollover threshold; generalized dynamic-position tracing is still a serving integration concern.
- `tt-perf-report` still reports trace-savings advice on an already traced decode window. This appears to be merged-device/op-gap attribution rather than a hidden host fallback.
- The LM-head `in0_block_w=1` recommendation remains because larger settings failed legal memory/config checks on this full vocab shard.

## Artifacts

- Optimized code: `../../tt/model.py`, `../../tt/generator.py`
- Profile harness: `../../tests/test_optimized_full_model.py`
- Accuracy logs: `artifacts/run_prefill_check_aime24_plain_optimized.log`, `artifacts/run_teacher_forcing_aime24_plain_optimized.log`, `artifacts/run_autoregressive_aime24_plain_masked_optimized.log`
- Performance logs: `artifacts/perf_batch1_aime24_plain_masked_host_argmax_optimized.log`, `artifacts/perf_batch1_aime24_plain_traced_on_device_sampling_optimized.log`
- Precision trials: `artifacts/precision_lm_head_decode_bfp{8,4,4_lofi}_teacher_forcing.log`, `artifacts/perf_batch1_aime24_plain_traced_on_device_sampling_lm_head_*.log`
- Profiler: `tracy/full_model_2_layers/`
- Watcher: `watcher_eth_no_dispatch/generated/watcher/watcher.log`
