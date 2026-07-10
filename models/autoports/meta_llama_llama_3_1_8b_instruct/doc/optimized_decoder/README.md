# Optimized Decoder

This stage adds `tt/optimized_decoder.py` for `meta-llama/Llama-3.1-8B-Instruct`.

## Selected Runtime

- Prefill preserves the functional public contract: input shape `[1, batch, seq_len, hidden]`, batch 32, valid non-aligned logical sequence lengths. Projection weights are stored as `bfloat8_b`; RMSNorm weights remain BF16.
- Prefill now places the matmul input activations for QKV, gate, up, and down projection in L1 by default after a warmed tt-perf-report candidate beat the all-DRAM candidate. The o-proj input remains DRAM because the L1 conversion inserted an extra device copy and was slower.
- Decode uses the common optimized Llama 1D block with packed QKV, an autoport-local packed gate/up MLP candidate selected as default, sharded decode residual layout, DRAM-sharded decode matmuls, paged KV update, and `SdpaDecodeDeviceOperation`.
- Decode precision starts from the repo-local `performance_decoder_config.json` and applies the selected layer-local candidate in `OptimizedDecoder`: QKV/O, packed gate/up, and FF2/down weights use BFP4/LoFi. The measured path keeps BF16 residual/projection activations, emits the gated MLP intermediate as BFP8, feeds FF2 as BFP8 x BFP4, and allocates KV cache as BFP8.
- Paged KV uses `page_block_size=32`; tests validate a contiguous page table sized for `max_seq_len=64`.
- MoE/expert execution is not applicable to Llama-3.1-8B.

## Correctness

Command:

```bash
timeout 1200 pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py --tb=short -s --junitxml=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/test_reports/optimized_decoder_pytest.xml
```

Result: 12 passed in 74.72s. Artifacts:
`test_reports/optimized_decoder_pytest.xml` and
`test_reports/optimized_decoder_pytest_stdout.log`.

Watcher command:

```bash
TT_METAL_WATCHER=10 timeout 1200 pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py --tb=short -s --junitxml=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/test_reports/optimized_decoder_watcher_pytest.xml
```

Result: 12 passed in 77.82s; watcher attached to devices 0/1 and detached cleanly. Artifacts:
`test_reports/optimized_decoder_watcher_pytest.xml`,
`test_reports/optimized_decoder_watcher_stdout.log`, and
`test_reports/optimized_decoder_watcher.log`.

Observed PCC:

- Real prefill, seq_len 4: `0.999998`
- Real prefill, seq_len 8: `0.999998`
- Synthetic paged decode diagnostic, seq_len 1: `0.972351`
- Real paged decode, seq_len 1: `0.999991`
- Real paged decode, position 1 after one prior paged update: `0.999968`
- Repeated decode determinism: `1.000000`
- Real traced decode replay: `0.999991`
- KV cache dtype asserted by tests: `DataType.BFLOAT8_B`

The suite also contains warmed perf-only prefill tests for functional baseline, optimized default, and the DRAM candidate. These tests warm the model once and signpost only the second runtime path plus `ttnn.synchronize_device`, with no PCC readback or `ttnn.to_torch` inside the measured window.

## Performance

Artifacts:

- Baseline BF16 prefill: `tracy/baseline_prefill/prefill_perf_report.txt`
- Optimized prefill: `tracy/prefill/prefill_perf_report.txt`
- Prefill DRAM candidate: `tracy/prefill_warmed_default/prefill_perf_report.txt`
- Final current-source prefill: `tracy/prefill_current_final/prefill_perf_report.txt` and headline copy `tracy/prefill/prefill_perf_report.txt`
- Prefill L1 screening candidate: `tracy/prefill_warmed_l1/prefill_perf_report.txt` and `tracy/prefill_warmed_final/prefill_perf_report.txt`
- Rejected prefill o-proj L1 candidate: `tracy/prefill_warmed_o_proj_l1/prefill_perf_report.txt`
- Baseline eager decode: `tracy/baseline_decode/decode_perf_report.txt`
- Baseline traced decode candidate: `tracy/baseline_decode_trace/decode_trace_perf_report.txt`
- Repo performance-policy traced decode candidate: `tracy/decode_trace_repo_performance/decode_trace_perf_report.txt`
- Final traced decode: `tracy/decode_trace/decode_trace_perf_report.txt`
- Final current-source traced decode: `tracy/decode_trace_current_final/decode_trace_perf_report.txt`
- Packed gate/up candidate: `tracy/decode_trace_packed_gate_up/decode_trace_perf_report.txt`
- Separate-W1/W3 MLP core geometry candidates: `tracy/decode_trace_mlp48/` and `tracy/decode_trace_mlp72/`
- Packed-gate/up MLP core geometry candidate: `tracy/decode_trace_packed_mlp48/`
- Packed-gate/up invalid geometry logs: `test_reports/packed_mlp32_invalid.log`, `test_reports/packed_mlp72_invalid.log`, and `test_reports/packed_mlp128_invalid.log`
- Precision candidates: `tracy/decode_trace_ff2_bfp4/` and `tracy/decode_trace_attention_ff2_bfp4/`

Summary:

- Clean warmed prefill device time improved from `70,898 us` to `40,259 us` while preserving PCC. The measured signpost window excludes correctness readback and was regenerated from the current test/source with `PERF_PREFILL` signposts.
- The optimized all-DRAM prefill candidate was `40,696 us`; the selected L1 activation candidate was faster at `40,259 us` in the final current-source run.
- The o-proj L1 candidate removed the remaining L1-input advice but inserted one extra device op and was slower at `40,346 us`, so it is rejected.
- Best correct traced decode candidate before BFP4/LoFi was `1,130 us`.
- Repo performance-policy traced decode was `987 us`.
- FF2/down BFP4+LoFi improved traced decode to `872 us`.
- Final current-source traced decode with attention projections plus packed gate/up and FF2/down on BFP4+LoFi is `786 us`, 22 device ops, `16 us` total op-to-op gap.
- Separate-W1/W3 BFP4/LoFi core-geometry candidates did not beat the selected packed default under the final precision policy: 48 cores was `804 us` and inserted one extra reshard; 72 cores was `1,053 us`.
- Packed-gate/up BFP4/LoFi core-geometry candidates did not beat the selected default: the final default packed program was `786 us`, the packed 48-core override was correct but slower at `789 us`, packed 32 failed `in0_block_w=4` divisibility for the 2-tile input shard, packed 72 mapped to target 69 cores and exceeded the 64-worker N150 limit, and packed 128 exceeded the same limit directly.

## tt-perf-report Conclusions

- Final traced decode uses BFP4/LoFi for QKV, O, packed gate/up, and FF2/down. The selected policy is based on real-weight traced decode: repo-performance was `987 us`, FF2/down BFP4 was `872 us`, attention+FF2 BFP4 with separate W1/W3 was `795 us`, packed gate/up screened at `785 us`, and the final current-source rerun is `786 us`. The synthetic/random decode check is retained as a diagnostic and is not used to veto the faster real-weight policy.
- Same-policy separate-W1/W3 geometry variants were measured and rejected: 48 cores was slower and added a reshard; 72 cores lowered `in0_block_w` to 1 and was much slower. A 32-core override, which would have raised `in0_block_w` to 4, failed the legal-shape check: `shard_shape[1] / in0_tile.get_width() (2) must be divisible by in0_block_w (4)`. A 128-core override exceeded the N150 worker-core limit: target cores 112 > available 64.
- Same-policy packed-gate/up geometry variants were then measured after the packed topology became default. Packed 48 cores was correct but slower at `789 us`; packed 32 failed the same 2-tile input-shard divisibility check; packed 72 failed because its packed program mapped to target 69 cores, greater than the 64 available worker cores; packed 128 failed because target 128 cores exceeds 64.
- Prefill uses warmed perf-only signposts and remains eager, because the optimized-decoder requirement calls for warmed prefill latency and traced warmed decode latency. The final warmed prefill report has 37 device ops, 0 host ops, and `23 us` total op-to-op gap.
- Prefill report recommended placing input 0 in L1. The accepted candidate places the QKV, gate, up, and down projection inputs in L1 and improves the warmed path from the DRAM candidate `40,696 us` to final current-source `40,259 us`. The remaining o-proj L1 advice was tried separately and rejected because it added a `CopyDeviceOperation`, increased the path to 38 device ops, and measured `40,346 us`.
- Common optimized prefill remains rejected for this stage because it is single-user and asserts `seq_len % 128 == 0`; the stage must preserve valid non-aligned public sequence lengths.
- The current `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` path used by the selected decode FF1/FF3 program exposes core count, `in0_block_w`, per-core M/N, and fused activation knobs; it does not expose an output-subblock override for this DRAM-sharded decode configuration. The tt-perf-report "No output subblock size found" advice is therefore recorded as not actionable in this path.
- Forge sharding recommendations from `doc/functional_decoder/forge_sharding_recommendations.json` were used as seeds. The current legal N150 path keeps the matching high-value intents: width-sharded residual/norm layout, packed QKV, paged cache update, SDPA decode, DRAM-sharded decode matmuls, packed gate/up with on-device slices, fused gate SiLU in the elementwise multiply, and width-sharded residual adds. The emitted 90/96/109-core grids are not legal on the 64-worker N150 target and were mapped to the common model-config grids instead.
- Packed gate/up is selected by default because an autoport-local packed MLP candidate was legal and faster. It packs W1/W3 into one BFP4 DRAM-sharded weight, runs one `32 x 4096 x 28672` LoFi matmul, slices gate/up on device, and preserves real trace replay PCC `0.999991`; the final measured packed path is `786 us` versus `795 us` for the legal separate W1/W3 candidate.

## Limitations

The context contract remains short-context validated. Optimized decode validates paged KV semantics and trace replay, but this stage does not claim HF 131072-token context readiness.
