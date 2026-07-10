# Optimized Decoder Work Log

## 2026-07-10 Current Checkout Refresh

- Restored the repo-local Llama autoport source/docs/tests from checkpoint `4595b4ee2c5` because the current checkout only contained stale `__pycache__` files for this autoport.
- Hardware health: `tt-smi` is not on PATH in this environment. A bounded TTNN mesh smoke passed with `ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)` followed by close, printing `MESH_SMOKE_OK`.
- Current optimized correctness command:

```bash
timeout 1200 pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py --tb=short -s --junitxml=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/test_reports/optimized_decoder_pytest.xml
```

Result: 12 passed in 74.72s. Current artifacts:
`test_reports/optimized_decoder_pytest.xml` and
`test_reports/optimized_decoder_pytest_stdout.log`.

- Current watcher command:

```bash
TT_METAL_WATCHER=10 timeout 1200 pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py --tb=short -s --junitxml=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/test_reports/optimized_decoder_watcher_pytest.xml
```

Result: 12 passed in 77.82s. Watcher attached to devices 0/1 and detached cleanly. Current artifacts:
`test_reports/optimized_decoder_watcher_pytest.xml`,
`test_reports/optimized_decoder_watcher_stdout.log`, and
`test_reports/optimized_decoder_watcher.log`.

Current PCC lines match the selected optimized contract: real prefill seq_len 4 and 8 both `0.999998`; real paged decode `0.999991`; nonzero-position paged decode `0.999968`; determinism `1.000000`; real traced decode replay `0.999991`; KV-cache dtype `DataType.BFLOAT8_B`.

## Stage Review Remediation

An independent stage review returned `more-work-needed` with two required fixes:

- Prefill performance evidence had to use a warmed runtime-only signpost and exclude PCC readback / `ttnn.to_torch`.
- The prefill tt-perf-report recommendation to place input 0 in L1 had to be tried or rejected with exact evidence.

Remediation:

- Added warmed perf-only tests for functional baseline, optimized default, and optimized DRAM candidate. They run one warmup prefill, synchronize, then signpost only a second prefill plus `ttnn.synchronize_device`.
- Reprofiled functional baseline prefill through that clean window: `70,898 us`, 37 device ops, 0 host ops, `23 us` total op-to-op gap. Artifacts: `tracy/prefill_warmed_baseline/` and headline copy `tracy/baseline_prefill/`.
- Reprofiled optimized DRAM candidate with the same clean window: `40,696 us`, 37 device ops, 0 host ops, `23 us` total op-to-op gap. Artifacts: `tracy/prefill_warmed_default/`.
- Added default L1 activation placement for QKV, gate, up, and down projection inputs. It preserves real prefill PCC `0.999998` for seq_len 4 and 8 and improves clean warmed prefill to `40,259 us` in the final current-source run, with 37 device ops, 0 host ops, and `23 us` total op-to-op gap. Artifacts: `tracy/prefill_current_final/` and headline copy `tracy/prefill/`. Earlier L1 screening artifacts are preserved under `tracy/prefill_warmed_l1/` and `tracy/prefill_warmed_final/` with `40,288 us`.
- Tried the remaining o-proj L1 input advice by converting the attention output before o-proj. It removed the remaining L1-input report advice but inserted one extra `CopyDeviceOperation`, increased the path to 38 device ops, and measured `40,346 us`; the code change was reverted. Artifact: `tracy/prefill_warmed_o_proj_l1/`.

Focused final prefill verification after reverting the o-proj L1 trial:

```bash
timeout 600 pytest -q \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_prefill_warmed_perf_only \
  'models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_prefill_real_weight_pcc[4]' \
  'models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_prefill_real_weight_pcc[8]' \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_prefill_runtime_source_has_no_functional_fallbacks \
  --tb=short -s
```

Result: 4 passed; real prefill PCC stayed `0.999998` for seq_len 4 and 8. This focused check was superseded by the saved full 12-test correctness and watcher runs above after the final source was frozen.

## Topology Audit

Measured functional/baseline path:

- Prefill: 37 TTNN device ops, packed QKV matmul, explicit slices/reshapes/transposes for Q/K/V, separate gate/up/down MLP matmuls, BF16 projection weights, all DRAM interleaved.
- Decode: functional decoder had no implementation. The optimized candidate uses the common Llama 1D block plus an autoport-local packed gate/up MLP: packed QKV, packed gate/up with on-device slices, paged KV cache updates, SDPA decode, sharded residuals, and DRAM-sharded decode matmuls.

Candidates considered:

- Packed QKV: kept. The common decode block emits one QKV matmul and validates paged KV decode.
- Forge sharding recommendations: seeded from `doc/functional_decoder/forge_sharding_recommendations.json`. Kept intents after legalizing to N150: packed QKV, width-sharded residual/norms, paged cache update, SDPA decode, DRAM-sharded decode matmuls, packed gate/up with fused SiLU on gate after on-device slices, and width-sharded residual adds. Rejected literal grids because the emitted 90/96/109-core grids exceed the 64-worker N150 target.
- Packed/fused gate/up: kept. AutoFix experiment added an autoport-local `PackedGateUpMLP1D` that packs W1/W3 into one BFP4 DRAM-sharded `LazyWeight`, runs one doubled-width LoFi matmul, slices gate/up on device, then uses the existing fused-SiLU multiply and FF2/down path. Focused trace replay passed with real PCC `0.999991`; traced device time improved from `795 us` for the legal separate W1/W3 candidate to final current-source `786 us`, so the packed path is selected by default.
- Common optimized prefill: rejected for this stage because it is single-user and asserts `seq_len % 128 == 0`. The stage must preserve valid non-aligned public sequence lengths, including the validated real-weight seq_len 4 and 8 cases.
- BFP8 prefill projection weights: kept. Real prefill PCC stayed `0.999998`; clean warmed device time improved from `70,898 us` to final current-source `40,259 us`.
- Prefill L1 activation inputs: kept for QKV, gate, up, and down projection inputs. The all-DRAM candidate measured `40,696 us`; the selected L1 candidate measured `40,259 us` in the final current-source profile.
- Prefill o-proj L1 input: rejected. It added a copy op, measured `40,346 us`, and did not beat the selected L1 candidate.
- BFP4/LoFi FF1/FF3 decode: kept. Real traced decode PCC was `0.999996`; traced device time improved from the best correct HiFi2/BFP8 traced candidate at `1,130 us` to `987 us`.
- BFP4/LoFi FF2/down decode: kept. Real traced decode PCC stayed `0.999996`; traced device time improved to `872 us`.
- BFP4/LoFi attention QKV/O plus packed gate/up and FF2/down decode: kept as final. Real paged decode PCC is `0.999991`, nonzero-position paged decode PCC is `0.999968`, real traced decode replay PCC is `0.999991`, and current-source traced device time is `786 us`.
- KV cache dtype: kept BFP8. `Attention1D` allocates KV cache from `TensorGroup.KV_CACHE`, which defaults to BFP8; optimized tests assert `DataType.BFLOAT8_B`. The `PagedUpdateCacheDeviceOperation` profiler rows show BF16 operands because Q/K/V tensors entering cache update are BF16 after head creation/rotary, not because the allocated cache policy changed to BF16.
- Separate-W1/W3 BFP4/LoFi FF1/FF3 geometry sweep: selected packed default kept. Same-policy 48-core override was correct but slower at `804 us` and inserted an extra reshard. Same-policy 72-core override was correct but slower at `1,053 us`; its FF1/FF3 matmuls fell to about `311 us` each. A 32-core override failed because `in0_block_w=4` was illegal for a 2-tile input shard, and a 128-core override exceeded the 64-worker-core N150 limit.
- Packed BFP4/LoFi gate/up geometry sweep: selected default kept. Same-policy packed 48-core override was correct but slower at `789 us` and kept the packed `32 x 4096 x 28672` matmul. Packed 32 failed because `in0_block_w=4` was illegal for the 2-tile input shard. Packed 72 failed because the program mapped to target 69 cores, exceeding the 64-worker N150 limit. Packed 128 failed because target 128 cores exceeds the 64-worker limit.
- Output-subblock advice: rejected for the selected DRAM-sharded decode matmul path because the current `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` constructor exposes `in0_block_w`, per-core M/N, core count, and fused activation only; there is no output-subblock knob for this path.
- Host fallback / torch conversions in measured runtime methods: rejected by source guards. `OptimizedDecoder.decode_forward` and `OptimizedDecoder.prefill_forward` contain no torch/from_torch/to_torch and no functional fallback calls.

## Candidate Matrix

| Candidate | QKV/O | FF1/FF3 | FF2/down | KV cache | Trace device time | Real PCC | Decision |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| HiFi2/BFP8 baseline | BFP8/HiFi2 | BFP8/HiFi2 | BFP8/HiFi2 | BFP8 | `1,130 us` | `0.999996` | Rejected, slower |
| Repo performance | BFP8/HiFi2 | BFP4/LoFi | BFP8/HiFi2 | BFP8 | `987 us` | `0.999996` | Rejected, slower than later real-weight candidates |
| FF2/down BFP4 | BFP8/HiFi2 | BFP4/LoFi | BFP4/LoFi | BFP8 | `872 us` | `0.999996` | Rejected only because attention+FF2 was faster |
| Attention+FF2 BFP4 separate W1/W3 | BFP4/LoFi | BFP4/LoFi separate | BFP4/LoFi | BFP8 | `795 us` | `0.999991` | Rejected, slower than packed gate/up |
| Attention+FF2 BFP4 packed gate/up | BFP4/LoFi | BFP4/LoFi packed W1/W3 | BFP4/LoFi | BFP8 | `786 us` | `0.999991` | Kept |

Prefill candidate matrix:

| Candidate | Main change | Device ops | Device time | PCC | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| Functional BF16 baseline | Functional prefill with BF16 weights | 37 | `70,898 us` | N/A in perf-only test | Baseline |
| Optimized DRAM candidate | BFP8 weights, DRAM activations | 37 | `40,696 us` | `0.999998` | Rejected, slower than L1 activation candidate |
| Optimized L1 activation final | BFP8 weights; QKV/gate/up/down inputs in L1 | 37 | `40,259 us` | `0.999998` | Kept |
| O-proj L1 candidate | Also converts o-proj input to L1 | 38 | `40,346 us` | Not retained; runtime candidate only | Rejected, extra copy and slower |

## Commands

Correctness:

```bash
timeout 1200 pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py --tb=short -s --junitxml=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/test_reports/optimized_decoder_pytest.xml
```

Watcher correctness:

```bash
TT_METAL_WATCHER=10 timeout 1200 pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py --tb=short -s --junitxml=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/test_reports/optimized_decoder_watcher_pytest.xml
```

Result: 12 passed; watcher attached to devices 0/1 and detached cleanly.
Stdout artifacts preserve exact `OPT_*` PCC and KV-cache dtype lines:
`test_reports/optimized_decoder_pytest_stdout.log` and
`test_reports/optimized_decoder_watcher_stdout.log`.

Clean warmed prefill baseline profiling:

```bash
timeout 1200 python -m tracy -r -p -v --output-folder models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/prefill_warmed_baseline -m pytest models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_baseline_prefill_warmed_perf_only -q -s --tb=short
/localdev/mvasiljevic/tt-metal/python_env/bin/tt-perf-report models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/prefill_warmed_baseline/prefill_ops.csv --start-signpost PERF_PREFILL_BASELINE --end-signpost PERF_PREFILL_BASELINE_END --csv models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/prefill_warmed_baseline/prefill_perf_report.csv
```

Clean warmed optimized prefill profiling:

```bash
timeout 1200 python -m tracy -r -p -v --output-folder models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/prefill_current_final -m pytest models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_prefill_warmed_perf_only -q -s --tb=short
/localdev/mvasiljevic/tt-metal/python_env/bin/tt-perf-report models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/prefill_current_final/prefill_ops.csv --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END
/localdev/mvasiljevic/tt-metal/python_env/bin/tt-perf-report models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/prefill_current_final/prefill_ops.csv --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END --csv models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/prefill_current_final/prefill_perf_report.csv
```

Traced decode profiling:

```bash
timeout 1200 python -m tracy -r -p -v -m pytest models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_decode_trace_replay_real_weight_pcc -q -s --tb=short
/localdev/mvasiljevic/tt-metal/python_env/bin/tt-perf-report models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/decode_trace/decode_trace_ops.csv --start-signpost PERF_DECODE_TRACE --end-signpost PERF_DECODE_TRACE_END --csv models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/decode_trace/decode_trace_perf_report.csv
```

## Evidence

- `tracy/baseline_prefill/prefill_perf_report.txt`: clean warmed BF16 prefill baseline, `70,898 us`.
- `tracy/prefill_warmed_default/prefill_perf_report.txt`: clean warmed optimized DRAM prefill candidate, `40,696 us`.
- `tracy/prefill_current_final/prefill_perf_report.txt` and headline `tracy/prefill/prefill_perf_report.txt`: clean warmed optimized L1 activation final regenerated from current source/test, `40,259 us`.
- `tracy/prefill_warmed_l1/prefill_perf_report.txt`: earlier L1 activation screening candidate, `40,288 us`.
- `tracy/prefill_warmed_o_proj_l1/prefill_perf_report.txt`: rejected o-proj L1 candidate, `40,346 us` and 38 device ops.
- `tracy/baseline_decode_trace/decode_trace_perf_report.txt`: HiFi2/BFP8 traced decode, `1,130 us`.
- `tracy/decode_trace_repo_performance/decode_trace_perf_report.txt`: repo performance-policy traced decode candidate, `987 us`.
- `tracy/decode_trace_current_final/decode_trace_perf_report.txt` and headline `tracy/decode_trace/decode_trace_perf_report.txt`: final current-source attention+packed-MLP BFP4/LoFi traced decode, `786 us`.
- `tracy/decode_trace_packed_gate_up/decode_trace_perf_report.txt`: packed gate/up screening profile, `785 us`.
- `tracy/decode_trace_packed_mlp48/decode_trace_perf_report.txt`: packed gate/up 48-core geometry candidate, `789 us`.
- `tracy/decode_trace_ff2_bfp4/decode_trace_perf_report.txt`: FF2/down BFP4/LoFi candidate, `872 us`.
- `tracy/decode_trace_attention_ff2_bfp4/decode_trace_perf_report.txt`: final selected attention+FF2 BFP4/LoFi screening profile, `796 us`.
- `tracy/decode_trace_mlp48/decode_trace_perf_report.txt`: separate-W1/W3 48-core BFP4/LoFi FF1/FF3 candidate under the final precision policy, `804 us`.
- `tracy/decode_trace_mlp72/decode_trace_perf_report.txt`: separate-W1/W3 72-core BFP4/LoFi FF1/FF3 candidate under the final precision policy, `1,053 us`.
- `test_reports/packed_mlp32_invalid.log`, `test_reports/packed_mlp72_invalid.log`, and `test_reports/packed_mlp128_invalid.log`: exact packed-gate/up invalid geometry blockers.
- `test_reports/optimized_decoder_pytest.xml` and `test_reports/optimized_decoder_pytest_stdout.log`: final 12-test optimized correctness run with exact PCC/KV dtype output.
- `test_reports/optimized_decoder_watcher_pytest.xml`, `test_reports/optimized_decoder_watcher_stdout.log`, and `test_reports/optimized_decoder_watcher.log`: watcher-clean 12-test run with exact PCC/KV dtype output.

## Commit Log

- Stage checkpoint commit: `3d6a972a5c2` (`Add optimized Llama 3.1 8B decoder`).

## Checklist

- Operation-topology audit completed.
- Correctness tested for real prefill seq_len 4 and 8, synthetic diagnostic and real paged decode, nonzero-position paged KV decode, trace replay, BFP8 KV-cache allocation, determinism, and source-level no-fallback guards.
- Warmed perf-only prefill tests exercise the optimized path and exclude correctness readback from measured windows.
- Watcher-clean optimized correctness run completed.
- BFP4/LoFi tried on decode attention and MLP projections and kept because it beats the best correct traced candidate; legal geometry candidates and exact invalid blockers are recorded above.
- Prefill tt-perf-report L1-input advice tried. QKV/gate/up/down L1 activation placement is kept; o-proj L1 placement is rejected with before/after evidence.
- `tt-perf-report` CSV and text artifacts recorded for prefill and decode.
- Context contract preserved without increasing or reducing advertised capability.
- Final stage review after commit-hook formatting and refreshed artifacts: clean-pass from subagent `019f4b1a-f598-7f23-8104-046d6e8fb3bb`. Hard-check gaps recorded by review but not blockers: context remains short-context validated only, and trace replay coverage does not prove arbitrary refreshed-input replay beyond the saved test shape.
