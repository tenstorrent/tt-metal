# Optimized Decoder Work Log

Model: `microsoft/Phi-3.5-mini-instruct`

Autoport directory: `models/autoports/microsoft_phi_3_5_mini_instruct`

Date: 2026-06-15 UTC

## Implementation

Created:

- `tt/optimized_decoder.py`
- `tests/test_optimized_decoder.py`
- `doc/optimized_decoder/tracy/`
- `doc/optimized_decoder/watcher/`
- `doc/optimized_decoder/README.md`
- `doc/optimized_decoder/work_log.md`

The implementation is a single dense Phi-3.5-mini decoder layer. It preserves the functional decoder's paged prefill, paged decode, LongRoPE short/long table behavior, tensor `current_pos`, and trace-safe decode replay contract.

The final optimized path uses BF16 activations/norms, BFP8_B attention weights, BFP8_B prefill MLP weights, BFP4_B decode MLP weights, BFP8_B paged KV cache, width-sharded L1 decode residual activations, height-sharded L1 decode Q/K/V tensors, DRAM width-sharded decode weights, and DRAM-sharded decode matmul program configs.

## Final Commands And Results

Syntax and default suite:

```bash
python -m py_compile models/autoports/microsoft_phi_3_5_mini_instruct/tt/optimized_decoder.py models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py
pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py -s
```

Result: `4 passed, 2 skipped in 26.67s`.

Printed PCCs:

- Synthetic stats-derived weights: prefill PCC `0.9999909427141687`, decode PCC `0.9998422357977581`.
- Real layer-0 weights: prefill PCC `0.9999880147414637`, decode PCC `0.999796077448235`.
- Determinism test HF comparisons in each repeated run: prefill PCC `0.9999911722192633`, decode PCC `0.9998535538636368`; repeated identical TTNN outputs asserted PCC >= 0.9999 and passed.

Full context decode:

```bash
PHI35_RUN_LONG_CONTEXT=1 pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py::test_full_context_decode_current_position_and_page_table -s
```

Result: `1 passed in 18.76s`.

Long prefill:

```bash
PHI35_RUN_LONG_PREFILL=1 PHI35_LONG_PREFILL_LEN=32768 pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py::test_long_prefill_page_table -s
```

Result: `1 passed in 13.60s`.

Watcher run:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_APPEND=1 TT_METAL_LOGS_PATH=models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_decoder/watcher/2026_06_15_1334_final_split pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py::test_optimized_dense_layer_real_weights_prefill_decode_pcc -s
```

Result: `1 passed in 12.00s`.

Watcher audit:

```bash
rg -n -i "TT_FATAL|TT_THROW|exception|assert|out.of.bounds|overflow|sanit|stack overflow|noc .*bad|bad noc|l1 .*overflow|watcher.*error" models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_decoder/watcher/2026_06_15_1334_final_split/generated/watcher/watcher.log
```

Result: no matches.

## Performance Collection

Final profiler command:

```bash
PHI35_READ_DEVICE_PROFILER=1 PHI35_SKIP_MESH_CLOSE=1 PHI35_HOST_TIMING_ITERS=100 python -m tracy -r -p -v --dump-device-data-mid-run -o models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_decoder/tracy/raw_real -m pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py::test_optimized_dense_layer_real_weights_prefill_decode_pcc -s
```

Result: `1 passed in 11.00s`.

Profiler log:

`models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_decoder/tracy/raw_real/profile_split_mlp_policy_final.log`

Printed final profiler PCC and host timing:

- Prefill PCC `0.9999880147414637`.
- Decode PCC `0.999796077448235`.
- `PHI35_HOST_TIMED_TRACE_DECODE_E2E_US: 983.737`.

Source profiler CSV:

`models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_decoder/tracy/raw_real/reports/2026_06_15_13_30_42/ops_perf_results_2026_06_15_13_30_42.csv`

Final report commands:

```bash
tt-perf-report models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_decoder/tracy/dense/ops_perf_results_2026_06_15_13_30_42.csv --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END --csv models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_decoder/tracy/dense/prefill_perf_report.csv
tt-perf-report models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_decoder/tracy/dense/ops_perf_results_2026_06_15_13_30_42.csv --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END --no-summary
tt-perf-report models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_decoder/tracy/dense/ops_perf_results_2026_06_15_13_30_42.csv --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --csv models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_decoder/tracy/dense/decode_perf_report.csv --tracing-mode
tt-perf-report models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_decoder/tracy/dense/ops_perf_results_2026_06_15_13_30_42.csv --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --no-summary --tracing-mode
```

Summary from `tracy/dense/perf_summary.json`:

| Window | Device time | Op-to-op gap | Device + gap | Rows | Host ops |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefill | 1328.898 us | 1283.197 us | 2612.095 us | 46 | 0 |
| decode | 840.860 us | 82.610 us | 923.470 us | 56 | 0 |

Functional baseline from `doc/functional_decoder/tracy/dense/perf_summary.json`:

| Window | Device time | Device + gap |
| --- | ---: | ---: |
| prefill | 1807.085 us | 2752.018 us |
| decode | 1752.534 us | 1826.376 us |

The final optimized runtime path was measured with 0 host ops in both signposted windows.

## Optimization Iterations

Initial optimized path:

- Moved decode residual stream to width-sharded L1.
- Moved decode Q/K/V tensors to height-sharded L1 for cache update and paged SDPA.
- Stored decode weights in DRAM width-sharded layout.
- Used DRAM-sharded matmul program configs for decode QKV, O, gate/up, and down.
- Preserved BFP8_B weights and BFP8_B KV cache from the functional quality baseline.

Short-prefill DRAM-sharded trial:

- Source CSV: `tracy/raw_real/reports/2026_06_15_13_05_42/ops_perf_results_2026_06_15_13_05_42.csv`.
- Prefill device time `1918.781 us`, device + gap `3053.170 us`.
- Decode device time `894.532 us`, device + gap `977.352 us`.
- Decision: rejected for the short-prefill acceptance workload because default interleaved prefill matmul weights were faster.

L1 prefill placement trial:

- Source CSV: `tracy/raw_real/reports/2026_06_15_13_10_52/ops_perf_results_2026_06_15_13_10_52.csv`.
- Artifacts: `tracy/l1_prefill_trial/`.
- Decision: accepted for short O/down prefill matmul inputs.

Large prefill fix:

- Long prefill initially exposed L1 circular-buffer pressure in the large MLP matmul configs.
- The accepted fix chunks QKV at 2048 tokens, chunks output/MLP at 1024 tokens, and caps `in0_block_w` for large-N MLP matmuls.
- Evidence: `PHI35_RUN_LONG_PREFILL=1 PHI35_LONG_PREFILL_LEN=32768 ... test_long_prefill_page_table` passed in `13.60s`.

BFP8 MLP split baseline:

- Source CSV: `tracy/raw_real/reports/2026_06_15_13_24_48/ops_perf_results_2026_06_15_13_24_48.csv`.
- Prefill device time `1329.602 us`, device + gap `2547.156 us`.
- Decode device time `892.678 us`, device + gap `976.069 us`.
- Decision: kept until the BFP4 decode-only MLP policy improved decode without harming prefill.

BFP4 MLP prefill+decode trial:

- Source CSV: `tracy/raw_real/reports/2026_06_15_13_28_40/ops_perf_results_2026_06_15_13_28_40.csv`.
- Artifacts: `tracy/bfp4_mlp_trial/`.
- Prefill device time `1211.751 us`, device + gap `2688.945 us`, PCC `0.9997876497432253`.
- Decode device time `840.708 us`, device + gap `923.786 us`, PCC `0.999796077448235`.
- Decision: rejected as an all-path policy because it lowered prefill quality and worsened untraced prefill total; retained BFP4_B for decode MLP only.

Final split MLP policy:

- Source CSV: `tracy/raw_real/reports/2026_06_15_13_30_42/ops_perf_results_2026_06_15_13_30_42.csv`.
- Artifacts: `tracy/dense/`.
- Prefill device time `1328.898 us`, device + gap `2612.095 us`, PCC `0.9999880147414637`.
- Decode device time `840.860 us`, device + gap `923.470 us`, PCC `0.999796077448235`.
- Decision: accepted. BFP8_B prefill MLP preserves prefill quality while BFP4_B decode MLP improves traced decode.

## tt-perf-report Advice

Final prefill advice:

- High op-to-op gap advice reports that tracing could save up to `1088 us` for this signposted prefill window. This was not applied in this stage because the module prefill path is variable-length and one-shot; traced decode is the latency-critical traced path required here.
- QKV and MLP gate/up prefill matmuls still suggest DRAM-sharded program configs. This was tried in the short-prefill DRAM-sharded trial and rejected because it slowed the acceptance workload.
- O and MLP down short prefill matmul input placement was tried with L1 placement and accepted in the final path.
- HiFi4 matmul advice was not applied because the final BFP8/BFP4 HiFi2 matmul policy preserves PCC above the acceptance bar and improves latency.

Final decode advice:

- QKV and O decode matmuls are marked optimized.
- MLP gate/up and down decode matmuls are FLOP-bound BFP4_B matmuls and advise increasing grid size from 12.
- The grid-size advice was rejected for this decoder module because the accepted path uses DRAM-sharded weights with `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`; in this TTNN API path the effective grid is tied to the DRAM-bank sharded layout and there is no independent compute-grid override. Replacing it with a non-DRAM-sharded path would discard the accepted DRAM-sharded decode-weight optimization and was not justified by the tested evidence.
- The only remaining high op-to-op gap item in final decode is the first layer norm, with about `4 us` possible saving. The decode window is already traced and has 0 host ops.

Data movement and RoPE audit:

- Final prefill and decode reports have 0 host ops.
- The optimized hot path has no torch/from_torch/to_torch calls; `test_runtime_forward_fallback_audit_static` passed.
- Remaining decode layout ops are tied to current TTNN op contracts for width-sharded decode matmuls, head creation, paged cache update, paged SDPA, and Phi LongRoPE.
- `ttnn.experimental.rotary_embedding_hf` was inspected and rejected because it requires a padded `head_dim` of 32 or a multiple of 64. Phi-3.5 mini uses `head_dim=96`, so the HF split-half midpoint is not tile-aligned.
- `ttnn.experimental.rotary_embedding_llama` and `rotary_embedding_llama_fused_qk` were rejected for this model because their tile-local transformation-matrix contract does not implement Phi's HF split-half LongRoPE mapping at `head_dim=96`.
- No remaining measured layout conversion was identified as an unnecessary host fallback or removable sharding mismatch within current TTNN capabilities.

## Artifacts

Final optimized perf artifacts:

- `doc/optimized_decoder/tracy/dense/ops_perf_results_2026_06_15_13_30_42.csv`
- `doc/optimized_decoder/tracy/dense/prefill_perf_report.csv`
- `doc/optimized_decoder/tracy/dense/prefill_perf_report.txt`
- `doc/optimized_decoder/tracy/dense/prefill_perf_report.console.log`
- `doc/optimized_decoder/tracy/dense/prefill_perf_report_stacked.csv`
- `doc/optimized_decoder/tracy/dense/prefill_perf_report_stacked.png`
- `doc/optimized_decoder/tracy/dense/decode_perf_report.csv`
- `doc/optimized_decoder/tracy/dense/decode_perf_report.txt`
- `doc/optimized_decoder/tracy/dense/decode_perf_report.console.log`
- `doc/optimized_decoder/tracy/dense/decode_perf_report_stacked.csv`
- `doc/optimized_decoder/tracy/dense/decode_perf_report_stacked.png`
- `doc/optimized_decoder/tracy/dense/perf_summary.json`
- `doc/optimized_decoder/tracy/raw_real/profile_split_mlp_policy_final.log`

Trial artifacts:

- `doc/optimized_decoder/tracy/l1_prefill_trial/`
- `doc/optimized_decoder/tracy/bfp4_mlp_trial/`
- `doc/optimized_decoder/tracy/raw_real/profile_bfp4_mlp_trial.log`
- `doc/optimized_decoder/tracy/raw_real/profile_optimized_final_after_longfix.log`

Watcher artifact:

- `doc/optimized_decoder/watcher/2026_06_15_1334_final_split/generated/watcher/watcher.log`

## Limitations And Non-Applicable Items

- Batch-size-1 decode remains the contract for this autoport stage.
- Prefill requires sequence length to be a multiple of the paged-cache block size.
- Prefill is not traced in this module test; final reports still include host/device op-to-op gaps for prefill.
- Full 131072-token prefill is not used as a module test because full causal attention materialization is too large. The final optimized long-prefill stress length is 32768.
- MoE active-expert execution is not applicable because Phi-3.5-mini is dense.
- Fused CCL is not applicable because this is a single-device decoder layer.
- LM-head and sampling optimization are not applicable because this goal excludes full-model and vLLM work.
- Multichip, full-model generator, and vLLM serving integration were not started in this goal.
