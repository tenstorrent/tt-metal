# Optimized Decoder Work Log

Date: 2026-06-15

Model: `meta-llama/Llama-3.1-8B-Instruct`

Autoport directory: `models/autoports/meta_llama_llama_3_1_8b_instruct`

Repo commit: `86f8bc022e6d526d9766539c6ea50137cabec799`

Runtime identifiers:

- `torch 2.10.0+cpu`
- `transformers 5.13.0.dev0`
- `ttnn` Python package reports no `__version__`
- N300 Wormhole board, 8 visible UMD chips, tests use one 1x1 mesh

## Files Added Or Updated

- `tt/optimized_decoder.py`
- `tests/test_optimized_decoder.py`
- `doc/optimized_decoder/README.md`
- `doc/optimized_decoder/work_log.md`
- `doc/optimized_decoder/precision_trials.log`
- `doc/optimized_decoder/final_full_optimized_run.log`
- `doc/optimized_decoder/final_real_weights_run.log`
- `doc/optimized_decoder/tracy/dense/*`
- `doc/optimized_decoder/watcher/synthetic_disable_eth/*`

## Implementation Notes

The optimized decoder keeps the functional residual order:

1. input RMSNorm
2. paged self-attention
3. residual add
4. post-attention RMSNorm
5. SwiGLU MLP
6. residual add

The optimized policy is `llama31_8b_single_chip_bfp8_attn_bfp4_mlp_decode_v1`:

- BF16 activations and norm weights;
- BFP8 attention weights;
- BFP8 paged KV cache;
- BFP4 MLP gate/up/down weights;
- BFP8 MLP multiply intermediate;
- LoFi MLP matmul kernels;
- width-sharded L1 decode residual stream;
- DRAM-sharded decode MLP weights and matmul program configs;
- explicit large-prefill 2D matmul program configs.

The common `models.common.modules.mlp.mlp_1d` import failed in this checkout
because it depends on unavailable `models.tt_transformers`, so the optimized
decoder uses an autoport-local `_OptimizedMLP`.

Measured hot-path movement is limited to TTNN device-side operations. The final
prefill `tt-perf-report` window has no tilize, untilize, copy, host, or reshard
operations. Decode has no tilize, untilize, copy, or host operations; the two
layout transitions in the decode report are the 3 us
`ShardedToInterleavedDeviceOperation` and 1 us
`InterleavedToShardedDeviceOperation` required by current TTNN decode attention
head op contracts.

## Commands And Results

### Syntax

```bash
python_env/bin/python -m py_compile \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tt/optimized_decoder.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py
```

Result: passed.

### Full Optimized Test File

Final command:

```bash
python_env/bin/pytest \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py \
  -vv -s \
  2>&1 | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/final_full_optimized_run.log
```

Result: 6 passed in 40.45 s.

Passed tests:

- `test_optimized_decoder_contract_and_policy`
- `test_optimized_decoder_full_context_cache_contract`
- `test_optimized_decoder_synthetic_paged_prefill_decode_trace`
- `test_optimized_decoder_repeated_trace_stress`
- `test_optimized_decoder_synthetic_long_context_paged_prefill_decode_trace`
- `test_optimized_decoder_real_weights_paged_prefill_decode_trace`

Final metrics from this run:

| Case | Seq len | Decode context | Prefill PCC | Decode trace PCC | Decode avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| synthetic | 128 | 129 | 0.9995298149814705 | 0.9995144101749035 | 0.800335081294179 ms |
| 8-replay stress | 128 | 129 | 0.9995298149814705 | 0.9995144101749035 | 0.7938862545415759 ms |
| long-context synthetic | 2048 | 2049 | 0.999534819041018 | 0.9995242130611273 | 0.8550598286092281 ms |
| real weights | 128 | 129 | 0.9994865842273941 | 0.9995098207830138 | 0.8087046444416046 ms |

### Additional Real-Weight Artifact Run

```bash
python_env/bin/pytest \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py \
  -k real_weights_paged_prefill_decode_trace -vv -s \
  2>&1 | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/final_real_weights_run.log
```

Result: passed.

Metrics:

- `prefill_pcc=0.9994865842273941`
- `decode_trace_pcc=0.9995098207830138`
- `determinism_pcc=1.0`
- `eager_trace_pcc=1.0`
- `prefill_ms_e2e=2.9665050096809864`
- `decode_ms_e2e_avg=0.7993229664862156`

### Precision Trials

Artifact: `precision_trials.log`.

All trials used real HF layer-0 weights and the optimized prefill/decode trace
harness.

| Trial | Prefill PCC | Decode trace PCC | Prefill avg | Decode avg | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| BFP8 attention, BFP8 MLP, BFP8 KV | 0.9999772171300764 | 0.9999838591470852 | 4.324990790337324 ms | 1.1400850489735603 ms | rejected, slower |
| BFP8 attention, BFP4 gate/up, BFP8 down | 0.9996526916909572 | 0.9996571433021519 | 3.3168280497193336 ms | 1.0319964494556189 ms | rejected, slower than full BFP4 |
| BFP8 attention, BFP4 gate/up/down | 0.9994865842273941 | 0.9995098207830138 | 2.950129099190235 ms | 0.9890336077660322 ms | promoted to final policy with LoFi kernels |

After these trials, LoFi MLP kernels were enabled. Final full-run real-weight
coverage preserved the same PCC level and kept decode around 0.8 ms.

### Long-Context And Cache

Default long-context optimized test:

```bash
python_env/bin/pytest \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py \
  -k synthetic_long_context_paged_prefill_decode_trace -vv -s
```

Final full-run metrics:

- `seq_len=2048`
- `decode_context=2049`
- `prefill_pcc=0.999534819041018`
- `decode_trace_pcc=0.9995242130611273`
- `determinism_pcc=1.0`
- `eager_trace_pcc=1.0`

Full-cache contract:

- `max_seq_len=131072`
- `page_block_size=64`
- `max_num_blocks=2048`
- key/value cache dtype BFP8
- key/value cache block count 2048

### Performance Profile

Final Tracy command:

```bash
python_env/bin/python -m tracy -r -p -v \
  -o models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/dense/.logs \
  -m pytest \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py \
  -k synthetic_paged_prefill_decode_trace -q -s \
  2>&1 | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/dense/tracy_run.log
```

Result: passed.

Raw report:

- `tracy/dense/.logs/reports/2026_06_15_13_41_52/ops_perf_results_2026_06_15_13_41_52.csv`
- `tracy/dense/.logs/reports/2026_06_15_13_41_52/profile_log_device.csv`

Stable copies:

- `tracy/dense/optimized_ops_perf_results.csv`
- `tracy/dense/optimized_profile_log_device.csv`
- `tracy/dense/prefill_ops.csv`
- `tracy/dense/decode_ops.csv`

Rendered reports:

```bash
tt-perf-report \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/dense/prefill_ops.csv \
  --start-signpost PERF_PREFILL \
  --end-signpost PERF_PREFILL_END \
  --csv models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/dense/prefill_perf_report.csv \
  > models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/dense/prefill_perf_report.console.log

tt-perf-report \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/dense/decode_ops.csv \
  --start-signpost PERF_DECODE \
  --end-signpost PERF_DECODE_END \
  --csv models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/dense/decode_perf_report.csv \
  > models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/dense/decode_perf_report.console.log

tt-perf-report \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/dense/prefill_ops.csv \
  --start-signpost PERF_PREFILL \
  --end-signpost PERF_PREFILL_END \
  --no-summary \
  > models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/dense/prefill_perf_report.txt

tt-perf-report \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/dense/decode_ops.csv \
  --start-signpost PERF_DECODE \
  --end-signpost PERF_DECODE_END \
  --no-summary \
  > models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/dense/decode_perf_report.txt
```

Final report summary:

- warmed prefill: 20 device ops, 0 host ops, 2387 us device time;
- traced warmed decode: 19 device ops, 0 host ops, 750 us device time;
- decode matmul advice: all five decode matmuls marked optimized.

### Before/After Latency

Functional baseline device timing from `doc/functional_decoder`:

- warmed prefill: 3494.848 us;
- traced warmed decode: 2482.910 us.

Final optimized timing:

- warmed prefill: 2387 us device, 2.5199800729751587 ms host timing;
- traced warmed decode: 750 us device, 0.8243415504693985 ms host timing.

Intermediate BFP8 optimized policy:

- real-weight prefill host timing: 4.324990790337324 ms;
- real-weight traced decode host timing: 1.1400850489735603 ms.

Final BFP4/LoFi policy from the durable full-suite log:

- real-weight prefill host timing: 2.5178100913763046 ms;
- real-weight traced decode host timing: 0.8087046444416046 ms.

### Performance Accounting

Decode bytes estimate:

- attention QKV BFP8 weights: 25,165,824 bytes;
- attention output BFP8 weights: 16,777,216 bytes;
- MLP gate/up BFP4 weights: 58,720,256 bytes total;
- MLP down BFP4 weights: 29,360,128 bytes;
- BFP8 K/V reads at context 129: about 264,192 bytes.

Total: about 130.3 MB/token.

At 288 GB/s single-chip DRAM bandwidth, roofline is about 0.452 ms/token.
The final signposted device decode is 0.750 ms/token and final warmed traced
end-to-end decode is 0.824 ms/token. The device gap is explained by non-matmul
decode ops plus measured matmul bandwidth at 208-232 GB/s; the host gap is
about 74 us in the final traced replay.

### Advice Tried Or Rejected

- BFP8 KV cache: kept. Full-cache test verifies BFP8 cache allocation and all
  optimized prefill/decode tests pass.
- DRAM-sharded decode matmuls: kept. Final decode report marks all decode
  matmuls optimized.
- Large prefill 2D configs: kept. Attention prefill matmuls use `in0_block_w=4`
  and `out_subblock_w=4`; MLP uses `in0_block_w=8` for short prefill and
  conservative `in0_block_w=4`, `out_subblock_w=2` for long prefill to avoid L1
  circular-buffer overflow.
- Short-prefill MLP L1 input: tried and rejected. It added a
  `CopyDeviceOperation` and produced `2437 us` prefill device time versus
  `2387 us` final, with extra movement.
- Attention prefill L1 input: rejected for this stage. `Attention1D` exposes a
  static `prefill_input_memcfg`; making it L1 would not preserve long-context
  semantics because large prefill inputs exceed practical L1 residency.
- HiFi2/HiFi4 for final BFP4 MLP: rejected. LoFi preserves PCC above threshold
  and is faster.
- Fused matmul-CCL: not applicable to the single-chip decoder stage.
- MoE active expert path: not applicable to dense Llama 3.1 8B.
- LM head and sampling: not applicable to decoder-only goal.

### Watcher

Command:

```bash
mkdir -p models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/watcher/synthetic_disable_eth
TT_METAL_WATCHER=10 \
TT_METAL_WATCHER_DISABLE_ETH=1 \
TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_LOGS_PATH=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/watcher/synthetic_disable_eth \
python_env/bin/pytest \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py \
  -k synthetic_paged_prefill_decode_trace -vv -s \
  2>&1 | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/watcher/synthetic_disable_eth/watcher_run.log
```

Result: passed.

Watcher log scan:

```bash
rg -n -i \
  "fatal|assert|exception|error|timeout|hang|watcher.*(warn|fail)|failed|fault|illegal|noc|erisc|arc|heartbeat|overflow|out.of.bounds" \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/watcher/synthetic_disable_eth
```

Result: no matches.

## Artifact Index

Correctness:

- `tests/test_optimized_decoder.py`
- `doc/optimized_decoder/final_full_optimized_run.log`
- `doc/optimized_decoder/final_real_weights_run.log`
- `doc/optimized_decoder/precision_trials.log`

Perf:

- `doc/optimized_decoder/tracy/dense/optimized_ops_perf_results.csv`
- `doc/optimized_decoder/tracy/dense/optimized_profile_log_device.csv`
- `doc/optimized_decoder/tracy/dense/prefill_ops.csv`
- `doc/optimized_decoder/tracy/dense/decode_ops.csv`
- `doc/optimized_decoder/tracy/dense/prefill_perf_report.txt`
- `doc/optimized_decoder/tracy/dense/decode_perf_report.txt`
- `doc/optimized_decoder/tracy/dense/prefill_perf_report.csv`
- `doc/optimized_decoder/tracy/dense/decode_perf_report.csv`
- `doc/optimized_decoder/tracy/dense/prefill_perf_report.console.log`
- `doc/optimized_decoder/tracy/dense/decode_perf_report.console.log`
- `doc/optimized_decoder/tracy/dense/prefill_perf_report_stacked.csv`
- `doc/optimized_decoder/tracy/dense/decode_perf_report_stacked.csv`
- `doc/optimized_decoder/tracy/dense/prefill_perf_report_stacked.png`
- `doc/optimized_decoder/tracy/dense/decode_perf_report_stacked.png`
- `doc/optimized_decoder/tracy/dense/tracy_run.log`
- `doc/optimized_decoder/tracy/dense/.logs/reports/2026_06_15_13_41_52/*`

Watcher:

- `doc/optimized_decoder/watcher/synthetic_disable_eth/watcher_run.log`
- `doc/optimized_decoder/watcher/synthetic_disable_eth/generated/watcher/watcher.log`
- `doc/optimized_decoder/watcher/synthetic_disable_eth/generated/watcher/kernel_names.txt`
- `doc/optimized_decoder/watcher/synthetic_disable_eth/generated/watcher/kernel_elf_paths.txt`
- `doc/optimized_decoder/watcher/synthetic_disable_eth/generated/inspector/kernels.yaml`
- `doc/optimized_decoder/watcher/synthetic_disable_eth/generated/inspector/mesh_devices_log.yaml`
- `doc/optimized_decoder/watcher/synthetic_disable_eth/generated/inspector/mesh_workloads_log.yaml`
- `doc/optimized_decoder/watcher/synthetic_disable_eth/generated/inspector/programs_log.yaml`
- `doc/optimized_decoder/watcher/synthetic_disable_eth/generated/inspector/startup.yaml`

## Closed Limitations

- No optimized decoder work is deferred from this stage. Remaining non-applicable
  checklist items are tied to later stages: multichip CCL, full-model LM head,
  full-model token sampling, MoE routing, and vLLM serving.
