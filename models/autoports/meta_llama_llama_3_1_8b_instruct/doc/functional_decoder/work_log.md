# Functional Decoder Work Log

Date: 2026-06-15

Model: `meta-llama/Llama-3.1-8B-Instruct`

Autoport directory: `models/autoports/meta_llama_llama_3_1_8b_instruct`

Repo commit: `86f8bc022e6d526d9766539c6ea50137cabec799`

Runtime identifiers:

- `torch 2.10.0+cpu`
- `transformers 5.13.0.dev0`
- `ttnn` Python package reports no `__version__`
- N300 Wormhole board, 8 visible UMD chips, tests use one 1x1 mesh

## Files Added

- `tt/functional_decoder.py`
- `tests/test_functional_decoder.py`
- `__init__.py`
- `tt/__init__.py`
- `tests/__init__.py`
- `doc/functional_decoder/README.md`
- `doc/functional_decoder/work_log.md`

## Implementation Notes

`FunctionalDecoder` mirrors HF `LlamaDecoderLayer` residual order:

1. input RMSNorm
2. paged self-attention
3. residual add
4. post-attention RMSNorm
5. SwiGLU MLP
6. residual add

The HF Q/K projection weights are converted to Meta/TTNN RoPE head order with
`_reverse_permute`, then Q/K/V are concatenated into the `Attention1D` `wqkv`
contract. Runtime prefill/decode operations only consume TTNN tensors.

`from_state_dict` validates the Llama 3.1 8B config shape exactly:

- hidden size 4096
- intermediate size 14336
- attention heads 32
- KV heads 8
- head dim 128
- RMSNorm eps 1e-5
- SiLU activation
- no attention or MLP bias

## Commands And Results

### Syntax

```bash
python_env/bin/python -m py_compile \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tt/functional_decoder.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py
```

Result: passed.

### Full Cache Contract

```bash
python_env/bin/pytest \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py \
  -k full_context_cache_contract -vv -s
```

Result: passed.

Evidence:

- `max_seq_len=131072`
- `page_block_size=64`
- `max_num_blocks=2048`
- key cache block count 2048
- value cache block count 2048

### Synthetic Paged Prefill/Decode Trace

```bash
python_env/bin/pytest \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py \
  -k synthetic_paged_prefill_decode_trace -vv -s
```

Result: passed.

Metrics:

- `seq_len=128`
- `decode_context=129`
- `prefill_pcc=0.9999777881890652`
- `decode_trace_pcc=0.9999841394751932`
- `determinism_pcc=1.0`
- `eager_trace_pcc=1.0`
- `runtime_fallback_audit=prefill_decode_clean`

### Real Weights Paged Prefill/Decode Trace

```bash
python_env/bin/pytest \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py \
  -k real_weights -vv -s
```

Result: passed.

Metrics:

- `seq_len=128`
- `decode_context=129`
- `prefill_pcc=0.9999812906688174`
- `decode_trace_pcc=0.9999836008747124`
- `determinism_pcc=1.0`
- `eager_trace_pcc=1.0`
- `runtime_fallback_audit=prefill_decode_clean`

The test loads real local HF layer-0 weights with:

```python
AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    local_files_only=True,
    dtype=torch.bfloat16,
    device_map="cpu",
)
```

### Long-Context Probes

Commands used the same long-context test with
`LLAMA31_8B_FUNCTIONAL_DECODER_LONG_SEQ_LEN=<seq>` and
`LLAMA31_8B_FUNCTIONAL_DECODER_LONG_MAX_SEQ_LEN=<seq+128>`.

Passed probes:

| Seq len | Decode context | Prefill PCC | Decode trace PCC |
| ---: | ---: | ---: | ---: |
| 512 | 513 | 0.9999820419667054 | 0.999986338281037 |
| 2048 | 2049 | 0.9998910863406064 | 0.9999854097355967 |
| 4096 | 4097 | 0.9998915403322384 | 0.9999813933054436 |
| 8192 | 8193 | 0.9998919058058718 | 0.9999858092744297 |
| 16384 | 16385 | 0.9998919829177182 | 0.9999802190191369 |
| 32768 | 32769 | 0.9998921568476432 | 0.9999840371223072 |

Largest verified command:

```bash
LLAMA31_8B_FUNCTIONAL_DECODER_LONG_SEQ_LEN=32768 \
LLAMA31_8B_FUNCTIONAL_DECODER_LONG_MAX_SEQ_LEN=32896 \
python_env/bin/pytest \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py \
  -k synthetic_long_context -vv -s
```

Result: passed in 161.64 s.

Failed capacity probe:

```bash
timeout 420 env \
  LLAMA31_8B_FUNCTIONAL_DECODER_LONG_SEQ_LEN=65536 \
  LLAMA31_8B_FUNCTIONAL_DECODER_LONG_MAX_SEQ_LEN=65664 \
  python_env/bin/pytest \
    --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
    models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py \
    -k synthetic_long_context -vv -s
```

Result: process killed by host OOM before a pytest report.

Kernel log excerpt:

```text
[Mon Jun 15 12:48:17 2026] python3 invoked oom-killer
[Mon Jun 15 12:48:17 2026] oom-kill:constraint=CONSTRAINT_NONE,...task=python3,pid=4071657
[Mon Jun 15 12:48:17 2026] Out of memory: Killed process 4071657 (python3) total-vm:567087984kB, anon-rss:506740044kB, file-rss:0kB, shmem-rss:160kB
```

### Runtime Fallback Audit

The audited hot prefill and decode passes are inside
`_assert_no_host_fallback()`, which patches these APIs to raise if they are
called:

- `ttnn.from_torch`
- `ttnn.as_tensor`
- `ttnn.to_torch`
- `torch.tensor`
- `torch.as_tensor`
- `torch.empty`
- `torch.zeros`
- `torch.ones`
- `torch.arange`
- `torch.full`
- `torch.cat`
- `torch.stack`
- `torch.matmul`
- `torch.nn.functional.linear`

Result: synthetic, real-weight, and long-context test cases pass the audit.

### Performance

Tracy attempt:

```bash
python_env/bin/python -m tracy -r -p -v \
  -o models/autoports/meta_llama_llama_3_1_8b_instruct/doc/functional_decoder/tracy/dense \
  -m pytest \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py \
  -k synthetic -q -s
```

The synthetic test passed, but Tracy teardown aborted with
`tcache_thread_shutdown(): unaligned tcache chunk detected`, and report
generation then failed because a host/device ID mismatch left device 4 in host
logs but missing from `cpp_device_perf_report.csv`.

Fallback device profiler run:

```bash
TT_METAL_DEVICE_PROFILER=1 \
python_env/bin/pytest \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py \
  -k synthetic_paged_prefill_decode_trace -vv -s
```

Result: passed with the same synthetic PCC metrics as above.

Postprocess:

```bash
python_env/bin/python tools/tracy/process_ops_logs.py --date
```

Created:

- `generated/profiler/reports/2026_06_15_12_50_36/ops_perf_results_2026_06_15_12_50_36.csv`
- `generated/profiler/reports/2026_06_15_12_50_36/per_core_op_to_op_times_2026_06_15_12_50_36.csv`
- `generated/profiler/reports/2026_06_15_12_50_36/profile_log_device.csv`

Normalized artifacts in this directory:

- `tracy/dense/prefill_ops.csv`
- `tracy/dense/decode_ops.csv`
- `tracy/dense/raw_device_only_ops_perf_results.csv`
- `tracy/dense/raw_profile_log_device.csv`

The normalized inputs use signposts recovered from the host Tracy capture and
device timing rows from the device-profiler fallback. Timing columns are real
device-profiler values; tensor metadata and FLOP/DRAM percentages are neutral
placeholders for `tt-perf-report` rendering.

Rendered reports:

```bash
tt-perf-report \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/functional_decoder/tracy/dense/prefill_ops.csv \
  --start-signpost PERF_PREFILL \
  --end-signpost PERF_PREFILL_END \
  --csv models/autoports/meta_llama_llama_3_1_8b_instruct/doc/functional_decoder/tracy/dense/prefill_perf_report.csv \
  --no-advice \
  > models/autoports/meta_llama_llama_3_1_8b_instruct/doc/functional_decoder/tracy/dense/prefill_perf_report.console.log

tt-perf-report \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/functional_decoder/tracy/dense/prefill_ops.csv \
  --start-signpost PERF_PREFILL \
  --end-signpost PERF_PREFILL_END \
  --no-summary \
  --no-advice \
  > models/autoports/meta_llama_llama_3_1_8b_instruct/doc/functional_decoder/tracy/dense/prefill_perf_report.txt

tt-perf-report \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/functional_decoder/tracy/dense/decode_ops.csv \
  --start-signpost PERF_DECODE \
  --end-signpost PERF_DECODE_END \
  --tracing-mode \
  --csv models/autoports/meta_llama_llama_3_1_8b_instruct/doc/functional_decoder/tracy/dense/decode_perf_report.csv \
  --no-advice \
  > models/autoports/meta_llama_llama_3_1_8b_instruct/doc/functional_decoder/tracy/dense/decode_perf_report.console.log

tt-perf-report \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/functional_decoder/tracy/dense/decode_ops.csv \
  --start-signpost PERF_DECODE \
  --end-signpost PERF_DECODE_END \
  --tracing-mode \
  --no-summary \
  --no-advice \
  > models/autoports/meta_llama_llama_3_1_8b_instruct/doc/functional_decoder/tracy/dense/decode_perf_report.txt
```

Perf summary:

- warmed prefill: 24 device ops, 0 host ops, 3494.848 us summed `Device Time`;
- traced warmed decode replay: 22 device ops, 0 host ops, 2482.910 us summed
  `Device Time`;
- decode report has large op-to-op gaps because the final rendered report is
  based on device-only replay rows rather than a fully merged stable Tracy host
  timeline.

### Watcher

First watcher attempt:

```bash
TT_METAL_WATCHER=10 \
TT_METAL_LOGS_PATH=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/functional_decoder/watcher/synthetic \
python_env/bin/pytest \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py \
  -k synthetic_paged_prefill_decode_trace -vv -s
```

Result: failed during `ttnn.open_mesh_device` before the decoder ran:

```text
idle_erisc.elf: segment[0] [0x3f10,+0x5a88) overflows region:0 limit of 0x54c0 bytes
```

The stale watcher pytest process was terminated and hardware was reset:

```bash
kill 373268
tt-smi -r
```

Clean watcher run:

```bash
TT_METAL_WATCHER=10 \
TT_METAL_WATCHER_DISABLE_ETH=1 \
TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_LOGS_PATH=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/functional_decoder/watcher/synthetic_disable_eth \
python_env/bin/pytest \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py \
  -k synthetic_paged_prefill_decode_trace -vv -s
```

Result: passed.

Metrics:

- `prefill_pcc=0.9999777881890652`
- `decode_trace_pcc=0.9999841394751932`
- `decode_trace_repeated_input_pcc=1.0`
- `decode_eager_vs_trace_pcc=1.0`

Watcher log scan:

```bash
rg -n -i \
  'fatal|assert|exception|error|noc|l1|stack|sanitize|overflow|out.of.bounds|watcher.*(fail|fault)|hang|timeout' \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/functional_decoder/watcher/synthetic_disable_eth/generated/watcher
```

Result: no matches.

The clean watcher log ends with normal detach lines for devices 0 through 7 and
zero Ethernet retraining events.

## Artifact Index

Correctness and implementation:

- `tt/functional_decoder.py`
- `tests/test_functional_decoder.py`
- `doc/functional_decoder/README.md`
- `doc/functional_decoder/work_log.md`

Perf:

- `doc/functional_decoder/tracy/dense/prefill_ops.csv`
- `doc/functional_decoder/tracy/dense/decode_ops.csv`
- `doc/functional_decoder/tracy/dense/prefill_perf_report.txt`
- `doc/functional_decoder/tracy/dense/decode_perf_report.txt`
- `doc/functional_decoder/tracy/dense/prefill_perf_report.csv`
- `doc/functional_decoder/tracy/dense/decode_perf_report.csv`
- `doc/functional_decoder/tracy/dense/prefill_perf_report.console.log`
- `doc/functional_decoder/tracy/dense/decode_perf_report.console.log`
- `doc/functional_decoder/tracy/dense/raw_device_only_ops_perf_results.csv`
- `doc/functional_decoder/tracy/dense/raw_profile_log_device.csv`
- `doc/functional_decoder/tracy/dense/.logs/tracy_profile_log_host.tracy`
- `doc/functional_decoder/tracy/dense/.logs/cpp_device_perf_report.csv`
- `doc/functional_decoder/tracy/dense/.logs/profile_log_device.csv`
- `doc/functional_decoder/tracy/dense/.logs/tracy_ops_data.csv`
- `doc/functional_decoder/tracy/dense/.logs/tracy_ops_times.csv`

Watcher:

- `doc/functional_decoder/watcher/synthetic_disable_eth/generated/watcher/watcher.log`
- `doc/functional_decoder/watcher/synthetic_disable_eth/generated/watcher/kernel_names.txt`
- `doc/functional_decoder/watcher/synthetic_disable_eth/generated/watcher/kernel_elf_paths.txt`
- `doc/functional_decoder/watcher/synthetic_disable_eth/generated/inspector/kernels.yaml`
- `doc/functional_decoder/watcher/synthetic_disable_eth/generated/inspector/mesh_devices_log.yaml`
- `doc/functional_decoder/watcher/synthetic_disable_eth/generated/inspector/mesh_workloads_log.yaml`
- `doc/functional_decoder/watcher/synthetic_disable_eth/generated/inspector/programs_log.yaml`
- `doc/functional_decoder/watcher/synthetic_disable_eth/generated/inspector/startup.yaml`

## Open Limitations

- The stage is deliberately functional-only and single-chip.
- No non-BF16 dtype sweep was attempted.
- The 65K long-context probe failed on host memory in the HF eager reference,
  not on TTNN cache allocation. The full 128K cache geometry test passed.
- The normalized perf CSVs contain real timing and signpost windows but neutral
  tensor metadata for `tt-perf-report` rendering.
- Root pytest collection is blocked by this checkout's root `conftest.py`
  importing missing `models.tt_transformers.demo.trace_region_config`; use the
  autoport-local `--confcutdir` shown above.
