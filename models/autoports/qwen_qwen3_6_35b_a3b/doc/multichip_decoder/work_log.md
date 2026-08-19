# Multichip Decoder Work Log

Model: `Qwen/Qwen3.6-35B-A3B`

Pre-commit repo SHA: `dc616b36652edcb0f4361629360e099370bca37b`

## Implementation

- Added `tt/multichip_decoder.py`.
- `MultichipDecoder` subclasses `OptimizedDecoder` and records
  `single_chip_baseline_cls = OptimizedDecoder`.
- Targeted the local `2x2` Blackhole mesh only.
- Implemented TP=2 across mesh columns and EP=2 across mesh rows.
- Kept decoder layer boundary activations replicated.
- Full-attention Q/K/V projections, output projection, paged KV cache update,
  and paged decode are sharded by TP columns.
- Linear-attention packed input projection, conv state, recurrent state, and
  output projection are sharded by TP columns.
- MoE shared expert is TP-sharded. Routed MoE uses active-expert decode with
  `moe_routing_remap`, EP-row masks, exact per-row decode `nnz`, EP reduction,
  and TP reduction.
- Added `tests/test_multichip_decoder.py` coverage for graph summary, source
  fallback audit, synthetic parity, non-aligned lengths, batch 2, real weights,
  traced decode, cache/state layout, and signposted perf.

## Commands And Evidence

Cheap checks:

```bash
./python_env/bin/python -m py_compile \
  models/autoports/qwen_qwen3_6_35b_a3b/tt/multichip_decoder.py \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_multichip_decoder.py \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/conftest.py

pytest --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --collect-only -q \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_multichip_decoder.py
```

Final collect-only result after adding batch-2 coverage: 14 tests collected.

Synthetic correctness:

```bash
pytest --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=900 -q -s \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_multichip_decoder.py \
  -k 'test_synthetic_multichip_decoder_prefill_decode_against_optimized or test_synthetic_multichip_decoder_non_aligned_lengths'
```

Result before the batch-2 test was added: 5 selected, 2 deselected, 5 passed in
74.82s including source fallback audit. PCCs are recorded in `README.md`.

Batch-2 routing/state validation:

```bash
timeout 900 pytest --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=600 -q -s \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_multichip_decoder.py \
  -k 'test_synthetic_multichip_decoder_batch2_against_optimized'
```

Final result: 2 passed, 12 deselected in 35.68s.

Real weights and runtime fallback:

```bash
timeout 1800 env TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
  RUN_QWEN36_REAL_WEIGHTS=1 \
  pytest --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=900 -q -s \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_multichip_decoder.py \
  -k 'test_multichip_runtime_fallback_audit_source or test_synthetic_multichip_decoder_prefill_decode_against_optimized or test_synthetic_multichip_decoder_non_aligned_lengths or test_synthetic_multichip_decoder_batch2_against_optimized or test_real_weight_multichip_decoder_prefill_decode_against_optimized' \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/multichip_decoder/logs/runtime_fallback_audit_exact_nnz.log
```

Result after the stage-review finding was fixed: 9 passed, 5 deselected in
142.09s. This covers source audit, synthetic multi-token prefill/decode,
non-aligned logical sequence lengths, batch 2, and real-weight layer checks
under `throw_exception_on_fallback=true`.

Watcher correctness:

```bash
timeout 1800 env TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 \
  TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
  RUN_QWEN36_REAL_WEIGHTS=1 \
  pytest --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=900 -q -s \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_multichip_decoder.py \
  -k 'not test_perf' \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/multichip_decoder/logs/watcher_correctness_disable_eth.log
```

Final result: 10 passed, 4 deselected in 184.54s. This covers source audit,
synthetic batch 1, non-aligned logical sequence lengths, synthetic batch 2,
real weights for layer 0 and layer 3, and traced decode for the batch-1 cases.

Device snapshot:

```bash
tt-smi -ls --local 2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/multichip_decoder/logs/tt_smi_final.log
```

Result: four local Blackhole p300c chips were visible and resettable.

Perf:

```bash
timeout 1200 env RUN_QWEN36_MULTICHIP_PERF=1 RUN_QWEN36_PERF_REAL_WEIGHTS=1 \
  ./python_env/bin/python -m tracy -r -p -v --no-runtime-analysis \
  --op-support-count=5000 --check-exit-code \
  --output-folder models/autoports/qwen_qwen3_6_35b_a3b/doc/multichip_decoder/tracy/final_exact_nnz_raw \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=900 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_multichip_decoder.py \
  -k test_perf_qwen36_multichip -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/multichip_decoder/logs/tracy_perf_final_exact_nnz_summary.log
```

Result: 4 passed, 8 deselected in 111.22s. Tracy report generated:

- `tracy/final_exact_nnz_raw/reports/2026_08_19_03_59_38/ops_perf_results_2026_08_19_03_59_38.csv`

The raw CSV was normalized for `tt-perf-report` by filling blank `DEVICE ARCH`
with `blackhole` and blank `AVAILABLE WORKER CORE COUNT` with `110`.

Report generation:

```bash
./python_env/bin/tt-perf-report --start-signpost BASE_LINEAR_PREFILL --end-signpost BASE_LINEAR_PREFILL_END --arch blackhole --active-experts 8 --no-color --raw-op-codes ...
./python_env/bin/tt-perf-report --start-signpost MC_LINEAR_PREFILL --end-signpost MC_LINEAR_PREFILL_END --arch blackhole --active-experts 8 --no-color --raw-op-codes ...
./python_env/bin/tt-perf-report --start-signpost BASE_FULL_PREFILL --end-signpost BASE_FULL_PREFILL_END --arch blackhole --active-experts 8 --no-color --raw-op-codes ...
./python_env/bin/tt-perf-report --start-signpost MC_FULL_PREFILL --end-signpost MC_FULL_PREFILL_END --arch blackhole --active-experts 8 --no-color --raw-op-codes ...
./python_env/bin/tt-perf-report --start-signpost BASE_LINEAR_DECODE --end-signpost BASE_LINEAR_DECODE_END --arch blackhole --active-experts 8 --no-color --raw-op-codes ...
./python_env/bin/tt-perf-report --start-signpost MC_LINEAR_DECODE --end-signpost MC_LINEAR_DECODE_END --arch blackhole --active-experts 8 --no-color --raw-op-codes ...
./python_env/bin/tt-perf-report --start-signpost BASE_FULL_DECODE --end-signpost BASE_FULL_DECODE_END --arch blackhole --active-experts 8 --no-color --raw-op-codes ...
./python_env/bin/tt-perf-report --start-signpost MC_FULL_DECODE --end-signpost MC_FULL_DECODE_END --arch blackhole --active-experts 8 --no-color --raw-op-codes ...
```

Output tables and CSVs are under `tracy/final_exact_nnz/`; summarized metrics
are in `tracy/final_exact_nnz/perf_summary.csv`.

## Triage Notes

An early linear synthetic run appeared hung. `tt-triage` showed only device 3
executing optimized baseline operations from the old 1x1 submesh baseline
harness, not the multichip decoder. The fix was to run the single-chip baseline
serially through `ttnn.CreateDevice(0)`, close it, then open the `2x2` mesh for
the multichip path.

Artifacts:

- `triage/linear_reshape_summary.txt`
- `triage/linear_reshape_tt-triage.txt.gz.parts`
- `triage/linear_after_conv_state_fix_summary.txt`
- `triage/linear_after_conv_state_fix_tt-triage.txt.gz.parts`

## Artifact Packaging

The profiler emitted multi-GB `profile_log_device.csv` and Tracy GUI traces.
Those transient files are not committed. The generated raw op CSV and normalized
Blackhole op CSV are gzip-split into `.parts` directories with SHA256 manifests.

Reconstruction pattern:

```bash
cat <file>.gz.parts/part_* > <file>.gz
sha256sum -c <file>.gz.parts/SHA256SUMS
gunzip -c <file>.gz > <file>
```

## Known Limitations

- Only the local `2x2` mesh is supported.
- Active Ethernet watcher mode hit a fabric-router teardown assertion after the
  correctness body passed. Worker/NOC watcher coverage with
  `TT_METAL_WATCHER_DISABLE_ETH=1` is clean.
- Full 262144-token context was not rerun on mesh in this stage. The contract is
  preserved from optimized/functional evidence and the per-device KV math
  reduces memory pressure; no hard context limit was observed.
