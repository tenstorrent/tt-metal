# Functional Decoder Work Log

Target: `meta-llama/Llama-3.2-1B-Instruct`

Autoport directory: `models/autoports/meta_llama_llama_3_2_1b_instruct`

## Implementation

- Added `tt/functional_decoder.py`.
- Implemented one dense Llama decoder layer kind: RMSNorm -> self-attention -> residual -> RMSNorm -> MLP -> residual.
- Used common `Attention1D` and `RMSNorm1D`.
- Added local `_LlamaMLP` because the shared MLP path imports missing legacy `models.tt_transformers` modules in this checkout.
- `from_state_dict` is the host setup boundary. Runtime `prefill_forward` and `decode_forward` accept TTNN tensors and page-table/current-position tensors.

## Commands Run

Syntax/import checks:

```bash
python -m py_compile models/autoports/meta_llama_llama_3_2_1b_instruct/tt/functional_decoder.py
python -m py_compile models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_functional_decoder.py
pytest --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct -q --collect-only models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_functional_decoder.py
```

Correctness and audit:

```bash
pytest --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_functional_decoder.py::test_functional_decoder_contract_and_runtime_fallback_audit
pytest --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_functional_decoder.py::test_runtime_fallback_audit_measured_prefill_and_traced_decode
pytest --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_functional_decoder.py::test_synthetic_paged_prefill_decode_trace_and_determinism
pytest --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_functional_decoder.py::test_real_weights_paged_prefill_and_decode_trace
FD_PREFILL_SEQ_LEN=8192 pytest --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_functional_decoder.py::test_real_weights_paged_prefill_and_decode_trace
```

Long-context probes:

```bash
for seq in 1024 2048 4096 8192 16384 32768; do
  FD_LONG_SEQ_LEN=$seq pytest --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_functional_decoder.py::test_longest_feasible_prefill_probe
done
```

Performance:

```bash
FD_PERF_PREFILL_SEQ_LEN=8192 python -m tracy -r -p -v -o models/autoports/meta_llama_llama_3_2_1b_instruct/doc/functional_decoder/tracy/run_prefill_decode -m pytest --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_functional_decoder.py::test_perf_artifact_signposted_prefill_and_decode
tt-perf-report --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END --no-color --no-advice models/autoports/meta_llama_llama_3_2_1b_instruct/doc/functional_decoder/tracy/run_prefill_decode/ops_perf_results_raw.csv > models/autoports/meta_llama_llama_3_2_1b_instruct/doc/functional_decoder/tracy/run_prefill_decode/prefill_8192_tt_perf_report.txt
tt-perf-report --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --no-color --no-advice models/autoports/meta_llama_llama_3_2_1b_instruct/doc/functional_decoder/tracy/run_prefill_decode/ops_perf_results_raw.csv > models/autoports/meta_llama_llama_3_2_1b_instruct/doc/functional_decoder/tracy/run_prefill_decode/decode_8192_tt_perf_report.txt
```

Watcher:

```bash
tt-smi -r all
timeout 180s env TT_METAL_WATCHER=1 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_APPEND=1 pytest --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_functional_decoder.py::test_runtime_fallback_audit_measured_prefill_and_traced_decode
rg -n "ERROR|FATAL|ASSERT|WATCHER|exception|failed|overflow" generated/watcher models/autoports/meta_llama_llama_3_2_1b_instruct/doc/functional_decoder/watcher || true
```

## Results

- Synthetic 128-token paged prefill PCC: 0.9999880265.
- Synthetic traced decode at current_pos 128 PCC: 0.9999892573.
- Real-weight 128-token paged prefill PCC: 0.9999887963.
- Real-weight traced decode at current_pos 128 PCC: 0.9999900152.
- Real-weight 8192-token paged prefill PCC: 0.9999664355.
- Real-weight traced decode at current_pos 8192 PCC: 0.9999890750.
- Longest HF-vs-TTNN prefill probe: seq_len 32768, PCC 0.9999893172.
- Runtime fallback audit: passed with `ttnn.from_torch` and `ttnn.to_torch` guarded during measured prefill and traced decode.
- Watcher/noinline run: passed, copied to `watcher/`.

## Performance Results

- Warmed prefill 8192: 21 device ops, 0 host ops, 36560 us device time.
- Warmed traced decode replay at current_pos 8192: 22 device ops, 0 host ops, 864 us device time.
- Raw Tracy ops CSV and bounded `tt-perf-report` outputs are under `tracy/run_prefill_decode/`.

## Limitations

- The full 131072 context HF-vs-TTNN comparison was not run. The HF reference path needs a dense causal mask and dense attention score tensors; at 131072 tokens, the float32 causal mask is 64 GiB and bf16 32-head scores are about 1 TiB. See `context_capacity_note.json`.
- Root-level pytest collection in this checkout needs `--confcutdir` for these tests because root `conftest.py` imports missing legacy `models.tt_transformers` modules.
- The first watcher attempt without `TT_METAL_WATCHER_NOINLINE=1` hit an instrumented code-size failure and left device firmware state unhealthy; `tt-smi -r all` recovered the device before the clean watcher/noinline run.
