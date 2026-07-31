# Functional Decoder: meta-llama/Llama-3.2-1B-Instruct

This directory records the functional-decoder bringup evidence for the repo-local TTNN autoport:

- Implementation: `models/autoports/meta_llama_llama_3_2_1b_instruct/tt/functional_decoder.py`
- Tests: `models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_functional_decoder.py`
- Target layer kind: HuggingFace `LlamaDecoderLayer`, dense Llama layer only.

The runtime contract is documented in `functional_decoder.py`. The hot `prefill_forward` and `decode_forward` paths accept TTNN tensors, use paged KV cache/page tables, and do not call torch, `ttnn.from_torch`, or `ttnn.to_torch`.

## Correctness

Acceptance bar: PCC >= 0.995.

| Evidence | Shape / position | Prefill PCC | Decode PCC | Repeated trace PCC | Artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| Synthetic stats-based weights | prefill 128, decode current_pos 128 | 0.9999880265 | 0.9999892573 | 1.0 | `synthetic_correctness.json` |
| Real layer-0 safetensors weights | prefill 128, decode current_pos 128 | 0.9999887963 | 0.9999900152 | 0.9999999999 | `real_weight_correctness.json` |
| Real layer-0 safetensors weights | prefill 8192, decode current_pos 8192 | 0.9999664355 | 0.9999890750 | 1.0 | `real_weight_correctness_prefill_8192.json` |

Long prefill probes passed with synthetic stats-based weights:

| Sequence length | PCC | Artifact |
| ---: | ---: | --- |
| 1024 | 0.9999664353 | `long_prefill_1024.json` |
| 2048 | 0.9999666862 | `long_prefill_2048.json` |
| 4096 | 0.9999667028 | `long_prefill_4096.json` |
| 8192 | 0.9999668094 | `long_prefill_8192.json` |
| 16384 | 0.9999670454 | `long_prefill_16384.json` |
| 32768 | 0.9999893172 | `long_prefill_32768.json` |

The full 131072-token HF-vs-TTNN PCC run was reduced because the HF dense attention reference would require 64 GiB for the float32 causal mask alone, and about 1 TiB for bf16 32-head attention scores. See `context_capacity_note.json`.

## Runtime Audit

`runtime_fallback_audit.json` records a guarded hardware run where Python `ttnn.from_torch` and `ttnn.to_torch` were monkeypatched to raise during the measured prefill and traced decode pass. The run passed.

The source audit also checks `FunctionalDecoder.prefill_forward`, `FunctionalDecoder.decode_forward`, `FunctionalDecoder.kv_cache`, and `_LlamaMLP._forward` for forbidden host fallback strings.

## Performance

Perf command:

```bash
FD_PERF_PREFILL_SEQ_LEN=8192 python -m tracy -r -p -v -o models/autoports/meta_llama_llama_3_2_1b_instruct/doc/functional_decoder/tracy/run_prefill_decode -m pytest --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_functional_decoder.py::test_perf_artifact_signposted_prefill_and_decode
```

Bounded `tt-perf-report` results:

| Pass | Signposts | Device ops | Host ops | Device time | Report |
| --- | --- | ---: | ---: | ---: | --- |
| Warmed prefill 8192 | `PERF_PREFILL` to `PERF_PREFILL_END` | 21 | 0 | 36560 us | `tracy/run_prefill_decode/prefill_8192_tt_perf_report.txt` |
| Warmed traced decode replay at current_pos 8192 | `PERF_DECODE` to `PERF_DECODE_END` | 22 | 0 | 864 us | `tracy/run_prefill_decode/decode_8192_tt_perf_report.txt` |

CSV/provenance artifacts:

- `tracy/run_prefill_decode/ops_perf_results_raw.csv`
- `tracy/run_prefill_decode/prefill_8192_summary.csv`
- `tracy/run_prefill_decode/decode_8192_summary.csv`
- `tracy/run_prefill_decode/perf_provenance.json`

## Watcher

Watcher-clean run:

```bash
timeout 180s env TT_METAL_WATCHER=1 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_APPEND=1 pytest --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_functional_decoder.py::test_runtime_fallback_audit_measured_prefill_and_traced_decode
```

Result: passed in 41.70s. The copied watcher logs are in `watcher/`, and a log scan for error/fatal/assert/failure markers returned no matches. See `watcher/summary.json`.

## Notes

- Use `--confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct` when running these tests from this checkout; the repo root `conftest.py` imports a missing legacy `models.tt_transformers` module.
- The real-weight tests use the locally cached HuggingFace snapshot for `meta-llama/Llama-3.2-1B-Instruct`.
- This work is limited to the functional decoder. No optimized-decoder, multichip, full-model, or vLLM work is included here.
