# Final-source evidence manifest

- Frozen UTC run/artifact window: 2026-07-30 00:19–01:03 UTC.
- Decoder SHA-256:
  `134ca812394d2548687bccdfc2524fc75c7c69bb52c0609f4b8ac67ab022972f`.
- Functional decoder SHA-256:
  `6fdec7f771d77e14d99aa201a49099c0f00a0abc600ae489e27afccf3df59e2e`.
- Fused test SHA-256:
  `8b6e627b0667e88e1dbc1069f2c03a961881455005e3e1d70b85803ad2e63d92`.
- Shared harness SHA-256:
  `86a456b8e03dc8088822bd6aad07461f6a69d90f446e8319193e4d557a4572ec`.
- TTNN extension SHA-256:
  `ed2bf1a78109396ab2411c7f7bd0a0fa341c3eb3c8b4f82a40985fadb3dc26a2`.
- Hardware: four visible P300 Blackhole chips; tests selected device 3.
- Firmware bundle: 19.8.0.
- Artifact hashes: `final_manifest.sha256`. Verification:
  `sha256sum -c models/autoports/google_gemma_4_26b_a4b_it/doc/fused_decoder/final_manifest.sha256`.

## Accepted gates

| Gate | Result | Artifacts |
| --- | --- | --- |
| fused topology/dispatch/provenance | 7 passed | `tests/test_fused_decoder.py` |
| real-weight PCC/cache views | passed | `pcc_layer*.json` |
| batch-2 prefill | passed | `prefill_batch2_*.json` |
| non-aligned logical lengths | passed | `prefill_boundaries_*.json` |
| bounded tail-cache integrity | passed | `bounded_modulo_tail_cache_integrity.json` |
| eager/capture/replay equivalence | passed | `trace_replay_pcc.json` |
| repeated b1/b32 trace determinism | passed | `trace_*.json` |
| advertised context decode | passed | `advertised_context_decode_*.json` |
| watcher-clean final source | 9 passed | `watcher.log` |
| functional/fused performance matrix | fused wins 6/6 rows | `layer*_host_timings.json`, functional baseline peers |
| Blackhole prefill op reports | passed | `tracy/final_ops_*_b1/prefill_report.{csv,txt}` |
| Blackhole traced replay device timing | passed equivalent | `tracy/final_*/.logs/cpp_device_perf_report.csv`, `profiler_summary.md` |

The advertised-context artifacts remain applicable because graph fusion changes
only expert activation dispatch and router constant setup; attention layout,
paged-cache allocation, dtype, and capacity are unchanged. The final subclass
was nevertheless rerun at the advertised decode position and at representative
non-aligned prefill boundaries.
