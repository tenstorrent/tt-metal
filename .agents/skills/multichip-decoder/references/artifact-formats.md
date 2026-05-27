# Multichip Decoder Evidence Packet

This reference is a lightweight checklist, not a schema. Use it when shaping the durable evidence for `multichip-decoder`.

## Durable Files

Prefer this compact tree:

```text
models/autoports/<model>/tt/multichip_decoder.py
models/autoports/<model>/doc/multichip_decoder/
  multichip_decoder_bringup_log.md
  multichip_decoder.md
  tracy/<layer_kind>/prefill_perf_report.csv
  tracy/<layer_kind>/prefill_perf_report.txt
  tracy/<layer_kind>/decode_perf_report.csv
  tracy/<layer_kind>/decode_perf_report.txt
```

Keep failed mesh experiments, large raw profiler captures, and noisy logs in scratch space unless they explain the final result.

## Work Log

`multichip_decoder_bringup_log.md` is for the parallelization trail:

- single-chip baseline chosen and why;
- target hardware and mesh shape;
- parallelization hypotheses and strategy changes;
- sharding, collective, RMSNorm, KV-cache, and layout bugs found;
- commands run and where raw logs live;
- blockers if multi-chip bringup could not complete.

## Final Report

`multichip_decoder.md` should focus on the target mesh result:

- baseline implementation and reproduced baseline numbers;
- `MultichipDecoder` interface and mesh/layout contract;
- target mesh, strategy, and why alternatives were rejected;
- prefill/decode PCC against the single-chip TTNN baseline;
- paged KV-cache behavior on the target mesh;
- trace replay status for decode;
- sequence lengths preserved or measured capacity reductions;
- determinism/stress, watcher, and runtime fallback status;
- warmed single-chip and multi-chip latency, speedup, and efficiency;
- `tt-perf-report` findings, including communication/data-movement bottlenecks;
- remaining risks and next work.

## Optional Structured Files

Add compact JSON only when another tool consumes it: for example `mesh_contract.json`, `pcc_results.json`, or `performance_results.json`. Do not create a large manifest tree just to mirror the report.

## Keep Out

Do not store full model weights, binary tensors, program-cache directories, giant profiler captures, or every failed mesh experiment under `doc/multichip_decoder/`.
