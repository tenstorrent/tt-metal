# Mistral Small 24B optimized decoder

This directory records the completed single-device optimized-decoder stage for
`mistralai/Mistral-Small-24B-Instruct-2501`. It covers the repo-local TTNN
decoder layer only; multichip, full-model, LM-head, sampling, and vLLM work are
outside this stage.

## Result

The selected default in `tt/optimized_decoder.py` owns both prefill and decode
graphs. It uses flattened logical tokens, packed QKV, BFP4/LoFi attention and
MLP weights, BFP8 KV caches in the optimized test/generator contract,
DRAM-sharded decode matmuls, an advisor-seeded L1 residual/norm chain, explicit
SDPA configs, fused SiLU in the gate/up product, and large-M prefill configs.
Longer prefills transparently use general QKV/O configs and 576-token MLP
chunks; there is no public sequence-alignment restriction.

On one Blackhole p300c logical device (`TT_VISIBLE_DEVICES=2,3`), batch 32,
prefill length 18, representative dense layer 20:

| Metric | Functional | Selected optimized | Change |
| --- | ---: | ---: | ---: |
| real prefill PCC | 0.9999699 | 0.9997972 | -0.0001727; passes 0.99 bar |
| real decode PCC | 0.9999697 | 0.9998262 | -0.0001436; passes 0.99 bar |
| warmed prefill | 94.153 ms | 5.387 ms | 17.48x faster |
| traced warmed decode | 93.315 ms | 1.288 ms | 72.44x faster |

The final decode profile contains 31 device ops and no host ops: 1.2815 ms
summed device-kernel time, 0.0193 ms intra-window gaps, and 217 GB/s / 42.4%
modeled aggregate DRAM roofline. Its profiled host signpost span is 1.4783 ms;
the separate non-profiled 100-replay result is 1.2881 ms. See
`tracy/final/decode_perf_report.txt` and the full decision history in
`work_log.md`.

## Correctness and contract

- Prefill lengths 1, 17, 18, and 33 pass on the selected optimized path.
- Five consecutive decode positions pass and update paged BFP8 KV caches.
- Eager and first trace replay have PCC 1.0; 100 identical trace replays and
  their key/value cache writes are bitwise deterministic.
- The model has one meaningful decoder kind: all 40 layers are dense Mistral
  layers. Layer 20 is the functional/advisor/real-weight representative.
- A separate watcher run is clean; its environment, attach/detach output,
  passing test, and signature scan are saved under `evidence/`.
- Batch-32 optimized capacity passes at 3,584 and 4,096 tokens. The shared
  `doc/context_contract.json` now advertises the conservative tested value
  4,096 without claiming a hard maximum.

## Artifacts

- `shard_advise/report.json` and `shard_advise/final_ir.mlir`: mandatory
  current-pass advisor outputs.
- `shard_advise/advise_mistral.py`: bounded rewritten dense capture.
- `tracy/functional/`: BF16 functional profiler summaries.
- `tracy/optimized/`: intermediate profiler evidence that drove the rewrite.
- `tracy/final/`: selected default raw ops CSV (stored as `ops.csv.gz`) and
  advice-enabled prefill/decode `tt-perf-report` tables.
- `evidence/`: current-source JUnit logs plus watcher/device transcripts and
  precision-locked MLP block-16 expected-failure runner evidence.

Exact commands, candidate results, applied/rejected advisor choices,
performance accounting, limitations, and the completed optimization checklist
are in `work_log.md`.
