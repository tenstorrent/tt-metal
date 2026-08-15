# Gemma-4 26B A4B optimized multichip decoder

This stage optimizes the completed decoder in place on four Blackhole P300C
devices arranged as a `1x4` `FABRIC_1D_RING`. It does not add full-model or
vLLM code. The default remains tensor parallel: QKV/gate/up are column split,
O/down are row split, KV heads are local, and MoE executes only the gate's
top-eight active experts.

## Final default

- Sliding attention weights use BFP8 with HiFi2; full attention remains BFP8
  with LoFi. Dense and expert weights remain BFP8. Residuals and CCL payloads
  remain BF16 because lower precision missed PCC or lost latency after casts.
- Decode O, packed dense gate/up, and dense down use TP-local DRAM-sharded
  weights with block widths 4, 11, and 17. QKV remains packed/interleaved:
  wiring its block-11 DRAM candidate into the actual decode call reduced
  sliding decode PCC to 0.992642. Dense gate/up
  stays packed and its decode-only copy uses BFP4 from an independent host
  upload; explicit execution-phase selection keeps that copy out of prefill.
- Sparse expert gate/up uses `in0_block_w=44`, `per_core_N=2`; sparse down uses
  block width 6 and `per_core_N=2`. Execution remains `active=8/128`.
- Row-parallel contractions use a two-link ring. Full-attention decode uses
  three preallocated width-sharded-L1 async all-reduce buffers and semaphores;
  sliding attention retains standard all-reduce because the conversion cost
  outweighed the async saving there.

The inter-layer residual contract is replicated BF16 tile-layout DRAM
`[1,1,M,2816]`. O, dense-down, and expert-down complete their collective
inside the layer. Consequently there is no gather, reshard, or all-reduce at
a decoder-layer boundary. Exact hardware repros prove a fractured residual can
cross distributed RMSNorm and fused AG+matmul into QKV, packed dense gate/up,
replicated router logits, and a fixed selected expert. It is not selected
because the exact dense/expert-down fused-RS producers remain illegal and the
gate-selected rank-5 sparse expert op has no fused AG consumer. Dense
all-expert adaptation is not an acceptable substitute.

## Correctness and performance

All final numbers below are from the no-override default path, real weights,
sequence 1024, batch 1, five additional warmups, and 30 trace replays. PCC
thresholds are 0.995.

| Layer kind | Baseline prefill/decode PCC | Final prefill/decode PCC | Prefill before/after (ms) | Traced decode before/after (ms) |
| --- | ---: | ---: | ---: | ---: |
| sliding, layer 0 | 0.998613 / 0.999653 | 0.998493 / 0.997778 | 77.524 / 77.817 | 1.146865 / 1.068942 |
| full, layer 5 | 0.997088 / 0.999786 | 0.997408 / 0.998347 | 85.728 / 86.199 | 1.252950 / 1.110293 |

Decode improves 6.79% for sliding and 11.38% for full attention. Prefill is
0.38% slower for sliding and 0.55% slower for full; the selected
changes target the traced decode path and do not alter prefill kernels.

Final default also passes logical sequence 33 prefill plus repeated traced
decode for both layer kinds, advertised current position 262143, and warmed
batch-32 trace replay. `doc/context_contract.json` remains at context 262144;
the persistent buffers consume 2.0625 MiB/device/full layer and do not reduce
the physical memory envelope.

## Profiles and runtime safety

Four-device Tracy captures were converted by `tt-perf-report` into separately
signpost-scoped `prefill_analyzed.csv` and `decode_analyzed.csv` plus summary
tables under `artifacts/final_profile_{sliding,full}`. The
reports show packed projections, BFP8+HiFi2 sliding attention, DRAM-sharded
decode matmuls, `active=8/128` sparse matmuls, ordinary AG/reduce pairs for
sliding, and `AllReduceAsyncDeviceOperation` for full attention. Raw commands
and checkout/runtime provenance are in `artifacts/provenance.json`.
Same-run device/host accounting, theoretical bytes/token, and the cumulative
decode contract table are in `profile_accounting.md`. Candidate evidence is
indexed by `artifacts/candidates/manifest.json`.

Fallback-raising runs passed throughout. Watcher was run separately from the
profiler and passed both layer kinds with worker NoC/CB/assert checks. Ethernet
watcher instrumentation must be disabled because the checkout's instrumented
ACTIVE_ETH fabric firmware is 27920 bytes versus a 25600-byte config buffer;
fabric and collectives remain active. See `artifacts/watcher_clean.json`.

Rejected families and adapted retries—including whole-activation BFP8,
BFP8 KV cache, lower-precision CCL, separate projections, geometry,
persistent-buffer shapes, and fused CCL contract analysis—are recorded in
`work_log.md`. `$autofix` found that the apparent decode-only BFP4 corruption
was ambiguous phase selection at a 32-row prefill boundary; the fixed BFP4
path is retained. Its report is `AUTODEBUG.md`.
