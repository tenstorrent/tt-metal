# Final independent stage review

Date: 2026-07-17

Verdict: **clean-pass**

Required work: none.

The fresh reviewer directly inspected the original goal and selected skill
contracts; `multichip_decoder.py`, its tests, the optimized baseline, fused CCL
validators, and matmul program sources; `context_contract.json`; canonical
correctness/watcher logs and XML; every geometry/topology candidate; and the
final merged profiler CSV plus human-readable reports.

Controlled observations:

- The four decode matmuls remain marked `SLOW`, but all exposed DRAM-sharded
  geometry controls were measured with real weights. O2/G1 improved whole-
  layer latency; QKV/down alternatives regressed or were noise.
- A fused hidden-residual family was not hardware-measured. The compiler 2D
  hidden-sharded control loses at decode; a non-fused flat form has equal
  large-payload wire volume plus norm gathers; current fused validators reject
  the selected DRAM-sharded program/layout, with a Blackhole nondeterminism
  blocker on the minimal fused family.
- Full 131072-token evidence is a decoder-stage peak-capacity proof and local
  cache allocation, not an 80-layer construction, consistent with the goal's
  prohibition on beginning full-model work.
- ETH watcher instrumentation is disabled due to fabric-router binary size;
  Tensix watcher remained attached to devices 0--3 and passed the 100-replay
  persistent-CCL stress.

Residual follow-on risks belong to the next full-model stage: recheck aggregate
L1 occupancy when 80 decoder objects own persistent CCL buffers, and revisit
fused CCL/matmul if Blackhole validators/support change. Within this decoder
stage, dynamic advancing trace, paging/remapping, nonaligned chunking,
two-layer composition, full-context arithmetic, final reproduction, profiler
provenance, runtime fallback audit, and watcher evidence were judged internally
consistent and sufficient.
