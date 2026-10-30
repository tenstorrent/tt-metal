# AutoFix: profiler capacity

Failing contract: complete whole-layer prefill and128 traced decode windows for
B32. AutoDebug isolated lost prefill markers, not decode or model failure.
See AUTODEBUG.md for exact producer warnings, ID-set comparison and source chain.

Fix: run_fused_profile_queue.py supplies Tracy --op-support-count4096. Existing
CLI sets the profiler capacity before device initialization. No runtime/library
change or relaxation of strict merging. Final captures use tracy_final, preserving
the failed capture. No intra-layer drains or shortened workload.

Validation passed on all eight final-source profiles: no dropped-marker warnings,
complete strict report merging, full prefill coverage, and128 complete decode
op-ID sets each matching the original captured body. Native tt-perf-report tables
and CSV exist for both phases in each profile directory. profile_audit.json and
profile_provenance.json identify the results and exact source/command provenance.
The profiler-capacity repair is verified; no runtime or numerical relaxation was needed.

AutoDebug launcher preflight failed because bubblewrap was unavailable. Retried
with explicit AUTODEBUG_SKIP_CHILD_SANDBOX=1 under the already-authorized unrestricted
execution policy, personal CODEX_HOME, gpt-6-astra/medium and ambient API keys removed.
The isolated investigator performed read-only source/artifact inspection only.

Other layout repairs, verified by candidate correctness before final acceptance:
- Full-attention direct32-head concat requires2208896B static L1 >1572864B.
  Splitting into two16-head groups passes (full_lifetime_retry4.json).
- Removing K/V temporaries before SDPA reduces the collision but retaining full Q
  still fails (full_lifetime_retry2.log); placing full Q in DRAM passes. Sliding Q
  remains sharded. The attention numerical configuration is not weakened.
- Decode concat emits32 logical padded rows; applying metadata-only logical view
  repairs B1 reshape. The adapted op is measured slower and rejected.
- Height-sharded head-split outputs are explicitly rejected by RMSNorm
  (layernorm_device_operation.cpp:227). Interleaved norm inputs are the tested
  adaptation, with their full transfer cost included in the winning batch timings.
- Incorrect scalar enum `MUL` was corrected to the binding's `MUL_UNARY_SFPU`;
  packed_retry.json and all subsequent candidate correctness pass.

These are diagnosed op/shape contracts with passing adaptations, not unresolved
runtime failures. No unrelated cleanup or low-level runtime edits are included.

Final profiler warning classification: the optional browser-copy helper looks for
its host .tracy in the default generated/profiler path even with a custom output
root. Every actual .tracy remains under its capture .logs/ and timestamped report
directory; raw host/device CSVs and strict reports are complete. The optional
browser-copy warning is therefore an output-copy-path limitation, not lost data.
Pandas mixed-column DtypeWarning concerns mixed metadata columns; the calculation
parses explicit device timestamp/ID fields and independently verifies capture IDs.
The controlled single-device motherboard/MMIO-subset notices are unchanged from
baseline. Compiler flags containing -Werror are not runtime errors. No dropped
markers, watcher corruption or device failure is accepted.

## Large full-attention batch: inherited capacity and K placement

The additional B55/S33 full-attention check failed: fused cache update static L1
end2,110,912B exceeds1,572,864B. The unchanged functional path also fails there
(individual update end2,114,696B). Both factories allocate B*Wt output tiles;
at D512, Wt=16 and BF16 tiles are2048B. Changing to individual updates alone
therefore does not fix this limit.

Narrow controls exposed allocation placement as well as static size. Before the
repair, B37 fused static end1,521,024B collided with K at1,409,024B; functional
static end1,524,872B collided with a resident shard at1,507,328B. K was allocated
while RoPE Q/cos/sin shards were live, so freeing those tensors left K below holes
at the top of L1. For full-attention B>=34, copy K to DRAM, explicitly deallocate
its original shard, then reshard and release the DRAM temporary. The explicit
release precedes the new allocation; a nested conversion expression would keep
the original alive and defeat relocation. Configured B1/8/16/32 graphs do not change.

Verified: B36/S33 passes functional and fused HF PCC, repeated traced replay and
changed positions/pages. At B37/S33 both fail the inherited physical L1 boundary.
The repaired fused B37 reports K at1,507,328B, proving the intended relocation
from1,409,024B, while static end remains1,521,024B. See full_batch_capacity_manifest.json,
full_capacity_36_*.json, full_capacity_37_*.log and before_relocation_* artifacts.
This is a limit of the measured S33/page-table geometry, not a universal batch cap;
metadata size varies with context. Advertised context and configured batches are
unchanged. No public restriction or relaxed numerical bar was introduced.
