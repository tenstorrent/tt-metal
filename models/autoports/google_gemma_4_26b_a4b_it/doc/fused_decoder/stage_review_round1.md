# Stage review round 1

Verdict: `more-work-needed`

The independent reviewer held completion for four evidence gaps:

1. The available dedicated `ttnn.experimental.moe_compute` path had been
   dismissed without a target-shaped attempt.
2. The claim that the final graph beat the strongest correct candidate was not
   supported by a source-bound, same-run comparison across required cases.
3. Context, host-timing, and candidate artifacts did not consistently bind
   themselves to the exact fused decoder and test sources.
4. Remaining decode RMSNorm layout conversions lacked a direct feasibility
   test.

Resolution:

- `rejected_moe_compute_candidate.json` records the real-weight target-shaped
  attempt, numerical failure, and output-contract blocker.
- `final_vs_dense_split_layer*.json` records four alternating, 21-replay
  candidate comparisons with numerical equivalence.
- Current evidence contains decoder/test SHA-256 provenance and the default
  suite produces `source_binding.json`.
- `rejected_sharded_decode_rmsnorm.json` records the exact device rejection.

The stage was sent to a fresh reviewer only after these findings were fixed
and the complete default suite passed.
