# Advisor contribution: Phi-3.5 Mini decode batch 32

## Headline

The estimated 32-layer decoder moved from **18,210.88 us to 16,616.75 us, a measured -1,594.13 us/model, with a ±13.76 us uncertainty band**. The saving is well outside the band. This is a per-layer measured delta scaled over the model's 32 identical dense layers, not an end-to-end full-model timing.

The frozen incumbent measured 0.656989 ms/layer. The shipped L1 RoPE chain measured 0.607172 ms/layer and confirmed at 0.607902 ms/layer in a fresh process. Every candidate and confirmation repeat beat every incumbent repeat. The real-weight differential oracle passed at PCC 0.9999988 against the frozen incumbent (bar 0.995).

## Control and capture

- Batch is 32 throughout: `decode_batch=32`, `requested_decode_batch=32`, and `capture_batch=32`.
- The checkpoint config has `num_hidden_layers=32`; the decoder builds one dense topology, recorded as `layer_counts={"dense": 32}`.
- Incumbent `measured_at` precedes capture `captured_at`.
- Timing used the fixed `harness_template.py` protocol: 10 untimed warmups, five block means, and 50 traced replays per block, in a fresh process per configuration.
- Capture used the challenger `capture_template.py` contract with BFP4 attention/gate-up/down, BFP8 KV, and LoFi compute, matching executed profiler rows. Advisor pin: `618cd4e75d`.
- The single-replay incumbent window is 569.090 us and reconciliation closes at 100%; it is not DEGRADED. Feasibility is `measurable`: 70.381 us/layer ceiling versus a 0.430 us floor.

## What the advisor contributed

The ranked candidates were a connected explicit RoPE chain. Applying only its first generic restore produced `bad optional access`, so the chain was extended to the known legal exact batch geometry: L1-interleaved embeddings/slices/rotate-half arithmetic, restored to a rectangular 32-core height shard. This tests above the advisor's 22-core placement and uses an exactly dividing batch grid.

That combined candidate covers the ranked RoPE boundaries `dense:b30`, `dense:b15`, `dense:4`, `dense:9`, `dense:3`, `dense:8`, `dense:2`, `dense:10`, `dense:11`, `dense:7`, `dense:5`, `dense:6`, `dense:b39`, `dense:b40`, and `dense:1`. Their separate application is not a legal independent knob: they share neighbour layout constraints inside `_apply_rope`.

The candidate series was 0.608324, 0.608693, 0.607060, 0.606833, and 0.607172 ms. The incumbent series was 0.657027, 0.656989, 0.656822, 0.656754, and 0.657184 ms. Fresh confirmation was 0.608125, 0.607902, 0.607923, 0.607608, and 0.608084 ms.

## Accounting and exclusions

Reconciliation assigns 224.287 us/layer (39.41% of the window) to `agrees_with_shipped`. The advisor-dropped-boundary upper bound is 70.381 us/layer. `advised_boundaries.us_advisor_agrees=1.227 us/layer` is reported and never screened or credited.

The layer handoff is also reported but not screened: the layer enters from DRAM and exits in L1, pricing 1.399 us/layer or 43.369 us/model. The advisor was not asked about inter-layer boundaries, so this is upstream optimization work and contributes zero here.

The SDPA-to-concat 1.559 us conversion remains because `nlp_concat_heads_decode` requires a sharded input. Five DRAM-sharded matmuls already agree with the shipped family; DS was not re-credited. There are no material <=2-core attributable ops and no starved non-attributable ops.

Rejected configurations remain off: generic height-shard restore (`bad optional access`) and removing the required SDPA-to-concat sharding conversion. The winning `use_advisor_decode_rope_l1` policy is now the shipped default.
