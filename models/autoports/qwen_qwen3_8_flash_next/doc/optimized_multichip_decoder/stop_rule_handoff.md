# Not tried (stop rule)

The owner records optimize Stop Rule2: the last two retained QSA refinements
each improved the target B32 workload by less than3%.

| Retained refinement | Before B32 ms | After B32 ms | Gain | Evidence |
|---|---:|---:|---:|---|
| Compact QSA projection output | 74.706688 | 74.268517 | 0.5865% | `dram_ablation_qsa_l3.json` → `dram_compact_qsa_l3.json` |
| Remaining QSA attention HiFi2 after role composition | 72.503128 | 71.619102 | 1.2193% | `dram_final_candidate_l3.json` → `dram_final_candidate_hifi2_l3.json` |

The selected roles are now integrated into the default. Other role integration
composes already measured choices; it does not introduce an unmeasured new search.
Both accepted optimizations preserve the gate contract. Candidate clean timing
is not final-default reproduction or final stage acceptance.

The following are performance handoff notes bounded by that rule, not blockers
or claims of measured rejection. Expected gains are unknown unless specified.
Do not repeat completed async/persistent/global and subgroup CCL, DRAM reader/
geometry/packing/fidelity, activation8, real KV8, or coherent QSA hidden-family
experiments merely because their alternatives were not selected.

| Untried extension | Existing evidence and next useful experiment | Expected gain |
|---|---|---|
| Hidden residual for PLE/GDN with sensitive exact WO | Whole-QSA native RS/fused AG-MM loses B32 despite component gains. A future extension must retain hidden slices through exact GR, norm, residual and the next decoder; no immediate restore to the old stream contract. | Unknown; QSA evidence is negative for the tested whole paths |
| Other whole-layer fused MM-RS combinations | Native/fused producer families pass through next GR in the component; whole-QSA native RS and fused AG-MM were integrated. Fused MM-RS with a different full-layer policy remains a distinct extension, requiring best-control B1/B32 and state gates. | Unknown |
| Packed-only active-expert execution | Current gate-selected execution remains correct. Dual expert formats are physically excluded by the prior per-device ledger; a packed-only implementation may change dispatch/layout cost without storing both formats. Keep selected experts and measure diverse B32; dense all-expert work requires evidence. | Unknown; use final MoE share to bound benefit |
| Additional phase-specific weight/layout or pipeline schedules | Packed shared now uses one immutable policy in both phases after measuring separate/phase-specific candidates. Further schedules need complete prefill/decode and persistent-memory accounting, not isolated MM speedups. | Unknown; current role gains diminish |
| Additional subgroup CCL dtype combinations or worker/chunk/link tuning | Global BFP8 regresses; real MoE/QSA subgroup gains are below0.2%; persistent ordered async regresses. New combinations must include casts, buffers and actual consumers. | Unknown; tested subgroup gains below0.2% |
| BFP8 KV at a long-context bandwidth break-even | Exact selected-row unpacking and short whole decoder pass, but B1 is12.09% slower. A future long-context branch must prove benefit, cache semantics and changed-format maximum context before selection. | Unknown; short-workload result is negative |
| New graph fusion/layout opportunities from the final selected-default profile | Starting material topology families were audited and tried. Preserve final advice output and map each new recommendation to existing experiments or a precise future experiment; do not invent a rejection from source/API availability. | Unknown until final advice tables exist |

The following are **mandatory completion work, not deferred optimization ideas**:
final-default PCC and warmed prefill/traced B1/B32, actual dtype/fidelity and
advice-enabled profiler tables/CSV/provenance, nonaligned tails and32K–64K context
smoke with unchanged BF16 KV, selected-memory/context ledger, changed-input and
mutable-state tracked replay, runtime fallback audit, watcher-clean stress,
independent stage-review with fixes/rereview, and local stage-owned commit SHAs.
Any failed gate still requires repair. This document neither marks the stage
complete nor overrides the user's requirement to persist through failed gates.
