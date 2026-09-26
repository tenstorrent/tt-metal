# Direct projection and MLP geometry candidates

Snapshot of `actual_direct_geometry_commands.json` and
`actual_direct_mlp_geometry_commands.json`: **33 completed candidates, 29 passes,
4 L1 errors**. All use real checkpoint weights and the recorded actual-text
4096/128 fixtures; the unchanged HF threshold is **0.995**. Every passing row
covers all 128 unique positions 4096..4223, with deterministic first-position
replay, clean runtime audits, and the program-cache guard enabled.

Times are **warmed traced host-wall medians**, in microseconds: five batches of
30 replays at the final position, after accuracy checking. They are not device
profile durations or the refreshed 128-step loop mean. Final complete 4096/128
device profiles and native operator-row audits remain pending at this snapshot.
Small differences from one timing series are not a robust speedup claim.

`direct_geometry_results.json` retains every exact result/log path and result
hash, journal hashes, all requested policy overrides (base plus per-row delta),
reported policy/program geometry (base plus per-row delta), source factory
defaults, runtime/fixture hashes, timing ranges, accuracy minima and errors.
It excludes large per-step curves. Every candidate records runtime SHA256
`4a9050497d5d4734eb40715d8942adfc33413b958358a209dbbd31c9132a146e`,
which matched the source inspected for this ledger. Requested or reported
program fields are not proof of native dispatch; the operator audits must
verify dtype, fidelity, core count, K block and layout in actual profiler rows.

## Common per-layer policy

| Group | Sliding, layer 0 | Full, layer 5 |
| --- | --- | --- |
| Decode QKV | Direct native FP32 input/output, BFP8 weights, HiFi4 | Direct native FP32 input/output, BFP8 weights, LoFi |
| QKV baseline program | Grid 8x8, K11, subblock 1x1, per-core M1/N4 | Grid 9x8, K11, subblock 1x4, per-core M1/N4 |
| QKV/output weight storage | BFP8/BFP8 | BFP8/BFP8 |
| Decode native SDPA | Grid 8x8, HiFi4, FP32 destination/full synchronization | Grid 8x8, LoFi, FP32 destination/full synchronization |
| KV cache / prefill attention | BFP8 / LoFi | BFP8 / LoFi |
| Router | Generalized top-8 with FP32 row centering before BF16 conversion | Generalized top-8 without centering |
| Expert decode gate-up/down | BFP8/BFP4, LoFi; fused gate-up; grid 11x4, K11 | BFP4/BFP4, LoFi; fused gate-up; grid 11x4, K11 |
| Expert decode activation | Original activation dtype | Explicit BFP8 |
| Expert prefill | BFP4 gate-up/down, LoFi; 32-token groups, grid 11x4, K11 | Same |
| Shared decode | BFP8 gate-up/down, LoFi; fused gate-up; DRAM reader1, K11 | BFP4 gate-up/down, LoFi; fused gate-up; DRAM reader1, K11 |
| Residual / hidden norms | FP32, sharded residual; all guarded sharded norm sites | FP32, sharded residual; factory post+common norm sites |

Direct QKV uses FP32 destination accumulation with packer L1 accumulation off,
no lane decomposition/partial-row reduction, and setup cleanup enabled.
The baseline weights are interleaved DRAM, output is L1; DRAM-reader variants
change the storage/program geometry as recorded in their JSON. Unlisted
parameters retain the captured factory defaults. In the MLP campaign, both
layer kinds additionally use packed QKV grid11x10/K11 with automatic subblock
selection; each MLP row changes only its stated override from that base.

## QKV geometry campaign

Exact result filenames are
`actual_direct_geometry_{candidate}_layer{0,5}.json`; logs use the same stem.
`packed110` requests grid11x10, automatic subblock selection. `separate`
requests separate projections on grid8x4 sliding/grid8x8 full, also automatic
subblocks. `dramrN` uses N workers per DRAM bank; K is the program tile block.

| Candidate | Sliding host us | Sliding min PCC / status | Full host us | Full min PCC / status |
| --- | ---: | --- | ---: | --- |
| base | 1095.624 | 0.995336754 / pass | 1191.896 | 0.995221962 / pass |
| packed110k11 | 1094.523 | 0.995336754 / pass | 1188.826 | 0.995221962 / pass |
| packed110k22 | 1097.998 | 0.995336754 / pass | 1190.876 | 0.995221962 / pass |
| separatek11 | 1101.589 | 0.995336754 / pass | 1209.763 | 0.995221962 / pass |
| separatek22 | 1111.417 | 0.995336754 / pass | 1208.808 | 0.995221962 / pass |
| dramr1k11 | — | L1 error | — | L1 error |
| dramr2k11 | 1116.366 | 0.995295986 / pass | 1202.969 | 0.995284292 / pass |
| dramr3k11 | 1122.874 | 0.995295986 / pass | 1219.755 | 0.995284292 / pass |
| dramr1k22 | — | L1 error | — | L1 error |

All completed QKV rows retain prefill PCC .998669385 sliding/.998846359 full.
The reader-1 errors occur before a completed decode measurement:

| Layer / candidate | Exact observed L1 limit |
| --- | --- |
| Sliding `dramr1k11` | Static CB region ends at 1,481,728 B and overlaps an L1 allocation beginning at 1,378,304 B |
| Full `dramr1k11` | Static CB size 1,641,728 B exceeds 1,572,864 B L1 |
| Sliding `dramr1k22` | Static CB size 2,720,768 B exceeds 1,572,864 B L1 |
| Full `dramr1k22` | Static CB size 3,024,384 B exceeds 1,572,864 B L1 |

These are program-allocation failures, not PCC failures or measured slow
candidates. Successful reader2/3 controls do not establish that all DRAM-sharded
geometries are valid.

## Expert and shared-MLP campaign

Exact result filenames are `actual_direct_mlp_{candidate}_layer{0,5}.json`.
`expert44/22` means requested expert gate-up grid11x4/11x2. `down88` changes
only down-grid to11x8 while keeping gate-up grid11x4/K11. `expert_separate`
splits gate and up; its K value replaces expert K11. `shared_separate` splits
shared gate/up. `shared_rN` changes workers per bank; `shared_r1k22` changes
shared K block to22. Other expert/shared defaults remain as described above.

| Candidate | Sliding host us | Sliding min PCC / status | Full host us | Full min PCC / status |
| --- | ---: | --- | ---: | --- |
| expert44k22 | 1091.684 | 0.995336754 / pass | 1179.688 | 0.995221962 / pass |
| expert22k22 | 1104.955 | 0.995336754 / pass | 1197.879 | 0.995221962 / pass |
| down88 | 1094.664 | 0.995336754 / pass | 1188.484 | 0.995221962 / pass |
| expert_separatek11 | 1163.729 | 0.995336754 / pass | 1247.120 | 0.995221962 / pass |
| expert_separatek22 | 1113.009 | 0.995336754 / pass | 1208.932 | 0.995221962 / pass |
| shared_separate | 1106.943 | 0.995336754 / pass | 1198.931 | 0.995221962 / pass |
| shared_r2k11 | — | Not run | 1185.447 | 0.995221962 / pass |
| shared_r3k11 | — | Not run | 1188.346 | 0.995221962 / pass |
| shared_r1k22 | — | Not run | 1192.615 | 0.995049947 / pass |

Every listed MLP candidate passes all128 actual checks and retains prefill PCC
.998669385 sliding/.998846359 full. Full shared reader1/K22 has the narrowest
margin, .995049947; it remains recorded as a pass at the unchanged threshold,
not rounded into a stronger accuracy claim. Full shared reader2/K11 is the
fastest recorded shared alternative (1185.447 us), versus its packed-QKV/K11
base1188.826 us. Shared reader3/K11, reader1/K22 and separate shared gate/up do
not beat reader2 in this series. These alternatives were not combined with the
expert K22 change in their recorded commands.

Across these rows, fused expert grid11x4/K22 has the lowest recorded host median
for each kind:1091.684/1179.688 us. Separate-K22 is the fastest separate-expert
candidate for both kinds, but remains slower in this series; grid11x2/K22 also
remains slower than grid11x4/K22. These observations justify the bounded native
operator audit in `operator_audit_plan.md`. Final selection still requires the
complete selected-policy accuracy/contracts and device/host reconciliation;
constructor metadata and one-replay audits do not replace those measurements.

This ledger was assembled and checked on CPU from the saved journals, JSONs,
logs and matching source. No runtime changes or hardware execution occurred.
