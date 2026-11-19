# Prefill program tuning after residual narrowing

Source-only follow-up on selected162. No device was opened and no main/shared/frozen source was edited. Artifact: `candidates/prefill_split_role_configs.py`, generated immutably from integrated162 by `make_prefill_split_role_configs.py`.

## Measured context

Profile163 is a pre-narrow134 profile (both correction terms still have physical N16640), not a profile of selected162. Its B32/S33 prefill has4.313 ms across the four primary/residual input matmuls (640/416-row chunks),2.073 ms SwiGLU,0.789 ms attention-output matmul,0.930 ms MLP down, and3.711 ms reshape operations. Therefore it supports targeting correction matmuls but must not be quoted as the narrowed candidate's operation cost.

Parent subsequently reports selected162 B32 prefill24.471418 ms and traced512-run decode1.681551 ms, min PCC0.995542746; integrated167 B32 prefill24.599744 ms and decode1.671731 ms at the same minimum PCC. Profile169 is queued for the narrowed selected path. The older B32/S512 prefill188.435 ms versus fused165.624 ms belongs to134; the long-prefix gap must be remeasured after narrowing before attributing all of it to the remaining selected path.

Prior phase139 HiFi2/single-correction reductions fail the retained quality gate. Larger-row147 configs lose latency. Packed correction pair135 and SwiGLU input-L1 also lose total prefill. None is reintroduced here; BFP8 values, correction terms, math fidelity, packer policy, row cap, recurrence and cache semantics remain unchanged.

## Concrete new geometry

Primary physical N16640 has per_core_N48 on11x10. Narrow residual physical N11520 instead has per_core_N33. Existing automatic subblock selection only considers widths1/2/4, so the narrowed residual uses2x1 at640 rows: two FP32 destination tiles. A legal1x3 uses three destination tiles. Native matmul permits width3 (divides33 and area3 fits four FP32 half-sync DST tiles), as independently validated in the earlier packed-pair study.

The existing prefill config lookup strips `_residual` unconditionally. That prevents tuning the new narrow residual independently of the primary. The candidate first checks the exact projection key, then preserves the legacy stripped-key fallback. It also adds `out_block_h` as a per-projection **ceiling**: choose the largest divisor of current per_core_M that does not exceed it, then clamp subblock height to that chosen block. Arbitrarily short final chunks remain legal. Output width remains the entire per_core_N; the known DRAM WIDTH_SHARDED partial-width reader defect is never used.

Suggested isolated policies under `prefill_projection_configs`:

| Trial | Override | Main CB estimate at640 rows |
|---|---|---:|
| Residual DST utilization | `packed_gdn_residual: {subblock_w: 3}` | 910,336 B |
| Residual K10 | `packed_gdn_residual: {subblock_w: 3, in0_block_w: 10}` | 1,070,336 B |
| Residual K16, height block1 | `packed_gdn_residual: {subblock_w: 3, in0_block_w: 16, out_block_h: 1}` | 1,349,632 B |
| Primary K10 only | `packed_gdn: {in0_block_w: 10}, packed_gdn_residual: {}` | 1,519,616 B primary |

An explicit empty residual mapping in the last row prevents legacy inheritance from the primary key. The source candidate does not select any trial by default; an empty config mapping produces the exact selected162 geometry for every row count1..640.

The CB estimates use the native factory's double-buffered BF16 A, BFP8 B, and shared FP32 output/partial buffer: `2*BH*K*2048 + 2*PN*K*1088 + BH*PN*4096`. Source: `matmul_multicore_reuse_mcast_2d_program_factory.cpp:411-434,758-803`; buffering depth2 is in `matmul_utilities.hpp:23`. Actual ancillary allocations/reserved L1 are additional. Primary K10 is tight and must be measured rather than claimed to fit. Residual K16 with full height2 would need1,550,336 main-CB bytes; height1 reduces that to1,349,632 and has a source-supported K-offset reset in the DRAM reader.

K10 is legal: K5120 is160 tiles, divisible by10. Validation requires divisibility, not a power-of-two block. It reduces K blocks20→16; K16 reduces20→10. This is not a claim of20%/50% speedup: arithmetic count remains unchanged and smaller/full height blocks change reload/multicast cost. Different FP32 spill boundaries can change results, so long correctness remains required.

## Verification and decision

`tests/test_prefill_split_role_configs_source.py` passes3,840 geometry cases, including all logical row counts1..640 and each proposed residual/primary variant. It proves exact default equivalence, legal tail divisors, full output width, per-config DST bounds, exact-key precedence and deterministic source hashes. AST comparison shows only `_linear_raw` config selection changes; all other class methods, helper functions, embedded recurrence kernels and defaults remain exact to162. Python compilation passes. Artifact: `prefill_split_role_configs_cpu_checks.json`.

Compare isolated variants against the same162 baseline first on B1/S512 and B32/S33, then B32/S512 to address the long-prefix gap. Keep complete warmed prefill, final PCC/state and512-step stress results; use profile169 for the new actual operation costs. Decode runtime/configs are unchanged, but prefix changes can propagate into later decode, so gate the whole sequence. No numerical tolerance is reduced and no hardware speedup is claimed here.

A further movement hand-off, not implemented: corrected prefill could place each BF16 activation chunk in L1 once and share it across the two terms, analogous to the selected decode activation sharing. The earlier input-L1 test changed out/down or SwiGLU, not this shared correction input. Independent input-L1 flags currently risk two conversions. For640x5120 the shared activation is6.55 MB total (roughly60 KiB/core over110 cores), so test it with roomy K8 configs rather than the tight primary K10 config. A new candidate and full movement accounting would be needed before selection.
