# Final profiler findings

Final selected-default captures and advice-enabled reports exist for L0 GDN,
L1 PLE/GDN and L3 QSA, both prefill and decode. The collector reports
`evidence_ready:true`, zero issues, with independent review/commit still pending
(`final_acceptance_summary.json`). This is CPU analysis of existing evidence;
no new device execution or artifact hashing was performed.

## Scope and accounting

Each report uses warmed eager operations between `PERF_PREFILL` or `PERF_DECODE`
signposts. Prefill33 comprises a logical32 chunk followed by a logical1 tail;
decode is logical1. No B32 profiler capture is claimed. `ops.csv` contains
construction/warmup outside these signposts, so its complete-file sum is not a
forward latency. Here each report Global Call Count is joined exactly to its
raw operation; sibling counts divided by256 identify the four device rows.
All selected-policy checks include device IDs0,1,2,3.

`final_operator_summary.json` sums merged worst-rank device rows. This is neither
a critical path nor trace wall time and cannot be multiplied by four to estimate
performance. Segmented eager synchronization creates op gaps; the human reports'
large Total% denominators/gaps are not sustained traced-decode overhead. Device
operation totals help rank kernels. Clean final-default warm/traced timing remains
the headline source (`real_mesh_final_l{0,1,3}.json`).

| Layer | Prefill report rows / summed device ms | Decode report rows / summed device ms | Clean warmed prefill33 ms | Clean traced B1 ms | Clean traced B32 ms |
|---|---:|---:|---:|---:|---:|
| 0 | 764 / 21.034082 | 293 / 1.937342 | 21.417217 | 2.081643 | 52.452465 |
| 1 | 1164 / 25.649672 | 497 / 2.874258 | 25.954905 | 3.136728 | 87.054184 |
| 3 | 1743 / 40.692352 | 409 / 2.624779 | 41.624786 | 2.858865 | 71.598373 |

This is cross-regime attribution, not a subtraction yielding dispatch overhead.
Final B32 and MoE share come from separate clean traces, not from this B1 profile.

## Actual policy reached the measured ops

`measured_policy_rows.json` contains135 exact row joins,42 selected-policy checks,
raw shapes/dtypes/memory/attributes, four-rank coverage, role identification,
shape counts and159 advice clauses with individual dispositions. All42 policy
checks pass. Shape matching and source call sites identify roles; source intent
alone was not used as proof. DRAM matmul raw attributes show output storage grid
x0..7/y0, L1 WIDTH_SHARDED activation, DRAM WIDTH_SHARDED weights, L1 sharded
output, FP32 destination accumulation and packer accumulation enabled. Reader
count is a separate program field; total operation core count is not8 storage
cores. Effective matmul configurations are:

| Role and physical local weight K×N | Measured input / weight / output | Fidelity | K block / readers per bank | Example decode report IDs |
|---|---|---|---|---|
| GR injection10240×32 (logical projection ultimately sliced to1) | BF16 / BF16 / BF16 | HiFi4 |5 /3| L0:19109,19767; L3:40068,41186 |
| Compact GDN Z/A/B2560×1568 (logical payload1560) | BF16 / BF16 / BF16 | HiFi4 |2 /2| L0:19146; L1:30449 |
| Packed shared gate/up2560×320 | BF16 / BFP8 / BF16 | HiFi4 |5 /3| L0:19993; L3:41414 |
| Compact QSA QKV2560×3584 | BF16 / BFP8 / BF16 | HiFi4 |2 /3| L3:40142 |
| QSA WO1536×2560 | BF16 / BFP8 / BF16 | HiFi4 |2 /3| L3:40912 |
| QSA SDPA, head dimension256, selected K2080 | BF16 / BF16 / BF16 | **HiFi2** | SDPA q32/k128 program | L3:40890 |

The SDPA human table displays only `BF16, BF16 => BF16`; the raw `MATH FIDELITY`
and compute attributes prove HiFi2. QKV/WO remain HiFi4 as independently selected.
Ordinary residual shared-down160×2560 and shared-route2560×1 are interleaved
BF16×BFP8→BF16 HiFi4; they are not falsely described as DRAM-sharded winners.
PLE key/value2560×2560 remain interleaved BFP8 HiFi4 (two local projections).

Exact GDN QKV, GR down/up ordered products and QSA ordered indexer are not native
matmul precision knobs. Generic SFPU rows sometimes carry raw `HiFi4` compiler
metadata; that does not show a native tensor-core matmul or justify switching
its arithmetic fidelity. GR exact trees appear as binary/reduce/typecast rows.
The QSA indexer raw compute source is `parallel_ordered_projection_compute.cpp`
with BF16 operands and FP32 output, followed by the exact combine kernel.
GDN QKV is the custom fused exact2560×2560 projection; recurrence is the custom
FP32 `gdn_compute_sfpu.cpp`. Fused active-expert SwiGLU raw inputs show inherited
BFP4 expert weights/BF16 activations, but raw fidelity is blank: do not invent
measured LoFi from that row. The10240-row expert workspace shape is capacity,
not evidence of dense all-expert execution; routing/count evidence is separate.

## Top costs and chunk shapes

| Layer / phase | Largest operation families (summed device microseconds) |
|---|---|
| 0 / prefill | BinaryNg 7604.438; GenericOp 4354.691; Reduce 3883.814; ReshapeCodegen 1218.105 |
| 0 / decode | BinaryNg 422.859; Reduce 308.736; GenericOp 249.184; AllGather 136.620 |
| 1 / prefill | BinaryNg 7985.222; GenericOp 4788.887; Reduce 4075.981; ReshapeCodegen 1672.511 |
| 1 / decode | BinaryNg 593.857; Reduce 380.535; GenericOp 327.491; Matmul 190.915 |
| 3 / prefill | GenericOp 25765.853; BinaryNg 4241.403; Reduce 2153.975; TilizeWithValPadding 1129.612 |
| 3 / decode | GenericOp 729.008; BinaryNg 386.266; Reduce 252.776; Typecast 172.794 |

The largest QSA single row is the exact indexer: prefill32 row33557 costs
24090.856us, tail1 row38712 costs438.424us; decode1 row40354 costs438.470us.
It remains a dominant exact-arithmetic cost after the successful parallelization.
GDN exact QKV costs2273.807us for32 rows and132.425us for the prefill tail;
recurrence costs1748.352/71.301us. L1 adds PLE lookup and two ordinary projections.
Final decode lookup rows29492/29712 cost48.496/27.815us, versus the old one-core
UINT32 lookup family ~133us identified in the starting audit; these are different
lookup roles and must not be merged into a fabricated single-op speedup.

Measured role invocation counts and summed kernel times below separate logical
rows from physical32-row tile padding. This is a per-chunk kernel table, not a
claim that each kernel's logical row count equals the public batch (SDPA and
expert workspaces have different internal shapes). All selected projection
activations report physical M32 for both logical32 and logical1.

| Layer / role | Prefill chunk32 count / us | Prefill tail1 count / us | Decode1 count / us |
|---|---:|---:|---:|
| 0 / GR_injection | 2 / 24.852 | 2 / 24.894 | 2 / 25.014 |
| 0 / GDN_compact_ZAB | 1 / 20.996 | 1 / 20.954 | 1 / 21.163 |
| 0 / GDN_exact_QKV | 1 / 2273.807 | 1 / 132.425 | 1 / 132.566 |
| 0 / shared_packed_gate_up | 1 / 10.296 | 1 / 10.749 | 1 / 10.629 |
| 1 / PLE_key_or_value | 2 / 101.446 | 2 / 101.414 | 2 / 101.480 |
| 3 / GR_injection | 2 / 24.846 | 2 / 25.029 | 2 / 24.876 |
| 3 / QSA_compact_QKV | 1 / 23.961 | 1 / 24.005 | 1 / 23.970 |
| 3 / QSA_exact_ordered_indexer | 1 / 24090.856 | 1 / 438.424 | 1 / 438.470 |
| 3 / QSA_WO | 1 / 13.096 | 1 / 13.313 | 1 / 13.456 |
| 3 / shared_packed_gate_up | 1 / 10.045 | 1 / 10.736 | 1 / 11.231 |

## Every actionable advice item has a disposition

All six text reports contain two advice sections: High Op-to-Op Gap and Matmul
Optimization. CSV advice has five distinct clauses: L1 input, larger K block,
larger output subblock, HiFi2, and the informational “block5 looks good”. The
JSON maps every individual clause to its actual role/row; none is silently
dropped. A suggestion for a residual small projection is not declared tried
merely because another large projection was swept.

| Advice / actual role | Measured disposition or exact Stop Rule2 handoff |
|---|---|
| High op-to-op gap, all phases | Decode already uses captured traces, with final changed-input/output/state replay. Eager segmented gaps include deliberate drains; no throughput rejection from those gaps. Any additional prefill scheduling/fusion is the final-profile opportunity under the recorded Stop Rule2. |
| HiFi2 on GR injection | `dram_precision_b1_gr`/`b32_gr` crossed fidelity and geometry, plus real whole-role DRAM/injection comparisons. HiFi4 selected. A dedicated real whole-decoder GR-only HiFi2 candidate remains untried; exact handoff, not a synthetic quality rejection. |
| HiFi2 on packed shared gate/up | Native packed/split/fidelity and real DRAM shared HiFi2/LoFi were measured; B32 did not beat selected packed HiFi4 combination. See `real_dram_shared_*`, `dram_packed_shared_l0_v2`, combined final candidates. |
| L1 input / output subblock / HiFi2 on shared-down160×2560 and shared-route2560×1 | Final-profile residual-role handoff: compare complete input conversion + projection + next consumer, legal subblocks and independent real-input fidelity. Existing gate/up screens do not cover these roles. Decode kernels sum ~29us/layer, an upper bound on kernel-only benefit before conversion cost. |
| L1 input / block2 / larger subblock / HiFi2 on active-expert weighted sum | Logical K10 pads to one tile, so block2 requires additional internal K padding and matching weights. Current selected-order reduction must preserve whole-output and diverse routing. Defer exact padding/layout/subblock/fidelity experiment under Stop Rule2; ~3.5us decode kernel limits kernel-only gain. |
| L1 input / larger subblock / HiFi2 on PLE key/value | Two2560×2560 projections consume the same embeddings (`tt/ple.py:47,49`). Packing these projections is an additional untried topology opportunity. Defer a packed-versus-separate PLE family with L1 input, legal subblocks/fidelity, both outputs/next consumers and real PLE quality; ~101us combined decode kernel cost bounds direct benefit. Lookup or shared/QKV packing evidence does not measure this PLE change. |
| “in0_block_w=5 looks good” | Informational. Dominant DRAM geometry and1/2/3 readers were swept; GR block40 was separately retried. No additional action requested by this clause. |

These precise residual-role handoffs instantiate the existing final-profile
advice item in `stop_rule_handoff.md`; the owner may merge them into work-log
handoff. Larger subblocks are untried for the listed residual roles, not rejected
on hypothetical L1 pressure. No new performance search is required after the
recorded Stop Rule2; acceptance failures would still require repair.

Final CCL and movement opportunities beyond automatic matmul advice already have
coherent evidence: async/persistent/global and subgroup CCL, packed/separate
projections, activation sharding/dtype, fused/native producer+nextGR families,
whole hidden QSA layouts, selected buffers and KV dtype. The default keeps stream
ownership between layers with no boundary conversion. These observations do
not replace state/context/watcher gates; those are independently recorded in the
collector and tracked/watcher artifacts.

## Final evidence state

Final-default clean latency, logical32/33/1024, tracked all1..32 tails and stacks,
B1/B32 changed-input replay, runtime fallback audit,32769-token PLE/GDN and QSA
context smoke, memory reservation, allocation-tracker positive control and
separate all-device watcher evidence are present. The collector reports zero
issues and no watcher error-like lines. Long tests reserve model/static/state
pressure; this is not an instantiated full model or a physically reserved2GiB
runtime envelope. Review the exact accepted context scope in the context contract.

`optimization_checklist.md` now records completed evidence and this measured-policy
verification. Independent stage-review/fixes/rereview and local stage-owned commit
SHAs remain pending; this analysis does not mark the stage complete.
