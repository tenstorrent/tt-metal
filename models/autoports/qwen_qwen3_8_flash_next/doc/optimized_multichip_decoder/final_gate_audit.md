# Final gate coverage audit

Closure note: all five preparation actions below are now complete. See
`final_acceptance_summary.json`, `evidence_manifest.json`, `measured_policy_rows.json`
and the final README. The historical audit is preserved to show what was required
before the final runs; its future-tense descriptions are not current blockers.

CPU-only source/plan audit, 2026-11-13. This is a preparation audit, not the
independent stage review or a claim that pending final gates passed. No TTNN
module was imported, device opened, or hardware job launched for this audit.
Selection was still running; final DRAM policies were not yet installed in the
default decoder when inspected. Runtime and test sources were not edited.

## Actions before final acceptance

1. **Freeze and install the selected default before final jobs.** The final
   plans call `probe_multichip_decoder` directly, so they correctly exercise the
   default rather than `apply_dram_candidate`. Candidate artifacts and current
   `dram_integrated_v1_l*.json` passes cannot stand in for those final runs.
   Record each role's decode/rows geometry, weight/activation dtype, fidelity,
   packing, and native fallback policy beside the final artifacts. If the
   default changes after a gate, rerun the affected gate and timing pair.
2. **Run the paired final length timings.** Added
   `plans/final_lengths.json`: six mesh jobs for layers 0/1/3 at logical lengths
   32 and 1024. They preserve the references, capacity, repeats, clean-timing
   regime, and trace options from the six passed `before_mesh_l*_n*.json`
   artifacts; only output/log/provenance names change. Existing final-real jobs
   cover length33 and real-routing B1/B32 but do not replace these paired
   chunk-boundary and sustained-prefill measurements. No baseline rerun or
   candidate application is embedded in the new plan.
3. **Reconcile final memory policy and reservation before the long jobs.**
   `final_long.json` still requests the old 31,203,443,520-byte per-rank target.
   `final_context_memory_plan.{json,md}` already derives prospective updated
   targets, correctly counting 96 GR injections and phase weight deduplication.
   Select/recompute the exact final policy, update the long-plan target, then
   update `doc/context_contract.json` with final steady weights, setup overlap,
   transient allocations, and completed smoke evidence. Do not copy an all-role
   target if final selection retains only some roles. Preserve the existing
   1-GiB default and 2-GiB stress runtime planning envelopes and bounded-stack
   IO requirement. Reservation tests measure representative execution under
   resident-budget pressure, not an instantiated full stack or measured 2-GiB
   free-runtime guarantee. BF16 KV is unchanged in this audit; selecting the
   BFP8-KV candidate would require the separate maximum-context gate.
4. **Close or explicitly account for the grouped changed-input trace gap.**
   Full-decoder B1 traces cycle three changed hidden inputs and PLE token
   inputs. B32 traces restore state and replay the same input batch each time.
   Thus B32 verifies lifetime/repeatability, but does not directly demonstrate
   fresh data through a distinct rows-phase shared DRAM policy. A focused
   tracked B32 run with two diverse batches, refreshing the same input/token
   buffers and comparing each replay with its own eager output/state, closes
   this gap. This matters especially if rows and decode have different packing
   or activation dtype. No stale-buffer defect was found by source inspection;
   the missing evidence is narrower than general trace safety. Editing the
   test is outside this audit's authorized file scope.
5. **Use whole-output PCC for the final decision.** Parent corrected the legacy
   B32 predicate to `whole_output_pcc >= 0.995`; individual rows remain reported
   diagnostics. State, trace, finite-output, tail and stack gates remain in
   force. The historical `real_dram_gdn-compact_LoFi_native.json` exit1 with
   whole-output approximately0.99907 and minimum row approximately0.99447 is
   not a whole-output accuracy rejection. Preserve that artifact and label the
   old gate provenance rather than silently rewriting it.

## Existing coverage and remaining obligations

| Risk | Existing evidence / planned final coverage | Assessment |
|---|---|---|
| Ordered indexer changes floating-point summation order | `parallel_indexer_all_counts_v3.json` passed; all Q1..32, six captured real-HF input windows, BF16 exactness, explicitly gated FP32 bit equality | Strong component coverage; both valid-row and copy-once flags are forced true in the wrapper, matching the new default |
| Valid-row face masks and K tails | Q1 uses five512-element partials; Q2..32 uses1024/1024/512; all-count component exercises masks at1/2/4/8/16/32 boundaries | Final layer3 tail/watcher gates integrate this with real mesh weights and consumers |
| Partial scratch, program cache and trace output lifetime | Ordered component warms all counts, retains a trace while another prepared count executes, changes inputs, checks preallocated output identity and stable program-cache count | Reuse evidence unless relevant kernel/runtime source changes; a single Q1 component run would be insufficient |
| Pipelined lookup NoC scratch reuse and output padding | `pipelined_lookup_component.json` passed: four ranks with different tables, UINT32/BF16/FP32 exact bits, signed zero, changed stripe/column inputs, tile-edge columns, nonaligned H/W, pipeline batches16/32/64; all variant traces retained until teardown | Covers asynchronous read/barrier reuse and bit-preserving stores; no numeric tolerance masks address errors |
| Actual large PLE table addressing | Component tables are small; final layer1 contract/watcher/long jobs use the real packed table | Final layer1 gates remain required even with a component pass |
| DRAM per-rank weight ownership and consumer field ordering | Probe sources actual rank partitions; runtime QSA compacts Q3072/K256/V256, GDN logical Z1536/A12/B12 (or compact width1560), no padded gaps exposed to consumers | Final layers0/1/3 validate real consumers; synthetic projection PCC alone cannot select precision |
| Activation conversion and reshard overhead | DRAM measured groups include optional BF8 input cast, activation reshard, matmul, output restore and field slicing; integrated decoder includes any additional compaction | Report integrated default timing; component matmul speedup is not decoder speedup |
| Logical nonaligned public input | `final_contract.json` and `final_watcher.json` use length33 plus eager tails1..32; long jobs use32769 with capacity32896 | Covers every current <=32 projection logical count and public chunk boundary; no aligned-input requirement is added |
| DRAM shape/phase trace coverage | All logical tails eager, B1 and B32 tracked full-decoder traces; physical height32 and fixed geometry per phase | Component ordered traces cover every count; grouped changed-input caveat is item4 above |
| Final runtime fallback audit | Contract/watcher runs leave `--clean-timing` off; runtime audit wraps model calls | Clean-timing or isolated DRAM candidate runs are not substitutes |
| Inter-layer contract | `--stack` checks pairs0→3,1→0,3→0 with direct producer output; final profiles expose material gathers/reshards | Explicitly document residual ownership/dtype/layout and any measured necessary boundary collective; this audit does not establish a new distributed residual family |
| Diverse MoE B1/B32 and share of step | `final_real.json` uses the real prompt fixture, B32,32 repeats and `--moe-timing` | Report actual route diversity and selected active-expert execution, not only batch size; current serial attention/grouped MLP semantics must be stated |
| Long-context state/cache/page mapping | `final_long.json` tests GDN+PLE layer1 and QSA layer3 from empty at32769, reversed page table, then three decodes | Covers meaningful cache/state kinds; needs revised reservation from item3 and final default |
| Watcher and profiler interference | Final watcher runs all three layer kinds with tracking1 and profiler variables cleared; final timing/profile runs clear watcher | Preserve separation; inspect watcher logs as well as command exit and JSON pass |

## Scope details that prevent misleading claims

- Public decoder prefill splits lengths above32 into chunks. Consequently the
  DRAM group's optional internal >32 chunk helper is not exercised through
  today's public decoder. It has CPU control-flow/compaction checks; test it on
  hardware before exposing it directly in a future caller. It does not require
  an additional long-input path for this decoder stage.
- B32 decoder attention/state still executes one request at a time; the MLP is
  grouped across32 rows. Thus B32 does not automatically prove a Q32 attention
  projection was used. Prefill tails and the32-row prefill chunk exercise that
  phase. Distinguish request batch from projection row count in reports.
- `final_profiles.json` signposts eager warmed prefill33 and B1 decode, not
  traced B32 decode. Prefill33 contains both Q32 and Q1 chunks and therefore
  exposes both selected projection phases in op tables. Use `final_real.json`
  for traced B32 latency/MoE share; label profile windows accurately. Add a B32
  profile only if a selected operation is exclusive to grouped decode or its
  material cost is not otherwise explained by the existing windows.
- Weight-cache sharing uses `(weight dtype, packed)` and intentionally ignores
  activation dtype, fidelity and geometry. Different packing or weight dtype
  retains separate prepared tensors. Fully selected phases must release native
  callable/weight captures; decode-only policies retain them. Setup overlap is
  not steady allocation, and arithmetic overlap is not a measured allocator
  high-water mark.
- Parallel indexer invocation-local FP32 partials are409,600 bytes at Q1 or
  245,760 bytes at Q2..32, plus81,920-byte FP32 result. Keep these in runtime
  reserve accounting, not persistent per-layer weight multiplication. The
  pipelined lookup scratch is per-core L1, not an extra full PLE table.
- The all-count ordered probe's hardware execution is single-device component
  evidence. It cannot independently satisfy the real-mesh decoder requirement;
  final layer3 contract/perf/watcher jobs supply that integration evidence.

## Plan change and execution order

Only `plans/final_lengths.json` was added. Existing references were checked to
exist and all six matching before JSONs report `passed: true`. No runtime/test,
existing final plan, or running `selection_remaining.json` was modified.

After final selection: install default; reconcile memory contract/long target;
run tracked contract gates (including the grouped changed-input supplement);
run watcher separately; run final-real and final-length clean timings; collect
final profiles/tables/CSV and long smoke with the exact selected target. Retain
all command/provenance logs, then reconcile README/work log and request the
separate independent stage review. Any final source change requires the
affected correctness/performance artifacts to be regenerated.

Prepared timing-plan command, for the parent-controlled device queue only:

```bash
python -m models.autoports.qwen_qwen3_8_flash_next.tests.run_multichip_stage_jobs \
  models/autoports/qwen_qwen3_8_flash_next/doc/optimized_multichip_decoder/plans/final_lengths.json
```
