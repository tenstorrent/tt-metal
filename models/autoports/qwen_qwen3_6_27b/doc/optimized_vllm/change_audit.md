# Targeted Stage 10 change audit

Inspection date: 2026-11-17. Baseline: `d67807951e5411a4c1a79144da81b7fbc7852a43`. Inspected current HEAD: `8df7e416f5c8bf097ae84f1a8660cb0ef4e22951`, with no working-tree runtime changes in the three audited files. This is a CPU-only source/evidence audit, not a final stage-review verdict. No TTNN import, hardware command, server, or benchmark was run by this audit.

## Findings

No concrete new runtime correctness defect was found in the three-file diff. The reduced evidence supports the narrow retained change: single-row full attention can use the vector path when an existing chunk begins on a 32-token boundary. It does **not** support a claim that every unaligned continuation now peels to 32, that multi-row continuation changed, or that the four diagnostic JSONs establish complete serving acceptance.

Two evidence/wording boundaries matter:

1. The adapter diagnostic changes page-table column 3 while the active request advances from position 67 to 68 (page 2). It proves a page-only refresh preserves resident token/position state and executes a replay, but this particular case does not prove use of a newly allocated active page. Automatic page growth across boundaries and request reordering must be established by the serving evidence already required for final acceptance; the `page_only_refresh: true` field alone is insufficient.
2. `adapter_optimized_final.log` contains telemetry-enabled page/update lines. It is not execution evidence for the new telemetry-disabled branch. Source shows that branch removes only diagnostic calculation/printing, with no refresh or scheduling operation inside the conditional. Final telemetry-disabled serving benchmarks/checks should be the runtime evidence for that selected mode. No new stale-input mechanism is introduced by the inspected conditional.

## Exact runtime semantics

### C1: tile-aligned vector continuation

`bounded_prefill.py::chunk_intervals` retains the baseline 512-boundary partition formula exactly; only its docstring changed. A public call starting at 31 still processes the old 481-token head through token-by-token full attention until 512. Starting at 33, 481, 962, or 2015 likewise retains the old head behavior. There is no general 32-token peel in the final source.

Inside `run`, vector eligibility is now `absolute % 512 == 0 or (physical_prefill_batch == 1 and absolute % 32 == 0)`. The new branch therefore affects starts such as 32, 384, 480, 896, 992, and 2016 only for the single-row helper. This includes a single request in a nonzero slot of a larger generator. Uniform multi-row helpers retain the original 512-alignment rule. GDN receives the same logical intervals, imports state once before the public call, and publishes once after its iterator completes; the change does not introduce extra recurrent publication boundaries.

For a newly eligible vector interval, the page-table origin is `absolute // 32`. The fill table is kept at `[batch,16]` by appending `-1` entries if the context ends before 16 pages. Only the fill table receives these skip sentinels; the full SDPA page table remains the existing valid mapping. Query length remains the real interval rounded only by the existing tile behavior. With capacity 2048 and start 2016, the actual interval cannot exceed 32, and the fill table has one destination plus 15 sentinels. Prefix rows are not overwritten by an offset rounded down from a non-aligned token position.

Warmup adds a length-32, start-32 single-row invocation using the existing persistent `self.start` and shifted fixed-width fill buffer, then restores the start and fill table. It does not add trace invalidation, capture, retirement, or request-time storage rebinding. The existing public preparation covers lengths 1–512; this extra invocation exercises the new branch rather than establishing a new shape family. The new `tile_aligned` log field is descriptive; the old `aligned` field still means 512 alignment, not actual vector eligibility for every batch.

### C2: page-bound validation

`generator.py::_bind_pages` now calculates the host maximum once and compares it against every non-recurrent cache allocation before uploading any device table. It removes repeated maximum reductions and duplicate inner-loop bound checks. Minimum, shape, cache-object identity, certification, prefix/full upload, and dirty-state behavior remain intact.

The exact unchanged/changed-map contract is:

- A non-forced equal host table returns early only if `prefix_only` is true or the full tables are clean.
- A changed table validates every attention allocation bound before its first upload.
- Prefix-only refresh updates the prepared B1 table, copies the host map, and marks full tables dirty. A later full-table consumer must upload even if the host map is equal.
- Decode calls `_bind_pages(..., force=True)` for either authoritative `reload_inputs` or `reload_page_table`. Such a call intentionally bypasses the equal-map fast return. Therefore “unchanged maps always return immediately” is too broad; resident steps with neither reload flag skip binding entirely, and non-forced clean equal bindings return early.

No bound check was removed for a distinct attention cache. Recurrent state remains outside KV-cache validation, as before.

### C3: telemetry disabled

`generator_vllm.py::decode_forward` now places only `torch.nonzero(page_table != generator.page_host)`, construction of the change list/JSON, and its print behind `self.telemetry`. `_pages`, padding partial tables to fixed capacity, computing the refresh predicate, updating `active_decode_rows` on authoritative inputs, and the generator call stay outside that conditional. `changed` has no other consumer. Telemetry is not used as an input-validity or page-refresh predicate.

With both reload flags false, page/token/position host arguments remain advisory/stale and the generator consumes persistent state. With page-only reload, the table is normalized and forwarded while token/position replacement stays disabled. With full reload, token/position/table state remains authoritative. C3 does not alter any of these cases. Disabling diagnostic `.tolist()` work removes host inspection of CPU scheduler metadata, not a required device completion boundary.

## Re-derived diagnostic coverage

| Artifact | What the artifact establishes | Limits |
|---|---|---|
| `sdpa_tile_start.json` | 52 passing primitive executions, Q6/KV1/head256, BF16 Q/BFP8 KV, Q32/K512, shuffled pages, capacity2048; starts0/32/480/512/992/2016 and valid lengths1/31/32/33, plus long bridges; persistent/alternate start buffers and changed Q; all program counts remain8 | First mesh replica numerical comparison, primitive only, no model or trace acceptance |
| `tile_continuation_final.json` | Nine passing public-call sequences at slot3/capacity4 using real layers0/3/4 and selected policy; exact prefix/outside-write-footprint/peer-state guards and stable cache addresses; downstream GDN state and all logical logits compared with baseline | Reduced context2048, no captured traces, no standalone decode-after-continuation comparison in this probe |
| `tile_uniform_final.json` | B3 `0→32→545` and `0→2016→2048` are bit-equal to old logits/GDN state, with stable cache addresses and passing guards | Demonstrates preserved multi-row path; it is not evidence for enabling tile-offset vector prefill at B3 |
| `adapter_optimized_final.json` | Full preparation, serving-sized virtual context262144/cache `[8196,1,32,256]`, reduced layers0/3/4; resident stale arguments, async deferred-read storage, page-only refresh, prefix/full dirty-table flush, B4/B1 transitions, retained decode during tile-start continuation | Direct adapter diagnostic, not real plugin/scheduler HTTP serving; new active-page consumption is not tested by the page-only case |

The primitive's minimum whole-output PCC is `0.999991917312674`; its minimum per-head/query-row PCC is `0.9999858844858621`. All 52 cases are finite. Start481 is a host eligibility rejection, not an unsupported device invocation.

For the single-row real-model probe, the newly changed `32→545` interval has logit PCC `0.999977682265002`, exact-reference-maximum-set top1 agreement `0.9922027587890625`, and strict lowest-index top1 agreement `0.9863547682762146`. `2016→2048` has PCC `0.9999928510821332`, maximum-set agreement1.0, and strict agreement0.96875 (one of32 rows selects a different exact baseline maximum). These definitions are visibly distinct in JSON and the probe code; no epsilon or top-k tolerance is used to call a tie. The minimum recorded downstream GDN PCC over the changed intervals is `0.9999908570743803`. All other sequences, including the repeated481 scheduler progression, remain bit-equal and thus primarily serve as regression controls.

The single-row artifact names commit `db13d017...`, before the final B1-only eligibility guard. Comparing that commit's helper to current code shows only narrowing the new branch/warmup to batch1; this artifact's physical batch1 path is unchanged. The B3 artifact names `fff3a763...`, whose helper has no diff against current source. C2/C3 do not modify prefill arithmetic, so these results are not invalidated by them.

The adapter artifact records five retained trace IDs (`decode_b1`, `decode`, `sample`, `sample_penalties`, `reset`), zero retirements, and program count23839 unchanged through `0→32→65` and `0→32→513` at slot3 while slot0 repeatedly decodes. It ends with a decode replay at position76. The positive-control log proves tracking rejected a deliberately live late allocation; the probe asserts program-cache tracking was not excluded. Near-context-end fill is covered by the reduced2048 primitive/model probes rather than by a new max-context adapter request.

## Remaining acceptance scope

These diagnostics contain no serving speed claim. Completion-inclusive timings include model/state/cache readback and must not become TTFT or TPOT comparisons. They also do not replace final all-layer sampling/qualitative runs, automatic allocator-driven page-growth control, mixed scheduler traffic with actual chunk offsets, same-config primary/CI benchmark JSON, or eval readiness. The main stage owns those existing/final artifacts.

The retained implementation is narrower than the original AutoDebug repair proposal: arbitrary non-tile-aligned heads and multi-row tile-offset heads remain unchanged. This is a rejected/unimplemented optimization boundary, not evidence of a new correctness fault. Any future general peel or B>1 vector adoption needs fresh recurrent-state and numerical gates.

Commands used were read-only `git diff`/`git rev-parse`, bounded `sed`/`rg` inspection, and Python-standard-library JSON extraction. Runtime source and tests were not edited by this audit; only this report was written.
