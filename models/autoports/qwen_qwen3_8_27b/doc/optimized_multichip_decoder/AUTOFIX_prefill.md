# AutoFix: prefill collective candidates

## Failing contract and diagnosis

A valid TP4 fused AG/matmul projection at4096 rows must finish and agree with the accepted reference. H2 on an8x8 grid hung in the input0 sender. `AUTOTRIAGE_prefill.md` derives finite AG readiness production (two increments/direction) versus a receiver reused across eight M blocks (second block requests three). It also identifies the lost current-row base. Current source was checked before the repair.

The native-weight parser issue from the earlier decode investigation is distinct: these prefill columns are explicitly interleaved BFP4. Both H2 layer-kind hangs were captured before termination, followed by bounded reset/list and passing mesh use.

## Ranked repairs and discriminating checks

1. One M/N output block per core avoids receiver reuse and the lost-base path. Adapter policy H16 yields actual H16/H8/H4 for B1 chunks4096/2048/1024. Both chunked AG and mix-only4096 paths complete, preserve state/output PCC and pass traced replay. Raising H merely to8 at4096 would still repeat a K traversal and was not used as a repair.
2. Wide gate/up capacity was adapted with owned internal chunks, BF8 activation/output, and a wider11x8 compute grid with CCL on row8. The11x8 BF8-input path processes full4096 chunks correctly for both layer kinds. BF8 output can need an additional output CB rather than sharing BF16 partials; the retained error records show why that version still exceeded L1.
3. Coherent fused/L1 families retain1280 hidden channels/device through RS, residual and norm.1024-token BF16 and BF8 variants pass; they are slower in complete4096-layer timing. There is no immediate replicated-layout restore hiding their benefit.

Commands and results: `prefill_ccl_fused_smoke_*`, `prefill_ccl_fused_repair_*`, `prefill_ccl_fused_grid_*`; all use `tests.run_prefill_ccl_candidate`, seq4096,steps128,B1. Example full-grid repair: `--policy '{"pf_fused":"ag","pf_fused_block_h":16,"pf_ag_grid":[11,8],"pf_workers":8,"pf_fused_activation":"bfloat8_b"}'`. No shared C++ implementation was modified.

## Candidate/harness repairs

The original persistent-buffer prototype used unsupported `ttnn.Shape` slicing; converting to `list(shape)` repairs it, and both layer kinds subsequently pass persistent AG/RS trials. Adapter v4 fixes a precision-propagation error: the base `_linear` recast supposedly retained BF8 input to BF16. Role-local policy now makes the intended input dtype real, restoring the policy after each call; previous rows are explicitly classified as payload-only.

The harness now frees the eager prefill output after its synchronous host read and the captured output after comparison and trace retirement. Both outputs were unused during decode, and the eager output was unused during capture. Independent review checked no state alias, later consumer or live-trace ownership depended on them. Snapshots/hash receipts in `prefill_source_provenance` identify both lifetime changes. Ordinary trace-cache/allocation gates remain enabled; no acknowledgment or validation bypass was added.

## Result and limits

The caller/configuration repair resolves the measured AG hang and yields correctness/performance evidence instead of rejecting a first API/runtime error. Adapted AG remains slower. Subsequent matched-precision RS trials with the winning BF8 columns/CCLs required H8 to fit BFP8 output; reverse-order paired comparisons showed a small repeatable gain and selected native RS. `prefill_selection.json` records paired values and the55,705,600-byte shared pool. Larger-K/output-subblock closure and final source acceptance are recorded separately.

H16 is not a general larger-batch solution: a promoted AG implementation would need H=per_core_M or explicit owned M blocking. The generic AG factory also hardcodes transfer metadata inconsistent with TP4's watcher assertion. These limitations are preserved as source findings; the production decoder does not select fused AG or disable watcher assertions. Passing normal-run candidates are performance/correctness experiments, not claims that this generic API is watcher-clean.
