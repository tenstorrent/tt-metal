# Prediction-register and value-column composition audit

Source/CPU result: **composition is valid on the inspected current sources** for grain1 and grain2. After `cumulative_exp_scalar` selection, prediction-then-column and column-then-prediction produce byte-identical reader/compute/writer source, for both resolved grouped-QK values. This is not a combined hardware or performance pass. No main/helper/probe source was edited; no Torch/TTNN import, build, or device operation was performed.

Compact executable-check results and exact input/output hashes: [prediction_column_composition_audit23.cpu.json](prediction_column_composition_audit23.cpu.json). The parent-reported `prediction_smoke_v1` result establishes isolated prediction evidence; it does not prove the composed schedule. The ongoing whole-layer column comparison remains separate and its source provenance was not modified.

## Checked composition and anchors

The audit read `_LOCAL12_READER`, `_fused_delta_row_GDN_COMPACT_COMPUTE`, and `_fused_delta_row_GDN_COMPACT_WRITER` from the current model files using Python AST literal extraction, without importing those model modules. Only the four CPU-safe transformation/partition modules were imported from their file paths. `sys.modules` contained neither Torch nor TTNN at exit.

For `grouped=False` and `grouped=True`, this assembly succeeds:

```python
reader, compute = recurrent_variants.sources(
    base_reader, base_compute, "cumulative_exp_scalar", grouped=resolved_group_qk
)
reader, compute = recurrent_prediction_candidate.prediction_sources(reader, compute)
reader, compute, writer = column_recurrent_runtime.column_sources(reader, compute, base_writer)
```

Prediction's `HELPER_ANCHOR`, `WRAPPER_ANCHOR`, and complete `PREDICTION_LOOP` each occur exactly once after variant selection. Column's fail-closed transformations also find every required reader, compute, writer, and output-padding anchor exactly as expected. Applying either transformation a second time rejects rather than silently duplicating it.

An equally valid order is:

```python
reader, compute = recurrent_variants.sources(
    base_reader, base_compute, "cumulative_exp_scalar", grouped=resolved_group_qk
)
reader, compute, writer = column_recurrent_runtime.column_sources(reader, compute, base_writer)
reader, compute = recurrent_prediction_candidate.prediction_sources(reader, compute)
```

The two complete triples compare byte-for-byte equal. Both also equal the triple generated through `recurrent_column_candidate.column_sources`. Removing prediction's helper/wrapper and replacing its one specialized call with the original loop recovers the column-only compute source exactly. Prediction leaves the column reader and writer byte-for-byte unchanged.

The recommended documented order is variant selection → prediction → column, because each intermediate has the natural two-source or three-source interface. The audit does **not** recommend applying prediction directly to arbitrary variants. Its static requirement is the fused accumulator, fused decay/prediction, and hoisted column-init path. Current `_fused_delta_row_GDN_FUSE_ACCUMULATOR`, `_fused_delta_row_GDN_FUSE_DECAY_PREDICT`, and `_fused_delta_row_GDN_HOIST_COL_INIT` are all true. Supplying `variant="cumulative_prediction_registers"` to `recurrent_variants.sources` is invalid: that diagnostic name is not registered there. Keep variant selection `cumulative_exp_scalar` and apply the explicit transformation afterward.

## Why arithmetic and ownership do not conflict

Column scheduling restricts the outer value-tile index `n`. Each owner still loads all four key-row tiles for every value column it owns into DST0..3; compact K is DST5, prediction result DST6, and decay DST7. Prediction's specialized helper only visits these four local state tiles and their packed K segments. It does not reference a global value-column index or assume ownership of adjacent value columns. Its m0 product initialization and ordered m1/m2/m3 separate product/add fold remain unchanged by the outer-loop restriction.

The composition adds no prediction/update/readout arithmetic beyond the already isolated prediction transformation. It adds no FMA, transpose, dtype change, reduction, collective, output layout conversion, global buffer, CB, or launch. Later delta, update, BF16 persistence rounding, readout reduction, and pack mask sequences remain the existing column compute body. Full-DST capacity and SFPU register allocation are inherited unchanged. Register values do not survive across the existing value-column release, so grain1/2 introduce no new register-lifetime dependency.

Column readers fetch only owned `(head,m,n)` state pages; compute operates only on owned `n`; writers persist only those pages and emit only their output columns, including tail-padding stores. Prediction does not alter any of these boundaries. Partial heads on different workers have disjoint state/output addresses and independent reductions over all K rows. Grouped Q/K preparation remains core-local and refreshes at the local first head or global three-head boundary. Prediction consumes the retained compact K tile but does not change its CB waits/pops or retention policy.

The CPU partition audit covered B1..32 × grain1/2 × grouped request off/on × worker caps1/2/3/8/55/110 on the target110-worker geometry: **768 cases, 608,256 owned value-column units checked**. Static and generated partitions matched exactly. Every case covered all `B*48` value-column units once, with no gaps/duplicates. Every unit retained its four K-row state pages; unit starts/counts obeyed the grain boundary. Grouping resolved identically as `grouped_request and batch >= 3`.

## Static runtime interface and minimum integration scope

`column_recurrent_runtime.column_recurrent` already accepts `source_overrides=(reader, compute, writer)`; these must be the **complete column-aware triple**, not prediction's two-source output. It independently derives the partition and GROUP_QK compile argument. The caller must therefore assemble source with exactly that same resolved grouping bit. Passing a source pair built with `grouped=True` while runtime resolves false (for example B1/B2 or an explicit grouping-off control) is unsafe: reader Q/K production and compute consumption could diverge. Do not cache one grouped reader globally for all batches. State dtype affects existing CB formats/compile arguments, not this source composition; both BF16 and FP32 persistence still need hardware coverage.

Grain is a runtime ownership decision; the composed source triple is identical for grain1/2, but partition/runtime arguments differ. The existing static wrapper already forwards `first_n/last_n` to reader, compute and writer. There is no need for runtime AST extraction, `bind_local`, function rebinding, monkeypatching, or a new writer.

**Only if the combined benchmark wins**, the smallest maintainable optional integration is an explicit disabled-by-default prediction-register flag forwarded by the existing decoder recurrence-policy branch to the static column wrapper. Apply prediction to the already-selected pair inside the wrapper after its local partition resolves GROUP_QK and before `column_sources`; retain automatic source assembly and all present host assertions/CB descriptors. Alternatively, an external setup-time assembly can pass the complete override triple, provided its grouping cache key is correct. A new default variant or global registration is unnecessary. This audit implements neither option and selects no production policy.

## Required hardware discriminator before any selection

Compare three coherent paths with identical raw packed QKV/g/beta, persistence dtype, reset state, grouping, worker cap, output layout and timing boundary: current cumulative recurrence; current column-only grain; column plus prediction at that same grain. First use B1 and B3 at grain1/2, BF16/FP32 state, grouped-on/off, plus one-core B3 to exercise partial heads, user boundaries and retained Q/K. Then extend to the risk matrix, including B9/10/17/31/32 and changed-input queued recurrence.

Retain the existing exact finite output and complete-state equality checks; no aggregate PCC substitution is justified for this exact scheduling family. Run changed-state/input replay, distinct-owner bindings, preallocated output slots, tracking1/skip0 positive control, and watcher checks before uninstrumented timings. Isolated prediction and isolated column passes cannot stand in for their combination. Measure warmed whole-recurrence and real-weight complete-layer boundaries; a reduction in one SFPU loop can lose through occupancy, duplicated Q/K work or schedule effects. Only then consider the optional hook above. No speedup is predicted from this source audit.

## CPU evidence details

The audit was an inline `python3` stdlib script. It used `ast.literal_eval` for the three current kernel constants; `importlib.util.spec_from_file_location` for `recurrent_variants`, `recurrent_prediction_candidate`, `recurrent_column_candidate`, and `column_recurrent_runtime`; direct calls to the pure source/partition transforms; inverse-transform and byte-equality assertions; exhaustive ownership enumeration for the bounded matrix; and SHA256 hashes of all six source files plus both final triples. Results were written only to the new `.cpu.json` artifact. No existing artifacts were overwritten. All assertions passed. The JSON explicitly records `hardware_composition_validated=false`.
