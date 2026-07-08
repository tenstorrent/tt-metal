# Gotchas — sparse MLA perf report

Traps hit while building the report, each with the symptom and the fix. Read before trusting any number.

## Data / Tracy

**Summary CSVs are overwritten per `(scenario, mode)`.** Every `run_device_profiler` run rewrites
`deepseek_v32_<mode>_mla_perf_<scenario>.csv`. A saved cold CSV can silently be from a *different*
experiment (different `DS_PERF_CHUNK`/cache) than warm/long, so a sparse-vs-dense comparison built from
the top-level CSVs may be apples-to-oranges.
- *Symptom:* sparse cold total ~380× dense cold; `..._cold_by_iter.csv` cache-depth step 25,600 vs 1,280.
- *Fix:* verify each dump's real parameters from its **raw report**, not the filename:
  `LayerNorm INPUT_0_Y_PAD[LOGICAL]` = per-chip seq (the chunk); count the signature op
  (`SparseSDPAOperation` / `RingJointSDPADeviceOperation`) per `DEVICE ID` = number of forwards. The
  matched run usually still exists in an older `reports/<timestamp>/`; re-derive its summary (see
  `scripts/recover_cold.py`) instead of re-running the board.

**Dumps must postdate the last code change.** `git log` the last commit touching `tt/mla/` and the perf
test; compare to dump dir mtimes. HEAD being newer is fine if it doesn't touch MLA. Otherwise re-run tracy.

**Device-collapse with `merge_device_rows`, never by hand.** Multi-chip rows are one-per-(op×chip);
summing over-counts wall-clock ~8×. The convention (compute=max = critical path, collectives=avg) lives in
`models/tt_transformers/tests/test_utils.py::merge_device_rows`. Slice to the signpost region first with
`tests/nightly/sdpa_perf_utils.py::post_process_ops_log(has_signposts=True)`.

## Per-call attribution

**ttnn relabels CCL/rope ops.** The full-prefix KVPE gather (`all_gather_async`) is logged as
`AllBroadcastDeviceOperation`; a minimal reduce-scatter can log as `ReduceScatterDeviceOperation`; the
indexer query rope (`rotary_embedding` interleaved branch) logs as `RotaryEmbeddingLlamaDeviceOperation`
even though the code path is the indexed one. Match with **alias sets**, not exact codes.

**Do NOT advance the walk pointer on a pinned async anchor.** Pin once-per-forward structural ops
(SparseSDPA, RingJointSDPA, IndexerScore, Topk, FastReduceNC, BinaryNg, NLPConcatHeads, Typecast) to their
node so same-code CCL noise can't displace them — but **label only**. The profiler can list `topk` (async)
*before* the logits CCL round-trip that program-order precedes it; if pinning topk advances the pointer,
the trailing in-order ops (and the whole tail: gather, sdpa, o_proj) race into the wrong blocks.
- *Symptom:* the 7 ms `AllBroadcast` gather lands in the sdpa block; a stray `Matmul` lands in the cache
  block. Block sums still equal the total (every call counted once) but the **distribution is wrong**.
- *Validate:* block sums == scenario total to the ns AND unique anchors == summary op totals to the cent.

**Composites/relabels have real time but inferred wiring.** `MeshPartition`, `Copy`, `FillPad`,
`TilizeWithValPadding`, `UntilizeWithUnpadding`, extra `Slice`/`Concat`/`Permute` come from
`to_memory_config` / CCL internals / `nlp_create_qkv_heads`, not a distinct source line. Place them in
their issuing block with real duration, but mark them "composite" (dashed) — don't fabricate a tensor edge.

## Graph correctness

**Verify against source, cite `file:line`.** Delegate the breadth (both mode paths) to agents, but the
graph must be checked against `mla.py`/`indexer.py`, not proxied from an existing report. Reconcile each
block's emitted op codes against the per-forward Tracy counts; flag anything you inferred vs saw.

**Branch-specific dispatch facts change the graph.** `_needs_head_to_seq_reshard` is False for DeepSeek-128
(heads/chip=32) so the GLM head→seq transpose in `_sparse_mla` does NOT fire; indices stay SP-sharded so
the `mesh_partition` reshard also doesn't fire. Check the actual config before drawing those ops.

## UI / HTML

**Class-name collisions in the single stylesheet.** A `.caveat.info` card inherited an unrelated `.info`
tooltip-button rule (`display:inline-flex; width:19px; border-radius:50%`) and collapsed into a 19px box;
`overflow:hidden` on the panel had been *hiding* the breakage. *Fix:* scope utility classes uniquely
(`.infobtn`). When a fix "makes it worse", suspect a selector matching more than intended.

**Preserve pan/zoom across same-layout redraws.** Opening the node drawer re-renders the graph (for the
selection highlight); guard the viewBox reset with a layout signature (`mode|scenario|view|expanded`) so it
only resets to fit on a real layout change, not on selection/theme toggles.

**Reset a shared "was-dragging" flag on every mousedown**, before any early-return for buttons — otherwise
a stale drag flag from a previous pan swallows the next click on the expand (＋) button.

**No browser in this env.** Validate JS through a stubbed-DOM node harness (see `validation.md`). `node`
may only be at `/proj_sw/user_dev/bsheikh/nodejs/bin/node`.
