# Functional comparison and best-control timing comparability

**Use separate matched pure timings for best-correct ranking, while retaining every existing final functional-equivalence check.** No existing harness/checker/production source needs modification for this plan. This assessment supersedes only the candidate-path recommendation in `v6_d810_candidate_selection_assessment.md` and its pending manifest: use new `final_pure_0.json`/`final_pure_3.json` for best-control timing comparisons, not mixed-setup `final_0.json`/`final_3.json`. Those original final artifacts and all their checks remain mandatory.

## Demonstrated setup mismatch

- `tests/validate_fused.sh:45` invokes `run_decoder` with `--compare-functional` for final layers0/3; line47 does the same for nonaligned lengths. `tests/benchmark_fused_snapshot.py:61` instead passes `compare_functional=False` unconditionally. Thus no command-line omission or provenance interpretation can make the current snapshot wrapper perform the extra functional branch.
- `tests/run_decoder.py:96–120` creates a second functional decoder and cache, performs functional prefill/decode, uploads its setup constants/weights/positions and computes equivalence before warmed timing. The timed prefill loop starts at line123 and traced decode timings at line174. Functional arithmetic is outside both timing windows, but its preceding execution/allocation history differs.
- Line118 deletes `control`, `control_d`, `baseline` and `control_state`. It does not clear the mesh program cache. `tt_metal/distributed/mesh_device_impl.hpp:197` stores the program cache on the device implementation; `tt_metal/api/tt-metalium/program_cache.hpp` defines a device-handle cache holding owned program/workload/shared-variable objects, enabled by default. Mesh cache clearing is explicit (`tt_metal/distributed/mesh_device.cpp:1333–1343`). The harness never calls it between the functional branch and measurement. Deleting a Python decoder therefore does not establish identical cache history.
- There is also direct full-kind device-tensor retention: `control_plan` remains a live local throughout timing. `functional_decoder.py:343–366` puts `prepare_positions` results into each full-attention plan entry; `prepare_positions:240–261` uploads cos/sin/current_pos tensors. Those plan references are not among the four deleted names. Linear plans contain `None` for positions and do not have this particular retention.

These are confirmed source-level lifetime/setup differences. The number of additional cache entries, persistent device allocation bytes, cache reuse overlap, and resulting latency change have **not** been measured. Program ownership alone does not prove every intermediate allocation remains live. No direction or magnitude of performance impact is asserted. In particular, this is not evidence that d810's approximately0.1% advantage is caused by the functional branch: the observed44/85/d810 snapshot runs all disabled it and are internally matched on that point.

## What the existing checker accepts

`check_fused_evidence.py:175–286` validates source identity, device/versions, layer/kind/length/context, repetition counts, uninstrumented serial execution, positive samples, PCC, determinism and comparator endurance, then requires strictly faster traced decode. It does **not** require matching functional-comparison mode, benchmark wrapper identity, run-decoder source hash, or equivalent premeasurement allocation history. There is no explicit exception or measured justification for mixing the regimes; the mismatch is simply not tested.

`check_all:321–339` separately requires `final_0/3.json` to include all functional output/contribution equivalence fields at≥0.995 and to beat the historical functional baseline. Lines356–378 retain equivalent checks for lengths127/128/129/257/4097. Those checks must stay. The manifest's candidate timing path is independently selectable, and `check_best_correct` requires its measured source SHA to equal current production rather than requiring a particular filename. Consequently separate pure-timing artifacts are already supported without relaxing any gate.

## Existing pure timing provenance is matched

Read-only inspection of adjacent provenance confirms all four records below used the same frozen snapshot wrapper and current shared timing/reference/runtime-audit sources, with10prefill and60decode samples and no `unfused_*` result fields:

- `v6_transport_conv_norm_decay_l0_s256_10pf.json` (44f87)
- `v6_transpose_l0_s256_10pf.json` (85ad)
- `v6_transpose_interleaved_l0_s256_10pf.json` (d810)
- `v4_padded_views_layer3_s256.json` (a478 full-kind control)

| Source | SHA256, equal in all four provenance records and current file |
|---|---|
|`tests/run_decoder.py`|`b8242ff98ce14e7465f9aae093f1f85767a73bc4d0905eab5066c472f200bbeb`|
|`tests/benchmark_fused_snapshot.py`|`0cbc2c554be8bd13d3a9264d3cd05f3f2ceda8f29f5f68ab1e23670c8c0a0ec7`|
|`tests/reference.py`|`8c60281ea7849c7463f16f721ff8d553ee3014c7324db49b38fb8fa45f1b2dc3`|
|`tests/runtime_audit.py`|`79462d0d9b834541a2e53852f630ec0fd50267b49fba2562056584fb3b98380c`|

The wrapper fixes seed42, chunk-size=None, profile=False, and compare-functional=False (`benchmark_fused_snapshot.py:54–64`). The same shared harness generates HF reference outputs, checks state/output PCC, applies runtime-fallback guards, restores the prefix state outside each measured decode, and checks deterministic replay. Pure here means no additional functional-decoder execution; it does not mean removing HF correctness checks.

## Minimal adoption and manifest plan

1. Continue the planned rotated44/85/d810 repeats through the **unchanged** `benchmark_fused_snapshot.py` with identical10PF/60decode settings. This retains the already-matched regime. Preserve every run, including reversals/noise. Complete85's source-bound endurance eligibility independently.
2. If a source is selected, freeze production to its exact snapshot bytes. Keep `validate_fused.sh` and `run_decoder.py` unchanged: final_0/3 and nonaligned final artifacts still execute `--compare-functional`, including all≥0.995 functional output/contribution checks. Do not reuse pure timing files as these final correctness artifacts.
3. Add two serial, clean, fresh-process snapshot-wrapper measurements of the **exact selected production source**, writing `doc/fused_decoder/final_pure_0.json` and `final_pure_3.json`. Use the selected frozen snapshot as `--snapshot`, verify its raw SHA equals current `tt/fused_decoder.py` before and after, and use layer0 or3, length256, context512,10prefill and60decode repetitions. Preserve original wrapper metadata, including its diagnostic/acceptance flags: this artifact supplies matched timing only and is not represented as a completed final validation suite.
4. In the eventual best-correct manifest, change linear `candidate` paths to `../final_pure_0.json` and full-kind candidate to `../final_pure_3.json` when the manifest resides under endurance. Keep44/85/a478 control sources and strict endurance fields unchanged. Append every complete matched control repeat, as in the selection assessment. Do not route functional-branch timing into these pure-control comparisons.
5. Before invoking the unchanged checker, review/assert the comparison regime from immutable source/provenance: exact wrapper hash above, exact shared run-decoder/reference/runtime-audit hashes, command basename `benchmark_fused_snapshot.py`, wrapper-enforced seed42/chunk=None/profile=False/compare=False, matching10/60counts, same device/versions and clean environment. Check candidate snapshot SHA equals selected production and no source changed while running. Record this supplemental regime comparison in the final selection evidence; the current checker does not enforce every one of these fields itself.
6. Run the unchanged full checker. It will independently check the pure matched candidate paths from the manifest **and** every existing final functional-equivalence/fresh-HF stress/context/watcher/profile gate. If either class of evidence fails, continue work. The final functional-baseline timing assertions remain as written; this plan neither removes nor interprets them as proof of equal allocation histories.

Exact additional timing command pattern, after selection and after the active device owner closes (not executed by this source task):

```bash
QWEN_DECODER=fused PYTHONPATH=. python_env/bin/python \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/benchmark_fused_snapshot.py \
  --snapshot <exact-selected-snapshot> --layer <0-or-3> --length 256 --context 512 \
  --prefill-repeats 10 --decode-repeats 60 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/final_pure_<layer>.json
```

Use the existing device-owner/clean-environment protocol, unsetting Watcher, device profiler and trace-allocation tracking before the command. Placeholders must be resolved to the selected pin; this is a plan, not a launch authorization for this agent.

## Alternative and rejected shortcuts

A new separately pinned snapshot wrapper with `compare_functional=True` would align controls with existing final_0/3 and preserve all functional checks. It would require fresh44/85/a478 control timings in that mode and repeated d810 comparisons, plus provenance recording the new wrapper/mode. It is valid but unnecessary given the existing support for separate pure timing. Do not silently change the frozen wrapper or relabel old false-mode artifacts as true-mode runs.

Deleting more Python locals or calling `clear_program_cache()` is not a proved substitute for matched workloads: clearing affects both functional and candidate caches and may alter what the first measured prefill includes. Such changes require separate source review, warmup definition and remeasurement of both arms. They are not part of this minimal plan.

No existing source, manifest, device evidence, production, or checker was edited. Only this new assessment was written; no Torch/TTNN import, hardware operation, large evidence load or heavy CPU task was performed. All writes are closed.
