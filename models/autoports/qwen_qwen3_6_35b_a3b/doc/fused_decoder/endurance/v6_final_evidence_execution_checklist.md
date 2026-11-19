# Final evidence execution checklist — pending source selection

This is a source-only execution plan, not a stage-pass report. Run hardware commands only after the current device owner and host postprocessor close. Preserve prior outputs before using fixed final paths; no command below was executed by this preparation.

## 1. Freeze selection and host setup

- Resolve matched44/85/d810 repeats as described in [selection assessment](v6_d810_candidate_selection_assessment.md).44 already passes strict comparator endurance;85 still needs its own layer0 seeds913/42/154, not transferred44 evidence. Do not use803 as the sole control for a later candidate.
- Adopt exact selected bytes as `tt/fused_decoder.py`, preserve a selected snapshot and record SHA. If d810 is selected, expected SHA is `d8106611f65edccdf069baab44acc9c0088baa709090b017a6a6df75f0a7bb87`. Switch `audit_fused_source.py` to the matching supplemental auditor: currently it imports **decay44**; d810 requires `audit_fused_interleaved_tail_v6.audit`,85 requires `audit_fused_transpose_v6.audit`. Preserve/review this entrypoint change before final provenance. No production change is made by this checklist.
- Activate the current environment and freeze all measured harness/reference sources before final stress. Restore archived inputs only while evidence writers are closed; do not overwrite conflicting originals.

```bash
source python_env/bin/activate
source ../lane-env.sh
export QWEN_DECODER=fused
export PYTHONPATH=.
unset TT_VISIBLE_DEVICES
MODEL=models/autoports/qwen_qwen3_6_35b_a3b
DOC=$MODEL/doc/fused_decoder
E=$DOC/endurance
MOD=models.autoports.qwen_qwen3_6_35b_a3b.tests
python -m "$MOD.restore_fused_evidence"
python -m "$MOD.audit_fused_source"
```

`STOP_QUEUE` is intentionally present during experiments. Parent must preserve/remove that stop request only when ready for the authorized serial final queue. Check no existing device owner before every standalone hardware job; use the established bounded launcher and retain complete stdout/stderr/native close evidence. The checklist's standalone command bodies do not replace that device-owner protocol.

## 2. Comparator completeness and matched selection

For85, execute each missing ordinary snapshot stream (seed913,42,154), fresh or with a matching validated cached HF output oracle; never inject oracle state/output:

```bash
env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python -m "$MOD.stress_fused_snapshot" \
  --snapshot "$E/v6_transport_conv_norm_decay_transpose_integration.py.snapshot" \
  --layer 0 --seed <913-or-42-or-154> --prompt-length 128 --decode-length 8192 \
  --continuation-length 33 --repeats 1 \
  --output "$E/v6_transpose_layer0_seed<seed>.json"
```

Unset Watcher/profiler for these streams; optional `--oracle-cache <matching-cache>` is allowed for comparator selection only. Run unchanged `check_comparator_endurance` with the85 source SHA for these three paths and `v6_transpose_layer34_seed154.json`. The44 equivalent four paths are `v6_transport_conv_norm_decay_layer0_seed{913,42,154}.json` and `...layer34_seed154.json`; their strict validation already passed. Both kinds require all positions, output/contribution PCC≥0.995, tracking including program cache, explicit fused/device/serial provenance and current harness/reference hashes. Full-kind a478 control remains `v4_padded_views_layer3_s256.json`, source `v4_padded_views.py.snapshot`, correctness `v6_full_a478_layer3_seed913.json`.

Run the three rotated clean timing blocks44/85/d810 through unchanged `benchmark_fused_snapshot.py` (10PF/60decode, layer0/S256/context512, seed42 fixed by wrapper), retain every sample and resolve repeatability before selection. Do not infer a repeatable gain from the current0.104% d810-versus85 single-run difference.

## 3. Final correctness, semantic and watcher queue

```bash
bash "$MODEL/tests/validate_fused.sh"
```

This queue is serial and checks `STOP_QUEUE` between jobs. It produces fresh-HF three-repeat seed913 stress for layers0/3 at decode128,1024,8192; extra layer0 seeds42/1548192+33; layer34 seed1548192+33; batch32/boundaries/resume; Watcher-enabled trace reuse; layer0 full-context and full-kind trailing-row context regression; paged extent/padded trace guards; the two synthetic pytest tests; final functional-equivalence runs at S256 and127/128/129/257/4097; source and watcher audits. Fresh final stress must have `cpu_oracle_cache:null` and no `--oracle-cache`; comparator cached runs cannot substitute. All measured source/harness/reference provenance must match the frozen source. Watcher runs and clean latency/profiler runs are separate.

Supplement **selected-source layer34 maximum context**, not included in the script:

```bash
env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python -m "$MOD.long_context" --layer 34 --length 262144 \
  --oracle-cache "$DOC/oracle_cache/v6_layer34_262144.pt" \
  --output "$E/final_selected_layer34_long.json"
```

The shown layer34 cache path exists; the harness must still validate its weights/config/reference identity, or omit `--oracle-cache` to compute it fresh. No cache payload was loaded by this preparation. Required results: prefill262143/262144 and deterministic traced decode atposition262143/context262144, all selected-row PCC≥0.995. Final `linear_long.json` must still be layer0 for the existing checker, and `full_long.json` must retain all64trailing-row checks for both lengths. Supplemental S262143 timing is not a capacity correctness substitute.

## 4. Separate pure performance and manifest

Keep `final_0/3.json` from the queue with every functional-equivalence check. Their premeasurement functional execution differs from historical pure controls; use additional matched files for best-control ranking ([regime assessment](v6_final_timing_regime_comparability.md)). After adoption, set `SELECTED_SNAPSHOT` to the frozen snapshot whose raw SHA equals production, then run each layer separately with no tracking/Watcher/profiler:

```bash
env -u TT_METAL_TRACE_ALLOC_TRACKING -u TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE \
  -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER \
  python -m "$MOD.benchmark_fused_snapshot" --snapshot "$SELECTED_SNAPSHOT" \
  --layer <0-or-3> --length 256 --context 512 --prefill-repeats 10 --decode-repeats 60 \
  --output "$DOC/final_pure_<layer>.json"
```

Create a new completed manifest from `v6_d810_best_correct_pending_template.json`, preserving/rebasing relative paths. **Change candidate paths to `../final_pure_0.json` and `../final_pure_3.json` when stored under endurance.** The old pending template still says `final_0/3`; do not execute it unmodified. Each comparison needs:

```json
{
  "candidate": "../final_pure_0.json",
  "comparator": "v6_transpose_l0_s256_10pf.json",
  "comparator_source": "v6_transport_conv_norm_decay_transpose_integration.py.snapshot",
  "comparator_correctness": ["v6_transpose_layer0_seed913.json", "v6_transpose_layer0_seed42.json", "v6_transpose_layer0_seed154.json"],
  "representative_comparator_correctness": ["v6_transpose_layer34_seed154.json"],
  "required_endurance_seeds": [913, 42, 154]
}
```

Include44 and85 comparisons plus a478 full-kind; append every completed matched control repeat, not only convenient slow ones. Root fields are nonempty `selection_rationale` and `comparisons`. Full-kind seeds are `[913]`, representative list may be empty. Candidate measured SHA must equal final production; comparator measured SHA must equal its distinct frozen source. Layer/kind/length/context/device/versions and PF/decode sample counts must match. Record exact shared wrapper/harness/reference/runtime-audit hashes and false functional-comparison mode separately: checker does not enforce that regime equality. Both pure timing and final functional artifacts retain HF PCC/determinism checks. Do not remove wrapper metadata to pretend a timing diagnostic completes final correctness.

## 5. Final source profiler evidence and schema assembly

For d810 only, after production is adopted and frozen, the reviewed standard warmed capture plan is:

```bash
bash "$E/v6_profile_d810_warmed_s256_plan/capture.sh" 0
bash "$E/v6_profile_d810_warmed_s256_plan/capture.sh" 3
```

Run separately with device ownership; preserve logs through host export completion. For another selected SHA prepare/review a new pinned plan, not edited historical pins. Strict validation must accept the actual captures:1warmedPF,20identical total decode sequences/16measured, four signposts, positive complete durations, no drops/overlap, actual tt-perf-report CSV/text with active-experts8. The full-kind four `to_layout` warnings require fresh capture-specific mapping and metadata review if they recur; historical warning approval does not transfer automatically. Preserve initial failures.

**Path/schema mismatch:** capture plans write under endurance, whereas the gate expects `tracy/layer_{0,3}` and root summaries. After strict validation, assemble exact copies of the final capture's `correctness.json`, adjacent provenance, `ops.csv`, `prefill_perf_report.{csv,txt}`, `decode_perf_report.{csv,txt}` and detailed validator records into preserved/fresh `$DOC/tracy/layer_0` and `layer_3`. Copy each original `capture.log` byte-for-byte to `$DOC/logs/profile_0.log`/`profile_3.log`. Record origin paths and hashes; retain original rawTracy/export artifacts. Do not rewrite provenance to fix a stale source. The checker's profiler provenance requires `source_sha256["tt/fused_decoder.py"]` equal current production, so captures made before adoption cannot simply be relabeled as final.

```bash
python -m "$MOD.summarize_perf"
```

This produces root `perf_summary.json` with **four** measurements containing `latency_us`, and `profiler_completeness.json` with `valid:true`. Per-capture reports instead use `device_time_us_per_pass` and their own completeness schema; copying one directly to root does not satisfy the checker. Keep strict capture validation alongside the aggregate; the summarizer alone is less comprehensive. Reconcile actual final topology/conversions/sharding and every graph-fusing pattern with selected-source evidence.

## 6. Finish gates, accounting and review

```bash
python -m "$MOD.audit_fused_source"
python -m "$MOD.audit_watcher"
python -m "$MOD.check_fused_evidence" \
  --best-correct-manifest "$E/final_selected_best_correct_comparisons.json"
python -m "$MOD.summarize_fused_device_time" \
  --after 2026-11-12T19:38:27+00:00 --output "$E/final_selected_device_time_ledger.json"
```

Run accounting after relevant native sessions and host writers close; active/unproved/launcher-transcript intervals remain excluded and the union is deduplicated. Do not claim8hours from elapsed wall time. The alternative StopRule needs≥70%roofline plus last2kept gains<3%; neither branch is established by a hard-gates PASS alone. No threshold is changed.

Before independent `$stage-review`, update selected-source README/work log/pattern assessment and `context_contract.json`: currently the shared validated entry still names historicalV5, with44's layer34 proof in a pending section. Preserve history; attach actual selected-source layer0/full-kind and representative capacity proofs/hashes, cache/temporary-footprint conclusions, capability262144 and no reduction. Also freeze pure-perf regime/ranking evidence and all rejected options/limitations. `check_all` does not independently enforce context evidence identity, every pattern, StopRule or stage-review verdict; those remain explicit requirements.

Request independent review of the complete selected bundle, fix and rereview findings until clean-pass (or applicable pass-with-carried after3rounds). Only then close archives between writers, run CPU archive/restore verification and repository hooks, commit stage-owned changes locally and record SHA; never push. Archive restore is needed before checker reruns because several expected JSON/text paths are intentionally stored losslessly as gzip. Do not let an intermediate inherited functional PASS or historical hard_gates.json stand in for the final fused result.

All command placeholders must be resolved before execution. This checklist changed only this new memo; existing code, manifests, production and evidence are untouched. No device, Torch/TTNN import or heavy scan occurred. All writes are closed.
