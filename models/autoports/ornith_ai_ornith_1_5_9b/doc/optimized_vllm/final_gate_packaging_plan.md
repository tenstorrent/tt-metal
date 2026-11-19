# Guarded final gates and evidence packaging

The accepted candidate must use the page-refresh failure guard, applied after the preceding server flow finished.160 host tests/hooks pass; prior source-bound results remain historical. The original-source baseline is `stage10_final_before`; accepted targets are `stage10_guarded_after` and `stage10_guarded_tracked`, budget4096/responsive1:2/context262144. If the repeated sweep selects another budget, obtain a matching baseline before comparing performance.

Original-HF batch10 and HTTP story controls are complete; independent blind scoring found0/10 flagged HF samples and0/10 flagged HTTP samples under the fixed rubric. The small-sample limitation is recorded, and the interrupted serial attempt remains preserved with its supersession record. Guarded hardware acceptance completed with exit0; strict final audit passed with zero gaps, canonical promotion completed, and cleanup found no owned server processes or plugin changes. The packaged gate returned advisory-only exit1, dispositioned in `final_gate_disposition.json`; explicit current-output degeneracy passed. [Optimization Stop Rule3](optimization_stop.json) records the reached user wall cap: no new experiments; finish only the remaining required gates once, then audit/review/cleanup/local commits. Stage-review and local commits remain separate final steps.

## Collection and control ordering

This records the required sequence; completed steps are not instructions to rerun them after the cap. Root owns the active flow and remaining gates.

1. Complete the original-HF batch10 control and its mandatory fixed-logit/native batch1 equivalence gates. Retain full256-token samples, effective EOS scoring and explicit batch10 numerical scope. The preparation agent must not inspect labelled outputs. The serial attempt is historical, not an extra denominator sample.
2. Root runs `guarded_validation_frozen.sh` under serialized device ownership. It repeats source-bound budgets512/2048/4096, requires selected4096, runs corrected-stream tracked validation, requires all five guarded device checks, repeats guarded full-model/standalone qualitative controls and runs guarded acceptance using the matching original burst reference.
3. Guarded acceptance collects ten HTTP story controls after canonical sampling/qualitative and before mixed/load/benchmark windows. Source, loaded/native snapshots, exact rendered chat and generation settings remain bound to artifacts. No sampling recipe, context or performance workload changes accompany the control.
4. Export `story_control/batched_run` with `story_control/blind_review.py`. Give a fresh reviewer only anonymous samples and fixed rubric; score all20 before unblinding. A strictly greater than20-point HTTP excess is stage work. Within-margin results require explicit observed rates/limitations, not a universal quality claim. Then complete the final served qualitative review against current standalone/HF controls.
5. Require final shutdown/owned-process cleanup, complete source/config audit, promote canonical artifacts, run stage gates, finish documentation/packaging and obtain independent stage-review. Fix findings and rereview before local commits; never push.

The completed guarded full-profile sampling suite has67 passes,75 skips,zero failures/errors in344.74s. [Guarded skip scope](sampling_skip_scope_guarded.json) binds the exact log/command and every skipped test:71 speculative-only skips, three chunk-prefill fixture skips evaluated separately by mixed/P5/exact controls, and one all-vocabulary chat-logprobs rejection. This is142 collected tests, not142 passing tests. The [pre-guard scope](sampling_skip_scope_pre_guard.json) remains historical.

## Exact final audit and gate commands

After all accepted guarded artifacts and explicit served review exist, from the checkout root:

```bash
model=models/autoports/ornith_ai_ornith_1_5_9b
stage="$model/doc/optimized_vllm"
skill=/home/ttuser/ornith-9b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24
qualitative_review="$stage/guarded_served_qualitative_review.json"
python "$stage/audit_final_serving.py" \
  --before-label stage10_final_before --after-label stage10_guarded_after \
  --tracked-label stage10_guarded_tracked --budget 4096 --sweep-budgets 512 2048 4096
python "$stage/promote_stage10.py" \
  --before-label stage10_final_before --after-label stage10_guarded_after \
  --tracked-label stage10_guarded_tracked --budget 4096 --qualitative-review "$qualitative_review"
if MODEL_DIR="$model" HF_MODEL=ornith-ai/Ornith-1.5-9B bash "$skill/prompts/model_bringup_multigoal/10-optimized-vllm.check.sh" > "$stage/final_stage_gate.log" 2>&1; then
    printf '0\n' > "$stage/final_stage_gate.exit"
else
    gate_status=$?
    printf '%s\n' "$gate_status" > "$stage/final_stage_gate.exit"
fi
python "$skill/runtime/readiness_check/check_degenerate_output.py" \
  --root "$model/readiness_vllm/stage10_guarded_after_b32_untracked/vllm_qualitative_outputs.json" \
  --scope vllm --missing-artifacts critical --json "$stage/final_served_degenerate.json"
```

The served review at that path is complete. Promotion requires `status=pass`, `reader`, exact served-output `source`, all six coherence/topic/repetition/gibberish/language/contamination checks true, and one nonempty verdict per prompt. Do not create these flags before reading current outputs and resolving the story control.

The packaged gate runs degeneracy, context, prefill features and P4/P5 readiness checks. Worst exit wins:0pass,1advisory,2critical,3checker error. Read actual output and fix critical/errors; explicitly disposition advisories. The `if` branch preserves return status, not an automatic pass. The single-file degeneracy check supplements the mandatory model-scoped gate.

## Artifacts and schema checks

| Requirement | What final evidence must prove |
|---|---|
| Source and measured routes | Strict `final_serving_audit.json` must bind guarded manifests, full B1/B32 launch commands, identical before/after configuration/workloads, source/native/environment identity, sampling and workload windows. Device sampling, async decode/deferred read routes, six initial traces and no captures/releases inside measured/load windows remain mandatory. Finish helper edits before audit; later input/script changes invalidate promotion. |
| Primary and capacity | Primary128input/128output/1request/concurrency1 and CI100input/100output/32requests/concurrency unset, with TTFT/TPOT/ITL/throughput before/after. Only primary TPOT supplies headline t/s/u. Keep workload/config beside numbers. |
| Mixed traffic | Corrected-stream tracked and untracked tests must pass supported chunked/nonaligned shapes and preserve traces. Earlier bytewise-reader long-prompt latency is rejected measurement evidence, even where correctness checks passed. Preserve the diagnosis and original files. |
| Sampling/quality | Current full-profile sampling log and command, explicit skip scopes, served outputs, guarded standalone outputs and review, and completed independently scored ten-seed story control. Inherited quality limitations remain labelled; no silent waiver. |
| Full-model comparison | Guarded full32-layer TP4 selected-policy traced decode,128/128/B1/context262144, streamed/resident timings and repetitions, compared to like-work final serving TPOT. |
| P4/P5 | `long_answer_load.json` and `long_context_timing.json`:schema_version1, matching probe, unique nonempty check IDs/status/severity. P4 uses32users×6000output tokens and eval profile. P5 includes timing lengths1024/8192/32768/131072 plus127/128,context262144 and exactly three guarded sweep budgets with launch/result sidecars. Report forecast assumptions separately. |
| Context | Keep `doc/context_contract.json` target/supported262144. Historical512-token fixture/log mentions are scope annotations, not permission to lower serving/eval context. The prior scan's0critical/8advisory counts are historical, not final gate results. |
| Prefix cache | Preserve declared `supports_prefix_caching=not_supported` with `recurrent_state` reason. Do not relabel APC as supported. |
| Prefill burst | Final128input/32output/32requests/concurrency32 result must include matching `serial_reference`,config_matches,mean/p99 TTFT ratios and prefix-cache state. The audit's median ratio does not substitute for packaged mean/p99 fields. Thresholds>0.6p99/>0.5mean are advisory for the optional dense batching optimization; report honestly. |
| Existing multirow boundary | `batched_prefill=supported` evidence `doc/full_model/batch32_full.json` covers generator multirow/solo/order correctness. No newly optimized dense batched-prefill path was adopted; retain that distinction. |
| Limitations | Generic waiver schema uses `items[{probe,check,reason,evidence,original_model,fix_hours}]`. Waivers require eligible checks and actual supporting controls/repair time. Strict audit/promotion require critical pass; do not rewrite failed statuses. |
| Recursive degeneracy | The mandatory gate scans historical `vllm_qualitative_outputs.json` too. Preserve history; never move filenames to evade checks. Fresh current-output review remains necessary. |

## Historical package already complete

[raw_evidence_historical_v1/manifest.json](raw_evidence_historical_v1/manifest.json) preserves893 raw paths as376 deduplicated gzip objects,2,884,338 bytes total. Every original restored and byte/hash checked; originals remain untouched. Largest bundle file428,126 bytes fits the512,000-byte hook limit. [Validation](historical_package_validation.json) and [explicit bundle paths](stage_commit_historical_bundle_paths.txt) record the result. No staging/commit occurred.

Do not rebuild or overwrite that bundle. It excludes live/guarded/story-control/final-after artifacts and mutable final manifests. Retain failed attempts/diagnoses and rejected timing evidence. Exclude duplicate `readiness_vllm/before_stage10_*` backups, copied inherited score tensors, pycache, weights and PID files; do not delete originals.

## Final extension after completion

`prepare_guarded_inventory.py` completed successfully after all hardware flows stopped, writing `guarded_commit_inventory_final/`. It refuses incomplete guarded flows, source drift, a missing/wrong final audit, incomplete HF/HTTP controls or unresolved blind findings. It treats text as opaque bytes and never scores/prints it. It enumerates explicit paths; unchanged historical raw paths are omitted by content hash. New/changed guarded execution, acceptance, control, P4/P5 and relevant diagnostic artifacts remain recoverable.

```bash
python "$stage/prepare_guarded_inventory.py" --output-dir "$stage/guarded_commit_inventory_final"
# Inspect direct_paths.txt, raw_paths.txt and summary.json for ownership/completeness first.
python "$stage/package_raw_evidence.py" pack \
  --paths-file "$stage/guarded_commit_inventory_final/raw_paths.txt" \
  --output "$stage/raw_evidence_guarded_v1"
python "$stage/package_raw_evidence.py" restore \
  --bundle "$stage/raw_evidence_guarded_v1" --destination /tmp/ornith-stage10-guarded-evidence-verify
```

Use fresh inventory/bundle/restore destinations. The inventory helper writes only path lists/statistics; it neither packages nor stages. Root reviews its classification, then the packer checks source stability, exact decompression hashes and512,000-byte object/manifest limits. Verify restored bytes and retain a validation record. If a single compressed object is too large, resolve retention explicitly; do not split it merely to bypass hooks. Semantic historical versus guarded bundles preserve lifecycle boundaries, not an oversized-blob workaround.

The final commit set combines reviewed direct-path lists, every historical/guarded bundle object and manifest, and later gate/review/resolution/cleanup/commit records. Add later records explicitly; do not assume the earlier inventory captured them. Run normal hooks on the exact complete stage-owned set, obtain `clean-pass` or the permitted third-round `pass-with-carried`, commit touched repositories locally and log actual SHAs. Do not change global ignores, stage the entire readiness tree, bypass hooks or push. Raw paths absent in a fresh checkout must be restored before re-running audits that consume them.

## Completed guarded packaging

`raw_evidence_guarded_v1` preserves530 paths as293 unique objects (1,830,624 bytes). Full restoration and original SHA256/size checks passed; every bundle file fits512,000 bytes. `guarded_package_validation.json` records hashes. `raw_evidence_guarded_metadata_v1` additionally preserves27 exact command/gate payloads (36,229 bytes), with independent full restoration recorded in `guarded_metadata_package_validation.json`. These original payloads are omitted from duplicate direct staging where formatting hooks would change hash-bound bytes. Originals remain untouched.

Explicit lists are `stage_commit_final_direct_paths.txt`, `stage_commit_historical_bundle_paths.txt`, `stage_commit_guarded_bundle_paths.txt`, and `stage_commit_guarded_metadata_bundle_paths.txt`; their union is `stage_commit_final_all_paths.txt`. Normal hooks passed on1,050 explicit paths; `final_owned_precommit.json` lists pending concurrent documentation/canonical packaging transforms requiring final hooks. Root owns compact canonical mixed-trace retention with a pointer to its complete raw bundle object. Later review/resolution/commit records must be appended explicitly. Nothing has been staged or committed by the packaging task.
