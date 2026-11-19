# Round2 diff-only review brief

Stage7 optimized-full-model, ornith-ai/Ornith-1.5-9B, baseline
ebd521f744bb61f787a164f6b207c6defb8dd306. Follow-up scope only.
No hardware work remains running. Do not re-review the whole stage.

Previous Required Work and Other Concerns, verbatim:

## Required Work

- P1: Reject resident replay windows that cross an unmapped causal page before submitting any trace.
  Evidence: `tt/generator.py:973` adds `decode_resident`, but its only position/page admission check is the configured-context comparison at lines 993–997; it never checks `state.host_tables`. The existing reload preflight explicitly permits `-1` for unused pages (lines 801–802) and checks mappings only through the position being submitted (lines 814–820). Consequently, with context 64 and table `[[0, -1]]`, an ordinary valid decode at position 31 succeeds and leaves `host_positions=(31,)`, `position_advances=1`. `decode_resident(1)` then passes its context check and submits position 32 against page `-1`. This is a source-derived counterexample; the unsafe device call was not run. The actual full-attention path calls `paged_fused_update_cache` (`tt/multichip_decoder.py:1272`). Its writer reads the page as an unsigned physical block and computes an address without an unmapped-page guard (`ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp:108`); only a negative **position** skips the update. The sibling `decode_forward(..., reload_inputs=False, reload_page_table=False)` also omits resident context/page validation in `_decode_preflight`.
  Why this matters: Partially allocated scheduler-owned page tables are admitted by the explicit cache contract. A valid initial decode can therefore be followed by an accepted public resident call that produces invalid cache addressing, rather than a host rejection or a request for the next mapped page. Fully mapped benchmark tables do not cover this case.
  Required next step: Validate every active slot's entire requested decode window against the authoritative resident host position and mapped causal-page prefix before queuing work. Preserve inactive rows and zero device transfers. Apply the same admission invariant to ordinary one-step resident decode so the sibling path cannot bypass it. Add focused host checks for the exact last mapped position, first unmapped position, a multi-step window crossing a hole, inactive rows with unmapped tables, and unchanged counters on rejection. Then exercise a valid mapped page-boundary transition and changed-table continuation against explicit-feedback control with allocation tracking, retaining trace IDs/program counts. Rerun affected resident performance/contract evidence after the guard change; the invalid-address call itself must remain host-only negative coverage. No decoder arithmetic or broad accuracy/geometry sweep is requested for this admission fix.

## Other Concerns

- Primary warmed TTFT is honestly reported as 90.017 to 92.049 ms. The three samples overlap (before 89.992/90.017/92.559, after 92.049/95.309/89.602 ms), and the separate capacity-matched head control does not establish a head-induced regression. Retain the reported primary number and its limitation; do not claim a TTFT improvement.
- The shared-suite output remains cap-limited. I read all six current texts and both current HF/TT autoregressive pairs. The reasoning is coherent, prompt05 completes its translation, and the other cap-limited outputs are not evidence of completed answers or runnable code. Exact current/prior token, prompt, seed, and sampling equality holds for all six shared prompts. The existing ten-seed blind prompt04 control records HF 4/10 versus TT 1/10 content failures and 10/10 cap-limited outputs on each side; the corresponding limitations carry `original_model: "same"`. These are controlled model limitations, not Required Work.
- Local commit creation and recording the resulting SHA remain the explicitly planned post-review bookkeeping. The user prohibited pushing and vLLM work. Neither action was performed by this review.


## Changes since round1

- Production: only tt/generator.py; see resident_guard_diff.md. CPU causal-prefix
  window guard called by bulk resident API and ordinary resident preflight.
  No decoder, kernels, sampler, trace body, dtype, cache layout or allocations changed.
- tests/full_model_resident_test.py: four new host cases,49 host tests total.
- tests/full_model_resident_page_probe.py: real layers0/3 B3 tracked boundary,
  sparse admission, changed table, explicit feedback, no-host-boundary checks.
- New results: resident_page_guard.json, optimized_guard_final.json,
  batch32_guard_final.json, serving_guard_final.json; logs with corresponding
  names under logs/. Initial diagnostic assertion error remains in
  resident_page_guard_probe_assertion.json and resident_page_guard.log.gz.
- Docs: model root README; stage README/work_log/runtime_audit/perf_summary;
  final_audit explicitly marks earlier generator hashes as historical.
- Command journal is now commands.jsonl.xz, lossless XZ to stay below repository
  file-size cap. Other logs/windows remain gzip. archive_manifest updated.
- Stage/degenerate gates,49 host tests and pre-commit pass. Commands and logs:
  resident_guard_host_final, resident_page_guard_final, optimized_guard_final,
  batch32_guard_final, serving_guard_final, stage_gate_guard_final,
  degenerate_guard_final, health_guard_final, precommit_guard_pass2.
  Exact argv/env/source hashes in commands.jsonl.xz. Device owner check empty.

## Requested review

Read the stage-review skill and this diff-only brief. For each previous finding,
report closed/open with evidence, then inspect only changed files for new issues.
Treat docs as claims; re-derive new metrics from JSONs and validate provenance.
Keep prior accepted accuracy/profiler/qualitative findings scoped to unchanged
compute bodies; no broad reruns were requested. Preserve known HF-controlled
limitations. Source guard negatives must not issue invalid-address device work.

Primary current B1 metrics come from optimized_guard_final.json. Earlier
92.049ms TTFT is retained and no material TTFT gain is claimed. Earlier all-active
B32 control is explicitly labelled as predating the guard. Masked B32 rerun is
batch32_guard_final.json. Exact decoder policy and262144 context unchanged.

Local stage commits and SHA logging remain post-review bookkeeping; no push or
vLLM integration is authorized. Write stage_review_round2.md in this directory
with the required verdict sections. No implementation edits, device use,
servers or further delegation. Read-only local artifact-analysis scripts are OK.
