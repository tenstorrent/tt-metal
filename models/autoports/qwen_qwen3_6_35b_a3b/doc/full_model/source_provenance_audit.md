# Historical source-provenance audit

CPU-only inspection at 2026-11-14T16:33:17+00:00. No devices, target imports, raw-artifact edits, gate reruns or inference changes. This audit qualifies historical wording; it does not change numerical findings or waive acceptance. Hash identity is a point-in-time file comparison, not proof of installed native-library identity.

## Resumed-prefill assessment

`AUTOFIX_RESUME.md` was assessed at 2026-11-14T14:09:39+00:00. Its former present-tense statement that all six parity source hashes match current files has been corrected to that original assessment time. The retained [parity artifact](prefill_equivalence/resume_legacy_parity_drain.json) remains the authority for the two-layer experiment and its recorded code. Comparison at this audit:

| File | Recorded SHA256 | Inspected SHA256 | Comparison |
|---|---|---|---|
| `tests/run_prefill_equivalence_probe.py` | `f3b08fdcb19852fe01d0fcabfd93f781561e005e96eb051e5cc6aa774399ad76` | `555887c2e21aedebe24583223cab91b12d1018030fd29eb450d1546b1c594003` | Changed |
| `tt/model.py` | `a0e5eb31075ea40326d4be423e907aca216f43abb25ff86c80b402ccf93e1317` | `a0e5eb31075ea40326d4be423e907aca216f43abb25ff86c80b402ccf93e1317` | Identical |
| `tt/multichip_decoder.py` | `a57112adc86cb65b42f57974c22dc561cf792a384501b866eaafb5c21269923f` | `d08412bdbd7bfbb2bcdef2974b1c3f5b924c8a7ffca4590be80316997cf0bf1b` | Changed |
| `tt/prefill.py` | `1979a8ad1e4c2cf4fd3adb1c50f99ffa047d4d7c18eae5d86c95d33643b58a15` | `388e773878ecaa0638097db2ba15c7c1cd4dc12459986f5203a7134452db10d2` | Changed |
| `tt/prefill_transport.py` | `1976f19160b973baea83739c8ce27f4b4645641651c9f7dfcc0265f40de35bdd` | `bf436f0a8b82daa47bf0ffd6bc06f09b6ad876aaba7abbd555f432e606240061` | Changed |
| `tt/slot_state.py` | `8265153a603c37ff6705e315a4e3efc60942f81930d0da1fd5d2d9db2f135350` | `8265153a603c37ff6705e315a4e3efc60942f81930d0da1fd5d2d9db2f135350` | Identical |

The raw all-40 failed resumed-prefill artifact separately pins six inputs. At this audit, the installed checker, `tt/model.py` and compressed book still match; its runner, `tt/generator.py` and `tt/prefill.py` differ. That artifact did not individually pin the decoder, transport, every helper or native binaries. Its three failed PCC gates remain failed; later capacity evidence does not supply resumed-prefill numerical acceptance.

## Subsequent changes and their evidence

- **Formatting/import order:** `work_log.md` at14:16 UTC and `logs/precommit_check.log` record Black on the resume runner and decoder, and isort on the parity runner, generator, decoder, prefill and prefill transport. The log says autoflake changed nothing. These documented edits explain byte-hash changes without an intended operation-graph change; this audit does not reconstruct every historical file or infer complete semantic equivalence from a hash mismatch alone.
- **Production allocation lifetime:** the later decoder repair explicitly releases `k`, `v` and surviving loop alias `src` after ordered paged-cache updates, before singleton SDPA. `AUTOFIX_CONTEXT.md` and the14:39 work-log entry record the reduced failure/pass pressure experiment. This is a production lifetime edit, not merely formatting and not the earlier test-only legacy attention drain. It preserves the selected computation policy; it is not a proven repair for the three resumed-prefill PCC failures.
- **Later maximum-context evidence:** `context_acceptance/final.json` is PASS on the repaired decoder. All eight recorded non-runner source files match at this audit. The runner differs because its inventory visitor subsequently removed the `callable(value)` exclusion and added a comment, as documented in `AUTOFIX_CONTEXT.md`. That change corrects future reporting only; the raw run inventory is partial, and its actual allocator census remains separate. No second maximum-context pass is implied.
- **Documentation/reporting:** `preserved_policy.md` now includes full-model boundary shapes/communication and Watcher status, so its old hash in the AutoFix assessment is a historical snapshot. Performance harness teacher-forcing/reporting and profiler-preparation RSS observability also postdate older measurements; their later byte hashes do not relabel previous timing artifacts as final-source runs.

## Other historical wording found

`AUTODEBUG_RESUME.md`'s “matches the inspected bytes,” “Current decoder” and Git HEAD statements describe its original pre-format/pre-lifetime inspection. `AUTODEBUG_L1.md`'s “Current relevant SHA-256” and matching-sidecar statements likewise describe the constructor-L1 investigation's earlier tree, before deduplication and later full-model work. These source-pinned diagnostic reports are preserved; neither is a claim that those bytes equal today's files. `work_log.md`'s13:35 “final-source” prefill entry is final relative to that dated run, not to later edits. Prior README accuracy/performance figures are already labeled historical and pending refresh.

`AUTOFIX_CONTEXT.md` and `context_acceptance/memory_audit.md` explicitly distinguish the later reporting-only runner edit; their current inference-match statements were verified at this audit. Runtime/serving audits and the context contract remain with their respective owners. Profiler reports' “inspected” and “post-edit” hashes are snapshots, not manifests for an ongoing or future capture. Raw metrics, outcome thresholds and the exact numerical interpretation of all historical experiments remain unchanged.
