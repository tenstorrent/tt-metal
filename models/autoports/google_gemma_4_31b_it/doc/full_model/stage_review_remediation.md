# Stage Review

Verdict: clean-pass

Independent review of stage 06 runner-advisory remediation `d7b18330-d611-4cfb-96da-36be99fea1f6`, 2026-10-31. This verdict covers the evidence-only remediation and its telemetry packet; the broader implementation review remains [stage_review.md](stage_review.md). The live checkout is at `92dd531223c20daded7e3954b61779c3b822a551`, with implementation checkpoint `a17d62a76a1c919b8184a0e69ba60f77de7e866b`. The user expressly permits concrete refutation evidence instead of an exit-0 result.

## Required Work

None. The blocking interpretation of the advisory is refuted by the active skill contract, checker source, implementation and measured evidence. The advisory itself remains valid and visible.

- The installed `agentic-serving/SKILL.md` permits an additional weight representation with a measured `copy_reason`. `scripts/check_agentic_serving.py:3` defines reason-backed shortfalls as advisory and explicitly permits them to proceed to later stages; `check_memory` emits that advisory whenever `copies > 1`. The original stage06 goal requires a memory ledger, preserved capability and independent clean-pass; it does not impose single-copy storage. Neither checker nor ledger was altered to suppress the finding.
- The reviewer independently ran the original `06-full-model.check.sh`: **exit 1**, exactly one tied-weight advisory, no critical findings. Degeneracy and full-context checks pass. This agrees with both the runner's `06-06-full-model.check-1.log` and [stage_check_remediation.log](stage_check_remediation.log). It is not an exit-0 gate pass.
- `tt/model.py:54` uploads the same pinned HF weight as BF16 row-major hidden shards and as a transposed BFP8 tiled vocabulary head. The current embedding device operation explicitly requires BF16 row-major weights (`ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp:34`). These two selected physical representations cannot alias. The work log correctly limits this claim to the current path and does not claim that every possible single-representation design is impossible or slower.
- Recomputed construction deltas in [memory_clean_b32.json](memory_clean_b32.json) are **704,643,072** bytes/ASIC for lookup and **374,341,632** for the head. The independent physical-buffer inventory includes both, and its sum plus the recorded 393,728-byte allocator remainder equals the constructed allocator total. `doc/agentic_serving.json` carries the same complete ledger as [memory_plan.json](memory_plan.json).
- Recomputed prepared free memory is **8,796,461,056 bytes/ASIC**, leaving **7,991,154,688** after the explicitly analytical 805,306,368-byte transient budget. The full-attention pool is exactly **1,048,576 = 32 × 32,768 tokens**; its KV storage plus the 32 sliding caches exactly matches the inventory. Individual context remains 262144. The duplicate representations do not force a declared context or target-pool reduction.
- Recomputed the terminal window from the original raw profiler CSV, using each device's complete FW cycle span and median cycles/ns conversion. The maximum is **1255.1956782812285 µs**, identical to [terminal_attribution.json](terminal_attribution.json). This is reduced-model terminal attribution, not a single-copy comparison or a full-stack device measurement; the remediation states those limits.
- The telemetry packet preserves every exact template identifier and the 4096/128/B1/concurrency-1 workload. Its autoregressive **659.3691239831969 ms TTFT** and **34.21531976214834 tokens/s/user** reproduce the three-sample medians in [performance_b1.json](performance_b1.json). All six accuracy values reproduce match-count/100 in [readiness_accuracy.json](readiness_accuracy.json); sample scope is explicitly token positions from one prompt, not a dataset score. Missing device/roofline metrics remain null with accurate reasons. The full-model skill explicitly prohibits all-layer device-profiler collection.
- All six production SHA256 values match [final_source_manifest.json](final_source_manifest.json). Relative to the implementation checkpoint, tracked autoport changes are only completion-status bookkeeping and the work-log addition. No production or test behavior changed, so the remediation invalidates none of the prior runtime evidence.

## Other Concerns

- No new concern requires stage06 work. The duplicate representation remains a carried memory cost that goal12 must price in its suite-hour survey, as required by the existing skill.

## Hard-Check Gaps

- The checker treats any nonempty `copy_reason` as sufficient for advisory severity; it does not verify its measurements. Source inspection, allocator reconciliation and raw CSV recomputation supply that verification here. Its exit 1 remains part of the record.

## Anomaly Ledger

- Observed anomaly: the runner requested remediation after an otherwise accepted stage because the aggregate checker returned 1.
  Evidence: original and remediation check logs; independent checker reproduction; `check_agentic_serving.py` advisory branch.
  Affected path: stage-completion interpretation of the memory ledger.
  Control or comparison: active skill allowance for a measured copy reason; actual tensor formats, allocator deltas, target-pool capacity and reduced terminal measurements.
  Likely subsystem: orchestration handling of a reason-backed advisory.
  Investigation performed: independently inspected the original goal and checker, verified unchanged source, recomputed memory/profile/performance values and checked the telemetry packet against its template and evidence.
  Resolution: controlled. The advisory is retained; its interpretation as an unmet stage06 requirement is refuted under the explicit user instruction.

## Scope Inspected

- Goal/skills: `bringup/run_contract.json`, `bringup/agentic_evals/inputs.json`, original `06-06-full-model.prompt.txt`; enabled model-bringup startup, stage-review, full-model, agentic-serving and telemetry evidence instructions; review core/router plus performance-claim and comment-hygiene guidance.
- Artifacts: prior stage review; remediation work-log section and both check logs; B32 memory snapshot, memory plan and agentic ledger; terminal attribution and its raw CSV; B1 performance samples, readiness accuracy, current source manifest; exact remediation telemetry template and packet. Read the actual HF and TT AIME completions, which remain coherent capped continuations without mechanical repetition.
- Code: `tt/model.py`, memory snapshot/inventory producers in `tests/measure_full_clean.py` and `tests/run_full_batch.py`, generic embedding validation, stage checker shell and `check_agentic_serving.py`.
- Commands: read-only `git status/diff/log`, `rg`, file reads, host-only JSON/hash/arithmetic/CSV analyses, and the existing artifact-only stage checker. No hardware, server, benchmark, profiler collection, reset or implementation edit was performed. The only reviewer-authored file is this report.

## Residual Risk

This review does not establish a speed advantage over an unimplemented single-representation design. The transient reserve remains an analytical bound backed by reduced-layer measurements and full-stack persistent snapshots. Existing broader stage limitations remain in `stage_review.md`; this evidence-only remediation adds no runtime coverage or new model-quality claim.
