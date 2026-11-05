# Stage Review

Verdict: clean-pass

Independent review of stage 09 runner-advisory remediation at HEAD `66561ef59f28cb3d1454da3aff8e85ca4dd85631`. This verdict accepts the requested evidence-backed refutation alternative; it does not assert that the check exits zero. No hardware, servers, new benchmarks, or subagents were used.

## Required Work

None.

## Other Concerns

- Stage 10 still owes genuine batched prefill, batched-equals-solo evidence, and the burst comparison gates. A self-comparison of the serial baseline is not evidence of batching and should not be manufactured to satisfy a JSON field.
- Preserve the explicit exit-1 result when reporting completion. The installed check documents exit 1 as advisory and exits at the prefill check before reaching the agentic check.

## Hard-Check Gaps

- Runner exit status alone conflates an advisory with incomplete stage work. The specific `prefill-features.md` stage split explicitly permits serial prefill in stage 09 and assigns batching/equality/ratio gates to stage 10, despite broader stage-review wording.
- B1 has one measured request; packet values are valid measurements, not a latency-distribution study.

## Anomaly Ledger

- Observed anomaly: Stage-09 check exits 1 for missing `serial_reference`.
  Evidence: `09-vllm.remediation-reproduce.log`; installed `09-vllm.check.sh` and `check_prefill_features.py`.
  Affected path: Runner completion interpretation.
  Control or comparison: `prefill-features.md` Stage split and Gate sections; generator.py iterates prefill rows. Both burst summaries have SHA256 `ca24bae0d970aa57d84cf24c9c037bfd26dee3683a44a97da1345ef9f0103c2b`. Raw burst results independently confirm 32 completed, zero failed, 4096 input/1024 output tokens, mean TTFT 1987.4401922461402 ms and P99 2638.7717707885895 ms. Summary records cache reset and zero hits.
  Likely subsystem: Advisory propagation, not an established runtime defect.
  Investigation performed: Compared original goal, stage-specific requirements, shell exit propagation, Python advisory sink, implementation and raw measurements. No batched arm or ratio is claimed.
  Resolution: controlled; refutation valid for stage 09.

- Observed anomaly: Separately executed agentic checker exits 1.
  Evidence: `09-vllm.remediation-agentic.log`, `doc/agentic_serving.json`, `pool_plan.json`, tracked runtime audit and growing-traffic evidence.
  Affected path: Weight representations and pre-load fixed KV budgeting.
  Control or comparison: Agentic-serving explicitly allows recorded reasons for these advisories. Revised tracked allocator audit shows 4031009088 bytes/bank across eight banks, matching 32248072704 bytes/device and 1796440576 free; earlier 24608-ID observations are explicitly distinguished from the revised 36896-ID pool.
  Likely subsystem: Documented representation and worker-sizing constraints.
  Investigation performed: Read both advisory lines, permitted-reason contract, pool plan and supporting audit; no critical result is hidden by the shell's early exit.
  Resolution: controlled under existing stage evidence.

## Scope Inspected

- Goal/skill paths: startup run_contract.json and agentic_evals/inputs.json; `09-09-vllm.prompt.txt`; stage-review skill; vllm-integration prefill-features reference; agentic-serving memory/advisory requirements; installed stage checker and prefill checker source.
- Artifact paths: latest work_log.md section, prior stage_review.md, both remediation logs under `bringup/artifacts/multigoal-runs/v3-20261105T172350Z`; burst summaries and raw result; B1 final raw result; final sampling log, mixed_async.json, final_source_manifest.json; pool plan, tracked runtime audit and growing_traffic.json.
- Telemetry: packet `telemetry/packets/1fe2e3d3-6678-46e2-ac37-658276eb9974.json` under that run matches its template's schema, run, stage, attempt, kind and workload exactly. TTFT 754.8702870262787 ms equals raw B1; decode 35.3212917843517 equals 1000 / raw mean TPOT 28.311535322811366 ms. Raw B1 confirms one completed, zero failed, 4096 input/128 output, concurrency one. Unknown device/roofline metrics stay null with reasons, and no synthetic accuracy or burst ratio is inserted.
- Code paths: generator.py serial prefill loop; all ten TT source files checked against final_source_manifest.json, with zero mismatches. Runtime remains unchanged; prior full-suite closure evidence is therefore not invalidated by this documentation remediation.
- Commands run: read-only cat/sed/rg/git inspection and Python JSON/hash analysis. Only this review file was written. No publication action was performed or inferred; prior work log records local checkpoints and no push.

## Residual Risk

This is a focused remediation review, not a rerun of the entire serving stage. Prior independent clean-pass and unchanged runtime retain the earlier correctness scope. Historical selected-policy language drift and single-request metric limitations remain as previously documented. Stage 10 performance work is not waived. The exact telemetry packet is supported by existing raw evidence and clearly labels reuse rather than a new measurement.
