# Stage Review

Verdict: clean-pass

Independent focused review of stage 06 full-model advisory remediation for
Qwen/Qwen3.8-27B, attempt `795b8327-1563-4a7d-ba4c-022ce9e3254c`, on
2026-10-25. The live `exp26-qwen38-bringup` checkout was based on
`3059d6e58b4`; runtime implementation remains identical to the accepted checkpoint
`8d2790658451b296dd63da7d17a670fc42a61400`. The prior full independent review in
`stage_review_final.md` is reused for unchanged runtime and quality scope.

The reported failure is reproduced and refuted as an implementation blocker.
The stock stage checker still exits **1**, with six justified weight-copy
advisories; this verdict does not claim checker exit 0. No acceptance criterion,
checker implementation, weight-copy count, or runtime policy was weakened.

## Required Work

- None.

## Other Concerns

- Retained representations remain a real memory cost. The agentic-serving skill
  explicitly accepts measured `copy_reason` evidence, and the original stage
  requires preserving the selected optimized decoder strategy. The checker
  deliberately reports every `copies > 1` row with a reason as advisory.
- The prior review's material quality and scope limitations remain: the original
  long prompt reaches its answer later than HF and truncates at 1024 tokens;
  native 262144-token capacity is established at B1, while larger-batch context
  pairs are separately scoped. This remediation adds no model-quality or
  larger-context claim.
- The top-level `stage05_prefill_policy` retains historical stage05 geometry.
  Current stage06 geometry and the required rotary correction are described by
  the current source and prior full review; historical fields are not a new
  claim that current B1 prefill uses out K48.

## Hard-Check Gaps

- The stock checker inspects whether copy reasons exist; it does not verify their
  source, measurements, or arithmetic. This review did that independently.
- The status repair reports reconciled measurements, not independent review
  acceptance. `complete_clean` requires all four batch artifacts, tracking off,
  and at least three records each. Current artifacts meet that condition. The
  existing review remains separately documented.
- No new runtime execution was needed: `git diff` against the accepted
  implementation checkpoint is empty for the model runtime and rotary backend.
  All 12 B1, B8, and B16 benchmark source hashes match current files. B32 retains
  the two historical source differences already investigated in the full review;
  this review does not relabel them as fresh matching-source measurements.

## Anomaly Ledger

- **Observed anomaly:** The runner reports a failed check although the stage was
  previously accepted. **Evidence:** `resume_795b8327_check.log`, independent
  rerun returning 1, installed `06-full-model.check.sh` and
  `scripts/check_agentic_serving.py`. **Affected path:** Stage evidence gate.
  **Control or comparison:** Degeneration and context checks pass; only six
  advisory weight rows remain. **Likely subsystem:** Deliberate advisory exit
  semantics. **Investigation performed:** Read `check_memory`, `shortfall`, and
  maximum-level return logic against the agentic skill and original goal.
  **Resolution:** Controlled and accurately disclosed; no claim of exit 0.

- **Observed anomaly:** `gate`, `up`, and `gateup` are held in two representations.
  **Evidence:** `tt/multichip_decoder.py` construction and `_finish`, stage05
  `prefill_projection_confirm{0,3}_results.json` and
  `prefill_projection_selection.json`. **Affected path:** Multi-row prefill
  versus packed decode/one-row projection. **Control or comparison:** Independently
  recomputed forward/reverse mean prefill host latency: linear packed
  6197.314011 us versus separate 6002.467009 us; full-attention packed
  6634.489517 us versus separate 6271.886523 us. All comparison rows exit 0 and
  have minimum PCC above 0.9998. **Likely subsystem:** Intentional phase-specific
  projection packing. **Investigation performed:** Verified both allocations,
  actual branch selection, and paired measurement arithmetic. These remain
  inherited layer host timings, not new full-model device timings.
  **Resolution:** Controlled; retention preserves the selected measured strategy.

- **Observed anomaly:** `in_norm`, `post_norm`, and `norm_shards` retain full and
  hidden-sharded representations. **Evidence:** Decoder construction at lines
  135–138, `_norm` distributed consumer, stage05 `residual_contract.md`, and final
  B1/B8/B16/B32 artifacts. **Affected path:** Replicated B1–7 decode residuals
  and hidden-sharded prefill/B8–32 decode residuals. **Control or comparison:**
  Actual source dispatch consumes the appropriate representation; measured
  workloads exercise the retained contracts. **Likely subsystem:** Inherited
  optimized residual layout. **Investigation performed:** Traced allocation and
  consumer branches, checked the residual contract, and recomputed all four
  inventories. Each has 1253 tensors; grouped bytes match every ledger family
  exactly and sum to 6,522,077,184 bytes/device. The separate gate/up rows add
  1,604,321,280 bytes/device across 64 layers; full norms total 41,943,040 and
  norm shards 10,485,760 bytes/device. `copies` is explicitly representation
  metadata, not another multiplier. Target pool 1,056,768 exceeds the required
  1,048,576 tokens. **Resolution:** Controlled and accounted; no accidental
  unmeasured duplicate was identified.

- **Observed anomaly:** Agentic status remained `implementation_in_progress`
  after completed clean measurements. **Evidence:** The one-line summarizer
  change and corresponding one-field JSON change. **Affected path:** Evidence
  status reporting only. **Control or comparison:** Current complete batch
  artifacts satisfy `complete_clean`; incomplete evidence retains the previous
  in-progress label. **Likely subsystem:** Stale summary metadata.
  **Investigation performed:** Inspected the full condition and final diff.
  **Resolution:** Fixed without asserting a review verdict or altering runtime.

## Scope Inspected

- **Goal/skills:** Original external `06-06-full-model.prompt.txt`; installed
  `stage-review`, `model-bringup`, `full-model`, and `agentic-serving` skills
  under `.codex/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.21/`.
- **Evidence:** Original and current full-model work log, README, prior full
  review, reproduced check log, current agentic/context ledgers,
  `memory_capacity_plan.json`, all four `benchmark_b*_perf.json` inventories and
  records, prefill/teacher-forcing accuracy, stage05 residual/topology/projection
  selection and raw paired comparison artifacts.
- **Code:** Installed check scripts; decoder allocation, norm and MLP consumers;
  `tests/summarize_full_model.py`; runtime/backend diff and benchmark provenance.
- **Telemetry:** External packet
  `telemetry/packets/795b8327-1563-4a7d-ba4c-022ce9e3254c.json` compared with its
  supplied `.template.json`. Identity and workload fields are preserved. The
  packet is compact and contains no raw captures. Recomputed exact B1
  autoregressive 4096/128/concurrency-1 medians are **433.81215003319085 ms TTFT**
  and **35.186164937945414 tokens/s/user**. All six accuracy fractions agree
  with the original 100-position JSON records: 0.99 top1, 1.0 top5, 1.0 top100
  for each phase. Device-time/roofline values remain unavailable with reasons;
  diagnostic teacher-forcing or reduced-profile times are not substituted.
- **Commands:** Read-only file/source reads, `rg`, `git diff/status/log`, small
  Python JSON/hash/arithmetic analyses, and a CPU-only rerun of the unchanged
  stock stage checker. No device access, benchmarks, profiling, servers, reset,
  or runtime edits. Only this review file was written by the reviewer.

## Residual Risk

- This is focused remediation review, not a fresh hardware acceptance run.
  Unchanged runtime acceptance rests on the existing original artifacts and
  `stage_review_final.md`, whose stated limits remain applicable.
- The stock checker will continue returning advisory 1 while these justified
  representations are retained. Suppressing that result by deleting copy counts
  or altering the checker would make the evidence less accurate.
- Stage-owned remediation changes still require the owner's local checkpoint
  and receipt after this clean-pass; nothing is authorized for push.
