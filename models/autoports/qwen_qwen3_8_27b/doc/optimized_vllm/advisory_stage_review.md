# Stage Review

Verdict: clean-pass

Independent review of Stage 10 advisory remediation for `Qwen/Qwen3.8-27B`,
attempt `324904ee-c02d-4ce7-a174-6b2a14ee5d8a`, on 2026-10-25. Reviewed the live
checkout at `0ae60620d22022d98222455075c77f33675ebda2`; current changes are the
remediation documentation and this report. The standalone TT plugin remains
clean at `d5ee19c6754905e7a9a69c79d9e2ee4b99b84295`.

The user's explicit alternative, reproduce and fix to zero **or refute the
reported failure with concrete evidence**, is satisfied by the refutation.
The checker still exits **1**, and its genuine advisory limitations remain.
This verdict does not turn that result into exit 0 or establish the unmeasured
32-by-32768 planning workload. The original full-stage `stage_review.md` remains
applicable to unchanged runtime and accepted evidence, whose hashes were checked
again here. The stage owner must still checkpoint the remediation locally and
record its commit receipt; this review does not perform that step.

## Required Work

- None for this remediation and its supplied acceptance alternative.

## Other Concerns

- I independently ran the supplied shell checker and each component against
  the current artifacts. Combined exit: **1**. Context: **0**, degeneracy: **0**,
  agentic: **1**. The combined output contains no critical findings. The
  nonzero component produces exactly six weight-copy advisories and one pool
  advisory. `check_agentic_serving.py` explicitly defines a shortfall with a
  recorded reason as advisory, with its cost measured in goal 12. The stage
  shell propagates the highest component result. The refutation explains this
  documented advisory behavior; it does not allege a serving regression or
  claim that the warnings are absent.
- The context scanner reads text logs recursively and repeats earlier checker
  messages. The originating `plugin_state_after.log:88-109` entries are passing
  `test_chunked_prefill_policy.py` tests. The test source constructs `_FakeModel`
  with a 16384 fixture. The actual accepted B1, B8, B16, B32 performance and B32
  tracked manifests all specify native context 262144. Their 44 source hashes
  apiece match the current checkout: **220 of 220 entries match**.
- The weight-copy explanations have implementation and measurement backing.
  `multichip_decoder.py:163-170` allocates replicated/sharded norms and separate
  gate/up plus packed gateup weights; `_norm` and `_mlp` select the relevant
  phase/layout representation. Recomputing the forward/reverse means in
  `prefill_projection_confirm{0,3}_results.json` gives separate versus packed
  prefill of **6002.467009 versus 6197.314011 us** for layer 0 and
  **6271.886523 versus 6634.489517 us** for layer 3, at B1/4096 input. All compared
  rows exit 0 with minimum PCC at least 0.9998216779. These support the inherited
  MLP representation choice; the remediation changes neither weights nor ledger
  accounting. The checker deliberately emits an advisory even with a reason.
- The pool shortfall is real. Both the recorded B32 configuration and the raw
  selected `after_b32_events.jsonl` preparation event report 289 scheduler IDs,
  block size 4096, attention allocation 10303307776 bytes/device, recurrent
  snapshots 11122458624 bytes/device, and free memory after preparation
  **4731160064 bytes/device**. Thus each extra dense pool ID costs
  `10303307776/289 + 11122458624/289 = 35651584 + 38486016` bytes/device.
  The stated 353-ID lower bound needs 64 more IDs, or **4744806400 bytes/device**,
  exceeding that free-memory snapshot by **13646336 bytes** before scratch.
  Usable planning capacity `(289 - 96 - 1) * 4096 = 786432` is below
  `32 * 32768 = 1048576`. This is a bound for the current dense layout and
  reservations, not a universal hardware/model limit. Native single-request
  context has not been reduced.
- The measured agentic fixture remains explicitly smaller: 32 first turns of
  4130 tokens, then 32 second turns of 4163 tokens. Those lengths appear in the
  raw request records, and the selected event stream retains four captures and
  zero retirements. The ledger's 131072/132160 reuse fraction is for that
  fixture. It is not evidence of successful 32-by-32768 traffic.
- The new telemetry packet matches its adjacent template exactly for schema,
  run, stage, attempt, kind and workload identifiers. Its 4096-input/128-output,
  physical-B1/concurrency-1 result is the existing selected measurement:
  **447.39251397550106 ms** median TTFT and
  `1000 / 27.621631607721287 = 36.20350941616578` decode tokens/s/user.
  The raw result completes one request with zero failures and exact token counts.
  Device times and rooflines are null with the profiler-prohibition reason;
  dataset accuracy and the planning workload are not invented. No new benchmark
  is required for this artifact-only remediation.

## Hard-Check Gaps

- The shell's exit 1 does not itself distinguish explained limitations from
  missing acceptance evidence; inspect the component classification and reasons.
  Here there are no missing-evidence or unexplained-shortfall critical rows.
- The broad text scan cannot distinguish a fake-model unit-test fixture from a
  serving configuration. The source test and accepted command manifests supply
  that distinction. No checker edit or deletion of historical logs was needed.
- Allocator checkpoints are not runtime peaks, and fixed single-request
  benchmarks do not prove planning-workload capacity. The remediation preserves
  both limitations and makes neither stronger claim.

## Anomaly Ledger

- Observed anomaly: the runner reports nonzero despite the prior stage pass.
  Evidence: supplied `10-10-optimized-vllm.check-1.log`, new
  `10-optimized-vllm.advisory-reproduction.log`, and this reviewer's rerun.
  Affected path: stage evidence aggregation.
  Control or comparison: component exits 0/0/1; zero critical findings; seven
  explained memory advisories; original accepted source/evidence unchanged.
  Likely subsystem: advisory-to-stage-exit propagation.
  Investigation performed: inspected shell and Python checker branches, fixture
  tests, current manifests, raw allocation event, and weight-selection results.
  Resolution: controlled and refuted under the user's explicit alternative;
  exit 1 and real planning limitations are retained.

- Observed anomaly: repeated historical context warnings mention 16384.
  Evidence: `plugin_state_after.log:88-109` and recursive log scanning.
  Affected path: context advisory reporting.
  Control or comparison: `_FakeModel` test configuration versus five native
  262144 accepted manifests and `doc/context_contract.json`.
  Likely subsystem: text scanning without fixture provenance.
  Investigation performed: traced every originating warning to its passing
  unit-test case and checked current manifest hashes.
  Resolution: controlled; no reduced served-context finding.

## Scope Inspected

- Original goal: external run root
  `/home/mvasiljev/agentic-research-data/experiment-26/bringup-logs/qwen38-27b/20261023-astra-medium-exp26-clean/10-10-optimized-vllm.prompt.txt`;
  supplied user remediation alternative and profiler/benchmark prohibitions.
- Skills: installed model-bringup 0.1.21 `stage-review`, `model-bringup` startup
  and evidence-gate semantics, `agentic-serving`, and relevant serving,
  optimization and device-usage requirements from `vllm-integration`, `optimize`
  and `tt-device-usage`.
- Remediation artifacts: README/work-log diff, `advisory_recheck.json`, original
  and reproduced checker logs, individual component logs, original independent
  `stage_review.md`, and the new telemetry packet and adjacent template under
  the external run root's `telemetry/packets/` directory.
- Supporting evidence: all five accepted source manifests; all **32** accepted
  artifact-index checksums, with no mismatches; raw/normalized selected primary
  benchmark and accepted before configuration; context/agentic ledgers; raw B32
  allocation and lifecycle events/requests; batch configurations; projection
  selection and its forward/reverse measurement rows. Spot-read actual selected
  packaged/shared qualitative outputs; their unchanged checksums and prior full
  independent output/control review remain the full quality acceptance basis.
- Code: stage shell checker, `check_context_contract.py`,
  `check_agentic_serving.py`, plugin fixture tests, and relevant decoder weight
  construction and consumption branches. Generator/adapter hashes equal those
  in the original review:
  `91c8ad90268bcbb42185e9c3ad4cb143d798917316c93ba7b464410812ccacbb` and
  `315bd0fe2ac27fb0c9cbfc6efa655664ddee53bfb8c880d0c10d9b9f2b746307`.
- Commands: read-only git/rg/cat/sed and standard-library JSON/hash/arithmetic
  analysis; supplied shell checker and its three artifact-only components.
  No hardware, server, benchmark, profiler, reset, implementation edit or
  subagent was used. This requested report is the reviewer's sole write.

## Residual Risk

- The checker still returns advisory exit 1. An orchestrator that insists on
  literal exit 0 must recognize the user's accepted refutation; this review
  does not change runner policy or claim that it was changed.
- The planning 32-by-32768 workload, runtime memory peaks and long-context
  multi-turn eviction behavior remain unmeasured. Goal 12 must assess the
  recorded shortfall's workload cost.
- The prior review's other explicit limits remain: performance parity rather
  than established speedup, long mixed-prefill pauses, capped qualitative
  responses and inherited native-context evidence. No runtime or accepted
  measurement change in this remediation invalidates that review.
