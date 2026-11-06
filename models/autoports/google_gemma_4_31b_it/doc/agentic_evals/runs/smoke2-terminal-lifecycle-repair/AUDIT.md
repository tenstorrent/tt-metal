# Terminal smoke trajectory audit

Valid smoke at two tasks / C2 / B8. Score **0/2 (0%; Wilson95 [0%, 65.76%])**; verified same-version reference unavailable. Nightly10 is eligible immediately, subject to coordinator scheduling/admission. This is a valid score, not a score gate.

Evaluation makespan 2295.204564s (0.637557h); CI device job 0.886111h; submission-to-completion 1.960833h. Fivefold measured makespan suggests 3.188h eval for10 at C2 as an exploratory estimate, not a release projection.

- terminal-bench/llm-inference-batching-scheduler: valid reward0, 5 turns, 436.563550s agent / 457.429368s trial; max completion4669 tokens, max request140.174s. bucket1 sequential_timecost273126882.8496 exceeds270000000.
- terminal-bench/mailman: valid reward0, 42 turns, 2182.950288s agent / 2295.080568s trial; max completion3447 tokens, max request112.143s. announcement delivery absent after successful join; verifier1failed2passed.

Both end via explicit mark_task_complete with no Harbor exceptions/retries, then independent verifiers record task-quality failures. Limits remain21600s agent,3600s request,32768 output. No observed cutoff. CPU/memory overrides absent. All49 trajectory steps (47agent turns) were included in command/observation review. Four invalid-JSON responses recover through stock parser feedback; repeated Mailman checks and false completion are model-quality problems, not validity blockers or limit-ended loops.

No external retrieval or fetched fix is observed in either agent trajectory. Mailman performs task-required localhost SMTP/LMTP operations; external-network quarantine is not inferred from those necessary task-local calls. URLs in installed source/comments are not fetches. Package downloads in verifier/test-stdout.txt occur in official verifier setup after agent completion, not in agent trajectories. No packet-level audit is claimed. Per-response finish_reason is absent; low completion counts and complete final actions support no cap truncation.

Exact full TT 08ee8790bcbcd9e85cb799593b2bbaf40b785134 and TTI 2c4a2628b974acc7ebc7a4fa867db68f0b108019 match collected report metadata. Plugin dispatch manifest pins ca7a18246e4111629ed6cb4a7d1192002749fccf; collected build/runtime metadata corroborates abbreviation ca7a182. Harbor job lock records actual commit e76c762fca5267dfb36e1f900f0a7724ed72e735 version0.25.0. HF and tokenizer revision842da3794eaa0b77d5f08bae87a17459d91ff475, fresh autoport adapter, context262144, native gemma4 parsers, served top_k32, and C2/B8 are verified in runtime/config artifacts. Exact task package digests and trajectory hashes are in audit.json. Original cohort matches; stock terminus-2 agent reports version2.0.0 (agent version, not benchmark version).

No central docs, source, launch or device state changed. The original Docker lifecycle defect is confirmed resolved by this normal two-task run. Later nightly candidate preserves the periodic unique phase export absent from this immutable smoke. Detailed compact evidence: audit.json and adjacent collected task result/trajectory/lock/verifier files.
