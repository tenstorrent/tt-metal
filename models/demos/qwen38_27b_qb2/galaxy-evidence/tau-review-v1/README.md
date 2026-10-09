# Tau3 failure review, Oct 9 2026 UTC

The official pilot remains **3/12 (25%)**. This audit makes no model calls,
changes no benchmark data or reward, and does not remove failed tasks from the
denominator. The run uses local Qwen as agent, user simulator and assertion
judge; it is not a matched published-reference qualification.

The three naturally completed failures have distinct causes. Message indices
below refer to the original `completed-result.json` simulation messages. Their
SHA256 values, raw-call hashes and pinned upstream source hashes are retained
in [audit.json](audit.json). Full conversations remain on the allocated host.

| Task | Recorded evidence | Interpretation |
|---|---|---|
| 019 | Agent supplied the correct opaque user ID when handing over the dispute tool. The simulated user first invented a tool name, then substituted a name-like user ID on four dispute submissions. Tool responses reported success, but the resulting DB did not match. | Simulator/action validity contaminates this outcome; zero malformed JSON does not imply correct tool arguments. No score repair is justified from this observation alone. |
| 051 | Agent logged a verification timestamp different from the benchmark clock. Later, denial succeeded but the original request stayed pending; three identical resubmissions returned an error. | There is both an agent error and a reproducible upstream tool-state blocker. Neither can explain all other failed tasks. |
| 102 | Task instructions intentionally give misleading customer recollections and require verification. The agent permitted three referrals where the expected final state contained one. The local judge accepted an explanation based on the misleading company-age premise, but the DB check failed. | Genuine verification/action failure, with unreliable assertion-judge reasoning. The final reward of zero was retained correctly. |

The initial review hypothesis that task 102's age was invented by the user
simulator was rejected after reading its original instructions: that premise
was deliberately misleading in the task itself. This correction matters when
attributing benchmark defects versus agent failures.

Task 047 hit the step limit. Tasks 057, 060, 066 and 097 hit task timeouts;
task 026 had the 300-second request timeout. These six outcomes prevent
interpreting 25% as a clean model-only accuracy measure. The audit has not
manually validated every successful answer or every incomplete trajectory.

## Standalone upstream tool reproduction

The unchanged upstream revision is `17e07b1da2bbc0cadfddeea36412686e0604127b`.
The [audit utility](../../tests/tau_failure_audit.py) invokes the actual
`KnowledgeTools` implementation against a fresh synthetic in-memory database:

1. Submit a credit-limit request: succeeds and creates a PENDING record.
2. Deny it: reports success and creates a different DENIED record.
3. Submit the same amount again: fails because the original deterministic
   request ID still exists. The database contains both PENDING and DENIED.

This reproduced without inference or devices. Source inspection agrees:
`deny_credit_limit_increase_5848` inserts a denial under an ID generated using
zero amount instead of updating the original record, while submission's
deterministic ID excludes attempt/time and `add_to_db` rejects existing IDs.
This proves the narrow resubmission blocker; it does not establish that fixing
it alone would make task 051 pass, especially with the wrong verification time.

All completed tool messages have `error=false`, including the literal `Error:`
responses in tasks 019 and 051. The audit records those textual responses
separately. These are semantic tool failures, not malformed wire JSON, and the
prefix count is deliberately not presented as exhaustive error detection.

The first audit invocation failed during import because its shell omitted the
task-owned PortAudio library path. Reusing the benchmark's existing library
path fixed the audit environment; no package or native installation changed.

Reproduce on the allocated host with its existing Tau environment:

```bash
OMP_NUM_THREADS=1 \
LD_LIBRARY_PATH=/home/ttuser/qwen38-artifacts-20261007/tau-environment-v1/portaudio/root/usr/lib/x86_64-linux-gnu \
timeout 60 /home/ttuser/qwen38-artifacts-20261007/tau-environment-v1/venv/bin/python \
  models/demos/qwen38_27b_qb2/tests/tau_failure_audit.py \
  --source /home/ttuser/qwen38-artifacts-20261007/tau-environment-v1/tau2 \
  --results /home/ttuser/qwen38-artifacts-20261007/overnight-v2/native-control/evaluation/tau \
  --output /path/to/new-audit.json
```

Next agentic qualification should keep the official result intact, separate
simulator/judge validity from agent behavior, and measure timeout sensitivity
after serving-latency work. It should use a validated simulator/judge protocol
before claiming a published-score comparison. GPQA remains the current primary
accuracy gate and has not been affected by this audit.
