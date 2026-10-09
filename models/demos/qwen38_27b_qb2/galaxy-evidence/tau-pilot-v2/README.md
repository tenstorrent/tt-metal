# Completed bounded Tau3 pilot

The corrected twelve-task banking pilot completed all attempts in **32m36s**:
**3/12 successes (25%)**, with all twelve tasks retained in the denominator.
This is a short integration/agentic pilot, not a matched published reference
score. Qwen supplies the agent, user simulator and assertion verifier; manual
review of the simulated conversations and judgments is still pending.

It made 298 model calls and emitted 277 tool calls. No malformed tool-call
arguments or output-budget truncations were recorded. Successful tool parsing
does not imply task success. The five incomplete trials also prevent treating
the headline reward as a clean measure of model quality.

| Outcome | Tasks |
|---|---:|
| Reward 1, user stop | 3 |
| Reward 0, user stop | 3 |
| Reward 0, 60-step limit | 1 |
| Per-task 20-minute timeout | 4 |
| One 300-second model-request timeout | 1 |

The request error was verified in `task_026/raw-calls.jsonl` at row index 10:
`Timeout`, 300.119971 seconds, 8,192-token output budget, 22 message entries.
The upstream log identifies `litellm.Timeout: APITimeoutError`. This is the
summary's one `infrastructure_error`; it is not evidence of malformed JSON
or proof that a device failed. No retry or score repair was performed.

`summary.json` and `protocol.json` preserve numeric results and the executed
protocol. Raw conversations stay on the host beneath
`/home/ttuser/qwen38-artifacts-20261007/overnight-v2/native-control/evaluation/tau`.
The persistent queue advanced to physical Galaxy HTTP measurements afterward.
The earlier pilot with missing simulator data made zero model calls and remains
a separate setup failure, not a previous valid score.
