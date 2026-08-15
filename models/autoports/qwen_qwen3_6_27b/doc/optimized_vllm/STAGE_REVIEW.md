# Optimized vLLM stage review

Verdict: `clean-pass`

Required work: none.

The fresh rereview inspected the user contract, relevant skills, context
contract, optimized-vLLM documentation and all before/after/chat artifacts,
current readiness logs, inherited stale/non-aligned evidence, full-model and
datatype controls, the adapter/full generator/common sampler, and the sibling
plugin async controller, runner, platform gating, RNG fix, and regression test.

The comparable primary and CI workloads are substantiated. Primary decode
improves from 14.138 to 16.157 t/s/u and reaches 92.5% of the comparable 17.467
t/s/u full-model path. Context 262144, non-aligned support, persistent traced
state, stale page-table behavior, on-device split sampling, nonblocking replay,
deferred async readback, and clean process teardown have supporting code and
live evidence. No prohibited profiler was collected.

The initial review's only blocker was corrected: raw-completion qualitative
text is diagnostic-only, while `artifacts/after_chat/` contains final optimized
chat-template rendering/token IDs, six greedy and six fixed-seed sampled
outputs, and an exit-0 degeneration check. All outputs were read and compared
with prompt-correct datatype/full-model and prior serving controls.

Controlled anomalies:

- All 12 final chat outputs reach the 256-token cap during visible reasoning
  before a concise answer. Controls show the same reasoning-first behavior; it
  remains an explicit short-budget presentation limitation.
- Earlier raw-completion sampled text contained malformed phrases. Prompt-format
  misuse was isolated and that artifact is not used for quality gating.
- Some startup attempts hit device-0 Ethernet heartbeat stalls before model
  execution. Bounded reset plus mesh smoke recovered them; successful gates
  were rerun and final process audits were clean.

No runtime sampler-refresh counter was invented. Source inspection, focused
contract tests, two repeatable primary measurements, live sampling, and the
before/after result adequately connect refresh suppression to the measured
adapter path.
