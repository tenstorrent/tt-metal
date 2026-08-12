# Independent full-model stage review

Verdict: `clean-pass`

The final fresh reviewer found no required work, no other concern requiring
stage remediation, and no material hard-check gap. It independently inspected
the full-model and generator code, multichip changes, shared readiness helpers,
the CCL writer repair, root/stage documentation, context arithmetic, all
final-source accuracy and trace logs, qualitative outputs, profiler evidence,
watcher reruns, and final device health.

The review specifically recomputed the canonical full-context live set as
31,378,609,024 bytes/rank before the explicit 1.5 GiB runtime reserve, leaving
1,189,509,248 bytes/rank at context 32,768 and a page-aligned ceiling of 34,464.
It verified that the physical gate retains/represents both prefill and decode
weights for all 40 layers, all 40 BFP8 K/V cache pairs and endpoints, the
200,000,000-byte-per-bank trace reservation, and the runtime reserve while
executing batch-32 prefill and position-32,767 decode.

The reviewer also verified the resolved Sampling1D watcher failure, qualitative
HF/TT output pairs, split-trace device token feedback, changed page-table replay,
and the canonical/rejected common sampler comparison. Remaining characteristics
are documented limitations rather than stage failures: long prompt ingestion
after the optimized prefill window is sequential device decode, and active-ETH
watcher inspection is unavailable under the installed firmware while worker
watcher and live fabric/CCL execution pass.
