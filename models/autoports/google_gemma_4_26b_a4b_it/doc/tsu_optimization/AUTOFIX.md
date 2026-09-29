# Decode-only eager-prefill reuse experiment

Diagnosis: `AUTODEBUG_decode_reuse.md` identifies an unnecessary dependency
between serving decode reuse and a prepared prefill trace. Matched baseline
recaptures decode for293–302ms per4K request. Long prefill tracing removes the
recapture but introduces6.1s first-shape setup and exceeds the1GB trace region
at8K. The latter candidate is rejected.

The selected alternative keeps the original1K prefill-trace default and
defaults `GEMMA4_EAGER_PREFILL_DECODE_REUSE` to1 (explicit0 disables it).
The initial experiments opted in explicitly. Adapter metadata records an exact
eager-prefill signature only after its blocking first-token read completes,
then attaches that signature to the first successful B1 greedy decode trace.
The key includes model identity, cache container and every nested tensor's
identity/address/spec, logical and physical prompt shape, dtype/stride/device,
and all hybrid table specifications. Token values and page IDs may change.
Only the same live trace, slot0/B1, cache binding and canonical greedy mode can
reuse it. Mismatches release before eager prefill; no long prefill trace or new
persistent device staging is introduced. Host sampling, resumed generated
prefixes, penalties, logprobs and other batches retain their original path.

Verified so far:

- 109 initial CPU contracts passed, including five temporary eligibility cases;
  after removing the rejected production knob,119 retained contracts pass.
- 15 new CPU key, invalidation and prefill-to-first-decode lifecycle cases pass.
- Reduced0/5-layer allocation-tracked4096/4097 probe passes11 cases, including
  changed tokens, reversed live pages, and4096→4097→4096 transitions.
  Repeated exact signatures preserve the trace, perform0 captures, and create
  0 program-cache entries with misses forbidden. No corruptible scope or
  program-cache tracking exclusion is used.

Expanded Watcher now passes31 cases through16384 context, including the three
shape transitions, with zero warmed program-cache growth. Full-model matched
sync4K serving reaches47.98/47.79TSU versus42.95/43.03, exact eight output texts,
and lower total request latency. Both8K cohorts also improve with exact texts.
All18 official chat greedy replays match accepted controls; C32/N32 completes.
Independent bounded review is `clean-pass` in `review_decode_retention.md`.

Final-source/default qualification is running; remote exact-image qualification
is still pending. Matched async experiments pass local performance and guard
checks but are not covered by the earlier bounded sync review verdict. A fresh
selected-local review is clean-pass in `review_selected_local.md`; remote gates
remain pending. See `work_log.md` for commands/artifacts.
