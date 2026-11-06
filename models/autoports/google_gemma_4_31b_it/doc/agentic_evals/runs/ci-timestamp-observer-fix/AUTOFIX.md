# AutoFix: CI timestamp precision in private observer

The observer failed freshness arithmetic on valid CI timestamps such as2026-11-06T21:39:06.4653537Z. Its date() helper passed seven fractional digits directly to host Python3.10 datetime.fromisoformat, which accepts three/six fractional digits rather than arbitrary CI precision. The saved42296943044 error independently shows the same failure on21:43:20.5438920Z. Source plus traceback establishes the causal chain without a separate speculative AutoDebug run.

Minimal repair: normalize fractional seconds to six digits solely inside date() before arithmetic, padding shorter fractions and truncating submicrosecond remainder. Original timestamps in parsed log events, state, request records and output remain unchanged. No request matching, boundary, control, no-score or dry-verdict logic changed. Invalid date strings still fail. No dependency or runtime/harness change.

Changed files: private bringup/artifacts/gpqa-first-agent-observer/observe.py and test_observe.py. Added fractional-precision regression (one through nine digits), actual failing timestamp, original-string preservation, freshness delta and invalid-input checks. Existing native-log test now asserts without overwriting its prior actual-local-check.json evidence.

Validation: `python -m unittest discover -s bringup/artifacts/gpqa-first-agent-observer -p test_observe.py -v` passes all9 tests. Fresh in-memory state parses the actual saved42296943044 current.log (4212534bytes, digest in RESULT.json), retaining timestamp21:43:20.5438920Z exactly and measuring6.779697s lag against saved fetch-end21:43:27.323589+00:00. Initial replies remain unconfirmed, no_score_currently remains unknown; no clean-dry claim. Prior observer error/state artifacts remain untouched. Wrappers load the fixed source on their normal next invocation; no process restart/reset or network/device command performed.

Arithmetic remains microsecond-resolution: discarded remainder is less than1microsecond per timestamp, not material for minute-scale freshness and explicitly reported. No claim about current run status beyond the saved snapshot. Device charge0; host effort within0.25h box.
