# Private incremental observation helper

Host-only helper: `bringup/artifacts/gpqa-first-agent-observer/observe.py`. It performs no API requests, dispatch, cancellation or dry decisions. Keep state and output private. Existing coordinator capture commands and immutable source/config/image proof remain prerequisites; this helper does not replace them.

Capture successive complete cumulative job-log snapshots with known fetch start/end UTC times as in `observation-handoff.md`. Select an exact, invocation-specific evaluator boundary string **before its initial requests** by inspecting the private log; a progress marker appearing after requests is too late. Do not use an arbitrary earlier server-start marker if preflight traffic intervenes. For a fresh invocation/state:

```bash
python bringup/artifacts/gpqa-first-agent-observer/observe.py \
  --snapshot bringup/artifacts/ACTUAL_ARM/console-SNAPSHOT.log \
  --state bringup/artifacts/ACTUAL_ARM/observer-state.json \
  --output bringup/artifacts/ACTUAL_ARM/observation-SNAPSHOT.json \
  --boundary-marker 'EXACT VERIFIED EVALUATOR BOUNDARY BEFORE REQUESTS' \
  --clients 4 --year 2026 \
  --fetch-start 'ACTUAL_FETCH_START_ISO_UTC' \
  --fetch-end 'ACTUAL_FETCH_END_ISO_UTC'
```

For subsequent snapshots use the same state and boundary parameters but a new output filename. Full prefix SHA verification detects replacement/rotation/truncation; failures leave the prior state intact and require separate observation state, not silent continuation. Same stale snapshot produces no duplicate events and its lag grows. Unterminated lines and split UTF-8 are buffered. State can contain incomplete raw JSON, so it is private even though output omits content/token arrays. The16MiB pending-body ceiling records incompleteness instead of accepting a prefix. Invalid UTF-8 fails explicitly. Do not edit snapshots to splice gaps.

The first C unique Received IDs after the selected boundary are recorded with timestamps. Response matching requires a complete valid JSON object, the stock logger's extra closing wrapper, and a single finished chat choice. Multiline bodies with CI timestamp prefixes are supported; a new logger event interrupts an unfinished body. Truncation warnings mark coverage compromised. Missing/truncated response records never become observed replies. Full JSON hashes, sizes, IDs, finish reasons and empty-visible status are compact metadata; no raw prompts/reasoning/tokens enter output. Non-JSON metrics/streaming response bodies are ignored as unsupported.

Timestamps prefer the CI UTC prefix; local INFO timestamps explicitly assume UTC and the supplied year. Inspect negative/future lag or clock mismatch before use. Output includes actual supplied API fetch duration and fetch-end minus newest timestamped log event, independent HTTP-status counts and successful-call progress observations. Request IDs are not attached to HTTP200 or progress counters, so all `client_acknowledgement` values stay unknown. Four observed server bodies do not prove four client acknowledgements. This is scoped to accepted nonstreaming GPQA logging and requires independently established isolated evaluator traffic and successful request admission.

`no_score_currently` **always stays unknown**, even when no markers are found. Score markers are observations requiring context review, not automatic assertion that aggregation executed. New model logs may help establish recency but cannot eliminate REST publication lag or cancellation race. Root owns any strengthened conclusion, final dry criteria and cancellation. `all_initial_server_replies_observed=true` is only a server-body observation; check `source_coverage_compromised`, errors, identity isolation and remaining gates independently.

Host validation: eight tests passed, including actual accepted local four-probe logs (4/4 initial IDs with complete empty-visible native response JSON), incremental partial final line, multiline CI body, split UTF-8, explicit truncation, nonappend snapshot rejection, missing boundary, stale-snapshot lag and separate HTTP/progress semantics. Actual-log test prepends a synthetic boundary only around the known private four-probe segment; it is a parser control, not a normal CI observation or dry proof. Private tests/output: `bringup/artifacts/gpqa-first-agent-observer/{test_observe.py,tests.log,actual-local-check.json}`. No device use or new live measurement.
