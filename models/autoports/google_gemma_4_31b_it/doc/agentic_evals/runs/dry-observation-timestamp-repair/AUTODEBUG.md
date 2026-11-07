# Timestamp parsing interrupted observation-summary preservation

Inspection-only diagnosis: the reported Python 3.10 rejection is explained by passing a CI timestamp with seven fractional-second digits to `datetime.fromisoformat`. Replacing `Z` with `+00:00` fixes the timezone spelling but does not fix fractional precision. This affects the private preservation helper's lag calculation, not model execution.

## Direct observations

Source: `bringup/artifacts/stage12-preserve-dry-observation.py`, lines 28–38. Evidence base: `bringup/artifacts/stage12-eval-observation/42339332447/`.

- Both `pre-stop-20261107T062525974258Z/observer.json:193` and `pre-stop-20261107T062539710523Z/observer.json:193` retain `2026-11-07T06:23:55.7885751Z`. Both record snapshot size 35,789,544 and SHA-256 `0c7294d30d61f6a13b2e5335b05f2b562e08ca70a93910d2cef869a741f26baa` (lines 198–199; recorded hashes, not independently recomputed here).
- The earlier directory contains snapshot, observer, fetch and metadata files but no summary. The later directory contains all five files. This matches source ordering: lines 31–34 write the four artifacts before timestamp parsing at line 35; summary writing happens at line 37. An uncaught parsing exception therefore leaves a partial preservation directory.
- The current line 35 applies `re.sub(r'(\.\d{6})\d+', r'\1', ...)` only to the string passed to the parser, then converts `Z` to `+00:00`. It does not assign back into the observer object or rewrite the snapshot.
- The later summary records preservation at `2026-11-07T06:25:39.710523+00:00`, lag `103.921948` seconds, progress 33/40, `dry_pass: null`, `cancelled: false`, and required coordinator review. Its progress timestamp also retains seven fractional digits.

## Cause, repair scope, and precision

The reported failing parse argument was `2026-11-07T06:23:55.7885751+00:00`. The current repair turns it into `2026-11-07T06:23:55.788575+00:00`, compatible with the reported Python 3.10 parser's microsecond precision. The smallest intervention is this parse-boundary normalization; modifying recorded timestamps or upstream logs is unnecessary.

For this timestamp, truncation discards 0.1 microsecond. Subtracting the truncated event time from the recorded preservation time yields the stored `103.921948` seconds; retaining the seventh digit would yield `103.9219479`. Thus the derived lag increases by 0.0000001 second. Generally this truncation loses less than one microsecond of event-time precision. Raw timestamp values remain intact; JSON serialization does not promise byte-identical JSON formatting.

The artifact sequence supports successful summary preservation after the repair. The first exception text, original pre-repair source, Python interpreter behavior, and cancellation at 06:25:48 are requester-reported rather than independently reproduced here. The later recorded preservation timestamp precedes that reported cancellation; `cancelled: false` is a hard-coded helper field, not an independent cancellation-status check. The script itself performs no cancellation.

## Review and limits

Rechecked the diagnosis against write ordering, the retained timestamp, the replacement expression, and the resulting lag. The partial first directory and complete second summary fit the reported failure/repair boundary without requiring a model, device, observer, or fetch failure. No additional defect is established within this bounded inspection. This report makes no dry-run validity or admission claim.

No target execution, probes, devices, APIs, subagents, or source/evidence modifications were used. Only this report was written; root reports were untouched.
