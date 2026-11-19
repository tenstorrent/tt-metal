# Decode page-refresh failure self-check

CPU/source self-check, not independent stage review. No production runtime or test file was edited while final acceptance was active.

The final manifest matched all 23 runtime files before preparation. The six production diffs and related host tests were inspected for borrowed output lifetime, changed/unchanged decode metadata, and prefill input cache ownership.

## Concrete defect

The changed-row decode upload loop commits `host_tables` only after all copies succeed. If one row succeeds and another transfer fails, some device rows have changed but the old host snapshot remains. A retry using the old table skips all copies. CPU injection into the actual production AST reproduced zero retry copies, device first row `[0,1,7]`, and host first row `[0,1,-1]`.

The proposed patch permanently marks the affected DecodeState invalid on a page-refresh or forward-input-remap exception. Public decode, replay, resident decode, capture, prepare, and internal execute/admission paths reject the invalid state before further work. The message instructs callers to recreate the generator. No retry or reset sets validity back to true; the dataclass initializes it only for a new state. Normal close remains available to drain queued work and release traces. This avoids assuming that an interrupted scheduler transition can be retried: tokens, positions, sampling state or slot ownership may already have changed before the page failure.

## Other writers audited

- Initial page uploads and preparation restore copies run before publication in `self.states`; failure does not publish a reusable state.
- `_remap_forward_inputs` copies page snapshots and replaces page rows before committing host metadata. It has the same partial-failure risk; the candidate poisons that state on an exception too.
- Successful changed-row refresh and remap retain existing tensor addresses, copy decisions, counters, scalar bounds, trace selection and nonblocking execution.
- Prefill masks, row indices and attention fields already invalidate metadata before transfer, commit after success, and have partial-copy retry tests. Their owner lifetimes and all post-construction writers were checked; no additional stale-cache defect was found.
- Deferred reads enqueue owning copies before any later trace, and unqueued borrowed outputs reject the next decode/replay. Completion masks use immutable submission state. Existing host and device controls cover output ownership, changed token/position inputs, changed/unchanged pages and sampled-token feedback. The previously fixed replay guard remains intact.

## Artifacts and tests

- `page_refresh_failclosed_candidate.patch`: proposed runtime change plus new `tests/vllm_page_failure_test.py`; not applied.
- `page_refresh_failclosed_generator.py.source`, `page_refresh_failclosed_test.py.source`: frozen proposed files.
- `page_refresh_failclosed_red.log`: current source fails the partial-refresh poison regression.
- `page_refresh_failclosed_green.log`: six proposed failure/guard tests pass against the source artifact.
- `page_refresh_failclosed_focused_host.log`: 53 CPU tests pass from an isolated temporary source/test copy: new failure cases, page-row reload, resident window, seed continuity, and deferred-output tests.
- The source artifacts were formatted with repository-pinned Black 23.10.1 and isort 5.13.2.
- An attempted full-suite run in the shallow temporary fixture lacked unrelated repository/module/probe paths (17 setup/import errors, recorded in `page_refresh_failclosed_full_host.log`). This is not a full-suite pass; root must run the ordinary full host suite and affected-file hooks after applying to the real checkout.

Tests inject failure after the destination has already changed, require old/new table retries to remain rejected, cover failed metadata commit and page remap, exercise public rejection before preflight/execute/capture, and retain successful unchanged/changed copy-elision controls. No device run, numerical claim, or new performance claim is made by this patch preparation.
