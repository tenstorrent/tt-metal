# AutoDebug: native reply prefixes are missing; observer silently ignores their tails

## Finding

**The snapshot contains damaged reply-log records, not merely missing completions.** Eight container-prefixed lines contain long hexadecimal suffixes without `TTNATIVEHEX1`. After skipping the first half-byte, every suffix decodes as valid UTF-8 and ends in the native envelope's escaped response-JSON tail, including `finish_reason=stop` and completion usage. These are positive evidence of lost record prefixes. The observer's zero malformed-envelope count is misleading as a coverage indicator: its validation only runs when the marker survives.

The leading explanation is fragmentation/interleaving or prefix loss along the stdout/container/CI path. The exact stage that loses the prefix remains unproved. In particular, concurrent ordinary INFO output combined with the CI whole-line INFO filter can produce precisely this shape. This is a demonstrated mechanism, not a proven attribution to a particular writer or Docker/GitHub component.

**Do not infer that the six unmatched initial IDs remain running, or that they completed.** The eight damaged tails have no request IDs. The 29/40 client counter is global, not an initial-eight counter. Neither timing nor arrival order supplies the missing correspondence.

## Evidence and scope

Inspected snapshot: `bringup/artifacts/stage12-eval-observation/42316311286/current.log`, 27,370,208 bytes, 105,637 lines, SHA-256 `73fb566226420174ff97ac02c5d4004b4234406fc46e68f481d319629fb1e5f9`. This matches `observer-state.json`'s offset and digest. The last event is 2026-11-07 01:32:06.7983403 UTC; collection ended at 01:36:18.661103 UTC. Conclusions concern this snapshot, not current remote execution.

After stripping ANSI codes, an independent in-memory scan found:

| Observation | Count / detail |
|---|---|
| Valid marked native envelopes | 64: 39 request-logger records, 25 middleware records |
| Received chat requests after the eval boundary | 38 distinct IDs |
| Complete chat reply JSON records | 22, all distinct IDs; 20 belong to later requests |
| Other middleware records | Three pre-boundary empty-body records |
| Malformed marked envelopes | 0 |
| Unmarked response hex tails | 8 |
| Latest client progress | 29/40, line 104246 |
| Initial-eight complete reply matches | 2 |

The initial matches are `chatcmpl-a36ef45b15eadb8e` at line 6680 and `chatcmpl-98aa2ab35651eb31` at line 11280. The unmatched initial IDs are `chatcmpl-81c70f9dda0cb165`, `chatcmpl-a3d9bbe6d895576c`, `chatcmpl-a7b8e48a699f00ae`, `chatcmpl-b93140f00b0a8245`, `chatcmpl-a504ac531c40ba18`, and `chatcmpl-9335cdc02b4394a2`.

| Damaged line | Hex characters remaining | Completion tokens in decoded suffix |
|---|---:|---:|
| 6819 | 23803 | 4953 |
| 6881 | 14319 | 5008 |
| 26864 | 33499 | 8506 |
| 43563 | 4127 | 3495 |
| 58710 | 10895 | 4387 |
| 59442 | 13633 | 15845 |
| 71967 | 20507 | 7747 |
| 95793 | 15399 | 4335 |

All eight tails have odd hex length; removing the first nibble permits decoding. All contain a stop finish reason, usage, and the closing envelope syntax. No raw response content is reproduced here. These lines precede nearby client progress events, but that adjacency is **not** used to assign request IDs. The observed 22 intact replies plus eight damaged tails are consistent with 30 response emissions and a delayed 29 progress display; they do not establish 30 unique acknowledged request IDs.

The CI log identifies tt-metal `eef534e5cd6bcaf37957647de4d8d4eff4f0e047`, TTI `e5a9a7638a1f46f22c0d89acf2ac333da5d53c02`, and vLLM ref `ca7a18246e4111629ed6cb4a7d1192002749fccf` at lines 64–66. It explicitly selects this model's `native_logging.json` at line 1329. The published formatter was inspected at the named metal SHA and matches the local implementation. The TTI checkout HEAD is `af1ef17abfecb059da13d0a1287c00eace0f210e`; diffs against the run SHA are empty for the logging utility, subprocess collector, log setup, and Docker launcher inspected below. Installed vLLM/Starlette source is local-path evidence, not independently verified CI-image byte identity.

## Logging chain and completion boundaries

1. Installed `python_env/lib/python3.10/site-packages/vllm/entrypoints/serve/middleware/log_response.py:146–160` awaits `call_next`, exhausts `response.body_iterator` into a list, resets the iterator for forwarding, and logs before returning. Thus a middleware record is after response-body construction but before this wrapper forwards the body to the client; it is not client acknowledgement.
2. `tt/native_log_formatter.py:13–22` serializes the entire `record.getMessage()` into one JSON envelope and hex-encodes it into one unbounded line. It performs no truncation. Hex doubles the serialized byte length.
3. `tt/native_logging.json:17–22,32–44` sends both native logger names directly to a synchronous stdout `logging.StreamHandler`, with propagation disabled. They do **not** traverse `AsyncLogHandler` or its rotating file.
4. TTI `utils/logging_utils.py:69–91` uses an unbounded `Queue(-1)` and a background listener for other vLLM logs, with a separate stdout handler. Standard `StreamHandler.emit` writes the formatted string plus newline and flushes; handler locks do not establish process-wide, cross-handler, or cross-process atomicity of arbitrary-size writes.
5. The actual CI shell printed in `current.log:1172–1174`, also cached in `doc/agentic_evals/setup/shield-run-tests.yml:384–387`, follows Docker logs, applies `grep -v -i "INFO"`, and colors/prefixes each surviving line with `sed`. The collector is in this workflow, not in TTI's Python async handler. An intact native hex line cannot contain the letters `INFO`. A fragment sharing a physical line with ordinary INFO text can be discarded wholesale.
6. TTI `workflow_module/proc.py:26–29,103–130` independently reads evaluator stdout/stderr with ordinary text `readline` and relogs it. It has no explicit line-size cap. `workflows/log_setup.py:18–22` preserves raw messages. This is the client-progress path, distinct from the colored Docker-log path.
7. `bringup/artifacts/stage12-watch-eval-log.py:18–43` saves successful job-log downloads without a line filter. The observer verifies append-only input, but that cannot detect loss already present upstream.
8. `bringup/artifacts/gpqa-first-agent-observer/observe.py:78–98` checks hex only if the marker exists. Lines 125–175 parse only recognizable response records and retain only initial-ID matches. Unmarked hex tails hit neither check. Therefore `truncation_events=0` and `source_coverage_compromised=false` do not establish intact source coverage. This is a concrete observer blind spot.

The cached harness source `bringup/artifacts/stage12-lm-api-models.py:690–727,865–874` awaits the HTTP JSON body, parses generations, returns answers, and increments progress after task completion. Its counter contains no request ID. The run logs installation and verification of lm-eval `af401ef3dd3d2d90e55680990ed1f0c9f7892448`; the cached source is explanatory evidence, not a newly verified import from the CI process.

## Competing explanations, adjudicated

| Candidate | Evidence and limits | Verdict |
|---|---|---|
| Lost prefixes / fragmented transport | Eight response suffixes survive without their markers or IDs. This directly explains absent complete JSON and the observer's zero malformed count. | Confirmed record damage; exact losing component unresolved. |
| Large writes interleave with INFO; collector drops the mixed prefix line | Unbounded native lines share output infrastructure with other logging; CI filters any line containing INFO. Forced interleaving reproduces prefix removal. Actual interleaving is not captured before the filter. | Leading mechanism to discriminate, not proven root component. |
| A fixed CI line-size cutoff | Large surviving marked lines exist, including 79,643 characters after ANSI removal at line 47535. Prefix-loss tails vary in length. | No specific cutoff established; do not claim a 64-KiB limit. |
| Genuinely unfinished initial IDs | Six have no intact matching reply. Eight active rows are aggregate and do not identify requests; damaged tails lack IDs. | Possible for some/all unmatched IDs; unresolved, not an explanation for damaged tails. |
| TTI async queue loss | Native loggers bypass that queue; generic queue is unbounded. | Not a supported cause of these native omissions. Abrupt termination/reconfiguration is generic risk, without supporting evidence here. |
| First-section-only middleware logging | Installed `_log_non_streaming_response` at lines 137–143 decodes only `response_body[0]`. A multipart ASGI body would lose later sections; a UTF-8 split could instead produce the binary placeholder. | Real general limitation, but does not explain markerless suffixes. |

For the last candidate, installed `vllm/entrypoints/openai/chat_completion/api_router.py:69–73` returns `JSONResponse` for a non-streaming completion. Installed `starlette/responses.py:165–169,183–203` sends its serialized body as one ASGI message; `starlette/middleware/base.py:176–188` yields the message body without a size-based split. Network packet boundaries occur downstream and do not imply multiple ASGI sections here. No multipart response record, binary placeholder after the boundary, or marked partial JSON was observed. Do not patch the shared middleware on this evidence.

## Smallest host discriminator and supported repair boundary

A cheap local check was performed entirely in memory, without model imports: a synthetic native line of 180,503 characters round-tripped exactly through `grep -v -i INFO`. Injecting an ordinary INFO line after byte 16,384 removed the marker-bearing prefix and preserved a markerless hex suffix. This proves the filter mechanism can explain the shape and falsifies an inherent formatter/grep length cap at that size. It does **not** prove the real stream interleaved.

Next host-only test: use synthetic non-streaming JSON replies with unique IDs, known byte counts and hashes at 8, 32, 64, 96 and 256 KiB. Exercise the stock middleware and configured native handler while a second writer emits ordinary INFO and sampling lines into the same pipe; drain it continuously through the exact cached grep/sed filter. Capture bytes immediately before and after the filter and compare IDs/hashes, not body contents. Include a forced split/interleaving control and a single-writer control. Separately feed one-section and deliberately two-section ASGI bodies to distinguish transport loss from the first-section limitation. No vLLM engine, weights, devices, Docker, network, or workflow change is needed. If only actual CI transport loses records, a later transport-only CI probe requires separate authorization; it was not run here.

The observed damage supports a **model-owned framed logging repair**, not a shared plugin/workflow patch: replace the unbounded native-line emission with bounded, independently recognizable frames carrying record identity, chunk index/count, total byte length and checksum. Include a small completion manifest with request ID/status and payload hash so an absent payload is distinguishable from an unfinished request. Emit each bounded ASCII frame as one write; choose its bound from the actual pipe's atomic-write limit, and validate the downstream transport separately. A frame-local envelope must remain INFO-filter-safe. Reassembly must reject missing, duplicate-conflicting or reordered/incomplete data as appropriate and never call partial payloads complete. Keep this confined to the model's logging module/config and the private observer; do not monkeypatch stock vLLM/plugin runtime. Existing single-record formatter test (`tests/test_native_log_formatter.py:12–28`) covers encoding/filter round-trip, not large concurrent emission.

Also teach the private observer to flag unmarked long hex suffixes on the expected container channel as suspicious coverage loss. Do not fabricate IDs or accept tail reconstruction as a complete reply. A formatter alone cannot recover body sections never supplied by the middleware; if the host discriminator proves that separate problem, it needs a separately justified model-owned full-body capture point.

## Review and limitations

Claims were revisited against the producer, handler routing, collector, observer and reply-ID ledger. The first-section defect is deliberately secondary; the exact transport mechanism and initial-six completion states remain unresolved. No model execution, device access, subagents, launches, source changes, shared workflow/plugin edits, or root-report writes were performed. Only this report was written.

Tooling exception: a read-only `git cat-file` query for the unavailable CI vLLM ref unexpectedly triggered the partial clone's lazy fetch, which failed (`not our ref`). No remote source was obtained or used. No further fetch was attempted. The CI middleware revision is therefore explicitly unverified rather than inferred from that query.
