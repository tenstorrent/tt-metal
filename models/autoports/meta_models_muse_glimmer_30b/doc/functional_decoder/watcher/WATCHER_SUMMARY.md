# Watcher run — functional decoder

`TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1`,
logs redirected with `TT_METAL_LOGS_PATH` so this run is not mixed with any other.
Watcher and the profiler were run separately, as `$tt-device-usage` requires.

Code under test: fingerprint `68b4a0ca63d305c9`, git head `9876c3547604` — the same
fingerprint every record in `../pcc/pcc_results.json` carries, so this run certifies the
shipped configuration and not an earlier one.

Selection: 26 of 103 collected tests — chunked prefill (8256),
non-aligned prefill (100), sub-tile prefill (1/7/31), traced decode, batch-32
prefill+decode, ragged slots/positions, batched multi-chunk prefill out of a shared pool at
block_size 128, the continued-prefill contract, page block sizes 32/64/128, and the
real-weight tests — both layer kinds. Result: **26 passed**
(see `pytest_watcher.log`).
Watcher log: 19416 lines.

| signature | occurrences |
|---|---|
| watcher-detected errors | 0 |
| kernel asserts | 0 |
| NOC sanitization failures | 0 |
| CB out-of-bounds | 0 |
| L1 overflow | 0 |
| stack overflow | 0 |
| hangs / tripped waypoints | 0 |

Stack headroom: minimum 1312 bytes free, across 17 `Stack usage summary` blocks
(85 per-RISC lines; watcher reports *free* bytes, so no overflow).

The only matches for a broad error-word grep are the benign `Stack usage summary:` blocks and
a few `NOC` mentions in the legend/waypoint headers; the log otherwise contains only the
normal attach / dump / kernel-id / detach lines. No fatal watcher exception, no sanitization
failure, no CB/L1/stack finding.

Not covered here, for runtime reasons: the full-context (131072) tests.

The raw log is committed gzipped (`generated/watcher/watcher.log.gz`) because the repo's
pre-commit hook rejects files over 500 KB; the counts above were taken from the uncompressed
log before compression. The `generated/inspector/` dumps that the same run produced are not
committed: they are runtime bookkeeping, not watcher evidence, and two of them are also over
the limit.
