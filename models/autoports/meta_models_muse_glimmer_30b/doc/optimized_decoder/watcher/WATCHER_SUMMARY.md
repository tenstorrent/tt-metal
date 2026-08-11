# Watcher run — optimized decoder

`TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1`, with
`TT_METAL_LOGS_PATH` pointing here so this run is not mixed with any other. Watcher and the
profiler were run as separate hardware runs, as `$tt-device-usage` requires; no profiler
environment was set for this run and no Tracy session was active.

Code under test: fingerprint `0c001f8b379c2783` — the same fingerprint every record in
[`../pcc/pcc_results.json`](../pcc/pcc_results.json) carries, so this run certifies the
shipped configuration and not an earlier one. (The fingerprint, not a commit SHA, is the
identity: these runs necessarily precede the commit that contains them.)

Selection: 56 of the 105 collected tests — paged prefill+decode PCC at all nine sequence
lengths, page block sizes 32/64/128, batch 4 and 32, ragged slots and ragged decode
positions, sub-tile prompts (1/7/31), batched multi-chunk prefill from a shared pool at
`block_size=128`, the continued-prefill contract, sliding-window enforcement, traced decode
capture+replay, determinism, the runtime host-fallback tripwire, the real-weight tests and
the repeated-run stress test, the real-weight batch-4 test and the real-weight
length-independence test — both layer kinds, at both the shipped `bfp4_all` policy and the
`bfp8_all_lofi` structural policy. Result: **56 passed** (see
[`pytest_watcher.log`](pytest_watcher.log)). Watcher log: 62526 lines.

| signature | occurrences |
|---|---|
| watcher-detected errors | 0 |
| kernel asserts | 0 |
| NOC sanitization failures | 0 |
| CB out-of-bounds | 0 |
| L1 overflow | 0 |
| stack overflow | 0 |
| hangs / tripped waypoints | 0 |

A case-insensitive grep for `error|assert|fail|overflow|sanitiz|hang|tripped|corrupt` over
the whole log returns **zero** lines.

Stack headroom: minimum **1312 bytes free** across 57 `Stack usage summary` blocks (watcher
reports *free* bytes, so no overflow) — the same floor the functional stage measured, i.e.
the sharded/DRAM-sharded kernels this stage introduced did not eat into it.

Not covered here, for runtime reasons: the full-context (131072) path. That path *is*
exercised on this stage's code, at both precision policies, by
`test_full_context_prefill_and_decode` in the ordinary suite
([`../logs/full_suite.log`](../logs/full_suite.log)) — just not under watcher.

The raw log is committed gzipped (`generated/watcher/watcher.log.gz`) because the repo's
pre-commit hook rejects files over 500 KB; `generated/watcher/kernel_names.txt` and
`generated/watcher/kernel_elf_paths.txt` are the watcher's own symbol side-files.
