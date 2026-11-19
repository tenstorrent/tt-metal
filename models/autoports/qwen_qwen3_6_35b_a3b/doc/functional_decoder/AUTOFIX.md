# AutoFix: full-context decoder validation

Two separate issues were investigated: diagnostic-induced device halt and long-context BF16 accumulation error. Recovery is complete. The final numerical repair passes full-context and affected-path validation; no context or PCC threshold was reduced. All hard gates pass; independent round1 review returned clean-pass with no Required Work.

## Diagnostic halt: resolved by authorized recovery

The first full-attention run completed its HF oracle in 751.83 seconds and was executing TTNN prefill when intrusive triage was invoked after quiet logs. Triage completed at 16:35:01 UTC; stable host snapshots were taken only afterward, at 16:36:19 and 16:37:50. These observations do not establish any pre-triage hang.

`tools/triage/triage.py:959` disables `cont` and `continue_without_debug` on Blackhole/Wormhole because HALT → READ/WRITE → CONTINUE can break cores (`triage_session.py:13`). The diagnostic left the consumer halted. The later full command ring is consistent with that halt:

- CQ0 write pointer and fence were 0x1d33c; credit address 0x1b7c0 (`triage/host_queue_values.txt`).
- Ring had 1535 nonzero entries and one zero at 0x1d33c; next slot 0x1d340 contained 0xa88 (`triage/fetchqueue_words.json`).
- Actual BRISC cursor was 0x1d340, with zero in-flight work, and the core was already halted (`triage/prefetch_private.json`).
- The sampled MMIO instruction had completed; its caller PC was a saved return address (`autofix_dispatch_audit.md`).

Installed AutoTriage and separate source audits investigated dispatch and decoder address bounds. `AUTOTRIAGE.md` marks the initial interpretation as superseded; `autofix_dispatch_audit.md` and `autofix_decoder_audit.md` contain the controls. The installed launcher's namespace sandbox preflight failed; retry used `AUTODEBUG_SKIP_CHILD_SANDBOX=1` in the unrestricted environment. Exact commands and logs are in `work_log.md`. No speculative dispatch patch or unsafe continue bypass was applied.

The owner subsequently authorized terminating PID 2087868 and `tt-smi -r`. Verified sole ownership, SIGKILL, bounded list/reset/list, all four chips visible, and a chip-0 BF16 add/readback/open/close smoke all succeeded. No second reset or lock cleanup was needed. Evidence: `recovery.json`, `logs/recovery_{list_before,reset,list_after,smoke}.log`. Subsequent quiet long passes were monitored only with host-side py-spy snapshots, which showed advancing token offsets.

## Long-context accumulation: experimentally isolated and repaired

The recovered run closed the device cleanly but failed the numerical gate: prefill262143 aggregate PCC0.999269557, last row262142 PCC0.945939623. Evidence: `logs/full_long_recovered.log`. This was a correctness failure, independent of device recovery.

Installed AutoDebug completed source-only investigation in `long_tail_debug/AUTODEBUG.md`; `logs/autodebug_long_tail.log` records the launcher. `long_tail_address_audit.md` rules out the proposed uint32/address overflow at the target shapes. Native decode factory hard-codes BF16 intermediate/statistics storage despite FP32 destination accumulation. One core at key chunk32 performs8192 serial rounded updates. Native chunked prefill similarly keeps its running output in BF16, although its denominator is FP32.

Controlled evidence:

| Experiment | Original K32 / one-core or original prefill | Grid8 / K256 | Artifact |
|---|---:|---:|---|
| Synthetic full-context decode PCC | 0.770067 | 0.999722 | `sdpa_precision_probe.json` |
| Real-weight final-token decode PCC, HF-prepared KV | 0.842910 | 0.999945 | `real_tail_probe.json` |
| Real-weight final256 prefill, minimum per-row PCC | 0.990283 | 0.998581 | `real_prefill_tail_probe.json` |

A grid-only decode repair passed the originally sampled long rows, but the expanded final64-row gate found chunked-prefill rows262105/262106 at0.994884/0.992146. This intermediate result is retained as `full_long_grid_only.json`; the failure is `long_tail_expanded_failure.json`. It is not final capability evidence. The attempted K512 diagnostic rejected start261888 because it is not512-aligned; the legal K32/K256 comparison was rerun successfully (`logs/real_prefill_tail_probe_fixed.log`).

The final model-local repair uses key chunk256 on both full-attention paths and an8x8 decode grid. Full-prefill interiors begin at256-aligned absolute positions; unaligned starts/tails execute exact one-token kernels. Tables pad to an8-entry multiple with read-only aliases of the last owned finite page. Default physical pools also round up to that width, satisfying native cache-update validation; the first small-capacity probe exposed and corrected this requirement. Additional default K/V storage is at most458752 bytes. Logical capacity remains262144 and the acceptance bar remains0.995. Linear-attention arithmetic is unchanged.

Final validation artifacts are `full_long.json` (both lengths and all64 trailing rows), `page_extent_regression.json`, `padded_trace_regression.json`, refreshed full-attention trace/continuation/shared-pool/partition tests, and profiler correctness results. `padding_contract_audit.md` independently inspected the final allocation and masking rules. `source_audit.json` verifies unchanged linear runtime and planning. The final README and hard-gate result determine stage acceptance; historical failed runs above are retained for reproducibility.

## Profiler marker overflow: resolved measurement failure

Initial profiling completed but dropped markers after its DRAM buffers filled. The incomplete summary is explicitly invalid in `tracy/overflow_summary.json`; historical reports/logs are retained. Draining before windows and between complete trace replays resolved the loss without changing decoder execution. `profiler_completeness.json` proves20 identical operation sequences per kind (four correctness/warmup,16 measured),144 operations per linear replay and99 per full replay, with no dropped markers. The final four timing rows are in `functional_decoder.md`. Host drain overhead is excluded from device-operation duration and included only in the separately reported gap column.
