# Investigate expert placement and MoE cost variation in Mistral Small 4 prefill

Owner: Sonnet. Draft for review with Alina; not posted.

## Finding

Dispatch and Combine account for most of the historical difference in MoE cost between L18 and L23 on the same PP4 stage. Controlled replays now show that changing expert placement reduces isolated Combine time by **38.36% for L18 and 25.82% for L23**.

This demonstrates placement sensitivity in the controlled replay. It does not yet establish the cause of the historical full-model slowdown or an end-to-end speedup.

## Evidence

The September 8 eager 36-layer PP4 capture shows MoE costs of 6.129–10.377 ms (+69.31%). L18 and L23, on the same devices, differ by 4.005 ms; Dispatch and Combine account for 3.413 ms (85.2%). These values sum each operation's maximum device duration; they are not elapsed layer times. The plan's original +54% came from a different capture or metric.

The September 11 replay uses all 128 experts on eight Galaxy devices at SP8/TP1, with 16 experts per chip, 5,120 tokens and top-4 routing. Twelve successful captures cover two processes per layer and routing case. Each retains nine iterations after discarding the first. Results average the two process means, using the maximum device kernel duration for each iteration.

| Routing source | Captured placement: Combine | Balanced placement: Combine | Combine change | Dispatch change |
|---|---:|---:|---:|---:|
| L18 | 1.723 ms | 1.062 ms | −38.36% | −2.09% |
| L23 | 1.477 ms | 1.096 ms | −25.82% | −2.13% |

Shuffling source tokens preserves destination totals and fanout distribution but changes source traffic and ordering. L18 Combine falls 4.06%; L23 changes +0.10%, within run variation. Destination totals alone do not explain every effect.

## Limits

- Balanced placement changes load, tile padding, ordering, fanout and locality together. Load imbalance alone is not a proven cause.
- The routing came from a separate TP run. Replaying it with PP placement does not directly explain the historical PP timings.
- The worker replaces FFN with layout conversion and omits concurrent shared-expert work and downstream weighted reduction. It does not check numerical outputs.
- Successful captures pass device-count, iteration, duration and operation-order checks. Failed attempts are excluded; some repetitions required targeted resets. Startup failures and an initial stall remain unexplained.

## Next steps

1. **Capture PP routing and timings together.** Start with L18/L23 on the same request, chunk and rank. Record expert mapping, valid tokens and per-device Dispatch, Combine and FFN durations. Export indices after measurement and check whether instrumentation changes timing.
2. **Separate placement effects.** Permute expert slots within each device to test ordering without changing destination traffic or padded totals. Compare placements that balance raw assignments versus tile-rounded load, and record fanout and locality.
3. **Separate source traffic from token ordering.** Compare whole-source-shard permutations with within-source token permutations, using fixed seeds and randomized run order.
4. **Validate any production change.** Move expert weights consistently, check correctness, then measure matched traced throughput for TP and PP4 separately.

Completion requires an explanation linking routing and mapping to cost variation in the same full-model run, or evidence that narrows the cause another way. A production speedup additionally needs correctness and end-to-end measurements. These results do not rule out improvements to static program settings.

## Supporting material

- [B1 finding and historical provenance](https://github.com/tenstorrent/tt-metal/blob/ssalice/mistral4-b1-investigation/profiling_reports/2026-09-11/B1_FINDING.md)
- [Local findings and run ranges](https://github.com/tenstorrent/tt-metal/blob/ssalice/mistral4-b1-investigation/profiling_reports/2026-09-11/b1-controlled-routing/LOCAL_FINDINGS.md)
- [Validated capture matrix](https://github.com/tenstorrent/tt-metal/blob/ssalice/mistral4-b1-investigation/profiling_reports/2026-09-11/b1-controlled-routing/MATRIX_RESULTS.md) and [timing data](https://github.com/tenstorrent/tt-metal/blob/ssalice/mistral4-b1-investigation/profiling_reports/2026-09-11/b1-controlled-routing/matrix_timings.json)
- [Validation and recovery history](https://github.com/tenstorrent/tt-metal/blob/ssalice/mistral4-b1-investigation/profiling_reports/2026-09-11/b1-controlled-routing/VALIDATION.md)
- [Plan to capture PP routing and timings together](https://github.com/tenstorrent/tt-metal/blob/ssalice/mistral4-b1-investigation/profiling_reports/2026-09-11/b1-controlled-routing/PAIRED_CAPTURE_PLAN.md)
