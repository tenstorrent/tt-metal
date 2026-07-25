# Stage Review: round 1

Verdict: more-work-needed

Independent reviewer: `/root/stage_review`, fresh read-only subagent. No hardware
or server commands were run by the reviewer.

## Required work

### P1: repeated decode CCL closure

The selected attention and EP paths call the inherited bare
`ttnn.all_reduce`; the constructed `CCLManager` does not supply persistent
semaphores/buffers to these calls. Six AllBroadcast rows account for 12.81% of
the three-replay decode profile. The coherent sharded residual probe does not
measure persistent/preallocated RS+AG, an alternate async all-reduce family,
or gathered-input/local-output O projection.

Required: measure legal persistent/preallocated or alternate projection/CCL
families under warmed traced decode, keep the fastest correct path, and retain
exact blockers for unavailable fused families. The Blackhole fused MM+RS race
in GPT-OSS source is a valid blocker only for that one family.

### P2: multichip prefill regression

Four-chip EP4 prefill is 26.6769 ms versus 13.4083 ms on one chip. Sparse
matmul is 53.04% of the profile, followed by routing typecast/reshape/fill
movement. EP4 beats TP4, but no EP4 program/layout/chunk/memory sweep or exact
blocker closes the dominant rows.

Required: test same-contract EP4 sparse program config, chunk, memory placement,
and data-movement candidates; retain measurements or exact blockers.

### P2: Ethernet watcher disabled

The final watcher log says `disabled features: ETH`, and the command sets
`TT_METAL_WATCHER_DISABLE_ETH=1`, without a retained full-watcher failure.

Required: run full watcher coverage on the selected path, or retain and scope
the exact incompatibility artifact.

## Other concerns / hard-check gaps

- Batch one is intentionally specialized and remains a later serving risk.
- Performance is for the default layer-12 sliding path; full-attention latency
  is not separately reported.
- Trace uses a reversed page table and mutable positions but does not replace
  the page-table tensor after capture.
- The topology probe is eager while accepted decode timing is traced.

## Anomaly dispositions accepted by the reviewer

- The deliberately near-tied fourth/fifth route swap is controlled by exact
  attention and deterministic repetition.
- Historical TP4 receiver-grid and EP4 tiled-partition failures are fixed by
  passing successors.
- The RoPE capacity omission is fixed; current total is 13.24 GiB/device.
- Correctness, cache ownership, arbitrary reversed paging, trace replay,
  active-expert semantics, and canonical PCC evidence otherwise appear
  internally consistent.

## Follow-up

`$autofix` is mandatory for the three required-work items. A later independent
review must return `clean-pass`; this round does not satisfy stage completion.
