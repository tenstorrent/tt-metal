# Stage Review: round 2

Verdict: clean-pass

Independent reviewer: `/root/stage_review_round2`, fresh read-only subagent.
No TT hardware was opened and no files were modified by the reviewer.

## Required work

None.

## Accepted scoped concerns

- Full Ethernet watcher instrumentation cannot fit the Blackhole ACTIVE_ETH
  kernel-config buffer: 27,920 bytes versus 25,600 bytes. Exact failure logs
  are retained; the maximal legal worker/Tensix watcher passes both layer
  kinds, followed by clean device health.
- Page-table contents are capture-static. Future allocation changes require
  trace release, warmup, and recapture; the passing successor validates this
  contract for sliding and full attention at position 192.
- S=128 multichip prefill remains slower than one chip after a measured 15.2%
  improvement. Traced decode provides the primary gain: 1.62987x speedup at
  40.75% four-chip efficiency.
- The implementation is deliberately fixed to batch one and this machine's
  1x4 P300c ring.

## Hard-check gaps accepted as limitations

- Full-attention latency is not reported separately.
- Full-context validation proves physical cache allocation and a final-position
  update, not one 131,072-token prefill invocation.
- The 24-layer memory plan uses calculated weights/cache/RoPE plus conservative
  reserve rather than loading all layers simultaneously in this decoder-only
  stage.

## Anomaly disposition

- Page-table replay without recapture fails (PCC 0.431/0.449); the valid
  release/warm/recapture control is PCC 1.0 for both layer kinds.
- Full Ethernet watcher fails before decoder execution on the physical program
  buffer limit; worker watcher, reset, and 1x4 mesh reopen pass.
- The intentionally near-tied fourth/fifth route swap is isolated to TP
  attention rounding; exact attention restores routing PCC 1.0, and canonical
  end-to-end seeds pass.
- Initial 26.6769 ms multichip prefill is fixed to 22.6186 ms through the
  measured 9x10/BF16 policy; residual slowdown is documented.
- Persistent-semaphore RS+AG is correct and trace-safe but slower than
  production all-reduce (0.638599 versus 0.598641 ms), so rejection is
  controlled by wall and profiler evidence.

## Scope inspected

The reviewer inspected the user contract, `$stage-review`, `$multichip`,
`$tt-device-usage`, the stage code/tests/docs, round-1 findings,
AUTODEBUG/AUTOFIX, context/compiler provenance, correctness/cache/trace/context
JUnit, timing JSON, compact profiler CSV/logs, watcher/recovery logs, and final
device health. The unrelated dirty skill-file edit remained separable.

## Residual risk

Later full-model work must preserve the replicated BF16 stack boundary and
page-table recapture rule. This goal intentionally did not begin full-model or
vLLM work.
