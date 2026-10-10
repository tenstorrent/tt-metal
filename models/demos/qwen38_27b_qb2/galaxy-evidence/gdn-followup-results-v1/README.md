# Completed compact GDN follow-up hardware tests

All three independently requeued tests completed with clean teardown on October
10, 2026, by 21:51:55 UTC. No configuration was promoted to serving.

- Compact versus qualified flat-boundary comparison passed exactly for 4096
  updates at B16 and B32. This compares two implementations; it is not a new
  independent dense-reference or whole-model qualification.
- Resident FP32 state passed its standalone correctness suite, including the
  independent dense-reference test. At B16 the bracket moved from 99.041 to
  88.131 us (1.1238x throughput, 0.256% control drift), projecting **0.524 ms**
  across 48 layers. B32 moved 166.788 to 149.599 us (1.1149x), projecting 0.825 ms.
  The state-only bandwidth floor plus 10 us target is **not met** in either case.
- Compact gate packing passed correctness for the tested batch/placement matrix.
  B16/L1 moved from 66.985 to 58.346 us (1.1481x, 0.169% control drift), projecting
  **0.415 ms** across 48 layers. B16/DRAM projects 0.431 ms; B32/L1 0.575 ms.
  B1 is effectively unchanged. Raw per-case values and source hashes are retained.

The stage speedups are not full-model speedups. Relative to the qualified
49.913-ms B16/32K step, each measured saving projects roughly 1% individually.
They require combined full-model timing, accuracy and memory checks before
promotion; do not add them and call the result measured performance.
Current qualified native performance remains **20.035 TSU at B16/32K/TP4** and
compact full GPQA **177/198**. All units in this test chain are now terminal;
there is no further hardware test queued by these launchers.
