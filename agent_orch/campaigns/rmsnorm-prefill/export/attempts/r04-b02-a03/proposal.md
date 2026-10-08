# r04-b02-a03: port the round-4 stack (posted drain + ack-free stick push + streamed gamma from r04-b04-a02, PRE at HiFi2 from r04-b01-a02) onto the streamed-multicast AG release

## Motivation
The parent r04-b02-a02 (1.3175) streams the gathered pages to the workers by forwarder multicast. That brings the
combine forward: C_POST starts 1.34-1.39 µs after the last arrival, versus 1.64 µs on the pull path (r03-b02-a02).
The gain never reached the kernel end. With a multicast go, all 20 drains start within 0.11 µs of each other, and
the non-posted drain is contention-bound. So POST-start -> drain-end grew by 0.17-0.32 µs and ate the gain (parent
reflection #3).

Meanwhile every other branch landed orthogonal wins on the pull path, and this lineage has none of them:
- r04-b04-a01: posted drain writes. Per-core drain ~7% faster, -0.2..-0.46 µs drain tail.
- r04-b03-a01: flush-then-inc stick push. W_PUSH 0.63 -> 0.25 µs.
- r04-b01-a01: streamed gamma chunks. Removes the dev-0 cross-call straggler.
- r04-b01-a02: PRE x*x + row-sum matmul at HiFi2. PRE tail -0.2..-0.6 µs.

The first three are stacked in r04-b04-a02 (1.3997) and HiFi2 alone is in r04-b01-a02 (1.3911).
The parent's reflection #1 asks for exactly this port. Its purpose is to answer: once the drain is posted (less
ack-bound), does the multicast release's ~0.3 µs earlier POST survive to the kernel end?

## Mechanism
Mechanical port, no new kernel logic:
- `dit_rmsnorm_fused_worker_writer.cpp`: the r03-b02-a02 -> r04-b04-a02 writer diff, applied with git 3-way:
  - `push_stick`: `async_writes_flushed` then the arrival inc, and the atomic barrier moves to kernel end. In this
    lineage the stick goes to the gather_mcast tile-row-0 slot offset in the forwarder's sharded scratch. It uses
    the same unicast write + inc to the same core on the same NoC/VC, so the same ordering argument holds.
  - W_GAMMA: 8-page sticky-trid chunks, pushed 2 chunks behind the issue front.
  - W_DRAIN: `NocOptions::POSTED` writes plus a posted flush before each pop and at the end.
- `dit_rmsnorm_fused_compute.cpp`: the r04-b01-a02 PRE diff (explicit HiFi2 ELWMUL init/exec and ones*S^T matmul).
  POST and the combine stay at HiFi4.
- Forwarder, factory and the gather_mcast compute indexing come from the parent and are untouched (kernel-only
  change, no rebuild).

## Why this is not a repeat
- No node has combined the multicast release (r04-b02-a01/a02) with any of the round-4 wins.
- r04-b04-a02 = the three writer wins on the pull release. r04-b01-a02 = HiFi2 on the pull release with
  non-posted drain and acked push.
- This node's code differs from "r04-b04-a02 + HiFi2" only in the AG release (forwarder multicast of the data and
  the go vs. 20 go incs + 8 pulled face-rows). Its score against that stack measures the multicast release
  directly.

## Expected effect and risk
- Versus the parent: the sum of the ports. Roughly -0.3 µs PRE (HiFi2), -0.3..-0.4 µs push, -0.2..-0.5 µs posted
  drain, and h7168 straggler removal. Expected ~11.7 / 13.2 / 16.6 / 17.5 µs, score ~1.41-1.43.
- Against a pull-path "r04-b04-a02 + HiFi2" stack, if a sibling builds one:
  - If posted drains tolerate synchronized starts, this should be ~0.2-0.3 µs better per shape.
  - If not, it will be equal.
- Risks:
  - The ack-free push into the forwarder's sharded scratch. It is the same unicast path as on the pull lineage, so
    a stale slot would show as accuracy_fail.
  - HiFi2 max_abs was 0.022-0.024 on its own node, with a gate of 0.05.
  - A merge slip, e.g. the gamma-loop poll_stick. That would show up as a hang or jit_compile_error.
- To read the result: rerun the parent's `tail.py` (POST start after the last arrival, per-core drain end - POST
  start) and r04-b03-a01's `push.py` on the report.
