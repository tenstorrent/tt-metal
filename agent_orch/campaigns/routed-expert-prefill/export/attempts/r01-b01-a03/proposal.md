# r01-b01-a03: expert-group pacing: groups more than ~3% ahead (fraction of their own weight K-blocks) hold their weight reads until the slowest group catches up

## Motivation
Since r01-b01-a01 the grid runs as 2-4 expert row groups that share DRAM but nothing else. Per-core
TRISC1 end times from the parent's profile (`reports/r01-b01-a02/.logs/profile_log_device.csv`,
10 measured calls per shape, groups = 2-row bands) show the groups doing identical work
(dsv3: every expert per_core_M 3, 2 experts per group) finish far apart, and which group is late
changes from call to call (no position effect when averaged over calls):

| shape | experts/group | mean over calls of (slowest group / mean group) |
|---|---|---|
| mistral | 1 | 1.10 (up to 1.15) |
| dsv3 | 2 | 1.065 (up to 1.10: 471/567/570/485 µs) |
| glm | 2 | 1.06 |
| dsv4-flash | 2 | 1.065 |
| minimax-m3 | 1 | 1.066 |
| kimi-k2.7 | 3 | 1.043 |
| kimi-k3 | 7 | 1.032 |
| dsv4-pro | 3 | 1.028 |
| gpt-oss | 2 (2 groups) | 1.008 |

Total DRAM traffic (bf4 weights + bf16 x + bf8 out) is already ~400-420 GB/s on dsv3, glm, kimi-k2.7
and dsv4-pro, so those shapes are DRAM-bound and the kernel time is set by the LAST group. The
early groups get more of the shared bandwidth, then idle while the late groups run the tail with
fewer streams. On dsv3 the mean group already finishes at ~523 µs against 563 µs measured. The
r01-b01-a02 reflection named this imbalance as the largest remaining prize.

## Mechanism
Reader kernel only plus one tiny host CB (`unified_routed_expert_ffn_reader.cpp`,
`unified_routed_expert_ffn_program_factory.cpp`):
- New 128 B CB `CB_PACE` (c_21) used as a progress board: one 16 B slot per group (max 4) plus a
  staging word. Every reader clears the slots at kernel start, before the counts handshake.
- Every core computes each group's total weight K-blocks for this call
  (Σ chunks × (gate/up blocks + down blocks)) with the same scheduler the kernels already run.
- Publisher per group = its (column 0, first row) core. At the top of every gate/up and down
  K-block its reader stores the number of blocks started into its own slot and multicasts it
  (4-byte non-posted semaphore-style write) into that slot on the whole grid (the counts-broadcast
  rectangle). It publishes before it may itself wait.
- Every weight sender (`is_in1_sender`), at the top of each block, waits while its group's
  PUBLISHED fractional progress exceeds some other active group's by more than
  slack = max(2, total/32) of its own blocks. The writer's `up` and down-half reads are gated by
  the reader's up_go/down_go, so they wait too. Receivers aren't touched.
- No deadlock: all senders compare the same published values, so the group furthest behind
  never waits; it keeps publishing, which releases the others. A per-block wall-clock bound
  (64k cycles) caps any wait, so a stale slot can only slow a block. Publishers do a write
  barrier at kernel end so no board write lands after the program.
Single-group launches (rows 8) and zero-token calls skip all of it. No math, CB format or data
path
change, so numerics stay bit-identical.

## Why this is not a repeat
Nothing so far touched cross-group scheduling at runtime. r01-b01-a01/r01-b02-a0x chose the group
size and a static expert deal. r01-b03-a01/r01-b04-a01 reordered reads inside one sender's
pipeline (both lost). This changes when whole groups issue reads relative to each other, and it
reads no data earlier.

## Expected effect and risk
If the late group is late because it gets less bandwidth, holding the leaders gives it more and
the makespan moves toward the mean: about -3 to -6% on dsv3/glm/dsv4-flash/mistral/minimax,
-1 to -3% on kimi-k2.7/kimi-k3/dsv4-pro, ~0 on gpt-oss. If the late group is slow for its own
reasons (its own latency chain), holding the leaders doesn't help it, and the result is neutral
apart from the publish/poll overhead (one 4-byte multicast per group per block). Risks: a
JIT compile error; a hang if the waiting logic is wrong (the bound should prevent it); regression
if pacing lines the groups' phase transitions up so that DRAM idles at the same moments.
Accuracy can't change.
