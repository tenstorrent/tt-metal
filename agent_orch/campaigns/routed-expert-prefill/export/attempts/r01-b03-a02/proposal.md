# r01-b03-a02: batch x over several gate/up K-blocks at small per_core_M (one x read/mcast/tilize per batch, row-major path)

## Motivation
The parent (r01-b03-a01) and r01-b04-a01 both moved weight DRAM reads earlier and both lost 5-9%.
So the sender read->multicast chain is not the bottleneck. This node starts by **reverting the
parent's read-ahead** (kernels back to root code) and then tests a different hypothesis that both
reflections named: the fixed per-K-block cost on the x / compute side.

Baseline per-expert arithmetic (dsv3 100 us, glm 88 us, dsv4-flash 66 us per expert) gives about
28 cycles per tile-mult plus about 16 us per expert. b01's result (time ∝ per-core M-rows) shows the
time scales with per-core compute, not with DRAM. In the gate/up K-loop on the ROW_MAJOR path
(the only path the benchmark runs), every K-block of 16 tiles pays, on all 88 cores:
- an x handshake round (receiver ready-ack, sender DRAM stick reads, multicast, valid sem). This
  couples the 11 cores of a row in lock-step once per K-block,
- `compute_kernel_lib::tilize` with InitAndUninit (fast-tilize init: unpack/math remap/pack config,
  then uninit), then `reconfig_data_format_srca` + `matmul_block_init` + pack reconfig + L1_ACC
  reprogramming before the matmul can resume,
- 32 separate 1 KB DRAM stick reads per tile-row on the x sender.

At per_core_M = 1 (nearly every benchmark expert) the x CBs are sized for per_core_M_max = 4, so
3/4 of every x block is padding that the kernels only pointer-bump.

## Mechanism
Use that unused x capacity to carry B = min(4, per_core_M_max / per_core_M) consecutive K-blocks of
x per CB block. B is computed at runtime on every RISC from the same per_core_M, so reader and
compute agree. Per batch of nb <= B K-blocks (the last batch may be shorter):
- **Reader** (`unified_routed_expert_ffn_reader.cpp`): the x reserve, ready-ack, DRAM read,
  multicast, valid and push happen only at a batch start. Each token stick is read once,
  nb*16 tiles wide (one 4 KB read instead of four 1 KB reads). The strip of a tile-row is laid out
  `nb*w` tiles wide. Weight blocks (gate/up) still stream per K-block, unchanged.
- **Compute** (`fused_swiglu.cpp`): one tilize call per batch, with strip width nb*w (a
  compile-time switch over nb = 1..4). Then one matmul re-init, then nb gate/up K-blocks that read x
  at tile offset `j*w`, with row stride `nb*w`. Per K-block: weight waits/pops, L1_ACC, and the
  packer drain stay as they are.
- A shared helper `adaptive_chunk::x_kblock_batch()` defines B. The TILE x path keeps B = 1 (its x
  CB is double-buffered and filled by the reader, with no tilize), so its behaviour is byte-identical.

CB sizes, formats and math are unchanged: the batch fits exactly in the existing
per_core_M_max * w block. There is no host change (kernel-only, JIT).

## Why this is not a repeat
- r01-b03-a01 / r01-b04-a01 changed the timing of weight (in1) DRAM reads relative to their
  multicast. This node doesn't touch the weight path at all (it is reverted to root). It changes the
  x path and the compute's per-K-block re-init.
- r01-b01-a01 / r01-b02-a01 change the row/expert schedule. This keeps the root schedule and is
  orthogonal, so it could compose with them later (with per_core_M = 3 there, B = 1, but their M=1/2
  tails benefit).
- b04's suggestion was "wider in0_block_w_gu at per_core_M=1". That needs larger weight CBs. This
  widens only x (free L1) and keeps the weight blocks as they are.

## Expected effect and risk
If the per-K-block x/tilize/re-init overhead is ~10-20% of the gate/up phase (gate/up is ~65% of
the time), expect roughly 3-8% on the M=1 shapes and less on gpt-oss (M=2 experts get B=2). If
in1 unpack is the binding limit, the gain will be within noise. That result would itself confirm
that compute-side per-block overhead is not the issue at M=1.
Risks: a wrong in0 offset or stick layout shows up as accuracy_fail (bit-identical results are
expected, since the same tiles feed the same MACs in the same K order). A CB ring mismatch shows up
as a hang. Each batch pushes/pops exactly one max x block, so the ring arithmetic doesn't change.
