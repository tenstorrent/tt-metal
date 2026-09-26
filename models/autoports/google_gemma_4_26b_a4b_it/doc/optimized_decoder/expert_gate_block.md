# Unapplied gate-only K-block control

`expert_gate_block.patch` is prepared **but not applied**. It adds
`expert_gate_block_w=None` to the decoder factory and forwards it as
`gate_block_w` to `OptimizedExperts`. None preserves the current common
gate/down block. An explicit value must be a positive integer divisor of
H/32=88; it changes the decode gate configuration after construction and also
feeds the separate gate/up program when `expert_split=True`. Decode down and
all prefill configurations are unchanged. The policy adds actual
`expert_gate_k_block` and `expert_down_k_block` values; the existing
`expert_k_block` remains the common/down setting.

- Base runtime SHA-256: `d6f4d858d7358f6d52332d9d1747a8bdb25cfa359f5c6f5aedde2f2e4f5d9f93`
- Proposed runtime SHA-256: `fd833e7c0d2882a2f6e5cfb8a03fb888f6c398758576df71f668a845ee779cf3`

After applying, bounded controls require only
`--default-overrides '{"expert_gate_block_w":44}'` or
`--default-overrides '{"expert_gate_block_w":88}'` on the existing actual-input
runner. The selected expert grid stays `(11,4)`, precision stays unchanged,
and down K remains 22. No wrapper change is needed.

## Source legality and buffers

The native sparse factory is
`ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp`.
Its K check requires `Kt % in0_block_w == 0` (line 160). The packed gate has
M=1 tile, K=88 tiles, N=44 tiles; `(11,4)` supplies 44 workers, each with
per-core M/N=1/1, output block 1x1, and subblock 1x1. K44 and K88 satisfy
those geometry checks without changing output distribution. The down
projection has K=22, which explains why setting the previous common block
to 44 or 88 could not test the desired gate geometry.

Factory lines 198–228 allocate `2*block` tiles for each operand, plus one
output tile. `MCAST_INPUT_BUFFERING_DEPTH=2` is defined in
`device/utilities/matmul_utilities.hpp:23`. Both K choices remain double
buffered: batchB=128 makes `batchA*batchB*num_blocks > 1` even at K88.
With selected FP32-destination and packer accumulation both disabled, the
BF16 output and intermediate buffers share one 2,048-byte allocation
(factory lines 632–666). Two row-major BF16 sparsity pages add 256 bytes
each. Standard 32x32 BFP8/BFP4 tiles occupy 1,088/576 bytes and already meet
Blackhole's 64-byte alignment.

| Kind and gate K block | Input A CB | Weight B CB | Output/intermediate + sparsity | Total per core |
| --- | ---: | ---: | ---: | ---: |
| Sliding K44, BF16 A / BFP8 B | 180,224 | 95,744 | 2,560 | **278,528 bytes = 272 KiB** |
| Sliding K88, BF16 A / BFP8 B | 360,448 | 191,488 | 2,560 | **554,496 bytes = 541.5 KiB** |
| Full K44, BFP8 A / BFP4 B | 95,744 | 50,688 | 2,560 | **148,992 bytes = 145.5 KiB** |
| Full K88, BFP8 A / BFP4 B | 191,488 | 101,376 | 2,560 | **295,424 bytes = 288.5 KiB** |

Blackhole's `hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h:33` defines
1,536 KiB L1 per core. The largest CB estimate is below that capacity by
994.5 KiB. The packed BF16 gate output occupies 11,534,336 bytes distributed
across L1 banks. Conservatively budgeting only the 44 active worker banks
adds 128 tiles = 262,144 bytes per bank; a larger interleaved bank pool
reduces that contribution. This still leaves substantial room, but firmware,
other live tensors, semaphores, and trace allocations are not included here;
actual live allocation must be confirmed by the parent hardware control.
No performance or accuracy outcome is inferred from these estimates.

Preparation checks passed: candidate compilation, Black equivalence,
`git apply --check`, and CPU execution of the changed validation/configuration
statements for None, all divisors of 88, separate-path propagation, unchanged
down K22, and invalid values. The live runtime hash was checked unchanged.
No TTNN import or device execution was performed.
