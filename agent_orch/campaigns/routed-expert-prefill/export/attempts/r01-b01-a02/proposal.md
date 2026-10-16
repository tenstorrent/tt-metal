# r01-b01-a02: row-shared x tilize: each gate/up K-block of row-major x is tilized ONCE per M-row by a rotating owner core and multicast as bf8 tiles

## Motivation
On the row-major (benchmark) path every one of the 11 cores in an M-row receives the same bf16
row-major x K-block (multicast from the row's x sender) and then **tilizes the same block itself**.
That puts a per-K-block `tilize init -> tilize per_core_M strips -> uninit -> matmul_block_init ->
pack reconfig -> L1_ACC reconfig` sequence in front of every gate/up K-block on every core. The
sequence drains the unpack/math/pack pipeline at each block boundary.

Evidence that compute efficiency, not DRAM, is what bounds the per-group pipeline at the parent
(r01-b01-a01, profile `reports/r01-b01-a01/.logs/profile_log_device.csv`):
- Per-core tile-mults / per-core kernel time on the fastest row group comes to ~28-32 cycles per
  tile-mult on every non-bias shape (mistral 29, minimax-m3 28, dsv3 ~30). The LoFi math rate is 16
  (tech_reports/GEMM_FLOPS: BH bf4 LoFi big-matmul utilisation reaches ~80%). The groups are
  far from DRAM-bound: mistral pulls ~95 GB/s per group, and gpt-oss pulls 147 GB/s in total.
- The reflections of r01-b03-a01 and r01-b04-a01 found that DM-side read reordering only hurt. Both
  pointed at the compute's per-K-block fixed cost (tilize init/uninit, matmul re-init, pack
  reconfigs) as the likely limit.
Per K-block at per_core_M=3, w=16 the tilize is 48 tiles plus ~4 reconfigs, against 576 tile-mults.
That is roughly 10-15% of the gate/up phase, and it is paid by all 11 cores for identical work.

## Mechanism
Only on the row-major path (TILE x keeps the existing code byte-for-byte):
- Global gate/up block counter `g` (same on all cores of a row, advanced by num_blocks_gu per
  chunk, in reader and compute). Owner of block g = column `g % GRID_X` of the row.
- Reader of the owner core: reads the RM x block from DRAM into `cb_x_rm` two blocks ahead (now a
  single-block local ring, no longer a multicast source). After its own weight-sender work, it
  waits for the tilized block in a new `cb_x_tl` (bf8, 1 block). It waits for the 10 row-mates'
  ready, then multicasts the bf8 tiles (loopback included) into every row core's `cb_in0_x` slot.
  The slot address is the same on all cores because they all push the same sequence. Then it sends
  the valid semaphore (loopback), does a write barrier, pops `cb_x_tl` and pushes `cb_in0_x`.
- Row-mates: reserve the `cb_in0_x` slot, ack the block's owner (row NoC table), wait valid, push.
- Compute: before block 0 it tilizes block 0 if it owns it. At block j it tilizes block j+1
  (`cb_x_rm -> cb_x_tl`) if it owns it, then runs the matmul on `cb_in0_x` (now double-buffered).
  It only restores the matmul / pack config after it has actually tilized. Each core does the
  tilize + reconfig on ~1/11 of the K-blocks instead of all of them.
- Host: `cb_in0_x` double-buffered on both paths, `cb_x_rm` single block, new `CB_X_TL` (c_20,
  RM only). The footprint/fit is updated: +128 B per M*w tile, and the fitted (per_core_M, w) is
  unchanged on all 9 shapes. Reader gets `cb_x_tl` as CT arg 36 (accessor offset 37). Compute gets
  named `cb_x_tl`, `grid_x` and runtime arg 3 = gx.

Files: `unified_routed_expert_ffn/device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp`,
`.../kernels/compute/fused_swiglu.cpp`, `.../unified_routed_expert_ffn_program_factory.cpp`.

## Why this is not a repeat
- r01-b02-a01 rotated the x **DRAM read** across the row, but every core still tilized every block.
  It gained ~3% on the K=128 shapes. Here the **tilize work and its per-block reconfigs** are
  de-duplicated, and the multicast carries bf8 tiles (1088 B) instead of bf16 sticks (2048 B per
  tile).
- r01-b03-a01 and r01-b04-a01 reordered weight DRAM reads on the sender RISCs and lost. This change
  doesn't move any weight read.

## Expected effect and risk
- Bit-identical numerics expected: the same tilize (bf16 RM -> bf8 tiles) runs on one core instead
  of eleven, and the matmul inputs and accumulation order are unchanged. Any PCC change means a
  slot or race bug.
- Expect 4-10% on the compute-bound shapes (mistral, minimax-m3, dsv3, glm, dsv4-flash, gpt-oss)
  and less on the DRAM-bound ones (kimi-k2.7, dsv4-pro, kimi-k3).
- Costs: the owner core pays the tilize (+reconfig) once per 11 blocks. At each chunk start the row
  waits for block 0's tilize + multicast (one extra ~2-3 us bubble per expert per group). The owner
  reader does a write barrier per owned block.
- Risks: a hang from a semaphore or ordering mistake (eval `hang`), or accuracy_fail from a wrong
  slot address. L1 margin on dsv4-pro is ~17 KB by my estimate. An overflow would show as a host
  TT_FATAL / allocation error.
