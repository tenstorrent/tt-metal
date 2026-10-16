# r01-b03-a02 result: 0.9626 (ok)

## What happened vs expected
I expected +3-8% on the per_core_M=1 shapes from paying the x handshake/multicast, the fast-tilize
init/uninit and the matmul re-init once per batch of up to 4 gate/up K-blocks instead of once per
K-block. Instead the result is a regression, though smaller than the parent's. Per shape: dsv3 0.954,
kimi-k2.7 0.961, kimi-k3 0.998, glm 0.947, mistral 0.943, dsv4-flash 0.953, dsv4-pro 0.976,
minimax-m3 0.935, gpt-oss 1.000. PCC / max_abs / mean_abs are bit-identical to baseline on all 9
shapes, so the batched stick layout, the wider tilize (nb = 1..4 instantiations), the j*w in0
offsets and the batch-granular CB pushes/pops are correct. No hang.
Note that this node also reverted the parent's read-ahead (0.9416). Against the root this
mechanism alone is roughly -4%. Against the parent it is +2%, and that gain is only the removal of
a01's regression.

## Why (best explanation)
- The work removed (per-block tilize init/uninit + matmul_block_init + one x sem round) is clearly
  small. If it were ~10% of gate/up time, the gain would have shown despite any side effect.
  Together with b03-a01 and b04-a01 this is the third change to per-K-block delivery/overhead that
  failed to help. At M=1 the root is limited by the per-tile matmul/unpack work itself (~28
  cycles per bf4 tile-mult with 1xN subblocks) and the lock-step grid, not by per-block fixed costs.
- The cost came from making x arrival **bursty while its lead time stayed the same**. In the reader,
  x and the weight blocks are handled in the same kb iteration, and receivers wait on in0_valid
  before pushing in1 block kb. The reader can only run ahead of compute by the 2-slot in1 ring,
  so a batch's x is still requested only ~2 K-blocks before it is needed. Now it takes ~4x longer to
  arrive: the x sender reads 32 sticks x 4 KB and multicasts up to 128 KB, and compute then tilizes
  64 tiles before the first matmul of the batch. That latency is exposed once per batch on every row
  and, through the lock-step, on the whole grid. The shapes with the least compute per K-block
  (mistral, minimax, glm, dsv4-flash) lost the most, which fits an exposed latency.
- gpt-oss (bias, per_core_M 1-2, compute-heavy) and kimi-k3 (M_max=8 / narrow w, more K-blocks per
  x batch relative to compute) were neutral. Where compute per block is large the burst is hidden,
  and the saved inits buy nothing measurable.

## Classification
flawed idea (as implemented). Per-K-block init/handshake amortization on the x path isn't worth
anything measurable, and batching x without decoupling its prefetch from the weight loop costs
~4-6%. Implementation correct.

## What a child of this node should try next
- Don't spend more attempts on per-K-block overhead or weight/x read timing in the root schedule.
  Three nodes (b03-a01, b04-a01, this one) say the root's M=1 time is per-tile compute plus lock-step.
- The proven lever is fewer phantom M-rows / more independent pipelines (r01-b01-a01 1.44,
  r01-b02-a01 1.40). The natural next compute lever on top of them is a taller gate/up and down
  subblock (h = per_core_M, w shrunk to fit DST 8), so each bf4 in1 tile is unpacked once per
  h rows. With per_core_M = 2-3 per group there, that attacks the ~28-cycle/tile-mult cost directly.
- If x batching is revisited (only on a lineage where x latency matters), the x sender must prefetch
  batch g+1 right after pushing batch g (x_rm is already double-buffered at batch granularity), and
  receivers should not gate the in1 push of kb on in0_valid unless kb starts a batch whose x is
  still missing.
