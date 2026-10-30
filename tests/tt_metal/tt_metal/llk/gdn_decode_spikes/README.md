# GDN decode v2: spike notes (2026-10-30, p150 `bh-qbae-12`, tt-metal main 7aad8aacd5b tree + spike branch `iwrosz/gdn-decode-spikes`)

Throwaway experiments that answer one design question each before the step-1 design of the kernel-first plan is frozen.
Code: `scratchpad/spikes/` (G) and the branch `iwrosz/gdn-decode-spikes` (A, B); results below.

## Spike G: host emulation of the device numerics (done)

`spike_g_numerics.py`, torch. Model: FPU operands truncated to 10 (in0/SrcB) and 9 (in1/SrcA) explicit mantissa bits,
exact products, fp32 accumulation in k order, SFPU ops fp32 with one rounding, copy_tile exact (UnpackToDestFp32).
HV 12, K = V = 128, 64 tokens vs an fp64 recurrence, two decay regimes.

| form | what it models | max rel err of S after 64 tokens, g in [-2, 0] | heavy decay g in [-6, -4] | T 8 == 8 x T 1 |
|---|---|---|---|---|
| direct | today's kernel: FPU decay pass (S truncated), FPU residual, rank-1 on a seed, q.S'' | 3.0e-3 | 2.7e-3 | yes |
| two_row | tier B: [q;k].S_old, SFPU residual, SFPU decay in the acquire + rank-1, FPU q.k | 1.8e-3 | 1.5e-3 | yes |
| two_row_sfpu_dot | same, q.k on the SFPU | 1.8e-3 (o PCC no better) | 1.5e-3 | yes |
| two_row_S10 | hypothetical: S and u read at 10 bits (SrcB) instead of 9 | 1.2e-3 | 1.2e-3 | yes |
| lazy_persist | running scale c on the stored state, persisted across calls | 2.0e-3 | 2.0e-3 | yes |
| lazy_fold | running scale folded into the state at every call boundary | 2.0e-3 | 2.0e-3 | **no** (max dS 9.4e-4 / 7.0e-4) |

Every form keeps PCC >= 0.999999 on o and S at every step (gate N-2: 0.9999). Findings:
- The two-pass form is ~40 % more accurate than today's (one truncating pass over S fewer, exact decay); the SFPU dot
  product for q.k buys nothing over the FPU one.
- Operand placement is worth ~1.5x: S and u at 10 bits would halve the error again, but the row-vector-times-matrix
  layout forces S into in1/SrcA (9 bits). HiFi2 keeps the full SrcA precision against ~7 bits of SrcB (S-6 item 5),
  so HiFi2 with S in SrcA is the variant to measure on silicon later, not a default.
- The lazy decay only keeps the T = K + 1 vs chained T = 1 identity (N-3) if the scale is persisted with the state;
  folding it at a call boundary rounds differently. Renormalisation horizon (c < 2^-60): 42 / 14 / 8 tokens at a mean
  g of -1 / -3 / -5. Decision 4 of the plan: SFPU decay inside the update acquire for v2; the lazy scale is a later
  lever that changes the pool format.
- Caveat: the real MVMUL's 16-term face accumulation holds ~12-13 bits (prefill hi/lo probe), so absolute device
  errors will sit above these; the ranking of the forms does not depend on it.

## Spike F: dispatch cost of common versus unique runtime args (done)

`test_pgm_dispatch` (Google Benchmark, `tests/tt_metal/tt_metal/perf_microbenchmark/dispatch`), all processors, all
cores, under trace, kernel sizes 256 and 2048 B (identical results). Grid (0,0)-(10,8) = 99 cores on this p150, 1.35 GHz.

| program | per launch |
|---|---|
| no runtime args, no CBs | 0.95 us |
| 1 CB | 1.64 us |
| 1 common runtime arg | 1.64 us |
| 32 common runtime args | 1.65 us |
| 1 unique runtime arg per core | 4.18 us |
| 32 unique runtime args per core | 4.39 us |
| 128 unique runtime args per core | 5.12 us |

Pass criteria: common-only <= 2.0 us: **met** (1.64). Unique minus common >= 3 us: 2.54 us at 99 cores (~25.6 ns per
core per launch, ~2.8 us extrapolated to 110 cores), so the effect is a little smaller than the plan's figure but of the
same kind: one unique runtime arg per core costs 2.5x the whole launch. The rule stands: unit ids from core coordinates
and common args; buffers through CB aliasing and common args (S-6 item 8).

## Spike A: tiny-tile matmuls (done)

`tests/tt_metal/tt_metal/llk/test_gdn_tiny_tiles.cpp` + `test_kernels/compute/gdn_tiny_tiles.cpp` +
`test_kernels/dataflow/reader_gdn_tiny_tiles.cpp` (spike branch; `unit_tests_llk --gtest_filter='*GdnTinyTiles*'`,
4 tests, all logging). One core, K = V = 128 (4 x 4 fp32 tiles), HiFi4, fp32 DEST, `matmul_tiles` with the tile dims
taken from the CB descriptors (`set_tile_dims`), 16 repetitions, MATH-thread wall clock.

| experiment | result |
|---|---|
| two-row read, in0 [q;k] as a **2x32 / 4x32 / 8x32** tile (partial face) x S 32x32 | **wrong values** (max rel err 0.8-1.0 vs double; k.k came out 0.49 for 1.0: half the terms lost), finite, not bit-equal to the padded form |
| two-row read, in0 as **16x32** (the SDPA-decode shape) | **bit-identical** to the padded 32x32 in0; max rel err 1.33e-3; NaN in rows 2-15 leaves rows 0-1 unchanged |
| cycles per in0-tile x S-tile product, in0 height 2 / 4 / 8 / 16 / 32 | 167 / 170 / 173 / 179 / 197 (gate was <= 24): the cost is the 4 KB fp32 S-tile unpack and the DEST pipeline, not the MVMUL row count |
| rank-1 update, in0 k^T **32x16** x in1 u **16x32** onto an exactly seeded S tile | **bit-identical** to the padded 32x32 x 32x32 form; max rel err 3.8e-4 |
| the same with a tf32 seed (copy_tile without UnpackToDestFp32, today's kernel) | max rel err 8.5e-4 (2.2x worse) |
| NaN in k^T's padding columns, u's padding rows zero | no leak, bit-identical to the clean run (0 x NaN reads as 0 on the FPU) |
| NaN in both paddings | finite but not bit-identical: garbage. Rule: zero at least one side; the plan's "u built in a zeroed DEST tile" suffices |
| cycles per S tile, groups of 4: matmul alone tiny / padded; copy seed alone; copy + matmul | 525 / 540; 566; 592 (gate <= 48 was for the MVMUL alone) |
| copy_tile of an fp32 tile with / without UnpackToDestFp32 | 0 / 16381 of 16384 words changed; 577 / 525 cycles per tile (copy + fp32 pack) |
| scalar q.k with in0 2x32 / 16x32 / 32x32 | wrong / right / right (k.k 0.489 / 0.99898 / 0.99898 for 1.0) |

Conclusions for step 1a:
- **Partial-face in0 tiles (height < 16) are not usable from `matmul_tiles` on Blackhole** in this configuration
  (the in-tree tiny-tile matmul matrix was verified on Wormhole). The token vectors go in **16x32 tiles**: [q; k] in
  rows 0-1 with zero padding, which is bit-identical to the padded form at 179 instead of 197 cycles per product.
- The two-row read costs ~180 cycles per S tile whatever the in0 height, so the read pass is ~2.1 us per head-token
  at Vtl 4 (16 products), bound by unpacking S, not by the MVMULs.
- The rank-1 update with 32x16 / 16x32 operands works and saves only ~3 % over padded operands; its ~590 cycles per
  S tile are the fp32 tile moving L1 -> DEST (copy seed) and DEST -> L1 (pack). The state write-back is the floor of
  the update pass: ~16 x 590 = 9500 cycles = 7.0 us per head-token at Vtl 4, which matches today's "~7 us of math".
  The hardware model's 96 cycles per S tile (S-6 item 3) is 6x too optimistic; the V-split (step 3) is the lever that
  divides it, and the trajectory table's "after step 1" at B 1 should read ~15-20 us rather than 10-15.
- Today's kernel seeds the rank-1 update through a tf32 copy (no UnpackToDestFp32) and also truncates the state in
  the FPU decay pass: two avoidable precision losses per token; the exact seed is a factory flag away (N-6 item).
- The DEST pipeline moves an fp32 32x32 tile in or out at ~250-290 cycles, not the 64 cycles (64 B/cycle) assumed in
  S-6 item 2; every per-token estimate that counts 4 KB tile moves should use the measured figure.

## Spike B / B2: in-place state pool (B in progress, B2 by inspection)

**B2 (DRAM pool + indices) is already in today's op as "ring mode"**: `initial_state_block_idx` ([BH] int32/uint32,
one page) makes head h start from block idx[h] of a `[T*BH, K, V]` ring, every per-token state is written back in
place (`create_output_tensors` returns the input tensor as the state output; the writer writes block t*BH + h), and
the Qwen3.6 verify path runs it under trace. What the ring lacks for step 2 is the general slot layout (the contract
`idx[h] % BH == h` and the exact ring shape), the negative index that skips a row (the reader can skip its reads, but
the compute kernel must also idle: a one-word control CB from the reader, or T = 0 for that core through common args,
is the design item), and `num_accepted_tokens`. The in-place aliasing, the index read and the cache-hit re-point of
the index and pool buffers are proven by the model's use; the spike adds nothing there.

**B (L1-sharded pool aliased by the state CB)**: implemented on today's op as `inplace_final_state=True` (spike
branch): the pool is `[BH, K, V]` fp32 height-sharded in L1 with one [K, V] shard per head core (the shard grid is the
factory's head -> core map, column-major), the state CB carries `CBDescriptor.buffer = pool.buffer()`, the compute
kernel's pack thread publishes the resident CB at start and packs the last token's state back into it, the reader no
longer reads the state and the writer no longer writes it, and the op returns the pool tensor.
Finding on the way: the first version let the *reader* publish the resident CB and the *pack thread* later push the
new state into the same CB; that is two producers for one CB, and since each producer keeps its received count
locally, the unpacker's final `wait_front` can never be satisfied: the device hung. The CB protocol allows exactly one
producer thread per CB; the publish moved to the pack thread. (Each hang needed `tt-smi -r 0,1,2,3` plus ~25 s before
the board accepted work again; a reset used too early left even a trivial add hanging.)
Results (`spike_b_inplace_pool.py`, p150, HV 12, B 1): one step `torch.equal` to the fresh-tensor path on o and the
state, the returned tensor is the pool (same address); a program-cache hit with the pool reallocated at a new address
(0x170000 -> 0x160000) gives the same bits, so the descriptor path re-points the globally allocated CB on cache hits
(13 program-cache entries, 0.7 ms per eager call); 100 chained steps in place equal 100 fresh-tensor steps at every
checkpoint; a trace of one in-place step replayed 100 times equals 100 eager in-place steps, and the replay's result
is the pool. Step 4's plumbing (S-6 item 7) is proven on today's op: shard = CB alias, in-place pack, cache-hit
re-point, trace; what remains for step 4 is the NV axis in the shard layout and the capacity policy.

## Spike E: conv state update (done)

Results (p150, `unit_tests_llk --gtest_filter='*GdnConvUpdate*'`, 3 passed): y bit-exact to the host fp32 reference
on all 1024 words (0 bf16 ulp: the device's exp polynomial agrees with std::exp after the bf16 rounding), the slot
holds the last W-1 inputs, the other slots are byte-identical, a negative slot index leaves the pool and the y rows
untouched, and one call of T 4 equals four chained T 1 calls bit for bit on y and on the state. The step-5 design is
therefore settled on the contract side: pool `[slots, W-1 (+num_spec), D]`, token-major x, a slot index with a skip
value, the window shifted by T, SiLU in fp32 before one bf16 rounding. What this spike does not say is the cost: the
software FIR on a RISC is a probe; the op will do the taps on the compute engine (W tile-rows of bf16 x a broadcast
weight row, SFPU SiLU), where the 11-program FIR it replaces is the baseline.

`tests/tt_metal/tt_metal/llk/test_gdn_conv_update.cpp` + `test_kernels/dataflow/gdn_conv_update.cpp`: one
data-movement RISC, conv-state pool `[slots, W-1, D]` bf16 row-major in DRAM, x `[T, D]`, weight `[D, W]`, bias, a
one-row int32 slot index (negative = skip), FIR + SiLU in software fp32 (exp by range reduction + degree-6 Taylor),
bf16 RNE on the way out, the slot shifted by T rows. Three tests: one call vs the host reference (<= 1 bf16 ulp, slot
holds the last W-1 inputs, other slots untouched), a skipped row (pool and y untouched), one T = K + 1 call vs K + 1
chained T = 1 calls bit for bit. D 256, W 4, 4 slots.

## Spike C: bit identity on today's op (done, lands with #59526)

`test_multi_token_equals_chained_single_token`: K 3 / 7 / 11, B 1 / 8, `torch.equal` on o and every state; 6 passed
on the p150 today in the rebased #59526.

## Spike E: conv state update (pending)
