# wo_a bank-local weight reads: design (WA.1, CPU only, 2026-11-27)

(MB.NC, 2026-11-28: brought from dnijemcevic/model_bringup_batching as written there. The roofline script and its test
(tests/woa/woa_roofline.py, test_woa_design.py) read that branch's ledger results (PB.MM256) and were not brought, nor
the packed-path model A/B tests/woa/test_woa_offon.py; on model_bringup the layout is checked at op level by
tests/woa/test_gmm_native_layer.py, model-wide by MB.NCL.)

**Upper bound (best case) of the gain, device ms a chunk, layers 0-20 (20 wo_a calls, DECODER_SKIP=decode):**
**8192 chunk (256 rows a chip): 1.19 ms. 5120 chunk (160 rows a chip): 1.00 ms.**
Realistic (all bytes at 410 GB/s, compute at 80 % of HiFi2): 0.73 ms (8192), 0.58 ms (5120).

**Verdict vs the KILL LINE (upper bound < 0.5 ms a 8192 chunk -> stop): NOT killed (1.19 ms >= 0.5).** Go to WA.2,
with one caveat that WA.2 must settle first (section 2, step 0 of section 7): today's numbers do not separate the
reader from the compute. If WA.2's "no weight reads" stub runs within ~10 % of today, wo_a is compute-bound at 256
rows and the layout gains ~0: stop there.

Numbers: `python3 -m models.demos.deepseek_v41_flash_d_p.tests.woa.woa_roofline` (shapes from the checkpoint's
inference config and safetensors header, the mesh from the spec, the core cap from tt/settings.py; today's 256-row
time read from results/PB.MM256.json); gated by tests/woa/test_woa_design.py.

| | 160 rows | 256 rows |
|---|---|---|
| weights MB / x MB / y MB (bfp8, a call a chip) | 35.65 / 5.57 / 1.39 | 35.65 / 8.91 / 2.23 |
| floor: weights at 512 GB/s (us) | 69.6 | 69.6 |
| weights at 410 GB/s (us) (P.81's floor) | 87.0 | 87.0 |
| all bytes at 512 GB/s (us) | 83.2 | 91.4 |
| all bytes at 410 GB/s (us) | 104.0 | 114.2 |
| compute, busiest core (3 column tiles) at HiFi2 peak (us) | 45.5 | 72.8 |
| best case = max(all bytes 512, compute) (us) | 83.2 | 91.4 |
| realistic = max(all bytes 410, compute / 0.8) (us) | 104.0 | 114.2 |
| today (us) | 133.0 | 151.0 |
| today / weights floor (512) | 1.91x | 2.17x |
| upper bound ms a chunk (20 calls) | 1.00 | 1.19 |
| realistic ms a chunk | 0.58 | 0.73 |

Hardware used: Blackhole, 8 DRAM banks (GDDR6 channels; `dram_grid_size().x`, flat plan.cpp:241) x 64 GB/s = 512
GB/s a chip (X.7_roofline.md "Hardware"); this ledger's measurement for bank-local reads: DS41-249 (the flat experts'
bank-local weight stream fits A = 19.9 MB / 512 GB/s, rmse lowest at the DRAM rate, 1.5x / 2x worse), so 512 GB/s is
the rate the layout we copy has been seen to reach; 410 GB/s (80 %) is X.7's planning fraction and P.81's floor.
Compute: 4096 FLOP / cycle a core at LoFi, HiFi2 = 32 cycles a 32x32x32 tile product, 1.35 GHz (GEMM_FLOPS.md).

## 1. Today's wo_a, as the model holds and calls it

- Weight: checkpoint `layers.L.attn.wo_a.weight` fp8 e4m3 [8192, 4096] (= o_groups 8 x rank 1024, hpg 8 x 512);
  tt/attention.py:1571-1579 reshapes to [1, G 8, hpg*512 = 4096, rank 1024] (K rows permuted to [nope | re | ro]),
  `_upload` (attention.py:149): `from_torch(..., dtype=dense_dtype(...) = bfloat8_b (DENSE_WEIGHTS=bfp8, rule 7),
  TILE, memory_config=DRAM = ttnn.DRAM_MEMORY_CONFIG (INTERLEAVED), ReplicateTensorToMesh)` (ATTN_SP32_PROJ=on: every
  group on all 32 chips). 8 x 128 x 32 = 32768 bfp8 tiles of 1088 B = 35.65 MB a layer a chip. Converted at every load
  (no disk cache).
- Call: attention.py `_out_wo` (~2275-2297): `ttnn.bringup.grouped_matmul(oa [1, 8, rows, 4096] bfp8, self.wo_a,
  dtype=bfp8, compute_kernel_config=ckc (ATTN_FIDELITY: HiFi2, fp32 DEST), output_tensor=host_alloc buffer,
  max_grid=max_grid() (11 x 10, MAX_WORKER_CORES 110), block_w, depth, rt)` when ATTN_ROOFLINE and `supported`.
  Knobs: `mm.gmm_knobs` (mm_configs.py:672): at 256 rows GMM256 (MM_TUNED256=on, default) `(block_w 2, depth 2, rt 1)`;
  at 160 rows the settings `WO_A_GMM_BLOCK_W 4`, `WO_A_GMM_DEPTH 4`, rt 1.
- Program (grouped_matmul.py `program`): one grid row a group (G = 8 rows), GX = min(11, Nt 32) = 11 columns,
  `column_split(32, 11)` = 10 cores x 3 column tiles + 1 x 2: 88 cores. Per core: all MT row tiles x its PN columns
  of one group, K in NBLK = 128 / BW blocks.
- Reader (kernels/gmm_reader.cpp), per K block: every core issues BW x PN single-tile `noc_async_read`s (page 1088 B,
  TensorAccessor on the interleaved weight) = 6 reads (256 rows, bw 2) or 12 (160 rows, bw 4); the row's column-0 core
  also issues MT x BW x reads (16 or 20 tiles); then `noc_async_read_barrier()`: **one block of reads in flight a core,
  then a full drain**. Then the x multicast handshake (sender waits READY from the 10 receivers, multicasts the
  [MT x BW] block, sets VALID; receivers inc READY, wait VALID): the 11 cores of a row advance in lockstep with their
  sender.
- Which banks: an interleaved page p sits in bank p % 8. Weight page = g * 4096 + k * 32 + n, and 4096, 32 are
  multiples of 8, so **tile (g, k, n) is in bank n % 8**: core (cx, g) reads its 3 columns n0..n0+2 from 3 fixed banks
  (n0 % 8 ..), the same 3 banks for all 128 K tiles, and the 8 cores of a grid column (one a group) read the same 3
  banks. Over the chip each bank serves 4 columns x 8 groups: balanced on average, but every core reads banks across
  the chip (the cores sit in columns 0-10, the bank anchors in columns 0 and 7), one tile at a time. x page =
  g * MT * 128 + m * 128 + kb * BW + k: bank (kb * BW + k) % 8, so in block kb all 8 senders read the same BW banks.
- Measured: **160 rows 133.0 us** (P.81 attempt 2 op probe, 8x4, rt profiler, slowest chip, 20 iterations, bw 4
  depth 4; grouped_matmul CHANGELOG, BREADCRUMBS.0004 "P.81 (perf, attempt 2)", DS41-320); **256 rows 150.96 us**
  (results/PB.MM256.json `mm256_confirm_us_attn_wo_a_gmm_256x4096x1024_on_tagentic_cc`, bw 2 depth 2, every chip;
  the in-model off side bw 4: 3147.5 us / 20 calls = 157.4 us, PB.MM256_ladder.md section 1). 20 calls a chunk.
- Versus the floor: the "2.0x" of the plan is P.81 attempt 1's stock 2D program (0.175 ms vs 0.087 = weights at 410
  GB/s). Today's fork is 1.53x / 1.73x of that floor, and **1.91x / 2.17x of the weights at 512 GB/s** (69.6 us); it
  moves 320 / 310 GB/s of all its bytes (weights + x + y) and runs 0.92 / 1.29 TFLOP/s a core (34 % / 48 % of the
  HiFi2 peak) at 160 / 256 rows.

## 2. Where the gap goes

Per K block, today (measured time / NBLK): 160 rows 133 / 32 = 4.16 us a block; 256 rows 151 / 64 = 2.36 us.

| candidate | estimate a block (160 / 256 rows) | evidence | the bank layout |
|---|---|---|---|
| compute (HiFi2, busiest core: MT x BW x 3 products) | 1.42 / 1.14 us at peak; 1.78 / 1.42 at 80 % | 34 % / 48 % of today; DS41-299: 84 % of peak seen on the flat expert cores | does not change it (same products a core: 256 column units over <= 110 cores is >= 3 a core) |
| one block of weight reads in flight + full drain (12 / 6 tiles, 13 / 6.5 KB a core) | ~1 us DRAM + NoC round trip a block, not overlapped with the next block's reads | Little: 512 GB/s x ~1 us needs ~0.5 MB in flight; today 88 x 6.5 KB = 0.57 MB only at the wave's peak, 0 at each drain | fixes it with 2+ blocks in flight, each ONE contiguous read (DS41-319: 2 in flight) |
| single-tile reads across the chip (NoC distance, 1088 B packets) | 6-12 packets a block a core, each crossing the grid from a far bank | cores in columns 0-10 read banks whose anchors are columns 0 / 7; 8 cores share the same 3 banks | fixes it: one 6.5-13 KB read from a bank 0-2 hops away |
| x sender chain + multicast lockstep (sender: MT x BW x tiles + own weights, then READY / mcast / VALID) | sender reads 2.7x (160) / 3.7x (256) a receiver's bytes a block, then multicasts 21.8 / 17.4 KB | the 11 cores of a row wait for their sender every block | does not fix it by itself: needs x read one block ahead (x_prefetch) |
| x bank hotspot (block kb's x pages on BW consecutive banks for all 8 senders) | 139 KB (256 rows) / 174 KB (160) on 2 / 4 banks a block | bank = (kb BW + k) % 8; but bw 2 measured faster than bw 4 at 256 rows (149.6 vs 154.2 us), the opposite of what a hotspot predicts: minor | clusters run independently (no cross-row lockstep), so the senders drift apart |
| pipeline fill / drain (first block read before any compute, last compute after the last read) | ~1 block | P.81: bw 4 / 8 / 16 / 32 = 0.133 / 0.140 / 0.144 / 0.150 ms (larger blocks, longer fill) | unchanged |

What existing data cannot tell: whether the steady state is reader- or compute-bound. Both scale with the bytes a
block (bw 2 vs bw 4 at 256 rows: 2.34 vs 4.82 us a block, ~2.06x for 2x the bytes), so the bw sweep does not
separate them. At 256 rows the compute at peak is 72.8 us, 80 % of the 91.4 us DRAM best case: a compute efficiency
below ~64 % (72.8 / 114.2) makes the call compute-bound above the realistic DRAM time, and the layout would gain
little there. At 160 rows compute (45.5 us at peak) is far below DRAM (83-104 us): reads dominate, the layout's case
is strongest there.

The bank-local layout fixes the weight side (distance, packet count, reads in flight, per-instant bank balance); it
does not fix the compute, the x stream (x stays interleaved: it is the previous op's output), the output writes or
the fill / drain. The bound above assumes the reader changes that come with it (2 weight blocks in flight, x read one
block ahead) and that x and y also move at 512 GB/s.

## 3. The per-bank layout

- Units: (group g, column tile n), 8 x 32 = 256 units of 128 K tiles (139 KB each). Bank b holds the units
  [32 b, 32 b + 32): **with G = 8 = banks, bank b holds group b entirely** (all 32 N columns, all 128 K tiles: 4096
  tiles = 4.46 MB; every bank the same size, no padding, OV-3). On a box with another G a chip the same run split
  applies (G = 4: half a group a bank; G = 16: two groups a bank, a sub-cluster each).
- Cores of bank b: its group's 32 columns split by `column_split(32, c)`: **c = 12 (default): 8 x 3 + 4 x 2 = 96
  cores**; c = 11: 10 x 3 + 1 x 2 = 88 cores (today's split). Either is >= 88 and <= 110 (MAX_WORKER_CORES, FIX.15,
  rule 18). No core count <= 110 brings the busiest core below 3 columns (256 / 110 > 2; 86 cores is the minimum at 3).
- Region of a core in its bank: its PN columns **K-major**: tile (k, j) at tile0 + k * PN + j; the bank's cores'
  regions back to back (tile0 = the earlier cores' 128 x PN tiles). A K block of any BW is then ONE contiguous read of
  BW x PN tiles at tile0 + kb * BW * PN, already in today's CB slot order [k][j] (PN = PNX = 3: one read a block; a
  PN = 2 core reads BW rows of 2 tiles at L1 stride PNX, as today's narrow core). The layout does not depend on BW, so
  one copy serves 160 rows (bw 4) and 256 rows (bw 2).
- DRAM tensor: WIDTH_SHARDED DRAM, one (rows, 32) shard a bank (bank b = column block b of a [131072, 256] bfp8 host
  matrix), exactly `flat_routed_expert_weights.bank_host_tensor` / `_bank_memory_config` / `bank_sharded(...,
  replicate=True)` (reused as is). The reader addresses `get_noc_addr_from_bank_id<true>(bank, w_addr + tile0 * 1088 +
  kb * BW * PN * 1088)` as gate_up_weight_read.hpp does.
- Placement: the bank's optimal worker (`mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(NOC_0)`,
  Python-bound, distributed_nanobind.cpp:594) is its anchor; anchors sit in logical columns 0 (west banks) and 7 (east
  banks) on a Galaxy chip's 12 x 10 grid (flat plan.cpp:247). Cluster of bank b: a **6 x 2 rectangle** on its side
  (west columns 0-5, east 6-11) in the 2-row band nearest its anchor (4 disjoint bands a side of the 10 rows; 2 spare
  rows to align the bands with the anchors). With c = 11 the rectangle's 12th core idles (x is multicast to the
  rectangle; keep it out of any other work) or the cluster is 2 rectangles. The experts place cores by NoC0 hop
  distance one by one (plan.cpp:545) because their x arrives by relays; wo_a needs x[g] at every core of the cluster,
  hence rectangles (one multicast).
- How x reaches them: x [1, 8, rows, 4096] stays interleaved (written by `gemm_input`). In each cluster one sender
  (the core nearest the anchor; option x_senders: 2 split the K block's x reads) reads x[g]'s [MT x BW] block and
  multicasts it to its rectangle (loopback write, the sender is inside), reading block kb + 1 while kb is multicast
  (x_prefetch). Total x bytes unchanged (each x tile read once); the 8 clusters do not wait on each other.
- Same tiles, same K order, same fidelity / accumulation (owner constraint 1): the compute kernel is unchanged
  (gmm_compute.cpp: per block matmul_block over k < BW into fp32 DEST, the packer adds into Float32 partials in L1,
  one copy + pack at the end), BW and the block order are the call's (today's knobs). Each output tile's sum does not
  depend on which core computes it (DS41-320: the 2-column core is bit-identical to the 3-column ones). Tiles move
  whole and keep their orientation, so each bfp8 tile (16-value shared exponents inside the tile) has the same bytes.
  The CPU model of the layout (tests/woa: `bank_regions` + `grouped_mm_from_banks`) reads every block from its bank
  region and reproduces x @ w exactly for BW 1-8 and both core counts.
- Reused as is from the experts: `bank_host_tensor`, `_bank_memory_config`, `bank_sharded(replicate=True)`
  (flat_routed_expert_weights.py), the reader's bank-id addressing and transaction-id window
  (gate_up_weight_read.hpp: DEPTH reads in flight under TRIDs). Not reused: `flat_weight_regions` (expert-shaped:
  gate/up pairs, down rotation) and plan.cpp's placement (C++ inside the flat op; its rule, nearest by NoC0 hops, is
  re-stated in Python for rectangles).

## 4. The switch, DRAM and load cost

- **WOA_BANK = off | on, call-time, both layouts resident** (owner constraint 2: call-time preferred if the DRAM
  allows). The bank copy is +35.65 MB a layer a chip: +0.75 GB (0.70 GiB) for the perf load's layers 0-20, +1.43 GB
  (1.33 GiB) for all 40 layers. Budget at 1M history with packed segments (results/PB.8RP_packed_walls.md, 4 x 2k
  lanes, slots at capacity 1048576): 40-layer model 24.32 GiB of the allocator's 31.83 GiB, headroom 7.51 GiB -> 25.65
  GiB, headroom 6.18 GiB with both copies. Fits; one load serves the A/B (off vs on in one process, as test_mm256_offon
  does). Both sides are one prepared program a call (no extra launch; the program cache keys on the layout); the cold
  no-rebuild test must stay PASS. After acceptance the interleaved copy can become load-time only (owner decision),
  returning the 1.33 GiB.
- Load cost (constraint 4): the region cut + bank_host_tensor on the host measured 23-38 ms a layer here (CPU, a
  [8, 4096, 1024] bf16 tensor; tests/woa), ~1-1.5 s for 40 layers, plus one more bfp8 conversion and replicated upload
  of 35.65 MB a layer (today's `_upload` of the same bytes; the bfp8 statistics of dense_dtype need not run twice).
  Estimate: a few seconds a load; WA.3 measures the load-time delta. No cache unless it measures > ~10 s.

## 5. Roofline of the new layout

Per call (table above): bytes = weights 35.65 MB + x (5.57 / 8.91 MB) + y (1.39 / 2.23 MB); compute: the busiest core
does 3 x MT x 128 tile products (45.5 / 72.8 us at the HiFi2 peak). Best case max(83.2, 45.5) = **83.2 us at 160
rows**, max(91.4, 72.8) = **91.4 us at 256 rows**, DRAM-bound both; realistic 104.0 / 114.2 us. Gains a call: 49.8 /
59.6 us best, 29.0 / 36.7 us realistic; x 20 calls: 1.00 / 1.19 ms best, 0.58 / 0.73 ms realistic a chunk. At 256
rows the compute floor is 80 % of the DRAM floor: the bound is tight to compute.

## 6. Options of ttnn.bringup.grouped_matmul and the fork tests

Extend the fork (constraint 3), defaults = today's behaviour:
- `w_layout=None` (default: today's interleaved path, unchanged) | a `BankLayout` from the new
  `grouped_matmul_ttnn/bank.py`: `plan(device, G, K, N, max_cores, cores_per_bank=12)` (anchors, rectangles, the
  CoreSlot table: bank, core, g, n0, pn, tile0; refuses > max_cores), `pack(w [1, G, K, N] torch, layout)` (the K-major
  region cut + bank_host_tensor), `upload(host, mesh)` (bank_sharded replicate). The call checks the tensor is
  WIDTH_SHARDED DRAM with the layout's shard shape.
- Bank path knobs: `w_inflight` (weight K blocks in flight a core under transaction ids; default 2), `x_prefetch`
  (sender reads block kb + 1 before multicasting kb; default True), `x_senders` (1 | 2). `block_w`, `depth`, `rt` as
  today. The program-cache key adds the layout and the knobs; the prepared program patches x, w, y (same patch count).
- Fork tests (tests/unit/test_grouped_matmul.py, new cases): bank vs interleaved **bit-identical** at the 8x4's wo_a
  shapes (8 x [160 | 256, 4096] @ [4096, 1024] bfp8, bw 2 / 4, depth 2 / 4, c 11 / 12, every knob value); every chip
  equal; a repeat on fresh buffers the same bytes (determinism) and a prepared-program cache hit; a small G = 4 /
  2-rectangle case; refusals (interleaved tensor with a layout, a layout over max_cores); a CPU-only pack test (every
  tile once, bank sizes equal, contiguous blocks for every BW). Full regression of the existing cases with the option
  off vs the recorded baseline (skill section 3).

## 7. WA.2: the probe and its knob sweep

- File: tests/perf/test_woa_bank_probe.py, `BATCH_SHARED = True`, mesh only (no model load) through the resident
  runner; random bfp8 weights / x of the model's shapes (the layer's bfp8 values change nothing in the timing),
  rt profiler, slowest chip, 20 iterations a config.
- Step 0 (the decider, before building the bank path): today's op with diagnostic stubs (compile-time defines of the
  probe, marked diagnostic): (a) as is, (b) no weight reads, (c) no compute, (d) no x read / multicast, at 160 and 256
  rows. If (b) >= 0.9 x (a) at 256 rows: compute-bound, the bank layout cannot reach its bound at 8192: report, stop.
- Step 1: the bank path at 160 and 256 rows vs today, output bitwise equal on every chip, 2 repeats bitwise.
- Sweep in parallel across chips: one `MeshProgramDescriptor` maps a different knob set to each of the 32 mesh
  coordinates (each chip compares with today's op on its own data), so one launch times 32 configs: w_inflight
  {1, 2, 3} x x_prefetch {0, 1} x x_senders {1, 2} x block_w {2, 4, 8} x depth {2, 3, 4} (108 a row count, 4 launches
  each at 160 / 256 rows); cores_per_bank {11, 12} per launch (it changes the packed tensor). If per-coordinate
  programs do not prepare, fall back to one knob set a launch and sweep in a loop (same opens: 0, the runner's mesh).
- KILL (WA.2) if the best measured gain is < half the upper bound (8192: < 0.60 ms a chunk, i.e. > 121.2 us a call at
  256 rows; 5120: < 0.50 ms, > 108.1 us at 160 rows) or if any output is not bit-identical or not repeatable.

## WA.2: what the build measured (2026-11-27; results/WA.2_probe.md)

- Step 0 decided go: today's no-weight-reads stub 0.707x (256 rows) / 0.540x (160), no-compute 0.99x (reader-bound).
- Departures from sections 3, 6 and 7, each measured: the x sender is the rectangle's FIRST core, not the core next to
  the anchor (a NoC-0 multicast from inside the rectangle wraps round the chip: 342 vs 155 us at 256 rows); the x
  read-ahead is non-blocking (`x_ahead`, 0-3 blocks, never waits for a slot) instead of `x_prefetch`; `x_senders` 2 not
  built; the weight goes through `bank.upload` (plain bfp8 conversion, then the sharded placement; `bank_sharded`'s
  route re-quantizes bfp8); 10 cores a bank (5 x 2) beat 12; the sweep ran as one mesh program with a knob set a chip.
- Result: 151.3 -> 125.0 us (256 rows), 131.5 -> 106.9 us (160 rows), bit-identical; gain 0.552 / 0.517 ms a chunk at
  21 calls vs the KILL line 0.595 / 0.50: killed at 8192 by 3.8 us a call, passes at 5120.

## WA.3: in the model (2026-11-27; results/WA.3_offon.md)

- Departures from sections 3-6, each measured: the bank cores stay in the matmul-capped 11 x 10 grid (bank.plan
  `grid`; a packed chunk's wo_a runs in the gather-hide region's compute sub-device, DS41-382): 130.3 / 114.2 us a call
  instead of WA.2's 124.9 / 104.1; one kernel group (`one_group`: the column tiles a runtime arg) so the launch has 4
  core ranges, not 24 (the host delta, DS41-394); today's core set ("rect") measured and not used (147.7 / 121.2 us).
- Switch: WOA_BANK call-time; WOA_BANK_COPIES=selected (one copy, the shipped load) | both (A/B) per OV-77.
- Result: off == on bit-identical (single 8192, packed 4 x 2k, single 5120, real-input 160 rows), repeats exact, cold
  no-rebuild PASS; wo_a 3.06 -> 2.65-2.74 ms a chunk; host a call unchanged; load +1.39 s (21 layers).
