# moe_compute: the decode MoE kernel with per-expert weight residency (C1)

`ttnn.experimental.moe_compute` (the shared op under `ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute`) is the
largest compute item of the Gemma 4 26B-A4B decode step on the Galaxy lane engine: 416 us per layer, 12.7 ms per
128-user step, against a 26 us DRAM floor (`_logs/galaxy_decode_roofline_analysis.md` C1). This folder is its
`ttnn.generic_op` port with the weight stream rewritten so each expert's weights cross DRAM once per call, whatever the
token count, and a deeper DRAM prefetch. `op.py` builds the same program the shared op's factory builds for its
ComputeOnly path (same cores, CBs, semaphores, compile-time and runtime args), `kernels/` are the shared op's kernels
copied and modified (the headers `moe_ring_common.h` / `hostdevcommon/config.hpp` / `moe_utils.hpp` are included from
the shared tree by path, so the weight layout stays the one `prepare_w0_w1_tensor_for_moe_compute` /
`prepare_w2_tensor_for_moe_compute` pack).

## 1. The shared op's data flow per core (gemma_4_26b: H = 2816 = 88 tiles, N = 704 = 22 tiles, 4 experts/chip, 8-bank ring)

Three core groups on each chip, one program (`select_moe_compute_cores`):

| group | cores (BH 13x10 grid) | kernels | role |
|---|---|---|---|
| tilize | 4 (2x2 block, top-right; `tilize_cores[0]` = drain) | tilize_reader (NCRISC/NOC1), tilize_writer (BRISC/NOC1), tilize_compute | scan the routing (indices/scores on the drain core, mapping) into `e_t` (token list per local expert, 16 B entries: token id, k slot), the activation table and the per-expert counts; then per expert, per chunk of 32 tokens: gather the token rows of the dispatch buffer (each tilize core a quarter of the hidden dim), tilize, multicast the `[32, H]` bf16 chunk into the ring cores' shared CB `c_0` (two halves, ping-pong), signal `matmul_chunk_ready` |
| matmul ring | 8 (one next to each DRAM bank) | dm0 (NCRISC/NOC0: weights), compute, dm1 (BRISC/NOC1: ring a2a + output) | per chunk: gate/up for this core's 3 (or 2) of the 22 intermediate columns, GELU * up packed to `c_6`; a2a ring exchange of `c_6` so every core holds the chunk's full `[32, N]` intermediate; this core's 11 of the 88 output columns of `in2 @ W2`, untilized into `c_1`; dm1 copies the chunk's rows to the combine cores' staging L1 |
| combine | 16 (hs 4 x ws 4) | (FullCcl) selective_reduce_combine reader/writer + fabric mux | send each staged row `(token, k)` back to the token's home chip |

Per ring core per chunk the shared dm0 streams the expert's whole slice again:

| stream | per core per expert | per chip per expert | layout |
|---|---|---|---|
| W0/W1 (gate, up) | 20 blocks x 28 tiles (cores with 3 cols; 13 blocks for 2 cols) = 560 tiles = 322,560 B | 146 blocks = 2.35 MB | compact bank-balanced: 4-wide block-columns (W0 c, W1 c, W0 c+1, W1 c+1) x 7 K rows, zero-padded to 19 blocks per bank; a core's slice may straddle two banks |
| W2 (down) | 12 blocks = 336 tiles = 193,536 B | 1.55 MB | per core: 3 a2a iterations x 4 K-blocks of 4 output tiles x 7 K rows |

`dm0.cpp` keeps `weight_cb_slots - 1 = 2` blocks (2 x 16 KB) of DRAM reads in flight per core (one transaction id per
block, 2 x 8 KB transactions per block) and re-walks the expert's W0/W1 then W2 stream for every 32-token chunk
(`for expert / for chunk / for block`). The compute kernel consumes block by block (`cb_wait_front(28)`, 4-wide
`matmul_block` over K, `cb_pop_front(28)`), so the weights never stay in L1. With 256 frame rows the pad rows (ids
0..7, score 0) make device 0's four experts see 128 extra tokens each: 5 chunks per expert = 20 streams of 3.35 MB
instead of 4.

Bytes per call today: `chunks x 13.4 MB` per chip (device 0: 67 MB at 128 users, 107 MB at 4 users; the other
chips 13.4 MB), at 2 x 16 KB in flight per core = latency-bound at ~24 GB/s per core (~190 GB/s per chip).

## 2. New streaming order (this port)

Per ring core, per expert with at least one token, in `dm0_moe_compute.cpp`:

1. `cb_reserve_back(c_3, w01_tiles)`: the whole W0/W1 slice of the expert (560 tiles, 322 KB) is one CB region,
   reserved at once; the reads of all 20 blocks are issued back to back, one 16,128 B NoC read per block on Blackhole
   (a block never straddles a bank piece: pieces are whole blocks), `prefetch_depth` (default 8) transaction ids in
   flight; as the oldest lands it is pushed (`cb_push_back(28)`), so the compute of chunk 0 chases the stream.
2. `cb_reserve_back(c_10, w2_tiles)`: the W2 slice (336 tiles, 194 KB) the same way into its own CB.
3. The compute kernel (`compute_moe_compute.cpp`) addresses tiles by absolute index inside the resident region
   (`in1_index = block * 28 + k`), waits cumulatively (`cb_wait_front(c_3, (block + 1) * 28)`) only while the first
   chunk chases the stream, and pops the region once: W0/W1 after the expert's LAST chunk's gate/up phase, W2 after
   its last chunk's down phase. Every later chunk of the expert re-reads L1.
4. dm0 therefore streams expert e+1's W0/W1 while the compute is in expert e's last chunk's down phase (the W0/W1
   region is free), and e+1's W2 as soon as e is finished.

Chunk order, matmul_block sequence, operands, K order, activation and pack calls are unchanged: the math is
bit-identical to the shared op (the unit test asserts it). Bytes per call: `13.4 MB` per chip regardless of the
chunk count (device 0: 67 -> 13.4 MB). In flight per core: 8 x 16 KB = 128 KB (was 32 KB).

Pad rows: `skip_zero_scores=1` (env `GEMMA4_MOE_COMPUTE_SKIP_ZERO_SCORES=1`, default off) makes the tilize scan drop
every `(token, k)` whose score is exactly 0 (the frame's padding), so the pad tokens are neither tilized nor computed
nor staged; the fast reduce multiplies those output rows by 0 (the persistent combine output buffer is zero-filled at
allocation and only ever holds previous finite outputs). The multi-chip combine protocol is per-device (each home chip
waits for one barrier increment per sender chip, not for a row count), so dropped rows cannot hang it.

## 3. L1 budget per ring core (new CBs)

| CB | shared op | this port |
|---|---|---|
| `c_3` weights (Bfp4_b, 576 B tiles) | 84 tiles = 48,384 B (3 blocks, W0/W1 and W2 alternate) | W0/W1 region: 560 tiles = 322,560 B |
| `c_10` W2 (Bfp4_b) | - | 336 tiles = 193,536 B |
| `c_6` a2a (bf16) | 24 tiles = 49,152 B | same |
| `c_4`, `c_5`, `c_7` | 3 x 16-64 B | same |
| `c_0`/`c_1`/`c_2` | tensor-backed (the shared `[64, H]` bf16 shard, 360,448 B; counts) | same |

CB total on a ring core 565,392 B (was 97,600 B); with the 360 KB shared tensor and the model's ~200 KB of resident L1
that is ~1.13 MB of the 1.5 MB Blackhole L1 (the op asserts the fit from the allocator base). Tilize and combine cores
are unchanged. Nothing is double-buffered across experts: the next expert's W0/W1 lands in the region the previous
expert just released (stream time 322 KB / ~45 GB/s = ~7 us per core, hidden behind the ~8 us gate/up phase).

## 4. What the port covers, and the fused fabric combine

`ttnn.generic_op` cannot build the shared op's FullCcl program: the fused `selective_reduce_combine` appends Fabric Mux
V2 kernels and serializes the mux client runtime args through `FabricMuxV2Config` (C++ only, not bound to Python). So:

* this op = the shared op's ComputeOnly program (tilize + ring + dm1 staging, no combine kernels), with two staging
  layouts selected by `staging_layout`: `"fused"` = the shared op's double-buffered halves (what its compute_only
  returns; used for the bit-identity test) and `"standalone"` = one block per expert at
  `e * (T / hs) * (H / ws) * 2` bytes, the layout the standalone `ttnn.experimental.selective_reduce_combine` op
  (`tests/nightly/tg/ccl/moe/test_selective_combine_6U.py`) consumes;
* on a mesh the production path is this op + that standalone combine op (one extra program per layer; the fused
  overlap of the combine with the next expert's compute is lost, which is why the (2,4) measurement below reports
  both programs). `moe_decode_custom.py` (`TTMoEDecodeCustom`) is the `TTMoEDecode` subclass that swaps the one
  `ttnn.experimental.moe_compute` call for the pair and returns the same 6-tuple; `tt/experts/moe_decode.py` picks it
  under `GEMMA4_MOE_COMPUTE_CUSTOM_OP=1`.

`layer_id` is a runtime arg here (the shared op compiles one program per layer).

Not ported (asserted off): `has_bias`, shared experts, the SingleCluster `local_output` writer, the SingleDevice fused
local combine (the 1x1 bit-identity test reads the staging back instead), Wormhole (one 16 KB read per block needs the
Blackhole NoC burst; the 2 x 8 KB form is kept under `txns_per_read = 2` but untested).

## 5. Validation

Single chip (`TT_VISIBLE_DEVICES=6`, done here): `tests/unit/custom_ops/test_moe_compute.py`

* `bit_identity_fused`: gemma shapes, 2 experts (the fused halves hold two experts), this op (fused layout) vs
  `ttnn.experimental.moe_compute(compute_only=True)`: counts, activation, e_t and every staged row bitwise equal.
* `bit_identity_standalone`: 4 experts, 256 tokens, distinct top-8 over 128 global experts (4 local), this op
  (standalone layout) gathered to `[k, T, H]` on the host vs the shared op's SingleDevice fused local combine output,
  bitwise; plus the torch golden (PCC) and the program-cache hit on fresh buffers.
* `pad_rows`: 128 real + 128 pad rows (ids 0..7, score 0): 5 chunks per expert on device 0's experts; same checks, and
  with `skip_zero_scores` the pad rows are dropped and the real rows stay bitwise equal.
* `perf`: device profiler kernel durations (`TT_METAL_DEVICE_PROFILER=1`, private `TT_METAL_PROFILER_DIR`), shared op
  vs this op, 1-chunk and 5-chunk routing.

Multi-chip, for the lead on the (2,4) block (`TT_VISIBLE_DEVICES=0,1,4,5,8,9,12,13`, p150_x8 descriptor; 16 experts per
chip there, frame 64):

```
# 1. the pair vs the shared FullCcl op, bitwise, on the block (each chip its own experts, dispatch axis 0 of extent 2)
TT_METAL_OPERATION_TIMEOUT_SECONDS=60 pytest models/demos/gemma4/tests/unit/custom_ops/test_moe_compute.py -k mesh_block -p no:cacheprovider
# 2. the decode kernel through the model wrapper, shared vs custom
GEMMA4_MOE_DECODE_KERNEL=1 GEMMA4_MOE_COMPUTE_CUSTOM_OP=1 pytest models/demos/gemma4/tests/unit/test_experts.py -k "decode_kernel" -p no:cacheprovider
# 3. the traced step: ops/step and ms/step, custom vs shared
GEMMA4_MOE_DECODE_KERNEL=1 GEMMA4_MOE_COMPUTE_CUSTOM_OP=1 pytest models/demos/gemma4/tests/unit/test_decode_roofline_block_ring.py -p no:cacheprovider
```

(`-k mesh_block` is written against the (2,4) block in the test file; it opens the mesh with `FABRIC_1D` and
`l1_small_size=16384` as the fused combine requires.)

## 5b. Results on chip 6 (2026-10-23, `_logs/a13_test3..6.log`, `_logs/a13_perf_{one,five}.log`, profiler dirs `_logs/a13_prof_{one,five}`)

Bit-identity (standalone layout, 4 experts, 256 tokens, distinct top-8 over 128 experts; both the program-cache miss
and the hit on fresh buffers): the custom op's staged rows gathered to `[K, T, H]` are bitwise equal to the shared
op's SingleDevice fused local-combine output in all three cases (1 chunk per expert; 128 pad rows = 5 chunks per
expert; pad rows skipped), PCC vs the torch golden 0.9990-0.9991 per expert. The fused layout (2 experts) is bitwise
equal to the shared op's compute_only staging, counts, activation table and e_t lists.

Device kernel duration per call (device profiler, median of 5, after warm-up; the shared op in compute_only mode,
which runs the same tilize + ring + staging work as FullCcl minus the fused combine):

| routing (4 experts, 256 tokens) | shared op | custom (prefetch 8) | custom, prefetch 4 / 12 | custom + skip_zero_scores |
|---|---|---|---|---|
| distinct top-8, 1 chunk per expert | 77.8 us | 74.3 us (-4.5%) | 74.2 / 74.6 us | 74.7 us |
| + 128 pad rows (ids 0..7, score 0): 5 chunks per expert = device 0's load | 321.0 us | 290.5 us (-9.5%) | 290.3 / 290.6 us | **74.7 us (-77%)** |

Per-core, per-RISC kernel spans of the median call (`scratchpad/a13_spans.py`): in the custom op the reader NCRISC
(dm0) finishes at 63 of 74 us (1 chunk) and 227 of 290 us (5 chunks) while the ring cores' compute threads and the
writer BRISC run to the end; in the shared op dm0 runs to the end in lock-step with the compute. The tilize cores
finish ~20 us before the ring cores in every case. So after this port the expert weights are no longer on the
critical path at all (the prefetch depth is irrelevant from 4 to 12) and the call is bounded by the ring cores'
per-chunk work: ~14.5 us per 32-token chunk (5-chunk case) = 792 tile matmuls (LoFi bf16 x bf4), the GELU, the
24 a2a ring steps and the staging writes, plus a ~15 us tilize/metadata lead-in.

What this says about C1: the roofline note's "13.4 MB at 32 GB/s, 26 us floor" counted bytes for one stream, but the
416 us on device 0 is 5 chunks x ~16 us of per-chunk ring work caused by the frame's 128 pad rows, not DRAM
bandwidth; the other 30 chips run the same op in ~78 us. The weight-residency rewrite buys the 10% that the per-chunk
re-stream wait cost; dropping the zero-score pad routes (`GEMMA4_MOE_COMPUTE_SKIP_ZERO_SCORES=1`, or the Python-side
equivalent of spreading the pad ids over all 128 experts / the frame-128 path) takes device 0 from 321 to 75 us, which
is the step-level saving C1 was priced at (12.7 -> ~2.3 ms per 128-user step, every chip at ~78 us), since the lane
step is gated by the slowest chip. The next kernel-side cut is the per-chunk cost itself: 64-token chunks (rt_dim 2,
8 DST tiles, the weights unpacked once per two row tiles), which halves the matmul_block count per token.

## 6. Self-review (ttnn-op-kernel-review / tt-l1-memory-review categories)

| category | finding |
|---|---|
| CB ownership (4.1) | `c_3` (W0/W1 region) and `c_10` (W2 region): one producer (dm0, NCRISC) and one consumer (the compute UNPACK thread) each; `c_0`/`c_1`/`c_6`/`c_4`/`c_5`/`c_7` keep the shared op's ownership. |
| CB capacity vs counts (4.2, 4.5) | `c_10`: capacity 336 = 12 blocks x 28, pushes of 28, one pop of 336. `c_3`: capacity 560 = the largest core's slice; a 2-column core pushes 13 x 28 + a 196-tile unfilled tail (never read) and pops 560, so every expert's pushes land the write pointer exactly on the CB limit before it wraps (the hardware condition the "divides the capacity" rule guarantees). Cumulative `cb_wait_front(c, (b + 1) * 28)` calls only on an expert's first chunk, no pop between them; `llk_wait_tiles` compares received - acked against the count, so counts up to the capacity are valid. |
| control-flow balance (7) | dm0 and compute skip the same experts (zero tokens); dm0 reserves and fills one region per expert and compute pops it once (after the last chunk's gate/up phase for W0/W1, after its down phase for W2), so the reserve/pop counts balance per expert whatever the chunk count. No early return holds pages or DST. |
| init / reconfig (1) | unchanged from the shared kernel: `compute_kernel_hw_startup<Reverse>(c_0, c_3, c_6)`, srcA Bfp4_b for both weight CBs (same format, so no reconfig between the gate/up and down phases), the GELU LUT re-init per activation call. |
| semaphores (6) | unchanged (ids allocated as `add_semaphore_descriptor` does: 0..3 tilize, 0 ring, 4..6 tilize+matmul, 1..2 matmul); the ComputeOnly path never increments the combine semaphores. |
| program cache (10a/10b) | the descriptor (kernels, CT args, CBs, semaphores) is the hash; buffer addresses are runtime args and CB bindings rebuilt per call, re-applied on a hit by `apply_descriptor_runtime_args` (`UpdateDynamicCircularBufferAddress` for the five tensor-backed CBs); `layer_id` is a runtime arg. The unit tests run every case twice, the second on fresh buffers. |
| L1 inventory | two new regions, each with one reason: W0/W1 resident so the chunks after the first re-read L1, W2 in its own CB so the next expert's W0/W1 can stream while the current expert's down phase reads W2. Nothing double-buffered across experts (the stream hides behind the gate/up phase). Ring-core static CB region ends at 677,408 B (base 100 KB + 565 KB); the L1 buffers (the 360 KB shared shard + the model's residents) must stay above it: the op asserts the fit from the allocator base, and a stale 360 KB shard left alive by the caller reproduces the "static circular buffers clash with L1 buffers" error (seen once from a test that forgot to free it). |
| tier crossings | per chip per call: 13.4 MB of bf4 weights cross DRAM once (was once per chunk); the chunks' activations cross the NoC as before (tilize multicast, a2a ring, staging writes). |
