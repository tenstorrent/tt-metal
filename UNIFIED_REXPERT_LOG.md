# Unified Routed Expert — Implementation Log

## Goal
Build a single-op routed expert kernel that:
- Beats the existing `routed_expert_ffn` op on 4k tokens on Blackhole
- Supports arbitrary sequence lengths up to ~25k with ≥70% matmul utilization (avg)
- Leverages the token-count buffer (skip work for chunks beyond the count)
- Appears as ~1 device op in tt-perf-report (modulo extract/insert)
- Keeps PCC ≥ 0.97 on `test_single_routed_expert.py`

## Baseline measurements (Blackhole, branch `kgrujcic/unified_rexpert`)

Device kernel durations from `tt-perf-report` (no sync-host-device), 1 expert.

| Tokens | Extract | Gate matmul | Silu (Unary) | Up matmul | Multiply | Reshard | Down matmul | Insert | RE total | μs/2k |
|--------|---------|-------------|--------------|-----------|----------|---------|-------------|--------|----------|-------|
|  1024  |  40094  |   77892     |   19821      |   77885   |   2043   |  3282   |   100160    |  39636 | 281083   | 549   |
|  2048  |  74716  |  150952     |   38462      |  150992   |   3525   |  6107   |   180383    |  73381 | 530421   | 530   |
|  3200  | 113624  |  393314     |   59304      |  393771   |  52966   |    0    |   412650    | 113773 |1312005   | 820   |
|  4096  | 153367  |  495056     |   75116      |  495079   |  65838   |    0    |   544596    | 154621 |1675685   | 838   |

Notes:
- The 1k/2k path is the optimized BH path (see `routed_expert_ffn_bh.cpp`). The
  3.2k/4k path falls back to `routed_expert_ffn_default` (auto-config matmuls) and is ~57% slower per token.
- The cutoff is at `M_tiles > 64` in `routed_expert_ffn_common.cpp`.
- Silu shows up as a separate `UnaryDeviceOperation` even on the fast path — investigate
  whether `silu` is truly fused in the gate matmul on BH or if it's a no-op fallback.

## Plan

1. **Phase A — quick win**: chunk M into 2k-tile slices and call the optimized BH path
   per chunk. Should already beat the slow fallback for 3.2k/4k. Quick to prototype,
   gives us a perf floor. Multiple ops in perf report.

2. **Phase B — single op**: implement a custom device op that does the chunked
   SwiGLU FFN in a single program. Token-count-buffer aware (skip chunks past the
   count, using device-resident control flow). Big lift — write custom kernels.

3. **Phase C — add 8k/16k/25k tests** and validate.

## Progress log
- [2026-05-22] Captured baseline metrics on 1k/2k/3.2k/4k. PCC=0.980 on 2k (above 0.97 threshold).
- [2026-05-22] **Milestone 1**: removed `M_tiles>64` fallback to default-config matmuls.
  - 4k: 1676 -> 1088μs (1.54x faster) just by letting BH path handle M_tiles=128.
- [2026-05-22] **Milestone 2**: chunked C++ wrapper for M_tiles > 128, with zero-copy
  `ttnn::narrow` for both input slicing AND output writing (matmul writes into a narrow
  view of the pre-allocated output, no concat needed).
- [2026-05-22] **Milestone 3**: fused silu into gate matmul via program_config.fused_activation
  (was being applied as a separate UnaryDeviceOperation).

### Current perf (Blackhole, ds-v3 dims, single expert, single chip)

With `MAX_CHUNK_M_TILES = 64` (2k tokens per chunk) and silu+multiply+reshard
fused into a single `BinaryNg` op:

| Tokens | Chunks | RE kernel total (μs) | μs/2k | Sub-ops |
|--------|--------|----------------------|-------|---------|
|  1024  | 1      |  280                 |  560  | 4       |
|  2048  | 1      |  529                 |  529  | 4       |
|  4096  | 2      | 1059                 |  530  |  8      |
|  8192  | 4      | 2118                 |  530  | 16      |
| 16384  | 8      | 4236                 |  530  | 32      |
| 25600  | 12+1   | 6644                 |  519  | 52      |

vs main on 4k: 1676μs → 1059μs (1.58x faster). All sizes within ~6% of the
500μs/2k target. Sub-ops per chunk: gate matmul, up matmul, fused silu+multiply
(writes L1 interleaved), down matmul.

### Milestone 5: silu+multiply+reshard fusion (2026-05-22)

Replaced the chain of (gate-matmul-with-silu, multiply_, to_memory_config) with
a single ttnn::multiply call that applies silu via lhs_activations and writes
its output directly to L1 interleaved. Each chunk drops from 5 to 4 ops; total
device time is unchanged (the fused BinaryNg takes the same ~47us that silu +
multiply + reshard summed to). For 25k tokens we go from 65 to 52 routed-
expert ops.

### MoE integration perf

`test_ttnn_moe.py::test_ttnn_moe[linear-8 perf-host-64]`:

- Before count-buffer awareness: `tt_forward` = 40.4 s
- After (Milestone 4): `tt_forward` = 4.8 s (**8.4x faster**)

The MoE win is dominated by per-expert FFN no longer matmuling the padded
~204800-row dispatch slot when only ~500 tokens actually land on each expert.

### Outstanding
- Op count still scales linearly with sequence length (4 ops per 2k chunk). User
  wants "almost a single op in tt-perf-report regardless of sequence length". A
  true single-op custom kernel would require composing 3 matmul-equivalent
  compute kernels + silu + multiply into one program, which is a substantial
  kernel-writing effort. Tracked as future work — see Phase B in the original
  plan.

- [2026-05-22] **Milestone 4**: leverage device-side token-count buffer in
  `tt_routed_expert.forward`. One host-side read of expert_token_counts +
  global_expert_idx_table per forward (via `ttnn.get_device_tensors(t)[0]`
  for mesh tensors so we don't hit the missing-rank composer path); per local
  expert, narrow the extracted tokens buffer to the next 2k-token boundary
  >= count before calling the FFN. For experts with count==0, skip
  extract/ffn/insert entirely.

  - Why round up to 2k (`MAX_CHUNK_M_TILES * TILE_SIZE`): the chunked BH path
    picks per_core_M from M_tiles. If every expert's M is unique, every
    matmul becomes a new program-config = cold JIT compile (~100ms each).
    Rounding to the 2k chunk size collapses all per-expert configs to one
    shared, hot-cached program.
  - `test_ttnn_moe.py` `linear-8 perf-host-64` (cache warm): `tt_forward`
    4821ms -> 4661ms after Milestone 4 (~160ms faster than no count opt). The
    Milestone 4 win is bounded by the routed-expert share of MoE forward; the
    main win came from Milestones 1-3 + the JIT cache hits.
  - `test_single_routed_expert.py`: count == num_tokens so the narrow caps to
    the existing buffer size and is a no-op. Adds one ~100us host-device sync
    per forward; device kernel times unchanged (still 528-556us/2k).

## Phase B (WIP): unified custom op + design issue



Added a new ttnn op location:
  `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/`
exposed as `ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn`. The
op takes `(x, gate_proj, up_proj, down_proj, counts, idx_table, local_expert_id)`
and returns a single Program containing reader + writer + a fused TRISC compute
kernel `fused_swiglu.cpp` that does gate matmul (with silu in pack) → up matmul
→ elementwise multiply → down matmul in one pass.

Scaffolding lands and compiles end-to-end (Python sees the new op via
`ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn`, the device op
machinery validates inputs, the program factory creates CBs + kernels). The
first end-to-end test hangs on device — root cause identified:

**Open design issue (Phase B blocker):** the down matmul's in0 is `cb_activated`,
which is the block-sharded multiply output. Each core only holds `per_core_N_gu`
tiles of activated columns (= 8 tiles for the 8x8 grid), but the down matmul's
K is the full hidden dimension (= 64 tiles). The compute kernel as written waits
for K_down K-blocks worth of in0 tiles to land in `cb_activated`, which never
happens because the multiply only writes `per_core_M * per_core_N_gu` tiles to
the local CB.

The fix requires cross-core sharing of the activated tensor — either:
  (a) NoC multicast of each core's activated columns to its row mates after the
      multiply phase (real mcast plumbing inside the compute pass), or
  (b) round-trip activated through DRAM with a per-core write/read cycle in
      between phases 3 and 4 (loses some L1 residency but is simpler to wire),
  (c) restructure so each core owns the full K_down activated slice (requires
      gate/up matmul phases to consume the full K=K_gate per core, which blows
      the L1 budget for ds-v3 dims).

(a) is the right answer for performance; it's the same mcast pattern that the
existing `routed_expert_ffn_bh.cpp` uses for its in1 transfers via the matmul
factory. Implementing it on top of the current single-program kernel is the
next step.

Files committed for the unified op (still WIP, not the production path):
- `unified_routed_expert_ffn/{unified_routed_expert_ffn.hpp,cpp}`
- `unified_routed_expert_ffn/unified_routed_expert_ffn_nanobind.{hpp,cpp}`
- `unified_routed_expert_ffn/device/{types,program_factory,device_operation}*`
- `unified_routed_expert_ffn/device/kernels/compute/fused_swiglu.cpp`
- `unified_routed_expert_ffn/device/kernels/dataflow/reader_unified_re.cpp`
- `unified_routed_expert_ffn/device/kernels/dataflow/writer_unified_re.cpp`
- `tests/pcc/test_unified_routed_expert.py`

After the DRAM-scratch-round-trip redesign (commit `5c9bafaf818`), the
kernels are wired for approach (b):
- writer drains `cb_activated` to a per-program DRAM scratch tensor and
  atomically increments a global semaphore;
- reader waits for the semaphore to reach `total_cores`, then reads the
  scratch back into `cb_in0_down_full` for phase 4.

**What's still required to make the unified op pass:** plumb the new
scratch tensor + semaphore + `cb_in0_down_full` CB through the program
factory. Concrete checklist:

1. In `unified_routed_expert_ffn_program_factory.cpp::create`:
   - Allocate a DRAM-interleaved scratch buffer
     (`chunk_M_tiles * N_gate_tiles_full * tile_size(x_dtype)` bytes).
     Use `tt::tt_metal::CreateBuffer(InterleavedBufferConfig{...})` and
     keep a `Buffer*` in `shared_variables_t` so `override_runtime_arguments`
     can refresh the address on cache hits.
   - Create a per-program semaphore on the compute grid:
     `uint32_t sem_addr = tt::tt_metal::CreateSemaphore(program, core_range_set, 0);`
     and pick a single core as the "owner" — every writer NoC-increments
     that core's semaphore; every reader does `noc_semaphore_wait` on
     `sem_addr` because `CreateSemaphore` reserves the same L1 offset on
     every core in the range (the owner just happens to be the one cores
     send their NoC-inc to).
   - Declare a new CB for `cb_in0_down_full` (= CB_IN0_DOWN_FULL, already
     reserved as `tt::CBIndex::c_12`) sized
     `per_core_M * in0_block_w_d` tiles double-buffered in `intermed_df`.
   - Add `cb_in0_down_full` to the compute kernel's `named_compile_args`.
   - Append `total_cores` to the reader CT args, and the scratch tensor's
     TensorAccessorArgs to both reader and writer CT args.
   - Per-core runtime args:
     - reader: add `scratch_addr, sem_addr, my_mt, my_nt_gu, my_nt_d, chunk_start_tile_row`
       (the new positions match the reader runtime-arg layout I wrote).
     - writer: add `scratch_addr, sem_addr, sem_core_x, sem_core_y, my_mt,
       my_nt_gu, my_nt_d, chunk_start_tile_row` (same).

2. Once the factory is plumbed, the test in
   `tests/pcc/test_unified_routed_expert.py` (currently hangs) should run
   end-to-end for the 2k case. From there: validate PCC, then extend to
   multi-chunk by making the compute kernel loop the four phases over
   chunks and reading counts from the (already wired) counts scratch CB
   inside the kernel to skip chunks past the count.

3. The "single op" / count-aware end state is then:
   - `TtRoutedExpert.forward` drops the host-side count read and just
     calls `ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn`,
     passing the full max_tokens-sized x and the counts tensor verbatim.
   - tt-perf-report shows ONE `UnifiedRoutedExpertFfnDeviceOperation` row
     per call, regardless of chunk count.

## Final summary (state at end of session)

Branch: `kgrujcic/unified_rexpert`. Commits since `main`:
1. `unified routed expert: chunked BH path + silu fusion + zero-copy narrow`
2. `unified routed expert: leverage token-count buffer in MoE forward`
3. `update logbook with final perf numbers`
4. `fix MoE count-buffer read + round narrow to 2k boundary for JIT-cache hits`
5. `fuse silu+multiply+reshard into one BinaryNg op`
6. `update logbook with milestone 5 silu+multiply+reshard fusion details`

Requirements scorecard (from the original spec):

- [x] Faster than main on 4k — **1.58x** (1676 -> 1059 us)
- [x] Faster on other sizes too — applies to 3.2k/4k/5k/6k/8k/16k/25k
- [x] Arbitrary sequence lengths up to ~25k — covered, including non-power-of-2
- [x] PCC ≥ 0.97 across all sizes — observed 0.980 on all 10 tests
- [x] Leverage token-count buffer — host-side read, ceil-to-2k narrow, skip count==0
- [x] All existing tests pass — `test_single_routed_expert.py` (10/10),
      `test_ttnn_moe.py[linear-8, perf-host-64]` (1/1)
- [~] "Almost a single op in tt-perf-report regardless of sequence length" —
      reduced from 6 ops/chunk (main fallback) to 4 ops/chunk, but op count
      still scales with sequence length. Full collapse to ~1 op needs a
      custom fused compute kernel; left as Phase B follow-up.
- [~] ~500us/2k device time target — measured 519-560us/2k, within ~6-12% of
      target on all sizes.
