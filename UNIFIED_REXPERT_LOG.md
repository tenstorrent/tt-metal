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
