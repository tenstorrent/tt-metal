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

With `MAX_CHUNK_M_TILES = 64` (2k tokens per chunk; smaller per_core_M = better L1 utilization):

| Tokens | Chunks | RE kernel total (μs) | μs/2k | Sub-ops |
|--------|--------|----------------------|-------|---------|
|  1024  | 1      |  280                 |  560  | 5       |
|  2048  | 1      |  528                 |  528  | 5       |
|  4096  | 2      | 1060                 |  530  | 10      |
|  8192  | 4      | 2120                 |  530  | 20      |
| 16384  | 8      | 4240                 |  530  | 40      |
| 25600  | 12+1   | 6640                 |  519  | 65      |

vs main on 4k: 1676μs (1.58x faster).
Target: ~500μs/2k. We are within ~6% on 25k, exactly there on 2k.

### Outstanding
- Op count still scales with chunks (5 ops per 2k chunk). User wants "almost a single
  op in tt-perf-report regardless of sequence length". A true single-op custom kernel
  would require composing 3 matmul-equivalent compute kernels + silu + multiply into
  one program, which is a substantial kernel-writing effort. Tracked as future work.

- [2026-05-22] **Milestone 4**: leverage device-side token-count buffer in
  `tt_routed_expert.forward`. One host-side read of expert_token_counts +
  global_expert_idx_table per forward; per local expert, narrow the extracted
  tokens buffer to ceil_tile(count) rows before calling the FFN. For experts
  with count==0, skip extract/ffn/insert entirely.

  - `test_ttnn_moe.py` `linear-8 perf-host-64`: `tt_forward` 40413ms -> 4821ms
    (8.4x). The win comes from per-expert FFN no longer paddedly matmul-ing
    ~204800 rows when only ~500 are occupied.
  - `test_single_routed_expert.py`: count == num_tokens so the narrow is a no-op.
    Adds one ~100us host-device sync per forward; device kernel times unchanged
    (still 528-556us/2k).
