# Quasar SDPA (prefill) Bring-up — WH-green + craq-sim

**Goal:** prefill `ttnn.experimental.quasar.transformer.scaled_dot_product_attention` (a) stays green on WH for ALL its tests, (b) runs on Quasar via craq-sim as far as possible. Shared tree/build/WH card with the forked `sdpa_decode` session (`tt-metal-fb`); commits tagged `Quasar sdpa (prefill):`.

## Status: DONE on WH (13/13 green, multi-chunk validated). craq-sim walls at dispatch (emulator territory).

**Tests (all PASS on WH n150):** ops regular 3/3 (seq128/512/1024), ops chunked 3/3, prototype_ops sdpa 3/3 + chunked 3/3, graph 1/1. (`models/experimental/llama32_1b_quasar/tests/{ops,prototype_ops,graph_ops}/`.) Module `test_attention_1d` 9/10 — the 1 failure is a HF gated-repo `OSError` (env/auth), not the op.

## Fixes applied (all committed)
1. **SFPU swap** (`ARCH_QUASAR`-gated): fused `ckernel_sfpu_sdpa.h` → generic Quasar SFPU. WH keeps fused.
2. **Streaming guard**: `#ifndef ARCH_QUASAR` around the `compute_streaming.hpp` include + branch; factory forces `use_streaming_compute=false` on Quasar (Quasar uses non-streaming `compute_common.hpp::sdpa_inner_loop`; the huge streaming file is NOT ported — deferred).
3. **fp32-off + Os**: `fp32_dest_acc_en=false` (+ `enable_32_bit_dest(compute_hw)=false`, the value JIT reads) and compute `opt_level=Os` on Quasar.
4. **QK^T transpose-free** (`89ac69008ac`): Quasar's matmul can't transpose SrcA → `transpose_block` K-chunk into a dedicated `kt` DFB, then `matmul_blocks(transpose=false)`. Ungated (WH validates). `matmul_blocks` needs `ct=N` (per-column `ct=1` is silently wrong).
5. **8-DFB fit = merge SUM, keep MAX separate** (`cb2268c00c6`) — see below.

## The DFB-merge finding (the big lesson)
`kt` pushed the STANDARD non-streaming path to 9 compute self-loop DFBs > Quasar's cap of 8. Final = merge sum → 8: `qk_im, kt, out_im_A, out_im_B, max_A, max_B, sum_A, exp_max_diff`.

- **DO NOT merge MAX**: `reduce_c<MAX>(cur,prev,eltwise)` needs `prev_dfb==out_dfb` (reads prev@front while reserving cur@back of the *same* DFB) — in-place hazard + offset-read ring-wrap. Multi-chunk-broken (WH PCC 0.955; peer's decode bisect: max-merged=0.748 independently). Committed max-merge `bf7466bffa0` was **REVERTED**.
- **Merge SUM** (discipline matches peer's decode fix): merged `sum_A` keeps prev@ring-front `[0,Sq)`; the producer (`sub_exp_block_bcast_cols_inplace`, out-of-order `pack_tile<true>` = reserve-relative → no offset) appends cur `[Sq,2Sq)`. The correction's `mul_tiles_bcast_cols_inplace(prev)`+`add_block_inplace` pair → one `fma_block_merged_sum` (`cur + prev·emd`, re-bases running sum to front; no `std::swap`, no extra `move_block`). Gated by path not arch: `merged_sum=(sdpa_type==STANDARD)&&!use_attention_sink`; factory `merge_sum=!use_streaming_compute&&!use_attention_sink`. JOINT/RING/streaming/sink keep two sum DFBs; **max stays separate on every path**.
- **Two prefill gotchas vs decode's fma** (decode: column-reduced + bf16 sums → hits neither): (a) prefill sums are FULL tiles at correction (final `matmul_reduce` after the K-loop), emd is a column vector → prev·emd MUST be `mul_tiles_bcast_cols` (plain `mul_tiles` pulls emd's junk cols into the later reduce → catastrophic). (b) **DST overflow**: WH runs `fp32_dest_acc_en=true` → `dst_size=4` (factory:411); a fma holding `num_tiles+1`=5 DST tiles overflows → silent corruption. **Fix = DST-frugal fma**: two single-tile L1-accum passes (`reserved[i]=cur[i]`; `reserved[i]+=prev[i]·bcast(emd[i])`) on a **3-deep** `sum_A` (`prev|cur|running`) + `move_block` running→front. 1 DST tile/op; correct on WH-fp32 (4) and Quasar (fp32-off → 8).
- **LESSON:** both ops' committed merges shipped multi-chunk-broken behind weak tests (WH used the separate path via an ARCH_QUASAR gate; Quasar walled on the sim before compute). **Any DFB-merge must run + validate on WH multi-chunk (seq≥256), never Quasar-gated-only.**

## craq-sim frontier
Regular prefill (`is_chunked=false` → no `read_tile_value`) compiles + program-creates + EXECUTES to SDPA program launch, then walls at `UndefinedBehavior: qsr_cache_validate_l1_range: addr=0x460bc0 size=1` — fires at *dispatch* (right after per-core DFB configs), before any compute, byte-identical across compute-kernel changes. A craq-sim single-element-L1-read limitation (same class as decode's `cur_pos`/GH#50135 mailbox gap, different trigger). Emulator/sim-internal, not an op bug. This is the frontier; runtime numerics need the ZEBU emulator.

## Deferred
- craq-sim regular-prefill PCC (blocked on the dispatch UB above).
- dtype BFLOAT16-on-Quasar device-op restriction (test uses bf16; add for robustness).
- Streaming path Quasar port (only for the model/module path).
- Commit hygiene: `bf7466bffa0` (broken max-merge) + `b9ec0f06405` (its doc) precede `cb2268c00c6`, which reverts/replaces them — drop or squash when reordering.
