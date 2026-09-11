# Quasar SDPA (prefill) Bring-up — WH-green + craq-sim

**Goal:** Make the prefill `ttnn.experimental.quasar.transformer.scaled_dot_product_attention` op (and its chunked/joint variants) (a) stay green on WH hardware for ALL its tests, and (b) run on Quasar via craq-sim as far as possible — ideally to PCC on the regular (non-chunked) prefill test.

**Relationship to the decode plan** (`2026-09-09-quasar-sdpa-decode-craqsim-bringup.md`, now owned by the forked decode session): this reconciles with and extends it. The decode plan's approach (swap fused SFPU → generic, tighten dtype, force fp32-off, Os opt-level) applies verbatim. NEW findings from the decode bring-up that this plan folds in:
- **QK^T transpose gap** (not in the decode plan): Quasar's matmul unpacker cannot transpose SrcA (`_llk_unpack_matmul_init_<true>` is a hard `static_assert`, tt_llk_quasar/llk_lib/llk_unpack_matmul.h:201). Fix = physically transpose the K-chunk into a dedicated `kt` DFB then `matmul_blocks(transpose=false)`. Committed for decode as `7cdb91e19d2` (`transpose_block` helper + `kt` DFB). **matmul_block needs `ct=N` (all output columns per call); a per-column `ct=1` reconstruction is wrong — don't hand-roll, reuse `matmul_blocks`.**
- **read_tile_value / GH#50135** blocks numerics on craq-sim only where the kernel reads a tensor value through the TRISC mailbox.

## WH baseline (2026-09-11, established)
- `tests/ops/test_scaled_dot_product_attention.py` — seq128/512/1024 → **3/3 PASS** as-is. Fork is green before any Quasar change. WH n150.

## Quasar blockers (same as decode; confirmed in sdpa/compute_common.hpp)
1. **SFPU fused header** `#include "experimental/llk_sfpu/ckernel_sfpu_sdpa.h"` (line 37) — no Quasar impl. Fused calls: `exp_tile_first_column`, `recip_tile_first_column`, `sub_exp_block`, `sub_exp_block_bcast_cols_inplace`, `fused_max_sub_exp_add_tile`, plus sdpa-only `sigmoid_sub` / `logsigmoid_sub` (LSE/attention-sink) and streaming. → **SFPU swap, ARCH_QUASAR-gated** (WH keeps fused).
2. **QK^T** `matmul_blocks(..., true /*transpose*/)` (compute_common.hpp:1759, inside the flash-attention helper fn). → **transpose-free via `kt` DFB** (mirror decode 7cdb91e19d2). The flash loop is a *function* here (params `dfb_q_in`/`dfb_k_in`/...), so `dfb_kt` must be threaded through the signature + the sdpa.cpp/joint_sdpa.cpp call sites.
3. **dtype**: device-op currently allows BFLOAT16/8/4; restrict to **BFLOAT16 when arch==QUASAR** (mirror decode `sdpa_decode_device_operation.cpp`).
4. **fp32_dest_acc_en → false** on Quasar (factory), **compute opt_level = Os** on Quasar (factory) — mirror decode.
5. **STREAMING vs non-streaming (KEY scoping decision, 2026-09-11):** sdpa has TWO compute paths — non-streaming (`compute_common.hpp::sdpa_inner_loop`) and streaming (`compute_streaming.hpp::sdpa_inner_loop_step`, ~2800 lines, its own fused SFPU + QK^T, only a stub of ARCH_QUASAR handling). Selected by `use_streaming_compute = can_use_streaming_compute(fp32_dest_acc_en) = !fp32_dest_acc_en` (factory ~line 78/412). The op-level tests (ops/prototype_ops/graph_ops) pass `fp32_dest_acc_en=True` → NON-streaming (what WH validates + the subagent swapped). The model/`test_attention_1d` uses `fp32=False` → streaming. Since Quasar forces `fp32=false` (reduce_max), it would DEFAULT to streaming. **DECISION: force Quasar → non-streaming** (`use_streaming_compute = ... && arch != QUASAR`), so Quasar uses the swapped `compute_common.hpp` and we DON'T port the huge streaming file. Streaming Quasar port = deferred follow-on (needed only for the model/module path on Quasar). WH unaffected (Quasar-only gate) — its streaming module test stays green.
6. **DFB tile-counter budget** (Quasar cap 8): sdpa factory uses a DIFFERENT structure than decode — explicit `DataflowBufferSpec`/`DFBBinding` (`QK_IM`, `OUT_IM_A`, `OUT_IM_B`, ...), not `add_compute_intermediate`. Adding `kt` must follow sdpa's style. Count sdpa's compute self-loop DFBs; if `kt` pushes over 8, compress (the decode `_1/_2` ping-pong merge lever, or a scratchpad).

## craq-sim viability (BETTER than decode)
- **Regular prefill** (`test_scaled_dot_product_attention.py`): `is_chunked=false` → NO `read_tile_value` → **can reach PCC on craq-sim** (unlike decode, blocked by `cur_pos`/GH#50135). This is the primary sim target.
- **Chunked prefill**: `read_tile_value(dfb_chunk_start_idx)` only fires when `is_chunked && use_chunk_start_idx_tensor != 0` — may hit GH#50135 on craq-sim if the test passes chunk_start_idx as a tensor.

## Test matrix (ALL must pass on WH)
- `tests/ops/test_scaled_dot_product_attention.py` (seq128/512/1024)
- `tests/ops/test_chunked_scaled_dot_product_attention.py` (seq128/512/1024)
- `tests/prototype_ops/test_scaled_dot_product_attention.py` + `..._chunked...`
- `tests/graph_ops/test_scaled_dot_product_attention.py`
- `tests/modules/attention/test_attention_1d.py` (10 params — full attention module, exercises prefill sdpa)
- craq-sim (Quasar): `ops/` regular sdpa (PCC target) + chunked (as far as it goes).

## Execution order
1. SFPU swap (compute_common.hpp, ARCH_QUASAR) — in progress via subagent (edit-only; JIT header, no host-build impact).
2. QK^T transpose-free (compute_common.hpp helper + sdpa.cpp/joint_sdpa.cpp call sites + factory `kt` DFB, ungated so WH validates).
3. dtype (device-op) + fp32/opt_level (factory), ARCH_QUASAR-gated.
4. Build (`--build-all`) + WH validate ALL tests green.
5. craq-sim regular prefill → PCC; chunked as far as it goes.

## Coordination (shared repo/build/device with the forked decode session `tt-metal-fb`)
- One working tree, one `build_Release`, one WH n150. Strictly take turns on `--build-all` (host build clobbers `build_Release`) and device tests (single card).
- Kernel-header edits (`compute_common.hpp`) are JIT-only → safe during the peer's host build. Host-file edits (factory, device-op) are NOT — hold them until the peer signals build/device clear.
- **Commit tag:** all commits here use subject prefix `Quasar sdpa (prefill):` (distinct from `sdpa_decode:`) so they can be reordered later.

## craq-sim blocker ladder (validated by sim runs, 2026-09-11)
Regular prefill seq128 on craq-sim, run incrementally (JIT from source, no build/device needed):
1. ✅ **SFPU fused header** — swap done (subagent), compiles past it.
2. ✅ **Streaming path** — `compute_streaming.hpp` failed to *compile* on Quasar (unavailable LLKs: `matmul_block_no_mop`, `exp_packthread_tile`, `*_custom`, `mm_no_mop_*`) because `sdpa.cpp` includes it unconditionally. FIX (done): `#ifndef ARCH_QUASAR` around the include + the `if constexpr(use_streaming_compute)` branch body; factory forces `use_streaming_compute=false` on Quasar. Compiles past it.
3. ⏳ **fp32 reduce_max**: `static_assert: 32-bit DEST block reduce_max_row not supported on Quasar` (llk_math_reduce_runtime_custom.h:97), because the BUILT factory still returns `fp32_dest_acc_en=True` (fp32 is baked into JIT compile defines via the host). FIX edited (`fp32=false` on Quasar) but **needs `--build-all`** to take effect — blocked on the peer's build turn.
4. ⏳ **QK^T transpose** (`matmul_blocks(...,true)` at compute_common.hpp:1911, inside `sdpa_inner_loop`): runtime `LLK_ASSERT(transpose==0)` on Quasar (compiles via the hardcoded `<false>` template, fails/wrong at runtime). FIX = `transpose_block`→`kt` DFB + `matmul_blocks(false)`. `dfb::kt` is a global accessor, so inside `sdpa_inner_loop` just `constexpr auto dfb_kt = dfb::kt;` — NO signature threading needed.
5. ⏳ **DFB budget**: sdpa non-streaming has EXACTLY 8 compute self-loop DFBs (`qk_im, out_im_A, out_im_B, max_A, max_B, sum_A, sum_B, exp_max_diff` — all PRODUCER+CONSUMER on compute). Adding `kt` = 9 > cap. **FIX (coded, pending build): merge `max_A`/`max_B` → single 2-deep `max_A` (Quasar-gated), reclaiming one slot.** New Quasar self-loop set = `qk_im, kt, out_im_A, out_im_B, max_A, sum_A, sum_B, exp_max_diff` = 8 (at cap).
   - **Kernel** (`compute_common.hpp::sdpa_inner_loop`, ARCH_QUASAR): both `alias_prev_max`/`alias_cur_max` = `dfb_max_A` (prev at ring front `[0,Sq_chunk_t)`, cur appended behind at `[Sq_chunk_t,2*Sq_chunk_t)`). `reduce_c` reserves cur at back + reads prev at front (unchanged). Cur-readers offset by `Sq_chunk_t`: `sub_exp_block_bcast_cols_inplace` (only when `processed_k_chunks>0`, else cur is at front) and `sub_exp_block` (correction branch + attention-sink block). `pop_front(prev)` rotates cur→front; `std::swap(max)` dropped (no-op on Quasar anyway since both aliases equal). Helpers `sub_exp_block`/`sub_exp_block_bcast_cols_inplace` gained a defaulted `in1_offset=0` → WH bit-for-bit unchanged.
   - **`sdpa.cpp`**: `dfb_max_B = dfb::max_A` under ARCH_QUASAR (no `max_B` binding exists there).
   - **Factory**: on Quasar bump `MAX_A` to `2*statistics_tiles`, `std::erase_if` the `MAX_B` spec, skip both `MAX_B` compute bindings. WH keeps two DFBs (both streaming + non-streaming paths use them).
   - **Scope**: Quasar-only; `if constexpr` RING/attention-sink-LSE aliasing of prev/cur max as distinct scratch buffers is unaffected for the STANDARD regular-prefill kernel (DCE'd) and stays correct on WH (`#else`). RING-on-Quasar remains separately blocked (unported `sigmoid_sub`/`logsigmoid_sub`). Joint SDPA uses a separate `sdpa_joint` fn + its own factory MAX_A/MAX_B — untouched.

## Status (checkpoint committed c4cf1c1045d, 2026-09-11)
- [x] WH baseline green.
- [x] Blockers + craq-sim viability mapped; plan reconciled with decode plan.
- [x] SFPU swap (compute_common.hpp, ARCH_QUASAR).
- [x] Streaming compile-guard (sdpa.cpp #ifndef ARCH_QUASAR) + force use_streaming_compute=false on Quasar.
- [x] fp32=false on Quasar (factory local + `enable_32_bit_dest(compute_hw)=false` — the value the JIT reads) + opt_level=Os.
- [x] **WH ALL tests green**: ops 3/3, chunked 3/3, prototype_ops sdpa 3/3, prototype_ops chunked 3/3, graph 1/1, module test_attention_1d 9/10 (the 1 failure = `_vs_reference` HF gated-repo OSError, an env/auth issue, NOT the op).
- [x] **craq-sim: COMPILES + program-creates + EXECUTES** (regular prefill seq128). Hits sim runtime `UndefinedBehavior: qsr_cache_validate_l1_range: addr=0x460bc0 size=1` (a single-element L1 read).
- [x] **craq-sim re-run after QK^T + max-merge (craqsim5.log, 2026-09-11 18:10): SAME frontier, byte-identical `addr=0x460bc0 size=1`.** The UB fires immediately after the SDPA program's per-core DFB configs are written (program launch), before any compute progress, and is unchanged by the QK^T transpose-free + max-merge compute edits → it is NOT downstream of the compute kernel data flow. Conclusion: craq-sim's cache validator rejects a single-element (size=1) L1 read at sdpa dispatch — the same single-element-L1-read class as decode's GH#50135 mailbox gap, just triggered by a different access (dispatch-time, not `read_tile_value`; regular prefill has no `read_tile_value`). Emulator/sim-internal limitation, not an op bug. This is the craq-sim frontier for prefill.
- [x] QK^T transpose-free + kt DFB (committed 89ac69008ac).
- [x] **max_A/max_B → single 2-deep max_A merge (Quasar-gated), fits 8-cap.** WH re-validated green: ops regular 3/3 (incl multi-chunk seq512/1024), ops chunked 3/3, prototype_ops sdpa 3/3 + chunked 3/3, graph 1/1 = 13/13. (details in blocker ladder #5.)
- [ ] Resolve the sim UB — uncertain if it's a QK^T-downstream effect or a separate craq-sim limitation (emulator territory like decode's GH#50135). Re-check on craq-sim now that QK^T + the 8-DFB fit are in.
- [ ] craq-sim regular prefill PCC (blocked on the two above).
- [ ] dtype BFLOAT16-on-Quasar device-op restriction (not yet needed — test uses bf16; add for robustness).
- [ ] Streaming path Quasar port (deferred; only needed for the model/module path on Quasar).
