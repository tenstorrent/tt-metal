# Debugging Analysis — Flash Attention (scaled_dot_product_attention) From-Scratch Eval Run

- **Run**: `flash_attention_run1` (clean clone, op dir written from scratch — no drafter)
- **Clone**: `/localdev/dnijemcevic/2026_07_06/0904_nuke-sdpa-0623/clones/flash_attention_run1/tt-metal`
- **Source data**:
  - Implementer breadcrumbs: `ttnn/ttnn/operations/scaled_dot_product_attention/agent_logs/ttnn-implementer_breadcrumbs.jsonl` (107 entries)
  - Expert-debugger breadcrumbs: `…/agent_logs/ttnn-expert-debugger_breadcrumbs.jsonl` (6 entries)
  - Refinement log: `results/flash_attention/run_1/refinements.log`
- **Outcome**: 7 refinements complete (R1, R1b, R2, R3, R3b, R4, R5, R6). Golden 1790/2269 pass, 0 hangs, 10 pre-existing precision near-misses. Acceptance 29/29. Refinement tests 254/254.

---

## Section 1: Debugging episodes (enumerated, grouped by theme)

Timestamps are from breadcrumbs (UTC). Commit SHAs are abbreviated.

### Theme A — CB synchronization: double-pop / pop-when-helper-already-pops

This is the single most consequential class of bug in the run. It appeared **three times**, each time the same shape: an eltwise/reduce/matmul helper internally pops its input CB tiles, and the kernel then issues a *redundant manual* `cb_pop_front` on the same CB, corrupting the read pointer and deadlocking the next iteration.

**Episode A1 — Phase 0 init: missing `cb_push_back` + `tile_regs` handshake (IMPLEMENTER)**
- Symptom: Hang on `single_tile_32x32` (dispatch timeout). Commit `c7e5d1b10e` ("compiles, hangs on CB sync").
- Hypothesis (crumb 6, `09:55:17`): reader pushes partial blocks (1 tile for 32×32) but compute expects full `B_q=4`.
- Fix applied (crumb 10, `10:16:30`; commit `2cb609b6af`): four fixes from static analyzer — F1 add `cb_push_back` after `pack_tile` in Phase 0 init (tiles packed but never made visible); F2 add `tile_regs_commit/wait` (incomplete MATH→PACK DST handshake); F3 remove `cb_pop_front(cb_m_new)` before copy (destroyed data); F4 `OperandKind::Block` instead of `Scalar` for 1D `tiles(B_q)` chains.
- Result (crumb 11, `10:17:23`): hang fixed; now PCC=0.124 (numerical). Static analyzer caught all four.
- Note: F3 is a "pop that destroys data the helper expects to retain" — a variant of the double-pop theme, just on the *retain* side rather than the *over-pop* side.

**Episode A2 — Multi-KV-block hang: double-pop of `cb_k` / `cb_v` (EXPERT-DEBUGGER)**
- Symptom: single KV block passes; multi KV block hangs (crumb 20, `11:05:19`). NCRISC at `wait_for_brisc_notification`, TRISC0 at `mm_block_init_short`, TRISC2 at `reserve_back(cb_scores)`.
- Debugger root cause #3 (debugger crumb 4, `12:10:37`; commit `7fcfa2991a`): `matmul_block` with `WaitAndPopPerKBlock` *already pops* K/V tiles, but the KV-block-loop cleanup also did `cb_pop_front(cb_k/v, B_kv*D_t)` → double-pop → CB underflow → `tiles_received` wraps unsigned → subsequent `wait_front` succeeds on an empty CB → matmul reads garbage → pipeline deadlocks.
- Result: fixed by removing the redundant manual pops.

**Episode A3 — Full-suite hang: double-pop of `cb_attn_mask` (IMPLEMENTER)**
- Symptom (crumb 43/44, `18:19:31`–`19:13:56`): golden suite times out; hang in PV matmul on shapes with `D_t > 4` (see Episode E2 below) masked this; once that was fixed, a full-suite hang on `cb_attn_mask` surfaced.
- Hypothesis (crumb 46, `21:05:15`; commit `2176b043ec`): `BinaryFpu<cb_scores, cb_attn_mask, Add>` with default `InputLifecycle::Streaming` already pops all `B_q*B_kv` mask tiles internally per-tile. Line 324 then did `cb_pop_front(cb_attn_mask, B_q*B_kv)` again — double-pop corrupts the CB read pointer, reader deadlocks on `cb_reserve_back` for the next KV block.
- Result (crumb 47/48, `21:14:43`/`21:24:57`): 138/140 golden pass, 0 hangs. **Bonus**: this *also* fixed the mask precision issue (PCC 0.9657 → 0.9999) — the "systematic ~3.4% correlation loss" (Episode C2) was *not* numerical at all; it was corrupted mask data from the double-pop. The implementer had spent ~3 hours chasing a numerical-precision hypothesis (Approx::Fast→Exact, finite -inf init, FPU-mode reordering) for what was actually a CB-sync bug.

### Theme B — CB write-pointer alignment / mixed push-count patterns

**Episode B1 — `cb_scores` mixed 1-tile / 2-tile pushes → LLK assert (IMPLEMENTER)**
- Symptom (crumb 29/30, `17:20:31`/`17:27:55`): hang *inside Phase 12 PV matmul on the FIRST KV block* (not a block-boundary issue). TRISC0 at Phase 13 add, TRISC1 at matmul_init, TRISC2 at push_back. LLK assert: `remaining=128 < num_words=256` on `cb_scores` push_back.
- Hypothesis trail (crumbs 30→31→32→34→35, `17:27`–`17:55`): five hypotheses in sequence — (a) TileRowMajor reserve_back blocked because cb_scores full from reduce `WaitUpfrontNoPop`; (b) `DataFormatReconfig::NONE` reconfig issue; (c) cb_scores drain issue; (d) `fifo_wr_ptr` alignment after 1-tile push/pop cycles; (e) pad CB to `2 * max_push_size`. The last hypothesis (e) noted the CB *was* already sized to `2 * max_push_size` but still failed.
- Fix (crumb 36, `17:56:48`; commit `12a79f1800`): **separate `cb_pv_out` CB for PV matmul output**. `cb_scores` now only holds QK^T scores (uniform 1-tile pushes). The PV matmul outputs to `cb_pv_out`, consumed by Phase 13. Root cause confirmed: mixed 1-tile (QK^T scores) and multi-tile (PV output) push patterns on one CB left `fifo_wr_ptr` at a position where only 1 contiguous page was free before `fifo_limit`, but the PV matmul needed 2. The `llk_push_tiles` assert fires *before* the push (preventing mid-push wrapping) rather than the push being split across the wrap boundary.
- Result: hang fixed; all 14 phases complete on multiple KV blocks. This was the critical unblocker for the whole op.

### Theme C — Numerical precision / FPU-mode confusion

**Episode C1 — Normalize phase: `DivBinary` SFPU doesn't broadcast across columns (IMPLEMENTER)**
- Symptom (crumb 13, `10:19:48`): all-ones input → output `[1, inf, inf, inf]`. First element correct, rest inf.
- DEVICE_PRINT (crumb 15, `10:24:55`): `l_i` has value 31.375 only in col 0 (REDUCE_ROW output). `DivBinary` (SFPU) doesn't broadcast across columns → div-by-zero in other columns.
- Fix (commit `5276a022e4`): `recip(l_i)` via SFPU, then `BinaryFpu<Mul, Col>` which broadcasts the B operand across columns at FPU level.
- Result: single_tile_32x32 passes. This is a correct and well-targeted fix.

**Episode C2 — Mask PCC ~0.96: a 3-hour numerical false scent (IMPLEMENTER)**
- Symptom (crumb 37, `17:58:17`): 16/29 acceptance pass; 13 fail (all PCC). Mask tests ~0.96, long-context no-mask accumulating error (S=1024→0.988, 2048→0.944, 4096→0.839).
- Hypothesis C2a (crumb 38, `18:00:25`; commit `369bbec750`): `Approx::Fast` exp approximation. Changed all exp to `Approx::Exact`. → No-mask long-context *fixed* (20/29 pass), but 9 mask tests still fail at PCC ~0.96.
- Hypothesis C2b (crumb 40, `18:03:23`): `BinaryFpu<Add>` producing `score*scale` instead of `score+mask` — FPU still in MUL mode from QK^T matmul; `add_tiles_init` doesn't restore FPU add mode. DEVICE_PRINT confirmed: scores before mask=4.21875, after mask=0.52734 = 4.21875×0.125.
- Hypothesis C2c (crumb 41, `18:04:22`): reorder mask before scale; tried `BinaryFpu<Mul>` with `cb_scale` tile. Same PCC — concluded issue is in `add_tiles_init` not restoring FPU mode.
- Hypothesis C2d (crumb 42, `18:06:13`): SFPU `exp(-inf)` producing NaN instead of 0. Changed `m_i` init from `-inf` (0xFF800000) to `-1e38f` (0xFE967699).
- **Actual root cause** (crumb 46, `21:05:15`; Episode A3): the double-pop of `cb_attn_mask`. The mask data was *corrupted*, not numerically imprecise. PCC 0.9657 → 0.9999 after the CB-sync fix. C2a–C2d were all treating a structural bug as a precision problem. `Approx::Exact` and finite `-1e38f` init were kept (they're defensible), but neither addressed the actual failure.

### Theme D — DEST overflow / subblocking for large head_dim

**Episode D1 — PV matmul DEST overflow at `D_t > 4` (IMPLEMENTER)**
- Symptom (crumb 44, `19:13:56`): hang in PV matmul on shapes with `D_t > 4` (D=256, 512, 1024). `MatmulBlockShape::of(1,1,B_q,D_t,B_kv,1)` creates a subblock of `B_q*D_t` tiles; with `fp32_dest_acc_en=True`, DEST limit is 4 tiles.
- Fix (crumb 44; commit `bcb07a2de5`): compute `PV_SUBBLOCK_W = min(D_t, 4/B_q)` and `PV_NUM_SUBBLOCKS_N` to split the N dimension into subblocks that fit DEST. No-op for `D_t <= 4`.
- Result: hang resolved. This is the same class as the debugger's bug #1 (see Episode E1) but at a different call site — the debugger had already corrected *one* PV matmul shape; this was the DEST-tile-count dimension of the same matmul.

### Theme E — Expert-debugger findings the implementer missed

The expert-debugger was invoked once (crumb 21, `12:56:47`), for the multi-KV-block hang after the implementer's 5 failed fix attempts (commits `a8564383d8`, `9fbf3de456`, `47bfc293ed`). It found **4 root causes** in one pass (debugger crumb 5, `12:56:06`; commit `7fcfa2991a`):

**Episode E1 — Phase 12 PV matmul `MatmulBlockShape` doubled the N dimension (DEBUGGER bug #1)**
- `MatmulBlockShape::of(B_q,D_t,B_q,D_t,B_kv,1)` set `in1_num_subblocks=D_t` instead of 1, producing `B_q*D_t*D_t` tiles instead of `B_q*D_t`. With `B_q=1,D_t=2`: 4 tiles instead of 2. Single-tile test passed because `D_t=1` makes the bug inert.
- Fix: `of(1,1,B_q,D_t,B_kv,1)`.
- This is a **MatmulBlockShape::of() argument-order footgun**: the 6 positional args are `(in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, in0_block_k, num_k_blocks)`. The implementer passed `D_t` as `in1_num_subblocks` when it should have been `1`.

**Episode E2 — `cb_scores` too small for PV output (DEBUGGER bug #2)**
- Sized for `B_q*B_kv` (QK^T output = 1 page) but Phase 12 PV matmul outputs `B_q*D_t` (2 pages). Single-tile passes because `B_q*B_kv == B_q*D_t == 1`.
- Fix: `max(B_q*B_kv, B_q*D_t)`. (Later superseded by the separate `cb_pv_out` CB — Episode B1.)

**Episode E3 — double-pop `cb_k`/`cb_v` (DEBUGGER bug #3)** — same as Episode A2.

**Episode E4 — `cb_scaler` too small → reader deadlock (DEBUGGER bug #4)**
- `cb_scaler` had only 2 pages; reader runs ahead of compute, pushes 2 scaler tiles per KV block, fills CB, blocks on `reserve_back` for the next KV block's scalers.
- Fix (initial): size to `2*num_kv_blocks` pages. (Later refined in R2 — Episode G2 — to always use Float16_b.)

**Episode E5 — Q block boundary hang: UNRESOLVED by debugger (DEBUGGER, MEDIUM confidence)**
- After bugs 1–4, Q block 0 with all 4 KV blocks completes; Q block 1 hangs at Phase 3 reduce. Debugger tried `compute_kernel_hw_startup` (unsafe mid-kernel per docs) and `eltwise_chain FillBitcast` — neither fixed it. Diagnosed as `WaitUpfrontNoPop` reduce on `cb_scores` leaving CB in a state preventing subsequent streaming eltwise_chain access. **No fix applied by debugger.**
- The implementer eventually resolved this via the `cb_pv_out` split (Episode B1, commit `12a79f1800`), which removed the mixed-push-pattern problem on `cb_scores` entirely.

### Theme F — Multi-block / multi-work-unit iteration

**Episode F1 — Multi-work-unit: init placement deadlock (IMPLEMENTER)**
- Symptom (crumb 70, `00:14:02`): MQA `H_q=71` needs >1 work unit per core (56 worker cores on 8×7 grid). Hang: TRISC2 at `cb_reserve_back` in `init_cb_constant_f` (Phase 0 re-init), BRISC at `cb_wait_front(cb_output)`.
- Hypothesis (crumb 72, `00:17:33`; commit `b2ee257b93`): Q-block-loop re-init of `cb_m/cb_l/cb_o` at the *end* of each Q block leaves `cb_o` full of `B_q*D_t` tiles; the next work unit's `init_cb_constant_f(cb_o,…)` calls `cb_reserve_back` and deadlocks.
- Fix: move init to the *start* of each Q block; remove end-of-loop re-init and separate Phase 0 init.
- Result (crumb 73, `00:18:42`): MQA H_q=71 PCC=0.999968.

**Episode F2 — `long_context_1024` PCC=0.0009 → stale JIT cache (IMPLEMENTER)**
- Symptom (crumb 77, `00:42:26`): S=1024 MHA catastrophic PCC=0.0009 (was passing before Refinement 3). S=512 and S=896 work.
- Hypothesis trail (crumbs 78→79→80→81, `00:44`–`00:51`): (a) scaler/CB sync or init placement; (b) online-softmax accumulator overflow/underflow over 32 KV blocks; (c) init_cb_constant_f disrupting matmul state; (d) **race condition** — "test passes with DEVICE_PRINT, fails without" → attributed to 8×8→8×7 grid change or init-inside-loop TRISC race.
- Fix attempt (crumb 82, `01:19:09`): move init back before the loop. PCC 0.0009 → -0.04 (still failing).
- **Actual root cause** (crumb 84, `01:31:26`; commit `4b40eaafc1`): **stale JIT cache binary**. The Refinement 3 kernel code was correct; the JIT cache served an old compiled binary that produced NaN/garbage. The DEVICE_PRINT "fix" was actually forcing a *recompile* (different binary hash), not adding timing delays — the race-condition hypothesis was a red herring. Clearing `~/.cache/ttnn/ttnn/generated/kernels/` resolved it.
- Result (crumb 85/86, `01:55:30`/`01:56:54`): 1046/2269 golden pass, 0 hangs, long_context_1024 passes consistently.

**Episode F3 — D_CHUNK K-blocking: CB deadlock from push ordering (IMPLEMENTER)**
- Symptom (crumb 100, `04:17:05`): hang on D=1024 bf16 after D_CHUNK changes. Timed out after 5min.
- Hypothesis (crumb 101, `04:22:46`): reader pushes ALL Q K-blocks before ANY K K-blocks; `cb_q` fills up (`2*D_CHUNK=16` pages), reader blocks on `reserve_back`, compute blocks on `wait_front(cb_k)`.
- Hypothesis (crumb 102, `04:30:09`): compute waits for `cb_scaler` at start of KV block, but reader pushes scalers *after* Q/K/V; reader blocks on `cb_reserve_back(cb_q)` because `cb_q` full of K-blocks.
- Fix (crumb 103, `04:31:18`; commit `f006770fc0`): interleave Q and K pushes per K-block; move `cb_wait_front(cb_scaler, 2)` to *after* QK^T matmul (scalers only needed for reduce ops).
- Result: fp32+D=1024 passes. 1790/2269 golden (+6).

### Theme G — Numerical configurability (Refinement 2)

**Episode G1 — pack_reconfig missing in `init_cb_constant_f` (IMPLEMENTER)**
- Symptom (crumb 55, `22:13:28`): PCC=0.916 (was 0.995+) on BFLOAT16+True 128×64. Regression from changing intermediate CB format to Float32.
- Hypothesis (crumb 56, `22:13:34`): intermediate CB format change; matmul helper reconfigs data format and the mismatch causes format corruption. Considered reverting to input dtype for all CBs.
- Fix (crumb 61, `22:40:01`; commit `7426bac050`): add `pack_reconfig_data_format(cb_id)` to `init_cb_constant_f` before `pack_tile`. Root cause: packer was left in the previous operation's format, causing silent corruption when writing to CBs with a different format.
- This is a **packer-format-state leak** — a structural correctness rule the implementer had to rediscover.

**Episode G2 — bfloat8_b intermediate CBs lose precision (IMPLEMENTER)**
- Symptom (crumb 58, `22:23:34`): BFLOAT8+True PCC=1.0 but RMS=1.0006 — systematic scale issue. bfloat8_b intermediate CBs can't represent running max/sum accurately (shared exponent).
- Hypothesis→fix (crumb 59/62, `22:23`–`22:43`; commit `7426bac050`): intermediate CBs (m, l, o, scores, pv, psum) that carry running state across KV blocks must be Float32 (or Float16_b) even when input is bfloat8_b. When `fp32_dest_acc_en=False`, use Float16_b for intermediates (best format for 16-bit dest, no shared-exponent loss).
- Also discovered (crumb 57, `22:20:42`): `prepare_reduce_scaler` only supports Float16_b and Float32 — scaler CB must use one of those, not the input dtype.

### Theme H — Non-aligned shapes (Refinement 5)

**Episode H1 — truncating vs ceildiv tile counts (IMPLEMENTER)**
- Hypothesis (crumb 94, `03:01:12`; commit `ac2a3ed056`): program descriptor used `S//32` (truncating) for tile counts, missing the partially-filled last tile. Padded K/V rows are zeros → `exp(0)=1` → non-zero softmax weight → output corruption.
- Fix: ceildiv for `S_q_t/S_kv_t/D_t`; generate padding mask in reader for non-aligned `S_kv` when `mask_mode=none`; overlay padding on custom/causal mask tiles; set padded KV columns to `-1e9`.
- Insight: padded K positions are *columns* in the scores tile (Q @ K^T), not rows.

**Episode H2 — bf8b + custom mask + non-aligned: block-float can't be written directly (IMPLEMENTER)**
- Fix (crumb e94212b639, `03:21:35`): typecast mask tensor to `intermediate_format` when custom mask + non-aligned S_kv. Reader overlays `-1e9` by writing raw bits, which only works for non-block-float formats (fp32/bf16). bf8b's block-float layout can't be written to directly. Force mask CB to `intermediate_format` when custom mask + kv_padding.

---

## Section 2: Patterns

### P1 — Double-pop / redundant-pop is the dominant recurring bug
It appeared **3 times** (A2 `cb_k/cb_v`, A3 `cb_attn_mask`, and structurally A1-F3 `cb_m_new`). In every case a helper (matmul_block `WaitAndPopPerKBlock`, or eltwise_chain `InputLifecycle::Streaming`) *already pops* its input CB internally, and the kernel added a manual `cb_pop_front` on top. The implementer hit this on `cb_attn_mask` (A3) *after* the debugger had already found it on `cb_k/cb_v` (A2) — the lesson did not generalize within the run. A3 alone cost ~3 hours because its symptom (PCC 0.96) was misread as numerical (see P3).

### P2 — MatmulBlockShape::of() argument-order footgun
The debugger's bug #1 (E1) was passing `D_t` as `in1_num_subblocks` instead of `1`. The 6 positional args `(in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, in0_block_k, num_k_blocks)` are easy to get wrong because `B_q, D_t, B_q, D_t, B_kv, 1` *looks* symmetrical. The correct call `of(1,1,B_q,D_t,B_kv,1)` sets subblock counts to 1 and uses `out_subblock_h/w` for the actual dimensions. This is a genuine API usability issue.

### P3 — Numerical-precision false scents masking structural bugs
Two major time sinks were precision hypotheses for what were actually CB-sync bugs:
- **C2** (mask PCC 0.96, ~3h): chased `Approx::Fast→Exact`, FPU-mode, `-inf` handling. Actual cause: double-pop of `cb_attn_mask` (A3) corrupting mask data.
- **F2** (long_context_1024 PCC 0.0009, ~1h): chased accumulator overflow, init placement, race condition, grid change. Actual cause: stale JIT cache binary.

The implementer's instinct when seeing a bad PCC is to reach for numerical hypotheses (exp approximation, dest format, init values) *before* verifying CB integrity. The breadcrumbs show DEVICE_PRINT being used to check *values* (m_i, l_i, scores) rather than *CB tile counts / pop counts*.

### P4 — Single-tile tests hide multi-tile bugs
A striking recurring pattern: bugs were inert at `D_t=1` or `B_q*B_kv=1` and only surfaced on multi-tile shapes. E1 (PV matmul shape), E2 (cb_scores sizing), and the `single_tile_32x32` → `multi_tile_128x64` progression all show this. The implementer's workflow (test single-tile first, then multi-tile) is correct, but the *lesson* — "if it passes at 1 tile, suspect that the bug is masked by the degenerate case" — was not internalized; each multi-tile hang required fresh diagnosis.

### P5 — The expert-debugger was dramatically more efficient than the implementer on the hard hang
The implementer made 5 failed fix attempts on the multi-KV-block hang (commits `a8564383d8`, `9fbf3de456`, `47bfc293ed`, plus 2 uncommitted) over ~2 hours before escalating. The debugger found 4 root causes in one pass (~2h including investigation). The debugger's `observe→hypothesize→experiment→diagnose` discipline and systematic CB producer/consumer tracing caught bugs the implementer's scattershot fixes missed. However, the debugger's one *unresolved* issue (E5, Q-block-boundary hang) was ultimately fixed by the implementer's `cb_pv_out` split (B1) — the debugger correctly diagnosed the symptom class but not the specific fix.

### P6 — Repeated hypothesis cycles before finding the fix
- B1: 5 hypotheses (crumbs 30→31→32→34→35) before the `cb_pv_out` split worked.
- C2: 4 hypotheses (C2a→d) before the real cause (A3) was found.
- F2: 4 hypotheses (crumbs 78→79→80→81) before the JIT-cache root cause.
- F3: 2 hypotheses (crumbs 101→102) for the push-ordering deadlock.

In each case the *first* hypothesis was plausible but wrong, and the agent cycled through related-but-incorrect ideas before either finding the root cause or stumbling onto the right fix.

### P7 — False-fix cycles (fix that appeared to work but broke something else)
- C2a (`Approx::Exact`) appeared to improve things (20/29 from 16/29) but didn't address the real issue; it was kept because it's defensible, but it masked A3 for hours.
- F2 init-before-loop (crumb 82): PCC 0.0009 → -0.04, "helped but still failing" — a partial improvement that was also a red herring.
- G1 intermediate-CB-to-Float32 caused a PCC=0.916 regression (crumb 55) that needed a separate fix (pack_reconfig).

### P8 — Bugs the expert-debugger caught that the implementer missed
E1 (MatmulBlockShape doubled N), E2 (cb_scores too small), E3 (double-pop K/V), E4 (cb_scaler too small). All four were invisible to single-tile testing. The implementer's 5 attempts were all on the *symptom* (DataFormatReconfig, init placement, wait policies) rather than the *CB sizing and matmul shape* — the debugger went straight to the CB producer/consumer chain and the matmul shape spec.

---

## Section 3: Action items — helper comments

Concrete, file:line-targeted comment additions. Each is motivated by a specific episode above.

### AI-1 — `eltwise_chain.hpp`: make the "Streaming pops internally" warning unmissable
**File**: `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp`, line ~166-167 (the `InputLifecycle::Streaming` definition).

The existing note ("Streaming pops each tile after reading...") is present but was missed twice (A2 on `cb_k/cb_v` via matmul, A3 on `cb_attn_mask`). The agent treated `BinaryFpu<...,Streaming>` as "just does the add" and added a manual pop. Strengthen to a warning block:

```cpp
// ⚠ DO NOT add a manual cb_pop_front for any Streaming/HeldBulk/HeldStream input
// after an eltwise_chain call — the chain already pops internally (Streaming: per-tile,
// Bulk: at-end). A redundant pop corrupts the CB read pointer and deadlocks the next
// iteration. This was the root cause of a 3-hour debug session (PCC 0.96 misread as
// numerical; actual cause: double-pop of cb_attn_mask). If a CB must survive the call
// for reuse, use HeldBulk/HeldStream (PopPolicy::None) and pop manually AFTER all reuse.
```

**Episode ref**: A2, A3. This is the highest-value single comment.

### AI-2 — `matmul_block_helpers.hpp`: warn about `WaitAndPopPerKBlock` already popping
**File**: `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp`, line ~67 (the `InputPolicy` doc block).

Add to the `WaitAndPopPerKBlock` description:

```cpp
// ⚠ WaitAndPopPerKBlock (default) pops in0/in1 tiles each K-block. Do NOT add a
// manual cb_pop_front(cb_in0/cb_in1, ...) after the matmul_block loop — that
// double-pops, wraps tiles_received unsigned, and the next wait_front succeeds
// on an empty CB (garbage read → deadlock). For data reused across matmul calls
// (e.g. Q across KV blocks), use WaitAndRetainOnLastBlock or NoWaitNoPop + external mgmt.
```

**Episode ref**: A2 (debugger bug #3). Same class as AI-1, different helper.

### AI-3 — `matmul_block_helpers.hpp`: `MatmulBlockShape::of()` argument-order warning + worked example
**File**: `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp`, line ~156 (the `of()` factory).

The debugger found (E1) that `of(B_q,D_t,B_q,D_t,B_kv,1)` was used instead of `of(1,1,B_q,D_t,B_kv,1)` — passing `D_t` as `in1_num_subblocks`. Add:

```cpp
// ⚠ COMMON MISTAKE: passing output dims as subblock COUNTS. of(B_q,D_t,B_q,D_t,...)
// sets in1_num_subblocks=D_t, producing B_q*D_t*D_t tiles instead of B_q*D_t and
// overflowing DEST. The first two args are SUBBLOCK COUNTS (usually 1,1 for a single
// subblock); out_subblock_h/out_subblock_w carry the actual tile dimensions.
// Correct: of(1, 1, B_q, D_t, B_kv, 1) → one subblock of B_q×D_t tiles.
// Single-tile tests (D_t=1) HIDE this bug — it only overflows at D_t>1.
```

**Episode ref**: E1. The single-tile-hiding-multi-tile-bug pattern (P4) makes this especially costly.

### AI-4 — `matmul_block_helpers.hpp`: CB sizing rule for mixed push-count patterns
**File**: `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp`, near the `OutputCBLayout` / `TileRowMajor` docs (line ~55-60 region or a new note near `LastBlockTarget`).

B1 showed that a single CB receiving both 1-tile pushes (QK^T scores) and multi-tile pushes (PV output) can deadlock on `fifo_wr_ptr` alignment even when sized to `2 * max_push_size`. Add:

```cpp
// ⚠ CB PUSH-COUNT UNIFORMITY: a CB that receives pushes of different tile counts
// (e.g. 1-tile QK^T scores then 2-tile PV output) can deadlock on fifo_wr_ptr
// alignment: after N single-tile push/pop cycles, wr_ptr may leave <max_push
// contiguous pages before fifo_limit, and llk_push_tiles asserts (pre-wrap) rather
// than splitting. Either (a) use a SEPARATE CB per push-count, or (b) size the CB
// to a multiple of max_push_size AND ensure all pushes in the cycle are uniform.
```

**Episode ref**: B1. This was the critical unblocker for the whole op and took 5 hypotheses.

### AI-5 — `reduce_helpers_compute.hpp`: `WaitUpfrontNoPop` leaves tiles — warn about reuse
**File**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`, line ~101 (the `WaitUpfrontNoPop` doc).

The debugger (E5) and implementer (B1) both struggled with `WaitUpfrontNoPop` on `cb_scores` leaving tiles that blocked subsequent streaming eltwise access. The current doc says "don't pop (persistent, for tile reuse)" but doesn't warn about the downstream interaction. Add:

```cpp
// ⚠ WaitUpfrontNoPop LEAVES B_q*B_kv tiles in the CB after reduce. If a subsequent
// eltwise_chain on the SAME CB uses Streaming/HeldStream, it will see stale tiles
// unless you explicitly pop them between the reduce and the chain. For reduce→eltwise
// on one CB, prefer WaitAndPopPerTile (streaming) or insert an explicit
// cb_pop_front(cb, reduced_count) after the reduce. Mixing WaitUpfrontNoPop with
// multi-KV-block iteration is a known deadlock source.
```

**Episode ref**: E5, B1.

### AI-6 — `reduce_helpers_dataflow.hpp`: document `prepare_reduce_scaler` format restriction
**File**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp`, line ~66 (the `prepare_reduce_scaler` signature).

The implementer discovered (crumb 57, `22:20:42`) that `prepare_reduce_scaler` only supports Float16_b and Float32 — not the input dtype (e.g. bfloat8_b). This should be a stated precondition, not discovered at runtime:

```cpp
// ⚠ prepare_reduce_scaler ONLY accepts Float16_b or Float32 as the format. It does
// NOT support bfloat8_b or other block-float formats (the scaler is a matmul-layout
// constant tile). When the input dtype is bfloat8_b, the scaler CB MUST be Float16_b
// (or Float32 when fp32_dest_acc_en=True). The same applies to intermediate
// accumulator CBs that carry running max/sum across blocks — bfloat8_b's shared
// exponent loses precision in accumulators.
```

**Episode ref**: G2 (crumb 57).

### AI-7 — `dfb_helpers_compute.hpp` / `eltwise_fill.hpp`: pack_reconfig before pack_tile after a format change
**File**: `ttnn/cpp/ttnn/kernel_lib/eltwise_fill.hpp` (the `init_cb_constant_f` helper) or `dfb_helpers_compute.hpp`.

G1 showed that `init_cb_constant_f` (fill_tile + pack_tile) after a matmul silently corrupts output when the CB format differs from the packer's current format — because the packer was left in the previous op's format. Add to the `init_cb_constant_f` doc:

```cpp
// ⚠ init_cb_constant_f calls pack_tile without reconfiguring the packer format. If the
// preceding op left the packer in a different data format (e.g. matmul → this), the
// packed tiles are silently corrupted (PCC ~0.916 observed). Call
// pack_reconfig_data_format(cb_id) before this helper when the preceding op's output
// format differs from the target CB format. This is especially needed after matmul
// when intermediate CBs are Float32.
```

**Episode ref**: G1 (crumb 61).

---

## Section 4: Action items — agent prompt improvements

### AP-1 — `ttnn-implementer.md`: add a "check for double-pop before chasing PCC" triage rule
The single most impactful prompt edit. The implementer spent ~3h on numerical hypotheses (C2) for a CB-sync bug (A3). Add to the triage/decision section:

> **PCC-debugging order**: Before attributing a PCC failure to numerical precision (exp approximation, dest format, init values), verify CB integrity: (1) grep the kernel for `cb_pop_front` calls on any CB consumed by a helper (`matmul_block`, `eltwise_chain`, `reduce`) — if the helper's policy already pops (`WaitAndPopPerKBlock`, `Streaming`, `Bulk`), the manual pop is a double-pop and the "numerical" symptom is actually corrupted data. (2) Check that no CB receives mixed push-counts without a dedicated output CB. This check takes 2 minutes and would have saved ~3 hours in the SDPA run.

### AP-2 — `ttnn-implementer.md`: escalate to expert-debugger after 2 failed fixes, not 5
The implementer made 5 failed fix attempts on the multi-KV-block hang before escalating (crumb 21, `12:56:47`, ~2h after first hang at `10:52:39`). The debugger found 4 root causes in one pass. Add:

> **Escalation threshold**: If you have applied ≥2 fix hypotheses for a hang and neither resolved it, invoke the `ttnn-expert-debugger` subagent. Do not accumulate a 3rd, 4th, 5th hypothesis — the debugger's systematic CB producer/consumer tracing is more reliable than incremental scattershot fixes. Cost of late escalation in the SDPA run: ~2 hours.

### AP-3 — `ttnn-implementer.md`: "single-tile passing" is a red flag, not a green light
P4 shows bugs inert at `D_t=1`/`B_q*B_kv=1` recurred. Add:

> **Single-tile caveat**: When a test passes at `D_t=1` or `B_q*B_kv=1` but fails at multi-tile, the first hypothesis should be "the bug is masked by the degenerate case," not "the multi-tile path has a new bug." Specifically check: (1) MatmulBlockShape `of()` argument order (subblock counts vs dims), (2) CB sizing (`B_q*B_kv` vs `B_q*D_t`), (3) DEST tile-count limits. Single-tile tests hide all three.

### AP-4 — `ttnn-implementer.md`: JIT-cache hygiene after kernel edits
F2 cost ~1h because a stale JIT binary produced garbage that looked like a race condition. The MEMORY.md already has an "SDPA Single-Core: Kernel Install Path" note about this, but it's op-specific. Add a general rule:

> **JIT cache**: After editing any compute/dataflow kernel, if a test produces NaN/garbage/PCC≈0 and the code looks correct, clear the JIT cache (`rm -rf built/tt-metal-cache*` and/or `~/.cache/ttnn/ttnn/generated/kernels/`) before debugging further. A stale cached binary is indistinguishable from a race condition. If adding a `DEVICE_PRINT` "fixes" the bug, suspect a forced recompile, not a timing fix.

### AP-5 — `ttnn-expert-debugger.md`: emit a "single-tile hides this" note per finding
The debugger's 4 bugs (E1-E4) were all inert at single-tile. Adding a per-finding annotation would help the implementer internalize P4. Add to the debugger's output format:

> For each root cause found, state whether it is **inert at D_t=1 / B_q*B_kv=1** (i.e. hidden by single-tile tests). If so, note what minimum shape is needed to reproduce it.

### AP-6 — `ttnn-static-analyzer.md`: add a double-pop / mixed-push-count check
The static analyzer caught A1 (Phase 0 init) but missed A2/A3 (double-pop on `cb_k/cb_v` and `cb_attn_mask`) and B1 (mixed push-counts). These are structural and detectable. Add to the analyzer's checklist:

> 1. **Double-pop detection**: for each `cb_pop_front(cb, N)` in the kernel, check whether any helper consuming `cb` already pops it (`matmul_block` with `WaitAndPopPerKBlock`, `eltwise_chain` with `Streaming`/`Bulk`, `reduce` with `WaitAndPopPerTile`). Flag redundant pops.
> 2. **Mixed push-count detection**: for each CB, collect all `cb_push_back`/`push_back` sites and their tile counts. If a CB receives pushes of different counts (e.g. 1 and 2), flag it — it can deadlock on `fifo_wr_ptr` alignment.
> 3. **MatmulBlockShape::of() argument check**: flag any call where `in0_num_subblocks` or `in1_num_subblocks` equals a dimension that should be `out_subblock_h`/`out_subblock_w` (heuristic: if arg 1 or 2 equals arg 4 or 5, it's likely a subblock-count/dim confusion).

---

## Section 5: Summary

The Flash Attention from-scratch run comprised **~20 distinct debugging episodes** across 7 refinements (R1, R1b, R2, R3, R3b, R4, R5, R6) spanning roughly 19 hours (09:39 → 04:56 next day). The dominant pattern was **CB synchronization bugs** — specifically, redundant manual `cb_pop_front` calls on CBs that helpers already pop internally (`WaitAndPopPerKBlock`, `Streaming`), which appeared 3 times (A2, A3, structurally A1) and was the root cause of the run's two biggest time sinks (the multi-KV-block hang and the full-suite "mask precision" hang). The second pattern was **numerical-precision false scents**: the implementer consistently reached for exp-approximation / dest-format / init-value hypotheses (C2, F2, G1) before verifying CB integrity, costing ~4-5 hours total across the run. The expert-debugger, invoked once, was dramatically more efficient (4 root causes in one pass vs. 5 failed implementer attempts), catching MatmulBlockShape argument-order and CB-sizing bugs that single-tile tests had masked — but its one unresolved issue (Q-block-boundary hang) was ultimately fixed by the implementer's insight to split `cb_pv_out` into a dedicated CB (B1), the single most important fix of the run. The highest-leverage interventions are: (1) a warning comment on `InputLifecycle::Streaming` and `InputPolicy::WaitAndPopPerKBlock` that they pop internally and must not be double-popped, (2) an implementer-prompt triage rule to check for double-pops *before* chasing PCC, and (3) a static-analyzer check for double-pops and mixed push-counts.
