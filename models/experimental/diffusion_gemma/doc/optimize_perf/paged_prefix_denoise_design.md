# capture-once paged-prefix traced denoise (recapture fix)

Owner branch: `diffusion-gemma-function`.
Golden = the current per-block-**recapture** path (already growing-prefix correct via `ec5b64b4891`).
Supersedes `denoise_replay_recovery_plan.md` §1–7 (recon record); §8 `DG_DENOISE_FROZEN_PREFIX`
stays as the speed-over-fidelity single-block fallback.

## STATUS (2026-07-19) — Phase 1 DONE + device-verified; Phase 2 (true paged) remains

**Phase 1 (reveal-mask, no new kernel) is COMPLETE and BIT-EXACT on the full model (QB2).**
- `demo/serving_smoke.py` full 30-layer 26B, 3 blocks: `committed_sha256` **identical** across eager /
  recapture-golden / reveal-mask (`31a59b3e…`). Reveal is **capture-once** (`reveal_mask_reuse`
  every block, 0 recapture) and **1.68× faster** (13.0 vs 21.9 s/block; 19.6 vs 11.7 tok/blk/s).
- Gated `DG_DENOISE_REVEAL_MASK` (default OFF). `DG_DENOISE_REVEAL_PMAX` overrides p_max (default =
  KV-cache seq len). Note: full-span read `[0:cache_len]` clones the cache (`read_prompt_kv_cache_slice`)
  to avoid aliasing/freeing it.
- Tests: `test_paged_prefix_reveal_mask.py` (52) + `test_paged_prefix_plumbing.py` (8) + device
  non-regression `test_device_denoise_loop.py` traced 2/2.

**Lazy capture (`DG_DENOISE_LAZY_CAPTURE`, default OFF) — DONE + device bit-exact.** EarlyHalt
records window `w` on-demand (record-then-`execute_trace`; `begin_trace_capture` is record-only),
so block-0 ttft pays only for windows early-halt runs. Coherent prompt (halts [8,2]), full 30L:
reveal+lazy `committed_sha256` == golden (`19ada7e9…`) BIT-EXACT; `capture_events` 48→8; **ttft
108.8→54.0s (~2×)**; steady-state unchanged. NB: early-halt does NOT fire on the `live_context_sweep`
repetitive FILLER prompt (degenerate → full 48 steps); it fires on coherent prompts.

**Phase 2 primitives DONE + device-verified** (ready for the T8 paged swap):
- **T6 `return_lse`** SDPA kernel (12 C++ files, guarded) — `test_return_lse.py` **6/6** on QB2:
  `return_lse=False` byte-identical + LSE ≈ `torch.logsumexp`. Impl notes in `return_lse_kernel_plan.md`.
- **T7 `merge_attention_partials`** (`tt/attention_merge.py`) — `test_attention_merge.py` **3/3** incl device.

**Remaining:** T8 = swap the 5 full-attn layers to paged read + merge (needs the paged-cache infra
over DG's contiguous cache — the hard part; §1a); T10 = true-sliding #48291 re-run; T11 = 128K/256K
perf+mem. See §6 task table for the split.

Grounded in source: `ttnn/.../transformer/sdpa/*` (op surface + compute kernels),
`models/demos/gemma4/tt/attention/operations.py` (paged + sliding slicers),
DG `tt/{traced_denoise,diffusion_attention,denoise_forward}.py`.

## 0. Problem & contract
The traced denoise loop **recaptures every block** (steady 71–84 s/block ≈ 3.6 tok/s vs eager ~7),
because the denoise attention's prefix K/V and mask **grow +canvas_len (256) per block** and a Metal
trace bakes tensor shapes at capture. Root cause = the growing concatenated prefix read
(`read_prompt_kv_cache_slice`) + the `prompt_len`-keyed `invalidate_prefix_growth` guard. RoPE is
already trace-fixed via the constant-shape `canvas_rope_provider` written input.

Contract for the fix:
- **C1 — no recapture.** Every per-block-varying shape becomes a **constant-shape written input**
  refreshed OUTSIDE capture → one capture replays every block.
- **C2 — early-halt unchanged.** `EarlyHaltTracedDenoiseController` (group_size=1) untouched: all
  read spans / page tables / tails / masks are **constant within a block**, change only BETWEEN
  blocks (outside capture). Per-step 8-byte halt readback + `eval_halt` byte-identical.
- **C3 — multi-block, no leak.** Later blocks attend earlier blocks' **committed** KV; uncommitted /
  future positions are **never read** (frozen_prefix violates this; this design does not).

## 1. Architecture (per attention-layer kind)

DG has **30 layers = 5 full-attention (K=V-tied, head_dim 512, every 6th) + 25 sliding-window
(head_dim 256, window W=1024)**. The two kinds get different treatments.

### 1a. FULL-ATTENTION 5 layers — paged chunked SDPA + LSE-merge (Phase 2)
Canvas Q (C=256 rows, abs pos `prompt_len .. prompt_len+C-1`) attends `[prefix(P committed) ++
canvas(C)]`, bidirectional on canvas. Decompose into two partials + a merge:
- **Prefix partial** `canvas→prefix` is causal-equivalent (canvas is entirely after prefix): with
  `chunk_start_idx = prompt_len`, paged `ttnn.transformer.chunked_scaled_dot_product_attention`
  over the **prefix pages** gives each query all P committed keys. `page_table_tensor` (fixed shape)
  + `chunk_start_idx_tensor` ([1] int32 device tensor, runtime offset, no recompile — **trace-safe**,
  proven in llama3_70b_galaxy) are the written inputs. Op exists, paged, head_dim 512 OK
  (gemma4 `chunked_prefill_sdpa` uses grid `(8,4)`, `q_chunk=k_chunk=128`, HiFi4+fp32-acc).
  No leak: `page_table` exposes only committed pages **and** `chunk_start_idx=prompt_len` causal-bounds
  the read to committed rows.
- **Canvas partial** `canvas→canvas` is the existing non-causal C×C local SDPA
  (`_sdpa_q_chunked`, `is_causal=False`). Fixed shape, no kernel.
- **Merge** = flash online-softmax combine of the two partials, needs each partial's **LSE**.

**MERGE decision: LSE-kernel-extension (C++/LLK build required).** Both SDPA partials emit their
internal flash log-sum-exp via a new `return_lse` (default False → byte-identical to today); the
2-way combine is a **pure-ttnn** `merge_attention_partials` (max/sub/exp/mul/add/div), no kernel.
Rejected alternatives: (a) two-pass recompute re-materializes the C×P prefix scores — defeats the
long-context reason paged exists; (b) mask-into-chunked-op adds a general additive-mask + non-causal
mode to the causal-only paged kernel = a strictly larger LLK surface than emitting a statistic it
already computes.

**Why this is a *localized* kernel change, not a new flash kernel:** the LSE machinery already
exists in the shared `compute_streaming.hpp` and is emitted by the **ring-joint** path
(`ring_joint_sdpa.cpp` `cb_lse_out`, "eager norm: LSE"; the running max `m` + `normalize_row_streaming`
already holds the running sum `l`). `sdpa.cpp` (plain + chunked) `#include`s the same
`compute_streaming.hpp`. So `return_lse` = expose `lse = m + log(l)` from the existing reduction
into an optional second output CB/tensor. Keep `return_lse=False` bit-identical for all existing
callers (gemma4).

Minimal kernel API (stub downstream against this NOW):
```
# ttnn/.../transformer/sdpa/{sdpa.cpp,sdpa.hpp,sdpa_nanobind.cpp} + sdpa program factory + compute/sdpa.cpp
scaled_dot_product_attention(..., return_lse: bool = False)          # default → identical to today
chunked_scaled_dot_product_attention(..., return_lse: bool = False)
#   return_lse=True → (output[1,H,C,vhd], lse[1,H,C,1] fp32 = m + log(l))   # flash running max + log(exp-sum), post-scale
```
Pure-ttnn merge (unit-testable now vs mocked LSE):
```
merge_attention_partials(out_a, lse_a, out_b, lse_b) -> out
#   m = max(lse_a, lse_b); wa = exp(lse_a - m); wb = exp(lse_b - m)
#   out = (out_a*wa + out_b*wb) / (wa + wb)
# Algebraically exact; bf16 rescale drift → gate on decision-agreement, not bitwise.
```

### 1b. SLIDING 25 layers — NO kernel (windowed SDPA + tail buffer)
A sliding query at `prompt_len+j` attends prefix only in `(prompt_len+j-W, prompt_len]`; since
`C=256 < W=1024`, the union over all j = the **last W committed prefix rows**. So no paging is
needed. **Tail buffer**: persistent `[1, kv, W, hd]` K and V per sliding layer-type, constant shape,
allocated before capture, refreshed per block OUTSIDE capture via `ttnn.copy` of the last-W committed
rows (W=1024, commit=256 both 32-aligned → exact). Concat `[tail ++ canvas]` = `[1,kv,W+C,hd]`,
run the existing masked non-causal SDPA. O(W), context-independent, no leak.

Regular `scaled_dot_product_attention(sliding_window_size=W, is_causal=False)` = bidirectional
window `abs(q−kv)≤W` on-device (nanobind: "if !is_causal, window centered at current position"),
so the window can be enforced by the op OR by a purpose-built additive mask.

### 1c. Phase split (memory SURPRISE — do not skip)
**Today's traced path (`_trace_safe_call`) is maskless ALL-ATTEND — it ignores the sliding window.**
It equals HF only while `prompt_len+C-1 ≤ W`. Therefore:
- **Phase 1 sliding = maskless** (bit-exact to the current all-attend golden).
- **Phase 2 sliding = true bidirectional-sliding overlay** (a *decision change* vs today → its own
  #48291 re-validation). Gate separately.

## 2. Plumbing (all no-kernel)
- Persistent written inputs allocated **BEFORE** `begin_trace_capture`, refreshed OUTSIDE capture:
  `page_table_tensor` `[1,num_pages]` int32; `chunk_start_idx_tensor` `[1]` int32 = committed
  `prompt_len`; sliding tail K/V `[1,kv,W,hd]`; (Phase-1 full-attn) reveal-mask `[1,1,C,P_max]`.
  Add `prepare/update_paged_prefix_buffers` on the adapter **mirroring**
  `prepare/update_canvas_rope_buffers`.
- **commit → page write**: existing commit (`tt/commit_decode.py`) writes KV into the cache OUTSIDE
  the trace; `update_paged_prefix_buffers` appends pages / bumps `chunk_start_idx` / `ttnn.copy`s the
  last-W rows into the tails — all constant-shape device ops.
- **Decouple `prompt_len`** (it currently drives read-span AND `q_rope_offset` AND mask anchor
  simultaneously): read-span → `page_table` (P2) / fixed `P_max` (P1); reveal → `chunk_start_idx`;
  RoPE offset + mask anchor **stay** `prompt_len`. Assert `revealed_len == chunk_start_idx == prompt_len`.
- **Guard removal**: demote `invalidate_prefix_growth` to a no-op in both `denoise_block` guards
  (base + early-halt); `advance_prefix_after_commit` updates written inputs, not the read-span.
  Keep a `DG_FROZEN_PREFIX` escape hatch.
- **session-8 buffer-lifetime trap**: allocate EVERY persistent cross-replay buffer in `prepare_*`
  before capture, else trace scratch clobbers it every replay.

## 3. Correctness gates
1. **Multi-block decision-agreement vs recapture golden** (QB2): ≥3 blocks, committed argmax
   bit-exact (CROSSBLOCK_OK 100%), `recapture_after_block0=false`, 1 capture (0
   `invalidate_prefix_growth`), 3.6→~18 tok/s (+early-halt ~47).
2. **Sliding #48291 re-validation** (QB2): P1 maskless bit-exact to golden; P2 true-sliding is a
   decision change → own run, argmax-agreement ≥5 seeds at/above the bf16 floor (≥0.992 @prod-48;
   strict 0.95 gate is unreachable by any bf16 impl — mis-specified gate per project decision; do
   not chase precision). Gated.
3. **Bit-exactness**: P1 full-attn reveal-mask bit-exact (tile-aligned masked tail = no-op tiles;
   `exp_approx_mode=False`; verify SFPU `exp(0)==1.0`); P1 sliding maskless bit-exact; P2 paged
   prefix partial bit-exact to a contiguous read; P2 LSE-merge exact in fp32, bf16 rescale →
   decision-agreement (zero argmax flips).
4. **No-leak / determinism**: updated-input→output-changes replay test; read-never-exposes-
   uncommitted predicate (`j < prompt_len`); replay determinism; **zero-init uncommitted rows**
   (`NaN + -inf = NaN` poisons softmax).

## 4. Gotchas
- Do NOT reuse `build_canvas_denoise_mask(P_max,C)` for reveal (it anchors canvas at `P_max` and
  reveals the tail) — purpose-built builder anchored at `prompt_len`, explicit `j < prompt_len`.
- Force the sliding mask **always-present** (drop the `_sliding_layer_needs_denoise_mask`
  short-circuit) else a sometimes-None mask makes the graph non-constant → breaks trace-safety.
- The masked SDPA has **no `_manual_gqa_attention` L1-CB fallback** (that fires only when
  `attn_mask=None`) → caps reveal-mask `P_max` on the head_dim-512 layers → a reason P2 paged is the
  real long-context answer.
- Refresh via persistent buffer + `ttnn.copy`, **never** `ttnn.from_torch` in-trace.
- Chunked SDPA Q-chunk length must be a multiple of 128 (pad/slice — gemma4 `operations.py:286`).

## 5. Sequencing — CORRECTNESS-FIRST, kernel as drop-in speedup
- **Phase 1 (no build, runnable now)** — satisfies C1+C2+C3 on bounded contexts:
  sliding 25 → tail-buffer windowed SDPA (scales to any context); full-attn 5 → contiguous-fixed-max
  + reveal-mask over the existing masked SDPA (correct multiblock, bit-exact, O(P_max) waste OK while
  bounded); plumbing + guard removal. Controllers unchanged except the demoted guard → capture-once,
  multiblock-correct, early-halt intact → **3.6→~18 tok/s (+early-halt ~47)**. Ship first.
- **Phase 2 (C++/LLK) — drop-in for the 5 full-attn layers**: add `return_lse`, implement
  `merge_attention_partials`, swap those 5 to paged-causal-prefix + LSE-merge, enable true sliding on
  the 25. Removes the O(P_max) waste → true 128K/256K. **Controller/adapter interface identical**;
  only the per-layer read+SDPA internals swap.

Everything behind `DG_DENOISE_PAGED_PREFIX` / `DG_DENOISE_REVEAL_MASK` / `DG_DENOISE_SLIDING_TAIL`
(default OFF) until QB2.

## 6. Task list (dependency-ordered)
| id | status | bucket | title | files |
|----|--------|--------|-------|-------|
| T1 | ✅ DONE | no-kernel | reveal-mask written inputs (adapter `prepare/update_reveal_mask_buffers` + reader `set_read_span`) | `tt/denoise_forward.py` |
| T2 | ⚠️ Phase-1 done via unified reveal-mask (all 30 layers hide-tail); dedicated sliding-tail buffer is a P2 opt | `no-kernel` | sliding tail-buffer windowed SDPA | `tt/denoise_forward.py` |
| T3 | ✅ DONE | no-kernel | reveal-mask builder (`build_canvas_reveal_denoise_mask`) + device wrapper + forward threading | `reference/attention_mask.py`, `tt/denoise_forward.py` |
| T4 | ✅ DONE | no-kernel | controller capture-once: demote guard→`reveal_mask_reuse`, `_prepare_reveal_if_enabled`, decouple prompt_len | `tt/traced_denoise.py`, `tt/denoise_forward.py` |
| T5 | ✅ DONE | no-kernel | host unit tests (reveal builder 52, plumbing 8, no-leak, golden-equiv) | `tests/test_paged_prefix_reveal_mask.py`, `tests/test_paged_prefix_plumbing.py` |
| T7 | ✅ DONE (device 3/3) | no-kernel | `merge_attention_partials` pure-ttnn + unit test | `tt/attention_merge.py`, `tests/test_attention_merge.py` |
| T6 | ✅ DONE (device 6/6) | cpp-kernel | `return_lse` on chunked + regular SDPA (expose `m+log(l)`) | `ttnn/.../sdpa/*` (12 files), `tests/test_return_lse.py` |
| T8 | ⬜ REMAINS | cpp-kernel | swap 5 full-attn layers to paged prefix + `merge_attention_partials` (+ paged-cache infra over DG contiguous cache) | `tt/diffusion_attention.py`, `tt/denoise_forward.py`, `tt/commit_decode.py` |
| T9 | ✅ DONE (bit-exact full 30L) | qb2-device | Phase-1 verify (capture-once, 3-block bit-exact, tok/s) — `serving_smoke` reveal==recapture==eager | `doc/optimize_perf/` |
| T10 | ⬜ REMAINS | qb2-device | sliding #48291 re-validation (true window) | `doc/optimize_perf/` |
| T11 | ⬜ REMAINS | qb2-device | Phase-2 long-context perf+memory (128K/256K) | `doc/vllm_integration/` |
