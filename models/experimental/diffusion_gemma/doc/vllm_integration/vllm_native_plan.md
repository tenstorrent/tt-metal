# DiffusionGemma → vLLM-native serving: technical plan (#47466)

> **DESIGN DOCUMENT, not current launch/performance status.** Produced 2026-07-07; file:line
> references and intermediate capability statements may be stale. The keystone (vLLM-owned paged
> KV plus non-zero-position write and paged denoise read) remains open. Use `README.md` and
> `plan.md` Part 0 for the 2026-07-17 execution contract.

Make DG's serving features **vLLM-native** (engage the framework's APC / chunked-prefill /
continuous-batching / speculative-decode) instead of the current model-side prototypes, on the
tenstorrent/vllm TT plugin. Grounded in the real code (file:line); load-bearing claims independently
verified (see **Verification** below). Produced 2026-07-07.

> Context (Viktor Pus, TT vLLM plugin owner, Slack): *"If the model can prefill from a NON-ZERO token
> position, APC should work. Chunked prefill will likely use a similar model-side mechanism, so in
> principle doable. But the greatest benefit is to avoid decode stalls during long prefills — best
> achieved with mixed prefill + decode batches. Major model code change. Without that, we can perhaps
> simulate something via scheduler modification."* Ben Goel's asks: proper vLLM chunked prefill →
> continuous batching; native 256K context; APC to cut TTFT on multi-turn agentic; native vLLM
> spec-decode (not model-side hacks).

## TL;DR — the keystone

**Move DG off its model-owned CONTIGUOUS KV cache onto a vLLM-owned (hybrid) PAGED block pool, and make
`prefill_forward` write only the uncached suffix/chunk at absolute RoPE positions, while the denoise
frozen-prefix read comes from the paged pool via per-request page tables.** Every native feature (APC,
chunked-prefill/continuous-batching, 256K, spec-decode) is a corollary of this one change; none has
non-zero value until it lands. It is three coupled moves:

1. **OWNERSHIP** — today `initialize_vllm_model` builds with `create_kv_cache=True`
   (`generator_vllm.py:163`) and `allocate_kv_cache` / `allocate_kv_cache_per_layer`
   (`generator_vllm.py:265-285`, `_model_owned_kv_handles`) return the model's own contiguous cache, so
   vLLM's BlockPool/KVCacheManager is **bookkeeping-only with no physical blocks to hash/reuse/evict**.
   Change to `create_kv_cache=False` + return real paged tensors `[max_num_blocks, num_local_kv_heads,
   block_size, head_dim]` (the `tt_transformers/generator_vllm.py` `allocate_vllm_kv_cache_per_layer`
   shape), using a **diffusion_gemma-local COPY** of gemma4 `kv_cache.py` `init_kv_cache` +
   `kv_cache_hybrid.py` (no shared edits).
2. **WRITE from non-zero position** — `prefill_forward` currently `del`s `kv_cache, start_pos,
   page_tables_per_layer, sampling_params` (`generator_vllm.py:342`) and prefills the WHOLE prompt in
   one call. It must instead write only `tokens[num_computed:num_computed+num_new]` at `chunk_start_idx`
   via `paged_fill_cache(chunk_page_table)`. **This mechanism is ALREADY device-proven** in
   `tt/chunked_prefill.py` (`chunk_start_idx` RoPE offset; `paged_fill_cache`;
   `chunked_scaled_dot_product_attention(chunk_start_idx=...)`; **PCC 0.99997 at 2048=8×256** incl
   sliding-past-window) — it is just env-gated OFF (`DG_CHUNKED_PREFILL`) and not wired to vLLM.
   **Viktor's precondition is therefore already met at the kernel level.**
3. **READ paged** (the load-bearing model change) — `diffusion_attention.py:387-395` materializes the
   frozen prefix as `[1,1,P,H]` via `ttnn.slice` over the contiguous cache then plain SDPA; this must
   become a **paged read** of the prefix blocks via `page_table`.

**Key leverage:** the runner ALREADY delivers everything needed — `model_runner.py:919`
(`input_positions = num_computed_tokens_cpu[...]` = `chunk_start_idx`), passed as `start_pos`
(`:2337`) + `page_tables_per_layer` (`:2345`) into `prefill_forward` — **DG just throws it away today.**
So the change is smaller than it looks. Landing this = #47488 (PR-ready, `zni/dg-47488-block-granular`)
+ the cache-ownership swap; #47557 (batched canvas decode) then unlocks >1 concurrent sequence.

## Feature dependency tree

```
#47488 (block-granular runner/scheduler, PR-ready) ──▶ land first
        │
KEYSTONE: paged KV ownership + prefill-from-non-zero
        ├─ Prefill-from-non-zero (stateless per-chunk prefill)   [prototype-able NOW, offline]
        ├─ Paged denoise prefix read  ★ hardest, correctness gate (#48291)
        │        └─ #47557 batched canvas decode ──▶ concurrency
        ├─ APC (framework block-hash prefix reuse)
        ├─ Chunked prefill → continuous batching (incl mixed-batch question)
        └─ Native 256K context (hybrid paged KV)
Spec-decode ──▶ research spike, likely OUT of native scope (see below)
```

## Sequenced steps

- **Step 0 — quick wins (no keystone dependency, do now):**
  (a) fix `get_kv_cache_spec` TP=4 degenerate head count — full-attn `num_global_key_value_heads=2`,
  `2//4 = 0` per device (`generator_vllm.py:~231-232,255`); declare `num_kv_heads=1` to match the
  physical cache + give uniform hybrid page size. *(Sliding `num_key_value_heads=8` is fine: 8//4=2.)*
  (b) add `model_capabilities.supports_chunked_prefill` + `supports_prefix_caching_with_sliding_window`
  flags and convert the three hard-disables (`platform.py:470-472`, `model_runner.py:147-148`,
  `platform.py:659-666`) into capability gates left OFF — no behavior change, unblocks later flips.
  (c) refactor `tt/chunked_prefill.py` into a stateless `prefill_one_chunk()` + re-prove PCC 0.99997
  offline via `DG_CHUNKED_PREFILL=1`.
- **Step 1 — land #47488** (block-granular runner/scheduler + n-token block accounting).
- **Step 2 — KEYSTONE A (ownership):** `create_kv_cache=False`; copy gemma4 `kv_cache.py` +
  `kv_cache_hybrid.py` into `diffusion_gemma/tt/`; return real paged handles; stop `del`-ing
  `start_pos`/`page_tables_per_layer`.
- **Step 3 — KEYSTONE B (write):** wire stateless chunked prefill into `prefill_forward`
  (`paged_fill_cache` at `chunk_start_idx=num_computed_tokens`); runner slices
  `[num_computed:num_computed+num_new]` (`model_runner.py:927-929`); persist the per-request
  sliding-window buffer across engine steps.
- **Step 4 — KEYSTONE C (read, hardest):** rewire the denoise prefix read to paged-gather +
  bidirectional-mask SDPA; run the **#48291 fidelity gate** to confirm diffusion argmax decisions do
  not drift vs the contiguous baseline.
- **Step 5 — #47557 batched canvas decode:** remove the `num_reqs>1` raise; batched denoise-block
  forward + per-request paged reads. Unlocks >1 concurrent sequence.
- **Step 6 — APC:** flip `supports_prefix_caching=True`; enable the sliding-window APC carve-out; keep
  the hybrid manager ON; validate multi-turn TTFT reduction + partial-hit bf16 fidelity.
- **Step 7 — chunked prefill → continuous batching:** enable `supports_chunked_prefill`; relax
  `TTScheduler` forced-mode; scheduler-level interleave of prefill chunks with denoise blocks; scope to
  `tt_data_parallel=1`.
- **Step 8 — 256K:** confirm the fork hybrid manager gives the full-attn group `max_model_len/block_size`
  blocks (not an even split); pick `block_size` dividing `sliding_window=1024` and aligned to 256/32;
  device-validate 256K on QB2.
- **Step 9 — spec-decode:** research spike only; do not build against the AR spec_decode driver; confirm
  scope with Viktor/Ben first.

## The block ≠ token reinterpretations (DG-specific, sharp)

DG is **block-diffusion**: a "decode step" is a whole 256-token denoise BLOCK (≤48 sub-steps of
Gumbel-max → entropy-budget accept → renoise), **not** a token. This changes the autoregressive framing:

- **Continuous batching = advance N *blocks* per step, not N tokens.** The batched unit (#47557) is a
  256-token block.
- **"Mixed prefill+decode batch" (Viktor's greatest benefit) is a category error for DG if taken
  literally:** a fused kernel with a many-query prefill row + 1-query decode rows is incoherent because a
  DG "decode row" is a 48-step *bidirectional* canvas loop with on-device sampling — a different graph,
  48× the steps of a 1-pass causal chunk. **The DG-coherent analog is scheduler-level interleave of
  prompt-prefill CHUNKS with denoise BLOCKS**, which captures most of the anti-stall benefit with only
  the stateless-chunk model change. A fully-fused mixed kernel is not worth it.
- **APC caches PROMPT-prefix blocks only** — clean fit (causal-prefill K/V at position `i` is a pure
  function of `tokens[0:i]` + absolute RoPE `i`). **Never hash in-flight canvas blocks** (re-noised
  across ≤48 steps).
- **Spec-decode has no AR draft-token analogue:** DG already emits 256 tokens/block, and the
  entropy-budget accept loop is *already* an internal draft/accept mechanism. Native vLLM spec-decode
  framing likely doesn't map; reframe as fewer-denoise-step drafts verified by a full-step target
  (model-side research), and confirm scope before investing.

## Open questions for Viktor

1. **Hybrid KV manager allocation (MAKE-OR-BREAK for 256K):** does the fork allocate the full-attn group
   `max_model_len/block_size` blocks, or split the pool evenly across groups (the ~23K-ISL starvation
   gemma4 escaped via all-FullAttentionSpec)? Determines whether hybrid-ON can admit 256K at all.
2. **Mixed prefill+decode for DG:** is scheduler-level interleave of prompt-prefill CHUNKS with denoise
   BLOCKS acceptable as "chunked prefill → avoid decode stalls", or does Ben want a single fused
   mixed-batch kernel (incoherent for block-diffusion — see above)?
3. **Sliding-window APC gate (`platform.py:659-666`):** OK to gate behind a new capability
   (`supports_prefix_caching_with_sliding_window`) rather than remove it (so non-paged TT sliding models
   stay safe)? And do you prefer the hybrid 2-group spec for APC, or the unified-to-FullAttentionSpec
   fallback (simpler, caches full prefix for sliding layers)?
4. **Chunked-prefill gates:** OK to gate `platform.py:470-472` + `model_runner.py:147` behind
   `supports_chunked_prefill` and have `TTScheduler` bypass the forced all-prefill/all-decode mode
   (delegate to base `AsyncScheduler.schedule()`)?
5. **Shared paged-read work:** the denoise-step KV READ move onto the paged cache is the same change for
   APC + chunked-prefill + 256K — confirm we do it once for all three.
6. **`block_size`:** does the TT plugin pin one, or can we choose a value dividing `sliding_window=1024`
   and aligned to DG's 256-canvas / 32-tile granularity?
7. **APC fidelity:** is standard block-hash APC acceptable if partial-prefix reuse is bf16
   fidelity-approximate (#48291 can flip a diffusion argmax), or must DG gate APC to exact-full-prefix
   hits only?
8. **DP scope:** confirm mixed/chunked batches are scoped to `tt_data_parallel=1` only (which DG serving
   already enforces), never mixed under DP (`lane_scheduler.py:362-371` negotiates one mode across lanes).
9. **Spec-decode:** is native vLLM spec-decode in scope for block-diffusion at all, given DG already
   emits 256 tokens/step and its entropy-budget accept is already a draft/accept mechanism?
10. **#47488 follow-on:** do you want the INPUT-chunking half (runner slicing
    `[num_computed:num_computed+num_new]`) as a follow-on PR on the same branch, or a separate PR?

## Biggest risks

1. **Sequencing hazard:** APC/continuous-batching value is ZERO until paged ownership lands. Flipping
   `supports_prefix_caching=True` / `enable_chunked_prefill` while the cache is still model-owned
   contiguous makes vLLM believe blocks are reusable when they physically are not — a correctness hazard.
   Gates must flip strictly AFTER the keystone.
2. **Paged denoise prefix read (Step 4) is the highest-risk device change:** a NEW read-only,
   *bidirectional* attention path over paged blocks; the proven chunked SDPA is causal-only, so this
   likely needs paged-gather-then-full-SDPA and must reproduce the exact prefix K/V ordering or diffusion
   decisions drift.
3. **bf16 non-bit-exactness:** partial-prefix / shorter-prefix reuse is not bit-exact in bf16 and can
   flip a diffusion argmax (#48291) — a fidelity failure mode absent for AR models. APC partial hits and
   chunked-prefix cross-attention both need an explicit fidelity gate before shipping.
4. **256K three-way tension:** bounded sliding memory needs hybrid + chunked prefill, but hybrid may
   starve the full-attn group below 256K admission (open Q #1), and hybrid also trips the sliding-window
   APC disable. The all-Full collapse fixes admission + APC but re-inflates KV to ~15 GiB (only ~3.6 GiB
   headroom — insufficient at 256K on QB2). Resolution hinges on the fork's per-group allocation.
5. **Everything is blocked behind #47488** (PR-ready, not merged) and #47557 (concurrency); slip on
   either stalls the program.
6. **Cross-step state lifetime:** persisting the per-sliding-layer rolling K/V window buffer across vLLM
   engine steps (was within-call) introduces leak/lifetime risk — release on request completion.
7. **Scheduler invariant relaxation:** dropping `TTScheduler`'s homogeneous mode can break the DP
   `lane_scheduler` (single agreed mode across lanes) and could livelock under DP; scope to
   `tt_data_parallel=1`.
8. **Effort trap:** a true single-dispatch fused mixed prefill+decode kernel is large model work for
   marginal gain over scheduler-level interleave.
9. **QB2 256K device validation is flaky** (eth-core teardown re-hang, board fw 19.9.0 quirk); needs
   `tt-smi -r` between runs.
10. **no-shared-edits drift:** gemma4 `kv_cache`/`kv_cache_hybrid` must be COPIED into
    `diffusion_gemma/`, so upstream gemma4 fixes won't propagate automatically.

## Quick wins (do now, independent of the keystone)

- Fix the `get_kv_cache_spec` TP=4 degenerate head count (full-attn `num_global_key_value_heads=2` →
  `2//4=0`; declare `num_kv_heads=1`) — pure spec fix.
- Add the two `model_capabilities` flags + convert the three hard-disables to capability gates (left OFF)
  — no behavior change, unblocks later flips.
- Refactor `tt/chunked_prefill.py` into a stateless `prefill_one_chunk()` and re-prove PCC 0.99997
  offline — de-risks the write path early.
- Land #47488 (PR-ready).
- Write a bf16 diffusion-argmax fidelity harness now (contiguous baseline vs paged/partial-prefix) so
  Steps 4/6 have a ready gate instead of discovering #48291 drift late.

## Verification

Load-bearing claims independently verified against the code on QB2 (`bhqb`), 2026-07-07 — **all
CONFIRMED**, keystone thesis factually sound. Two corrections applied above vs the first draft:
- The `prefill_forward` del list is `kv_cache, start_pos, page_tables_per_layer, sampling_params`;
  `page_table` is accepted-and-ignored (not `del`'d). Load-bearing point (position/page info discarded)
  holds.
- The TP=4 `2//4=0` degeneracy is on the **full-attention** `num_global_key_value_heads=2`, not
  `num_key_value_heads=8` (which is a healthy `8//4=2`).

Method: 6-agent code-grounded design workflow (keystone/APC/chunked/spec/256K/reference) → synthesis →
adversarial verification. 256K KV math cross-checked: all-full contiguous ≈15 GiB/chip, hybrid ≈2.5
GiB/chip (real geometry ≈1.4), weights ≈13.25 GiB of ≈31.87 GiB/chip — order-of-magnitude holds.
