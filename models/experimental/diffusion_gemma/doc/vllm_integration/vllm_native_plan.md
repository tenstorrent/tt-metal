# vLLM-native serving: the paged-KV keystone (#47466)

Status: open — design document, not launch/performance status. Produced 2026-07-07 by a 6-agent
code-grounded design workflow, then adversarially verified against the code on bhqb; all
load-bearing claims CONFIRMED. **Every `generator_vllm.py` file:line anchor below is stale** (the
file has grown to ~56 KB since), so treat them as "where it was", never as precise refs. Over the
100-line cap because it also absorbs the deleted `prefix_cache/` pair, and its refutations, open
questions and the cache-ownership contract are never cut for length.
Owns: cache ownership and the three-phase KV machine, the paged-KV keystone, the
prefill-from-non-zero dependency, tenstorrent/vllm TT-plugin constraints, and the frozen
prompt-prefix reuse evidence (prototype **DELETED** 2026-07-28).
See also: [refuted list](../REFUTED.md) · [serving README](README.md) ·
[chunked prefill](chunked_prefill/README.md) · [PR #47488](PR_47488.md) · [plan](../../plan.md)

## The keystone (still open)

Move DG off its model-owned **contiguous** KV onto a vLLM-owned **paged** block pool, make
`prefill_forward` write only the uncached suffix/chunk at absolute RoPE positions, and make the
denoise frozen-prefix read come from the paged pool via per-request page tables.

**SEQUENCING FACT:** APC, chunked-prefill/continuous-batching, native 256K and spec-decode are all
corollaries of this one change, and **none has non-zero value until it lands**.

1. **OWNERSHIP** — switch `create_kv_cache=False` and return real paged tensors shaped
   `[max_num_blocks, num_local_kv_heads, block_size, head_dim]`, using a diffusion_gemma-local
   **COPY** of gemma4 `kv_cache.py` + `kv_cache_hybrid.py` ([no shared edits](../../AGENTS.md)).
2. **WRITE from non-zero position** — already **device-proven** in `tt/chunked_prefill.py`
   (`chunk_start_idx` RoPE offset, `paged_fill_cache`,
   `chunked_scaled_dot_product_attention(chunk_start_idx=...)`, PCC 0.99997 at 2048 = 8x256
   including sliding-past-window), simply not wired to vLLM. The plugin owner's
   prefill-from-non-zero precondition is therefore **already met at the kernel level**.
3. **READ paged** (the load-bearing model change) — `diffusion_attention.py` materializes the frozen
   prefix as `[1,1,P,H]` via `ttnn.slice` over the contiguous cache then plain SDPA; this must
   become a paged read via `page_table`.

**KEY LEVERAGE:** the runner already delivers everything needed —
`input_positions = num_computed_tokens_cpu[...]` (= `chunk_start_idx`) is passed as `start_pos`
together with `page_tables_per_layer` into `prefill_forward` — and DG throws it away by
`del kv_cache, start_pos, page_tables_per_layer, sampling_params`. **VERIFIED CORRECTION:**
`page_table` is accepted-and-ignored rather than `del`'d; the load-bearing point that
position/page info is discarded holds.

Feature order: land #47488 → keystone A (ownership) → B (write) → C (paged read, hardest, gated by
the #48291 fidelity harness) → #47557 batched canvas decode → APC → chunked prefill/continuous
batching → 256K → spec-decode spike. **Recommended de-risking step:** write the bf16
diffusion-argmax fidelity harness (contiguous baseline vs paged/partial-prefix) **before** the paged
read and APC steps, so #48291-class drift is not discovered late.

DONE since this plan was written: the `get_kv_cache_spec` TP=4 degenerate head count.
**VERIFIED CORRECTION:** that `2//4 = 0` degeneracy was on the FULL-attention
`num_global_key_value_heads=2`, not `num_key_value_heads=8` (a healthy `8//4=2`).
`tt/generator_vllm.py` now emits `FullAttentionSpec` with `full_kv_heads_per_dev` and
`tests/test_vllm_live_context_sweep.py` asserts `num_kv_heads == 1` on the full-attention layer. NOT
done and NOT started: the stateless `prefill_one_chunk()` refactor of `tt/chunked_prefill.py` — no
such function exists.

## Cache ownership and the three-phase KV machine

The denoise read path reads the frozen prompt prefix from the model-owned **contiguous**
`tt_model.tt_kv_cache` via `ttnn.slice` (`tt/denoise_forward.py`
`read_prompt_kv_cache_by_layer` / `read_prompt_kv_cache_slice`), **not** from a vLLM paged block
pool. The model owns its `max_model_len` cache (`create_kv_cache=True`) and is driven with
`page_table=None`; `allocate_kv_cache[_per_layer]` return those existing handles via
`_model_owned_kv_handles`, so there is no double allocation.

| phase | direction | KV effect |
|---|---|---|
| prefill | causal | **writes** `[0:cache_len]` |
| denoise | bidirectional | **read-only** slice of `[0:cache_len]`; the canvas recomputes its own K/V every step |
| commit | causal | **appends** at `[cache_len : cache_len + 256*N]`, strictly after the prompt region |

Consequence: a session's own commits only touch positions `>= cache_len`, so a prompt prefix it
reused stays valid throughout its generation.

## Frozen prompt-prefix reuse — measured, then deleted

The `DG_PREFIX_CACHE` prototype (`tt/prefix_cache.py`, `demo/prefix_cache_smoke.py`,
`tests/test_prefix_cache.py`, branch `dg-vllm-apc` SHAs `cbddeef07ca` and `09f7f075e52`) **no longer
exists**; it was deleted 2026-07-28 and `tt/serving.py` now hard-sets `prefill_reused = False`. Its
reproduction command is dead. Its acceptance bar was bit-exact committed argmax with the cache ON vs
OFF for two requests sharing a long prefix, plus a logged prefill-time saving — met only by the
exact-full-match tier. What survives is the measurement and the argument.

**WHY REUSE IS SOUND AT ALL (three-part argument):** RoPE is absolute and prefix-anchored, so
position `i`'s rotation depends only on `i`; causal prefill makes position `i`'s K/V a pure function
of `tokens[0:i]` plus position `i`, regardless of total prefill length; and **both** sliding-window
layers (window 1024, theta=1e4) and full-attention layers (theta=1e6) attend to a subset of `[0:i]`,
so no layer type breaks the prefix invariant.

**ALIGNMENT RULE (the one non-obvious constraint):** prefill pads the prompt to a 32-tile multiple
and the denoise read covers the aligned `cache_len` **including** the pad positions, so a reuse span
must match the resident byte-for-byte over the entire `new_cache_len`. A non-32-aligned proper
prefix does not qualify at all, because its zero-pad would claim positions holding the resident's
real-token K/V.

Device result (QB2, 2026-07-06, `--num-layers 2`, 1 block, 2 steps,
`prefix_cache/prefix_cache_smoke_reduced.json`): `DG_PREFIX_CACHE_SMOKE_SUCCESS
required_bit_exact_pass=True all_reused=True exact:bit_exact=True,mismatch=0,reused=True,saved_s=0.127
prefix:bit_exact=False,mismatch=57,reused=True,saved_s=0.168`

- **Exact-full-match reuse IS bit-exact on device: 0/256** committed-argmax mismatches with the
  prefill forward skipped entirely (`prefill_time_on_s=0.0`).
- **REFUTED — aligned *proper*-prefix reuse:** fp32-correct but **not** bit-exact in bf16; **57 of
  256** committed-argmax tokens flip, because the resident wrote that span inside a *longer* prefill
  and the SDPA reduction over the longer masked key span rounds differently than a standalone
  shorter prefill. A bit-exact shorter-prefix reuse would need a length-independent attention
  reduction — a kernel property, not changeable at this layer — so it belongs to the paged path.
- **SCALE OF THE PRIZE:** 0.127 s saved at 2 layers; at full depth prefill is ~60 s, so exact-match
  reuse would skip approximately that whole cost. **NEVER COMPLETED:** the full-depth realistic
  saving run was blocked by the eth core 29-25 fault at `open_mesh_device` and never re-run, so that
  number does not exist. The bit-exactness result was taken at 2 layers on the argument that the
  reuse decision, cache skip, RoPE offset, denoise read and commit are layer-count-independent. The
  prototype quantified what a paged path could capture by logging
  `partial-prefix miss: matched N aligned tokens, suffix differs -> full prefill`.

**BLOCKED ON #47488:** the productionally valuable shared-prefix-with-differing-suffix case (same
system prompt, different user turn) cannot be made bit-exact at the DG serving layer, because the
suffix tokens must cross-attend to the cached prefix during prefill — i.e. chunked/prefix prefill,
which the shared backbone discards ([the gemma4 bug](chunked_prefill/README.md)) and which DG may
not fix by editing it. **REFUTED alternative:** a DG-local commit-decode "suffix prefill"
(token-by-token via the decode path) is functionally correct but not bit-exact to the batched
prefill matmul geometry in bf16, so it fails the bar. General APC belongs in the paged path.

## TT-plugin constraints (tenstorrent/vllm), already respected

Spec decode is hard-blocked (`platform.py:342-344`); the scheduler does not support chunked prefill
(`platform.py:339-341`, re-asserted `model_runner.py:251-252`); batching is phase-based — a step is
all-prefill OR all-decode (`scheduler.py:30-100`, `model_runner.py:911-914`); APC is force-disabled
for sliding-window models (`platform.py:512-521`); there is no prompt-length divisibility
requirement (`model_runner.py:924-928`). The model owns forward, attention and KV.

Plugin-owner context (Viktor Pus, Slack): if the model can prefill from a **non-zero** token
position, APC should work; chunked prefill would use a similar model-side mechanism; the greatest
benefit is avoiding decode stalls during long prefills via mixed prefill+decode batches, which is
major model work. Stakeholder asks (Ben Goel): proper vLLM chunked prefill leading to continuous
batching, native 256K context, APC to cut multi-turn agentic TTFT, and native vLLM spec-decode
rather than model-side hacks.

## Block != token (DG-specific)

- **Continuous batching = advance N 256-token BLOCKS per step, not N tokens.** The batched unit is
  #47557.
- **REFUTED AS A CATEGORY ERROR:** "mixed prefill+decode batch" taken literally. A DG decode row is a
  48-step **bidirectional** canvas loop with on-device sampling — a different graph with 48x the
  steps of a 1-pass causal chunk. The DG-coherent analog is scheduler-level interleave of
  prompt-prefill CHUNKS with denoise BLOCKS; a fully-fused mixed kernel is not worth it.
- **APC may cache PROMPT-prefix blocks only**, because causal-prefill K/V at position `i` is a pure
  function of `tokens[0:i]` plus absolute RoPE `i`. **In-flight canvas blocks must never be hashed**
  — they are re-noised across up to 48 steps.
- **REFUTED / OUT OF SCOPE:** native vLLM spec-decode. DG has no AR draft-token analogue — it
  already emits 256 tokens per block and its entropy-budget accept loop is already an internal
  draft/accept mechanism. Confirm scope before investing.

## Risks

1. **SEQUENCING HAZARD:** flipping `supports_prefix_caching=True` or `enable_chunked_prefill` while
   the cache is still model-owned contiguous makes vLLM believe blocks are reusable when they
   physically are not — a correctness hazard. The gates must flip strictly AFTER the keystone.
2. The **paged denoise prefix read** is the highest-risk device change: a NEW read-only
   *bidirectional* attention path over paged blocks, while the proven chunked SDPA is causal-only.
   It likely needs paged-gather-then-full-SDPA and must reproduce the exact prefix K/V ordering or
   diffusion decisions drift.
3. **bf16 non-bit-exactness:** partial- or shorter-prefix reuse is not bit-exact in bf16 and can flip
   a diffusion argmax (#48291) — a fidelity failure mode absent for AR models. APC partial hits and
   chunked-prefix cross-attention each need an explicit fidelity gate before shipping.
4. **256K three-way tension:** bounded sliding memory needs hybrid + chunked prefill, but hybrid may
   starve the full-attn group below 256K admission and also trips the sliding-window APC disable;
   the all-Full collapse fixes admission and APC but re-inflates KV to ~15 GiB, leaving only ~3.6
   GiB headroom — insufficient at 256K on QB2.
5. **Cross-step state lifetime:** persisting the per-sliding-layer rolling K/V window buffer across
   vLLM engine steps (it was within-call) introduces leak/lifetime risk; release on request
   completion.
6. Dropping `TTScheduler`'s homogeneous mode can break the DP `lane_scheduler`, which negotiates one
   agreed mode across lanes, and could livelock under DP — scope to `tt_data_parallel=1`.
7. QB2 256K device validation is flaky (eth-core teardown re-hang, board fw 19.9.0 quirk) and needs
   `tt-smi -r` between runs.
8. **no-shared-edits drift:** gemma4 `kv_cache`/`kv_cache_hybrid` must be COPIED into
   `diffusion_gemma/`, so upstream gemma4 fixes will not propagate automatically.

256K KV math cross-check: all-full contiguous ~15 GiB/chip, hybrid ~2.5 GiB/chip (real geometry
~1.4), weights ~13.25 GiB of ~31.87 GiB/chip.

## Open questions for the plugin owner (unanswered)

1. **MAKE-OR-BREAK for 256K:** does the fork's hybrid KV manager allocate the full-attn group
   `max_model_len/block_size` blocks, or split the pool evenly across groups (the ~23K-ISL
   starvation gemma4 escaped via all-`FullAttentionSpec`)?
2. Is scheduler-level interleave of prefill CHUNKS with denoise BLOCKS acceptable as "chunked
   prefill", or is a single fused mixed-batch kernel wanted?
3. May the sliding-window APC disable be gated behind a new
   `supports_prefix_caching_with_sliding_window` capability rather than removed, and is the hybrid
   2-group spec preferred over the unified-to-`FullAttentionSpec` fallback?
4. May the chunked-prefill hard-disables be gated behind `supports_chunked_prefill`, with
   `TTScheduler` delegating to base `AsyncScheduler.schedule()`?
5. The denoise-step KV **read** move onto the paged cache is the same change for APC, chunked prefill
   and 256K — confirm it is done once for all three.
6. Does the TT plugin pin `block_size`, or may DG choose a value dividing `sliding_window=1024` and
   aligned to the 256-canvas / 32-tile granularity?
7. Is standard block-hash APC acceptable when partial-prefix reuse is bf16 fidelity-approximate, or
   must DG gate APC to exact-full-prefix hits only?
8. Are mixed/chunked batches scoped to `tt_data_parallel=1` only, never mixed under DP?
9. Should the #47488 **input-chunking** half (runner slicing `[num_computed:num_computed+num_new]`)
   be a follow-on PR on the same branch, or a separate PR?
