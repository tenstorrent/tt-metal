# DFlash for Gemma4-31B — understanding, reference status, and ttnn design

Branch: `ign/gemma4_31B_dflash_reference`. Covers what DFlash is and how it works, the
validated torch/HF reference already in this repo, and the design/plan for a ttnn port —
no ttnn code has been written yet.

## 1. What DFlash is

DFlash ("Block Diffusion for Flash Speculative Decoding", [arXiv:2602.06036](https://arxiv.org/abs/2602.06036),
[github.com/z-lab/dflash](https://github.com/z-lab/dflash)) is a speculative-decoding drafter.
Like any speculative decoder, it drafts cheaply and verifies against the real target model —
the speedup comes from checking several candidate tokens in one target forward pass instead
of one real decode step per token (the general mechanism, decode-vs-verify-chunk code paths,
and causal-masking argument for *why* one bigger pass costs about the same as one small pass
on this hardware, are documented in `models/demos/blackhole/qwen36/docs/mtp_batched_verify_investigation.md`
for the Qwen3.6/GDN case — the reasoning transfers directly here).

**What's different from Qwen3.6's MTP head (a normal autoregressive drafter):**

| | Qwen3.6 MTP | DFlash |
|---|---|---|
| Drafts | one token per forward pass, chained K times | a whole `block_size`-token block in **one** forward pass |
| Draft input | the real previous token's embedding | a placeholder "mask token" embedding, identical at every draft position |
| Context from target | only the final hidden state | hidden states tapped from **6 different depths** of the target |

### The core mechanism: guess a whole block at once, using borrowed hints

The target model's internal hidden states, well before its final layer, already carry a rough
sense of where the sequence is heading — not just the immediate next token. DFlash taps that
signal at 6 points inside the target's depth, and uses it as "context" that its own small
drafter reads to fill in `block_size` blank positions **simultaneously**, rather than
guessing one and feeding it back in to guess the next.

Mechanically, every draft position's attention keys/values are the **concatenation** of the
context's K/V and the draft block's own K/V:

```python
# Qwen3DFlashAttention.forward (dflash.py)
k_ctx, v_ctx     = k_proj(target_hidden), v_proj(target_hidden)   # real context
k_noise, v_noise = k_proj(hidden_states), v_proj(hidden_states)   # the block's own positions
k = cat([k_ctx, k_noise]);  v = cat([v_ctx, v_noise])
```

So every draft position's query attends over both the real context and every other draft
position in the same call — that single mechanism is what makes one-shot block drafting
possible at all.

**How much the block positions actually "see" each other** is controlled per layer by a
causal mask, driven by `config.layer_types`. For the Gemma4-31B-DFlash checkpoint
(`layer_types = [sliding, sliding, sliding, sliding, full]`, 5 layers):

- Layers 1-4 (`sliding_attention`, `is_causal=True`): position *i* sees only positions `≤ i`
  in the block (plus the full context, always) — information drifts forward only.
- Layer 5 (`full_attention`, `is_causal=False`): every position sees every other position,
  both directions — one round of full mixing right before the final logits.

What gets shared across these layers is each position's **continuous hidden vector**, not a
decided word — no token id exists until the very last step (`compute_logits`, after the final
norm). Nothing here is iterative in the image-diffusion sense either: the drafter's `forward()`
runs exactly **once** per block (5 layers of internal mixing), not many refinement steps.

### Verify — same idea as any speculative decoder, at block scale

`dflash_generate` (in the vendored `dflash.py`) feeds `[real tokens so far] + [block_size draft
tokens]` through the **real** target model in one causally-masked pass, compares the target's
own argmax at each position against the draft, and keeps the longest matching prefix plus one
"bonus" token from the target's own output at the point of first disagreement — so every
iteration advances by at least 1 real, guaranteed-correct token, and by up to `block_size + 1`
when every draft is confirmed.

### Is it "the same attention" as Gemma4's own?

No — different code entirely (`Qwen3DFlashAttention`, not Gemma4's own attention class), and
the same generic classes back the *unrelated* Kimi-K2.6-DFlash checkpoint in
`models/demos/deepseek_v3_d_p/reference/dflash_prefill/` with different config values. Only
the drafter's **layer-type pattern** echoes Gemma4's own local/global alternation — everything
else (dimensions, RoPE, the K=V constraint, the context-concat mechanism) is independent:

| parameter | gemma4-31b (target) | dflash drafter |
|---|---|---|
| layer pattern | 5 sliding + 1 full, ×10 across 60 layers | 4 sliding + 1 full, once (5 layers total) |
| head_dim | 256 local / 512 global | 128, uniform |
| heads / kv-heads | 32 / 16 | 64 / 8 |
| rope theta | 10000 sliding / 1e6 full | 1e6, uniform |
| K = V constraint | yes | no (separate k_proj/v_proj) |
| sliding_window | 1024 | 2048 |
| K/V source | its own past tokens only | context K/V ⊕ block K/V (the DFlash-specific trick) |

### This checkpoint's exact config (`z-lab/gemma-4-31B-it-DFlash/config.json`)

```
architecture: DFlashDraftModel (the plain variant, not DFlash2DraftModel)
model_type: qwen3          # the drafter's OWN 5 layers reuse Qwen3-style blocks
hidden_size: 5376          # matches Gemma4-31B's own hidden_size (context feature width)
num_hidden_layers: 5
num_attention_heads: 64, num_key_value_heads: 8, head_dim: 128
block_size: 16             # tokens drafted per iteration
mask_token_id: 4
target_layer_ids: [1, 12, 23, 35, 46, 57]   # taps into the 60-layer target
num_target_layers: 60
final_logit_softcapping: 30.0
sliding_window: 2048, layer_types: [sliding, sliding, sliding, sliding, full]
rope_theta: 1000000, rope_scaling: null
vocab_size: 262144, tie_word_embeddings: true
```

Target: [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) (60 layers,
hidden 5376, `attention_k_eq_v: true`, per-layer-type head_dim/RoPE, multimodal —
`Gemma4ForConditionalGeneration`).

## 2. Reference implementation — done and validated

`models/demos/gemma4/reference/dflash/`:

- `dflash.py` — verbatim copy of `github.com/z-lab/dflash @ 07ebd93db9f472af339b644bb70221ad8428328a`,
  `dflash/model.py` (MIT). Same generic classes used for Kimi-K2.6-DFlash elsewhere in this repo.
- `dflash_e2e_check.py` — loads both real checkpoints, runs the real `dflash_generate` loop on
  a real prompt, reports generated text + acceptance stats. No synthetic weights anywhere.
- `README.md` — required environment setup (see below) and the verified run output.

**Required environment:** a separate venv pinned to `transformers==5.15.0` exactly — the
shared `tt-metal/python_env`'s `5.12.1` is missing `DynamicCache.activate_past_recording()`,
which `dflash_generate` calls unconditionally. Documented rather than patched around, to keep
the vendored reference faithful to upstream:

```bash
uv venv --python 3.10 /path/to/dflash_venv
uv pip install --python /path/to/dflash_venv/bin/python3 torch torchvision "transformers==5.15.0" accelerate
source /path/to/dflash_venv/bin/activate
python3 -m models.demos.gemma4.reference.dflash.dflash_e2e_check \
    --max-new-tokens 32 --prompt "Write a one-line Python function that reverses a string."
```

**Verified result** (CPU, no GPU in this environment — this is a correctness check, not a
throughput benchmark; 31B on CPU is slow, ~35s for 16 tokens):

```
GENERATED: def reverse_string(s): return s[::-1]
block_size=16  iterations=2  tokens/iteration: [10, 5]  avg=7.50
```

Correct output, and 7.50 tokens/iteration lines up with the checkpoint's own published
HumanEval-style acceptance-length numbers (~8.0) — real evidence the wiring (context taps,
sliding-window masking, softcap, block accept) is faithful to the real checkpoint.

Commit `d1506661e42` on this branch.

## 3. ttnn port — design

No ttnn code exists yet for this. Survey of what's reusable vs. greenfield, from the current
`models/demos/gemma4/tt/` and the sibling Kimi-K2.6-DFlash port in `deepseek_v3_d_p/`:

| Piece | Status |
|---|---|
| Generic GQA attention, config-driven head_dim/heads, sliding-window support | **Reusable as-is** — `models/tt_transformers/tt/attention.py::Attention`; already proven with non-Gemma dims (`qwen3_vl` drives the same class) |
| Standard paged KV cache | **Confirmed, no new risk** — Gemma4 is a plain transformer; none of the GDN-style recurrent-state fragility from the Qwen3.6/MTP work applies here |
| tanh logit softcap | **Reusable verbatim** — `gemma4/tt/model.py:1275-1279`, 3 elementwise ops (`mul`/`tanh`/`mul`) |
| Multi-layer hidden-state tapping (6 taps → fc → context) | **Partially there.** `layer_probe` hook exists (`gemma4/tt/model.py`, called after every decoder layer) but is explicitly documented "UNTRACED RUNS ONLY" / debug-only — a production version needs the on-device tap()/additive-FC-decomposition pattern from `deepseek_v3_d_p/tt/dflash_prefill/tt_dflash_drafter.py`, adapted to Gemma4's dims (60 layers, hidden 5376, `target_layer_ids=[1,12,23,35,46,57]`) |
| The DFlash draft-generation forward pass itself (5 layers, 16-position block, context+own-KV concat attention) | 🔴 **Greenfield.** Doesn't exist anywhere in tt-metal yet — even the existing Kimi/DeepSeek port only has the prefill/context-KV-caching half built (the directory is literally named `dflash_prefill`); the actual block-drafting forward pass has no precedent for any model |
| Verify → accept → commit loop | **Partially reusable.** Gemma4's existing `spec_decode.py` decouples verify/accept/commit from the drafter via a `draft_fn` seam — reusable — but it threads a single final-layer "anchor hidden" between iterations, which needs generalizing to DFlash's 6-tap context representation |
| The existing EAGLE/MTP assistant (`tt/assistant/`) | **Not reusable for drafting.** Recurrent, one token at a time, KV-shared with the target (`is_kv_shared=True`, zeroed K/V weights). DFlash needs a parallel block pass with its own separate context K/V — architecturally incompatible as a drafter, though its verify loop (above) is still usable |

### Two-phase execution

```
PREFILL (once per request)                    DECODE (once per block, repeated)
───────────────────────────                   ─────────────────────────────────
Target runs its 60 layers normally.            Drafter: 5-layer parallel forward
At layers {1,12,23,35,46,57}, tap the           over 16 mask-token positions,
hidden state and accumulate into the            attention K/V = concat(cached
fc-decomposition sum (port tap()/                context K/V, this block's own K/V)
finalize() from tt_dflash_drafter.py).                │
        │                                             ▼
        ▼                                      16 draft tokens
context = hidden_norm(sum)                            │
        │                                             ▼
        ▼                                      Verify: real tokens + 16 drafts
Project once per drafter layer to               through target in one pass (reuse
k_ctx/v_ctx, cache it (fixed for the            Gemma4's existing verify machinery)
whole request — computed once, read                   │
many times per decode block)                          ▼
                                                Accept longest matching prefix +
                                                bonus token; repeat from decode
```

### Component decisions

1. **Drafter's own 5 layers** — instantiate `tt_transformers/tt/attention.py::Attention` (+
   paired MLP) with the drafter's own config (head_dim=128, 64 heads, 8 kv,
   `layer_types=[sliding,sliding,sliding,sliding,full]`, sliding_window=2048). A normal
   instantiation of the existing generic module, not a fork of it.
2. **Context-K/V-concat attention** — the one piece with no template to lift wholesale: cache
   `k_ctx`/`v_ctx` per drafter layer (computed once at prefill), and at each decode block
   compute `k_noise`/`v_noise` fresh from the 16 mask-position embeddings, concatenate, run
   ordinary SDPA. Same underlying idea as `tt_dflash_drafter.py`'s shared padded KV cache
   (context written once, block written per iteration), but the attention-forward that
   *consumes* it is new — that file only builds the cache.
3. **Context tap mechanism** — port `tap()` / `_finalize_sharded_partial()` / `export_partial()`
   / `import_partial()` from `tt_dflash_drafter.py`, swapping in Gemma4's dims and
   `target_layer_ids`. Already TP/pipeline-aware in the source file.
4. **Verify/accept/commit** — reuse `spec_decode.py`'s loop structure and
   `ttnn_verify_forward`/`ttnn_packed_verify_forward`, generalizing the anchor threaded
   between iterations from a single final hidden vector to DFlash's cached context
   (`k_ctx`/`v_ctx`, refreshed only when the context window itself grows — not every
   iteration, since it represents the prompt/committed history, not the draft block).
5. **Softcap** — reuse `model.py:1275-1279` verbatim.

### TP strategy — decided: shard, don't replicate

Two options were on the table; sharding won:

- **Memory.** The drafter is ~1.5B params (~3GB bf16). T3K's Wormhole chips have 12GB each,
  and the 31B target already occupies a large share of that once sharded 8 ways — before
  KV cache, which this branch already pushes hard for long-context serving (up to 256k per
  the main README). Replicating a full 3GB drafter on every chip competes directly with that
  budget; sharding 8 ways brings it to ~375MB/chip.
- **Precedent.** `tt_dflash_drafter.py` (Kimi-K2.6-DFlash, already in this repo) is inherently
  built TP+SP-sharded — sequence-parallel context build, tensor-parallel attention heads,
  `reduce_scatter`/`all_gather` for the FC accumulation. Following it means porting a
  validated pattern; full replication of a ~1.5B model has no precedent anywhere in this repo
  (the closest analog, the EAGLE assistant, only replicates two small linear layers, not a
  whole model).
- **CCL overhead is proportionally small.** The verify step against the 31B target already
  pays CCL cost on every one of its own layers; a handful of small collectives across 5 tiny
  drafter layers isn't the dominant cost in an iteration whose expensive part is the one
  target-model verify pass.

Consequence: TP/SP sharding is load-bearing from the first component built, not bolted on at
the end. Steps 2-3 below are built directly against T3K's mesh dimensions.

## 4. Step-by-step build plan

1. **Weight loading** — map `z-lab/gemma-4-31B-it-DFlash`'s safetensors keys to ttnn tensors
   (mirrors `load_qwen36_mtp_state_dict` from the Qwen3.6 MTP work).
2. **Context-tap + context-KV-cache module** — port from `tt_dflash_drafter.py`, TP/SP-sharded
   from the start. Validate: PCC-compare the built context sequence against the torch
   reference's `extract_context_feature`/`fc`/`hidden_norm` output, same real prompt.
3. **Single drafter layer, isolated** — one `Qwen3DFlashAttention`-equivalent layer in ttnn,
   fed a real cached context + a real 16-position mask block. Validate via layer-bisection PCC
   against the torch reference (same method used for the Qwen3.6 GDN bug hunt this session).
4. **Full 5-layer drafter forward, one block** — chain all 5; confirm the sliding/causal vs.
   full/bidirectional masking matches the torch reference exactly (not just PCC — the
   masking pattern is a hard correctness boundary, not a numerical-tolerance one).
5. **Logits + softcap + argmax** — validate token-for-token against the torch reference on a
   real prompt (exact token IDs, not just PCC — same bar as Qwen3.6's `generate_tp` 10/10 match).
6. **Wire into verify/accept/commit** — adapt `spec_decode.py`'s loop to call the new drafter
   instead of `assistant.step()`; generalize the anchor-seeding logic (point 4 above).
7. **End-to-end demo, no-MTP vs. with-MTP** — same comparison pattern as the Qwen3.6 work: real
   prompt, both modes, tok/s and coherence, on real T3K hardware.
8. **TP validation throughout** — gather TP-sharded outputs with
   `ConcatMeshToTensor`/`ConcatMesh2dToTensor` and PCC-compare against the single-device torch
   reference at every step above, rather than building single-device-first and porting later.

## 5. Status

Reference (section 2): done, validated, committed.

ttnn port (section 3-4): **Steps 1-6 done and validated on real T3K hardware**
(`models/demos/gemma4/tt/dflash/`, tests in `models/demos/gemma4/tests/dflash/`):

1. Weight loading — exact round-trip + shape checks, all 58 tensors, T3K TP=8. PASSED.
2. Context extraction (6-tap → fc → hidden_norm) — PCC 0.998 against the torch reference.
3. Single isolated drafter layer — PCC 0.9995.
4. Full 5-layer drafter chain — PCC 0.9985.
5. Final norm + shared LM head + softcap + argmax — PCC 0.998, 94% exact token match (the
   one mismatch is a confirmed genuine bf16 logit tie, not a defect).
6. Verify/accept/commit, wired against Gemma4's existing `ttnn_verify_forward` — exact
   match against the torch reference's accept/commit decision.

**Known Gemma4 kernel bug found and worked around during Step 6** (general to Gemma4, not
DFlash-specific): the first decode-style KV write immediately after a prefill whose real
token count isn't a multiple of 32 reads leftover garbage from the unused rows of the last
KV-cache tile and returns a wrong prediction; the same call one position later is correct.
Root-caused via hardware A/B testing (a tile-aligned 32-token prompt has no such garbage
and is correct on the very first call) plus static tracing (a pad-removal branch in
`attention/prefill.py` fails to fire exactly when the valid content ends at a tile
boundary). `tt/dflash/verify.py` documents this and requires callers to keep the prefill
context length tile-aligned (32) before calling `dflash_verify`; the underlying kernel bug
in `attention/prefill.py` itself is still unfixed and would be worth a separate bug report.

**Multi-iteration generation, validated on real hardware**
(`tt/dflash/generate.py::dflash_generate`, tests in `tests/dflash/test_dflash_generate.py`
and `test_dflash_generate_traced.py`): runs the full draft→verify→accept→commit loop
across multiple blocks with NO static torch reference needed at runtime -- noise-block
embeddings, RoPE cos/sin, and the drafter's context are all built from whatever tokens
the loop itself produces.

CORRECTED (an earlier version of this section, and of `generate.py`'s own module
docstring, claimed the drafter's "context" is a sliding window replaced each iteration,
citing reference `dflash/dflash.py:305` -- that was a misread of the reference, caught
via external review). The reference's drafter attends to the FULL history back to
prefill: its `past_key_values_draft` cache is created once, outside the generation loop,
and `past_key_values.update()` APPENDS every iteration's projected context-tap K/V onto
everything from every prior iteration; `_crop_to(past_key_values_draft, start)` (using
`start`'s pre-increment value) only strips that same call's transient noise-block K/V,
never the context contribution. Line 305 is only the NEW delta fed into one call, not
the full attended set. Losing this (as the TT port did) starves the drafter of context
past the first iteration or two and tanks acceptance. Fixed by making `generate.py`'s
context a FIXED-size (`max_seq_len`-wide) persistent buffer whose first `context_len`
rows are real and grow every iteration (never sliding), with the remainder masked out
via a dynamic (tensor-valued) valid-length -- see `generate.py`'s module docstring and
`attention.py`'s `build_attention_mask_static_parts`/`combine_attention_mask_dynamic`.
Confirmed exact match against the 2-iteration torch reference on real T3K hardware
(both the eager and Metal-traced paths), including a verify call at a non-tile-aligned
position inside an already-touched KV-cache tile, and a measured acceptance-rate
improvement on a longer benchmark (0.29→0.55 mean accepted/15) consistent with the
drafter now seeing real history instead of a truncated recent window.

KNOWN GAP vs. the reference (not yet closed): the TT port recomputes each layer's
context K/V projection over the FULL `max_seq_len`-wide buffer every iteration (mostly
re-deriving unchanged historical rows), rather than caching already-projected K/V and
only projecting the new delta like the reference's real incremental cache does. This is
mathematically identical (confirmed) but computationally wasteful -- measured ~2.7x
eager slowdown from widening context 16→128 rows. A true per-layer incremental KV cache
(project+write only the new delta each iteration, read the rest straight from a
persistent per-layer cache) would close most of this gap; scoped but not yet built.

KNOWN LATENT LIMITATION (separate from the above, not yet fixed, currently unobservable
at any tested scale): the attention mask's position grid is relative to each call's own
[context, noise] layout, not true absolute sequence position. Causal-among-noise and the
valid-length check are unaffected, but the sliding-window distance check for a noise
query attending a context key is off by a constant that grows with generation length --
benign while `sliding_window` (2048) comfortably exceeds `max_seq_len`, but would
mis-mask context in a longer-context deployment where `max_seq_len` approaches or
exceeds `sliding_window`. See `generate.py`'s module docstring for the fix shape.

**Step 7 (demo) done**: `demo/dflash_demo.py`, toggled via `GEMMA4_USE_DFLASH` (1 = DFlash,
0 = plain greedy decode baseline) and `GEMMA4_USE_TRACE` (1 = Metal-traced steady state,
0 = eager), both paths through the same real target model instantiation for a matched
comparison. Real T3K run, 64 generated tokens (EOS-stopping disabled via
`GEMMA4_DFLASH_STOP_AT_EOS=0` to force enough iterations to amortize trace capture --
this depresses acceptance below what a natural-length generation would show, since it
pushes past the model's actual stopping point into degenerate continuation):

| | tokens | verify iters | mean accepted | ms/token | tok/s/user |
|---|---|---|---|---|---|
| Plain          | 64 | -- | -- | 298 | 3.35 |
| DFlash, eager  | 64 | 42 | 0.55/15 | 1882 | 0.53 |
| DFlash, traced | 64 | 42 | 0.55/15 | 256 | 3.90 |

DFlash eager is well below plain decode here -- expected, since eager pays the drafter's
full (now correctly wider) per-iteration compute plus host-dispatch overhead every call,
with only ~1.55 tokens/iteration to amortize it against. Metal trace capture (see below)
removes the host-dispatch overhead and gets DFlash modestly ahead of plain decode
(1.16x); closing the gap further needs either a genuinely incremental drafter KV cache
(see above) or a higher-acceptance (non-degenerate) generation to amortize against.

**Metal trace capture, validated on real hardware** (`generate.py`'s `use_trace` param /
`_traced_steady_state`, mirroring `spec_decode.py`'s `_capture_fused_trace`/
`_generate_fused_traced`): captures the steady-state draft→verify→tap iteration as ONE
trace, replayed for every subsequent block. First attempt hit
`TT_FATAL: Writes are not supported during trace capture` -- root cause: the attention
mask was being rebuilt from scratch (`ttnn.arange`/`ones`/`zeros`/`full`, each of which
does a host->device write internally) on every captured call. Fixed by splitting mask
construction into a one-time static builder (run before capture) and a per-replay
combiner using only elementwise ops on already-built tensors
(`build_attention_mask_static_parts`/`combine_attention_mask_dynamic`). Confirmed exact
match against the torch reference with no capture-time errors and no hang.

`_tile_align_prompt` pads any non-tile-aligned prompt with repeated newline filler tokens
as a documented, logged fallback -- found to bias generation toward repeating the filler
(the padding becomes real, attended-to trailing context), so the demo's default prompt is
chosen to already tokenize to a 32-token multiple, sidestepping this entirely. The
underlying fix belongs in `attention/prefill.py`'s pad-removal condition (see verify.py),
not in caller-side prompt engineering.
