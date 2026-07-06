# Batched canvas decode — design note (#47557, Agent C)

**Goal.** Today the DiffusionGemma serving path runs a **single active canvas** (batch=1): the
denoise loop processes one 256-token canvas per block. #47557 = process **B>1 canvases through the
denoise loop simultaneously**. The Gemma-4 backbone is already batch-capable on the leading batch
dim (standard prefill/decode carries `[B, …]`); what this note pins down is the **diffusion delta**:
which DG-local ops assume batch=1, and how each generalizes to a batch dim **without cross-canvas
leakage**.

Scope (per task): **model-side batched denoise only**. The vLLM-level multi-request wiring
(per-request paged KV caches + block tables) is #47488 and is explicitly *out of scope*. This note
therefore separates ops that generalize purely on the batch dim (in scope) from ops that need B
independent KV caches (the #47488 boundary).

## The two flavors of "batch"

There are two distinct things B could mean:

1. **B canvases sharing one prompt prefix** (e.g. B diffusion samples of the same request, or the
   common denoise fan-out). One frozen prompt KV cache (batch=1), broadcast across B canvas rows.
   The canvases have independent init/noise and denoise independently. **This is the in-scope,
   testable model-side batched denoise** — it exercises every diffusion-delta op on a real batch dim
   and the independence property (row i must not see row j) is exactly what the correctness check
   asserts.
2. **B canvases with different prompts** (true multi-request serving). Each canvas attends to its own
   prompt prefix → B separate KV caches with per-request page tables. That is the **paged-cache
   ownership half of #47488**, not this task.

This note's prototype uses flavor (1): a single prefill, prefix KV **broadcast** to B rows, B
independent canvases. That is sufficient to generalize and prove every diffusion-delta op on the
batch dim; wiring B distinct prompt caches is a mechanical extension once #47488 lands the paged
read path.

## Op-by-op: what assumes batch=1, and how it generalizes

Legend: **✓ already batched** (carries the batch dim today), **△ generalize** (in-scope change),
**⊘ #47488** (needs B KV caches; out of scope).

### Canvas init / re-noise — `tt/generate.py`
- `make_seeded_host_canvas_init_fn`, `make_seeded_host_noise_tokens_fn`, `make_host_canvas_init_fn`,
  `make_host_noise_tokens_fn` — **✓**. All already take a `batch` param and emit `[batch, canvas_len]`
  (validators accept `[batch, canvas_len]`).
- `host_canvas_to_device` — **✓**. `[batch, canvas_len] → [batch, 1, canvas_len, 1]` via
  `view(batch, 1, canvas_len, 1)`; replicate mesh-mapper.
- `make_seeded_*_gumbel_noise_fn` / `host_gumbel_noise_to_device` — **✓**. Shapes carry
  `[batch, 1, canvas_len, vocab]`. (Argmax/RUN-first uses `gumbel=None`, so no batch concern there.)

### Sampling — `tt/sampling.py`
- `argmax_last_dim`, `gumbel_max`, `token_entropy`, `gumbel_max_with_chunked_noise`,
  `canvas_sample`, `sample_gumbel_noise*` — **✓**. Every op reduces/perturbs the **last (vocab)
  axis** and is elementwise/`slice`/`max`/`argmax` over leading dims. A `[B, 1, C, V]` logits tensor
  flows through unchanged: reductions are per `(b, c)` row, no cross-batch coupling. `_select_by_mask`
  in the chunked path is elementwise on `[B, 1, C, 1]`. Nothing keys on batch=1.

### Denoise step + decision kernels — `tt/denoise_loop.py`
- `denoise_step`, `entropy_budget_accept`, `renoise`, `make_denoise_constants`,
  `denoise_step_next_canvas`, `run_fixed_denoise_steps`, `_sample_and_argmax` — **✓**. The step math
  is written batch-generically:
  - `denoise_step` reshapes entropy `[B,1,C,1] → [B*1, C]` for the accept, then back to
    `[B,1,1,C]`, then to `sampled.shape = [B,1,C,1]`. Each reshape is a bit-exact reinterpret that
    preserves the per-`(b, c)` mapping (dim products match; leading-1 dims fold cleanly).
  - `entropy_budget_accept` does `sort/cumsum/subtract/le/scatter` on **dim=-1** of a `[B, C]`
    tensor → each row's cutoff is computed over its own C entropies, no cross-row mixing.
  - `renoise` is uint32 elementwise select on `[B,1,C,1]`.
  - `make_denoise_constants(batch=…)` already allocates `[B, C]` / `[B,1,C,1]` constants.
- **Early-halt (`denoise_block`)** — **△ (control-flow coupling).** `denoise_block`'s halt check is
  `torch.equal(argmax, h)` over the **whole `[B, C]` tensor** and a scalar `entropy.mean()`. With
  B>1 that couples the batch: the loop halts only when **all** rows are simultaneously stable+
  confident, and the mean entropy is averaged across rows. So a batched run could take a different
  number of steps than a standalone row → a different commit. **Generalization:** the trace-safe /
  serving shape is a **fixed step budget** (`run_fixed_denoise_steps`, `device_loop_denoise_block`),
  which has **no early-halt** and is per-row independent by construction. RUN-first (#48291) already
  makes early-halt a no-op (mean entropy never clears the 0.005 threshold), and the trace-safe loop
  is the shipping path, so the prototype uses the fixed-step loop. A *per-row* early-halt (a device
  mask that freezes converged rows and stops when all are frozen) is the future batched-eager form;
  it is not needed for the fixed-budget serving contract.

### Denoise forward (embed + attention + norms) — `tt/denoise_forward.py`
- `embed_canvas_tokens` — **△**. Hard guard `if canvas_tokens.shape[0] != 1: raise` **and** a
  hardcoded `reshape(canvas_hidden, (1, 1, canvas_len, H))`. `Gemma4Model.embed_tokens([B,C])`
  returns `[1, B, C, H]` (TP all-gather path); the fix reshapes to **`[B, 1, C, H]`** (a bit-exact
  reinterpret because dim0=1) and drops the guard. Byte-identical for B=1.
- `_denoise_layer_forward` / `denoise_hidden_forward` / `denoise_attention_forward` — **✓ once the
  prefix is broadcast** (below). The per-layer norms (`_chunked_norm_forward`, `_rms_norm_dram`) slice
  on **dim=-2 (seq)** and carry the batch dim; MoE router/experts, shared MLP, `ttnn.add` residuals
  are all per-`(b, seq)`.
- **Prefix-KV concat** (`tt/diffusion_attention.py::denoise_attention`) — **△**. The canonical path
  concatenates the frozen prompt `(K, V)` in front of the canvas `(K, V)` on **dim=2 (seq)**:
  `ttnn.concat([prefix_k, canvas_k], dim=2)`. The prompt prefix read from the batch=1 cache is
  `[1, kv_heads, P, d]`; the canvas is `[B, kv_heads, C, d]` → concat shape mismatch on dim0.
  **Generalization:** broadcast the shared prefix to B rows (`ttnn.repeat(prefix, [B,1,1,1])`) before
  the concat, gated on a batch mismatch (B=1 → no repeat → byte-identical). Because attention is a
  **per-batch-element** op (QK^T / softmax / PV never cross the batch axis), each row i then attends
  to `[its prompt copy ; canvas_i]` and to nothing in row j — **this is the structural reason there
  is no cross-canvas leakage.** The legacy `kv_hidden` recompute path
  (`ttnn.concat([prompt_source, normed], dim=2)`) has the same broadcast requirement; the production
  path is prefix-KV, so only that path is generalized here.
- `read_prompt_kv_cache_by_layer` / `read_prompt_kv_cache_slice` — **✓/⊘**. For the shared-prompt
  case they read the batch=1 prefix once and it is broadcast (in scope). For **distinct** prompts per
  canvas they must read B different caches → **⊘ #47488**.

### Self-conditioning — `tt/self_conditioning.py`
- `TtSelfConditioning.forward` — **✓**. RMSNorm (`_rms_norm_dram`, seq-chunked, carries batch),
  `gate/up/down` linears `[B,1,C,H] @ [1,1,H,I]`, `gelu`, `mul`, `add` — all per-`(b, seq)`.
- `soft_embedding` — **△ (broadcast).** `matmul(probs[B,1,C,V], embed_weight[1,1,V,H]) → [B,1,C,H]`,
  and the chunked form `matmul(exp_chunk[B,1,C,V'], embed_chunk[1,1,V',H])`. The tied embedding table
  is a single `[1,1,V,H]` operand shared across the batch; the matmul must **broadcast the batch=1
  weight across B**. TTNN matmul broadcasts a batch-1 rhs, so no data change is needed — but this is
  the one op where B>1 relies on matmul batch-broadcast, so it is called out and covered by the
  device test (self-cond ON). `condition(prev_logits=None)` (step 0 / encoder pass) is
  `post_norm(embeds)`, per-row. Self-cond is **zeroed on encoder passes** (prefill/commit) — unchanged.
- `DenoiseLogitsAdapter` (`denoise_forward.py`) — **✓** for the eager `__call__` path once embed +
  prefix broadcast + soft_embedding are batched: it threads `prev_logits` `[B,1,C,V]` across steps
  (one shared object; per-row content). The **trace-safe** signal buffer
  (`prepare_trace_safe_self_conditioning`) is allocated `[1,1,C,H]` and would need `[B,1,C,H]` for a
  batched *traced* loop — noted, not required for the eager prototype.

### Commit-append — `tt/generate.py`, `tt/commit_decode.py`, `tt/commit_batched.py`
- `commit_canvas_tokens` / `commit_canvas_tokens_batched` — **⊘ #47488**. Hard guard
  `if canvas_tokens.shape[0] != 1: raise`. Commit appends the committed clean-argmax K/V to **the
  model's single contiguous cache**. B canvases each have their own committed history → B caches with
  per-request page tables. That is the paged-cache ownership half of #47488, out of scope. **The
  batched-denoise correctness check therefore validates the committed argmax *tokens* emitted by the
  denoise loop** (`run_fixed_denoise_steps` → `[B, C]`), not the KV append. This is the exact model
  boundary the task draws ("do not attempt the vLLM-level multi-request wiring").
- `denoise_and_commit_block` — **△/⊘**. `_validate_committed_block_shape(..., batch_size=1)` is
  hardcoded and it calls the commit; the batched prototype bypasses it and drives the denoise loop
  directly (no commit), since commit is ⊘.

### Prefill — `tt/generate.py`
- `prefill_prompt_tokens` — **✓ (shared prompt) / ⊘ (distinct prompts).** Guard `batch==1`. The
  shared-prompt prototype prefills once (batch=1) and broadcasts; distinct per-canvas prompts →
  ⊘ #47488. `generate_blocks` / `generate_from_prompt_tokens` already thread a `batch_size` param and
  validate `[batch_size, canvas_len]` committed blocks.

## Summary table

| Op / module | Status | Generalization |
|---|---|---|
| canvas init / re-noise host fns, `host_canvas_to_device` | ✓ | already `[B, C]` |
| sampling (`argmax`, `gumbel_max`, `token_entropy`, chunked) | ✓ | last-axis reductions, per-row |
| `denoise_step`, `entropy_budget_accept`, `renoise`, constants | ✓ | reshapes preserve per-row map; sort/cumsum/scatter on dim=-1 |
| `denoise_block` early-halt | △ | use fixed-step loop (per-row, no halt); per-row halt is future |
| `embed_canvas_tokens` | △ | drop guard; reshape embed `[1,B,C,H] → [B,1,C,H]` |
| `denoise_attention` prefix-KV concat | △ | broadcast prefix `[1,…] → [B,…]`; attention per-batch ⇒ no leak |
| per-layer norms / MoE / MLP / residual | ✓ | carry batch on dim0 |
| `self_conditioning.forward` | ✓ | linears/norms per-row |
| `self_conditioning.soft_embedding` | △ | matmul broadcasts batch-1 embedding weight |
| `DenoiseLogitsAdapter.__call__` (eager) | ✓ | `prev_logits [B,1,C,V]` threaded |
| trace-safe self-cond signal buffer | △ (future) | `[1,1,C,H] → [B,1,C,H]` for batched traced loop |
| `commit_canvas_tokens*`, `denoise_and_commit_block` | ⊘ | needs B KV caches = #47488 |
| `prefill_prompt_tokens` (distinct prompts) | ⊘ | needs B caches = #47488 |

## Independence / no-leakage argument

Cross-canvas leakage could only enter through an op that mixes the **batch axis**. Enumerating the
batch-axis touch points:

1. **Attention** — QK^T, softmax, PV, per-head norm, RoPE, output proj, all-reduce are per-batch-
   element (batch is an outer loop of the matmuls; softmax normalizes over the KV axis within a row).
   The only place the prompt is *shared* is the prefix, and it is **replicated** to B identical rows
   before the concat, so row i's KV = `[prompt_i ; canvas_i]` with no reference to canvas_j.
2. **Sampling / entropy / accept / renoise** — reductions on the vocab or canvas axis, per `(b, ·)`.
   The entropy-budget sort/cumsum/scatter is on **dim=-1 (canvas)** for a `[B, C]` tensor → each
   row's accept cutoff is a function of that row's entropies only.
3. **soft_embedding matmul** — `[B,1,C,V] @ [1,1,V,H]`: the shared weight is broadcast, output row i
   depends only on `probs_i`.
4. **Early-halt** — the *only* genuine batch-coupling in the current code (whole-tensor `torch.equal`
   + scalar mean). Removed by using the fixed-step loop.

Hence with the fixed-step loop, argmax sampling, and a broadcast (not shared-in-place) prefix, the
B=2 committed argmax for row i is **bit-identical** to the standalone B=1 run of canvas i. The device
correctness check asserts exactly this.

## What the prototype builds (behind `DG_BATCH_DECODE`, default OFF)

- `tt/batched_decode.py` — `run_batched_denoise_block(...)`: prefill one shared prompt, build the
  denoise adapter, stack B host canvases → `[B,1,C,1]`, run `run_fixed_denoise_steps` → committed
  `[B, C]` argmax. Host canvas/noise stacking helpers.
- The three △ op generalizations (`embed_canvas_tokens`, `denoise_attention` prefix broadcast,
  `soft_embedding`), each **byte-identical at B=1**.
- `demo/batched_decode_smoke.py` + `tests/test_batched_denoise.py` — the device correctness check:
  B=2 committed argmax == two B=1 runs, bit-exact per row (independence).
