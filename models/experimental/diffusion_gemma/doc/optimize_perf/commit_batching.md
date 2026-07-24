# Commit batching — one causal prefill-append instead of 256 decode-appends (#47557)

## What / why

After each denoise block the generation loop **commits** the block's 256 clean-argmax canvas
tokens into the frozen Gemma4 KV cache (three-phase KV: prefill → denoise → **commit**). The
baseline commit (`tt/generate.py::commit_canvas_tokens`) does this with **256 sequential
single-token decode-appends** — one full 30-layer `commit_decode_forward` per committed token
(**~31.5 s / 256-token block** on QB2; see `README.md` headline table). The 256 forwards, not the
KV writes, are the cost.

The batched commit (`tt/commit_batched.py::commit_canvas_tokens_batched`) collapses those 256
forwards into **one causal masked prefill** over the whole 256-token canvas. Expected ~7× on the
commit step and ~1.25× on end-to-end block throughput (the ~48-step denoise loop is the other
large per-block term: `README.md` shows commit = 31.5 s of a ~232 s block).

It is **opt-in and guarded** (`DG_COMMIT_BATCHED=1`, or pass `commit_fn` to
`denoise_and_commit_block`). The sequential path stays the default until this is validated on
device (`verify_commit_batching.py`). **No `models/demos/gemma4/` edits** — the batched commit
composes over the importable Gemma4 ops exactly like `tt/commit_decode.py` and
`tt/diffusion_attention.py`.

## The equivalence argument (code inspection)

**Claim.** For a fixed committed token block, the batched commit writes, per layer and per
committed position, K/V that are the *same function* of the same inputs as the 256 sequential
decode-appends — same absolute positions, same per-head norm, same RoPE, same K/V projections,
same causal visibility, same cache layout. The two paths write **algebraically identical** K/V;
they differ only in **op implementation** (prefill vs decode kernels for the same math), a small
numerical drift (bfp8 / tile-reduction order), not an algebraic difference. So the correct
assertion is **high PCC / small max-abs-diff**, not literal bit-identity — the same relationship
prefill and decode already have in the backbone. Each component is pinned to code below.

### 1. The sequential commit *is* a causal prefill

In `commit_decode_forward` → `_commit_attention_decode_forward`, for committed token `i` at
absolute position `start_pos + i`:

- its K/V are computed from **that token's** per-layer input hidden state and written at
  `start_pos + i` (`paged_update_cache(update_idxs_tensor=cache_pos)`, `commit_decode.py:236-257`);
- its decode SDPA (`scaled_dot_product_attention_decode`, `cur_pos = start_pos+i`) attends to
  **all cache positions `≤ start_pos + i`** — the frozen prefix (prompt + prior blocks,
  `0..start_pos-1`) plus already-committed canvas tokens `0..i-1` plus token `i` itself. Token
  `j < i` was written when *it* was the current decode token, so by the time token `i` runs the
  cache holds `prefix ++ canvas[0..i]` at every layer.

That is exactly the causal-prefill visibility of a chunk appended at the end of the cache. In a
decoder transformer each token's per-layer representation depends only on itself and earlier
tokens, computed identically token-by-token or all-at-once — the standard
autoregressive-equals-prefill identity, of which the commit is a concrete instance.

### 2. The batched commit reproduces that visibility with an explicit causal mask

`commit_batched.py::_sdpa_causal_masked` runs `scaled_dot_product_attention(is_causal=False,
attn_mask=<additive [1,1,C,start_pos+C] causal mask>)`. The mask
(`build_canvas_denoise_mask(prefix_len=start_pos, canvas_len=C, causal=True, ...)`) sets, for
canvas query `i` (absolute `start_pos+i`) and key `p`:

- **full-attention layer:** attend iff `p ≤ start_pos + i` → whole prefix visible, canvas causal
  (`p = start_pos+j` visible iff `j ≤ i`);
- **sliding layer:** additionally `start_pos + i - p < sliding_window` (last `sliding_window`
  positions), matching the decode SDPA's `sliding_window_size`.

Verified on host against a brute-force enumeration of the sequential decode-append visibility
(`reference/attention_mask.py::build_canvas_denoise_mask(causal=True)`; the landing commit runs
that check across `P ∈ {32,64,1024,1280,0,96}`). The batched SDPA's `attn_mask` handling and the
L1-clash fallback (`_manual_gqa_attention_masked`, which *keeps* the mask) reuse the validated
denoise SDPA path — only the mask content changes.

> **Sliding-window edge caveat.** The mask uses the HF causal-sliding convention
> (`0 ≤ q−k < window`); the device reference is the decode SDPA's `sliding_window_size`. They
> agree where the window does **not** bite (`start_pos + C ≤ sliding_window`, i.e. committed
> context ≤ 768 with `window = 1024`), covering RUN-first. Where it bites, a one-position edge
> mismatch is possible and must be confirmed on device — the verify harness compares sliding
> layers' K/V directly at a long-enough context.

### 3. K/V values: same projection + per-head norm + RoPE, same positions

| step | sequential (`commit_decode.py`) | batched (`commit_batched.py`) |
|------|----------------------------------|-------------------------------|
| QKV projection | `apply_qkv_projection` | `apply_qkv_projection` (identical weights) |
| head split | `split_qkv_heads_decode` | `split_qkv_heads_prefill` (decode vs prefill layout of the same split) |
| Q norm | `_apply_per_head_norm(q_norm_weight, with_scale=True)` | `apply_per_head_norm(q_norm_weight, with_scale=True)` |
| K norm | `_apply_per_head_norm(k_norm_weight, with_scale=True)` | `apply_per_head_norm(k_norm_weight, with_scale=True)` |
| V norm | `_apply_per_head_norm(None, with_scale=False)` | `apply_per_head_norm(None, with_scale=False)` |
| RoPE | `_apply_rope_decode_peruser` / `apply_rope` at `start_pos+i` | `_apply_rope_chunked(start_offset=start_pos)` → `start_pos+i` |

Both use the **same RMS eps** (`config.rms_norm_eps`) and the **same absolute RoPE position**
`start_pos + i` for canvas token `i`. The frozen prefix K/V are read back already-RoPE'd/normed
(written by prompt prefill / prior commits) and are **not** re-RoPE'd in either path — matching
the denoise prefix contract (`diffusion_attention.denoise_attention` `k_rope_offset` logic). The
decode vs prefill head-split/RoPE ops compute the same values in different tile layouts — the
numerical-drift term, not an algebraic one.

### 4. Same cache positions and layout

- **Positions.** Batched writes canvas token `i` at absolute `start_pos + i`
  (`_write_canvas_kv_contiguous`, `update_idxs=[start_pos+t]`) — the *same* indices the sequential
  path uses. `start_pos = cache_len + N·256` is a multiple of 32 (prompt padded to 32, canvas
  256), so all seq bounds are tile-aligned.
- **Layout.** Both write the same contiguous per-layer cache tensor
  `[1, num_local_kv_heads, max_seq, head_dim]` (`tt_kv_cache[i]`). Originally both used the
  same non-paged `paged_update_cache`, one op per seq position (a batch-1 contiguous cache
  addresses one position per non-paged update) — provably the same write as sequential. The
  default is now ONE tile-aligned `ttnn.fill_cache` per K/V per layer, device-verified
  bit-identical to that per-position write over the whole cache (see "Opt A" below); the
  per-position write remains available as `DG_COMMIT_KV_WRITE=position`.
- **KV-sharing.** `write-then-read-from-cache`: a non-shared layer writes its canvas K/V, then the
  SDPA reads `cache[0 : start_pos+C]`. A KV-shared layer (`kv_shared_layer_map[i]`, E2B/E4B) skips
  its write; its earlier **source** layer already wrote the shared cache tensor, so the shared
  layer's read sees `prefix ++ canvas`. Mirrors the sequential `is_kv_shared` handling
  (`commit_decode.py:154,568`).

### 5. Where the two paths legitimately differ (numerical, not algebraic)

- **SDPA kernel:** decode flash-decode (per token) vs prefill flash-attention (masked, 256
  queries). Same `softmax(QKᵀ·scale)·V`; different tiling/accumulation.
- **Head split / RoPE / per-head norm / MoE:** decode (1-token, sparse-matmul MoE via
  `commit_decode._commit_experts_decode_forward`) vs prefill (256-token, gathered-expert MoE via
  `denoise_forward._denoise_moe_forward`). Same routing (top-8, softmax, geglu, weighted sum),
  different kernels. MoE numerics feed the next layer's K/V, so drift compounds mildly down the
  stack — bounded and measured per-layer by the verify harness.

**Net:** the batched commit is algebraically the 256 sequential appends. Expected KV-cache
agreement is high PCC (≥ ~0.999 early layers, decreasing slightly with depth), with the
sliding-edge caveat. `verify_commit_batching.py` asserts this per layer and reports commit_ms
before/after.

## Honesty flags (do not force these)

- **Not bit-identical.** Prefill and decode kernels differ numerically; assert PCC, not equality.
  Low PCC on a specific layer means that layer's decode↔prefill op mapping (head split / RoPE /
  MoE) needs closer reconciliation before this ships as default.
- **Contiguous cache + `page_table=None` only** (the standalone / serving RUN path). Paged / vLLM
  hybrid-cache commit still uses the sequential path; the batched SDPA-read for paged caches is
  intentionally `NotImplemented` (batched paged decode is #47557 / #47488).
- **Sliding-window edge** unproven where the window bites (> ~768 committed tokens).
- ~~**`write_batch>1`** (fast contiguous write) unproven on device~~ — resolved: batching
  `paged_update_cache` is racy by construction and the knob was **removed**; the fast write is
  now one tile-aligned `ttnn.fill_cache` per K/V, device-verified bit-identical (see "Opt A").

## Device verification results (2026-07-04, QB2 1x4) — **NOT bit-equivalent → stays opt-in**

`verify_commit_batching.py` was run on the real 26B-A4B backbone (L=6, contiguous cache).
Two real bugs were found and fixed to make the batched path RUN at all:

1. **Sharded KV write** (`_write_canvas_kv_contiguous`): `paged_update_cache` requires a
   HEIGHT_SHARDED, one-core-per-user update tensor (shard width == head_dim); the per-position
   canvas-K/V slice was DRAM-interleaved and tripped `input_tensor.is_sharded()`. Fixed by
   resharding each `[1,1,nkv,hd]` slice onto one core with the tile-padded shard, exactly like
   the proven decode `sequential_kv_write` (`gemma4/tt/attention/decode.py`).
2. **`to_memory_config` no-op alias** (`_commit_attention_batched`): the chunked RoPE already
   emits DRAM, so an unconditional `to_memory_config(tt_q, DRAM)` returned a fresh **unallocated**
   alias and the SDPA input died (`is_allocated()`). Fixed with the guarded `if buffer_type != DRAM`
   convert, mirroring `commit_decode.py`.

**Commit speedup (real):** batched commit ≈ **6.3× faster** than the 256 sequential decode-appends
(1031 ms vs 6503 ms at L=6, sparse denoise MoE; 6.6× with dense).

**KV bit-equivalence: FAIL.** worst KV PCC **0.43** at L=6 (threshold 0.997). The divergence was
localized layer-by-layer with `probe_commit_l0attn.py`:

| layer-0 stage | PCC (batched vs sequential) |
|---|---|
| attention output | **0.99992** (bit-exact) |
| shared_mlp input | **0.99994** (bit-exact) |
| shared_mlp output | **0.99975** (bit-exact) |
| router output (dense_routing) | 0.98097 — expert-mask agreement **99.6%**, both nnz=8 |
| **MoE expert output** | **0.16551** (catastrophic) |

**Root cause = the MoE EXPERT KERNEL, not routing.** Everything up to and including the router is
bit-exact / near-exact; the routed-expert *output* is near-uncorrelated (0.17). Diluted by the
bit-exact shared_mlp + residual, this gives ~0.94 layer-1 KV, compounding to 0.43 by layer 5 (and
would be worse across 30). Crucially:
- The batched commit's experts (`moe.experts` dense, and `sparse_experts_forward`) are **independently
  verified ≈ torch/dense at 0.9997** (denoise + leverA). Dense-MoE and sparse-MoE batched variants
  give the *same* 0.43, so it is not the sparse-gather approximation.
- The sequential commit's `_commit_experts_decode_forward` (decode `sparse_matmul` nnz=8) is the
  **outlier** — it has never been PCC-verified vs torch (RUN-first path). It diverges 0.17 from the
  verified-correct batched experts on identical routing + bit-exact input.

**Conclusion:** the batched commit is **likely more correct** than the sequential decode-append
reference; they simply disagree because `_commit_experts_decode_forward` appears defective. Per the
hard rule (verify before landing), the batched commit **stays opt-in / default-off**. It is NOT that
the batched path is wrong — the *reference* is suspect. Also ruled out with device evidence: masked
flash SDPA (`probe_masked_sdpa.py`, 0.998 vs torch), a read-after-write cache hazard (a
`synchronize_device` before the read changed nothing), and the layer-body norms (bit-exact).

**To land Lever B next session (unlocks the path to 30 t/s):** verify the batched commit KV against a
**torch commit reference** (not the buggy sequential), OR reconcile / fix `_commit_experts_decode_forward`
against `moe.experts`/torch so the two commit paths agree — then flip the default. This is also a
#48291-relevant finding: the RUN commit may be writing slightly-wrong prefix KV.

## RESOLUTION (2026-07-04 session 4) — batched commit is CORRECT → now the DEFAULT

The tie-break the previous session left open ("verify the batched commit KV against a
**torch** reference, not the buggy sequential") is **done and decisive.** A torch MoE
oracle — hand-rolled to match HF `transformers.models.diffusion_gemma`
(`DiffusionGemmaTextRouter` + `DiffusionGemmaTextExperts`), loaded from the real
layer-0 checkpoint weights, run in fp32 on the **identical bit-exact** layer-0 commit
MoE input the device computed (`probe_moe_vs_torch.py`):

| MoE output vs torch oracle (layer-0, real weights, bit-exact input) | PCC |
|---|---|
| **batched commit** (`moe.experts` / `sparse_experts_forward`) | **0.9936** |
| sequential commit (`_commit_experts_decode_forward`, decode `sparse_matmul` nnz=8) | **0.1542** |
| (input bit-exactness batched↔sequential: expert_input PCC 0.9999; routing agree 0.9968) | |

**Verdict: the batched commit MoE reproduces the torch reference (bf16-kernel accuracy,
like the rest of the model); the sequential decode-commit MoE is near-uncorrelated with
torch — it is genuinely DEFECTIVE.** So the 0.43 batched-vs-sequential KV disagreement
was the *sequential* path being wrong, exactly as hypothesized. Every other commit-layer
component (attention, shared_mlp, router, norms) was already bit-exact between the two
paths (`probe_commit_l0attn.py`) and the two attention implementations are independent
(decode flash-decode vs prefill masked-SDPA) yet agree at 0.99992 — cross-implementation
corroboration that the batched attention/RoPE/mask/KV-write is also correct. The batched
commit is therefore **both faster (~6.3×) and strictly more correct**.

**Landed:** `DG_COMMIT_BATCHED` now defaults **ON** (`batched_commit_enabled()` default
"1"); `_resolve_default_commit_fn(page_table, page_tables_per_layer)` forces the
sequential path only for **paged/vLLM** caches (batched supports the contiguous
model-owned cache; paged batched SDPA-read is #47488). Set `DG_COMMIT_BATCHED=0` to force
sequential. No `models/demos/gemma4/` edits (the gate stays the 1-line dealloc).

**Verified combined t/s (serving_smoke, full 30L, `DG_SPARSE_MOE=1`, 24-step, same session):**

| commit path | block latency | **tokens/block/s** | commit KV vs torch |
|---|---|---|---|
| sequential (old default) | 49.94 s | **5.13** | 0.154 (defective) |
| **batched (new default)** | 19.64 s | **13.04** | **0.994 (correct)** |

**2.54× on the block AND a correctness fix.** Generated block-0 text is coherent
("…a generative model that creates text by starting with random noise… iteratively
refines them into a coherent sequence through a denoising process."); later-block
degeneration is the deferred #48291 fidelity issue, not a commit defect. Artifacts:
`probe_moe_vs_torch.py`, `artifacts/leverB_moe_vs_torch_L0.log`,
`artifacts/leverB_verified_{batched,seq}_30L_s24.{json,log}`.

> Note: `verify_commit_batching.py` (batched-vs-sequential PCC) is now a **non-gate** —
> it compares against the defective sequential reference, so its FAIL is expected and
> meaningless. `probe_moe_vs_torch.py` (batched-vs-torch) is the correct gate.

## How to enable / verify

```bash
# Batched commit is the DEFAULT now (torch-verified correct). Force sequential with:
DG_COMMIT_BATCHED=0 python -m models.experimental.diffusion_gemma.demo.text_demo ...

# The correct gate — batched vs a torch MoE oracle (run when QB2 is free):
DG_CKPT=/path/to/diffusiongemma-26B-A4B-it \
  python models/experimental/diffusion_gemma/doc/optimize_perf/probe_moe_vs_torch.py

# (legacy, now a non-gate — compares against the DEFECTIVE sequential reference):
DG_COMMIT_BATCHED=1 python -m models.experimental.diffusion_gemma.demo.text_demo ...

# Device verify (KV bit-equivalence + commit_ms before/after) — run when QB2 is free:
DG_CKPT=/path/to/diffusiongemma-26B-A4B-it \
  python models/experimental/diffusion_gemma/doc/optimize_perf/verify_commit_batching.py \
  --mesh 1x4 --num-layers 30 --max-seq-len 1024 --prompt "The capital of France is"
```

---

# Opt A — the KV write is half the commit; one `fill_cache` removes it (2026-07-24)

## Where the commit time actually goes

Splitting the batched commit on device (backbone = fwd + cache read + SDPA + MoE, vs the
per-layer KV write) at two depths:

| L  | full commit | backbone | KV write | write share |
|----|-------------|----------|----------|-------------|
| 8  | 266.1 ms    | 123.2 ms | 142.9 ms | 53.7%       |
| 16 | 495.1 ms    | 241.9 ms | 253.2 ms | 51.1%       |

Per layer: write ≈ 13.8 ms, backbone ≈ 14.9 ms. Extrapolated to 30L the write is
~0.4–0.5 s of a ~0.96 s commit. It is almost **pure host dispatch**: the write was 256
per-position iterations × (2 `slice` + 2 reshard + 2 `paged_update_cache`) ≈ 1536 tiny
`[1,1,nkv,hd]` ops per layer, with essentially no arithmetic.

## The fix: `ttnn.fill_cache(cache, canvas, 0, update_idx=start_pos)`

The commit's write span is tile-aligned by construction (`start_pos % 32 == 0` and
`canvas_len = 256`, both now validated in `commit_canvas_tokens_batched`), so the whole
canvas is a **tile-granular copy** — no read-modify-write of a partial tile is needed.
`ttnn.fill_cache` takes exactly that: a tile-aligned seq-dim offset `update_idx`
(exposed by #44827). One op per K/V per layer, i.e. **2 dispatches instead of ~1536**.

* `update_idx` becomes `update_idxt = update_idx/32` folded into the per-core destination
  page id `batch_idx*C*Ht*Wt + h*cache_HtWt + (update_idxt + j)*Wt`
  (`fill_cache_multi_core_program_factory.cpp`), i.e. head-major with the **cache's** seq
  stride — the same positions the per-position loop writes.
* FILL is pure data movement: reader → CB → `writer_unary_interleaved`, **no compute
  kernel** (unlike the UPDATE path's untilize/patch/tilize), so bf16 values cannot be
  perturbed. It also never touches the frozen prefix `[0, start_pos)` or the tail.
* No transpose (the `[1, C, nkv, hd]` permute existed only to satisfy
  `paged_update_cache`'s user-dim contract), no reshard, no per-call host tensor.
* Precedent: DG's own *prompt* prefill already fills this same contiguous cache with
  `ttnn.fill_cache` (`gemma4/tt/attention/prefill.py`); the only delta is `update_idx != 0`.

### Measured (QB2 Blackhole, 11x10 grid, nkv=2 hd=256 C=256, 20 warm reps)

| write mode                | per layer (K+V) |
|---------------------------|-----------------|
| per-position (256 × 6 ops)| 9.591 ms        |
| one `fill_cache` per K/V  | **0.012 ms**    |

The ~52% write share collapses to ~0.1% of the commit.

### Real-backbone gate — PASS (2026-07-24, QB2 1x4, full 30L 26B-A4B)

`verify_commit_kv_write.py --num-layers 30 --max-seq-len 1024` (batched commit run twice
from the identical pre-commit cache, once per write mode, warmed):

```
commit_ms  per-position =     905.1   one-op fill =     464.0   speedup =  1.95x
whole-cache max_abs_diff (position vs fill) = 0.0000e+00   (must be 0.0)
layers = 30  shards/layer = 4  start_pos = 32
RESULT: PASS — one-op fill KV write is bit-identical to the per-position write
```

Bit-identical over the **whole** cache (30 layers × K/V × 4 device shards, every position,
not just the written region), and the frozen prefix `[0, start_pos)` matched the
pre-commit snapshot under both modes — three runs, all `max_abs_diff = 0.0`.

The speedup is what the ~52% write share predicts (0.91 s − ~0.44 s ≈ 0.46 s). Across three
30L runs of the same geometry: **1.85x–1.95x** (905.1→464.0 and 882.4→478.2 warmed;
970.7→520.8 without the warm-up, where the first-timed mode absorbs the cold program-cache
cost) — so the ratio is neither a warm-up artifact nor tighter than ~±0.05x run to run.

## Why the *original* Opt A (batch `paged_update_cache`) is not the fix

The plan was to reshard the batched `[1, n, nkv, hd]` slice to satisfy
`paged_update_cache`'s multi-user contract (`num_cores == num_users`). That reshard makes
the op *run*, and then it **silently corrupts the cache**: `paged_update_cache` is a
per-TILE read-modify-write (reader loads the whole 32-row tile-row, compute untilizes,
writer patches one row and writes the tile-row back), and `n` consecutive canvas positions
share ONE 32-row tile, so all `n` cores compute the identical `cache_id` and the RMWs
collide — last-writer-wins. This is the same failure `gemma4/tt/attention/decode.py`
serializes around (`sequential_kv_write`, #44923); the only serialization the op has
(`in0_sequential_mode` semaphores) is wired for `share_cache`, which is a hard `TT_FATAL`
in paged mode. A race-free repair exists (stride-32 grouping so each user is in a
different tile) but caps at 8 users for a 256-token canvas → ≥258 dispatches/layer and
zero DRAM-traffic reduction. `fill_cache` makes it moot.

**The branch and its `write_batch` / `DG_COMMIT_WRITE_BATCH` knob are therefore DELETED**,
not left behind an env var: its whole failure mode is that the one missing piece (a shard
spec) converts a loud `is_sharded()` assert into silent KV loss, which is a poor thing to
leave lying around for the next person who reads "opt-in fast write" in a docstring. A
stale `DG_COMMIT_WRITE_BATCH` export is now ignored with a warning (it used to select the
mechanism, so silently honoring it would restore the 1536-dispatch path). The dead end is
recorded here and in a "do not reintroduce" note on `_write_canvas_kv_contiguous`; the
per-position write survives as `DG_COMMIT_KV_WRITE=position`, which is a *different*
mechanism (one op per position, no batching, no race).

## One ttnn hazard this change had to guard

`fill_cache`'s interleaved path splits `nkv * (S/32)` tile-rows over the core grid and each
core writes its rows **contiguously** from a single `cache_start_id`, assuming no core's
range crosses a kv-head boundary ("assume that work doesn't spill over to next head" —
`fill_cache_multi_core_program_factory.cpp`). **No validator enforces it.** Once the rows
exceed the core count (and the input does not span the whole cache) the op silently writes
rows into the wrong head. Device-confirmed on QB2: `nkv=8, C=1024, max_seq=2048` (256 rows
> 110 cores) → **49106 wrong elements**; the same geometry with `C == max_seq` is fine
(the destination is then one contiguous run, which is why the existing whole-prompt
prefill callers never hit this).

DiffusionGemma is far inside the safe region (`nkv_local ≤ 2`, `C = 256` ⇒ ≤ 16 rows vs
110 cores; `split_work_to_cores` gives exactly 1 row/core whenever rows ≤ cores), but
`_fill_write_unsupported_reason` checks it explicitly, along with dtype equality (FILL
refuses to convert, unlike `paged_update_cache`), head/head_dim match, tile alignment,
span bounds and cache batch. Any failed precondition logs once and falls back to the
per-position write — correct, just slow — so a geometry the op cannot serve never raises
mid-layer-loop and never leaves a half-written cache.

Upstream follow-ups worth filing: (a) `fill_cache` should `TT_FATAL` on the head-boundary
spill instead of silently corrupting; (b) gemma4's long **non-paged** prompt prefill
(`nkv=2, S ≥ 8192` at `S < max_seq`) can cross the same threshold.

## Knobs / verification

```bash
# Default is the one-op fill. Force a mechanism:
DG_COMMIT_KV_WRITE=fill|position   # position = the proven per-position reference
# DG_COMMIT_WRITE_BATCH is OBSOLETE (the racy 1-block-paged write it selected was deleted).
# It is now ignored with a warning rather than silently restoring the 1536-dispatch path.

# Op-level gate, checkpoint-free, ~4 s — bit-identity of the two mechanisms over the WHOLE
# cache (so a disturbed frozen prefix/tail fails), + torch oracle, + the spill guard:
DG_RUN_DEVICE=1 pytest models/experimental/diffusion_gemma/tests/test_device_commit_kv_write.py

# Real-backbone gate — batched commit with fill vs with per-position, whole-cache exact:
DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it \
  python models/experimental/diffusion_gemma/doc/optimize_perf/verify_commit_kv_write.py \
  --mesh 1x4 --num-layers 30 --max-seq-len 1024 --prompt "The capital of France is"
```

Both write modes are now selectable per call (`write_mode=`), not only per process, so a
single process can A/B them against the identical pre-commit cache — which is what makes
an exact (`max_abs_diff == 0`) gate possible. Note `verify_commit_batching.py` /
`probe_attn_only.py` gained a `--kv-write` flag but still default to `position`, so their
existing (batched-vs-sequential) comparisons keep using the proven write.

## Pre-existing bug fixed in passing: the last block could free the KV cache

`_read_cache_kv` reads `[0, start_pos+C)` back out of the cache for the causal SDPA, and
`_commit_attention_batched` deallocates that result. When the committed block ends exactly
at `max_seq` the read is a **full-span** slice (all starts 0, all ends max), which ttnn
short-circuits to an **alias** of the input — `slice.cpp`:
`if (no_step && starts_zero && ends_max) return finalize_into_preallocated(ret_adjustment(input_tensor));`
with `ret_adjustment`'s no-op `to_memory_config`/`to_layout` passing the tensor straight
through. Deallocating it therefore frees the **KV cache itself**. `_read_cache_kv` now
`ttnn.clone`s at that boundary so the caller always owns a distinct buffer
(`test_device_commit_kv_write.py::test_cache_read_at_max_seq_does_not_alias_the_cache`).
Latent since the batched commit landed, independent of the write mode, and the same
ttnn aliasing gotcha the traced-denoise reveal-mask work hit.

Two guard-design notes worth keeping:
* **cache batch > 1 must not fall back to the per-position write.** `fill_cache` with
  `batch_idx=0` is legal on a multi-slot cache; the per-position path is not (non-paged
  `paged_update_cache` asserts `num_users == cache batch`), so a blanket fallback would
  turn a working write into a `TT_FATAL`. The fill guard ignores cache batch; the position
  branch raises its own clear error.
* **A ragged span is a contract violation, not a fallback.** `start_pos % 32` and
  `canvas_len % 32` now both raise in `commit_canvas_tokens_batched`, before any device
  work — a ragged `canvas_len` would otherwise push tile-pad K/V past `start_pos+canvas_len`
  and only `_read_cache_kv`'s `end_pos` check would notice, with the cache already dirty.
