# DiffusionGemma optimize-perf (#47465) — work log

> Historical topology/audit log. Section-local “current” statements describe
> their measurement date. Selected-current evidence is
> `selfcond_logits_l1_e2e.json` and `selfcond_logits_l1.md`.

Stage: dg-08-optimize-perf (PERF). Device: QB2 / `bh-qbge-06` / P150x4, mesh `(1,4)`, TP=4.
Branch: `diffusion-gemma-function`. Baseline SHA at stage start: `aff8f2105d3`.
Env: venv `/home/zni/venvs/tt-diffusion-gemma` (Python 3.12, ttnn + transformers 5.12.1).
Build: `build_Release`; current CMake cache has `ENABLE_TRACY=OFF`, so synchronized component timing
and Metal trace capture/replay are the approved profiling substitutes.

**Optimization unit = the DENOISE STEP over the 256-token canvas (≤48 steps/block) + the commit.**
NOT per-token autoregressive decode. Precision policy is preserved from the context contract:
BF16 weights / activations / KV and the established BF16 ordered online-chunk self-conditioning
and terminal-entropy arithmetic.
Diffusion decisions preserved: temperature schedule 0.8→0.4, Gumbel-max, entropy-budget accept,
random-token renoise, commit = clean argmax.

---

## 1. Operation-topology audit of the measured denoise step

The measured denoise step is `DenoiseLogitsAdapter.__call__(canvas_tokens, step)` (the logits path)
followed by `denoise_loop.denoise_step(logits, ...)` (the terminal decision path). Source:
`tt/denoise_forward.py`, `tt/diffusion_attention.py`, `tt/self_conditioning.py`, `tt/denoise_loop.py`,
`tt/sampling.py`.

### 1a. Logits path (per step) — `denoise_logits_from_tokens`

| # | Stage | Op sequence (DiffusionGemma-local) | Cost class | Candidate action |
|---|---|---|---|---|
| L0 | `embed_canvas_tokens` | reshape → to_layout(ROW_MAJOR) → `embed_tokens` → reshape → to_layout(TILE) | embedding lookup [1,1,256,2816] | layout churn; minor |
| L1 | self-conditioning `condition` | `soft_embedding` (prev_logits): production vocab 262144 > 8192 → `_soft_embedding_chunked`: **32 vocab chunks** × (L1 slice/subtract/exp/denominator sum + persistent embed chunk + DRAM matmul[256,8192]@[8192,2816] + ordered adds), then gated MLP (pre_norm, gate/up/down linears, gelu, mul) + scaleless post_norm | 32 chunked matmuls + full-vocab streaming per step (denoise only); selected defaults remove embedding slices and retain the dynamic-logits/denominator chain in L1 | selected 2026-07-10: prechunk + logits/denominator-L1, exact decisions, final reviewed 18.844 t/s @48; `ttnn.split` lost and larger chunks changed commits |
| L2 | backbone `denoise_hidden_forward` ×30 layers | per layer: chunked input_layernorm → denoise_attention → chunked post_attn_norm + residual → chunked pre_ff_norm → shared_mlp → (MoE: chunked router norm + linear + softmax + topk(128,8) + scatter + sparse experts) → chunked norms + adds + layer_scalar | **dominant** — 30× MoE-A4B over 256 tokens | see 1b/1c |
| L3 | final norm | `_chunked_norm_forward(tt_model.norm)` | 8 slice+norm+concat | de-chunk |
| L4 | LM head | `_apply_lm_head(hidden, is_decode=False)` → hidden[256,2816] → vocab 262144, softcap 30 | **large DRAM-bound matmul** | backbone knob; vocab-shard |

### 1b. `_chunked_norm_forward` (norms over the 256 canvas) — `tt/denoise_forward.py:187`, `tt/self_conditioning.py:177`

Every RMSNorm on the 256-token canvas is split into `256/32 = 8` slices, each width-sharded-normed,
then `ttnn.concat`-ed back. With ~6 norms/layer × 30 layers + router norms + final norm, this is
**~1600+ slice/norm/concat ops per step** where a full-256 norm would be ~1 op each. The 32-row
chunking was a decode-footprint workaround; at 256 tokens the full norm should fit.
→ **Candidate: run RMSNorm over the full 256 rows in one shot** (block_h = 256/32 = 8), measure L1 fit.

### 1c. `denoise_attention` per layer — `tt/diffusion_attention.py:314`

| Step | Op | Cost class | Candidate |
|---|---|---|---|
| QKV | `apply_qkv_projection` (packed) + `split_qkv_heads_prefill` | backbone matmul (already packed ✓) | keep |
| per-head norm | q_norm (scale), k_norm (scale), v_norm (scaleless) | small | keep |
| RoPE | `_apply_rope_chunked`: **manual** RoPE built from slice(x1)/slice(x2)/mul(-1)/concat/mul(cos)/mul(sin)/add, chunked 32 tokens × 1 head | **many small DRAM ops** (8 seq-chunks × heads) | **replace with fused/prefill RoPE over full 256** |
| prefix KV | `to_memory_config` ×2 + concat prompt K/V in front of canvas K/V | movement | keep (prefix-style, needed) |
| move to DRAM | `to_memory_config` q/k/v → DRAM | movement | fold into producer layout |
| SDPA | `_sdpa_q_chunked` → **ttnn SDPA misses L1 static-CB by <1 tile → `_manual_gqa_attention`**: per kv-head slice + clone/concat (GQA expand) + permute + matmul(scores) + softmax + matmul(out) + concat | **staged fallback = many ops, single-core-ish** | **fix SDPA L1 CB clash to use the real kernel** (kill the fallback) |
| out | `concat_heads` + `apply_output_projection` + `apply_allreduce` (CCL) | matmul + all-reduce | keep; audit CCL |

### 1d. Terminal decision path (per step) — `denoise_loop.denoise_step` + `sampling.py`

| Step | Op sequence | Axis | Cost class | Candidate |
|---|---|---|---|---|
| gumbel_max | `temperature_scale`(mul over vocab) → add(noise) → `argmax` over vocab | vocab 262144 | full-vocab mul+add+argmax | reuse temp-scaled z; RUN-first argmax path |
| clean argmax | `ttnn.argmax(logits, dim=-1)` (again) | vocab 262144 | **2nd full-vocab argmax** | **dedup vs gumbel argmax when noise≈0 / share z** |
| token_entropy | `max` → `subtract`(DRAM) → `exp`(DRAM) → `sum`(DRAM) → `log` → `multiply`(DRAM) → `sum`(DRAM) → `div`(DRAM) → `subtract` | vocab 262144 | **~4 full-vocab [1,1,256,262144] intermediates in DRAM** (shifted/exp_shifted/expected_terms) → heavy DRAM read+write | fuse / reduce intermediates / placement |
| entropy_budget_accept | reshape → `sort` → `cumsum` → `subtract` → `ttnn.full`(budget) → `le` → `typecast` → `zeros_like` → `scatter` | **256 canvas axis** | small; but `full`+`zeros_like` allocated every call | **candidate table (net-new): sort/cumsum/scatter placement + config; persistent budget/zeros** |
| renoise | `typecast` → `ttnn.full`(ones) → `subtract` → `mul` → `mul` → `add` | 256×1 | small; allocs every call | persistent ones; fuse select |

**Repeated-input observations:** `logits` is consumed by 3 separate full-vocab passes
(gumbel temperature-scale, clean argmax, entropy temperature-scale) — the temperature scaling and
the argmax are duplicated. `ttnn.full`/`ttnn.zeros_like` constants are re-allocated every step in
both `entropy_budget_accept` and `renoise` (candidates for persistent/preallocated device tensors,
important for a trace-safe fixed-step loop where constants should be captured once).

**Collectives:** one `apply_allreduce` per layer (attention output projection, TP=4). No
per-step LM-head all-gather in the current path (logits stay device-resident, decision tensors
read back small). CCL audit: the all-reduce is inside the shared backbone op; driven, not edited.

### 1e. Trace-safety gaps (current loop is EAGER)

`denoise_block` (`tt/denoise_loop.py:179`) reads back host tensors **every step**
(`_ids_to_torch`, `_entropy_to_torch`, `_accept_to_torch`, `_ids_to_torch(canvas)`) and does a
host-side `torch.equal` early-halt with Python `return`. That is a per-step host sync + data-dependent
control flow — not trace-safe and not acceptable optimized evidence. The fixed ≤48-step trace-safe
scheme (on-device cutoff mask, tensor-valued scatter indices, no host readback of the halt, warmed
program cache) is built in section 3.

---

## 2. Measurements

### 2a. Eager per-op device times at real shapes (logits `[1,1,256,262144]` bf16, QB2 (1,4))

Artifact: `artifacts/diag_sampling_ops.log`.

| Op | device ms (eager, 3-rep avg) | note |
|---|---|---|
| `argmax` over vocab (TILE input) | **1239.7** | single-core `ArgMaxDeviceOperation` on TILE (the doc: TILE last-dim argmax = single-core) |
| `max` over vocab (value only) | 3.5 | multi-core reduction — fast |
| temperature_scale (`multiply`) over vocab | 0.83 | |
| `token_entropy` (full, 3 full-vocab DRAM intermediates) | 13.5 | |
| `sort` over 256 axis | 0.128 | net-new accept chain — tiny |
| `cumsum` over 256 axis | 0.205 | |
| `entropy_budget_accept` (sort+cumsum+sub+le+scatter) over 256 | **0.566** | the whole net-new sort/cumsum/scatter chain is ~1.3% of the step |

**Finding:** the net-new entropy-accept chain (sort/cumsum/scatter over 256) is negligible. The
terminal-path bottleneck is `ttnn.argmax` over the 262144 vocab at **1240 ms**, called **twice per
step** (gumbel sample + clean commit argmax) → ~2.48 s/step from argmax alone.

### 2b. argmax-over-vocab alternatives (logits `[1,1,256,262144]` bf16)

Artifact: `artifacts/diag_argmax_alt.log`. Root cause (ttnn doc): last-dim argmax is **single-core
on TILE input** but **multi-core on ROW_MAJOR input**; output is always UINT32 ROW_MAJOR.

| method | device ms | vs torch fp32 | vs TILE argmax |
|---|---|---|---|
| `argmax` TILE (baseline) | 1239.7 | (bf16) | — |
| **`argmax` via ROW_MAJOR** | **14.4** | 94.53% | **exact match (1.0)** — bit-identical, 86× faster |
| `topk` k=1 (TILE) | 42.5 | 94.53% | equivalent |
| `topk` k=32 (TILE) | 42.5 | 94.53% | equivalent |

(94.53% vs fp32 torch is bf16 tie-breaking on random data — identical across all methods; ROW_MAJOR
argmax is a bit-exact replacement for TILE argmax.) **Chosen: convert the argmax input to ROW_MAJOR**
(`sampling.argmax_last_dim`), wired into `gumbel_max` (both paths) and `denoise_step` clean argmax.

### 2c. Trace-safety fix

The original terminal chain is **not trace-capturable**: `entropy_budget_accept` (`ttnn.full` budget,
`ttnn.zeros_like`) and `renoise` (`ttnn.full` ones) issue host→device writes, which raise
`TT_FATAL: Writes are not supported during trace capture`. Fix: `make_denoise_constants(...)`
preallocates `budget_t`/`accept_zeros`/`renoise_ones` once outside the trace; `entropy_budget_accept`,
`renoise`, `denoise_step`, and the new `run_fixed_denoise_steps` accept them (eager path still
auto-allocates). This makes the whole terminal decision path trace-safe.

### 2d. Traced terminal decision step (per denoise step, 1-capture / 20-replay, warmed)

Artifacts: `artifacts/bench_*.log`. Path = `denoise_step` (gumbel/argmax over vocab + entropy +
sort/cumsum/scatter accept + renoise) at real `[1,1,256,262144]` shape, TP=4 replicated.

| variant (argmax RUN-first path) | traced ms/step | note |
|---|---|---|
| terminal chain, TILE argmax + per-call ttnn.full (ORIGINAL) | **untraceable** | ttnn.full write TT_FATAL; eager ≈ 2×1240 + 13.5 ≈ 2494 ms |
| baseline (ROW_MAJOR argmax + preallocated consts) | **43.06** | ~58× faster than the eager original, and now trace-safe |
| **share_z** (reuse z across gumbel/clean argmax + entropy) | **42.30** | **kept** — drops one redundant full-vocab temperature-scale (~0.8 ms) |
| chunked_entropy (L1-resident vocab-chunked entropy) | 45.43 | **rejected** — extra slice/op launches (2 passes × 8 chunks) outweigh the DRAM-write saving |
| baseline, **gumbel** path (materialized full-vocab noise) | 44.26 | gumbel add over vocab ≈ +1.2 ms vs argmax path |

**Terminal-path reconciliation:** traced 43.06 ms ≈ eager op sum (2×14.4 ROW_MAJOR argmax + 13.5
entropy + ~0.8 temp-scale + 0.57 accept ≈ 43.7 ms). The remaining cost is the two ROW_MAJOR argmaxes
(28.8 ms) + entropy (13.5 ms); the sort/cumsum/scatter accept chain is ~1.3%. For the RUN-first
argmax path (`gumbel_noise=None`) the sampled token equals the clean argmax, so the two argmaxes are
in principle dedupable to one — a further ~14 ms/step is available and noted as remaining work; it is
deferred to avoid touching the general Gumbel decision path late in the stage.

### 2e. sort/cumsum/scatter placement candidate table (net-new accept chain over the 256 axis)

The entropy-accept chain operates on a tiny `[1, 256]` tensor (1 tile row × 8 tile cols), so the
knob is DRAM-vs-L1 placement + preallocated constants, not matmul geometry. Artifact:
`artifacts/diag_accept_placement.log`.

| candidate | ms (eager, 20-rep) | decision |
|---|---|---|
| accept, DRAM entropy, per-call `ttnn.full`/`zeros_like` (original) | 0.416 | not trace-safe (write during capture) |
| accept, DRAM entropy, **preallocated constants** | **0.321** | trace-safe + 23% faster → **chosen** |
| accept, L1 entropy, preallocated constants | 0.345 | rejected (slower than DRAM — tensor is one tile row, L1 reshard adds cost) |
| `sort` over 256 (component) | 0.114 | — |
| `cumsum` over 256 (component) | 0.195 | — |

The whole accept chain is ~0.32 ms of a ~43 ms step; placement is not the bottleneck, but the
preallocated-constants form is both faster and required for trace-safety, so it is the chosen config.
Artifact: `artifacts/diag_accept_placement.log`.

### 2f. Trace-safe fixed-step loop verification

`run_fixed_denoise_steps` (device-only, no host readback, fixed ≤48-step count, preallocated
constants) captured as a Metal trace and replayed vs the eager run with a canvas-dependent synthetic
`logits_fn`: **committed argmax 100.00% identical** (`TRACE_SAFE_OK`). This proves the accepted canvas
from step N feeds step N+1 entirely on device with no host readback of the cutoff. Artifact:
`artifacts/verify_trace_safe_loop.log`.

### 2g. Correctness floor (unchanged decisions)

Existing device sampling-decision tests pass against the optimized `sampling.py` / `denoise_loop.py`:
`test_device_entropy_accept.py` + `test_tt_sampling.py` → **18 passed** (`DG_RUN_DEVICE=1`, QB2).
The ROW_MAJOR argmax is bit-identical to the TILE argmax (verified exact), so the Gumbel-max /
entropy-budget-accept / renoise decisions are preserved. Artifact: `artifacts/pytest_sampling.log`.


## 3. Roofline for the diffusion path

Unlike autoregressive decode (one new token, incremental single-token KV read), each denoise step
**re-reads the model weights and recomputes over the full 256-token canvas against the frozen
prefix** — there is no incremental single-token KV read. So the per-step floor is set by
**weight traffic**, not KV traffic.

Weight-byte model (bf16, 2 bytes; from `config.json`: H=2816, 30 layers, qh16/kv8/hd256, shared
intermediate 2112, 128 experts × moe_inter 704, vocab 262144):

| component | params | bf16 GB total | GB/chip (TP=4) |
|---|---|---|---|
| per layer, dense (attn QKV+O, shared MLP, router, norms) | 52.8M | — | — |
| per layer, experts (128 × gate/up/down) | 761.3M | — | — |
| **model total** | **25.16B** | **50.3** | **12.58** (matches 13.24 GiB build) |

**Per-step weight traffic** (dense read once/step + activated experts):

| scenario | GB/chip/step | @256 GB/s·chip | @512 | @1024 |
|---|---|---|---|---|
| all 128 experts activated (256-token canvas saturates routing) | 12.58 | 49.1 ms | 24.6 ms | 12.3 ms |
| 8/128 experts (per-token active, perfect reuse — lower bound) | 1.88 | 7.3 ms | 3.7 ms | 1.8 ms |
| dense only (no experts) | 1.16 | 4.5 ms | 2.3 ms | 1.1 ms |

For a 256-token canvas with top-8 routing there are 256×8 = 2048 expert-token assignments over 128
experts, so by coupon-collector the unique-expert count saturates near 128 → the **all-experts row is
the realistic per-step weight floor** (the routed `sparse_matmul` reads roughly the whole expert bank
per step). Peak per-chip GDDR6 BW is stated as a range pending calibration from the measured
per-layer time (§3a); the achieved fraction is derived below.

## 3a. Reduced-layer measured per-step + full-generation projection (eager warmed)

Real 26B DiffusionGemma checkpoint, reduced layers, canvas 256, short prompt (cache_len=32), QB2
(1,4) TP=4, fabric FABRIC_1D. Artifacts: `artifacts/prof_denoise_L{1,2,4}_eager.log`. RUN-first argmax
sampler (`gumbel_noise=None`), with the ROW_MAJOR argmax fix + preallocated trace-safe constants.

| num_layers | prefill TTFT (ms) | eager denoise ms/step |
|---|---|---|
| 1 | 74.7 | 193.14 |
| 2 | 351.2* | 331.48 |
| 4 | 384.8 | 607.21 |

Linear fit (excellent): **per-layer = (607.21−193.14)/3 = 138.0 ms/layer**; **fixed overhead
(embed + self-cond + LM-head over 262144 + final norm + terminal sampling) = 55.1 ms** (check: L2 =
193.1+138.0 = 331.2 ≈ 331.5). (*L2 TTFT includes a first-touch alloc; L1/L4 TTFT scale ~linearly.)

**Full 30-layer projection (eager, post-argmax-fix):**
- per denoise step ≈ 55.1 + 30 × 138.0 = **4196 ms/step**.
- per block (fixed trace-safe 48 steps + commit) ≈ 48 × 4196 ms + commit ≈ **~201 s/block** (matches
  the plan's 272.8 s full-canvas one-block RUN once build/prefill/commit are added).
- tokens-per-block = 256; blocks-per-second ≈ 1 / (per-block seconds) ≈ **0.005 blocks/s** at this
  eager op-dispatch-bound baseline (NOT `1000/mean_tpot_ms`).

**Impact of the landed argmax fix at full depth:** the terminal sampling had 2× 1240 ms single-core
argmax = +2480 ms/step before the fix. Post-fix per-step ≈ 4196 ms; pre-fix ≈ 4196 + 2480 = **6676
ms/step**. The argmax fix cuts the full-depth step by **~37%** (and the terminal sub-path by ~58×).

## 3b. Roofline reconciliation

Measured post-fix per-step ≈ 4196 ms vs roofline weight floor 24.6 ms (@512 GB/s·chip, all experts)
/ 49.1 ms (@256): the measured step is **~85–170× the bandwidth roofline**. The denoise per-layer
(138 ms) is therefore **dispatch / op-count bound, not DRAM-bandwidth bound**. Roofline says a
weight-traffic-bound step should be ~25–50 ms; the 138 ms/layer comes from the DiffusionGemma-local
per-layer op *count*, not bytes:
- `_chunked_norm_forward`: every canvas RMSNorm split into 8 slices + 8 sharded-norms + concat →
  ~6 norms/layer → ~90 slice/norm/concat ops/layer.
- `_apply_rope_chunked`: manual RoPE as slice/concat/mul/add chunked 32-tokens × per-head → dozens of
  tiny DRAM ops per q/k per layer.
- `_manual_gqa_attention`: the real ttnn SDPA misses L1 by <1 tile and falls back to a staged
  per-kv-head matmul→softmax→matmul (single-core-ish) → many ops replacing one flash-attention op.

**Conclusion:** the terminal path (argmax) was the first-order fix and is done (traced, ~58×). The
dominant remaining cost is the per-layer denoise backbone op-count; the roofline shows ~3–5× headroom
to bandwidth if op count is collapsed (de-chunk norms/RoPE, fix the SDPA L1 clash to use the real
flash kernel and kill the staged GQA fallback). These are DiffusionGemma-local (`tt/diffusion_attention.py`,
`tt/denoise_forward.py`) and are the next optimization target; §4 records what was tried.

## 4. Advice tried / rejected, and limitations

### 4a. Applied (kept)
- **ROW_MAJOR argmax** (`sampling.argmax_last_dim`): 1240 ms → 14.4 ms/op (86×), bit-identical to
  TILE argmax; 2×/step → ~2480 ms/step saved at full depth. Wired into `gumbel_max` + `denoise_step`.
- **Preallocated accept/renoise constants** (`make_denoise_constants`): removes per-step
  `ttnn.full`/`zeros_like` host writes → makes the terminal chain trace-safe **and** 23% faster on
  the accept chain (0.416 → 0.321 ms).
- **share_z** (reuse temperature-scaled z across gumbel/clean argmax + entropy): 43.06 → 42.30 ms.
- **Trace-safe fixed-step loop** (`run_fixed_denoise_steps`): device canvas feedback, no host
  readback, fixed ≤48-step count; traced replay == eager (100%).

### 4b. Tried and rejected (before/after)
- **chunked_entropy** (vocab-chunked, L1-resident entropy): 45.43 ms > 43.06 ms baseline — the extra
  slice/op launches (2 passes × 8 chunks) outweigh the DRAM-write saving. Rejected.
- **L1 placement of the accept entropy tensor**: 0.345 ms > 0.321 ms (DRAM) — resharding a one-tile-row
  tensor to L1 costs more than it saves. Rejected; DRAM + preallocated constants kept.

### 4c. Identified next target (dominant remaining cost) — per-layer denoise backbone
Measured 138 ms/layer is ~170× the ~0.8 ms/layer bandwidth roofline → dispatch/op-count bound. The
DiffusionGemma-local candidates (all in `tt/denoise_forward.py` / `tt/diffusion_attention.py`):
1. **De-chunk `_chunked_norm_forward`** (8 slice+norm+concat per canvas norm → one full-256 norm).
2. **Replace `_apply_rope_chunked`** (manual per-32-token/per-head RoPE) with a single fused RoPE.
3. **Fix the denoise SDPA L1 static-CB clash** so the real flash SDPA runs and the staged
   `_manual_gqa_attention` fallback (single-core-ish per-kv-head matmuls) is eliminated.
These are the next optimization pass; §4d records the attempt status.

### 4e. Environment limitation — tt-perf-report / Tracy unavailable (hardware-profiler-limited)
The shared `build_Release` was compiled with `ENABLE_TRACY:BOOL=OFF` (confirmed in
`build_Release/CMakeCache.txt`), so `TT_METAL_DEVICE_PROFILER=1` raises `TT_FATAL: ... requires a
Tracy-enabled build` and `python -m tracy` / `tt-perf-report` op-CSV collection is unavailable.
Enabling it needs a full tt-metal + ttnn rebuild (`build_metal.sh` without `--disable-profiler`,
~30–60 min) that would replace the shared venv's `_ttnn` bindings — a disruptive, hard-to-reverse
action on a shared QB2. Per the `optimize` skill's `hardware-profiler-limited` provision, the op-level
breakdown is provided instead as **synchronized per-op device-time tables** (§2a/§2b/§2e) plus the
reduced-layer per-layer sweep (§3a) and the traced e2e terminal microbench (§2d) — these give the same
"which ops dominate + before/after" signal a `tt-perf-report` table would. `perf_summary.json` records
device-time fields with this named limitation.

## 3c. TRACED full denoise step + per-block + full-generation (supersedes §3a eager)

The full reduced-layer denoise step (real 26B backbone adapter + terminal decision path) **does
capture and replay as a Metal trace** (my earlier SDPA-fallback-blocks-trace concern was wrong — the
staged GQA fallback ops replay fine once the accept/renoise constants are preallocated). Warmed traced
replay, canvas 256, QB2 (1,4) TP=4. Artifacts: `artifacts/prof_denoise_L{1,2,4}_traced.log`.

| num_layers | TRACED ms/step | eager ms/step | commit ms (256 decodes) | TTFT ms |
|---|---|---|---|---|
| 1 | 186.79 | 193.27 | 1248.3 | 76.0 |
| 2 | 324.12 | 334.05 | 2306.7 | 95.6 |
| 4 | 599.45 | 607.37 | 4375.0 | 141.7 |

- **Traced per-layer = (599.45−186.79)/3 = 137.55 ms/layer**; **fixed overhead = 49.24 ms** (check L2:
  186.79+137.55 = 324.34 ≈ 324.12). Traced ≈ eager (~3% gain) → the denoise path is **op-cost bound,
  not dispatch-gap bound**; tracing removes little because the per-layer ops themselves are the cost.
- **Full 30-layer TRACED per denoise step = 49.24 + 30×137.55 = 4175.7 ms.**
- **Commit per-layer = (4375−1248)/3 = 1042 ms/layer**, fixed 206 ms → full-30L commit ≈ 206 + 30×1042
  = **31.5 s/block** (256 sequential single-token decode-appends).
- **Per block** (fixed trace-safe 48 steps + commit) = 48 × 4175.7 ms + 31.5 s = **231.9 s/block**;
  **tokens/block = 256**; **blocks/s ≈ 0.0043** (NOT `1000/mean_tpot_ms`).
- **Full generation (1 block, 256 tokens)** = TTFT(30L, 32-tok prompt) ≈ 0.71 s + Σ48 steps 200.4 s +
  commit 31.5 s ≈ **232.6 s**.
- **Argmax-fix impact at full depth (traced):** pre-fix would add 2×~1233 ms single-core argmax to the
  fixed overhead → ≈ 6642 ms/step; post-fix 4175.7 ms/step → **~37% faster per step** (~58× on the
  terminal sub-path). The dominant remaining cost is the per-layer backbone (137.55 ms/layer × 30 =
  4126 ms = 98.8% of the traced step).

**Correction to §4e:** the tt-perf-report/Tracy op *CSV* is unavailable (ENABLE_TRACY=OFF), but Metal
**trace capture/replay works**, so the per-step / per-block / full-generation numbers above are from a
**traced measured path**, not eager. Only the op-level attribution table (which would come from Tracy)
is substituted by synchronized per-op device times (§2a/§2b) + the per-layer sweep (§3c).

## 5. Runtime fallback audit + optimize checklist evidence

### 5a. Runtime fallback audit (clean)
- The only runtime fallback in the denoise path is the pre-existing **denoise SDPA → staged GQA**
  (`_manual_gqa_attention`), triggered by the L1 static-CB clash and emitting an expected caught
  `TT_THROW` (RUN-first, documented in the plan). My optimizations add **no** new fallback and remove
  none; the traced full step replays the same fallback ops deterministically.
- **No host-argmax / full-logits-readback fallback**: logits stay on device; `argmax_last_dim`
  returns device tokens. The trace-safe `run_fixed_denoise_steps` reads back nothing per step (the
  eager `denoise_block` host trajectory harness is a separate correctness/oracle path, not the
  optimized loop).
- The ROW_MAJOR-argmax change is a layout conversion (not a fallback) and is bit-identical to TILE.

### 5b. Optimize checklist (applicable items) — evidence
- Operation-topology audit completed — §1.
- Denoise loop trace-safe with device canvas feedback, no host readback of the cutoff — §2f, verified 100%.
- Explicit `memory_config` for the terminal ops (DRAM entropy intermediates; preallocated accept/renoise
  constants; ROW_MAJOR argmax input) — §2b/§2e. Backbone matmul/CCL program configs are the shared
  backbone's (driven via its knobs, not edited — gate preserved).
- Functional correctness floor preserved — `test_device_entropy_accept` + `test_tt_sampling` 18 passed
  (§2g); ROW_MAJOR argmax bit-identical to TILE.
- Performance accounting reconciled (roofline vs traced device time vs e2e) — §3/§3c.
- Warmed traced replay measured (per-step / per-block / full-generation) — §3c; terminal ranking traced — §2d.
- Precision policy preserved (BF16 weights/activations/KV and established BF16 ordered online-chunk
  self-conditioning/terminal entropy arithmetic) — unchanged.
- `perf_summary.json` written — `doc/optimize_perf/perf_summary.json`.
- tt-perf-report / Tracy op CSV — hardware-profiler-limited (ENABLE_TRACY=OFF), §4e; substituted
  synchronized per-op device times.
- Watcher-clean — `artifacts/watcher_pytest.log` + `artifacts/watcher_verify.log` (TT_METAL_WATCHER=10,
  separate from any profiler run).
- Batch: batch-1 single-user latency is the target (context contract batch_size_default=1); the
  optimized terminal path and `make_denoise_constants` are parameterized by `batch`/`canvas_len` (not
  hard-coded to 1), so batch>1 shape handling is preserved.

### 5c. gemma4 isolation gate
`git diff <baseline aff8f2105d3>..HEAD -- models/demos/gemma4/` is empty (only
`models/experimental/diffusion_gemma/` changed). The four decode-footprint files remain clean vs main.

## 6. Reusable up-front denoise capture (2026-07-22)

Implemented the default-OFF `DG_UPFRONT_CAPTURE` path entirely under DiffusionGemma:

- model startup runs a one-token mock prefill + one block decode and retains the captured adapter;
- request prefill overwrites cache `[0:cache_len]`, resets the mutable fixed-span reader even when
  the new prompt shrinks, and refreshes reveal-mask + canvas-RoPE contents in place;
- request reset detaches a borrowed persistent adapter without releasing traces or trace-baked
  buffers; the wrapper destructor calls the idempotent best-effort release method before inherited
  model/mesh teardown;
- startup rejects missing reveal mask, missing traced serving, zero/invalid trace region, missing or
  unaligned explicit `p_max`, and lazy capture;
- block-capacity validation enforces `prompt + generated <= p_max` before device execution.

CPU:

```text
pytest -q tests/test_upfront_capture.py tests/test_serving_block_contract.py -k 'not test_device'
23 passed, 1 skipped, 4 deselected
```

Device mechanics/decision gate, full 30 layers, K=2, argmax, `p_max=1024`, 1 GiB trace region:

```text
DG_SPARSE_MOE=1 DG_SPARSE_MOE_TUNED=1 DG_DEDUP_ARGMAX=1 \
DG_RUN_DEVICE=1 DG_TRACE_REGION_SIZE=1073741824 \
DG_DENOISE_REVEAL_PMAX=1024 DG_UPFRONT_NUM_LAYERS=full DG_UPFRONT_STEPS=2 \
pytest --timeout=600 -q \
  tests/test_upfront_capture.py::test_device_upfront_trace_reuses_one_capture_across_different_prompt_lengths
1 passed in 29.48s
```

Results: aligned prompt spans 32→320→32 reused `MeshTraceId(0..1)` with
`capture_events=1`; A repeated exactly and differed from B. The 320-token real prefill is longer
than the mock's `32 + 256` span and therefore directly proves it overwrites the committed mock KV.
Up-front, per-request reveal trace, and eager all committed SHA256
`924ae03b6111734d8ab1d2d4c88ec6a7da5ba6612c50b2f0e3c27d0511980e0f`.
The bit-exact test was then rerun at the same full-depth K=2 configuration with
`DG_UPFRONT_GUMBEL_MODE=chunked`; it passed in 50.39 s with the same three-way digest. The persisted
`upfront_bit_exactness.json` is this production-Gumbel run; the reuse artifact remains the argmax
stale-input control. Both runs construct the real `DiffusionGemmaForCausalLM` wrapper (including its
startup guard resolution) rather than a method-only shell. Eager and per-request controls run first;
the model-lifetime capture runs last so its release remains terminal.

Full qualitative stale-state gate, 30 layers, K=48, tuned argmax, `p_max=1024`, 10 GiB trace region:

```text
DG_SPARSE_MOE=1 DG_SPARSE_MOE_TUNED=1 DG_DEDUP_ARGMAX=1 \
DG_RUN_DEVICE=1 DG_TRACE_REGION_SIZE=10737418240 \
DG_DENOISE_REVEAL_PMAX=1024 DG_UPFRONT_NUM_LAYERS=full DG_UPFRONT_STEPS=48 \
pytest --timeout=900 -q \
  tests/test_upfront_capture.py::test_device_upfront_multi_request_smoke_has_no_stale_cross_request_state
1 passed in 154.02s
```

The persistent controller captured 48 traces once, replayed four blocks (startup + A + B + A),
and executed 192 traces without recapture. A round-tripped exactly; B differed. Decoded outputs were
coherent (`"你好！ How can I help you today?"` and a prompt-correct black-hole explanation).
The artifact records the checkpoint chat template, rendered token ids, and output hashes.

The prompt-B output was then compared against the existing per-request reveal-mask trace in a
fresh process (the release API is terminal shutdown, so the control deliberately reopens the mesh):

```text
DG_UPFRONT_BASELINE_CONTROL=1 DG_RUN_DEVICE=1 DG_TRACE_REGION_SIZE=10737418240 \
DG_DENOISE_REVEAL_PMAX=1024 DG_UPFRONT_NUM_LAYERS=full DG_UPFRONT_STEPS=48 \
pytest --timeout=900 -q \
  tests/test_upfront_capture.py::test_device_per_request_prompt_b_matches_upfront_qualitative_artifact
1 passed in 90.68s
```

The baseline committed SHA256 exactly matched the up-front prompt-B digest
`82dac3229b72134447b6ad8f1571a6520215c9e0642b07c8a5a715d3706075b4`.

An initial full-depth K=48 run without sparse-MoE tuning hit pytest's 300 s timeout while
synchronizing the first 48-trace replay after a successful capture. Cleanup released all traces and
closed the mesh; `tt-smi -ls --local` then showed all four p300c devices healthy. The tuned rerun
above passed. This was a workload-timeout control, not a trace correctness failure.

An attempted *test-only* same-process sequence of terminal
`release_persistent_capture()` followed by a new per-request K=48 capture stalled in
`AllBroadcastDeviceOperation` during the post-release control prefill. Live triage showed all four
devices in the broadcast, with the broadcast writer waiting on its semaphore; ARC, Ethernet, L1,
watcher, and lightweight-assert checks passed. Evidence:
`triage/upfront_control_hang_{tt-triage,summary}.txt`. The process group was stopped, devices were
reset, and a `(1,4)` mesh open/close smoke passed. This does not affect serving: the release method is
now explicitly documented as terminal shutdown immediately before mesh close. The fresh-process
baseline above passed and exactly matched prompt B.

The compact triage summary reports script execution status, not an idle-device integrity verdict.
The raw capture's binary-integrity/NoC mismatch rows were sampled while the broadcast was actively
running; they are not being claimed as post-recovery checks. Reset + mesh open/close is the
post-recovery health evidence.

### Independent review

A fresh stage-review pass after wiring destructor cleanup returned `clean-pass` with no required
work. It rechecked the live diff, production shutdown call site, request detachment, long-prompt
mock-span overwrite, three-way chunked-Gumbel exactness, K=48 A→B→A evidence, prompt-B baseline
control, triage anomaly, and the DiffusionGemma-local isolation gate.

## 7. Up-front + early-halt GPQA hang repair (2026-07-22)

The reported second-request hang was reproduced with the real
`r1_gpqa_diamond` task. Request 0 completed at K=13, then request 1 stopped
between `prefill_device_begin` and `prefill_device_end`. Exact live triage
(`triage/upfront_earlyhalt_gpqa_hang_tt-triage.txt`) placed all four devices in
the causal-prefill broadcast writer waiting on its semaphore.

Hypothesis controls:

1. Direct wrapper A→B with early halt passed (K=17, K=19), refuting a universal
   early-controller request-reset bug.
2. Replaying all skipped windows after a K=13 halt still hung on the next GPQA
   prefill, refuting incomplete trace/CCL advancement.
3. Fixed-K up-front capture hung on the same second GPQA prefill, refuting an
   early-halt-specific root cause.
4. Warming the 160-token prefill shape before denoise capture made both fixed-K
   GPQA requests pass.

Root cause: vLLM's compile-only phase previously deferred without compiling
real prefill shapes, so the first real 160-token prompt compiled/allocated a new
prefill program after 48 denoise traces were active. That violated trace address
stability and corrupted CCL state; the following prefill stalled in
`AllBroadcast`.

The fix:

- honors vLLM's two-phase warmup (`enable_trace=False` compiles configured
  prefill lengths; `enable_trace=True` captures denoise);
- makes DiffusionGemma decode warmup a no-op because the block-denoise path is
  captured by prefill warmup;
- requires `DG_UPFRONT_PREFILL_WARMUP_LENS` in vLLM mode;
- rejects unseen aligned runtime prefill lengths before device execution;
- leaves boundary markers around device prefill and adapter rebind.

Full 30-layer validation used
`DG_UPFRONT_PREFILL_WARMUP_LENS=160,192,256,384,480` and ran eight sequential
real GPQA-Diamond requests with traced early halt. All eight passed and released;
realized K was 10–43, TTFT 5.38–18.63 s, and `capture_events` stayed 48 with no
recapture. The lm-eval exact-match score was 0 because this lifecycle run capped
generation at one 256-token block, truncating reasoning before answer
extraction. See `upfront_earlyhalt_gpqa_20260722.{json,md}`.

A fresh independent stage review returned `clean-pass` with no required work.
It verified the exact-hang triage, fixed-K/drain refutations, two-phase warmup
ordering, unseen-length admission guard, eight-request GPQA evidence, and
DiffusionGemma-local isolation.
