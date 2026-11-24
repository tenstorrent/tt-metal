# What can be copied from `aperezvicente-TT/glm-5.3-flash`

Survey of <https://github.com/aperezvicente-TT/glm-5.3-flash> (Apache-2.0, same copyright holder — no
licensing obstacle) against this port. Surveyed at `main` = `18457222c2c`, tag `glm53-flash-0.1.0`
= `10e8b4e0ae3`.

## What the repo is

A **working, production GLM-5.3-Flash implementation for Blackhole**: ~60k lines of Python plus
~250 hand-written TTNN kernels (dataflow/compute C++ for KDA, MLA, MoE, mHC, fabric streams,
megakernel phases). It is substantially more mature than this port: it has a megakernel, an 8-chip
layer pipeline, split-K DSA decode, speculative decode (DSpark/MTP), a prefix cache, vLLM
integration and a sampler.

Layout: `tt/` (Python modules) + `tt/kernels/` (the C++ kernels) + `mesh/` (10 `.textproto` mesh
descriptors). **No README, no tests directory, no docs** — the docstrings reference
`doc/ttnn/...` design notes that are not in the repo.

It expects to be imported as `models.experimental.glm53_flash.tt.*`, which is the
`autoports/` convention this checkout already has (`models/autoports/aleph_alpha_kolibri_1_bf16`).

## Verified: 9 of 13 core modules import into `metal_1` unchanged

Copied `tt/` + `mesh/` to `models/experimental/glm53_flash/` and imported each:

| module | result |
|---|---|
| `config` | **OK** — `Glm5NextTextConfig`, no transformers, no ttnn |
| `precision` | **OK** — dtype map + HiFi2/HiFi4 kernel configs |
| `mlp` | **OK** — SiLU/SwiGLU dense FFN |
| `indexer` | **OK** — DSA dense-exact regime check |
| `hyperconnection` | **OK** — mHC, uses `deepseek_prefill.mhc_split_sinkhorn` |
| `weights` | **OK** — tile-cache + state-dict sources |
| `moe` | **OK** — router, routed decode/prefill, shared expert |
| `mla` | **OK** — NoPE MLA, paged latent cache |
| `model`, `stack_model` | **OK** (import only; `model` needs `tests.common` at call time) |
| `kda`, `decoder_layer`, `functional_decoder` | **FAIL** — one missing import, below |

Op availability checked against the compiled `ttnn` in `python_env`:

| op | status |
|---|---|
| `experimental.deepseek_prefill.mhc_split_sinkhorn` | present |
| `experimental.fast_reduce_nc` | present |
| `experimental.paged_fill_cache` / `paged_update_cache` | present |
| `transformer.chunked_scaled_dot_product_attention` | present |
| `transformer.paged_flash_multi_latent_attention_decode` | present |
| `experimental.indexer_score_dsa` | present |
| `transformer.fused_recurrent_gated_delta_rule` | present |
| `experimental.kda.qkv_causal_conv1d_silu` / `prepare_chunk_recurrence` / `sigmoid_gated_rms_norm` / `recurrent_chunk_scan` | present |
| **`experimental.kda.chronological_selections`** | **missing** |

So the Python surface of this `ttnn` build covers everything except one KDA helper.

## The single blocker, and its cause

`tt/kda.py` (line 36) imports `models.demos.deepseek_v3_d_p.tt.kda.chronological_selections`. That
file **was removed from `main`** by the "[KDA] Fuse the QKV split, convolution and history exchange
into one op" change (#60450). Recoverable from `a6ea480da8a`, but the recovered file also needs
`ttnn.experimental.kda.chronological_selections` and `ttnn._ttnn...kda._selection_layout`, and
**neither exists in this build** — the C++ op directory has no `chronological_selections/` target.

The replacement is `ttnn.experimental.kda.qkv_causal_conv1d_silu_with_carry`, whose docstring says it
does the history exchange itself and returns the replacement carry. So the fix is a **local patch to
`kda.py`'s `prefill`** (one call site: lines 266-268 plus the `select_local_final_history` call),
not a copy. That is the only porting work needed to get `kda.py`, `decoder_layer.py` and
`functional_decoder.py` importing.

## What to copy, in priority order

**Copy as-is — this is the biggest win.** `config.py`, `precision.py`, `weights.py`, `mlp.py`,
`moe.py`, `mla.py`, `indexer.py`, `hyperconnection.py`. Together these replace most of this port's
TODO list with working code, and they carry knowledge the stubs only name:

* `precision.py` — the exact dtype policy: routed experts `bfloat4_b`, shared expert / dense MLP /
  KDA q-k-v-o / MLA projections `bfloat8_b`, everything else bf16, and `A_log` / `dt_bias` / router
  bias / `hc_base` / `hc_scale` **fp32**. Plus residuals and all mHC quantities fp32, sublayer
  inputs bf16, router logits fp32, KDA state fp32.
* `mla.py` — the NoPE MLA absorbed form, and the crucial simplification: **`q_lat = W_uk^T q` is
  exact only because `qk_rope_head_dim = 0`** (no rope tail to absorb). Its DSA handling is
  explicitly limited to `kv_len <= 2051` and *rejects longer contexts* rather than silently computing
  dense attention where HF would attend sparsely (`DSA_DENSE_EXACT_MAX_KV`).
* `moe.py` — `Router` + routed decode/prefill + shared expert with the `swiglu_limit=10` clamp.
* `hyperconnection.py` — mHC with the Sinkhorn split done by the existing `mhc_split_sinkhorn`
  kernel, verified equal to GLM's `modeling:313-324`.
* `weights.py` — the whole weight story: tile-cache manifest format, per-tensor dtype map, and the
  `[in, out]` storage convention, `W_uk [64,256,512]` / `W_uv [64,512,256]` pre-split.

**Copy with the KDA patch.** `kda.py`, `decoder_layer.py`, `functional_decoder.py`. Note
`functional_decoder.py` is only 404 lines and is the reference for the whole layer contract
(`allocate_cache`, `begin_prefill_chunk`, `prefill_chunk`, `decode_forward`, `capture_decode_trace`).

**Mine for technique, don't copy wholesale.** `stack_model.py` (3116 lines), `prefill_tp.py`
(4962), `optimized_decoder.py`, `multichip_decoder.py`, `pmk_*` (the megakernel: 2831 + 2942
lines), `generator.py`, `spec_mtp.py`, `dflash2*`. These are tuned for homelab-1 / an 8x1 Blackhole
mesh with a specific `pipeline8_ring_wrap` partition; their value here is the *design record* in
their docstrings — which is unusually good. `stack_model.py` and `prefill_tp.py` in particular answer
questions this port's TODOs raise.

**Not usable.** `tt/kernels/` — the hand-written kernels are bound to a specific ttnn build's
generic_op ABI; and `mesh/*.textproto` — homelab-1's Ethernet mesh, not your Galaxy.

## Corrections this repo makes to the existing skeleton

Worth fixing in `models/experimental/glm53_flash/`, independent of whether the code is copied:

1. **KDA math is `sigmoid`, not `softplus`+`exp`.** My `tt/kda.py` docstring has
   `g = -exp(A_log) * softplus(f + dt_bias)`; the real form is
   `g = lower_bound * sigmoid(exp(A_log) * (f + dt_bias))` with `lower_bound = -5.0`. Also the
   sublayer RMSNorm is **sigmoid-gated** (`o_norm(o) * w * sigmoid(g_b(g_a(x)))`), and `q`/`k` are
   **L2-normalized** with `q *= 1/sqrt(128)`.
2. **`metal_1` already has a working KDA layer**: `models/demos/deepseek_v3_d_p/tt/kda/kda.py`
   (683 lines, class `ttKDA`, `forward` / `forward_slot` / `allocate_state`). It is
   DeepSeek-flavored (softplus decay, `tt_ccl` TP) but the op sequence is the same family. Prefer
   adapting it over porting `tt/kda.py` — it has no missing-op problem.
3. **The DSA dense-exact limit.** `kv_len <= 2051` is where top-k selects every visible token, so
   dense attention *is* exact. Above it you need real sparse select. This port's `tt/decode/
   attention.py` TODO should say so instead of "sparse SDPA over the selected rows" unqualified.
4. **The `deepseek_sparse_attention -> indexed_attention` remap is confirmed** as upstream
   `configuration_glm5_next.py:170-172`. My `config.py` guessed the right rename; this pins the
   provenance.
5. **Weight dtype policy is not bf16 throughout** (see `precision.py` above). Any cache/conversion
   code should adopt it.

## Recommendation

Copy the eight easy modules first and delete the corresponding stubs — that turns the majority of
this port from TODO into working code with no porting risk. Then decide on KDA separately, because
`metal_1`'s existing `ttKDA` may be the better base than patching the external one.
