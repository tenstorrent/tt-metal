# Findings from wiring Mistral-Small-4-119B into the prefill test suite

Ordered by how much they matter for model correctness.

## F0 — Mistral's MoE router is softmax, and this stack has no softmax router (real correctness gap)

This is the most consequential finding, and it was reached independently by two separate
investigations before being cross-checked in three places:

| layer | what it says | evidence |
|---|---|---|
| the real model | `router_logits.softmax(-1)` | `transformers/models/mistral4/modeling_mistral4.py:226` |
| host reference | `raise ValueError(f"Unsupported score_func '{score_func}'")` — only `sigmoid` / `sqrtsoftplus` | `tt/moe/validation_helpers.py:27` |
| device op | `TT_THROW("Unsupported score_func '{}'. Expected 'sigmoid' or 'sqrtsoftplus'.")` | `ttnn/cpp/.../moe_grouped_topk/moe_grouped_topk.cpp:23` |

`Mistral4Small119BConfig` declares no `SCORE_FUNC`, so `TtMoEGateConfig.score_func` falls to its
`"sigmoid"` default (`tt_moe_gate_prefill.py:118`).

**Why it matters, precisely:**
- Top-k *indices* can still agree — both scoring functions are monotone in the logit, and Mistral's
  router has no `e_score_correction_bias` to perturb the order. But that agreement is conditional on
  the bias being **zero**, and `tt_moe_gate_prefill.py:213-214` allocates it with `torch.empty(...)`
  — uninitialised memory, not zeros.
- Top-k *weights* differ unconditionally: after `norm_topk_prob` normalisation,
  `sigmoid(l_i) / Σ sigmoid(l_j) ≠ exp(l_i) / Σ exp(l_j)`.
- **This gap is context-length-independent.** Unlike F1 below, it does not need a long sequence to
  show up — it is wrong at seq 5120 today.

It has not yet been caught because `moe_pcc_threshold = 0.971` is read only at
`tests/pcc/test_ttnn_moe.py:573`, and that test's variants are `deepseek_v3` and `kimi_k2_6`.
**There was no mistral4 MoE test at all**, which is why adding one was the highest-value entry
in this batch.

### Measured: PCC does NOT catch this, which makes it worse

`test_mistral4_prefill_block` was wired faithfully against the softmax reference, deliberately
without lowering any threshold, on the expectation that the MoE case would go red. **It passed.**

| case | block output PCC | threshold |
|---|---|---|
| `mistral4-dense-seq5120-mesh-8x4` | **0.999878** | 0.98 |
| `mistral4-moe-seq5120-mesh-8x4` | **0.995178** | 0.98 |

So the wrong routing rule costs roughly **0.005 PCC** at seq 5120 on random weights — comfortably
inside a 0.98 gate, and inside the adapter's stricter 0.971 MoE gate too.

Why so small: with the correction bias zeroed, both scoring functions are monotone in the logit, so
top-4 **selection** is identical; only the **weights** differ. After `norm_topk_prob` renormalises
them to sum to 1, and with top-4-of-128 typically having one dominant expert, the residual
difference is second-order.

### And over depth it is model-breaking

The one-layer number was misleading. `test_mistral4_prefill_transformer` chains the reference per
layer and measures each stage:

| stage | PCC | Δ per layer |
|---|---|---|
| `embed` | **1.000000** | — |
| `layer_0` | 0.975813 | −0.0242 |
| `layer_1` | 0.942922 | −0.0329 |
| `layer_2` | 0.906779 | −0.0361 |
| `layer_3` | 0.870295 | −0.0365 |
| `layer_4` | 0.834688 | −0.0356 |
| `norm` | 0.834414 | −0.0003 |
| `lm_head` | 0.769880 | −0.0645 |

Three controls make this conclusive:
- `embed` is exactly **1.000000** — harness, sharding and comparator are exact, so all loss comes
  from the layers.
- `layer_0` / `layer_1` are **bit-identical** across the 2-layer and 5-layer runs — deterministic,
  not noise.
- the marginal loss **converges to ≈ −0.0355 per layer** (−0.0361 / −0.0365 / −0.0356) after a
  two-layer transient.

A steady ~0.0355 PCC per layer over a **36-layer** model puts the full stack's output far outside any
usable range. So the finding is not "PCC can't see it" — it is that **PCC can't see it in one layer,
and it is model-breaking over depth.** Anyone reading the passing one-layer
`test_mistral4_prefill_block` (0.995) as evidence that mistral4's MoE routing is correct would be
wrong; `test_mistral4_prefill_transformer` at 2 layers already fails a 0.99 gate.

**The fix** is a `softmax` `score_func` in `moe_grouped_topk.cpp` (which currently `TT_THROW`s on
anything but `sigmoid`/`sqrtsoftplus`), the matching host branch in `validation_helpers.py`, and
`SCORE_FUNC = "softmax"` on `Mistral4Small119BConfig`. That is C++ kernel work on an op shared by
DeepSeek / Kimi / GLM, so it was deliberately not attempted here. Until it lands, mistral4 MoE
output is wrong by construction, and no threshold choice makes that untrue.

Consequence for wiring: `pcc/test_moe_gate_prefill2d.py` is left **deliberately unwired**. Adding a
mistral4 entry there would validate the device gate against a sigmoid + `noaux_tc` reference the
model never uses — a green PCC on the wrong routing rule, the same trap
`mistral_small_4_119b_config.py:21-28` documents for the YaRN mscale. A correct entry needs a
bias-free softmax branch on both host and device, which is new mechanism, not test wiring.

## F1 — ttMLA is missing Mistral's position-dependent query scale (real correctness gap)

HF's `Mistral4Attention` scales the query by a position-dependent factor that ttMLA has no
equivalent for:

```python
# transformers/models/mistral4/modeling_mistral4.py:367-369
def get_llama_4_attn_scale(positions_ids, beta, max_position_embeddings):
    scaling = 1 + beta * torch.log(1 + torch.floor(positions_ids / max_position_embeddings))
    return scaling[:, None, :, None]

# :456-460 — applied to query_states, with:
#   beta                    = rope_parameters["llama_4_scaling_beta"]              = 0.1
#   max_position_embeddings = rope_parameters["original_max_position_embeddings"]  = 8192
```

Note the third argument is **`original_max_position_embeddings` (8192)**, not
`max_position_embeddings` (1048576). That distinction is what makes the gap real:

| token position | `floor(pos/8192)` | query scale |
|---|---|---|
| 0 – 8191 | 0 | **1.0000** (exact) |
| 8192 – 16383 | 1 | 1.0693 |
| 16384 – 24575 | 2 | 1.1099 |
| 24576 – 32767 | 3 | 1.1386 |

**Consequence for testing.** Below 8192 the scale is exactly 1.0, so a comparison against the true
HF reference is *valid*. Above 8192 ttMLA and HF genuinely diverge by 7–14% on the query.

Today's `test_mistral4_mla[...seq25k...]` passes at PCC 0.9986 — but it passes because the
in-tree `MLAReference` also omits the scale. That is the same "both sides wrong, so PCC stays
green" failure mode that commit 94b0499e483 called out for the YaRN mscale. Any strict-PCC
comparison against real `Mistral4Attention` must therefore be restricted to seq ≤ 8192 until
ttMLA implements the scale.

**Fix location:** ttMLA applies its softmax scale in `models/demos/deepseek_v3_d_p/tt/mla/mla.py`
(`self.scale`, passed to `ring_joint_scaled_dot_product_attention`). A position-dependent Q
multiply would need to go in before attention. Deliberately NOT attempted here: `mla.py` is shared
by DeepSeek / Kimi / GLM, so touching it would invalidate every other model's PCC baseline.

## F2 — scaled-FP8 KV cache path hardcodes the DeepSeek/Kimi 576-wide latent

`models/demos/common/prefill/runners/prefill_producer.py`:
- `_SCALED_FP8_LATENT_DIM = 512` (`:383`)
- guard `packed scaled-FP8 KV requires head_dim 576` (`:395`)
- `if head_dim == 576 and ...` (`:429`)

Meanwhile `MlaKvCacheGeometry.num_scales` only requires `latent_dim % 128 == 0`, which mistral4's
256 satisfies — so the geometry helper already supports a 320-wide scaled-FP8 row that
`prefill_producer` would reject. **Not currently hit**, because mistral4's KV cache is BFP8_TILE
and that path is generic (`head_dim % 32 == 0`). Becomes a blocker only if mistral4 moves to
scaled-FP8 KV.

## F3 — mistral4's kvpe latent is 320 wide, and only TP-replication makes that work

`kv_lora_rank + qk_rope_head_dim = 256 + 64 = 320` = 10 tiles of 32
(DeepSeek/Kimi: 512 + 64 = 576 = 18 tiles).

320 / 4 = 80, which is **not** tile-aligned — so a TP-sharded kvpe cache would be illegal at TP=4.
It works because `init_kvpe_cache` stamps `PlacementReplicate()` on the TP axis and the readback's
`ConcatMesh2dToTensor(dims=(2,1))` concatenates SP on dim 2 while treating TP as replicated
copies, leaving head_dim (dim 3) untouched. Worth knowing before anyone tries to TP-shard it.

Derived chunk size: `(320 // 32) * 1088 = 10880` bytes, cross-checked against the two existing
data points in `test_kv_cache_table.py` — `(576//32)*1088 = 19584` (DeepSeek/Kimi) and
`(128//32)*1088 = 4352` (GLM index cache).

## F4 — mistral4 has no indexer, confirmed by code not inference

`resolve_has_indexer(mistral4_hf_config())` returns **False**, so `mla_tt._indexer` never exists.
This is what makes the whole `sparse_mla/` directory (5 files, 137 params) inapplicable rather
than merely unwired, and it is why `test_glm52_kv_cache_table` could not be used as a template
(it reaches into `mla_tt._indexer.index_args.index_head_dim`).

## F5 — the checkpoint directory is read-only, and its weight cache is empty

`/data/kmabee/models/Mistral-Small-4-119B-2603` is read-only, and its
`tensor_cache_bh_32dev/` is present but **empty**. The adapter's `ttnn_cache_default = ""`
(`mistral_small_4_119b.py:39`) means no cache is used unless `TT_MISTRAL4_PREFILL_TTNN_CACHE` is
exported. So every pretrained run re-converts weights from fp8. Tolerable for MLA-only
(8–38 s); not for MoE. This sweep exports the cache to `/data/ssalice/mistral4_ttnn_cache`.

## F6 — pretrained MoE is blocked by the packed-expert checkpoint layout

`packed_expert_checkpoint = True` (`mistral_small_4_119b.py:84`): routed experts are stacked as one
`mlp.experts.gate_up_proj` `[128, 4096, 4096]` fp8 tensor rather than per-expert
`gate_proj`/`up_proj`. The pretrained fixture therefore loads **attention only**, leaving
`routed_expert_weights = None`. Every MoE-on-pretrained-weights case is blocked until that stacked
tensor is split — a checkpoint-layout gap, not something a better reference fixes. MoE tests here
are consequently authored with **random weights**.

## F7 — the MoE op unit tests' expert scale-down is incompatible with a 32-chip mesh (pre-existing)

`op_unit_tests/test_prefill_dispatch.py:440-444` states the design assumption plainly: *"these models
deploy their routed experts across a 32-chip Galaxy … but this op test runs on at most 8 chips."*

The `-pcc` params scale experts by `// 16`; the `-perf_no_pcc` params by `// 4`. On a 32-chip mesh:

| param | mistral4 experts after scaledown | experts/chip on 8x4 | outcome |
|---|---|---|---|
| `mistral4-pcc` | 128 // 16 = 8 | 8 // 32 = **0** | `ZeroDivisionError` at `tt/moe/init_helpers.py:245` |
| `mistral4-perf_no_pcc` | 128 // 4 = 32 | 32 // 32 = 1 | runs |

**This is not a mistral4 bug.** Verified directly: the baseline `dsv3-pcc` on `mesh-8x4` fails with
the *identical* `ZeroDivisionError` (`diag_dsv3_dispatch_8x4.log`). DeepSeek's 256 // 16 = 16 experts
over 32 chips is also 0.

Compounding it on this hardware: the ≤8-chip meshes these `-pcc` params need cannot be opened on a
Blackhole galaxy at all — `mesh-2x2` / `mesh-2x4` skip with *"Blackhole only supports 32-device mesh
configs (requested 4 / 8)"*. So on this box the `-pcc` params are unreachable for **every** model,
and only the `// 4` params are runnable.

Same root cause explains the other skips observed in the shakeout:
`cache/test_mla_cache.py`, `pcc/test_parallel_embedding.py` and `op_unit_tests/test_reduce.py` all
skipped for every mistral4 case with *"Blackhole only supports 32-device mesh configs (requested
4 / 8)"* — their mesh axes offer only 4- and 8-chip shapes. The mistral4 entries there are correct
and collect; they are simply unreachable on a 32-chip Blackhole galaxy and would exercise on a
Wormhole T3K or a 4/8-chip carve.

`op_unit_tests/test_ttnn_dispatch_combine.py` is the counter-example that proves the diagnosis:
its scaledown is `// 4`, so mistral4 runs and **passes** on `mesh-8x4` (8/8 cases).

## F0b — the softmax fix already exists in-tree, and it halves the error (but does not close it)

`GateComputeMode.GPT_DEVICE` is already implemented and is **exactly** Mistral's routing rule.
`_device_gpt_gate` (`tt/moe/tt_moe_gate_prefill.py:864-880`):

```python
biased = ttnn.add(logits_tiled, self.bias)                       # bias == 0 for Mistral -> identity
values, indices = ttnn.topk(biased, k=n_activated_experts, dim=-1, sorted=True)
scores = ttnn.softmax(values, dim=-1, numeric_stable=True)       # == HF's weights
```

**Why a k-wide softmax is sufficient** (proven on CPU, max weight error **1.19e-07** vs the current
sigmoid path's **5.34e-01**): softmax is strictly monotone, so `topk(softmax(l)) == topk(l)`, and the
128-wide normaliser cancels in the `norm_topk_prob` renormalisation —
`p_i / Σ_{j∈topk} p_j = exp(l_i) / Σ_{j∈topk} exp(l_j)`. No 128-wide reduction is needed.
Corollary: sum-normalising an activation `f` gives softmax weights **iff `f(l) = c·exp(l)`**, so
`sigmoid` and `sqrtsoftplus` are *wrong*, not merely different — there is no way to make an existing
`score_func` equivalent.

### Measured on device

Single block, seq 5120 (`block_gate_compare.log`):

| gate mode | rule | block output PCC | error |
|---|---|---|---|
| `DEVICE_FP32` | sigmoid + noaux_tc | 0.995192 | 0.004808 |
| `GPT_DEVICE` | **softmax over top-k** | **0.998185** | 0.001815 |
| (`dense` layer, no MoE at all) | — | 0.999877 | 0.000123 |

2-layer transformer (`gate_gpt_device_2L.log`, `gate_gpt_host_2L.log`):

| stage | `DEVICE_FP32` | `GPT_DEVICE` | `GPT_HOST` |
|---|---|---|---|
| `embed` | 1.000000 | 1.000000 | 1.000000 |
| `layer_0` | 0.976044 | **0.987098** | 0.987135 |
| `layer_1` | 0.943361 | **0.968872** | 0.968953 |
| `norm` | 0.943421 | 0.968698 | 0.968778 |
| `lm_head` | 0.942780 | **0.968680** | 0.968950 |

### What this settles, and what it doesn't

**Settled — the router is a large, real error source.** Switching to the correct rule cuts per-layer
error by ~62% on one block and ~46% across two layers. This is a **one-line change**
(`adapters/mistral_small_4_119b.py:40`, `default_gate_mode = "DEVICE_FP32"` → `"GPT_DEVICE"`) with
zero blast radius — DeepSeek/Kimi/GLM/V4 stay on `moe_grouped_topk` and are untouched.

**Settled — the residual is NOT the gate's top-k/softmax arithmetic.** `GPT_HOST` (host fp32 top-k +
softmax) and `GPT_DEVICE` (device bf16) agree to four decimals, so moving that arithmetic to fp32
buys nothing.

**Not settled — the residual still accumulates ~0.018 PCC/layer**, so a 36-layer model still will not
hold, and the PCC test still fails at 2 layers. It sits upstream of the gate arithmetic. Two
candidates, not separated by these runs:
1. **bf16 gate matmul → different expert selection.** Measured on CPU: fp32-reference vs
   bf16-device logits agree on only **98.02%** of top-4 sets (99.51% recall). ~2% of tokens routing
   to a different 4th expert is a plausible ~0.018/layer.
2. **Expert-FFN numerics.** The dense block leaves only 0.000123 error, so MLA is essentially exact;
   the MoE path contributes 0.0017 even with correct routing.

### Recommended sequence

1. **Now, one line:** flip `default_gate_mode` to `GPT_DEVICE`. Halves the error, no C++, no risk to
   other models. (Residual risk: GPT modes have only ever been exercised in the gate unit test, never
   through `TtMoe.forward → routing_setup → dispatch → combine` — though both device runs above did
   exactly that and passed, which is new evidence they work downstream.)
2. **Next, likely small:** there is **no `GPT_DEVICE` + fp32-logits mode**. `DEVICE_FP32` is fp32
   logits with the *wrong* rule; `GPT_DEVICE` is the right rule with *bf16* logits. A
   `GPT_DEVICE_FP32` that typecasts the logits before `ttnn.topk` — mirroring what `DEVICE_FP32`
   already does — is the obvious next experiment and would test candidate 1 directly.
3. **Production, ~15 lines of C++:** add `score_func = "softmax"` to `moe_grouped_topk`, implemented
   as `exp` (the kernel already does gather-then-normalise, so `exp` makes it softmax). Keeps
   padding-aware dispatch and `route_scale`, which `GPT_DEVICE` silently drops
   (`_device_gpt_gate` is the only gate path that never reads `self.config.route_scale` — harmless at
   Mistral's 1.0, a trap otherwise). Caveat: normalising unnormalised `exp(l)` is exact only for
   logits in roughly `[-35, +80]`; outside that the `1e-20` epsilon or fp32 overflow bites.

### Two adjacent bugs found while investigating

- **`torch.empty` for gate weights** (`tt_moe_gate_prefill.py:211,214`). On a cache *hit* it is inert
  (`ttnn/ttnn/operations/core.py:718-728` discards the passed tensor). But on a cache **miss in load
  mode** it writes **uninitialised memory into the `.tensorbin` permanently**, poisoning every later
  run. One-line `torch.zeros` fix. Not what is biting mistral4 today, but a real latent bug.
- **Real-weights MoE will `KeyError`** at `utils/transformer_helpers.py:641` (and `:238`) on
  `mlp.gate.e_score_correction_bias` — `Mistral4TopkRouter` has only `weight`. This needs a zeros
  injection regardless of which router fix lands, and is **independent** of the router bug. Don't let
  it get absorbed into "the softmax fix".
