# Nomic Embed Text v2 MoE: TTNN bring-up plan (single Blackhole chip)

Phase sequencing, the device measurements that constrain the design, and the bring-up gates.
The verified architecture, the silent-failure list and the operator mapping are in
[`ARCHITECTURE.md`](ARCHITECTURE.md); the generated inventory is in
[`MODEL_ANALYSIS.md`](MODEL_ANALYSIS.md); the file layout and test commands are in
[`../README.md`](../README.md). This document does not restate them.

## Context

`nomic-ai/nomic-embed-text-v2-moe` is a multilingual encoder-only text-embedding model with a
Mixture-of-Experts FFN, 475M total and 305M active parameters. Umbrella issue
[#54916](https://github.com/tenstorrent/tt-metal/issues/54916) tracks running it on one
Blackhole chip as a TTNN port of the golden PyTorch reference, under `models/experimental/`.

| Issue | Phase | Scope |
|---|---|---|
| [#54917](https://github.com/tenstorrent/tt-metal/issues/54917) | 0: architectural overview | Understand the architecture and the PyTorch implementation; run the model on small inputs. |
| [#54919](https://github.com/tenstorrent/tt-metal/issues/54919) | 0: PyTorch reference | Reference implementation using transformers, following the existing Gemma and Llama patterns. |
| [#54918](https://github.com/tenstorrent/tt-metal/issues/54918) | 1: first working TTNN PoC | The TTNN variant becomes functional on one Blackhole chip. Performance out of scope; only correct output matters. |

Decisions taken up front: Phase 0 in its own PR, a vendored reference under `reference/` as the
golden, weights downloaded and assumptions settled before any TTNN code, the router gated on
index agreement plus routed-subset PCC, and no `tests/perf/` until Phase 1.

Every load-bearing assumption has been executed rather than inferred from the model name, the
paper or a code read. What remains open is the register in
[`ARCHITECTURE.md`](ARCHITECTURE.md#open-questions), and none of it blocks Phase 0.

# Device measurements

Run on the p300c. These are the findings that constrain the Phase 1 design; the model-side
findings are in [`ARCHITECTURE.md`](ARCHITECTURE.md).

| Gate | Result |
|---|---|
| Full MoE layer at real dims (E=8, I=3072, H=768, T=512), batched dense-all-experts | PCC 0.998942 |
| Broadcast-batch matmul `[1,1,T,768] x [1,8,768,3072]` | PCC 0.999979 |
| `fast_reduce_nc(dims=[1])` | PCC 0.999992 |
| `reshape [1,1,T,H]` to `[B,1,S,H]` and back | exact round trip |
| `nlp_create_qkv_heads` on three-major | PCC 0.999999; head-major control 0.082 |
| `rotary_embedding_hf`, NeoX, batch folded into heads | PCC 0.999995; interleaved control 0.287 |
| SDPA `is_causal=False` with an additive mask | PCC 0.999776 |
| `layer_norm(residual_input_tensor=...)` and `embedding` | PCC 0.999995 and 0.999999 |
| `ttnn.topk` on W=8 | fp32 and bf16 both work, indices all below 8, exact index match against torch. `bfloat8_b` fails. |

## Four findings that change the design

1. `ttnn.softmax` needs an explicit HiFi4 `compute_kernel_config`. Default max-abs against
   torch is 1.91e-2; with HiFi4 it is 1.31e-3, a 14x improvement. `numeric_stable=True`
   changes nothing. Not optional for the router.
2. `ttnn.scatter` rejects `float32` in both TILE and ROW_MAJOR
   (`TT_FATAL ... !(input_dtype == FLOAT32 && input_layout == TILE)`). bf16 works in both, so
   the router cannot stay fp32 end to end; cast before the scatter.
3. `TT_VISIBLE_DEVICES=0` fails on this p300c with `TT_FATAL: Custom fabric mesh graph
   descriptor path must be specified for CUSTOM cluster type`. Use the plain `device` fixture
   or `open_device(device_id=0)` with no `TT_VISIBLE_DEVICES` set.
4. The compute grid is 11x10 = 110 cores and `dram_grid_size().x` is 8, not the (8, 10) that
   `models/tt_transformers/tt/model_config.py:1815` implies. Always derive both from
   `mesh_device.compute_with_storage_grid_size()`.

## GELU, quantified

| | max-abs vs exact erf |
|---|---|
| `ttnn.gelu` default (`Accurate`), fp32 | 9.5e-7, essentially exact |
| `ttnn.gelu` default (`Accurate`), bf16 | 1.58e-2, the bf16 noise floor |
| `fast_and_approximate_mode=True` (`FastLut`) | 2.34e-2, independent of dtype |

The LUT error exceeds the bf16 noise floor, so it is not swamped by it. The repo's BERT idiom
`fused_activation=(ttnn.UnaryOpType.GELU, True)` selects that LUT. Do not copy it.

# Phase 0

## The vendored reference

Three fidelity pillars, all validated:

1. Name isomorphism instead of a remapper. The module tree mirrors upstream's names (`Wqkv`,
   `emb_ln`, `mlp.experts.mlp.w1`), so `load_state_dict(strict=True)` is itself the structural
   proof. Verified clean: 0 missing, 0 unexpected.
2. A validating config. `from_hf_config` raises on every one of the 23 fields in
   `REQUIRED_FIELDS`, whether the field is missing or holds a value the reference does not
   implement, plus three structural checks: `n_embd` divisible by `n_head`, an even head
   dimension for the rotary halves, and `vocab_size` a multiple of `pad_vocab_size_multiple`.
   It converts silent divergence into a loud error.
3. Per-layer parity. Forward hooks at `emb_ln` and each `encoder.layers.{i}` give a 13-point
   PCC ladder, because end-to-end PCC alone can mask compensating errors.

Around 380 lines covering only the inference path. Excluded, roughly 85% of upstream's 2556
lines: the vision tower, all task heads, the pooler, gated-MLP variants, DynamicNTK rotary,
xPos, the megablocks bridge, the custom `from_pretrained`, KV cache, gradient checkpointing,
the pre-norm branch, and every `use_flash_attn` and `fused_*` path. `einops` is dropped so
layout choices are explicit.

The megablocks dependency is not needed: the model card's 0.9118 reproduces at 0.911788 on the
pure-torch path, so the card's number was not produced by a fused kernel this port would have
to match.

## Tests

Coverage per file is tabulated in [`../README.md`](../README.md#tests). What matters for the
500 KB pre-commit cap: nothing committed is a tensor. The committed data is `config.json` at
2.4 KB, three integers, two revision SHAs and one float. Measured numbers live in
`ARCHITECTURE.md` as documentation, and only contracts that can be regenerated are asserted.

Acceptance for #54917: `ARCHITECTURE.md` documents the verified architecture and the operator
mapping, the vendored reference matches upstream on the 13-point ladder, and the model runs on
small inputs and reproduces 0.911788.

# Phase 1: TTNN design (#54918)

Correctness only. `bfloat16`, `TILE_LAYOUT`, `DRAM_MEMORY_CONFIG`, no sharding. The operator
mapping is [`ARCHITECTURE.md` section 7](ARCHITECTURE.md#7-operator-mapping-for-the-ttnn-port).

The canonical activation layout is a flattened token axis `[1, 1, B*S, 768]`. Every sub-block
except attention is token-wise, and flattening removes all batch-broadcast ambiguity from the
MoE matmuls; the reshape round trip is exact. Do not copy
`models/demos/blackhole/sentence_bert/ttnn/common.py`'s program configs: that file is
byte-identical to the Wormhole one and hardcodes a (6, 8) grid.

## MoE, the verified formulation

Chosen over concat-dense: it holds the same arithmetic position as the reference, needs a third
of the peak memory, and requires no static `expand` matrix.

```python
w1_tt = w1.view(8, 3072, 768).transpose(1, 2).unsqueeze(0)   # [1, 8, 768, 3072], one transpose
w2_tt = w2.view(8, 3072, 768).unsqueeze(0)                   # [1, 8, 3072, 768], pure view
bias = experts_bias.reshape(1, 1, 1, 768)

logits = ttnn.linear(typecast(h, fp32), rw, compute_kernel_config=HIFI)  # fp32
p = ttnn.softmax(logits, dim=-1, compute_kernel_config=HIFI)             # HiFi4 required
p_bf16 = typecast(p, bf16)                               # ttnn.scatter rejects fp32
weights, experts = ttnn.topk(p_bf16, k=2, dim=-1)
dense = ttnn.scatter(ttnn.zeros_like(p_bf16), dim=-1, index=experts, src=weights)

h1 = ttnn.matmul(h, w1_tt, compute_kernel_config=HIFI)   # [1, 8, T, 3072] broadcast-batch
a = ttnn.gelu(h1)                                        # Accurate variant
o = ttnn.matmul(a, w2_tt, compute_kernel_config=HIFI)    # [1, 8, T, 768]
o = ttnn.mul(o, ttnn.permute(dense, (0, 3, 2, 1)))       # [1, 8, T, 1] broadcast
y = ttnn.experimental.fast_reduce_nc(o, dims=[1])        # [1, 1, T, 768]
return ttnn.add(y, bias)                                 # one shared bias, after the sum
```

Measured at PCC 0.998942 at real dims on Blackhole. Peak transient about 50 MB at B=1, S=512;
resident weights about 951 MB. The transient scales with batch times sequence length, not
sequence length alone, so a larger batch changes the budget: see
[`MODEL_ANALYSIS.md` section 5](MODEL_ANALYSIS.md#5-parameters-and-memory).

## Bring-up order and gates

| # | Step | Gate |
|---|---|---|
| 1 | Embeddings | PCC >= 0.999 plus token-type-fold exactness |
| 2 | Rotary | PCC >= 0.999, an analytic single-position probe, and the interleaved oracle must fail |
| 3 | Attention, no mask then 25% padding | PCC >= 0.99 on kept positions only |
| 4 | Dense MLP | PCC >= 0.999 |
| 5 | Router | set agreement >= 99%, every disagreeing token shown to have a near-tied softmax margin, weights >= 0.999 on agreeing tokens, and the not-renormalized assertion |
| 6 | Experts, injected routing | PCC >= 0.998 and max-abs against the bias-inside oracle |
| 7 | MoE layer | PCC >= 0.998 on correctly-routed tokens plus both negative controls |
| 8 | One dense and one MoE block | PCC >= 0.99 |
| 9 | 12-layer encoder | PCC >= 0.98 |
| 10 | Pool, truncate, normalize | PCC >= 0.99 |
| 11 | End to end | PCC >= 0.99, cosine against the reference within 0.005, and top-1 retrieval agreement |

Steps 5 to 7 carry non-PCC assertions because PCC provably cannot catch their two most likely
bugs, measured at 0.9999998 and 0.993 in
[`ARCHITECTURE.md`](ARCHITECTURE.md#32-the-shared-expert-bias-is-added-once-after-the-weighted-sum)
section 3.2.
Step 5 is the hardest: routing is a discrete decision and a flip changes a token's output
entirely. Step 9 is the least worrying, because post-norm re-centres after every sub-block, so
bf16 error cannot compound the way it does in a pre-norm decoder.

## Blackhole specifics

`fp32_dest_acc_en=False` plus HiFi4 for matmuls; `fp32_dest_acc_en=True` for norms and softmax;
`ttnn.Tile([32, 32])` only, since TinyTile is broken (#31385); derive the grid (11x10) and
`dram_grid_size().x` (8) from the device; avoid `topk_router_gpt`, unavailable on Blackhole, and
`moe_compute`, which fails PCC there; set no `TT_VISIBLE_DEVICES` on this p300c.

# Phase 2: device performance

`@run_for_blackhole()` plus `@pytest.mark.models_device_performance_bare_metal` driving
`run_device_perf`, then `check_device_perf(assert_on_fail=True)` and
`prep_device_perf_report`, with `command` pointing at the Phase 1 end-to-end PCC test so the
number always describes shipped code. `device_params = {"l1_small_size": 16384,
"trace_region_size": 0, "num_command_queues": 1}`.

Trace and 2CQ are not worth it in Phase 1: around 200 DRAM-interleaved ops at M=512 are
compute-bound, the input is 2 KB, and an oversized trace reservation is itself an OOM risk.

# Verification

```bash
cd /path/to/tt-metal && source python_env/bin/activate
pytest models/experimental/nomic_embed_text_v2_moe/tests/pcc/ -v      # no TT_VISIBLE_DEVICES
pre-commit run --from-ref origin/main --to-ref HEAD
```

Phase 1 adds `TT_METAL_WATCHER=10 pytest .../tests/pcc/test_ttnn_model.py -v`. Run Watcher and
the profiler in separate invocations; they contend for debug resources.

# Out of scope

No decoder, generation, KV cache or multi-chip infrastructure. No optimization before
correctness passes. No `tt/` code before Phase 1.
