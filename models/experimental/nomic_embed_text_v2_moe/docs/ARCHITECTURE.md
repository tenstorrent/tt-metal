# Nomic Embed Text v2 MoE: Architecture and Operator Mapping

Phase 0 deliverable for [#54917](https://github.com/tenstorrent/tt-metal/issues/54917).

Hand-written and hand-verified: every number was measured against the pinned checkpoint,
input-dependent values say so, and open assumptions sit in [Open questions](#open-questions)
rather than being stated as fact. The mechanical inventory the port is built against, meaning
the graph, module hierarchy, operator list, per-module shapes and memory figures, is generated
into [`MODEL_ANALYSIS.md`](MODEL_ANALYSIS.md). Phase sequencing, device measurements and the
bring-up gates are in [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

Reproduce with `pytest models/experimental/nomic_embed_text_v2_moe/tests/pcc/ -v`.

## 1. Model

Encoder-only multilingual text-embedding transformer with a Mixture-of-Experts FFN on every
other layer. 475M total parameters, ~305M active per token. Emits sentence embeddings; no
decoder, no KV cache, no generation.

| | |
|---|---|
| Weights | `nomic-ai/nomic-embed-text-v2-moe` @ `1066b6599d099fbb93dfcb64f9c37a7c9e503e85` |
| Modelling code | `nomic-ai/nomic-bert-2048` @ `7710840340a098cfb869c4f65e87cf2b1b70caca` |
| Tokenizer | `XLMRobertaTokenizerFast`, 250002 tokens, pad 1 / bos 0 / eos 2 |
| Backbone | 12 layers, hidden 768, 12 heads x 64, FFN 3072, LayerNorm eps 1e-5 |
| Position | Rotary (GPT-NeoX halves), base 10000, full head width. No learned position table. |
| MoE | 8 experts, top-2, on layers 1/3/5/7/9/11 |
| Pooling | mask-weighted mean, optional Matryoshka truncation, L2 normalize |

Two repositories must be pinned. The weights repo's `auto_map` points at a different repo for
the code, so pinning only the weights leaves the model definition floating on `main`.

The full graph, with the mask path and both residual edges of each block, is
[`MODEL_ANALYSIS.md` section 1](MODEL_ANALYSIS.md#1-model-graph).

## 2. Checkpoint contract

148 tensors, 475,292,928 parameters, all `float32`, flat key names. Generated from the config
by `loader.expected_checkpoint_keys` and asserted against the real file, so the generator is
what is under test.

```
embeddings.word_embeddings.weight            [250048, 768]
embeddings.token_type_embeddings.weight      [1, 768]
emb_ln.{weight,bias}                         [768]

encoder.layers.{0..11}.attn.Wqkv.weight      [2304, 768]     .bias [2304]
encoder.layers.{0..11}.attn.out_proj.weight  [768, 768]      .bias [768]
encoder.layers.{0..11}.norm1.{weight,bias}   [768]
encoder.layers.{0..11}.norm2.{weight,bias}   [768]

# dense layers: i in {0, 2, 4, 6, 8, 10}
encoder.layers.{i}.mlp.fc1.weight            [3072, 768]     .bias [3072]
encoder.layers.{i}.mlp.fc2.weight            [768, 3072]     .bias [768]

# MoE layers: i in {1, 3, 5, 7, 9, 11}
encoder.layers.{i}.mlp.router.layer.weight   [8, 768]        # no bias
encoder.layers.{i}.mlp.experts.mlp.w1        [24576, 768]    # = [8, 3072, 768]
encoder.layers.{i}.mlp.experts.mlp.w2        [24576, 768]    # = [8, 3072, 768]
encoder.layers.{i}.mlp.experts.bias          [768]           # one shared bias
```

Verified absent: `position_embeddings`, `pooler`, `cls.`, `lm_head`, `ln_f`, `inv_freq`,
`norm_factor`, router bias, anything vision. Each absence is a separate test; a hit means the
reference is dropping a real weight.

Per-group parameter counts and the fp32 and bf16 footprint are in
[`MODEL_ANALYSIS.md` section 5](MODEL_ANALYSIS.md#5-parameters-and-memory).

## 3. What is easy to get wrong

Each item was measured and each has a negative control in the test suite.

### 3.1 MoE top-k weights are not renormalized

`moe_normalize_expert_weights` is false. Softmax over all 8 experts, top-2 taken, used as is.
They sum to less than 1, so the MoE branch is attenuated relative to the residual.

Measured at the router on real text:

| layer | 1 | 3 | 5 | 7 | 9 | 11 |
|---|---|---|---|---|---|---|
| mean top-2 sum | 0.700 | 0.772 | 0.800 | 0.810 | 0.834 | 0.302 |

Mean 0.703 across the six MoE layers. Renormalizing makes every one exactly 1.0. Mixtral and
Switch both renormalize, so this is the most likely thing to be copied in by reflex.

### 3.2 The shared expert bias is added once, after the weighted sum

One `[768]` vector per MoE layer, shared by all eight experts. Folding it into the per-expert
loop scales it by the routed-weight sum, giving an offset of `(sum(w) - 1) * bias`.

PCC cannot see this: it mean-centres, and the offset is nearly constant.

| implementation | PCC vs correct | max-abs |
|---|---|---|
| bias inside the expert loop | 0.99999 (synthetic) / 0.9999998 (real weights) | 7.9e-2 |
| renormalized top-2 | 0.989 to 0.993 depending on input | 2.8e+01 |
| `w2` viewed `(E, H, F)` | -0.0006 | 3.2e+01 |

The bias bug passes any PCC threshold. The renormalization bug sits on a 0.99 gate and whether
it passes depends on the input. Both are gated on max-abs.

### 3.3 Expert weight orientation

`w1` and `w2` are both `[E*3072, 768]`, expert axis outer: expert `e` owns rows
`e*3072 .. (e+1)*3072`. Both stored `[F, H]` per expert:

```python
x1  = x @ w1[e].T          # w1[e] is [3072, 768], transposed
out = gelu(x1) @ w2[e]     # w2[e] is [3072, 768], not transposed
```

`E*F*H` is symmetric in F and H, so viewing `w2` as `(E, H, F)` succeeds and every downstream
matmul typechecks. Nothing raises. Output is uncorrelated noise (PCC -0.0006).

### 3.4 Rotary: NeoX halves, cos/sin cached at half width

`rotate_half` splits the last axis in half: `(x1, x2) -> (-x2, x1)`. Not the GPT-J even-odd
pairing; `rotary_emb_interleaved` is false.

The cache holds `[S, 32]`, widened at apply time by concatenation (`torch.cat([cos, cos])`).
`repeat_interleave` instead gives the GPT-J lane layout, which combined with NeoX
`rotate_half` is not a rotation: it stops preserving the per-plane norm and scores PCC 0.61.

### 3.5 Wqkv is three-major

`[q(768) | k(768) | v(768)]`, heads contiguous within each block. A head-major reading strides
`3*head_dim` across blocks that are 768 wide, so its "q" slice straddles all three.

### 3.6 GELU is exact erf, not tanh

`activation_function: "gelu"` maps to `nn.GELU(approximate="none")`. The tanh approximation
differs by 4.7e-4: small enough to pass a loose PCC gate, large enough to look like a device
precision problem later.

### 3.7 The `<pad>` embedding row is not zero

`nn.Embedding(padding_idx=1)` zeroes row 1 at init; loading the checkpoint overwrites it.
Trained row absmax 1.499232e-02, all 768 elements non-zero. Do not pass `padding_idx` on
device.

### 3.8 Post-norm, not pre-norm

`h = norm1(attn(x) + x)`, `y = norm2(mlp(h) + h)`. Residual added before the norm, so every
sub-block output is re-centred. This is why error does not compound over 12 layers the way it
does in a pre-norm decoder. Verified by zeroing each branch and checking the block collapses
to `norm2(norm1(x))`.

### 3.9 The MoE attention_mask is inverted and ignored

Upstream's block passes `torch.where(attention_mask.squeeze() == 0, 1, 0)` into the MoE layer,
a mask where 1 means pad, and `NomicMoELayer.forward` ignores it. Applying it would zero the
real tokens. The reference does not thread it through.

## 4. The transformers native-class trap

`transformers` >= 5 ships a native `transformers.models.nomic_bert` targeting
nomic-embed-text-v1.5: separate q/k/v/o, no biases, SwiGLU `gate_proj`/`up_proj`/`down_proj`,
no MoE, `layer_norm_eps` 1e-12 and rope theta 1000.0. Registered for
`model_type == "nomic_bert"`, which is what this checkpoint declares.

Measured on transformers 5.12.1 at the pinned revision:

| call | resolves to | consequence |
|---|---|---|
| `AutoConfig.from_pretrained(MODEL_ID)` | native config | mild: only `use_cache` is dropped |
| `AutoModel.from_pretrained(MODEL_ID)` | native model | severe, and it does not raise |

The model case reports every MoE tensor, every `mlp.fc1/fc2` and all q/k/v/o biases as
UNEXPECTED (silently discarded), and `gate_proj`/`up_proj`/`down_proj` as MISSING (randomly
initialised). It returns a working 136-parameter model with no MoE that computes finite,
plausible, wrong numbers.

Containment: pass `trust_remote_code=True` and `code_revision`, then assert the resolved
class's module starts with `transformers_modules`. The assertion is the load-bearing part;
without it a future release that changes resolution order silently downgrades the reference.

The tokenizer half is inert: `tokenizer_config.json`'s `tokenizer_class` outranks the
model-type mapping, so `AutoTokenizer` is safe. There is a canary on that precedence anyway.

## 5. Parity with upstream

Bit-exact at the pinned revisions: max-abs 0.0, PCC 1.0000000000, at all 13 capture points
(`emb_ln` plus each block) and end to end.

Two upstream behaviours are deliberately not reproduced:

- Upstream requires `attention_mask` and raises `AttributeError` without it. The reference
  defaults it to all-ones.
- Upstream's `matryoshka_dim` slices `sequence_output[:, :matryoshka_dim]`, the sequence axis,
  dropping tokens while keeping 768-wide features. Truncation belongs after pooling, on the
  feature axis, and lives in `pipeline.py`.

Per-layer activation magnitudes, real text, S=22. A relative tolerance calibrated on layer 0
would be far too loose from layer 1 onwards.

| | emb_ln | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kind | | dense | MoE | dense | MoE | dense | MoE | dense | MoE | dense | MoE | dense | MoE |
| absmax | 3.5 | 7.6 | 21.2 | 22.0 | 23.6 | 21.1 | 20.9 | 17.7 | 15.4 | 14.1 | 15.3 | 18.3 | 8.3 |
| std | 0.23 | 0.42 | 0.63 | 0.73 | 0.65 | 0.69 | 0.69 | 0.71 | 0.67 | 0.68 | 0.66 | 0.70 | 0.69 |

## 6. Embedding pipeline

From the checkpoint's `modules.json`, `1_Pooling/config.json` and
`config_sentence_transformers.json`:

```
task prefix -> tokenize -> encoder -> mask-weighted mean pool -> [truncate] -> L2 normalize
```

Task prefixes are trained-in, not decoration: `search_query: `, `search_document: `,
`classification: `, `clustering: `, each with a trailing space. `include_prompt` is true, so
prefix tokens are pooled too.

Pooling is mean, not CLS. Padded positions must be excluded, since `<pad>` carries a non-zero
embedding and including it makes an embedding depend on its batch-mates.

Matryoshka ordering is a free choice. Truncate-then-normalize and normalize-then-truncate give
different norms (1.0 vs ~0.57 at d=256) but identical directions, and the declared similarity
is cosine.

Model card check: cosine similarity between the passage-prefixed pair in
`common.MODEL_CARD_SENTENCES` reproduces at 0.911788 against the card's 0.9118. The exact
strings live in that constant rather than being repeated here, since one of them carries a
non-ASCII character that the input must preserve to reproduce the number.

## 7. Operator mapping for the TTNN port

Phase 1 target, not implemented in this PR. Correctness only: `bfloat16`, `TILE_LAYOUT`,
`DRAM_MEMORY_CONFIG`, no sharding.

| Reference | TTNN |
|---|---|
| `word_embeddings` | `ttnn.embedding(ids, weight, layout=TILE)`, no `padding_idx` (3.7) |
| `token_type_embeddings` | fold to a constant: `type_vocab_size == 1`, pre-add the `[1,768]` row into the table in fp32 at load |
| `LayerNorm` + residual | `ttnn.layer_norm(x, epsilon=1e-5, weight, bias, residual_input_tensor=h)` |
| fused `Wqkv` | `ttnn.linear(h, W, bias=b)` with `W = cat([Wq.T, Wk.T, Wv.T], -1)` |
| three-major split | `ttnn.experimental.nlp_create_qkv_heads(num_heads=12, num_kv_heads=12, transpose_k_heads=False)` |
| rotary | `ttnn.experimental.rotary_embedding_hf(x, cos, sin, is_decode_mode=False)`, cos/sin concat-duplicated (3.4) |
| SDPA | `ttnn.transformer.scaled_dot_product_attention(..., is_causal=False, scale=1/8)`; `is_causal` defaults True |
| head concat | `ttnn.transformer.concatenate_heads`, which takes rank-4 and returns rank-3, so the canonical rank-4 activation layout needs a reshape after it |
| GELU | `ttnn.gelu(x)` as its own op, never `fused_activation=(GELU, True)`; the LUT error 2.3e-2 exceeds the bf16 noise floor |
| router | fp32 `ttnn.linear`, `ttnn.softmax(dim=-1, compute_kernel_config=HiFi4)`, cast bf16, `ttnn.topk(k=2)`, `ttnn.scatter`. No sum-normalization (3.1); `models/demos/gemma4/tt/router.py` divides by the top-k sum, which is exactly what must not be copied here |
| experts | 2 broadcast-batch `ttnn.matmul`, `ttnn.gelu`, `ttnn.permute`, `ttnn.mul`, `ttnn.experimental.fast_reduce_nc(dims=[1])`, `ttnn.add(bias)` |
| padding mask | `eq`, `to_layout(TILE)`, `reshape`, `where(pad, -100000., 0.)`, `expand`, `typecast`. Built once per forward from the tokenizer's `attention_mask`, not from `input_ids != pad_token_id`, and shared across all 12 layers |
| mean pool | `matmul(keep[B,1,1,S], hidden[B,1,S,D])` divided by `clip(sum(keep), 1., S)` |
| L2 normalize | no single op: `mul`, `sum`, `rsqrt(+1e-12)`, `mul` |

Device-side prerequisites (HiFi4 softmax, `ttnn.scatter` rejecting fp32, measured MoE-layer
PCC, grid geometry) are in
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md#device-measurements).

`NomicExperts.dense_forward` implements the device-shaped MoE in PyTorch and is asserted equal
to the upstream loop, so it is the bridge between the two:

```python
w1_tt = w1.view(8, 3072, 768).transpose(1, 2)   # [8, 768, 3072], one transpose
w2_tt = w2.view(8, 3072, 768)                   # [8, 3072, 768], pure view
# every token through every expert, weighted by a dense [T, 8] routing tensor that is zero
# off the top-2, reduced over the expert axis, then one shared bias.
```

## Open questions

Carried to Phase 1; none blocks Phase 0.

| | Question | Status |
|---|---|---|
| 1 | Router index agreement between torch fp32 and device bf16 | Top-2 over 8 experts is a discrete decision and near-ties flip. 99.41% set-agreement measured in device probing at T=512. Needs a gate on agreement plus a margin analysis, not exact match. |
| 2 | End-to-end 12-layer TTNN PCC | Not measurable until the TTNN model exists. Post-norm (3.8) is the reason to expect it not to compound; the 0.98 target at bring-up step 9 is informed by the per-layer magnitudes above and by bge_m3 reaching 0.94 at bfloat8. |
| 3 | Blackhole DRAM headroom for ~951 MB resident plus transient | No documented per-chip figure in-repo. Confirm by allocation at Phase 1 step 1. |
| 4 | `fp32_dest_acc_en` on BH expert matmuls | From in-repo findings (#49068), not independently reproduced. Defaulting it off for matmuls is the safe side. |
