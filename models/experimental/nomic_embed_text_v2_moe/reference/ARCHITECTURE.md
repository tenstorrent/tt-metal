# Nomic Embed Text v2 MoE — Architecture and Operator Mapping

Phase 0 deliverable for [#54917](https://github.com/tenstorrent/tt-metal/issues/54917).

Everything in this document was **measured against the pinned checkpoint**, not read off the
model card or the paper. Where a number is input-dependent it says so. Where something is
still an assumption it is in [Open questions](#open-questions), not stated as fact.

Reproduce any of it with `pytest models/experimental/nomic_embed_text_v2_moe/tests/pcc/ -v`.

---

## 1. What this model is

An **encoder-only** multilingual text-embedding transformer with a **Mixture-of-Experts** FFN
on every other layer. 475M total parameters, ~305M active per token. It emits sentence
embeddings, not tokens — there is no decoder, no KV cache and no generation anywhere in it.

| | |
|---|---|
| Weights | `nomic-ai/nomic-embed-text-v2-moe` @ `1066b6599d099fbb93dfcb64f9c37a7c9e503e85` |
| Modelling code | `nomic-ai/nomic-bert-2048` @ `7710840340a098cfb869c4f65e87cf2b1b70caca` |
| Tokenizer | `XLMRobertaTokenizerFast`, 250002 tokens, pad 1 / bos 0 / eos 2 |
| Backbone | 12 layers, hidden 768, 12 heads × 64, FFN 3072, LayerNorm eps 1e-5 |
| Position | Rotary (GPT-NeoX halves), base 10000, full head width. **No learned position table.** |
| MoE | 8 experts, top-2, on layers 1/3/5/7/9/11 |
| Pooling | mask-weighted **mean**, then optional Matryoshka truncation, then L2 normalize |

**Two repositories have to be pinned.** The weights repo's `auto_map` points at a *different*
repo for the code. Pinning only the weights leaves the reference model definition floating on
`main`.

### Structure

```
input_ids ─► word_embeddings ─(+)─ token_type_embeddings[0] ─► emb_ln ─┐
                                                                       │
   ┌───────────────────────────────────────────────────────────────────┘
   │
   ├─► block 0  (dense MLP)  ─► block 1  (MoE) ─► ... ─► block 11 (MoE) ─► last_hidden_state
   │
   └─ each block is POST-norm:   h = norm1(attn(x) + x)
                                 y = norm2(mlp(h) + h)

last_hidden_state ─► mask-weighted mean pool ─► [truncate] ─► L2 normalize ─► embedding
```

---

## 2. Checkpoint contract

**148 tensors, 475,292,928 parameters, all `float32`, flat key names.** Generated from the
config by `loader.expected_checkpoint_keys` and asserted against the real file — so the
*generator* is what is under test, not a captured list compared to itself.

```
embeddings.word_embeddings.weight            [250048, 768]
embeddings.token_type_embeddings.weight      [1, 768]
emb_ln.{weight,bias}                         [768]

encoder.layers.{0..11}.attn.Wqkv.weight      [2304, 768]     .bias [2304]
encoder.layers.{0..11}.attn.out_proj.weight  [768, 768]      .bias [768]
encoder.layers.{0..11}.norm1.{weight,bias}   [768]
encoder.layers.{0..11}.norm2.{weight,bias}   [768]

# dense layers: i ∈ {0, 2, 4, 6, 8, 10}
encoder.layers.{i}.mlp.fc1.weight            [3072, 768]     .bias [3072]
encoder.layers.{i}.mlp.fc2.weight            [768, 3072]     .bias [768]

# MoE layers: i ∈ {1, 3, 5, 7, 9, 11}
encoder.layers.{i}.mlp.router.layer.weight   [8, 768]        # NO bias
encoder.layers.{i}.mlp.experts.mlp.w1        [24576, 768]    # = [8, 3072, 768]
encoder.layers.{i}.mlp.experts.mlp.w2        [24576, 768]    # = [8, 3072, 768]
encoder.layers.{i}.mlp.experts.bias          [768]           # ONE shared bias
```

**Verified absent:** `position_embeddings`, `pooler`, `cls.`, `lm_head`, `ln_f`, `inv_freq`,
`norm_factor`, any router bias, anything vision. Each absence is a separate test — a hit would
mean the reference is silently dropping a real weight.

Resident weights: **1901 MB** at fp32, **951 MB** at bf16.

---

## 3. The parts that are easy to get wrong

Each of these was measured, and each has a **negative control** in the test suite: an
implementation of the plausible wrong choice, asserted to differ. A test that only checks the
right answer cannot tell you whether the wrong answer would also have passed.

### 3.1 MoE top-k weights are NOT renormalized

`moe_normalize_expert_weights` is `false`. Softmax runs over all 8 experts, top-2 is taken,
and those two weights are used **as they are** — they sum to less than 1, so the MoE branch is
attenuated relative to the residual.

Measured at the router on real text:

| layer | 1 | 3 | 5 | 7 | 9 | 11 |
|---|---|---|---|---|---|---|
| mean top-2 sum | 0.700 | 0.772 | 0.800 | 0.810 | 0.834 | 0.302 |

Mean over the six MoE layers **0.703**; renormalizing would make every one of them exactly
1.0. Nearly every other MoE implementation — Mixtral, Switch — *does* renormalize, so this is
the most likely thing to be copied in by reflex.

### 3.2 The shared expert bias is added once, after the weighted sum

One `[768]` vector per MoE layer, shared by all eight experts, added after the reduction — not
inside the per-expert loop. Folding it into the loop scales it by the routed-weight sum,
giving an almost-constant offset of `(Σw − 1)·bias`.

**PCC cannot see this.** PCC mean-centres before correlating, and a near-constant offset is
almost exactly what mean-centring removes:

| implementation | PCC vs correct | max-abs |
|---|---|---|
| correct | — | — |
| bias inside the expert loop | **0.99999** (synthetic) / **0.9999998** (real weights) | 7.9e-2 |
| renormalized top-2 | 0.989–0.993 depending on input | 2.8e+01 |
| `w2` viewed `(E, H, F)` | −0.0006 | 3.2e+01 |

The bias bug passes *any* PCC threshold. The renormalization bug lands right around 0.99 and
whether it passes a 0.99 gate depends on the input. **Both must be gated on max-abs**, which
is what the tests do.

### 3.3 Expert weight orientation

`w1` and `w2` are both `[E·3072, 768]`, expert axis **outer** — expert `e` owns rows
`e·3072 … (e+1)·3072`. Both are stored `[F, H]` per expert, so:

```python
x1  = x @ w1[e].T          # w1[e] is [3072, 768]  -> transposed
out = gelu(x1) @ w2[e]     # w2[e] is [3072, 768]  -> NOT transposed
```

This is the **silent-failure** case. `E·F·H` is symmetric in F and H, so viewing `w2` as
`(E, H, F)` succeeds, the downstream matmul typechecks, and nothing raises. The output is
uncorrelated noise (PCC −0.0006). Only a numerical check catches it.

### 3.4 Rotary: NeoX halves, cos/sin cached at half width

`rotate_half` splits the last axis in half: `(x1, x2) → (-x2, x1)`. Not the GPT-J/interleaved
even-odd pairing (`rotary_emb_interleaved` is `false`).

The cache holds `[S, 32]` and is widened at apply time by **concatenation** —
`torch.cat([cos, cos])`, upstream's `repeat(cos, "... d -> ... 1 (2 d)")`. Using
`repeat_interleave` instead gives the GPT-J lane layout, which composed with NeoX
`rotate_half` **is not a rotation at all**: it stops preserving the per-plane norm, and scores
PCC 0.61 against the correct result.

### 3.5 `Wqkv` is three-major

`[q(768) | k(768) | v(768)]`, heads contiguous within each block — upstream's
`rearrange(qkv, "... (three h d) -> ... three h d", three=3)`, where `three` is the **outer**
factor. A head-major reading (`(h, three, d)`) strides 192 across a layout whose blocks are 768
wide, so its "q" slice straddles all three blocks.

### 3.6 GELU is exact-erf, not tanh

`activation_function: "gelu"` maps to `nn.GELU(approximate="none")`. The tanh approximation
differs by **4.7e-4** — small enough to pass a loose PCC gate, large enough to be misread later
as a device precision problem.

### 3.7 The `<pad>` embedding row is not zero

`nn.Embedding(padding_idx=1)` zeroes row 1 at *init*; loading the checkpoint overwrites it, and
the trained row has absmax ~1.5e-2. **Do not pass `padding_idx` on device** — it would zero a
row that upstream uses.

### 3.8 Post-norm, not pre-norm

`h = norm1(attn(x) + x)`, `y = norm2(mlp(h) + h)` — residual added *before* the norm. Every
sub-block output is re-centred, which is why numerical error does not compound over the 12
layers the way it does in a pre-norm decoder. Verified structurally by zeroing each branch and
checking the block collapses to `norm2(norm1(x))`.

### 3.9 The MoE `attention_mask` is inverted and ignored

Upstream's block passes `torch.where(attention_mask.squeeze() == 0, 1, 0)` into the MoE layer
— a mask in which **1 means pad** — and `NomicMoELayer.forward` then ignores it entirely.
Applying it would zero the real tokens. The vendored reference does not thread it through at
all.

---

## 4. The transformers native-class trap

`transformers` ≥ 5 ships a **native** `transformers.models.nomic_bert` targeting
**nomic-embed-text-v1.5**: separate q/k/v/o projections, no biases, SwiGLU
`gate_proj`/`up_proj`/`down_proj`, no MoE. It is registered for `model_type == "nomic_bert"`,
which is exactly what this checkpoint declares.

Measured on transformers 5.12.1 at the pinned revision:

| call | resolves to | consequence |
|---|---|---|
| `AutoConfig.from_pretrained(MODEL_ID)` | native config class | **mild** — every `config.json` field survives except `use_cache` |
| `AutoModel.from_pretrained(MODEL_ID)` | native model class | **severe, and it does not raise** |

The model case reports every MoE tensor (`mlp.experts.mlp.w1/w2`, `mlp.router.layer.weight`,
`mlp.experts.bias`), every `mlp.fc1/fc2` and all q/k/v/o biases as **UNEXPECTED** — silently
discarded — and `gate_proj`/`up_proj`/`down_proj` as **MISSING**, i.e. randomly initialised.
It returns a working 136-parameter model with no MoE that computes finite, plausible, entirely
wrong numbers.

**Containment:** always pass `trust_remote_code=True` *and* `code_revision`, then assert the
resolved class's module starts with `transformers_modules`. The assertion is the part that
matters — see `hf_reference.py` and its test. Without it, a future release that changes
resolution order would silently downgrade the golden reference.

The tokenizer half of this is **inert**: `tokenizer_config.json`'s explicit `tokenizer_class`
outranks the model-type mapping, so `AutoTokenizer` is safe. There is a canary on that
precedence anyway.

---

## 5. Parity with upstream

The vendored reference is **bit-exact** with the upstream HF model at the pinned revisions —
max-abs `0.0`, PCC `1.0000000000`, at all 13 capture points (`emb_ln` plus each of the 12
blocks) and end to end.

Two upstream behaviours are deliberately **not** reproduced:

- Upstream *requires* `attention_mask` and raises `AttributeError` without one. The reference
  defaults it to all-ones.
- Upstream's `matryoshka_dim` slices `sequence_output[:, :matryoshka_dim]` — the **sequence**
  axis, dropping tokens while keeping 768-wide features. That is a bug; truncation belongs
  after pooling, on the feature axis, and lives in `pipeline.py`.

### Per-layer activation magnitudes

Real text, S=22. Useful for setting Phase 1 tolerances — a relative tolerance calibrated on
layer 0 would be far too loose from layer 1 onwards.

| | emb_ln | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kind | — | dense | **MoE** | dense | **MoE** | dense | **MoE** | dense | **MoE** | dense | **MoE** | dense | **MoE** |
| absmax | 3.5 | 7.6 | 21.2 | 22.0 | 23.6 | 21.1 | 20.9 | 17.7 | 15.4 | 14.1 | 15.3 | 18.3 | 8.3 |
| std | 0.23 | 0.42 | 0.63 | 0.73 | 0.65 | 0.69 | 0.69 | 0.71 | 0.67 | 0.68 | 0.66 | 0.70 | 0.69 |

---

## 6. The embedding pipeline

From the checkpoint's own `modules.json`, `1_Pooling/config.json` and
`config_sentence_transformers.json`:

```
task prefix ─► tokenize ─► encoder ─► mask-weighted mean pool ─► [truncate] ─► L2 normalize
```

**Task prefixes are mandatory** — the model was trained with them and dropping one measurably
moves the embedding: `search_query: `, `search_document: `, `classification: `,
`clustering: ` (note the trailing space on each). `include_prompt` is true, so prefix tokens
are pooled too.

Pooling is **mean**, not CLS. Padded positions must be excluded — `<pad>` carries a non-zero
embedding, so including it would make a sentence's embedding depend on its batch-mates.

**Matryoshka ordering is a free choice.** Truncate-then-normalize and normalize-then-truncate
give different norms (1.0 vs ~0.57 at d=256) but identical directions, and the declared
similarity function is cosine — so the two are equivalent for every intended use. Pinned as a
lemma in the tests rather than left to be rederived.

**Model card check:** cosine similarity between passage-prefixed `"Hello!"` and `"¡Hola!"`
reproduces at **0.911788** against the card's 0.9118.

---

## 7. Operator mapping for the TTNN port

Phase 1 target, for reference — **not implemented in this PR**. Correctness only: `bfloat16`,
`TILE_LAYOUT`, `DRAM_MEMORY_CONFIG`, no sharding.

| Reference | TTNN |
|---|---|
| `word_embeddings` | `ttnn.embedding(ids, weight, layout=TILE)` — **no `padding_idx`** (§3.7) |
| `token_type_embeddings` | fold to a constant: `type_vocab_size == 1`, so pre-add the `[1,768]` row into the table in fp32 at load |
| `LayerNorm` + residual | `ttnn.layer_norm(x, epsilon=1e-5, weight=…, bias=…, residual_input_tensor=h)` — 2 torch ops → 1 |
| fused `Wqkv` | `ttnn.linear(h, W, bias=b)` with `W = cat([Wq.T, Wk.T, Wv.T], -1)` → `[768, 2304]` |
| three-major split | `ttnn.experimental.nlp_create_qkv_heads(num_heads=12, num_kv_heads=12, transpose_k_heads=False)` — not GQA, no permutation |
| rotary | `ttnn.experimental.rotary_embedding_hf(x, cos, sin, is_decode_mode=False)`; cos/sin built by **concat**-duplicating `[S,32]` → `[1,1,S,64]` (§3.4) |
| SDPA | `ttnn.transformer.scaled_dot_product_attention(..., is_causal=False, scale=1/8)` — `is_causal` **defaults True**, so passing False is mandatory |
| head concat | `ttnn.transformer.concatenate_heads` |
| GELU | `ttnn.gelu(x)` (default `Accurate`) as its own op — **never** the repo's `fused_activation=(GELU, True)` LUT, whose 2.3e-2 error exceeds the bf16 noise floor |
| router | fp32 `ttnn.linear` → `ttnn.softmax(dim=-1, compute_kernel_config=HiFi4)` → cast bf16 → `ttnn.topk(k=2)` → `ttnn.scatter`. **No sum-normalization** (§3.1) |
| experts | 2 broadcast-batch `ttnn.matmul` + `ttnn.gelu` + `ttnn.permute` + `ttnn.mul` + `ttnn.experimental.fast_reduce_nc(dims=[1])` + `ttnn.add(bias)` |
| padding mask | built once per forward from the tokenizer's `attention_mask` (not `input_ids != pad`), shared across all 12 layers |
| mean pool | `matmul(keep[B,1,1,S], hidden[B,1,S,D])` ÷ `clip(sum(keep), 1., S)` |
| L2 normalize | no single op: `mul → sum → rsqrt(+1e-12) → mul` |

The device-side prerequisites for this mapping (softmax needing an explicit HiFi4 config,
`ttnn.scatter` rejecting fp32, the measured MoE-layer PCC, grid geometry) are recorded in
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §"Device side".

### The MoE formulation that maps onto the device

`NomicExperts.dense_forward` implements this in PyTorch and is asserted equal to the upstream
loop, so it is the bridge between the two:

```python
w1_tt = w1.view(8, 3072, 768).transpose(1, 2)   # [8, 768, 3072]  one transpose
w2_tt = w2.view(8, 3072, 768)                   # [8, 3072, 768]  pure view
# run every token through every expert, weight by a dense [T, 8] routing tensor
# that is zero off the top-2, reduce over the expert axis, then add the shared bias once.
```

Same arithmetic as the reference, no ragged gather/scatter, and two broadcast-batch matmuls on
device.

---

## Open questions

Carried forward to Phase 1; none blocks this PR.

| | Question | Why it is still open |
|---|---|---|
| 1 | Router index agreement between torch fp32 and device bf16 | Top-2 over 8 experts is a **discrete** decision; near-ties can flip. Measured ~99.4% set-agreement in device probing. Needs a gate on agreement plus a margin analysis of the disagreeing tokens, not an exact-match assertion. |
| 2 | End-to-end 12-layer TTNN PCC | Not measurable until the TTNN model exists. Post-norm (§3.8) is the reason to expect it not to compound. |
| 3 | Blackhole DRAM headroom for ~951 MB resident + transient | No documented per-chip figure in-repo; confirm by allocation at Phase 1 step 1. |
| 4 | `fp32_dest_acc_en` on BH expert matmuls | Taken from in-repo findings (#49068), not independently reproduced here. Defaulting it **off** for matmuls is the safe side either way. |
