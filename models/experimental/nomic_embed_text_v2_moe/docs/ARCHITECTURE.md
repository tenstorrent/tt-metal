# Nomic Embed Text v2 MoE: Architecture

Encoder-only multilingual text-embedding transformer; every other layer replaces the dense FFN
with a Mixture-of-Experts FFN. 475M parameters, ~305M active per token. No decoder, no KV
cache, no generation. The reference in [`../reference/modeling_nomic_moe.py`](../reference/modeling_nomic_moe.py) is bit-exact against
upstream at the pinned revisions (PCC 1.0, max-abs 0.0, at all 13 capture points and end to
end).

## 1. Dimensions

Read from the vendored [`../reference/config.json`](../reference/config.json), pinned at the
weights revision and the source of truth; nothing is restated in Python.

| quantity | config.json key | value |
|---|---|---|
| encoder layers | `n_layer` | 12 |
| hidden size | `n_embd` | 768 |
| attention heads | `n_head` | 12 |
| head dimension | `n_embd / n_head` | 64 |
| FFN hidden size | `n_inner` | 3072 |
| experts per MoE layer | `num_experts` | 8 |
| experts routed per token | `moe_top_k` | 2 |
| MoE layer period | `moe_every_n_layers` | 2 |
| vocabulary, padded | `vocab_size` | 250048 |
| max positions | `max_trained_positions` | 2048 |
| LayerNorm epsilon | `layer_norm_epsilon` | 1e-5 |
| rotary base | `rotary_emb_base` | 10000 |

Each block is post-norm, `h = norm1(attn(x) + x)` then `y = norm2(mlp(h) + h)`, so every
sub-block output is re-centred. Attention packs q, k and v into one `Wqkv` of width
`3 * 768 = 2304`, three-major as `[q | k | v]` with heads contiguous inside each block.
Position is rotary over the full head width, GPT-NeoX half-splitting, no learned position
table.

The FFN alternates on `i % 2 == 1`, so layers 1, 3, 5, 7, 9 and 11 route and the even layers
are dense. Note the offset: upstream uses `== 1`, not `== 0`, which puts a dense layer first.
A MoE layer softmaxes over all 8 experts and takes the top 2 without renormalizing, so the
weights sum to less than 1 and the branch is attenuated against the residual.

Tokenizer is `XLMRobertaTokenizerFast`, 250002 tokens, pad 1 / bos 0 / eos 2, shorter than
`vocab_size` (padded to a multiple of 64), so the embedding table has unreachable trailing rows.

## 2. Provenance

Weights, config and tokenizer come from `nomic-ai/nomic-embed-text-v2-moe` at revision
`1066b6599d099fbb93dfcb64f9c37a7c9e503e85`, pinned in [`../common.py`](../common.py) and passed
on every load.

That repository ships no model definition. Its `config.json` `auto_map` sends `transformers` to
a second repository for the classes
(`"AutoModel": "nomic-ai/nomic-bert-2048--modeling_hf_nomic_bert.NomicBertModel"`), which is
fetched from that repository's `main`.

`transformers` >= 5 also ships a native `nomic_bert` registered for this checkpoint's
`model_type` that targets a different model and discards the expert weights without raising.
Load only through `reference/hf_reference.load_hf_model`, which forces the remote code and
asserts what it resolved to. The repo pins `transformers == 5.12.1`, which is the version this
was measured on; the `transformers_version: 4.44.2` inside `config.json` records what upstream
serialized the file with and constrains nothing at runtime.

## 3. Checkpoint contract

148 tensors, 475,292,928 parameters, all `float32`, flat keys. Generated from the config by
`loader.expected_checkpoint_keys` and asserted against the real file, so the generator is what
is under test.

```
embeddings.word_embeddings.weight                   [250048, 768]
embeddings.token_type_embeddings.weight             [1, 768]
emb_ln.{weight,bias}                                [768]

encoder.layers.{0..11}.attn.Wqkv.{weight,bias}      [2304, 768] / [2304]
encoder.layers.{0..11}.attn.out_proj.{weight,bias}  [768, 768]  / [768]
encoder.layers.{0..11}.norm{1,2}.{weight,bias}      [768]

# dense FFN, i in {0, 2, 4, 6, 8, 10}
encoder.layers.{i}.mlp.fc1.{weight,bias}            [3072, 768] / [3072]
encoder.layers.{i}.mlp.fc2.{weight,bias}            [768, 3072] / [768]

# MoE FFN, i in {1, 3, 5, 7, 9, 11}
encoder.layers.{i}.mlp.router.layer.weight          [8, 768]      # no bias
encoder.layers.{i}.mlp.experts.mlp.{w1,w2}          [24576, 768]  # = [8, 3072, 768]
encoder.layers.{i}.mlp.experts.bias                 [768]         # shared by all 8
```

Both expert blocks carry the expert axis outer: expert `e` owns rows `e*3072 .. (e+1)*3072`,
each stored `[F, H]`. Apply as `x @ w1[e].T`, then `gelu(...) @ w2[e]` untransposed. The `[768]`
bias is shared by all eight experts and added once, after the weighted sum. Verified absent,
each by its own test: `position_embeddings`, `pooler`, `cls.`, `lm_head`, `ln_f`, `inv_freq`,
`norm_factor`, router bias, anything vision.

## 4. Operator inventory

42 distinct aten operators, captured with `TorchDispatchMode` over a forward pass on
`input_ids [2, 16]`, ragged padding `[0, 5]`. Shapes are one representative call. SDPA was
pinned to `SDPBackend.MATH`, which decomposes attention into its matmul, scale, softmax and
matmul rather than dispatching one fused kernel; leaving the backend to auto-selection would
make this list a property of the host rather than of the model.

| operator | input shapes | output shape |
|---|---|---|
| `aten.view.default` | `[2, 16, 768]` | `[32, 768]` |
| `aten.select.int` | `[2, 16, 3, 12, 64]` | `[2, 16, 12, 64]` |
| `aten.mul.Tensor` | `[2, 1, 1, 16]` | `[2, 1, 1, 16]` |
| `aten.slice.Tensor` | `[16, 32]` | `[16, 32]` |
| `aten.unsqueeze.default` | `[2, 16]` | `[2, 1, 16]` |
| `aten.t.default` | `[2304, 768]` | `[768, 2304]` |
| `aten.mm.default` | `[32, 768], [768, 8]` | `[32, 8]` |
| `aten._unsafe_view.default` | `[2, 12, 16, 64]` | `[24, 16, 64]` |
| `aten.cat.default` | `[16, 32], [16, 32]` | `[16, 64]` |
| `aten.index.Tensor` | `[32, 768], [16]` | `[16, 768]` |
| `aten.add.Tensor` | `[2, 16, 768], [16, 768]` | `[2, 16, 768]` |
| `aten.permute.default` | `[2, 16, 12, 64]` | `[2, 12, 16, 64]` |
| `aten.alias.default` | `[2, 16, 12, 64]` | `[2, 16, 12, 64]` |
| `aten.clone.default` | `[2, 12, 16, 64]` | `[2, 12, 16, 64]` |
| `aten.expand.default` | `[2, 12, 16, 64]` | `[2, 12, 16, 64]` |
| `aten.nonzero.default` | `[2, 32]` | `[16, 2]` |
| `aten.unbind.int` | `[16, 2]` | `[16]` |
| `aten.gelu.default` | `[2, 16, 3072]` | `[2, 16, 3072]` |
| `aten.addmm.default` | `[2304], [32, 768], [768, 2304]` | `[32, 2304]` |
| `aten.index_add_.default` | `[32, 768], [16], [16, 768]` | `[32, 768]` |
| `aten.native_layer_norm.default` | `[2, 16, 768], [768], [768]` | `[2, 16, 768]` |
| `aten.bmm.default` | `[24, 16, 64], [24, 64, 16]` | `[24, 16, 16]` |
| `aten.mul.Scalar` | `[2, 12, 16, 64]` | `[2, 12, 16, 64]` |
| `aten.neg.default` | `[2, 16, 12, 32]` | `[2, 16, 12, 32]` |
| `aten.split.Tensor` | `[2, 16, 12, 64]` | `[2, 16, 12, 32]` |
| `aten._local_scalar_dense.default` | `[]` | `-` |
| `aten._safe_softmax.default` | `[2, 12, 16, 16]` | `[2, 12, 16, 16]` |
| `aten.arange.default` | `-` | `[16]` |
| `aten.cos.default` | `[16, 32]` | `[16, 32]` |
| `aten.sin.default` | `[16, 32]` | `[16, 32]` |
| `aten.stack.default` | `[2, 16, 12, 64], [2, 16, 12, 64], [2, 16, 12, 64]` | `[2, 16, 3, 12, 64]` |
| `aten.transpose.int` | `[2, 12, 16, 64]` | `[2, 12, 64, 16]` |
| `aten.zeros.default` | `-` | `[16]` |
| `aten._softmax.default` | `[32, 8]` | `[32, 8]` |
| `aten.max.default` | `[32, 2]` | `[]` |
| `aten.min.default` | `[32, 2]` | `[]` |
| `aten.scatter_.value` | `[32, 2, 8], [32, 2, 1]` | `[32, 2, 8]` |
| `aten.topk.default` | `[32, 8]` | `[32, 2]` |
| `aten.zeros_like.default` | `[32, 768]` | `[32, 768]` |
| `aten.embedding.default` | `[250048, 768], [2, 16]` | `[2, 16, 768]` |
| `aten._to_copy.default` | `[2, 1, 1, 16]` | `[2, 1, 1, 16]` |
| `aten.rsub.Scalar` | `[2, 1, 1, 16]` | `[2, 1, 1, 16]` |

`aten.nonzero`, `aten.index`, `aten.index_add_` and `aten._local_scalar_dense` come from
upstream's ragged expert loop, which gathers each expert's tokens by value. Being
data-dependent, the model is not `torch.fx`-traceable.
`NomicExperts.dense_forward` replaces all four with two broadcast-batch matmuls, a multiply and
a reduce. It gives every expert every token, so it materialises
`(num_experts, tokens, ffn_hidden)` per MoE layer, where tokens is batch times sequence length
rather than sequence length alone. `test_dense_forward_matches_the_ragged_loop` asserts the two
formulations agree.

## 5. Embedding pipeline

From the checkpoint's `modules.json`, `1_Pooling/config.json` and
`config_sentence_transformers.json`:

```
task prefix -> tokenize -> model -> mask-weighted mean pool -> [truncate] -> L2 normalize
```

Prefixes are trained-in, not decoration: `search_query: `, `search_document: `,
`classification: `, `clustering: `, each with a trailing space. `include_prompt` is true, so
prefix tokens are pooled too.

Pooling is mean over unpadded positions, not CLS. Padding must be excluded because the `<pad>`
embedding row is not zero (trained absmax 1.499232e-02); including it makes an embedding depend
on its batch-mates. `nn.Embedding(padding_idx=...)` zeroes that row only at init, and loading
the checkpoint overwrites it, so the trained values are what the model actually uses.

Matryoshka truncation happens after pooling, on the feature axis. Truncate-then-normalize and
normalize-then-truncate give different norms (1.0 vs ~0.57 at d=256) but identical directions,
and the declared similarity is cosine, so the ordering is free.
