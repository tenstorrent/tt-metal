# Nomic Embed Text v2 MoE: aten to TTNN

Maps the operator inventory in [`ARCHITECTURE.md` section 4](ARCHITECTURE.md#4-operator-inventory)
onto TTNN, with the settings each operator runs under and the PCC it reaches.

Measured on a Blackhole p300c, bfloat16 unless stated, weights read from the pinned
checkpoint. Section 1 is uniformly at `B=2, S=512` (`T=1024`); section 5 states its own shapes,
since several controls only bite off-tile. The tests gate these rather than print them:

```bash
pytest models/experimental/nomic_embed_text_v2_moe/tests/pcc/test_ttnn_operators.py \
       models/experimental/nomic_embed_text_v2_moe/tests/pcc/test_ttnn_operators_attention.py \
       models/experimental/nomic_embed_text_v2_moe/tests/pcc/test_ttnn_operators_moe.py -v
```

Settings live in [`../tt/model_config.py`](../tt/model_config.py): one compute kernel config
throughout, HiFi4 with fp32 destination accumulation, with the measurements behind each half in
its module docstring. Tensor preparation is in [`../tt/common.py`](../tt/common.py).

## 1. Mapping

Gate is PCC >= 0.999 per operator. SDPA sits closest at 0.99976; everything else clears
0.99999.

### Embeddings and normalization

| aten | TTNN | PCC | max-abs |
|---|---|---|---|
| `embedding` | `ttnn.embedding` | 0.999999 | 1.95e-03 |
| `native_layer_norm` | `ttnn.layer_norm` | 0.999996 | 7.01e-02 |
| `add` + `native_layer_norm` | `ttnn.layer_norm(residual_input_tensor=)` | 0.999996 | 6.22e-02 |
| `_to_copy` | `ttnn.typecast`, fp32 to bf16 | 0.999999 | 1.54e-02 |
| `view`, `_unsafe_view` | `ttnn.reshape` | exact | 0 |

### Projections

`aten.t` is folded into weight preparation by `transpose_linear_weight`: the checkpoint stores
`[out, in]` and `ttnn.linear` wants `[in, out]`. The router is the one bias-free projection, so
it lowers from `mm` rather than `addmm`.

| aten | TTNN | PCC | max-abs |
|---|---|---|---|
| `addmm` | `ttnn.linear`, `attn.Wqkv` 768 to 2304 | 0.999996 | 5.85e-02 |
| `addmm` | `ttnn.linear`, `attn.out_proj` 768 to 768 | 0.999996 | 7.91e-02 |
| `addmm` | `ttnn.linear`, `mlp.fc1` 768 to 3072 | 0.999996 | 8.53e-02 |
| `addmm` | `ttnn.linear`, `mlp.fc2` 3072 to 768 | 0.999996 | 9.75e-02 |
| `mm` | `ttnn.linear`, `mlp.router.layer` 768 to 8, fp32, no bias | 1.000000 | 4.30e-03 |

### Attention

| aten | TTNN | PCC | max-abs |
|---|---|---|---|
| `view` + `select` + `permute` | `ttnn.experimental.nlp_create_qkv_heads` | 0.999999 | 1.48e-02 |
| `cat` + `neg` + `split` + `mul` | `ttnn.experimental.rotary_embedding_hf` | 0.999994 | 4.29e-02 |
| `bmm` + `mul` + `_safe_softmax` + `bmm` | `ttnn.transformer.scaled_dot_product_attention` | 0.999758 | 1.73e-02 |
| same, 25% padding, kept rows | same, additive mask | 0.999756 | 2.02e-02 |
| `permute` + `reshape` | `ttnn.experimental.nlp_concat_heads` | 0.999999 | 1.53e-02 |

### Dense FFN

| aten | TTNN | PCC | max-abs |
|---|---|---|---|
| `gelu` | `ttnn.gelu`, accurate | 0.999996 | 1.61e-02 |

### Router

| aten | TTNN | PCC | max-abs |
|---|---|---|---|
| `_softmax` | `ttnn.softmax`, HiFi4 + fp32 dest acc | 1.000000 | 1.53e-03 |
| `topk` | `ttnn.topk`, k=2 over 8, fp32 | exact indices | 0 |
| dense routing weights | `ttnn.zeros_like` + `ttnn.scatter` | 0.999998 | 1.95e-03 |

The last row has no aten counterpart: it implements `NomicRouter.dense_weights`, which the trace
never reached because that forward took the ragged path. The inventory's own `zeros_like` and
`scatter_` belong to that path and are eliminated (section 2).

### Experts

Cumulative: each row consumes the previous row's device output, so the last figure is the whole
dense-all-experts chain. The tests isolate each operator instead.

| aten | TTNN | PCC | max-abs |
|---|---|---|---|
| `matmul` | `ttnn.matmul`, `[1,1,T,768] x [1,8,768,3072]` | 0.999996 | 7.65e-02 |
| `gelu` | `ttnn.gelu`, `[1,8,T,3072]` | 0.999993 | 7.50e-02 |
| `matmul` | `ttnn.matmul`, `[1,8,T,3072] x [1,8,3072,768]` | 0.999994 | 4.94e-01 |
| `mul` | `ttnn.multiply`, gate `[1,8,T,1]` broadcast | 0.999991 | 2.61e-01 |
| `sum` over experts | `ttnn.experimental.fast_reduce_nc(dims=[1])` | 0.999992 | 2.88e-01 |
| `add` | `ttnn.add`, shared bias, once after the sum | 0.999991 | 2.95e-01 |

### Pooling and output

Not in the aten inventory, which covers the backbone only. From
[`ARCHITECTURE.md` section 5](ARCHITECTURE.md#5-embedding-pipeline).

| stage | TTNN | PCC | max-abs |
|---|---|---|---|
| mask-weighted mean pool | `ttnn.multiply` + `ttnn.sum` + `ttnn.divide` | 0.999997 | 5.85e-04 |
| Matryoshka truncation | `ttnn.slice` on the feature axis | 1.000000 | 0 |
| L2 normalize | `ttnn.multiply` + `ttnn.sum` + `ttnn.rsqrt` | 0.999995 | 8.30e-04 |

## 2. Operators with no TTNN equivalent

None require a fallback. Each is eliminated by a change of formulation, folded into host-side
preparation, absorbed into a fused device op, or a no-op on this path.

| aten | Disposition |
|---|---|
| `nonzero`, `index`, `index_add_`, `_local_scalar_dense`, `max`, `min`, `unbind`, `zeros_like`, `scatter_` | Eliminated. From upstream's ragged expert loop, which gathers each expert's tokens by value. `NomicExperts.dense_forward` replaces them with two broadcast-batch matmuls, a multiply and a reduce; `test_dense_forward_matches_the_ragged_loop` proves the two agree. |
| `t` | Host. `transpose_linear_weight` reorients each `nn.Linear` weight once at load. |
| `arange`, `cos`, `sin`, `slice` | Host. The rotary tables are built once by `rotary_tables`, already trimmed to seqlen, so the reference's `cos[:seqlen]` has nothing to do on device. |
| `rsub`, `mul`, `unsqueeze` on the mask | Host. `additive_attention_mask` turns the (B, S) keep-mask into the additive `(B, 1, S, S)` form. |
| `stack`, `split`, `neg`, `cat` | Absorbed into `rotary_embedding_hf`. |
| `mul.Scalar`, `transpose`, `_safe_softmax` | Absorbed into SDPA, which applies its own `1/sqrt(head_dim)` scale. |
| `alias`, `clone`, `expand` | No-ops on this path. |
| `zeros` | Folded away. The `token_type_ids` default, `torch.zeros(S)`. `type_vocab_size` is 1, so the lookup returns one constant row per token and collapses into a bias on the word embedding. |

All 42 inventory rows are accounted for by this table and section 1, over 41 distinct names:
`mul` appears as both `mul.Tensor` and `mul.Scalar`.

## 3. API differences

| Operator | Difference |
|---|---|
| `ttnn.linear` | Wants `[in, out]`; torch stores `[out, in]` and applies `x @ w.T`. Forgetting the transpose raises on four of the five projections, but `attn.out_proj` is 768x768, so it typechecks and returns noise (PCC 0.001 to 0.004). |
| `ttnn.layer_norm` | `residual_input_tensor=` fuses the post-norm residual add. All 24 encoder norms use it. |
| `ttnn.embedding` | Index tensor must be `uint32` in `ROW_MAJOR`; `layout=` selects the output layout. `padding_idx=` does not zero the row on this path, so it is neither a hazard nor a safeguard. |
| `ttnn.transformer.scaled_dot_product_attention` | `is_causal` defaults to `True`; torch's defaults to `False`. Omitting it on this encoder applies a decoder mask and drops PCC to 0.44. |
| `ttnn.topk` | Returns `(values, indices)`; index dtype follows the input, `uint32` from fp32 and `uint16` from bf16. Both feed `ttnn.scatter` directly. |
| `ttnn.experimental.fast_reduce_nc` | Replaces `sum` over dim 0, 1 or both, keeping the reduced dim at size 1. Returns the **tile-padded** row count: `T=74` gives 96 rows, the trailing 22 zero, so slice back to `T`. `ttnn.sum` reaches the same PCC without the quirk. |
| `ttnn.experimental.rotary_embedding_hf` | Prefill mode needs a leading batch of 1, so `(B, A, S, D)` is folded to `(1, B*A, S, D)`. cos/sin are `(1, 1, S, D)` and broadcast over heads. |
| `ttnn.experimental.nlp_create_qkv_heads` | Takes `(B, 1, S, 3H)` and returns all three heads at once. `transpose_k_heads=False`, since SDPA wants K as `(B, A, S, D)`. |

## 4. Shape, dtype and layout constraints

- Tiles are 32x32 only. TinyTile is broken on Blackhole (#31385).
- Canonical activation layout is `(1, 1, B*S, H)`. Attention and pooling need `(B, 1, S, H)`:
  attention mixes tokens along S, pooling reduces along it, and both would cross the batch
  boundary if it were flattened away. The reshape between the two is exact, and free only when
  B is 1 or S is a multiple of 32.
- `rotary_embedding_hf` requires a padded head_dim of 32 or a multiple of 64. This model's 64
  qualifies.
- **SDPA rejects a `(B, 1, 1, S)` mask** with `mask_shape[2] == q_shape[2]`. Torch broadcasts
  that shape over queries; ttnn does not. Only the head axis may stay singleton, so
  `additive_attention_mask` materialises `(B, 1, S, S)`. At `B=2, S=512` that is 1 MB.
- **The mask's tile padding must be dtype-min, not the 0 a TILE conversion defaults to.** S
  rounds up to a multiple of 32 and 0 means "attend here", so SDPA counts the pad columns in
  the softmax denominator. At `S=37` that took the output norm to 0.69x. Build the mask in
  ROW_MAJOR and convert with `ttnn.to_layout(..., pad_value=finfo.min)`.
- `ttnn.scatter` rejects fp32 in both TILE and ROW_MAJOR
  (`!(input_dtype == DataType::FLOAT32 && input_layout == Layout::TILE)`). Only the destination
  and the scattered values need casting; the index does not.
- `ttnn.topk` accepts fp32, bf16 and `bfloat8_b`, but is only correct on the first two.
- The MoE transient is `(1, E, T, F)`, about 50 MB at `T=1024` in bf16. It scales with batch
  times sequence length, not sequence length alone.

## 5. Negative controls

Nine ways to get an operator wrong that produce finite, plausible output. Each has a test.

| Control | Measured | Test |
|---|---|---|
| `ttnn.softmax` without the full config | max-abs 2.7e-02 to 3.0e-02 stock, 5.4e-03 to 6.8e-03 with HiFi4 alone, 1.4e-03 to 1.9e-03 with both. Budget is 5e-03, so HiFi4 on its own does not reach it, though only just. `numeric_stable=True` measures identically to stock. | `test_softmax_needs_both_hifi4_and_fp32_accumulation` |
| `ttnn.gelu(fast_and_approximate_mode=True)` | max-abs 2.34e-02 in fp32 vs 1.19e-06 accurate. Above the bf16 noise floor of 1.6e-02, so it is not swamped by it. The repo's BERT idiom `fused_activation=(ttnn.UnaryOpType.GELU, True)` selects this LUT. | `test_gelu_fast_mode_is_worse_than_the_bfloat16_noise_floor` |
| Head-major Wqkv view | PCC 0.082 | `test_head_major_qkv_view_is_decorrelated` |
| Attention mask left with 0 tile padding | Norm 0.69x at `S=37`, 0.96x at `S=513`. PCC is near-blind to it, moving 0.9998 to 0.9974 at `S=37` and not at all at `S=513`, so this is gated on the norm. | `test_all_ones_mask_matches_no_mask` |
| SDPA left at its default `is_causal=True` | PCC 0.44. Keeps more correlation than the layout controls, since it averages a subset of the same values rather than the wrong ones. | `test_causal_attention_is_decorrelated` |
| Interleaved (GPT-J) rotary tables | PCC 0.209 | `test_interleaved_rotary_tables_are_decorrelated` |
| Rounding router probabilities to bf16 before `topk` | Reroutes 0.34% to 0.59% of tokens. bf16 quantizes an 8-wide row coarsely enough to turn near-ties into exact ties, which torch and ttnn order differently. | `test_rounding_probabilities_to_bfloat16_before_topk_flips_routing` |
| `ttnn.topk` on `bfloat8_b` | Accepted, 1010/1024 rows wrong. A tile-wide shared exponent flattens an 8-wide probability row. | `test_topk_on_bfloat8_b_is_silently_wrong` |
| Shared expert bias added inside the loop | PCC 0.9999+, so only max-abs sees it. | `test_shared_bias_must_be_added_after_the_weighted_sum` |

Two more raise instead, and are asserted as such:

| Control | Error | Test |
|---|---|---|
| `w2` viewed `(E, H, F)` | `a_shape[-1] == b_shape[-2]`. In torch this view succeeds and computes noise, because `E*F*H` is symmetric in F and H. Packing to `(1, E, F, H)` moves the mistake into the matmul's inner-dimension check. | `test_transposed_expert_weights_are_a_shape_error` |
| `ttnn.scatter` on fp32 | `!(input_dtype == DataType::FLOAT32 && input_layout == Layout::TILE)` | `test_scatter_rejects_float32` |
