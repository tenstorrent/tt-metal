# Nomic Embed Text v2 MoE: aten to TTNN

Maps the operator inventory in [`ARCHITECTURE.md` section 4](ARCHITECTURE.md#4-operator-inventory)
onto TTNN, with the settings each operator runs under and the PCC it reaches.

Measured on a Blackhole p300c, bfloat16 unless stated, weights read from the pinned
checkpoint. Section 1 is at `B=2, S=512` (`T=1024`) unless a line says otherwise; section 5 states
its own shapes, since several controls only bite off-tile. The tests gate these rather than print
them:

```bash
pytest models/experimental/nomic_embed_text_v2_moe/tests/pcc/test_ttnn_operators.py \
       models/experimental/nomic_embed_text_v2_moe/tests/pcc/test_ttnn_operators_attention.py \
       models/experimental/nomic_embed_text_v2_moe/tests/pcc/test_ttnn_operators_moe.py -v
```

Settings live in [`../tt/model_config.py`](../tt/model_config.py): one compute kernel config per
op group, all with fp32 destination accumulation, and each group's fidelity and weight dtype,
with the measurements behind them in its module docstring. Tensor preparation is in
[`../tt/common.py`](../tt/common.py).

The tables below map each aten op to the TTNN op its operator test checks. The matmul tests
call the module code, so they run the program configs chosen per call shape in
[`../tt/matmul_config.py`](../tt/matmul_config.py):
- the dense projections through `ttnn.experimental.minimal_matmul` above 32 tile rows of M, and
  through `ttnn.linear` with a multicast config below. Above, the QKV, out_proj and fc1 outputs go
  to L1 when they fit;
- the experts in one of two layouts per pass of tokens. Up to 128 tokens they stay on the rows,
  with the pass's tensors in L1: w1 through `ttnn.sparse_matmul` with every expert enabled, w2
  through a 1D `ttnn.matmul` that reads its `(E, H, F)` weight with `transpose_b`. Above, the pass
  runs transposed, tokens on the columns: w1 as one unbatched `minimal_matmul` of the stacked
  `(E*F, H)` checkpoint weight and `x^T`, w2 as a batched 2D `ttnn.matmul` writing bfloat8_b, the
  gate cast to match, and the result transposed back.

## 1. Mapping

Gate is PCC >= 0.999 per operator. The expert chain, on bfloat8_b weights and intermediates, and
SDPA sit closest, at 0.99975 to 0.99994 and at 0.99976; everything else clears 0.99999.

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
| `addmm` | `ttnn.linear`, `attn.Wqkv` 768 to 2304 | 0.999996 | 7.06e-02 |
| `addmm` | `ttnn.linear`, `attn.out_proj` 768 to 768 | 0.999996 | 5.69e-02 |
| `addmm` | `ttnn.linear`, `mlp.fc1` 768 to 3072 | 0.999992 | 1.09e-01 |
| `addmm` | `ttnn.linear`, `mlp.fc2` 3072 to 768 | 0.999996 | 1.02e-01 |
| `mm` | `ttnn.linear`, `mlp.router.layer` 768 to 8, bf16 input, fp32 weight and output, no bias | 1.000000 | 3.24e-03 |

At this size M is 32 tile rows, the last that runs `ttnn.linear`. At `B=4, S=512` the four
projections run `minimal_matmul`, three of them into L1, and measure 0.999991 to 0.999996, max-abs
7.06e-02 to 1.48e-01.

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
| `_softmax` | `ttnn.softmax`, HiFi4 + fp32 dest acc, over 64 scored columns | 1.000000 | 1.53e-03 |
| `topk` | `ttnn.topk`, k=2 over the 64, fp32 | exact indices | 0 |
| dense routing weights | `ttnn.matmul` 0/1 one-hot of the indices, `ttnn.eq`, `ttnn.multiply` | 0.999998 | 1.95e-03 |

The router matmul scores 64 columns, the 8 experts and 56 at -1e30, so that `ttnn.topk` has
nothing to pad; softmax turns the 56 into exact zeros and leaves the experts' probabilities
bit-identical to an 8-wide softmax. The dense row is bit-identical to `ttnn.scatter` of the top-k
values into zeros (`test_one_hot_reproduces_the_scatter`), at 27 against 126 us a MoE layer at
8x512. It has no aten counterpart: it implements `NomicRouter.dense_weights`, which the trace never
reached because that forward took the ragged path. The inventory's own `zeros_like` and
`scatter_` belong to that path and are eliminated (section 2).

### Experts

Cumulative: each row consumes the previous row's device output, so the last figure is the whole
dense-all-experts chain. `test_expert_matmuls` chains the first three the same way; the other
tests isolate their operator.

| aten | TTNN | PCC | max-abs |
|---|---|---|---|
| `matmul` | `ttnn.experimental.minimal_matmul`, `[1,1,8*3072,768] x [1,1,768,T]` | 0.999941 | 3.07e-01 |
| `gelu` | `ttnn.gelu`, `[1,1,8*3072,T]` | 0.999860 | 3.09e-01 |
| `matmul` | `ttnn.matmul`, `[1,8,768,3072] x [1,8,3072,T]` | 0.999782 | 5.02e+00 |
| `mul` | `ttnn.multiply`, gate `[1,8,1,T]` broadcast | 0.999746 | 1.61e+00 |
| `sum` over experts | `ttnn.experimental.fast_reduce_nc(dims=[1])` | 0.999815 | 1.65e+00 |
| `add` | `ttnn.add`, shared bias, once after the sum | 0.999814 | 1.64e+00 |

At this size a pass runs transposed, everything before the sum in bfloat8_b. Up to 128 tokens
it runs token-major, w1 through `ttnn.sparse_matmul` and everything from the w2 output on in bf16:
at `B=1, S=128` the rows measure 0.999936, 0.999852, 0.999827, 0.999821, 0.999867 and 0.999866.

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
| `ttnn.topk` | Returns `(values, indices)`; index dtype follows the input, `uint32` from fp32 and `uint16` from bf16. Widens a last dim under 64 to 64 with -inf before its device op, on one core for fp32: 95 us at 4096 rows of 8. |
| `ttnn.linear` | A fused bias rounds every fp32 output, not only the biased ones: a zero bias on the router's 8 experts moved their logits by up to 3e-3. The router adds its padding row with its own `ttnn.add`, which is exact. |
| `ttnn.slice` | A column at a non-tile-aligned offset goes through untilize, a row-major slice and tilize: 49 us for column 1 of a `(4096, 2)` top-k output, against 2 to 5 us at offset 0. |
| `ttnn.experimental.fast_reduce_nc` | Replaces `sum` over dim 0, 1 or both, keeping the reduced dim at size 1. Left to allocate its output, returns the **tile-padded** row count: `T=74` gives 96 rows, the trailing 22 zero. `tt/experts.py` passes an output of the logical shape instead. `ttnn.sum` reaches the same PCC without the quirk. |
| `ttnn.experimental.rotary_embedding_hf` | Prefill mode needs a leading batch of 1, so `(B, A, S, D)` is folded to `(1, B*A, S, D)`. cos/sin are `(1, 1, S, D)` and broadcast over heads. |
| `ttnn.experimental.nlp_create_qkv_heads` | Takes `(B, 1, S, 3H)` and returns all three heads at once. `transpose_k_heads=False`, since SDPA wants K as `(B, A, S, D)`. |

## 4. Shape, dtype and layout constraints

- Tiles are 32x32 only. TinyTile is broken on Blackhole (#31385).
- Activations cross block boundaries as `(B, 1, S, H)`; the flat `(1, 1, B*S, H)` form is taken
  inside `tt/moe.py` only, where the expert matmuls require it. Attention mixes tokens along S
  and pooling reduces along it, so both would cross the batch boundary if it were flattened
  away. The reshape between the two is exact, and free only when B is 1 or S is a multiple
  of 32.
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
- The MoE transient is the w1 output, `(1, E, T, F)` or `(1, 1, E*F, T)` transposed, about 27 MB
  at `T=1024` in bfloat8_b, and the GELU's copy of it. It scales with batch times sequence length,
  not sequence length alone.
- **`ttnn.matmul` deadlocks on broadcast-batch operands** (`in0_B == 1`, `in1_B > 1`) once the
  block geometry has `num_blocks_h_dim * num_blocks_w_dim > 1`. The in0 reuse path replays
  `num_blocks_inner_dim` blocks per extra batch, dropping the `h_dim * w_dim` factor the compute
  kernel applies, so the reader under-produces and every core waits in `cb_wait_front` on in0.
  At `(1, 1, T, 768) x (1, E, 768, 3072)` on an 11x10 grid, T=3520 passes and T=3552 hangs for
  every E > 1 tried. The bound is joint in M and N: at M=32 tiles, N=3072 passes and N=4096
  hangs. It is a hang rather than an exception, so recovery needs `tt-smi -r`. No program in
  `tt/experts.py` is broadcast-batch any more: the token-major w1 is a `sparse_matmul` and the
  transposed one puts the shared input in in1. `tests/hangRepro` reproduces it standalone.
- **`ttnn.transformer.scaled_dot_product_attention` returns wrong output at some batch shapes.**
  Measured on random q, k, v against float64 attention, masked and not: wrong at 8x264, 8x528,
  8x544, 12x352 and 16x528 (errors up to 1e+38), correct at 8x512, 8x384, 8x256, 4x528, 4x264,
  16x512, 6x704 and 3x1408. The failures pair an odd number of 32-token tiles per sequence (9, 11
  or 17) with 8 or more sequences. In the model a row's embedding then depends on the rest of its
  batch. q and k chunks of 128 correct B=8 but not 16x528. Not worked around here.

## 5. Negative controls

Ten ways to get an operator wrong that produce finite, plausible output. Each has a test.

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
| A `bfloat8_b` cast of a tensor with stale tile padding | The padding enters the shared exponents. The permuted expert gate's padding held stale memory, and casting it crushed the 12 real gates of the last tile at 300 tokens: module PCC 0.984. Zeroed first with `ttnn.fill_implicit_tile_padding`, which writes in place, the cast stays within 1e-2. The padding of `x` reaches the bfloat8_b w1 output the same way on a transposed pass, so it is zeroed too. | `test_bfloat8_b_cast_reads_the_tile_padding`, and `test_the_padding_of_x_does_not_reach_the_output` in `test_ttnn_experts.py` |
| Shared expert bias added inside the loop | PCC 0.9999+, so only max-abs sees it. | `test_shared_bias_must_be_added_after_the_weighted_sum` |

Two more raise instead, and are asserted as such:

| Control | Error | Test |
|---|---|---|
| `w2` viewed `(E, H, F)` | `a_shape[-1] == b_shape[-2]`. In torch this view succeeds and computes noise, because `E*F*H` is symmetric in F and H. Packing to `(1, E, F, H)` moves the mistake into the matmul's inner-dimension check. `TtNomicExperts` stores w2 transposed per expert, the mistake's own shape, so at module level `test_misoriented_w2_decorrelates` catches it by PCC instead. | `test_transposed_expert_weights_are_a_shape_error` |
| `ttnn.scatter` on fp32 | `!(input_dtype == DataType::FLOAT32 && input_layout == Layout::TILE)` | `test_scatter_rejects_float32` |
