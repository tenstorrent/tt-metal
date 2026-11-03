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
- the dense projections, from 32 tile rows of M, through this port's own matmuls: QKV while M <= N
  through `grouped_matmul_w1` with one expert, the others through `pirotski_matmul`
  (`ttnn.experimental.minimal_matmul` only where its buffers do not fit); below 32 tile rows through
  `ttnn.linear` with a multicast config. fc1 runs `grouped_matmul_w1` with one expert at every size.
  Their outputs go to L1 when they fit;
- the experts routed, each token through its two experts only: `routing_metadata` sorts the top-2
  slots by expert into 32-row regions, `routed_combine` without weights gathers each slot's token row,
  `grouped_matmul_w1` runs each 32-row block through its expert's w1 with the GELU fused,
  `grouped_matmul_w2` runs its w2 where `grouped_matmul_w1` wrote its input in w2's layout (above 512
  tokens on the 11x10 grid, where that layout fits L1) and `grouped_matmul_w1`'s transposed path
  elsewhere, and `routed_combine` weights and sums each token's two rows and adds the shared bias.
  `routing_metadata`, `routed_combine`, `grouped_matmul_w1`, `grouped_matmul_w2` and `pirotski_matmul`
  are this port's own ops, in `ttnn.experimental.nomic_embed_text_v2_moe`.

## 1. Mapping

Gate is PCC >= 0.999 per operator. The routed expert pass, on bfloat8_b weights and intermediates
with its w2 at LoFi, and SDPA sit closest, at 0.99990 and 0.99976; everything else clears 0.99999.

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
it lowers from `mm` rather than `addmm`; it runs in one op with its softmax (Router below).

| aten | TTNN | PCC | max-abs |
|---|---|---|---|
| `addmm` | `grouped_matmul_w1` (one expert), `attn.Wqkv` 768 to 2304 | 0.999992 | 1.02e-01 |
| `addmm` | `pirotski_matmul`, `attn.out_proj` 768 to 768 | 0.999996 | 5.69e-02 |
| `addmm` | `grouped_matmul_w1` (one expert), `mlp.fc1` 768 to 3072 | 0.999992 | 1.09e-01 |
| `addmm` | `pirotski_matmul`, `mlp.fc2` 3072 to 768 | 0.999991 | 1.97e-01 |

The four biased rows are measured at `B=2, S=512`, 32 tile rows of M, the first where all four run this port's own
matmuls (layer 0's weights, a randn input, as `test_linear` calls them); below it QKV, out_proj and fc2 run
`ttnn.linear`, and fc1 `grouped_matmul_w1` at every size.
At `B=4, S=512` the same ops measure 0.999991 to 0.999996, max-abs 6.69e-02 to 1.97e-01.

### Attention

| aten | TTNN | PCC | max-abs |
|---|---|---|---|
| `view` + `select` + `permute` | `ttnn.experimental.nlp_create_qkv_heads` | 0.999999 | 1.48e-02 |
| `cat` + `neg` + `split` + `mul` | `ttnn.experimental.rotary_embedding_llama`, on pair-permuted q and k | 0.999994 | 4.14e-02 |
| `bmm` + `mul` + `_safe_softmax` + `bmm` | `ttnn.transformer.scaled_dot_product_attention` | 0.999758 | 1.73e-02 |
| same, 25% padding, kept rows | same, additive mask | 0.999756 | 2.02e-02 |
| `permute` + `reshape` | `ttnn.experimental.nlp_concat_heads`, standalone in `test_nlp_concat_heads`; the model folds it into SDPA (`output_concat_heads=True`, `tt/attention.py`) | 0.999999 | 1.53e-02 |

### Dense FFN

| aten | TTNN | PCC | max-abs |
|---|---|---|---|
| `gelu` | `ttnn.gelu`, accurate | 0.999996 | 1.61e-02 |

### Router

| aten | TTNN | PCC | max-abs |
|---|---|---|---|
| `mm` + `_softmax` | `router_front_end`: `mlp.router.layer` 768 to 8, bf16 input, fp32 weight, K in one block in the fp32 destination, then `ttnn.softmax`'s steps over the 8 experts, HiFi4 + fp32 dest acc; it also writes the input's rows for the dispatch | 1.000000 | 1.93e-03 |
| `topk` | `routing_metadata(scores=..., top_k=2)`: each token's top 2 of the experts' 8 columns, fp32, a tie to the lower column | exact: the metadata of `ttnn.topk`'s selection, bit for bit | 0 |

Measured at `B=2, S=512` against float64, layer 1's weight, a randn input. The scores are 8 columns
of a padded tile: the softmax masks the other 24 with -inf, as `ttnn.softmax` does for a width off
the tile grid, so they come out as exact zeros. They are bit for bit what `ttnn.linear` with K in one
block and `ttnn.softmax` return under the same config (`test_router_front_end.py`, nightly), and the
rows are the input's bits. `routing_metadata` selects from the experts' columns itself, in `ttnn.topk`'s order
(`test_routing_metadata.py`, `test_ttnn_router.py::test_routing_metadata_from_the_router`), and
copies the chosen probabilities as they are.

### Experts

The routed pass runs upstream's ragged loop itself rather than its dense equivalent. Each op is checked against its own definition by the nightly tests in
`tests/ttnn/nightly/unit_tests/operations/experimental/nomic_embed_text_v2_moe/`, and the pass
against the reference's loop in `test_ttnn_experts.py`:

| aten | TTNN | Check |
|---|---|---|
| `nonzero`, `index`, `_local_scalar_dense` | `routing_metadata` from the router's scores, then `routed_combine` without weights or bias gathering each slot's row; no count is read back to the host | exact: `test_routing_metadata.py`, `test_dispatch_copies_each_slot_row` |
| `matmul`, `gelu`, per expert | `grouped_matmul_w1`, w1 with the GELU fused | bit-identical to a stock `ttnn.matmul` of each expert's rows with the tanh GELU; the polynomial GELU the model runs (`GELU_TANH` parameter 2) within its error of the accurate form: `test_grouped_matmul_w1.py` |
| `matmul`, per expert | `grouped_matmul_w2`, or `grouped_matmul_w1`'s transposed path where w1 did not write w2's layout; w2 read transposed, rows written row-major | bit-identical to each other and to a stock `ttnn.matmul` of each expert's rows: `test_grouped_matmul_w2.py`, `test_grouped_matmul_w1.py` |
| `mul`, `index_add_`, `add` | `routed_combine`, the shared bias added once | exact at one-hot weights: `test_routed_combine.py`, `test_combine_returns_each_slot_row` |
| the whole loop | the four above | PCC 0.99990 at worst over seeds 0..7, a token's error at most 2.59e-02 of its norm, the outputs scaled by 0.9886 (w2 at LoFi): `test_routed_pass_with_device_experts` |

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
| `nonzero`, `index`, `index_add_`, `_local_scalar_dense`, `max`, `min`, `unbind`, `zeros_like`, `scatter_` | From upstream's ragged expert loop, which gathers each expert's tokens by value. The routed pass runs the loop itself (section 1), its indices computed on the device. `NomicExperts.dense_forward`, the loop's dense equivalent, is the module tests' oracle; `test_dense_forward_matches_the_ragged_loop` proves the two agree. |
| `t` | Host. `transpose_linear_weight` reorients each `nn.Linear` weight once at load. |
| `arange`, `cos`, `sin`, `slice` | Host. The rotary tables are built once by `rotary_tables`, already trimmed to seqlen, so the reference's `cos[:seqlen]` has nothing to do on device. |
| `rsub`, `mul`, `unsqueeze` on the mask | Host. `additive_attention_mask` turns the (B, S) keep-mask into the additive `(B, 1, S, S)` form. |
| `stack`, `split`, `neg`, `cat` | Absorbed into `rotary_embedding_llama`, which rotates each adjacent pair of head dims through a 32x32 matmul. |
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
| `ttnn.linear` | A fused bias rounds every fp32 output, not only the biased ones: a zero bias on the router's 8 experts moved their logits by up to 3e-3. |
| `ttnn.slice` | A column at a non-tile-aligned offset goes through untilize, a row-major slice and tilize: 49 us for column 1 of a `(4096, 2)` top-k output, against 2 to 5 us at offset 0. |
| `ttnn.experimental.rotary_embedding_llama` | Rotates adjacent pairs (2i, 2i + 1), where the reference rotates i with i + 32. `Wqkv`'s q and k rows are permuted at load (new 2i = old i, new 2i + 1 = old i + 32, `rotary_pair_permutation`) and the tables are pair-interleaved, `[c0, c0, c1, c1, ...]`. The rotated q and k are then bit-identical to `rotary_embedding_hf`'s on the unpermuted ones; SDPA sums q.k in another order. Prefill splits work over batch and sequence tiles only, so `(B, A, S, D)` runs as the view `(B*A/k, k, S, D)`, a free reshape that rotates every row alike, with k from `tt/attention.py::rotary_heads_per_view`: at S <= 128 from 32 tile rows of M (B x ceil(S / 32)), the largest divisor of A that leaves each core at most 8 rows (sequence tiles x heads); otherwise 1 below 96 tile rows and A, the native view, from 96. cos/sin are `(1, 1, S, D)` and broadcast. The 32x32 matrix is sharded one copy a core, which the prefill program reads in place. |
| `ttnn.experimental.nlp_create_qkv_heads` | Takes `(B, 1, S, 3H)` and returns all three heads at once. `transpose_k_heads=False`, since SDPA wants K as `(B, A, S, D)`. |

## 4. Shape, dtype and layout constraints

- Tiles are 32x32 only. TinyTile is broken on Blackhole (#31385).
- Activations cross block boundaries as `(B, 1, S, H)`; the flat `(1, 1, B*S, H)` form is taken
  inside `tt/moe.py` only, where the expert matmuls require it. Attention mixes tokens along S
  and pooling reduces along it, so both would cross the batch boundary if it were flattened
  away. The reshape between the two is exact, and free only when B is 1 or S is a multiple
  of 32.
- `rotary_embedding_llama` takes bfloat16 only, for the input, the tables and the matrix, and an
  interleaved input in prefill. With fp32 accumulation head_dim may be at most 128; this model's 64
  qualifies.
- **SDPA rejects a `(B, 1, 1, S)` mask** with `mask_shape[2] == q_shape[2]`. Torch broadcasts
  that shape over queries; ttnn does not. Only the head axis may stay singleton, so
  `additive_attention_mask` materialises `(B, 1, S, S)`. At `B=2, S=512` that is 1 MB.
- **The mask's tile padding must be dtype-min, not the 0 a TILE conversion defaults to.** S
  rounds up to a multiple of 32 and 0 means "attend here", so SDPA counts the pad columns in
  the softmax denominator. At `S=37` that took the output norm to 0.69x. Build the mask in
  ROW_MAJOR and convert with `ttnn.to_layout(..., pad_value=finfo.min)`.
- `ttnn.topk` accepts fp32, bf16 and `bfloat8_b`, but is only correct on the first two.
- The MoE transient is the routed pass's w1 output, `(1, 1, R, F)` with R about 2T rows: 7.4 MB
  at `T=1024` in bfloat8_b, where every token through every expert would take 27 MB. It scales
  with batch times sequence length, not sequence length alone; a window of at most 32768 tokens
  caps it at about 215 MB.
- **`ttnn.matmul` deadlocks on broadcast-batch operands** (`in0_B == 1`, `in1_B > 1`) once the
  block geometry has `num_blocks_h_dim * num_blocks_w_dim > 1`. The in0 reuse path replays
  `num_blocks_inner_dim` blocks per extra batch, dropping the `h_dim * w_dim` factor the compute
  kernel applies, so the reader under-produces and every core waits in `cb_wait_front` on in0.
  At `(1, 1, T, 768) x (1, E, 768, 3072)` on an 11x10 grid, T=3520 passes and T=3552 hangs for
  every E > 1 tried. The bound is joint in M and N: at M=32 tiles, N=3072 passes and N=4096
  hangs. It is a hang rather than an exception, so recovery needs `tt-smi -r`. No program in
  `tt/experts.py` is broadcast-batch any more: the routed pass runs `grouped_matmul_w1` and `grouped_matmul_w2`. `tests/hangRepro` reproduces it standalone.
- **`ttnn.transformer.scaled_dot_product_attention` returns wrong output at some batch shapes.**
  Measured with no program config on random q, k, v against float64 attention, masked and not:
  wrong at 8x264, 8x528, 8x544, 12x352 and 16x528 (errors up to 1e+38), correct at 8x512, 8x384,
  8x256, 4x528, 4x264, 16x512, 6x704 and 3x1408. The failures pair an odd number of 32-token tiles
  per sequence (9, 11 or 17) with 8 or more sequences, and a row's embedding then depends on the
  rest of its batch. q and k chunks of 128 correct B=8 but not 16x528. Explicit chunks fail too
  where a head runs as three query chunks at 10 or more sequences (q128 with a whole-sequence key
  chunk: 10x384, 12x384, 16x384, 12x352). The model never calls SDPA without a program config:
  `tt/attention.py::sdpa_program_config` and `attention_placement` leave out an odd query-chunk
  count above one at 10 or more sequences, and their pairs are right at 8x264, 8x528 and 12x352
  and at the four three-chunk shapes (`test_sdpa_at_shapes_where_its_default_chunks_are_wrong`,
  `test_sdpa_where_three_query_chunks_a_head_are_wrong`).

## 5. Negative controls

Ten ways to get an operator wrong that produce finite, plausible output. Each has a test.

| Control | Measured | Test |
|---|---|---|
| `router_front_end` without the full config | Scores' max-abs 9.6e-03 to 1.1e-02 at HiFi2 with fp32 accumulation, 1.6e-03 to 2.4e-03 at HiFi4 (HiFi3 1.8e-03 to 2.4e-03), at layer 1's weight over 1024 randn tokens, seeds 0..7. Budget is 5e-03. The op refuses a bfloat16 destination; `ttnn.softmax` on its own measured 5.4e-03 to 6.8e-03 there with HiFi4. | `test_router_front_end_needs_hifi4_and_fp32_accumulation` |
| `ttnn.gelu(fast_and_approximate_mode=True)` | max-abs 2.34e-02 in fp32 vs 1.19e-06 accurate. Above the bf16 noise floor of 1.6e-02, so it is not swamped by it. The repo's BERT idiom `fused_activation=(ttnn.UnaryOpType.GELU, True)` selects this LUT. | `test_gelu_fast_mode_is_worse_than_the_bfloat16_noise_floor` |
| Head-major Wqkv view | PCC 0.082 | `test_head_major_qkv_view_is_decorrelated` |
| Attention mask left with 0 tile padding | Norm 0.69x at `S=37`, 0.96x at `S=513`. PCC is near-blind to it, moving 0.9998 to 0.9974 at `S=37` and not at all at `S=513`, so this is gated on the norm. | `test_all_ones_mask_matches_no_mask` |
| SDPA left at its default `is_causal=True` | PCC 0.44. Keeps more correlation than the layout controls, since it averages a subset of the same values rather than the wrong ones. | `test_causal_attention_is_decorrelated` |
| Concat-widened rotary tables, the reference's own widening, under the adjacent-pair kernel | PCC 0.295 | `test_concat_widened_rotary_tables_are_decorrelated` |
| Rounding router probabilities to bf16 before `topk` | Reroutes 0.34% to 0.59% of tokens. bf16 quantizes an 8-wide row coarsely enough to turn near-ties into exact ties, which torch and ttnn order differently. | `test_rounding_probabilities_to_bfloat16_before_topk_flips_routing` |
| `ttnn.topk` on `bfloat8_b` | Accepted, 1010/1024 rows wrong. A tile-wide shared exponent flattens an 8-wide probability row. | `test_topk_on_bfloat8_b_is_silently_wrong` |
| A `bfloat8_b` cast of a tensor with stale tile padding | The padding enters the shared exponents. The permuted expert gate of a dense pass this port no longer runs held stale memory there, and casting it crushed the 12 real gates of the last tile at 300 tokens: module PCC 0.984. Zeroed first with `ttnn.fill_implicit_tile_padding`, which writes in place, the cast stays within 1e-2. | `test_bfloat8_b_cast_reads_the_tile_padding` |
| Shared expert bias added inside the loop | PCC 0.9999+, so only max-abs sees it. | `test_shared_bias_must_be_added_after_the_weighted_sum` |

One more raises instead, and is asserted as such:

| Control | Error | Test |
|---|---|---|
| `w2` viewed `(E, H, F)` | `a_shape[-1] == b_shape[-2]`. In torch this view succeeds and computes noise, because `E*F*H` is symmetric in F and H. Packing to `(1, E, F, H)` moves the mistake into the matmul's inner-dimension check. `TtNomicExperts` stores w2 transposed per expert, the mistake's own shape, so at module level `test_misoriented_w2_decorrelates` catches it by PCC instead. | `test_transposed_expert_weights_are_a_shape_error` |
