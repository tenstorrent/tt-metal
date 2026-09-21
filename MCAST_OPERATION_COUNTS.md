# Additional multicast consumer argument counts

Companion to [MCAST_ARGUMENT_REPORT.md](MCAST_ARGUMENT_REPORT.md).
Source states, architecture, workload details, historical-layout derivations,
and limitations are documented there. Pre/v1 counts below are source-calculated;
v2 counts were captured from lowered Programs at `5745f4b6d71` with temporary
measurement hooks. No hooks remain in the implementation. Each CT row is one
distinct kernel variant, including zero-placement variants. `P+N` counts
positional and named CT once; resource bindings/defines are not counted again.

RT entries are `words × placed cores`, including common RT on each core and
zero-RT compute placements. Δ columns give per-core changes and multiplicity.
`H1→H2` shows helper-only RT; the remaining words are operation fields and padding.
All totals are logical uint32 entries, not aligned dispatch bytes. Full v2 CT
values, defines, compile hashes, and counts are in
[MCAST_OPERATION_COUNTS.json](MCAST_OPERATION_COUNTS.json).

## 1. Conv3D compact chain

Factory: `conv3d_program_factory.cpp`.

Captured test: `tests/ttnn/unit_tests/operations/conv/test_conv3d.py::test_conv3d_weight_sharing_groups[compact-chain]`.

| Kernel / variant | Pre CT P+N | v1 CT P+N | v2 CT P+N | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. `reader_vol2col.cpp` | 53+0 | 53+0 | 53+0 | 0 | 0 |
| 2. `compute.cpp` | 31+0 | 31+0 | 31+0 | 0 | 0 |
| 3. `writer.cpp` | 34+0 | 43+2 | 50+2 | +18 | +7 |

| Variant | H1→H2 | Pre RT | v1 RT | v2 RT | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0→0 | 11×64 | 11×64 | 11×64 | 0 | 0 |
| 2 | 0→0 | 12×64 | 12×64 | 12×64 | 0 | 0 |
| 3 | 11→11 | 26×64 | 26×64 | 26×64 | 0 | 0 |

Variants pre/v1/v2: 3/3/3. Aggregate CT: 118 / 129 / 136;
aggregate RT: 3136 / 3136 / 3136. V2 CT Δ vs pre/v1:
+18/+7; RT Δ:
0/0.

## 2. Conv3D compact mixed/idle

Factory: `conv3d_program_factory.cpp`.

Captured test: `tests/ttnn/unit_tests/operations/conv/test_conv3d.py::test_conv3d_weight_sharing_groups[compact-mixed]`.

| Kernel / variant | Pre CT P+N | v1 CT P+N | v2 CT P+N | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. `reader_vol2col.cpp` | 53+0 | 53+0 | 53+0 | 0 | 0 |
| 2. `compute.cpp` | 31+0 | 31+0 | 31+0 | 0 | 0 |
| 3. `writer.cpp` | 34+0 | 43+2 | 50+2 | +18 | +7 |

| Variant | H1→H2 | Pre RT | v1 RT | v2 RT | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0→0 | 11×64 | 11×64 | 11×64 | 0 | 0 |
| 2 | 0→0 | 12×64 | 12×64 | 12×64 | 0 | 0 |
| 3 | 11→11 | 26×4, 30×60 | 30×64 | 30×64 | +4×4, 0×60 | 0 |

Variants pre/v1/v2: 3/3/3. Aggregate CT: 118 / 129 / 136;
aggregate RT: 3376 / 3392 / 3392. V2 CT Δ vs pre/v1:
+18/+7; RT Δ:
+16/0.

## 3. Conv3D rectangular/passive

Factory: `conv3d_program_factory.cpp`.

Captured test: `tests/ttnn/unit_tests/operations/conv/test_conv3d.py::test_conv3d_weight_sharing_groups[rectangular-passive]`.

| Kernel / variant | Pre CT P+N | v1 CT P+N | v2 CT P+N | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. `reader_vol2col.cpp` | 53+0 | 53+0 | 53+0 | 0 | 0 |
| 2. `compute.cpp` | 31+0 | 31+0 | 31+0 | 0 | 0 |
| 3. `writer.cpp` | 34+0 | 42+2 | 49+2 | +17 | +7 |

| Variant | H1→H2 | Pre RT | v1 RT | v2 RT | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0→0 | 11×64 | 11×64 | 11×64 | 0 | 0 |
| 2 | 0→0 | 12×64 | 12×64 | 12×64 | 0 | 0 |
| 3 | 13→7 | 26×4, 30×60 | 32×64 | 26×64 | 0×4, -4×60 | -6×64 |

Variants pre/v1/v2: 3/3/3. Aggregate CT: 118 / 128 / 135;
aggregate RT: 3376 / 3520 / 3136. V2 CT Δ vs pre/v1:
+17/+7; RT Δ:
-240/-384.

## 4. GroupNorm wrapped, legacy

Factory: `groupnorm_sharded_program_factory.cpp`.

Captured test: `tests/ttnn/unit_tests/operations/fused/test_group_norm_family.py::test_group_norm_wrapped_family_cache[rows-legacy]`.

| Kernel / variant | Pre CT P+N | v1 CT P+N | v2 CT P+N | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. `reader_mcast_sender_unary_sharded_gn_v2.cpp` | 10+0 | 19+3 | 26+3 | +19 | +7 |
| 2. `reader_mcast_receiver_unary_sharded_gn_v2.cpp` | 8+0 | 17+3 | 24+3 | +19 | +7 |
| 3. `writer_unary_sharded_gn_rm_gb_v2.cpp` | 21+1 | 21+1 | 21+1 | 0 | 0 |
| 4. `groupnorm_sharded_v2.cpp` | 28+1 | 28+1 | 28+1 | 0 | 0 |
| 5. `groupnorm_sharded_v2.cpp` | 28+1 | 28+1 | 28+1 | 0 | 0 |

| Variant | H1→H2 | Pre RT | v1 RT | v2 RT | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 27→19 | 30×6, 35×1 | 45×7 | 37×7 | +7×6, +2×1 | -8×7 |
| 2 | 27→2 | 2×56 | 27×56 | 2×56 | 0 | -25×56 |
| 3 | 0→0 | 10×63 | 10×63 | 10×63 | 0 | 0 |
| 4 | 0→0 | 0×7 | 0×7 | 0×7 | 0 | 0 |
| 5 | 0→0 | 0×56 | 0×56 | 0×56 | 0 | 0 |

Variants pre/v1/v2: 5/5/5. Aggregate CT: 98 / 122 / 136;
aggregate RT: 957 / 2457 / 1001. V2 CT Δ vs pre/v1:
+38/+14; RT Δ:
+44/-1456.

## 5. GroupNorm rectangular, legacy

Factory: `groupnorm_mcast_program_factory.cpp`.

Captured test: `tests/ttnn/unit_tests/operations/fused/test_group_norm_family.py::test_group_norm_interleaved_family_cache[multicast-legacy]`.

| Kernel / variant | Pre CT P+N | v1 CT P+N | v2 CT P+N | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. `reader_mcast_sender_unary_gn.cpp` | 4+27 | 15+28 | 22+28 | +19 | +7 |
| 2. `reader_mcast_receiver_unary_gn.cpp` | 4+26 | 15+27 | 22+27 | +19 | +7 |
| 3. `writer_unary_gn_rm_gb.cpp` | 8+31 | 8+31 | 8+31 | 0 | 0 |
| 4. `groupnorm.cpp` | 0+36 | 0+36 | 0+36 | 0 | 0 |
| 5. `groupnorm.cpp` | 0+36 | 0+36 | 0+36 | 0 | 0 |

| Variant | H1→H2 | Pre RT | v1 RT | v2 RT | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13→4 | 28×4 | 34×4 | 25×4 | -3×4 | -9×4 |
| 2 | 13→2 | 7×28 | 18×28 | 7×28 | 0 | -11×28 |
| 3 | 0→0 | 11×32 | 11×32 | 11×32 | 0 | 0 |
| 4 | 0→0 | 0×4 | 0×4 | 0×4 | 0 | 0 |
| 5 | 0→0 | 0×28 | 0×28 | 0×28 | 0 | 0 |

Variants pre/v1/v2: 5/5/5. Aggregate CT: 172 / 196 / 210;
aggregate RT: 660 / 992 / 648. V2 CT Δ vs pre/v1:
+38/+14; RT Δ:
-12/-344.

## 6. GroupNorm local, legacy

Factory: `groupnorm_no_mcast_program_factory.cpp`.

Captured test: `tests/ttnn/unit_tests/operations/fused/test_group_norm_family.py::test_group_norm_interleaved_family_cache[local-legacy]`.

| Kernel / variant | Pre CT P+N | v1 CT P+N | v2 CT P+N | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. `reader_mcast_sender_unary_gn.cpp` | 4+27 | 15+27 | 22+27 | +18 | +7 |
| 2. `reader_mcast_sender_unary_gn.cpp` | 4+27 | 15+27 | 22+27 | +18 | +7 |
| 3. `writer_unary_gn_rm_gb.cpp` | 8+31 | 8+31 | 8+31 | 0 | 0 |
| 4. `groupnorm.cpp` | 0+36 | 0+36 | 0+36 | 0 | 0 |
| 5. `groupnorm.cpp` | 0+36 | 0+36 | 0+36 | 0 | 0 |

| Variant | H1→H2 | Pre RT | v1 RT | v2 RT | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13→0 | 14×32 | 20×32 | 7×32 | -7×32 | -13×32 |
| 2 | 13→3 | — | — | — | — | — |
| 3 | 0→0 | 11×32 | 11×32 | 11×32 | 0 | 0 |
| 4 | 0→0 | 0×32 | 0×32 | 0×32 | 0 | 0 |
| 5 | 0→0 | — | — | — | — | — |

Variants pre/v1/v2: 5/5/5. Aggregate CT: 173 / 195 / 209;
aggregate RT: 800 / 992 / 576. V2 CT Δ vs pre/v1:
+36/+14; RT Δ:
-224/-416.

## 7. Attention Q10, interleaved

Factory: `group_attn_matmul_program_factory.cpp`.

Captured test: `tests/ttnn/nightly/unit_tests/operations/matmul/test_attn_matmul.py::test_group_attn_matmul_with_program_cache[num_loops=5-in0_dtype=DataType.BFLOAT16-in1_dtype=DataType.BFLOAT16-output_dtype=DataType.BFLOAT16-sharded=False]`.

| Kernel / variant | Pre CT P+N | v1 CT P+N | v2 CT P+N | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. `reader_mcast_transformer_group_attn_matmul.cpp` | 5+0 | 16+2 | 23+2 | +20 | +7 |
| 2. `writer_transformer_group_attn_matmul.cpp` | 7+0 | 7+0 | 7+0 | 0 | 0 |
| 3. `transformer_group_attn_matmul.cpp` | 4+0 | 4+0 | 4+0 | 0 | 0 |

| Variant | H1→H2 | Pre RT | v1 RT | v2 RT | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 75→73 | 47×110 | 95×110 | 93×110 | +46×110 | -2×110 |
| 2 | 0→0 | 17×110 | 17×110 | 17×110 | 0 | 0 |
| 3 | 0→0 | 14×110 | 14×110 | 14×110 | 0 | 0 |

Variants pre/v1/v2: 3/3/3. Aggregate CT: 16 / 29 / 36;
aggregate RT: 8580 / 13860 / 13640. V2 CT Δ vs pre/v1:
+20/+7; RT Δ:
+5060/-220.

## 8. Attention Q50, interleaved

Factory: `group_attn_matmul_program_factory.cpp`.

Captured test: `tests/ttnn/nightly/unit_tests/operations/matmul/test_attn_matmul.py::test_group_attn_matmul_with_program_cache[num_loops=5-in0_dtype=DataType.BFLOAT16-in1_dtype=DataType.BFLOAT16-output_dtype=DataType.BFLOAT16-sharded=False]`.

| Kernel / variant | Pre CT P+N | v1 CT P+N | v2 CT P+N | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. `reader_mcast_transformer_group_attn_matmul.cpp` | 5+0 | 16+2 | 23+2 | +20 | +7 |
| 2. `writer_transformer_group_attn_matmul.cpp` | 7+0 | 7+0 | 7+0 | 0 | 0 |
| 3. `transformer_group_attn_matmul.cpp` | 4+0 | 4+0 | 4+0 | 0 | 0 |

| Variant | H1→H2 | Pre RT | v1 RT | v2 RT | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 75→10 | 47×110 | 95×110 | 30×110 | -17×110 | -65×110 |
| 2 | 0→0 | 17×110 | 17×110 | 17×110 | 0 | 0 |
| 3 | 0→0 | 14×110 | 14×110 | 14×110 | 0 | 0 |

Variants pre/v1/v2: 3/3/3. Aggregate CT: 16 / 29 / 36;
aggregate RT: 8580 / 13860 / 6710. V2 CT Δ vs pre/v1:
+20/+7; RT Δ:
-1870/-7150.

## 9. Attention Q10, sharded

Factory: `group_attn_matmul_program_factory.cpp`.

Captured test: `tests/ttnn/nightly/unit_tests/operations/matmul/test_attn_matmul.py::test_group_attn_matmul_with_program_cache[num_loops=5-in0_dtype=DataType.BFLOAT16-in1_dtype=DataType.BFLOAT16-output_dtype=DataType.BFLOAT16-sharded=True]`.

| Kernel / variant | Pre CT P+N | v1 CT P+N | v2 CT P+N | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. `reader_mcast_transformer_group_attn_matmul.cpp` | 25+0 | 36+2 | 43+2 | +20 | +7 |
| 2. `writer_transformer_group_attn_matmul.cpp` | 25+0 | 25+0 | 25+0 | 0 | 0 |
| 3. `transformer_group_attn_matmul.cpp` | 4+0 | 4+0 | 4+0 | 0 | 0 |

| Variant | H1→H2 | Pre RT | v1 RT | v2 RT | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 75→73 | 47×110 | 95×110 | 93×110 | +46×110 | -2×110 |
| 2 | 0→0 | 17×110 | 17×110 | 17×110 | 0 | 0 |
| 3 | 0→0 | 14×110 | 14×110 | 14×110 | 0 | 0 |

Variants pre/v1/v2: 3/3/3. Aggregate CT: 54 / 67 / 74;
aggregate RT: 8580 / 13860 / 13640. V2 CT Δ vs pre/v1:
+20/+7; RT Δ:
+5060/-220.

## 10. Attention Q50, sharded

Factory: `group_attn_matmul_program_factory.cpp`.

Captured test: `tests/ttnn/nightly/unit_tests/operations/matmul/test_attn_matmul.py::test_group_attn_matmul_with_program_cache[num_loops=5-in0_dtype=DataType.BFLOAT16-in1_dtype=DataType.BFLOAT16-output_dtype=DataType.BFLOAT16-sharded=True]`.

| Kernel / variant | Pre CT P+N | v1 CT P+N | v2 CT P+N | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. `reader_mcast_transformer_group_attn_matmul.cpp` | 25+0 | 36+2 | 43+2 | +20 | +7 |
| 2. `writer_transformer_group_attn_matmul.cpp` | 65+0 | 65+0 | 65+0 | 0 | 0 |
| 3. `transformer_group_attn_matmul.cpp` | 4+0 | 4+0 | 4+0 | 0 | 0 |

| Variant | H1→H2 | Pre RT | v1 RT | v2 RT | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 75→10 | 47×110 | 95×110 | 30×110 | -17×110 | -65×110 |
| 2 | 0→0 | 17×110 | 17×110 | 17×110 | 0 | 0 |
| 3 | 0→0 | 14×110 | 14×110 | 14×110 | 0 | 0 |

Variants pre/v1/v2: 3/3/3. Aggregate CT: 94 / 107 / 114;
aggregate RT: 8580 / 13860 / 6710. V2 CT Δ vs pre/v1:
+20/+7; RT Δ:
-1870/-7150.

## 11. LayerNorm ordinary, two-stage

Factory: `layernorm_op_multi_core_sharded.cpp`.

Captured test: `tests/ttnn/unit_tests/operations/fused/test_layer_norm_sharded.py::test_layer_norm_sharded_with_residual[dtype=torch.bfloat16-tensor_type=ascending_values_repeated_rows-two_stage=True-use_welford=False]`.

| Kernel / variant | Pre CT P+N | v1 CT P+N | v2 CT P+N | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. `reader_mcast_sender_unary_sharded_ln.cpp` | 0+14 | 0+33 | 0+47 | +33 | +14 |
| 2. `reader_mcast_receiver_unary_sharded_ln.cpp` | 0+12 | 0+32 | 0+46 | +34 | +14 |
| 3. `reader_mcast_receiver_unary_sharded_ln.cpp` | 0+12 | 0+32 | 0+46 | +34 | +14 |
| 4. `writer_unary_sharded_ln.cpp` | 0+4 | 0+4 | 0+4 | 0 | 0 |
| 5. `writer_unary_sharded_ln.cpp` | 0+4 | 0+4 | 0+4 | 0 | 0 |
| 6. `layernorm_sharded.cpp` | 0+9 | 0+9 | 0+9 | 0 | 0 |
| 7. `layernorm_sharded.cpp` | 0+9 | 0+9 | 0+9 | 0 | 0 |

| Variant | H1→H2 | Pre RT | v1 RT | v2 RT | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 26→8 | 13×1 | 35×1 | 17×1 | +4×1 | -18×1 |
| 2 | 26→4 | 12×7 | 38×7 | 16×7 | +4×7 | -22×7 |
| 3 | 26→4 | 7×2 | 33×2 | 11×2 | +4×2 | -22×2 |
| 4 | 0→0 | 4×8 | 4×8 | 4×8 | 0 | 0 |
| 5 | 0→0 | 4×2 | 4×2 | 4×2 | 0 | 0 |
| 6 | 0→0 | 4×8 | 4×8 | 4×8 | 0 | 0 |
| 7 | 0→0 | 1×2 | 1×2 | 1×2 | 0 | 0 |

Variants pre/v1/v2: 7/7/7. Aggregate CT: 64 / 123 / 165;
aggregate RT: 185 / 441 / 225. V2 CT Δ vs pre/v1:
+101/+42; RT Δ:
+40/-216.

## 12. LayerNorm ordinary, per-line

Factory: `layernorm_op_multi_core_sharded.cpp`.

Captured test: `tests/ttnn/unit_tests/operations/fused/test_layer_norm_sharded.py::test_layer_norm_sharded_with_residual[dtype=torch.bfloat16-tensor_type=ascending_values_repeated_rows-two_stage=False-use_welford=False]`.

| Kernel / variant | Pre CT P+N | v1 CT P+N | v2 CT P+N | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. `reader_mcast_sender_unary_sharded_ln.cpp` | 0+14 | 0+33 | 0+47 | +33 | +14 |
| 2. `reader_mcast_receiver_unary_sharded_ln.cpp` | 0+12 | 0+32 | 0+46 | +34 | +14 |
| 3. `reader_mcast_receiver_unary_sharded_ln.cpp` | 0+12 | 0+32 | 0+46 | +34 | +14 |
| 4. `writer_unary_sharded_ln.cpp` | 0+4 | 0+4 | 0+4 | 0 | 0 |
| 5. `writer_unary_sharded_ln.cpp` | 0+4 | 0+4 | 0+4 | 0 | 0 |
| 6. `layernorm_sharded.cpp` | 0+9 | 0+9 | 0+9 | 0 | 0 |
| 7. `layernorm_sharded.cpp` | 0+9 | 0+9 | 0+9 | 0 | 0 |

| Variant | H1→H2 | Pre RT | v1 RT | v2 RT | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 26→8 | 12×2 | 34×2 | 16×2 | +4×2 | -18×2 |
| 2 | 26→4 | 11×6 | 37×6 | 15×6 | +4×6 | -22×6 |
| 3 | 26→4 | 7×2 | 33×2 | 11×2 | +4×2 | -22×2 |
| 4 | 0→0 | 4×8 | 4×8 | 4×8 | 0 | 0 |
| 5 | 0→0 | 4×2 | 4×2 | 4×2 | 0 | 0 |
| 6 | 0→0 | 4×8 | 4×8 | 4×8 | 0 | 0 |
| 7 | 0→0 | 1×2 | 1×2 | 1×2 | 0 | 0 |

Variants pre/v1/v2: 7/7/7. Aggregate CT: 64 / 123 / 165;
aggregate RT: 178 / 430 / 218. V2 CT Δ vs pre/v1:
+101/+42; RT Δ:
+40/-212.

## 13. LayerNorm pre-allgather

Factory: `layernorm_op_multi_core_sharded.cpp`.

Captured test: `tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_sharded.py::test_pre_allgather_layernorm[fuse_residual=False-max_atol_ex2=0.04-min_pcc_ex2=0.982-min_pcc_residual_add=0.997-min_pcc_ex=0.9997-max_atol_ex=0.01-core_grid=(8, 4)-mean=0-std=1-input_df=DataType.BFLOAT8_B-num_devices=4-input_width=2048-seed=0-is_rmsnorm=True]`.

| Kernel / variant | Pre CT P+N | v1 CT P+N | v2 CT P+N | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. `reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp` | 0+14 | 0+23 | 0+30 | +16 | +7 |
| 2. `reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp` | 0+12 | 0+22 | 0+29 | +17 | +7 |
| 3. `reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp` | 0+12 | 0+22 | 0+29 | +17 | +7 |
| 4. `writer_unary_sharded_ln_pre_all_gather.cpp` | 0+2 | 0+2 | 0+2 | 0 | 0 |
| 5. `writer_unary_sharded_ln_pre_all_gather.cpp` | 0+2 | 0+2 | 0+2 | 0 | 0 |
| 6. `layernorm_sharded_pre_allgather.cpp` | 0+9 | 0+9 | 0+9 | 0 | 0 |
| 7. `layernorm_sharded_pre_allgather.cpp` | 0+9 | 0+9 | 0+9 | 0 | 0 |

| Variant | H1→H2 | Pre RT | v1 RT | v2 RT | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13→4 | 18×1 | 27×1 | 18×1 | 0 | -9×1 |
| 2 | 13→2 | 17×3 | 30×3 | 19×3 | +2×3 | -11×3 |
| 3 | 13→2 | 7×28 | 20×28 | 9×28 | +2×28 | -11×28 |
| 4 | 0→0 | 2×4 | 2×4 | 2×4 | 0 | 0 |
| 5 | 0→0 | 2×28 | 2×28 | 2×28 | 0 | 0 |
| 6 | 0→0 | 4×4 | 4×4 | 4×4 | 0 | 0 |
| 7 | 0→0 | 1×28 | 1×28 | 1×28 | 0 | 0 |

Variants pre/v1/v2: 7/7/7. Aggregate CT: 60 / 89 / 110;
aggregate RT: 373 / 785 / 435. V2 CT Δ vs pre/v1:
+50/+21; RT Δ:
+62/-350.

## 14. LayerNorm post-allgather

Factory: `layernorm_op_multi_core_sharded.cpp`.

Captured test: `tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_sharded.py::test_post_allgather_layernorm[core_grid=(8, 2)-mean=0-std=1-weights_df=DataType.BFLOAT8_B-output_df=DataType.BFLOAT8_B-input_df=DataType.BFLOAT8_B-num_devices=4-input_width=2048-min_pcc=0.9997-max_atol=0.45-eps=1e-06-seed=0-is_rmsnorm=True]`.

| Kernel / variant | Pre CT P+N | v1 CT P+N | v2 CT P+N | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. `reader_mcast_sender_unary_sharded_ln_post_allgather.cpp` | 0+14 | 0+23 | 0+30 | +16 | +7 |
| 2. `reader_mcast_receiver_unary_sharded_ln_post_allgather.cpp` | 0+12 | 0+22 | 0+29 | +17 | +7 |
| 3. `reader_mcast_receiver_unary_sharded_ln_post_allgather.cpp` | 0+12 | 0+22 | 0+29 | +17 | +7 |
| 4. `writer_unary_sharded_ln.cpp` | 2+4 | 2+4 | 2+4 | 0 | 0 |
| 5. `writer_unary_sharded_ln.cpp` | 2+4 | 2+4 | 2+4 | 0 | 0 |
| 6. `layernorm_sharded_post_allgather.cpp` | 0+9 | 0+9 | 0+9 | 0 | 0 |
| 7. `layernorm_sharded_post_allgather.cpp` | 0+9 | 0+9 | 0+9 | 0 | 0 |

| Variant | H1→H2 | Pre RT | v1 RT | v2 RT | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13→4 | 16×1 | 25×1 | 16×1 | 0 | -9×1 |
| 2 | 13→2 | 15×1 | 28×1 | 17×1 | +2×1 | -11×1 |
| 3 | 13→2 | 7×14 | 20×14 | 9×14 | +2×14 | -11×14 |
| 4 | 0→0 | 5×2 | 5×2 | 5×2 | 0 | 0 |
| 5 | 0→0 | 5×14 | 5×14 | 5×14 | 0 | 0 |
| 6 | 0→0 | 5×2 | 5×2 | 5×2 | 0 | 0 |
| 7 | 0→0 | 1×14 | 1×14 | 1×14 | 0 | 0 |

Variants pre/v1/v2: 7/7/7. Aggregate CT: 68 / 97 / 118;
aggregate RT: 233 / 437 / 263. V2 CT Δ vs pre/v1:
+50/+21; RT Δ:
+30/-174.

## 15. Sparse matmul compact output

Factory: `sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp`.

Captured test: `tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py::test_sparse_matmul_compact_optional_output`.

| Kernel / variant | Pre CT P+N | v1 CT P+N | v2 CT P+N | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. `reader_bmm_tile_layout_in0_sender_padding.cpp` | 33+4 | 40+6 | 47+6 | +16 | +7 |
| 2. `reader_bmm_tile_layout_in0_receiver.cpp` | 8+1 | 17+3 | 24+3 | +18 | +7 |
| 3. `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | 41+5 | 38+7 | 38+7 | -1 | 0 |
| 4. `bmm_large_block_zm_fused_bias_activation.cpp` | 18+6 | 18+6 | 18+6 | 0 | 0 |

| Variant | H1→H2 | Pre RT | v1 RT | v2 RT | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13→7 | 8×1 | 17×1 | 11×1 | +3×1 | -6×1 |
| 2 | 13→7 | 2×5 | 13×5 | 7×5 | +5×5 | -6×5 |
| 3 | 0→0 | 26×6 | 22×6 | 22×6 | -4×6 | 0 |
| 4 | 0→0 | 0×6 | 0×6 | 0×6 | 0 | 0 |

Variants pre/v1/v2: 4/4/4. Aggregate CT: 116 / 135 / 149;
aggregate RT: 174 / 214 / 178. V2 CT Δ vs pre/v1:
+33/+14; RT Δ:
+4/-36.

## 16. Sparse matmul wide subblock

Factory: `sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp`.

Captured test: `tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py::test_sparse_matmul_wide_subblock`.

| Kernel / variant | Pre CT P+N | v1 CT P+N | v2 CT P+N | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. `reader_bmm_tile_layout_in0_sender_padding.cpp` | 33+4 | 40+6 | 47+6 | +16 | +7 |
| 2. `reader_bmm_tile_layout_in0_receiver.cpp` | 8+1 | 17+3 | 24+3 | +18 | +7 |
| 3. `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | 41+5 | 38+7 | 38+7 | -1 | 0 |
| 4. `bmm_large_block_zm_fused_bias_activation.cpp` | 18+6 | 18+6 | 18+6 | 0 | 0 |

| Variant | H1→H2 | Pre RT | v1 RT | v2 RT | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13→7 | 8×1 | 17×1 | 11×1 | +3×1 | -6×1 |
| 2 | 13→7 | 2×15 | 13×15 | 7×15 | +5×15 | -6×15 |
| 3 | 0→0 | 26×16 | 22×16 | 22×16 | -4×16 | 0 |
| 4 | 0→0 | 0×16 | 0×16 | 0×16 | 0 | 0 |

Variants pre/v1/v2: 4/4/4. Aggregate CT: 116 / 135 / 149;
aggregate RT: 454 / 564 / 468. V2 CT Δ vs pre/v1:
+33/+14; RT Δ:
+14/-96.

## 17. TopK local/final

Factory: `topk_multi_core_program_factory.cpp`.

Captured test: `tests/ttnn/unit_tests/operations/reduce/test_topk.py::test_topk_multicore_local_write_correctness[largest=True]`.

| Kernel / variant | Pre CT P+N | v1 CT P+N | v2 CT P+N | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. `reader_create_index_local_topk.cpp` | 9+0 | 9+0 | 9+0 | 0 | 0 |
| 2. `reader_final_topk.cpp` | 11+0 | 16+2 | 23+2 | +14 | +7 |
| 3. `writer_local_topk.cpp` | 11+0 | 21+2 | 28+2 | +19 | +7 |
| 4. `writer_final_topk.cpp` | 8+0 | 8+0 | 8+0 | 0 | 0 |
| 5. `topk_local.cpp` | 15+0 | 15+0 | 15+0 | 0 | 0 |
| 6. `topk_final.cpp` | 15+0 | 15+0 | 15+0 | 0 | 0 |

| Variant | H1→H2 | Pre RT | v1 RT | v2 RT | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0→0 | 5×32 | 5×32 | 5×32 | 0 | 0 |
| 2 | 13→4 | 0×1 | 13×1 | 4×1 | +4×1 | -9×1 |
| 3 | 13→2 | 1×32 | 14×32 | 3×32 | +2×32 | -11×32 |
| 4 | 0→0 | 2×1 | 2×1 | 2×1 | 0 | 0 |
| 5 | 0→0 | 1×32 | 1×32 | 1×32 | 0 | 0 |
| 6 | 0→0 | 0×1 | 0×1 | 0×1 | 0 | 0 |

Variants pre/v1/v2: 6/6/6. Aggregate CT: 69 / 88 / 102;
aggregate RT: 226 / 655 / 294. V2 CT Δ vs pre/v1:
+33/+14; RT Δ:
+68/-361.

## 18. Conv2D width-sharded

Factory: `conv2d_op_width_sharded_program_factory.cpp`.

Captured test: `tests/ttnn/unit_tests/operations/conv/test_conv2d.py::test_conv_features[output_layout=Layout.TILE-math_fidelity=MathFidelity.HiFi4-filter=3-padding=(1, 2, 2, 3)-packer_l1_acc=True-fp32_accum=True-input_dtype=DataType.BFLOAT16-output_dtype=DataType.BFLOAT16-output_channels=353-input_channels=384-input_height=8-input_width=8-shard_layout=TensorMemoryLayout.WIDTH_SHARDED-config=None-batch_size=2-stride=2-device_params={'l1_small_size': 16384}]`.

| Kernel / variant | Pre CT P+N | v1 CT P+N | v2 CT P+N | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. `activation_reader_width_sharded.cpp` | 27+0 | 32+2 | 39+2 | +14 | +7 |
| 2. `weights_reader_width_sharded.cpp` | 18+0 | 18+0 | 18+0 | 0 | 0 |
| 3. `conv_bmm_tilize.cpp` | 38+0 | 38+0 | 38+0 | 0 | 0 |

| Variant | H1→H2 | Pre RT | v1 RT | v2 RT | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 55→11 | 24×22 | 58×22 | 14×22 | -10×22 | -44×22 |
| 2 | 0→0 | 4×12 | 4×12 | 4×12 | 0 | 0 |
| 3 | 0→0 | 0×12 | 0×12 | 0×12 | 0 | 0 |

Variants pre/v1/v2: 3/3/3. Aggregate CT: 83 / 90 / 97;
aggregate RT: 576 / 1324 / 356. V2 CT Δ vs pre/v1:
+14/+7; RT Δ:
-220/-968.

## 19. Conv2D block-sharded

Factory: `conv2d_op_sharded_program_factory.cpp`.

Captured test: `tests/ttnn/unit_tests/operations/conv/test_conv2d.py::test_conv_features[output_layout=Layout.TILE-math_fidelity=MathFidelity.HiFi4-filter=3-padding=(1, 2, 2, 3)-packer_l1_acc=True-fp32_accum=True-input_dtype=DataType.BFLOAT16-output_dtype=DataType.BFLOAT16-output_channels=128-input_channels=128-input_height=32-input_width=32-shard_layout=TensorMemoryLayout.BLOCK_SHARDED-config=None-batch_size=2-stride=2-device_params={'l1_small_size': 16384}]`.

| Kernel / variant | Pre CT P+N | v1 CT P+N | v2 CT P+N | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. `writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` | 40+0 | 51+2 | 58+2 | +20 | +7 |
| 2. `writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp` | 40+0 | 51+2 | 58+2 | +20 | +7 |
| 3. `reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp` | 36+0 | 36+0 | 36+0 | 0 | 0 |
| 4. `conv_bmm_tilize.cpp` | 38+0 | 38+0 | 38+0 | 0 | 0 |

| Variant | H1→H2 | Pre RT | v1 RT | v2 RT | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13→5 | 14×44 | 19×44 | 11×44 | -3×44 | -8×44 |
| 2 | 13→2 | 5×36 | 14×36 | 3×36 | -2×36 | -11×36 |
| 3 | 0→0 | 17×80 | 17×80 | 17×80 | 0 | 0 |
| 4 | 0→0 | 1×80 | 1×80 | 1×80 | 0 | 0 |

Variants pre/v1/v2: 4/4/4. Aggregate CT: 154 / 180 / 194;
aggregate RT: 2236 / 2780 / 2032. V2 CT Δ vs pre/v1:
+40/+14; RT Δ:
-204/-748.

## 20. Conv2D height-sharded

Factory: `conv2d_op_sharded_program_factory.cpp`.

Captured test: `tests/ttnn/unit_tests/operations/conv/test_conv2d.py::test_conv_features[output_layout=Layout.TILE-math_fidelity=MathFidelity.HiFi4-filter=3-padding=(1, 2, 2, 3)-packer_l1_acc=True-fp32_accum=True-input_dtype=DataType.BFLOAT16-output_dtype=DataType.BFLOAT16-output_channels=16-input_channels=16-input_height=256-input_width=256-shard_layout=TensorMemoryLayout.HEIGHT_SHARDED-config={'act_block_h': 32}-batch_size=2-stride=2-device_params={'l1_small_size': 16384}]`.

| Kernel / variant | Pre CT P+N | v1 CT P+N | v2 CT P+N | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. `reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` | 43+0 | 54+2 | 61+2 | +20 | +7 |
| 2. `reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp` | 43+0 | 54+2 | 61+2 | +20 | +7 |
| 3. `reader_conv_activations_padded_with_halo_3x3_weights_v2.cpp` | 41+0 | 41+0 | 41+0 | 0 | 0 |
| 4. `conv_bmm_tilize.cpp` | 38+0 | 38+0 | 38+0 | 0 | 0 |

| Variant | H1→H2 | Pre RT | v1 RT | v2 RT | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13→4 | 12×1 | 18×1 | 9×1 | -3×1 | -9×1 |
| 2 | 13→2 | 5×109 | 15×109 | 4×109 | -1×109 | -11×109 |
| 3 | 0→0 | 2×105 | 2×105 | 2×105 | 0 | 0 |
| 4 | 0→0 | 0×105 | 0×105 | 0×105 | 0 | 0 |

Variants pre/v1/v2: 4/4/4. Aggregate CT: 165 / 191 / 205;
aggregate RT: 767 / 1863 / 655. V2 CT Δ vs pre/v1:
+40/+14; RT Δ:
-112/-1208.

## 21. DiT GroupNorm local

Factory: `dit_fused_distributed_groupnorm_program_factory.cpp`.

Captured test: `models/tt_dit/tests/unit/test_normalization.py::test_distributed_group_norm[blackhole-no_act-c128_h64-local_1x1]`.

| Kernel / variant | Pre CT P+N | v1 CT P+N | v2 CT P+N | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. `welford_reader_mcast_sender_unary_gn.cpp` | 4+27 | 15+29 | 22+29 | +20 | +7 |
| 2. `welford_reader_mcast_receiver_unary_gn.cpp` | 2+27 | 15+29 | 22+29 | +22 | +7 |
| 3. `welford_writer_unary_gn_rm_gb.cpp` | 8+22 | 8+22 | 8+22 | 0 | 0 |
| 4. `welford_groupnorm.cpp` | 0+20 | 0+20 | 0+20 | 0 | 0 |

| Variant | H1→H2 | Pre RT | v1 RT | v2 RT | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13→7 | 44×4 | 50×4 | 44×4 | 0 | -6×4 |
| 2 | 13→7 | 7×60 | 18×60 | 12×60 | +5×60 | -6×60 |
| 3 | 0→0 | 10×64 | 10×64 | 10×64 | 0 | 0 |
| 4 | 0→0 | 0×64 | 0×64 | 0×64 | 0 | 0 |

Variants pre/v1/v2: 4/4/4. Aggregate CT: 110 / 138 / 152;
aggregate RT: 1236 / 1920 / 1536. V2 CT Δ vs pre/v1:
+42/+14; RT Δ:
+300/-384.

## 22. Dedicated DRAM-sharded matmul

Factory: `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`.

Captured test: `tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_padding[dram_sharded-no_padding-input_a_value=4.0-input_b_value=2.0]`.

| Kernel / variant | Pre CT P+N | v1 CT P+N | v2 CT P+N | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. `reader_bmm_tile_layout_in0_sender_dram_sharded.cpp` | 16+2 | 16+2 | 16+2 | 0 | 0 |
| 2. `reader_bmm_tile_layout_in1_sender_dram_sharded.cpp` | 12+4 | 12+4 | 12+4 | 0 | 0 |
| 3. `bmm_large_block_zm_fused_bias_activation.cpp` | 18+10 | 18+10 | 18+10 | 0 | 0 |

| Variant | H1→H2 | Pre RT | v1 RT | v2 RT | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0→0 | 1×26, 9×4 | 1×26, 9×4 | 1×26, 9×4 | 0 | 0 |
| 2 | 0→0 | 1×29, 11×1 | 1×29, 11×1 | 1×29, 11×1 | 0 | 0 |
| 3 | 0→0 | 1×30 | 1×30 | 1×30 | 0 | 0 |

Variants pre/v1/v2: 3/3/3. Aggregate CT: 62 / 62 / 62;
aggregate RT: 132 / 132 / 132. V2 CT Δ vs pre/v1:
0/0; RT Δ:
0/0.
