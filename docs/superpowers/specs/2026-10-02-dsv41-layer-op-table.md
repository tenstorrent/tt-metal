# DSV4.1-Flash decode layer op table (BH Galaxy 4x8, batch 16 = 4 per mesh row)

Source: tests/test_layer_op_table.py (ttnn call recorder + device profiler digest) + tests/op_table_build_v2.py. Data on .46, dirs /mnt/tt-data/ssinghal/dsv4-logs/optable_L2 and optable_L3.
Kernel = DEVICE KERNEL DURATION from the trace replay (5-8 clean replay sessions x 32 devices, median over devices; max = slowest device). Gap = OP TO OP LATENCY in the trace (median). 'eager' = same op measured in the eager pass.
Layer 2 = MoE layer with compressor (ratio 2, kv source); layer 3 = MoE layer, no compressor/Engram in the layer forward. Section names follow the profile marks; 'CCL' ops are listed under the section that calls them and tagged (CCL).

## Layer 2: traced 1502 us/layer; sum kernel 1436 us + sum gap 60 us over 85 programs

| section | programs | kernel us | gap us | of which CCL kernel us | % layer |
|---|---|---|---|---|---|
| mhc_attn_mixes | 2 | 67.8 | 18.2 | 0.0 | 5.7% |
| mhc_attn_collapse+norm | 2 | 24.3 | 1.1 | 0.0 | 1.7% |
| attention | 38 | 355.7 | 28.0 | 43.8 | 25.5% |
| mhc_attn_expand | 1 | 26.7 | 0.5 | 0.0 | 1.8% |
| mhc_ffn_mixes | 2 | 68.2 | 1.2 | 0.0 | 4.6% |
| mhc_ffn_collapse+norm | 2 | 27.9 | 1.1 | 0.0 | 1.9% |
| moe | 31 | 700.9 | 24.0 | 204.4 | 48.3% |
| moe_allgather | 1 | 25.6 | 0.5 | 25.6 | 1.7% |
| shared_expert | 5 | 111.4 | 2.7 | 0.0 | 7.6% |
| mhc_ffn_expand | 1 | 27.6 | 0.5 | 0.0 | 1.9% |

| # | section | op | caller | cores | kernel us (trace med / max) | eager us | gap us | out |
|---|---|---|---|---|---|---|---|---|
| 0 | mhc_attn_mixes | generic_op [mhc_proj] | mhc_mixes.py:51:mhc_proj | 32 | 21.5 / 22.5 | 21.7 | 17.55 | [32x1x32x32] FLOAT32 T DRAM |
| 1 | mhc_attn_mixes | generic_op [mhc_post] | mhc_mixes.py:79:mhc_post | 1 | 46.2 / 46.5 | 46.4 | 0.60 | [4x1x4x4] FLOAT32 T DRAM |
| 2 | mhc_attn_collapse+norm | generic_op [mhc_collapse] | mhc_collapse.py:74:mhc_collapse | 32 | 12.5 / 12.9 | 12.5 | 0.52 | [32x1x32x32] FLOAT32 T DRAM |
| 3 | mhc_attn_collapse+norm | generic_op [mhc_norm] | mhc_collapse.py:111:mhc_norm_apply | 32 | 11.8 / 12.1 | 11.4 | 0.53 | [1x1x4x5120] BFLOAT16 T DRAM |
| 4 | attention | linear | attention.py:221:_lin | 16 | 26.2 / 26.4 | 25.4 | 0.55 | [1x1x4x1024] FLOAT32 T DRAM |
| 5 | attention | subtract | attention.py:417:_compress_step | 120 | 2.7 / 3.0 | 2.3 | 0.53 | [1x1x4x1024] FLOAT32 T DRAM |
| 6 | attention | slice | attention.py:415:<lambda> | 16 | 1.8 / 1.9 | 1.2 | 0.59 | [1x1x4x512] FLOAT32 T DRAM |
| 7 | attention | slice | attention.py:415:<lambda> | 16 | 1.7 / 1.8 | 1.2 | 0.60 | [1x1x4x512] FLOAT32 T DRAM |
| 8 | attention | sigmoid | attention.py:418:_compress_step | 120 | 3.3 / 3.5 | 2.7 | 0.45 | [1x1x4x1024] FLOAT32 T DRAM |
| 9 | attention | slice | attention.py:415:<lambda> | 16 | 1.8 / 1.9 | 1.2 | 0.57 | [1x1x4x512] FLOAT32 T DRAM |
| 10 | attention | addcmul | attention.py:418:_compress_step | 120 | 3.1 / 3.5 | 2.5 | 0.45 | [1x1x4x512] FLOAT32 T DRAM |
| 11 | attention | typecast | attention.py:419:_compress_step | 16 | 2.2 / 2.2 | 1.5 | 0.58 | [1x1x4x512] BFLOAT16 T DRAM |
| 12 | attention | rms_norm | attention.py:419:_compress_step | 1 | 7.2 / 7.3 | 7.1 | 1.25 | [1x1x4x512] BFLOAT16 T DRAM |
| 13 | attention | copy | attention.py:421:_compress_step | 32 | 1.7 / 1.9 | 1.5 | 0.58 | [1x1x4x1024] FLOAT32 T DRAM |
| 14 | attention | multiply | attention.py:225:_rope_rows | 120 | 2.9 / 3.0 | 2.1 | 0.51 | [1x1x4x512] BFLOAT16 T DRAM |
| 15 | attention | linear | attention.py:221:_lin | 16 | 9.3 / 9.5 | 8.4 | 0.53 | [1x1x4x512] BFLOAT16 T DRAM |
| 16 | attention | addcmul | attention.py:225:_rope_rows | 120 | 2.9 / 3.0 | 2.1 | 0.52 | [1x1x4x512] BFLOAT16 T DRAM |
| 17 | attention | linear | attention.py:221:_lin | 56 | 34.2 / 36.1 | 34.1 | 0.59 | [1x1x4x1792] BFLOAT16 T DRAM |
| 18 | attention | slice | attention.py:236:_qkv | 40 | 1.7 / 1.8 | 1.2 | 0.64 | [1x1x4x1280] BFLOAT16 T DRAM |
| 19 | attention | rms_norm | attention.py:236:_qkv | 1 | 13.6 / 14.1 | 13.5 | 0.74 | [1x1x4x1280] BFLOAT16 T DRAM |
| 20 | attention | linear | attention.py:221:_lin | 32 | 21.0 / 21.6 | 21.0 | 0.53 | [1x1x4x4096] BFLOAT16 T DRAM |
| 21 | attention | slice | attention.py:238:_qkv | 16 | 1.7 / 1.9 | 1.0 | 0.66 | [1x1x4x512] BFLOAT16 T DRAM |
| 22 | attention | rms_norm | attention.py:238:_qkv | 1 | 7.2 / 7.3 | 7.1 | 1.34 | [1x1x4x512] BFLOAT16 T DRAM |
| 23 | attention | concat | attention.py:240:_qkv | 120 | 2.5 / 3.0 | 2.4 | 0.51 | [1x1x4x5632] BFLOAT16 T DRAM |
| 24 | attention | experimental.nlp_create_qkv_heads_decode | attention.py:241:_qkv | 4 | 17.4 / 19.1 | 16.7 | 0.99 | [1x4x9x512] BFLOAT16 T L1-sh |
| 25 | attention | to_memory_config | attention.py:244:_qkv | 4 | 1.5 / 1.9 | 1.1 | 1.04 | [1x4x9x512] BFLOAT16 T DRAM |
| 26 | attention | multiply | attention.py:273:_rope_heads | 120 | 3.1 / 3.2 | 2.8 | 0.52 | [1x4x9x512] BFLOAT16 T DRAM |
| 27 | attention | linear | attention.py:221:_lin | 8 | 17.6 / 17.7 | 16.7 | 0.59 | [1x4x9x512] BFLOAT16 T DRAM |
| 28 | attention | addcmul | attention.py:273:_rope_heads | 120 | 3.1 / 3.7 | 3.0 | 0.54 | [1x4x9x512] BFLOAT16 T DRAM |
| 29 | attention | to_memory_config | attention.py:247:_qkv | 4 | 1.1 / 1.3 | 1.0 | 1.25 | [1x4x9x512] BFLOAT16 T L1-sh |
| 30 | attention | experimental.paged_update_cache | attention.py:251:_write_cache | 4 | 3.8 / 3.9 | 3.6 | 1.36 | [4x1x256x512] BFLOAT16 T DRAM |
| 31 | attention | experimental.paged_update_cache | attention.py:251:_write_cache | 4 | 3.8 / 3.9 | 3.7 | 1.38 | [4x1x256x512] BFLOAT16 T DRAM |
| 32 | attention | transformer.scaled_dot_product_attention_decode | attention.py:433:forward | 4 | 38.1 / 40.6 | 38.1 | 2.36 | [1x4x9x512] BFLOAT16 T DRAM |
| 33 | attention | multiply | attention.py:273:_rope_heads | 120 | 3.0 / 3.4 | 2.7 | 0.52 | [1x4x9x512] BFLOAT16 T DRAM |
| 34 | attention | linear | attention.py:221:_lin | 8 | 17.5 / 17.7 | 16.7 | 0.60 | [1x4x9x512] BFLOAT16 T DRAM |
| 35 | attention | addcmul | attention.py:273:_rope_heads | 120 | 3.3 / 3.4 | 3.5 | 0.54 | [1x4x9x512] BFLOAT16 T L1-sh |
| 36 | attention | experimental.nlp_concat_heads_decode | attention.py:265:_finish | 9 | 3.3 / 3.5 | 3.0 | 0.57 | [1x1x32x4608] BFLOAT16 T L1-sh |
| 37 | attention | to_memory_config | attention.py:266:_finish | 9 | 1.9 / 2.0 | 1.4 | 0.69 | [1x1x32x4608] BFLOAT16 T DRAM |
| 38 | attention | linear | attention.py:221:_lin | 16 | 23.6 / 23.7 | 23.2 | 0.60 | [1x1x32x1024] BFLOAT16 T DRAM |
| 39 | attention | linear | attention.py:221:_lin | 32 | 21.6 / 22.6 | 21.5 | 0.62 | [1x1x32x5120] BFLOAT16 T DRAM |
| 40 | attention | experimental.reduce_scatter_minimal_async (CCL) | config.py:145:allreduce | 12 | 20.4 / 25.6 | 34.8 | 0.54 | [1x1x32x640] BFLOAT16 T DRAM |
| 41 | attention | experimental.all_gather_async (CCL) | config.py:166:allreduce | 12 | 23.4 / 26.1 | 33.0 | 0.54 | [1x1x32x5120] BFLOAT16 T DRAM |
| 42 | mhc_attn_expand | generic_op [mhc_expand] | mhc_expand.py:62:mhc_expand | 32 | 26.7 / 27.9 | 26.8 | 0.52 | [4x1x4x5120] FLOAT32 T DRAM |
| 43 | mhc_ffn_mixes | generic_op [mhc_proj] | mhc_mixes.py:51:mhc_proj | 32 | 21.9 / 23.3 | 21.6 | 0.59 | [32x1x32x32] FLOAT32 T DRAM |
| 44 | mhc_ffn_mixes | generic_op [mhc_post] | mhc_mixes.py:79:mhc_post | 1 | 46.2 / 46.6 | 46.3 | 0.58 | [4x1x4x4] FLOAT32 T DRAM |
| 45 | mhc_ffn_collapse+norm | generic_op [mhc_collapse] | mhc_collapse.py:74:mhc_collapse | 32 | 12.5 / 13.0 | 12.4 | 0.52 | [32x1x32x32] FLOAT32 T DRAM |
| 46 | mhc_ffn_collapse+norm | generic_op [mhc_norm] | mhc_collapse.py:111:mhc_norm_apply | 32 | 15.4 / 15.8 | 15.1 | 0.53 | [4x1x1x5120] BFLOAT16 RM DRAM |
| 47 | moe | matmul | router.py:120:_forward_exact_fast | 12 | 24.2 / 24.9 | 23.4 | 0.55 | [1x1x4x384] FLOAT32 T DRAM |
| 48 | moe | softplus | router.py:122:_forward_exact_fast | 120 | 4.3 / 4.6 | 3.5 | 0.53 | [1x1x4x384] FLOAT32 T DRAM |
| 49 | moe | sqrt | router.py:122:_forward_exact_fast | 120 | 2.8 / 3.0 | 1.9 | 0.53 | [1x1x4x384] FLOAT32 T DRAM |
| 50 | moe | add | router.py:124:_forward_exact_fast | 120 | 3.1 / 3.2 | 2.3 | 0.53 | [1x1x4x384] FLOAT32 T DRAM |
| 51 | moe | topk | router.py:124:_forward_exact_fast | 12 | 4.7 / 4.9 | 4.4 | 0.63 | [1x1x4x6] FLOAT32 T DRAM |
| 52 | moe | topk | router.py:124:_forward_exact_fast | 1 | 63.0 / 63.3 | 63.0 | 1.54 | [1x1x4x6] FLOAT32 T DRAM |
| 53 | moe | typecast | router.py:125:_forward_exact_fast | 1 | 1.2 / 1.3 | 1.2 | 1.47 | [1x1x4x6] FLOAT32 T DRAM |
| 54 | moe | reshape | router.py:125:_forward_exact_fast | 4 | 2.7 / 2.8 | 1.9 | 0.61 | [4x1x6x1] FLOAT32 T DRAM |
| 55 | moe | eq | router.py:125:_forward_exact_fast | 120 | 7.6 / 8.1 | 7.8 | 0.51 | [4x1x6x384] FLOAT32 T DRAM |
| 56 | moe | reshape | router.py:126:_forward_exact_fast | 48 | 2.5 / 2.7 | 2.3 | 0.60 | [4x1x1x384] FLOAT32 T DRAM |
| 57 | moe | multiply | router.py:126:_forward_exact_fast | 120 | 3.3 / 3.5 | 3.2 | 0.51 | [4x1x6x384] FLOAT32 T DRAM |
| 58 | moe | sum | router.py:126:_forward_exact_fast | 48 | 5.1 / 5.3 | 4.8 | 0.61 | [4x1x6x1] FLOAT32 T DRAM |
| 59 | moe | sum | router.py:126:_forward_exact_fast | 4 | 5.1 / 5.5 | 4.6 | 0.82 | [4x1x6x1] FLOAT32 T DRAM |
| 60 | moe | sum | router.py:128:_forward_exact_fast | 4 | 12.0 / 12.1 | 11.2 | 0.68 | [4x1x1x1] FLOAT32 T DRAM |
| 61 | moe | sum | router.py:128:_forward_exact_fast | 4 | 2.3 / 2.5 | 1.6 | 0.69 | [4x1x1x1] FLOAT32 T DRAM |
| 62 | moe | add | router.py:128:_forward_exact_fast | 120 | 4.3 / 4.4 | 3.4 | 0.51 | [4x1x1x1] FLOAT32 T DRAM |
| 63 | moe | div | router.py:129:_forward_exact_fast | 120 | 6.0 / 6.2 | 5.1 | 0.51 | [4x1x6x1] FLOAT32 T DRAM |
| 64 | moe | multiply | router.py:129:_forward_exact_fast | 120 | 4.3 / 4.4 | 3.4 | 0.52 | [4x1x6x1] FLOAT32 T DRAM |
| 65 | moe | typecast | router.py:129:_forward_exact_fast | 4 | 2.2 / 2.3 | 1.3 | 0.61 | [4x1x6x1] BFLOAT16 T DRAM |
| 66 | moe | to_layout | router.py:130:_forward_exact_fast | 4 | 3.3 / 3.4 | 2.4 | 0.63 | [4x1x6x1] BFLOAT16 RM DRAM |
| 67 | moe | reshape | router.py:130:_forward_exact_fast | 120 | 4.0 / 4.2 | 3.4 | 0.52 | [4x1x1x6] BFLOAT16 RM DRAM |
| 68 | moe | typecast | router.py:131:_forward_exact_fast | 1 | 1.2 / 1.3 | 1.2 | 1.48 | [1x1x4x6] UINT16 T DRAM |
| 69 | moe | to_layout | router.py:131:_forward_exact_fast | 1 | 1.9 / 2.0 | 1.9 | 1.46 | [1x1x4x6] UINT16 RM DRAM |
| 70 | moe | to_memory_config | tt_moe_decode.py:900:_format_dispatch_inputs | 4 | 2.0 / 2.2 | 1.5 | 0.61 | [4x1x1x5120] BFLOAT16 RM L1 |
| 71 | moe | to_memory_config | tt_moe_decode.py:908:_format_dispatch_inputs | 4 | 1.6 / 1.7 | 0.9 | 0.68 | [4x1x1x6] UINT16 RM L1 |
| 72 | moe | to_memory_config | tt_moe_decode.py:919:_format_dispatch_inputs | 4 | 1.3 / 1.4 | 0.7 | 0.68 | [4x1x1x6] BFLOAT16 RM L1-sh |
| 73 | moe | experimental.all_to_all_dispatch_metadata (CCL) | tt_moe_decode.py:972:forward | 2 | 10.0 / 27.1 | 18.8 | 0.97 | [1x16x5120] BFLOAT16 RM DRAM |
| 74 | moe | experimental.moe_compute (CCL) | tt_moe_decode.py:1001:forward | 32 | 281.5 / 463.0 | 342.2 | 2.25 | [120x12] UINT32 RM L1-sh |
| 75 | moe | tilize_with_val_padding | tt_moe_decode.py:1040:forward | 80 | 24.5 / 24.9 | 24.4 | 0.49 | [6x1x4x5120] BFLOAT16 T L1-sh |
| 76 | moe | experimental.deepseek_moe_fast_reduce_nc_fused | tt_moe_decode.py:1049:forward | 120 | 14.3 / 15.0 | 14.2 | 0.52 | [1x1x4x5120] BFLOAT16 T DRAM |
| 77 | moe | reduce_scatter (CCL) | tt_moe_decode.py:1089:forward | 12 | 194.4 / 307.7 | 48.5 | 0.73 | [1x1x4x640] BFLOAT16 T DRAM |
| 78 | moe_allgather | experimental.all_gather_async (CCL) | config.py:194:allgather | 12 | 25.6 / 37.2 | 38.1 | 0.52 | [1x1x4x5120] BFLOAT16 T DRAM |
| 79 | shared_expert | linear | shared_expert_v2.py:90:<lambda> | 36 | 65.4 / 67.2 | 65.7 | 0.51 | [1x1x4x4608] BFLOAT16 T DRAM |
| 80 | shared_expert | slice | shared_expert_v2.py:112:forward | 72 | 1.8 / 2.0 | 1.4 | 0.56 | [1x1x4x2304] BFLOAT16 T DRAM |
| 81 | shared_expert | slice | shared_expert_v2.py:112:forward | 72 | 1.8 / 1.9 | 1.4 | 0.56 | [1x1x4x2304] BFLOAT16 T DRAM |
| 82 | shared_expert | multiply | shared_expert_v2.py:92:<lambda> | 120 | 4.5 / 4.8 | 4.2 | 0.53 | [1x1x4x2304] BFLOAT16 T DRAM |
| 83 | shared_expert | linear | shared_expert_v2.py:90:<lambda> | 40 | 37.9 / 39.9 | 37.8 | 0.54 | [1x1x4x5120] FLOAT32 T DRAM |
| 84 | mhc_ffn_expand | generic_op [mhc_expand] | mhc_expand.py:62:mhc_expand | 32 | 27.6 / 29.3 | 27.8 | 0.54 | [4x1x4x5120] FLOAT32 T DRAM |

Ops with kernel < 30 us: 76 of 85, kernel 629 us + gap 68 us. Ops < 5 us: 45 (kernel floor ~1.1-1.8 us, gap 0.5-1.5 us each, i.e. launch-bound).

## Layer 3: traced 1456 us/layer; sum kernel 1395 us + sum gap 44 us over 63 programs

| section | programs | kernel us | gap us | of which CCL kernel us | % layer |
|---|---|---|---|---|---|
| mhc_attn_mixes | 2 | 67.7 | 18.1 | 0.0 | 5.9% |
| mhc_attn_collapse+norm | 2 | 24.5 | 1.1 | 0.0 | 1.8% |
| attention | 24 | 284.5 | 17.9 | 43.7 | 20.8% |
| mhc_attn_expand | 1 | 26.7 | 0.5 | 0.0 | 1.9% |
| mhc_ffn_mixes | 2 | 68.1 | 1.2 | 0.0 | 4.8% |
| mhc_ffn_collapse+norm | 2 | 28.1 | 1.1 | 0.0 | 2.0% |
| moe | 23 | 731.9 | 18.4 | 196.3 | 51.5% |
| moe_allgather | 1 | 24.9 | 0.5 | 24.9 | 1.7% |
| shared_expert | 5 | 111.5 | 2.7 | 0.0 | 7.8% |
| mhc_ffn_expand | 1 | 27.7 | 0.5 | 0.0 | 1.9% |

| # | section | op | caller | cores | kernel us (trace med / max) | eager us | gap us | out |
|---|---|---|---|---|---|---|---|---|
| 0 | mhc_attn_mixes | generic_op [mhc_proj] | mhc_mixes.py:51:mhc_proj | 32 | 21.5 / 22.6 | 21.5 | 17.52 | [32x1x32x32] FLOAT32 T DRAM |
| 1 | mhc_attn_mixes | generic_op [mhc_post] | mhc_mixes.py:79:mhc_post | 1 | 46.2 / 46.5 | 46.3 | 0.59 | [4x1x4x4] FLOAT32 T DRAM |
| 2 | mhc_attn_collapse+norm | generic_op [mhc_collapse] | mhc_collapse.py:74:mhc_collapse | 32 | 12.6 / 12.9 | 12.5 | 0.52 | [32x1x32x32] FLOAT32 T DRAM |
| 3 | mhc_attn_collapse+norm | generic_op [mhc_norm] | mhc_collapse.py:111:mhc_norm_apply | 32 | 12.0 / 12.4 | 11.6 | 0.53 | [1x1x4x5120] BFLOAT16 T DRAM |
| 4 | attention | linear | attention.py:221:_lin | 56 | 34.2 / 35.3 | 34.1 | 0.54 | [1x1x4x1792] BFLOAT16 T DRAM |
| 5 | attention | slice | attention.py:236:_qkv | 40 | 1.7 / 1.8 | 1.2 | 0.64 | [1x1x4x1280] BFLOAT16 T DRAM |
| 6 | attention | rms_norm | attention.py:236:_qkv | 1 | 13.6 / 14.0 | 13.5 | 0.80 | [1x1x4x1280] BFLOAT16 T DRAM |
| 7 | attention | linear | attention.py:221:_lin | 32 | 21.1 / 21.6 | 21.0 | 0.53 | [1x1x4x4096] BFLOAT16 T DRAM |
| 8 | attention | slice | attention.py:238:_qkv | 16 | 1.6 / 1.8 | 1.0 | 0.66 | [1x1x4x512] BFLOAT16 T DRAM |
| 9 | attention | rms_norm | attention.py:238:_qkv | 1 | 7.2 / 7.3 | 7.1 | 1.31 | [1x1x4x512] BFLOAT16 T DRAM |
| 10 | attention | concat | attention.py:240:_qkv | 120 | 2.5 / 3.1 | 2.5 | 0.51 | [1x1x4x5632] BFLOAT16 T DRAM |
| 11 | attention | experimental.nlp_create_qkv_heads_decode | attention.py:241:_qkv | 4 | 17.2 / 17.6 | 16.8 | 0.97 | [1x4x9x512] BFLOAT16 T L1-sh |
| 12 | attention | to_memory_config | attention.py:244:_qkv | 4 | 1.6 / 2.0 | 1.1 | 0.97 | [1x4x9x512] BFLOAT16 T DRAM |
| 13 | attention | multiply | attention.py:273:_rope_heads | 120 | 3.0 / 3.2 | 2.7 | 0.52 | [1x4x9x512] BFLOAT16 T DRAM |
| 14 | attention | linear | attention.py:221:_lin | 8 | 17.5 / 17.7 | 16.7 | 0.61 | [1x4x9x512] BFLOAT16 T DRAM |
| 15 | attention | addcmul | attention.py:273:_rope_heads | 120 | 3.1 / 3.5 | 3.1 | 0.54 | [1x4x9x512] BFLOAT16 T DRAM |
| 16 | attention | to_memory_config | attention.py:247:_qkv | 4 | 1.1 / 1.3 | 1.0 | 1.23 | [1x4x9x512] BFLOAT16 T L1-sh |
| 17 | attention | experimental.paged_update_cache | attention.py:251:_write_cache | 4 | 3.8 / 4.0 | 3.6 | 1.37 | [4x1x256x512] BFLOAT16 T DRAM |
| 18 | attention | transformer.scaled_dot_product_attention_decode | attention.py:433:forward | 4 | 37.7 / 38.4 | 38.1 | 1.46 | [1x4x9x512] BFLOAT16 T DRAM |
| 19 | attention | multiply | attention.py:273:_rope_heads | 120 | 3.0 / 3.1 | 2.8 | 0.52 | [1x4x9x512] BFLOAT16 T DRAM |
| 20 | attention | linear | attention.py:221:_lin | 8 | 17.5 / 17.9 | 16.6 | 0.60 | [1x4x9x512] BFLOAT16 T DRAM |
| 21 | attention | addcmul | attention.py:273:_rope_heads | 120 | 3.2 / 3.5 | 3.5 | 0.54 | [1x4x9x512] BFLOAT16 T L1-sh |
| 22 | attention | experimental.nlp_concat_heads_decode | attention.py:265:_finish | 9 | 3.3 / 3.6 | 2.9 | 0.57 | [1x1x32x4608] BFLOAT16 T L1-sh |
| 23 | attention | to_memory_config | attention.py:266:_finish | 9 | 1.9 / 2.0 | 1.4 | 0.67 | [1x1x32x4608] BFLOAT16 T DRAM |
| 24 | attention | linear | attention.py:221:_lin | 16 | 23.5 / 24.0 | 23.3 | 0.60 | [1x1x32x1024] BFLOAT16 T DRAM |
| 25 | attention | linear | attention.py:221:_lin | 32 | 21.6 / 22.6 | 21.4 | 0.62 | [1x1x32x5120] BFLOAT16 T DRAM |
| 26 | attention | experimental.reduce_scatter_minimal_async (CCL) | config.py:145:allreduce | 12 | 20.3 / 25.5 | 36.2 | 0.54 | [1x1x32x640] BFLOAT16 T DRAM |
| 27 | attention | experimental.all_gather_async (CCL) | config.py:166:allreduce | 12 | 23.4 / 27.8 | 33.5 | 0.54 | [1x1x32x5120] BFLOAT16 T DRAM |
| 28 | mhc_attn_expand | generic_op [mhc_expand] | mhc_expand.py:62:mhc_expand | 32 | 26.7 / 28.2 | 26.7 | 0.52 | [4x1x4x5120] FLOAT32 T DRAM |
| 29 | mhc_ffn_mixes | generic_op [mhc_proj] | mhc_mixes.py:51:mhc_proj | 32 | 21.8 / 23.6 | 21.4 | 0.59 | [32x1x32x32] FLOAT32 T DRAM |
| 30 | mhc_ffn_mixes | generic_op [mhc_post] | mhc_mixes.py:79:mhc_post | 1 | 46.3 / 46.6 | 46.4 | 0.61 | [4x1x4x4] FLOAT32 T DRAM |
| 31 | mhc_ffn_collapse+norm | generic_op [mhc_collapse] | mhc_collapse.py:74:mhc_collapse | 32 | 12.6 / 13.2 | 12.5 | 0.53 | [32x1x32x32] FLOAT32 T DRAM |
| 32 | mhc_ffn_collapse+norm | generic_op [mhc_norm] | mhc_collapse.py:111:mhc_norm_apply | 32 | 15.5 / 15.9 | 15.1 | 0.53 | [4x1x1x5120] BFLOAT16 RM DRAM |
| 33 | moe | matmul | router.py:158:_forward_exact_fast2 | 6 | 22.1 / 22.5 | 21.6 | 1.01 | [1x1x4x384] FLOAT32 T L1 |
| 34 | moe | add | router.py:160:_forward_exact_fast2 | 120 | 6.3 / 6.5 | 5.5 | 0.54 | [1x1x4x384] FLOAT32 T L1 |
| 35 | moe | add | router.py:162:_forward_exact_fast2 | 120 | 3.1 / 3.2 | 2.3 | 0.53 | [1x1x4x384] FLOAT32 T L1 |
| 36 | moe | topk | router.py:163:_forward_exact_fast2 | 12 | 4.7 / 5.1 | 4.4 | 0.62 | [1x1x4x6] FLOAT32 T L1 |
| 37 | moe | topk | router.py:163:_forward_exact_fast2 | 1 | 62.9 / 63.0 | 62.8 | 1.60 | [1x1x4x6] FLOAT32 T L1 |
| 38 | moe | typecast | router.py:165:_forward_exact_fast2 | 1 | 1.1 / 1.1 | 1.0 | 1.45 | [1x1x4x6] FLOAT32 T L1 |
| 39 | moe | reshape | router.py:170:_forward_exact_fast2 | 4 | 2.1 / 2.2 | 1.4 | 0.61 | [4x1x1x6] FLOAT32 T L1 |
| 40 | moe | eq | router.py:170:_forward_exact_fast2 | 120 | 7.5 / 7.6 | 7.3 | 0.51 | [4x1x384x6] FLOAT32 T L1 |
| 41 | moe | reshape | router.py:171:_forward_exact_fast2 | 48 | 2.2 / 2.3 | 1.8 | 0.61 | [4x1x1x384] FLOAT32 T L1 |
| 42 | moe | matmul | router.py:171:_forward_exact_fast2 | 4 | 9.5 / 10.3 | 9.1 | 1.00 | [4x1x1x6] FLOAT32 T L1 |
| 43 | moe | matmul | router.py:173:_forward_exact_fast2 | 4 | 3.1 / 3.2 | 2.5 | 0.61 | [4x1x1x6] FLOAT32 T L1 |
| 44 | moe | div | router.py:174:_forward_exact_fast2 | 120 | 3.9 / 4.0 | 3.0 | 0.52 | [4x1x1x6] BFLOAT16 T L1 |
| 45 | moe | to_layout | router.py:175:_forward_exact_fast2 | 4 | 1.9 / 1.9 | 1.1 | 0.61 | [4x1x1x6] BFLOAT16 RM DRAM |
| 46 | moe | typecast | router.py:182:_forward_exact_fast2 | 1 | 1.1 / 1.1 | 1.1 | 1.46 | [1x1x4x6] UINT16 T L1 |
| 47 | moe | to_layout | router.py:182:_forward_exact_fast2 | 1 | 1.8 / 1.9 | 1.8 | 1.46 | [1x1x4x6] UINT16 RM DRAM |
| 48 | moe | to_memory_config | tt_moe_decode.py:900:_format_dispatch_inputs | 4 | 2.0 / 2.2 | 1.5 | 0.61 | [4x1x1x5120] BFLOAT16 RM L1 |
| 49 | moe | to_memory_config | tt_moe_decode.py:908:_format_dispatch_inputs | 4 | 1.6 / 1.7 | 0.9 | 0.68 | [4x1x1x6] UINT16 RM L1 |
| 50 | moe | to_memory_config | tt_moe_decode.py:919:_format_dispatch_inputs | 4 | 1.4 / 1.5 | 0.6 | 0.68 | [4x1x1x6] BFLOAT16 RM L1-sh |
| 51 | moe | experimental.all_to_all_dispatch_metadata (CCL) | tt_moe_decode.py:972:forward | 2 | 14.9 / 27.5 | 22.3 | 0.97 | [1x16x5120] BFLOAT16 RM DRAM |
| 52 | moe | experimental.moe_compute (CCL) | tt_moe_decode.py:1001:forward | 32 | 358.0 / 522.0 | 406.7 | 0.62 | [120x12] UINT32 RM L1-sh |
| 53 | moe | tilize_with_val_padding | tt_moe_decode.py:1040:forward | 80 | 24.5 / 26.6 | 24.4 | 0.50 | [6x1x4x5120] BFLOAT16 T L1-sh |
| 54 | moe | experimental.deepseek_moe_fast_reduce_nc_fused | tt_moe_decode.py:1049:forward | 120 | 14.7 / 16.9 | 14.6 | 0.52 | [1x1x4x5120] BFLOAT16 T DRAM |
| 55 | moe | reduce_scatter (CCL) | tt_moe_decode.py:1089:forward | 12 | 181.4 / 414.0 | 53.5 | 0.68 | [1x1x4x640] BFLOAT16 T DRAM |
| 56 | moe_allgather | experimental.all_gather_async (CCL) | config.py:194:allgather | 12 | 24.9 / 29.9 | 43.1 | 0.54 | [1x1x4x5120] BFLOAT16 T DRAM |
| 57 | shared_expert | linear | shared_expert_v2.py:90:<lambda> | 36 | 65.5 / 66.6 | 65.5 | 0.52 | [1x1x4x4608] BFLOAT16 T DRAM |
| 58 | shared_expert | slice | shared_expert_v2.py:112:forward | 72 | 1.8 / 2.1 | 1.4 | 0.56 | [1x1x4x2304] BFLOAT16 T DRAM |
| 59 | shared_expert | slice | shared_expert_v2.py:112:forward | 72 | 1.8 / 1.9 | 1.4 | 0.56 | [1x1x4x2304] BFLOAT16 T DRAM |
| 60 | shared_expert | multiply | shared_expert_v2.py:92:<lambda> | 120 | 4.5 / 4.7 | 4.2 | 0.53 | [1x1x4x2304] BFLOAT16 T DRAM |
| 61 | shared_expert | linear | shared_expert_v2.py:90:<lambda> | 40 | 37.9 / 40.0 | 37.9 | 0.54 | [1x1x4x5120] FLOAT32 T DRAM |
| 62 | mhc_ffn_expand | generic_op [mhc_expand] | mhc_expand.py:62:mhc_expand | 32 | 27.7 / 29.5 | 28.0 | 0.54 | [4x1x4x5120] FLOAT32 T DRAM |

Ops with kernel < 30 us: 54 of 63, kernel 525 us + gap 55 us. Ops < 5 us: 28 (kernel floor ~1.1-1.8 us, gap 0.5-1.5 us each, i.e. launch-bound).

## Fusion / removal candidates (per layer, kernel+gap before -> assumed target after)

Saving per layer = (sum of kernel+gap of listed ops) - target, averaged over layers 2 and 3 (both listed). x40 = per decode token over 40 layers (assumes layers like 2/3; layers with Engram or ratio-4 attention not measured). Effort 1 (trivial) .. 5 (new kernel + validation). Targets are estimates, not measured.

| rank | candidate | L2 before us | L3 before us | target us | saving/layer us | saving/token (x40) ms | effort | saving/effort |
|---|---|---|---|---|---|---|---|---|
| 1 | Fused router gate (generalized_moe_gate) replacing softplus/sqrt/add/topk/eq/sum/div chain | 164 | 123 | 20 | 123 | 4.94 | 3 | 41 |
| 2 | Overlap shared expert with moe_compute (sub-device / separate cores) | 114 | 114 | 15 | 99 | 3.96 | 3 | 33 |
| 3 | mHC: fuse proj+post+collapse+norm (x2 per layer); single-core post 46us -> multicore | 210 | 210 | 90 | 120 | 4.79 | 4 | 30 |
| 4 | RoPE as multiply+linear+addcmul (x2 pairs) -> one fused rotary op | 51 | 51 | 16 | 35 | 1.39 | 2 | 17 |
| 5 | Remove to_memory_config layout shuffles (choose producer layout/sharding) | 14 | 14 | 0 | 14 | 0.57 | 1 | 14 |
| 6 | mHC expand + next-layer proj (cross-layer fusion) | 55 | 55 | 0 | 55 | 2.22 | 4 | 14 |
| 7 | L2 only: compressor/indexer-score elementwise chain (subtract, slices, sigmoid, addcmul, typecast, rms_norm, copy, multiply, addcmul) -> one eltwise kernel | 38 | 0 | 12 | 26 | 1.04 | 2 | 13 |
| 8 | q/kv lora: slice+rms_norm(1 core)+slice+rms_norm+concat -> split matmuls + fused multicore dual rmsnorm | 31 | 30 | 12 | 18 | 0.74 | 2 | 9 |
| 9 | MoE tail: tilize_with_val_padding + fast_reduce -> emit tiled from moe_compute / fuse | 40 | 40 | 15 | 25 | 1.00 | 3 | 8 |
| 10 | Attn allreduce reduce_scatter+all_gather -> single all_reduce / fused CCL | 45 | 45 | 30 | 15 | 0.59 | 3 | 5 |

Other findings
- Trace replay shows kernel time dominates: gaps are 44-60 us/layer (3-4%). Only ops after single-core ops (topk, rms_norm, nlp heads, paged_update_cache, SDPA) show 1-2.4 us gaps; the 17.5 us gap on op 0 is the layer-entry wait, not an op gap.
- Launch-bound ops (kernel 1-3 us, gap 0.5-1.5 us): slice, typecast, to_memory_config, reshape, to_layout, multiply/add/addcmul on [4x..] tiles. They cost ~2-4 us each end to end; removing one saves about that.
- Kernel-bound: moe_compute (282 us L2 / 358 us L3 median, slowest device 463 / 522), shared-expert linears (65 + 38 us), SDPA 38 us, mHC post 46 us on ONE core (x2), router topk 63 us on ONE core, wq/wkv linears 21-34 us.
- MoE reduce_scatter reads 181-194 us in trace (48-53 us eager): it is the straggler wait for the slowest device's moe_compute (max 463-522 us vs median 282-358 us), not communication. Expert-load imbalance across devices costs ~100-150 us/layer; balancing/ dropping the sync is a bigger win than any small-op fusion but is not a fusion.
- moe_compute + its reduce_scatter + dispatch + allgather = ~540 us (L2) / ~600 us (L3) = ~37-41% of the layer.
- Profiler: the first run dropped markers in the long 50-iteration trace-timing and warm-up phases only; the eager pass (85/63 programs x 32 devices) and trace sessions 4+ are complete for both layers. Digest op names are empty, names come from the recorder.
