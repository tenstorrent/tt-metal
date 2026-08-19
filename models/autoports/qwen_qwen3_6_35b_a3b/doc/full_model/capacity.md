# Full Model Capacity

The full-model context remains the HF advertised `262144` tokens. No capability
reduction was made because the modeled per-device DRAM footprint and the real
full-model load plus full-context cache allocation both passed.

## Real Allocation Evidence

Artifact: `logs/real_full_model_load_context_alloc.log`

Result:

- `REAL_FULL_MODEL_LOAD_OK`
- `load_s=325.024`
- `cache_alloc_s=4.460`
- `mesh_shape=(2, 2)`
- `num_devices=4`
- `dram_grid=8-1`
- `max_seq_len=262144`
- `full_layers=10`
- `linear_layers=30`
- `page_table_shape=(1, 8192)`

## Modeled Per-Device DRAM

| Component | Bytes | GiB |
| --- | ---: | ---: |
| Transformed TT weights under selected policy | 15,985,073,536 | 14.8873 |
| Runtime state excluding weights | 2,768,207,872 | 2.5781 |
| Total weights plus runtime state | 18,753,281,408 | 17.4654 |

Runtime state breakdown:

| Component | Bytes | GiB |
| --- | ---: | ---: |
| Full-attention KV cache, one layer | 268,435,456 | 0.2500 |
| Full-attention KV cache, 10 layers | 2,684,354,560 | 2.5000 |
| Linear conv plus recurrent state, 30 layers | 16,711,680 | 0.0156 |
| Page table | 32,768 | 0.00003 |
| Decode RoPE tables | 67,108,864 | 0.0625 |

Weight breakdown:

| Component | Bytes |
| --- | ---: |
| Embedding BF16 replicated | 1,017,118,720 |
| Final norm BF16 replicated | 4,096 |
| LM head BF8 flat vocab 4-way | 127,139,840 |
| Full-attention BF8 TP-sharded decoder weights | 136,314,880 |
| Full-attention norms | 92,160 |
| Linear-attention BF8 TP-sharded decoder weights | 505,282,560 |
| Linear conv BF16 TP-sharded | 983,040 |
| Linear state params BF16 sharded | 1,920 |
| Linear norms | 253,440 |
| MoE router/gate BF16 replicated | 42,106,880 |
| MoE shared BF8 TP-sharded | 62,914,560 |
| MoE routed full-attention layers BF4 TP-sharded | 2,013,265,920 |
| MoE routed linear-attention layers BF8 TP-sharded | 12,079,595,520 |

HF checkpoint text tensors contain `693` tensors and `34,660,610,688` text
parameters, corresponding to `69,321,221,376` raw BF16 bytes (`64.5604 GiB`)
before the selected transformed TT policy and mesh sharding.
