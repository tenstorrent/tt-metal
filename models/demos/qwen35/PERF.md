# Qwen3.5-27B Performance

Performance collected from [demo/demo.py](demo/demo.py) on single P100A Blackhole.

## Decode Performance

| Model | Device | Precision | Speed (t/s/u) |
|---|---|---|---|
| Qwen3.5-27B | P100A | bfp8 weights, f32 recurrence | 2.88 |

## DRAM Usage

| Component | Dtype | Size |
|---|---|---|
| DeltaNet projections (48 layers) | bfp8 | 5.4 GB |
| MLP w1+w2+w3 (64 layers) | bfp8 | 17.1 GB |
| Attention QKV+WO+gate (16 layers) | bf16 | 2.2 GB |
| Other | bf16 | ~0.3 GB |
| **Total** | | **~25 GB / 28 GB** |

## Known Bottlenecks

- DeltaNet recurrence on host (~15% overhead): 5 tensor transfers per layer per token
- GatedAttention custom RoPE on host: partial rotation for head_dim=256
- Host embedding + CPU LM head: 248K vocab too large for device
