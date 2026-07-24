# Attention Module

Generic attention implementation with clean decode/prefill separation and configurable matmul program configs.

## Structure

```
attention/
├── __init__.py      # Main Attention class with auto-dispatch
├── config.py        # AttentionConfig + ProgramConfig base
├── weights.py       # Weight loading (fused QKV)
├── kv_cache.py      # KV cache initialization
├── operations.py    # Common ops (RoPE, head ops, projections)
├── decode.py        # Decode forward (seq_len=1)
└── prefill.py       # Prefill forward (seq_len>1)
```

## Usage

```python
from models.demos.gpt_oss.tt.attention import Attention, AttentionConfig
from models.demos.gpt_oss.tt.attention_configs import GPTOSSAttentionProgramConfig

# Create config
config = AttentionConfig(
    hidden_size=4096,
    num_heads=32,
    num_kv_heads=8,
    head_dim=128,
    max_seq_len=131072,
    sliding_window=4096,
)

# Create attention with model-specific program config
attention = Attention(
    mesh_device=mesh_device,
    config=config,
    state_dict=state_dict,
    ccl_manager=ccl_manager,
    mesh_config=mesh_config,
    program_config=GPTOSSAttentionProgramConfig(),
    layer_idx=0,
    transformation_mats=transformation_mats,
    weight_dtype=ttnn.bfloat8_b,
)

# Forward (auto-detects decode/prefill)
output = attention(hidden_states, rope_mats, position_idx, page_table)
```

## Customization

Override `ProgramConfig` for different models or matmul configs:

```python
# models/demos/your_model/tt/attention_configs.py
@dataclass
class YourModelProgramConfig:
    # SDPA configs
    decode_k_chunk_size: int = 256
    prefill_q_chunk_size_large: int = 512

    # Matmul program configs (optional)
    decode_qkv_cores: tuple[int, int] | None = (8, 8)
    decode_qkv_in0_block_w: int = 2
    decode_qkv_out_subblock_w: int = 2

    decode_out_cores: tuple[int, int] | None = (6, 6)
    decode_out_in0_block_w: int = 4
```
