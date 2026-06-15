# MoE Experts Module

Generic, reusable MoE expert implementation with clean decode/prefill separation.

## Structure

```
experts/
├── __init__.py      # Main Experts class with auto-dispatch
├── config.py        # ExpertConfig + ProgramConfig base
├── weights.py       # Weight loading and sharding
├── operations.py    # Core operations (swiglu, allreduce, etc)
├── decode.py        # Decode forward (seq_len=1)
└── prefill.py       # Prefill forward (seq_len>1)
```

## Usage

```python
from models.demos.gpt_oss.tt.experts import Experts, ExpertConfig
from models.demos.gpt_oss.tt.expert_configs import GPTOSSProgramConfig

# Create config
config = ExpertConfig(
    intermediate_size=8192,
    num_experts=8,
    hidden_size=4096,
    num_experts_per_tok=2,
    swiglu_limit=30.0,
)

# Create experts with model-specific program config
experts = Experts(
    mesh_device=mesh_device,
    config=config,
    state_dict=state_dict,
    ccl_manager=ccl_manager,
    mesh_config=mesh_config,
    program_config=GPTOSSProgramConfig(),  # Model-specific
    weight_dtype=ttnn.bfloat4_b,
)

# Forward (auto-detects decode/prefill)
output = experts(hidden_states, routing_weights)
```

## Customization

Override `ProgramConfig` for different models:

```python
# models/demos/your_model/tt/expert_configs.py
@dataclass
class YourModelProgramConfig:
    # Core grid sizes
    decode_gate_up_cores: tuple[int, int] = (4, 6)
    decode_down_cores: tuple[int, int] = (6, 8)

    # Sparse matmul parameters
    decode_gate_up_in0_block_w: int = 32
    decode_gate_up_subblock_w: int = 2
    decode_down_in0_block_w: int = 16
    decode_down_subblock_w: int = 4

    # Chunking
    sequence_chunk_size: int = 2048
```
