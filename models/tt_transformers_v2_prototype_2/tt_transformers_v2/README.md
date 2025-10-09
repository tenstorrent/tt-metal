# TT-Transformers V2 (TTTv2)

**Library, not Framework** — Pure Python building blocks for transformer models on Tenstorrent hardware.

## Quick Start

```bash
pip install tt-transformers-v2
```

## Usage

```python
from tt_transformers_v2.building_blocks import AttentionSpec, FFNSpec
from tt_transformers_v2.testing import TestSuite

# Define model components
attn_spec = AttentionSpec(
    hidden_dim=4096,
    num_heads=32,
    num_kv_heads=8,  # GQA support
    max_seq_len=2048
)

# Get default implementation config
from tt_transformers_v2.building_blocks.attention import get_default_impl_config
impl_config = get_default_impl_config(attn_spec, device="N150", mode="prefill")

# Use building blocks directly
output, cache = attention_prefill_forward(
    hidden_states,
    spec=attn_spec,
    impl_config=impl_config
)
```

## Key Features

- **Modular Building Blocks**: Attention, FFN, Normalization, Embeddings
- **Separation of Concerns**: Mathematical specs vs. TTNN implementation configs
- **Device-Aware Defaults**: Automatic optimization for different Tenstorrent hardware
- **Stateless & Functional**: Pure functions for easy testing and composition
- **Optional Patterns**: Pre-built decoder/encoder layers (use if helpful)

## Architecture

TTTv2 provides building blocks that models compose:
- Core owns NO models — models live in separate repos
- Models pin to specific TTTv2 versions
- Adding model N+1 is O(1), not O(N)

## Documentation

See [TTTv2 Design Document](docs/TTTv2_design.md) for detailed architecture and design principles.

## License

Apache 2.0
