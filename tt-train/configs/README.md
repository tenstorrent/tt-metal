# TT-Train Configuration Guide

This directory contains YAML configuration files for training transformer models with TT-Metal. This README explains all valid parameters for each configuration type.

## Configuration Types

There are four main configuration types:
- **Training Config**: Training hyperparameters and optimization settings
- **Device Config**: Device mesh and distributed training setup
- **Transformer Config**: Model architecture parameters
- **MultiHost Config**: Multi-process execution settings

## Training Configuration (`training_config`)

Training hyperparameters and optimization settings.

### Core Training Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | 42 | Random seed for reproducibility |
| `batch_size` | int | 4 | Batch size for training |
| `max_steps` | int | 1000 | Maximum number of training steps |
| `eval_every` | int | 200 | Evaluate model every N steps |
| `gradient_accumulation_steps` | int | 1 | Number of steps to accumulate gradients |
| `model_config` | str | null | Path to model configuration file |
| `use_bpe` | bool | true | Whether to use Byte Pair Encoding |

### Optimizer Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | float | 3e-4 | Learning rate |
| `beta1` | float | 0.9 | Adam beta1 parameter |
| `beta2` | float | 0.999 | Adam beta2 parameter |
| `eps` | float | 1e-8 | Adam epsilon parameter |
| `weight_decay` | float | 0.01 | Weight decay for regularization |

### Example
```yaml
training_config:
  seed: 5489
  batch_size: 8
  gradient_accumulation_steps: 8
  max_steps: 5000
  eval_every: 500
  lr: 0.0003
  weight_decay: 0.01
  model_config: "configs/model_configs/tinyllama.yaml"
  use_bpe: true
```

## Device Configuration (`device_config`)

Device mesh and distributed training configuration.

### Device Mesh Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh_shape` | [int, int] | [1, 1] | Device mesh shape [rows, cols] |
| `device_ids` | [int] | [] | Specific device IDs to use |

### Distributed Training Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_tp` | bool | false | Enable Tensor Parallelism |
| `enable_ddp` | bool | false | Enable Distributed Data Parallelism |

### Notes
- DDP and TP cannot both be enabled simultaneously
- Only `[1, N]` mesh shapes are currently supported (single row)
- Total devices = `mesh_shape[0] * mesh_shape[1]`

### Example
```yaml
device_config:
  enable_tp: true
  mesh_shape: [1, 32]
  device_ids: []
```

## Transformer Configuration (`transformer_config`)

Model architecture parameters for transformer models.

### Base Architecture Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `runner_type` | str | "default" | Type of model runner (`default` or `memory_efficient`)|
| `num_heads` | int | 6 | Number of attention heads |
| `embedding_dim` | int | 384 | Embedding/hidden dimension |
| `dropout_prob` | float | 0.2 | Dropout probability |
| `num_blocks` | int | 6 | Number of transformer blocks |
| `vocab_size` | int | 96 | Vocabulary size |
| `max_sequence_length` | int | 128 | Maximum sequence length |
| `weight_tying` | any | false | Weight tying configuration |

### LLaMA-Specific Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `intermediate_dim` | int | null | Feed-forward intermediate dimension |
| `theta` | float | null | RoPE theta parameter |
| `num_groups` | int | 3 | Number of groups for grouped attention |

### RoPE Scaling (`rope_scaling`)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scaling_factor` | float | null | Scaling factor for RoPE |
| `high_freq_factor` | float | null | High frequency scaling factor |
| `low_freq_factor` | float | null | Low frequency scaling factor |
| `original_context_length` | int | null | Original context length for scaling |

### Example
```yaml
transformer_config:
  num_heads: 32
  embedding_dim: 2048
  num_blocks: 22
  vocab_size: 32000
  max_sequence_length: 2048
  intermediate_dim: 5632
  theta: 10000.0
  rope_scaling:
    scaling_factor: 1.0
    high_freq_factor: 4.0
    low_freq_factor: 1.0
    original_context_length: 8192
```

## MultiHost Configuration (`multihost_config`)

Multi-process execution and pipeline parallelism settings.

### Core MultiHost Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | false | Enable multihost execution |
| `num_workers` | int | 1 | Number of worker processes |
| `socket_type` | str | "mpi" | Communication backend (`mpi`, `fabric`) |

### Pipeline Parallel Configuration (`pipeline_parallel_config`)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_blocks` | int | 0 | Total number of pipeline blocks |
| `blocks_per_rank` | dict | {} | Mapping of rank ID to number of blocks |

### Example
```yaml
multihost_config:
  enabled: true
  num_workers: 4
  socket_type: "fabric"
  pipeline_parallel_config:
    num_blocks: 24
    blocks_per_rank:
      0: 6
      1: 6
      2: 6
      3: 6
```

## File Organization

```
tt-train/configs/
├── training_configs/          # Training configuration files
├── model_configs/            # Model architecture configurations  
├── device_configs/           # Device mesh configurations
├── multihost_configs/        # MultiHost execution configurations
└── README.md                 # This file
```

## Loading Configurations

Configurations can be loaded using the provided utility functions:

```python
from ttml.common.config import (
    get_training_config,
    get_device_config, 
    get_model_config,
    get_multihost_config
)

# Load configurations
training_config = get_training_config("training_shakespeare_tinyllama.yaml")
device_config = get_device_config("device_config.yaml") 
model_config = get_model_config("configs/model_configs/tinyllama.yaml")
multihost_config = get_multihost_config("multihost_config.yaml")
```

## Notes

- All paths in existing configuration files are relative to `TT_METAL_HOME/tt-train/configs/`, but absolute paths are supported
- YAML files support comments using `#`
- Boolean values should be lowercase: `true`/`false`
- Numeric values don't need quotes
- String values should be quoted

## Examples

See the files in each subdirectory for complete examples of each configuration type.