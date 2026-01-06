# TT-Train Configuration Guide

This directory contains YAML configuration files for training transformer models with TT-Metal. This README explains all valid parameters for each configuration type based on the actual implementation.

## Configuration Types

There are four main configuration types:
- **Training Config**: Training hyperparameters and optimization settings
- **Device Config**: Device mesh and distributed training setup; *this is expected to be in the same file as the training config*
- **Model Config**: Model type and architecture configuration
- **MultiHost Config**: Multi-process execution and pipeline parallelism settings

## Training Configuration (`training_config`)

Training hyperparameters and optimization settings.

### Core Training Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_name` | str | "tt_train_nano_gpt" | Project name for tracking |
| `seed` | int | 5489 | Random seed for reproducibility |
| `model_save_interval` | int | 500 | Save model every N steps |
| `batch_size` | int | 4 | Batch size for training |
| `num_epochs` | int | 1 | Number of training epochs |
| `max_steps` | int | 1000 | Maximum number of training steps |
| `gradient_accumulation_steps` | int | 1 | Number of steps to accumulate gradients |
| `model_config` | str | "" | Path to model configuration file |
| `data_path` | str | "DATA_FOLDER/shakespeare.txt" | Path to training data |
| `scheduler_type` | str | "identity" | Learning rate scheduler ("identity", "warmup_linear") |
| `tokenizer_type` | str | "char" | Tokenizer type ("char" or "bpe") |

### Optimizer Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | 3e-4 | Learning rate |
| `beta1` | float | 0.9 | Adam beta1 parameter |
| `beta2` | float | 0.999 | Adam beta2 parameter |
| `eps` | float | 1e-8 | Adam epsilon parameter |
| `weight_decay` | float | 1e-2 | Weight decay for regularization |
| `use_no_op` | bool | false | Use no-op optimizer (no parameter updates) |
| `use_moreh_adamw` | bool | false | Use Moreh AdamW optimizer |
| `use_kahan_summation` | bool | false | Use Kahan summation in AdamW |

### Gradient Clipping Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_clip_grad_norm` | bool | false | Enable gradient norm clipping |
| `clip_grad_norm_max_norm` | float | 1.0 | Maximum gradient norm for clipping |

### Example
```yaml
training_config:
  project_name: "my_training_project"
  model_type: "llama"
  seed: 5489
  batch_size: 8
  gradient_accumulation_steps: 8
  num_epochs: 1
  max_steps: 5000
  learning_rate: 0.0003
  weight_decay: 0.01
  use_moreh_adamw: true
  use_kahan_summation: false
  use_clip_grad_norm: false
  clip_grad_norm_max_norm: 1.0
  model_config: "configs/model_configs/tinyllama.yaml"
  data_path: "data/my_dataset.txt"
  scheduler_type: "warmup_linear"
  tokenizer_type: "bpe"
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
| `enable_ddp` | bool | false | Enable Distributed Data Parallelism |
| `enable_tp` | bool | false | Enable Tensor Parallelism |

### Constraints
- DDP and TP cannot both be enabled simultaneously
- For DDP: batch_size must be divisible by number of devices
- For TP: vocab_size is automatically rounded up to be divisible by (num_devices * 32)

### Device Mesh Shapes
- Single-device (N150, P150): [1, 1]
- Dual-device (N300, P300): [1, 2]
- LoudBox: [1, 8]
- Single Galaxy: [1, 32]

### Example
```yaml
device_config:
  enable_tp: true
  mesh_shape: [1, 32]
  device_ids: []
```

## Model Configuration (`model_config`)

Model type and architecture configuration loaded from separate files.

### Model Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | str | "gpt2" | Model architecture ("gpt2" or "llama") |
| `model_path` | str | "" | Path to saved model parameters (as HF SafeTensors) |
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
  model_type: "llama"
  model_path: "saved_models/my_model.safetensors"
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

# Additional transformer-specific configuration follows...
```

## MultiHost Configuration (`multihost_config`)

Multi-process execution and pipeline parallelism settings.

### Core MultiHost Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | false | Enable multihost execution |
| `num_workers` | int | 0 | Number of worker processes |
| `socket_type` | str | "mpi" | Communication backend ("mpi" or "fabric") |

### Pipeline Parallel Configuration (`pipeline_parallel_config`)
Optional configuration for pipeline parallelism:

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_blocks` | int | Total number of pipeline blocks |
| `blocks_per_rank` | dict | Mapping of rank ID to number of blocks |

### Training Mode Effects
- **Three-tier training**: When `enabled=true` and no pipeline_parallel_config
- **Pipeline parallel**: When `enabled=true` and pipeline_parallel_config is provided
- **Standard training**: When `enabled=false`

### Constraints
- Gradient clipping is not supported with multihost training
- Model save/load is handled differently in multihost mode
- Seeds are automatically adjusted per rank

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

## Command Line Options

The main executable accepts these command line arguments:

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Training config file path | `$TT_METAL_HOME/tt-train/configs/training_configs/training_shakespeare_nanogpt.yaml` |
| `--multihost` | | Multihost config file path | "" (optional) |
| `--name` | `-n` | Run name | "" |
| `--add_time_to_name` | `-t` | Add timestamp to run name | true |
| `--save_and_exit` | `-s` | Save model and exit (msgpack path) | "" |
| `--safetensors` | | Load model from safetensors path | "" |

## File Organization

```
tt-train/configs/
├── training_configs/          # Training configuration files
├── model_configs/            # Model architecture configurations
├── multihost_configs/        # MultiHost execution configurations (if separated)
└── README.md                 # This file
```

## Configuration Loading

Configurations are loaded using YAML::LoadFile in the main application:

```cpp
// Load configurations from files
TrainingConfig training_config = parse_config(YAML::LoadFile(training_config_name));
DeviceConfig device_config = parse_device_config(YAML::LoadFile(training_config_name));
ModelConfig model_config = parse_model_config(YAML::LoadFile(training_config.model_config));

// Optional multihost config
MultihostConfig multihost_config;
if (!multihost_config_name.empty()) {
    multihost_config = parse_multihost_config(YAML::LoadFile(multihost_config_name));
}
```

## Complete Example Configuration

```yaml
# Complete training configuration file
training_config:
  project_name: "shakespeare_training"
  seed: 5489
  model_save_interval: 500
  batch_size: 8
  gradient_accumulation_steps: 8
  num_epochs: 1
  max_steps: 5000
  learning_rate: 0.0003
  weight_decay: 0.01
  use_moreh_adamw: true
  use_kahan_summation: false
  use_clip_grad_norm: false
  clip_grad_norm_max_norm: 1.0
  model_config: "configs/model_configs/tinyllama.yaml"
  data_path: "data/shakespeare.txt"
  scheduler_type: "warmup_linear"
  tokenizer_type: "char"

device_config:
  enable_tp: true
  mesh_shape: [1, 32]
  device_ids: []

# Optional multihost configuration (separate file)
multihost_config:
  enabled: false
  num_workers: 1
  socket_type: "mpi"
```

## Notes

- Environment variable `TT_METAL_HOME` must be set
- For BPE tokenization, data should be pre-tokenized (space-separated tokens)
- Vocabulary size is automatically adjusted to be divisible by 32 (or 32 * num_devices for TP)
- Model parameters are automatically loaded if both safetensors_path and model_path exist
- Training state (optimizer, scheduler) is saved/loaded separately from model weights

## Supported Model Types

- **GPT-2**: Traditional transformer with learned positional embeddings
- **LLaMA**: Transformer with RMSNorm, SwiGLU activation, and RoPE positional encoding

See the respective model configuration files for architecture-specific parameters.
