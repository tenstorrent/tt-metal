# Model Configuration System

This directory contains a new Pydantic-based model configuration system that standardizes configuration parsing across different LLM formats.

## Overview

The system supports multiple model configuration formats and converts them to a standardized format that can be consumed by the existing `model_config.py` without breaking changes.

### Supported Formats

- **Meta Llama** (`params.json`): Original Meta format with fields like `dim`, `n_layers`, etc.
- **HuggingFace Llama** (`config.json`): HuggingFace format with `hidden_size`, `num_hidden_layers`, etc.
- **Qwen2.5** (`config.json`): Qwen model format with sliding window attention
- **DeepSeek V3** (`config.json`): DeepSeek format with MoE (Mixture of Experts) support

## Files

- `model_configs.py`: Core Pydantic models and parsing logic
- `config_integration.py`: Integration helpers for existing codebase
- `test_model_configs.py`: Test script to verify parsing works correctly
- `CONFIG_README.md`: This documentation file

## Quick Start

### 1. Install Dependencies

```bash
pip install pydantic>=2.0.0
```

### 2. Basic Usage

```python
from tt.model_configs import parse_model_config, get_standard_config

# Parse from file
standard_config = parse_model_config("path/to/config.json")

# Parse from directory (auto-detects config.json or params.json)
standard_config = get_standard_config("path/to/checkpoint_dir")

# Parse from dictionary
config_dict = {...}  # Your config dictionary
standard_config = parse_model_config_from_dict(config_dict)

# Access standardized fields
print(f"Model dimension: {standard_config.dim}")
print(f"Number of layers: {standard_config.n_layers}")
print(f"Number of heads: {standard_config.n_heads}")
```

### 3. Integration with Existing ModelArgs

```python
from tt.config_integration import enhanced_set_model_params

# Drop-in replacement for existing _set_model_params
enhanced_set_model_params(model_args, checkpoint_dir)
```

## Architecture

### StandardModelConfig

The `StandardModelConfig` class is the unified output format that all specific model configurations convert to. It includes:

- **Core dimensions**: `dim`, `n_layers`, `n_heads`, `n_kv_heads`, `vocab_size`
- **MLP configuration**: `hidden_dim`, `ffn_dim_multiplier`, `multiple_of`
- **RoPE parameters**: `rope_theta`, `rope_scaling`, `rope_scaling_factor`
- **Vision parameters**: For multimodal models
- **MoE parameters**: For mixture of experts models
- **Metadata**: `model_type`, `architecture`

### Format Detection

The system automatically detects the configuration format based on:

1. **Field presence**: Meta format has `dim` and `n_layers`, HF has `hidden_size` and `num_hidden_layers`
2. **Architecture field**: HF configs have `architectures` list
3. **Model type**: Some configs have explicit `model_type` field

### Specific Model Classes

Each supported format has its own Pydantic model:

- `MetaLlamaConfig`: For Meta's `params.json` format
- `HuggingFaceLlamaConfig`: For HF Llama `config.json` format
- `QwenConfig`: For Qwen2.5 models
- `DeepSeekV3Config`: For DeepSeek V3 models with MoE support

## Adding New Model Support

To add support for a new model type:

### 1. Create a new Pydantic model

```python
class NewModelConfig(BaseModel):
    """Configuration for NewModel."""
    # Define all fields with appropriate types
    some_field: int
    another_field: str
    optional_field: Optional[float] = None
    
    def to_standard(self) -> StandardModelConfig:
        """Convert to standard format."""
        return StandardModelConfig(
            dim=self.some_field,
            n_layers=self.another_field,
            # ... map all relevant fields
            architecture=ModelArchitecture.NEW_MODEL,
            model_type="new_model"
        )
```

### 2. Update the detection logic

Add detection logic in `detect_config_format()`:

```python
def detect_config_format(config_data: Dict[str, Any]) -> str:
    # ... existing logic ...
    
    # Check for new model
    if "architectures" in config_data:
        architectures = config_data["architectures"]
        if any("NewModel" in arch for arch in architectures):
            return "new_model"
    
    if "model_type" in config_data:
        model_type = config_data["model_type"]
        if model_type == "new_model":
            return "new_model"
```

### 3. Update the parser functions

Add the new format to `parse_model_config_from_dict()`:

```python
def parse_model_config_from_dict(config_data: Dict[str, Any]) -> StandardModelConfig:
    format_type = detect_config_format(config_data)
    
    if format_type == "new_model":
        config = NewModelConfig(**config_data)
    # ... existing formats ...
    
    return config.to_standard()
```

### 4. Add to ModelArchitecture enum

```python
class ModelArchitecture(str, Enum):
    LLAMA = "llama"
    QWEN2 = "qwen2"
    DEEPSEEK_V3 = "deepseek_v3"
    NEW_MODEL = "new_model"  # Add this
```

## Testing

Run the test script to verify your configuration parsing:

```bash
cd models/tt_transformers/tt
python test_model_configs.py
```

Add test cases for new model types in `test_model_configs.py`.

## Migration Guide

### For Existing Code

The system is designed to be backward compatible. You can migrate gradually:

1. **No changes required**: Existing code continues to work
2. **Optional enhancement**: Use `enhanced_set_model_params()` for improved parsing
3. **Full migration**: Use standardized configs directly

### Example Migration

```python
# OLD: Direct dictionary access with fallbacks
dim = config.get("dim", config.get("hidden_size"))
n_layers = config.get("n_layers", config.get("num_hidden_layers"))

# NEW: Standardized access
standard_config = parse_model_config_from_dict(config)
dim = standard_config.dim
n_layers = standard_config.n_layers
```

## Benefits

1. **Type Safety**: Pydantic provides runtime type checking and validation
2. **Extensibility**: Easy to add new model formats without changing consumer code
3. **Maintainability**: Clear separation of parsing logic from business logic
4. **Documentation**: Self-documenting with field descriptions and types
5. **Backward Compatibility**: Existing code continues to work unchanged

## Error Handling

The system includes robust error handling:

- **Graceful fallback**: If Pydantic parsing fails, falls back to legacy parsing
- **Clear error messages**: Descriptive errors for debugging
- **Optional dependency**: Works even if Pydantic is not installed (with warnings)

## Examples

See `test_model_configs.py` for comprehensive examples of:

- Parsing Meta Llama configurations
- Parsing HuggingFace Llama configurations  
- Parsing Qwen2.5 configurations
- Parsing DeepSeek V3 configurations with MoE

Each example shows the input format and the resulting standardized output. 