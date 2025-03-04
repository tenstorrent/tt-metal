# Qwen2.5-VL Conversion Framework

This repository contains a framework for converting the Qwen2.5-VL model from its original PyTorch module implementation to a clean, functional implementation that's easier to understand and modify.

## Installation

Install the Qwen2.5-VL model dependencies:

```bash
pip install -r requirements.txt
```

Note that this uses the main branch of transformers, not a release. It also requires python 3.9 or higher.

## Overview

The conversion process follows these steps:

1. **Instrumentation**: Using `instrument.py` to capture inputs and outputs of module calls in the original model
2. **Recording**: Running `record.py` to generate test data from model execution
3. **State Dict Conversion**: Using `convert.py` to transform flat state dicts to nested dictionaries
4. **Functional Reimplementation**: Creating clean functional implementations in `functional.py`
5. **Testing**: Using `test_functional.py` to validate functional implementations against recorded data

## Instrumentation (instrument.py)

`instrument.py` provides a decorator that records the inputs, outputs, and settings of model modules during execution:

```python
@instrument("ModuleName")
def forward(self, *args, **kwargs):
    # Original module code...
```

When applied to module forward methods, this decorator:

- Creates a timestamped directory for each call
- Saves input tensors and parameters
- Captures module configuration
- Records output tensors
- Stores metadata in a JSON file

This instrumentation is critical for understanding how the original model works and generates data for testing functional reimplementations.

### Directory Structure

Instrumented calls create this directory structure:
```
module_io_data/
    ModuleName_TIMESTAMP/
        metadata.json      # Contains settings and references to tensors
        inputs/           # Directory for input tensors
            arg_0.pt      # Input tensors
            kwarg_name.pt
        outputs/          # Directory for output tensors
            output.pt
```

## Recording (record.py)

`record.py` loads the model and runs it on sample inputs to generate the instrumented data:

```python
python record.py
```

This script:
1. Loads the Qwen2.5-VL model with instrumented modules
2. Processes a sample image input
3. Performs model inference
4. Generates instrumented data for each module's execution

After running, the `module_io_data` directory will contain recordings for each module's execution, providing test data for the functional implementations.

## State Dict Conversion (convert.py)

`convert.py` transforms the model's flat state dictionaries into nested dictionaries that mirror the module hierarchy:

```python
python convert.py
```

### Purpose

The script serves these critical functions:

1. **Structure Transformation**: Converts flat state_dict keys like `'model.blocks.0.norm1.weight'` into intuitive nested structures: `model['blocks']['0']['norm1']['weight']`
2. **Vision Component Extraction**: Isolates just the vision-related components from the full model
3. **Weight Preparation**: Creates a structured weights file used by the functional implementations

### How It Works

The conversion process follows these steps:

1. Load the pretrained Qwen2.5-VL model
2. Extract the flat state dictionary
3. Recursively build a nested dictionary structure by splitting key paths
4. Extract only vision-related components (patch embedding, blocks, rotary embeddings, etc.)
5. Save the nested dictionary to `weights/vision_weights.pt`

Example of the transformation:
```python
# Original flat dict
flat_dict = {
    'visual.patch_embed.proj.weight': tensor(...),
    'visual.blocks.0.norm1.weight': tensor(...)
}

# Converted nested dict
nested_dict = {
    'patch_embed': {
        'proj': {
            'weight': tensor(...)
        }
    },
    'blocks': {
        '0': {
            'norm1': {
                'weight': tensor(...)
            }
        }
    }
}
```

This structured format makes the functional implementations much more intuitive and easier to debug, as they can directly access the weights with a hierarchy that matches the original module structure.

## Functional Reimplementation (functional.py)

`functional.py` contains clean, functional implementations of the model's modules:

- Each module is reimplemented as a simple function
- Functions match input/output signatures of original modules
- Implementations use pure PyTorch operations
- Code is simplified and well-documented

For example, the RMSNorm implementation:

```python
def qwen2_rms_norm(x: torch.Tensor, state_dict: Dict, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm with weight scaling."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * state_dict['weight']
```

### State Dict to Nested Dict Conversion

A key aspect of the functional implementation is converting PyTorch's flat state dictionaries to nested dictionaries that match the module hierarchy. This makes the functional implementations more intuitive:

1. The original model uses a flat state dict with keys like `"blocks.0.norm1.weight"`
2. The functional implementation converts this to a nested structure: `state_dict['blocks']['0']['norm1']['weight']`

This conversion happens when loading the weights for testing:

```python
# Example of nested structure in the vision weights
weights_path = 'weights/vision_weights.pt'
vision_weights = torch.load(weights_path, weights_only=False)

# Using the nested structure
block_weights = vision_weights['blocks']['0']
```

## Testing (test_functional.py)

`test_functional.py` validates the functional implementations against the recorded data:

```python
pytest test_functional.py
```

For each module:
1. Loads the earliest recorded run from `module_io_data`
2. Extracts inputs and expected outputs
3. Runs the functional implementation with the same inputs
4. Compares results using Pearson correlation coefficient
5. Verifies the correlation exceeds 0.999

This approach ensures that the functional implementations faithfully reproduce the behavior of the original model.

## Usage Example

To convert and test a new module:

1. Apply the `@instrument` decorator to the module's forward method
2. Run `record.py` to generate test data
3. Run `convert.py` to create the nested weights structure
4. Implement the functional version in `functional.py`
5. Add a test case in `test_functional.py`
6. Run the test to validate the implementation

## Vision Transformer Specifics

The conversion particularly focuses on the vision components of Qwen2.5-VL, including:

- Patch embedding
- Self-attention mechanisms
- MLP blocks
- Rotary position embeddings
- Window attention mechanisms

These components handle the visual processing in Qwen2.5-VL before integration with the language model.

## Conclusion

This framework enables a systematic conversion of the complex Qwen2.5-VL model into a more understandable functional implementation, facilitating research, optimization, and educational purposes.
