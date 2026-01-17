# Model Trace to Test Automation Guide

## Overview

Automatically extracts real-world operation configurations from model tests and integrates them into sweep tests. Tests run with the exact configurations that production models use.

**Benefits:**
- ✅ Test with real model configurations (EfficientNet, ResNet, BERT, etc.)
- ✅ Automatic extraction and deduplication
- ✅ Simple 2-step integration into sweep tests
- ✅ Captures shapes, dtypes, layouts, and exact shard specs

### Currently Traced Models

The following models have been traced and their configurations are available in `ttnn_operations_master.json`:

**N300 Machine:**

| Model | Purpose | Pytest command used for tracing |
|-------|---------|---------------------------------|
| meta-llama/Llama-3.2-11B-Vision-Instruct | Vision-language model | `HF_MODEL=meta-llama/Llama-3.2-11B-Vision-Instruct python model_tracer/generic_ops_tracer.py models/tt_transformers/demo/simple_text_demo.py::test_demo_text` |
| meta-llama/Llama-3.2-11B-Vision | Vision model | `HF_MODEL=meta-llama/Llama-3.2-11B-Vision python model_tracer/generic_ops_tracer.py models/tt_transformers/demo/simple_text_demo.py::test_demo_text` |
| meta-llama/Llama-3.2-1B-Instruct | Small instruction-tuned model | `HF_MODEL=meta-llama/Llama-3.2-1B-Instruct python model_tracer/generic_ops_tracer.py models/tt_transformers/demo/simple_text_demo.py::test_demo_text` |
| Qwen/Qwen2.5-Coder-7B-Instruct | Code generation model | `HF_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct python model_tracer/generic_ops_tracer.py models/tt_transformers/demo/simple_text_demo.py::test_demo_text` |
| deepseek-ai/deepseek-llm-7b-chat | Chat model | `HF_MODEL=deepseek-ai/deepseek-llm-7b-chat python model_tracer/generic_ops_tracer.py models/tt_transformers/demo/simple_text_demo.py::test_demo_text` |
| ilsp/Llama-Krikri-8B-Instruct | Instruction-tuned model | `HF_MODEL=ilsp/Llama-Krikri-8B-Instruct python model_tracer/generic_ops_tracer.py models/tt_transformers/demo/simple_text_demo.py::test_demo_text` |
| microsoft/Phi-3-mini-128k-instruct | Small instruction model | `HF_MODEL=microsoft/Phi-3-mini-128k-instruct python model_tracer/generic_ops_tracer.py models/tt_transformers/demo/simple_text_demo.py::test_demo_text` |
| mistralai/Mistral-7B-Instruct-v0.3 | Instruction-tuned model | `HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3 python model_tracer/generic_ops_tracer.py models/tt_transformers/demo/simple_text_demo.py::test_demo_text` |
| efficientnetb0 | EfficientNet-B0 vision model | `python model_tracer/generic_ops_tracer.py models/experimental/efficientnetb0/tests/pcc/test_ttnn_efficientnetb0.py::test_efficientnetb0_model` |
| vit | Vision Transformer | `python model_tracer/generic_ops_tracer.py models/demos/wormhole/vit/demo/demo_vit_performant_imagenet_inference.py` |
| ssd512 | Object detection | `python model_tracer/generic_ops_tracer.py models/experimental/SSD512/tests/perf/test_device_perf_ssd.py` |
| stable-diffusion-xl | Image generation | `python model_tracer/generic_ops_tracer.py models/experimental/stable_diffusion_xl_base/demo/demo_img2img.py` |
| whisper | Audio transcription | `python model_tracer/generic_ops_tracer.py models/demos/whisper/demo/demo.py` |
| gemma-3 | Language model | `python model_tracer/generic_ops_tracer.py models/demos/gemma3/demo/text_demo.py` |
| falcon7b | Language model | `python model_tracer/generic_ops_tracer.py models/demos/wormhole/falcon7b/demo_wormhole.py` |
| sentence-bert | Sentence embeddings | `python model_tracer/generic_ops_tracer.py models/demos/wormhole/sentence_bert/demo/demo.py` |
| segmentation | Image segmentation | `python model_tracer/generic_ops_tracer.py models/demos/segmentation_evaluation/test_segmentation_eval.py` |

**T3K Machine:**

| Model | Purpose | Pytest command used for tracing |
|-------|---------|---------------------------------|
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | Distilled reasoning model | `HF_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B python model_tracer/generic_ops_tracer.py models/tt_transformers/demo/simple_text_demo.py::test_demo_text` |
| Qwen/Qwen2.5-Coder-32B | Large code generation model | `HF_MODEL=Qwen/Qwen2.5-Coder-32B python model_tracer/generic_ops_tracer.py models/tt_transformers/demo/simple_text_demo.py::test_demo_text` |

**WH Galaxy (32 cards):**

| Model | Purpose | Pytest command used for tracing |
|-------|---------|---------------------------------|
| **DeepSeek V3** | 671B parameter MoE model | `python model_tracer/generic_ops_tracer.py models/demos/deepseek_v3/demo/demo.py --prompts-file models/demos/deepseek_v3/demo/test_prompts.json --output-path deepseek_tt_out_batch_4.json --max-new-tokens 128 --model-path $DEEPSEEK_V3_HF_MODEL --cache-dir $DEEPSEEK_V3_CACHE --num-prompts 128` |


These traced configurations provide real-world operation patterns from production models, ensuring sweep tests validate against actual usage scenarios.

*Last updated: January 9, 2026*

---

## Quick Reference

### Common Commands

| Task | Command |
|------|---------|
| **Trace a model** | `python model_tracer/generic_ops_tracer.py <test_path>` |
| **View configurations** | `python model_tracer/analyze_operations.py <operation_name>` |
| **Generate sweep vectors** | `python3 tests/sweep_framework/sweeps_parameter_generator.py --module-name <op_name>` |
| **Run single sweep test** | `python3 tests/sweep_framework/sweeps_runner.py --module-name <op_name> --suite-name model_traced --vector-source file --file-path <vector_file> --result-dest results_export` |

### Key Files

- **Tracer**: `model_tracer/generic_ops_tracer.py` - Employs methodology described in the [graph tracing tech report](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ttnn/graph-tracing.md)
- **Master JSON**: `model_tracer/traced_operations/ttnn_operations_master.json` - Contains all traced configurations
- **Analyzer**: `model_tracer/analyze_operations.py` - Query and view configurations
- **Config Loader**: `tests/sweep_framework/master_config_loader.py` - Converts JSON configs to sweep test parameters

---

## Master JSON Format

The `ttnn_operations_master.json` file stores traced configurations in a structured format. The loader supports both legacy and new formats for backward compatibility.

### Configuration Formats

**Legacy Format (Single Source):**
```json
{
  "operations": {
    "ttnn::silu": {
      "configurations": [
        {
          "arguments": [...],
          "source": "models/demos/model_name/demo.py",
          "machine_info": [
            {
              "board_type": "Wormhole",
              "device_series": "n300",
              "card_count": 1
            }
          ]
        }
      ]
    }
  }
}
```

**New Format (Contexts with Multiple Sources):**

The new format supports multiple execution contexts per configuration, enabling the same operation arguments to be traced from different models and hardware configurations:

```json
{
  "operations": {
    "ttnn::silu": {
      "configurations": [
        {
          "arguments": [...],
          "contexts": [
            {
              "source": ["models/demos/deepseek_v3/demo/demo.py"],
              "machine_info": [
                {
                  "board_type": "Wormhole",
                  "device_series": "tt-galaxy-wh",
                  "card_count": 32,
                  "tensor_placements": [
                    {
                      "mesh_device_shape": "[4, 8]"
                    }
                  ]
                }
              ]
            }
          ]
        }
      ]
    }
  }
}
```

### Multi-Chip Mesh Configuration

For multi-chip operations (Galaxy, T3K, etc.), the `machine_info` should include `tensor_placements` with `mesh_device_shape` to enable proper runner assignment in CI:

```json
{
  "machine_info": [
    {
      "board_type": "Wormhole",
      "device_series": "tt-galaxy-wh",
      "card_count": 32,
      "tensor_placements": [
        {
          "mesh_device_shape": "[4, 8]"
        }
      ]
    }
  ]
}
```

**Note:** Only `mesh_device_shape` is used by the sweep framework for runner assignment. Other tensor placement fields (like `shard_mesh`, `tensor_layout`) may be captured during tracing for informational purposes but are not used for CI routing.

**Mesh Shape Values:**
| Mesh Shape | Description | Runner Assignment |
|------------|-------------|-------------------|
| `[1, 1]` | Single-chip | N150 runner |
| `[1, 2]` | 2-chip | N300/Galaxy runner |
| `[1, 4]` | 4-chip | T3K/Galaxy runner |
| `[2, 4]` | 8-chip | T3K/Galaxy runner |
| `[4, 8]` | 32-chip | Galaxy runner |
| `[8, 4]` | 32-chip (alt layout) | Galaxy runner |

The sweep framework automatically:
1. Groups vectors by mesh shape during generation
2. Creates separate JSON files per mesh (e.g., `op__mesh_4x8.json`)
3. Routes tests to appropriate hardware runners in CI

---

## Integration Pattern

**Just 2 simple steps:**

```python
# 1. Import and load
from tests.sweep_framework.master_config_loader import MasterConfigLoader

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("your_operation_name")

# 2. Add to parameters
parameters = {
    "nightly": { ... },
    "model_traced": model_traced_params,  # Add this line!
}
```

**Test Modes:**
- **Default (`all_cases=False`)**: Runs exact N traced configs with deduplication (fast, real-world patterns)
  - Deduplicates configurations with identical input tensor specs
  - Example: 81 configs → ~67 unique configs tested
- **All cases (`all_cases=True`)**: Cartesian product of all parameter combinations (comprehensive, slower)
  - Generates all combinations of shapes, dtypes, layouts, etc.
  - Use with caution - can generate thousands of test vectors

---

## Usage

### 1. Trace a Model

The tracer automatically detects whether to run as pytest or standalone Python script:

```bash
# Pytest test (with ::test_name or if pytest test cases detected)
python model_tracer/generic_ops_tracer.py /path/to/model/test.py::test_function

# Standalone Python script (if no pytest test cases detected)
python model_tracer/generic_ops_tracer.py /path/to/model/demo.py

# Examples - Pytest
python model_tracer/generic_ops_tracer.py models/experimental/efficientnetb0/tests/pcc/test_ttnn_efficientnetb0.py::test_efficientnetb0_model

# Examples - Standalone scripts
python model_tracer/generic_ops_tracer.py models/demos/wormhole/resnet50/demo/demo.py
python model_tracer/generic_ops_tracer.py models/experimental/some_model/run_inference.py

# Keep trace files (default: auto-deleted after adding to master)
python model_tracer/generic_ops_tracer.py <test_path> --store
```

**Detection Logic:**
- If path contains `::` → runs as pytest
- Otherwise, uses `pytest --collect-only` to detect test cases
- If no pytest tests found → runs as standalone Python script

**Captures:**
- Operation names (e.g., `sigmoid_accurate`, `add`)
- Tensor shapes (e.g., `[1, 1, 12544, 32]`)
- Data types (e.g., `BFLOAT8_B`, `BFLOAT16`)
- Memory layouts (e.g., `HEIGHT_SHARDED`, `INTERLEAVED`)
- Exact shard specifications (grid, shard_shape, orientation)
- Machine information (board type and device series, e.g., `Wormhole n300`, `Blackhole tt-galaxy-bh`)
- Mesh device shape for multi-chip configurations (e.g., `[4, 8]` for 32-chip Galaxy)
- Tensor placements for distributed operations

**Output:**
- Updates `model_tracer/traced_operations/ttnn_operations_master.json`
- Shows summary of unique configurations added

**Note:** After tracing, if the summary shows operations that don't have corresponding `*_model_traced.py` test files in `tests/sweep_framework/sweeps/model_traced/`, you'll need to create test files for those operations to use the traced configurations in sweep tests.

**Note:** Some operations may require custom parameter extraction code in `model_tracer/operation_parameter_extractors.py` if the default extraction doesn't properly parse operation-specific parameters (e.g., `output_dtype` for `typecast`, `dims` for `permute`). Check existing extractors in that file for examples.


### 2. View Configurations

```bash
# View all configs for an operation
python model_tracer/analyze_operations.py sigmoid_accurate
```

**Example output:**
```
📊 Operation: ttnn::sigmoid_accurate
📊 Configurations: 30

📋 Configuration 1:
  arg0: Tensor(shape=[1, 1, 12544, 32], dtype=BFLOAT8_B,
               memory=L1_HEIGHT_SHARDED,
               shard(shard_shape=[224, 32], grid=[(0,0)→(7,6)], orientation=ROW_MAJOR))
```

### 3. Run Sweep Tests

**Run individual operation:**
```bash
# Generate test vectors
python3 tests/sweep_framework/sweeps_parameter_generator.py \
  --module-name model_traced.pad_model_traced

# Run model_traced suite
python3 tests/sweep_framework/sweeps_runner.py \
  --module-name model_traced.pad_model_traced \
  --suite model_traced \
  --vector-source vectors_export \
  --result-dest results_export
```

---

## File Structure

```
tt-metal/
├── model_tracer/
│   ├── generic_ops_tracer.py          # Main tracing script
│   ├── analyze_operations.py          # Query tool
│   └── traced_operations/
│       └── ttnn_operations_master.json # Master config storage
└── tests/sweep_framework/
    ├── master_config_loader.py        # Config loader & utilities
    ├── sweeps_parameter_generator.py  # Generate test vectors
    ├── sweeps_runner.py               # Run individual sweep tests
    └── sweeps/
        └── model_traced/              # Model-traced sweep tests
            └── *_model_traced.py      # Individual operation tests
```

---

## Complete Example

See any `*_model_traced.py` file in `tests/sweep_framework/sweeps/model_traced/` for a full working example, such as:
- `tests/sweep_framework/sweeps/model_traced/add_model_traced.py`
- `tests/sweep_framework/sweeps/model_traced/reshape_model_traced.py`
- `tests/sweep_framework/sweeps/model_traced/pad_model_traced.py`

---

## Workflow

```
1. Trace Model → 2. Store in Master JSON → 3. Use in Sweep Tests
     ⬇️                      ⬇️                        ⬇️
   Real-world          Deduplicated           Model-driven
   configs            configurations         validation
```

**Simple as:**
```python
from tests.sweep_framework.master_config_loader import MasterConfigLoader
loader = MasterConfigLoader()
parameters = {"model_traced": loader.get_suite_parameters("your_op")}
```

---

For complete documentation on running sweep tests, see [Sweep Framework README](tests/sweep_framework/README.md).
