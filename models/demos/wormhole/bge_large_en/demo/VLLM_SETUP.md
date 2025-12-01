# vLLM Setup Instructions for BGE-Large-EN-v1.5

This document provides step-by-step instructions for setting up BGE-Large-EN-v1.5 in the vLLM repository.

## Prerequisites

- Access to the vLLM repository at `/home/ttuser/ashai/main/vllm`
- TT-Metal repository at `/home/ttuser/ashai/main/tt-metal` (this repo)
- Python environment with vLLM dependencies installed

## Required Changes in vLLM Repository

### 1. Add Model to Supported Models List

**File**: `vllm/platforms/tt.py`

**Location**: In the `check_tt_model_supported()` function

**Change**: Add `"BAAI/bge-large-en-v1.5"` to the `supported_models` list:

```python
def check_tt_model_supported(model):
    supported_models = [
        "meta-llama/Llama-3.1-70B",
        # ... existing models ...
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "deepseek-ai/DeepSeek-R1-0528",
        "BAAI/bge-large-en-v1.5",  # ADD THIS LINE
    ]
    assert model in supported_models, (
        f"{model} is not in list of supported TT models")
```

### 2. Register the Model

**File**: `vllm/platforms/tt.py`

**Location**: In the `register_tt_models()` function (usually near the end of the function)

**Change**: Add model registration:

```python
def register_tt_models():
    # ... existing registrations ...

    # BGE Embedding Model
    # Note: vLLM constructs model class names as "TT" + architecture name
    # For BertModel architecture, it expects "TTBertModel"
    ModelRegistry.register_model(
        "TTBertModel",
        "models.demos.wormhole.bge_large_en.demo.generator_vllm:BGEForEmbedding",
    )
```

**Note**: Make sure `TT_METAL_HOME` environment variable points to `/home/ttuser/ashai/main/tt-metal` so vLLM can import the model.

## Verification

After making these changes, verify the setup:

```bash
cd /home/ttuser/ashai/main/vllm

# Check that the model is in the supported list
python -c "from vllm.platforms.tt import check_tt_model_supported; check_tt_model_supported('BAAI/bge-large-en-v1.5'); print('✓ Model is supported')"

# Try starting the server (will fail if device not available, but should pass validation)
python -m vllm.entrypoints.openai.api_server \
    --model BAAI/bge-large-en-v1.5 \
    --tensor-parallel-size 1 \
    --max-model-len 384 \
    --host 0.0.0.0 \
    --port 8000
```

## Troubleshooting

### Error: "BAAI/bge-large-en-v1.5 is not in list of supported TT models"

- **Solution**: Make sure you added the model to the `supported_models` list in `check_tt_model_supported()` function

### Error: "ModuleNotFoundError: No module named 'models.demos.wormhole.bge_large_en.demo.generator_vllm'"

- **Solution**: Ensure `TT_METAL_HOME` is set correctly:
  ```bash
  export TT_METAL_HOME=/home/ttuser/ashai/main/tt-metal
  export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
  ```

### Error: "ModelRegistry: Model not found"

- **Solution**: Make sure you added the `ModelRegistry.register_model()` call in `register_tt_models()` function

## Testing

Once the server starts successfully, test the embedding API:

```bash
curl http://localhost:8000/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
        "model": "BAAI/bge-large-en-v1.5",
        "input": "This is a test sentence."
    }'
```

Expected response:
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, -0.456, ...],
      "index": 0
    }
  ],
  "model": "BAAI/bge-large-en-v1.5",
  "usage": {
    "prompt_tokens": 6,
    "total_tokens": 6
  }
}
```

## Next Steps

After successful setup:
1. Test with multiple concurrent requests
2. Monitor performance metrics
3. Adjust `max_batch_size` and `max_seq_len` as needed
4. Consider adding to CI/CD pipeline
