# vLLM Integration for BGE-Large-EN-v1.5

This document describes how to integrate the BGE-Large-EN-v1.5 embedding model with vLLM for OpenAI Embedding API compatibility.

## Overview

The vLLM integration enables serving the BGE embedding model through vLLM's OpenAI-compatible API server, eliminating the need to build a custom HTTP frontend. This provides:

- **OpenAI Embedding API Compatibility**: Standard `/v1/embeddings` endpoint
- **Continuous Batching**: Efficient handling of multiple concurrent requests
- **Production-Ready Serving**: Built on vLLM's proven serving infrastructure

## Architecture

The integration consists of:

1. **`generator_vllm.py`**: vLLM-compatible wrapper class (`BGEForEmbedding`) that implements the required interface
2. **Model Registration**: Registration with vLLM's `ModelRegistry` to make the model available
3. **OpenAI API Server**: vLLM's built-in server that exposes the embedding endpoint

## Usage

### 1. Direct Usage (Without vLLM Server)

You can use the vLLM-compatible interface directly:

```python
import ttnn
from models.demos.wormhole.bge_large_en.demo.generator_vllm import BGEForEmbedding
import transformers

# Initialize device
device = ttnn.open_device(device_id=0)

# Load config
config = transformers.BertConfig.from_pretrained("BAAI/bge-large-en-v1.5")

# Initialize model
bge_model = BGEForEmbedding.initialize_vllm_model(
    hf_config=config,
    mesh_device=device,
    max_batch_size=8,
    max_seq_len=384,
)

# Generate embeddings
input_ids = torch.tensor([[101, 7592, 2088, ...]])  # Tokenized input
embeddings = bge_model.forward(input_ids=input_ids)
```

### 2. vLLM Server Integration

To serve the model via vLLM's OpenAI-compatible API:

#### Step 1: Add to Supported Models List

**CRITICAL**: Add `"BAAI/bge-large-en-v1.5"` to the supported models list in `vllm/platforms/tt.py`:

In the `check_tt_model_supported` function, add to the `supported_models` list:

```python
def check_tt_model_supported(model):
    supported_models = [
        "meta-llama/Llama-3.1-70B",
        # ... other models ...
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "deepseek-ai/DeepSeek-R1-0528",
        "BAAI/bge-large-en-v1.5",  # Add this line
    ]
    assert model in supported_models, (
        f"{model} is not in list of supported TT models")
```

#### Step 2: Register the Model

In `vllm/platforms/tt.py`, in the `register_tt_models()` function, add:

```python
# BGE Embedding Model
ModelRegistry.register_model(
    "BGEForEmbedding",
    "models.demos.wormhole.bge_large_en.demo.generator_vllm:BGEForEmbedding",
)
```

Or alternatively, you can call the `register_model()` function from the generator_vllm module:

```python
from models.demos.wormhole.bge_large_en.demo.generator_vllm import register_model

# In register_tt_models() function:
register_model()
```

#### Step 3: Start vLLM Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model BAAI/bge-large-en-v1.5 \
    --tensor-parallel-size 1 \
    --max-model-len 384
```

#### Step 4: Use OpenAI-Compatible API

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",  # vLLM doesn't require real API key
)

response = client.embeddings.create(
    model="BAAI/bge-large-en-v1.5",
    input="Your text here",
)

embedding = response.data[0].embedding
```

Or using `curl`:

```bash
curl http://localhost:8000/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
        "model": "BAAI/bge-large-en-v1.5",
        "input": "Your text here"
    }'
```

## Implementation Details

### BGEForEmbedding Class

The `BGEForEmbedding` class implements the vLLM interface:

- **`initialize_vllm_model()`**: Class method called by vLLM's model loader
- **`forward()`**: Main inference method that takes tokens and returns embeddings
- **`get_embedding_dim()`**: Returns embedding dimension (1024 for BGE-large)
- **`get_max_seq_len()`**: Returns maximum sequence length (384)
- **`get_max_batch_size()`**: Returns maximum batch size

### Key Features

1. **Lazy Initialization**: The runner is initialized on first forward pass to optimize memory usage
2. **Trace Capture**: Uses TTNN trace capture for optimized inference
3. **Batch Handling**: Supports variable batch sizes up to `max_batch_size`
4. **Sequence Length**: Supports sequences up to `max_seq_len` (384)

## Testing

Run the vLLM integration test:

```bash
pytest models/demos/wormhole/bge_large_en/demo/demo.py::test_bge_vllm_demo -v
```

## Performance

The vLLM integration maintains the same performance characteristics as the direct runner:

- **Throughput**: ~175 sentences/sec (batch_size=8, seq_len=384)
- **PCC**: ~93% compared to PyTorch reference
- **Latency**: Optimized via TTNN trace capture

## Limitations

1. **Fixed Sequence Length**: Currently supports up to 384 tokens (BGE-large default)
2. **Batch Size**: Limited by device memory and `max_batch_size` parameter
3. **Single Device**: Current implementation supports single device (can be extended for multi-device)

## Future Enhancements

- [ ] Support for variable sequence lengths
- [ ] Multi-device data parallelism
- [ ] Dynamic batch size handling
- [ ] Integration with vLLM's continuous batching optimizations

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Integration Guide](../../../../tech_reports/LLMs/vLLM_integration.md)
- [OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings)
