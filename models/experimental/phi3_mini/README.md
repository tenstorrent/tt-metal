# Phi 3 Mini Model

This codebase includes the bring up of Phi 3 Mini model on Tenstorrent hardware.

The current version supports the following Phi 3 Mini models:
- Phi-3-mini-128k-instruct

All the above qwen models are compatible and tested on the following Tenstorrent hardware:
- N150 (1-chip)
- N300 (2-chip)

## Dependencies
Dependecies for TT-Transformers are needed to run this model. Install them from:

```
pip install -r models/tt_transformers/requirements.txt
```

## How to Run

### 1. Set up environment variables

- `HF_MODEL` determines the Hugging Face model to run

```
export HF_MODEL=microsoft/Phi-3-mini-128k-instruct
```

- `WH_ARCH_YAML` environment variable sets the dispatch over ethernet cores, this is optional for N150

```
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

- `MAX_PREFILL_CHUNK_SIZE` - this determines how many thousands of tokens are prefilled in one go. For optimal performance pick 128. Depending on the model dimensions and hardware you're running on, there may not be enough L1 to prefill 128K tokens at once, in which case you can reduce this in powers of 2 down to 4.

For N300, we support maximum prefill size of 128k
```
export MAX_PREFILL_CHUNK_SIZE=128
```

For N150, we support maximum perfill size of 1k
```
export MAX_PREFILL_CHUNK_SIZE=1
```

- `TT_CACHE_PATH` is optional. It sets the path for ttnn's weight cache files. See below for more details.

- `MESH_DEVICE` is optional. It allows you to use fewer devices than are available. See below for more details.

### 2. Run the demo

The `simple_text_demo.py` script includes the following main modes of operation and is parametrized to support other configurations.

- `batch-1`: Runs a small prompt (128 tokens) for a single user

```
# Examples of how to run the demo:

# Batch-1
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"
```

The above examples are run in `ModelOptimizations.performance` mode. You can override this by setting the `optimizations` argument in the demo to run in `ModelOptimizations.accuracy` mode.

The expected performance results can be found in `PERF.md`

## Implementation details over TT-Transformer

- The presented implementation of RoPE uses a dynamic scaling factor.
- It switches the scaling when sequence length > 4k to support till 128k.

## Known Issues

1. N300 devices may hit DRAM limit with 64k context length.
2. N150 devices may hit DRAM limit with 32k context length.


### Max Prefill Chunk Sizes (text-only)
|         Model Name        |      N150     |      N300     |
|---------------------------|---------------|---------------|
| Phi-3-mini-128k-instruct  |   1k tokens   |  128k tokens  |
