# Llama3 Models

This codebase includes the Llama3 family of models.

The current version supports the following Llama3 models:
- Llama3.1-8B
- Llama3.2-1B
- Llama3.2-3B

All the above llama models are compatible and tested on the following Tenstorrent hardware:
- N150 (1-chip)
- N300 (2-chips)
- T3000 (8-chips)

## How to Run

### Download the weights

Download the weights [directly from Meta](https://llama.meta.com/llama-downloads/), this will mean accepting their license terms.

The downloaded directories include weight files (e.g. `consolidated.00.pth`), the tokenizer `tokenizer.model` and configuration file `params.json`.

### Setup TT environment

1. Set up environment variables:
```
export LLAMA_DIR=<meta_llama_model_dir>
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

- `$LLAMA_DIR` sets the path for the Llama3 model weights and caches.

- `$WH_ARCH_YAML` sets the dispatch over ethernet cores. This is optional for N150 and required for N300 and T3000, enabling a full core grid utilization (8x8), allowing for maximum performance of LLama3 models.

On the first execution of each model, TTNN will create weight cache files for that model, to speed up future runs.
These cache files only need to be created once for each model and each weight (i.e. new finetuned weights will need to be cached) and will be stored accordingly to the machine you are running the models:
```
$LLAMA_DIR/N150  # For N150
$LLAMA_DIR/N300  # For N300
$LLAMA_DIR/T3K   # For T3000
```


### Run the demo

The current demo is setup for a single user (batch=1) that loads a prompt file (around 128 tokens), prefills the encoded prompt and then runs decode for 120 iterations.

The demo is also parametrized to run for 1 or 3 continuous batch of users, i.e. to simulate multiple users generating text one after another.

The input prompts are based on the general or instruct (fine-tuned) weights. The prompts are included in the demo folder `models/demos/llama3/demo`.

When running the demo, do not forget to setup the `$LLAMA_DIR` environment variable to the corresponding Llama3 model weights.

```
# Examples of how to run the demo for any supported Llama3 models

# Run a single continuous batch with instruct weights
pytest models/demos/llama3/demo/demo.py -k 'instruct and 1_batch'

# Run 3 continuous batches with general weights
pytest models/demos/llama3/demo/demo.py -k 'general and 3_batch'
```
