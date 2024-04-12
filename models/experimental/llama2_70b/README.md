# Llama2-70B Demo

## How to Run
```bash
# Download the Llama2-70B weights locally

# Run our weight repacking script. This concatenates the sharded checkpoints and makes it easier for us to load.
python scripts/repack_weights.py <path_to_checkpoint_dir> <repacked_output_dir>

# Make a directory for us to cache weights into. This speeds up subsequent runs.
mkdir <weight_cache_dir>
# Set up some environment variables for the demo
export LLAMA_CKPT_DIR=<repacked_output_dir>
export LLAMA_TOKENIZER_PATH=<path_to_checkpoint_dir>
export LLAMA_CACHE_PATH=<weight_cache_dir>

# First time only: run a model test to cache the weights. This will take some time.
pytest -svv models/demos/llama2_70b/tests/test_llama_model.py::test_LlamaModel_inference[BFLOAT16-DRAM-decode-8chip-T3000-0.9-80]

# Run the demo
pytest -svv models/demos/llama2_70b/demo/demo.py::test_LlamaModel_demo[BFLOAT16-DRAM-sampling-decode-tt-70b-T3000-80]
```

## Details
This version of the demo supports batch size 32. We use `./demo/data/multi_prompt.json` as our input. The model is the pretrained model.

The demo runs on an 8-chip T3000 machine using tensor parallelism. We implement decode to prefill, meaning that the prompts are consumed token-by-token to produce KV caches. Once a user's prompt has been prefilled, tokens are generated in decode mode.

Your host machine must have at least 512 GB of memory to run this demo.
