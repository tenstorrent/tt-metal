# Qwen2-7B Demo

Demo showcasing Qwen2-7B running on Wormhole, using ttnn.

## How to Run

### Download the weights

Download the weights from Hugging Face:
- General weights: [Qwen/Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B)
- Finetune instruct weights: [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)

We also include a script to download the weight files inside `models/demos/wormhole/qwen2_7b/scripts/get_qwen2_weights.py`.

```
# Download general weights
python models/demos/wormhole/qwen2_7b/scripts/get_qwen2_weights.py --weights_path=<FOLDER_TO_SAVE_WEIGHTS>

# To download instruct weights add --instruct flag
python models/demos/wormhole/qwen2_7b/scripts/get_qwen2_weights.py --weights_path=<FOLDER_TO_SAVE_WEIGHTS> --instruct
```

### Set up environment

1. Prepare the weight cache directory:

```
# Make a directory for ttnn to cache weights into. This speeds up subsequent runs.
mkdir <weight_cache_dir>
```

2. Set up environment variables:
```
export QWEN2_CKPT_DIR=<weights_dir>
export QWEN2_TOKENIZER_PATH=<path_to_tokenizer_dir>
export QWEN2_CACHE_PATH=<weights_cache_dir>
```

A typical environment will have all the above point to the same folder.

Note that the cached weights folder structure will contain, after being generated, the general and instruct cached weights in separate directories, like so:

```
<weights_cache_dir>
  /tensor_cache_bfp8
  /tensor_cache_instruct_bfp8
  ...
```

3. Cache the weights (first-time setup).
If the cached weights have not yet been created the first execution will take care of generating them. You can run the model test for this step:

```
# Build a full layer model to cache the weights. This will take some time (1 time only).
pytest models/demos/wormhole/qwen2_7b/tests/test_qwen2_model.py::test_qwen2_model_inference
```

### Run the demo

Qwen2-7B runs fast prefill upto sequence lengths of 4096.

For decode-only, the largest context length supported is currently 1024 tokens.

Qwen2-7B is running on a single chip. If you are running on a T3000 please set the following: `export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml`

Note that while running the demo you might see the warning: `Op | WARNING  | TILE layout does not have multicore implementation yet. Falling back to 1 core.` This is expected and can be ignored; the demo will run after the warning.

Batches of 1 to 5 are supported. To set the number of batches for the demo, replace `<number_of_batches>` with a value between 1 and 5 in the CLI command.

```
# Run the demo with a pre-written batch of 8 user prompts:

# Prefill & Decode demo
pytest models/demos/wormhole/qwen2_7b/demo/demo_with_prefill.py::test_qwen2_7B_demo[general_weights-<number_of_batches>_batch]

# Decode-only demo
pytest models/demos/wormhole/qwen2_7b/demo/demo.py::test_qwen2_7B_demo[general_weights-<number_of_batches>_batch]
```

We also provide an input file with 32 user question-prompt for instruct weights (don't forget to update your env flags to the correct instruct weights folder):
```
# Run the demo with a pre-written batch of 8 user question-prompts:

# Prefill & Decode demo
pytest models/demos/wormhole/qwen2_7b/demo/demo_with_prefill.py::test_qwen2_7B_demo[instruct_weights-<number_of_batches>_batch]

# Decode-only demo
pytest models/demos/wormhole/qwen2_7b/demo/demo.py::test_qwen2_7B_demo[instruct_weights-<number_of_batches>_batch]
```

Both input files are provided inside `models/demos/wormhole/qwen2_7b/demo/`.

If you wish you to run the model using a different set of input prompts you can provide a different path to pytest inside the demo code. Keep in mind that for the instruct-mode, the prompts are automatically prefixed and suffixed by `[INST]` and `[/INST]`, respectively, so there's no need to add them to your file.

## Known Issues

### 1. Variation in the PCC scores.

PCC (Pearson Correlation Coefficient) is used to measure the inference differences between the TT model and the reference model.

Specifically, we measure the PCC score for each token used during inference. While some tokens achieve high PCC scores, indicating close alignment with the reference implementation, others do not perform as well.

At present, we set the PCC score of 0.8 as the pass threshold in our tests.

To replicate the experiment, execute the command below.
```
pytest models/demos/wormhole/qwen2_7b/tests/test_qwen2_model.py::test_qwen2_model_inference
```

The PCC scores for the first 10 tokens are as follows.

| Token ID | PCC |
|----------|-----|
| 0 | 0.9846506775387012 |
| 1 | 0.8799932753123203 |
| 2 | 0.9878372520955442 |
| 3 | 0.9849852922234221 |
| 4 | 0.9038908947493449 |
| 5 | 0.8499090047667848 |
| 6 | 0.9512573486160867 |
| 7 | 0.9490127247677106 |
| 8 | 0.9695236893895862 |
| 9 | 0.9545147233692451 |

### 2. Sequence length limitation and incomplete RoPE implementation due to hardware constraints.

The Qwen2 model was intended to support sequences up to a maximum length of 131,072 tokens, but the current implementation can only process up to 4,096 tokens.

Since the Qwen2 model is designed to handle very long sequences, a specialized RoPE approach is required for such extreme cases. However, this method only takes effect when the sequence length exceeds a certain threshold. In our implementation, the current limit of 4096 tokens is well below that threshold, making the specialized RoPE implementation unnecessary for now. We plan to incorporate this feature in future development once the hardware constraints are resolved, allowing support for larger sequence lengths.

### 3. Further analysis of the PPL results.

We have conducted a PPL (perplexity) evaluation experiment to assess the performance of our model. However, the results of this experiment require further analysis, and their full significance is not yet clear.

|     | Reference Model | TT Model |
|-----|-----------------|----------|
| PPL | 19.3004 | 26.2477 |

To replicate the results, run the commands:

```
# PPL for referencee model.
pytest models/demos/wormhole/qwen2_7b/tests/test_qwen2_perplexity.py::test_qwen2_reference_perplexity

# PPL for TT model.
pytest models/demos/wormhole/qwen2_7b/tests/test_qwen2_perplexity.py::test_qwen2_perplexity
```
