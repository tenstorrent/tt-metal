# Qwen2-7b

## Platforms:
    Wormhole n150

## Introduction
This codebase includes the Qwen2 family of models and currently supports the model ['Qwen/Qwen2-7B'](https://huggingface.co/Qwen/Qwen2-7B)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run
### Download the weights
#### Option 1: Straight from Hugging Face:
- General weights: [Qwen/Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B)
- Finetune instruct weights: [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)

#### Option 2: Use the script provided in `models/demos/qwen/scripts/get_weights.py`
- General weights:
```
python models/demos/qwen/scripts/get_weights.py --weights_path=<FOLDER_TO_SAVE_WEIGHTS>
```
- Finetune instruct weights:
```
python models/demos/qwen/scripts/get_weights.py --weights_path=<FOLDER_TO_SAVE_WEIGHTS> --instruct
```

### Setup model path
- `$QWEN_DIR` sets the path for the Qwen model weights and caches:
```
export QWEN_DIR=<meta_qwen_model_dir>
```

### Run the demo
For a single continuous batch with instruct weights:
```
pytest models/demos/qwen/demo/demo.py -k 'instruct_weights-1_batch'
```

For 2 continuous batches with general weights:
```
pytest models/demos/qwen/demo/demo.py -k 'general_weights-2_batch'
```

## Details
- On the first execution of each model, TTNN will create weight cache files for that model, to speed up future runs.
These cache files only need to be created once for each model and each weight (i.e. new finetuned weights will need to be cached) and will be stored accordingly to the machine you are running the models, for a Wormhole N150 it will be stored as: '$QWEN_DIR/N150'

- The current demo is setup for a single user (batch=1) that loads a prompt file (around 128 tokens), prefills the encoded prompt and then runs decode for 120 iterations. The demo is also parametrized to run for 1 or 3 continuous batch of users, i.e. to simulate multiple users generating text one after another. The input prompts are based on the general or instruct (fine-tuned) weights. The prompts are included in the demo folder `models/demos/qwen/demo`.

- Currently, this model is only supported on Wormhole N150 (single-device). To run this model on a multi-chip device, you can add 'FAKE_DEVICE=N150' to the commands:
```
# Run a single continuous batch with instruct weights
FAKE_DEVICE=N150 pytest models/demos/qwen/demo/demo.py -k 'instruct_weights-1_batch'

# Run 2 continuous batches with general weights
FAKE_DEVICE=N150 pytest models/demos/qwen/demo/demo.py -k 'general_weights-2_batch'
```

### Known Issues
#### 1. Variation in the PCC scores.
PCC (Pearson Correlation Coefficient) is used to measure the inference differences between the TT model and the reference model.

Specifically, we measure the PCC score for each token used during inference. While some tokens achieve high PCC scores, indicating close alignment with the reference implementation, others do not perform as well.

At present, we set the PCC score of 0.79 as the pass threshold in our tests.


#### 2. Sequence length limitation and incomplete RoPE implementation due to hardware constraints.
The Qwen2 model was intended to support sequences up to a maximum length of 131,072 tokens, but the current implementation can only process up to 4,096 tokens.

Since the Qwen2 model is designed to handle very long sequences, a specialized RoPE approach is required for such extreme cases. However, this method only takes effect when the sequence length exceeds a certain threshold. In our implementation, the current limit of 4096 tokens is well below that threshold, making the specialized RoPE implementation unnecessary for now. We plan to incorporate this feature in future development once the hardware constraints are resolved, allowing support for larger sequence lengths.

#### 3. Fast prefill not supported
We currently do prefill via decode. In the future, we will support fast prefill.
