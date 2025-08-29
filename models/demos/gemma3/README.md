# Gemma3

Note: This model is currently functional with vLLM integration on branch pending merging PRs in tt-metal and vLLM. The branches are:
- [tt-metal:handrews/gemma3-27b-demo-PR](https://github.com/tenstorrent/tt-metal/tree/handrews/gemma3-27b-demo-PR)
- [vllm:rdraskic/gemma3-vllm-support](https://github.com/tenstorrent/vllm/tree/rdraskic/gemma3-vllm-support)

## Platforms:
    Wormhole (n150, n300, t3k)

## Introduction
Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models. Gemma 3 models are multimodal, handling text and image input and generating text output, with open weights for both pre-trained variants and instruction-tuned variants. Gemma 3 has a large, 128K context window, multilingual support in over 140 languages, and is available in more sizes than previous versions. Gemma 3 models are well-suited for a variety of text generation and image understanding tasks, including question answering, summarization, and reasoning. Their relatively small size makes it possible to deploy them in environments with limited resources such as laptops, desktops or your own cloud infrastructure, democratizing access to state of the art AI models and helping foster innovation for everyone.

Resource link - [source](https://huggingface.co/google/gemma-3-27b-it)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`

## How to Run

### TT-Metal
Use the following command(s) to run the model:
- Vision Demo
```
HF_MODEL=google/gemma-3-27b-it pytest models/demos/gemma3/demo/vision_demo.py -k “batch1-trace”
```
- Text only Demo
```
HF_MODEL=google/gemma-3-27b-it pytest models/demos/gemma3/demo/text_demo.py -k “performance and batch1”
```

### vLLM offline
```
MESH_DEVICE=T3K python examples/offline_inference_tt.py --model "google/gemma-3-27b-it" --multi_modal --max_seqs_in_batch 32 --override_tt_config "{\"l1_small_size\": 768, \"fabric_config\": \"FABRIC_1D\"}"
```

## Details
- The entry point to the model is located at:`models/demos/gemma3/tt/gemma_e2e_model.py`
- Batch Size :1-32
- The image pre-processing is performed using PyTorch on host before passing to the on-device vision model.

## Performance

Current model performance metrics can be found in [PERF.md](PERF.md)

## Inputs and outputs
### Input:
- Text string, such as a question, a prompt, or a document to be summarized
- Images, normalized to 896 x 896 resolution and encoded to 256 tokens each
- Total input context of 128K tokens for the 4B, 12B, and 27B sizes, and 32K tokens for the 1B size
### Output:

- Generated text in response to the input, such as an answer to a question, analysis of image content, or a summary of a document
- Total output context of 8192 tokens
