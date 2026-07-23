# Janus-Pro

## Platforms:
    Blackhole (QuietBox / P300, P150), Wormhole (N150)

## Introduction
Janus-Pro is a multimodal understanding model from DeepSeek. It pairs a SigLIP-style vision tower with a LLaMA-style text decoder: images and text go in, text comes out. This directory implements Janus-Pro-7B multimodal understanding on TT-NN (image → text). Image generation is out of scope.

Resource link - [source](https://huggingface.co/deepseek-community/Janus-Pro-7B)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Real weights under `HF_MODEL=deepseek-community/Janus-Pro-7B` (dummy weights produce garbage text)

## How to Run

### TT-Metal
Use the following command(s) to run the model:
- Vision Demo
```
MESH_DEVICE=N150 HF_MODEL=deepseek-community/Janus-Pro-7B pytest models/experimental/janus_pro/demo/vision_demo.py -k "trace and single"
```
- Text only Demo
```
MESH_DEVICE=N150 HF_MODEL=deepseek-community/Janus-Pro-7B pytest models/experimental/janus_pro/demo/text_demo.py -k notrace
```
Notes:
- Use `-k notrace` for text Top-1/Top-5 accuracy (Debug build is fine); use `-k trace` for perf (Release build required).
- Text accuracy needs a one-time HF reference: see [demo/README.md](demo/README.md).
- Demos are manual-run only (not wired into CI e2e yaml). More detail: [demo/README.md](demo/README.md).

## Details
- The entry point to the model is located at:`models/experimental/janus_pro/tt/janus_pro_e2e_model.py`
- Batch Size :1
- Precision: vision tower `bfloat16`, decoder `bfloat8_b` (single fixed config — no separate accuracy/performance precision modes)
- Image pre-processing is performed using PyTorch / the HF processor on host (384×384) before passing to the on-device vision model.

## Performance

Current model performance metrics can be found in [PERF.md](PERF.md)

## Inputs and outputs
### Input:
- Text string, such as a question, a prompt, or a document to be summarized
- Images, normalized to 384 × 384 resolution and encoded to 576 tokens each
- Multimodal understanding only (image → text); text-only decode is also supported via the text demo
### Output:

- Generated text in response to the input, such as an answer to a question, analysis of image content, or OCR of an image

## Example

Example model inputs using the sample image at `models/tt_transformers/demo/sample_prompts/llama_models/dog.jpg` with the vision-demo haiku prompt:

> "Write a haiku for this image."

Run the vision demo (`-k "trace and single"`) to print generated text plus TTFT and decode tok/s/user. Benchmark plan and accuracy/perf methodology: [docs/benchmark/README.md](docs/benchmark/README.md).
