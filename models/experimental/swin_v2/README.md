# Swin_V2

## Platforms:
Wormhole N150, N300

## Introduction
Swin Transformer v2 builds upon the original Swin Transformer to tackle key challenges in large-scale vision models, including training stability, handling high-resolution inputs, and the scarcity of labeled data. It introduces advanced techniques such as residual-post-norm with cosine attention, log-spaced continuous position bias, and SimMIM-based self-supervised pretraining. The core idea of the Swin Transformer is to integrate essential visual priors—like hierarchy, locality, and translation invariance—into the standard Transformer encoder. This combination leverages the strong modeling capabilities of Transformers while making the architecture more effective and adaptable for a wide range of vision tasks.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`
- Login to huggingface with: `huggingface-cli login` or by setting the token with the command `export HF_TOKEN=<token>`
   - To obtain a huggingface token visit: https://huggingface.co/docs/hub/security-tokens

## How to Run
- Use the following command to run the `swin_v2` model:
  ```
  pytest models/experimental/swin_v2/tests/pcc/test_ttnn_swin_v2_s.py
  ```

### Model performant running with Trace+2CQ
- For `512x512` resolution, end-2-end perf is `7` FPS

    ```sh
    pytest models/experimental/swin_v2/tests/perf/test_e2e_performant.py
    ```

### Performant Demo on ImageNet:
- Make sure your HuggingFace token is set ([See Prerequisites](#prerequisites) for instructions).
- Use the following command to run the demo using `Imagenet-1K` dataset:

  ```bash
  pytest models/experimental/swin_v2/demo/demo.py
  ```

## Details
- Entry point for the model is `models/experimental/swin_v2/tt/tt_swin_transformer.py`
- Batch Size: `1` (Single Device).
- Support Input Resolution: `(512, 512)` - (Height, Width).
- Dataset used for evaluation - **imagenet-1k**.
