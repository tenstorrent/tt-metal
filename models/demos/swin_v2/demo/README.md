# Swin_V2Add commentMore actions
# Platforms:

Wormhole N150, N300

Note: On N300, make sure to use WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml with the pytest.

Or, make sure to set the following environment variable in the terminal:

```
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```
To obtain the perf reports through profiler, please build with following command:

```
./build_metal.sh -p
```

# Introduction
Swin Transformer v2 builds upon the original Swin Transformer to tackle key challenges in large-scale vision models, including training stability, handling high-resolution inputs, and the scarcity of labeled data. It introduces advanced techniques such as residual-post-norm with cosine attention, log-spaced continuous position bias, and SimMIM-based self-supervised pretraining. The core idea of the Swin Transformer is to integrate essential visual priors—like hierarchy, locality, and translation invariance—into the standard Transformer encoder. This combination leverages the strong modeling capabilities of Transformers while making the architecture more effective and adaptable for a wide range of vision tasks.


# Details
- Entry point for the model is models/experimental/swin_v2/tt/tt_swin_transformer.py
- Batch Size: 1
- Support Input Resolution: 512x512 (Height, Width)

# How to run
To run the inference, make sure to build the project, activate the environment, and set the appropriate environment variables. For more information, refer to the [installation and build guide](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/get_started/get_started.html).

Command to run the inference pipeline with random weights and random tensor:

pytest models/experimental/swin_v2/tests/test_ttnn_swin_v2_s.py



## Model performant running with Trace+2CQ
Use the following command to run the e2e perf:
- end-2-end perf is 6 FPS

-  For overall rutime inference (end-2-end), use the following command to run the demo:

    ```sh
    pytest --disable-warnings models/experimental/swin_v2/tests/test_e2e_performant.py
    ```
