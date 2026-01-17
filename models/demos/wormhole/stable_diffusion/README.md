# Stable_diffusion

## Platforms:
    Wormhole (n150, n300)

## Introduction
Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run
- Run the demo:
```sh
pytest --disable-warnings --input-path="models/demos/wormhole/stable_diffusion/demo/input_data.json" models/demos/wormhole/stable_diffusion/demo/demo.py::test_demo
```

- If you wish to run the demo with a different input:
```sh
pytest --disable-warnings --input-path="<address_to_your_json_file.json>" models/demos/wormhole/stable_diffusion/demo/demo.py::test_demo

```

- If you would like to run an interactive demo which will prompt you for the input:
```sh
pytest models/demos/wormhole/stable_diffusion/demo/demo.py::test_interactive_demo
```

- Our second demo is designed to run poloclub/diffusiondb dataset, run this with:
```sh
pytest --disable-warnings models/demos/wormhole/stable_diffusion/demo/demo.py::test_demo_diffusiondb
```

- If you wish to run for `num_prompts` samples and `num_inference_steps` denoising steps:
```sh
pytest --disable-warnings models/demos/wormhole/stable_diffusion/demo/demo.py::test_demo_diffusiondb[<num_prompts>-<num_inference_steps>]
```

## Details
Note: ttnn stable diffusion utilizes `PNDMScheduler` and requires `num_inference_steps to be greater than or equal to 4`. [Reference](https://arxiv.org/pdf/2202.09778)

### Inputs
Inputs by default are provided from `input_data.json`. If you wish to change the inputs, provide a different path to test_demo.We do not recommend modifying `input_data.json` file.
The entry point to  functional_stable_diffusion model is UNet2DConditionModel in `models/demos/wormhole/stable_diffusion/tt/ttnn_functional_unet_2d_condition_model.py`. The model picks up certain configs and weights from huggingface pretrained model. We have used `CompVis/stable-diffusion-v1-4` version from huggingface as our reference.

### Metrics  Interpretation
`FID Score (Fréchet Inception Distance)` evaluates the quality of generated images by measuring the similarity between their feature distributions and those of real images. A lower FID score indicates better similarity between generated and real images.
For more information, refer [FID Score](https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html).

`CLIP Score` measures the similarity between the generated images and the input prompts. Higher CLIP scores indicate better alignment between the generated images and the provided text prompts.
For more information, refer [CLIP Score](https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_score.html).
