## functional_stable_diffusion Demo
## How to Run

To run the demo, make sure to build the project, activate the environment, and set the appropriate environment variables.
For more information, refer [installation and build guide](https://tenstorrent-metal.github.io/tt-metal/latest/get_started/get_started.html#install-and-build).

Use `pytest --disable-warnings --input-path="models/experimental/functional_stable_diffusion/demo/input_data.json" models/experimental/functional_stable_diffusion/demo/demo.py::test_demo` to run the demo.

If you wish to run the demo with a different input use `pytest --disable-warnings --input-path="<address_to_your_json_file.json>" models/experimental/functional_stable_diffusion/demo/demo.py::test_demo`

Our second demo is designed to run poloclub/diffusiondb dataset, run this with `pytest --disable-warnings models/experimental/functional_stable_diffusion/demo/demo.py::test_demo_diffusiondb`.

If you wish to run for `num_prompts` samples and `num_inference_steps` denoising steps, use `pytest --disable-warnings models/experimental/functional_stable_diffusion/demo/demo.py::test_demo_diffusiondb[<num_prompts>-<num_inference_steps>]`

# Inputs
Inputs by default are provided from `input_data.json`. If you wish you to change the inputs, provide a different path to test_demo.

We do not recommend modifying `input_data.json` file.

# Details
The entry point to  functional_stable_diffusion model is UNet2DConditionModel in `models/experimental/functional_stable_diffusion/tt/ttnn_functional_unet_2d_condition_model.py`. The model picks up certain configs and weights from huggingface pretrained model. We have used `CompVis/stable-diffusion-v1-4` version from huggingface as our reference.

# Metrics  Interpretation
`FID Score (Fréchet Inception Distance)` evaluates the quality of generated images by measuring the similarity between their feature distributions and those of real images. A lower FID score indicates better similarity between generated and real images.
For more information, refer [FID Score](https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html).

`CLIP Score` measures the similarity between the generated images and the input prompts. Higher CLIP scores indicate better alignment between the generated images and the provided text prompts.
For more information, refer [CLIP Score](https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_score.html).
