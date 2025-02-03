#! /usr/bin/env python
import json
import os
import time

import click
import numpy as np
import torch
import ttnn
import pytest

from genmo.lib.progress import progress_bar
from genmo.lib.utils import save_video
from genmo.mochi_preview.pipelines import (
    DecoderModelFactory,
    MochiSingleGPUPipeline,
    T5ModelFactory,
    linear_quadratic_schedule,
)
from models.experimental.mochi.factory import TtDiTModelFactory

pipeline = None
model_dir_path = os.getenv("MOCHI_DIR")
lora_path = None
num_gpus = 1
cpu_offload = True


def load_model(mesh_device):
    global num_gpus, pipeline, model_dir_path, lora_path
    if pipeline is None:
        MOCHI_DIR = model_dir_path
        print(f"Launching with {num_gpus} GPUs.")
        pipeline = MochiSingleGPUPipeline(
            text_encoder_factory=T5ModelFactory(),
            dit_factory=TtDiTModelFactory(
                model_path=f"{MOCHI_DIR}/dit.safetensors",
                model_dtype="bf16",
                lora_path=lora_path,
                mesh_device=mesh_device,
            ),
            decoder_factory=DecoderModelFactory(
                model_path=f"{MOCHI_DIR}/decoder.safetensors",
            ),
            cpu_offload=cpu_offload,
            decode_type="tiled_spatial",
            fast_init=not lora_path,
            strict_load=not lora_path,
            decode_args=dict(overlap=8),
        )


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_generate_video(mesh_device, generation_args, use_program_cache):
    """Generate video using the provided mesh device and arguments."""
    mesh_device.enable_async(True)
    load_model(mesh_device)

    # sigma_schedule should be a list of floats of length (num_inference_steps + 1),
    # such that sigma_schedule[0] == 1.0 and sigma_schedule[-1] == 0.0 and monotonically decreasing.
    sigma_schedule = linear_quadratic_schedule(generation_args["num_steps"], 0.025)

    # cfg_schedule should be a list of floats of length num_inference_steps.
    cfg_schedule = [generation_args["cfg_scale"]] * generation_args["num_steps"]

    args = {
        "height": generation_args["height"],
        "width": generation_args["width"],
        "num_frames": generation_args["num_frames"],
        "sigma_schedule": sigma_schedule,
        "cfg_schedule": cfg_schedule,
        "num_inference_steps": generation_args["num_steps"],
        "batch_cfg": False,
        "prompt": generation_args["prompt"],
        "negative_prompt": generation_args["negative_prompt"],
        "seed": generation_args["seed"],
    }

    with progress_bar(type="tqdm"):
        final_frames = pipeline(**args)
        final_frames = final_frames[0]
        assert isinstance(final_frames, np.ndarray)
        assert final_frames.dtype == np.float32

        os.makedirs("outputs", exist_ok=True)
        output_path = os.path.join("outputs", f"output_{int(time.time())}.mp4")

        save_video(final_frames, output_path)
        json_path = os.path.splitext(output_path)[0] + ".json"
        json.dump(args, open(json_path, "w"), indent=4)

        return output_path


from textwrap import dedent

DEFAULT_PROMPT = dedent(
    """
A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl
filled with lemons and sprigs of mint against a peach-colored background.
The hand gently tosses the lemon up and catches it, showcasing its smooth texture.
A beige string bag sits beside the bowl, adding a rustic touch to the scene.
Additional lemons, one halved, are scattered around the base of the bowl.
The even lighting enhances the vibrant colors and creates a fresh,
inviting atmosphere.
"""
)


@click.command()
@click.option("--prompt", default=DEFAULT_PROMPT, help="Prompt for video generation.")
@click.option("--negative_prompt", default="", help="Negative prompt for video generation.")
@click.option("--width", default=848, type=int, help="Width of the video.")
@click.option("--height", default=480, type=int, help="Height of the video.")
@click.option("--num_frames", default=163, type=int, help="Number of frames.")
@click.option("--seed", default=1710977262, type=int, help="Random seed.")
@click.option("--cfg_scale", default=6.0, type=float, help="CFG Scale.")
@click.option("--num_steps", default=64, type=int, help="Number of inference steps.")
@click.option("--model_dir", required=True, help="Path to the model directory.")
@click.option("--lora_path", required=False, help="Path to the lora file.")
@click.option("--cpu_offload", is_flag=True, help="Whether to offload model to CPU")
def main(
    prompt, negative_prompt, width, height, num_frames, seed, cfg_scale, num_steps, model_dir, lora_path, cpu_offload
):
    """CLI interface for video generation."""
    # configure_model(model_dir, lora_path, cpu_offload)

    # Create a dictionary of generation arguments
    generation_args = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "seed": seed,
        "cfg_scale": cfg_scale,
        "num_steps": num_steps,
    }

    # Set the generation args as an environment variable
    os.environ["GENERATION_ARGS"] = json.dumps(generation_args)

    # Run pytest for the generate_video function
    pytest.main(
        [
            __file__,
            "-v",
            "-k",
            "test_generate_video",
            "--capture=no",  # Show output
            "-s",  # Don't capture stdout
        ]
    )


# Add a pytest fixture to provide the generation args
@pytest.fixture
def generation_args():
    """Fixture to provide generation arguments from environment variable."""
    args_json = os.environ.get("GENERATION_ARGS")
    if args_json is None:
        return {}
    return json.loads(args_json)


if __name__ == "__main__":
    main()
