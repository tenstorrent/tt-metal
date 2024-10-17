from pathlib import Path
from typing import Optional
from loguru import logger

from PIL import Image as PIL_Image
from termcolor import cprint

import importlib

llama_reference_generation = importlib.import_module(
    "models.demos.t3000.llama2_70b.reference.llama-models.models.llama3.reference_impl.generation"
)

# Must import from reference for formatter to understand type of ImageMedia
datatypes = importlib.import_module("models.demos.t3000.llama2_70b.reference.llama-models.models.llama3.api.datatypes")
ImageMedia = datatypes.ImageMedia

# THIS_DIR = Path(__file__).parent.resolve()
# TODO: Generalize not to cglagovich home :)
THIS_DIR = Path("/home/cglagovich/tt-metal/models/demos/t3000/llama2_70b/reference/llama-models/models/scripts/")

import torch
import pytest
import os
import ttnn


def create_multimodal_model(model_args, mesh_device, dtype=ttnn.bfloat16):
    from models.demos.llama3.tt.multimodal.llama_vision_model import CrossAttentionTransformer
    from models.demos.llama3.tt.model_config import TtModelArgs

    tt_model_args = TtModelArgs(mesh_device)
    checkpoint = torch.load(tt_model_args.consolidated_weights_path, map_location="cpu", weights_only=True)
    model = CrossAttentionTransformer(
        model_args,
        mesh_device,
        checkpoint,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=tt_model_args,
    )
    model.setup_cache(model_args.max_batch_size, torch.float32)
    return model


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "target",
    ("tt", "cpu"),
)
def test_llama_multimodal_demo_text(
    mesh_device,
    target,
    temperature: float = 0,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    model_parallel_size: Optional[int] = None,
):
    mesh_device.enable_program_cache()
    mesh_device.enable_async(True)
    ckpt_dir = os.environ["LLAMA_DIR"]
    tokenizer_path = str(Path(ckpt_dir) / "tokenizer.model")

    logger.info(f"Creating reference model from checkpoint in '{ckpt_dir}'")
    generator = llama_reference_generation.Llama.build(
        ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size,
    )

    if target == "tt":
        logger.info(f"Creating TT model on {len(mesh_device.get_devices())} devices")
        model = create_multimodal_model(generator.args, mesh_device)
        generator.model = model

    with open(THIS_DIR / "resources/dog.jpg", "rb") as f:
        img = PIL_Image.open(f).convert("RGB")

    with open(THIS_DIR / "resources/pasta.jpeg", "rb") as f:
        img2 = PIL_Image.open(f).convert("RGB")

    with open(THIS_DIR / "resources/ocr_image.jpeg", "rb") as f:
        ocr_image = PIL_Image.open(f).convert("RGB")
    # with open(THIS_DIR / "resources/clutter.jpeg", "rb") as f:
    #     clutter = PIL_Image.open(f).convert("RGB")

    interleaved_contents = [
        # text only
        # "The color of the sky is blue but sometimes it can also be",
        # image understanding
        # [
        #     ImageMedia(image=img),
        #     "If I had to write a haiku for this one",
        # ],
        [ImageMedia(image=ocr_image), "The full text in this image is as follows"],
        # [
        #     ImageMedia(image=clutter),
        #     "The count of vases, books, and miscellaneous items in this image is",
        # ]
    ]

    print(f"Running text completion on {target}")
    for content in interleaved_contents:
        result = generator.text_completion(
            content,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        cprint(f"{content}", end="")
        cprint(f"{result.generation}", color="yellow")
        print("\n==================================\n")
