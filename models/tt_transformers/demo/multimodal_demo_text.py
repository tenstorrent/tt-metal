# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Optional

import llama_models.llama3.reference_impl.generation as llama_reference_generation
import pytest
from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.datatypes import ImageMedia
from llama_models.llama3.api.tokenizer import Tokenizer
from loguru import logger
from PIL import Image as PIL_Image
from pkg_resources import resource_filename
from termcolor import cprint

import ttnn

IMG_PATH = Path(resource_filename("llama_models", "scripts/resources/"))

from models.tt_transformers.demo.simple_vision_demo import create_multimodal_model
from models.tt_transformers.tt.generator import Generator


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "target",
    ("tt", "cpu"),
)
@pytest.mark.parametrize(
    "warmup_iters",
    (0, 1),
)
def test_multimodal_demo_text(
    mesh_device,
    target,
    warmup_iters,
    temperature: float = 0.5,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = 200,
    model_parallel_size: Optional[int] = None,
):
    ckpt_dir = os.environ["LLAMA_DIR"]
    tokenizer_path = str(Path(ckpt_dir) / "tokenizer.model")

    logger.info(f"Creating reference model from checkpoint in '{ckpt_dir}'")
    if target == "cpu":
        generator = llama_reference_generation.Llama.build(
            ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            model_parallel_size=model_parallel_size,
        )
    else:
        logger.info(f"Creating TT model on {mesh_device.get_num_devices()} devices")

        model_args, model, _ = create_multimodal_model(
            mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        formatter = ChatFormat(tokenizer)
        generator = Generator([model], [model_args], mesh_device, tokenizer=tokenizer, formatter=formatter)

    with open(IMG_PATH / "dog.jpg", "rb") as f:
        img = PIL_Image.open(f).convert("RGB")

    with open(IMG_PATH / "pasta.jpeg", "rb") as f:
        img2 = PIL_Image.open(f).convert("RGB")

    with open(IMG_PATH / "ocr_image.jpeg", "rb") as f:
        ocr_image = PIL_Image.open(f).convert("RGB")

    with open(IMG_PATH / "clutter.jpeg", "rb") as f:
        clutter = PIL_Image.open(f).convert("RGB")

    interleaved_contents = [
        # image understanding
        [ImageMedia(image=img), "If I had to write a haiku for this one"],
        [ImageMedia(image=img2), "Couting the number of individual spaghetti strands in this image"],
        [ImageMedia(image=ocr_image), "The full text in this image is as follows"],
        [ImageMedia(image=clutter), "The count of vases, books, and miscellaneous items in this image is"],
    ]

    print(f"Running text completion on {target}")
    for _ in range(warmup_iters + 1):
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
