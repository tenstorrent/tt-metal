# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Optional
from loguru import logger

from PIL import Image as PIL_Image
from termcolor import cprint

from models.demos.llama3.demo.multimodal_demo_text import create_multimodal_model
import llama_models.llama3.reference_impl.generation as llama_reference_generation

from llama_models.llama3.api.datatypes import ImageMedia, UserMessage

from pkg_resources import resource_filename

IMG_PATH = Path(resource_filename("llama_models", "scripts/resources/"))

import torch
import pytest
import os
import ttnn


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
@pytest.mark.parametrize(
    "warmup_iters",
    (0, 1),
)
def test_llama_multimodal_demo_chat(
    mesh_device,
    target,
    warmup_iters,
    temperature: float = 0.5,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = 200,
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

    # image understanding
    dialogs = []
    with open(IMG_PATH / "dog.jpg", "rb") as f:
        img = PIL_Image.open(f).convert("RGB")

    dialogs = [
        [
            UserMessage(
                content=[
                    ImageMedia(image=img),
                    "Describe this image in two sentences",
                ],
            )
        ],
    ]
    # text only
    dialogs += [
        [UserMessage(content="what is the recipe of mayonnaise in two sentences?")],
    ]

    print(f"Running text completion on {target}")
    for _ in range(warmup_iters + 1):
        for dialog in dialogs:
            result = generator.chat_completion(
                dialog,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            for msg in dialog:
                print(f"{msg.role.capitalize()}: {msg.content}\n")

            out_message = result.generation
            print(f"> {out_message.role.capitalize()}: {out_message.content}")
            for t in out_message.tool_calls:
                print(f"  Tool call: {t.tool_name} ({t.arguments})")
            print("\n==================================\n")
