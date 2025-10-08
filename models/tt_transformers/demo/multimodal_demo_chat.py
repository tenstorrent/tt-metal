# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Optional

import pytest
from loguru import logger
from PIL import Image as PIL_Image
from pkg_resources import resource_filename
from transformers import AutoProcessor

import ttnn

IMG_PATH = Path(resource_filename("llama_models", "scripts/resources/"))

from models.common.llama_models import GeneratorChat
from models.tt_transformers.demo.simple_vision_demo import create_multimodal_model
from models.tt_transformers.tt.generator import Generator


@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "target",
    ("tt", "cpu"),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_multimodal_demo_chat(
    mesh_device,
    target,
    temperature: float = 0.5,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = 200,
    model_parallel_size: Optional[int] = None,
):
    ckpt_dir = os.environ["HF_MODEL"]

    logger.info(f"Creating reference model from checkpoint in '{ckpt_dir}'")
    if target == "cpu":
        generator = GeneratorChat(ckpt_dir, max_batch_size=max_batch_size)
    else:
        logger.info(f"Creating TT model on {mesh_device.get_num_devices()} devices")

        model_args, model, _ = create_multimodal_model(
            mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len
        )
        processor = AutoProcessor.from_pretrained(ckpt_dir)
        tokenizer = processor.tokenizer
        generator = Generator([model], [model_args], mesh_device, processor=processor, tokenizer=tokenizer)

    # image understanding
    dialogs = []
    with open(IMG_PATH / "dog.jpg", "rb") as f:
        img = PIL_Image.open(f).convert("RGB")

    dialogs = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Describe this image in two sentences"},
                ],
            }
        ],
    ]

    print(f"Running text completion on {target}")
    for dialog in dialogs:
        result = generator.chat_completion(
            dialog,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")

        out_message = result
        print(f"> {out_message.role.capitalize()}: {out_message.content}")
        # TODO: add tool_calls functionality
        # for t in out_message.tool_calls:
        #     print(f"  Tool call: {t.tool_name} ({t.arguments})")
        print("\n==================================\n")
