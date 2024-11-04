# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Optional
from loguru import logger

from PIL import Image as PIL_Image
from termcolor import cprint

import llama_models.llama3.reference_impl.generation as llama_reference_generation
from llama_models.llama3.api.tokenizer import Tokenizer
from llama_models.llama3.api.chat_format import ChatFormat, ModelInput

from llama_models.llama3.api.datatypes import ImageMedia, UserMessage

from pkg_resources import resource_filename

IMG_PATH = Path(resource_filename("llama_models", "scripts/resources/"))

import torch
import pytest
import os
import ttnn


class LlamaVision:
    def __init__(self, model, model_args, mesh_device, vllm=False):
        """
        Creating a LlamaVision wrapper requires only a mesh_device and model_args.
        With model_args you have the checkpoint location, can specify max batch size
        and max seqlen, and other model specific parameters.

        LlamaVision is general to text and chat.

        For bringup, make this class general to any backend implementation, as long as it takes torch tensors and returns torch tensors.

        """
        self.model = model
        self.model_args = model_args
        self.mesh_device = mesh_device
        self.vllm = vllm

    def get_prefill_inputs(self, model_input):
        """
        Responsible for taking model_input: ModelInput and returning vision_images, vision_mask, tokens
        """
        images = model_input.vision.images
        mask = model_input.visiom.mask
        tokens = model_input.tokens

        return images, mask, tokens

    def forward_prefill(self, vision_images, vision_mask, tokens, total_len, text_only_inference=False):
        """
        Performs vision encode step then text prefill.
        Returns (xattn_caches, cross_attention_masks, full_text_row_masked_out_mask, logits)
        """
        xattn_caches, cross_attention_masks, full_text_row_masked_out_mask = self.model.compute_vision_tokens_masks(
            batch_images=[vision_images],
            batch_masks=[vision_mask],
            total_len=total_len,
        )

        position_ids = torch.arange(tokens.shape[-1], dtype=torch.long)

        logits = self.model.forward(
            position_ids,
            tokens,
            cross_attention_masks,
            full_text_row_masked_out_mask,
            xattn_caches,
            text_only_inference,
        )

        return xattn_caches, cross_attention_masks, full_text_row_masked_out_mask, logits

    def forward_decode(
        self,
        position_ids,
        tokens,
        cross_attention_masks,
        full_text_row_masked_out_mask,
        xattn_caches,
        text_only_inference=False,
    ):
        """
        Performs text decode step.
        Returns logits
        """
        pass


def get_sampler(temperature, top_p, tokenizer):
    def sample(logits):
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = llama_reference_generation.sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        next_token = next_token.reshape(-1)
        token = next_token[0].item()
        text = tokenizer.decode(next_token.tolist())
        return token, text

    return sample


def create_multimodal_model(mesh_device, dtype=ttnn.bfloat16):
    from models.demos.llama3.tt.multimodal.llama_vision_model import CrossAttentionTransformer
    from models.demos.llama3.tt.model_config import TtModelArgs

    tt_model_args = TtModelArgs(mesh_device)
    checkpoint = torch.load(tt_model_args.consolidated_weights_path, map_location="cpu", weights_only=True)
    model = CrossAttentionTransformer(
        mesh_device,
        checkpoint,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=tt_model_args,
    )
    model.setup_cache(tt_model_args.max_batch_size, torch.float32)  # TODO: is a no-op
    return tt_model_args, model


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
def test_llama_multimodal_demo_text(
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
    """
    Simple multimodal demo with limited dependence on reference code.
    """
    ckpt_dir = os.environ["LLAMA_DIR"]
    tokenizer_path = str(Path(ckpt_dir) / "tokenizer.model")

    if target == "cpu":
        generator = llama_reference_generation.Llama.build(
            ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            model_parallel_size=model_parallel_size,
        )
        model_args = generator.args
        model = LlamaVision(generator.model, model_args, None)
        tokenizer = generator.tokenizer
        formatter = generator.formatter
    else:
        mesh_device.enable_program_cache()
        mesh_device.enable_async(True)
        model_args, model = create_multimodal_model(mesh_device)
        model = LlamaVision(model, model_args, mesh_device)
        tokenizer = Tokenizer(model_path=tokenizer_path)
        formatter = ChatFormat(tokenizer)

    with open(IMG_PATH / "dog.jpg", "rb") as f:
        img = PIL_Image.open(f).convert("RGB")

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

    sampler = get_sampler(temperature, top_p, tokenizer)

    print(f"Running text completion on {target}")
    for _ in range(warmup_iters + 1):
        for dialog in dialogs:
            # result = generator.chat_completion(
            #     dialog,
            #     max_gen_len=max_gen_len,
            #     temperature=temperature,
            #     top_p=top_p,
            # )
            for msg in dialog:
                print(f"{msg.role.capitalize()}: {msg.content}\n")

            model_input = formatter.encode_dialog_prompt(dialog, tool_prompt_format=False)

            # Do initial prefill
            vision_images, vision_mask, tokens = model.get_prefill_inputs(model_input)
            total_len = len(tokens) + max_gen_len  # Prepares mask for full length of output
            # Create tokens tensor
            pad_id = tokenizer.pad_id
            bsz = 1
            tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long)
            tokens[0, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            xattn_caches, cross_attention_masks, full_text_row_masked_out_mask, logits = model.forward_prefill(
                vision_images, vision_mask, tokens, total_len
            )

            next_token, text = sampler(logits)
            logger.info(f"Prefill output: {text}")

            # Iterate over decode
            # for gen_idx in range(max_gen_len-1):

            # out_message = result.generation
            # print(f"> {out_message.role.capitalize()}: {out_message.content}")
            # for t in out_message.tool_calls:
            #     print(f"  Tool call: {t.tool_name} ({t.arguments})")
            # print("\n==================================\n")
