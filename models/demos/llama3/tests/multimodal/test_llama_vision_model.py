# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Optional
from loguru import logger

from PIL import Image as PIL_Image
from termcolor import cprint

import llama_models.llama3.reference_impl.generation as llama_reference_generation

from llama_models.llama3.api.datatypes import ImageMedia

from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)

THIS_DIR = Path(__file__).parent.parent.parent.resolve() / "reference/llama_models/models/scripts/"

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
def test_llama_vision_model(
    mesh_device,
    temperature: float = 0,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = 50,
    model_parallel_size: Optional[int] = None,
):
    """
    This test runs the Llama3.2 vision model on CPU and TT concurrently.
    It does not use teacher forcing and compares output logits at every token.
    """
    mesh_device.enable_program_cache()
    mesh_device.enable_async(True)
    ckpt_dir = os.environ["LLAMA_DIR"]
    tokenizer_path = str(Path(ckpt_dir) / "tokenizer.model")

    logger.info(f"Creating reference model from checkpoint in '{ckpt_dir}'")
    generator_pt = llama_reference_generation.Llama.build(
        ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size,
    )

    generator_tt = llama_reference_generation.Llama(generator_pt.model, generator_pt.tokenizer, generator_pt.args)
    logger.info(f"Creating TT model on {len(mesh_device.get_devices())} devices")
    model = create_multimodal_model(generator_tt.args, mesh_device)
    generator_tt.model = model

    # with open(THIS_DIR / "resources/dog.jpg", "rb") as f:
    #     img = PIL_Image.open(f).convert("RGB")

    # with open(THIS_DIR / "resources/pasta.jpeg", "rb") as f:
    #     img2 = PIL_Image.open(f).convert("RGB")

    with open(THIS_DIR / "resources/ocr_image.jpeg", "rb") as f:
        ocr_image = PIL_Image.open(f).convert("RGB")

    # with open(THIS_DIR / "resources/clutter.jpeg", "rb") as f:
    #     clutter = PIL_Image.open(f).convert("RGB")

    interleaved_contents = [
        # text only
        # "The color of the sky is blue but sometimes it can also be",
        # image understanding
        # [ImageMedia(image=img), "If I had to write a haiku for this one"],
        # [ImageMedia(image=img2), "Couting the number of individual spaghetti strands in this image"],
        [ImageMedia(image=ocr_image), "The full text in this image is as follows"],
        # [ImageMedia(image=clutter), "The count of vases, books, and miscellaneous items in this image is"],
    ]

    for content in interleaved_contents:
        logger.info(f"Generating text for content: {content}")
        model_input = generator_pt.formatter.encode_content(content)
        gen_pt = generator_pt.generate(
            model_input, max_gen_len=max_gen_len, temperature=temperature, return_logits=True
        )
        gen_tt = generator_tt.generate(
            model_input, max_gen_len=max_gen_len, temperature=temperature, return_logits=True
        )

        for out_idx, (token_pt, token_tt) in enumerate(zip(gen_pt, gen_tt)):
            logger.info(f"Comparing output token {out_idx}")
            out_pt, out_tt = token_pt[1], token_tt[1]
            out_pt = out_pt[0, -1]
            out_tt = out_tt[0, -1]
            passing, pcc_message = comp_pcc(out_pt, out_tt, 0.90)
            print(f"PCC: {pcc_message}")
            # Check shapes of logprobs

            ref_argmax = torch.argmax(out_pt).item()
            ref_logprob = out_pt[ref_argmax].item()
            ref_token = generator_pt.tokenizer.decode([ref_argmax])

            # Reference model: top-5 tokens
            ref_top5_vals, ref_top5_idxs = torch.topk(out_pt, 5)
            ref_top5_tokens = [generator_pt.tokenizer.decode([idx.item()]) for idx in ref_top5_idxs]
            ref_top5_logprobs = ref_top5_vals.tolist()

            # Test model: top-5 tokens
            top5_vals, top5_idxs = torch.topk(out_tt, 5)
            top5_tokens = [generator_pt.tokenizer.decode([idx.item()]) for idx in top5_idxs]
            top5_logprobs = top5_vals.tolist()

            def entropy(logits):
                probs = torch.softmax(logits, dim=-1)
                return -(probs * torch.log(probs)).sum().item()

            # Print the information
            print(f"Token Position {out_idx}:")
            print(f"  Reference | Test")
            print(f"  Entropy: {entropy(out_pt):.4f} | {entropy(out_tt):.4f}")
            print(f"  Top-5 Tokens:")
            for rank in range(5):
                print(
                    f"    {rank+1}. Token='{ref_top5_tokens[rank]}' @ {ref_top5_logprobs[rank]:.2f} | '{top5_tokens[rank]}' @ {top5_logprobs[rank]:.2f}"
                )
            print()
