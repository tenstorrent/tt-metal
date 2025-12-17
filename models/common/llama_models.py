# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, pipeline


class Role(Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    ipython = "ipython"


class StopReason(Enum):
    end_of_turn = "end_of_turn"
    end_of_message = "end_of_message"
    out_of_tokens = "out_of_tokens"


@dataclass
class TokenResult:
    token: int
    text: str
    logprobs: Optional[List[float]] = None


@dataclass
class CompletionMessage:
    content: str
    role: Role = Role.assistant.value


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    From: https://github.com/meta-llama/llama-models/blob/v0.1.5/models/llama3/reference_impl/generation.py#L450-L472
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def extract_images_from_messages(messages):
    images = []
    for message in messages:
        if "content" in message:
            contents = message["content"]
            for content in contents:
                if (content["type"] == "image") and ("image" in content):
                    images.append(content["image"])
    return images


def create_vision_mask(
    tokens: List[int],
    vision_token: int,
) -> List[List[int]]:
    """From: https://github.com/meta-llama/llama-models/blob/v0.1.5/models/llama3/api/chat_format.py#L253-L276"""
    vision_token_locations = [i for i, token in enumerate(tokens) if token == vision_token]
    if len(vision_token_locations) == 0:
        return []

    if len(vision_token_locations) == 1:
        # only one image present, unmask until end of sequence
        return [[vision_token_locations[0], -1]]
    vision_masks = [[loc1, loc2] for loc1, loc2 in zip(vision_token_locations[:-1], vision_token_locations[1:])]
    # last image will attend to all subsequent text
    vision_masks.append([vision_token_locations[-1], len(tokens)])

    # if there are two or more consecutive vision tokens,
    # they should all attend to all subsequent
    # text present
    last_mask_end = vision_masks[-1][1]
    for vision_mask in vision_masks[::-1]:
        if vision_mask[0] == vision_mask[1] - 1:
            vision_mask[1] = last_mask_end
        last_mask_end = vision_mask[1]
    return vision_masks


def encode_content(content, images, image_token):
    if isinstance(content, Image):
        images.append(content)
        assert image_token is not None
        return image_token
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "\n".join(encode_content(item, images) for item in content)
    if isinstance(content, dict):
        content_type = content.get("type")
        if content_type == "text":
            return content["text"]
        if content_type == "image":
            # TBD: support url
            images.append(content["image"])
            assert image_token is not None
            return image_token
    raise ValueError(f"Unknown content format: {content}")


class GeneratorChat:
    def __init__(self, model_name, max_batch_size=1):
        self.pipe = pipeline("image-text-to-text", model=model_name, batch_size=max_batch_size)

    def chat_completion(
        self,
        messages,
        temperature=0.6,
        top_p: float = 0.9,
        max_gen_len=None,
    ):
        generation_output = self.pipe(
            text=messages, temperature=temperature, top_p=top_p, max_new_tokens=max_gen_len, return_full_text=False
        )

        if len(generation_output) == 1:
            return CompletionMessage(content=generation_output[0]["generated_text"])
        return [CompletionMessage(content=output[0]["generated_text"]) for output in generation_output]


class GeneratorText:
    def __init__(self, model_name):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(model_name)

    def text_completion(
        self,
        content: Union[str, Image.Image, Dict, List[Dict]],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len=None,
    ):
        images = []
        text = encode_content(content, images, self.processor.image_token)
        model_input = self.processor(text=text, images=images or None, return_tensors="pt", add_special_tokens=False)
        tokens = self.model.generate(**model_input, temperature=temperature, top_p=top_p, max_new_tokens=max_gen_len)[0]
        tokens = tokens[model_input["input_ids"].shape[-1] :]
        return self.processor.decode(tokens, skip_special_tokens=True)
