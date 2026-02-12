# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torchvision.transforms as tv
from PIL import Image
from torchvision.transforms import functional as F
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


class VariableSizeImageTransform(object):
    """
    This class accepts images of any size and dynamically resize, pads and chunks it
    based on the image aspect ratio and the number of image chunks we allow.

    The algorithm will NOT distort the image fit a certain aspect ratio, because
    that leads to a significant degradation in image quality.

    It can be summarized in 6 steps:
    1. Find all possible canvas combinations of max_num_chunks;
    2. Find the best canvas to fit the image;
    3. Resize without distortion
    4. Pad
    5. Normalize
    6. Chunk

    For example, if an input image is of size 300x800, patch_size of 224,
    and max_num_chunks = 8, it will find the closest aspect ratio that
    is allowed within 8 image chunks, with some restrictions.
    In this case, 2:4 = 2 horizontal patches and 4 vertical patches,
    giving a total of 8 chunks.

    If resize_to_max_canvas, the image will be resized (without distortion),
    to the largest possible resolution. In this case, 388:896, and padded to 448:896,
    where we maintain the original aspect ratio and pad with zeros value for the rest.
    This approach minimizes the amount of padding required for any arbitrary resolution.

    However, if limit_upscaling_to_patch_size is set to True,
    the upscaling will be limited to the patch size. In the example above,
    the image would remain 300x800 (no upscaling), and then padded to 448:896.

    The final output will therefore be of shape (8, 3, 224, 224), where 2x4
    patches are coming from the resizing and chunking.
    """

    def __init__(self, size: int = 224) -> None:
        self.size = size  # image resolution is the size of the patch that the image will be split into defaulted to 224 for LLaMA-3
        self.to_tensor = tv.ToTensor()
        self._mean = (0.48145466, 0.4578275, 0.40821073)
        self._std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = tv.Normalize(
            mean=self._mean,
            std=self._std,
            inplace=True,
        )
        self.resample = tv.InterpolationMode.BILINEAR

    @staticmethod
    def get_factors(n: int) -> Set[int]:
        """
        Calculate all factors of a given number, i.e. a dividor that leaves
        no remainder. For example, if n=12, it will return {1, 2, 3, 4, 6, 12}.

        Args:
            n (int): The number to find factors for.

        Returns:
            set: A set containing all factors of the number.
        """
        factors_set = set()

        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                factors_set.add(i)
                factors_set.add(n // i)
        return factors_set

    def find_supported_resolutions(self, max_num_chunks: int, patch_size: int) -> torch.Tensor:
        """
        Computes all of the allowed resoltuions for a fixed number of chunks
        and patch_size. Useful for when dividing an image into chunks.

        Args:
            max_num_chunks (int): Maximum number of chunks for processing.
            patch_size (int): Size of the side of the patch.

        Returns:
            torch.Tensor: List of possible resolutions as tuples (height, width).

        Example:
            >>> max_num_chunks = 5
            >>> patch_size = 224
            >>> find_supported_resolutions(max_num_chunks, patch_size)
            tensor([(224, 896), (448, 448), (224, 224), (896, 224), (224, 672),
            (672, 224), (224, 448), (448, 224)])

            Given max_num_chunks=4, patch_size=224, it will create a dictionary:
            {
            0.25: [(1, 4)],
            1.0: [(2, 2), (1, 1)],
            4.0: [(4, 1)],
            0.33: [(1, 3)],
            3.0: [(3, 1)],
            0.5: [(1, 2)],
            2.0: [(2, 1)]
            }

            and return the resolutions multiplied by the patch_size:
            [(1*224, 4*224), (2*224, 2*224), ..., (2*224, 1*224)]
        """
        asp_dict = defaultdict(list)
        for chunk_size in range(max_num_chunks, 0, -1):
            _factors = sorted(self.get_factors(chunk_size))
            _asp_ratios = [(factor, chunk_size // factor) for factor in _factors]
            for height, width in _asp_ratios:
                ratio_float = height / width
                asp_dict[ratio_float].append((height, width))

        # get the resolutions multiplied by the patch_size
        possible_resolutions = []
        for key, value in asp_dict.items():
            for height, depth in value:
                possible_resolutions.append((height * patch_size, depth * patch_size))

        return possible_resolutions

    @staticmethod
    def get_max_res_without_distortion(
        image_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> Tuple[int, int]:
        """
        Determines the maximum resolution to which an image can be resized to without distorting its
        aspect ratio, based on the target resolution.

        Args:
            image_size (Tuple[int, int]): The original resolution of the image (height, width).
            target_resolution (Tuple[int, int]): The desired resolution to fit the image into (height, width).
        Returns:
            Tuple[int, int]: The optimal dimensions (height, width) to which the image should be resized.
        Example:
            >>> _get_max_res_without_distortion([200, 300], target_size = [450, 200])
            (134, 200)
            >>> _get_max_res_without_distortion([800, 600], target_size = [450, 1300])
            (450, 338)
        """

        original_width, original_height = image_size
        target_width, target_height = target_size

        scale_w = target_width / original_width
        scale_h = target_height / original_height

        if scale_w < scale_h:
            new_width = target_width
            new_height = min(math.floor(original_height * scale_w), target_height)
        else:
            new_height = target_height
            new_width = min(math.floor(original_width * scale_h), target_width)

        return new_width, new_height

    def _pad(self, image: Image.Image, target_size) -> Image.Image:
        new_width, new_height = target_size
        new_im = Image.new(mode="RGB", size=(new_width, new_height), color=(0, 0, 0))  # type: ignore
        new_im.paste(image)
        return new_im

    def _split(self, image: torch.Tensor, ncw: int, nch: int) -> torch.Tensor:
        # Split image into number of required tiles (width x height)
        num_channels, height, width = image.size()
        image = image.view(num_channels, nch, height // nch, ncw, width // ncw)
        # Permute dimensions to reorder the axes
        image = image.permute(1, 3, 0, 2, 4).contiguous()
        # Reshape into the desired output shape (batch_size * 4, num_channels, width/2, height/2)
        image = image.view(ncw * nch, num_channels, height // nch, width // ncw)
        return image

    def resize_without_distortion(
        self,
        image: torch.Tensor,
        target_size: Tuple[int, int],
        max_upscaling_size: Optional[int],
    ) -> torch.Tensor:
        """
        Used to resize an image to target_resolution, without distortion.

        If target_size requires upscaling the image, the user can set max_upscaling_size to
        limit the upscaling to a maximum size. In this case, since we rescale without distortion,
        modifying target_size works as a boundary for the image's largest side.

        Args:
            resample (str): Resampling method used when resizing images.
                Supports "nearest", "nearest_exact", "bilinear", "bicubic".
            max_upscaling_size (int): The maximum size to upscale the image to.
                If None, there is no limit.
        Examples:
        >>> target_size = (1000, 1200)
        >>> max_upscaling_size = 600
        >>> image_size = (400, 200)
        >>> resize_without_distortion(image_size, target_size, max_upscaling_size)
        (600, 300)  # new_size_without_distortion

        >>> target_size = (1000, 1200)
        >>> max_upscaling_size = 600
        >>> image_size = (2000, 200)
        >>> resize_without_distortion(image_size, target_size, max_upscaling_size)
        (1000, 100)  # new_size_without_distortion

        >>> target_size = (1000, 1200)
        >>> max_upscaling_size = 2000
        >>> image_size = (400, 200)
        >>> resize_without_distortion(image_size, target_size, max_upscaling_size)
        (1000, 500)  # new_size_without_distortion

        >>> target_size = (1000, 1200)
        >>> max_upscaling_size = None
        >>> image_size = (400, 200)
        >>> resize_without_distortion(image_size, target_size, max_upscaling_size)
        (1000, 500)  # new_size_without_distortion
        """

        image_width, image_height = image.size
        image_size = (image_width, image_height)

        # If target_size requires upscaling, we might want to limit the upscaling to max_upscaling_size
        if max_upscaling_size is not None:
            new_target_width = min(max(image_width, max_upscaling_size), target_size[0])
            new_target_height = min(max(image_height, max_upscaling_size), target_size[1])
            target_size = (new_target_width, new_target_height)

        # resize to target_size while preserving aspect ratio
        new_size_without_distortion = self.get_max_res_without_distortion(image_size, target_size)

        image = F.resize(
            image,
            (new_size_without_distortion[1], new_size_without_distortion[0]),
            interpolation=self.resample,
        )

        return image

    def get_best_fit(
        self,
        image_size: Tuple[int, int],
        possible_resolutions: torch.Tensor,
        resize_to_max_canvas: bool = False,
    ) -> Tuple[int, int]:
        """
        Determines the best canvas possible from a list of possible resolutions to, without distortion,
        resize an image to.

        For each possible resolution, calculates the scaling factors for
        width and height, and selects the smallest one, which is the limiting side.
        E.g. to match the canvas you can upscale height by 2x, and width by 1.5x,
        therefore, the maximum upscaling you can do is min(2, 1.5) = 1.5.

        If upscaling is possible (any of the scaling factors is greater than 1),
        then picks the smallest upscaling factor > 1, unless resize_to_max_canvas is True.

        If upscaling is not possible, then picks the largest scaling factor <= 1, i.e.
        reduce downscaling as much as possible.

        If there are multiple resolutions with the same max scale, we pick the one with the lowest area,
        to minimize padding. E.g., the same image can be upscaled to 224x224 and 224x448, but the latter
        has more padding.

        Args:
            image_size (Tuple[int, int]): A tuple containing the height and width of the image.
            possible_resolutions (torch.Tensor): A tensor of shape (N, 2) where each
                row represents a possible resolution (height, width).
            use_max_upscaling (bool): If True, will return the largest upscaling resolution.

        Returns:
            List[int]: The best resolution [height, width] for the given image.

        Example:
            >>> image_size = (200, 300)
            >>> possible_resolutions = torch.tensor([[224, 672],
            ...                                     [672, 224],
            ...                                     [224, 448],
            ...                                     [448, 224],
            ...                                     [224, 224]])
            >>> _get_smallest_upscaling_possibility(image_size, possible_resolutions)
            [224, 448]

            We have:
                scale_w = tensor([2.2400, 0.7467, 1.4933, 0.7467, 0.7467])
                scale_h = tensor([1.1200, 3.3600, 1.1200, 2.2400, 1.1200])
                scales = tensor([1.1200, 0.7467, 1.1200, 0.7467, 0.7467])
            Only one of the scales > 1:
                upscaling_possible = tensor([1.1200, 1.1200])
                smallest_rescale = tensor(1.1200)
            So we pick the resolution with the smallest smallest area:
                areas = tensor([150528, 100352]) # [672, 224], [224, 448]
                optimal_canvas = tensor([224, 448])
        """

        original_width, original_height = image_size

        # get all possible resolutions heights/widths
        target_widths, target_heights = (
            possible_resolutions[:, 0],
            possible_resolutions[:, 1],
        )

        # get scaling factors to resize the image without distortion
        scale_w = target_widths / original_width
        scale_h = target_heights / original_height

        # get the min scale between width and height (limiting side -> no distortion)
        scales = torch.where(scale_w > scale_h, scale_h, scale_w)

        # filter only scales that allow upscaling
        upscaling_options = scales[scales >= 1]
        if len(upscaling_options) > 0:
            if resize_to_max_canvas:
                selected_scale = torch.max(upscaling_options)
            else:
                selected_scale = torch.min(upscaling_options)
        else:
            # no upscaling possible,
            # get the minimum downscaling (max scale for scales<1)
            downscaling_options = scales[scales < 1]
            selected_scale = torch.max(downscaling_options)

        # get all resolutions that support this scaling factor,
        # e.g. you can upscale to 224x224, 224x448, 224x672 without distortion
        chosen_canvas = possible_resolutions[scales == selected_scale]

        # if there are multiple resolutions,
        # get the one with minimum area to reduce padding
        if len(chosen_canvas) > 1:
            areas = chosen_canvas[:, 0] * chosen_canvas[:, 1]
            optimal_idx = torch.argmin(areas)
            optimal_canvas = chosen_canvas[optimal_idx]
        else:
            optimal_canvas = chosen_canvas[0]

        return tuple(optimal_canvas.tolist())

    def __call__(
        self,
        image: Image.Image,
        max_num_chunks: int,
        normalize_img: bool = True,
        resize_to_max_canvas: bool = False,
    ) -> Tuple[Any, Any]:
        """
        Args:
            image (PIL.Image): Image to be resized.
            max_num_chunks (int): Maximum number of chunks to split the image into.
            normalize_img (bool): Whether to normalize the image.
            resize_to_max_canvas (bool): Whether to resize the image to the maximum canvas size.
            If True, picks the canvas the allows the largest resizing without distortion.
            If False, downsample as little as possible, including no resizing at all,
            but never upsample, unless the image is smaller than the patch size.
        """
        assert max_num_chunks > 0
        assert isinstance(image, Image.Image), type(image)
        w, h = image.size

        possible_resolutions = self.find_supported_resolutions(max_num_chunks=max_num_chunks, patch_size=self.size)
        possible_resolutions = torch.tensor(possible_resolutions)

        best_resolution = self.get_best_fit(
            image_size=(w, h),
            possible_resolutions=possible_resolutions,
            resize_to_max_canvas=resize_to_max_canvas,
        )

        max_upscaling_size = None if resize_to_max_canvas else self.size
        image = self.resize_without_distortion(image, best_resolution, max_upscaling_size)
        image = self._pad(image, best_resolution)

        image = self.to_tensor(image)

        if normalize_img:
            image = self.normalize(image)

        ratio_w, ratio_h = (
            best_resolution[0] // self.size,
            best_resolution[1] // self.size,
        )

        image = self._split(image, ratio_w, ratio_h)  # type: ignore

        ar = (ratio_h, ratio_w)
        return image, ar
