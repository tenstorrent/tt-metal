# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import collections
import logging
import math
from collections import defaultdict
from functools import partial
from typing import Any, List, Optional, Set, Tuple

import torch
import torchvision.transforms as tv
from PIL import Image as PIL_Image
from torch import Tensor
from torchvision.transforms import functional as F

import ttnn
from models.common.utility_functions import nearest_32
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode, copy_host_to_device, get_padded_prefill_len
from models.tt_transformers.tt.multimodal.llama_cross_attention_transformer_text import (
    TtLlamaCrossAttentionTransformerText,
)
from models.tt_transformers.tt.multimodal.llama_cross_attention_transformer_vision import (
    TtLlamaCrossAttentionTransformerVision,
)
from models.tt_transformers.tt.rope import get_rot_mats

logger = logging.getLogger(__name__)
MP_SCALE = 8


def get_negative_inf_value(dtype):
    return torch.finfo(dtype).min


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


def build_encoder_attention_mask(
    x: torch.Tensor,
    ar: torch.Tensor,
    ntok: int,
    num_chunks: int,
    n_heads: int,
):
    """
    Build vision encoder attention mask that omits padding tokens.
    """
    masks = []
    for arx in ar:
        mask_i = torch.ones((num_chunks, x.shape[2], 1), dtype=x.dtype)
        mask_i[: arx[0] * arx[1], :ntok] = 0
        mask_i = mask_i.view(num_chunks * x.shape[2], -1)
        mask_i = mask_i @ mask_i.T * get_negative_inf_value(x.dtype)
        mask_i = mask_i.unsqueeze(0)
        masks.append(mask_i)
    masks = torch.stack(masks).to(x.device).expand(-1, n_heads, -1, -1)
    return masks


def _get_full_row_masked_out_mask(
    attn_bias,
    negative_inf_value,
):
    """
    attn_bias should be a 4D tensor of shape [B, H, S1, S2]
    where B is the batch size, H is the number of heads,
    and S1/S2 are the sequence lengths. This returns
    a 4D tensor of shape [B, H, S1, 1] which stores boolean
    values which are 0 if the a full row in the last dimension
    contains negative infinity values, otherwise it's 1.
    """
    return (attn_bias != negative_inf_value).any(dim=-1).type_as(attn_bias)[..., None]


def _get_xattn_mask(
    text_device,
    text_dtype,
    vision_tokens,
    cross_attention_masks,
) -> Tuple[Tensor, Tensor]:
    assert vision_tokens is not None, "Vision tokens must be provided"
    vision_seqlen = vision_tokens.shape[3]
    assert (
        vision_tokens.shape[1] == cross_attention_masks.shape[2]
    ), f"Mismatch in number of images given and number of masks given {vision_tokens.shape} {cross_attention_masks.shape}"
    assert (
        vision_tokens.shape[2] == cross_attention_masks.shape[3]
    ), f"Vision tokens shape {vision_tokens.shape} mismatch with xattn shape {cross_attention_masks.shape}"

    _, _, _, num_image_tokens, image_token_dim = tuple(vision_tokens.shape)
    bsz, ntext, nimg, nchunks = cross_attention_masks.shape
    cross_attention_masks = (
        cross_attention_masks.repeat_interleave(vision_seqlen, dim=3).view(bsz, ntext, -1).unsqueeze(1)
    )
    full_text_row_masked_out_mask = _get_full_row_masked_out_mask(
        cross_attention_masks,
        get_negative_inf_value(cross_attention_masks.dtype),
    )
    cross_attention_masks *= full_text_row_masked_out_mask

    return (
        cross_attention_masks.to(device=text_device, dtype=text_dtype),
        full_text_row_masked_out_mask,
    )


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

    def _pad(self, image: PIL_Image.Image, target_size) -> PIL_Image.Image:
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
        image: PIL_Image.Image,
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
        assert isinstance(image, PIL_Image.Image), type(image)
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


class CrossAttentionTransformer(torch.nn.Module):
    def __init__(
        self,
        mesh_device,
        state_dict,
        weight_cache_path,
        dtype,
        configuration,
        use_paged_kv_cache=False,
    ) -> None:
        super().__init__()

        self.model_dim = configuration.dim

        self.mesh_device = mesh_device
        self.tt_ccl = TT_CCL(self.mesh_device)
        self.weight_cache_path = weight_cache_path
        self.dtype = dtype
        self.configuration = configuration

        return_intermediate = "3,7,15,23,30"
        return_intermediate = [int(l) for l in return_intermediate.split(",")]

        self.vision_model = TtLlamaCrossAttentionTransformerVision(
            mesh_device,
            self.tt_ccl,
            state_dict,
            "vision_model.",
            weight_cache_path=configuration.weight_cache_path(dtype),
            dtype=dtype,
            configuration=configuration,
            return_intermediate=return_intermediate,
        )

        self.text_model = TtLlamaCrossAttentionTransformerText(
            mesh_device,
            self.tt_ccl,
            state_dict,
            state_dict_prefix="text_model.",
            weight_cache_path=configuration.weight_cache_path(ttnn.bfloat8_b),
            dtype=ttnn.bfloat8_b,
            configuration=configuration,
            use_paged_kv_cache=use_paged_kv_cache,
        )
        self.image_res = configuration.vision_chunk_size
        self.max_num_chunks = configuration.vision_max_num_chunks
        self.num_vision_tokens = self.max_num_chunks * nearest_32(self.configuration.vision_chunk_ntok)
        self.image_transform = partial(
            VariableSizeImageTransform(size=configuration.vision_chunk_size),
            max_num_chunks=configuration.vision_max_num_chunks,
        )

    def setup_cache(self, max_batch_size):
        return self.text_model.setup_cache(max_batch_size)

    def compute_vision_tokens_masks(
        self,
        batch_images: List[List[PIL_Image.Image]],
        batch_masks: List[List[List[int]]],
        total_len: int,
        prefill_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        skip_vision_encoder = False

        assert len(batch_images) == len(batch_masks), "Images and masks must have the same length"

        max_num_images = max(len(x) for x in batch_images)
        bsz = len(batch_images)

        if max_num_images == 0:
            num_chunks = [[self.max_num_chunks] for _ in batch_images]
            skip_vision_encoder = True
        else:
            images_and_aspect_ratios = [[self.image_transform(im) for im in row] for row in batch_images]
            transformed_images = [[x[0] for x in row] for row in images_and_aspect_ratios]

            aspect_ratios = torch.ones(bsz, max_num_images, 2, dtype=torch.int64)
            for i, row in enumerate(images_and_aspect_ratios):
                if len(row) > 0:
                    aspect_ratios[i, : len(row)] = torch.stack([torch.tensor(x[1]) for x in row])

            stacked_images, num_chunks = _stack_images(
                transformed_images,
                max_num_chunks=self.max_num_chunks,
                image_res=self.configuration.vision_chunk_size,
                max_num_images=max_num_images,
            )

            max_actual_num_chunks = max([i for chunk in num_chunks for i in chunk])
            max_actual_num_chunks = max_actual_num_chunks if max_actual_num_chunks <= 2 else self.max_num_chunks

        if skip_vision_encoder:
            vision_tokens = torch.zeros(
                (
                    bsz,
                    max_num_images,
                    self.max_num_chunks,
                    int((self.vision_model.image_res / self.vision_model.patch_size) ** 2 + 1),
                    self.model_dim,
                ),
            )
        else:
            # TT vision_model
            vision_tokens = self.vision_model(stacked_images, aspect_ratios, max_actual_num_chunks)
            chunk_seq_len = self.configuration.vision_chunk_ntok
            # NOTE: slicing up to chunk_seq_len is necessary because padding information is lost by this point
            vision_tokens = ttnn.reshape(
                vision_tokens[0, :, :chunk_seq_len], (bsz, max_num_images, max_actual_num_chunks, -1, self.model_dim)
            )

        bsz, nimg, nchunk, ntok, image_token_dim = tuple(vision_tokens.shape)
        padded_seq_len = self.num_vision_tokens

        # Prepare vision tokens for TT text_model
        vision_tokens_squeeze = ttnn.reshape(vision_tokens, (1, bsz, -1, image_token_dim))
        vision_tokens_squeeze = ttnn.pad(
            vision_tokens_squeeze, [(0, 0), (0, 0), (0, padded_seq_len - vision_tokens_squeeze.shape[2]), (0, 0)], 0
        )

        prefill_padded_masks, decode_padded_masks = _pad_masks(  # torch.Size([1, 512, 1, 4])
            batch_masks,
            num_chunks,
            total_len,
            max_actual_num_chunks,
            prefill_len,
        )

        # torch.Size([1, 1, 512, 4100]), torch.Size([1, 1, 512, 1])
        prefill_cross_attention_masks, prefill_full_text_row_masked_out_mask = _get_xattn_mask(
            text_device="cpu",
            text_dtype=torch.float32,  # next(self.text_model.parameters()).dtype,
            vision_tokens=vision_tokens,
            cross_attention_masks=prefill_padded_masks,
        )
        decode_cross_attention_masks, decode_full_text_row_masked_out_mask = _get_xattn_mask(
            text_device="cpu",
            text_dtype=torch.float32,  # next(self.text_model.parameters()).dtype,
            vision_tokens=vision_tokens,
            cross_attention_masks=decode_padded_masks,
        )

        return (
            vision_tokens_squeeze,
            prefill_cross_attention_masks,
            prefill_full_text_row_masked_out_mask,
            decode_cross_attention_masks,
            decode_full_text_row_masked_out_mask,
        )

    def validate_inputs(self, tokens, position_ids):
        batch, seq_len = tokens.shape[:2]
        assert (
            seq_len <= self.configuration.max_seq_len
        ), f"Sequence length {seq_len} exceeds max sequence length {self.configuration.max_seq_len}"
        assert len(position_ids.shape) == 1, f"Position ids must be 1D, got {len(position_ids.shape)}"

    def prepare_inputs_common(self, position_ids, tokens):
        self.validate_inputs(tokens, position_ids)
        h = self.text_model.get_partially_trainable_embedding(tokens)
        return h

    def prepare_inputs_prefill(
        self,
        tokens,
        cross_attention_masks,
        full_text_row_masked_out_mask,
        prefill_len,
        page_table=None,
        cross_page_table=None,
        text_only_inference=False,
    ):
        B = tokens.shape[0]
        assert B == 1, f"Only batch 1 is supported, got {B}"
        S = tokens.shape[1]
        position_ids = torch.arange(S, dtype=torch.long)
        h = self.prepare_inputs_common(position_ids, tokens)
        padded_seq_len = get_padded_prefill_len(S)

        if not text_only_inference:
            # Prepare cross attention mask
            xattn_mask = cross_attention_masks[:, :, position_ids]
            xattn_mask = torch.nn.functional.pad(
                xattn_mask,
                (0, self.num_vision_tokens - xattn_mask.shape[3], 0, padded_seq_len - xattn_mask.shape[2]),
                "constant",
                get_negative_inf_value(torch.float32),
            )
            tt_xattn_mask = ttnn.from_torch(
                xattn_mask,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            tt_xattn_mask = ttnn.to_layout(tt_xattn_mask, ttnn.TILE_LAYOUT)
            tt_xattn_mask = ttnn.typecast(tt_xattn_mask, ttnn.bfloat4_b)

            # Prepare text masks
            full_text_mask = full_text_row_masked_out_mask[:, :, position_ids]
            full_text_mask = torch.nn.functional.pad(
                full_text_mask, (0, 0, 0, padded_seq_len - full_text_mask.shape[2]), "constant", 0
            )
            full_text_mask_expand_1NSH = full_text_mask.expand(
                -1, self.configuration.n_heads // self.configuration.num_devices, -1, self.configuration.head_dim
            )
            tt_full_text_mask_expand_1NSH = ttnn.from_torch(
                full_text_mask_expand_1NSH,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            tt_full_text_mask_expand_1NSH = ttnn.to_layout(tt_full_text_mask_expand_1NSH, ttnn.TILE_LAYOUT)
            tt_full_text_mask_expand_1NSH = ttnn.typecast(tt_full_text_mask_expand_1NSH, ttnn.bfloat4_b)

            full_text_mask_expand_11SD = full_text_mask.expand(-1, -1, -1, self.configuration.dim)
            tt_full_text_mask_expand_11SD = ttnn.from_torch(
                full_text_mask_expand_11SD,
                device=self.mesh_device,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
            )

            if isinstance(cross_page_table, torch.Tensor):
                # Support vLLM tensor cross_page_table input
                cross_page_table = ttnn.as_tensor(
                    cross_page_table,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
        else:
            assert cross_attention_masks is None and full_text_row_masked_out_mask is None
            tt_xattn_mask = None
            tt_full_text_mask_expand_1NSH = None
            tt_full_text_mask_expand_11SD = None

        h = torch.nn.functional.pad(h, (0, 0, 0, padded_seq_len - h.shape[1]), "constant", 0)
        tt_h = self.configuration.prepare_residual_tensor_prefill(
            h,
        )
        rot_mats = get_rot_mats(
            head_dim=self.configuration.head_dim,
            device=self.mesh_device,
            seq_len=S,
            theta=self.configuration.rope_theta,
            rope_scaling=self.configuration.rope_scaling,
        )

        if isinstance(page_table, torch.Tensor):
            # Support vLLM tensor page_table input
            page_table = ttnn.as_tensor(
                page_table,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        return (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
            rot_mats,
            page_table,
            cross_page_table,
        )

    def prepare_inputs_decode(
        self,
        tokens,
        prefill_cross_attention_masks,
        prefill_full_text_row_masked_out_mask,
        decode_cross_attention_masks,
        decode_full_text_row_masked_out_mask,
        position_id,
        page_table=None,
        cross_page_table=None,
    ):
        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
            tt_position_id,
            tt_rope_id,
            tt_page_table,
            tt_cross_page_table,
        ) = self.prepare_decode_inputs_host(
            tokens,
            prefill_cross_attention_masks,
            prefill_full_text_row_masked_out_mask,
            decode_cross_attention_masks,
            decode_full_text_row_masked_out_mask,
            position_id,
            page_table=page_table,
            cross_page_table=cross_page_table,
        )

        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
            tt_position_id,
            tt_rope_id,
            tt_page_table,
            tt_cross_page_table,
        ) = copy_host_to_device(
            (
                tt_h,
                tt_xattn_mask,
                tt_full_text_mask_expand_1NSH,
                tt_full_text_mask_expand_11SD,
                tt_position_id,
                tt_rope_id,
                tt_page_table,
                tt_cross_page_table,
            ),
            mesh_device=self.mesh_device,
        )

        (
            tt_h,
            tt_rot_mats,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
        ) = self.transform_decode_inputs_device(
            tt_h,
            tt_rope_id,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
            B=tokens.shape[0],
        )

        return (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
            tt_position_id,
            tt_rot_mats,
            tt_page_table,
            tt_cross_page_table,
        )

    def prepare_decode_inputs_host(
        self,
        tokens,
        prefill_cross_attention_masks,
        prefill_full_text_row_masked_out_mask,
        decode_cross_attention_masks,
        decode_full_text_row_masked_out_mask,
        position_id,
        page_table=None,
        cross_page_table=None,
    ):
        B = tokens.shape[0]
        assert (
            B == self.configuration.max_batch_size
        ), f"Batch size must match max batch size. Got {B}, expected {self.configuration.max_batch_size}"
        unpadded_batch_size = len(prefill_cross_attention_masks)
        assert unpadded_batch_size == len(
            prefill_full_text_row_masked_out_mask
        ), f"prefill_cross_attention_masks batch dim ({unpadded_batch_size}) does not match prefill_full_text_row_masked_out_mask batch dim ({len(prefill_full_text_row_masked_out_mask)})"
        assert unpadded_batch_size == len(
            decode_cross_attention_masks
        ), f"decode_cross_attention_masks batch dim ({unpadded_batch_size}) does not match decode_full_text_row_masked_out_mask batch dim ({len(decode_full_text_row_masked_out_mask)})"
        h = self.prepare_inputs_common(position_id, tokens)
        tt_h = self.configuration.prepare_residual_tensor_decode(
            h,
            None,
            on_host=True,
        )

        tt_position_id = ttnn.from_torch(
            position_id,
            device=None,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        rot_position_id = torch.maximum(
            position_id, torch.tensor(0, dtype=torch.int64)
        )  # Ensure position indices are non-negative
        tt_rope_id = self.text_model.rope_setup.get_rot_idxs(rot_position_id, on_host=True)

        xattn_mask = []
        full_text_mask = []
        for i in range(unpadded_batch_size):
            text_only_user = (
                prefill_cross_attention_masks[i] is None and prefill_full_text_row_masked_out_mask[i] is None
            )
            if not text_only_user:
                if prefill_cross_attention_masks[i].shape[2] > position_id[i].item():
                    xattn_mask_i = torch.nn.functional.pad(
                        prefill_cross_attention_masks[i][:, :, position_id[i]],
                        (0, self.num_vision_tokens - prefill_cross_attention_masks[i].shape[3]),
                        "constant",
                        get_negative_inf_value(torch.float32),
                    )
                    xattn_mask.append(xattn_mask_i)
                    full_text_mask.append(prefill_full_text_row_masked_out_mask[i][:, :, position_id[i]])
                else:
                    xattn_mask_i = torch.nn.functional.pad(
                        decode_cross_attention_masks[i][:, :, 0],
                        (0, self.num_vision_tokens - decode_cross_attention_masks[i].shape[3]),
                        "constant",
                        get_negative_inf_value(torch.float32),
                    )
                    xattn_mask.append(xattn_mask_i)
                    full_text_mask.append(decode_full_text_row_masked_out_mask[i][:, :, 0])
            else:
                xattn_mask.append(torch.zeros(1, 1, self.num_vision_tokens))
                full_text_mask.append(torch.zeros(1, 1, 1))

        xattn_mask = torch.cat(xattn_mask, dim=1).unsqueeze(0)
        # Pad xattn_mask along batch if tokens have been padded
        if B > unpadded_batch_size:
            xattn_mask = torch.cat(
                [xattn_mask, torch.zeros(1, 1, B - unpadded_batch_size, xattn_mask.shape[-1])], dim=2
            )
        xattn_mask_expand = xattn_mask.expand(-1, self.configuration.n_heads // self.configuration.num_devices, -1, -1)
        xattn_mask_expand = xattn_mask_expand.transpose(1, 2).contiguous()
        tt_xattn_mask = ttnn.from_torch(
            xattn_mask_expand,
            device=None,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        full_text_mask = torch.cat(full_text_mask, dim=1).unsqueeze(0)
        # Pad full_text_mask along batch if tokens have been padded
        if B > unpadded_batch_size:
            full_text_mask = torch.cat(
                [full_text_mask, torch.zeros(1, 1, B - unpadded_batch_size, full_text_mask.shape[-1])], dim=2
            )
        full_text_mask_expand_1NSH = full_text_mask.expand(
            -1, self.configuration.n_heads // self.configuration.num_devices, -1, self.configuration.head_dim
        )
        full_text_mask_expand_1NSH = full_text_mask_expand_1NSH.transpose(1, 2).contiguous()
        tt_full_text_mask_expand_1NSH = ttnn.from_torch(
            full_text_mask_expand_1NSH,
            device=None,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        full_text_mask_expand_11SD = full_text_mask
        if B < self.configuration.tile_size:
            full_text_mask_expand_11SD = torch.cat(
                [full_text_mask_expand_11SD, torch.zeros(1, 1, self.configuration.tile_size - B, 1)], dim=2
            )
        full_text_mask_expand_11SD = full_text_mask_expand_11SD.expand(
            -1, -1, -1, self.configuration.dim // self.configuration.num_devices
        )
        tt_full_text_mask_expand_11SD = ttnn.from_torch(
            full_text_mask_expand_11SD,
            device=None,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        if isinstance(page_table, torch.Tensor):
            # Support vLLM tensor page_table input
            page_table = ttnn.as_tensor(
                page_table,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        if isinstance(cross_page_table, torch.Tensor):
            # Support vLLM tensor cross_page_table input
            cross_page_table = ttnn.as_tensor(
                cross_page_table,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        return (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
            tt_position_id,
            tt_rope_id,
            page_table,
            cross_page_table,
        )

    def transform_decode_inputs_device(
        self, tt_h, tt_rope_id, tt_xattn_mask, tt_full_text_mask_expand_1NSH, tt_full_text_mask_expand_11SD, B
    ):
        """
        Does any transformations on device tensors which are necessary before ttnn_decode_forward
        """
        assert (
            B == self.configuration.max_batch_size
        ), f"Batch size must match max batch size. Got {B}, expected {self.configuration.max_batch_size}"
        S = 1

        tt_h = ttnn.to_memory_config(tt_h, self.configuration.get_residual_mem_config(Mode.DECODE))

        tt_rot_mats = self.text_model.rope_setup.get_rot_mats(tt_rope_id)

        tt_xattn_mask = ttnn.to_layout(tt_xattn_mask, ttnn.TILE_LAYOUT)
        tt_xattn_mask = ttnn.reshape(
            tt_xattn_mask,
            [S, B, self.configuration.n_heads // self.configuration.num_devices, tt_xattn_mask.shape[-1]],
            [S, B, 32, tt_xattn_mask.shape[-1]],
        )
        tt_full_text_mask_expand_1NSH = ttnn.to_layout(tt_full_text_mask_expand_1NSH, ttnn.TILE_LAYOUT)
        tt_full_text_mask_expand_1NSH = ttnn.reshape(
            tt_full_text_mask_expand_1NSH,
            [S, B, self.configuration.n_heads // self.configuration.num_devices, self.configuration.head_dim],
            [S, B, 32, self.configuration.head_dim],
        )
        tt_full_text_mask_expand_11SD = ttnn.to_layout(tt_full_text_mask_expand_11SD, ttnn.TILE_LAYOUT)

        return (tt_h, tt_rot_mats, tt_xattn_mask, tt_full_text_mask_expand_1NSH, tt_full_text_mask_expand_11SD)

    def process_output_prefill(self, tt_out, B, last_token_idx):
        tt_out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
        tt_out = tt_out[0, 0, last_token_idx, :]
        return tt_out

    def process_output_decode(self, tt_out, B, S, is_tokens=False):
        """
        Input is ttnn device tensor of logits if is_tokens=False, otherwise tokens. Output is the corresponding torch tensor.
        """
        tt_out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
        tt_out = tt_out[:, :, :B, :].reshape(B, S, -1)
        return tt_out

    def ttnn_prefill_forward(
        self,
        h,
        xattn_mask,
        full_text_mas_expand_1NSH,
        full_text_mask_expand_11SD,
        xattn_caches,
        rot_mats,
        user_id,
        vision_tokens,
        page_table=None,
        kv_cache=None,
        get_last_token=-1,
        cross_page_table=None,
        text_only_inference=False,
    ):
        """
        This method runs prefill forward. It takes ttnn tensors in, returns ttnn tensors.
        """

        if cross_page_table is not None:
            assert (
                xattn_caches is None and kv_cache is not None
            ), "no separate xattn_caches should be allocated when using cross_page_table with paged kv cache"

        logits = self.text_model.forward(
            h,
            xattn_mask=xattn_mask,
            full_text_row_masked_out_mask_1NSH=full_text_mas_expand_1NSH,
            full_text_row_masked_out_mask_11SD=full_text_mask_expand_11SD,
            xattn_caches=xattn_caches,
            current_pos=None,
            rot_mats_global=rot_mats,
            user_id=user_id,
            mode=Mode.PREFILL,
            page_table=page_table,
            kv_cache=kv_cache,
            cross_page_table=cross_page_table,
            text_only_inference=text_only_inference,
            vision_tokens=vision_tokens,
            get_last_token=get_last_token,
        )
        tt_out = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        tt_out = ttnn.to_memory_config(tt_out, ttnn.DRAM_MEMORY_CONFIG)
        return tt_out

    def ttnn_decode_forward(
        self,
        h,
        xattn_mask,
        full_text_mas_expand_1NSH,
        full_text_mask_expand_11SD,
        xattn_caches,
        position_id,
        rot_mats,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
    ):
        """
        This method runs decode forward. It takes ttnn tensors in, returns ttnn tensors.
        """

        if cross_page_table is not None:
            assert (
                xattn_caches is None and kv_cache is not None
            ), "no separate xattn_caches should be allocated when using cross_page_table with paged kv cache"

        logits = self.text_model.forward(
            h,
            xattn_mask=xattn_mask,
            full_text_row_masked_out_mask_1NSH=full_text_mas_expand_1NSH,
            full_text_row_masked_out_mask_11SD=full_text_mask_expand_11SD,
            xattn_caches=xattn_caches,
            current_pos=position_id,
            rot_mats_global=rot_mats,
            mode=Mode.DECODE,
            page_table=page_table,
            kv_cache=kv_cache,
            cross_page_table=cross_page_table,
        )
        tt_out = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)

        # Return logits and None for log-probs for compatibility with generator interface
        return tt_out, None


def _stack_images(
    images: List[List[PIL_Image.Image]],
    max_num_chunks: int,
    image_res: int,
    max_num_images: int,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Takes a list of list of images and stacks them into a tensor.
    This function is needed since images can be of completely
    different resolutions and aspect ratios.
    """
    out_images, out_num_chunks = [], []
    for imgs_sample in images:
        out_images_i = torch.zeros(
            max_num_images,
            max_num_chunks,
            3,
            image_res,
            image_res,
        )
        _num_chunks = []
        for j, chunks_image in enumerate(imgs_sample):
            out_images_i[j, : chunks_image.shape[0]] = chunks_image
            _num_chunks.append(chunks_image.shape[0])
        out_images.append(out_images_i)
        out_num_chunks.append(_num_chunks)
    return torch.stack(out_images), out_num_chunks


def _pad_masks(
    all_masks: List[List[List[int]]],
    all_num_chunks: List[List[int]],
    total_len: int,
    max_num_chunks: int,
    prefill_len: int,
) -> torch.Tensor:
    # dtype = torch.bfloat16
    dtype = torch.float32
    inf_value = get_negative_inf_value(dtype)

    bsz = len(all_masks)
    max_num_media = max([len(m) for m in all_masks])
    max_mask_len = max([max(m[1]) if len(m) == 2 and m[1] != -1 else prefill_len for m in all_masks])
    max_mask_len = min(max_mask_len, total_len)

    prefill_out_masks = torch.full(
        (bsz, max_mask_len, max_num_media, max_num_chunks),
        inf_value,
        dtype=dtype,
    )

    decode_out_masks = torch.full(
        (bsz, 1, max_num_media, max_num_chunks),
        inf_value,
        dtype=dtype,
    )

    for idx, (mask, num_chunks) in enumerate(zip(all_masks, all_num_chunks)):
        for mask_idx, (mask_elem, mask_num_chunks) in enumerate(zip(mask, num_chunks)):
            if len(mask_elem) == 2:
                mask_elem[1] = min(mask_elem[1], max_mask_len)
                if mask_elem[1] == -1:
                    decode_out_masks[idx, 0, mask_idx, :mask_num_chunks].fill_(0.0)
                    mask_elem[1] = max_mask_len
                prefill_out_masks[idx, mask_elem[0] : mask_elem[1], mask_idx, :mask_num_chunks].fill_(0.0)

    return prefill_out_masks, decode_out_masks
