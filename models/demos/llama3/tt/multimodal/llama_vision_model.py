# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import logging
import math
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import collections

from PIL import Image as PIL_Image

from torch import nn, Tensor

BFLOAT = False

import importlib

llama_reference_model = importlib.import_module(
    "models.demos.t3000.llama2_70b.reference.llama-models.models.llama3.reference_impl.multimodal.model"
)
llama_reference_image_transforms = importlib.import_module(
    "models.demos.t3000.llama2_70b.reference.llama-models.models.llama3.reference_impl.multimodal.image_transform"
)

import ttnn
from models.demos.llama3.tt.multimodal.llama_image_transformer_vision import TtLlamaCrossAttentionTransformerVision
from models.demos.llama3.tt.multimodal.llama_cross_attention_transformer_text import (
    TtLlamaCrossAttentionTransformerText,
)
from models.demos.llama3.tt.llama_common import (
    prepare_inputs_ttnn_prefill,
    prepare_inputs_ttnn,
    get_prefill_rot_mat,
    prepare_inputs_ttnn_prefill,
    get_rot_transformation_mat,
    get_single_rot_mat,
)

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
    num_tokens,
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
    assert (
        num_tokens == cross_attention_masks.shape[1]
    ), f"Mismatch in text sequence length and cross attention mask sequence length {num_tokens} {cross_attention_masks.shape}"
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


class CrossAttentionTransformer(torch.nn.Module):
    def __init__(
        self,
        args: llama_reference_model.ModelArgs,
        mesh_device,
        state_dict,
        weight_cache_path,
        dtype,
        configuration,
    ) -> None:
        super().__init__()
        self.params = args

        self.model_dim = args.dim

        self.mesh_device = mesh_device
        self.state_dict = state_dict
        self.weight_cache_path = weight_cache_path
        self.dtype = dtype
        self.configuration = configuration

        return_intermediate = "3,7,15,23,30"
        return_intermediate = [int(l) for l in return_intermediate.split(",")]

        self.vision_model = TtLlamaCrossAttentionTransformerVision(
            mesh_device,
            state_dict,
            "vision_model.",
            weight_cache_path=configuration.weight_cache_path(dtype),
            dtype=dtype,
            configuration=configuration,
            return_intermediate=return_intermediate,
        )

        # self.text_model = CrossAttentionTransformerText(args)
        self.text_model = TtLlamaCrossAttentionTransformerText(
            mesh_device,
            state_dict,
            state_dict_prefix="text_model.",
            weight_cache_path=configuration.weight_cache_path(ttnn.bfloat8_b),
            dtype=ttnn.bfloat8_b,
            configuration=configuration,
        )
        self.image_res = args.vision_chunk_size
        self.max_num_chunks = args.vision_max_num_chunks
        self.image_transform = partial(
            llama_reference_image_transforms.VariableSizeImageTransform(size=args.vision_chunk_size),
            max_num_chunks=args.vision_max_num_chunks,
        )

    def setup_cache(self, max_batch_size: int, dtype: torch.dtype):
        self.text_model.setup_cache(max_batch_size, dtype)

    def compute_vision_tokens_masks(
        self,
        batch_images: List[List[PIL_Image.Image]],
        batch_masks: List[List[List[int]]],
        total_len: int,
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
                image_res=self.params.vision_chunk_size,
                max_num_images=max_num_images,
            )

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
            vision_tokens = self.vision_model(stacked_images, aspect_ratios)
            # Back to torch
            vision_tokens = ttnn.to_torch(vision_tokens, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
            vision_tokens = (
                vision_tokens[0].reshape(bsz, max_num_images, self.max_num_chunks, -1, self.model_dim).float()
            )

        bsz, nimg, nchunk, ntok, image_token_dim = tuple(vision_tokens.shape)

        # Prepare vision tokens for TT text_model
        vision_tokens_squeeze = vision_tokens.view(1, bsz, -1, image_token_dim)
        vision_tokens_squeeze = torch.nn.functional.pad(
            vision_tokens_squeeze, (0, 0, 0, 4224 - vision_tokens_squeeze.shape[2]), "constant", 0
        )
        vision_tokens_tt = ttnn.from_torch(
            vision_tokens_squeeze,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        xattn_caches = [
            layer.compute_xattn_kv_cache(vision_tokens_tt) for layer in self.text_model.cross_attention_layers
        ]
        padded_masks = _pad_masks(  # torch.Size([1, 512, 1, 4])
            batch_masks,
            num_chunks,
            total_len,
            self.max_num_chunks,
        )

        # torch.Size([1, 1, 512, 4100]), torch.Size([1, 1, 512, 1])
        cross_attention_masks, full_text_row_masked_out_mask = _get_xattn_mask(
            num_tokens=total_len,
            text_device="cpu",
            text_dtype=torch.float32,  # next(self.text_model.parameters()).dtype,
            vision_tokens=vision_tokens,
            cross_attention_masks=padded_masks,
        )

        cross_attention_masks = torch.nn.functional.pad(
            cross_attention_masks,
            (0, 4224 - cross_attention_masks.shape[3]),
            "constant",
            get_negative_inf_value(torch.float32),
        )
        return (xattn_caches, cross_attention_masks, full_text_row_masked_out_mask)

    def forward(
        self,
        position_ids: torch.Tensor,
        tokens: torch.Tensor,
        cross_attention_masks: torch.Tensor,
        full_text_row_masked_out_mask: torch.Tensor,
        xattn_caches: torch.Tensor,
        text_only_inference: bool = False,
    ) -> torch.Tensor:
        h = self.text_model.get_partially_trainable_embedding(tokens[:, position_ids])
        batch, seq_len = h.shape[:2]
        if seq_len == 1:
            mode = "decode"
        else:
            mode = "prefill"
        # Prepare TT inputs for text_model
        tt_position_id = ttnn.from_torch(
            position_ids.reshape(batch, seq_len),
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        xattn_mask = cross_attention_masks[:, :, position_ids]
        xattn_mask_expand = xattn_mask.expand(-1, self.configuration.n_heads // self.configuration.num_devices, -1, -1)
        if mode == "prefill":
            xattn_mask_expand = torch.nn.functional.pad(
                xattn_mask_expand,
                (0, 0, 0, 128 - xattn_mask_expand.shape[2]),
                "constant",
                get_negative_inf_value(torch.float32),
            )
        tt_xattn_mask = ttnn.from_torch(
            xattn_mask_expand,
            device=self.mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        full_text_mask = full_text_row_masked_out_mask[:, :, position_ids]
        if mode == "prefill":
            full_text_mask = torch.nn.functional.pad(
                full_text_mask, (0, 0, 0, 128 - full_text_mask.shape[2]), "constant", 0
            )
        full_text_mask_expand_1NSH = full_text_mask.expand(
            -1, self.configuration.n_heads // self.configuration.num_devices, -1, self.configuration.head_dim
        )
        tt_full_text_mask_expand_1NSH = ttnn.from_torch(
            full_text_mask_expand_1NSH,
            device=self.mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        full_text_mask_expand_11SD = full_text_mask.expand(-1, -1, -1, self.configuration.dim)
        tt_full_text_mask_expand_11SD = ttnn.from_torch(
            full_text_mask_expand_11SD,
            device=self.mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        # Check mask shapes, pad if in prefill?
        if mode == "prefill":
            # DEBUG: pad h seqlen to 128
            h = torch.nn.functional.pad(h, (0, 0, 0, 128 - h.shape[1]), "constant", 0)
            tt_h = prepare_inputs_ttnn_prefill(
                h,
                self.mesh_device,
            )
            rot_mats = get_prefill_rot_mat(
                self.configuration.head_dim, self.configuration.max_seq_len, self.mesh_device, seq_len=seq_len
            )
            transformation_mat_torch = get_rot_transformation_mat(self.configuration.head_dim)
            transformation_mats = ttnn.as_tensor(
                transformation_mat_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            tt_h = prepare_inputs_ttnn(
                h,
                self.configuration.dim,
                self.mesh_device,
            )
            rot_mats, rot_matrix = get_single_rot_mat(
                self.configuration.head_dim,
                self.mesh_device,
                self.configuration.num_devices,
                start_pos=position_ids.item() - 1,  # TODO: Change function to support decode batch > 1
            )
            transformation_mats = None

            tt_xattn_mask = ttnn.reshape(
                tt_xattn_mask,
                shape=ttnn.Shape(
                    [
                        batch,
                        self.configuration.n_heads // self.configuration.num_devices,
                        seq_len,
                        xattn_mask.shape[-1],
                    ],
                    [batch, self.configuration.n_heads // self.configuration.num_devices, 32, xattn_mask.shape[-1]],
                ),
            )
            tt_full_text_mask_expand_1NSH = ttnn.reshape(
                tt_full_text_mask_expand_1NSH,
                shape=ttnn.Shape(
                    [
                        batch,
                        self.configuration.n_heads // self.configuration.num_devices,
                        seq_len,
                        self.configuration.head_dim,
                    ],
                    [
                        batch,
                        self.configuration.n_heads // self.configuration.num_devices,
                        32,
                        self.configuration.head_dim,
                    ],
                ),
            )

            tt_full_text_mask_expand_11SD = ttnn.reshape(
                tt_full_text_mask_expand_11SD,
                shape=ttnn.Shape(
                    [batch, 1, seq_len, self.configuration.head_dim],
                    [batch, 1, 32, self.configuration.head_dim],
                ),
            )

        logits = self.text_model.forward(
            tt_h,
            xattn_mask=tt_xattn_mask,
            full_text_row_masked_out_mask_1NSH=tt_full_text_mask_expand_1NSH,
            full_text_row_masked_out_mask_11SD=tt_full_text_mask_expand_11SD,
            xattn_caches=xattn_caches,
            current_pos=tt_position_id,
            rot_mat=rot_mats,
            transformation_mats=transformation_mats,
            user_id=0,
            mode=mode,
            text_only_inference=text_only_inference,
        )

        tt_out = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).float()
        if mode == "prefill":
            tt_out = tt_out[0].reshape(batch, 128, -1)[:, :seq_len, :]  # DEBUG: undo padding
        else:
            tt_out = tt_out[0, ..., :batch, :].transpose(0, 1).reshape(batch, seq_len, -1)

        return tt_out


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
) -> torch.Tensor:
    # dtype = torch.bfloat16
    dtype = torch.float32
    inf_value = get_negative_inf_value(dtype)

    bsz = len(all_masks)
    max_num_media = max([len(m) for m in all_masks])

    out_masks = torch.full(
        (bsz, total_len, max_num_media, max_num_chunks),
        inf_value,
        dtype=dtype,
    )

    for idx, (mask, num_chunks) in enumerate(zip(all_masks, all_num_chunks)):
        for mask_idx, (mask_elem, mask_num_chunks) in enumerate(zip(mask, num_chunks)):
            if len(mask_elem) == 2:
                mask_elem[1] = min(mask_elem[1], total_len)
                if mask_elem[1] == -1:
                    mask_elem[1] = total_len
                out_masks[idx, mask_elem[0] : mask_elem[1], mask_idx, :mask_num_chunks].fill_(0.0)

    return out_masks
