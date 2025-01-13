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

import llama_models.llama3.reference_impl.multimodal.model as llama_reference_model
import llama_models.llama3.reference_impl.multimodal.image_transform as llama_reference_image_transforms

import ttnn
from models.demos.llama3.tt.multimodal.llama_cross_attention_transformer_vision import (
    TtLlamaCrossAttentionTransformerVision,
)
from models.demos.llama3.tt.multimodal.llama_cross_attention_transformer_text import (
    TtLlamaCrossAttentionTransformerText,
)
from models.demos.llama3.tt.llama_common import (
    get_prefill_rot_mat,
    get_rot_transformation_mat,
    get_single_rot_mat,
    copy_host_to_device,
    get_padded_prefill_len,
)
from models.utility_functions import (
    nearest_32,
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

        self.text_model = TtLlamaCrossAttentionTransformerText(
            mesh_device,
            state_dict,
            state_dict_prefix="text_model.",
            weight_cache_path=configuration.weight_cache_path(ttnn.bfloat8_b),
            dtype=ttnn.bfloat8_b,
            configuration=configuration,
            use_paged_kv_cache=use_paged_kv_cache,
        )
        self.image_res = configuration.vision_chunk_size
        self.max_num_chunks = configuration.vision_max_num_chunks
        self.image_transform = partial(
            llama_reference_image_transforms.VariableSizeImageTransform(size=configuration.vision_chunk_size),
            max_num_chunks=configuration.vision_max_num_chunks,
        )

    def setup_cache(self, max_batch_size):
        return self.text_model.setup_cache(max_batch_size)

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
                image_res=self.configuration.vision_chunk_size,
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
            chunk_seq_len = self.configuration.vision_chunk_ntok
            # NOTE: slicing up to chunk_seq_len is necessary because padding information is lost by this point
            vision_tokens = (
                vision_tokens[0, :, :chunk_seq_len]
                .reshape(bsz, max_num_images, self.max_num_chunks, -1, self.model_dim)
                .float()
            )

        bsz, nimg, nchunk, ntok, image_token_dim = tuple(vision_tokens.shape)
        padded_seq_len = self.max_num_chunks * nearest_32(self.configuration.vision_chunk_ntok)

        # Prepare vision tokens for TT text_model
        vision_tokens_squeeze = vision_tokens.view(1, bsz, -1, image_token_dim)
        vision_tokens_squeeze = torch.nn.functional.pad(
            vision_tokens_squeeze, (0, 0, 0, padded_seq_len - vision_tokens_squeeze.shape[2]), "constant", 0
        )
        vision_tokens_tt = ttnn.from_torch(
            vision_tokens_squeeze,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

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
            (0, padded_seq_len - cross_attention_masks.shape[3]),
            "constant",
            get_negative_inf_value(torch.float32),
        )
        return (vision_tokens_tt, cross_attention_masks, full_text_row_masked_out_mask)

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
    ):
        B = tokens.shape[0]
        assert B == 1, f"Only batch 1 is supported, got {B}"
        S = tokens.shape[1]
        position_ids = torch.arange(S, dtype=torch.long)
        h = self.prepare_inputs_common(position_ids, tokens)
        padded_seq_len = get_padded_prefill_len(S)

        xattn_mask = cross_attention_masks[:, :, position_ids]
        xattn_mask = torch.nn.functional.pad(
            xattn_mask,
            (0, 0, 0, padded_seq_len - xattn_mask.shape[2]),
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

        h = torch.nn.functional.pad(h, (0, 0, 0, padded_seq_len - h.shape[1]), "constant", 0)
        tt_h = self.configuration.prepare_residual_tensor_prefill(
            h,
        )
        rot_mats = get_prefill_rot_mat(
            self.configuration.head_dim,
            self.configuration.max_seq_len,
            self.mesh_device,
            seq_len=S,
            scale_factor=self.configuration.rope_scaling_factor,
        )

        full_text_mask_expand_11SD = full_text_mask.expand(-1, -1, -1, self.configuration.dim)
        tt_full_text_mask_expand_11SD = ttnn.from_torch(
            full_text_mask_expand_11SD,
            device=self.mesh_device,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
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
        cross_attention_masks,
        full_text_row_masked_out_mask,
        position_id,
        page_table=None,
        cross_page_table=None,
    ):
        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_position_id,
            tt_rope_id,
            tt_page_table,
            tt_cross_page_table,
        ) = self.prepare_decode_inputs_host(
            tokens,
            cross_attention_masks,
            full_text_row_masked_out_mask,
            position_id,
            page_table=page_table,
            cross_page_table=cross_page_table,
        )

        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_position_id,
            tt_rope_id,
            tt_page_table,
            tt_cross_page_table,
        ) = copy_host_to_device(
            (
                tt_h,
                tt_xattn_mask,
                tt_full_text_mask_expand_1NSH,
                tt_position_id,
                tt_rope_id,
                tt_page_table,
                tt_cross_page_table,
            ),
            mesh_device=self.mesh_device,
        )

        tt_h, tt_rot_mats, tt_xattn_mask, tt_full_text_mask_expand_1NSH = self.transform_decode_inputs_device(
            tt_h,
            tt_rope_id,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            B=tokens.shape[0],
        )

        return (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_position_id,
            tt_rot_mats,
            tt_page_table,
            tt_cross_page_table,
        )

    def prepare_decode_inputs_host(
        self,
        tokens,
        cross_attention_masks,
        full_text_row_masked_out_mask,
        position_id,
        page_table=None,
        cross_page_table=None,
    ):
        B = tokens.shape[0]
        assert (
            B == self.configuration.max_batch_size
        ), f"Batch size must match max batch size. Got {B}, expected {self.configuration.max_batch_size}"
        unpadded_batch_size = len(cross_attention_masks)
        assert unpadded_batch_size == len(
            full_text_row_masked_out_mask
        ), f"cross_attention_masks batch dim ({unpadded_batch_size}) does not match full_text_row_masked_out_mask batch dim ({len(full_text_row_masked_out_mask)})"
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

        xattn_mask = torch.cat(
            [cross_attention_masks[i][:, :, position_id[i]] for i in range(unpadded_batch_size)], dim=1
        ).unsqueeze(0)
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

        full_text_mask = torch.cat(
            [full_text_row_masked_out_mask[i][:, :, position_id[i]] for i in range(unpadded_batch_size)], dim=1
        ).unsqueeze(0)
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
            tt_position_id,
            tt_rope_id,
            page_table,
            cross_page_table,
        )

    def transform_decode_inputs_device(self, tt_h, tt_rope_id, tt_xattn_mask, tt_full_text_mask_expand_1NSH, B):
        """
        Does any transformations on device tensors which are necessary before ttnn_decode_forward
        """
        assert (
            B == self.configuration.max_batch_size
        ), f"Batch size must match max batch size. Got {B}, expected {self.configuration.max_batch_size}"
        S = 1

        tt_h = ttnn.to_memory_config(tt_h, self.configuration.model_config["DECODE_RESIDUAL_MEMCFG"])

        tt_rot_mats = self.text_model.rope_setup.get_rot_mats(tt_rope_id)

        tt_xattn_mask = ttnn.to_layout(tt_xattn_mask, ttnn.TILE_LAYOUT)
        tt_xattn_mask = ttnn.reshape(
            tt_xattn_mask,
            shape=ttnn.Shape(
                [
                    S,
                    B,
                    self.configuration.n_heads // self.configuration.num_devices,
                    tt_xattn_mask.shape[-1],
                ],
                [S, B, 32, tt_xattn_mask.shape[-1]],
            ),
        )
        tt_full_text_mask_expand_1NSH = ttnn.to_layout(tt_full_text_mask_expand_1NSH, ttnn.TILE_LAYOUT)
        tt_full_text_mask_expand_1NSH = ttnn.reshape(
            tt_full_text_mask_expand_1NSH,
            shape=ttnn.Shape(
                [
                    S,
                    B,
                    self.configuration.n_heads // self.configuration.num_devices,
                    self.configuration.head_dim,
                ],
                [
                    S,
                    B,
                    32,
                    self.configuration.head_dim,
                ],
            ),
        )

        return (tt_h, tt_rot_mats, tt_xattn_mask, tt_full_text_mask_expand_1NSH)

    def process_output_prefill(self, tt_out, B, last_token_idx):
        tt_out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
        tt_out = tt_out[0, 0, last_token_idx, :]
        return tt_out

    def process_output_decode(self, tt_out, B, S):
        tt_out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
        tt_out = tt_out[:, :, :B, :].reshape(B, S, -1)
        return tt_out

    def forward(
        self,
        position_ids: torch.Tensor,
        tokens: torch.Tensor,
        cross_attention_masks: torch.Tensor,
        full_text_row_masked_out_mask: torch.Tensor,
        xattn_caches,  # list of ttnn tensors
        text_only_inference: bool = False,
        user_id=0,
        vision_tokens=None,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
    ) -> torch.Tensor:
        """
        This method takes torch tensors in, returns torch tensors.
        It also determines whether or not to run prefill or decode.
        """
        B = tokens.shape[0]
        S = position_ids.shape[0]  # TODO: Get B, S from tokens when we don't pass full tokens around
        mode = "decode" if S == 1 else "prefill"

        # pos_arg is used in preparation in different ways based on mode
        pos_arg = S if mode == "prefill" else position_ids.item()
        prepare_fn = self.prepare_inputs_decode if mode == "decode" else self.prepare_inputs_prefill
        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
            tt_position_id,
            rot_mats,
            tt_page_table,
            tt_cross_page_table,
        ) = prepare_fn(
            tokens,
            cross_attention_masks,
            full_text_row_masked_out_mask,
            pos_arg,
            page_table=page_table,
            cross_page_table=cross_page_table,
        )

        logits = self.text_model.forward(
            tt_h,
            xattn_mask=tt_xattn_mask,
            full_text_row_masked_out_mask_1NSH=tt_full_text_mask_expand_1NSH,
            full_text_row_masked_out_mask_11SD=tt_full_text_mask_expand_11SD,
            xattn_caches=xattn_caches,
            current_pos=tt_position_id,
            rot_mats=rot_mats,
            user_id=user_id,
            mode=mode,
            page_table=tt_page_table,
            kv_cache=kv_cache,
            text_only_inference=text_only_inference,
            vision_tokens=vision_tokens,
            cross_page_table=tt_cross_page_table,
        )
        tt_out = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)

        output_fn = self.process_output_decode if mode == "decode" else self.process_output_prefill
        return output_fn(tt_out, B, S)

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
            rot_mats=rot_mats,
            user_id=user_id,
            mode="prefill",
            page_table=page_table,
            kv_cache=kv_cache,
            cross_page_table=cross_page_table,
            vision_tokens=vision_tokens,
            get_last_token=get_last_token,
        )
        tt_out = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        return tt_out

    def ttnn_decode_forward(
        self,
        h,
        xattn_mask,
        full_text_mas_expand_1NSH,
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
            full_text_row_masked_out_mask_11SD=None,
            xattn_caches=xattn_caches,
            current_pos=position_id,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table,
            kv_cache=kv_cache,
            cross_page_table=cross_page_table,
        )
        tt_out = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
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
