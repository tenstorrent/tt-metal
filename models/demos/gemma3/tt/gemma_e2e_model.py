# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import List

import torch
from loguru import logger

import ttnn
from models.demos.gemma3.tt.gemma_vision_model import TtGemmaTransformerVision
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.multimodal.llama_vision_model import _stack_images


def _stack_images(
    images: List[List[torch.Tensor]],  # batch of samples, each with list of image embeddings
) -> List[torch.Tensor]:
    """
    Concatenate image embeddings per sample into a single 2D tensor.

    Args:
        images: List of samples, each being a list of [num_patches, hidden_dim] tensors

    Returns:
        List of [total_patches, hidden_dim] tensors, one per sample
    """
    return [torch.cat(image_list, dim=0) for image_list in images]


class TtGemmaModel(Transformer):
    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        super().__init__(
            args,
            dtype,
            mesh_device,
            state_dict,
            weight_cache_path,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
        )

        self.vision_model = TtGemmaTransformerVision(
            mesh_device=mesh_device,
            state_dict=state_dict,
            tt_ccl=self.tt_ccl,
            state_dict_prefix="model.vision_tower.vision_model.",
            dtype=dtype,
            configuration=args,
            weight_cache_path=weight_cache_path,
        )

    def prepare_inputs_prefill(self, pt_tokens, start_pos=0, page_table=None, chunk_page_table=None, **kwargs):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        TODO: Debate whether this function is responsible for padding
        """

        S = pt_tokens.shape[-1]
        tokens = ttnn.from_torch(
            pt_tokens.reshape(1, 1, 1, -1),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        # self.embed_scale = self.args.dim**0.5
        tokens_embd = self.embd(tokens)
        tokens_embd = ttnn.to_torch(tokens_embd, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))

        if "pixel_values" in kwargs and kwargs.get("pixel_values", None) is not None:
            vision_output = self.compute_vision_token(kwargs.get("pixel_values", None))
            comp_vision_output = ttnn.to_torch(
                vision_output, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
            )[: vision_output.shape[0], :]

            image_features = comp_vision_output.squeeze(0)
            special_image_mask = (pt_tokens == self.args.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(tokens_embd)
            image_features = image_features.to(tokens_embd.device, tokens_embd.dtype)
            tokens_embd = tokens_embd.masked_scatter(special_image_mask, image_features)

        tokens_embd = self.args.prepare_residual_tensor_prefill(
            tokens_embd,
        )

        tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)
        # Slice the rot mats to the prefill seqlen
        assert (
            self.rope_setup.cos_matrix.shape[2] >= start_pos + S
        ), f"Padded prefill end idx {start_pos + S} exceeds max seq len {self.rope_setup.cos_matrix.shape[2]}"

        tt_rot_mats_prefill_global = [
            self.rope_setup.cos_matrix[:, :, start_pos : start_pos + S, :],
            self.rope_setup.sin_matrix[:, :, start_pos : start_pos + S, :],
        ]

        tt_rot_mats_prefill_local = [
            self.rope_local_setup.cos_matrix[:, :, start_pos : start_pos + S, :],
            self.rope_local_setup.sin_matrix[:, :, start_pos : start_pos + S, :],
        ]

        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_page_table = None

        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_chunk_page_table = None

        return tokens_embd, tt_rot_mats_prefill_global, tt_rot_mats_prefill_local, tt_page_table, tt_chunk_page_table

    def compute_vision_token(self, pixel_values, batch_size=3):
        """
        Process vision tokens in batches to avoid OOM for large number of images.

        Args:
            pixel_values: torch.Tensor of shape (B, C, H, W) where B is number of images
            batch_size: Number of images to process in one batch (max 3, or else device runs OOM)

        Returns:
            Combined vision output tensor
        """

        assert 0 < batch_size <= 3, "Device runs OOM with batch size > 3"

        if not isinstance(pixel_values, list):
            pixel_values = [pixel_values]

        pixel_values_batches = []
        total_num_images = 0
        for image in pixel_values:
            num_images = image.shape[0]
            total_num_images += num_images
            if num_images < batch_size:
                pixel_values_batches.append(image)
            else:
                # If image was too big it was split into several, but still in one tensor
                for i in range(0, num_images, batch_size):
                    end_idx = min(i + batch_size, num_images)
                    pixel_values_batches.append(image[i:end_idx])

        logger.info(f"Starting vision encoder for {total_num_images} image(s) in {len(pixel_values_batches)} batch(es)")

        # Process images in batches
        vision_outputs = []
        for batch_idx, batch_pixel_values in enumerate(pixel_values_batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(pixel_values_batches)}")
            batch_vision_output = self.vision_model(batch_pixel_values)
            vision_outputs.append(batch_vision_output)

        # Combine all vision outputs along the batch dimension
        combined_vision_output = ttnn.concat(vision_outputs, dim=1)
        logger.info(f"Vision encoder done")
        return combined_vision_output
