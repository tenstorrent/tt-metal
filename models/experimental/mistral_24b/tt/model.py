# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
This is the end-to-end pipeline for the Mistral-Small-3.1-24B-Instruct-2503 model.

The `MistralTransformer` class inherits from the `Transformer` class in tt_transformers.
It overrides `prepare_inputs_prefill` to run inference on the vision model and
pass the resulting visual tokens to the text model along with text tokens.
"""


import ttnn
import torch

from models.tt_transformers.tt.model import Transformer
from ttnn import ConcatMeshToTensor


class MistralTransformer(Transformer):
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

    def prepare_inputs_prefill(self, tokens, start_pos=0, page_table=None, chunk_page_table=None, **kwargs):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        TODO: Debate whether this function is responsible for padding
        """

        tokens = tokens.reshape(1, 1, 1, -1)
        S = tokens.shape[-1]
        tokens = ttnn.from_torch(
            tokens,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        tokens_embd = self.embd(tokens)

        pixel_values = kwargs["processed_inputs"]["pixel_values"]
        input_ids = kwargs["processed_inputs"]["input_ids"]
        image_sizes = kwargs["processed_inputs"]["image_sizes"]

        if pixel_values is not None:
            vision_model = kwargs["vision_model"]
            vision_output = vision_model(pixel_values, image_sizes)
            vision_output_torch = ttnn.to_torch(
                vision_output, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=-1)
            )[:, : vision_output.shape[-1]]
            tokens_embd = ttnn.to_torch(tokens_embd, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=-1))
            sliced_token_embds = tokens_embd[: tokens_embd.shape[0]]

            image_features = vision_output_torch

            input_ids = torch.nn.functional.pad(
                input_ids, (0, tokens_embd.shape[1] - input_ids.shape[1]), "constant", 0
            )
            special_image_mask = (input_ids == 10).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(tokens_embd)
            image_features = image_features.to(tokens_embd.device, tokens_embd.dtype)
            tokens_embd = tokens_embd.masked_scatter(special_image_mask, image_features)

            tokens_embd = ttnn.from_torch(
                tokens_embd,
                dtype=ttnn.bfloat16,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, dims=(None, 2), mesh_shape=list(self.mesh_device.shape)
                ),
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

        if hasattr(self, "rope_local_setup"):
            tt_rot_mats_prefill_local = [
                self.rope_local_setup.cos_matrix[:, :, start_pos : start_pos + S, :],
                self.rope_local_setup.sin_matrix[:, :, start_pos : start_pos + S, :],
            ]
        else:
            tt_rot_mats_prefill_local = None

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
