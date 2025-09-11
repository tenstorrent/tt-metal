"""
This is the Gemma3 end-to-end model.
"""

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.experimental.gemma3_4b.tt.gemma_text_model import Gemma3Transformer
from models.experimental.gemma3_4b.tt.gemma_vision_model import TtSiglipGemmaVisionModel
from models.experimental.gemma3_4b.tt.mmp import TtGemma3MultiModalProjector
from models.tt_transformers.tt.ccl import TT_CCL


class TtGemma3Model(Gemma3Transformer):
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
        self.tt_ccl = TT_CCL(mesh_device)

        super().__init__(
            args=args,
            dtype=dtype,
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
        )

        self.vision_encoder = TtSiglipGemmaVisionModel(
            mesh_device,
            state_dict=state_dict,
            tt_ccl=self.tt_ccl,
            state_dict_prefix=args.state_dict_vision_prefix,
            weight_cache_path=args.weight_cache_path(dtype),
            dtype=dtype,
            configuration=args,
        )

        self.mmp = TtGemma3MultiModalProjector(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix="multi_modal_projector",
            image_size=args.image_size,
            patch_size=args.vision_patch_size,
            hidden_size=args.vision_hidden_dim,
            mm_tokens_per_image=args.mm_tokens_per_image,
            weight_cache_path=args.weight_cache_path(dtype),
            layer_norm_eps=1e-06,  # layer_norm_eps
            dtype=dtype,
            configuration=args,
        )

    def prepare_inputs_prefill(self, pt_tokens, start_pos=0, page_table=None, chunk_page_table=None, **kwargs):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        TODO: Debate whether this function is responsible for padding
        """

        tokens_embd, *kwargs_out = super().prepare_inputs_prefill(
            pt_tokens, start_pos, page_table, chunk_page_table, **kwargs
        )

        if kwargs.get("pixel_values") is not None:
            vision_output = self.compute_vision_token(kwargs["pixel_values"])

            # TODO: Move tokens merging to device

            tokens_embd = ttnn.to_torch(tokens_embd)
            comp_vision_output = ttnn.to_torch(ttnn.from_device(vision_output))
            comp_vision_output = torch.nn.functional.pad(
                comp_vision_output, (0, 0, 0, tokens_embd.shape[1] - comp_vision_output.shape[1]), "constant", 0
            )

            input_ids = torch.nn.functional.pad(
                pt_tokens, (0, tokens_embd.shape[1] - pt_tokens.shape[1]), "constant", 0
            )
            image_features = comp_vision_output.squeeze(0)
            special_image_mask = (input_ids == self.args.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(tokens_embd)
            image_features = image_features.to(tokens_embd.device, tokens_embd.dtype)
            tokens_embd = tokens_embd.masked_scatter(special_image_mask, image_features)
            tokens_embd = ttnn.from_torch(
                tokens_embd,
                dtype=ttnn.bfloat16,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        return tokens_embd, *kwargs_out

    def compute_vision_token(self, pixel_values):
        vision_tokens = self.vision_encoder(pixel_values)[0, :, :, :]
        vision_output = self.mmp(vision_tokens)
        return vision_output
