# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.mistral_24b.tt.vision_conv2d import TtMistralConv2dPatch
from models.common.rmsnorm import RMSNorm as RMSNorm
from models.tt_transformers.tt.distributed_norm import DistributedNorm

from models.tt_transformers.tt.common import position_ids_in_meshgrid_tt, generate_block_attention_mask_tt
from models.experimental.mistral_24b.tt.vision_rope import VisionRotarySetup as RotarySetup

from models.experimental.mistral_24b.tt.vision_pixtral_transformer import TtPixtralTransformer
from models.utility_functions import comp_allclose, comp_pcc
from loguru import logger
import torch


def position_ids_in_meshgrid(patch_embeds_list, max_width):
    positions = []
    for patch in patch_embeds_list:
        height, width = patch.shape[-2:]
        mesh = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        h_grid, v_grid = torch.stack(mesh, dim=-1).reshape(-1, 2).chunk(2, -1)
        ids = h_grid * max_width + v_grid
        positions.append(ids[:, 0])
    return torch.cat(positions)


def generate_block_attention_mask(patch_embeds_list, tensor):
    dtype = tensor.dtype
    device = tensor.device
    seq_len = tensor.shape[1]
    d_min = torch.finfo(dtype).min
    causal_mask = torch.full((seq_len, seq_len), fill_value=d_min, dtype=dtype, device=device)

    block_end_idx = torch.tensor(patch_embeds_list).cumsum(-1)
    block_start_idx = torch.tensor([0] + patch_embeds_list[:-1]).cumsum(-1)
    for start, end in zip(block_start_idx, block_end_idx):
        causal_mask[start:end, start:end] = 0

    causal_mask = causal_mask[None, None, :, :].expand(tensor.shape[0], 1, -1, -1)
    return causal_mask


class MistralVisionTower(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        dtype,
        configuration,
        return_intermediate=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.config = configuration

        self.image_size = configuration.vision_chunk_size
        self.patch_size = configuration.vision_patch_size
        self.width = configuration.vision_dim
        self.layers = configuration.vision_n_layers
        self.heads = configuration.vision_attn_n_heads
        self.vision_head_dim = configuration.vision_head_dim
        self.mlp_ratio = configuration.vision_mlp_ratio
        self.act_layer = configuration.vision_act_layer
        self.in_channels = configuration.vision_in_channels
        self.n_global_layers = configuration.vision_n_global_layers
        self.max_seq_len = configuration.max_seq_len
        self.return_intermediate = return_intermediate
        self.n_layers = configuration.vision_n_layers

        in_channels, out_channels, kernel_size, stride, bias = (
            3,
            configuration.vision_dim,
            configuration.vision_patch_size,
            configuration.vision_patch_size,
            False,
        )

        self.patch_conv = TtMistralConv2dPatch(
            mesh_device=self.mesh_device,
            state_dict=self.state_dict,
            state_dict_prefix=f"{state_dict_prefix}patch_conv.",
            dtype=self.dtype,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )

        layer_norm = RMSNorm(
            device=mesh_device,
            dim=self.width,
            state_dict=self.state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_dtype=dtype,
            weight_key="ln_pre",
            is_distributed=configuration.is_distributed_norm,
        )

        self.ln_pre = DistributedNorm(
            layer_norm,
            configuration,
            TG=configuration.is_galaxy,
        )

        image_size = configuration.vision_image_size
        patch_size = configuration.vision_patch_size
        dim = configuration.vision_head_dim
        num_patches_per_dim = image_size // patch_size
        num_patches = num_patches_per_dim * num_patches_per_dim
        self.num_patches = num_patches

        self.patch_positional_embedding = RotarySetup(
            self.mesh_device,
            1,
            dim,
            image_size,
            patch_size,
            num_patches,
            configuration.vision_rope_theta,
            scale_factor=None,
            orig_context_len=num_patches,
            datatype=dtype,
        )

        self.transformer = TtPixtralTransformer(
            mesh_device=self.mesh_device,
            state_dict=self.state_dict,
            state_dict_prefix=f"{state_dict_prefix}transformer.",
            weight_cache_path=configuration.weight_cache_path(dtype),
            dtype=self.dtype,
            configuration=configuration,
            layers=self.n_layers,
        )

    def forward(self, input_tensor, image_sizes=None, ref_model=None):
        """
        input_tensor shape: (B, C, H, W)
        """
        print("MistralVisionTower forward called with input_tensor shape:", input_tensor.shape)
        ref_patch_conv = ref_model.patch_conv(input_tensor)
        patch_embeds = self.patch_conv(input_tensor)
        patch_embeds = ttnn.transpose(patch_embeds, 1, 2)
        height, width = image_sizes[0]
        patch_embeds = ttnn.reshape(
            patch_embeds,
            [patch_embeds.shape[0], self.width, height // self.patch_size, width // self.patch_size],
        )

        pcc_required = 0.99
        passing, pcc_message = comp_pcc(ref_patch_conv, ttnn.to_torch(patch_embeds), pcc_required)

        logger.info(comp_allclose(ref_patch_conv, ttnn.to_torch(patch_embeds)))
        logger.info(f"========= Stage1 ref_patch_conv PCC: {pcc_message}")
        assert passing, f"========= Stage1 ref_patch_conv PCC below {pcc_required}. {pcc_message}"

        ref_patch_embeds_list = [
            embed[..., : (size[0] // self.patch_size), : (size[1] // self.patch_size)]
            for embed, size in zip(ref_patch_conv, image_sizes)
        ]
        # flatten to a single sequence
        ref_patch_embeds = torch.cat([p.flatten(1).T for p in ref_patch_embeds_list], dim=0).unsqueeze(0)

        patch_embeds_list = [
            ttnn.slice(
                patch_embeds,
                [0, 0, 0, 0],
                [1, self.width, size[0] // self.patch_size, size[1] // self.patch_size],
            )
            for size in image_sizes
        ]

        reshaped_patches = []
        for p in patch_embeds_list:
            p = ttnn.reshape(p, (1, self.width, -1))
            p = ttnn.transpose(p, 1, 2)
            reshaped_patches.append(p)

        patch_embeds = ttnn.concat(reshaped_patches, dim=0)

        passing, pcc_message = comp_pcc(ref_patch_embeds, ttnn.to_torch(patch_embeds), pcc_required)
        logger.info(comp_allclose(ref_patch_embeds, ttnn.to_torch(patch_embeds)))
        logger.info(f"========= Stage2 patch_embeds PCC: {pcc_message}")
        assert passing, f"========= Stage2 patch_embeds PCC below {pcc_required}. {pcc_message}"

        passing, pcc_message = comp_pcc(
            ref_patch_embeds_list[0], ttnn.to_torch(patch_embeds_list[0]).squeeze(0), pcc_required
        )
        logger.info(comp_allclose(ref_patch_embeds_list[0], ttnn.to_torch(patch_embeds_list[0]).squeeze(0)))
        logger.info(f"========= Stage3 Patch_embeds_list PCC: {pcc_message}")
        assert passing, f"========= Stage3 Patch_embeds_list PCC below {pcc_required}. {pcc_message}"

        # ln_pre RMS Norm
        ref_patch_embeds = ref_model.ln_pre(ref_patch_embeds)
        mode = "prefill"  # if self.max_seq_len <= 32 else "prefill"
        patch_embeds = self.ln_pre(patch_embeds, mode=mode)

        passing, pcc_message = comp_pcc(ref_patch_embeds, ttnn.to_torch(patch_embeds), pcc_required)
        logger.info(comp_allclose(ref_patch_embeds, ttnn.to_torch(patch_embeds)))
        logger.info(f"========= Stage4 ln_pre PCC: {pcc_message}")
        assert passing, f"========= Stage4 ln_pre PCC below {pcc_required}. {pcc_message}"

        ref_position_ids = position_ids_in_meshgrid(
            ref_patch_embeds_list,
            max_width=self.config.vision_image_size // self.config.vision_patch_size,
        )

        # # positional embeddings
        position_ids = position_ids_in_meshgrid_tt(
            patch_embeds_list,
            max_width=self.config.vision_image_size // self.config.vision_patch_size,
            device=self.mesh_device,
        )
        passing, pcc_message = comp_pcc(ref_position_ids, ttnn.to_torch(position_ids), pcc_required)
        logger.info(comp_allclose(ref_position_ids, ttnn.to_torch(position_ids)))
        logger.info(f"========= Stage5 position_ids PCC: {pcc_message}")
        assert passing, f"========= Stage5 position_ids PCC below {pcc_required}. {pcc_message}"

        ref_position_embeddings = ref_model.patch_positional_embedding(ref_patch_embeds, ref_position_ids)
        position_embeddings = self.patch_positional_embedding.get_rot_mats(ttnn.to_torch(position_ids))

        passing, pcc_message = comp_pcc(
            ref_position_embeddings[0], ttnn.to_torch(position_embeddings[0]).squeeze(0), pcc_required
        )
        logger.info(comp_allclose(ref_position_embeddings[0], ttnn.to_torch(position_embeddings[0]).squeeze(0)))
        logger.info(f"========= Stage6 position_embeddings PCC: {pcc_message}")
        assert passing, f"========= Stage6 position_embeddings PCC below {pcc_required}. {pcc_message}"

        ref_attention_mask = generate_block_attention_mask(
            [p.shape[-2] * p.shape[-1] for p in ref_patch_embeds_list], ref_patch_embeds
        )

        attention_mask = generate_block_attention_mask_tt(
            [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], patch_embeds, tt_device=self.mesh_device
        )

        passing, pcc_message = comp_pcc(ref_attention_mask, ttnn.to_torch(attention_mask), pcc_required)
        logger.info(comp_allclose(ref_attention_mask, ttnn.to_torch(attention_mask)))
        logger.info(f"========= Stage7 attention_mask PCC: {pcc_message}")
        assert passing, f"========= Stage7 attention_mask PCC below {pcc_required}. {pcc_message}"

        ref_out = ref_model.transformer(
            ref_patch_embeds,
            attention_mask=ref_attention_mask,
            position_embeddings=ref_position_embeddings,
            output_hidden_states=None,
            output_attentions=None,
            return_dict=None,
        )

        patch_embeds = ttnn.unsqueeze(patch_embeds, 0)
        out = self.transformer(patch_embeds, mask=attention_mask, position_embeddings=position_embeddings)
        passing, pcc_message = comp_pcc(ref_out.last_hidden_state, ttnn.to_torch(out).squeeze(0), pcc_required)
        logger.info(comp_allclose(ref_out.last_hidden_state, ttnn.to_torch(out).squeeze(0)))
        logger.info(f"========= Stage8 transformer out PCC: {pcc_message}")

        return out
