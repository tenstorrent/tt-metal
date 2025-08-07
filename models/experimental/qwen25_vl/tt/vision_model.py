"""
This is the end-to-end architecture of the Qwen-VL 2.5 vision model.

It brings together all components—patch embedding, vision blocks, rotary embeddings,
and patch merger for visual input processing.
"""

import ttnn
from tqdm import tqdm
from models.common.lightweightmodule import LightweightModule
from models.experimental.qwen25_vl.tt.vision_block import TtQwen2_5_VLVisionBlock
from models.experimental.qwen25_vl.tt.patch_embed import TTQwen2_5_VisionPatchEmbed
from models.experimental.qwen25_vl.tt.rope import TTQwen2_5_VisionRotaryEmbedding
from models.experimental.qwen25_vl.tt.patch_merger import TTQwen2_5_VLPatchMerger

import torch
import torch.nn.functional as F


class TtQwen2_5_VisionTransformerPretrainedModel(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        model_args,
        layers,
        block_key="",
        gated=False,
    ):
        self.spatial_merge_size = model_args.spatial_merge_size
        self.patch_size = model_args.vision_patch_size
        self.fullatt_block_indexes = model_args.fullatt_block_indexes
        self.window_size = model_args.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.mesh_device = mesh_device
        hidden_size = model_args.vision_dim
        n_heads = model_args.vision_attn_n_heads
        out_hidden_size = model_args.out_hidden_size
        temporal_patch_size = model_args.temporal_patch_size

        self.patch_embed = TTQwen2_5_VisionPatchEmbed(
            device=mesh_device,
            patch_size=self.patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=3,
            embed_dim=hidden_size,
            state_dict=state_dict,
            weight_key="patch_embed.",
            layer_num=None,
            state_dict_prefix=state_dict_prefix,
            weight_cache_path=None,
            weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            weight_dtype=ttnn.bfloat16,
        )

        head_dim = hidden_size // n_heads

        self.rotary_pos_emb = TTQwen2_5_VisionRotaryEmbedding(
            device=mesh_device,
            dim=head_dim // 2,
            theta=10000.0,
        )

        self.blocks = [
            TtQwen2_5_VLVisionBlock(
                mesh_device=mesh_device,
                state_dict=state_dict,
                state_dict_prefix=f"{state_dict_prefix}blocks.{i}.",
                weight_cache_path=weight_cache_path,
                dtype=dtype,
                model_args=model_args,
            )
            for i in tqdm(range(layers), desc=f"Loading vision transformer blocks")
        ]

        self.merger = TTQwen2_5_VLPatchMerger(
            device=mesh_device,
            dim=5120,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_key="merger.",
            layer_num=None,
            weight_cache_path=None,
            weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            weight_dtype=ttnn.bfloat16,
            is_distributed=None,
            eps=1e-06,
            dims=out_hidden_size,
            context_dim=hidden_size,
            spatial_merge_size=self.spatial_merge_size,
        )

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb_full = ttnn.to_torch(rotary_pos_emb_full, device=self.mesh_device)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(self, hidden_states, grid_thw):
        hidden_states = self.patch_embed(hidden_states)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            # device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len = hidden_states.shape[-2]

        hidden_states = ttnn.reshape(hidden_states, [seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1])
        # hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        tt_index = ttnn.from_torch(
            window_index.view(-1, 1, 1).expand(-1, hidden_states.shape[-2], hidden_states.shape[-1]).permute(1, 2, 0),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            # memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        hidden_states = ttnn.gather(ttnn.permute(hidden_states, (1, 2, 0)), dim=-1, index=tt_index)
        hidden_states = ttnn.permute(hidden_states, (2, 0, 1))

        hidden_states = ttnn.reshape(hidden_states, [seq_len, -1])
        # hidden_states = hidden_states.reshape(seq_len, -1)

        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)

        cos_tensor = ttnn.from_torch(emb.cos(), device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        sin_tensor = ttnn.from_torch(emb.sin(), device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        position_embeddings = (cos_tensor, sin_tensor)

        ttnn.deallocate(tt_index)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

        ttnn.deallocate(cos_tensor)
        ttnn.deallocate(sin_tensor)
        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)

        tt_reverse_indices = ttnn.from_torch(
            reverse_indices.view(-1, 1).expand(-1, hidden_states.shape[-1]).transpose(0, 1),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
        )
        hidden_states = ttnn.gather(ttnn.permute(hidden_states, (1, 0)), dim=-1, index=tt_reverse_indices)
        hidden_states = ttnn.permute(hidden_states, (1, 0))

        return hidden_states
