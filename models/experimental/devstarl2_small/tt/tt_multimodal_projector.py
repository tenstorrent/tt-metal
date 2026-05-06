from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.devstarl2_small.tt.tt_patchmerger import TTMistral3PatchMerger
from models.experimental.devstarl2_small.tt.tt_rmsnorm import RMSNorm


class TTMistral3MultiModalProjector(LightweightModule):
    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        state_dict_prefix="multi_modal_projector.",
        weight_cache_path=None,
        dtype=ttnn.bfloat16,
        eps=1e-5,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args
        self.state_dict_prefix = state_dict_prefix
        self.dtype = dtype

        norm_dim = state_dict[f"{state_dict_prefix}norm.weight"].numel()
        self.norm = RMSNorm(
            device=mesh_device,
            dim=norm_dim,
            state_dict=state_dict,
            weight_key=f"{state_dict_prefix}norm",
            weight_cache_path=weight_cache_path,
            weight_dtype=dtype,
            eps=eps,
        )
        self.patch_merger = TTMistral3PatchMerger(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}patch_merger.",
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )

        def as_linear_weight(name):
            return ttnn.as_tensor(
                torch.transpose(state_dict[f"{state_dict_prefix}{name}.weight"], -2, -1),
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=None
                if weight_cache_path is None
                else weight_cache_path / f"{state_dict_prefix}{name}.weight",
            )

        self.linear_1_weight = as_linear_weight("linear_1")
        self.linear_2_weight = as_linear_weight("linear_2")

    def forward(self, image_features: ttnn.Tensor, image_sizes) -> ttnn.Tensor:
        # tt_rmsnorm expects rank-4 tensors; projector input is [tokens, hidden].
        x = ttnn.reshape(image_features, (1, 1, image_features.shape[0], image_features.shape[1]))
        x = self.norm(x, mode="prefill")
        x = ttnn.reshape(x, image_features.shape)
        x = self.patch_merger(x, image_sizes)
        x = ttnn.linear(x, self.linear_1_weight, dtype=self.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.gelu(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.linear(x, self.linear_2_weight, dtype=self.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x


__all__ = ["TTMistral3MultiModalProjector"]
