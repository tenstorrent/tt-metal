from functools import partial
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.mochi.tt.common import unsqueeze_to_4d, matmul_2d_config


class PatchEmbed(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        weight_cache_path,
        dtype,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        flatten: bool = True,
        bias: bool = True,
        dynamic_img_pad: bool = False,
        state_dict_prefix=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        assert self.patch_size[0] == self.patch_size[1], "Patch size must be square"
        self.flatten = flatten
        self.dynamic_img_pad = dynamic_img_pad
        assert not self.dynamic_img_pad, "Dynamic image padding not supported"

        # Load weights from state dict
        weight_name = f"{state_dict_prefix}.proj.weight" if state_dict_prefix else "proj.weight"
        weight = state_dict[weight_name]
        out_chan, in_chan = weight.shape[:2]
        assert out_chan == embed_dim
        assert in_chan == in_chans
        weight = weight.permute(1, 2, 3, 0).reshape(-1, out_chan)
        self.weight = ttnn.as_tensor(
            unsqueeze_to_4d(weight),
            dtype=dtype,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=weight_cache_path / (weight_name + "flattened"),
        )
        if bias:
            bias_name = f"{state_dict_prefix}.proj.bias" if state_dict_prefix else "proj.bias"
            bias_tensor = state_dict[bias_name]
            assert len(bias_tensor.shape) == 1
            assert bias_tensor.shape[0] == embed_dim
            self.bias = ttnn.as_tensor(
                unsqueeze_to_4d(bias_tensor),
                dtype=dtype,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=weight_cache_path / (bias_name + "flattened"),
            )
        else:
            self.bias = None

        self.mm_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.mm_config = partial(matmul_2d_config, n=embed_dim, grid_size=(8, 8))

    def forward(self, x_1BNI: ttnn.Tensor) -> ttnn.Tensor:
        """
        ttnn TMs aren't working here yet (Issue #17535) so we'll need the
        input to come in shape (1, B, N, I)
        x_1BNI: (1, B, N, I)
        Batch must be 1 and implied
        Input is replicated
        Output is replicated
        """
        assert not self.dynamic_img_pad, "Dynamic image padding not supported"
        if self.dynamic_img_pad:
            pass
            # pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            # pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            # if pad_h > 0 or pad_w > 0:
            #     x_BCTHW = ttnn.pad(x_BCTHW, (0, pad_w, 0, pad_h))
            #     H += pad_h
            #     W += pad_w
        else:
            pass
            # assert H % self.patch_size[0] == 0, f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
            # assert W % self.patch_size[1] == 0, f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."

        N, I = x_1BNI.shape[2], x_1BNI.shape[3]
        x_1BNX = ttnn.linear(
            x_1BNI,
            self.weight,
            bias=self.bias,
            compute_kernel_config=self.mm_compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.mm_config(m=N, k=I),
        )
        return x_1BNX
