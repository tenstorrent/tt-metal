import ttnn
from models.common.lightweightmodule import LightweightModule
import torch
from models.experimental.mochi.common import matmul_config


class TtFeedForward(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        in_features: int,
        hidden_size: int,
        multiple_of: int,
        ffn_dim_multiplier: float = None,
        state_dict_prefix=None,
    ):
        super().__init__()
        assert len(mesh_device.get_devices()) == 1, "Only single-device inference is supported for feedforward layers"

        # Calculate hidden size according to Mochi specs
        hidden_size = int(2 * hidden_size / 3)
        if ffn_dim_multiplier is not None:
            hidden_size = int(ffn_dim_multiplier * hidden_size)
        hidden_size = multiple_of * ((hidden_size + multiple_of - 1) // multiple_of)

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        torch_weight = lambda name: torch.transpose(self.state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)

        cache_name = lambda name: weight_cache_path / (state_dict_prefix + f".{name}")

        # TODO: Handle swizzling data when fracturing w1 on columns
        as_tensor = lambda name, type, dim: ttnn.as_tensor(
            torch_weight(name),
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=dim),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(name),
        )
        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        # Sharded weights
        self.w1 = as_tensor("w1", ttnn.bfloat16, dim=-1)
        self.w2 = as_tensor("w2", ttnn.bfloat16, dim=-2)

        # Vision MLP has longer seqlen than text MLP
        self.MM_SEQ_LEN = 512 if "mlp_x" in state_dict_prefix else 256

        self.w1_config = matmul_config(self.MM_SEQ_LEN, in_features, 2 * hidden_size, (8, 8))
        self.w2_config = matmul_config(self.MM_SEQ_LEN, hidden_size, in_features, (8, 8))

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        B = x.shape[1]
        S = x.shape[2]
        D = x.shape[3]
        assert S % self.MM_SEQ_LEN == 0, "Sequence length must be divisible by {}, got {}".format(self.MM_SEQ_LEN, S)
        assert B == 1, "Batch size must be 1, got {}".format(B)

        x = ttnn.reshape(x, (B, S // self.MM_SEQ_LEN, self.MM_SEQ_LEN, D))

        # W1 computation (includes both x and gate paths)
        w1_out = ttnn.linear(
            x,
            self.w1,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat16,
            memory_config=x.memory_config(),
            program_config=self.w1_config,
        )

        w1_out = ttnn.reshape(w1_out, (1, B, S, w1_out.shape[-1]))
        # Split into x and gate paths
        x_path, gate_path = ttnn.split(w1_out, 2, dim=-1)

        # Apply SiLU and multiply with gate
        w2_in = ttnn.multiply(
            x_path,
            gate_path,
            input_tensor_a_activation=ttnn.UnaryOpType.SILU,
            dtype=ttnn.bfloat16,
            memory_config=x_path.memory_config(),
        )

        w2_in = ttnn.reshape(w2_in, (B, S // self.MM_SEQ_LEN, self.MM_SEQ_LEN, w2_in.shape[-1]))

        # W2 computation
        result = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat16,
            memory_config=w2_in.memory_config(),
            program_config=self.w2_config,
        )

        # # All reduce for multi-chip setups
        # if self.args.is_multichip:
        #     result = ttnn.reduce_scatter(
        #         w2_out,
        #         dim=3,
        #         math_op=ttnn.ReduceType.Sum,
        #         num_links=1,
        #         memory_config=result.memory_config(),
        #     )

        result = ttnn.reshape(result, (1, B, S, D))
        return result
