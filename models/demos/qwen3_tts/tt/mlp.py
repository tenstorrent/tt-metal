# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
SwiGLU MLP implementation for Qwen3-TTS.
"""
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_tts.tt.linear_1d_program_config import make_linear_1d_program_config


class MLP(LightweightModule):
    """
    SwiGLU MLP for Qwen3-TTS.

    Architecture: down_proj(silu(gate_proj(x)) * up_proj(x))

    This is a simplified implementation for single device (N150/N300).
    """

    def __init__(
        self,
        device,
        hidden_size: int,
        intermediate_size: int,
        state_dict: dict,
        layer_prefix: str,
        weight_dtype=ttnn.bfloat16,
        weight_cache_path=None,
    ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        is_mesh_device = device.__class__.__name__ == "MeshDevice"

        def get_cache_name(name):
            if weight_cache_path is None:
                return None
            return weight_cache_path / f"{layer_prefix}_{name}".replace(".", "_")

        # Load and transpose weights (for matmul: x @ W.T -> x @ W where W is transposed)
        # gate_proj_weight = torch.transpose(state_dict[f"{layer_prefix}.mlp.gate_proj.weight"], -2, -1)
        # up_proj_weight = torch.transpose(state_dict[f"{layer_prefix}.mlp.up_proj.weight"], -2, -1)
        # down_proj_weight = torch.transpose(state_dict[f"{layer_prefix}.mlp.down_proj.weight"], -2, -1)
        #
        # Make weights 4D for TTNN [1, 1, in_features, out_features]
        # gate_proj_weight = gate_proj_weight.unsqueeze(0).unsqueeze(0)
        # up_proj_weight = up_proj_weight.unsqueeze(0).unsqueeze(0)
        # down_proj_weight = down_proj_weight.unsqueeze(0).unsqueeze(0)
        #
        # self.gate_proj = ttnn.as_tensor(
        #     gate_proj_weight,
        #     device=device,
        #     dtype=weight_dtype,
        #     layout=ttnn.TILE_LAYOUT,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        #     cache_file_name=get_cache_name("gate_proj"),
        #     mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        # )
        #
        # self.up_proj = ttnn.as_tensor(
        #     up_proj_weight,
        #     device=device,
        #     dtype=weight_dtype,
        #     layout=ttnn.TILE_LAYOUT,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        #     cache_file_name=get_cache_name("up_proj"),
        #     mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        # )
        #
        # self.down_proj = ttnn.as_tensor(
        #     down_proj_weight,
        #     device=device,
        #     dtype=weight_dtype,
        #     layout=ttnn.TILE_LAYOUT,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        #     cache_file_name=get_cache_name("down_proj"),
        #     mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        # )

        _mesh_mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None
        _dram = ttnn.DRAM_MEMORY_CONFIG

        def _build_proj_weight(weight_key: str, cache_name: str):
            weight_torch = state_dict[weight_key]
            # Keep layout conversion on host once at load time:
            # [out, in] -> [1, 1, in, out] for ttnn.linear.
            # This avoids device-side transpose/reshape churn during model init.
            weight_host = weight_torch.transpose(-2, -1).unsqueeze(0).unsqueeze(0).contiguous()

            cache_file = get_cache_name(cache_name)
            if cache_file is not None:
                return ttnn.as_tensor(
                    weight_host,
                    device=device,
                    dtype=weight_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=_dram,
                    cache_file_name=cache_file,
                    mesh_mapper=_mesh_mapper,
                )
            return ttnn.from_torch(
                weight_host,
                device=device,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=_dram,
                mesh_mapper=_mesh_mapper,
            )

        self.gate_proj = _build_proj_weight(f"{layer_prefix}.mlp.gate_proj.weight", "gate_proj")
        self.up_proj = _build_proj_weight(f"{layer_prefix}.mlp.up_proj.weight", "up_proj")
        self.down_proj = _build_proj_weight(f"{layer_prefix}.mlp.down_proj.weight", "down_proj")

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        grid = device.compute_with_storage_grid_size()
        self.short_seq_limit = 32
        _fp32 = self.compute_kernel_config.fp32_dest_acc_en
        # Targeted Stage-1 tuning:
        # dominant short-seq matmuls in decode/prefill32 are
        #   - hidden(2048) -> intermediate(6144)  [gate/up]
        #   - intermediate(6144) -> hidden(2048)  [down]
        # Decode (seq_len == 1): use m=1 so per_core_M matches single-token rows.
        # Short prefill (2 <= seq <= 32): keep m=32 (tile-era baseline) for stable PCC.
        self._decode_gate_up_progcfg = make_linear_1d_program_config(
            m=1,
            k=hidden_size,
            n=intermediate_size,
            grid_x=grid.x,
            grid_y=grid.y,
            fp32_dest_acc_en=_fp32,
        )
        self._decode_down_progcfg = make_linear_1d_program_config(
            m=1,
            k=intermediate_size,
            n=hidden_size,
            grid_x=grid.x,
            grid_y=grid.y,
            fp32_dest_acc_en=_fp32,
        )
        self._short_seq_gate_up_progcfg = make_linear_1d_program_config(
            m=self.short_seq_limit,
            k=hidden_size,
            n=intermediate_size,
            grid_x=grid.x,
            grid_y=grid.y,
            fp32_dest_acc_en=_fp32,
        )
        self._short_seq_down_progcfg = make_linear_1d_program_config(
            m=self.short_seq_limit,
            k=intermediate_size,
            n=hidden_size,
            grid_x=grid.x,
            grid_y=grid.y,
            fp32_dest_acc_en=_fp32,
        )

    def forward(self, x: ttnn.Tensor, mode: str = "prefill") -> ttnn.Tensor:
        """
        Apply SwiGLU MLP.

        Args:
            x: Input tensor of shape [batch, 1, seq_len, hidden_size]
            mode: "prefill" or "decode" - decode uses L1 memory for speed

        Returns:
            Output tensor of same shape
        """
        seq_len = x.shape[-2]

        # Stage-1 memory-layout tuning for matmuls:
        # - Decode is always L1.
        # - Prefill with small/medium sequence lengths also benefits from L1.
        # - Fall back to DRAM for longer prefill sequences to avoid L1 pressure.
        if mode == "decode" or seq_len <= 256:
            mem_cfg = ttnn.L1_MEMORY_CONFIG
        else:
            mem_cfg = ttnn.DRAM_MEMORY_CONFIG
        if mode == "decode" or seq_len == 1:
            gate_up_progcfg = self._decode_gate_up_progcfg
            down_progcfg = self._decode_down_progcfg
        elif seq_len <= self.short_seq_limit:
            gate_up_progcfg = self._short_seq_gate_up_progcfg
            down_progcfg = self._short_seq_down_progcfg
        else:
            gate_up_progcfg = down_progcfg = None

        # Reshape for large sequences to fit on device
        if seq_len >= 1024:
            x = ttnn.reshape(x, [1, seq_len // 1024, 1024, -1])

        # Gate projection with SiLU activation
        gate_out = ttnn.linear(
            x,
            self.gate_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=mem_cfg,
            program_config=gate_up_progcfg,
        )

        # Up projection
        up_out = ttnn.linear(
            x,
            self.up_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=mem_cfg,
            program_config=gate_up_progcfg,
        )

        # SwiGLU: silu(gate) * up
        hidden = ttnn.mul(
            gate_out,
            up_out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=mem_cfg,
        )

        ttnn.deallocate(gate_out)
        ttnn.deallocate(up_out)

        # Down projection
        output = ttnn.linear(
            hidden,
            self.down_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=mem_cfg,
            program_config=down_progcfg,
        )

        ttnn.deallocate(hidden)

        # Reshape back if needed
        if seq_len >= 1024:
            output = ttnn.reshape(output, [1, 1, seq_len, -1])

        return output
