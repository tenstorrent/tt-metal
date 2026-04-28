# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
SwiGLU MLP implementation for Qwen3-TTS.
"""
import math

import ttnn
from models.common.lightweightmodule import LightweightModule


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
            in_features = int(weight_torch.shape[1])
            out_features = int(weight_torch.shape[0])

            # DRAM (not L1): transpose/reshape compile matmul-scale programs with large circular
            # buffers; L1 staging here overlaps allocator space with those CBs (Metal validate
            # "clash with L1 buffers" during model init). Same pattern as Attention weight prep.
            weight_tt = ttnn.from_torch(
                weight_torch,
                device=device,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=_dram,
                mesh_mapper=_mesh_mapper,
            )
            weight_tx = ttnn.transpose(weight_tt, -2, -1, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(weight_tt)
            weight_4d = ttnn.reshape(weight_tx, [1, 1, in_features, out_features], memory_config=ttnn.L1_MEMORY_CONFIG)

            # Read reshape output before freeing transpose input; reshape may alias weight_tx storage.
            weight_host = ttnn.to_torch(weight_4d).contiguous()

            ttnn.deallocate(weight_4d)
            ttnn.deallocate(weight_tx)

            #

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
        # Targeted Stage-1 tuning:
        # dominant short-seq matmuls in decode/prefill32 are
        #   - hidden(2048) -> intermediate(6144)  [gate/up]
        #   - intermediate(6144) -> hidden(2048)  [down]
        # Keep this narrow (seq <= 32) to avoid broad core forcing.
        self._short_seq_gate_up_progcfg = self._make_linear_1d_program_config(
            m=self.short_seq_limit,
            k=hidden_size,
            n=intermediate_size,
            grid_x=grid.x,
            grid_y=grid.y,
            fp32_dest_acc_en=self.compute_kernel_config.fp32_dest_acc_en,
        )
        self._short_seq_down_progcfg = self._make_linear_1d_program_config(
            m=self.short_seq_limit,
            k=intermediate_size,
            n=hidden_size,
            grid_x=grid.x,
            grid_y=grid.y,
            fp32_dest_acc_en=self.compute_kernel_config.fp32_dest_acc_en,
        )

    @staticmethod
    def _make_linear_1d_program_config(
        m: int, k: int, n: int, grid_x: int, grid_y: int, fp32_dest_acc_en: bool
    ) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        tile_h = 32
        tile_w = 32
        num_cores = max(1, grid_x * grid_y)

        per_core_m = max(1, m // tile_h)
        per_core_k = max(1, math.ceil((k / tile_w) / num_cores))
        per_core_n = max(1, math.ceil((n / tile_w) / num_cores))

        subblock_limit = 4 if fp32_dest_acc_en else 8
        out_subblock_w = max(i for i in range(1, subblock_limit + 1) if per_core_n % i == 0)
        out_subblock_h = max(
            i for i in range(1, subblock_limit + 1) if per_core_m % i == 0 and i * out_subblock_w <= subblock_limit
        )

        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=per_core_k,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
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
        gate_up_progcfg = self._short_seq_gate_up_progcfg if seq_len <= self.short_seq_limit else None
        down_progcfg = self._short_seq_down_progcfg if seq_len <= self.short_seq_limit else None

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
