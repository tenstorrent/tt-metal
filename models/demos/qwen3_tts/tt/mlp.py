# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
SwiGLU MLP implementation for Qwen3-TTS.
"""


import math

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_tts.tt.attention import build_multicast_linear_program_config


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
            packer_l1_acc=False,
        )

        cg = device.compute_with_storage_grid_size()
        self._mlp_grid_x = int(cg.x)
        self._mlp_grid_y = int(cg.y)

        TILE = ttnn.TILE_SIZE
        min_total_m_rows = max(
            (intermediate_size + 7) // 8,
            (hidden_size + 7) // 8,
        )
        need_top = ((min_total_m_rows + TILE - 1) // TILE) * TILE
        bank_r = max(need_top - TILE, 0)
        bank_r = ((bank_r + TILE - 1) // TILE) * TILE
        if bank_r > 0:
            # Last dim must match gate/up linear input width (tile-padded), not config hidden_size —
            # otherwise pad slice ends conflict with activations / matmul K mismatch.
            _pad_w = int(self.gate_proj.shape[-2])
            self._mlp_pad_bank = ttnn.zeros(
                [1, 1, bank_r, _pad_w],
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=_dram,
            )
            self._mlp_pad_bank_rows = bank_r
        else:
            self._mlp_pad_bank = None
            self._mlp_pad_bank_rows = 0

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

        # Use L1 for decode mode (small tensors fit in L1), DRAM for prefill
        mem_cfg = ttnn.L1_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG

        # Reshape for large sequences to fit on device
        if seq_len >= 1024:
            x = ttnn.reshape(x, [1, seq_len // 1024, 1024, -1])

        ps = list(x.padded_shape)
        rank = len(ps)
        batch_prod = math.prod([int(ps[i]) for i in range(rank - 2)])
        cur_m = int(ps[-2])
        TILE = ttnn.TILE_SIZE
        min_total_m_rows = max(
            (self.intermediate_size + 7) // 8,
            (self.hidden_size + 7) // 8,
        )
        need_m_dim = (min_total_m_rows + batch_prod - 1) // batch_prod
        need_m_dim = ((need_m_dim + TILE - 1) // TILE) * TILE
        pad_seq = need_m_dim - cur_m
        shape_ok = rank == 4 and int(ps[0]) == 1 and int(ps[1]) == 1
        padded_for_multicast = (
            shape_ok and self._mlp_pad_bank is not None and pad_seq > 0 and pad_seq <= self._mlp_pad_bank_rows
        )
        spatial_m_before = int(x.shape[-2])
        h_last = int(ps[-1])
        hidden_width_logical = int(x.shape[-1])
        if padded_for_multicast:
            # Never deallocate inputs before concat: L1→DRAM copy may still depend on the L1 buffer
            # until the op completes; deallocating early yields "Tensor is not allocated" on concat.
            # If x is already DRAM, to_memory_config may alias x — also do not free before concat.
            # After concat: free L1 source and the DRAM tile used as concat input when we created it
            # from L1; do not free the original DRAM x when it was passed through (may alias concat).
            moved_from_l1 = mem_cfg != ttnn.DRAM_MEMORY_CONFIG
            x_pre = x
            if moved_from_l1:
                x_dram = ttnn.to_memory_config(x_pre, ttnn.DRAM_MEMORY_CONFIG)
            else:
                x_dram = x_pre
            pad_part = ttnn.slice(
                self._mlp_pad_bank,
                [0, 0, 0, 0],
                [1, 1, pad_seq, h_last],
            )
            x = ttnn.concat([x_dram, pad_part], dim=rank - 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if moved_from_l1:
                ttnn.deallocate(x_pre)
                ttnn.deallocate(x_dram)

        lin_mem = ttnn.DRAM_MEMORY_CONFIG if padded_for_multicast else mem_cfg

        # Gate projection with SiLU activation
        gate_out = ttnn.linear(
            x,
            self.gate_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=lin_mem,
            program_config=build_multicast_linear_program_config(x, self.gate_proj, self._mlp_grid_x, self._mlp_grid_y),
        )

        # Up projection
        up_out = ttnn.linear(
            x,
            self.up_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=lin_mem,
            program_config=build_multicast_linear_program_config(x, self.up_proj, self._mlp_grid_x, self._mlp_grid_y),
        )

        # SwiGLU: silu(gate) * up
        hidden = ttnn.mul(
            gate_out,
            up_out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=lin_mem,
        )

        ttnn.deallocate(gate_out)
        ttnn.deallocate(up_out)

        # Down projection
        output = ttnn.linear(
            hidden,
            self.down_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=lin_mem,
            program_config=build_multicast_linear_program_config(
                hidden, self.down_proj, self._mlp_grid_x, self._mlp_grid_y
            ),
        )

        ttnn.deallocate(hidden)

        if padded_for_multicast:
            ends = [int(output.shape[i]) for i in range(rank)]
            ends[-2] = min(spatial_m_before, ends[-2])
            ends[-1] = min(hidden_width_logical, ends[-1])
            output = ttnn.slice(output, [0] * rank, ends)

        # Reshape back if needed
        if seq_len >= 1024:
            output = ttnn.reshape(output, [1, 1, seq_len, -1])

        return output
