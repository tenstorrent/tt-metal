# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
SwiGLU MLP implementation for Qwen3-TTS.
"""
import os

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_tts.tt.dram_sharded_matmul import (
    build_dram_sharded_weight,
    dram_sharded_program_config,
    find_grid_k_n,
    width_sharded_l1_memcfg,
)
from models.demos.qwen3_tts.tt.linear_1d_program_config import make_linear_1d_program_config

_DRAM_SHARD_DOWN = os.environ.get("QWEN3_TTS_MLP_DRAM_SHARD_DOWN", "0") == "1"
_DRAM_SHARD_GATE_UP = os.environ.get("QWEN3_TTS_MLP_DRAM_SHARD_GATE_UP", "0") == "1"
_DECODE_SHARDED = os.environ.get("QWEN3_TTS_DECODE_SHARDED", "0") == "1"
_PREFILL_OPTI = os.environ.get("QWEN3_TTS_PREFILL_OPTI", "0") == "1"
_PREFILL_SEQ = 128  # ISL target for prefill optimization


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

        # Optional decode-only DRAM-sharded MLP projections (env-gated).
        self.gate_proj_dram_sharded = None
        self.up_proj_dram_sharded = None
        self._decode_gate_up_dramshard_progcfg = None
        self._decode_gate_up_in0_memcfg = None
        self._decode_gate_up_out_memcfg = None
        self._decode_gate_up_n_padded = None
        self.down_proj_dram_sharded = None
        self._decode_down_dramshard_progcfg = None
        self._decode_down_in0_memcfg = None
        self._decode_down_out_memcfg = None
        self._decode_down_n_padded = None

        if _DRAM_SHARD_GATE_UP:
            # gate_proj and up_proj share K=hidden, N=intermediate.
            gate_w_kn = state_dict[f"{layer_prefix}.mlp.gate_proj.weight"].transpose(-2, -1).contiguous()
            up_w_kn = state_dict[f"{layer_prefix}.mlp.up_proj.weight"].transpose(-2, -1).contiguous()
            self.gate_proj_dram_sharded, k_gu, n_padded_gu = build_dram_sharded_weight(
                gate_w_kn, device, dtype=weight_dtype
            )
            self.up_proj_dram_sharded, _, _ = build_dram_sharded_weight(up_w_kn, device, dtype=weight_dtype)
            self._decode_gate_up_n_padded = n_padded_gu
            k_tiles = k_gu // 32
            n_tiles = n_padded_gu // 32
            rows, cols = find_grid_k_n(k_tiles, n_tiles)
            num_cores = rows * cols
            self._decode_gate_up_dramshard_progcfg = dram_sharded_program_config(
                m=32, k=k_gu, n=n_padded_gu, num_cores=num_cores
            )
            self._decode_gate_up_in0_memcfg = width_sharded_l1_memcfg(
                m_tiles=1, k_tiles=k_tiles, num_cores_x=cols, num_cores_y=rows
            )
            self._decode_gate_up_out_memcfg = width_sharded_l1_memcfg(
                m_tiles=1, k_tiles=n_tiles, num_cores_x=cols, num_cores_y=rows
            )
        if _DRAM_SHARD_DOWN:
            down_w_torch = state_dict[f"{layer_prefix}.mlp.down_proj.weight"]  # [hidden, intermediate]
            # transpose to [intermediate, hidden] = [K, N] for x @ W
            down_w_kn = down_w_torch.transpose(-2, -1).contiguous()
            self.down_proj_dram_sharded, k_down, n_padded_down = build_dram_sharded_weight(
                down_w_kn, device, dtype=weight_dtype
            )
            self._decode_down_n_padded = n_padded_down
            k_tiles = k_down // 32
            n_tiles = n_padded_down // 32
            rows, cols = find_grid_k_n(k_tiles, n_tiles)
            num_cores = rows * cols
            self._decode_down_dramshard_progcfg = dram_sharded_program_config(
                m=32, k=k_down, n=n_padded_down, num_cores=num_cores
            )
            self._decode_down_in0_memcfg = width_sharded_l1_memcfg(
                m_tiles=1, k_tiles=k_tiles, num_cores_x=cols, num_cores_y=rows
            )
            # Output: [1,1,32, n_padded_down] width-sharded across the same grid.
            self._decode_down_out_memcfg = width_sharded_l1_memcfg(
                m_tiles=1, k_tiles=n_tiles, num_cores_x=cols, num_cores_y=rows
            )

        # NOTE: DRAM-sharded matmul kernel only supports M=tile_height (32, decode-only).
        # For prefill at M=128, the kernel TT_FATALs ("currently only support in0 tensor
        # height of tile height"). Prefill matmuls keep using the existing 1D-mcast path;
        # the wins for prefill come from sharded layernorms and sharded NLP head ops.
        self._prefill_gate_up_dramshard_progcfg = None
        self._prefill_gate_up_in0_memcfg = None
        self._prefill_gate_up_out_memcfg = None
        self._prefill_down_dramshard_progcfg = None
        self._prefill_down_in0_memcfg = None
        self._prefill_down_out_memcfg = None

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

        # Gate / Up projections — pick decode (M=1) or prefill (M=ISL) configs.
        is_prefill_opti = (
            mode == "prefill" and seq_len == _PREFILL_SEQ and self._prefill_gate_up_dramshard_progcfg is not None
        )
        is_decode_opti = (
            (mode == "decode" or seq_len == 1) and self.gate_proj_dram_sharded is not None and seq_len < 1024
        )
        use_dram_shard_gate_up = is_prefill_opti or is_decode_opti
        gu_progcfg = (
            self._prefill_gate_up_dramshard_progcfg if is_prefill_opti else self._decode_gate_up_dramshard_progcfg
        )
        gu_in0_memcfg = self._prefill_gate_up_in0_memcfg if is_prefill_opti else self._decode_gate_up_in0_memcfg
        gu_out_memcfg = self._prefill_gate_up_out_memcfg if is_prefill_opti else self._decode_gate_up_out_memcfg

        is_prefill_opti_d = (
            mode == "prefill" and seq_len == _PREFILL_SEQ and self._prefill_down_dramshard_progcfg is not None
        )
        is_decode_opti_d = (
            (mode == "decode" or seq_len == 1) and self.down_proj_dram_sharded is not None and seq_len < 1024
        )
        use_dram_shard_down = is_prefill_opti_d or is_decode_opti_d
        d_progcfg = self._prefill_down_dramshard_progcfg if is_prefill_opti_d else self._decode_down_dramshard_progcfg
        d_in0_memcfg = self._prefill_down_in0_memcfg if is_prefill_opti_d else self._decode_down_in0_memcfg
        d_out_memcfg = self._prefill_down_out_memcfg if is_prefill_opti_d else self._decode_down_out_memcfg

        sharded_chain = use_dram_shard_gate_up and use_dram_shard_down

        if use_dram_shard_gate_up:
            # Width-shard x once and reuse for both gate and up matmuls.
            if x.memory_config() == gu_in0_memcfg:
                x_sharded = x
                _own_x_sharded = False
            else:
                x_sharded = ttnn.to_memory_config(x, gu_in0_memcfg)
                _own_x_sharded = True
            gate_sharded = ttnn.linear(
                x_sharded,
                self.gate_proj_dram_sharded,
                compute_kernel_config=self.compute_kernel_config,
                program_config=gu_progcfg,
                memory_config=gu_out_memcfg,
            )
            up_sharded = ttnn.linear(
                x_sharded,
                self.up_proj_dram_sharded,
                compute_kernel_config=self.compute_kernel_config,
                program_config=gu_progcfg,
                memory_config=gu_out_memcfg,
            )
            if _own_x_sharded:
                ttnn.deallocate(x_sharded)
        else:
            gate_sharded = ttnn.linear(
                x,
                self.gate_proj,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=mem_cfg,
                program_config=gate_up_progcfg,
            )
            up_sharded = ttnn.linear(
                x,
                self.up_proj,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=mem_cfg,
                program_config=gate_up_progcfg,
            )

        if sharded_chain:
            # gate_proj output sharding == down_proj input sharding (same 64-core
            # width-shard over K=intermediate). Mul preserves the layout; feed straight
            # into down without a round-trip through L1_INTERLEAVED.
            hidden_sharded = ttnn.mul(
                gate_sharded,
                up_sharded,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                memory_config=gu_out_memcfg,
            )
            ttnn.deallocate(gate_sharded)
            ttnn.deallocate(up_sharded)
            out_sharded = ttnn.linear(
                hidden_sharded,
                self.down_proj_dram_sharded,
                compute_kernel_config=self.compute_kernel_config,
                program_config=d_progcfg,
                memory_config=d_out_memcfg,
            )
            ttnn.deallocate(hidden_sharded)
            if _DECODE_SHARDED and mode == "decode" and self._decode_down_n_padded == self.hidden_size:
                # Return the width-sharded matmul output directly so decoder_layer can
                # do a sharded residual add, eliminating the S→I (1.5 µs) at this seam.
                output = out_sharded
            else:
                output_padded = ttnn.to_memory_config(out_sharded, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(out_sharded)
                if self._decode_down_n_padded != self.hidden_size:
                    output = ttnn.slice(
                        output_padded,
                        [0, 0, 0, 0],
                        [output_padded.shape[0], output_padded.shape[1], output_padded.shape[2], self.hidden_size],
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )
                    ttnn.deallocate(output_padded)
                else:
                    output = output_padded
        else:
            # Fallback: gate/up may be sharded or interleaved; mul + down go through L1_INTERLEAVED.
            if use_dram_shard_gate_up:
                gate_out = ttnn.to_memory_config(gate_sharded, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(gate_sharded)
                up_out = ttnn.to_memory_config(up_sharded, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(up_sharded)
            else:
                gate_out = gate_sharded
                up_out = up_sharded
            hidden = ttnn.mul(
                gate_out,
                up_out,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                memory_config=mem_cfg,
            )
            ttnn.deallocate(gate_out)
            ttnn.deallocate(up_out)
            if use_dram_shard_down:
                hidden_sharded = ttnn.to_memory_config(hidden, d_in0_memcfg)
                ttnn.deallocate(hidden)
                out_sharded = ttnn.linear(
                    hidden_sharded,
                    self.down_proj_dram_sharded,
                    compute_kernel_config=self.compute_kernel_config,
                    program_config=d_progcfg,
                    memory_config=d_out_memcfg,
                )
                ttnn.deallocate(hidden_sharded)
                output_padded = ttnn.to_memory_config(out_sharded, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(out_sharded)
                if self._decode_down_n_padded != self.hidden_size:
                    output = ttnn.slice(
                        output_padded,
                        [0, 0, 0, 0],
                        [output_padded.shape[0], output_padded.shape[1], output_padded.shape[2], self.hidden_size],
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )
                    ttnn.deallocate(output_padded)
                else:
                    output = output_padded
            else:
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
