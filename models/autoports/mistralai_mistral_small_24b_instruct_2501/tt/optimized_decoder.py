# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-device optimized decoder layer for Mistral Small 24B.

This stage starts from the proven functional layer but owns its optimized
runtime graph.  The initial graph-rewrite candidate packs the two same-input
SwiGLU projections and fuses SiLU into the following multiply.  Layout,
precision, program-config, and kernel-config candidates are layered onto this
class and retained only after same-harness PCC and latency comparison.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

import ttnn
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.functional_decoder import (
    EMITTED_BATCH,
    EMITTED_CACHE_LENGTH,
    EMITTED_PREFILL_SEQUENCE,
    FunctionalDecoder,
    _state_tensor,
    _to_device_tensor,
)


def _dram_sharded_weight_memory_config(mesh_device, k: int, n: int):
    """Shard a [K, N] constant evenly over the device DRAM banks."""

    dram_grid_size = mesh_device.dram_grid_size()
    if dram_grid_size.y != 1:
        raise ValueError(f"Expected a 1D DRAM grid, got {dram_grid_size}")
    dram_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(dram_grid_size.x - 1, 0),
            )
        }
    )
    padded_n = math.ceil(n / (ttnn.TILE_SIZE * dram_grid_size.x)) * ttnn.TILE_SIZE * dram_grid_size.x
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, [k, padded_n // dram_grid_size.x], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _l1_width_sharded_memory_config(mesh_device, height: int, width: int, num_cores: int):
    width_tiles = width // ttnn.TILE_SIZE
    if width % ttnn.TILE_SIZE or width_tiles % num_cores:
        raise ValueError(f"width={width} cannot be tile-sharded evenly across {num_cores} cores")
    device_grid = mesh_device.compute_with_storage_grid_size()
    grid = ttnn.num_cores_to_corerangeset(num_cores, device_grid, row_wise=True)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [height, width // num_cores], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _l1_padded_width_sharded_memory_config(mesh_device, height: int, width: int, num_cores: int):
    """Width-shard an output whose final core may contain logical padding."""

    device_grid = mesh_device.compute_with_storage_grid_size()
    grid = ttnn.num_cores_to_corerangeset(num_cores, device_grid, row_wise=True)
    shard_width = math.ceil(width / (ttnn.TILE_SIZE * num_cores)) * ttnn.TILE_SIZE
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [height, shard_width], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _advisor_1d_matmul_program_config(
    grid: tuple[int, int],
    *,
    per_core_n: int,
    out_subblock_w: int,
):
    """Materialize the exact 1D-multicast seed emitted by ttnn-advise."""

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _advisor_decode_norm_configs(mesh_device, hidden_size: int):
    """Reproduce the advisor's 11-core block-sharded decode RMSNorm choice."""

    num_cores = 11
    hidden_tiles = math.ceil(hidden_size / ttnn.TILE_SIZE)
    shard_width_tiles = math.ceil(hidden_tiles / num_cores)
    grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(num_cores - 1, 0),
            )
        }
    )
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            grid,
            [ttnn.TILE_SIZE, shard_width_tiles * ttnn.TILE_SIZE],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    subblock_w = next(divisor for divisor in range(4, 0, -1) if shard_width_tiles % divisor == 0)
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[num_cores, 1],
        subblock_w=subblock_w,
        block_h=1,
        block_w=shard_width_tiles,
        inplace=False,
    )
    return memory_config, program_config


def _dram_matmul_program_config(
    m: int,
    k: int,
    n: int,
    input_cores: int,
    output_cores: int,
    *,
    activation=None,
    max_in0_block_w: int = 8,
):
    k_tiles = k // ttnn.TILE_SIZE
    n_tiles = n // ttnn.TILE_SIZE
    if k_tiles % input_cores or n_tiles % output_cores:
        raise ValueError(
            f"DRAM matmul requires exact tile sharding: K={k}/cores={input_cores}, N={n}/cores={output_cores}"
        )
    in0_tiles = k_tiles // input_cores
    in0_block_w = next(divisor for divisor in range(min(max_in0_block_w, in0_tiles), 0, -1) if in0_tiles % divisor == 0)
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=math.ceil(m / ttnn.TILE_SIZE),
        per_core_N=n_tiles // output_cores,
        fused_activation=activation,
    )


def _prefill_matmul_program_config(m: int, k: int, n: int, grid: tuple[int, int], in0_block_w: int):
    m_tiles = math.ceil(m / ttnn.TILE_SIZE)
    n_tiles = math.ceil(n / ttnn.TILE_SIZE)
    per_core_m = math.ceil(m_tiles / grid[1])
    per_core_n = math.ceil(n_tiles / grid[0])
    out_subblock_h = 2 if per_core_m % 2 == 0 else 1
    out_subblock_w = 2 if per_core_n % 2 == 0 and out_subblock_h * 2 <= 4 else 1
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
    )


class OptimizedDecoder(FunctionalDecoder):
    """Decoder whose runtime MLP is the graph-rewritten optimized candidate."""

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        batch: int = EMITTED_BATCH,
        max_cache_len: int = EMITTED_CACHE_LENGTH,
        mlp_weight_dtype=ttnn.bfloat4_b,
        down_weight_dtype=None,
        mlp_math_fidelity=ttnn.MathFidelity.LoFi,
        use_dram_sharded_mlp: bool = True,
        mlp_geometry: str = "40x32",
        attention_weight_dtype=ttnn.bfloat4_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        use_dram_sharded_attention: bool = True,
        attention_geometry: str = "10x12_8x10",
        use_prefill_program_configs: bool = True,
        use_advisor_decode_layout: bool = True,
        use_advisor_1d_matmuls: bool = False,
        **kwargs,
    ) -> "OptimizedDecoder":
        decoder = super().from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            batch=batch,
            max_cache_len=max_cache_len,
            **kwargs,
        )

        decoder.use_dram_sharded_attention = use_dram_sharded_attention
        decoder.use_prefill_program_configs = use_prefill_program_configs
        decoder.use_advisor_decode_layout = use_advisor_decode_layout
        decoder.use_advisor_1d_matmuls = use_advisor_1d_matmuls
        if use_advisor_decode_layout and not (use_dram_sharded_attention and use_dram_sharded_mlp):
            raise ValueError(
                "use_advisor_decode_layout requires DRAM-sharded attention and MLP so the sharded residual chain "
                "has compatible dense-op boundaries"
            )
        decoder.decode_norm_mem_config, decoder.decode_norm_program_config = _advisor_decode_norm_configs(
            mesh_device, decoder.hidden_size
        )
        decoder.attention_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=attention_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        decoder.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        decoder.decode_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            q_chunk_size=0,
            k_chunk_size=0,
            exp_approx_mode=False,
        )
        decoder.prefill_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            q_chunk_size=64,
            k_chunk_size=64,
            exp_approx_mode=False,
        )
        if use_dram_sharded_attention:
            q = _state_tensor(state_dict, layer_idx, "self_attn.q_proj.weight").to(torch.bfloat16)
            k = _state_tensor(state_dict, layer_idx, "self_attn.k_proj.weight").to(torch.bfloat16)
            v = _state_tensor(state_dict, layer_idx, "self_attn.v_proj.weight").to(torch.bfloat16)
            o = _state_tensor(state_dict, layer_idx, "self_attn.o_proj.weight").to(torch.bfloat16)
            qkv = torch.cat((q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)), dim=-1)
            o = o.transpose(0, 1).contiguous()

            decoder.qkv_weight.deallocate(True)
            decoder.output_weight.deallocate(True)
            decoder.qkv_weight = ttnn.from_torch(
                qkv.contiguous(),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=attention_weight_dtype,
                memory_config=_dram_sharded_weight_memory_config(mesh_device, decoder.hidden_size, qkv.shape[-1]),
            )
            decoder.output_weight = ttnn.from_torch(
                o,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=attention_weight_dtype,
                memory_config=_dram_sharded_weight_memory_config(
                    mesh_device, decoder.attention_width, decoder.hidden_size
                ),
            )
            decoder.prefill_qkv_weight = _to_device_tensor(qkv, mesh_device, dtype=attention_weight_dtype)
            decoder.prefill_output_weight = _to_device_tensor(o, mesh_device, dtype=attention_weight_dtype)
            attention_geometries = {
                "80x96_64x80": (80, 96, 64, 80, 8),
                "40x48_32x40": (40, 48, 32, 40, 8),
                "20x24_16x20": (20, 24, 16, 20, 8),
                "10x12_8x10": (10, 12, 8, 10, 16),
            }
            if attention_geometry not in attention_geometries:
                raise ValueError(
                    f"Unknown attention_geometry={attention_geometry!r}; "
                    f"expected one of {sorted(attention_geometries)}"
                )
            qkv_input_cores, qkv_output_cores, o_input_cores, o_output_cores, max_in0_block_w = attention_geometries[
                attention_geometry
            ]
            decoder.attention_geometry = attention_geometries[attention_geometry]
            decoder.decode_qkv_input_mem_config = _l1_width_sharded_memory_config(
                mesh_device, ttnn.TILE_SIZE, decoder.hidden_size, qkv_input_cores
            )
            decoder.decode_qkv_output_mem_config = _l1_width_sharded_memory_config(
                mesh_device, ttnn.TILE_SIZE, qkv.shape[-1], qkv_output_cores
            )
            decoder.decode_qkv_program_config = _dram_matmul_program_config(
                ttnn.TILE_SIZE,
                decoder.hidden_size,
                qkv.shape[-1],
                qkv_input_cores,
                qkv_output_cores,
                max_in0_block_w=max_in0_block_w,
            )
            decoder.decode_o_input_mem_config = _l1_width_sharded_memory_config(
                mesh_device, ttnn.TILE_SIZE, decoder.attention_width, o_input_cores
            )
            decoder.decode_o_output_mem_config = _l1_width_sharded_memory_config(
                mesh_device, ttnn.TILE_SIZE, decoder.hidden_size, o_output_cores
            )
            decoder.decode_o_program_config = _dram_matmul_program_config(
                ttnn.TILE_SIZE,
                decoder.attention_width,
                decoder.hidden_size,
                o_input_cores,
                o_output_cores,
                max_in0_block_w=max_in0_block_w,
            )

        gate = _state_tensor(state_dict, layer_idx, "mlp.gate_proj.weight").to(torch.bfloat16)
        up = _state_tensor(state_dict, layer_idx, "mlp.up_proj.weight").to(torch.bfloat16)
        down = _state_tensor(state_dict, layer_idx, "mlp.down_proj.weight").to(torch.bfloat16)

        decoder.use_dram_sharded_mlp = use_dram_sharded_mlp
        geometries = {
            "80x64": (80, 64, 80, 2, 8),
            "40x32": (40, 32, 40, 4, 16),
            "20x16": (20, 16, 20, 8, 16),
            "10x32": (10, 32, 40, 16, 16),
            "10x64": (10, 64, 40, 16, 16),
        }
        if mlp_geometry not in geometries:
            raise ValueError(f"Unknown mlp_geometry={mlp_geometry!r}; expected one of {sorted(geometries)}")
        decoder.mlp_geometry = geometries[mlp_geometry]
        decoder.decode_mlp_output_mem_config = _l1_width_sharded_memory_config(
            mesh_device,
            ttnn.TILE_SIZE,
            decoder.hidden_size,
            decoder.mlp_geometry[2],
        )
        if use_dram_sharded_mlp:
            decoder.gate_weight.deallocate(True)
            decoder.up_weight.deallocate(True)
            decoder.down_weight.deallocate(True)
            decoder.gate_weight = ttnn.from_torch(
                gate.transpose(0, 1).contiguous(),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=mlp_weight_dtype,
                memory_config=_dram_sharded_weight_memory_config(
                    mesh_device, decoder.hidden_size, decoder.intermediate_size
                ),
            )
            decoder.up_weight = ttnn.from_torch(
                up.transpose(0, 1).contiguous(),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=mlp_weight_dtype,
                memory_config=_dram_sharded_weight_memory_config(
                    mesh_device, decoder.hidden_size, decoder.intermediate_size
                ),
            )
            decoder.down_weight = ttnn.from_torch(
                down.transpose(0, 1).contiguous(),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=down_weight_dtype or mlp_weight_dtype,
                memory_config=_dram_sharded_weight_memory_config(
                    mesh_device, decoder.intermediate_size, decoder.hidden_size
                ),
            )
            decoder.gate_up_weight = None
            # The dedicated DRAM-sharded program is decode-only (M=1 tile).
            # Keep one compressed interleaved copy for the large-M prefill
            # program; this is still smaller than the functional BF16 MLP.
            gate_up = torch.cat((gate.transpose(0, 1), up.transpose(0, 1)), dim=-1)
            decoder.prefill_gate_up_weight = _to_device_tensor(gate_up, mesh_device, dtype=mlp_weight_dtype)
            decoder.prefill_down_weight = _to_device_tensor(
                down.transpose(0, 1), mesh_device, dtype=down_weight_dtype or mlp_weight_dtype
            )
        else:
            gate_up = torch.cat((gate.transpose(0, 1), up.transpose(0, 1)), dim=-1)
            decoder.gate_up_weight = _to_device_tensor(gate_up, mesh_device, dtype=mlp_weight_dtype)

            # These tensors belong only to the functional split candidate.
            decoder.gate_weight.deallocate(True)
            decoder.up_weight.deallocate(True)
            decoder.gate_weight = None
            decoder.up_weight = None

        if down_weight_dtype is None:
            down_weight_dtype = mlp_weight_dtype
        if not use_dram_sharded_mlp and down_weight_dtype != ttnn.bfloat16:
            decoder.down_weight.deallocate(True)
            decoder.down_weight = _to_device_tensor(down.transpose(0, 1), mesh_device, dtype=down_weight_dtype)

        decoder.mlp_weight_dtype = mlp_weight_dtype
        decoder.down_weight_dtype = down_weight_dtype
        decoder.mlp_math_fidelity = mlp_math_fidelity
        decoder.mlp_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=mlp_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        if use_advisor_1d_matmuls:
            if not (use_dram_sharded_attention and use_dram_sharded_mlp and use_advisor_decode_layout):
                raise ValueError(
                    "use_advisor_1d_matmuls requires the optimized attention/MLP loaders and advisor residual layout"
                )
            for weight in (
                decoder.qkv_weight,
                decoder.output_weight,
                decoder.gate_weight,
                decoder.up_weight,
                decoder.down_weight,
            ):
                weight.deallocate(True)
            decoder.qkv_weight = _to_device_tensor(qkv, mesh_device, dtype=attention_weight_dtype)
            decoder.output_weight = _to_device_tensor(o, mesh_device, dtype=attention_weight_dtype)
            decoder.gate_weight = _to_device_tensor(gate.transpose(0, 1), mesh_device, dtype=mlp_weight_dtype)
            decoder.up_weight = _to_device_tensor(up.transpose(0, 1), mesh_device, dtype=mlp_weight_dtype)
            decoder.down_weight = _to_device_tensor(down.transpose(0, 1), mesh_device, dtype=down_weight_dtype)

            decoder.decode_qkv_input_mem_config = _l1_width_sharded_memory_config(
                mesh_device, ttnn.TILE_SIZE, decoder.hidden_size, 80
            )
            decoder.decode_qkv_output_mem_config = _l1_width_sharded_memory_config(
                mesh_device, ttnn.TILE_SIZE, qkv.shape[-1], 96
            )
            decoder.decode_qkv_program_config = _advisor_1d_matmul_program_config(
                (11, 9), per_core_n=2, out_subblock_w=2
            )
            decoder.decode_o_input_mem_config = ttnn.L1_MEMORY_CONFIG
            decoder.decode_o_output_mem_config = _l1_width_sharded_memory_config(
                mesh_device, ttnn.TILE_SIZE, decoder.hidden_size, 80
            )
            decoder.decode_o_program_config = _advisor_1d_matmul_program_config((11, 8), per_core_n=2, out_subblock_w=2)

            decoder.advisor_mlp_input_mem_config = _l1_width_sharded_memory_config(
                mesh_device, ttnn.TILE_SIZE, decoder.hidden_size, 80
            )
            decoder.advisor_mlp_intermediate_mem_config = _l1_padded_width_sharded_memory_config(
                mesh_device, ttnn.TILE_SIZE, decoder.intermediate_size, 103
            )
            decoder.advisor_mlp_down_input_mem_config = _l1_padded_width_sharded_memory_config(
                mesh_device, ttnn.TILE_SIZE, decoder.intermediate_size, 86
            )
            decoder.decode_mlp_output_mem_config = _l1_width_sharded_memory_config(
                mesh_device, ttnn.TILE_SIZE, decoder.hidden_size, 80
            )
            decoder.advisor_gate_up_program_config = _advisor_1d_matmul_program_config(
                (11, 10), per_core_n=10, out_subblock_w=5
            )
            decoder.advisor_down_program_config = _advisor_1d_matmul_program_config(
                (11, 8), per_core_n=2, out_subblock_w=2
            )
        return decoder

    def _mlp_forward(self, hidden_states):
        if self.use_dram_sharded_mlp:
            physical_height = math.ceil(int(hidden_states.shape[-2]) / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
            if physical_height != ttnn.TILE_SIZE:
                return self._packed_mlp_forward(hidden_states, self.prefill_gate_up_weight, self.prefill_down_weight)
            input_cores, intermediate_cores, output_cores, gate_max_block, down_max_block = self.mlp_geometry
            if self.use_advisor_1d_matmuls:
                input_mem_config = self.advisor_mlp_input_mem_config
                intermediate_mem_config = self.advisor_mlp_intermediate_mem_config
                gate_program_config = self.advisor_gate_up_program_config
                up_program_config = self.advisor_gate_up_program_config
                down_program_config = self.advisor_down_program_config
            else:
                input_mem_config = _l1_width_sharded_memory_config(
                    self.mesh_device, physical_height, self.hidden_size, input_cores
                )
                intermediate_mem_config = _l1_width_sharded_memory_config(
                    self.mesh_device, physical_height, self.intermediate_size, intermediate_cores
                )
                gate_program_config = _dram_matmul_program_config(
                    physical_height,
                    self.hidden_size,
                    self.intermediate_size,
                    input_cores,
                    intermediate_cores,
                    max_in0_block_w=gate_max_block,
                )
                up_program_config = _dram_matmul_program_config(
                    physical_height,
                    self.hidden_size,
                    self.intermediate_size,
                    input_cores,
                    intermediate_cores,
                    max_in0_block_w=gate_max_block,
                )
                down_program_config = _dram_matmul_program_config(
                    physical_height,
                    self.intermediate_size,
                    self.hidden_size,
                    intermediate_cores,
                    output_cores,
                    max_in0_block_w=down_max_block,
                )
            hidden_states = ttnn.to_memory_config(hidden_states, input_mem_config)
            gate = ttnn.matmul(
                hidden_states,
                self.gate_weight,
                dtype=ttnn.bfloat16,
                memory_config=intermediate_mem_config,
                program_config=gate_program_config,
                compute_kernel_config=self.mlp_compute_kernel_config,
            )
            up = ttnn.matmul(
                hidden_states,
                self.up_weight,
                dtype=ttnn.bfloat16,
                memory_config=intermediate_mem_config,
                program_config=up_program_config,
                compute_kernel_config=self.mlp_compute_kernel_config,
            )
            ttnn.deallocate(hidden_states)
            gated = ttnn.multiply(
                gate,
                up,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                dtype=ttnn.bfloat16,
                memory_config=intermediate_mem_config,
            )
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
            if self.use_advisor_1d_matmuls:
                gated = ttnn.to_memory_config(gated, self.advisor_mlp_down_input_mem_config)
            down = ttnn.matmul(
                gated,
                self.down_weight,
                dtype=ttnn.bfloat16,
                memory_config=input_mem_config,
                program_config=down_program_config,
                compute_kernel_config=self.mlp_compute_kernel_config,
            )
            if self.use_advisor_decode_layout:
                return down
            return ttnn.to_memory_config(down, ttnn.DRAM_MEMORY_CONFIG)

        return self._packed_mlp_forward(hidden_states, self.gate_up_weight, self.down_weight)

    def _packed_mlp_forward(self, hidden_states, gate_up_weight, down_weight):
        token_count = int(hidden_states.shape[-2])
        max_prefill_tokens = EMITTED_BATCH * EMITTED_PREFILL_SEQUENCE
        if token_count > max_prefill_tokens:
            chunks = []
            for start in range(0, token_count, max_prefill_tokens):
                end = min(start + max_prefill_tokens, token_count)
                chunk = ttnn.slice(
                    hidden_states,
                    [0, 0, start, 0],
                    [hidden_states.shape[0], hidden_states.shape[1], end, self.hidden_size],
                    [1, 1, 1, 1],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                chunks.append(self._packed_mlp_forward(chunk, gate_up_weight, down_weight))
            return ttnn.concat(chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        use_prefill_config = self.use_prefill_program_configs and token_count > ttnn.TILE_SIZE
        gate_up = ttnn.matmul(
            hidden_states,
            gate_up_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mlp_compute_kernel_config,
            program_config=(
                _prefill_matmul_program_config(
                    token_count,
                    self.hidden_size,
                    2 * self.intermediate_size,
                    (11, 10),
                    2,
                )
                if use_prefill_config
                else None
            ),
        )
        gate = ttnn.slice(
            gate_up,
            [0, 0, 0, 0],
            [gate_up.shape[0], gate_up.shape[1], gate_up.shape[2], self.intermediate_size],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        up = ttnn.slice(
            gate_up,
            [0, 0, 0, self.intermediate_size],
            [gate_up.shape[0], gate_up.shape[1], gate_up.shape[2], 2 * self.intermediate_size],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gated = ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.matmul(
            gated,
            down_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mlp_compute_kernel_config,
            program_config=(
                _prefill_matmul_program_config(
                    token_count,
                    self.intermediate_size,
                    self.hidden_size,
                    (10, 9),
                    4,
                )
                if use_prefill_config
                else None
            ),
        )

    def prefill_forward(self, hidden_states, key_cache, value_cache):
        seq_len = self._validate_hidden_states(hidden_states)
        self._validate_caches(key_cache, value_cache)
        token_count = seq_len * self.batch
        use_tuned_dense_config = (
            self.use_prefill_program_configs and token_count <= EMITTED_BATCH * EMITTED_PREFILL_SEQUENCE
        )

        # Put the emitted [batch, seq] token axes into one matrix dimension.
        # This removes the per-user sequence-tile padding from every dense op:
        # seq=18 executes M=576 rather than 32 independent padded M=32 batches.
        hidden_states = ttnn.permute(hidden_states, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.reshape(hidden_states, [1, 1, token_count, self.hidden_size])
        residual = hidden_states
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.input_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fused_qkv = ttnn.matmul(
            normed,
            self.prefill_qkv_weight if self.use_dram_sharded_attention else self.qkv_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attention_compute_kernel_config,
            program_config=(
                _prefill_matmul_program_config(
                    token_count,
                    self.hidden_size,
                    (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
                    (8, 9),
                    8,
                )
                if use_tuned_dense_config
                else None
            ),
        )
        fused_qkv = ttnn.reshape(
            fused_qkv,
            [1, seq_len, self.batch, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim],
        )
        fused_qkv = ttnn.permute(fused_qkv, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        fused_qkv = ttnn.reshape(
            fused_qkv,
            [self.batch, seq_len, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim],
        )
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused_qkv,
            None,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        cos = ttnn.slice(self.rotary_cos, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim], [1, 1, 1, 1])
        sin = ttnn.slice(self.rotary_sin, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim], [1, 1, 1, 1])
        query = ttnn.experimental.rotary_embedding(query, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.experimental.rotary_embedding(key, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        query = ttnn.slice(
            query,
            [0, 0, 0, 0],
            [self.batch, self.num_heads, seq_len, self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.slice(
            key,
            [0, 0, 0, 0],
            [self.batch, self.num_kv_heads, seq_len, self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cache_key = key
        cache_value = value
        if key_cache.dtype != key.dtype:
            cache_key = ttnn.typecast(key, key_cache.dtype)
        if value_cache.dtype != value.dtype:
            cache_value = ttnn.typecast(value, value_cache.dtype)

        for user_id in range(self.batch):
            key_user = ttnn.slice(
                cache_key,
                [user_id, 0, 0, 0],
                [user_id + 1, self.num_kv_heads, seq_len, self.head_dim],
                [1, 1, 1, 1],
            )
            value_user = ttnn.slice(
                cache_value,
                [user_id, 0, 0, 0],
                [user_id + 1, self.num_kv_heads, seq_len, self.head_dim],
                [1, 1, 1, 1],
            )
            ttnn.fill_cache(key_cache, key_user, user_id)
            ttnn.fill_cache(value_cache, value_user, user_id)

        attention = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.prefill_sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        attention = ttnn.transformer.concatenate_heads(attention, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.reshape(attention, [1, self.batch, seq_len, self.attention_width])
        attention = ttnn.permute(attention, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.reshape(attention, [1, 1, token_count, self.attention_width])
        attention = ttnn.matmul(
            attention,
            self.prefill_output_weight if self.use_dram_sharded_attention else self.output_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attention_compute_kernel_config,
            program_config=(
                _prefill_matmul_program_config(
                    token_count,
                    self.attention_width,
                    self.hidden_size,
                    (10, 9),
                    8,
                )
                if use_tuned_dense_config
                else None
            ),
        )
        hidden_states = ttnn.add(residual, attention, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        residual = hidden_states
        hidden_states = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.post_attention_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden_states = self._mlp_forward(hidden_states)
        hidden_states = ttnn.add(residual, hidden_states, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.reshape(hidden_states, [1, seq_len, self.batch, self.hidden_size])
        return ttnn.permute(hidden_states, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def decode_forward(self, hidden_states, key_cache, value_cache, *, current_pos: int):
        self._validate_hidden_states(hidden_states, expected_seq_len=1)
        self._validate_caches(key_cache, value_cache)
        if current_pos < 0 or current_pos >= self.max_cache_len:
            raise ValueError(f"current_pos must be in [0, {self.max_cache_len}), got {current_pos}")

        # A metadata reshape moves batch into the matmul M axis.  Dense ops now
        # execute one M=32 matrix instead of 32 independently padded M=1 jobs.
        hidden_states = ttnn.reshape(hidden_states, [1, 1, self.batch, self.hidden_size])
        if self.use_advisor_decode_layout:
            hidden_states = ttnn.to_memory_config(hidden_states, self.decode_norm_mem_config)
        residual = hidden_states
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.input_norm,
            memory_config=self.decode_norm_mem_config if self.use_advisor_decode_layout else ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.decode_norm_program_config if self.use_advisor_decode_layout else None,
        )
        if self.use_dram_sharded_attention:
            normed = ttnn.to_memory_config(normed, self.decode_qkv_input_mem_config)
            fused_qkv = ttnn.matmul(
                normed,
                self.qkv_weight,
                dtype=ttnn.bfloat16,
                memory_config=self.decode_qkv_output_mem_config,
                program_config=self.decode_qkv_program_config,
                compute_kernel_config=self.attention_compute_kernel_config,
            )
        else:
            fused_qkv = ttnn.matmul(
                normed,
                self.qkv_weight,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.attention_compute_kernel_config,
            )
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused_qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            overlap_qk_coregrid=None,
            memory_config=self.decode_heads_mem_config,
        )
        ttnn.deallocate(fused_qkv)

        query = ttnn.experimental.rotary_embedding(
            query,
            self.rotary_cos,
            self.rotary_sin,
            current_pos,
            memory_config=self.decode_heads_mem_config,
        )
        key = ttnn.experimental.rotary_embedding(
            key,
            self.rotary_cos,
            self.rotary_sin,
            current_pos,
            memory_config=self.decode_heads_mem_config,
        )

        position = ttnn.slice(self.position_indices, [current_pos], [current_pos + 1], [1])
        update_indices = ttnn.repeat(position, ttnn.Shape([self.batch]), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.experimental.paged_update_cache(
            key_cache,
            key,
            update_idxs_tensor=update_indices,
            share_cache=False,
            page_table=None,
        )
        ttnn.experimental.paged_update_cache(
            value_cache,
            value,
            update_idxs_tensor=update_indices,
            share_cache=False,
            page_table=None,
        )
        ttnn.deallocate(key)
        ttnn.deallocate(value)

        query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.transformer.scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            is_causal=True,
            cur_pos_tensor=update_indices,
            scale=self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.decode_sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        ttnn.deallocate(query)
        sharded_attention = ttnn.to_memory_config(attention, self.decode_heads_mem_config)
        ttnn.deallocate(attention)
        concatenated_attention = ttnn.experimental.nlp_concat_heads_decode(
            sharded_attention,
            num_heads=self.num_heads,
            sub_core_grids=self.decode_compute_core_grid,
        )
        ttnn.deallocate(sharded_attention)
        if self.use_dram_sharded_attention:
            attention = ttnn.to_memory_config(concatenated_attention, self.decode_o_input_mem_config)
            ttnn.deallocate(concatenated_attention)
            attention = ttnn.matmul(
                attention,
                self.output_weight,
                dtype=ttnn.bfloat16,
                memory_config=self.decode_o_output_mem_config,
                program_config=self.decode_o_program_config,
                compute_kernel_config=self.attention_compute_kernel_config,
            )
            if not self.use_advisor_decode_layout:
                attention = ttnn.to_memory_config(attention, ttnn.DRAM_MEMORY_CONFIG)
        else:
            attention = ttnn.to_memory_config(concatenated_attention, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(concatenated_attention)
            attention = ttnn.matmul(
                attention,
                self.output_weight,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.attention_compute_kernel_config,
            )
        if self.use_advisor_decode_layout:
            residual = ttnn.to_memory_config(residual, self.decode_o_output_mem_config)
            hidden_states = ttnn.add(
                residual,
                attention,
                dtype=ttnn.bfloat16,
                memory_config=self.decode_o_output_mem_config,
            )
            hidden_states = ttnn.to_memory_config(hidden_states, self.decode_norm_mem_config)
        else:
            hidden_states = ttnn.add(residual, attention, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        residual = hidden_states
        hidden_states = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.post_attention_norm,
            memory_config=self.decode_norm_mem_config if self.use_advisor_decode_layout else ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.decode_norm_program_config if self.use_advisor_decode_layout else None,
        )
        hidden_states = self._mlp_forward(hidden_states)
        if self.use_advisor_decode_layout:
            residual = ttnn.to_memory_config(residual, self.decode_mlp_output_mem_config)
            hidden_states = ttnn.add(
                residual,
                hidden_states,
                dtype=ttnn.bfloat16,
                memory_config=self.decode_mlp_output_mem_config,
            )
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        else:
            hidden_states = ttnn.add(
                residual, hidden_states, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        return ttnn.reshape(hidden_states, [1, self.batch, 1, self.hidden_size])


__all__ = [
    "EMITTED_BATCH",
    "EMITTED_CACHE_LENGTH",
    "EMITTED_PREFILL_SEQUENCE",
    "OptimizedDecoder",
]
