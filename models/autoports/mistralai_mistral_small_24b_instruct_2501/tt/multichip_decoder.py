# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TP4 decoder layer for Mistral-Small-24B-Instruct-2501.

The implementation keeps the optimized decoder's BFP4/LoFi dense kernels,
BFP8 KV-cache contract, flattened token matrices, and internal prefill MLP
chunking.  It restores the compiler-emitted 1x4 tensor parallelism: Q/K/V and
SwiGLU outputs are column sharded, output/down projections are row sharded,
and their partial hidden tensors are all-reduced before residual addition.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding

import ttnn
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.functional_decoder import (
    EMITTED_BATCH,
    EMITTED_CACHE_LENGTH,
    EMITTED_PREFILL_SEQUENCE,
    _config_value,
    _state_tensor,
)
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.optimized_decoder import (
    OptimizedDecoder,
    _advisor_decode_norm_configs,
    _dram_matmul_program_config,
    _dram_sharded_weight_memory_config,
    _l1_width_sharded_memory_config,
    _prefill_matmul_program_config,
)

TP_DEGREE = 4
TP_AXIS = 1


def _replicate_to_mesh(tensor: torch.Tensor, mesh_device, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _shard_to_mesh(
    tensor: torch.Tensor,
    mesh_device,
    *,
    dim: int,
    dtype,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
    )


def _rank_packed_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Return Q/K/V ordered so a width shard owns matching local heads."""

    q_chunks = q.transpose(0, 1).chunk(TP_DEGREE, dim=-1)
    k_chunks = k.transpose(0, 1).chunk(TP_DEGREE, dim=-1)
    v_chunks = v.transpose(0, 1).chunk(TP_DEGREE, dim=-1)
    return torch.cat(
        [torch.cat((q_chunks[rank], k_chunks[rank], v_chunks[rank]), dim=-1) for rank in range(TP_DEGREE)],
        dim=-1,
    ).contiguous()


def _rank_packed_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Return gate/up ordered so one shard contains both matching rank slices."""

    gate_chunks = gate.transpose(0, 1).chunk(TP_DEGREE, dim=-1)
    up_chunks = up.transpose(0, 1).chunk(TP_DEGREE, dim=-1)
    return torch.cat(
        [torch.cat((gate_chunks[rank], up_chunks[rank]), dim=-1) for rank in range(TP_DEGREE)], dim=-1
    ).contiguous()


class MultichipDecoder(OptimizedDecoder):
    """Optimized dense decoder layer specialized for the local 1x4 mesh."""

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
        attention_weight_dtype=ttnn.bfloat4_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        topology=ttnn.Topology.Linear,
        num_links: int = 2,
        use_prefill_program_configs: bool = True,
        attention_geometry: tuple[int, int, int, int, int, int] = (10, 12, 16, 8, 10, 4),
        mlp_geometry: tuple[int, int, int, int, int] = (10, 32, 40, 16, 16),
        shared_rope: tuple | None = None,
        **_kwargs,
    ) -> "MultichipDecoder":
        num_devices = mesh_device.get_num_devices()
        mesh_shape = tuple(mesh_device.shape)
        if mesh_shape != (1, TP_DEGREE) or num_devices != TP_DEGREE:
            raise ValueError(
                "MultichipDecoder requires the target logical 1x4 mesh, "
                f"got shape {mesh_shape} with {num_devices} devices"
            )
        if batch < 1 or batch > EMITTED_BATCH:
            raise ValueError(f"batch must be in [1, {EMITTED_BATCH}], got {batch}")
        if max_cache_len < 1:
            raise ValueError(f"max_cache_len must be positive, got {max_cache_len}")

        hidden_size = int(_config_value(hf_config, "hidden_size"))
        global_num_heads = int(_config_value(hf_config, "num_attention_heads"))
        global_num_kv_heads = int(_config_value(hf_config, "num_key_value_heads"))
        head_dim = int(_config_value(hf_config, "head_dim") or hidden_size // global_num_heads)
        global_intermediate_size = int(_config_value(hf_config, "intermediate_size"))
        rms_norm_eps = float(_config_value(hf_config, "rms_norm_eps"))
        advertised_context = int(_config_value(hf_config, "max_position_embeddings"))
        global_attention_width = global_num_heads * head_dim

        if max_cache_len > advertised_context:
            raise ValueError(f"max_cache_len={max_cache_len} exceeds max_position_embeddings={advertised_context}")
        for name, value in (
            ("num_attention_heads", global_num_heads),
            ("num_key_value_heads", global_num_kv_heads),
            ("attention_width", global_attention_width),
            ("intermediate_size", global_intermediate_size),
        ):
            if value % TP_DEGREE:
                raise ValueError(f"{name}={value} must divide evenly over TP{TP_DEGREE}")
        if global_num_heads % global_num_kv_heads:
            raise ValueError(
                f"num_attention_heads={global_num_heads} must be divisible by num_key_value_heads={global_num_kv_heads}"
            )
        if str(_config_value(hf_config, "hidden_act")) != "silu":
            raise ValueError(
                f"MultichipDecoder requires hidden_act='silu', got {_config_value(hf_config, 'hidden_act')!r}"
            )

        q = _state_tensor(state_dict, layer_idx, "self_attn.q_proj.weight").to(torch.bfloat16)
        k = _state_tensor(state_dict, layer_idx, "self_attn.k_proj.weight").to(torch.bfloat16)
        v = _state_tensor(state_dict, layer_idx, "self_attn.v_proj.weight").to(torch.bfloat16)
        o = _state_tensor(state_dict, layer_idx, "self_attn.o_proj.weight").to(torch.bfloat16)
        gate = _state_tensor(state_dict, layer_idx, "mlp.gate_proj.weight").to(torch.bfloat16)
        up = _state_tensor(state_dict, layer_idx, "mlp.up_proj.weight").to(torch.bfloat16)
        down = _state_tensor(state_dict, layer_idx, "mlp.down_proj.weight").to(torch.bfloat16)
        input_norm = _state_tensor(state_dict, layer_idx, "input_layernorm.weight").to(torch.bfloat16)
        post_attention_norm = _state_tensor(state_dict, layer_idx, "post_attention_layernorm.weight").to(torch.bfloat16)

        expected_shapes = {
            "q": (global_attention_width, hidden_size),
            "k": (global_num_kv_heads * head_dim, hidden_size),
            "v": (global_num_kv_heads * head_dim, hidden_size),
            "o": (hidden_size, global_attention_width),
            "gate": (global_intermediate_size, hidden_size),
            "up": (global_intermediate_size, hidden_size),
            "down": (hidden_size, global_intermediate_size),
            "input_norm": (hidden_size,),
            "post_attention_norm": (hidden_size,),
        }
        for name, tensor in {
            "q": q,
            "k": k,
            "v": v,
            "o": o,
            "gate": gate,
            "up": up,
            "down": down,
            "input_norm": input_norm,
            "post_attention_norm": post_attention_norm,
        }.items():
            if tuple(tensor.shape) != expected_shapes[name]:
                raise ValueError(f"{name} weight has shape {tuple(tensor.shape)}, expected {expected_shapes[name]}")

        local_num_heads = global_num_heads // TP_DEGREE
        local_num_kv_heads = global_num_kv_heads // TP_DEGREE
        local_attention_width = global_attention_width // TP_DEGREE
        local_intermediate_size = global_intermediate_size // TP_DEGREE
        local_qkv_width = (local_num_heads + 2 * local_num_kv_heads) * head_dim

        qkv = _rank_packed_qkv(q, k, v)
        gate_up = _rank_packed_gate_up(gate, up)
        o_t = o.transpose(0, 1).contiguous()
        down_t = down.transpose(0, 1).contiguous()
        gate_t = gate.transpose(0, 1).contiguous()
        up_t = up.transpose(0, 1).contiguous()

        if len(attention_geometry) != 6:
            raise ValueError(
                "attention_geometry must be "
                "(qkv_input_cores, qkv_output_cores, qkv_max_block, o_input_cores, o_output_cores, o_max_block)"
            )
        if len(mlp_geometry) != 5:
            raise ValueError(
                "mlp_geometry must be "
                "(input_cores, intermediate_cores, output_cores, gate_max_block, down_max_block)"
            )

        if shared_rope is None:
            rotary = MistralRotaryEmbedding(hf_config)
            positions = torch.arange(max_cache_len, dtype=torch.long).unsqueeze(0)
            rope_probe = torch.zeros((1, 1, max_cache_len, head_dim), dtype=torch.bfloat16)
            cos_host, sin_host = rotary(rope_probe, positions)
            cos_host = cos_host.to(torch.bfloat16)
            sin_host = sin_host.to(torch.bfloat16)
            cos = _replicate_to_mesh(cos_host.unsqueeze(1), mesh_device)
            sin = _replicate_to_mesh(sin_host.unsqueeze(1), mesh_device)
            decode_cos = _replicate_to_mesh(
                cos_host.reshape(max_cache_len, head_dim),
                mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
            )
            decode_sin = _replicate_to_mesh(
                sin_host.reshape(max_cache_len, head_dim),
                mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
            )
            position_indices = _replicate_to_mesh(
                torch.arange(max_cache_len, dtype=torch.int32),
                mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.int32,
            )
        else:
            if len(shared_rope) != 5:
                raise ValueError(
                    "shared_rope must contain "
                    "(rotary_cos, rotary_sin, position_indices, decode_rotary_cos, decode_rotary_sin)"
                )
            cos, sin, position_indices, decode_cos, decode_sin = shared_rope
            expected_rope_shape = (1, 1, max_cache_len, head_dim)
            if tuple(cos.shape) != expected_rope_shape or tuple(sin.shape) != expected_rope_shape:
                raise ValueError(
                    f"shared rotary cos/sin must have shape {expected_rope_shape}, "
                    f"got {tuple(cos.shape)} and {tuple(sin.shape)}"
                )
            if tuple(position_indices.shape) != (max_cache_len,):
                raise ValueError(
                    f"shared position_indices must have shape ({max_cache_len},), got {tuple(position_indices.shape)}"
                )
            expected_decode_shape = (max_cache_len, head_dim)
            if tuple(decode_cos.shape) != expected_decode_shape or tuple(decode_sin.shape) != expected_decode_shape:
                raise ValueError(
                    f"shared decode cos/sin must have shape {expected_decode_shape}, "
                    f"got {tuple(decode_cos.shape)} and {tuple(decode_sin.shape)}"
                )

        if down_weight_dtype is None:
            down_weight_dtype = mlp_weight_dtype

        decoder = cls(
            mesh_device=mesh_device,
            layer_idx=layer_idx,
            batch=batch,
            max_cache_len=max_cache_len,
            hidden_size=hidden_size,
            attention_width=local_attention_width,
            num_heads=local_num_heads,
            num_kv_heads=local_num_kv_heads,
            head_dim=head_dim,
            intermediate_size=local_intermediate_size,
            rms_norm_eps=rms_norm_eps,
            input_norm=_replicate_to_mesh(input_norm, mesh_device),
            post_attention_norm=_replicate_to_mesh(post_attention_norm, mesh_device),
            qkv_weight=_shard_to_mesh(
                qkv,
                mesh_device,
                dim=-1,
                dtype=attention_weight_dtype,
                memory_config=_dram_sharded_weight_memory_config(mesh_device, hidden_size, local_qkv_width),
            ),
            output_weight=_shard_to_mesh(
                o_t,
                mesh_device,
                dim=0,
                dtype=attention_weight_dtype,
                memory_config=_dram_sharded_weight_memory_config(mesh_device, local_attention_width, hidden_size),
            ),
            gate_weight=_shard_to_mesh(
                gate_t,
                mesh_device,
                dim=-1,
                dtype=mlp_weight_dtype,
                memory_config=_dram_sharded_weight_memory_config(mesh_device, hidden_size, local_intermediate_size),
            ),
            up_weight=_shard_to_mesh(
                up_t,
                mesh_device,
                dim=-1,
                dtype=mlp_weight_dtype,
                memory_config=_dram_sharded_weight_memory_config(mesh_device, hidden_size, local_intermediate_size),
            ),
            down_weight=_shard_to_mesh(
                down_t,
                mesh_device,
                dim=0,
                dtype=down_weight_dtype,
                memory_config=_dram_sharded_weight_memory_config(mesh_device, local_intermediate_size, hidden_size),
            ),
            rotary_cos=cos,
            rotary_sin=sin,
            position_indices=position_indices,
        )

        decoder.global_num_heads = global_num_heads
        decoder.global_num_kv_heads = global_num_kv_heads
        decoder.global_attention_width = global_attention_width
        decoder.global_intermediate_size = global_intermediate_size
        decoder.tp_degree = TP_DEGREE
        decoder.tp_axis = TP_AXIS
        decoder.collective_topology = topology
        decoder.num_links = num_links
        decoder.collectives_per_layer = 2
        decoder.activation_layout = "replicated_hidden"
        decoder.cache_layout = "kv_head_sharded"
        decoder.decode_rotary_cos = decode_cos
        decoder.decode_rotary_sin = decode_sin
        decoder.shared_rope = (
            decoder.rotary_cos,
            decoder.rotary_sin,
            decoder.position_indices,
            decoder.decode_rotary_cos,
            decoder.decode_rotary_sin,
        )

        decoder.use_dram_sharded_attention = True
        decoder.use_dram_sharded_mlp = True
        decoder.use_prefill_program_configs = use_prefill_program_configs
        decoder.use_advisor_decode_layout = True
        decoder.use_advisor_1d_matmuls = False
        decoder.mlp_geometry = tuple(int(value) for value in mlp_geometry)
        decoder.attention_geometry = tuple(int(value) for value in attention_geometry)
        decoder.mlp_weight_dtype = mlp_weight_dtype
        decoder.down_weight_dtype = down_weight_dtype
        decoder.attention_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=attention_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        decoder.mlp_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=mlp_math_fidelity,
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
            compute_with_storage_grid_size=(8, 8), q_chunk_size=0, k_chunk_size=0, exp_approx_mode=False
        )
        decoder.decode_paged_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8), q_chunk_size=0, k_chunk_size=128, exp_approx_mode=False
        )
        decoder.prefill_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8), q_chunk_size=64, k_chunk_size=64, exp_approx_mode=False
        )
        decoder.decode_norm_mem_config, decoder.decode_norm_program_config = _advisor_decode_norm_configs(
            mesh_device, hidden_size
        )

        # Interleaved local-weight copies are retained only for large-M prefill;
        # decode uses the DRAM-sharded tensors above, matching the optimized baseline.
        decoder.prefill_qkv_weight = _shard_to_mesh(qkv, mesh_device, dim=-1, dtype=attention_weight_dtype)
        decoder.prefill_output_weight = _shard_to_mesh(o_t, mesh_device, dim=0, dtype=attention_weight_dtype)
        decoder.prefill_gate_up_weight = _shard_to_mesh(gate_up, mesh_device, dim=-1, dtype=mlp_weight_dtype)
        decoder.prefill_down_weight = _shard_to_mesh(down_t, mesh_device, dim=0, dtype=down_weight_dtype)
        decoder.gate_up_weight = None

        (
            qkv_input_cores,
            qkv_output_cores,
            qkv_max_block,
            o_input_cores,
            o_output_cores,
            o_max_block,
        ) = decoder.attention_geometry
        decoder.decode_qkv_input_mem_config = _l1_width_sharded_memory_config(
            mesh_device, ttnn.TILE_SIZE, hidden_size, qkv_input_cores
        )
        decoder.decode_qkv_output_mem_config = _l1_width_sharded_memory_config(
            mesh_device, ttnn.TILE_SIZE, local_qkv_width, qkv_output_cores
        )
        decoder.decode_qkv_program_config = _dram_matmul_program_config(
            ttnn.TILE_SIZE,
            hidden_size,
            local_qkv_width,
            qkv_input_cores,
            qkv_output_cores,
            max_in0_block_w=qkv_max_block,
        )
        decoder.decode_o_input_mem_config = _l1_width_sharded_memory_config(
            mesh_device, ttnn.TILE_SIZE, local_attention_width, o_input_cores
        )
        decoder.decode_o_output_mem_config = _l1_width_sharded_memory_config(
            mesh_device, ttnn.TILE_SIZE, hidden_size, o_output_cores
        )
        decoder.decode_o_program_config = _dram_matmul_program_config(
            ttnn.TILE_SIZE,
            local_attention_width,
            hidden_size,
            o_input_cores,
            o_output_cores,
            max_in0_block_w=o_max_block,
        )
        mlp_output_cores = decoder.mlp_geometry[2]
        decoder.decode_mlp_output_mem_config = _l1_width_sharded_memory_config(
            mesh_device, ttnn.TILE_SIZE, hidden_size, mlp_output_cores
        )
        decoder.prefill_weights_released = False
        return decoder

    def release_prefill_weights(self) -> None:
        """Release interleaved prefill matrices before the long-lived decode phase."""

        if self.prefill_weights_released:
            return
        for name in (
            "prefill_qkv_weight",
            "prefill_output_weight",
            "prefill_gate_up_weight",
            "prefill_down_weight",
        ):
            tensor = getattr(self, name)
            if tensor is not None:
                ttnn.deallocate(tensor, force=True)
                setattr(self, name, None)
        self.prefill_weights_released = True

    def _validate_caches(self, key_cache, value_cache, *, paged: bool = False) -> None:
        key_shape = tuple(key_cache.shape)
        value_shape = tuple(value_cache.shape)
        if key_shape != value_shape:
            raise ValueError(f"key/value cache shapes differ: {key_shape} and {value_shape}")
        if paged:
            if len(key_shape) != 4 or key_shape[1] != self.num_kv_heads or key_shape[3] != self.head_dim:
                raise ValueError(
                    "paged caches must have local shape "
                    f"[num_blocks, {self.num_kv_heads}, block_size, {self.head_dim}], got {key_shape}"
                )
            if key_shape[0] < 1 or key_shape[2] < 1:
                raise ValueError(f"paged cache num_blocks and block_size must be positive, got {key_shape}")
            return
        expected = (self.batch, self.num_kv_heads, self.max_cache_len, self.head_dim)
        if key_shape != expected:
            raise ValueError(f"contiguous caches must have local shape {expected}, got {key_shape}")

    def _all_reduce_hidden(self, partial_hidden):
        if partial_hidden.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            partial_hidden = ttnn.to_memory_config(partial_hidden, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.all_reduce(
            partial_hidden,
            cluster_axis=self.tp_axis,
            topology=self.collective_topology,
            num_links=self.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _mlp_forward(self, hidden_states):
        """Run the tuned local SwiGLU with an explicit down-output shard contract."""

        physical_height = (int(hidden_states.shape[-2]) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
        if physical_height != ttnn.TILE_SIZE:
            return self._packed_mlp_forward(hidden_states, self.prefill_gate_up_weight, self.prefill_down_weight)

        input_cores, intermediate_cores, output_cores, gate_max_block, down_max_block = self.mlp_geometry
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
            program_config=gate_program_config,
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
        return ttnn.matmul(
            gated,
            self.down_weight,
            dtype=ttnn.bfloat16,
            memory_config=self.decode_mlp_output_mem_config,
            program_config=down_program_config,
            compute_kernel_config=self.mlp_compute_kernel_config,
        )

    def prefill_forward(self, hidden_states, key_cache, value_cache, *, page_table=None):
        if self.prefill_weights_released:
            raise RuntimeError("prefill weights were released for decode-phase capacity and cannot be reused")
        seq_len = self._validate_hidden_states(hidden_states)
        self._validate_caches(key_cache, value_cache, paged=page_table is not None)
        token_count = seq_len * self.batch
        use_tuned_dense_config = (
            self.use_prefill_program_configs and token_count <= EMITTED_BATCH * EMITTED_PREFILL_SEQUENCE
        )

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
            self.prefill_qkv_weight,
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
            fused_qkv, [self.batch, seq_len, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim]
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
        cache_key = key if key_cache.dtype == key.dtype else ttnn.typecast(key, key_cache.dtype)
        cache_value = value if value_cache.dtype == value.dtype else ttnn.typecast(value, value_cache.dtype)

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
            if page_table is None:
                ttnn.fill_cache(key_cache, key_user, user_id)
                ttnn.fill_cache(value_cache, value_user, user_id)
            else:
                user_page_table = page_table[user_id : user_id + 1, :]
                ttnn.experimental.paged_fill_cache(key_cache, key_user, user_page_table, batch_idx=0)
                ttnn.experimental.paged_fill_cache(value_cache, value_user, user_page_table, batch_idx=0)

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
            self.prefill_output_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attention_compute_kernel_config,
            program_config=(
                _prefill_matmul_program_config(token_count, self.attention_width, self.hidden_size, (10, 9), 8)
                if use_tuned_dense_config
                else None
            ),
        )
        attention = self._all_reduce_hidden(attention)
        hidden_states = ttnn.add(residual, attention, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        residual = hidden_states
        hidden_states = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.post_attention_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden_states = self._mlp_forward(hidden_states)
        hidden_states = self._all_reduce_hidden(hidden_states)
        hidden_states = ttnn.add(residual, hidden_states, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.reshape(hidden_states, [1, seq_len, self.batch, self.hidden_size])
        return ttnn.permute(hidden_states, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def decode_forward(
        self,
        hidden_states,
        key_cache,
        value_cache,
        *,
        current_pos: int | None = None,
        page_table=None,
        current_pos_tensor=None,
    ):
        self._validate_hidden_states(hidden_states, expected_seq_len=1)
        self._validate_caches(key_cache, value_cache, paged=page_table is not None)
        if (current_pos is None) == (current_pos_tensor is None):
            raise ValueError("decode requires exactly one of current_pos or current_pos_tensor")
        if current_pos is not None and (current_pos < 0 or current_pos >= self.max_cache_len):
            raise ValueError(f"current_pos must be in [0, {self.max_cache_len}), got {current_pos}")

        hidden_states = ttnn.reshape(hidden_states, [1, 1, self.batch, self.hidden_size])
        hidden_states = ttnn.to_memory_config(hidden_states, self.decode_norm_mem_config)
        residual = hidden_states
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.input_norm,
            memory_config=self.decode_norm_mem_config,
            program_config=self.decode_norm_program_config,
        )
        normed = ttnn.to_memory_config(normed, self.decode_qkv_input_mem_config)
        fused_qkv = ttnn.matmul(
            normed,
            self.qkv_weight,
            dtype=ttnn.bfloat16,
            memory_config=self.decode_qkv_output_mem_config,
            program_config=self.decode_qkv_program_config,
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
        if current_pos_tensor is None:
            position = ttnn.slice(self.position_indices, [current_pos], [current_pos + 1], [1])
            current_pos_tensor = ttnn.repeat(position, ttnn.Shape([self.batch]), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        elif tuple(current_pos_tensor.shape) != (self.batch,) or current_pos_tensor.dtype != ttnn.int32:
            raise ValueError(
                f"current_pos_tensor must be INT32 with local shape ({self.batch},), "
                f"got {current_pos_tensor.dtype} {tuple(current_pos_tensor.shape)}"
            )

        # One mutable per-user position tensor drives RoPE, cache writes, and
        # SDPA.  All lookup work stays on device and is safe to capture/replay.
        rotary_positions = ttnn.typecast(current_pos_tensor, ttnn.uint32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rotary_positions = ttnn.reshape(rotary_positions, [1, self.batch])
        cos = ttnn.embedding(
            rotary_positions,
            self.decode_rotary_cos,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sin = ttnn.embedding(
            rotary_positions,
            self.decode_rotary_sin,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cos = ttnn.unsqueeze_to_4D(cos)
        sin = ttnn.unsqueeze_to_4D(sin)
        cos = ttnn.transpose(cos, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sin = ttnn.transpose(sin, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cos = ttnn.to_memory_config(cos, self.decode_heads_mem_config)
        sin = ttnn.to_memory_config(sin, self.decode_heads_mem_config)
        query = ttnn.experimental.rotary_embedding_hf(
            query, cos, sin, is_decode_mode=True, memory_config=self.decode_heads_mem_config
        )
        key = ttnn.experimental.rotary_embedding_hf(
            key, cos, sin, is_decode_mode=True, memory_config=self.decode_heads_mem_config
        )

        ttnn.experimental.paged_update_cache(
            key_cache,
            key,
            update_idxs_tensor=current_pos_tensor,
            share_cache=False,
            page_table=page_table,
        )
        ttnn.experimental.paged_update_cache(
            value_cache,
            value,
            update_idxs_tensor=current_pos_tensor,
            share_cache=False,
            page_table=page_table,
        )
        ttnn.deallocate(key)
        ttnn.deallocate(value)

        query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
        if page_table is None:
            attention = ttnn.transformer.scaled_dot_product_attention_decode(
                query,
                key_cache,
                value_cache,
                is_causal=True,
                cur_pos_tensor=current_pos_tensor,
                scale=self.scale,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=self.decode_sdpa_program_config,
                compute_kernel_config=self.sdpa_compute_kernel_config,
            )
        else:
            attention = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                query,
                key_cache,
                value_cache,
                cur_pos_tensor=current_pos_tensor,
                page_table_tensor=page_table,
                is_causal=True,
                scale=self.scale,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=self.decode_paged_sdpa_program_config,
                compute_kernel_config=self.sdpa_compute_kernel_config,
            )
        ttnn.deallocate(query)
        sharded_attention = ttnn.to_memory_config(attention, self.decode_heads_mem_config)
        ttnn.deallocate(attention)
        concatenated_attention = ttnn.experimental.nlp_concat_heads_decode(
            sharded_attention, num_heads=self.num_heads, sub_core_grids=self.decode_compute_core_grid
        )
        ttnn.deallocate(sharded_attention)
        attention = ttnn.to_memory_config(concatenated_attention, self.decode_o_input_mem_config)
        # For eight local heads, concat can already produce the exact WO input
        # memory config.  to_memory_config then aliases its input; deallocating
        # concatenated_attention would invalidate attention on every rank.
        attention = ttnn.matmul(
            attention,
            self.output_weight,
            dtype=ttnn.bfloat16,
            memory_config=self.decode_o_output_mem_config,
            program_config=self.decode_o_program_config,
            compute_kernel_config=self.attention_compute_kernel_config,
        )
        attention = self._all_reduce_hidden(attention)
        attention = ttnn.to_memory_config(attention, self.decode_norm_mem_config)
        hidden_states = ttnn.add(residual, attention, dtype=ttnn.bfloat16, memory_config=self.decode_norm_mem_config)

        residual = hidden_states
        hidden_states = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.post_attention_norm,
            memory_config=self.decode_norm_mem_config,
            program_config=self.decode_norm_program_config,
        )
        hidden_states = self._mlp_forward(hidden_states)
        hidden_states = self._all_reduce_hidden(hidden_states)
        hidden_states = ttnn.to_memory_config(hidden_states, self.decode_norm_mem_config)
        hidden_states = ttnn.add(
            residual, hidden_states, dtype=ttnn.bfloat16, memory_config=self.decode_norm_mem_config
        )
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(hidden_states, [1, self.batch, 1, self.hidden_size])

    def forward(
        self,
        hidden_states,
        key_cache,
        value_cache,
        *,
        mode: str,
        current_pos: int | None = None,
        page_table=None,
        current_pos_tensor=None,
    ):
        if mode == "prefill":
            if current_pos is not None:
                raise ValueError("current_pos is decode-only")
            return self.prefill_forward(hidden_states, key_cache, value_cache, page_table=page_table)
        if mode == "decode":
            return self.decode_forward(
                hidden_states,
                key_cache,
                value_cache,
                current_pos=current_pos,
                page_table=page_table,
                current_pos_tensor=current_pos_tensor,
            )
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")


__all__ = [
    "EMITTED_BATCH",
    "EMITTED_CACHE_LENGTH",
    "EMITTED_PREFILL_SEQUENCE",
    "MultichipDecoder",
    "TP_AXIS",
    "TP_DEGREE",
]
