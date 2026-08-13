# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TP=4 Blackhole decoder layer for Qwen/Qwen3.6-27B.

The class deliberately derives from :class:`OptimizedDecoder`: numerical
policies and single-chip semantics stay anchored to that completed baseline,
while setup replaces replicated model tensors with head-/channel-local mesh
shards. Runtime tensors remain replicated at layer boundaries; this was faster
than the measured coherent fractured-residual alternative.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import ADVERTISED_CONTEXT, _require_tensor
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import (
    OptimizedDecoder,
    _decode_program,
    _dram_weight_memory_config,
    _l1_width_memory_config,
    _prefill_program,
    resolve_policy,
)

TARGET_MESH_SHAPE = (1, 4)
TARGET_TP = 4
TARGET_FABRIC = ttnn.FabricConfig.FABRIC_1D_RING
TARGET_TOPOLOGY = ttnn.Topology.Ring


def _validate_contract(hf_config, layer_idx: int, mesh_device, batch: int, max_context: int, page_size: int):
    if not isinstance(mesh_device, ttnn.MeshDevice):
        raise TypeError("MultichipDecoder requires a ttnn.MeshDevice")
    if tuple(mesh_device.shape) != TARGET_MESH_SHAPE:
        raise ValueError(f"MultichipDecoder requires mesh {TARGET_MESH_SHAPE}, got {tuple(mesh_device.shape)}")
    if not 0 <= layer_idx < int(hf_config.num_hidden_layers):
        raise ValueError(f"layer_idx={layer_idx} is outside the configured layer range")
    if not 1 <= batch <= 32:
        raise ValueError(f"batch must be in [1, 32], got {batch}")
    if not 1 <= max_context <= int(hf_config.max_position_embeddings):
        raise ValueError(f"max_context must be in [1, {hf_config.max_position_embeddings}], got {max_context}")
    if page_size < 32 or page_size % 32:
        raise ValueError(f"page_size must be a positive tile multiple, got {page_size}")
    expected = (5120, 17408, 24, 4, 256, 16, 48, 128, 128)
    actual = (
        int(hf_config.hidden_size),
        int(hf_config.intermediate_size),
        int(hf_config.num_attention_heads),
        int(hf_config.num_key_value_heads),
        int(hf_config.head_dim),
        int(hf_config.linear_num_key_heads),
        int(hf_config.linear_num_value_heads),
        int(hf_config.linear_key_head_dim),
        int(hf_config.linear_value_head_dim),
    )
    if actual != expected:
        raise ValueError(f"Qwen3.6-27B TP4 shape mismatch: expected {expected}, got {actual}")


def _replicate(tensor, *, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _shard(tensor, dim: int, *, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _shard_decode_weight(tensor, dim: int, *, mesh_device, dtype, k: int, n: int):
    """Mesh-shard semantically, then shard each rank over its eight DRAM banks."""
    staging = _shard(tensor, dim, mesh_device=mesh_device, dtype=dtype)
    result = ttnn.to_memory_config(staging, _dram_weight_memory_config(mesh_device, k=k, n=n))
    ttnn.deallocate(staging)
    return result


def _device_major(chunks: list[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """Concatenate already semantic-local chunks for ShardTensorToMesh."""
    if len(chunks) != TARGET_TP or len({tuple(x.shape[:-1]) for x in chunks}) != 1:
        raise ValueError("device-major packing requires four shape-compatible chunks")
    return torch.cat(chunks, dim=dim).contiguous()


class MultichipDecoder(OptimizedDecoder):
    """Real TP4 implementation targeting this machine's four-chip BH ring."""

    optimization_profile = {
        "name": "qwen3_6_27b_blackhole_1x4_tp4",
        "single_chip_baseline": "OptimizedDecoder",
        "mesh": TARGET_MESH_SHAPE,
        "tensor_parallel": 4,
        "residual_layout": "replicated_hidden_5120",
        "collective": "preallocated_ring_reduce_scatter_all_gather_inside_row_parallel_o_and_down",
        "cache": "one_local_kv_head_per_device_paged_bfp8",
        "moe": "not_applicable_dense_mlp",
    }
    residual_layout = "replicated_hidden_5120"

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, object],
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        batch: int = 1,
        max_context: int = ADVERTISED_CONTEXT,
        page_size: int = 64,
        candidate: str = "default",
        policy_override=None,
        **_kwargs,
    ):
        _validate_contract(hf_config, layer_idx, mesh_device, batch, max_context, page_size)
        kind = hf_config.layer_types[layer_idx]
        # Preserve the completed optimized baseline's selected policy.  A TP4
        # BFP4/LoFi full-attention candidate was faster on synthetic tensors
        # but failed the pinned official-weight PCC gate (0.9870 < 0.995).
        effective_candidate = candidate
        supported_multichip_candidates = {
            "multichip_baseline",
            "multichip_packed_mlp",
            "multichip_bfp8_ccl_attention",
            "multichip_bfp8_ccl_mlp",
            "multichip_bfp8_ccl_all",
            "multichip_preallocated_ccl",
            "multichip_prefill_l1",
            "multichip_linear_packed_l1",
        }
        if candidate.startswith("multichip_") and candidate not in supported_multichip_candidates:
            raise ValueError(f"unknown multichip candidate {candidate!r}")
        multichip_candidate = candidate if candidate.startswith("multichip_") else "default"
        policy_candidate = "default" if candidate == "default" or candidate.startswith("multichip_") else candidate
        policy = policy_override or resolve_policy(policy_candidate, kind)
        hidden, intermediate = int(hf_config.hidden_size), int(hf_config.intermediate_size)
        local_i = intermediate // TARGET_TP

        def host(suffix, *, transpose=False, add_one=False, dtype=torch.bfloat16):
            value = _require_tensor(state_dict, layer_idx, suffix).to(dtype)
            if transpose:
                value = value.transpose(-2, -1)
            if add_one:
                value = value + 1
            return value.contiguous()

        weights = {
            "input_norm": _replicate(
                host("input_layernorm.weight", add_one=True).reshape(1, 1, 160, 32),
                mesh_device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            "post_attention_norm": _replicate(
                host("post_attention_layernorm.weight", add_one=True).reshape(1, 1, 160, 32),
                mesh_device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            # Stack-compatible fractured-residual candidate. Distributed
            # RMSNorm consumes one hidden-width quarter and therefore needs a
            # semantic mesh shard rather than the replicated tiled norm view.
            "input_norm_fractured": _shard(
                host("input_layernorm.weight", add_one=True).reshape(1, 1, 1, hidden),
                -1,
                mesh_device=mesh_device,
            ),
            "post_attention_norm_fractured": _shard(
                host("post_attention_layernorm.weight", add_one=True).reshape(1, 1, 1, hidden),
                -1,
                mesh_device=mesh_device,
            ),
        }
        gate, up = host("mlp.gate_proj.weight", transpose=True), host("mlp.up_proj.weight", transpose=True)
        down = host("mlp.down_proj.weight", transpose=True)
        weights["mlp_gate_decode"] = _shard_decode_weight(
            gate, -1, mesh_device=mesh_device, dtype=policy.mlp_gate_up_dtype, k=hidden, n=local_i
        )
        weights["mlp_up_decode"] = _shard_decode_weight(
            up, -1, mesh_device=mesh_device, dtype=policy.mlp_gate_up_dtype, k=hidden, n=local_i
        )
        if multichip_candidate == "multichip_packed_mlp":
            packed_gate_up = _device_major(
                [
                    torch.cat(
                        [gate[:, rank * local_i : (rank + 1) * local_i], up[:, rank * local_i : (rank + 1) * local_i]],
                        dim=-1,
                    )
                    for rank in range(TARGET_TP)
                ]
            )
            weights["mlp_gate_up_decode"] = _shard_decode_weight(
                packed_gate_up,
                -1,
                mesh_device=mesh_device,
                dtype=policy.mlp_gate_up_dtype,
                k=hidden,
                n=2 * local_i,
            )
        weights["mlp_down_decode"] = _shard_decode_weight(
            down, -2, mesh_device=mesh_device, dtype=policy.mlp_down_dtype, k=local_i, n=hidden
        )
        weights["mlp_gate_prefill"] = _shard(gate, -1, mesh_device=mesh_device, dtype=policy.mlp_gate_up_dtype)
        weights["mlp_up_prefill"] = _shard(up, -1, mesh_device=mesh_device, dtype=policy.mlp_gate_up_dtype)
        weights["mlp_down_prefill"] = _shard(down, -2, mesh_device=mesh_device, dtype=policy.mlp_down_dtype)

        caches, rope = {}, {}
        if kind == "full_attention":
            qh, kvh, hd = (
                int(hf_config.num_attention_heads),
                int(hf_config.num_key_value_heads),
                int(hf_config.head_dim),
            )
            qg = host("self_attn.q_proj.weight", transpose=True).reshape(hidden, qh, 2 * hd)
            q, gate_h = qg[..., :hd], qg[..., hd:]
            k, v = host("self_attn.k_proj.weight", transpose=True), host("self_attn.v_proj.weight", transpose=True)
            chunks = []
            for rank in range(TARGET_TP):
                qs = q[:, rank * 6 : (rank + 1) * 6].reshape(hidden, 1536)
                gs = gate_h[:, rank * 6 : (rank + 1) * 6].reshape(hidden, 1536)
                chunks.append(
                    torch.cat([qs, k[:, rank * 256 : (rank + 1) * 256], v[:, rank * 256 : (rank + 1) * 256], gs], -1)
                )
            packed = _device_major(chunks)
            weights["qkv_gate_decode"] = _shard_decode_weight(
                packed, -1, mesh_device=mesh_device, dtype=policy.attention_weight_dtype, k=hidden, n=3584
            )
            weights["qkv_gate_prefill"] = _shard(
                packed, -1, mesh_device=mesh_device, dtype=policy.attention_weight_dtype
            )
            for name, parts in (
                ("q", [q[:, r * 6 : (r + 1) * 6].reshape(hidden, 1536) for r in range(4)]),
                ("gate", [gate_h[:, r * 6 : (r + 1) * 6].reshape(hidden, 1536) for r in range(4)]),
                ("k", [k[:, r * 256 : (r + 1) * 256] for r in range(4)]),
                ("v", [v[:, r * 256 : (r + 1) * 256] for r in range(4)]),
            ):
                weights[f"{name}_prefill_long"] = _shard(
                    _device_major(parts), -1, mesh_device=mesh_device, dtype=policy.attention_weight_dtype
                )
            o_proj = host("self_attn.o_proj.weight", transpose=True)
            weights["o_proj_decode"] = _shard_decode_weight(
                o_proj, -2, mesh_device=mesh_device, dtype=policy.attention_weight_dtype, k=1536, n=hidden
            )
            weights["o_proj_prefill"] = _shard(o_proj, -2, mesh_device=mesh_device, dtype=policy.attention_weight_dtype)
            weights["q_norm"] = _replicate(host("self_attn.q_norm.weight", add_one=True), mesh_device=mesh_device)
            weights["k_norm"] = _replicate(host("self_attn.k_norm.weight", add_one=True), mesh_device=mesh_device)
            blocks = batch * math.ceil(max_context / page_size)
            cache_shape = (blocks, kvh, page_size, hd)
            caches["key"] = _shard(
                torch.zeros(cache_shape, dtype=torch.bfloat16), 1, mesh_device=mesh_device, dtype=policy.cache_dtype
            )
            caches["value"] = _shard(
                torch.zeros(cache_shape, dtype=torch.bfloat16), 1, mesh_device=mesh_device, dtype=policy.cache_dtype
            )
            caches["batch_indices"] = _replicate(
                torch.arange(batch, dtype=torch.int32),
                mesh_device=mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            rotary = Qwen3_5TextRotaryEmbedding(hf_config)
            positions = torch.arange(max_context, dtype=torch.long).reshape(1, -1)
            cos, sin = rotary(torch.empty(1, 1, hidden, dtype=torch.bfloat16), positions)
            rope["cos"] = _replicate(cos.squeeze(0), mesh_device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)
            rope["sin"] = _replicate(sin.squeeze(0), mesh_device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)
        else:
            cls._load_linear_tensors(weights, caches, host, mesh_device, policy, batch)

        decode_attention_memory_config = None
        if kind == "full_attention":
            device_grid = mesh_device.compute_with_storage_grid_size()
            grid_x = min(batch, device_grid.x)
            while batch % grid_x or batch // grid_x > device_grid.y:
                grid_x -= 1
            decode_attention_memory_config = ttnn.create_sharded_memory_config(
                shape=(32, int(hf_config.head_dim)),
                core_grid=ttnn.CoreGrid(y=batch // grid_x, x=grid_x),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

        decoder = cls(
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            batch=batch,
            max_context=max_context,
            page_size=page_size,
            weights=weights,
            caches=caches,
            rope=rope,
            decode_attention_memory_config=decode_attention_memory_config,
        )
        decoder.policy, decoder.candidate = policy, effective_candidate
        decoder.local_intermediate_size = local_i
        decoder.local_num_heads, decoder.local_num_kv_heads = 6, 1
        decoder.local_q_width, decoder.local_kv_width = 1536, 256
        decoder.local_linear_key_heads, decoder.local_linear_value_heads = 4, 12
        decoder.local_linear_key_width, decoder.local_linear_value_width = 512, 1536
        decoder._configure_multichip_compute()
        decoder._row_collective_mode = "all_reduce"
        decoder._multichip_candidate = multichip_candidate
        decoder._ccl_buffers = {}
        if multichip_candidate == "multichip_preallocated_ccl":
            # Full-attention O reaches its CCL before the B1 logical view;
            # linear O and MLP-down reach theirs after that view.
            rows_by_role = {"token_mixer": 32 if kind == "full_attention" else batch, "mlp": batch}
            for role, rows in rows_by_role.items():
                zeros = torch.zeros((1, 1, rows, hidden), dtype=torch.bfloat16)
                decoder._ccl_buffers[role] = {
                    "reduce_scatter": _shard(zeros, -1, mesh_device=mesh_device),
                    "all_gather": _replicate(zeros, mesh_device=mesh_device),
                }
        return decoder

    @staticmethod
    def _load_linear_tensors(weights, caches, host, mesh_device, policy, batch):
        kh, vh, kd, vd = 16, 48, 128, 128
        qkv = host("linear_attn.in_proj_qkv.weight", transpose=True)
        z = host("linear_attn.in_proj_z.weight", transpose=True)
        b = host("linear_attn.in_proj_b.weight", transpose=True)
        a = host("linear_attn.in_proj_a.weight", transpose=True)
        chunks = []
        for rank in range(4):
            q = qkv[:, rank * 512 : (rank + 1) * 512]
            k = qkv[:, 2048 + rank * 512 : 2048 + (rank + 1) * 512]
            v = qkv[:, 4096 + rank * 1536 : 4096 + (rank + 1) * 1536]
            # Pad the two scalar groups to one tile each; runtime slices 12.
            bp = torch.nn.functional.pad(b[:, rank * 12 : (rank + 1) * 12], (0, 20))
            ap = torch.nn.functional.pad(a[:, rank * 12 : (rank + 1) * 12], (0, 20))
            chunks.append(torch.cat([q, k, v, z[:, rank * 1536 : (rank + 1) * 1536], bp, ap], -1))
        weights["linear_packed_decode"] = _shard_decode_weight(
            _device_major(chunks), -1, mesh_device=mesh_device, dtype=policy.linear_input_weight_dtype, k=5120, n=4160
        )
        linear_out = host("linear_attn.out_proj.weight", transpose=True)
        weights["linear_out_decode"] = _shard_decode_weight(
            linear_out, -2, mesh_device=mesh_device, dtype=policy.linear_output_weight_dtype, k=1536, n=5120
        )
        # Prefill's proven affine scan consumes the semantic projections separately.
        qkv_chunks = []
        for rank in range(4):
            q = qkv[:, rank * 512 : (rank + 1) * 512]
            k = qkv[:, 2048 + rank * 512 : 2048 + (rank + 1) * 512]
            v = qkv[:, 4096 + rank * 1536 : 4096 + (rank + 1) * 1536]
            qkv_chunks.append(torch.cat([q, k, v], -1))
        weights["in_qkv"] = _shard(
            _device_major(qkv_chunks), -1, mesh_device=mesh_device, dtype=policy.attention_weight_dtype
        )
        weights["in_z"] = _shard(z, -1, mesh_device=mesh_device, dtype=policy.attention_weight_dtype)
        weights["in_b"] = _shard(b, -1, mesh_device=mesh_device, dtype=policy.attention_weight_dtype)
        weights["in_a"] = _shard(a, -1, mesh_device=mesh_device, dtype=policy.attention_weight_dtype)
        weights["out_proj"] = _shard(linear_out, -2, mesh_device=mesh_device, dtype=policy.linear_output_weight_dtype)
        conv = host("linear_attn.conv1d.weight").reshape(1, 1, 10240, 4)
        conv_chunks = [
            torch.cat(
                [
                    conv[..., r * 512 : (r + 1) * 512, :],
                    conv[..., 2048 + r * 512 : 2048 + (r + 1) * 512, :],
                    conv[..., 4096 + r * 1536 : 4096 + (r + 1) * 1536, :],
                ],
                -2,
            )
            for r in range(4)
        ]
        weights["conv"] = _shard(torch.cat(conv_chunks, -2), -2, mesh_device=mesh_device)
        weights["dt_bias"] = _shard(
            host("linear_attn.dt_bias", dtype=torch.float32).reshape(1, 1, 1, vh),
            -1,
            mesh_device=mesh_device,
            dtype=ttnn.float32,
        )
        weights["a"] = _shard(
            (-host("linear_attn.A_log", dtype=torch.float32).float().exp()).reshape(1, 1, 1, vh),
            -1,
            mesh_device=mesh_device,
            dtype=ttnn.float32,
        )
        weights["gated_norm"] = _replicate(host("linear_attn.norm.weight"), mesh_device=mesh_device)
        weights["linear_identity"] = _replicate(
            torch.eye(vd, dtype=torch.bfloat16).reshape(1, 1, vd, vd), mesh_device=mesh_device
        )
        caches["conv"] = _shard(torch.zeros((1, batch, 10240, 4), dtype=torch.bfloat16), 2, mesh_device=mesh_device)
        caches["recurrent"] = _shard(
            torch.zeros((batch, vh, vd, vd), dtype=torch.bfloat16),
            1,
            mesh_device=mesh_device,
            dtype=policy.linear_recurrent_state_dtype,
        )

    def _configure_multichip_compute(self):
        def kernel(fidelity, fp32=False):
            return ttnn.init_device_compute_kernel_config(
                self.mesh_device.arch(),
                math_fidelity=fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32,
                packer_l1_acc=True,
            )

        p = self.policy
        self.attention_compute_kernel_config = kernel(p.attention_fidelity)
        self.qkv_compute_kernel_config = kernel(p.qkv_fidelity or p.attention_fidelity)
        self.o_compute_kernel_config = kernel(p.o_fidelity or p.attention_fidelity)
        self.mlp_compute_kernel_config = kernel(p.mlp_fidelity)
        self.linear_input_compute_kernel_config = kernel(p.linear_input_fidelity)
        self.linear_output_compute_kernel_config = kernel(p.linear_output_fidelity)
        self.linear_recurrent_compute_kernel_config = kernel(p.linear_recurrent_fidelity)
        self.norm_compute_kernel_config = kernel(ttnn.MathFidelity.HiFi4, True)
        self.decode_residual_memory_config = _l1_width_memory_config(rows=32, width=5120, cores=p.decode_storage_cores)
        self.decode_norm_memory_config = self.decode_residual_memory_config
        self.decode_norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(p.decode_storage_cores, 1),
            subblock_w=4,
            block_h=1,
            block_w=20,
            inplace=False,
        )
        self.decode_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0
        )

    def _all_reduce(self, tensor, *, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.all_reduce(
            tensor, num_links=1, topology=TARGET_TOPOLOGY, cluster_axis=1, memory_config=memory_config
        )

    def _reduce_scatter(self, tensor, *, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.reduce_scatter(
            tensor,
            dim=3,
            num_links=1,
            topology=TARGET_TOPOLOGY,
            cluster_axis=1,
            memory_config=memory_config,
        )

    def _tp_linear(
        self,
        hidden_states,
        weight_name,
        *,
        k,
        n,
        decode,
        row=False,
        fused_activation=None,
        compute_kernel_config=None,
        in0_block_w=1,
    ):
        kwargs = dict(
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config or self.mlp_compute_kernel_config,
        )
        if decode:
            input_memcfg = _l1_width_memory_config(rows=32, width=k, cores=8)
            if hidden_states.memory_config() != input_memcfg:
                hidden_states = ttnn.to_memory_config(hidden_states, input_memcfg)
            kwargs["memory_config"] = _l1_width_memory_config(rows=32, width=n, cores=8)
            kwargs["program_config"] = _decode_program(
                k=k, n=n, in0_block_w=in0_block_w, cores=8, fused_activation=fused_activation
            )
        else:
            rows = math.prod(tuple(hidden_states.padded_shape)[:-1])
            if self._multichip_candidate == "multichip_prefill_l1":
                hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
            if rows <= 2048:
                kwargs["program_config"] = _prefill_program(
                    rows=rows,
                    k=k,
                    n=n,
                    in0_block_w_limit=self.policy.prefill_in0_block_w,
                    grid_y=self.policy.prefill_grid_y,
                    fused_activation=fused_activation,
                )
        output = ttnn.linear(hidden_states, self.weights[weight_name], **kwargs)
        if not decode and fused_activation == ttnn.UnaryOpType.SILU:
            output = ttnn.silu(output)
        if not row:
            return output
        ccl_bfp8 = self._multichip_candidate == "multichip_bfp8_ccl_all"
        ccl_bfp8 |= self._multichip_candidate == "multichip_bfp8_ccl_attention" and weight_name in {
            "o_proj_decode",
            "linear_out_decode",
        }
        ccl_bfp8 |= self._multichip_candidate == "multichip_bfp8_ccl_mlp" and weight_name == "mlp_down_decode"
        if ccl_bfp8:
            output = ttnn.typecast(output, ttnn.bfloat8_b)
        if self._multichip_candidate == "multichip_preallocated_ccl" and decode:
            role = "mlp" if weight_name == "mlp_down_decode" else "token_mixer"
            buffers = self._ccl_buffers[role]
            scattered = ttnn.reduce_scatter(
                output,
                dim=3,
                num_links=1,
                topology=TARGET_TOPOLOGY,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tensor=buffers["reduce_scatter"],
            )
            return ttnn.all_gather(
                scattered,
                dim=3,
                num_links=1,
                topology=TARGET_TOPOLOGY,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tensor=buffers["all_gather"],
            )
        if self._row_collective_mode == "reduce_scatter":
            output = self._reduce_scatter(output)
        else:
            output = self._all_reduce(output)
        return ttnn.typecast(output, ttnn.bfloat16) if ccl_bfp8 else output

    def _distributed_rms_norm_decode(self, hidden_states, weight_name):
        """Normalize a semantic hidden-width mesh shard without replicating it."""
        stats = ttnn.rms_norm_pre_all_gather(hidden_states, dtype=ttnn.bfloat16)
        gathered_stats = ttnn.all_gather(
            stats,
            dim=3,
            num_links=1,
            topology=TARGET_TOPOLOGY,
            cluster_axis=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.rms_norm_post_all_gather(
            hidden_states,
            gathered_stats,
            epsilon=self.eps,
            weight=self.weights[weight_name],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _gather_fractured_hidden(self, hidden_states):
        return ttnn.all_gather(
            hidden_states,
            dim=3,
            num_links=1,
            topology=TARGET_TOPOLOGY,
            cluster_axis=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def decode_forward_fractured(self, *, hidden_states, page_table, current_positions):
        """Decode with a 1280-wide per-rank residual boundary.

        The caller owns the fractured input contract. The returned tensor stays
        fractured for the next decoder layer; comparison gathers belong in the
        test harness outside the measured stack.
        """
        if hidden_states.shape[-1] != 5120 // TARGET_TP:
            raise ValueError(f"fractured residual must have local width 1280, got {hidden_states.shape[-1]}")
        residual = hidden_states
        normalized = self._distributed_rms_norm_decode(residual, "input_norm_fractured")
        consumer_input = self._gather_fractured_hidden(normalized)
        previous_mode = self._row_collective_mode
        self._row_collective_mode = "reduce_scatter"
        try:
            mixed = self._token_mixer_decode(consumer_input, page_table, current_positions)
            residual = ttnn.add(residual, mixed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            normalized = self._distributed_rms_norm_decode(residual, "post_attention_norm_fractured")
            consumer_input = self._gather_fractured_hidden(normalized)
            mlp = self._mlp_decode(consumer_input)
            return ttnn.add(residual, mlp, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        finally:
            self._row_collective_mode = previous_mode

    def _mlp_decode(self, hidden_states):
        if self._multichip_candidate == "multichip_packed_mlp":
            packed = self._tp_linear(
                hidden_states,
                "mlp_gate_up_decode",
                k=5120,
                n=8704,
                decode=True,
                in0_block_w=4,
            )
            gate, up = ttnn.split(packed, (4352, 4352), dim=-1)
            product = ttnn.multiply(ttnn.silu(gate), up)
            return self._tp_linear(product, "mlp_down_decode", k=4352, n=5120, decode=True, row=True, in0_block_w=17)
        gate = self._tp_linear(
            hidden_states,
            "mlp_gate_decode",
            k=5120,
            n=4352,
            decode=True,
            fused_activation=ttnn.UnaryOpType.SILU,
            in0_block_w=4,
        )
        up = self._tp_linear(hidden_states, "mlp_up_decode", k=5120, n=4352, decode=True, in0_block_w=4)
        return self._tp_linear(
            ttnn.multiply(gate, up), "mlp_down_decode", k=4352, n=5120, decode=True, row=True, in0_block_w=17
        )

    def _mlp_prefill(self, hidden_states):
        sequence = hidden_states.shape[2]
        rows = math.prod(tuple(hidden_states.padded_shape)[:-1])
        if rows > 2048:
            chunk_sequence = max(32, (2048 // self.batch // 32) * 32)
            chunks = [
                self._mlp_prefill(
                    ttnn.slice(hidden_states, (0, 0, s, 0), (1, self.batch, min(sequence, s + chunk_sequence), 5120))
                )
                for s in range(0, sequence, chunk_sequence)
            ]
            return chunks[0] if len(chunks) == 1 else ttnn.concat(chunks, dim=2)
        gate = self._tp_linear(
            hidden_states, "mlp_gate_prefill", k=5120, n=4352, decode=False, fused_activation=ttnn.UnaryOpType.SILU
        )
        up = self._tp_linear(hidden_states, "mlp_up_prefill", k=5120, n=4352, decode=False)
        return self._tp_linear(ttnn.multiply(gate, up), "mlp_down_prefill", k=4352, n=5120, decode=False, row=True)

    def _linear_attention_decode(self, hidden_states):
        """Run gated-delta decode entirely on each device's owned heads."""
        key_heads, value_heads, key_dim, value_dim = 4, 12, 128, 128
        key_width, value_width, conv_width = 512, 1536, 2560
        # Two logical 12-wide scalar groups are independently tile padded.
        packed = self._tp_linear(
            hidden_states,
            "linear_packed_decode",
            k=5120,
            n=4160,
            decode=True,
            compute_kernel_config=self.linear_input_compute_kernel_config,
            in0_block_w=5,
        )
        # The recurrent path uses independently sharded state matmuls; do not
        # propagate the projection's eight-core L1 width layout into them.
        if self._multichip_candidate != "multichip_linear_packed_l1":
            packed = ttnn.to_memory_config(packed, ttnn.DRAM_MEMORY_CONFIG)
        mixed, z, beta_padded, decay_padded = ttnn.split(packed, (conv_width, value_width, 32, 32), dim=-1)
        if self._multichip_candidate == "multichip_linear_packed_l1":
            mixed, z, beta_padded, decay_padded = (
                ttnn.to_memory_config(part, ttnn.DRAM_MEMORY_CONFIG) for part in (mixed, z, beta_padded, decay_padded)
            )
        beta = beta_padded[..., :value_heads]
        decay = decay_padded[..., :value_heads]

        mixed = ttnn.permute(mixed, (0, 2, 3, 1))
        next_conv_state = ttnn.concat(
            [self.caches["conv"][..., 1:], mixed], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        mixed = ttnn.silu(ttnn.sum(ttnn.multiply(next_conv_state, self.weights["conv"]), dim=-1, keepdim=True))
        ttnn.copy(next_conv_state, self.caches["conv"])
        mixed = ttnn.permute(mixed, (0, 3, 1, 2))

        query = ttnn.reshape(mixed[..., :key_width], (self.batch, 1, key_heads, key_dim))
        key = ttnn.reshape(mixed[..., key_width : 2 * key_width], (self.batch, 1, key_heads, key_dim))
        value = ttnn.reshape(mixed[..., 2 * key_width :], (self.batch, 1, value_heads, value_dim))
        query = ttnn.repeat_interleave(ttnn.permute(query, (0, 2, 1, 3)), 3, dim=1)
        key = ttnn.repeat_interleave(ttnn.permute(key, (0, 2, 1, 3)), 3, dim=1)
        value = ttnn.permute(value, (0, 2, 1, 3))
        query = ttnn.multiply(self._l2_norm(query), key_dim**-0.5)
        key = self._l2_norm(key)

        beta = ttnn.sigmoid(beta)
        decay = ttnn.multiply(self.weights["a"], ttnn.softplus(ttnn.add(decay, self.weights["dt_bias"])))
        beta = ttnn.reshape(beta, (self.batch, value_heads, 1, 1))
        decay = ttnn.exp(ttnn.reshape(decay, (self.batch, value_heads, 1, 1)))

        state_dtype = self.policy.linear_recurrent_state_dtype
        recurrent_state = self.caches["recurrent"]
        if state_dtype != ttnn.float32:
            recurrent_state = ttnn.typecast(recurrent_state, ttnn.bfloat16)
            decay = ttnn.typecast(decay, ttnn.bfloat16)
            beta = ttnn.typecast(beta, ttnn.bfloat16)
        recurrent = ttnn.multiply(recurrent_state, decay)
        memory_value = self._linear_recurrent_matmul(key, recurrent)
        delta = ttnn.multiply(ttnn.subtract(value, memory_value), beta)
        update = ttnn.multiply(ttnn.transpose(key, -2, -1), delta, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        recurrent = ttnn.add(recurrent, update)
        output = self._linear_recurrent_matmul(query, recurrent)
        if state_dtype == ttnn.float32:
            ttnn.copy(recurrent, self.caches["recurrent"])
        else:
            ttnn.copy(ttnn.typecast(recurrent, state_dtype), self.caches["recurrent"])

        output = ttnn.rms_norm(
            output, epsilon=self.eps, weight=self.weights["gated_norm"], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        z = ttnn.reshape(z, (self.batch, value_heads, 1, value_dim))
        output = ttnn.multiply(output, ttnn.silu(z))
        output = ttnn.reshape(ttnn.permute(output, (2, 0, 1, 3)), (1, 1, self.batch, value_width))
        return self._tp_linear(
            output,
            "linear_out_decode",
            k=value_width,
            n=5120,
            decode=True,
            row=True,
            compute_kernel_config=self.linear_output_compute_kernel_config,
            in0_block_w=3,
        )

    def _linear_attention_prefill_chunk(self, hidden_states):
        """Local-head version of the optimized logarithmic affine scan."""
        key_heads, value_heads, key_dim, value_dim = 4, 12, 128, 128
        key_width, value_width = 512, 1536
        sequence = hidden_states.shape[2]
        groups = self.batch * value_heads

        mixed = self._tp_linear(
            hidden_states,
            "in_qkv",
            k=5120,
            n=2560,
            decode=False,
            compute_kernel_config=self.linear_input_compute_kernel_config,
        )
        z = self._tp_linear(
            hidden_states,
            "in_z",
            k=5120,
            n=value_width,
            decode=False,
            compute_kernel_config=self.linear_input_compute_kernel_config,
        )
        beta = self._tp_linear(
            hidden_states,
            "in_b",
            k=5120,
            n=value_heads,
            decode=False,
            compute_kernel_config=self.linear_input_compute_kernel_config,
        )
        decay = self._tp_linear(
            hidden_states,
            "in_a",
            k=5120,
            n=value_heads,
            decode=False,
            compute_kernel_config=self.linear_input_compute_kernel_config,
        )

        mixed = ttnn.permute(mixed, (0, 1, 3, 2))
        conv_input = ttnn.concat([self.caches["conv"], mixed], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        next_conv_state = conv_input[..., -self.caches["conv"].shape[-1] :]
        convolved = ttnn.multiply(conv_input[..., 1 : sequence + 1], self.weights["conv"][..., 0:1])
        for kernel_index in range(1, self.caches["conv"].shape[-1]):
            convolved = ttnn.add(
                convolved,
                ttnn.multiply(
                    conv_input[..., kernel_index + 1 : kernel_index + sequence + 1],
                    self.weights["conv"][..., kernel_index : kernel_index + 1],
                ),
            )
        ttnn.copy(next_conv_state, self.caches["conv"])
        mixed = ttnn.silu(ttnn.permute(convolved, (0, 1, 3, 2)))

        query = ttnn.reshape(mixed[..., :key_width], (self.batch, sequence, key_heads, key_dim))
        key = ttnn.reshape(mixed[..., key_width : 2 * key_width], (self.batch, sequence, key_heads, key_dim))
        value = ttnn.reshape(mixed[..., 2 * key_width :], (self.batch, sequence, value_heads, value_dim))
        query = ttnn.repeat_interleave(ttnn.permute(query, (0, 2, 1, 3)), 3, dim=1)
        key = ttnn.repeat_interleave(ttnn.permute(key, (0, 2, 1, 3)), 3, dim=1)
        value = ttnn.permute(value, (0, 2, 1, 3))
        query = ttnn.multiply(self._l2_norm(query), key_dim**-0.5)
        key = self._l2_norm(key)

        beta = ttnn.sigmoid(beta)
        decay = ttnn.multiply(self.weights["a"], ttnn.softplus(ttnn.add(decay, self.weights["dt_bias"])))
        beta = ttnn.permute(ttnn.reshape(beta, (self.batch, sequence, value_heads, 1)), (0, 2, 1, 3))
        decay = ttnn.exp(ttnn.permute(ttnn.reshape(decay, (self.batch, sequence, value_heads, 1)), (0, 2, 1, 3)))
        query = ttnn.reshape(query, (groups, sequence, 1, key_dim))
        key = ttnn.reshape(key, (groups, sequence, 1, key_dim))
        value = ttnn.reshape(value, (groups, sequence, 1, value_dim))
        beta = ttnn.typecast(ttnn.reshape(beta, (groups, sequence, 1, 1)), ttnn.bfloat16)
        decay = ttnn.typecast(ttnn.reshape(decay, (groups, sequence, 1, 1)), ttnn.bfloat16)

        identity = ttnn.repeat(self.weights["linear_identity"], ttnn.Shape([groups, sequence, 1, 1]))
        zero = ttnn.multiply(identity, 0.0)
        key_t = ttnn.transpose(key, -2, -1)
        transform = ttnn.multiply(
            decay,
            ttnn.subtract(identity, ttnn.multiply(beta, ttnn.matmul(key_t, key))),
        )
        bias = ttnn.multiply(beta, ttnn.matmul(key_t, value))
        distance = 1
        while distance < sequence:
            previous_transform = ttnn.concat([identity[:, :distance], transform[:, :-distance]], dim=1)
            previous_bias = ttnn.concat([zero[:, :distance], bias[:, :-distance]], dim=1)
            old_transform = transform
            transform = ttnn.matmul(old_transform, previous_transform)
            bias = ttnn.add(ttnn.matmul(old_transform, previous_bias), bias)
            distance *= 2

        initial = ttnn.typecast(self.caches["recurrent"], ttnn.bfloat16)
        initial = ttnn.repeat(ttnn.reshape(initial, (groups, 1, value_dim, value_dim)), ttnn.Shape([1, sequence, 1, 1]))
        states = ttnn.add(ttnn.matmul(transform, initial), bias)
        final_state = ttnn.reshape(states[:, -1:], (self.batch, value_heads, value_dim, value_dim))
        ttnn.copy(ttnn.typecast(final_state, self.policy.linear_recurrent_state_dtype), self.caches["recurrent"])

        output = ttnn.reshape(ttnn.matmul(query, states), (self.batch, value_heads, sequence, value_dim))
        output = ttnn.rms_norm(
            output, epsilon=self.eps, weight=self.weights["gated_norm"], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        z = ttnn.permute(ttnn.reshape(z, (self.batch, sequence, value_heads, value_dim)), (0, 2, 1, 3))
        output = ttnn.reshape(
            ttnn.permute(ttnn.multiply(output, ttnn.silu(z)), (0, 2, 1, 3)),
            (1, self.batch, sequence, value_width),
        )
        return self._tp_linear(
            output,
            "out_proj",
            k=value_width,
            n=5120,
            decode=False,
            row=True,
            compute_kernel_config=self.linear_output_compute_kernel_config,
        )

    def _full_attention_decode(self, hidden_states, page_table, current_positions):
        cache_positions = ttnn.typecast(current_positions, ttnn.int32)
        packed = self._tp_linear(
            hidden_states,
            "qkv_gate_decode",
            k=5120,
            n=3584,
            decode=True,
            compute_kernel_config=self.qkv_compute_kernel_config,
            in0_block_w=4,
        )
        qkv, gate = ttnn.split(packed, (2048, 1536), dim=-1)
        qkv = ttnn.to_memory_config(qkv, ttnn.L1_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=6, num_kv_heads=1, memory_config=self.decode_attention_memory_config
        )
        q = self._per_head_norm(q, "q_norm")
        k = self._per_head_norm(k, "k_norm")
        q = self._partial_rope_decode(q, current_positions)
        k = self._partial_rope_decode(k, current_positions)
        ttnn.experimental.paged_update_cache(
            self.caches["key"], k, update_idxs_tensor=cache_positions, page_table=page_table
        )
        ttnn.experimental.paged_update_cache(
            self.caches["value"], v, update_idxs_tensor=cache_positions, page_table=page_table
        )
        attention = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.caches["key"],
            self.caches["value"],
            cur_pos_tensor=cache_positions,
            page_table_tensor=page_table,
            scale=self.head_dim**-0.5,
            program_config=self.decode_sdpa_program_config,
            compute_kernel_config=self.attention_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.to_memory_config(attention, self.decode_attention_memory_config)
        attention = ttnn.experimental.nlp_concat_heads_decode(attention, num_heads=6)
        attention = ttnn.multiply(attention, ttnn.sigmoid(gate))
        partial = self._tp_linear(
            attention,
            "o_proj_decode",
            k=1536,
            n=5120,
            decode=True,
            row=True,
            compute_kernel_config=self.o_compute_kernel_config,
            in0_block_w=3,
        )
        output_width = 5120 // TARGET_TP if self._row_collective_mode == "reduce_scatter" else 5120
        return ttnn.reshape(partial, (1, 1, self.batch, output_width), (1, 1, 32, output_width))

    def _full_attention_prefill(self, hidden_states, page_table, current_positions):
        sequence = hidden_states.shape[2]
        if sequence > 32768:
            return self._full_attention_prefill_long(hidden_states, page_table, current_positions)
        packed = self._tp_linear(
            hidden_states,
            "qkv_gate_prefill",
            k=5120,
            n=3584,
            decode=False,
            compute_kernel_config=self.qkv_compute_kernel_config,
        )
        qkv, gate = ttnn.split(packed, (2048, 1536), dim=-1)
        q, k, v = ttnn.split(qkv, (1536, 256, 256), dim=-1)
        q = ttnn.permute(ttnn.reshape(q, (self.batch, sequence, 6, 256)), (0, 2, 1, 3))
        k = ttnn.permute(ttnn.reshape(k, (self.batch, sequence, 1, 256)), (0, 2, 1, 3))
        v = ttnn.permute(ttnn.reshape(v, (self.batch, sequence, 1, 256)), (0, 2, 1, 3))
        q = self._partial_rope_prefill(self._per_head_norm_prefill(q, "q_norm"), current_positions)
        k = self._partial_rope_prefill(self._per_head_norm_prefill(k, "k_norm"), current_positions)
        ttnn.experimental.paged_fill_cache(
            self.caches["key"],
            ttnn.typecast(k, self.policy.cache_dtype),
            page_table,
            batch_idx_tensor=self.caches["batch_indices"],
        )
        ttnn.experimental.paged_fill_cache(
            self.caches["value"],
            ttnn.typecast(v, self.policy.cache_dtype),
            page_table,
            batch_idx_tensor=self.caches["batch_indices"],
        )
        attention = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=256**-0.5,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8), exp_approx_mode=False, q_chunk_size=64, k_chunk_size=64
            ),
            compute_kernel_config=self.attention_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.experimental.nlp_concat_heads(attention, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.permute(attention, (1, 0, 2, 3))
        return self._tp_linear(
            ttnn.multiply(attention, ttnn.sigmoid(gate)),
            "o_proj_prefill",
            k=1536,
            n=5120,
            decode=False,
            row=True,
            compute_kernel_config=self.o_compute_kernel_config,
        )

    def _full_attention_prefill_long(self, hidden_states, page_table, current_positions):
        """TP-local, memory-bounded attention with an arbitrary logical tail."""
        sequence, chunk_size = hidden_states.shape[2], 32768
        page_size = self.caches["key"].shape[2]
        for start in range(0, sequence, chunk_size):
            end = min(sequence, start + chunk_size)
            length = end - start
            hidden = ttnn.slice(hidden_states, (0, 0, start, 0), (1, self.batch, end, 5120))
            positions = ttnn.slice(current_positions, (0, start), (self.batch, end))
            pages = ttnn.slice(page_table, (0, start // page_size), (self.batch, math.ceil(end / page_size)))
            for name in ("k", "v"):
                value = self._tp_linear(
                    hidden,
                    f"{name}_prefill_long",
                    k=5120,
                    n=256,
                    decode=False,
                    compute_kernel_config=self.qkv_compute_kernel_config,
                )
                value = ttnn.permute(ttnn.reshape(value, (self.batch, length, 1, 256)), (0, 2, 1, 3))
                if name == "k":
                    value = self._partial_rope_prefill(self._per_head_norm_prefill(value, "k_norm"), positions)
                value = ttnn.typecast(value, self.policy.cache_dtype)
                ttnn.experimental.paged_fill_cache(
                    self.caches["key" if name == "k" else "value"],
                    value,
                    pages,
                    batch_idx_tensor=self.caches["batch_indices"],
                )
                ttnn.deallocate(value)

        outputs = []
        for start in range(0, sequence, chunk_size):
            end = min(sequence, start + chunk_size)
            length = end - start
            hidden = ttnn.slice(hidden_states, (0, 0, start, 0), (1, self.batch, end, 5120))
            positions = ttnn.slice(current_positions, (0, start), (self.batch, end))
            q = self._tp_linear(
                hidden,
                "q_prefill_long",
                k=5120,
                n=1536,
                decode=False,
                compute_kernel_config=self.qkv_compute_kernel_config,
            )
            q = ttnn.permute(ttnn.reshape(q, (self.batch, length, 6, 256)), (0, 2, 1, 3))
            q = self._partial_rope_prefill(self._per_head_norm_prefill(q, "q_norm"), positions)
            padding = (-length) % ttnn.TILE_SIZE
            if padding:
                q = ttnn.pad(q, ((0, 0), (0, 0), (0, padding), (0, 0)), value=0.0)
            attention = ttnn.transformer.chunked_scaled_dot_product_attention(
                q,
                self.caches["key"],
                self.caches["value"],
                page_table,
                chunk_start_idx=start,
                scale=256**-0.5,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(q)
            if padding:
                attention = ttnn.slice(attention, (0, 0, 0, 0), (self.batch, 6, length, 256))
            attention = ttnn.permute(
                ttnn.experimental.nlp_concat_heads(attention, memory_config=ttnn.DRAM_MEMORY_CONFIG), (1, 0, 2, 3)
            )
            gate = self._tp_linear(
                hidden,
                "gate_prefill_long",
                k=5120,
                n=1536,
                decode=False,
                compute_kernel_config=self.qkv_compute_kernel_config,
            )
            outputs.append(
                self._tp_linear(
                    ttnn.multiply(attention, ttnn.sigmoid(gate)),
                    "o_proj_prefill",
                    k=1536,
                    n=5120,
                    decode=False,
                    row=True,
                    compute_kernel_config=self.o_compute_kernel_config,
                )
            )
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)


__all__ = ["MultichipDecoder", "TARGET_FABRIC", "TARGET_MESH_SHAPE", "TARGET_TOPOLOGY", "TARGET_TP"]
