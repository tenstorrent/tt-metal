# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Build DFlash context K/V from streamed Gemma decoder outputs."""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.gemma4_d_p.tt.attention.operations import apply_per_head_norm
from models.demos.gemma4_d_p.tt.attention.ring_prefill import _allocate_migration_ring_cache
from models.demos.gemma4_d_p.tt.ccl import ccl_allgather
from models.demos.gemma4_d_p.tt.dflash_config import DFlashConfig, dflash_tensor_cache_path


@dataclass(frozen=True)
class DFlashKVCache:
    k: ttnn.Tensor
    v: ttnn.Tensor
    config: DFlashConfig
    num_users: int
    max_seq_len: int


def allocate_dflash_kv_cache(mesh_config, config, *, num_users, max_seq_len):
    config.validate(mesh_config, max_seq_len)
    if num_users <= 0:
        raise ValueError("DFlash requires at least one user slot")
    shape = [
        num_users * config.num_hidden_layers,
        config.num_key_value_heads // mesh_config.tp_degree,
        max_seq_len // mesh_config.cp_degree,
        config.head_dim,
    ]
    k, v = (
        _allocate_migration_ring_cache(mesh_config.device, shape, ttnn.bfloat8_b, config.head_dim) for _ in range(2)
    )
    return DFlashKVCache(k, v, config, num_users, max_seq_len)


class DFlashPrefill:
    def __init__(self, mesh_config, config, state_dict, ccl_manager, kv_cache, *, max_seq_len, chunk_size):
        config.validate(mesh_config, max_seq_len)
        self.mesh_config = mesh_config
        self.config = config
        self.ccl_manager = ccl_manager
        self.kv_cache = kv_cache
        self.chunk_size = chunk_size
        self.max_seq_len = max_seq_len
        self.local_heads = config.num_key_value_heads // mesh_config.tp_degree
        if kv_cache.config != config or kv_cache.max_seq_len < max_seq_len:
            raise ValueError("DFlash cache configuration or capacity does not match the model")
        expected_shape = (
            kv_cache.num_users * config.num_hidden_layers,
            self.local_heads,
            kv_cache.max_seq_len // mesh_config.cp_degree,
            config.head_dim,
        )
        for tensor in (kv_cache.k, kv_cache.v):
            if tuple(tensor.shape) != expected_shape or tensor.dtype != ttnn.bfloat8_b:
                raise ValueError(f"DFlash cache must be BFP8 with per-device shape {expected_shape}")
        if chunk_size <= 0 or chunk_size % (mesh_config.cp_degree * ttnn.TILE_SIZE) or max_seq_len % chunk_size:
            raise ValueError("DFlash requires whole CP-local tiles and whole cache chunks")
        mesh_device = mesh_config.device
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        cache_path = dflash_tensor_cache_path(config, state_dict, mesh_config.mesh_shape)
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)

        def weight(value, name, mapper=replicate):
            return ttnn.as_tensor(
                value.contiguous(),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
                cache_file_name=str(cache_path / name),
            )

        self.fc = {}
        for index, layer_id in enumerate(config.target_layer_ids):
            block = state_dict["fc.weight"][:, index * config.hidden_size : (index + 1) * config.hidden_size]
            self.fc[layer_id] = weight(block.T, f"fc_{layer_id}", mesh_config.column_parallel())
        self.hidden_norm = weight(state_dict["hidden_norm.weight"].reshape(1, 1, 1, -1), "hidden_norm")
        self.k_proj, self.v_proj, self.k_norm = [], [], []
        for layer in range(config.num_hidden_layers):
            prefix = f"layers.{layer}.self_attn"
            self.k_proj.append(
                weight(state_dict[f"{prefix}.k_proj.weight"].T, f"k_{layer}", mesh_config.column_parallel())
            )
            self.v_proj.append(
                weight(state_dict[f"{prefix}.v_proj.weight"].T, f"v_{layer}", mesh_config.column_parallel())
            )
            self.k_norm.append(weight(state_dict[f"{prefix}.k_norm.weight"].reshape(1, 1, 1, -1), f"k_norm_{layer}"))

        inv_freq = config.rope_theta ** (-torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        angles = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
        angles = torch.cat((angles, angles), dim=-1)
        self.rope = tuple(
            ttnn.from_torch(
                values.to(torch.bfloat16),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate,
            )
            for values in (angles.cos(), angles.sin())
        )
        dims = [None, None]
        dims[mesh_config.cp_axis] = 1
        self.position_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_config.mesh_shape, dims=dims)
        self.positions = ttnn.to_device(self._host_positions(0), mesh_device)

    def _host_positions(self, start):
        return ttnn.from_torch(
            torch.arange(start, start + self.chunk_size, dtype=torch.int64).reshape(1, -1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self.position_mapper,
        )

    def stage_positions(self, start):
        if start < 0 or start + self.chunk_size > self.max_seq_len:
            raise ValueError("DFlash positions are outside the configured context")
        ttnn.copy_host_to_device_tensor(self._host_positions(start), self.positions)

    def tap(self, hidden_states, layer_idx, accumulator):
        """Consume a target output without retaining or deallocating it."""
        projected = ttnn.linear(
            hidden_states,
            self.fc[layer_idx],
            compute_kernel_config=self.compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if accumulator is None:
            return projected
        combined = ttnn.add(accumulator, projected, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        accumulator.deallocate(True)
        projected.deallocate(True)
        return combined

    def _heads(self, projection):
        heads, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            projection,
            num_heads=self.local_heads,
            num_kv_heads=0,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        projection.deallocate(True)
        return heads

    def write_kv(self, accumulator, metadata, *, positions=None, on_layer_complete=None):
        """Finish the context projection, then write and acknowledge each draft layer."""
        context = ccl_allgather(accumulator, self.mesh_config, self.ccl_manager)
        normed = ttnn.rms_norm(
            context,
            weight=self.hidden_norm,
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=self.compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        context.deallocate(True)
        positions = self.positions if positions is None else positions
        cos, sin = (
            ttnn.unsqueeze_to_4D(ttnn.embedding(positions, table, layout=ttnn.TILE_LAYOUT)) for table in self.rope
        )
        for layer in range(self.config.num_hidden_layers):
            k, v = (
                self._heads(
                    ttnn.linear(
                        normed,
                        weights[layer],
                        compute_kernel_config=self.compute_config,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                )
                for weights in (self.k_proj, self.v_proj)
            )
            k_normed = apply_per_head_norm(
                k, self.config.rms_norm_eps, weight=self.k_norm[layer], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            k.deallocate(True)
            rotated = ttnn.experimental.rotary_embedding(k_normed, cos, sin, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            k_normed.deallocate(True)
            for cache, value in ((self.kv_cache.k, rotated), (self.kv_cache.v, v)):
                packed = ttnn.typecast(value, cache.dtype)
                ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                    cache=cache,
                    input=packed,
                    slot_idx=metadata.slot_idx,
                    layer_idx=layer,
                    num_layers=self.config.num_hidden_layers,
                    kv_actual_global=metadata.kv_actual_global,
                    cluster_axis=self.mesh_config.cp_axis,
                )
                packed.deallocate(True)
                value.deallocate(True)
            if on_layer_complete is not None:
                on_layer_complete(layer)
        cos.deallocate(True)
        sin.deallocate(True)
        normed.deallocate(True)
