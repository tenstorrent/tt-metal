# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Attention: `MoonViTEncoderLayer.attention_qkvpacked` in the reference.

wqkv -> split into 12 heads of 128 for q, k, v (`nlp_create_qkv_heads`) -> RoPE on q and k
(`rotary_embedding_llama`, see `rope.py`) -> non-causal ring attention over the sequence split
along `sp_axis` (`ring_joint_scaled_dot_product_attention`, `logical_n` masking the padding) ->
merge heads (`nlp_concat_heads`) -> wo. Scale is `head_dim ** -0.5`.

`logical_n` is a device tensor (`logical_length_tensor`), not an int. As an int the op compiles it
into the kernel and its program cache key, so every image size would build new programs and could
not replay one trace. The tensor is read on device each call.

The ring attention runs on all but the last column of the worker grid; the last column runs the
K/V all-gather that feeds it (`ccl_core_grid_offset`, column-major CCL cores). Q and K chunks of 256
rows divide every chip's share of the padded sequence. With fp32 accumulation, K chunks of 512 need
more L1 for circular buffers than a Blackhole core has.
"""

from __future__ import annotations

import torch

import ttnn

from .linear import Linear
from .model_config import MoonViTConfig, VisionParallel
from .module import Module
from .rope import RopeTensors

Q_CHUNK_SIZE = 256
K_CHUNK_SIZE = 256


def logical_length_tensor(mesh_device: ttnn.MeshDevice, n: int) -> ttnn.Tensor:
    """The unpadded patch count as the `logical_n` the ring attention reads: [1, 1, 1, 1] uint32,
    row major, replicated."""
    return ttnn.from_torch(
        torch.tensor([n], dtype=torch.int64).reshape(1, 1, 1, 1),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


class TT_Vision_Attention(Module):
    def __init__(
        self, config: MoonViTConfig, *, mesh_device: ttnn.MeshDevice, parallel: VisionParallel, compute_kernel_config
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.parallel = parallel
        self.scale = config.head_dim**-0.5
        kwargs = dict(bias=config.attn_bias, mesh_device=mesh_device, compute_kernel_config=compute_kernel_config)
        self.wqkv = Linear(config.hidden_size, 3 * config.qkv_hidden_size, **kwargs)
        self.wo = Linear(config.qkv_hidden_size, config.hidden_size, **kwargs)
        self.compute_kernel_config = compute_kernel_config
        # Built on the first forward: they read the device's core grid, and construction stays
        # device-free (test_module.py builds the tree on a stand-in device).
        self._program_config: ttnn.SDPAProgramConfig | None = None
        self._ccl_core_grid_offset: ttnn.CoreCoord | None = None

    def _sdpa_grid(self) -> tuple[ttnn.SDPAProgramConfig, ttnn.CoreCoord]:
        """The SDPA program config on every worker column but the last, and the offset of that last
        column, where the K/V all-gather runs."""
        if self._program_config is None:
            grid = self.mesh_device.compute_with_storage_grid_size()
            self._ccl_core_grid_offset = ttnn.CoreCoord(grid.x - 1, 0)
            self._program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(grid.x - 1, grid.y),
                q_chunk_size=Q_CHUNK_SIZE,
                k_chunk_size=K_CHUNK_SIZE,
                exp_approx_mode=False,
            )
        return self._program_config, self._ccl_core_grid_offset

    def forward(self, x: ttnn.Tensor, rope: RopeTensors, logical_n: ttnn.Tensor) -> ttnn.Tensor:
        """[1, 1, rows, hidden_size] -> the same shape; rows are this device's shard of the padded,
        window-ordered sequence. `logical_n` holds the real (unpadded) patch count."""
        if not self.parallel.sp:
            raise ValueError("TT_Vision_Attention needs the sequence split over a ring (sp_factor > 1)")
        qkv = self.wqkv(x)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_attention_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # each [1, heads, rows, head_dim]
        ttnn.deallocate(qkv)

        q_rot = ttnn.experimental.rotary_embedding_llama(q, rope.cos, rope.sin, rope.trans_mat, is_decode_mode=False)
        k_rot = ttnn.experimental.rotary_embedding_llama(k, rope.cos, rope.sin, rope.trans_mat, is_decode_mode=False)
        ttnn.deallocate(q)
        ttnn.deallocate(k)

        attn = self._ring_attention(q_rot, k_rot, v, logical_n)
        ttnn.deallocate(q_rot)
        ttnn.deallocate(k_rot)
        ttnn.deallocate(v)

        merged = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, 1, rows, qkv]
        ttnn.deallocate(attn)
        out = self.wo(merged)
        ttnn.deallocate(merged)
        return out

    def _ring_attention(self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor, logical_n: ttnn.Tensor) -> ttnn.Tensor:
        """Non-causal attention of the local Q rows over the whole ring's K and V, padding masked."""
        ccl = self.parallel.ccl
        sp_axis = self.parallel.sp_axis
        program_config, ccl_core_grid_offset = self._sdpa_grid()
        k_buffer, v_buffer = ccl.ring_attention_buffers(k.shape, 2, sp_axis)
        out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            q,
            k,
            v,
            persistent_output_buffer_k=k_buffer,
            persistent_output_buffer_v=v_buffer,
            joint_strategy="rear",
            logical_n=logical_n,
            program_config=program_config,
            scale=self.scale,
            compute_kernel_config=self.compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=ccl.ring_attention_semaphores(),
            num_links=ccl.num_links,
            cluster_axis=sp_axis,
            mesh_device=self.mesh_device,
            topology=ccl.topology,
            subdevice_id=ccl.sub_device_id,
            ccl_core_grid_offset=ccl_core_grid_offset,
            use_column_major_ccl=True,
            is_causal=False,
            is_balanced=False,
        )
        return out
