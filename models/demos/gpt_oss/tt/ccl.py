# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn


class CCLManager:
    def __init__(self, mesh_device, num_links, topology=ttnn.Topology.Ring):
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology

        # Cache for ping pong buffers: key = (shape_tuple, dim, mesh_axis), value = [buffer1, buffer2]
        self._ping_pong_buffer_cache = {}
        self._ping_pong_buffer_indices = {}
        self._decode_boundary = {}
        self._decode_expert_stream = {}
        self._decode_stream_buffers = {}

        # Setup semaphores
        self._init_subdevice()

        # Initialize semaphores for reduce scatter and all gather
        self._init_semaphores()
        self.rs_ping_pong_idx = 0
        self.ag_ping_pong_idx = 0
        self.barrier_idx = 0

    def _init_subdevice(self):
        compute_grid_size = ttnn.CoreCoord(8, 8)
        self.ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )

        _worker_sub_device = ttnn.SubDevice(
            [
                self.ccl_cores,
            ]
        )
        self.ccl_sub_device_id = ttnn.SubDeviceId(0)

    def _init_semaphores(self):
        # Initialize semaphores for reduce scatter ping pong
        rs_n_sems = 3 * 2  # 3 semaphores * 2 for ping pong
        self.rs_ping_pong_semaphores = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(rs_n_sems)
        ]

        # Initialize semaphores for all gather ping pong
        ag_n_sems = 2 * 2  # 2 semaphores * 2 for ping pong (2 buffers)
        self.ag_ping_pong_semaphores = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(ag_n_sems)
        ]

        # Initialize barrier semaphores
        barrier_ns_sems = 2 * 1
        self.barrier_semaphore = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(barrier_ns_sems)
        ]

    def get_rs_ping_pong_semaphore(self):
        """
        Get semaphores for reduce scatter ping pong operations.

        Returns:
            List of 3 semaphores for the current ping pong cycle
        """
        cur_idx = self.rs_ping_pong_idx
        n_sems = 3
        self.rs_ping_pong_idx = (cur_idx + 1) % 2
        return self.rs_ping_pong_semaphores[cur_idx * n_sems : (cur_idx + 1) * n_sems]

    def get_ag_ping_pong_semaphore(self):
        """
        Get semaphores for all gather ping pong operations.

        Returns:
            List of 3 semaphores for the current ping pong cycle
        """
        cur_idx = self.ag_ping_pong_idx
        n_sems = 2
        self.ag_ping_pong_idx = (cur_idx + 1) % 2
        return self.ag_ping_pong_semaphores[cur_idx * n_sems : (cur_idx + 1) * n_sems]

    def get_barrier_semaphore(self):
        """
        Get semaphores for barrier operations.
        """
        cur_idx = self.barrier_idx
        self.barrier_idx = (cur_idx + 1) % 2
        return self.barrier_semaphore[cur_idx]

    def get_decode_boundary(self, hidden_size, eps=None, cluster_axis=None):
        """Layer boundary op (all-reduce + residual add + RMSNorm, decode_boundary.py) shared by every decoder
        layer's decode path, with its persistent receive buffers and semaphores. Created by the first decoder layer
        (eps, cluster_axis); the producer ops look it up by hidden size."""
        from .decode_boundary import DecodeBoundary

        if hidden_size not in self._decode_boundary:
            self._decode_boundary[hidden_size] = DecodeBoundary(self.mesh_device, hidden_size, eps, cluster_axis)
        return self._decode_boundary[hidden_size]

    def decode_boundary_send(self, hidden_size, site):
        """`send` argument of the producer stream ops: fuse the boundary's fabric send (None: the boundary sends)."""
        from .fused_decode import DECODE_BOUNDARY_CCL, DECODE_BOUNDARY_FUSED_SEND

        if DECODE_BOUNDARY_FUSED_SEND and DECODE_BOUNDARY_CCL == "fabric":
            return (self.get_decode_boundary(hidden_size), site)
        return None

    def get_decode_expert_stream(self, hidden, inter_pad, top_k, swiglu_limit, alpha):
        """Routed-expert stream ops (experts/stream.py) and their persistent buffers, shared by every decoder layer's
        decode path: the [k, I_pad] BF16 row-major activation and the flat partial sum (get_decode_partial)."""
        key = (hidden, inter_pad, top_k)
        if key not in self._decode_expert_stream:
            from .experts.stream import ExpertDownStream, ExpertGateUpStream
            from .fused_decode import DOWN_STREAM_READERS, GATE_UP_STREAM_READERS

            self._decode_expert_stream[key] = {
                "gate_up": ExpertGateUpStream(
                    self.mesh_device, hidden, inter_pad, top_k, swiglu_limit, alpha, readers=GATE_UP_STREAM_READERS
                ),
                "down": ExpertDownStream(self.mesh_device, hidden, inter_pad, top_k, readers=DOWN_STREAM_READERS),
                "act": self._persistent_zeros((top_k, inter_pad), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
                "partial": self.get_decode_partial(hidden),
            }
        return self._decode_expert_stream[key]

    def _persistent_zeros(self, shape, dtype, memory_config):
        import torch

        return ttnn.from_torch(
            torch.zeros(shape, dtype=torch.int32 if dtype == ttnn.uint16 else torch.float32),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT if len(shape) > 2 else ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def get_decode_partial(self, hidden):
        """Flat BF16 partial sum (decode_boundary.py: value h at byte 2 h of a [1, 1, 32, 32 * pages] tile tensor on
        the boundary core) written by the streamed o_proj and MoE down of every layer and all-reduced by the layer
        boundaries; the padding past hidden is never written and stays zero."""
        from .decode_boundary import flat_memory_config, flat_shape

        key = ("partial", hidden)
        if key not in self._decode_stream_buffers:
            self._decode_stream_buffers[key] = self._persistent_zeros(
                tuple(flat_shape(hidden)), ttnn.bfloat16, flat_memory_config(self.mesh_device, hidden)
            )
        return self._decode_stream_buffers[key]

    def get_decode_qkv_heads(self, num_heads, num_kv_heads, head_dim):
        """BF16 Q / K / V head tensors ([1, 1, heads, head_dim], head h in row h; Q and V on core (0, 0), K on (1, 0):
        the one-user layout nlp_create_qkv_heads_decode(overlap_qk_coregrid=False) produces), written by the streamed
        QKV of every layer (padding head rows stay zero)."""
        key = ("qkv_heads", num_heads, num_kv_heads, head_dim)
        if key not in self._decode_stream_buffers:

            def heads(n, core):
                memory_config = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet({ttnn.CoreRange(core, core)}),
                        (-(-n // ttnn.TILE_SIZE) * ttnn.TILE_SIZE, head_dim),
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                )
                return self._persistent_zeros((1, 1, n, head_dim), ttnn.bfloat16, memory_config)

            self._decode_stream_buffers[key] = (
                heads(num_heads, ttnn.CoreCoord(0, 0)),
                heads(num_kv_heads, ttnn.CoreCoord(1, 0)),
                heads(num_kv_heads, ttnn.CoreCoord(0, 0)),
            )
        return self._decode_stream_buffers[key]

    def get_decode_router_out(self):
        """UINT16 ids / BF16 scores [1, 32] row-major buffers holding the routed top-k (first k entries), written by
        the streamed router of every layer and read by the expert streams."""
        key = ("router_out",)
        if key not in self._decode_stream_buffers:
            self._decode_stream_buffers[key] = (
                self._persistent_zeros((1, 32), ttnn.uint16, ttnn.L1_MEMORY_CONFIG),
                self._persistent_zeros((1, 32), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
            )
        return self._decode_stream_buffers[key]

    def get_decode_linear_stream(self, name, *args, **kwargs):
        """LinearStream op (experts/stream.py) for one decode role, shared by every layer."""
        from .experts.stream import LinearStream

        if name not in self._decode_stream_buffers:
            self._decode_stream_buffers[name] = LinearStream(self.mesh_device, *args, **kwargs)
        return self._decode_stream_buffers[name]

    def reset_global_semaphores(self):
        """Reset all global semaphores to 0"""
        for sem in self.rs_ping_pong_semaphores:
            ttnn.reset_global_semaphore_value(sem, 0)
        for sem in self.ag_ping_pong_semaphores:
            ttnn.reset_global_semaphore_value(sem, 0)
