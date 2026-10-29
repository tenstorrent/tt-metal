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
        # boundary call sites; the model adds "entry" for a vocab-sharded embedding (fused_decode/inputs.py)
        self.decode_boundary_sites = ("attn", "moe")

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

    def get_batched_decode_all_reduce(self, hidden_size, cluster_axis):
        """Persistent one-op all-reduce shared by every layer of the batched decode path (batched_decode.py)."""
        from .batched_decode import DecodeAllReduce

        cache = self.__dict__.setdefault("_batched_decode_all_reduce", {})
        key = (hidden_size, cluster_axis)
        if key not in cache:
            cache[key] = DecodeAllReduce(self.mesh_device, hidden_size, cluster_axis, self.topology)
        return cache[key]

    def get_decode_boundary(self, hidden_size, eps=None, cluster_axis=None):
        """Layer boundary op (all-reduce + residual add + RMSNorm, fused_decode/boundary.py) shared by every decoder
        layer's decode path, with its persistent receive buffers and semaphores. Created by the first decoder layer
        (eps, cluster_axis); the producer ops look it up by hidden size."""
        from .fused_decode.boundary import DecodeBoundary

        if hidden_size not in self._decode_boundary:
            self._decode_boundary[hidden_size] = DecodeBoundary(
                self.mesh_device, hidden_size, eps, cluster_axis, sites=self.decode_boundary_sites
            )
        return self._decode_boundary[hidden_size]

    def decode_boundary_send(self, hidden_size, site):
        """`send` argument of the producer stream ops: fuse the boundary's fabric send (None: the boundary sends)."""
        from .fused_decode.config import DECODE_BOUNDARY_CCL, DECODE_BOUNDARY_FUSED_SEND

        if DECODE_BOUNDARY_FUSED_SEND and DECODE_BOUNDARY_CCL == "fabric":
            boundary = self.get_decode_boundary(hidden_size)
            # TP=1: nothing to all-reduce (the boundary reads the producer's partial directly)
            return (boundary, site) if boundary.ring > 1 else None
        return None

    def get_decode_expert_stream(
        self, hidden, inter_pad, top_k, swiglu_limit, alpha, weight_dtype=ttnn.bfloat4_b, k_grouped=False
    ):
        """Routed-expert stream ops (fused_decode/stream.py) and their persistent buffers, shared by every decoder
        layer's decode path: the [k, I_pad] BF16 row-major activation and the flat partial sum (get_decode_partial).
        k_grouped: the weights' tiles are stored transposed (K-grouped BFP4, ../bfp4_quant.py)."""
        key = (hidden, inter_pad, top_k, weight_dtype, k_grouped)
        if key not in self._decode_expert_stream:
            from .fused_decode.config import DOWN_STREAM_READERS, GATE_UP_STREAM_READERS
            from .fused_decode.stream import ExpertDownStream, ExpertGateUpStream

            self._decode_expert_stream[key] = {
                "gate_up": ExpertGateUpStream(
                    self.mesh_device,
                    hidden,
                    inter_pad,
                    top_k,
                    swiglu_limit,
                    alpha,
                    readers=GATE_UP_STREAM_READERS,
                    weight_dtype=weight_dtype,
                    k_grouped=k_grouped,
                ),
                "down": ExpertDownStream(
                    self.mesh_device,
                    hidden,
                    inter_pad,
                    top_k,
                    readers=DOWN_STREAM_READERS,
                    weight_dtype=weight_dtype,
                    k_grouped=k_grouped,
                ),
                "act": self._persistent_zeros((top_k, inter_pad), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
                "partial": self.get_decode_partial(hidden),
            }
        return self._decode_expert_stream[key]

    def get_decode_expert_nd(
        self, hidden, inter, top_k, swiglu_limit, alpha, weight_dtype, down_dtype=None, bias_pack=1
    ):
        """Routed-expert ops (fused_decode/nd.py: TP=1, gpt-oss-120b at TP=4) streaming the ND-sharded expert tensors,
        shared by every decoder layer, with the [k, I] BF16 row-major activation (I: this device's intermediate shard)
        and the flat partial sum (get_decode_partial). bias_pack: columns per bias tile (nd.py ND_BIAS_PACK)."""
        down_dtype = down_dtype or weight_dtype
        key = ("nd", hidden, inter, top_k, weight_dtype, down_dtype, bias_pack)
        if key not in self._decode_expert_stream:
            from .fused_decode.nd import NdExpertDown, NdExpertGateUp

            self._decode_expert_stream[key] = {
                "gate_up": NdExpertGateUp(
                    self.mesh_device,
                    hidden,
                    inter,
                    top_k,
                    swiglu_limit,
                    alpha,
                    weight_dtype=weight_dtype,
                    bias_pack=bias_pack,
                ),
                "down": NdExpertDown(
                    self.mesh_device, hidden, inter, top_k, weight_dtype=down_dtype, bias_pack=bias_pack
                ),
                "act": self._persistent_zeros((top_k, inter), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
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
        """Flat BF16 partial sum (fused_decode/boundary.py: value h at byte 2 h of a [1, 1, 32, 32 * pages] tile tensor
        on the boundary core) written by the streamed o_proj and MoE down of every layer and all-reduced by the layer
        boundaries; the padding past hidden is never written and stays zero."""
        from .fused_decode.boundary import flat_memory_config, flat_shape

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

    def get_decode_sda_out(self, num_heads, head_dim):
        """The split decode attention's output [1, 1, heads, head_dim] (BF16, L1) and constants (a BFP8 ones tile),
        shared by every layer and slice configuration: allocated at layer construction, never while a trace is live
        (fused_decode/split_attn.py)."""
        key = ("sda_out", num_heads, head_dim)
        if key not in self._decode_stream_buffers:
            from .fused_decode.split_attn import consts_tensor, output_tensor

            self._decode_stream_buffers[key] = (
                output_tensor(self.mesh_device, num_heads, head_dim),
                consts_tensor(self.mesh_device),
            )
        return self._decode_stream_buffers[key]

    def get_decode_split_attention(self, num_heads, num_kv_heads, head_dim, window, qkv_stream):
        """The split decode attention of a layer type (window 0: full attention), fused into the decode QKV program
        `qkv_stream` (fused_decode/split_attn.py): its slice cores are the cores that program leaves free (not the QKV
        stream's workers, the boundary core, the Q / K head buffers' cores (0, 0), (1, 0)) as few rectangles as
        possible. Slices per kv head and min chunks per slice: fused_decode/config.py sda_slices."""
        from .fused_decode.boundary import BOUNDARY_CORE
        from .fused_decode.config import sda_slices
        from .fused_decode.split_attn import SplitDecodeAttention, max_slices, pick_cores

        taken = {(0, 0), (1, 0)} | {(c.x, c.y) for c, *_ in qkv_stream.assignments} | {tuple(BOUNDARY_CORE)}
        slices, min_chunks = sda_slices(bool(window), max_slices(self.mesh_device, taken, num_kv_heads), num_kv_heads)
        key = ("sda", num_heads, num_kv_heads, head_dim, window, slices, min_chunks, tuple(sorted(taken)))
        if key not in self._decode_stream_buffers:
            self._decode_stream_buffers[key] = SplitDecodeAttention(
                self.mesh_device,
                num_heads,
                num_kv_heads,
                head_dim,
                window,
                pick_cores(self.mesh_device, taken, num_kv_heads, slices),
                slices,
                min_chunks=min_chunks,
                out=self.get_decode_sda_out(num_heads, head_dim)[0],
                consts=self.get_decode_sda_out(num_heads, head_dim)[1],
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
        """LinearStream op (fused_decode/stream.py) for one decode role, shared by every layer."""
        from .fused_decode.stream import LinearStream

        if name not in self._decode_stream_buffers:
            self._decode_stream_buffers[name] = LinearStream(self.mesh_device, *args, **kwargs)
        return self._decode_stream_buffers[name]

    def reset_global_semaphores(self):
        """Reset all global semaphores to 0"""
        for sem in self.rs_ping_pong_semaphores:
            ttnn.reset_global_semaphore_value(sem, 0)
        for sem in self.ag_ping_pong_semaphores:
            ttnn.reset_global_semaphore_value(sem, 0)
