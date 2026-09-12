# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP4 decoder on a 1x4 Blackhole mesh; optimized single-chip kernels are the baseline.

Only setup partitions Torch weights. Runtime attention, recurrence, paged cache,
and logical-tail handling reuse OptimizedDecoder with local head dimensions.
Residual ownership is fixed at construction; caller inputs and outputs match.
Layers in one ordered CQ0 stack share a TT_CCL context. Its workspace must stay
alive through replay and must not be shared by concurrently executing models.
"""

import copy

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.optimized_decoder import DEFAULT_POLICY, OptimizedDecoder
from models.common.modules.tt_ccl import TT_CCL


class MultichipDecoder(OptimizedDecoder):
    TP = 4

    def _role(self, name):
        if name.removesuffix(".weight") == "mlp.interleaved_gate_up":
            return "gate"
        return super()._role(name)

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, policy=None, ccl=None):
        import torch

        if mesh_device.get_num_devices() != 4 or tuple(mesh_device.shape) != (1, 4):
            raise ValueError("MultichipDecoder requires the target 1x4 Blackhole mesh")
        if ccl is not None and ccl.mesh_device is not mesh_device:
            raise ValueError("The shared CCL context must belong to this mesh")
        self = cls()
        self.device = mesh_device
        self.layer_idx = layer_idx
        self.kind = hf_config.layer_types[layer_idx]
        self.eps = hf_config.rms_norm_eps
        self.policy = dict(DEFAULT_POLICY)
        self.policy.update(
            # Measured TP4 policy; native descriptors resolve each physical coordinate.
            attention_readers=2,
            output_readers=2,
            gate_readers=3,
            up_readers=3,
            down_readers=2,
            attention_cores=10,
            attention_block=16,
            gate_cores=40,
            gate_block=4,
            up_cores=40,
            up_block=4,
            output_cores=8,
            output_block=6,
            down_cores=8,
            down_block=17,
            residual_cores=40,
            allreduce_cores=40,
            chunk_size=4096,
            sdpa_k=128,
            prefill_1d=True,
            prefill_1d_min=64,
            prefill_1d_max=256,
            prefill_1d_k=20 if self.kind == "full_attention" else 8,
            prefill_1d_output_k=24 if self.kind == "full_attention" else 8,
            prefill_1d_down_k=17 if self.kind == "full_attention" else 8,
            prefill_1d_l1=self.kind != "full_attention",
            carry_input=True,
            carry_output=True,
            carry_residual=True,
            residual_layout="replicated",
            ccl_dtype="bfloat16",
            num_links=2,
            ring=True,
            packed_mlp=True,
            persistent_ccl=True,
            direct_allreduce=True,
            # Public TILE [B,1,H] expands to B*32 rows. Keep these batched
            # boundaries out of L1 while the next layer retains its input.
            public_dram_batch=2,
        )
        self.policy.update(policy or {})
        self.CHUNK_SIZE = self.policy["chunk_size"]
        self.sharded_residual = self.policy["residual_layout"] == "sharded"
        if self.sharded_residual:
            # OptimizedDecoder's packed residual branch has a full-hidden contract.
            self.policy["carry_residual"] = False
        self.topology = ttnn.Topology.Ring if self.policy.get("ring", False) else ttnn.Topology.Linear
        self.ccl = ccl if ccl is not None else TT_CCL(mesh_device)
        self.config = copy.deepcopy(hf_config)
        for name in (
            "num_attention_heads",
            "num_key_value_heads",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "intermediate_size",
        ):
            value = getattr(hf_config, name)
            assert value % self.TP == 0, name
            setattr(self.config, name, value // self.TP)
        self.ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.projection_configs = {
            role: ttnn.WormholeComputeKernelConfig(
                math_fidelity=getattr(ttnn.MathFidelity, self.policy[role + "_fidelity"]),
                math_approx_mode=False,
                fp32_dest_acc_en=self.policy.get(role + "_fp32", True),
                packer_l1_acc=True,
            )
            for role in ("attention", "output", "gate", "up", "down")
        }

        def upload(t, dtype=ttnn.bfloat16, dim=None):
            return ttnn.from_torch(
                t.contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=(
                    ttnn.ReplicateTensorToMesh(mesh_device)
                    if dim is None
                    else ttnn.ShardTensorToMesh(mesh_device, dim=dim)
                ),
            )

        self.weights = {}
        self.projection_widths = {}

        def weight(name, pieces):
            # Each piece is a complete local [N,K] projection before transposition.
            self.projection_widths[name] = pieces[0].shape[0]
            readers = self.policy[self._role(name) + "_readers"]
            if readers > 1 and (
                self.policy.get("pad_reader_outputs", False)
                or (self.policy.get("pad_narrow_outputs", False) and pieces[0].shape[0] < 64)
            ):
                alignment = mesh_device.dram_grid_size().x * readers * 32
                pieces = [torch.nn.functional.pad(p, (0, 0, 0, (-p.shape[0]) % alignment)) for p in pieces]
            self.weights[name + ".weight"] = upload(
                torch.cat([p.T for p in pieces], dim=1), self._weight_dtype(name), dim=1
            )

        def split(t, dim=0):
            return t.chunk(self.TP, dim=dim)

        for name in ("input_layernorm", "post_attention_layernorm"):
            t = (state_dict[name + ".weight"].float() + 1).reshape(1, 1, -1)
            self.weights[name + ".weight"] = upload(t, dim=2 if self.sharded_residual else None)
        if not self.policy.get("packed_mlp", False):
            for name in ("gate", "up"):
                weight("mlp." + name + "_proj", split(state_dict["mlp." + name + "_proj.weight"]))
        weight(
            "mlp.down_proj",
            split(state_dict["mlp.down_proj.weight"], 0 if self.policy.get("output_scheme") == "agmm" else 1),
        )
        if self.kind == "full_attention":
            c = hf_config
            qg = state_dict["self_attn.q_proj.weight"].reshape(c.num_attention_heads, 2, c.head_dim, c.hidden_size)
            qs, gs = split(qg[:, 0].reshape(-1, c.hidden_size)), split(qg[:, 1].reshape(-1, c.hidden_size))
            ks, vs = split(state_dict["self_attn.k_proj.weight"]), split(state_dict["self_attn.v_proj.weight"])
            weight("self_attn.qkvg", [torch.cat(p, dim=0) for p in zip(qs, ks, vs, gs)])
            if self.policy.get("split_attention", False):
                weight("self_attn.split_qg", [torch.cat(p) for p in zip(qs, gs)])
                weight("self_attn.split_k", ks)
                weight("self_attn.split_v", vs)
            weight(
                "self_attn.o_proj",
                split(state_dict["self_attn.o_proj.weight"], 0 if self.policy.get("output_scheme") == "agmm" else 1),
            )
            for name in ("q", "k"):
                self.weights[f"self_attn.{name}_norm.weight"] = upload(
                    (state_dict[f"self_attn.{name}_norm.weight"].float() + 1).reshape(1, 1, -1)
                )
        else:
            c = hf_config
            q, k, v = state_dict["linear_attn.in_proj_qkv.weight"].split(
                [c.linear_num_key_heads * 128, c.linear_num_key_heads * 128, c.linear_num_value_heads * 128]
            )
            zs = split(state_dict["linear_attn.in_proj_z.weight"])
            bs, aa = [split(state_dict[f"linear_attn.in_proj_{n}.weight"]) for n in ("b", "a")]
            pieces = []
            for qp, kp, vp, zp, bp, ap in zip(split(q), split(k), split(v), zs, bs, aa):
                pieces.append(
                    torch.cat(
                        [
                            qp,
                            kp,
                            vp,
                            zp,
                            torch.nn.functional.pad(bp, (0, 0, 0, (-bp.shape[0]) % 32)),
                            torch.nn.functional.pad(ap, (0, 0, 0, (-ap.shape[0]) % 32)),
                        ]
                    )
                )
            weight("linear_attn.packed", pieces)
            if self.policy.get("split_attention", False):
                widths = [2560, 1536, 32, 32]
                for index, part in enumerate(("qkv", "z", "b", "a")):
                    weight("linear_attn.split_" + part, [p.split(widths)[index] for p in pieces])
            weight(
                "linear_attn.out_proj",
                split(
                    state_dict["linear_attn.out_proj.weight"], 0 if self.policy.get("output_scheme") == "agmm" else 1
                ),
            )
            conv = state_dict["linear_attn.conv1d.weight"]
            cq, ck, cv = conv.split([q.shape[0], k.shape[0], v.shape[0]])
            local_convs = [torch.cat(p) for p in zip(split(cq), split(ck), split(cv))]
            self.conv_taps = [
                upload(torch.cat([p[:, 0, i] for p in local_convs]).reshape(1, 1, -1), dim=2) for i in range(4)
            ]
            self.a_neg = upload(-state_dict["linear_attn.A_log"].float().exp().reshape(1, 1, -1), ttnn.float32, dim=2)
            self.dt_bias = upload(state_dict["linear_attn.dt_bias"].float().reshape(1, 1, -1), ttnn.float32, dim=2)
            self.weights["linear_attn.norm.weight"] = upload(state_dict["linear_attn.norm.weight"].reshape(-1))
            masks = torch.zeros(1, 1, 32, 96)
            masks[:, :, :16, :16] = 1
            masks[:, :, 16:, 48:64] = 1
            masks[:, :, 16:, 64:80] = 1
            self.delta_constants = {
                "eye": upload(torch.eye(32).reshape(1, 1, 32, 32), ttnn.float32),
                "tril": upload(torch.ones(32, 32).tril().reshape(1, 1, 32, 32), ttnn.float32),
                "ones": upload(torch.ones(1, 1, 32, 32), ttnn.float32),
                "masks": upload(masks, ttnn.float32),
            }
        if self.policy.get("packed_mlp", False):
            gate, up = split(state_dict["mlp.gate_proj.weight"]), split(state_dict["mlp.up_proj.weight"])
            weight("mlp.gate_up", [torch.cat(parts) for parts in zip(gate, up)])
        if self.policy.get("minimal_mlp", False):
            gate, up = split(state_dict["mlp.gate_proj.weight"]), split(state_dict["mlp.up_proj.weight"])
            weight(
                "mlp.interleaved_gate_up",
                [
                    torch.stack([g.T.reshape(5120, -1, 32), u.T.reshape(5120, -1, 32)], dim=2).reshape(5120, -1).T
                    for g, u in zip(gate, up)
                ],
            )
        self.dram_weights = {}
        banks = mesh_device.dram_grid_size().x
        bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        for name, tensor in self.weights.items():
            if len(tensor.shape) != 2:
                continue
            k, n = tensor.shape
            readers = self.policy[self._role(name) + "_readers"]
            width = ((n + 32 * banks * readers - 1) // (32 * banks * readers)) * 32 * readers
            memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(bank_grid, [k, width], ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.dram_weights[name] = ttnn.to_memory_config(tensor, memory)
        self.ccl_buffers = {}
        self.fused_buffers = {}
        if self.policy.get("fused_mm_l1", False):
            counters = getattr(self.ccl, "_qwen_mmrs_counters", None)
            if counters is None:
                grid = mesh_device.compute_with_storage_grid_size()
                slots = grid.x * grid.y
                cores = ttnn.num_cores_to_corerangeset(slots, grid, row_wise=True)
                memory = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(cores, [1, slots], ttnn.ShardOrientation.ROW_MAJOR),
                )
                counters = tuple(
                    ttnn.allocate_tensor_on_device(
                        ttnn.Shape([slots, slots]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device, memory
                    )
                    for _ in range(2)
                )
                self.ccl._qwen_mmrs_counters = counters
            self.mmrs_counters = counters
        if self.policy.get("direct_allreduce", False):
            # A layer stack shares this context; 64 independent L1 workspaces
            # would crowd out native matmul circular buffers.
            ar_cores = self.policy.get("allreduce_cores", 80)
            workspace_key = "_qwen_tp4_allreduce_buffer_" + str(ar_cores)
            workspace = getattr(self.ccl, workspace_key, None)
            if workspace is None:
                workspace = ttnn.empty(
                    [1, 1, 32, 20480],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=self._width_memory(ar_cores, 32, 20480 // ar_cores),
                )
                setattr(self.ccl, workspace_key, workspace)
            self.allreduce_buffer = workspace
        return self

    def allocate_state(self, *, batch_size, num_pages=None):
        state = super().allocate_state(batch_size=batch_size, num_pages=num_pages)
        if self.policy.get("fused_input", False):
            self.input_ag_buffer = ttnn.empty(
                [1, 1, batch_size, 5120],
                dtype=getattr(ttnn, self.policy["ccl_dtype"]),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if self.sharded_residual and self.policy.get("fused_norm", False):
            template = ttnn.empty(
                [1, 1, batch_size, 1280],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.fused_norm_buffers = tuple(
                ttnn.experimental.dit_fused_distributed_rmsnorm_create_stats_buffer(
                    template,
                    1,
                    self.device,
                    num_links=self.policy["num_links"],
                    weight=self.weights["input_layernorm.weight"],
                )
                for _ in range(2)
            )
            self.fused_norm_index = 0
            ttnn.deallocate(template)
        if self.policy.get("fused_persistent", False) and self.policy.get("output_scheme") == "agmm":
            for name, weight in self.weights.items():
                if self._role(name) in ("output", "down"):
                    self.fused_buffers[name.removesuffix(".weight")] = ttnn.empty(
                        [1, 1, batch_size, weight.shape[0]],
                        dtype=getattr(ttnn, self.policy["ccl_dtype"]),
                        layout=ttnn.TILE_LAYOUT,
                        device=self.device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
        if self.policy.get("persistent_ccl", False):
            dtype = getattr(ttnn, self.policy["ccl_dtype"])
            shapes = {(1, 1, batch_size, 5120), (1, batch_size, 1, 5120)}
            for shape in shapes:
                memory = (
                    ttnn.L1_MEMORY_CONFIG
                    if batch_size == 1 and self.policy.get("persistent_l1", False)
                    else ttnn.DRAM_MEMORY_CONFIG
                )
                gathered = ttnn.empty(
                    shape,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=memory,
                )
                scattered = ttnn.empty(
                    [*shape[:-1], 1280],
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=memory,
                )
                if self.topology == ttnn.Topology.Ring:
                    first, penult = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
                        gathered, dim=3, topology=self.topology, cluster_axis=1
                    )
                    buffers = [first, scattered, penult]
                else:
                    buffers = [ttnn.empty_like(gathered), scattered]
                self.ccl_buffers[(shape, dtype)] = (gathered, buffers)
        return state

    def _gather(self, x):
        full_shape = (*list(x.shape)[:-1], x.shape[-1] * 4)
        buffers = self.ccl_buffers.get((full_shape, x.dtype))
        memory = self._collective_memory(x)
        if buffers and buffers[0].memory_config() != memory:
            buffers = None
        return ttnn.experimental.all_gather_async(
            x,
            persistent_output_tensor=buffers[0] if buffers else None,
            dim=3,
            cluster_axis=1,
            mesh_device=self.device,
            topology=self.topology,
            num_links=self.policy["num_links"],
            multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(1),
            memory_config=memory,
        )

    def _collective_memory(self, x):
        if self.policy.get("sharded_l1", False) and all(d == 1 for d in tuple(x.shape)[:-1]):
            return ttnn.L1_MEMORY_CONFIG
        return ttnn.DRAM_MEMORY_CONFIG

    def _finish(self, x, attention):
        if not (self.sharded_residual and self.policy.get("sharded_l1", False) and tuple(x.shape)[:2] == (1, 1)):
            return super()._finish(x, attention)
        memory = ttnn.L1_MEMORY_CONFIG
        h = ttnn.add(x, attention, memory_config=memory)
        n = self._norm(h, "post_attention_layernorm")
        if self.policy["packed_mlp"]:
            packed = self._linear(n, "mlp.gate_up")
            width = self.config.intermediate_size
            product = ttnn.mul(
                packed[:, :, :width],
                packed[:, :, width:],
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                memory_config=memory,
            )
        else:
            gate = self._linear(n, "mlp.gate_proj", activation="silu")
            up = self._linear(n, "mlp.up_proj")
            product = ttnn.mul(gate, up, memory_config=memory)
        return ttnn.add(h, self._linear(product, "mlp.down_proj"), memory_config=memory)

    def _linear(self, x, name, activation=None, keep_sharded=False):
        if (
            self.policy.get("fused_input", False)
            and x.shape[-1] == 1280
            and self._role(name) in ("attention", "gate", "up")
        ):
            shape = list(x.shape)
            xx = ttnn.reshape(ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG), [1, 1, -1, 1280])
            xx = ttnn.typecast(xx, getattr(ttnn, self.policy["ccl_dtype"]))
            weight = self.weights[name + ".weight"]
            result = ttnn.experimental.all_gather_minimal_matmul_async(
                xx,
                ttnn.reshape(weight, [1, 1, *weight.shape]),
                config=ttnn.MinimalMatmulConfig(
                    M_block_size=1,
                    K_block_size=self.policy.get("fused_input_k", 8),
                    N_block_size=self.policy.get("fused_input_n", 8),
                    subblock_h=1,
                    subblock_w=4,
                    compute_with_storage_grid_size=(10, 8),
                ),
                multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(1),
                topology=self.topology,
                cluster_axis=1,
                num_links=self.policy["num_links"],
                persistent_output_buffer=self.input_ag_buffer,
                compute_kernel_config=self.projection_configs[self._role(name)],
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                force_transpose=False,
                num_workers_per_link=8 // self.policy["num_links"],
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if activation else None,
            )[0]
            return ttnn.reshape(result, [*shape[:-1], weight.shape[-1]])
        scheme = self.policy.get("output_scheme", "row")
        if self._role(name) in ("output", "down") and scheme in ("agmm", "mmrs"):
            shape = list(x.shape)
            xx = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            xx = ttnn.reshape(xx, [1, *shape] if len(shape) == 3 else shape)
            if self.policy["ccl_dtype"] != "bfloat16":
                xx = ttnn.typecast(xx, getattr(ttnn, self.policy["ccl_dtype"]))
            weight = self.weights[name + ".weight"]
            weight = ttnn.reshape(weight, [1, 1, *weight.shape])
            config = ttnn.MinimalMatmulConfig(
                M_block_size=1 if x.shape[1] == 1 else 4,
                K_block_size=self.policy.get("fused_" + self._role(name) + "_k", self.policy.get("fused_k", 8)),
                N_block_size=self.policy.get("fused_n", 8),
                subblock_h=1,
                subblock_w=4,
                compute_with_storage_grid_size=tuple(self.policy.get("fused_grid", [10, 8])),
            )
            if scheme == "agmm":
                persistent = self.fused_buffers.get(name) if x.shape[1] == 1 else None
                result = ttnn.experimental.all_gather_minimal_matmul_async(
                    xx,
                    weight,
                    config=config,
                    multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(1),
                    topology=self.topology,
                    cluster_axis=1,
                    num_links=self.policy["num_links"],
                    persistent_output_buffer=persistent,
                    barrier_semaphore=(
                        None if persistent is not None else self.ccl.get_and_cycle_barrier_semaphore_handle(1)
                    ),
                    compute_kernel_config=self.projection_configs[self._role(name)],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=xx.dtype,
                    num_workers_per_link=self.policy.get(
                        "fused_workers",
                        self.policy.get("fused_grid", [10, 8])[0 if self.policy.get("fused_transpose", True) else 1]
                        // self.policy["num_links"],
                    ),
                    force_transpose=self.policy.get("fused_transpose", True),
                )[0]
            else:
                l1 = self.policy.get("fused_mm_l1", False) and x.shape[1] == 1
                output_memory = self._collective_memory(xx)
                buffers = self.ccl_buffers.get(((*tuple(xx.shape)[:-1], 5120), xx.dtype))
                persistent = (
                    buffers[1][1]
                    if self.policy.get("fused_persistent", False)
                    and buffers
                    and buffers[0].memory_config() == output_memory
                    else None
                )
                result = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
                    xx,
                    weight,
                    dim=3,
                    multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(1),
                    reduce_scatter_core_grid_offset=ttnn.CoreCoord(0, 8),
                    config=config,
                    topology=self.topology,
                    cluster_axis=1,
                    num_links=self.policy["num_links"],
                    barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(1),
                    compute_kernel_config=self.projection_configs[self._role(name)],
                    memory_config_mm=ttnn.L1_MEMORY_CONFIG if l1 else ttnn.DRAM_MEMORY_CONFIG,
                    rs_output_mem_config=output_memory,
                    optional_rs_output_tensor=persistent,
                    using_persistent_buffers=persistent is not None,
                    **(
                        {
                            "mm_window_blocks": 2,
                            "mm_progress_counters": self.mmrs_counters[0],
                            "mm_credit_counters": self.mmrs_counters[1],
                        }
                        if l1
                        else {}
                    ),
                    dtype=xx.dtype,
                )[1]
            if result.dtype != ttnn.bfloat16:
                result = ttnn.typecast(result, ttnn.bfloat16)
            if not self.sharded_residual:
                result = self._gather(result)
            return ttnn.reshape(result, [*shape[:-1], 1280 if self.sharded_residual else 5120])
        if self.policy.get("prefill_1d", False) and self.policy.get("prefill_1d_min", 2) <= x.shape[
            1
        ] <= self.policy.get("prefill_1d_max", 256):
            grid = self.device.compute_with_storage_grid_size()
            n = self.weights[name + ".weight"].shape[-1]
            per_m = (x.shape[1] + 31) // 32
            per_n = (n // 32 + grid.x * grid.y - 1) // (grid.x * grid.y)
            sub_w = next(v for v in (4, 3, 2, 1) if per_n % v == 0)
            sub_h = next(v for v in (4, 2, 1) if per_m % v == 0 and v * sub_w <= 4)
            program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                in0_block_w=self.policy.get(
                    "prefill_1d_" + self._role(name) + "_k", self.policy.get("prefill_1d_k", 8)
                ),
                per_core_M=per_m,
                per_core_N=per_n,
                out_subblock_h=sub_h,
                out_subblock_w=sub_w,
                fuse_batch=False,
                mcast_in0=True,
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if activation else None,
            )
            group = "mlp" if name.startswith("mlp.") else "attention"
            activation_dtype = getattr(ttnn, self.policy.get(group + "_activation", "bfloat16"))
            if x.dtype != activation_dtype:
                x = ttnn.typecast(x, activation_dtype)
            if self.policy.get("prefill_1d_l1", self.policy.get("prefill_l1", False)):
                x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            output = ttnn.linear(
                x,
                self.weights[name + ".weight"],
                program_config=program,
                compute_kernel_config=self.projection_configs[self._role(name)],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
        else:
            output = super()._linear(x, name, activation=activation, keep_sharded=keep_sharded)
        width = self.projection_widths[name]
        if output.shape[-1] != width:
            output = output[..., :width]
        if self._role(name) not in ("output", "down"):
            return output
        shape = list(output.shape)
        if self.policy.get("direct_allreduce", False) and not self.sharded_residual and x.shape[1] == 1:
            batch = shape[0] if len(shape) == 3 else shape[-2]
            packed = ttnn.reshape(output, [1, 1, batch, 5120])
            ar_cores = self.policy.get("allreduce_cores", 80)
            memory = self._width_memory(ar_cores, 32, 5120 // ar_cores)
            packed = ttnn.to_memory_config(packed, memory)
            ccl_dtype = getattr(ttnn, self.policy["ccl_dtype"])
            if packed.dtype != ccl_dtype:
                packed = ttnn.typecast(packed, ccl_dtype)
            reduced = ttnn.experimental.all_reduce_async(
                packed,
                self.allreduce_buffer,
                cluster_axis=1,
                mesh_device=self.device,
                multi_device_global_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(1),
                topology=self.topology,
                num_links=self.policy["num_links"],
                memory_config=memory,
            )
            if reduced.dtype != ttnn.bfloat16:
                reduced = ttnn.typecast(reduced, ttnn.bfloat16)
            if len(shape) == 3:
                return self._public_rows(reduced, batch, 5120, keep_sharded=keep_sharded)
            return reduced
        collective_memory = self._collective_memory(output)
        output = ttnn.to_memory_config(output, collective_memory)
        output = ttnn.reshape(output, [1, *shape] if len(shape) == 3 else shape)
        dtype = getattr(ttnn, self.policy["ccl_dtype"])
        if output.dtype != dtype:
            output = ttnn.typecast(output, dtype)
        buffers = self.ccl_buffers.get((tuple(output.shape), output.dtype))
        if buffers and buffers[0].memory_config() != collective_memory:
            buffers = None
        output = ttnn.experimental.reduce_scatter_minimal_async(
            output,
            persistent_output_buffers=buffers[1] if buffers else None,
            dim=3,
            cluster_axis=1,
            topology=self.topology,
            num_links=self.policy["num_links"],
            multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(1),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(1),
            memory_config=collective_memory,
        )
        if not self.sharded_residual:
            output = self._gather(output)
        if output.dtype != ttnn.bfloat16:
            output = ttnn.typecast(output, ttnn.bfloat16)
        return ttnn.reshape(output, [*shape[:-1], 1280 if self.sharded_residual else 5120])

    def _norm(self, x, name):
        if not self.sharded_residual or not name.endswith("layernorm"):
            return super()._norm(x, name)
        shape = list(x.shape)
        memory = self._collective_memory(x)
        x = ttnn.to_memory_config(x, memory)
        x = ttnn.reshape(x, [1, *shape] if len(shape) == 3 else shape)
        if self.policy.get("fused_norm", False) and shape[1] == 1:
            buffer = self.fused_norm_buffers[self.fused_norm_index]
            self.fused_norm_index = (self.fused_norm_index + 1) % len(self.fused_norm_buffers)
            normalized = ttnn.experimental.dit_fused_distributed_rmsnorm(
                x,
                1,
                self.device,
                self.ccl.get_and_cycle_ag_semaphore_handles(1),
                topology=self.topology,
                epsilon=self.eps,
                weight=self.weights[name + ".weight"],
                compute_kernel_config=self.ckc,
                memory_config=memory,
                dtype=ttnn.bfloat16,
                persistent_output_buffer=buffer,
                num_preferred_links=self.policy["num_links"],
            )
        else:
            stats = ttnn.rms_norm_pre_all_gather(x, compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
            stats = self._gather(stats)
            normalized = ttnn.rms_norm_post_all_gather(
                x,
                stats,
                epsilon=self.eps,
                weight=self.weights[name + ".weight"],
                compute_kernel_config=self.ckc,
                memory_config=memory,
            )
        ccl_dtype = getattr(ttnn, self.policy["ccl_dtype"])
        if self.policy.get("fused_input", False) and shape[1] == 1:
            return ttnn.reshape(normalized, shape)
        if normalized.dtype != ccl_dtype:
            normalized = ttnn.typecast(normalized, ccl_dtype)
        gathered = self._gather(normalized)
        if gathered.dtype != ttnn.bfloat16:
            gathered = ttnn.typecast(gathered, ttnn.bfloat16)
        return ttnn.reshape(gathered, [*shape[:-1], 5120])
