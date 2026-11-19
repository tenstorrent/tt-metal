# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Optimized Qwen decoder with explicit per-group precision and sparse geometry.

The fused attention/state implementation is shared unchanged. All forward work
remains on device; weight conversion and policy construction happen only at load.
"""

import math

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.fused_decoder import DRAM, FusedDecoder, reshape


class _ExpertDecoder(FusedDecoder):
    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        expert_dtype="bfp8",
        expert_down_dtype=None,
        expert_fidelity="LoFi",
        expert_memory="l1",
        sparse_block_w=(16, 16),
        sparse_cores=(32, 64),
        sparse_subblock_w=(1, 1),
        shared_dtype="bfp8",
        shared_fidelity="HiFi2",
    ):
        dtypes = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
        fidelities = {name: getattr(ttnn.MathFidelity, name) for name in ("LoFi", "HiFi2", "HiFi4")}
        expert_down_dtype = expert_dtype if expert_down_dtype is None else expert_down_dtype
        if expert_dtype not in dtypes or expert_down_dtype not in dtypes or shared_dtype not in dtypes:
            raise ValueError("Unknown weight dtype")
        if expert_fidelity not in fidelities or shared_fidelity not in fidelities:
            raise ValueError("Unknown compute fidelity")
        if expert_memory not in ("l1", "dram"):
            raise ValueError("expert_memory must be l1 or dram")
        self = super().from_state_dict(state_dict, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh_device)
        self.expert_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=fidelities[expert_fidelity],
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.shared_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=fidelities[shared_fidelity],
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.expert_memory = ttnn.L1_MEMORY_CONFIG if expert_memory == "l1" else DRAM
        self.sparse_block_w = tuple(sparse_block_w)
        self.sparse_cores = tuple(sparse_cores)
        self.sparse_subblock_w = tuple(sparse_subblock_w)
        self.egu = ttnn.typecast(self.egu, dtypes[expert_dtype])
        self.ed = ttnn.typecast(self.ed, dtypes[expert_down_dtype])
        # Grouped prefill supports BF16/BFP8 only. BFP4 is a decode candidate,
        # never the default or a claim of BFP4 grouped-prefill execution.
        prefill_dtype = ttnn.bfloat8_b if expert_dtype == "bfp4" else dtypes[expert_dtype]
        self.grouped_gu = [ttnn.typecast(w, prefill_dtype) for w in self.grouped_gu]
        prefill_down_dtype = ttnn.bfloat8_b if expert_down_dtype == "bfp4" else dtypes[expert_down_dtype]
        self.grouped_down = [ttnn.typecast(w, prefill_down_dtype) for w in self.grouped_down]
        for key in ("shared_packed", "mlp.shared_expert.down_proj.weight"):
            self.w[key] = ttnn.typecast(self.w[key], dtypes[shared_dtype])
        self.precision_policy = dict(
            expert_dtype=expert_dtype,
            expert_down_dtype=expert_down_dtype,
            grouped_prefill_down_dtype="bfp8" if expert_down_dtype == "bfp4" else expert_down_dtype,
            expert_fidelity=expert_fidelity,
            grouped_prefill_dtype="bfp8" if expert_dtype == "bfp4" else expert_dtype,
            shared_dtype=shared_dtype,
            shared_fidelity=shared_fidelity,
            attention="inherited BF16/HiFi4 and ordered FP32 exceptions",
            router="inherited BF16/FP32; exact linear selector",
            kv="bf16",
            activations="bf16; inherited sensitive FP32 intermediates",
            expert_memory=expert_memory,
            sparse_block_w=list(sparse_block_w),
            sparse_cores=list(sparse_cores),
            sparse_subblock_w=list(sparse_subblock_w),
        )
        return self

    def linear(self, x, key, dtype=ttnn.bfloat16, **kwargs):
        if key in ("shared_packed", "mlp.shared_expert.down_proj.weight"):
            return ttnn.linear(
                x, self.w[key], dtype=dtype, compute_kernel_config=self.shared_compute, memory_config=DRAM
            )
        return super().linear(x, key, dtype, **kwargs)

    def _moe(self, x, *, ordered_router=False):
        if self.kind == "full_attention":
            s = x.shape[1]
            logits = reshape(self.linear(x, "mlp.gate.weight", ttnn.bfloat16 if s == 1 else ttnn.float32), (s, 256))
            if s == 1:
                face = reshape(logits, (1, 16, 16))
                face = ttnn.to_memory_config(face, self.router_shard)
                gate_scores, gate_ids = ttnn.experimental.deepseek.moe.generalized_moe_gate(
                    face,
                    bias_tensor=self.router_bias,
                    input_indices_tensor=self.router_indices,
                    output_tensor=self.router_out,
                    output_indices_tensor=self.router_out_ids,
                    scaling_factor=1.0,
                    enable_sigmoid=False,
                    topk=8,
                    output_softmax=True,
                )
                scores = reshape(ttnn.slice(gate_scores, (0, 0, 0), (1, 1, 8), memory_config=DRAM), (1, 8))
                indices = ttnn.reshape(ttnn.slice(gate_ids, (0, 0, 0), (1, 1, 8), memory_config=DRAM), (1, 8))
            else:
                values, indices = ttnn.topk(logits, k=8, dim=-1, sorted=True)
                scores = ttnn.softmax(values, dim=-1, numeric_stable=True, compute_kernel_config=self.compute)
            if s == 1:
                xx = reshape(x, (1, 1, 1, 2048))
                ids = ttnn.reshape(ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT), (1, 1, 1, 8))

                def sparse(a, w, sparse_a=False):
                    n = w.shape[-1]
                    role = 0 if n == 1024 else 1
                    cores = self.sparse_cores[role]
                    grid = (8, cores // 8)
                    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
                        in0_block_w=self.sparse_block_w[role],
                        out_subblock_h=1,
                        out_subblock_w=self.sparse_subblock_w[role],
                        out_block_h=1,
                        out_block_w=n // (32 * cores),
                        per_core_M=1,
                        per_core_N=n // (32 * cores),
                        fuse_batch=False,
                        fused_activation=None,
                        mcast_in0=True,
                    )
                    return ttnn.sparse_matmul(
                        a,
                        w,
                        sparsity=self.indexed_sparsity,
                        indices=ids,
                        is_input_a_sparse=sparse_a,
                        is_input_b_sparse=True,
                        program_config=pc,
                        compute_kernel_config=self.expert_compute,
                        memory_config=self.expert_memory,
                    )

                gate_up = ttnn.reshape(sparse(xx, self.egu), (1, 8, 1, 1024))
                gate, up = (gate_up[:, :, :, :512], gate_up[:, :, :, 512:])
                down = sparse(
                    ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]), self.ed, True
                )
                down = ttnn.reshape(down, (1, 8, 1, 2048))
                routing = reshape(scores, (1, 8, 1, 1))
                weighted = ttnn.multiply(down, routing)
                total = ttnn.experimental.fast_reduce_nc(
                    weighted, dims=[1], memory_config=DRAM, compute_kernel_config=self.compute
                )
                routed = ttnn.reshape(total, (1, 1, 2048), (1, 32, 2048))
            else:
                group_memory = ttnn.L1_MEMORY_CONFIG if s <= 256 else DRAM
                x_rows = ttnn.to_layout(reshape(x, (1, 1, s, 2048)), ttnn.ROW_MAJOR_LAYOUT)
                indices32 = reshape(ttnn.typecast(indices, ttnn.uint32), (1, 1, s, 8))
                weights = reshape(scores, (1, 1, s, 8))
                routed = None
                for bank in range(4):
                    valid = ttnn.logical_and(ttnn.ge(indices32, bank * 64), ttnn.lt(indices32, (bank + 1) * 64))
                    local_ids = ttnn.subtract(indices32, bank * 64)
                    local_weights = ttnn.where(valid, weights, 0.0)
                    (
                        gather,
                        combine,
                        combine_weights,
                        counts,
                    ) = ttnn.experimental.nomic_embed_text_v2_moe.routing_metadata(
                        local_ids, local_weights, num_experts=64, token_start=0, num_tokens=s
                    )
                    dispatched = reshape(
                        ttnn.embedding(gather, x_rows, layout=ttnn.TILE_LAYOUT, memory_config=group_memory),
                        (1, 1, gather.shape[-1], 2048),
                    )
                    gu = ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul(
                        dispatched,
                        self.grouped_gu[bank],
                        counts,
                        dtype=ttnn.bfloat16,
                        memory_config=group_memory,
                        compute_kernel_config=self.expert_compute,
                    )
                    del dispatched
                    activated = ttnn.multiply(
                        gu[:, :, :, :512], gu[:, :, :, 512:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
                    )
                    del gu
                    down = ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul(
                        activated,
                        self.grouped_down[bank],
                        counts,
                        dtype=ttnn.bfloat16,
                        output_layout=ttnn.ROW_MAJOR_LAYOUT if self.kind == "full_attention" else ttnn.TILE_LAYOUT,
                        memory_config=group_memory,
                        compute_kernel_config=self.expert_compute,
                    )
                    del activated
                    part = ttnn.experimental.nomic_embed_text_v2_moe.routed_combine(
                        (
                            down[:, :, :, :2048]
                            if self.kind == "full_attention"
                            else ttnn.to_layout(down, ttnn.ROW_MAJOR_LAYOUT)
                        ),
                        combine,
                        combine_weights,
                        self.combine_bias,
                        num_tokens=s,
                        memory_config=DRAM,
                        compute_kernel_config=self.compute,
                    )
                    del down
                    routed = part if routed is None else ttnn.add(routed, part)
                routed = reshape(routed, (1, s, 2048))
            shared_all = self.linear(x, "shared_packed")
            shared = ttnn.multiply(
                shared_all[:, :, :512], shared_all[:, :, 512:1024], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
            )
            shared = self.linear(shared, "mlp.shared_expert.down_proj.weight")
            shared = ttnn.multiply(
                shared, shared_all[:, :, 1024:1025], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]
            )
            return ttnn.add(routed, shared)
        s = x.shape[1]
        logits = reshape(
            self.linear(
                x,
                "mlp.gate.weight",
                ttnn.bfloat16 if s == 1 else ttnn.float32,
                ordered_router=ordered_router,
                terminal_scan=ordered_router,
            ),
            (s, 256),
        )
        if s == 1:
            routing, ids = self.inline_selector(logits)
            scores = reshape(routing, (1, 8))
            indices = ttnn.reshape(ids, (1, 8))
        else:
            values, indices = ttnn.topk(logits, k=8, dim=-1, sorted=True)
            scores = ttnn.softmax(values, dim=-1, numeric_stable=True, compute_kernel_config=self.compute)
        if s == 1:
            xx = reshape(x, (1, 1, 1, 2048))
            ids = ttnn.reshape(ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT), (1, 1, 1, 8))

            def sparse(a, w, sparse_a=False):
                n = w.shape[-1]
                role = 0 if n == 1024 else 1
                cores = self.sparse_cores[role]
                grid = (8, cores // 8)
                pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
                    in0_block_w=self.sparse_block_w[role],
                    out_subblock_h=1,
                    out_subblock_w=self.sparse_subblock_w[role],
                    out_block_h=1,
                    out_block_w=n // (32 * cores),
                    per_core_M=1,
                    per_core_N=n // (32 * cores),
                    fuse_batch=False,
                    fused_activation=None,
                    mcast_in0=True,
                )
                return ttnn.sparse_matmul(
                    a,
                    w,
                    sparsity=self.indexed_sparsity,
                    indices=ids,
                    is_input_a_sparse=sparse_a,
                    is_input_b_sparse=True,
                    program_config=pc,
                    compute_kernel_config=self.expert_compute,
                    memory_config=self.expert_memory,
                )

            gate_up = ttnn.reshape(sparse(xx, self.egu), (1, 8, 1, 1024))
            gate, up = (gate_up[:, :, :, :512], gate_up[:, :, :, 512:])
            down = sparse(ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]), self.ed, True)
            down = ttnn.reshape(down, (1, 8, 1, 2048))
            routing = reshape(scores, (1, 8, 1, 1))
            weighted = ttnn.multiply(down, routing)
            total = ttnn.experimental.fast_reduce_nc(
                weighted, dims=[1], memory_config=DRAM, compute_kernel_config=self.compute
            )
            routed = ttnn.reshape(total, (1, 1, 2048), (1, 32, 2048))
        else:
            group_memory = ttnn.L1_MEMORY_CONFIG if s <= 256 else DRAM
            x_rows = ttnn.to_layout(reshape(x, (1, 1, s, 2048)), ttnn.ROW_MAJOR_LAYOUT)
            indices32 = reshape(ttnn.typecast(indices, ttnn.uint32), (1, 1, s, 8))
            weights = reshape(scores, (1, 1, s, 8))
            routed = None
            for bank in range(4):
                valid = ttnn.logical_and(ttnn.ge(indices32, bank * 64), ttnn.lt(indices32, (bank + 1) * 64))
                local_ids = ttnn.subtract(indices32, bank * 64)
                local_weights = ttnn.where(valid, weights, 0.0)
                gather, combine, combine_weights, counts = ttnn.experimental.nomic_embed_text_v2_moe.routing_metadata(
                    local_ids, local_weights, num_experts=64, token_start=0, num_tokens=s
                )
                dispatched = reshape(
                    ttnn.embedding(gather, x_rows, layout=ttnn.TILE_LAYOUT, memory_config=group_memory),
                    (1, 1, gather.shape[-1], 2048),
                )
                gu = ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul(
                    dispatched,
                    self.grouped_gu[bank],
                    counts,
                    dtype=ttnn.bfloat16,
                    memory_config=group_memory,
                    compute_kernel_config=self.expert_compute,
                )
                del dispatched
                activated = ttnn.multiply(
                    gu[:, :, :, :512], gu[:, :, :, 512:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
                )
                del gu
                down = ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul(
                    activated,
                    self.grouped_down[bank],
                    counts,
                    dtype=ttnn.bfloat16,
                    output_layout=ttnn.ROW_MAJOR_LAYOUT if self.kind == "full_attention" else ttnn.TILE_LAYOUT,
                    memory_config=group_memory,
                    compute_kernel_config=self.expert_compute,
                )
                del activated
                part = ttnn.experimental.nomic_embed_text_v2_moe.routed_combine(
                    (
                        down[:, :, :, :2048]
                        if self.kind == "full_attention"
                        else ttnn.to_layout(down, ttnn.ROW_MAJOR_LAYOUT)
                    ),
                    combine,
                    combine_weights,
                    self.combine_bias,
                    num_tokens=s,
                    memory_config=DRAM,
                    compute_kernel_config=self.compute,
                )
                del down
                routed = part if routed is None else ttnn.add(routed, part)
            routed = reshape(routed, (1, s, 2048))
        shared_all = self.linear(x, "shared_packed")
        shared = ttnn.multiply(
            shared_all[:, :, :512], shared_all[:, :, 512:1024], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
        )
        shared = self.linear(shared, "mlp.shared_expert.down_proj.weight")
        shared = ttnn.multiply(
            shared, shared_all[:, :, 1024:1025], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]
        )
        return ttnn.add(routed, shared)


class _ProjectionDecoder(_ExpertDecoder):
    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        attention_dtype="bfp8",
        attention_fidelity="HiFi2",
        attention_topology="packed",
        shared_topology="packed_all",
        shared_scalar_dtype="bf16",
        shared_scalar_fidelity="HiFi4",
        projection_geometry=None,
        shared_dram=True,
        prefill_block_w=8,
        **kwargs,
    ):
        """Geometry is a per-role JSON mapping, plus optional ``default`` entry.

        Roles: attention_packed, attention_q, attention_k, attention_v,
        attention_gate, attention_out, shared_packed, shared_gu, shared_gate,
        shared_up, shared_scalar, shared_down. Geometry fields: readers (1..3),
        input_cores, block_w, per_core_n, and bank_tile_multiple. Reader 1 uses
        the legacy single-reader control; 2/3 use the multicore pipeline.
        """
        import torch

        dtypes = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
        if attention_dtype not in ("bf16", "bfp8"):
            raise ValueError("attention candidate dtype must be bf16 or bfp8")
        if shared_scalar_dtype not in ("bf16", "bfp8"):
            raise ValueError("shared scalar dtype must be bf16 or bfp8")
        if attention_topology not in ("packed", "split"):
            raise ValueError("attention_topology must be packed or split")
        if shared_topology not in ("packed_all", "packed_gu_scalar", "split"):
            raise ValueError("shared_topology must be packed_all, packed_gu_scalar, or split")
        self = super().from_state_dict(
            state_dict, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh_device, **kwargs
        )
        self.projection_geometry = projection_geometry or {}
        self.candidate_weights = {}
        self.candidate_decode = {}
        self.candidate_compute = {}
        self.effective_geometry = {}
        self.prefill_block_w = prefill_block_w
        self.attention_topology = attention_topology
        self.shared_topology = shared_topology

        def compute(fidelity):
            if fidelity not in ("LoFi", "HiFi2", "HiFi4"):
                raise ValueError("unknown projection compute fidelity")
            return ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=getattr(ttnn.MathFidelity, fidelity),
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )

        attention_compute = compute(attention_fidelity)
        scalar_compute = compute(shared_scalar_fidelity)

        def install(role, matrix, dtype, kernel, dram=True):
            weight = ttnn.from_torch(
                matrix.T.contiguous(),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=DRAM,
            )
            self.candidate_weights[role] = weight
            self.candidate_compute[role] = kernel
            if dram:
                self.candidate_decode[role] = self._make_dram_projection(weight, role)

        if self.kind == "full_attention":
            qg = state_dict["self_attn.q_proj.weight"].reshape(16, 2, 256, 2048)
            # HF Q projection interleaves Q/gate by head. The consumer requires
            # Q[4096], K[512], V[512], gate[4096], never HF's raw Q/gate order.
            attention = (
                qg[:, 0].reshape(4096, 2048),
                state_dict["self_attn.k_proj.weight"],
                state_dict["self_attn.v_proj.weight"],
                qg[:, 1].reshape(4096, 2048),
            )
            if attention_topology == "packed":
                install("attention_packed", torch.cat(attention), dtypes[attention_dtype], attention_compute)
            else:
                for role, matrix in zip(("attention_q", "attention_k", "attention_v", "attention_gate"), attention):
                    install(role, matrix, dtypes[attention_dtype], attention_compute)
            install("attention_out", state_dict["self_attn.o_proj.weight"], dtypes[attention_dtype], attention_compute)

        gate = state_dict["mlp.shared_expert.gate_proj.weight"]
        up = state_dict["mlp.shared_expert.up_proj.weight"]
        scalar = torch.nn.functional.pad(state_dict["mlp.shared_expert_gate.weight"], (0, 0, 0, 31))
        shared_dtype = dtypes[kwargs.get("shared_dtype", "bfp8")]
        if shared_topology == "packed_all":
            install("shared_packed", torch.cat((gate, up, scalar)), shared_dtype, self.shared_compute, shared_dram)
        else:
            if shared_topology == "packed_gu_scalar":
                install("shared_gu", torch.cat((gate, up)), shared_dtype, self.shared_compute, shared_dram)
            else:
                install("shared_gate", gate, shared_dtype, self.shared_compute, shared_dram)
                install("shared_up", up, shared_dtype, self.shared_compute, shared_dram)
            install("shared_scalar", scalar, dtypes[shared_scalar_dtype], scalar_compute, shared_dram)
        install(
            "shared_down",
            state_dict["mlp.shared_expert.down_proj.weight"],
            shared_dtype,
            self.shared_compute,
            shared_dram,
        )
        replaced = ["shared_packed", "mlp.shared_expert.down_proj.weight"]
        if self.kind == "full_attention":
            replaced += ["attention_packed", "self_attn.o_proj.weight"]
        for key in replaced:
            self.w.pop(key, None)
            self.decode_linear.pop(key, None)
        self.precision_policy["attention"] = (
            "BFP8/HiFi2 packed QKVG and output"
            if self.kind == "full_attention"
            else "inherited BF16/HiFi4 and ordered FP32 exceptions"
        )
        self.precision_policy["projection_candidate"] = dict(
            attention_dtype=attention_dtype if self.kind == "full_attention" else "inherited ordered policy",
            attention_fidelity=attention_fidelity if self.kind == "full_attention" else "inherited ordered policy",
            attention_topology=attention_topology if self.kind == "full_attention" else "inherited ordered policy",
            shared_topology=shared_topology,
            shared_scalar_dtype=(
                kwargs.get("shared_dtype", "bfp8") if shared_topology == "packed_all" else shared_scalar_dtype
            ),
            shared_scalar_fidelity=(
                kwargs.get("shared_fidelity", "HiFi2") if shared_topology == "packed_all" else shared_scalar_fidelity
            ),
            shared_dram=shared_dram,
            projection_geometry=self.projection_geometry,
            prefill_block_w=prefill_block_w,
            effective_geometry=self.effective_geometry,
        )
        return self

    def _make_dram_projection(self, weight, role):
        k, n = tuple(weight.shape)[-2:]
        spec = dict(
            readers=3, input_cores=min(k // 32, 16), block_w=4, per_core_n=math.ceil(n / 2048), bank_tile_multiple=1
        )
        spec.update(self.projection_geometry.get("default", {}))
        spec.update(self.projection_geometry.get(role, {}))
        readers, cores, block_w = spec["readers"], spec["input_cores"], spec["block_w"]
        if readers not in (1, 2, 3):
            raise ValueError("readers must be 1, 2, or 3")
        if cores <= 0 or k % (32 * cores):
            raise ValueError(f"{role}: K={k} requires an integral tiled input shard over {cores} cores")
        if block_w <= 0 or (k // 32) % block_w:
            raise ValueError(f"{role}: block_w must divide tiled K={k // 32}")
        shard_tiles = k // (cores * 32)
        if readers == 1 and (max(block_w, shard_tiles) % min(block_w, shard_tiles)):
            raise ValueError(f"{role}: single-reader block and shard widths must divide one another")
        if spec["per_core_n"] < 1 or spec["bank_tile_multiple"] < 1:
            raise ValueError("per_core_n and bank_tile_multiple must be positive")
        mesh = self.mesh_device
        grid = mesh.compute_with_storage_grid_size()
        banks = mesh.dram_grid_size().x
        bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        unit = 32 * spec["bank_tile_multiple"]
        wm = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(bank_grid, (k, math.ceil(n / (unit * banks)) * unit), ttnn.ShardOrientation.ROW_MAJOR),
        )

        def memory(count, width):
            return ttnn.create_sharded_memory_config(
                (32, width),
                ttnn.num_cores_to_corerangeset(count, grid, row_wise=True),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

        im = memory(cores, k // cores)
        per_n = spec["per_core_n"]
        om = memory(math.ceil(n / (32 * per_n)), 32 * per_n)
        pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=block_w,
            per_core_M=1,
            per_core_N=per_n,
            fused_activation=None,
            num_workers_per_dram_bank=readers,
            use_multicore_pipeline=readers > 1,
        )
        if not hasattr(self, "effective_geometry"):
            self.effective_geometry = {}
        self.effective_geometry[role] = dict(spec, k=k, n=n, input_shard_tiles=shard_tiles)
        return ttnn.to_memory_config(weight, wm), im, om, pc

    def _candidate_project(self, x, role, dtype):
        if x.shape[-2] == 1 and role in self.candidate_decode:
            w, im, om, pc = self.candidate_decode[role]
            out = ttnn.linear(
                ttnn.to_memory_config(x, im),
                w,
                dtype=dtype,
                program_config=pc,
                compute_kernel_config=self.candidate_compute[role],
                memory_config=om,
            )
            return ttnn.to_memory_config(out, DRAM)
        program = {}
        if self.prefill_block_w is not None and x.shape[-2] > 1:
            k, n = tuple(self.candidate_weights[role].shape)[-2:]
            if (k // 32) % self.prefill_block_w:
                raise ValueError("prefill block_w must divide tiled K")
            # Physical prefill chunks are <=256 rows (full) or <=64
            # (linear). Keep the large-N projections distributed in two axes.
            per_m = math.ceil(x.shape[-2] / (8 * 32))
            per_n = math.ceil(n / (8 * 32))
            program["program_config"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                in0_block_w=self.prefill_block_w,
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=per_m,
                out_block_w=per_n,
                per_core_M=per_m,
                per_core_N=per_n,
                fused_activation=None,
                transpose_mcast=False,
            )
        return ttnn.linear(
            x,
            self.candidate_weights[role],
            dtype=dtype,
            compute_kernel_config=self.candidate_compute[role],
            memory_config=DRAM,
            **program,
        )

    def linear(self, x, key, dtype=ttnn.bfloat16, **kwargs):
        if key == "attention_packed" and self.kind == "full_attention":
            if self.attention_topology == "packed":
                return self._candidate_project(x, "attention_packed", dtype)
            return ttnn.concat(
                [
                    self._candidate_project(x, role, dtype)
                    for role in ("attention_q", "attention_k", "attention_v", "attention_gate")
                ],
                dim=-1,
                memory_config=DRAM,
            )
        if key == "self_attn.o_proj.weight" and self.kind == "full_attention":
            return self._candidate_project(x, "attention_out", dtype)
        if key == "mlp.shared_expert.down_proj.weight":
            return self._candidate_project(x, "shared_down", dtype)
        if key == "shared_packed":
            if self.shared_topology == "packed_all":
                return self._candidate_project(x, "shared_packed", dtype)
            roles = (
                ("shared_gu", "shared_scalar")
                if self.shared_topology == "packed_gu_scalar"
                else ("shared_gate", "shared_up", "shared_scalar")
            )
            return ttnn.concat([self._candidate_project(x, role, dtype) for role in roles], dim=-1, memory_config=DRAM)
        return super().linear(x, key, dtype, **kwargs)


class OptimizedDecoder(_ProjectionDecoder):
    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        residual_mode="sharded",
        residual_return_dram=False,
        residual_moe_reshard=True,
        projection_geometry=None,
        **kwargs,
    ):
        if residual_mode not in ("sharded", "control"):
            raise ValueError("residual_mode must be sharded or control")
        geometry = {role: dict(spec) for role, spec in (projection_geometry or {}).items()}
        if residual_mode == "sharded" and hf_config.layer_types[layer_idx] == "full_attention":
            # Output N=2048 at two tiles/core gives exactly the existing
            # 32-core, 64-element-wide RMSNorm/residual layout.
            geometry.setdefault("attention_out", {})["per_core_n"] = 2
        self = super().from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            projection_geometry=geometry,
            **kwargs,
        )
        self.residual_mode = residual_mode
        self.residual_return_dram = residual_return_dram
        self.residual_moe_reshard = residual_moe_reshard
        self._residual_decode_active = False
        if residual_mode == "sharded" and self.kind == "full_attention":
            w, im, _, pc = self.candidate_decode["attention_out"]
            self.candidate_decode["attention_out"] = (w, im, self.norm_memory, pc)
        self.precision_policy["residual_candidate"] = dict(
            mode=residual_mode,
            applicable=self.kind == "full_attention",
            residual_layout="32-core width-sharded L1, shard 32x64" if residual_mode == "sharded" else "inherited",
            attention_out_per_core_n=2 if residual_mode == "sharded" else None,
            return_dram=residual_return_dram,
            moe_output_reshard=residual_moe_reshard,
            prefill_singletons="inherited interleaved-return path via cache_boundary=False",
        )
        return self

    def _candidate_project(self, x, role, dtype):
        if self._residual_decode_active and role == "attention_out" and x.shape[-2] == 1:
            w, im, om, pc = self.candidate_decode[role]
            return ttnn.linear(
                ttnn.to_memory_config(x, im),
                w,
                dtype=dtype,
                program_config=pc,
                compute_kernel_config=self.candidate_compute[role],
                memory_config=om,
            )
        return super()._candidate_project(x, role, dtype)

    def _forward(
        self, hidden_states, *, state, decode, positions=None, start_pos=0, cache_boundary=True, projection_k=512
    ):
        if self.kind != "full_attention" or not decode or not cache_boundary or self.residual_mode != "sharded":
            return super()._forward(
                hidden_states,
                state=state,
                decode=decode,
                positions=positions,
                start_pos=start_pos,
                cache_boundary=cache_boundary,
                projection_k=projection_k,
            )
        # The flag changes dispatch only during this synchronous Python call;
        # captured TT commands contain no host-dependent data or callback.
        self._residual_decode_active = True
        try:
            residual = ttnn.to_memory_config(hidden_states, self.norm_memory)
            normalized = self.norm(residual, "input_layernorm.weight")
            attention = self._full_attention(normalized, state, positions, decode=True, start_pos=start_pos)
            residual = ttnn.add(residual, attention, memory_config=self.norm_memory)
            normalized = self.norm(residual, "post_attention_layernorm.weight")
            # Router/sparse expert consumers retain their proven interleaved
            # contract. Only this branch input changes memory configuration.
            moe = self._moe(ttnn.to_memory_config(normalized, DRAM))
            if self.residual_moe_reshard:
                moe = ttnn.to_memory_config(moe, self.norm_memory)
            result = ttnn.add(residual, moe, memory_config=self.norm_memory)
            return ttnn.to_memory_config(result, DRAM) if self.residual_return_dram else result
        finally:
            self._residual_decode_active = False
