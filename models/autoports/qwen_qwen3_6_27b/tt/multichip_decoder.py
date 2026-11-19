# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6 TP4 decoder. Setup owns host conversion; forwards operate on devices.

The stack residual is [B,S,1280] hidden-sharded (replicated control:5120).
Each rank owns6 full-attention Q heads/1 KV head or4 GDN K heads/12 V heads.
"""

import math
import re

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import (
    _NARROW_GDN_COMBINE_COMPUTE,
    _NARROW_GDN_COMBINE_READER,
    _NARROW_GDN_COMBINE_WRITER,
    OptimizedDecoder,
    PagedState,
    RecurrentState,
    _fused_delta_row_compact_fp32_compute_config,
    _fused_delta_row_GDN_COMPACT_COMPUTE,
    _fused_delta_row_GDN_COMPACT_WRITER,
    _fused_delta_row_GDN_FUSE_ACCUMULATOR,
    _fused_delta_row_GDN_FUSE_DECAY_PREDICT,
    _fused_delta_row_GDN_HOIST_COL_INIT,
    _fused_delta_row_GDN_HOIST_DELTA,
    _fused_delta_row_GDN_HOIST_SCALAR_INIT,
)
from models.autoports.qwen_qwen3_6_27b.tt.page_table_ownership import (
    CertifiedPagedState,
    certify_host_page_rows,
    page_table_certified,
    page_table_for_attention,
)
from models.autoports.qwen_qwen3_6_27b.tt.prefill_rope_candidate import batched_prefill_rope
from models.demos.gpt_oss.tt.ccl import CCLManager

from .precision import group_for_weight


class _BlackholeCCLManager(CCLManager):
    def _init_subdevice(self):
        # CCL worker selection uses the whole default subdevice (11x10 here).
        # Every possible worker must own initialized global semaphore storage.
        grid = self.mesh_device.compute_with_storage_grid_size()
        self.ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )
        self.ccl_sub_device_id = ttnn.SubDeviceId(0)


# Selected at the explicit stage-5 wrap-up cap; measured final defaults.
# Experimental hooks remain opt-in and cannot select a runtime fallback.
_STAGE5_DEFAULTS = {
    "linear_attention": {
        "flat_prefill": True,
        "flat_finish": True,
        "flat_prefill_correction": True,
        "persistent_ccl": True,
        "recurrent_variant": "cumulative_exp_scalar",
        "ccl_decode_memory": "DRAM",
        "mmrs_chunk_rows": 16384,
        "mmrs_max_rows": 32768,
        "mmrs_min_rows": 256,
        "local_fused_conv": True,
        "batched_conv_b1": True,
        "packed_correction_options": {
            "prefill": True,
            "release_separate": True,
            "prefill_config": {"row_cap": 4096, "grid": [8, 8], "out_block_h": 4},
        },
        "fused_packed_swiglu": True,
        "local_affine_gate": True,
        "tiled_prefill_conv": True,
        "flat_packed_gdn": True,
        "batched_prefill_cache": True,
        "local_beta_decay": True,
        "recurrent_column_grain": 1,
        "distributed_decode_norm": "off",
        "attention_fidelity": "HiFi4",
    },
    "full_attention": {
        "flat_prefill": True,
        "flat_finish": True,
        "flat_prefill_correction": True,
        "persistent_ccl": True,
        "packed_qkvg": True,
        "direct_compact_attention": True,
        "gather_norm": "decode_or_single",
        "shard_hidden_norm": True,
        "ccl_decode_memory": "DRAM",
        "mmrs_chunk_rows": 16384,
        "mmrs_max_rows": 32768,
        "batched_prefill_rope": True,
        "mmrs_min_rows": 256,
        "fused_packed_swiglu": True,
        "local_affine_gate": True,
        "tiled_prefill_conv": True,
        "flat_packed_gdn": True,
        "batched_prefill_cache": True,
        "l1_mlp": True,
        "fused_compact_qk_norm": True,
        "decode_sdpa_grid": [8, 4],
        "fused_compact_qk_rope": True,
        "short_capacity_decode_k_chunk": 128,
        "distributed_decode_norm": "off",
    },
}


class MultichipDecoder(OptimizedDecoder):
    _owner_safe_joint_gdn = True

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, policy=None):
        import torch

        self = cls()
        self.config = getattr(hf_config, "text_config", hf_config)
        self.mesh = mesh_device
        assert tuple(mesh_device.shape) == (1, 4), "TP4 requires a 1x4 mesh"
        self.kind = self.config.layer_types[layer_idx]
        self.layer_idx = layer_idx
        self.policy = {
            "residual_layout": "sharded",
            "distributed_decode_norm": "off",
            "flat_ccl": True,
            "flat_norm": True,
            "flat_decode": True,
            "batched_conv": True,
            "batched_conv_b1": False,
            "local_fused_conv": False,
            "batched_prefill_cache": False,
            "fused_compact_qk_norm": False,
            "compact_attention": True,
            "attention_fidelity": "LoFi",
            "narrow_gdn_correction": True,
            "correction_l1": True,
            "fused_mmrs": True,
            "packed_mlp": True,
            "minimal_swiglu": "prefill",
            "gather_norm": "decode" if self.kind == "linear_attention" else "single",
            "prefill_chunk": 512,
            "prefill_grid": [11, 10],
            "prefill_k": 8,
            "dram_sharded": True,
            "readers": 2,
            "cores": 8,
            "block": 2,
            "pipeline": True,
            "prefill_subblock_h": True,
            "round_silu": self.kind == "linear_attention",
            "unfused_g": True,
            "explicit_sum_norm": self.kind == "linear_attention",
            "sfpu_hidden_norm": self.kind == "linear_attention",
            "exact_tree_hidden_norm": True,
            "explicit_fp32_norm": self.kind == "linear_attention",
            "norm_projection_l1": False,
            "sdpa_grid": [8, 4],
            "kv_dtype": "bfloat8_b",
            **_STAGE5_DEFAULTS[self.kind],
            **(policy or {}),
        }
        assert self.policy["residual_layout"] in ("sharded", "replicated")
        if self.policy.get("direct_prefill_qkv", False):
            assert self.kind == "full_attention" and self.policy.get(
                "packed_qkvg", False
            ), "Direct prefill QKV requires full attention with packed QKVG weights"
            assert type(self.policy.get("direct_prefill_qkv_cores", 110)) is int
            assert self.policy.get("direct_prefill_qkv_cores", 110) in (48, 110)
        assert self.policy.get("attention_gate_mode", "off") in ("off", "dram", "l1")
        assert self.policy.get("attention_gate_scope", "decode") in ("decode", "compact")
        assert self.policy.get("attention_gate_cores", 48) in (8, 48)
        if self.kind == "linear_attention" and (
            self.policy.get("recurrent_column_grain", 0) or self.policy.get("recurrent_prediction_registers", False)
        ):
            assert self.policy.get("batched_conv") and self.policy.get(
                "batched_conv_b1"
            ), "Specialized recurrence requires packed convolution for all public batches and S1 tails"
            assert self.policy.get("recurrent_column_grain", 0) in (0, 1, 2)
            if self.policy.get("recurrent_prediction_registers", False):
                assert self.policy.get("recurrent_variant") == "cumulative_exp_scalar"
        self.hidden_width = 1280 if self.policy["residual_layout"] == "sharded" else 5120
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.matmul_configs = {
            "attention": ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4 if self.kind == "linear_attention" else ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=self.kind != "linear_attention",
            ),
            "mlp": ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2 if self.kind == "linear_attention" else ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        }
        for role in ("attention", "mlp"):
            if self.policy.get(role + "_fidelity"):
                self.matmul_configs[role] = ttnn.init_device_compute_kernel_config(
                    mesh_device.arch(),
                    math_fidelity=getattr(ttnn.MathFidelity, self.policy[role + "_fidelity"]),
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=role == "mlp" or self.kind != "linear_attention",
                )
        for group, fidelity in self.policy.get("group_fidelities", {}).items():
            self.matmul_configs[group] = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=getattr(ttnn.MathFidelity, fidelity),
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=group.startswith("mlp") or self.kind != "linear_attention",
            )
        self.ccl = _BlackholeCCLManager(mesh_device, ttnn.get_num_links(mesh_device, 1), ttnn.Topology.Linear)
        self._ccl_buffers = {}
        self._prefill_batch_indices = {}
        self._prefill_cache_bulk_calls = self._prefill_cache_ordered_calls = 0
        self._packed_correction_candidate = None
        self.weights, self.weight_corrections, self.projection_info = {}, {}, {}
        self.precision_observations = {}
        self.narrow_gdn_correction = self.kind == "linear_attention" and self.policy["narrow_gdn_correction"]
        self.swiglu_weight = None
        self.gdn_program = None
        self.shard_norms = {}
        for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            v = (1 + state_dict[name].float()).reshape(1, 1, -1)
            self.weights[name] = self._upload(v, getattr(ttnn, self.policy.get("norm_weight_dtype", "float32")))
            self.shard_norms[name] = self._upload_shards(
                list(v.chunk(4, -1)), getattr(ttnn, self.policy.get("norm_weight_dtype", "float32"))
            )
        for name in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
            self._weight(name, list(state_dict[name].T.float().chunk(4, -1)))
        self._weight("mlp.down_proj.weight", list(state_dict["mlp.down_proj.weight"].T.float().chunk(4, 0)))
        if self.kind == "full_attention":
            qg = state_dict["self_attn.q_proj.weight"].float().reshape(24, 512, 5120)
            q = qg[:, :256].reshape(6144, 5120).T
            gate = qg[:, 256:].reshape(6144, 5120).T
            k, v = (state_dict[f"self_attn.{n}_proj.weight"].float().T for n in ("k", "v"))
            self._weight(
                "packed_qkv", [torch.cat([q.chunk(4, 1)[i], k.chunk(4, 1)[i], v.chunk(4, 1)[i]], 1) for i in range(4)]
            )
            self._weight("gate", list(gate.chunk(4, 1)))
            if self.policy.get("packed_qkvg", False):
                self._weight(
                    "packed_qkvg",
                    [torch.cat([t.chunk(4, 1)[i] for t in (q, k, v, gate)], 1) for i in range(4)],
                )
                self.weights.pop("packed_qkv")
                self.weights.pop("gate")
            self._weight("self_attn.o_proj.weight", list(state_dict["self_attn.o_proj.weight"].T.float().chunk(4, 0)))
            for name in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"):
                self.weights[name] = self._upload(
                    (1 + state_dict[name].float()).reshape(1, 1, -1),
                    getattr(ttnn, self.policy.get("norm_weight_dtype", "float32")),
                )
        else:
            qkv = state_dict["linear_attn.in_proj_qkv.weight"].float().split([2048, 2048, 6144], 0)
            z = state_dict["linear_attn.in_proj_z.weight"].float()
            ba = [state_dict[f"linear_attn.in_proj_{n}.weight"].float() for n in ("b", "a")]
            packed = [
                torch.cat(
                    [
                        *(v.chunk(4, 0)[i] for v in qkv),
                        z.chunk(4, 0)[i],
                        *(torch.nn.functional.pad(v.chunk(4, 0)[i], (0, 0, 0, 20)) for v in ba),
                    ],
                    0,
                ).T.contiguous()
                for i in range(4)
            ]
            self._weight("packed_gdn", packed, correction=True)
            self._weight(
                "linear_attn.out_proj.weight", list(state_dict["linear_attn.out_proj.weight"].T.float().chunk(4, 0))
            )
            for name in ("linear_attn.dt_bias", "linear_attn.A_log"):
                self.weights[name] = self._upload_shards(
                    [v.reshape(1, 1, -1) for v in state_dict[name].float().chunk(4)], ttnn.float32
                )
            self.neg_a = self._upload_shards(
                [(-v.exp()).reshape(1, 1, -1) for v in state_dict["linear_attn.A_log"].float().chunk(4)], ttnn.float32
            )
            self.weights["linear_attn.norm.weight"] = self._upload(
                state_dict["linear_attn.norm.weight"].float().reshape(1, 1, -1),
                getattr(ttnn, self.policy.get("gdn_norm_weight_dtype", "bfloat16")),
            )
            conv = state_dict["linear_attn.conv1d.weight"].float().split([2048, 2048, 6144], 0)
            self.taps = [
                self._upload_shards(
                    [torch.cat([v.chunk(4, 0)[i][:, 0, t] for v in conv]).reshape(1, 1, -1) for i in range(4)]
                )
                for t in range(4)
            ]
            self.taps_rm = [ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT) for t in self.taps]
            self.conv_start = self._upload(torch.tensor([0], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            self.conv_ends = {
                n: self._upload(torch.tensor([n], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                for n in range(1, max(512, self.policy.get("prefill_chunk", 512)) + 1)
            }
            core = ttnn._ttnn.operations.experimental.kda.qkv_causal_conv1d_silu_exchange_core(
                mesh_device.compute_with_storage_grid_size()
            )
            grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})
            self.exchange = self._upload(torch.zeros(1, 12, 2560), layout=ttnn.ROW_MAJOR_LAYOUT)
            self.conv_semaphores = [ttnn.create_global_semaphore(mesh_device, grid, 0) for _ in range(3)]
            self.eye = self._upload(torch.eye(32).reshape(1, 1, 32, 32), ttnn.float32)
            self.lower = self._upload(torch.ones(32, 32).tril().reshape(1, 1, 32, 32), ttnn.float32)
            self.ones = self._upload(torch.ones(1, 1, 32, 32), ttnn.float32)
            ii, jj = torch.arange(32).unsqueeze(1), torch.arange(32).unsqueeze(0)
            self.masks = self._upload(
                torch.cat(
                    [
                        ((ii < 16) & (jj < 16)).float(),
                        ((ii >= 16) & (jj >= 16)).float(),
                        ((ii >= 16) & (jj < 16)).float(),
                    ],
                    1,
                ).reshape(1, 1, 32, 96),
                ttnn.float32,
            )
        if self.policy.get("packed_mlp", False):
            gate, up = (state_dict[f"mlp.{name}_proj.weight"].T.float().chunk(4, -1) for name in ("gate", "up"))
            self._weight("mlp.gate_up.weight", [torch.cat((g, u), -1) for g, u in zip(gate, up)])
        if self.policy.get("minimal_swiglu", False):
            gate, up = (state_dict[f"mlp.{name}_proj.weight"].T.float().chunk(4, -1) for name in ("gate", "up"))
            paired = [
                torch.stack((g.reshape(5120, 136, 32), u.reshape(5120, 136, 32)), dim=2).reshape(5120, 8704)
                for g, u in zip(gate, up)
            ]
            self.swiglu_weight = self._upload_shards(
                paired,
                getattr(
                    ttnn,
                    self.policy.get("weight_group_dtypes", {}).get(
                        "mlp_gate_up", self.policy.get("mlp_dtype", "bfloat8_b")
                    ),
                ),
            )
        if self.policy.get("packed_mlp", False):
            self.weights.pop("mlp.gate_proj.weight")
            self.weights.pop("mlp.up_proj.weight")
        self._prepare_matmuls()
        self.row_parallel_weights = {}
        if self.policy.get("fused_mmrs", False):
            for name in (
                "mlp.down_proj.weight",
                "linear_attn.out_proj.weight" if self.kind == "linear_attention" else "self_attn.o_proj.weight",
            ):
                weight = ttnn.to_memory_config(self.weights[name], ttnn.DRAM_MEMORY_CONFIG)
                # Decode reader geometry can append N padding. MMRS scatters
                # its physical N, so remove that padding before partitioning
                # the logical 5120 channels into four 1280-channel outputs.
                logical_n = self.projection_info.get(name, {}).get("logical_n", 5120)
                self.row_parallel_weights[name] = weight if weight.shape[-1] == logical_n else weight[:, :logical_n]
        if self.kind == "linear_attention" and self.policy.get("packed_correction_options") is not None:
            from .packed_correction_candidate import install

            install(self, **dict(self.policy["packed_correction_options"]))
        from .fused_decode_norm import install as install_decode_norm

        install_decode_norm(self, layer_idx=self.layer_idx)
        return self

    def _upload(self, value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            value.contiguous(),
            device=self.mesh,
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def _upload_shards(self, values, dtype=ttnn.bfloat16):
        import torch

        return ttnn.from_torch(
            torch.cat([v.contiguous() for v in values], -1),
            device=self.mesh,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=-1),
        )

    def upload_residual(self, value):
        return self._upload_shards(list(value.chunk(4, -1))) if self.hidden_width == 1280 else self._upload(value)

    def upload_decode_residual(self, value):
        """Setup-only compact decode input; every layer preserves [1,B,H/TP]."""
        assert value.shape[1] == 1
        return self.upload_residual(value.reshape(1, value.shape[0], value.shape[-1]))

    def _combine_correction(self, primary, correction, key):
        if self.narrow_gdn_correction and key == "packed_gdn":
            return local_narrow_gdn_combine(primary, correction)
        return super()._combine_correction(primary, correction, key)

    def _weight(self, name, shards, correction=False):
        role = "mlp" if name.startswith("mlp.") else "attention"
        self.weights[name] = self._upload_shards(
            shards,
            getattr(
                ttnn,
                self.policy.get("weight_group_dtypes", {}).get(
                    group_for_weight(name), self.policy.get(role + "_dtype", "bfloat8_b")
                ),
            ),
        )
        if correction:
            values = ttnn.to_torch(self.weights[name], mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1)).chunk(
                4, -1
            )
            residuals = [a - b.float() for a, b in zip(shards, values)]
            for r in residuals:
                r[:, 2560:4096] = 0
            if self.narrow_gdn_correction:
                import torch

                # QKV and BA start on tile boundaries; removing zero Z columns
                # preserves every retained BFP8 quantization block exactly.
                residuals = [torch.cat([r[:, :2560], r[:, 4096:4160]], -1) for r in residuals]
            self.weights[name + "_residual"] = self._upload_shards(
                residuals, getattr(ttnn, self.policy.get("gdn_correction_dtype", "bfloat8_b"))
            )
            self.weight_corrections[name] = name + "_residual"

    def _prepare_matmuls(self):
        if not self.policy["dram_sharded"]:
            return
        grid = self.mesh.dram_grid_size()
        banks = grid.x * grid.y
        dram = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        for name, weight in list(self.weights.items()):
            if len(weight.shape) != 2:
                continue
            k, n = weight.shape
            opts = {**self.policy, **self.policy.get("geometry", {}).get(name, {})}
            cores, readers = opts["cores"], opts["readers"]
            assert k % (cores * 32) == 0
            width = math.ceil(n / (32 * math.lcm(banks * readers, cores))) * 32 * math.lcm(banks * readers, cores)
            if width != n:
                # Native tile pad does not accept BFP4. Repacking its exactly
                # BF16-representable values is a setup-only candidate adapter.
                dtype = weight.dtype
                if dtype == ttnn.bfloat4_b:
                    weight = ttnn.typecast(weight, ttnn.bfloat16)
                weight = ttnn.pad(weight, ((0, 0), (0, width - n)), value=0.0)
                if weight.dtype != dtype:
                    weight = ttnn.typecast(weight, dtype)
            memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(dram, (k, width // banks), ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.weights[name] = ttnn.to_memory_config(weight, memory)
            workers = ttnn.num_cores_to_corerangeset(cores, self.mesh.compute_with_storage_grid_size(), row_wise=True)
            self.projection_info[name] = {
                "input_mem": ttnn.create_sharded_memory_config(
                    (32, k // cores), workers, ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
                ),
                "program": ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=opts["block"],
                    per_core_M=1,
                    per_core_N=width // 32 // cores,
                    num_workers_per_dram_bank=readers,
                    use_multicore_pipeline=opts["pipeline"],
                ),
                "logical_n": n,
            }

    def _collective(self, x, *, reduce=False, placement_seq=None):
        shape = tuple(x.shape)
        placement = (
            ttnn.L1_MEMORY_CONFIG
            if (shape[-2] if placement_seq is None else placement_seq) <= 32
            else ttnn.DRAM_MEMORY_CONFIG
        )
        if shape[0] * shape[1] <= 32 and self.policy.get("ccl_decode_memory") == "DRAM":
            placement = ttnn.DRAM_MEMORY_CONFIG
        x = ttnn.to_memory_config(x, placement)
        original_dtype = x.dtype
        if self.policy.get("ccl_dtype") and shape[-1] >= 1280:
            x = ttnn.typecast(x, getattr(ttnn, self.policy["ccl_dtype"]))
        packed_shape = (1, shape[0] * shape[1], shape[2]) if self.policy.get("flat_ccl", False) else shape
        x = ttnn.reshape(x, (1, *packed_shape))
        kw = dict(
            dim=3,
            cluster_axis=1,
            topology=self.ccl.topology,
            num_links=self.ccl.num_links,
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            memory_config=x.memory_config(),
        )
        persistent = self.policy.get("persistent_ccl", False) and x.shape[-2] <= 32
        buffers = None
        if persistent:
            # Prepared by the exact-shape eager warmup before any live trace.
            # Each decoder owns its buffers. Calls on one command queue consume
            # each scratch result before the next call with the same signature.
            signature = (reduce, tuple(x.shape), str(x.dtype), str(x.memory_config()))
            if signature not in self._ccl_buffers:
                output_shape = (*tuple(x.shape)[:-1], x.shape[-1] // 4 if reduce else x.shape[-1] * 4)
                output = ttnn.empty(
                    output_shape, dtype=x.dtype, layout=x.layout, device=self.mesh, memory_config=x.memory_config()
                )
                intermediate = None
                if reduce:
                    # Linear RS stores forward/backward traffic in separate
                    # halves, matching the native compute_output_specs rule.
                    intermediate_shape = list(x.padded_shape)
                    intermediate_shape[0] *= 2
                    intermediate = ttnn.empty(
                        intermediate_shape,
                        dtype=x.dtype,
                        layout=x.layout,
                        device=self.mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                self._ccl_buffers[signature] = (intermediate, output)
            buffers = self._ccl_buffers[signature]
        if reduce:
            y = ttnn.experimental.reduce_scatter_minimal_async(
                x,
                multi_device_global_semaphore=self.ccl.get_rs_ping_pong_semaphore(),
                **({"persistent_output_buffers": list(buffers)} if buffers else {}),
                **kw,
            )
        else:
            y = ttnn.experimental.all_gather_async(
                x,
                mesh_device=self.mesh,
                multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
                **({"persistent_output_tensor": buffers[1]} if buffers else {}),
                **kw,
            )
        if y.dtype != original_dtype:
            y = ttnn.typecast(y, original_dtype)
        return ttnn.reshape(y, (*shape[:2], y.shape[-1]))

    def _prefill_packed_gdn(self, x):
        # Preserve the flat-prefill FP32 primary/correction combine when off.
        if self._packed_correction_candidate is not None:
            return self._packed_correction_candidate.project_prefill(self, x)
        key = "packed_gdn"
        if key in self.weight_corrections:
            return self._combine_correction(
                self._linear_raw(x, key), self._linear_raw(x, self.weight_corrections[key]), key
            )
        return self._linear(x, key)

    def _linear(self, x, key):
        if key == "packed_gdn" and self._packed_correction_candidate is not None:
            if len(x.shape) == 3 and x.shape[0] == 1 and 1 <= x.shape[1] <= 32:
                return self._packed_correction_candidate.project(self, x)
            return self._packed_correction_candidate.project_prefill(self, x)
        # Separate matmul/RS wins small and large prefill. Intermediate M
        # benefits from fusion; above8192 rows the repeated128-row weight
        # reads/dispatch outweigh it. Decode also takes the separate path.
        rows = x.shape[0] * x.shape[1]
        fused_rows = self.policy.get("mmrs_min_rows", 512) < rows <= self.policy.get("mmrs_max_rows", 8192) or (
            rows <= 32 and self.policy.get("fused_mmrs_decode", False)
        )
        if key not in self.row_parallel_weights or self.hidden_width != 1280 or not fused_rows:
            return super()._linear(x, key)
        batch, seq, width = tuple(x.shape)
        flat = ttnn.reshape(ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG), (1, rows, width))
        padded = math.ceil(rows / 32) * 32
        if rows != padded:
            flat = ttnn.pad(flat, ((0, 0), (0, padded - rows), (0, 0)), value=0.0)
        fidelity = (
            (ttnn.MathFidelity.HiFi2 if key.startswith("mlp.") else ttnn.MathFidelity.HiFi4)
            if self.kind == "linear_attention"
            else ttnn.MathFidelity.LoFi
        )
        role = "mlp" if key.startswith("mlp.") else "attention"
        fidelity = (
            getattr(ttnn.MathFidelity, self.policy.get(role + "_fidelity"), fidelity)
            if self.policy.get(role + "_fidelity")
            else fidelity
        )
        if group_for_weight(key) in self.policy.get("group_fidelities", {}):
            fidelity = getattr(ttnn.MathFidelity, self.policy["group_fidelities"][group_for_weight(key)])
        target_dtype = getattr(ttnn, self.policy.get(role + "_activation_dtype", "bfloat16"))
        if flat.dtype != target_dtype:
            flat = ttnn.typecast(flat, target_dtype)
        # The accepted baseline uses 128-row native calls. Larger pieces are
        # candidates after repairing native dedicated-NoC reader ownership.
        # This linear-only partition preserves the caller's logical B/S order.
        chunk_rows = self.policy.get("mmrs_chunk_rows", 128)
        assert chunk_rows >= 32 and chunk_rows % 32 == 0
        pieces = []
        for start in range(0, padded, chunk_rows):
            chunk = flat if start == 0 and chunk_rows >= padded else flat[:, start : min(start + chunk_rows, padded), :]
            if self.policy.get("mmrs_input_l1", False):
                chunk = ttnn.to_memory_config(chunk, ttnn.L1_MEMORY_CONFIG)
            pieces.append(
                ttnn.experimental.matmul_reduce_scatter_prefill(
                    chunk,
                    self.row_parallel_weights[key],
                    cluster_axis=1,
                    scatter_dim=-1,
                    topology=ttnn.Topology.Linear,
                    num_links=self.ccl.num_links,
                    compute_kernel_config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
        output = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=1)
        return ttnn.reshape(output[:, :rows, :], (batch, seq, 1280))

    def _reduce(self, x):
        if x.shape[-1] == 1280 and self.hidden_width == 1280:
            return x
        y = self._collective(x, reduce=True)
        return y if self.hidden_width == 1280 else self._collective(y)

    def _rope(self, x, rope):
        if self.policy.get("batched_prefill_rope", False) and x.shape[0] > 1 and x.shape[2] > 1:
            return batched_prefill_rope(self, x, rope)
        return super()._rope(x, rope)

    def _norm(self, x, key, *, decode=False, logical_batch=None):
        if x.shape[-1] != 1280:
            return super()._norm(x, key)
        if decode and self.policy.get("distributed_decode_norm", "off") != "off":
            assert self._fused_decode_norm is not None
            return self._fused_decode_norm.normalize(self, x, key)
        gather = self.policy.get("gather_norm", False)
        if gather == "decode":
            gather = decode
        elif gather == "decode_or_single":
            gather = decode or (logical_batch or x.shape[0]) == 1
        elif gather == "single":
            gather = (x.shape[0] * x.shape[1] == 1) if decode else (logical_batch or x.shape[0]) == 1
        if gather:
            gathered = self._collective(x)
            if decode and self.policy.get("fused_hidden_norm", False):
                from .hidden_norm_fused_candidate import hidden_norm_fused

                assert self.policy.get("sfpu_hidden_norm") and self.policy.get("exact_tree_hidden_norm")
                return hidden_norm_fused(
                    ttnn.to_memory_config(gathered, ttnn.DRAM_MEMORY_CONFIG),
                    self.weights[key],
                    epsilon=self.config.rms_norm_eps,
                    admission=self.policy.get("hidden_norm_admission", "software"),
                )
            return super()._norm(gathered, key)
        shape = tuple(x.shape)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        work_shape = (1, shape[0] * shape[1], shape[2]) if self.policy.get("flat_norm", False) else shape
        x4 = ttnn.reshape(x, (1, *work_shape))
        stats4 = ttnn.rms_norm_pre_all_gather(x4, compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
        stats = self._collective(ttnn.reshape(stats4, tuple(stats4.shape)[1:]))
        stats4 = ttnn.reshape(stats, (1, *tuple(stats.shape)))
        normalized = ttnn.rms_norm_post_all_gather(
            x4,
            stats4,
            epsilon=self.config.rms_norm_eps,
            weight=self.shard_norms[key],
            compute_kernel_config=self.ckc,
        )
        gathered = self._collective(ttnn.reshape(normalized, work_shape))
        return ttnn.reshape(gathered, (*shape[:2], 5120))

    def _observe_precision(self, label, tensor, field):
        # Boundary metadata is gathered by existing eager preparation/capture.
        # Preserve native arithmetic; checking a fixed dtype creates no op.
        expected = getattr(ttnn, self.policy.get(field, "bfloat16"))
        if tensor.dtype != expected:
            raise ValueError(f"Layer {self.layer_idx} {label}: {tensor.dtype} violates {field}={expected}")
        self.precision_observations.setdefault(
            label, dict(field=field, dtype=str(tensor.dtype), shape=list(tensor.shape), layout=str(tensor.layout))
        )
        return tensor

    def _residual_add(self, value, contribution, label):
        self._observe_precision("residual_input", value, "residual_dtype")
        # Keep the original native add invocation, inferred dtype and placement.
        # The selected contract is enforced against its actual output.
        return self._observe_precision(label, ttnn.add(value, contribution), "residual_dtype")

    def _mlp_input(self, value):
        dtype = getattr(ttnn, self.policy.get("mlp_activation_dtype", "bfloat16"))
        return value if value.dtype == dtype else ttnn.typecast(value, dtype)

    def _prefill_swiglu_with_options(self, h):
        """Opt-in bulk-only geometry/placement family; paired weights and compute config unchanged."""
        options = self.policy["prefill_swiglu_options"]
        assert isinstance(options, dict) and set(options) <= {"k_block", "chunk_rows", "input_l1"}
        kb = options.get("k_block", 4)
        cap = options.get("chunk_rows", 0)
        input_l1 = options.get("input_l1", False)
        assert type(kb) is int and kb in (4, 8)
        assert type(cap) is int and (cap == 0 or cap >= 32 and cap % 32 == 0)
        assert type(input_l1) is bool and (not input_l1 or 32 <= cap <= 2048)
        batch, seq, width = tuple(h.shape)
        rows = batch * seq
        assert width == 5120 and rows > 512 and self.swiglu_weight is not None
        assert h.dtype == ttnn.bfloat16
        flat = ttnn.reshape(ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG), (1, rows, width))
        padded = (rows + 31) // 32 * 32
        if padded != rows:
            flat = ttnn.pad(flat, ((0, 0), (0, padded - rows), (0, 0)), value=0.0)
        cap = cap or padded
        config = ttnn.MinimalMatmulConfig(
            M_block_size=4,
            K_block_size=kb,
            N_block_size=16,
            subblock_h=1,
            subblock_w=4,
            compute_with_storage_grid_size=(11, 10),
        )

        def project_piece(start, stop):
            # Local lifetime avoids retaining a previous L1 piece while admitting the next.
            piece = flat if start == 0 and stop == padded else flat[:, start:stop, :]
            if input_l1:
                piece = ttnn.to_memory_config(piece, ttnn.L1_MEMORY_CONFIG)
            return ttnn.experimental.minimal_matmul(
                self._mlp_input(piece),
                self.swiglu_weight,
                config=config,
                fuse_swiglu=True,
                compute_kernel_config=self.matmul_configs.get("mlp_gate_up", self.matmul_configs["mlp"]),
                dtype=getattr(ttnn, self.policy.get("activation_dtype", "bfloat16")),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        pieces = [project_piece(start, min(start + cap, padded)) for start in range(0, padded, cap)]
        act = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=1)
        if padded != rows:
            act = act[:, :rows, :]
        return ttnn.reshape(act, (batch, seq, 4352))

    def _finish(self, x, attention, prefill=False, *, decode=False):
        original_shape = tuple(x.shape)
        flat = prefill and self.policy.get("flat_finish", False) and x.shape[0] > 1 and x.shape[1] > 1
        if flat:
            rows = x.shape[0] * x.shape[1]
            x = ttnn.reshape(x, (1, rows, x.shape[-1]))
            attention = ttnn.reshape(attention, (1, rows, attention.shape[-1]))
        x = self._residual_add(x, self._reduce(attention), "residual_after_attention")
        h = self._norm(x, "post_attention_layernorm.weight", decode=decode, logical_batch=original_shape[0])
        self._observe_precision("normalized_mlp_input", h, "activation_dtype")
        swiglu_rows = h.shape[0] * h.shape[1]
        tail_packed = prefill and swiglu_rows <= 32 and self.policy.get("prefill_swiglu_tail_packed", False)
        if tail_packed:
            assert self.policy.get("packed_mlp") and self.policy.get(
                "fused_packed_swiglu"
            ), "Prefill tail requires the existing packed projection and fused epilogue"
            # Own flattening of either [B,1,H] or [1,B,H], including non-aligned public B.
            tail_shape = tuple(h.shape)
            h = ttnn.reshape(h, (1, swiglu_rows, 5120))
        if (
            self.swiglu_weight is not None
            and (prefill or self.policy.get("minimal_swiglu") is True)
            and not tail_packed
        ):
            if prefill and swiglu_rows > 512 and self.policy.get("prefill_swiglu_options"):
                act = self._prefill_swiglu_with_options(h)
            else:
                batch, seq, _ = h.shape
                flat = ttnn.reshape(ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG), (1, batch * seq, 5120))
                small = 32 < batch * seq <= 512
                act = ttnn.experimental.minimal_matmul(
                    self._mlp_input(flat),
                    self.swiglu_weight,
                    config=ttnn.MinimalMatmulConfig(
                        M_block_size=2 if small else 4,
                        K_block_size=8 if small else 4,
                        N_block_size=26 if small else 16,
                        subblock_h=1,
                        subblock_w=2 if small else 4,
                        compute_with_storage_grid_size=(11, 8) if small else (11, 10),
                    ),
                    fuse_swiglu=True,
                    compute_kernel_config=self.matmul_configs.get("mlp_gate_up", self.matmul_configs["mlp"]),
                    dtype=getattr(ttnn, self.policy.get("activation_dtype", "bfloat16")),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                act = ttnn.reshape(act, (batch, seq, 4352))
        else:
            fused_epilogue = (
                (not prefill or tail_packed)
                and self.policy.get("fused_packed_swiglu", False)
                and self.policy.get("packed_mlp", False)
            )
            if self.policy.get("packed_mlp", False):
                packed = self._linear(h, "mlp.gate_up.weight")
                if fused_epilogue:
                    from .packed_swiglu_candidate import packed_swiglu

                    act = packed_swiglu(packed, round_silu=self.policy["round_silu"])
                else:
                    gate, up = packed[:, :, :4352], packed[:, :, 4352:]
            else:
                gate = self._linear(h, "mlp.gate_proj.weight")
                up = self._linear(h, "mlp.up_proj.weight")
            if not fused_epilogue:
                act = (
                    ttnn.multiply(ttnn.silu(gate), up)
                    if self.policy["round_silu"]
                    else ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
                )
        if tail_packed:
            act = ttnn.reshape(act, (*tail_shape[:2], 4352))
        self._observe_precision("mlp_activation_output", act, "activation_dtype")
        result = self._residual_add(x, self._reduce(self._linear(act, "mlp.down_proj.weight")), "residual_after_mlp")
        return ttnn.reshape(result, original_shape) if flat else result

    def create_recurrent_state(self, batch=1):
        import torch

        return RecurrentState(
            self._upload(
                torch.zeros(batch, 3, 2560),
                getattr(ttnn, self.policy.get("activation_dtype", "bfloat16")),
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            self._upload(
                torch.zeros(batch, 12, 128, 128), getattr(ttnn, self.policy.get("recurrent_state_dtype", "bfloat16"))
            ),
        )

    def create_paged_state(self, page_table):
        import torch

        assert page_table.ndim == 2 and 1 <= page_table.shape[0] <= 32
        blocks = int(page_table.max()) + 1
        if page_table.shape[1] >= 16:
            extra = (-page_table.shape[1]) % 16
            padding = torch.arange(blocks, blocks + page_table.shape[0] * extra, dtype=page_table.dtype)
            page_table = torch.cat((page_table, padding.reshape(page_table.shape[0], extra)), 1)
            blocks += padding.numel()
        kv_dtype = self.policy["kv_dtype"]
        assert kv_dtype in ("bfloat16", "bfloat8_b", "bfloat4_b"), "Unsupported cache precision"
        key = self._upload(torch.zeros(blocks, 1, 32, 256), getattr(ttnn, kv_dtype))
        value = self._upload(torch.zeros(blocks, 1, 32, 256), getattr(ttnn, kv_dtype))
        table = self._upload(page_table.to(torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        batch = page_table.shape[0]
        if not self.policy.get("batched_prefill_cache", False) or batch == 1:
            return PagedState(key, value, table)
        if batch not in self._prefill_batch_indices:
            self._prefill_batch_indices[batch] = self._upload(
                torch.arange(batch, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
            )
        return CertifiedPagedState(key, value, table, certified=certify_host_page_rows(page_table.tolist()))

    @staticmethod
    def _attention_page_table(state):
        return page_table_for_attention(state)

    @staticmethod
    def _attention_state_view(state):
        return (
            PagedState(state.key, state.value, page_table_for_attention(state))
            if isinstance(state, CertifiedPagedState)
            else state
        )

    def _attention_chunk(self, state):
        return OptimizedDecoder._attention_chunk(self._attention_state_view(state))

    def _decode_attention_chunk(self, state, *, decode=False):
        return super()._decode_attention_chunk(self._attention_state_view(state), decode=decode)

    def _attention_prefill(self, x, state, *, start_pos, rope):
        # B1 and one-token tails retain the inherited ordered implementation.
        return super()._attention_prefill(x, self._attention_state_view(state), start_pos=start_pos, rope=rope)

    def replace_paged_table_from_host(self, state, page_table):
        """Setup only, before warmup/capture: upload a fresh private table lease.

        Existing traces must be released/recaptured after changing buffer addresses.
        This cannot certify an already-exposed device tensor in place.
        """
        import torch

        assert isinstance(state, CertifiedPagedState)
        assert not ttnn.is_trace_capture_active(self.mesh)
        assert tuple(page_table.shape) == state.page_table_shape
        assert int(page_table.min()) >= -1 and int(page_table.max()) < state.key.shape[0]
        private_host = page_table.to(torch.int32).contiguous().clone()
        table = self._upload(private_host, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        state._replace_with_fresh_private_table(table, certified=certify_host_page_rows(private_host.tolist()))

    def _prefill_cache_fill(self, state, key, value, *, start_pos):
        batch = key.shape[0]
        table = self._attention_page_table(state)
        if (
            batch > 1
            and self.policy.get("batched_prefill_cache", False)
            and page_table_certified(state)
            and batch in self._prefill_batch_indices
            and not ttnn.is_trace_capture_active(self.mesh)
        ):
            from .batched_cache_fill_candidate import batched_fill_pair

            self._prefill_cache_bulk_calls += 1
            batched_fill_pair(
                state.key,
                state.value,
                key,
                value,
                table,
                self._prefill_batch_indices[batch],
                start_pos=start_pos,
                disjoint_write_pages=True,
            )
        else:
            # Captured prefill also retains the ordered loop: a later public
            # table exposure cannot change an already-captured branch choice.
            self._prefill_cache_ordered_calls += 1
            shifted = table[:, start_pos // 32 :]
            for row in range(batch):
                ttnn.experimental.paged_fill_cache(state.key, key[row : row + 1], shifted, batch_idx=row)
                ttnn.experimental.paged_fill_cache(state.value, value[row : row + 1], shifted, batch_idx=row)

    def _qkv(self, x, rope, *, prefill=False, logical_shape=None):
        compact = logical_shape is None and self.policy.get("flat_attention", False) and x.shape[1] == 1
        batch = x.shape[0]
        packed_gate = self.policy.get("packed_qkvg", False)
        projected = self._linear(
            ttnn.reshape(x, (1, batch, 5120)) if compact else x, "packed_qkvg" if packed_gate else "packed_qkv"
        )
        if compact:
            projected = ttnn.reshape(projected, (batch, 1, 3584 if packed_gate else 2048))
        direct_prefill = (
            prefill
            and self.policy.get("direct_prefill_qkv", False)
            and not compact
            and (logical_shape is not None or x.shape[0] == 1)
        )
        if direct_prefill:
            from .prefill_qkv_candidate import prefill_qkv

            assert packed_gate, "Direct prefill QKV requires the packed QKVG projection"
            logical_batch, logical_seq = logical_shape if logical_shape is not None else tuple(x.shape)[:2]
            assert logical_shape is not None or logical_batch == 1, "Direct prefill requires compact rows"
            q, k, v, gate = prefill_qkv(
                ttnn.to_memory_config(projected, ttnn.DRAM_MEMORY_CONFIG),
                batch=logical_batch,
                seq=logical_seq,
                cores=self.policy.get("direct_prefill_qkv_cores", 110),
            )
        else:
            gate = projected[:, :, 2048:] if packed_gate else None
            projected = projected[:, :, :2048] if packed_gate else projected
            if logical_shape is not None:
                projected = ttnn.reshape(projected, (*logical_shape, 2048))
            q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                projected, num_heads=6, num_kv_heads=1, transpose_key=False
            )
        if compact:
            q = ttnn.reshape(
                self._norm(ttnn.reshape(q, (1, 1, batch * 6, 256)), "self_attn.q_norm.weight"), (batch, 6, 1, 256)
            )
            k = ttnn.reshape(
                self._norm(ttnn.reshape(k, (1, 1, batch, 256)), "self_attn.k_norm.weight"), (batch, 1, 1, 256)
            )
        else:
            q, k = self._norm(q, "self_attn.q_norm.weight"), self._norm(k, "self_attn.k_norm.weight")
        return self._rope(q, rope), self._rope(k, rope), v, gate

    def _local_compact_attention_decode(self, x, state, current_pos, rope, *, decode=False, decode_batch=None):
        batch = x.shape[0] if decode_batch is None else decode_batch
        xf = ttnn.reshape(x, (1, batch, 5120)) if decode_batch is None else x
        packed_gate = self.policy.get("packed_qkvg", False)
        projected = self._linear(xf, "packed_qkvg" if packed_gate else "packed_qkv")
        heads_mem = self._compact_height_mem(self._compact_cores(batch), 32, 256)
        if self.policy.get("packed_compact_qk_rope", False):
            from .packed_qk_rope_candidate import packed_qk_rope

            assert packed_gate, "Packed QK/RoPE requires packed QKVG projection"
            q, k, v, gate = packed_qk_rope(
                projected,
                self.weights["self_attn.q_norm.weight"],
                self.weights["self_attn.k_norm.weight"],
                *rope,
                epsilon=self.config.rms_norm_eps,
            )
            k_update = k
        else:
            gate = projected[:, :, 2048:] if packed_gate else None
            projected = projected[:, :, :2048] if packed_gate else projected
            q, k, v = _local_qkv_split(ttnn.reshape(projected, (1, 1, batch, 2048)), memory_config=heads_mem)
            if self.policy.get("fused_compact_qk_rope", False):
                from .parallel_qk_rope_candidate import compact_qk_rope

                q, k = compact_qk_rope(
                    q,
                    k,
                    self.weights["self_attn.q_norm.weight"],
                    self.weights["self_attn.k_norm.weight"],
                    *rope,
                    epsilon=self.config.rms_norm_eps,
                )
                k_update = k
            else:
                if self.policy.get("fused_compact_qk_norm", False):
                    from .parallel_qk_norm_candidate import compact_qk_norm

                    q, k = compact_qk_norm(
                        q,
                        k,
                        self.weights["self_attn.q_norm.weight"],
                        self.weights["self_attn.k_norm.weight"],
                        epsilon=self.config.rms_norm_eps,
                    )
                else:
                    q, k = (
                        ttnn.rms_norm(
                            ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG),
                            weight=self.weights[name],
                            epsilon=self.config.rms_norm_eps,
                            compute_kernel_config=self.ckc,
                        )
                        for t, name in ((q, "self_attn.q_norm.weight"), (k, "self_attn.k_norm.weight"))
                    )
                rot_mem = self._compact_rope_mem(batch, self.mesh.compute_with_storage_grid_size())
                trig = tuple(ttnn.to_memory_config(ttnn.permute(t, (2, 0, 1, 3)), rot_mem) for t in rope)

                def rotate(t):
                    prefix = ttnn.to_memory_config(t[..., :64], rot_mem)
                    prefix = ttnn.experimental.rotary_embedding_hf(
                        prefix, *trig, is_decode_mode=True, memory_config=rot_mem, compute_kernel_config=self.ckc
                    )
                    return ttnn.concat([ttnn.to_memory_config(prefix, ttnn.DRAM_MEMORY_CONFIG), t[..., 64:]], dim=-1)

                q, k = rotate(q), rotate(k)
                k_update = ttnn.to_memory_config(
                    k, self._compact_height_mem(self._compact_cores(batch, offset=32), 32, 256)
                )
        ttnn.experimental.paged_fused_update_cache(
            state.key,
            k_update,
            state.value,
            v,
            update_idxs_tensor=current_pos,
            page_table=self._attention_page_table(state),
        )
        del k_update, k, v
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            state.key,
            state.value,
            page_table_tensor=self._attention_page_table(state),
            cur_pos_tensor=current_pos,
            scale=256**-0.5,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(
                    *self.policy.get("decode_sdpa_grid", [8, 8] if batch > 1 else [8, 4])
                ),
                q_chunk_size=32,
                k_chunk_size=self._decode_attention_chunk(state, decode=decode),
                exp_approx_mode=False,
            ),
            compute_kernel_config=self.ckc,
        )
        gate_mode = self.policy.get("attention_gate_mode", "off")
        if gate_mode != "off" and (decode or self.policy.get("attention_gate_scope", "decode") == "compact"):
            from .attention_gate_candidate import attention_gate, canonical_output_memory

            destination = ttnn.DRAM_MEMORY_CONFIG
            if gate_mode == "l1":
                key = "self_attn.o_proj.weight"
                assert key not in self.weight_corrections
                assert self.policy.get("attention_activation_dtype", "bfloat16") == "bfloat16"
                fused_rows = self.policy.get("mmrs_min_rows", 512) < batch <= self.policy.get(
                    "mmrs_max_rows", 8192
                ) or (batch <= 32 and self.policy.get("fused_mmrs_decode", False))
                assert not fused_rows, "Direct L1 gate requires the separate DRAM-sharded projection"
                destination = self.projection_info[key]["input_mem"]
                assert destination == canonical_output_memory(self.mesh)
            out = attention_gate(
                out,
                gate if packed_gate else self._linear(xf, "gate"),
                memory_config=destination,
                cores=self.policy.get("attention_gate_cores", 48),
            )
        else:
            out = ttnn.to_memory_config(out, heads_mem)
            grid = self.mesh.compute_with_storage_grid_size()
            full_grid = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
            )
            out = ttnn.experimental.nlp_concat_heads_decode(out, num_heads=6, sub_core_grids=full_grid)
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
            out = ttnn.reshape(out[:, :, :batch, :], (1, batch, 1536))
            out = ttnn.multiply(
                gate if packed_gate else self._linear(xf, "gate"),
                out,
                input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID],
            )
        out = self._linear(out, "self_attn.o_proj.weight")
        if decode and self.policy.get("flat_decode", False):
            return out
        # A one-token prefill/resume returns the public [B,1,H] layout. Expanding
        # compact L1 rows adds tile padding per user and can collide with the
        # following reduce-scatter CBs. Release the private L1 projection before
        # that expansion; actual compact decode keeps its measured L1 path.
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(out, (batch, 1, 5120))

    def _attention_decode(self, x, state, current_pos, rope, *, decode=False):
        if self.policy.get("compact_attention", False):
            return self._local_compact_attention_decode(x, state, current_pos, rope, decode=decode)
        b = x.shape[0]
        q, k, v, gate = self._qkv(x, rope)
        inputs = []
        for offset, tensor in ((0, k), (b, v)):
            cores = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(i % 8, i // 8), ttnn.CoreCoord(i % 8, i // 8))
                    for i in range(offset, offset + b)
                }
            )
            mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(cores, [32, 256], ttnn.ShardOrientation.ROW_MAJOR),
            )
            inputs.append(ttnn.to_memory_config(ttnn.permute(tensor, (2, 0, 1, 3)), mem))
        ttnn.experimental.paged_fused_update_cache(
            state.key,
            inputs[0],
            state.value,
            inputs[1],
            update_idxs_tensor=current_pos,
            page_table=self._attention_page_table(state),
        )
        del inputs
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            ttnn.permute(q, (2, 0, 1, 3)),
            state.key,
            state.value,
            page_table_tensor=self._attention_page_table(state),
            cur_pos_tensor=current_pos,
            scale=256**-0.5,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(
                    *self.policy.get("decode_sdpa_grid", [8, 8] if b > 1 else [8, 4])
                ),
                q_chunk_size=32,
                k_chunk_size=self._decode_attention_chunk(state, decode=decode),
                exp_approx_mode=False,
            ),
            compute_kernel_config=self.ckc,
        )
        out = ttnn.permute(out[:, :, :6, :], (1, 2, 0, 3))
        if self.policy.get("flat_attention", False):
            flat = ttnn.reshape(x, (1, b, 5120))
            out = ttnn.reshape(ttnn.transformer.concatenate_heads(out), (1, b, 1536))
            out = ttnn.multiply(self._linear(flat, "gate"), out, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
            out = self._linear(out, "self_attn.o_proj.weight")
            return out if decode and self.policy.get("flat_decode", False) else ttnn.reshape(out, (b, 1, out.shape[-1]))
        return self._attention_out(x, out, gate)

    def _local_batched_conv(self, qkv, history, *, logical_batch=None):
        capacity, _, channels = history.shape
        batch = capacity if logical_batch is None else logical_batch
        if type(batch) is not int or not 1 <= batch <= capacity:
            raise ValueError(f"Convolution logical batch {batch!r} exceeds capacity {capacity}")
        if (
            self.policy.get("local_fused_conv", False)
            and 1 <= batch <= 32
            and channels == 2560
            and tuple(qkv.shape) == (1, batch, channels)
        ):
            from .recurrent_next_candidate import local_conv

            return local_conv(qkv, history, self.taps_rm, logical_batch=batch)
        if batch != capacity:
            raise ValueError("Prefix decode requires the local fused convolution backend")
        token = ttnn.reshape(ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT), (batch, 1, channels))
        inputs = [ttnn.reshape(history[:, j : j + 1], (1, batch, channels)) for j in range(3)]
        inputs.append(ttnn.reshape(token, (1, batch, channels)))
        products = [ttnn.multiply(x, tap, dtype=ttnn.float32) for x, tap in zip(inputs, self.taps_rm)]
        acc = ttnn.add(products[0], products[1], dtype=ttnn.float32)
        acc = ttnn.add(acc, products[2], dtype=ttnn.float32)
        result = ttnn.silu(ttnn.add(acc, products[3], dtype=ttnn.bfloat16))
        ttnn.copy(ttnn.concat([history[:, 1:3], token], dim=1), history)
        return ttnn.reshape(result, (batch, 1, channels))

    def _gdn(self, x, state, decode_batch=None, *, packed=None, flat_output=False):
        if packed is None:
            b, s = (decode_batch, 1) if decode_batch is not None else tuple(x.shape)[:2]
            packed = self._linear(x, "packed_gdn")
        else:
            assert decode_batch is None
            b, s = tuple(packed.shape)[:2]
        capacity = state.conv.shape[0]
        if b != capacity and (
            decode_batch is None
            or not 1 <= b <= capacity
            or not self.policy.get("local_fused_conv", False)
            or not self.policy.get("batched_conv", False)
            or (b == 1 and not self.policy.get("batched_conv_b1", False))
            or not self.policy.get("recurrent_column_grain", 0)
        ):
            raise ValueError(
                f"Prefix GDN batch {b} of {capacity} requires compact decode, local fused convolution, "
                "and the column recurrence backend"
            )
        qkv = packed[:, :, :2560]
        padded = math.ceil(s / 32) * 32
        use_batched_conv = (
            s == 1 and self.policy.get("batched_conv", False) and (b > 1 or self.policy.get("batched_conv_b1", False))
        )
        if (
            s > 1
            and self.policy.get("tiled_prefill_conv", False)
            and (b > 1 or self.policy.get("tiled_prefill_conv_b1", False))
        ):
            from .tiled_prefill_conv_candidate import tiled_prefill_conv

            q, k, v = tiled_prefill_conv(
                qkv,
                state.conv,
                self.taps,
                batch=b,
                seq=s,
                block_ct=self.policy.get("prefill_conv_block_ct", 8),
            )
        elif s > 1 and self.policy.get("batched_prefill_conv", False):
            from .prefill_conv_candidate import batched_prefill_conv

            q, k, v = batched_prefill_conv(qkv, state.conv, self.taps_rm)
        elif use_batched_conv:
            # One logical row per user avoids B padded convolution launches.
            q = k = v = self._local_batched_conv(qkv, state.conv, logical_batch=b)
        else:
            if s != padded:
                qkv = ttnn.pad(qkv, ((0, 0), (0, padded - s), (0, 0)), value=0.0)
            outputs = [
                ttnn.experimental.kda.qkv_causal_conv1d_silu_with_carry(
                    qkv[row : row + 1],
                    state.conv[row : row + 1],
                    *self.taps,
                    512,
                    512,
                    1536,
                    actual_start=self.conv_start,
                    actual_end=self.conv_ends[s],
                    exchange_buffer=self.exchange,
                    exchange_semaphores=self.conv_semaphores,
                    program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=256),
                    compute_kernel_config=self.matmul_configs["attention"],
                )
                for row in range(b)
            ]
            q, k, v = (ttnn.concat([o[i] for o in outputs], 0) if b > 1 else outputs[0][i] for i in range(3))
            ttnn.copy(ttnn.concat([o[3] for o in outputs], 0) if b > 1 else outputs[0][3], state.conv)
        if decode_batch is not None and self.policy.get("local_beta_decay", False):
            from .local_beta_decay_candidate import local_beta_decay

            beta, g = local_beta_decay(packed, self.weights["linear_attn.dt_bias"], self.neg_a)
        else:
            beta = ttnn.sigmoid(packed[:, :, 4096:4108])
            g = self._decay(packed[:, :, 4128:4140])
            if decode_batch is not None:
                beta, g = (ttnn.reshape(t, (b, 1, 12)) for t in (beta, g))
        if s == 1:
            # Exact optimized recurrence/readout arithmetic, with local head
            # addressing. Keep FP32 live readout and rounded persistence separate.
            recurrent = _local12_recurrent
            recurrent_kwargs = {}
            if self.policy.get("recurrent_column_grain", 0):
                from .column_recurrent_runtime import column_recurrent

                assert use_batched_conv, "Column recurrence requires the packed QKV owner"
                recurrent = column_recurrent
                recurrent_kwargs["grain"] = self.policy["recurrent_column_grain"]
                recurrent_kwargs["logical_batch"] = b
            if self.policy.get("recurrent_prediction_registers", False):
                from .recurrent_prediction_candidate import prediction_sources
                from .recurrent_variants import sources

                assert use_batched_conv and self.policy.get("recurrent_variant") == "cumulative_exp_scalar"
                pair = prediction_sources(
                    *sources(
                        _LOCAL12_READER,
                        _fused_delta_row_GDN_COMPACT_COMPUTE,
                        "cumulative_exp_scalar",
                        grouped=b >= 3,
                    )
                )
                if self.policy.get("recurrent_column_grain", 0):
                    from .column_recurrent_runtime import column_sources

                    recurrent_kwargs["source_overrides"] = column_sources(*pair, _fused_delta_row_GDN_COMPACT_WRITER)
                else:
                    recurrent_kwargs["source_overrides"] = pair
            out = recurrent(
                q,
                k,
                v,
                g,
                beta,
                # Preserve the allocated state address/extent; the column
                # kernel limits work to logical prefix heads when b < capacity.
                ttnn.reshape(state.recurrent, (capacity * 12, 128, 128)),
                packed_qkv=use_batched_conv,
                variant=self.policy.get("recurrent_variant", "baseline") if use_batched_conv else "baseline",
                **recurrent_kwargs,
            )
            if decode_batch is not None and self.policy.get("local_norm_gate", False):
                from .local_norm_gate_candidate import local_norm_gate

                out = local_norm_gate(
                    out,
                    self.weights["linear_attn.norm.weight"],
                    packed,
                    epsilon=self.config.rms_norm_eps,
                    ownership=self.policy["local_norm_gate"],
                )
                return self._linear(out, "linear_attn.out_proj.weight")
            out = ttnn.rms_norm(
                ttnn.typecast(out, ttnn.bfloat16), epsilon=self.config.rms_norm_eps, compute_kernel_config=self.ckc
            )
            if decode_batch is not None and self.policy.get("local_affine_gate", False):
                from .local_affine_gate_candidate import local_affine_gate

                out = local_affine_gate(out, self.weights["linear_attn.norm.weight"], packed)
            else:
                out = ttnn.multiply(out, self.weights["linear_attn.norm.weight"], dtype=ttnn.bfloat16)
                out = ttnn.reshape(out, (1, b, 1536) if decode_batch is not None else (b, 1, 1536))
                out = ttnn.multiply(
                    ttnn.typecast(packed[:, :, 2560:4096], ttnn.float32),
                    ttnn.typecast(out, ttnn.float32),
                    dtype=ttnn.bfloat16,
                    input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                )
            return self._linear(out, "linear_attn.out_proj.weight")
        if s != padded:
            beta, g = (ttnn.pad(t, ((0, 0), (0, padded - s), (0, 0)), value=0.0) for t in (beta, g))
        out, updated = ttnn.transformer.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=state.recurrent,
            output_final_state=True,
            chunk_size=32,
            num_key_heads=4,
            output_head_major=True,
            eye=self.eye,
            tril=self.lower,
            ones=self.ones,
            masks=self.masks,
            compute_kernel_config=self.ckc,
        )
        ttnn.typecast(updated, state.recurrent.dtype, output_tensor=state.recurrent)
        out = ttnn.reshape(out, ttnn.Shape((b, 12, s, 128)), ttnn.Shape((b, 12, padded, 128)))
        out = ttnn.rms_norm(
            ttnn.typecast(out, ttnn.bfloat16),
            weight=self.weights["linear_attn.norm.weight"],
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=self.ckc,
        )
        out = ttnn.transformer.concatenate_heads(out)
        out = ttnn.multiply(ttnn.silu(packed[:, :, 2560:4096]), out)
        if flat_output:
            out = ttnn.reshape(out, (1, b * s, 1536))
        return self._linear(out, "linear_attn.out_proj.weight")

    def prefill_forward(self, x, *, state, start_pos=0, rope=None):
        if self.policy.get("flat_prefill", False):
            from .flat_prefill import flat_prefill_forward

            return flat_prefill_forward(self, x, state=state, start_pos=start_pos, rope=rope)
        return super().prefill_forward(x, state=state, start_pos=start_pos, rope=rope)

    def decode_forward(self, x, *, state, current_pos=None, rope=None, logical_batch=None):
        capacity = (
            state.conv.shape[0] if self.kind == "linear_attention" else self._attention_page_table(state).shape[0]
        )
        batch = capacity if logical_batch is None else logical_batch
        if type(batch) is not int or not 1 <= batch <= capacity:
            raise ValueError(f"Decode logical batch {batch!r} exceeds state capacity {capacity}")
        if self.kind == "full_attention" and batch != capacity:
            raise ValueError("Prefix attention decode requires a matching page-table row view")
        native_compact = tuple(x.shape) == (1, batch, self.hidden_width)
        assert native_compact or tuple(x.shape) == (batch, 1, self.hidden_width)
        compact = self.policy.get("flat_decode", False)
        assert compact or not native_compact or batch == 1
        if compact:
            assert self.kind != "linear_attention" or self.policy.get("batched_conv", False)
            if not native_compact:
                x = ttnn.reshape(x, (1, batch, self.hidden_width))
        h = self._norm(x, "input_layernorm.weight", decode=True)
        if self.kind == "linear_attention":
            attention = self._gdn(h, state, decode_batch=batch if compact else None)
        elif compact and self.policy.get("direct_compact_attention", False):
            attention = self._local_compact_attention_decode(
                h, state, current_pos, rope, decode=True, decode_batch=batch
            )
        else:
            attention = self._attention_decode(
                ttnn.reshape(h, (batch, 1, 5120)) if compact else h, state, current_pos, rope, decode=True
            )
            if compact:
                attention = ttnn.reshape(attention, (1, batch, attention.shape[-1]))
        result = self._finish(x, attention, decode=True)
        return ttnn.reshape(result, (batch, 1, self.hidden_width)) if compact and not native_compact else result


# Local-12 address specialization; compute/writer arithmetic is imported unchanged
# from the validated optimized decoder. See doc/multichip_decoder/AUTODEBUG.md.
_LOCAL12_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t NORMALIZED=get_compile_time_arg_val(0);
    constexpr uint32_t G_FP32=get_compile_time_arg_val(1), B_FP32=get_compile_time_arg_val(2);
    constexpr uint32_t Q_STRIDE=get_compile_time_arg_val(3), V_STRIDE=get_compile_time_arg_val(4);
    constexpr bool ROW_MAJOR=get_compile_time_arg_val(5);
    constexpr bool PACKED=get_compile_time_arg_val(6);
    constexpr uint32_t STATE_BYTES=get_compile_time_arg_val(7);
    constexpr auto qa=TensorAccessorArgs<8>();
    constexpr auto ka=TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va=TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto ga=TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ba=TensorAccessorArgs<ga.next_compile_time_args_offset()>();
    constexpr auto sa=TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    const auto q=TensorAccessor(qa,get_arg_val<uint32_t>(0),PACKED?5120:(ROW_MAJOR?1024:(NORMALIZED?4096:2048)));
    const auto k=TensorAccessor(ka,get_arg_val<uint32_t>(1),PACKED?5120:(ROW_MAJOR?1024:(NORMALIZED?4096:2048)));
    const auto v=TensorAccessor(va,get_arg_val<uint32_t>(2),PACKED?5120:(ROW_MAJOR?3072:2048));
    const auto g=TensorAccessor(ga,get_arg_val<uint32_t>(3),G_FP32?4096:2048);
    const auto b=TensorAccessor(ba,get_arg_val<uint32_t>(4),B_FP32?4096:2048);
    const auto s=TensorAccessor(sa,get_arg_val<uint32_t>(5),STATE_BYTES);
    const uint32_t start=get_arg_val<uint32_t>(6),count=get_arg_val<uint32_t>(7);
    Noc noc; CircularBuffer scratch(15);
    scratch.reserve_back(1);
    const uint32_t scratch_addr=scratch.get_write_ptr();
    // Reader CB rings are exactly one head deep. Only active row values change;
    // clear padding once, then preserve it across every ring wrap.
    for(uint32_t id=0;id<5;id++) {
        CircularBuffer cb(id); const uint32_t n=id<3?4:1;
        cb.reserve_back(n); noc.async_write_zeros(cb,2*n*4096);
    }
    noc.write_zeros_l1_barrier();
    for(uint32_t id=6;id<=7;id++) {
        CircularBuffer cb(id); cb.reserve_back(1);
        volatile tt_l1_ptr uint32_t* p=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
        for(uint32_t i=0;i<1024;i++) p[i]=id==7?0x3f800000u:0u;
        cb.push_back(1);
    }
    auto faces=[&](const auto& a,uint32_t page0,uint32_t row,uint32_t element_bytes,uint32_t dst) {
        for(uint32_t tile=0;tile<4;tile++) for(uint32_t face=0;face<2;face++) {
            const uint32_t src=((row/16)*512+face*256+(row%16)*16)*element_bytes;
            // Blackhole DRAM reads require matching source/destination mod64.
            // Each face has its own64-byte slot, including BF16's32-byte row.
            // BF16's second row is harmless overfetch and is never consumed.
            noc.async_read(a,scratch,64,
                {.page_id=page0+tile,.offset_bytes=src},
                {.offset_bytes=dst+(tile*2+face)*64});
        }
    };
    auto gate=[&](const auto& a,uint32_t user,uint32_t head,uint32_t element_bytes,uint32_t dst) {
        const uint32_t page=user+head/32;
        const uint32_t offset=((head%32)/16)*256*element_bytes;
        noc.async_read(a,scratch,16*element_bytes,{.page_id=page,.offset_bytes=offset},{.offset_bytes=dst});
    };
    for(uint32_t h=start;h<start+count;h++) {
        const uint32_t user=h/12,hv=h%12,hk=hv/3;
        if(h!=start) for(uint32_t id=0;id<5;id++) {
            CircularBuffer cb(id); cb.reserve_back(id<3?4:1);
        }
        CircularBuffer state(5); state.reserve_back(16);
        for(uint32_t i=0;i<16;i++)
            noc.async_read(s,state,STATE_BYTES,{.page_id=h*16+i},{.offset_bytes=i*STATE_BYTES});
        if constexpr(ROW_MAJOR) {
            // One interleaved DRAM page is one complete BF16 channel row.
            // Every head spans256 contiguous bytes, naturally64-byte aligned.
            noc.async_read(q,scratch,256,{.page_id=user,.offset_bytes=hk*256},{.offset_bytes=0});
            noc.async_read(k,scratch,256,{.page_id=user,.offset_bytes=(PACKED?1024:0)+hk*256},{.offset_bytes=512});
            noc.async_read(v,scratch,256,{.page_id=user,.offset_bytes=(PACKED?2048:0)+hv*256},{.offset_bytes=1024});
        } else {
            faces(q,NORMALIZED?user*4:user*Q_STRIDE+hk*4,NORMALIZED?hk:0,NORMALIZED?4:2,0);
            faces(k,NORMALIZED?user*4:user*Q_STRIDE+hk*4,NORMALIZED?hk:0,NORMALIZED?4:2,512);
            faces(v,user*V_STRIDE+hv*4,0,2,1024);
        }
        gate(g,user,hv,G_FP32?4:2,1536); gate(b,user,hv,B_FP32?4:2,1600);
        noc.async_read_barrier();
        // Integer-only BF16 widening preserves all bits; no RISC floating math.
        for(uint32_t id=0;id<3;id++) {
            CircularBuffer cb(id);
            volatile tt_l1_ptr uint32_t* dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
            const uint32_t offset=id==0?0:(id==1?512:1024);
            volatile tt_l1_ptr uint16_t* src16=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch_addr+offset);
            volatile tt_l1_ptr uint32_t* src32=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_addr+offset);
            for(uint32_t i=0;i<128;i++) {
                const uint32_t tile=i/32,col=i%32;
                const uint32_t tiled=tile*1024+(col/16)*256+col%16;
                dst[tiled]=(id<2 && NORMALIZED)?src32[i]:(uint32_t(src16[ROW_MAJOR?i:(i/16)*32+i%16])<<16);
            }
            cb.push_back(4);
        }
        for(uint32_t id=3;id<5;id++) {
            const uint32_t offset=id==3?1536:1600, fp32=id==3?G_FP32:B_FP32;
            CircularBuffer cb(id);
            volatile tt_l1_ptr uint32_t* dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
            volatile tt_l1_ptr uint16_t* src16=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch_addr+offset);
            volatile tt_l1_ptr uint32_t* src32=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_addr+offset);
            dst[0]=fp32?src32[hv%16]:(uint32_t(src16[hv%16])<<16);
            cb.push_back(1);
        }
        state.push_back(16);
    }
}
"""


def _local12_head_partition(batch, available_cores, cores=None, grouped_qk=True):
    """Pure host partition: (grouped, [(global_head_start, head_count), ...])."""
    assert batch >= 1 and available_cores >= 1
    assert cores is None or cores >= 1
    grouped = bool(grouped_qk and batch >= 3)
    # Balance individual heads; compute still caches within global triples.
    unit = 1
    units = batch * 12 // unit
    ncores = min(units, available_cores, cores or available_cores)
    result = []
    start = 0
    for index in range(ncores):
        count = unit * (units // ncores + (index < units % ncores))
        result.append((start, count))
        start += count
    assert start == batch * 12
    return grouped, result


def _local12_recurrent(
    q,
    k,
    v,
    g,
    beta,
    state,
    *,
    normalized=False,
    packed_qkv=False,
    output=None,
    cores=None,
    grouped_qk=True,
    variant="baseline",
    source_overrides=None,
):
    """Accept raw RM[B,1,C] directly, or the original TILE interface."""
    row_major = q.layout == ttnn.ROW_MAJOR_LAYOUT
    assert not (row_major and normalized)
    assert not packed_qkv or row_major
    assert state.dtype in (ttnn.float32, ttnn.bfloat16)
    state_bf16 = state.dtype == ttnn.bfloat16
    state_bytes = 2048 if state_bf16 else 4096
    batch = state.shape[0] // 12
    assert tuple(state.shape) == (batch * 12, 128, 128)
    assert q.dtype == k.dtype == (ttnn.float32 if normalized else ttnn.bfloat16)
    assert v.dtype == ttnn.bfloat16
    assert tuple(q.shape) == tuple(k.shape)
    if packed_qkv:
        assert tuple(q.shape) == tuple(k.shape) == tuple(v.shape) == (batch, 1, 2560)
        assert q.buffer_address() == k.buffer_address() == v.buffer_address()
    elif normalized:
        assert tuple(q.shape) == (batch, 4, 128)
    else:
        assert q.shape[0] == batch and q.shape[-1] == 512 and q.shape[1] in (1, 32)
    if not packed_qkv:
        assert v.shape[0] == batch and v.shape[-1] == 1536 and v.shape[1] in (1, 32)
    assert tuple(g.shape) == tuple(beta.shape) == (batch, 1, 12)
    assert tuple(g.padded_shape) == tuple(beta.padded_shape) == (batch, 32, 32)
    inputs = [q, k, v, g, beta, state]
    for tensor in inputs:
        assert tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    for tensor in (q, k, v):
        assert tensor.layout == (ttnn.ROW_MAJOR_LAYOUT if row_major else ttnn.TILE_LAYOUT)
        if row_major:
            assert tensor.shape[1] == 1
            assert tuple(tensor.padded_shape) == tuple(tensor.shape)
    for tensor in (g, beta, state):
        assert tensor.layout == ttnn.TILE_LAYOUT
    if output is None:
        output = ttnn.empty(
            (1, batch * 12, 128),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=state.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    assert tuple(output.shape) == (1, batch * 12, 128)
    grid = state.device().compute_with_storage_grid_size()
    group_qk, partition = _local12_head_partition(batch, grid.x * grid.y, cores, grouped_qk)
    if source_overrides is not None or variant != "baseline":
        assert packed_qkv and not normalized
    if source_overrides is None:
        from .recurrent_variants import sources

        reader_source, compute_source = sources(
            _LOCAL12_READER, _fused_delta_row_GDN_COMPACT_COMPUTE, variant, grouped=group_qk
        )
    else:
        reader_source, compute_source = source_overrides
    ncores = len(partition)
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncores)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    for c, (start, count) in zip(coords, partition):
        rr[c.x][c.y] = [t.buffer_address() for t in inputs] + [start, count]
        wr[c.x][c.y] = [state.buffer_address(), output.buffer_address(), start, count]
        cr[c.x][c.y] = [count, start]
    # Raw q/k/v paddedS32 occupies one tile row per batch.
    rct = [
        int(normalized),
        int(g.dtype == ttnn.float32),
        int(beta.dtype == ttnn.float32),
        16,
        48,
        int(row_major),
        int(packed_qkv),
        state_bytes,
    ]
    for t in inputs:
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct = [state_bytes]
    for t in (state, output):
        wct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct.append(batch * 12)

    def kernel(source, config, rt, ct):
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )

    kernels = [
        kernel(reader_source, ttnn.ReaderConfigDescriptor(), rr, rct),
        kernel(_fused_delta_row_GDN_COMPACT_WRITER, ttnn.WriterConfigDescriptor(), wr, wct),
        kernel(
            compute_source,
            _fused_delta_row_compact_fp32_compute_config(state_bf16),
            cr,
            [
                int(normalized),
                int(state_bf16),
                int(group_qk),
                int(_fused_delta_row_GDN_FUSE_ACCUMULATOR),
                int(_fused_delta_row_GDN_HOIST_DELTA),
                int(_fused_delta_row_GDN_HOIST_SCALAR_INIT),
                int(_fused_delta_row_GDN_HOIST_COL_INIT),
                int(_fused_delta_row_GDN_FUSE_DECAY_PREDICT),
            ],
        ),
    ]
    sizes = {
        0: 8,
        1: 8,
        2: 8,
        3: 2,
        4: 2,
        5: 32,
        6: 1,
        7: 1,
        8: 1,
        9: 1,
        10: 1,
        11: 32,
        12: 8,
        13: 1,
        14: 1,
        15: 1,
        16: 1,
        17: 4,
        18: 1,
    }
    cbs = []
    for i, n in sizes.items():
        is_state = state_bf16 and i in (5, 11)
        page_size = 2048 if is_state else 4096
        data_format = ttnn.bfloat16 if is_state else ttnn.float32
        cbs.append(
            ttnn.CBDescriptor(
                total_size=n * page_size,
                core_ranges=ranges,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=data_format, page_size=page_size)
                ],
            )
        )
    ttnn.generic_op(inputs + [output], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return output


_LOCAL_QKV_STAGE = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
void kernel_main() {
    constexpr uint32_t PHASE=get_compile_time_arg_val(0);
    constexpr auto xa=TensorAccessorArgs<1>();
    const auto input=TensorAccessor(xa,get_arg_val<uint32_t>(0),2048);
    const uint32_t qb=get_arg_val<uint32_t>(1),kb=get_arg_val<uint32_t>(2),vb=get_arg_val<uint32_t>(3);
    const uint32_t user=get_arg_val<uint32_t>(4);
    Noc noc;
    CircularBuffer scratch(PHASE);
    const uint32_t raw=scratch.get_write_ptr();
    const uint32_t base=(raw+63u)&~63u;

    // Each RISC owns one horizontal face. Initialize the complete face before copying live rows;
    // the other RISC never writes these byte ranges. Complete zero mode before
    // issuing ordinary NoC operations or local scatter copies.
    const uint32_t buffers[3]={qb,kb,vb};
    const uint32_t heads[3]={6,1,1};
    for(uint32_t which=0;which<3;which++) for(uint32_t tile=0;tile<8;tile++) {
        for(uint32_t half=0;half<2;half++) {
            const uint32_t first=half*16;
            const uint32_t end=(half+1)*16;
            if(first<end) {
                const uint32_t dst=buffers[which]+tile*2048+half*1024+(first%16)*32+PHASE*512;
                noc.async_write_zeros(CoreLocalMem<uint32_t>(dst),(end-first)*32);
            }
        }
    }
    noc.write_zeros_l1_barrier();

    const uint32_t src_row=(user/16)*1024+(user%16)*32+PHASE*512;
    const uint32_t skew=src_row&63u,aligned_row=src_row-skew;
    // 6 Q + 1 K + 1 V heads, eight channel tiles/head. Every64-byte
    // destination slot is unique, so no per-head scratch-reuse barrier is needed.
    for(uint32_t tile=0;tile<64;tile++) {
        noc.async_read(input,CoreLocalMem<uint32_t>(base+tile*64),64,
                       {.page_id=tile,.offset_bytes=aligned_row},{});
    }
    noc.async_read_barrier();
    for(uint32_t tile=0;tile<64;tile++) {
        const uint32_t global_head=tile/8,channel_tile=tile%8;
        const uint32_t head=global_head<6?global_head:(global_head<7?global_head-6:global_head-7);
        const uint32_t out_base=global_head<6?qb:(global_head<7?kb:vb);
        const uint32_t dst=out_base+channel_tile*2048+(head/16)*1024+(head%16)*32+PHASE*512;
        tt::data_movement::common::tt_memmove<false,true,true,32>(
            noc,dst,base+tile*64+skew,32);
    }
    noc.async_read_barrier();
}
"""


def _local_qkv_split(input_tensor, *, num_heads=6, num_kv_heads=1, memory_config):
    assert num_heads == 6 and num_kv_heads == 1
    assert input_tensor.dtype == ttnn.bfloat16 and input_tensor.layout == ttnn.TILE_LAYOUT
    assert input_tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    batch = input_tensor.shape[2]
    assert tuple(input_tensor.shape) == (1, 1, batch, 2048) and 1 <= batch <= 32
    coords = [ttnn.CoreCoord(user % 8, user // 8) for user in range(batch)]
    core_ranges = []
    rows, tail = divmod(batch, 8)
    if rows:
        core_ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, rows - 1)))
    if tail:
        core_ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, rows), ttnn.CoreCoord(tail - 1, rows)))
    ranges = ttnn.CoreRangeSet(core_ranges)
    expected = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ranges, [32, 256], ttnn.ShardOrientation.ROW_MAJOR),
    )
    assert memory_config == expected, "QKV owner map must be first B cores in eight-column row-major order"
    outputs = [
        ttnn.empty(
            (1, batch, heads, 256),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=input_tensor.device(),
            memory_config=memory_config,
        )
        for heads in (6, 1, 1)
    ]
    runtime = ttnn.RuntimeArgs()
    for user, core in enumerate(coords):
        runtime[core.x][core.y] = [input_tensor.buffer_address(), *[t.buffer_address() for t in outputs], user]
    kernels = []
    for phase, config in enumerate((ttnn.ReaderConfigDescriptor(), ttnn.WriterConfigDescriptor())):
        compile_args = [phase, *ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=_LOCAL_QKV_STAGE,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=ranges,
                compile_time_args=compile_args,
                runtime_args=runtime,
                config=config,
            )
        )
    # Extra64 bytes permit explicit64-byte base alignment even if allocator's
    # CB alignment is only16 bytes. Two RISCs have entirely disjoint scratch.
    cbs = [
        ttnn.CBDescriptor(
            total_size=65 * 64,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=phase, data_format=ttnn.bfloat16, page_size=64)],
        )
        for phase in (0, 1)
    ]
    ttnn.generic_op([input_tensor, *outputs], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
    return tuple(outputs)


def _specialize(source, expected):
    """Replace whole integer tokens and fail closed if upstream kernel changes."""
    substitutions = {516: 130, 320: 80, 512: 128, 192: 48}
    for old, count in expected.items():
        source, found = re.subn(r"\b" + str(old) + r"\b", str(substitutions[old]), source)
        assert found == count, (old, found, count)
    for old in substitutions:
        assert re.search(r"\b" + str(old) + r"\b", source) is None, old
    return source


# Output columns: QKV tiles[0,80), Z[80,128), BA[128,130).
# Narrow residual columns: QKV[0,80), BA[80,82). Runtime pn/rn retain
# the true padded strides (normally144/96 with the selected N512 sharding).
_LOCAL_NARROW_READER = _specialize(_NARROW_GDN_COMBINE_READER, {516: 2, 320: 2, 512: 1, 192: 1})
_LOCAL_NARROW_COMPUTE = _specialize(_NARROW_GDN_COMBINE_COMPUTE, {516: 1, 320: 1, 512: 1})


def local_narrow_gdn_combine(primary, residual, *, output=None, cores=None):
    assert len(primary.shape) == len(residual.shape) == 3
    batch, seq, pwidth = tuple(primary.shape)
    rb, rs, rwidth = tuple(residual.shape)
    assert (batch, seq) == (rb, rs) and pwidth >= 4160 and rwidth >= 2624
    assert tuple(primary.padded_shape)[:2] == tuple(residual.padded_shape)[:2]
    for t in (primary, residual):
        assert t.dtype == ttnn.float32 and t.layout == ttnn.TILE_LAYOUT
        mem = t.memory_config()
        assert mem == ttnn.DRAM_MEMORY_CONFIG or (
            mem.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED and mem.buffer_type == ttnn.BufferType.L1
        )
    if output is None:
        output = ttnn.empty(
            (batch, seq, 4160),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=primary.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    assert (
        tuple(output.shape) == (batch, seq, 4160) and tuple(output.padded_shape)[:2] == tuple(primary.padded_shape)[:2]
    )
    assert (
        output.dtype == ttnn.bfloat16
        and output.layout == ttnn.TILE_LAYOUT
        and output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    )
    assert output.buffer_address() not in (primary.buffer_address(), residual.buffer_address())
    total = batch * (primary.padded_shape[-2] // 32) * 130
    grid = primary.device().compute_with_storage_grid_size()
    workers = min(total, cores or grid.x * grid.y)
    assert 1 <= workers <= grid.x * grid.y
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    start = 0
    for i, c in enumerate(coords):
        count = total // workers + (i < total % workers)
        rr[c.x][c.y] = [
            primary.buffer_address(),
            residual.buffer_address(),
            primary.padded_shape[-1] // 32,
            residual.padded_shape[-1] // 32,
            start,
            count,
        ]
        wr[c.x][c.y] = [output.buffer_address(), start, count]
        cr[c.x][c.y] = [start, count]
        start += count
    rct = []
    for t in (primary, residual):
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    def kernel(source, config, args, ct):
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=args,
            config=config,
        )

    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for i in (0, 1, 3):
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    kernels = [
        kernel(_LOCAL_NARROW_READER, ttnn.ReaderConfigDescriptor(), rr, rct),
        kernel(
            _NARROW_GDN_COMBINE_WRITER,
            ttnn.WriterConfigDescriptor(),
            wr,
            ttnn.TensorAccessorArgs(output).get_compile_time_args(),
        ),
        kernel(_LOCAL_NARROW_COMPUTE, config, cr, []),
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=size,
            core_ranges=ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=i,
                    data_format=ttnn.bfloat16 if i == 2 else ttnn.float32,
                    page_size=2048 if i == 2 else 4096,
                )
            ],
        )
        for i, size in ((0, 8192), (1, 8192), (2, 4096), (3, 4096))
    ]
    ttnn.generic_op([primary, residual, output], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return output
