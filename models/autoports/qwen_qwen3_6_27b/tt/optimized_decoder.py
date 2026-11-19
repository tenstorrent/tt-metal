# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Optimized single-device Qwen3.6 text decoder.

Weights and request buffers are prepared outside forward. Activations use
[batch, sequence, 5120]; each request owns its recurrent state or paged KV.
Text positions use the same position on all three MRoPE axes. Vision encoding
is outside this decoder's contract.
"""

import math
import struct
from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule

# Measured defaults; explicit overrides support isolated precision/config probes.
DEFAULT_POLICIES = {
    "linear_attention": {
        "prefill_projection_configs": {"packed_gdn_residual": {"in0_block_w": 10, "subblock_w": 3}},
        "gdn_fused_affine_gate": True,
        "gdn_affine_gate_cores": 64,
        "dram_sharded": True,
        "prefill_grid": [11, 10],
        "readers": 2,
        "pipeline": True,
        "block": 2,
        "fold_mlp": True,
        "batched_conv": True,
        "compact_gdn": True,
        "compact_residual": True,
        "shard_hidden_norm": True,
        "sfpu_compact": True,
        "sfpu_normalized": False,
        "minimal_swiglu": "prefill",
        "swiglu_blocks": [2, 8, 16],
        "weight_correction": ["packed_gdn"],
        "attention_fidelity": "HiFi4",
        "conv_functional": False,
        "conv_round_output": True,
        "hf_gated_norm": True,
        "round_silu": True,
        "explicit_fp32_norm": True,
        "unfused_g": True,
        "geometry": {
            "packed_gdn": {"cores": 40, "readers": 5, "block": 1},
            "packed_gdn_residual": {"cores": 40, "readers": 5, "block": 1},
            "linear_attn.out_proj.weight": {"cores": 32, "readers": 4, "block": 1},
        },
        "conv_rm": True,
        "correction_l1": True,
        "conv_packed_rm": True,
        "accurate_sharded_norm": False,
        "round_recurrent_state": False,
        "bf16_recurrent_cache": True,
        "sfpu_hidden_norm": True,
        "explicit_sum_norm": True,
        "packer_l1_acc": False,
        "flat_prefill_correction": True,
        "flat_prefill_mlp": True,
        "prefill_k": 8,
        "sfpu_decode_conv": True,
        "exact_tree_hidden_norm": True,
        "mlp_packer_l1_acc": True,
        "share_mlp_input": True,
        "grouped_recurrent_init": True,
        "decode_weight_correction": ["packed_gdn"],
        "chunk_add_prefill_correction": True,
        "prefill_subblock_h": True,
        "conv_untilize_pipeline": True,
        "decay_predict_fusion": True,
        "matmul_row_cap": 640,
        "zero_z_correction": True,
        "narrow_gdn_correction": True,
        "narrow_gdn_verify_setup": False,
        "narrow_gdn_fused_combine": True,
        "narrow_gdn_combine_cores": None,
        "prefill_unflatten_packed": True,
        "prefill_unflatten_cores": None,
        "fused_beta_decay": True,
        "attention_dtype": "bfloat8_b",
        "mlp_dtype": "bfloat8_b",
        "mlp_fidelity": "HiFi2",
        "attention_packer_l1_acc": False,
        "prefill_chunk": 512,
    },
    "full_attention": {
        "dram_sharded": True,
        "prefill_grid": [11, 10],
        "readers": 2,
        "pipeline": True,
        "block": 2,
        "fold_mlp": True,
        "compact_attention": True,
        "compact_residual": True,
        "shard_hidden_norm": True,
        "minimal_swiglu": "prefill",
        "swiglu_blocks": [2, 8, 16],
        "kv_dtype": "bfloat8_b",
        "sdpa_grid": [8, 8],
        "full_pack_all": "prefill",
        "flat_prefill_mlp": True,
        "attention_fidelity": "LoFi",
        "mlp_fidelity": "LoFi",
        "share_mlp_input": True,
        "single_user_classic_attention": True,
        "single_user_sdpa_grid": [8, 4],
        "norm_projection_l1": True,
        "short_capacity_decode_k_chunk": 64,
        "attention_dtype": "bfloat8_b",
        "mlp_dtype": "bfloat8_b",
        "attention_packer_l1_acc": True,
        "mlp_packer_l1_acc": True,
        "prefill_chunk": 512,
    },
}


@dataclass
class RecurrentState:
    conv: ttnn.Tensor
    recurrent: ttnn.Tensor


@dataclass
class PagedState:
    key: ttnn.Tensor
    value: ttnn.Tensor
    page_table: ttnn.Tensor


class OptimizedDecoder(LightweightModule):
    """Real HF layer weights; both linear_attention and full_attention kinds.

    ``prefill_forward(x, *, state, ...)`` accepts device activations and mutates
    request state. ``decode_forward`` consumes one token per row. The caller
    owns state restoration and stable inputs when capturing decode.
    """

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, policy=None):
        self = cls()
        self.config = getattr(hf_config, "text_config", hf_config)
        c = self.config
        assert (c.hidden_size, c.intermediate_size) == (5120, 17408)
        assert mesh_device.get_num_devices() == 1
        self.mesh = mesh_device
        self.kind = c.layer_types[layer_idx]
        self.ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.policy = {**DEFAULT_POLICIES[self.kind], **(policy or {})}
        assert self.policy.get("full_pack_all", False) in (
            False,
            True,
            "prefill",
        ), "full_pack_all must be False, True, or 'prefill'"
        self.matmul_configs = {
            group: ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=getattr(ttnn.MathFidelity, self.policy.get(group + "_fidelity", "HiFi2")),
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=self.policy.get(group + "_packer_l1_acc", self.policy.get("packer_l1_acc", True)),
            )
            for group in ("attention", "mlp")
        }
        self.weights = {}
        folded_norms = (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
        )
        for name, value in state_dict.items():
            # The loading boundary accepts a layer-local HF state dict.
            value = value.detach().float()
            if name.endswith(".weight") and value.ndim == 2:
                value = value.T.contiguous()
            elif name in folded_norms:
                value = 1.0 + value
            if value.ndim == 1:
                value = value.reshape(1, 1, -1)
            if "conv1d" not in name:
                self.weights[name] = self._upload(
                    value,
                    dtype=(
                        ttnn.float32
                        if name in folded_norms or name.endswith(("A_log", "dt_bias"))
                        else self._weight_dtype(name)
                        if value.ndim == 2
                        else ttnn.bfloat16
                    ),
                )
        if self.kind == "linear_attention":
            import torch

            assert (
                c.linear_num_key_heads,
                c.linear_num_value_heads,
                c.linear_key_head_dim,
                c.linear_value_head_dim,
                c.linear_conv_kernel_dim,
            ) == (16, 48, 128, 128, 4)
            packed = torch.cat(
                [
                    state_dict["linear_attn.in_proj_qkv.weight"],
                    state_dict["linear_attn.in_proj_z.weight"],
                    torch.nn.functional.pad(state_dict["linear_attn.in_proj_b.weight"], (0, 0, 0, 16)),
                    torch.nn.functional.pad(state_dict["linear_attn.in_proj_a.weight"], (0, 0, 0, 16)),
                ],
                dim=0,
            )
            self.weights["packed_gdn"] = self._upload(packed.T.contiguous(), dtype=self._weight_dtype("packed_gdn"))
            self.conv_start = ttnn.from_torch(
                torch.tensor([0], dtype=torch.int64), device=self.mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            exchange_core = ttnn._ttnn.operations.experimental.kda.qkv_causal_conv1d_silu_exchange_core(
                self.mesh.compute_with_storage_grid_size()
            )
            exchange_grid = ttnn.CoreRangeSet({ttnn.CoreRange(exchange_core, exchange_core)})
            self.exchange = ttnn.from_torch(
                torch.zeros(1, 12, 10240), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh
            )
            self.conv_semaphores = [ttnn.create_global_semaphore(self.mesh, exchange_grid, 0) for _ in range(3)]
            self.conv_ends = {
                n: ttnn.from_torch(
                    torch.tensor([n], dtype=torch.int64),
                    device=self.mesh,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                for n in range(1, max(512, self.policy.get("prefill_chunk", 512)) + 1)
            }
            self.eye = self._upload(torch.eye(32).reshape(1, 1, 32, 32), dtype=ttnn.float32)
            self.lower = self._upload(torch.ones(32, 32).tril().reshape(1, 1, 32, 32), dtype=ttnn.float32)
            self.ones = self._upload(torch.ones(1, 1, 32, 32), dtype=ttnn.float32)
            ii, jj = torch.arange(32).unsqueeze(1), torch.arange(32).unsqueeze(0)
            lo_i, lo_j = ii < 16, jj < 16
            masks = torch.cat([(lo_i & lo_j).float(), (~lo_i & ~lo_j).float(), (~lo_i & lo_j).float()], dim=1)
            self.masks = self._upload(masks.reshape(1, 1, 32, 96), dtype=ttnn.float32)
            conv = state_dict["linear_attn.conv1d.weight"].float()
            self.taps = [self._upload(conv[:, 0, k].reshape(1, 1, -1)) for k in range(4)]
            self.channel_taps = {}
            if self.policy.get("channel_conv", False):
                for batch in [self.policy.get("channel_batch", 32)]:
                    self.channel_taps[batch] = [
                        self._upload(
                            torch.cat(
                                [conv[a:z, 0, t].repeat(batch) for a, z in ((0, 2048), (2048, 4096), (4096, 10240))]
                            ).reshape(1, 1, -1)
                        )
                        for t in range(4)
                    ]
            self.taps_rm = (
                [ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT) for t in self.taps]
                if self.policy.get("batched_conv", False)
                else None
            )
            self.neg_a = self._upload(
                -state_dict["linear_attn.A_log"].float().exp().reshape(1, 1, -1), dtype=ttnn.float32
            )
        else:
            assert (c.num_attention_heads, c.num_key_value_heads, c.head_dim) == (24, 4, 256)
            # HF interleaves Q and output gate per head in q_proj.
            w = state_dict["self_attn.q_proj.weight"].reshape(24, 512, 5120)
            self.weights["query"] = self._upload(
                w[:, :256].reshape(6144, 5120).T.contiguous(), dtype=self._weight_dtype("query")
            )
            self.weights["gate"] = self._upload(
                w[:, 256:].reshape(6144, 5120).T.contiguous(), dtype=self._weight_dtype("gate")
            )
        if self.kind == "full_attention":
            import torch

            qw = state_dict["self_attn.q_proj.weight"].reshape(24, 512, 5120)[:, :256].reshape(6144, 5120)
            self.weights["packed_qkv"] = self._upload(
                torch.cat(
                    [qw, state_dict["self_attn.k_proj.weight"], state_dict["self_attn.v_proj.weight"]], dim=0
                ).T.contiguous(),
                dtype=self._weight_dtype("packed_qkv"),
            )
        obsolete = (
            [
                "linear_attn.in_proj_qkv.weight",
                "linear_attn.in_proj_z.weight",
                "linear_attn.in_proj_a.weight",
                "linear_attn.in_proj_b.weight",
            ]
            if self.kind == "linear_attention"
            else ["query", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"]
        )
        if self.policy.get("split_attention", False):
            obsolete = ["packed_gdn"] if self.kind == "linear_attention" else ["packed_qkv", "self_attn.q_proj.weight"]
        if self.policy.get("packed_mlp", False):
            import torch

            self.weights["mlp.gate_up"] = self._upload(
                torch.cat([state_dict["mlp.gate_proj.weight"], state_dict["mlp.up_proj.weight"]], dim=0).T.contiguous(),
                dtype=self._weight_dtype("mlp.gate_proj.weight"),
            )
            obsolete += ["mlp.gate_proj.weight", "mlp.up_proj.weight"]
        if self.kind == "full_attention" and self.policy.get("full_pack_all", False):
            self.weights["packed_qkvg"] = self._upload(
                torch.cat(
                    [
                        qw,
                        state_dict["self_attn.k_proj.weight"],
                        state_dict["self_attn.v_proj.weight"],
                        w[:, 256:].reshape(6144, 5120),
                    ],
                    dim=0,
                ).T.contiguous(),
                dtype=self._weight_dtype("packed_qkv"),
            )
            # Prefill-only mode retains the tuned separate decode projections.
            if self.policy.get("full_pack_all") is True:
                obsolete += ["packed_qkv", "gate"]
        self.swiglu_weight = None
        if self.policy.get("minimal_swiglu", False):
            import torch

            gate = state_dict["mlp.gate_proj.weight"].T.contiguous().reshape(5120, 544, 32)
            up = state_dict["mlp.up_proj.weight"].T.contiguous().reshape(5120, 544, 32)
            paired = torch.stack((gate, up), dim=2).reshape(5120, 34816)
            self.swiglu_weight = self._upload(paired, dtype=self._weight_dtype("mlp.gate_proj.weight"))
        for name in obsolete:
            self.weights.pop(name, None)
        self.gdn_program = (
            ttnn.ChunkGdnFusedProgramConfig(**self.policy["gdn_program"]) if self.policy.get("gdn_program") else None
        )
        self.residual_memory = ttnn.create_sharded_memory_config(
            (32, 640), ttnn.CoreGrid(x=8, y=1), ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
        )
        self.weight_corrections = {}
        correction_keys = self.policy.get("weight_correction", [])
        if correction_keys:
            for key in correction_keys:
                original = (
                    packed.T.contiguous().float() if key == "packed_gdn" else state_dict[key].T.contiguous().float()
                )
                quantized = ttnn.to_torch(self.weights[key]).float()
                residual = original - quantized
                if key == "packed_gdn" and self.policy.get("zero_z_correction", True):
                    # Numerical veto experiment only: retain full physical N.
                    # Primary BFP8 upload/readback above is entirely unchanged.
                    # Whole N tiles are zeroed before correction BFP8 packing;
                    # no exponent group straddles either aligned boundary.
                    assert tuple(residual.shape) == (5120, 16512)
                    residual[:, 10240:16384] = 0
                correction = key + "_residual"
                self.weights[correction] = self._upload(residual, dtype=ttnn.bfloat8_b)
                self.weight_corrections[key] = correction
        self.narrow_gdn_correction = self.policy.get("narrow_gdn_correction", False)
        if self.narrow_gdn_correction:
            assert self.kind == "linear_attention" and self.policy.get("zero_z_correction", True)
            correction_key = self.weight_corrections["packed_gdn"]
            full = self.weights[correction_key]
            assert tuple(full.shape) == (5120, 16512) and full.dtype == ttnn.bfloat8_b
            assert full.layout == ttnn.TILE_LAYOUT and full.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            qkv = ttnn.slice(full, (0, 0), (5120, 10240), (1, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ba = ttnn.slice(full, (0, 16384), (5120, 16512), (1, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            narrow = ttnn.concat([qkv, ba], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if self.policy.get("narrow_gdn_verify_setup", False):
                import torch

                host = ttnn.to_torch(full)
                expected = torch.cat([host[:, :10240], host[:, 16384:16512]], dim=-1)
                assert torch.equal(ttnn.to_torch(narrow), expected)
                assert torch.count_nonzero(host[:, 10240:16384]).item() == 0
                print("NARROW_GDN_SETUP_VERIFY exact", tuple(narrow.shape), flush=True)
            self.weights[correction_key] = narrow
        self.projection_info = {}
        if self.policy.get("dram_sharded", False):
            dram = mesh_device.dram_grid_size()
            banks = dram.x * dram.y
            dram_grid = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))}
            )
            for key, weight in list(self.weights.items()):
                if len(weight.shape) != 2:
                    continue
                k, n = weight.shape
                group = "mlp" if key.startswith("mlp.") else "attention"
                role = "down" if "down_proj" in key else "out" if "out_proj" in key or "o_proj" in key else "in"
                options = self.policy.get("geometry", {}).get(key, {})
                cores = options.get("cores", self.policy.get("cores", 8))
                readers = options.get("readers", self.policy.get("readers", 1))
                block = options.get("block", self.policy.get("block", 4))
                alignment = math.lcm(banks * readers, cores)
                shard_n = math.ceil(n / 32 / alignment) * 32 * alignment // banks
                mem = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(dram_grid, (k, shard_n), ttnn.ShardOrientation.ROW_MAJOR),
                )
                physical_n = shard_n * banks
                if physical_n != n:
                    weight = ttnn.pad(weight, ((0, 0), (0, physical_n - n)), value=0.0)
                self.weights[key] = ttnn.to_memory_config(weight, mem)
                grid = ttnn.num_cores_to_corerangeset(
                    cores, mesh_device.compute_with_storage_grid_size(), row_wise=True
                )
                inp = ttnn.create_sharded_memory_config(
                    (32, k // cores), grid, ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
                )
                program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=block,
                    per_core_M=1,
                    per_core_N=physical_n // 32 // cores,
                    num_workers_per_dram_bank=readers,
                    use_multicore_pipeline=options.get("pipeline", self.policy.get("pipeline", False)),
                )
                self.projection_info[key] = dict(input_mem=inp, program=program, logical_n=n)
        return self

    def _weight_dtype(self, key):
        group = "mlp" if key.startswith("mlp.") else "attention"
        return getattr(
            ttnn, self.policy.get("weight_dtypes", {}).get(key, self.policy.get(group + "_dtype", "bfloat8_b"))
        )

    def _upload(self, value, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            value, device=self.mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def create_recurrent_state(self, batch=1):
        import torch

        return RecurrentState(
            ttnn.from_torch(
                torch.zeros(batch, 3, 10240), device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            ),
            self._upload(
                torch.zeros(batch, 48, 128, 128),
                dtype=ttnn.bfloat16 if self.policy.get("bf16_recurrent_cache", False) else ttnn.float32,
            ),
        )

    def _combine_correction(self, primary, correction, key):
        if not (self.narrow_gdn_correction and key == "packed_gdn"):
            return ttnn.add(primary, correction, dtype=ttnn.bfloat16)
        if self.policy.get("narrow_gdn_fused_combine", False):
            return narrow_gdn_combine(primary, correction, cores=self.policy.get("narrow_gdn_combine_cores"))
        # First legal movement control. Keep FP32 add -> BF16 for every region,
        # including primary Z + scalar0; do not substitute a standalone cast.
        primary = ttnn.to_memory_config(primary, ttnn.DRAM_MEMORY_CONFIG)
        correction = ttnn.to_memory_config(correction, ttnn.DRAM_MEMORY_CONFIG)
        assert primary.dtype == correction.dtype == ttnn.float32
        b, s, _ = tuple(primary.shape)
        assert tuple(correction.shape)[:2] == (b, s)

        def cut(t, start, stop):
            return ttnn.slice(t, (0, 0, start), (b, s, stop), (1, 1, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)

        qkv = ttnn.add(cut(primary, 0, 10240), cut(correction, 0, 10240), dtype=ttnn.bfloat16)
        z = ttnn.add(cut(primary, 10240, 16384), 0.0, dtype=ttnn.bfloat16)
        ba = ttnn.add(cut(primary, 16384, 16512), cut(correction, 10240, 10368), dtype=ttnn.bfloat16)
        return ttnn.concat([qkv, z, ba], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _linear(self, x, key):
        if key not in self.weight_corrections:
            return self._linear_raw(x, key)
        if self.policy.get("chunk_add_prefill_correction", False) and x.shape[0] * x.shape[1] > 32:
            return corrected_projection_chunk_add_prefill(self, x, key)
        if self.policy.get("flat_prefill_correction", False) and x.shape[0] > 1:
            return corrected_projection_flat_prefill(self, x, key)
        if (
            self.policy.get("correction_l1", False)
            and x.shape[0] == 1
            and x.shape[1] <= 32
            and key in self.projection_info
        ):
            correction_key = self.weight_corrections[key]
            info = self.projection_info[key]
            assert info["input_mem"] == self.projection_info[correction_key]["input_mem"]
            activation = ttnn.to_memory_config(x, info["input_mem"])
            partials = [
                ttnn.linear(
                    activation,
                    self.weights[k],
                    program_config=self.projection_info[k]["program"],
                    compute_kernel_config=self.matmul_configs["attention"],
                    memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1),
                    dtype=ttnn.float32,
                )
                for k in (key, correction_key)
            ]
            result = self._combine_correction(*partials, key)
            result = ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG)
            return result[:, :, : info["logical_n"]]
        primary = self._linear_raw(x, key)
        correction = self._linear_raw(x, self.weight_corrections[key])
        return self._combine_correction(primary, correction, key)

    def _precision_group(self, key):
        from .precision import group_for_weight

        return group_for_weight(key)

    def _linear_raw(self, x, key):
        output_dtype = (
            ttnn.float32
            if key in self.weight_corrections or key.endswith("_residual")
            else getattr(ttnn, self.policy.get("activation_dtype", "bfloat16"))
        )
        b, seq, width = x.shape
        if b > 1:
            x = ttnn.reshape(x, (1, b * seq, width))
        group = "mlp" if key.startswith("mlp.") else "attention"
        target_dtype = getattr(ttnn, self.policy.get(group + "_activation_dtype", "bfloat16"))
        if x.dtype != target_dtype:
            x = ttnn.typecast(x, target_dtype)
        weight = self.weights[key]
        cap = 256 if weight.shape[-1] > 20000 else self.policy.get("matmul_row_cap", 512)
        if key in self.projection_info and b * seq > cap:
            parts = [self._linear_raw(x[:, start : min(start + cap, b * seq)], key) for start in range(0, b * seq, cap)]
            y = ttnn.concat([ttnn.to_memory_config(part, ttnn.DRAM_MEMORY_CONFIG) for part in parts], dim=1)
            return ttnn.reshape(y, (b, seq, y.shape[-1])) if b > 1 else y
        if key in self.projection_info and b * seq <= 32:
            info = self.projection_info[key]
            x = ttnn.to_memory_config(x, info["input_mem"])
            y = ttnn.linear(
                x,
                weight,
                program_config=info["program"],
                compute_kernel_config=self.matmul_configs.get(self._precision_group(key), self.matmul_configs[group]),
                memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1),
                dtype=output_dtype,
            )
            keep_l1 = (
                self.policy.get("l1_mlp", False)
                and b == 1
                and (key.startswith("mlp.") or "out_proj" in key or "o_proj" in key)
            )
            if not keep_l1:
                y = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
        else:
            program = None
            prefill_configs = self.policy.get("prefill_projection_configs", {})
            prefill_options = prefill_configs.get(key, prefill_configs.get(key.removesuffix("_residual"), {}))
            if key in self.projection_info or self.policy.get("prefill_2d", False):
                assert set(prefill_options) <= {
                    "grid",
                    "in0_block_w",
                    "subblock_h",
                    "subblock_w",
                    "input_l1",
                    "out_block_h",
                }
                gx, gy = prefill_options.get("grid", self.policy.get("prefill_grid", (8, 8)))
                pm = math.ceil(b * seq / 32 / gy)
                pn = math.ceil(weight.shape[-1] / 32 / gx)
                sw = prefill_options.get("subblock_w", max(v for v in (1, 2, 4) if pn % v == 0))
                requested_bh = prefill_options.get(
                    "out_block_h", 1 if self.policy.get("prefill_height_block", False) else pm
                )
                assert requested_bh >= 1
                # Height overrides are ceilings, preserving arbitrary short tails.
                bh = max(v for v in range(1, pm + 1) if v <= requested_bh and pm % v == 0)
                requested_h = min(
                    bh, prefill_options.get("subblock_h", 4 if self.policy.get("prefill_subblock_h", False) else 1)
                )
                sh = max(v for v in (1, 2, 4) if v <= requested_h and bh % v == 0 and v * sw <= 4)
                kb = prefill_options.get(
                    "in0_block_w", 2 if weight.shape[-1] > 20000 else self.policy.get("prefill_k", 4)
                )
                assert kb > 0 and (width // 32) % kb == 0
                assert sh > 0 and sw > 0 and pm % sh == 0 and pn % sw == 0 and sh * sw <= 4
                program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(gx, gy),
                    in0_block_w=kb,
                    out_subblock_h=sh,
                    out_subblock_w=sw,
                    per_core_M=pm,
                    per_core_N=pn,
                    out_block_h=bh,
                    out_block_w=pn,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=True,
                )
            if self.policy.get("prefill_minimal", False):
                mb, kb, nb = self.policy.get("minimal_blocks", [4, 4, 8])
                y = ttnn.experimental.minimal_matmul(
                    x,
                    weight,
                    config=ttnn.MinimalMatmulConfig(
                        M_block_size=mb,
                        K_block_size=kb,
                        N_block_size=nb,
                        subblock_h=1,
                        subblock_w=4,
                        compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
                    ),
                    compute_kernel_config=self.matmul_configs.get(
                        self._precision_group(key), self.matmul_configs[group]
                    ),
                    dtype=output_dtype,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if key in self.projection_info and y.shape[-1] != self.projection_info[key]["logical_n"]:
                    y = y[..., : self.projection_info[key]["logical_n"]]
                return ttnn.reshape(y, (b, seq, y.shape[-1])) if b > 1 else y
            if prefill_options.get("input_l1", self.policy.get("prefill_input_l1", False)):
                x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            y = ttnn.linear(
                x,
                weight,
                program_config=program,
                compute_kernel_config=self.matmul_configs.get(self._precision_group(key), self.matmul_configs[group]),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=output_dtype,
            )
        if key in self.projection_info and y.shape[-1] != self.projection_info[key]["logical_n"]:
            y = y[..., : self.projection_info[key]["logical_n"]]
        return ttnn.reshape(y, (b, seq, y.shape[-1])) if b > 1 else y

    def _norm(self, x, key):
        if (
            self.policy.get("sfpu_hidden_norm", False)
            and tuple(x.shape)[0] == 1
            and x.shape[1] <= 32
            and x.shape[-1] == 5120
        ):
            norm_function = (
                _norm_zero_meanhidden_rmsnorm_sfpu
                if self.policy.get("exact_tree_hidden_norm", False)
                else hidden_rmsnorm_sfpu
            )
            return norm_function(
                ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG), self.weights[key], epsilon=self.config.rms_norm_eps
            )
        if self.policy.get("sfpu_prefill_norm", False) and x.shape[-1] == 5120:
            return prefill_hidden_rmsnorm_sfpu(
                ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG), self.weights[key], epsilon=self.config.rms_norm_eps
            )
        if self.policy.get("explicit_sum_norm", False) and x.shape[-1] == 5120:
            xf = ttnn.typecast(x, ttnn.float32)
            variance = ttnn.multiply(ttnn.sum(ttnn.multiply(xf, xf), dim=-1, keepdim=True), 1.0 / 5120)
            norm = ttnn.multiply(xf, ttnn.rsqrt(ttnn.add(variance, self.config.rms_norm_eps)))
            return ttnn.multiply(norm, self.weights[key], dtype=ttnn.bfloat16)
        if self.policy.get("explicit_fp32_norm", False) and not (
            self.policy.get("accurate_sharded_norm", False)
            and tuple(x.shape)[0] == 1
            and x.shape[1] <= 32
            and x.shape[-1] == 5120
        ):
            y = ttnn.rms_norm(
                ttnn.typecast(x, ttnn.float32),
                weight=self.weights[key],
                epsilon=self.config.rms_norm_eps,
                compute_kernel_config=self.ckc,
            )
            return ttnn.typecast(y, ttnn.bfloat16)
        if (
            self.policy.get("fold_norm", False)
            and len(x.shape) == 3
            and x.shape[0] > 1
            and x.shape[1] == 1
            and x.shape[-1] == 5120
        ):
            batch = x.shape[0]
            y = self._norm(ttnn.reshape(x, (1, batch, 5120)), key)
            return ttnn.reshape(ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG), (batch, 1, 5120))
        # Single-row RMSNorm shares width across eight cores; the two required
        # transfers also widen/narrow, replacing separate dtype conversions.
        if (
            len(x.shape) == 3
            and x.shape[0] == 1
            and (
                x.shape[1] == 1
                or (self.policy.get("l1_mlp", False) or self.policy.get("shard_hidden_norm", False))
                and x.shape[1] <= 32
            )
            and x.shape[-1] == 5120
        ):
            grid = ttnn.CoreGrid(x=8, y=1)
            memory = ttnn.create_sharded_memory_config(
                (32, 640),
                grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            y = (
                ttnn.typecast(x, ttnn.float32)
                if x.is_sharded()
                else ttnn.interleaved_to_sharded(x, memory, output_dtype=ttnn.float32)
            )
            y = ttnn.rms_norm(
                y,
                weight=self.weights[key],
                epsilon=self.config.rms_norm_eps,
                compute_kernel_config=self.ckc,
                program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=(8, 1),
                    subblock_w=4,
                    block_h=1,
                    block_w=20,
                    inplace=False,
                ),
            )
            # Keep the same FP32 RMSNorm and affine, then round once to BF16.
            # Only the normalization result stays in L1; residuals retain their
            # existing memory policy. Projection to_memory_config is an alias
            # when its exact input MemoryConfig already matches this output.
            if self.kind == "full_attention" and self.policy.get("norm_projection_l1", True):
                if key == "post_attention_layernorm.weight":
                    consumers = ("mlp.gate_proj.weight", "mlp.up_proj.weight")
                elif key == "input_layernorm.weight":
                    if self.policy.get("full_pack_all", False) is True:
                        consumers = ("packed_qkvg",)
                    elif self.policy.get("split_attention", False):
                        consumers = ("query", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "gate")
                    else:
                        consumers = ("packed_qkv", "gate")
                else:
                    consumers = ()
                compatible = consumers and all(
                    name in self.projection_info and self.projection_info[name]["input_mem"] == memory
                    for name in consumers
                )
                if compatible:
                    # Match sharded_to_interleaved's default copy conversion:
                    # 16-bit DST, source-register unpack, no SFPU RNE typecast.
                    return ttnn.clone(
                        y,
                        dtype=ttnn.bfloat16,
                        memory_config=memory,
                        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                            math_fidelity=ttnn.MathFidelity.HiFi4,
                            math_approx_mode=False,
                            fp32_dest_acc_en=False,
                            packer_l1_acc=False,
                        ),
                    )
            if self.policy.get("l1_mlp", False) and key == "post_attention_layernorm.weight":
                return ttnn.typecast(y, ttnn.bfloat16)
            return ttnn.sharded_to_interleaved(y, ttnn.DRAM_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)
        # Native BF16 intake keeps FP32 intermediates/affine under self.ckc,
        # so widening/narrowing is absorbed into the norm's unpack/pack.
        return ttnn.rms_norm(
            x,
            weight=self.weights[key],
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=self.ckc,
        )

    def _finish(self, x, attention, prefill=False):
        if self.policy.get("fold_mlp", False) and x.shape[0] > 1 and x.shape[1] == 1:
            batch = x.shape[0]
            out = self._finish(
                ttnn.reshape(x, (1, batch, 5120)),
                ttnn.reshape(ttnn.to_memory_config(attention, ttnn.DRAM_MEMORY_CONFIG), (1, batch, 5120)),
                prefill=prefill,
            )
            return ttnn.reshape(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG), (batch, 1, 5120))
        if self.policy.get("l1_mlp", False) and x.shape[0] == 1 and x.shape[1] <= 32:
            x = ttnn.to_memory_config(x, self.residual_memory)
            attention = ttnn.to_memory_config(attention, self.residual_memory)
        x = ttnn.add(x, attention)
        h = self._norm(x, "post_attention_layernorm.weight")
        if self.swiglu_weight is not None and (self.policy.get("minimal_swiglu") != "prefill" or prefill):
            if prefill and self.policy.get("flat_prefill_mlp", False):
                return minimal_mlp_flat_prefill(self, x, h)
            batch, seq, width = h.shape
            h = ttnn.reshape(ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG), (1, batch * seq, width))
            mb, kb, nb = self.policy.get("swiglu_blocks", [4, 4, 16])
            activated = ttnn.experimental.minimal_matmul(
                h,
                self.swiglu_weight,
                config=ttnn.MinimalMatmulConfig(
                    M_block_size=mb,
                    K_block_size=kb,
                    N_block_size=nb,
                    subblock_h=1,
                    subblock_w=4,
                    compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
                ),
                fuse_swiglu=True,
                compute_kernel_config=self.matmul_configs["mlp"],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            activated = ttnn.reshape(activated, (batch, seq, 17408))
            return ttnn.add(x, self._linear(activated, "mlp.down_proj.weight"))
        if self.policy.get("packed_mlp", False):
            combined = self._linear(h, "mlp.gate_up")
            gate, up = combined[..., :17408], combined[..., 17408:]
        else:
            gate_info = self.projection_info.get("mlp.gate_proj.weight")
            up_info = self.projection_info.get("mlp.up_proj.weight")
            share = (
                self.policy.get("share_mlp_input", False)
                and not prefill
                and h.shape[0] == 1
                and h.shape[1] <= 32
                and gate_info is not None
                and up_info is not None
                and gate_info["input_mem"] == up_info["input_mem"]
                and h.dtype == getattr(ttnn, self.policy.get("mlp_activation_dtype", "bfloat16"))
            )
            projected_input = ttnn.to_memory_config(h, gate_info["input_mem"]) if share else h
            gate = self._linear(projected_input, "mlp.gate_proj.weight")
            up = self._linear(projected_input, "mlp.up_proj.weight")
            del projected_input
        if (
            self.kind == "full_attention"
            and self.policy.get("norm_projection_l1", True)
            and h.is_sharded()
            and not self.policy.get("l1_mlp", False)
        ):
            # Gate/up have both consumed this private norm result. Release it
            # before the large static buffers of MLP down are allocated.
            ttnn.deallocate(h)
        activated = (
            ttnn.multiply(ttnn.silu(gate), up)
            if self.policy.get("round_silu", False)
            else ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        )
        if self.policy.get("l1_mlp", False):
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
            ttnn.deallocate(h)
        down = self._linear(activated, "mlp.down_proj.weight")
        return ttnn.add(x, down)

    def _decay(self, a):
        value = ttnn.add(a, self.weights["linear_attn.dt_bias"], dtype=ttnn.float32)
        if self.policy.get("unfused_g", False):
            return ttnn.multiply(self.neg_a, ttnn.softplus(value))
        return ttnn.multiply(
            self.neg_a, value, input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)]
        )

    def _gdn(self, x, state):
        if self.policy.get("compact_gdn", False) and x.shape[1] == 1:
            return self._gdn_compact(x, state)
        b, s, _ = x.shape
        if self.policy.get("split_attention", False):
            packed = ttnn.concat(
                [
                    self._linear(x, "linear_attn.in_proj_qkv.weight"),
                    self._linear(x, "linear_attn.in_proj_z.weight"),
                    ttnn.pad(self._linear(x, "linear_attn.in_proj_b.weight"), ((0, 0), (0, 0), (0, 16)), value=0.0),
                    ttnn.pad(self._linear(x, "linear_attn.in_proj_a.weight"), ((0, 0), (0, 0), (0, 16)), value=0.0),
                ],
                dim=-1,
            )
        else:
            packed = self._linear(x, "packed_gdn")
        qkv = packed[:, :, :10240]
        padded = (s + 31) // 32 * 32
        conv_input = qkv if s == padded else ttnn.pad(qkv, ((0, 0), (0, padded - s), (0, 0)), value=0.0)
        if self.policy.get("channel_conv", False) and b in self.channel_taps and s == 1:
            q, k, v = self._channel_conv(qkv, state.conv)
        elif self.policy.get("batched_conv", False) and b > 1 and s == 1:
            q, k, v = self._batched_conv(qkv, state.conv)
        else:
            outputs = []
            for row in range(b):
                history = state.conv[row : row + 1]
                outputs.append(
                    ttnn.experimental.kda.qkv_causal_conv1d_silu_with_carry(
                        conv_input[row : row + 1],
                        history,
                        *self.taps,
                        2048,
                        2048,
                        6144,
                        actual_start=self.conv_start,
                        actual_end=self.conv_ends[s],
                        exchange_buffer=self.exchange,
                        exchange_semaphores=self.conv_semaphores,
                        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=256),
                        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                            math_fidelity=ttnn.MathFidelity.HiFi4,
                            math_approx_mode=False,
                            fp32_dest_acc_en=True,
                            packer_l1_acc=False,
                        ),
                    )
                )
            q, k, v = (ttnn.concat([o[i] for o in outputs], dim=0) if b > 1 else outputs[0][i] for i in range(3))
            ttnn.copy(ttnn.concat([o[3] for o in outputs], dim=0) if b > 1 else outputs[0][3], state.conv)
        beta = ttnn.sigmoid(packed[:, :, 16384:16432])
        a = packed[:, :, 16448:16496]
        g = self._decay(a)
        out = self._delta_chunks(q, k, v, beta, g, state.recurrent)
        out = ttnn.typecast(out, ttnn.bfloat16)
        out = ttnn.rms_norm(
            out,
            weight=self.weights["linear_attn.norm.weight"],
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=self.ckc,
        )
        out = ttnn.transformer.concatenate_heads(out)
        out = (
            ttnn.multiply(ttnn.silu(packed[:, :, 10240:16384]), out)
            if self.policy.get("round_silu", False)
            else ttnn.multiply(packed[:, :, 10240:16384], out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        )
        return self._linear(out, "linear_attn.out_proj.weight")

    def _linear_decode(self, x, key):
        selected = self.policy.get("decode_weight_correction", self.weight_corrections)
        if key in self.weight_corrections and key not in selected:
            return ttnn.typecast(self._linear_raw(x, key), ttnn.bfloat16)
        return self._linear(x, key)

    def _gdn_compact(self, x, state, flat=False, decode=False):
        projection = self._linear_decode if decode else self._linear
        batch = x.shape[1] if flat else x.shape[0]
        xf = x if flat else ttnn.reshape(x, (1, batch, 5120))
        if self.policy.get("split_attention", False):
            packed = ttnn.concat(
                [
                    projection(xf, "linear_attn.in_proj_qkv.weight"),
                    projection(xf, "linear_attn.in_proj_z.weight"),
                    ttnn.pad(projection(xf, "linear_attn.in_proj_b.weight"), ((0, 0), (0, 0), (0, 16)), value=0.0),
                    ttnn.pad(projection(xf, "linear_attn.in_proj_a.weight"), ((0, 0), (0, 0), (0, 16)), value=0.0),
                ],
                dim=-1,
            )
        else:
            packed = projection(xf, "packed_gdn")
        q, k, v = self._batched_conv(packed[:, :, :10240], state.conv)
        if decode and self.policy.get("fused_beta_decay", True):
            beta, g = _gate178_gdn_beta_decay(packed, self.weights["linear_attn.dt_bias"], self.neg_a)
        else:
            beta = ttnn.reshape(ttnn.sigmoid(packed[:, :, 16384:16432]), (batch, 1, 48))
            a = packed[:, :, 16448:16496]
            g = self._decay(a)
            g = ttnn.reshape(g, (batch, 1, 48))
        if self.policy.get("sfpu_compact", False):
            normalized = self.policy.get("sfpu_normalized", True)
            if normalized:
                q, k = (ttnn.reshape(t[:, :1, :], (batch, 16, 128)) for t in (q, k))
                q = ttnn.multiply(self._l2(q), 128**-0.5)
                k = self._l2(k)
            recurrent_op = (
                _fused_delta_row_recurrent_sfpu_compact
                if self.policy.get("decay_predict_fusion", False)
                else (
                    _grouped96_recurrent_sfpu_compact
                    if self.policy.get("grouped_recurrent_init", False)
                    else _recurrent_sfpu_compact
                )
            )
            out = recurrent_op(
                q,
                k,
                v,
                g,
                beta,
                ttnn.reshape(state.recurrent, (batch * 48, 128, 128)),
                normalized=normalized,
                packed_qkv=self.policy.get("conv_packed_rm", False),
                cores=self.policy.get("sfpu_cores"),
            )
        else:
            out = self._delta_chunks(q, k, v, beta, g, state.recurrent)
        out = ttnn.typecast(ttnn.reshape(out, (1, batch * 48, 128)), ttnn.bfloat16)
        if self.policy.get("hf_gated_norm", False):
            out = ttnn.rms_norm(out, epsilon=self.config.rms_norm_eps, compute_kernel_config=self.ckc)
            if decode and self.policy.get("gdn_fused_affine_gate", True):
                out = _affinev3_gdn_affine_gate_flat(
                    out, self.weights["linear_attn.norm.weight"], packed, cores=self.policy.get("gdn_affine_gate_cores")
                )
                out = projection(out, "linear_attn.out_proj.weight")
                return (
                    out if flat else ttnn.reshape(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG), (batch, 1, 5120))
                )
            out = ttnn.multiply(out, self.weights["linear_attn.norm.weight"], dtype=ttnn.bfloat16)
        else:
            out = ttnn.rms_norm(
                out,
                weight=self.weights["linear_attn.norm.weight"],
                epsilon=self.config.rms_norm_eps,
                compute_kernel_config=self.ckc,
            )
        out = ttnn.reshape(out, (1, batch, 6144))
        if self.policy.get("hf_gated_norm", False):
            out = ttnn.multiply(
                ttnn.typecast(packed[:, :, 10240:16384], ttnn.float32),
                ttnn.typecast(out, ttnn.float32),
                dtype=ttnn.bfloat16,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            )
        else:
            out = (
                ttnn.multiply(ttnn.silu(packed[:, :, 10240:16384]), out)
                if self.policy.get("round_silu", False)
                else ttnn.multiply(packed[:, :, 10240:16384], out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            )
        out = projection(out, "linear_attn.out_proj.weight")
        return out if flat else ttnn.reshape(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG), (batch, 1, 5120))

    def _batched_conv(self, qkv, history):
        """Return flat TILE q/k/v; update existing RM [B,3,C] history in place.

        Arithmetic is compact [1,B,C] ROW_MAJOR, with one row per logical user.
        No host conversion, Python user loop, or [B,32,C] arithmetic intermediates.
        Native FP32 accumulation/SILU is targeted, not asserted bit-identical.
        """
        b, _, c = history.shape
        if self.policy.get("sfpu_decode_conv", False):
            assert self.policy.get("conv_packed_rm", False)
            assert self.policy.get("conv_round_output", False)
            assert not self.policy.get("conv_functional", False)
            assert not self.policy.get("conv_round_products", False)
            conv_op = (
                _conv107_conv_decode_sfpu if self.policy.get("conv_untilize_pipeline", False) else conv_decode_sfpu
            )
            result = conv_op(qkv, history, self.taps_rm, cores=self.policy.get("sfpu_conv_cores"))
            return result, result, result
        assert qkv.shape[-1] == 10240 and qkv.shape[0] * qkv.shape[1] == b
        assert tuple(history.shape) == (b, 3, c)
        token = ttnn.reshape(ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT), (b, 1, c))
        inputs = [ttnn.reshape(history[:, j : j + 1, :], (1, b, c)) for j in range(3)]
        inputs.append(ttnn.reshape(token, (1, b, c)))
        products = [
            ttnn.multiply(
                x,
                tap,
                dtype=(
                    ttnn.bfloat16
                    if (self.policy.get("conv_round_products", False) or self.policy.get("conv_functional", False))
                    else ttnn.float32
                ),
            )
            for x, tap in zip(inputs, self.taps_rm)
        ]
        accumulation = ttnn.bfloat16 if self.policy.get("conv_functional", False) else ttnn.float32
        products = [ttnn.typecast(t, accumulation) if t.dtype != accumulation else t for t in products]
        acc = ttnn.add(products[0], products[1], dtype=accumulation)
        acc = ttnn.add(acc, products[2], dtype=accumulation)
        # SILU executes before the sole BF16 output pack; fp32 inputs keep DST fp32.
        result = ttnn.add(
            acc,
            products[3],
            dtype=ttnn.bfloat16,
            activations=(
                []
                if (
                    self.policy.get("conv_round_output", False)
                    or self.policy.get("conv_round_products", False)
                    or self.policy.get("conv_functional", False)
                )
                else [ttnn.UnaryOpType.SILU]
            ),
        )
        if (
            self.policy.get("conv_round_output", False)
            or self.policy.get("conv_round_products", False)
            or self.policy.get("conv_functional", False)
        ):
            result = ttnn.silu(result)
        result = ttnn.reshape(result, (b, 1, c))
        next_history = ttnn.concat([history[:, 1:3, :], token], dim=1)
        ttnn.copy(next_history, history)
        if self.policy.get("conv_packed_rm", False):
            return result, result, result
        if self.policy.get("conv_rm", False):
            return tuple(result[:, :, start:end] for start, end in ((0, 2048), (2048, 4096), (4096, 10240)))
        return tuple(
            ttnn.pad(ttnn.to_layout(result[:, :, start:end], ttnn.TILE_LAYOUT), ((0, 0), (0, 31), (0, 0)), value=0.0)
            for start, end in ((0, 2048), (2048, 4096), (4096, 10240))
        )

    def _channel_conv(self, qkv, history):
        b = qkv.shape[0]
        ranges = ((0, 2048), (2048, 4096), (4096, 10240))
        widths = (2048, 2048, 6144)
        token = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT)
        groups = [ttnn.reshape(token[:, :, a:z], (1, 1, b * (z - a))) for a, z in ranges]
        packed = ttnn.to_layout(
            ttnn.pad(ttnn.concat(groups, dim=2), ((0, 0), (0, 31), (0, 0)), value=0.0), ttnn.TILE_LAYOUT
        )
        groups = [ttnn.reshape(ttnn.permute(history[:, :, a:z], (1, 0, 2)), (1, 3, b * (z - a))) for a, z in ranges]
        carry = ttnn.concat(groups, dim=2)
        outputs = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            packed,
            carry,
            *self.channel_taps[b],
            *(b * w for w in widths),
            predecessor_carry=carry,
            actual_start=self.conv_start,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=256),
            compute_kernel_config=ttnn.init_device_compute_kernel_config(
                self.mesh.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.copy(ttnn.concat([history[:, 1:3, :], token], dim=1), history)
        result = []
        for y, width in zip(outputs[:3], widths):
            y = ttnn.to_layout(y[:, :1, :], ttnn.ROW_MAJOR_LAYOUT)
            y = ttnn.reshape(y, (b, 1, width))
            result.append(ttnn.pad(ttnn.to_layout(y, ttnn.TILE_LAYOUT), ((0, 0), (0, 31), (0, 0)), value=0.0))
        return tuple(result)

    def _round_recurrent_state(self, state):
        if (
            self.kind == "linear_attention"
            and self.policy.get("round_recurrent_state", False)
            and state.recurrent.dtype == ttnn.float32
        ):
            rounded = ttnn.typecast(state.recurrent, ttnn.bfloat16)
            ttnn.typecast(rounded, ttnn.float32, output_tensor=state.recurrent)

    def decode_forward(self, x, *, state, current_pos=None, rope=None):
        """One device token per row. Inputs/state must remain stable across replay."""
        compact_input = x.shape[0] == 1 and x.shape[1] > 1
        assert x.shape[1] == 1 or (compact_input and x.shape[1] <= 32)
        if (
            self.policy.get("compact_residual", False)
            and (self.kind == "full_attention" or self.policy.get("compact_gdn", False))
            and not self._single_user_classic_decode(x.shape[1] if compact_input else x.shape[0], decode=True)
        ):
            batch = x.shape[1] if compact_input else x.shape[0]
            xf = x if compact_input else ttnn.reshape(x, (1, batch, 5120))
            h = self._norm(xf, "input_layernorm.weight")
            attention = (
                self._compact_attention_decode(h, state, current_pos, rope, flat=True, decode=True)
                if self.kind == "full_attention"
                else self._gdn_compact(h, state, flat=True, decode=True)
            )
            if self.kind == "full_attention" and self.policy.get("norm_projection_l1", True) and h.is_sharded():
                # Input norm's last consumer is the completed attention call.
                ttnn.deallocate(h)
            out = self._finish(xf, attention)
            self._round_recurrent_state(state)
            return (
                out
                if compact_input
                else ttnn.reshape(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG), (batch, 1, 5120))
            )
        h = self._norm(x, "input_layernorm.weight")
        attention = (
            self._gdn(h, state)
            if self.kind == "linear_attention"
            else self._attention_decode(h, state, current_pos, rope, decode=True)
        )
        if self.kind == "full_attention" and self.policy.get("norm_projection_l1", True) and h.is_sharded():
            ttnn.deallocate(h)
        out = self._finish(x, attention)
        self._round_recurrent_state(state)
        return out

    def prefill_forward(self, x, *, state, start_pos=0, rope=None):
        """Chunked device prefill with arbitrary logical lengths and starts.

        A list of tensors/states/starts/rope tuples describes independent rows
        with different lengths. Recurrent continuation requires the exact state
        snapshot at start_pos; full attention requires all earlier KV pages.
        The caller must not share writable request state between rows.
        """
        if isinstance(x, (list, tuple)):
            starts = [start_pos] * len(x) if isinstance(start_pos, int) else start_pos
            rotations = [None] * len(x) if rope is None else rope
            if not x or not (len(x) == len(state) == len(starts) == len(rotations)):
                raise ValueError("Each prefill row needs its own state, start and rotations")
            return [
                self.prefill_forward(t, state=s, start_pos=p, rope=r) for t, s, p, r in zip(x, state, starts, rotations)
            ]
        length = x.shape[1]
        if not (
            0 <= start_pos < self.config.max_position_embeddings
            and 1 <= length <= self.config.max_position_embeddings - start_pos
        ):
            raise ValueError("Prefill is outside the HF context contract")
        published_state = state
        if self.kind == "linear_attention" and state.recurrent.dtype == ttnn.bfloat16:
            state = RecurrentState(state.conv, ttnn.typecast(state.recurrent, ttnn.float32))
        outputs = []
        offset = 0
        alignment = self._attention_chunk(state) if self.kind == "full_attention" else 1
        while offset < length:
            absolute = start_pos + offset
            # SDPA-block-unaligned continuation uses exact token updates until
            # aligned; no padded write can overwrite an existing prefix row.
            n = 1 if absolute % alignment else min(self.policy.get("prefill_chunk", 512), length - offset)
            chunk = x[:, offset : offset + n]
            h = self._norm(chunk, "input_layernorm.weight")
            if self.kind == "linear_attention":
                attention = self._gdn(h, state)
            else:
                rotations = tuple(t[:, :, offset : offset + n, :] for t in rope)
                if absolute % alignment:
                    # Creation.full uploads small int32 tensors from the host.
                    # moreh_full dispatches a device fill, including ROW_MAJOR.
                    # A full 32-int physical row also covers its aligned NoC
                    # write; the native fill's tiny-row CB is undersized.
                    position = ttnn.reshape(
                        ttnn.moreh_full((1, 32), absolute, self.mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)[
                            :, : x.shape[0]
                        ],
                        (x.shape[0],),
                    )
                    attention = self._attention_decode(h, state, position, rotations)
                else:
                    attention = self._attention_prefill(h, state, start_pos=absolute, rope=rotations)
            if self.kind == "full_attention" and self.policy.get("norm_projection_l1", True) and h.is_sharded():
                # Covers compact small-prefill chunks and continuations too.
                ttnn.deallocate(h)
            outputs.append(self._finish(chunk, attention, prefill=True))
            offset += n
        if state is not published_state:
            ttnn.typecast(state.recurrent, ttnn.bfloat16, output_tensor=published_state.recurrent)
        else:
            self._round_recurrent_state(state)
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=1)

    def create_paged_state(self, page_table):
        """Setup: 32-token pages; append finite owned pages for 512-key SDPA blocks."""
        import torch

        assert page_table.ndim == 2 and 1 <= page_table.shape[0] <= 32
        blocks = int(page_table.max()) + 1
        if page_table.shape[1] >= 16:
            extra = (-page_table.shape[1]) % 16
            if extra:
                padding = torch.arange(blocks, blocks + page_table.shape[0] * extra, dtype=page_table.dtype)
                page_table = torch.cat((page_table, padding.reshape(page_table.shape[0], extra)), dim=1)
                blocks += padding.numel()
        shape = (blocks, 4, 32, 256)
        table = ttnn.from_torch(
            page_table.to(torch.int32), device=self.mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        return PagedState(
            self._upload(
                torch.zeros(shape, dtype=torch.bfloat16), dtype=getattr(ttnn, self.policy.get("kv_dtype", "bfloat16"))
            ),
            self._upload(
                torch.zeros(shape, dtype=torch.bfloat16), dtype=getattr(ttnn, self.policy.get("kv_dtype", "bfloat16"))
            ),
            table,
        )

    @staticmethod
    def _attention_chunk(state):
        # Native SDPA repacks running state at each key-block boundary. A
        # 512-key block bounds long-context rounding; FP32 dest alone does not.
        pages = state.page_table.shape[1]
        if pages < 16:
            return 32
        assert pages % 16 == 0, "Large page tables require finite padding to 512-token extent"
        return 512

    def _rope(self, x, rope):
        cos, sin = rope
        if (
            self.policy.get("batched_rope", False)
            and x.shape[0] > 1
            and x.shape[2] == 1
            and x.shape[0] in (2, 4, 8, 16, 32)
        ):
            batch = x.shape[0]
            gx = max(v for v in range(1, min(batch, 8) + 1) if batch % v == 0)
            gy = batch // gx
            grid = ttnn.CoreGrid(x=gx, y=gy)
            mem = ttnn.create_sharded_memory_config(
                (32, 64), grid, ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
            )
            part = ttnn.to_memory_config(ttnn.permute(x[:, :, :, :64], (2, 0, 1, 3)), mem)
            trig = [ttnn.to_memory_config(ttnn.permute(t, (2, 0, 1, 3)), mem) for t in (cos, sin)]
            rot = ttnn.experimental.rotary_embedding_hf(
                part, *trig, is_decode_mode=True, compute_kernel_config=self.ckc
            )
            rot = ttnn.permute(ttnn.to_memory_config(rot, ttnn.DRAM_MEMORY_CONFIG), (1, 2, 0, 3))
            return ttnn.concat([rot, x[:, :, :, 64:]], dim=-1)
        rotated = [
            ttnn.experimental.rotary_embedding_hf(
                x[row : row + 1, :, :, :64],
                cos[row : row + 1],
                sin[row : row + 1],
                compute_kernel_config=self.ckc,
            )
            for row in range(x.shape[0])
        ]
        rot = rotated[0] if len(rotated) == 1 else ttnn.concat(rotated, dim=0)
        return ttnn.concat([rot, x[:, :, :, 64:]], dim=-1)

    def _qkv(self, x, rope, *, prefill=False):
        gate = None
        mode = self.policy.get("full_pack_all", False)
        if mode is True or (mode == "prefill" and prefill):
            all_proj = self._linear(x, "packed_qkvg")
            projected, gate = all_proj[..., :8192], all_proj[..., 8192:]
        elif self.policy.get("split_attention", False):
            projected = ttnn.concat(
                [self._linear(x, key) for key in ("query", "self_attn.k_proj.weight", "self_attn.v_proj.weight")],
                dim=-1,
            )
        else:
            projected = self._linear(x, "packed_qkv")
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            projected, num_heads=24, num_kv_heads=4, transpose_key=False
        )
        q = self._norm(q, "self_attn.q_norm.weight")
        k = self._norm(k, "self_attn.k_norm.weight")
        return self._rope(q, rope), self._rope(k, rope), v, gate

    def _attention_out(self, x, out, gate=None):
        out = ttnn.transformer.concatenate_heads(out)
        out = ttnn.multiply(
            self._linear(x, "gate") if gate is None else gate,
            out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID],
        )
        return self._linear(out, "self_attn.o_proj.weight")

    def _single_user_classic_decode(self, batch, *, decode=False):
        return (
            decode
            and self.kind == "full_attention"
            and batch == 1
            and self.policy.get("single_user_classic_attention", False)
        )

    def _decode_attention_chunk(self, state, *, decode=False):
        baseline = self._attention_chunk(state)
        if decode and baseline == 512 and state.page_table.shape[1] <= 64:
            chunk = self.policy.get("short_capacity_decode_k_chunk", 512)
            assert chunk in (32, 64, 128, 256, 512)
            return chunk
        return baseline

    def _decode_sdpa_grid(self, batch, *, decode=False):
        override = self.policy.get("single_user_sdpa_grid")
        if decode and self.kind == "full_attention" and batch == 1 and override is not None:
            return override
        return self.policy.get("sdpa_grid", [8, 4])

    def _attention_decode(self, x, state, current_pos, rope, *, decode=False):
        if self.policy.get("compact_attention", False) and not self._single_user_classic_decode(
            x.shape[0] * x.shape[1], decode=decode
        ):
            return self._compact_attention_decode(x, state, current_pos, rope, decode=decode)
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
            memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(cores, [32, 256], ttnn.ShardOrientation.ROW_MAJOR),
            )
            inputs.append(ttnn.to_memory_config(ttnn.permute(tensor, (2, 0, 1, 3)), memory))
        ttnn.experimental.paged_fused_update_cache(
            state.key,
            inputs[0],
            state.value,
            inputs[1],
            update_idxs_tensor=current_pos,
            page_table=state.page_table,
        )
        # Update and attention share the ordered queue. Release the temporary
        # L1 inputs before SDPA reserves its large-context circular buffers.
        del inputs
        q = ttnn.permute(q, (2, 0, 1, 3))
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            state.key,
            state.value,
            page_table_tensor=state.page_table,
            cur_pos_tensor=current_pos,
            scale=256**-0.5,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*self._decode_sdpa_grid(b, decode=decode)),
                q_chunk_size=32,
                k_chunk_size=self._decode_attention_chunk(state, decode=decode),
                exp_approx_mode=False,
            ),
            compute_kernel_config=self.ckc,
        )
        out = ttnn.permute(out[:, :, :24, :], (1, 2, 0, 3))
        return self._attention_out(x, out, gate)

    def _attention_prefill(self, x, state, *, start_pos, rope):
        assert start_pos % self._attention_chunk(state) == 0
        q, k, v, gate = self._qkv(x, rope, prefill=True)
        s = x.shape[1]
        padded = (s + 31) // 32 * 32
        if s != padded:
            q, k, v = (ttnn.pad(t, ((0, 0), (0, 0), (0, padded - s), (0, 0)), value=0.0) for t in (q, k, v))
        k, v = (ttnn.typecast(t, state.key.dtype) if t.dtype != state.key.dtype else t for t in (k, v))
        table = state.page_table[:, start_pos // 32 :]
        for row in range(x.shape[0]):
            ttnn.experimental.paged_fill_cache(state.key, k[row : row + 1], table, batch_idx=row)
            ttnn.experimental.paged_fill_cache(state.value, v[row : row + 1], table, batch_idx=row)
        out = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            state.key,
            state.value,
            state.page_table,
            chunk_start_idx=start_pos,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*self.policy.get("sdpa_grid", [8, 4])),
                q_chunk_size=32,
                k_chunk_size=self._attention_chunk(state),
                exp_approx_mode=False,
            ),
            compute_kernel_config=self.ckc,
        )
        return self._attention_out(x, out[:, :, :s, :], gate)

    def _l2(self, x):
        if self.policy.get("hf_qk_norm", False):
            square = ttnn.multiply(x, x, dtype=ttnn.bfloat16)
            norm = ttnn.sum(square, dim=-1, keepdim=True)
            norm = ttnn.add(norm, 1e-6, dtype=ttnn.bfloat16)
            inv = ttnn.rsqrt(norm)
            return ttnn.typecast(ttnn.multiply(x, inv, dtype=ttnn.bfloat16), ttnn.float32)
        x = ttnn.typecast(x, ttnn.float32)
        return ttnn.multiply(x, ttnn.rsqrt(ttnn.add(ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True), 1e-6)))

    def _delta_step(self, q, k, v, beta, g, state):
        b = q.shape[0]
        q = ttnn.repeat_interleave(self._l2(q), 3, dim=2)
        k = ttnn.repeat_interleave(self._l2(k), 3, dim=2)
        q = ttnn.permute(q, (0, 2, 3, 1))
        k = ttnn.permute(k, (0, 2, 3, 1))
        v = ttnn.permute(ttnn.typecast(v, ttnn.float32), (0, 2, 1, 3))
        beta = ttnn.reshape(ttnn.typecast(beta, ttnn.float32), (b, 48, 1, 1))
        g = ttnn.reshape(g, (b, 48, 1, 1))
        h = ttnn.multiply(state, ttnn.exp(g))
        prediction = ttnn.sum(ttnn.multiply(h, k), dim=-2, keepdim=True)
        delta = ttnn.multiply(ttnn.subtract(v, prediction), beta)
        h = ttnn.add(h, ttnn.multiply(k, delta), output_tensor=state)
        out = ttnn.sum(ttnn.multiply(h, ttnn.multiply(q, 128**-0.5)), dim=-2, keepdim=True)
        return out

    def _native_recurrent(self, q, k, v, beta, g, state):
        """At most 2 users/96 heads per call on 110 cores; no host tensor conversion.

        scatter_state=True stages each FP32 final slab into L1 and uses tiled
        sharded slice_write into the original state. False concatenates final slabs
        once and copies once, an easier correctness control.
        """
        b = q.shape[0]
        q = self._l2(ttnn.reshape(q[:, :1, :], (b, 1, 16, 128)))
        k = self._l2(ttnn.reshape(k[:, :1, :], (b, 1, 16, 128)))
        v = ttnn.typecast(ttnn.reshape(v[:, :1, :], (b, 1, 48, 128)), ttnn.float32)
        scatter_state = self.policy.get("recurrent_scatter", False)
        outputs, states = [], []
        ncores = self.mesh.compute_with_storage_grid_size()
        users_per_wave = min(2, (ncores.x * ncores.y) // 48)
        assert users_per_wave >= 1
        for start in range(0, b, users_per_wave):
            end = min(b, start + users_per_wave)
            out, updated = ttnn.transformer.fused_recurrent_gated_delta_rule(
                q[start:end],
                k[start:end],
                v[start:end],
                g[start:end],
                beta[start:end],
                scale=128**-0.5,
                initial_state=state[start:end],
                output_final_state=True,
                use_qk_l2norm=False,
                compute_kernel_config=self.ckc,
            )
            outputs.append(ttnn.permute(out, (0, 2, 1, 3)))
            if scatter_state:
                count = (end - start) * 48
                grid = ttnn.num_cores_to_corerangeset(count, ncores, row_wise=True)
                memory = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(grid, [128, 128], ttnn.ShardOrientation.ROW_MAJOR),
                )
                staged = ttnn.to_memory_config(updated, memory)
                # Keep source TILE+sharded: interleaved slice_write converts/replaces dst.
                ttnn.experimental.slice_write(
                    staged,
                    state,
                    [start, 0, 0, 0],
                    [end, 48, 128, 128],
                    [1, 1, 1, 1],
                )
            else:
                states.append(updated)
        if not scatter_state:
            updated = states[0] if len(states) == 1 else ttnn.concat(states, dim=0)
            ttnn.copy(updated, state)
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=0)

    def _delta_existing_ops(self, q, k, v, beta, g, state):
        l2 = self._l2
        compute_kernel_config = self.ckc
        use_addcmul = self.policy.get("delta_ternary", True)
        use_matmul = self.policy.get("delta_matmul", False)
        block_k = self.policy.get("delta_block_k", 4)
        per_core_n = self.policy.get("delta_per_core_n", 4)
        subblock_n = self.policy.get("delta_subblock_n", 4)
        """Ablate fused outer update, then batched matrix-vector reductions.

        q/k: [B,1,16,128], v: [B,1,48,128], beta/g: [B,1,48].
        State/output: [B,48,128,128] / [B,48,1,128], float32 TILE.
        `state` is overwritten without replacing its allocation.
        """
        b = q.shape[0]
        q = ttnn.permute(ttnn.repeat_interleave(l2(q), 3, dim=2), (0, 2, 1, 3))
        k = ttnn.permute(ttnn.repeat_interleave(l2(k), 3, dim=2), (0, 2, 1, 3))
        v = ttnn.permute(ttnn.typecast(v, ttnn.float32), (0, 2, 1, 3))
        beta = ttnn.reshape(ttnn.typecast(beta, ttnn.float32), (b, 48, 1, 1))
        g = ttnn.reshape(g, (b, 48, 1, 1))
        k_column = ttnn.transpose(k, -2, -1)
        decayed = ttnn.multiply(state, ttnn.exp(g))

        if use_matmul:
            # Interleaved reuse distributes arbitrarily many batched output blocks
            # over the grid; no one-core-per-head constraint applies.
            grid = state.device().compute_with_storage_grid_size()
            program = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                in0_block_w=block_k,
                out_subblock_h=1,
                out_subblock_w=subblock_n,
                per_core_M=1,
                per_core_N=per_core_n,
            )

            def contract(vector, matrix):
                return ttnn.matmul(
                    vector,
                    matrix,
                    dtype=ttnn.float32,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    compute_kernel_config=compute_kernel_config,
                    program_config=program,
                )

        else:

            def contract(vector, matrix):
                return ttnn.sum(
                    ttnn.multiply(matrix, ttnn.transpose(vector, -2, -1)),
                    dim=-2,
                    keepdim=True,
                )

        prediction = contract(k, decayed)
        delta = ttnn.multiply(ttnn.subtract(v, prediction), beta)
        if use_addcmul:
            # FP32 full+column+row is native ROW_COL_BCAST, not a composite fallback.
            ttnn.addcmul(decayed, k_column, delta, value=1.0, output_tensor=state)
        else:
            ttnn.add(decayed, ttnn.multiply(k_column, delta), output_tensor=state)
        return contract(ttnn.multiply(q, 128**-0.5), state)

    def _sfpu_recurrent(self, q, k, v, beta, g, state):
        batch = q.shape[0]
        q, k = (ttnn.reshape(t[:, :1, :], (batch, 1, 16, 128)) for t in (q, k))
        q = ttnn.multiply(self._l2(q), 128**-0.5)
        k = self._l2(k)
        q, k = (
            ttnn.reshape(ttnn.permute(ttnn.repeat_interleave(t, 3, dim=2), (0, 2, 3, 1)), (batch * 48, 128, 1))
            for t in (q, k)
        )
        v = ttnn.reshape(ttnn.typecast(v[:, :1, :], ttnn.float32), (batch * 48, 1, 128))
        beta = ttnn.reshape(ttnn.typecast(beta, ttnn.float32), (batch * 48, 1, 1))
        decay = ttnn.exp(ttnn.reshape(g, (batch * 48, 1, 1)))
        state_view = ttnn.reshape(state, (batch * 48, 128, 128))
        out = _recurrent_sfpu(q, k, v, decay, beta, state_view, cores=self.policy.get("sfpu_cores"))
        return ttnn.reshape(out, (batch, 48, 1, 128))

    def _delta_chunks(self, q, k, v, beta, g, state):
        b, s = q.shape[0], beta.shape[1]
        if s == 1 and self.policy.get("sfpu_recurrent", False):
            return self._sfpu_recurrent(q, k, v, beta, g, state)
        if s == 1 and self.policy.get("native_recurrent", False):
            return self._native_recurrent(q, k, v, beta, g, state)
        if s == 1 and self.policy.get("delta_direct", False):
            q, k = (ttnn.reshape(t[:, :1, :], (b, 1, 16, 128)) for t in (q, k))
            v = ttnn.reshape(v[:, :1, :], (b, 1, 48, 128))
            if self.policy.get("delta_ternary", False) or self.policy.get("delta_matmul", False):
                return self._delta_existing_ops(q, k, v, beta, g, state)
            return self._delta_step(q, k, v, beta, g, state)
        padded = (s + 31) // 32 * 32
        if b == 1 and s == 1:
            # Same phased topology as the public op at this shape. The prep
            # reader masks logical gate tails, removing two explicit pads.
            # Larger batches retain the public wave-capable native path.
            native = ttnn._ttnn.operations.transformer
            prep = native.chunk_gdn_prep(
                q,
                k,
                v,
                g,
                beta,
                self.eye,
                self.lower,
                self.ones,
                self.masks,
                chunk_size=32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.ckc,
                v_flat=True,
                HV=48,
                qk_norm=True,
                scale=128**-0.5,
                qk_flat=True,
                Hk=16,
                prep_serial=False,
                wy_inverse=ttnn.ChunkGdnWyInverse.AUTO,
            )
            out, updated = native.chunk_gdn_scan(
                *prep,
                initial_state=ttnn.reshape(state, (48, 128, 128)),
                chunk_size=32,
                output_final_state=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.ckc,
                use_mcast=True,
                force_serial=False,
            )
            ttnn.copy(ttnn.reshape(updated, (1, 48, 128, 128)), state)
            return ttnn.reshape(out, ttnn.Shape((1, 48, 1, 128)), ttnn.Shape((1, 48, 32, 128)))
        if s != padded:
            beta, g = (ttnn.pad(t, ((0, 0), (0, padded - s), (0, 0)), value=0.0) for t in (beta, g))
        if self.policy.get("inplace_gdn", False) and b > 1 and s == 1:
            normalize_in_kernel = True
            if self.policy.get("bf16_qk_norm", False):
                normalized = []
                for tensor in (q, k):
                    shaped = ttnn.reshape(tensor, (b, padded, 16, 128))
                    square = ttnn.multiply(shaped, shaped, dtype=ttnn.bfloat16)
                    summed = ttnn.sum(square, dim=-1, keepdim=True)
                    inv = ttnn.rsqrt(ttnn.add(summed, 1e-6))
                    shaped = ttnn.multiply(shaped, inv, dtype=ttnn.bfloat16)
                    shaped = ttnn.repeat_interleave(shaped, 3, dim=2)
                    shaped = ttnn.permute(shaped, (0, 2, 1, 3))
                    normalized.append(ttnn.reshape(shaped, (b * 48, 1, padded, 128)))
                q, k = normalized
                q = ttnn.multiply(q, 128**-0.5)
                normalize_in_kernel = False
            native = ttnn._ttnn.operations.transformer
            state_view = ttnn.reshape(state, (b * 48, 128, 128))
            out = ttnn.empty(
                (b * 48, 1, 32, 128),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            wave = self.policy.get("gdn_program", {}).get("wave_heads", 48)
            config = self.gdn_program or ttnn.ChunkGdnFusedProgramConfig(
                wave_heads=48, producer_pool=True, num_receivers=1
            )
            for head_offset in range(0, b * 48, wave):
                native.chunk_gdn_fused(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    self.eye,
                    self.lower,
                    self.ones,
                    self.masks,
                    state_view,
                    chunk_size=32,
                    output_final_state=True,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    compute_kernel_config=self.ckc,
                    program_config=config,
                    v_flat=True,
                    HV=48,
                    qk_norm=normalize_in_kernel,
                    scale=128**-0.5,
                    qk_flat=normalize_in_kernel,
                    Hk=16,
                    num_heads=min(wave, b * 48 - head_offset),
                    head_offset=head_offset,
                    o_out=out,
                    final_state_out=state_view,
                )
            return ttnn.reshape(out, ttnn.Shape((b, 48, s, 128)), ttnn.Shape((b, 48, padded, 128)))
        out, updated = ttnn.transformer.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=state,
            output_final_state=True,
            chunk_size=32,
            num_key_heads=16,
            output_head_major=True,
            program_config=self.gdn_program if b > 1 and s == 1 else None,
            eye=self.eye,
            tril=self.lower,
            ones=self.ones,
            masks=self.masks,
            compute_kernel_config=self.ckc,
        )
        ttnn.copy(updated, state)
        return ttnn.reshape(out, ttnn.Shape((b, 48, s, 128)), ttnn.Shape((b, 48, padded, 128)))

    def _compact_cores(self, count, offset=0):
        ranges = []
        while count:
            x, y = offset % 8, offset // 8
            rows = count // 8 if x == 0 else 0
            if rows:
                ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, y), ttnn.CoreCoord(7, y + rows - 1)))
                step = rows * 8
            else:
                step = min(count, 8 - x)
                ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x + step - 1, y)))
            count -= step
            offset += step
        return ttnn.CoreRangeSet(ranges)

    def _compact_height_mem(self, cores, height, width):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, [height, width], ttnn.ShardOrientation.ROW_MAJOR),
        )

    def _compact_rope_mem(self, batch, grid):
        # The rotary factory iterates a complete bounding rectangle, not sparse cores.
        # Choose an exact divisor of logical B: no fabricated users or mixed positions.
        choices = [
            (x * y, x, y)
            for x in range(1, grid.x + 1)
            for y in range(1, grid.y + 1)
            if x * y <= batch and batch % (x * y) == 0
        ]
        n, x, y = max(choices)
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))})
        return self._compact_height_mem(cores, (batch // n) * 32, 64)

    def _compact_attention_decode(self, x, state, current_pos, rope, flat=False, decode=False):
        b, seq, width = x.shape
        if flat:
            b, seq = seq, 1
        assert seq == 1 and 1 <= b <= 32
        # Fold before projection so intermediate projections remain compact token rows.
        xf = ttnn.reshape(x, (1, b, width))
        gate = None
        if self.policy.get("full_pack_all", False) is True:
            all_proj = self._linear(xf, "packed_qkvg")
            projected, gate = all_proj[..., :8192], all_proj[..., 8192:]
        elif self.policy.get("split_attention", False):
            projected = ttnn.concat(
                [self._linear(xf, name) for name in ("query", "self_attn.k_proj.weight", "self_attn.v_proj.weight")],
                dim=-1,
            )
        else:
            projected = self._linear(xf, "packed_qkv")
        heads_mem = self._compact_height_mem(self._compact_cores(b), 32, 256)
        q, k, v = qkv_split_batched_staging(
            ttnn.reshape(projected, (1, 1, b, 8192)),
            num_heads=24,
            num_kv_heads=4,
            memory_config=heads_mem,
        )
        # Native RMSNorm rejects HEIGHT_SHARDED: use compact interleaved data.
        q, k = (
            ttnn.rms_norm(
                ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG),
                weight=self.weights[name],
                epsilon=self.config.rms_norm_eps,
                compute_kernel_config=self.ckc,
            )
            for t, name in ((q, "self_attn.q_norm.weight"), (k, "self_attn.k_norm.weight"))
        )
        rot_mem = self._compact_rope_mem(b, self.mesh.compute_with_storage_grid_size())
        trig = tuple(ttnn.to_memory_config(ttnn.permute(t, (2, 0, 1, 3)), rot_mem) for t in rope)

        def rotate(t):
            prefix = ttnn.to_memory_config(t[..., :64], rot_mem)
            prefix = ttnn.experimental.rotary_embedding_hf(
                prefix,
                *trig,
                is_decode_mode=True,
                memory_config=rot_mem,
                compute_kernel_config=self.ckc,
            )
            return ttnn.concat([ttnn.to_memory_config(prefix, ttnn.DRAM_MEMORY_CONFIG), t[..., 64:]], dim=-1)

        q, k = rotate(q), rotate(k)
        # Native splitter puts V on cores [0,B); K update must be disjoint.
        k_update = ttnn.to_memory_config(k, self._compact_height_mem(self._compact_cores(b, offset=32), 32, 256))
        ttnn.experimental.paged_fused_update_cache(
            state.key,
            k_update,
            state.value,
            v,
            update_idxs_tensor=current_pos,
            page_table=state.page_table,
        )
        del k_update, k, v
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            state.key,
            state.value,
            page_table_tensor=state.page_table,
            cur_pos_tensor=current_pos,
            scale=256**-0.5,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*self._decode_sdpa_grid(b, decode=decode)),
                q_chunk_size=32,
                k_chunk_size=self._decode_attention_chunk(state, decode=decode),
                exp_approx_mode=False,
            ),
            compute_kernel_config=self.ckc,
        )
        # GQA currently requires interleaved SDPA output; this is a small required reshard.
        out = ttnn.to_memory_config(out, heads_mem)
        grid = self.mesh.compute_with_storage_grid_size()
        full_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        out = ttnn.experimental.nlp_concat_heads_decode(out, num_heads=24, sub_core_grids=full_grid)
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        # Concat advertises batch32 even for smaller logical B. Crop BEFORE gate/projection.
        out = ttnn.reshape(out[:, :, :b, :], (1, b, 6144))
        gate = self._linear(xf, "gate") if gate is None else gate
        out = ttnn.multiply(gate, out, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        out = self._linear(out, "self_attn.o_proj.weight")
        return out if flat else ttnn.reshape(out, (b, 1, 5120))


_GDN_SFPU_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto qa=TensorAccessorArgs<0>();
    constexpr auto ka=TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va=TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto da=TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ba=TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto sa=TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    const auto q=TensorAccessor(qa,get_arg_val<uint32_t>(0),4096);
    const auto k=TensorAccessor(ka,get_arg_val<uint32_t>(1),4096);
    const auto v=TensorAccessor(va,get_arg_val<uint32_t>(2),4096);
    const auto d=TensorAccessor(da,get_arg_val<uint32_t>(3),4096);
    const auto b=TensorAccessor(ba,get_arg_val<uint32_t>(4),4096);
    const auto s=TensorAccessor(sa,get_arg_val<uint32_t>(5),4096);
    const uint32_t start=get_arg_val<uint32_t>(6), count=get_arg_val<uint32_t>(7);
    Noc noc;
    auto read=[&](const auto& a,uint32_t id,uint32_t base,uint32_t n) {
        CircularBuffer cb(id); cb.reserve_back(n);
        for(uint32_t i=0;i<n;i++)
            noc.async_read(a,cb,4096,{.page_id=base+i},{.offset_bytes=i*4096});
        noc.async_read_barrier(); cb.push_back(n);
    };
    // Local constants: full zero and one tiles. No device tensor allocation.
    for(uint32_t id=6;id<=7;id++) {
        CircularBuffer cb(id); cb.reserve_back(1);
        volatile tt_l1_ptr uint32_t* p=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
        for(uint32_t i=0;i<1024;i++) p[i]=(id==7 ? 0x3f800000u : 0u);
        cb.push_back(1);
    }
    for(uint32_t h=start;h<start+count;h++) {
        // Full head read completes before compute can emit any state write.
        read(s,5,h*16,16);
        read(q,0,h*4,4); read(k,1,h*4,4); read(v,2,h*4,4);
        read(d,3,h,1); read(b,4,h,1);
    }
}
"""

_GDN_SFPU_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto sa=TensorAccessorArgs<0>();
    constexpr auto oa=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto s=TensorAccessor(sa,get_arg_val<uint32_t>(0),4096);
    const auto o=TensorAccessor(oa,get_arg_val<uint32_t>(1),4096);
    const uint32_t start=get_arg_val<uint32_t>(2),count=get_arg_val<uint32_t>(3);
    Noc noc; CircularBuffer updated(11),out(12);
    for(uint32_t h=start;h<start+count;h++) {
        updated.wait_front(16); out.wait_front(4);
        // Waiting for output guarantees compute finished reading updated CB.
        for(uint32_t i=0;i<16;i++)
            noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(updated),s,4096,{.offset_bytes=i*4096},{.page_id=h*16+i});
        for(uint32_t i=0;i<4;i++)
            noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),o,4096,{.offset_bytes=i*4096},{.page_id=h*4+i});
        noc.async_write_barrier(); updated.pop_front(16); out.pop_front(4);
    }
}
"""

_GDN_SFPU_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reduce.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"

void load(uint32_t cb,uint32_t tile,uint32_t dst) {
    copy_init(cb); copy_tile(cb,tile,dst);
}
void finish(uint32_t cb,uint32_t tile) {
    tile_regs_commit(); tile_regs_wait(); pack_tile<true>(0,cb,tile); tile_regs_release();
}
// Exactly the accurate FP32 reduction topology: fold the four H tiles with
// SFPU addition, then SFPU reduce their32 rows into row0. No TF32 FPU path.
void contract(uint32_t matrix,uint32_t column,uint32_t result) {
    CircularBuffer dst(result); dst.reserve_back(4);
    PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_COL,PackMode::Default>(result)));
    for(uint32_t n=0;n<4;n++) {
        tile_regs_acquire();
        for(uint32_t m=0;m<4;m++) {
            const uint32_t work=m==0 ? 0 : 2;
            load(matrix,m*4+n,work); load(column,m,1);
            sfpu_mul_bcast_col_init(); sfpu_mul_bcast_col(work,1);
            if(m) { add_binary_tile_init(); add_binary_tile(0,2,0); }
        }
        sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_COL>(0,1,1);
        finish(result,n);
    }
    PACK((llk_pack_reduce_mask_clear())); dst.push_back(4);
}
void kernel_main() {
    const uint32_t count=get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(5,6,12);
    CircularBuffer q(0),k(1),v(2),decay(3),beta(4),state(5),zero(6),one(7);
    CircularBuffer sd(8),pred(9),delta(10),updated(11);
    zero.wait_front(1); one.wait_front(1);
    for(uint32_t head=0;head<count;head++) {
        state.wait_front(16); decay.wait_front(1);
        sd.reserve_back(16);
        for(uint32_t i=0;i<16;i++) {
            tile_regs_acquire(); load(5,i,0); load(6,0,1); load(3,0,2);
            sfpu_normalize_bcast_scalar_init(); sfpu_normalize_bcast_scalar(0,1,2);
            finish(8,i);
        }
        sd.push_back(16); state.pop_front(16); decay.pop_front(1);
        sd.wait_front(16); k.wait_front(4);
        contract(8,1,9); pred.wait_front(4);
        v.wait_front(4); beta.wait_front(1); delta.reserve_back(4);
        for(uint32_t n=0;n<4;n++) {
            tile_regs_acquire(); load(2,n,0); load(9,n,1);
            sub_binary_tile_init(); sub_binary_tile(0,1,0);
            load(6,0,1); load(4,0,2);
            sfpu_normalize_bcast_scalar_init(); sfpu_normalize_bcast_scalar(0,1,2);
            finish(10,n);
        }
        delta.push_back(4); pred.pop_front(4); v.pop_front(4); beta.pop_front(1);
        delta.wait_front(4); updated.reserve_back(16);
        for(uint32_t m=0;m<4;m++) for(uint32_t n=0;n<4;n++) {
            tile_regs_acquire(); load(7,0,0); load(10,n,1);
            sfpu_mul_bcast_row_init(); sfpu_mul_bcast_row(0,1);
            load(1,m,1); sfpu_mul_bcast_col_init(); sfpu_mul_bcast_col(0,1);
            load(8,m*4+n,1); add_binary_tile_init(); add_binary_tile(1,0,0);
            finish(11,m*4+n);
        }
        updated.push_back(16); sd.pop_front(16); delta.pop_front(4); k.pop_front(4);
        updated.wait_front(16); q.wait_front(4);
        contract(11,0,12); q.pop_front(4);
        // Writer pops updated/output only after contract has finished.
    }
    zero.pop_front(1); one.pop_front(1);
}
"""


def _recurrent_sfpu(qcol, kcol, v, decay, beta, state, *, output=None, cores=None):
    """Enqueue one fused recurrent program; no synchronization or host fallback.

    Rebuild descriptors per call so new eager tensor addresses enter runtime args.
    Runtime values are excluded from generic_op's program hash and overridden by
    descriptor dispatch. Trace capture freezes the correct current addresses.
    The final io_tensors entry is the preallocated output, as generic_op requires.
    """
    bh = state.shape[0]
    expected = ((bh, 128, 1), (bh, 128, 1), (bh, 1, 128), (bh, 1, 1), (bh, 1, 1), (bh, 128, 128))
    inputs = [qcol, kcol, v, decay, beta, state]
    for t, shape in zip(inputs, expected):
        assert tuple(t.shape) == shape, (tuple(t.shape), shape)
        assert t.dtype == ttnn.float32 and t.layout == ttnn.TILE_LAYOUT
        assert t.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    if output is None:
        output = ttnn.empty(
            (bh, 1, 128),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=state.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    grid = state.device().compute_with_storage_grid_size()
    count = min(bh, grid.x * grid.y, cores or grid.x * grid.y)
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(count)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    reader_args, writer_args, compute_args = (ttnn.RuntimeArgs() for _ in range(3))
    start = 0
    for i, c in enumerate(coords):
        n = bh // count + (i < bh % count)
        reader_args[c.x][c.y] = [t.buffer_address() for t in inputs] + [start, n]
        writer_args[c.x][c.y] = [state.buffer_address(), output.buffer_address(), start, n]
        compute_args[c.x][c.y] = [n]
        start += n
    reader_ct = []
    for t in inputs:
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    writer_ct = []
    for t in (state, output):
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    def kernel(source, config, rt, ct):
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )

    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    compute_config.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] * 64
    kernels = [
        kernel(_GDN_SFPU_READER, ttnn.ReaderConfigDescriptor(), reader_args, reader_ct),
        kernel(_GDN_SFPU_WRITER, ttnn.WriterConfigDescriptor(), writer_args, writer_ct),
        kernel(_GDN_SFPU_COMPUTE, compute_config, compute_args, []),
    ]
    sizes = {0: 4, 1: 4, 2: 4, 3: 1, 4: 1, 5: 16, 6: 1, 7: 1, 8: 16, 9: 4, 10: 4, 11: 16, 12: 4}
    cbs = [
        ttnn.CBDescriptor(
            total_size=n * 4096,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096)],
        )
        for i, n in sizes.items()
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    ttnn.generic_op(inputs + [output], program)
    return output


_HIDDEN_NORM_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto xa=TensorAccessorArgs<0>();
    constexpr auto ga=TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0),2048);
    const auto gamma=TensorAccessor(ga,get_arg_val<uint32_t>(1),4096);
    const uint32_t user=get_arg_val<uint32_t>(2);
    Noc noc; CircularBuffer dense(0),weight(1),zero(4),scratch(5);
    dense.reserve_back(5); weight.reserve_back(5); scratch.reserve_back(5);
    zero.reserve_back(1);
    volatile tt_l1_ptr uint32_t* z=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero.get_write_ptr());
    for(uint32_t i=0;i<1024;i++) z[i]=0;
    zero.push_back(1);
    // Full vectors contain exactly five dense32x32 tiles, without padding.
    for(uint32_t tile=0;tile<160;tile++) for(uint32_t face=0;face<2;face++) {
        const uint32_t linear=tile*32+face*16;
        const uint32_t dt=linear/1024, dr=(linear%1024)/32, dc=linear%32;
        const uint32_t dense_offset=(dt*1024+(dr/16)*512+(dc/16)*256+(dr%16)*16)*4;
        const uint32_t x_offset=((user/16)*512+face*256+(user%16)*16)*2;
        // Source and L1 destination must match modulo64 for BH DRAM reads.
        // Read a pair of BF16 rows and select user%2 after the shared barrier.
        noc.async_read(x,scratch,64,{.page_id=tile,.offset_bytes=x_offset&~63u},
            {.offset_bytes=(tile*2+face)*64});
        noc.async_read(gamma,weight,64,{.page_id=tile,.offset_bytes=face*1024},
            {.offset_bytes=dense_offset});
    }
    noc.async_read_barrier();
    // Keep BF16 in L1. Copy eight 32-bit words per face row rather than
    // widening 5120 scalar elements on the reader RISC-V. Hardware unpack
    // widens these exact BF16 values to FP32 DST before any arithmetic.
    volatile tt_l1_ptr uint32_t* raw=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch.get_write_ptr());
    volatile tt_l1_ptr uint32_t* dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dense.get_write_ptr());
    for(uint32_t tile=0;tile<160;tile++) for(uint32_t face=0;face<2;face++) {
        const uint32_t linear=tile*32+face*16;
        const uint32_t dt=linear/1024, dr=(linear%1024)/32, dc=linear%32;
        const uint32_t target=(dt*1024+(dr/16)*512+(dc/16)*256+(dr%16)*16)/2;
        const uint32_t source=(tile*2+face)*16+(user%2)*8;
        #pragma GCC unroll 8
        for(uint32_t j=0;j<8;j++) dst[target+j]=raw[source+j];
    }
    dense.push_back(5); weight.push_back(5);
}
"""

_HIDDEN_NORM_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto oa=TensorAccessorArgs<0>();
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(0),2048);
    const uint32_t user=get_arg_val<uint32_t>(1),batch=get_arg_val<uint32_t>(2);
    Noc noc; CircularBuffer dense(3),zero(6);
    // Only the last user's writer zeroes unused output rows. Face-row ranges
    // never overlap any logical user, and source/destination both align to16B.
    if(user+1==batch && batch<32) {
        zero.reserve_back(1);
        volatile tt_l1_ptr uint32_t* z=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero.get_write_ptr());
        for(uint32_t i=0;i<512;i++) z[i]=0;
        zero.push_back(1); zero.wait_front(1);
    }
    dense.wait_front(5);
    for(uint32_t tile=0;tile<160;tile++) {
        for(uint32_t face=0;face<2;face++) {
            const uint32_t linear=tile*32+face*16;
            const uint32_t dt=linear/1024,dr=(linear%1024)/32,dc=linear%32;
            const uint32_t source=(dt*1024+(dr/16)*512+(dc/16)*256+(dr%16)*16)*2;
            const uint32_t target=((user/16)*512+face*256+(user%16)*16)*2;
            noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(dense),output,32,
                {.offset_bytes=source},{.page_id=tile,.offset_bytes=target});
        }
        if(user+1==batch && batch<32) {
            for(uint32_t row=batch;row<32;) {
                const uint32_t end=(row/16+1)*16;
                for(uint32_t face=0;face<2;face++) {
                    const uint32_t target=((row/16)*512+face*256+(row%16)*16)*2;
                    noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(zero),output,(end-row)*32,
                        {.offset_bytes=0},{.page_id=tile,.offset_bytes=target});
                }
                row=end;
            }
        }
    }
    noc.async_write_barrier(); dense.pop_front(5);
    if(user+1==batch && batch<32) zero.pop_front(1);
}
"""

_HIDDEN_NORM_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    constexpr uint32_t EPS_BITS=get_compile_time_arg_val(0), RECIPROCAL_BITS=get_compile_time_arg_val(1);
    compute_kernel_hw_startup(1,1,2);
    reconfig_data_format_srca(0); copy_init(0);
    CircularBuffer x(0),gamma(1),inverse(2),out(3),zero(4);
    x.wait_front(5); gamma.wait_front(5); zero.wait_front(1);
    inverse.reserve_back(1); tile_regs_acquire();
    // Fold five dense tiles first. Every operation is FP32 SFPU, including
    // square and the two dimensional scalar reduction; no FPU/TF32 path.
    for(uint32_t i=0;i<5;i++) {
        const uint32_t work=i?2:0;
        copy_tile(0,i,work); copy_tile(0,i,1);
        mul_binary_tile_init(); mul_binary_tile(work,1,work);
        if(i) { add_binary_tile_init(); add_binary_tile(0,2,0); }
    }
    sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_COL>(0,1,1);
    sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_ROW>(0,1,1);
    binop_with_scalar_tile_init(); mul_unary_tile(0,RECIPROCAL_BITS);
    binop_with_scalar_tile_init(); add_unary_tile(0,EPS_BITS);
    rsqrt_tile_init(); rsqrt_tile(0);
    tile_regs_commit(); tile_regs_wait(); pack_tile(0,2); tile_regs_release();
    inverse.push_back(1); inverse.wait_front(1);
    out.reserve_back(5); pack_reconfig_data_format(2,3);
    for(uint32_t i=0;i<5;i++) {
        tile_regs_acquire();
        reconfig_data_format_srca(0); copy_init(0);
        copy_tile(0,i,0);
        // zero, inverse, and gamma all have the same FP32/direct-unpack format.
        reconfig_data_format_srca(4); copy_init(4);
        copy_tile(4,0,1); copy_tile(2,0,2);
        sfpu_normalize_bcast_scalar_init(); sfpu_normalize_bcast_scalar(0,1,2);
        copy_tile(1,i,1); mul_binary_tile_init(); mul_binary_tile(0,1,0);
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>(0);
        tile_regs_commit(); tile_regs_wait(); pack_tile<true>(0,3,i); tile_regs_release();
    }
    out.push_back(5); inverse.pop_front(1); x.pop_front(5); gamma.pop_front(5); zero.pop_front(1);
}
"""


def hidden_rmsnorm_sfpu(x, gamma, *, epsilon=1e-6, output=None):
    """No allocation or address replacement when an output buffer is supplied."""
    assert len(x.shape) == 3 and x.shape[0] == 1 and 1 <= x.shape[1] <= 32 and x.shape[2] == 5120
    assert x.dtype == ttnn.bfloat16 and tuple(gamma.shape) == (1, 1, 5120) and gamma.dtype == ttnn.float32
    assert epsilon > 0
    for tensor in (x, gamma):
        assert tensor.layout == ttnn.TILE_LAYOUT and tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    if output is None:
        output = ttnn.empty(
            tuple(x.shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=x.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    assert tuple(output.shape) == tuple(x.shape) and output.dtype == ttnn.bfloat16
    assert output.layout == ttnn.TILE_LAYOUT and output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert output.buffer_address() not in (x.buffer_address(), gamma.buffer_address())
    batch = x.shape[1]
    grid = x.device().compute_with_storage_grid_size()
    assert grid.x * grid.y >= batch
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(batch)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    for user, c in enumerate(coords):
        rr[c.x][c.y] = [x.buffer_address(), gamma.buffer_address(), user]
        wr[c.x][c.y] = [output.buffer_address(), user, batch]
        cr[c.x][c.y] = []
    rct = []
    for t in (x, gamma):
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct = ttnn.TensorAccessorArgs(output).get_compile_time_args()

    def bits(value):
        return struct.unpack("<I", struct.pack("<f", value))[0]

    compute = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for i in (1, 2, 4, 5):
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute.unpack_to_dest_mode = modes

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
        kernel(_HIDDEN_NORM_READER, ttnn.ReaderConfigDescriptor(), rr, rct),
        kernel(_HIDDEN_NORM_WRITER, ttnn.WriterConfigDescriptor(), wr, wct),
        kernel(_HIDDEN_NORM_COMPUTE, compute, cr, [bits(epsilon), bits(1 / 5120)]),
    ]
    sizes = {0: 5, 1: 5, 2: 1, 3: 5, 4: 1, 5: 5, 6: 1}
    cbs = []
    for i, n in sizes.items():
        bf16 = i in (0, 3, 6)
        page = 2048 if bf16 else 4096
        cbs.append(
            ttnn.CBDescriptor(
                total_size=n * page,
                core_ranges=ranges,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=i, data_format=ttnn.bfloat16 if bf16 else ttnn.float32, page_size=page
                    )
                ],
            )
        )
    ttnn.generic_op([x, gamma, output], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return output


def corrected_projection_flat_prefill(layer, x, key):
    assert key in layer.weight_corrections and len(x.shape) == 3
    batch, seq, width = x.shape
    flat = ttnn.reshape(ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG), (1, batch * seq, width))
    primary = layer._linear_raw(flat, key)
    correction = layer._linear_raw(flat, layer.weight_corrections[key])
    # Keep the original FP32-partial -> BF16-add precision boundary.
    result = layer._combine_correction(primary, correction, key)
    return ttnn.reshape(result, (batch, seq, result.shape[-1]))


def minimal_mlp_flat_prefill(layer, residual, normalized):
    assert layer.swiglu_weight is not None and len(normalized.shape) == 3
    batch, seq, width = normalized.shape
    flat = ttnn.reshape(ttnn.to_memory_config(normalized, ttnn.DRAM_MEMORY_CONFIG), (1, batch * seq, width))
    mb, kb, nb = layer.policy.get("swiglu_blocks", [4, 4, 16])
    activated = ttnn.experimental.minimal_matmul(
        flat,
        layer.swiglu_weight,
        config=ttnn.MinimalMatmulConfig(
            M_block_size=mb,
            K_block_size=kb,
            N_block_size=nb,
            subblock_h=1,
            subblock_w=4,
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
        ),
        fuse_swiglu=True,
        compute_kernel_config=layer.matmul_configs["mlp"],
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    down = layer._linear(activated, "mlp.down_proj.weight")
    down = ttnn.reshape(ttnn.to_memory_config(down, ttnn.DRAM_MEMORY_CONFIG), (batch, seq, 5120))
    return ttnn.add(residual, down)


PREFILL_NORM_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto xa=TensorAccessorArgs<0>();
    constexpr auto ga=TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0),2048);
    const auto gamma=TensorAccessor(ga,get_arg_val<uint32_t>(1),4096);
    const uint32_t start=get_arg_val<uint32_t>(2),count=get_arg_val<uint32_t>(3);
    const uint32_t seq=get_arg_val<uint32_t>(4),padded_seq=get_arg_val<uint32_t>(5);
    Noc noc; CircularBuffer dense(0),weight(1),zero(4),scratch(5);
    weight.reserve_back(5); scratch.reserve_back(5); zero.reserve_back(1);
    volatile tt_l1_ptr uint32_t* z=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero.get_write_ptr());
    for(uint32_t i=0;i<1024;i++) z[i]=0;
    zero.push_back(1);
    // Gamma is invariant across every row owned by this core; stage it once.
    for(uint32_t tile=0;tile<160;tile++) for(uint32_t face=0;face<2;face++) {
        const uint32_t linear=tile*32+face*16;
        const uint32_t dt=linear/1024,dr=(linear%1024)/32,dc=linear%32;
        const uint32_t target=(dt*1024+(dr/16)*512+(dc/16)*256+(dr%16)*16)*4;
        noc.async_read(gamma,weight,64,{.page_id=tile,.offset_bytes=face*1024},{.offset_bytes=target});
    }
    noc.async_read_barrier(); weight.push_back(5);
    for(uint32_t logical=start;logical<start+count;logical++) {
        const uint32_t physical=(logical/seq)*padded_seq+logical%seq;
        const uint32_t tile_base=(physical/32)*160,row=physical%32;
        dense.reserve_back(5);
        for(uint32_t tile=0;tile<160;tile++) for(uint32_t face=0;face<2;face++) {
            const uint32_t source=((row/16)*512+face*256+(row%16)*16)*2;
            // Match BH64-byte alignment; each BF16 face-row fetch includes its neighbor.
            noc.async_read(x,scratch,64,{.page_id=tile_base+tile,.offset_bytes=source&~63u},
                {.offset_bytes=(tile*2+face)*64});
        }
        noc.async_read_barrier();
        volatile tt_l1_ptr uint32_t* raw=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch.get_write_ptr());
        volatile tt_l1_ptr uint32_t* dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dense.get_write_ptr());
        for(uint32_t tile=0;tile<160;tile++) for(uint32_t face=0;face<2;face++) {
            const uint32_t linear=tile*32+face*16;
            const uint32_t dt=linear/1024,dr=(linear%1024)/32,dc=linear%32;
            const uint32_t target=(dt*1024+(dr/16)*512+(dc/16)*256+(dr%16)*16)/2;
            const uint32_t source=(tile*2+face)*16+(row%2)*8;
            #pragma GCC unroll 8
            for(uint32_t j=0;j<8;j++) dst[target+j]=raw[source+j];
        }
        dense.push_back(5);
    }
}
"""

PREFILL_NORM_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto oa=TensorAccessorArgs<0>();
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(0),2048);
    const uint32_t start=get_arg_val<uint32_t>(1),count=get_arg_val<uint32_t>(2);
    const uint32_t seq=get_arg_val<uint32_t>(3),padded_seq=get_arg_val<uint32_t>(4);
    Noc noc; CircularBuffer dense(3),zero(6);
    if(seq%32) {
        zero.reserve_back(1);
        volatile tt_l1_ptr uint32_t* z=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero.get_write_ptr());
        for(uint32_t i=0;i<512;i++) z[i]=0;
        zero.push_back(1); zero.wait_front(1);
    }
    for(uint32_t logical=start;logical<start+count;logical++) {
        const uint32_t token=logical%seq,physical=(logical/seq)*padded_seq+token;
        const uint32_t tile_base=(physical/32)*160,row=physical%32;
        dense.wait_front(5);
        for(uint32_t tile=0;tile<160;tile++) {
            for(uint32_t face=0;face<2;face++) {
                const uint32_t linear=tile*32+face*16;
                const uint32_t dt=linear/1024,dr=(linear%1024)/32,dc=linear%32;
                const uint32_t source=(dt*1024+(dr/16)*512+(dc/16)*256+(dr%16)*16)*2;
                const uint32_t target=((row/16)*512+face*256+(row%16)*16)*2;
                noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(dense),output,32,
                    {.offset_bytes=source},{.page_id=tile_base+tile,.offset_bytes=target});
            }
            // The last logical token in each batch owns only its unused tail rows.
            // No logical row or other batch can overlap these deterministic zeros.
            if(token+1==seq && seq%32) {
                for(uint32_t padrow=seq%32;padrow<32;) {
                    const uint32_t end=(padrow/16+1)*16;
                    for(uint32_t face=0;face<2;face++) {
                        const uint32_t target=((padrow/16)*512+face*256+(padrow%16)*16)*2;
                        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(zero),output,(end-padrow)*32,
                            {.offset_bytes=0},{.page_id=tile_base+tile,.offset_bytes=target});
                    }
                    padrow=end;
                }
            }
        }
        noc.async_write_barrier(); dense.pop_front(5);
    }
    if(seq%32) zero.pop_front(1);
}
"""

PREFILL_NORM_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    constexpr uint32_t EPS_BITS=get_compile_time_arg_val(0), RECIPROCAL_BITS=get_compile_time_arg_val(1);
    compute_kernel_hw_startup(1,1,2);
    reconfig_data_format_srca(0); copy_init(0);
    CircularBuffer x(0),gamma(1),inverse(2),out(3),zero(4);
    const uint32_t count=get_arg_val<uint32_t>(0);
    gamma.wait_front(5); zero.wait_front(1);
    for(uint32_t row=0;row<count;row++) {
    x.wait_front(5);
    reconfig_data_format_srca(0); copy_init(0);
    if(row) pack_reconfig_data_format(3,2);
    inverse.reserve_back(1); tile_regs_acquire();
    // Fold five dense tiles first. Every operation is FP32 SFPU, including
    // square and the two dimensional scalar reduction; no FPU/TF32 path.
    for(uint32_t i=0;i<5;i++) {
        const uint32_t work=i?2:0;
        copy_tile(0,i,work); copy_tile(0,i,1);
        mul_binary_tile_init(); mul_binary_tile(work,1,work);
        if(i) { add_binary_tile_init(); add_binary_tile(0,2,0); }
    }
    sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_COL>(0,1,1);
    sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_ROW>(0,1,1);
    binop_with_scalar_tile_init(); mul_unary_tile(0,RECIPROCAL_BITS);
    binop_with_scalar_tile_init(); add_unary_tile(0,EPS_BITS);
    rsqrt_tile_init(); rsqrt_tile(0);
    tile_regs_commit(); tile_regs_wait(); pack_tile(0,2); tile_regs_release();
    inverse.push_back(1); inverse.wait_front(1);
    out.reserve_back(5); pack_reconfig_data_format(2,3);
    for(uint32_t i=0;i<5;i++) {
        tile_regs_acquire();
        reconfig_data_format_srca(0); copy_init(0);
        copy_tile(0,i,0);
        // zero, inverse, and gamma all have the same FP32/direct-unpack format.
        reconfig_data_format_srca(4); copy_init(4);
        copy_tile(4,0,1); copy_tile(2,0,2);
        sfpu_normalize_bcast_scalar_init(); sfpu_normalize_bcast_scalar(0,1,2);
        copy_tile(1,i,1); mul_binary_tile_init(); mul_binary_tile(0,1,0);
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>(0);
        tile_regs_commit(); tile_regs_wait(); pack_tile<true>(0,3,i); tile_regs_release();
    }
    out.push_back(5); inverse.pop_front(1); x.pop_front(5);
    }
    gamma.pop_front(5); zero.pop_front(1);
}
"""


def _prefill_row_partition(batch, seq, available_cores, cores=None):
    assert batch >= 1 and seq >= 1 and available_cores >= 1
    assert cores is None or cores >= 1
    rows = batch * seq
    workers = min(rows, available_cores, cores or available_cores)
    result, start = [], 0
    for index in range(workers):
        count = rows // workers + (index < rows % workers)
        result.append((start, count))
        start += count
    return result


def prefill_hidden_rmsnorm_sfpu(x, gamma, *, epsilon=1e-6, output=None, cores=None):
    """One device-only op; logical [B,S,5120] and per-batch padded rows preserved."""
    assert len(x.shape) == 3 and x.shape[0] >= 1 and x.shape[1] >= 1 and x.shape[2] == 5120
    assert x.dtype == ttnn.bfloat16 and tuple(gamma.shape) == (1, 1, 5120) and gamma.dtype == ttnn.float32
    assert epsilon > 0
    for tensor in (x, gamma):
        assert tensor.layout == ttnn.TILE_LAYOUT and tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    batch, seq, _ = x.shape
    padded_seq = (seq + 31) // 32 * 32
    assert tuple(x.padded_shape) == (batch, padded_seq, 5120), x.padded_shape
    if output is None:
        output = ttnn.empty(
            tuple(x.shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=x.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    assert tuple(output.shape) == tuple(x.shape) and tuple(output.padded_shape) == tuple(x.padded_shape)
    assert output.dtype == ttnn.bfloat16 and output.layout == ttnn.TILE_LAYOUT
    assert output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert output.buffer_address() not in (x.buffer_address(), gamma.buffer_address())
    grid = x.device().compute_with_storage_grid_size()
    partition = _prefill_row_partition(batch, seq, grid.x * grid.y, cores)
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(partition))]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    for c, (start, count) in zip(coords, partition):
        rr[c.x][c.y] = [x.buffer_address(), gamma.buffer_address(), start, count, seq, padded_seq]
        wr[c.x][c.y] = [output.buffer_address(), start, count, seq, padded_seq]
        cr[c.x][c.y] = [count]
    rct = []
    for t in (x, gamma):
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct = ttnn.TensorAccessorArgs(output).get_compile_time_args()

    def bits(value):
        return struct.unpack("<I", struct.pack("<f", value))[0]

    compute = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for i in (1, 2, 4, 5):
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute.unpack_to_dest_mode = modes

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
        kernel(PREFILL_NORM_READER, ttnn.ReaderConfigDescriptor(), rr, rct),
        kernel(PREFILL_NORM_WRITER, ttnn.WriterConfigDescriptor(), wr, wct),
        kernel(PREFILL_NORM_COMPUTE, compute, cr, [bits(epsilon), bits(1 / 5120)]),
    ]
    sizes = {0: 5, 1: 5, 2: 1, 3: 5, 4: 1, 5: 5, 6: 1}
    cbs = []
    for i, n in sizes.items():
        bf16 = i in (0, 3, 6)
        page = 2048 if bf16 else 4096
        cbs.append(
            ttnn.CBDescriptor(
                total_size=n * page,
                core_ranges=ranges,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=i, data_format=ttnn.bfloat16 if bf16 else ttnn.float32, page_size=page
                    )
                ],
            )
        )
    ttnn.generic_op([x, gamma, output], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return output


_CONV_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto xa=TensorAccessorArgs<0>();
    constexpr auto ha=TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto t0a=TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    constexpr auto t1a=TensorAccessorArgs<t0a.next_compile_time_args_offset()>();
    constexpr auto t2a=TensorAccessorArgs<t1a.next_compile_time_args_offset()>();
    constexpr auto t3a=TensorAccessorArgs<t2a.next_compile_time_args_offset()>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0),2048);
    const auto history=TensorAccessor(ha,get_arg_val<uint32_t>(1),20480);
    const auto t0=TensorAccessor(t0a,get_arg_val<uint32_t>(2),20480);
    const auto t1=TensorAccessor(t1a,get_arg_val<uint32_t>(3),20480);
    const auto t2=TensorAccessor(t2a,get_arg_val<uint32_t>(4),20480);
    const auto t3=TensorAccessor(t3a,get_arg_val<uint32_t>(5),20480);
    const uint32_t start=get_arg_val<uint32_t>(6),count=get_arg_val<uint32_t>(7);
    Noc noc; CircularBuffer scratch(8); scratch.reserve_back(2);
    for(uint32_t task=start;task<start+count;task++) {
        const uint32_t user=task/10,chunk=task%10;
        for(uint32_t id=0;id<8;id++) { CircularBuffer cb(id); cb.reserve_back(1); }
        for(uint32_t row=0;row<3;row++) {
            CircularBuffer cb(row);
            noc.async_read(history,cb,2048,{.page_id=user*3+row,.offset_bytes=chunk*2048},{.offset_bytes=0});
        }
        CircularBuffer cb4(4),cb5(5),cb6(6),cb7(7);
        noc.async_read(t0,cb4,2048,{.page_id=0,.offset_bytes=chunk*2048},{.offset_bytes=0});
        noc.async_read(t1,cb5,2048,{.page_id=0,.offset_bytes=chunk*2048},{.offset_bytes=0});
        noc.async_read(t2,cb6,2048,{.page_id=0,.offset_bytes=chunk*2048},{.offset_bytes=0});
        noc.async_read(t3,cb7,2048,{.page_id=0,.offset_bytes=chunk*2048},{.offset_bytes=0});
        for(uint32_t tile=0;tile<32;tile++) for(uint32_t face=0;face<2;face++) {
            const uint32_t offset=((user/16)*512+face*256+(user%16)*16)*2;
            noc.async_read(x,scratch,64,
                {.page_id=chunk*32+tile,.offset_bytes=offset&~63u},
                {.offset_bytes=(tile*2+face)*64});
        }
        noc.async_read_barrier();
        CircularBuffer token(3);
        volatile tt_l1_ptr uint32_t* src=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch.get_write_ptr());
        volatile tt_l1_ptr uint32_t* dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(token.get_write_ptr());
        for(uint32_t row=0;row<64;row++) {
            const uint32_t from=row*16+(user%2)*8;
            #pragma GCC unroll 8
            for(uint32_t j=0;j<8;j++) dst[row*8+j]=src[from+j];
        }
        for(uint32_t id=0;id<8;id++) { CircularBuffer cb(id); cb.push_back(1); }
        // Compute owns input pops, after the writer publishes a completion token.
        // Ring reuse is blocked until this chunk's carry writes finish.
    }
}
"""

_CONV_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto ha=TensorAccessorArgs<0>();
    constexpr auto oa=TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    const auto history=TensorAccessor(ha,get_arg_val<uint32_t>(0),20480);
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(1),20480);
    const uint32_t start=get_arg_val<uint32_t>(2),count=get_arg_val<uint32_t>(3);
    Noc noc; CircularBuffer out(9),done(10);
    for(uint32_t task=start;task<start+count;task++) {
        const uint32_t user=task/10,chunk=task%10;
        out.wait_front(1);
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),output,2048,
            {},{.page_id=user,.offset_bytes=chunk*2048});
        for(uint32_t row=0;row<3;row++) {
            CircularBuffer carry(row+1); carry.wait_front(1);
            noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(carry),history,2048,
                {},{.page_id=user*3+row,.offset_bytes=chunk*2048});
        }
        noc.async_write_barrier();
        out.pop_front(1);
        done.reserve_back(1); done.push_back(1);
    }
}
"""

_CONV_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"
// Native ttnn.silu(BF16) chooses calculate_silu<false>, including its BF16
// approximation/rounding. Keep that arithmetic while addressing our FP32 DST.
// silu_tile<false> would also select BF16 DST addressing, so do not use it here.
inline void conv_bf16_silu(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,true,calculate_silu,(false,8),idst,VectorMode::RC));
}
void kernel_main() {
    const uint32_t count=get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0,9); copy_init(0);
    CircularBuffer out(9),done(10);
    for(uint32_t task=0;task<count;task++) {
        for(uint32_t id=0;id<8;id++) { CircularBuffer cb(id); cb.wait_front(1); }
        out.reserve_back(1); tile_regs_acquire();
        for(uint32_t tap=0;tap<4;tap++) {
            const uint32_t work=tap?1:0;
            copy_tile(tap,0,work); copy_tile(tap+4,0,2);
            mul_binary_tile_init(); mul_binary_tile(work,2,work);
            if(tap) { add_binary_tile_init(); add_binary_tile(0,1,0); }
        }
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>(0);
        silu_tile_init(); conv_bf16_silu(0);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0,9); tile_regs_release();
        out.push_back(1);
        // Writer has read carry from these same one-entry input CBs. Keep
        // compute as their sole consumer so its local tiles_acked stays valid.
        done.wait_front(1);
        for(uint32_t id=0;id<8;id++) { CircularBuffer cb(id); cb.pop_front(1); }
        done.pop_front(1);
    }
}
"""


def conv_decode_sfpu(token, history, taps, *, output=None, cores=None):
    """TILE compact token -> packed RM output, stable RM carry allocation.

    Supports only the validated FP32 sum -> BF16 -> native BF16 SiLU policy.
    Current token is raw projection output; history stores three raw tokens.
    """
    batch = history.shape[0]
    assert 1 <= batch <= 32 and tuple(history.shape) == (batch, 3, 10240)
    assert tuple(token.shape) == (1, batch, 10240)
    assert token.layout == ttnn.TILE_LAYOUT and history.layout == ttnn.ROW_MAJOR_LAYOUT
    assert len(taps) == 4
    for tap in taps:
        assert tuple(tap.shape) == (1, 1, 10240) and tap.layout == ttnn.ROW_MAJOR_LAYOUT
    inputs = [token, history, *taps]
    for tensor in inputs:
        assert tensor.dtype == ttnn.bfloat16 and tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    if output is None:
        output = ttnn.empty(
            (batch, 1, 10240),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=token.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    assert tuple(output.shape) == (batch, 1, 10240) and output.dtype == ttnn.bfloat16
    assert output.layout == ttnn.ROW_MAJOR_LAYOUT and output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert output.buffer_address() not in [t.buffer_address() for t in inputs]
    grid = token.device().compute_with_storage_grid_size()
    tasks = batch * 10
    ncores = min(tasks, cores or grid.x * grid.y)
    assert 1 <= ncores <= grid.x * grid.y
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncores)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    start = 0
    for i, c in enumerate(coords):
        count = tasks // ncores + (i < tasks % ncores)
        rr[c.x][c.y] = [t.buffer_address() for t in inputs] + [start, count]
        wr[c.x][c.y] = [history.buffer_address(), output.buffer_address(), start, count]
        cr[c.x][c.y] = [count]
        start += count
    rct = []
    wct = []
    for t in inputs:
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    for t in (history, output):
        wct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

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
    kernels = [
        kernel(_CONV_READER, ttnn.ReaderConfigDescriptor(), rr, rct),
        kernel(_CONV_WRITER, ttnn.WriterConfigDescriptor(), wr, wct),
        kernel(_CONV_COMPUTE, config, cr, []),
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=(2 if i == 8 else 1) * 2048,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for i in range(11)
    ]
    ttnn.generic_op(inputs + [output], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return output


_GDN_FUSE_ACCUMULATOR = True
_GDN_HOIST_DELTA = True


def _compact_fp32_compute_config(state_bf16=False):
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True, math_approx_mode=False
    )
    # FP32 DEST alone still allows Float32 -> TF32 through source registers.
    # Explicit direct unpack is required for every FP32 CB copied into DST.
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for index in range(19):
        modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    if state_bf16:
        modes[5] = modes[11] = ttnn.UnpackToDestMode.Default
    config.unpack_to_dest_mode = modes
    return config


_GDN_COMPACT_READER = r"""
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
    const auto q=TensorAccessor(qa,get_arg_val<uint32_t>(0),PACKED?20480:((NORMALIZED || ROW_MAJOR)?4096:2048));
    const auto k=TensorAccessor(ka,get_arg_val<uint32_t>(1),PACKED?20480:((NORMALIZED || ROW_MAJOR)?4096:2048));
    const auto v=TensorAccessor(va,get_arg_val<uint32_t>(2),PACKED?20480:(ROW_MAJOR?12288:2048));
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
        const uint32_t page=user*2+head/32;
        const uint32_t offset=((head%32)/16)*256*element_bytes;
        noc.async_read(a,scratch,16*element_bytes,{.page_id=page,.offset_bytes=offset},{.offset_bytes=dst});
    };
    for(uint32_t h=start;h<start+count;h++) {
        const uint32_t user=h/48,hv=h%48,hk=hv/3;
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
            noc.async_read(k,scratch,256,{.page_id=user,.offset_bytes=(PACKED?4096:0)+hk*256},{.offset_bytes=512});
            noc.async_read(v,scratch,256,{.page_id=user,.offset_bytes=(PACKED?8192:0)+hv*256},{.offset_bytes=1024});
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

_GDN_COMPACT_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t STATE_BYTES=get_compile_time_arg_val(0);
    constexpr auto sa=TensorAccessorArgs<1>();
    constexpr auto oa=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto s=TensorAccessor(sa,get_arg_val<uint32_t>(0),STATE_BYTES);
    const auto o=TensorAccessor(oa,get_arg_val<uint32_t>(1),4096);
    constexpr uint32_t BH=get_compile_time_arg_val(oa.next_compile_time_args_offset());
    const uint32_t start=get_arg_val<uint32_t>(2),count=get_arg_val<uint32_t>(3);
    Noc noc; CircularBuffer updated(11),out(12);
    for(uint32_t h=start;h<start+count;h++) {
        updated.wait_front(16); out.wait_front(4);
        // Waiting for output guarantees compute finished reading updated CB.
        for(uint32_t i=0;i<16;i++)
            noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(updated),s,STATE_BYTES,{.offset_bytes=i*STATE_BYTES},{.page_id=h*16+i});
        for(uint32_t i=0;i<4;i++) for(uint32_t face=0;face<2;face++) {
            const uint32_t row=h%32;
            const uint32_t dest=((row/16)*512+face*256+(row%16)*16)*4;
            noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),o,64,
                {.offset_bytes=i*4096+face*1024},
                {.page_id=(h/32)*4+i,.offset_bytes=dest});
        }
        if(h+1==BH && BH%32) {
            // Preserve deterministic zero padding for non-tile-aligned head count.
            // Row1 in the reduced output tile is zero by the reduction pack mask.
            for(uint32_t row=BH%32;row<32;row++) for(uint32_t i=0;i<4;i++) for(uint32_t face=0;face<2;face++) {
                const uint32_t dest=((row/16)*512+face*256+(row%16)*16)*4;
                noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),o,64,
                    {.offset_bytes=i*4096+64+face*1024},
                    {.page_id=(h/32)*4+i,.offset_bytes=dest});
            }
        }
        noc.async_write_barrier(); updated.pop_front(16); out.pop_front(4);
    }
}
"""

_GDN_COMPACT_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reduce.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/transpose.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"


// Stage-local precise FP32 specialization of native scalar normalization:
// mean is known zero, so eliminate its unpack/broadcast and32 scalar adds.
namespace ckernel {
#ifdef TRISC_MATH
namespace sfpu {
inline void _gdn_mul_scalar_(uint32_t data, uint32_t scalar, uint32_t output) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    _broadcast_scalar_from_dest_(scalar*DEST_TILE_SIZE_RAW);
    TTI_SFPMOV(0,LREG_BCAST,LREG_SCALAR_INV_STD,0);
    for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {
        const uint32_t left=half*FACE2_BASE+band*ROW_BAND_STRIDE;
        const uint32_t right=left+FACE1_BASE;
        const uint32_t db=data*DEST_TILE_SIZE_RAW, ob=output*DEST_TILE_SIZE_RAW;
        TT_SFPLOAD(LREG_DATA0,IM,ADDR_MOD_7,db+left);
        TT_SFPLOAD(LREG_DATA1,IM,ADDR_MOD_7,db+left+ODD_COLS_OFFSET);
        TT_SFPLOAD(LREG_DATA2,IM,ADDR_MOD_7,db+right);
        TT_SFPLOAD(LREG_DATA3,IM,ADDR_MOD_7,db+right+ODD_COLS_OFFSET);
        TTI_SFPMUL(LREG_DATA0,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA0,0);
        TTI_SFPMUL(LREG_DATA1,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA1,0);
        TTI_SFPMUL(LREG_DATA2,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA2,0);
        TTI_SFPMUL(LREG_DATA3,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA3,0);
        TT_SFPSTORE(LREG_DATA0,IM,ADDR_MOD_7,ob+left);
        TT_SFPSTORE(LREG_DATA1,IM,ADDR_MOD_7,ob+left+ODD_COLS_OFFSET);
        TT_SFPSTORE(LREG_DATA2,IM,ADDR_MOD_7,ob+right);
        TT_SFPSTORE(LREG_DATA3,IM,ADDR_MOD_7,ob+right+ODD_COLS_OFFSET);
    }
}

// Replicate row0 without loading a tile of ones and multiplying by it.
// Native row-broadcast's SFPTRANSP mapping is retained exactly.
inline void _gdn_copy_row_(uint32_t data,uint32_t row,uint32_t output) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    const uint32_t rb=row*DEST_TILE_SIZE_RAW, ob=output*DEST_TILE_SIZE_RAW;
    TT_SFPLOAD(p_sfpu::LREG4,IM,ADDR_MOD_7,rb+COL_GROUP_OFFSETS[0]);
    TT_SFPLOAD(p_sfpu::LREG5,IM,ADDR_MOD_7,rb+COL_GROUP_OFFSETS[1]);
    TT_SFPLOAD(p_sfpu::LREG6,IM,ADDR_MOD_7,rb+COL_GROUP_OFFSETS[2]);
    TT_SFPLOAD(p_sfpu::LREG7,IM,ADDR_MOD_7,rb+COL_GROUP_OFFSETS[3]);
    TTI_SFPNOP;
    TTI_SFPTRANSP(0,0,0,0);
    TTI_SFPMOV(0,p_sfpu::LREG4,p_sfpu::LREG0,0);
    TTI_SFPMOV(0,p_sfpu::LREG4,p_sfpu::LREG1,0);
    TTI_SFPMOV(0,p_sfpu::LREG4,p_sfpu::LREG2,0);
    TTI_SFPMOV(0,p_sfpu::LREG4,p_sfpu::LREG3,0);
    TTI_SFPTRANSP(0,0,0,0);
    for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {
        const uint32_t base=ob+half*FACE2_BASE+band*ROW_BAND_STRIDE;
        TT_SFPSTORE(p_sfpu::LREG0,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[0]);
        TT_SFPSTORE(p_sfpu::LREG1,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[1]);
        TT_SFPSTORE(p_sfpu::LREG2,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[2]);
        TT_SFPSTORE(p_sfpu::LREG3,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[3]);
    }
}


// Preserve TWO rounding steps: product=delta*K, then updated=product+state.
// This deliberately uses SFPMUL followed by SFPADD, never SFPMAD/FMA.
inline void _gdn_col_mul_add_(uint32_t data,uint32_t column,uint32_t output,uint32_t residual) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    const uint32_t db=data*DEST_TILE_SIZE_RAW, cb=column*DEST_TILE_SIZE_RAW;
    const uint32_t ob=output*DEST_TILE_SIZE_RAW, rb=residual*DEST_TILE_SIZE_RAW;
    for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {
        const uint32_t left=half*FACE2_BASE+band*ROW_BAND_STRIDE;
        const uint32_t right=left+FACE1_BASE;
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,cb+left);
        lltt::replay(REPLAY_SLOT_BROADCAST,REPLAY_LEN_BROADCAST);
        _broadcast_stage3_with_data_prefetch_(db+left,db+left+ODD_COLS_OFFSET,db+right,db+right+ODD_COLS_OFFSET);
        TTI_SFPMUL(LREG_DATA0,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA0,0);
        TTI_SFPMUL(LREG_DATA1,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA1,0);
        TTI_SFPMUL(LREG_DATA2,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA2,0);
        TTI_SFPMUL(LREG_DATA3,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA3,0);
        // Broadcast is complete; its two scratch LREGs can carry residuals.
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,rb+left);
        TT_SFPLOAD(LREG_TMP,IM,ADDR_MOD_7,rb+left+ODD_COLS_OFFSET);
        TTI_SFPNOP;
        TTI_SFPADD(LREG_DATA0,p_sfpu::LCONST_1,LREG_BCAST,LREG_DATA0,0);
        TTI_SFPADD(LREG_DATA1,p_sfpu::LCONST_1,LREG_TMP,LREG_DATA1,0);
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,rb+right);
        TT_SFPLOAD(LREG_TMP,IM,ADDR_MOD_7,rb+right+ODD_COLS_OFFSET);
        TTI_SFPNOP;
        TTI_SFPADD(LREG_DATA2,p_sfpu::LCONST_1,LREG_BCAST,LREG_DATA2,0);
        TTI_SFPADD(LREG_DATA3,p_sfpu::LCONST_1,LREG_TMP,LREG_DATA3,0);
        TT_SFPSTORE(LREG_DATA0,IM,ADDR_MOD_7,ob+left);
        TT_SFPSTORE(LREG_DATA1,IM,ADDR_MOD_7,ob+left+ODD_COLS_OFFSET);
        TT_SFPSTORE(LREG_DATA2,IM,ADDR_MOD_7,ob+right);
        TT_SFPSTORE(LREG_DATA3,IM,ADDR_MOD_7,ob+right+ODD_COLS_OFFSET);
    }
}

} // namespace sfpu
#endif
ALWI void gdn_mul_scalar(uint32_t data,uint32_t scalar) {
    sfpu_normalize_bcast_scalar_init();
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_mul_scalar_,data,scalar,data,VectorMode::None)));
}

ALWI void gdn_copy_row(uint32_t data) {
    sfpu_bcast_row_init();
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_copy_row_,data,data,data,VectorMode::None)));
}


ALWI void gdn_col_mul_add(uint32_t data,uint32_t column,uint32_t residual) {
    sfpu_bcast_col_init();
    MATH((::ckernel::_sfpu_binary_check_<DST_SYNC_MODE>(data,residual,data,VectorMode::None),
        SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_col_mul_add_,data,column,data,VectorMode::None,residual)));
}

ALWI void gdn_col_mul_add_to(uint32_t data,uint32_t column,uint32_t residual,uint32_t output) {
    sfpu_bcast_col_init();
    MATH((::ckernel::_sfpu_binary_check_<DST_SYNC_MODE>(data,residual,output,VectorMode::None),
        SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_col_mul_add_,data,column,output,VectorMode::None,residual)));
}


// Native broadcast implementation supports distinct input/output registers;
// use that directly instead of copying a retained state before multiplying it.
ALWI void gdn_col_mul_to(uint32_t data,uint32_t column,uint32_t output) {
    sfpu_bcast_col_init();
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,
        _calculate_sfpu_binary_bcast_full_tile_,(ckernel::BinaryOp::MUL,ckernel::BroadcastType::COL),
        data,column,output,VectorMode::None)));
}
ALWI void gdn_copy_row_to(uint32_t row,uint32_t output) {
    sfpu_bcast_row_init();
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_copy_row_,row,row,output,VectorMode::None)));
}

} // namespace ckernel

void load(uint32_t cb,uint32_t tile,uint32_t dst) {
    copy_tile(cb,tile,dst);
}
void finish(uint32_t cb,uint32_t tile) {
    tile_regs_commit(); tile_regs_wait(); pack_tile<true>(0,cb,tile); tile_regs_release();
}
// Exactly the accurate FP32 reduction topology: fold the four H tiles with
// SFPU addition, then SFPU reduce their32 rows into row0. No TF32 FPU path.
void contract(uint32_t matrix,uint32_t column,uint32_t result) {
    CircularBuffer dst(result); dst.reserve_back(4);
    PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_COL,PackMode::Default>(result)));
    for(uint32_t n=0;n<4;n++) {
        tile_regs_acquire();
        for(uint32_t m=0;m<4;m++) {
            const uint32_t work=m==0 ? 0 : 2;
            load(matrix,m*4+n,work); load(column,m,1);
            sfpu_mul_bcast_col_init(); sfpu_mul_bcast_col(work,1);
            if(m) { add_binary_tile_init(); add_binary_tile(0,2,0); }
        }
        sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_COL>(0,1,1);
        finish(result,n);
    }
    PACK((llk_pack_reduce_mask_clear())); dst.push_back(4);
}

// Raw mode matches external FP32 square/sum(+epsilon)/rsqrt/multiply, then
// separate Q scaling. Normalized mode is a pure transpose control.
void prepare_vector(uint32_t input,uint32_t output,bool is_q) {
    constexpr bool NORMALIZED=get_compile_time_arg_val(0);
    CircularBuffer in(input),col(output),norm(16),rows(17);
    in.wait_front(4);
    copy_init(input);
    uint32_t transposed_source=input;
    if constexpr(!NORMALIZED) {
        norm.reserve_back(1); tile_regs_acquire();
        for(uint32_t i=0;i<4;i++) {
            const uint32_t work=i?2:0;
            load(input,i,work); load(input,i,1);
            mul_binary_tile_init(); mul_binary_tile(work,1,work);
            if(i) { add_binary_tile_init(); add_binary_tile(0,2,0); }
        }
        sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_ROW>(0,1,1);
        // Float32 encodings of1e-6 and128**-0.5. Scaling remains separate.
        binop_with_scalar_tile_init(); add_unary_tile(0,0x358637bdu);
        rsqrt_tile_init(); rsqrt_tile(0);
        finish(16,0); norm.push_back(1); norm.wait_front(1);
        rows.reserve_back(4);
        for(uint32_t i=0;i<4;i++) {
            tile_regs_acquire(); load(input,i,0); load(16,0,1);
            gdn_mul_scalar(0,1);
            if(is_q) { binop_with_scalar_tile_init(); mul_unary_tile(0,0x3db504f3u); }
            finish(17,i);
        }
        rows.push_back(4); rows.wait_front(4); norm.pop_front(1);
        transposed_source=17;
    }
    col.reserve_back(4); transpose_init(transposed_source);
    for(uint32_t i=0;i<4;i++) {
        tile_regs_acquire(); transpose_tile(transposed_source,i,0); finish(output,i);
    }
    col.push_back(4); in.pop_front(4);
    if constexpr(!NORMALIZED) rows.pop_front(4);
}


// Per value-column full-DST allocation: state0..3, work4, Q/K5,
// accumulator/delta6, decay/beta7. No retained state crosses a release.
void kernel_main() {
#ifdef TRISC_MATH
    static_assert(get_dest_max_tiles<DST_SYNC_MODE,true,DstTileShape::Tile32x32>()>=8,
                  "Retained GDN needs eight FP32 DST tiles and full synchronization");
#endif
    const uint32_t count=get_arg_val<uint32_t>(0);
    constexpr bool STATE_BF16=get_compile_time_arg_val(1);
    constexpr bool GROUP_QK=get_compile_time_arg_val(2);
    constexpr bool FUSE_ACCUMULATOR=get_compile_time_arg_val(3);
    constexpr bool HOIST_DELTA=get_compile_time_arg_val(4);
    compute_kernel_hw_startup(STATE_BF16?3:5,6,12);
    copy_init(STATE_BF16?3:5);
    CircularBuffer q(13),k(14),v(2),raw_g(3),decay(18),beta(4),state(5),zero(6),one(7);
    CircularBuffer updated(11),out(12),raw_q(0),raw_k(1);
    zero.wait_front(1); one.wait_front(1);
    for(uint32_t head=0;head<count;head++) {
        state.wait_front(16); raw_g.wait_front(1);
        decay.reserve_back(1); tile_regs_acquire(); load(3,0,0);
        exp_tile_init<false>(); exp_tile<false>(0); finish(18,0);
        decay.push_back(1); decay.wait_front(1); raw_g.pop_front(1);
        // Host partitions only complete aligned 3-head groups in grouped mode.
        // CB13/14 retain exactly the first head's prepared Q/K across its peers.
        if (!GROUP_QK || head%3==0) {
            prepare_vector(0,13,true); prepare_vector(1,14,false);
        } else {
            // Reader's rings remain one head deep: consume every duplicate input
            // to permit it to fetch the next state/value/gate without stalling.
            raw_q.wait_front(4); raw_k.wait_front(4);
            raw_q.pop_front(4); raw_k.pop_front(4);
        }
        copy_init(STATE_BF16?3:5);
        q.wait_front(4); k.wait_front(4); v.wait_front(4); beta.wait_front(1);
        updated.reserve_back(16); out.reserve_back(4);
        for(uint32_t n=0;n<4;n++) {
            tile_regs_acquire();
            load(18,0,7);
            if constexpr(STATE_BF16) {
                reconfig_data_format_srca(5); copy_init(5);
            }
            for(uint32_t m=0;m<4;m++) {
                load(5,m*4+n,m);
                gdn_mul_scalar(m,7);
            }
            if constexpr(STATE_BF16) {
                reconfig_data_format_srca(3); copy_init(3);
            }
            // Preserve the baseline four-K-tile accumulation order exactly.
            for(uint32_t m=0;m<4;m++) {
                load(14,m,5);
                if constexpr(FUSE_ACCUMULATOR) {
                    if(m) {
                        // Exact separate FP32 product then add into accumulator.
                        // Skip the full temporary-tile store/reload/add pass.
                        gdn_col_mul_add_to(m,5,6,6);
                    } else { gdn_col_mul_to(m,5,6); }
                } else {
                    const uint32_t work=m?4:6;
                    gdn_col_mul_to(m,5,work);
                    if(m) { add_binary_tile_init(); add_binary_tile(6,4,6); }
                }
            }
            sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
            sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_COL>(6,1,1);
            load(2,n,4);
            sub_binary_tile_init(); sub_binary_tile(4,6,6);
            load(4,0,7); gdn_mul_scalar(6,7);
            // State update writes only DST0..3, leaving broadcast delta DST4
            // unchanged. Reuse one broadcast across all four K-row tiles.
            if constexpr(HOIST_DELTA) gdn_copy_row_to(6,4);
            for(uint32_t m=0;m<4;m++) {
                if constexpr(!HOIST_DELTA) gdn_copy_row_to(6,4);
                load(14,m,5);
                // Separate FP32 multiply/add; write directly over retained state.
                gdn_col_mul_add_to(4,5,m,m);
            }
            for(uint32_t m=0;m<4;m++) {
                load(13,m,5);
                if constexpr(FUSE_ACCUMULATOR) {
                    if(m) {
                        // Exact separate FP32 product then add into accumulator.
                        // Skip the full temporary-tile store/reload/add pass.
                        gdn_col_mul_add_to(m,5,6,6);
                    } else { gdn_col_mul_to(m,5,6); }
                } else {
                    const uint32_t work=m?4:6;
                    gdn_col_mul_to(m,5,work);
                    if(m) { add_binary_tile_init(); add_binary_tile(6,4,6); }
                }
            }
            sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
            sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_COL>(6,1,1);
            if constexpr(STATE_BF16) {
                // Match TTNN Float32->BF16 RNE exactly, after current readout.
                typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();
                for(uint32_t m=0;m<4;m++)
                    typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>(m);
            }
            tile_regs_commit(); tile_regs_wait();
            PACK((llk_pack_reduce_mask_clear()));
            // Q readout is already complete in FP32 DST6. Only persistence rounds.
            if constexpr(STATE_BF16) pack_reconfig_data_format(12,11);
            for(uint32_t m=0;m<4;m++) pack_tile<true>(m,11,m*4+n);
            if constexpr(STATE_BF16) pack_reconfig_data_format(11,12);
            PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_COL,PackMode::Default>(12)));
            pack_tile<true>(6,12,n);
            PACK((llk_pack_reduce_mask_clear()));
            tile_regs_release();
        }
        updated.push_back(16); out.push_back(4);
        state.pop_front(16); decay.pop_front(1); v.pop_front(4); beta.pop_front(1);
        if (!GROUP_QK || head%3==2) {
            q.pop_front(4); k.pop_front(4);
        }
    }
    zero.pop_front(1); one.pop_front(1);
}
"""


def _head_partition(batch, available_cores, cores=None, grouped_qk=True):
    """Pure host partition: (grouped, [(global_head_start, head_count), ...])."""
    assert batch >= 1 and available_cores >= 1
    assert cores is None or cores >= 1
    grouped = bool(grouped_qk and batch >= 3)
    unit = 3 if grouped else 1
    units = batch * 48 // unit
    ncores = min(units, available_cores, cores or available_cores)
    result = []
    start = 0
    for index in range(ncores):
        count = unit * (units // ncores + (index < units % ncores))
        result.append((start, count))
        start += count
    assert start == batch * 48
    return grouped, result


def _recurrent_sfpu_compact(
    q, k, v, g, beta, state, *, normalized=False, packed_qkv=False, output=None, cores=None, grouped_qk=True
):
    """Accept raw RM[B,1,C] directly, or the original TILE interface."""
    row_major = q.layout == ttnn.ROW_MAJOR_LAYOUT
    assert not (row_major and normalized)
    assert not packed_qkv or row_major
    assert state.dtype in (ttnn.float32, ttnn.bfloat16)
    state_bf16 = state.dtype == ttnn.bfloat16
    state_bytes = 2048 if state_bf16 else 4096
    batch = state.shape[0] // 48
    assert tuple(state.shape) == (batch * 48, 128, 128)
    assert q.dtype == k.dtype == (ttnn.float32 if normalized else ttnn.bfloat16)
    assert v.dtype == ttnn.bfloat16
    assert tuple(q.shape) == tuple(k.shape)
    if packed_qkv:
        assert tuple(q.shape) == tuple(k.shape) == tuple(v.shape) == (batch, 1, 10240)
        assert q.buffer_address() == k.buffer_address() == v.buffer_address()
    elif normalized:
        assert tuple(q.shape) == (batch, 16, 128)
    else:
        assert q.shape[0] == batch and q.shape[-1] == 2048 and q.shape[1] in (1, 32)
    if not packed_qkv:
        assert v.shape[0] == batch and v.shape[-1] == 6144 and v.shape[1] in (1, 32)
    assert tuple(g.shape) == tuple(beta.shape) == (batch, 1, 48)
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
            (1, batch * 48, 128),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=state.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    assert tuple(output.shape) == (1, batch * 48, 128)
    grid = state.device().compute_with_storage_grid_size()
    group_qk, partition = _head_partition(batch, grid.x * grid.y, cores, grouped_qk)
    ncores = len(partition)
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncores)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    for c, (start, count) in zip(coords, partition):
        rr[c.x][c.y] = [t.buffer_address() for t in inputs] + [start, count]
        wr[c.x][c.y] = [state.buffer_address(), output.buffer_address(), start, count]
        cr[c.x][c.y] = [count]
    # Raw q/k/v paddedS32 occupies one tile row per batch.
    rct = [
        int(normalized),
        int(g.dtype == ttnn.float32),
        int(beta.dtype == ttnn.float32),
        64,
        192,
        int(row_major),
        int(packed_qkv),
        state_bytes,
    ]
    for t in inputs:
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct = [state_bytes]
    for t in (state, output):
        wct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct.append(batch * 48)

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
        kernel(_GDN_COMPACT_READER, ttnn.ReaderConfigDescriptor(), rr, rct),
        kernel(_GDN_COMPACT_WRITER, ttnn.WriterConfigDescriptor(), wr, wct),
        kernel(
            _GDN_COMPACT_COMPUTE,
            _compact_fp32_compute_config(state_bf16),
            cr,
            [int(normalized), int(state_bf16), int(group_qk), int(_GDN_FUSE_ACCUMULATOR), int(_GDN_HOIST_DELTA)],
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
        13: 4,
        14: 4,
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


def corrected_projection_chunk_add_prefill(layer, x, key):
    assert key in layer.weight_corrections and len(x.shape) == 3
    batch, seq, width = x.shape
    flat = ttnn.reshape(ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG), (1, batch * seq, width))
    correction_key = layer.weight_corrections[key]

    # Reproduce _linear_raw's row partitions exactly, including its wider-N cap.
    # A mismatched geometry must retain the baseline's independent partitions.
    def cap(k):
        return 256 if layer.weights[k].shape[-1] > 20000 else layer.policy.get("matmul_row_cap", 512)

    primary_cap, correction_cap = cap(key), cap(correction_key)
    chunked = (
        key in layer.projection_info
        and correction_key in layer.projection_info
        and primary_cap == correction_cap
        and batch * seq > primary_cap
    )
    if chunked:
        parts = []
        for start in range(0, batch * seq, primary_cap):
            tile_rows = flat[:, start : min(start + primary_cap, batch * seq)]
            primary = layer._linear_raw(tile_rows, key)
            correction = layer._linear_raw(tile_rows, correction_key)
            # Same FP32 partials and BF16 rounding boundary as the baseline.
            parts.append(
                ttnn.to_memory_config(layer._combine_correction(primary, correction, key), ttnn.DRAM_MEMORY_CONFIG)
            )
        result = ttnn.concat(parts, dim=1)
    else:
        primary = layer._linear_raw(flat, key)
        correction = layer._linear_raw(flat, correction_key)
        result = layer._combine_correction(primary, correction, key)
    if layer.policy.get("prefill_unflatten_packed", False) and key == "packed_gdn" and batch > 1 and seq % 32:
        return prefill_unflatten_bf16(
            ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG),
            batch,
            seq,
            cores=layer.policy.get("prefill_unflatten_cores"),
        )
    return ttnn.reshape(result, (batch, seq, result.shape[-1]))


_GROUPED96_GDN_FUSE_ACCUMULATOR = True
_GROUPED96_GDN_HOIST_DELTA = True
_GROUPED96_GDN_HOIST_SCALAR_INIT = True
_GROUPED96_GDN_HOIST_COL_INIT = True


def _grouped96_compact_fp32_compute_config(state_bf16=False):
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True, math_approx_mode=False
    )
    # FP32 DEST alone still allows Float32 -> TF32 through source registers.
    # Explicit direct unpack is required for every FP32 CB copied into DST.
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for index in range(19):
        modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    if state_bf16:
        modes[5] = modes[11] = ttnn.UnpackToDestMode.Default
    config.unpack_to_dest_mode = modes
    return config


_GROUPED96_GDN_COMPACT_READER = r"""
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
    const auto q=TensorAccessor(qa,get_arg_val<uint32_t>(0),PACKED?20480:((NORMALIZED || ROW_MAJOR)?4096:2048));
    const auto k=TensorAccessor(ka,get_arg_val<uint32_t>(1),PACKED?20480:((NORMALIZED || ROW_MAJOR)?4096:2048));
    const auto v=TensorAccessor(va,get_arg_val<uint32_t>(2),PACKED?20480:(ROW_MAJOR?12288:2048));
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
        const uint32_t page=user*2+head/32;
        const uint32_t offset=((head%32)/16)*256*element_bytes;
        noc.async_read(a,scratch,16*element_bytes,{.page_id=page,.offset_bytes=offset},{.offset_bytes=dst});
    };
    for(uint32_t h=start;h<start+count;h++) {
        const uint32_t user=h/48,hv=h%48,hk=hv/3;
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
            noc.async_read(k,scratch,256,{.page_id=user,.offset_bytes=(PACKED?4096:0)+hk*256},{.offset_bytes=512});
            noc.async_read(v,scratch,256,{.page_id=user,.offset_bytes=(PACKED?8192:0)+hv*256},{.offset_bytes=1024});
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

_GROUPED96_GDN_COMPACT_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t STATE_BYTES=get_compile_time_arg_val(0);
    constexpr auto sa=TensorAccessorArgs<1>();
    constexpr auto oa=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto s=TensorAccessor(sa,get_arg_val<uint32_t>(0),STATE_BYTES);
    const auto o=TensorAccessor(oa,get_arg_val<uint32_t>(1),4096);
    constexpr uint32_t BH=get_compile_time_arg_val(oa.next_compile_time_args_offset());
    const uint32_t start=get_arg_val<uint32_t>(2),count=get_arg_val<uint32_t>(3);
    Noc noc; CircularBuffer updated(11),out(12);
    for(uint32_t h=start;h<start+count;h++) {
        updated.wait_front(16); out.wait_front(4);
        // Waiting for output guarantees compute finished reading updated CB.
        for(uint32_t i=0;i<16;i++)
            noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(updated),s,STATE_BYTES,{.offset_bytes=i*STATE_BYTES},{.page_id=h*16+i});
        for(uint32_t i=0;i<4;i++) for(uint32_t face=0;face<2;face++) {
            const uint32_t row=h%32;
            const uint32_t dest=((row/16)*512+face*256+(row%16)*16)*4;
            noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),o,64,
                {.offset_bytes=i*4096+face*1024},
                {.page_id=(h/32)*4+i,.offset_bytes=dest});
        }
        if(h+1==BH && BH%32) {
            // Preserve deterministic zero padding for non-tile-aligned head count.
            // Row1 in the reduced output tile is zero by the reduction pack mask.
            for(uint32_t row=BH%32;row<32;row++) for(uint32_t i=0;i<4;i++) for(uint32_t face=0;face<2;face++) {
                const uint32_t dest=((row/16)*512+face*256+(row%16)*16)*4;
                noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),o,64,
                    {.offset_bytes=i*4096+64+face*1024},
                    {.page_id=(h/32)*4+i,.offset_bytes=dest});
            }
        }
        noc.async_write_barrier(); updated.pop_front(16); out.pop_front(4);
    }
}
"""

_GROUPED96_GDN_COMPACT_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reduce.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/transpose.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"


// Stage-local precise FP32 specialization of native scalar normalization:
// mean is known zero, so eliminate its unpack/broadcast and32 scalar adds.
namespace ckernel {
#ifdef TRISC_MATH
namespace sfpu {
inline void _gdn_mul_scalar_(uint32_t data, uint32_t scalar, uint32_t output) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    _broadcast_scalar_from_dest_(scalar*DEST_TILE_SIZE_RAW);
    TTI_SFPMOV(0,LREG_BCAST,LREG_SCALAR_INV_STD,0);
    for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {
        const uint32_t left=half*FACE2_BASE+band*ROW_BAND_STRIDE;
        const uint32_t right=left+FACE1_BASE;
        const uint32_t db=data*DEST_TILE_SIZE_RAW, ob=output*DEST_TILE_SIZE_RAW;
        TT_SFPLOAD(LREG_DATA0,IM,ADDR_MOD_7,db+left);
        TT_SFPLOAD(LREG_DATA1,IM,ADDR_MOD_7,db+left+ODD_COLS_OFFSET);
        TT_SFPLOAD(LREG_DATA2,IM,ADDR_MOD_7,db+right);
        TT_SFPLOAD(LREG_DATA3,IM,ADDR_MOD_7,db+right+ODD_COLS_OFFSET);
        TTI_SFPMUL(LREG_DATA0,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA0,0);
        TTI_SFPMUL(LREG_DATA1,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA1,0);
        TTI_SFPMUL(LREG_DATA2,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA2,0);
        TTI_SFPMUL(LREG_DATA3,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA3,0);
        TT_SFPSTORE(LREG_DATA0,IM,ADDR_MOD_7,ob+left);
        TT_SFPSTORE(LREG_DATA1,IM,ADDR_MOD_7,ob+left+ODD_COLS_OFFSET);
        TT_SFPSTORE(LREG_DATA2,IM,ADDR_MOD_7,ob+right);
        TT_SFPSTORE(LREG_DATA3,IM,ADDR_MOD_7,ob+right+ODD_COLS_OFFSET);
    }
}

// Replicate row0 without loading a tile of ones and multiplying by it.
// Native row-broadcast's SFPTRANSP mapping is retained exactly.
inline void _gdn_copy_row_(uint32_t data,uint32_t row,uint32_t output) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    const uint32_t rb=row*DEST_TILE_SIZE_RAW, ob=output*DEST_TILE_SIZE_RAW;
    TT_SFPLOAD(p_sfpu::LREG4,IM,ADDR_MOD_7,rb+COL_GROUP_OFFSETS[0]);
    TT_SFPLOAD(p_sfpu::LREG5,IM,ADDR_MOD_7,rb+COL_GROUP_OFFSETS[1]);
    TT_SFPLOAD(p_sfpu::LREG6,IM,ADDR_MOD_7,rb+COL_GROUP_OFFSETS[2]);
    TT_SFPLOAD(p_sfpu::LREG7,IM,ADDR_MOD_7,rb+COL_GROUP_OFFSETS[3]);
    TTI_SFPNOP;
    TTI_SFPTRANSP(0,0,0,0);
    TTI_SFPMOV(0,p_sfpu::LREG4,p_sfpu::LREG0,0);
    TTI_SFPMOV(0,p_sfpu::LREG4,p_sfpu::LREG1,0);
    TTI_SFPMOV(0,p_sfpu::LREG4,p_sfpu::LREG2,0);
    TTI_SFPMOV(0,p_sfpu::LREG4,p_sfpu::LREG3,0);
    TTI_SFPTRANSP(0,0,0,0);
    for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {
        const uint32_t base=ob+half*FACE2_BASE+band*ROW_BAND_STRIDE;
        TT_SFPSTORE(p_sfpu::LREG0,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[0]);
        TT_SFPSTORE(p_sfpu::LREG1,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[1]);
        TT_SFPSTORE(p_sfpu::LREG2,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[2]);
        TT_SFPSTORE(p_sfpu::LREG3,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[3]);
    }
}


// Preserve TWO rounding steps: product=delta*K, then updated=product+state.
// This deliberately uses SFPMUL followed by SFPADD, never SFPMAD/FMA.
inline void _gdn_col_mul_add_(uint32_t data,uint32_t column,uint32_t output,uint32_t residual) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    const uint32_t db=data*DEST_TILE_SIZE_RAW, cb=column*DEST_TILE_SIZE_RAW;
    const uint32_t ob=output*DEST_TILE_SIZE_RAW, rb=residual*DEST_TILE_SIZE_RAW;
    for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {
        const uint32_t left=half*FACE2_BASE+band*ROW_BAND_STRIDE;
        const uint32_t right=left+FACE1_BASE;
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,cb+left);
        lltt::replay(REPLAY_SLOT_BROADCAST,REPLAY_LEN_BROADCAST);
        _broadcast_stage3_with_data_prefetch_(db+left,db+left+ODD_COLS_OFFSET,db+right,db+right+ODD_COLS_OFFSET);
        TTI_SFPMUL(LREG_DATA0,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA0,0);
        TTI_SFPMUL(LREG_DATA1,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA1,0);
        TTI_SFPMUL(LREG_DATA2,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA2,0);
        TTI_SFPMUL(LREG_DATA3,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA3,0);
        // Broadcast is complete; its two scratch LREGs can carry residuals.
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,rb+left);
        TT_SFPLOAD(LREG_TMP,IM,ADDR_MOD_7,rb+left+ODD_COLS_OFFSET);
        TTI_SFPNOP;
        TTI_SFPADD(LREG_DATA0,p_sfpu::LCONST_1,LREG_BCAST,LREG_DATA0,0);
        TTI_SFPADD(LREG_DATA1,p_sfpu::LCONST_1,LREG_TMP,LREG_DATA1,0);
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,rb+right);
        TT_SFPLOAD(LREG_TMP,IM,ADDR_MOD_7,rb+right+ODD_COLS_OFFSET);
        TTI_SFPNOP;
        TTI_SFPADD(LREG_DATA2,p_sfpu::LCONST_1,LREG_BCAST,LREG_DATA2,0);
        TTI_SFPADD(LREG_DATA3,p_sfpu::LCONST_1,LREG_TMP,LREG_DATA3,0);
        TT_SFPSTORE(LREG_DATA0,IM,ADDR_MOD_7,ob+left);
        TT_SFPSTORE(LREG_DATA1,IM,ADDR_MOD_7,ob+left+ODD_COLS_OFFSET);
        TT_SFPSTORE(LREG_DATA2,IM,ADDR_MOD_7,ob+right);
        TT_SFPSTORE(LREG_DATA3,IM,ADDR_MOD_7,ob+right+ODD_COLS_OFFSET);
    }
}

} // namespace sfpu
#endif
template<bool INIT=true>
ALWI void gdn_mul_scalar(uint32_t data,uint32_t scalar) {
    if constexpr(INIT) sfpu_normalize_bcast_scalar_init();
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_mul_scalar_,data,scalar,data,VectorMode::None)));
}

ALWI void gdn_copy_row(uint32_t data) {
    sfpu_bcast_row_init();
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_copy_row_,data,data,data,VectorMode::None)));
}


ALWI void gdn_col_mul_add(uint32_t data,uint32_t column,uint32_t residual) {
    sfpu_bcast_col_init();
    MATH((::ckernel::_sfpu_binary_check_<DST_SYNC_MODE>(data,residual,data,VectorMode::None),
        SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_col_mul_add_,data,column,data,VectorMode::None,residual)));
}

template<bool INIT=true>
ALWI void gdn_col_mul_add_to(uint32_t data,uint32_t column,uint32_t residual,uint32_t output) {
    if constexpr(INIT) sfpu_bcast_col_init();
    MATH((::ckernel::_sfpu_binary_check_<DST_SYNC_MODE>(data,residual,output,VectorMode::None),
        SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_col_mul_add_,data,column,output,VectorMode::None,residual)));
}


// Native broadcast implementation supports distinct input/output registers;
// use that directly instead of copying a retained state before multiplying it.
template<bool INIT=true>
ALWI void gdn_col_mul_to(uint32_t data,uint32_t column,uint32_t output) {
    if constexpr(INIT) sfpu_bcast_col_init();
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,
        _calculate_sfpu_binary_bcast_full_tile_,(ckernel::BinaryOp::MUL,ckernel::BroadcastType::COL),
        data,column,output,VectorMode::None)));
}
ALWI void gdn_copy_row_to(uint32_t row,uint32_t output) {
    sfpu_bcast_row_init();
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_copy_row_,row,row,output,VectorMode::None)));
}

} // namespace ckernel

void load(uint32_t cb,uint32_t tile,uint32_t dst) {
    copy_tile(cb,tile,dst);
}
void finish(uint32_t cb,uint32_t tile) {
    tile_regs_commit(); tile_regs_wait(); pack_tile<true>(0,cb,tile); tile_regs_release();
}
// Exactly the accurate FP32 reduction topology: fold the four H tiles with
// SFPU addition, then SFPU reduce their32 rows into row0. No TF32 FPU path.
void contract(uint32_t matrix,uint32_t column,uint32_t result) {
    CircularBuffer dst(result); dst.reserve_back(4);
    PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_COL,PackMode::Default>(result)));
    for(uint32_t n=0;n<4;n++) {
        tile_regs_acquire();
        for(uint32_t m=0;m<4;m++) {
            const uint32_t work=m==0 ? 0 : 2;
            load(matrix,m*4+n,work); load(column,m,1);
            sfpu_mul_bcast_col_init(); sfpu_mul_bcast_col(work,1);
            if(m) { add_binary_tile_init(); add_binary_tile(0,2,0); }
        }
        sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_COL>(0,1,1);
        finish(result,n);
    }
    PACK((llk_pack_reduce_mask_clear())); dst.push_back(4);
}

// Raw mode matches external FP32 square/sum(+epsilon)/rsqrt/multiply, then
// separate Q scaling. Normalized mode is a pure transpose control.
void prepare_vector(uint32_t input,uint32_t output,bool is_q) {
    constexpr bool NORMALIZED=get_compile_time_arg_val(0);
    CircularBuffer in(input),col(output),norm(16),rows(17);
    in.wait_front(4);
    copy_init(input);
    uint32_t transposed_source=input;
    if constexpr(!NORMALIZED) {
        norm.reserve_back(1); tile_regs_acquire();
        for(uint32_t i=0;i<4;i++) {
            const uint32_t work=i?2:0;
            load(input,i,work); load(input,i,1);
            mul_binary_tile_init(); mul_binary_tile(work,1,work);
            if(i) { add_binary_tile_init(); add_binary_tile(0,2,0); }
        }
        sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_ROW>(0,1,1);
        // Float32 encodings of1e-6 and128**-0.5. Scaling remains separate.
        binop_with_scalar_tile_init(); add_unary_tile(0,0x358637bdu);
        rsqrt_tile_init(); rsqrt_tile(0);
        finish(16,0); norm.push_back(1); norm.wait_front(1);
        rows.reserve_back(4);
        for(uint32_t i=0;i<4;i++) {
            tile_regs_acquire(); load(input,i,0); load(16,0,1);
            gdn_mul_scalar(0,1);
            if(is_q) { binop_with_scalar_tile_init(); mul_unary_tile(0,0x3db504f3u); }
            finish(17,i);
        }
        rows.push_back(4); rows.wait_front(4); norm.pop_front(1);
        transposed_source=17;
    }
    col.reserve_back(4); transpose_init(transposed_source);
    for(uint32_t i=0;i<4;i++) {
        tile_regs_acquire(); transpose_tile(transposed_source,i,0); finish(output,i);
    }
    col.push_back(4); in.pop_front(4);
    if constexpr(!NORMALIZED) rows.pop_front(4);
}


// Per value-column full-DST allocation: state0..3, work4, Q/K5,
// accumulator/delta6, decay/beta7. No retained state crosses a release.
void kernel_main() {
#ifdef TRISC_MATH
    static_assert(get_dest_max_tiles<DST_SYNC_MODE,true,DstTileShape::Tile32x32>()>=8,
                  "Retained GDN needs eight FP32 DST tiles and full synchronization");
#endif
    const uint32_t count=get_arg_val<uint32_t>(0);
    constexpr bool STATE_BF16=get_compile_time_arg_val(1);
    constexpr bool GROUP_QK=get_compile_time_arg_val(2);
    constexpr bool FUSE_ACCUMULATOR=get_compile_time_arg_val(3);
    constexpr bool HOIST_DELTA=get_compile_time_arg_val(4);
    constexpr bool HOIST_SCALAR_INIT=get_compile_time_arg_val(5);
    constexpr bool HOIST_COL_INIT=get_compile_time_arg_val(6);
    compute_kernel_hw_startup(STATE_BF16?3:5,6,12);
    copy_init(STATE_BF16?3:5);
    CircularBuffer q(13),k(14),v(2),raw_g(3),decay(18),beta(4),state(5),zero(6),one(7);
    CircularBuffer updated(11),out(12),raw_q(0),raw_k(1);
    zero.wait_front(1); one.wait_front(1);
    for(uint32_t head=0;head<count;head++) {
        state.wait_front(16); raw_g.wait_front(1);
        decay.reserve_back(1); tile_regs_acquire(); load(3,0,0);
        exp_tile_init<false>(); exp_tile<false>(0); finish(18,0);
        decay.push_back(1); decay.wait_front(1); raw_g.pop_front(1);
        // Host partitions only complete aligned 3-head groups in grouped mode.
        // CB13/14 retain exactly the first head's prepared Q/K across its peers.
        if (!GROUP_QK || head%3==0) {
            prepare_vector(0,13,true); prepare_vector(1,14,false);
        } else {
            // Reader's rings remain one head deep: consume every duplicate input
            // to permit it to fetch the next state/value/gate without stalling.
            raw_q.wait_front(4); raw_k.wait_front(4);
            raw_q.pop_front(4); raw_k.pop_front(4);
        }
        copy_init(STATE_BF16?3:5);
        q.wait_front(4); k.wait_front(4); v.wait_front(4); beta.wait_front(1);
        updated.reserve_back(16); out.reserve_back(4);
        for(uint32_t n=0;n<4;n++) {
            tile_regs_acquire();
            load(18,0,7);
            if constexpr(STATE_BF16) {
                reconfig_data_format_srca(5); copy_init(5);
            }
            // copy_tile does not alter SFPU config, MOD7, LREGs or replay.
            if constexpr(HOIST_SCALAR_INIT) sfpu_normalize_bcast_scalar_init();
            for(uint32_t m=0;m<4;m++) {
                load(5,m*4+n,m);
                gdn_mul_scalar<!HOIST_SCALAR_INIT>(m,7);
            }
            if constexpr(STATE_BF16) {
                reconfig_data_format_srca(3); copy_init(3);
            }
            // Preserve the baseline four-K-tile accumulation order exactly.
            if constexpr(HOIST_COL_INIT && FUSE_ACCUMULATOR) sfpu_bcast_col_init();
            for(uint32_t m=0;m<4;m++) {
                load(14,m,5);
                if constexpr(FUSE_ACCUMULATOR) {
                    if(m) {
                        // Exact separate FP32 product then add into accumulator.
                        // Skip the full temporary-tile store/reload/add pass.
                        gdn_col_mul_add_to<!HOIST_COL_INIT>(m,5,6,6);
                    } else { gdn_col_mul_to<!HOIST_COL_INIT>(m,5,6); }
                } else {
                    const uint32_t work=m?4:6;
                    gdn_col_mul_to(m,5,work);
                    if(m) { add_binary_tile_init(); add_binary_tile(6,4,6); }
                }
            }
            sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
            sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_COL>(6,1,1);
            load(2,n,4);
            sub_binary_tile_init(); sub_binary_tile(4,6,6);
            load(4,0,7); gdn_mul_scalar(6,7);
            // State update writes only DST0..3, leaving broadcast delta DST4
            // unchanged. Reuse one broadcast across all four K-row tiles.
            if constexpr(HOIST_DELTA) gdn_copy_row_to(6,4);
            // Row broadcast overwrites LREG6: restore the column mask/replay here.
            if constexpr(HOIST_COL_INIT && HOIST_DELTA) sfpu_bcast_col_init();
            for(uint32_t m=0;m<4;m++) {
                if constexpr(!HOIST_DELTA) gdn_copy_row_to(6,4);
                load(14,m,5);
                // Separate FP32 multiply/add; write directly over retained state.
                gdn_col_mul_add_to<!(HOIST_COL_INIT && HOIST_DELTA)>(4,5,m,m);
            }
            if constexpr(HOIST_COL_INIT && FUSE_ACCUMULATOR) sfpu_bcast_col_init();
            for(uint32_t m=0;m<4;m++) {
                load(13,m,5);
                if constexpr(FUSE_ACCUMULATOR) {
                    if(m) {
                        // Exact separate FP32 product then add into accumulator.
                        // Skip the full temporary-tile store/reload/add pass.
                        gdn_col_mul_add_to<!HOIST_COL_INIT>(m,5,6,6);
                    } else { gdn_col_mul_to<!HOIST_COL_INIT>(m,5,6); }
                } else {
                    const uint32_t work=m?4:6;
                    gdn_col_mul_to(m,5,work);
                    if(m) { add_binary_tile_init(); add_binary_tile(6,4,6); }
                }
            }
            sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
            sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_COL>(6,1,1);
            if constexpr(STATE_BF16) {
                // Match TTNN Float32->BF16 RNE exactly, after current readout.
                typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();
                for(uint32_t m=0;m<4;m++)
                    typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>(m);
            }
            tile_regs_commit(); tile_regs_wait();
            PACK((llk_pack_reduce_mask_clear()));
            // Q readout is already complete in FP32 DST6. Only persistence rounds.
            if constexpr(STATE_BF16) pack_reconfig_data_format(12,11);
            for(uint32_t m=0;m<4;m++) pack_tile<true>(m,11,m*4+n);
            if constexpr(STATE_BF16) pack_reconfig_data_format(11,12);
            PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_COL,PackMode::Default>(12)));
            pack_tile<true>(6,12,n);
            PACK((llk_pack_reduce_mask_clear()));
            tile_regs_release();
        }
        updated.push_back(16); out.push_back(4);
        state.pop_front(16); decay.pop_front(1); v.pop_front(4); beta.pop_front(1);
        if (!GROUP_QK || head%3==2) {
            q.pop_front(4); k.pop_front(4);
        }
    }
    zero.pop_front(1); one.pop_front(1);
}
"""


def _grouped96_head_partition(batch, available_cores, cores=None, grouped_qk=True):
    """Pure host partition: (grouped, [(global_head_start, head_count), ...])."""
    assert batch >= 1 and available_cores >= 1
    assert cores is None or cores >= 1
    grouped = bool(grouped_qk and batch >= 3)
    unit = 3 if grouped else 1
    units = batch * 48 // unit
    ncores = min(units, available_cores, cores or available_cores)
    result = []
    start = 0
    for index in range(ncores):
        count = unit * (units // ncores + (index < units % ncores))
        result.append((start, count))
        start += count
    assert start == batch * 48
    return grouped, result


def _grouped96_recurrent_sfpu_compact(
    q, k, v, g, beta, state, *, normalized=False, packed_qkv=False, output=None, cores=None, grouped_qk=True
):
    """Accept raw RM[B,1,C] directly, or the original TILE interface."""
    row_major = q.layout == ttnn.ROW_MAJOR_LAYOUT
    assert not (row_major and normalized)
    assert not packed_qkv or row_major
    assert state.dtype in (ttnn.float32, ttnn.bfloat16)
    state_bf16 = state.dtype == ttnn.bfloat16
    state_bytes = 2048 if state_bf16 else 4096
    batch = state.shape[0] // 48
    assert tuple(state.shape) == (batch * 48, 128, 128)
    assert q.dtype == k.dtype == (ttnn.float32 if normalized else ttnn.bfloat16)
    assert v.dtype == ttnn.bfloat16
    assert tuple(q.shape) == tuple(k.shape)
    if packed_qkv:
        assert tuple(q.shape) == tuple(k.shape) == tuple(v.shape) == (batch, 1, 10240)
        assert q.buffer_address() == k.buffer_address() == v.buffer_address()
    elif normalized:
        assert tuple(q.shape) == (batch, 16, 128)
    else:
        assert q.shape[0] == batch and q.shape[-1] == 2048 and q.shape[1] in (1, 32)
    if not packed_qkv:
        assert v.shape[0] == batch and v.shape[-1] == 6144 and v.shape[1] in (1, 32)
    assert tuple(g.shape) == tuple(beta.shape) == (batch, 1, 48)
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
            (1, batch * 48, 128),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=state.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    assert tuple(output.shape) == (1, batch * 48, 128)
    grid = state.device().compute_with_storage_grid_size()
    group_qk, partition = _grouped96_head_partition(batch, grid.x * grid.y, cores, grouped_qk)
    ncores = len(partition)
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncores)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    for c, (start, count) in zip(coords, partition):
        rr[c.x][c.y] = [t.buffer_address() for t in inputs] + [start, count]
        wr[c.x][c.y] = [state.buffer_address(), output.buffer_address(), start, count]
        cr[c.x][c.y] = [count]
    # Raw q/k/v paddedS32 occupies one tile row per batch.
    rct = [
        int(normalized),
        int(g.dtype == ttnn.float32),
        int(beta.dtype == ttnn.float32),
        64,
        192,
        int(row_major),
        int(packed_qkv),
        state_bytes,
    ]
    for t in inputs:
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct = [state_bytes]
    for t in (state, output):
        wct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct.append(batch * 48)

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
        kernel(_GROUPED96_GDN_COMPACT_READER, ttnn.ReaderConfigDescriptor(), rr, rct),
        kernel(_GROUPED96_GDN_COMPACT_WRITER, ttnn.WriterConfigDescriptor(), wr, wct),
        kernel(
            _GROUPED96_GDN_COMPACT_COMPUTE,
            _grouped96_compact_fp32_compute_config(state_bf16),
            cr,
            [
                int(normalized),
                int(state_bf16),
                int(group_qk),
                int(_GROUPED96_GDN_FUSE_ACCUMULATOR),
                int(_GROUPED96_GDN_HOIST_DELTA),
                int(_GROUPED96_GDN_HOIST_SCALAR_INIT),
                int(_GROUPED96_GDN_HOIST_COL_INIT),
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
        13: 4,
        14: 4,
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


_conv107__CONV_RM_PIPE2_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto xa=TensorAccessorArgs<0>();
    constexpr auto ha=TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto t0a=TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    constexpr auto t1a=TensorAccessorArgs<t0a.next_compile_time_args_offset()>();
    constexpr auto t2a=TensorAccessorArgs<t1a.next_compile_time_args_offset()>();
    constexpr auto t3a=TensorAccessorArgs<t2a.next_compile_time_args_offset()>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0),20480);
    const auto history=TensorAccessor(ha,get_arg_val<uint32_t>(1),20480);
    const auto t0=TensorAccessor(t0a,get_arg_val<uint32_t>(2),20480);
    const auto t1=TensorAccessor(t1a,get_arg_val<uint32_t>(3),20480);
    const auto t2=TensorAccessor(t2a,get_arg_val<uint32_t>(4),20480);
    const auto t3=TensorAccessor(t3a,get_arg_val<uint32_t>(5),20480);
    const uint32_t start=get_arg_val<uint32_t>(6),count=get_arg_val<uint32_t>(7);
    Noc noc;
    for(uint32_t task=start;task<start+count;task++) {
        const uint32_t user=task/10,chunk=task%10;
        for(uint32_t id=0;id<8;id++) { CircularBuffer cb(id); cb.reserve_back(1); }
        for(uint32_t row=0;row<3;row++) {
            CircularBuffer cb(row);
            noc.async_read(history,cb,2048,{.page_id=user*3+row,.offset_bytes=chunk*2048},{.offset_bytes=0});
        }
        CircularBuffer cb4(4),cb5(5),cb6(6),cb7(7);
        noc.async_read(t0,cb4,2048,{.page_id=0,.offset_bytes=chunk*2048},{.offset_bytes=0});
        noc.async_read(t1,cb5,2048,{.page_id=0,.offset_bytes=chunk*2048},{.offset_bytes=0});
        noc.async_read(t2,cb6,2048,{.page_id=0,.offset_bytes=chunk*2048},{.offset_bytes=0});
        noc.async_read(t3,cb7,2048,{.page_id=0,.offset_bytes=chunk*2048},{.offset_bytes=0});
        // Native untilize has already placed each logical user's channels
        // contiguously. This is the identical physical CB3 sequence produced
        // by the old 64-small-read gather and RISC-V packed-word copy loop.
        CircularBuffer token(3);
        noc.async_read(x,token,2048,{.page_id=user,.offset_bytes=chunk*2048},{.offset_bytes=0});
        noc.async_read_barrier();
        for(uint32_t id=0;id<8;id++) { CircularBuffer cb(id); cb.push_back(1); }
        // Two input slots allow this reader to fill the next task while
        // compute/writer use the first. Compute alone pops the completed pair.
    }
}
"""

_conv107__CONV_RM_PIPE2_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto ha=TensorAccessorArgs<0>();
    constexpr auto oa=TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    const auto history=TensorAccessor(ha,get_arg_val<uint32_t>(0),20480);
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(1),20480);
    const uint32_t start=get_arg_val<uint32_t>(2),count=get_arg_val<uint32_t>(3);
    Noc noc; CircularBuffer out(9),done(10);
    for(uint32_t task=start;task<start+count;task++) {
        const uint32_t user=task/10,chunk=task%10,slot=(task-start)%2;
        out.wait_front(1);
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),output,2048,
            {.offset_bytes=0},{.page_id=user,.offset_bytes=chunk*2048});
        for(uint32_t row=0;row<3;row++) {
            // Output publication proves compute saw all inputs for this task.
            // This writer never pops input CBs, so its read pointer stays at
            // the ring base; choose the physical slot explicitly.
            CircularBuffer carry(row+1);
            noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(carry),history,2048,
                {.offset_bytes=slot*2048},{.page_id=user*3+row,.offset_bytes=chunk*2048});
        }
        noc.async_write_barrier();
        out.pop_front(1);
        done.reserve_back(1); done.push_back(1);
    }
}
"""

_conv107__CONV_RM_PIPE2_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"
// Native ttnn.silu(BF16) chooses calculate_silu<false>, including its BF16
// approximation/rounding. Keep that arithmetic while addressing our FP32 DST.
// silu_tile<false> would also select BF16 DST addressing, so do not use it here.
inline void conv_bf16_silu(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,true,calculate_silu,(false,8),idst,VectorMode::RC));
}
void kernel_main() {
    const uint32_t count=get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0,9); copy_init(0);
    CircularBuffer out(9),done(10);
    // Pop inputs only as complete groups. The unpack read pointer is at
    // the ring base for every group, so copy_tile(...,slot,...) never crosses
    // the physical CB end. A final singleton is safe because no task follows.
    for(uint32_t first=0;first<count;first+=2) {
        const uint32_t group=(count-first>=2)?2:1;
        for(uint32_t slot=0;slot<group;slot++) {
        for(uint32_t id=0;id<8;id++) { CircularBuffer cb(id); cb.wait_front(slot+1); }
        out.reserve_back(1); tile_regs_acquire();
        for(uint32_t tap=0;tap<4;tap++) {
            const uint32_t work=tap?1:0;
            copy_tile(tap,slot,work); copy_tile(tap+4,slot,2);
            mul_binary_tile_init(); mul_binary_tile(work,2,work);
            if(tap) { add_binary_tile_init(); add_binary_tile(0,1,0); }
        }
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>(0);
        silu_tile_init(); conv_bf16_silu(0);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0,9); tile_regs_release();
        out.push_back(1);
        }
        // Each token follows that task's complete NoC write barrier. Preserve
        // input storage through both consumers; compute is the sole pop owner.
        done.wait_front(group);
        for(uint32_t id=0;id<8;id++) { CircularBuffer cb(id); cb.pop_front(group); }
        done.pop_front(group);
    }
}
"""


def _conv107_conv_decode_sfpu(token, history, taps, *, output=None, cores=None):
    """TILE compact token -> packed RM output, stable RM carry allocation.

    Supports only the validated FP32 sum -> BF16 -> native BF16 SiLU policy.
    Current token is raw projection output; history stores three raw tokens.
    """
    batch = history.shape[0]
    assert 1 <= batch <= 32 and tuple(history.shape) == (batch, 3, 10240)
    assert tuple(token.shape) == (1, batch, 10240)
    assert token.layout == ttnn.TILE_LAYOUT and history.layout == ttnn.ROW_MAJOR_LAYOUT
    assert len(taps) == 4
    for tap in taps:
        assert tuple(tap.shape) == (1, 1, 10240) and tap.layout == ttnn.ROW_MAJOR_LAYOUT
    # Convert exactly once per invocation. to_layout uses multicore untilize
    # (untilize_with_unpadding for B<32), with no arithmetic or dtype change.
    raw_token = ttnn.to_layout(token, ttnn.ROW_MAJOR_LAYOUT)
    assert tuple(raw_token.shape) == tuple(token.shape) and raw_token.dtype == ttnn.bfloat16
    inputs = [raw_token, history, *taps]
    for tensor in inputs:
        assert tensor.dtype == ttnn.bfloat16 and tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    if output is None:
        output = ttnn.empty(
            (batch, 1, 10240),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=token.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    assert tuple(output.shape) == (batch, 1, 10240) and output.dtype == ttnn.bfloat16
    assert output.layout == ttnn.ROW_MAJOR_LAYOUT and output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert output.buffer_address() not in [t.buffer_address() for t in inputs]
    grid = token.device().compute_with_storage_grid_size()
    tasks = batch * 10
    ncores = min(tasks, cores or grid.x * grid.y)
    assert 1 <= ncores <= grid.x * grid.y
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncores)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    start = 0
    for i, c in enumerate(coords):
        count = tasks // ncores + (i < tasks % ncores)
        rr[c.x][c.y] = [t.buffer_address() for t in inputs] + [start, count]
        wr[c.x][c.y] = [history.buffer_address(), output.buffer_address(), start, count]
        cr[c.x][c.y] = [count]
        start += count
    rct = []
    wct = []
    for t in inputs:
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    for t in (history, output):
        wct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

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
    kernels = [
        kernel(_conv107__CONV_RM_PIPE2_READER, ttnn.ReaderConfigDescriptor(), rr, rct),
        kernel(_conv107__CONV_RM_PIPE2_WRITER, ttnn.WriterConfigDescriptor(), wr, wct),
        kernel(_conv107__CONV_RM_PIPE2_COMPUTE, config, cr, []),
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * 2048,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for i in (*range(8), 9, 10)
    ]
    ttnn.generic_op(inputs + [output], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return output


# Exact measured helper from narrow_gdn_combine.py
_NARROW_GDN_COMBINE_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto pa=TensorAccessorArgs<0>();
    constexpr auto ra=TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    const auto primary=TensorAccessor(pa,get_arg_val<uint32_t>(0),4096);
    const auto residual=TensorAccessor(ra,get_arg_val<uint32_t>(1),4096);
    const uint32_t pn=get_arg_val<uint32_t>(2),rn=get_arg_val<uint32_t>(3);
    const uint32_t start=get_arg_val<uint32_t>(4),count=get_arg_val<uint32_t>(5);
    Noc noc; CircularBuffer p(0),r(1),zero(3);
    zero.reserve_back(1); noc.async_write_zeros(zero,4096); noc.write_zeros_l1_barrier();zero.push_back(1);
    for(uint32_t tile=start;tile<start+count;tile++) {
        const uint32_t row=tile/516,col=tile%516;
        const bool has_residual=col<320 || col>=512;
        p.reserve_back(1);
        if(has_residual) r.reserve_back(1);
        noc.async_read(primary,p,4096,{.page_id=row*pn+col},{.offset_bytes=0});
        if(has_residual) {
            const uint32_t rc=col<320?col:col-192;
            noc.async_read(residual,r,4096,{.page_id=row*rn+rc},{.offset_bytes=0});
        }
        noc.async_read_barrier(); p.push_back(1);
        if(has_residual) r.push_back(1);
    }
}
"""
_NARROW_GDN_COMBINE_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto oa=TensorAccessorArgs<0>();
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(0),2048);
    const uint32_t start=get_arg_val<uint32_t>(1),count=get_arg_val<uint32_t>(2);
    Noc noc; CircularBuffer out(2);
    for(uint32_t tile=start;tile<start+count;tile++) {
        out.wait_front(1);
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),output,2048,
            {.offset_bytes=0},{.page_id=tile});
        noc.async_write_barrier();out.pop_front(1);
    }
}
"""
_NARROW_GDN_COMBINE_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    const uint32_t start=get_arg_val<uint32_t>(0),count=get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(0,2);copy_init(0);
    CircularBuffer p(0),r(1),out(2),zero(3);zero.wait_front(1);
    for(uint32_t tile=start;tile<start+count;tile++) {
        const uint32_t col=tile%516;const bool has_residual=col<320 || col>=512;
        p.wait_front(1);if(has_residual)r.wait_front(1);
        out.reserve_back(1);tile_regs_acquire();
        copy_tile(0,0,0);
        if(has_residual) copy_tile(1,0,1);else copy_tile(3,0,1);
        add_binary_tile_init();add_binary_tile(0,1,0);
        // Preserve native FP32 add -> RNE BF16 boundary, including Z +0.
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>(0);
        tile_regs_commit();tile_regs_wait();pack_tile(0,2);tile_regs_release();
        out.push_back(1);p.pop_front(1);if(has_residual)r.pop_front(1);
    }
    zero.pop_front(1);
}
"""


def narrow_gdn_combine(primary, residual, *, output=None, cores=None):
    assert len(primary.shape) == len(residual.shape) == 3
    batch, seq, pwidth = tuple(primary.shape)
    rb, rs, rwidth = tuple(residual.shape)
    assert (batch, seq) == (rb, rs) and pwidth >= 16512 and rwidth >= 10368
    assert tuple(primary.padded_shape)[:2] == tuple(residual.padded_shape)[:2]
    for t in (primary, residual):
        assert t.dtype == ttnn.float32 and t.layout == ttnn.TILE_LAYOUT
        mem = t.memory_config()
        assert mem == ttnn.DRAM_MEMORY_CONFIG or (
            mem.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED and mem.buffer_type == ttnn.BufferType.L1
        )
    if output is None:
        output = ttnn.empty(
            (batch, seq, 16512),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=primary.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    assert (
        tuple(output.shape) == (batch, seq, 16512) and tuple(output.padded_shape)[:2] == tuple(primary.padded_shape)[:2]
    )
    assert (
        output.dtype == ttnn.bfloat16
        and output.layout == ttnn.TILE_LAYOUT
        and output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    )
    assert output.buffer_address() not in (primary.buffer_address(), residual.buffer_address())
    total = batch * (primary.padded_shape[-2] // 32) * 516
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
        kernel(_NARROW_GDN_COMBINE_READER, ttnn.ReaderConfigDescriptor(), rr, rct),
        kernel(
            _NARROW_GDN_COMBINE_WRITER,
            ttnn.WriterConfigDescriptor(),
            wr,
            ttnn.TensorAccessorArgs(output).get_compile_time_args(),
        ),
        kernel(_NARROW_GDN_COMBINE_COMPUTE, config, cr, []),
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


# Exact measured helper from prefill_unflatten_bf16_v2.py

_PREFILL_UNFLATTEN_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto a=TensorAccessorArgs<0>();
    const auto input=TensorAccessor(a,get_arg_val<uint32_t>(0),2048);
    const uint32_t seq=get_arg_val<uint32_t>(1),nt=get_arg_val<uint32_t>(2);
    const uint32_t start=get_arg_val<uint32_t>(3),count=get_arg_val<uint32_t>(4);
    const uint32_t st=(seq+31)/32;
    Noc noc; CircularBuffer out(1);
    for(uint32_t tile=start;tile<start+count;tile++) {
        const uint32_t batch=tile/(st*nt),local=tile%(st*nt);
        const uint32_t tile_row=local/nt,column=local%nt;
        const uint32_t first=batch*seq+tile_row*32,offset=first%32;
        const uint32_t valid=seq-tile_row*32<32?seq-tile_row*32:32;
        const uint32_t source=(first/32)*nt+column;
        // BH DRAM read endpoints must have equal offsets modulo64. Each
        // BF16 face row is32B: shift the assembled tile by32B for odd first
        // rows. DRAM writes only require equal offsets modulo16, allowing
        // the writer to emit the shifted tile directly without local copies.
        const uint32_t shift=(offset&1u)*32;
        out.reserve_back(1);  // One4096B slot contains the2048B tile+shift.
        if(offset==0 && valid==32) {
            noc.async_read(input,out,2048,{.page_id=source},{.offset_bytes=0});
        } else {
            uint32_t row=0;
            while(row<valid) {
                const uint32_t sr=offset+row;
                uint32_t rows=16-(row%16);
                if(rows>16-(sr%16)) rows=16-(sr%16);
                if(rows>valid-row) rows=valid-row;
                for(uint32_t face=0;face<2;face++) {
                    const uint32_t src=((sr%32)/16)*1024+face*512+(sr%16)*32;
                    const uint32_t dst=shift+(row/16)*1024+face*512+(row%16)*32;
                    noc.async_read(input,out,rows*32,
                        {.page_id=source+(sr/32)*nt,.offset_bytes=src},{.offset_bytes=dst});
                }
                row+=rows;
            }
            // Pad only invalid rows. These writes are disjoint from every
            // valid segment above; all starts are32B-aligned (L1 requires16).
            row=valid;
            while(row<32) {
                const uint32_t rows=16-(row%16);
                for(uint32_t face=0;face<2;face++) {
                    const uint32_t dst=shift+(row/16)*1024+face*512+(row%16)*32;
                    noc.async_write_zeros(out,rows*32,{.offset_bytes=dst});
                }
                row+=rows;
            }
        }
        // On BH zeros are loopback reads, so this waits for both the valid
        // DRAM segments and disjoint pad-zero transactions in one barrier.
        if(valid<32) noc.write_zeros_l1_barrier();
        else noc.async_read_barrier();
        out.push_back(1);
    }
}
"""
_PREFILL_UNFLATTEN_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto a=TensorAccessorArgs<0>();
    const auto output=TensorAccessor(a,get_arg_val<uint32_t>(0),2048);
    const uint32_t start=get_arg_val<uint32_t>(1),count=get_arg_val<uint32_t>(2);
    const uint32_t seq=get_arg_val<uint32_t>(3),nt=get_arg_val<uint32_t>(4);
    const uint32_t tiles_per_batch=((seq+31)/32)*nt;
    Noc noc; CircularBuffer out(1);
    for(uint32_t tile=start;tile<start+count;tile++) {
        const uint32_t shift=(((tile/tiles_per_batch)*seq)&1u)*32;
        out.wait_front(1);
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),output,2048,
            {.offset_bytes=shift},{.page_id=tile});
        noc.async_write_barrier(); out.pop_front(1);
    }
}
"""


def prefill_unflatten_bf16(x, batch, seq, *, output=None, cores=None):
    assert len(x.shape) == 3 and tuple(x.shape)[:2] == (1, batch * seq)
    width = x.shape[-1]
    assert batch > 0 and seq > 0 and width % 32 == 0
    assert tuple(x.padded_shape) == (1, ((batch * seq + 31) // 32) * 32, width)
    assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT and x.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    if output is None:
        output = ttnn.empty(
            (batch, seq, width),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=x.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    assert tuple(output.shape) == (batch, seq, width) and tuple(output.padded_shape) == (
        batch,
        ((seq + 31) // 32) * 32,
        width,
    )
    assert (
        output.dtype == ttnn.bfloat16
        and output.layout == ttnn.TILE_LAYOUT
        and output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    )
    assert output.buffer_address() != x.buffer_address()
    total = batch * ((seq + 31) // 32) * (width // 32)
    grid = x.device().compute_with_storage_grid_size()
    workers = min(total, cores or grid.x * grid.y)
    assert 1 <= workers <= grid.x * grid.y
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    start = 0
    for i, c in enumerate(coords):
        count = total // workers + (i < total % workers)
        rr[c.x][c.y] = [x.buffer_address(), seq, width // 32, start, count]
        wr[c.x][c.y] = [output.buffer_address(), start, count, seq, width // 32]
        start += count

    def kernel(source, config, args, tensor):
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=ttnn.TensorAccessorArgs(tensor).get_compile_time_args(),
            runtime_args=args,
            config=config,
        )

    kernels = [
        kernel(_PREFILL_UNFLATTEN_READER, ttnn.ReaderConfigDescriptor(), rr, x),
        kernel(_PREFILL_UNFLATTEN_WRITER, ttnn.WriterConfigDescriptor(), wr, output),
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=8192,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=4096)],
        )
        for i in (1,)
    ]
    ttnn.generic_op([x, output], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return output


_QKV_BATCHED_STAGE = r"""
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

    // Each RISC owns one horizontal face. Only padding is initialized here;
    // the other RISC never writes these byte ranges. Complete zero mode before
    // issuing ordinary NoC operations or local scatter copies.
    const uint32_t buffers[3]={qb,kb,vb};
    const uint32_t heads[3]={24,4,4};
    for(uint32_t which=0;which<3;which++) for(uint32_t tile=0;tile<8;tile++) {
        for(uint32_t half=0;half<2;half++) {
            const uint32_t first=heads[which]>half*16?heads[which]:half*16;
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
    // 24 Q + 4 K + 4 V heads, eight channel tiles/head. Every64-byte
    // destination slot is unique, so no per-head scratch-reuse barrier is needed.
    for(uint32_t tile=0;tile<256;tile++) {
        noc.async_read(input,CoreLocalMem<uint32_t>(base+tile*64),64,
                       {.page_id=tile,.offset_bytes=aligned_row},{});
    }
    noc.async_read_barrier();
    for(uint32_t tile=0;tile<256;tile++) {
        const uint32_t global_head=tile/8,channel_tile=tile%8;
        const uint32_t head=global_head<24?global_head:(global_head<28?global_head-24:global_head-28);
        const uint32_t out_base=global_head<24?qb:(global_head<28?kb:vb);
        const uint32_t dst=out_base+channel_tile*2048+(head/16)*1024+(head%16)*32+PHASE*512;
        tt::data_movement::common::tt_memmove<false,true,true,32>(
            noc,dst,base+tile*64+skew,32);
    }
    noc.async_read_barrier();
}
"""


def qkv_split_batched_staging(input_tensor, *, num_heads=24, num_kv_heads=4, memory_config):
    assert num_heads == 24 and num_kv_heads == 4
    assert input_tensor.dtype == ttnn.bfloat16 and input_tensor.layout == ttnn.TILE_LAYOUT
    assert input_tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    batch = input_tensor.shape[2]
    assert tuple(input_tensor.shape) == (1, 1, batch, 8192) and 1 <= batch <= 32
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
        for heads in (24, 4, 4)
    ]
    runtime = ttnn.RuntimeArgs()
    for user, core in enumerate(coords):
        runtime[core.x][core.y] = [input_tensor.buffer_address(), *[t.buffer_address() for t in outputs], user]
    kernels = []
    for phase, config in enumerate((ttnn.ReaderConfigDescriptor(), ttnn.WriterConfigDescriptor())):
        compile_args = [phase, *ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=_QKV_BATCHED_STAGE,
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
            total_size=257 * 64,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=phase, data_format=ttnn.bfloat16, page_size=64)],
        )
        for phase in (0, 1)
    ]
    ttnn.generic_op([input_tensor, *outputs], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
    return tuple(outputs)


_fused_delta_row_GDN_FUSE_ACCUMULATOR = True
_fused_delta_row_GDN_HOIST_DELTA = True
_fused_delta_row_GDN_HOIST_SCALAR_INIT = True
_fused_delta_row_GDN_HOIST_COL_INIT = True
_fused_delta_row_GDN_FUSE_DECAY_PREDICT = True


def _fused_delta_row_compact_fp32_compute_config(state_bf16=False):
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True, math_approx_mode=False
    )
    # FP32 DEST alone still allows Float32 -> TF32 through source registers.
    # Explicit direct unpack is required for every FP32 CB copied into DST.
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for index in range(19):
        modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    if state_bf16:
        modes[5] = modes[11] = ttnn.UnpackToDestMode.Default
    config.unpack_to_dest_mode = modes
    return config


_fused_delta_row_GDN_COMPACT_READER = r"""
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
    const auto q=TensorAccessor(qa,get_arg_val<uint32_t>(0),PACKED?20480:((NORMALIZED || ROW_MAJOR)?4096:2048));
    const auto k=TensorAccessor(ka,get_arg_val<uint32_t>(1),PACKED?20480:((NORMALIZED || ROW_MAJOR)?4096:2048));
    const auto v=TensorAccessor(va,get_arg_val<uint32_t>(2),PACKED?20480:(ROW_MAJOR?12288:2048));
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
        const uint32_t page=user*2+head/32;
        const uint32_t offset=((head%32)/16)*256*element_bytes;
        noc.async_read(a,scratch,16*element_bytes,{.page_id=page,.offset_bytes=offset},{.offset_bytes=dst});
    };
    for(uint32_t h=start;h<start+count;h++) {
        const uint32_t user=h/48,hv=h%48,hk=hv/3;
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
            noc.async_read(k,scratch,256,{.page_id=user,.offset_bytes=(PACKED?4096:0)+hk*256},{.offset_bytes=512});
            noc.async_read(v,scratch,256,{.page_id=user,.offset_bytes=(PACKED?8192:0)+hv*256},{.offset_bytes=1024});
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

_fused_delta_row_GDN_COMPACT_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t STATE_BYTES=get_compile_time_arg_val(0);
    constexpr auto sa=TensorAccessorArgs<1>();
    constexpr auto oa=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto s=TensorAccessor(sa,get_arg_val<uint32_t>(0),STATE_BYTES);
    const auto o=TensorAccessor(oa,get_arg_val<uint32_t>(1),4096);
    constexpr uint32_t BH=get_compile_time_arg_val(oa.next_compile_time_args_offset());
    const uint32_t start=get_arg_val<uint32_t>(2),count=get_arg_val<uint32_t>(3);
    Noc noc; CircularBuffer updated(11),out(12);
    for(uint32_t h=start;h<start+count;h++) {
        updated.wait_front(16); out.wait_front(4);
        // Waiting for output guarantees compute finished reading updated CB.
        for(uint32_t i=0;i<16;i++)
            noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(updated),s,STATE_BYTES,{.offset_bytes=i*STATE_BYTES},{.page_id=h*16+i});
        for(uint32_t i=0;i<4;i++) for(uint32_t face=0;face<2;face++) {
            const uint32_t row=h%32;
            const uint32_t dest=((row/16)*512+face*256+(row%16)*16)*4;
            noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),o,64,
                {.offset_bytes=i*4096+face*1024},
                {.page_id=(h/32)*4+i,.offset_bytes=dest});
        }
        if(h+1==BH && BH%32) {
            // Preserve deterministic zero padding for non-tile-aligned head count.
            // Row1 in the reduced output tile is zero by the reduction pack mask.
            for(uint32_t row=BH%32;row<32;row++) for(uint32_t i=0;i<4;i++) for(uint32_t face=0;face<2;face++) {
                const uint32_t dest=((row/16)*512+face*256+(row%16)*16)*4;
                noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),o,64,
                    {.offset_bytes=i*4096+64+face*1024},
                    {.page_id=(h/32)*4+i,.offset_bytes=dest});
            }
        }
        noc.async_write_barrier(); updated.pop_front(16); out.pop_front(4);
    }
}
"""

_fused_delta_row_GDN_COMPACT_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reduce.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/transpose.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"


// Stage-local precise FP32 specialization of native scalar normalization:
// mean is known zero, so eliminate its unpack/broadcast and32 scalar adds.
namespace ckernel {
#ifdef TRISC_MATH
namespace sfpu {
inline void _gdn_mul_scalar_(uint32_t data, uint32_t scalar, uint32_t output) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    _broadcast_scalar_from_dest_(scalar*DEST_TILE_SIZE_RAW);
    TTI_SFPMOV(0,LREG_BCAST,LREG_SCALAR_INV_STD,0);
    for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {
        const uint32_t left=half*FACE2_BASE+band*ROW_BAND_STRIDE;
        const uint32_t right=left+FACE1_BASE;
        const uint32_t db=data*DEST_TILE_SIZE_RAW, ob=output*DEST_TILE_SIZE_RAW;
        TT_SFPLOAD(LREG_DATA0,IM,ADDR_MOD_7,db+left);
        TT_SFPLOAD(LREG_DATA1,IM,ADDR_MOD_7,db+left+ODD_COLS_OFFSET);
        TT_SFPLOAD(LREG_DATA2,IM,ADDR_MOD_7,db+right);
        TT_SFPLOAD(LREG_DATA3,IM,ADDR_MOD_7,db+right+ODD_COLS_OFFSET);
        TTI_SFPMUL(LREG_DATA0,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA0,0);
        TTI_SFPMUL(LREG_DATA1,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA1,0);
        TTI_SFPMUL(LREG_DATA2,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA2,0);
        TTI_SFPMUL(LREG_DATA3,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA3,0);
        TT_SFPSTORE(LREG_DATA0,IM,ADDR_MOD_7,ob+left);
        TT_SFPSTORE(LREG_DATA1,IM,ADDR_MOD_7,ob+left+ODD_COLS_OFFSET);
        TT_SFPSTORE(LREG_DATA2,IM,ADDR_MOD_7,ob+right);
        TT_SFPSTORE(LREG_DATA3,IM,ADDR_MOD_7,ob+right+ODD_COLS_OFFSET);
    }
}

// Replicate row0 without loading a tile of ones and multiplying by it.
// Native row-broadcast's SFPTRANSP mapping is retained exactly.
inline void _gdn_copy_row_(uint32_t data,uint32_t row,uint32_t output) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    const uint32_t rb=row*DEST_TILE_SIZE_RAW, ob=output*DEST_TILE_SIZE_RAW;
    TT_SFPLOAD(p_sfpu::LREG4,IM,ADDR_MOD_7,rb+COL_GROUP_OFFSETS[0]);
    TT_SFPLOAD(p_sfpu::LREG5,IM,ADDR_MOD_7,rb+COL_GROUP_OFFSETS[1]);
    TT_SFPLOAD(p_sfpu::LREG6,IM,ADDR_MOD_7,rb+COL_GROUP_OFFSETS[2]);
    TT_SFPLOAD(p_sfpu::LREG7,IM,ADDR_MOD_7,rb+COL_GROUP_OFFSETS[3]);
    TTI_SFPNOP;
    TTI_SFPTRANSP(0,0,0,0);
    TTI_SFPMOV(0,p_sfpu::LREG4,p_sfpu::LREG0,0);
    TTI_SFPMOV(0,p_sfpu::LREG4,p_sfpu::LREG1,0);
    TTI_SFPMOV(0,p_sfpu::LREG4,p_sfpu::LREG2,0);
    TTI_SFPMOV(0,p_sfpu::LREG4,p_sfpu::LREG3,0);
    TTI_SFPTRANSP(0,0,0,0);
    for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {
        const uint32_t base=ob+half*FACE2_BASE+band*ROW_BAND_STRIDE;
        TT_SFPSTORE(p_sfpu::LREG0,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[0]);
        TT_SFPSTORE(p_sfpu::LREG1,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[1]);
        TT_SFPSTORE(p_sfpu::LREG2,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[2]);
        TT_SFPSTORE(p_sfpu::LREG3,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[3]);
    }
}


// Only row0 survives the incumbent subtract/multiply/row-broadcast sequence.
// Transpose two four-row bands so LREG0 and LREG4 hold the useful V/pred rows.
// Keep subtraction and beta multiplication as distinct FP32 operations.
inline void _gdn_delta_row_(uint32_t value,uint32_t prediction,uint32_t output,uint32_t beta) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    const uint32_t vb=value*DEST_TILE_SIZE_RAW,pb=prediction*DEST_TILE_SIZE_RAW;
    const uint32_t ob=output*DEST_TILE_SIZE_RAW;
    TT_SFPLOAD(p_sfpu::LREG0,IM,ADDR_MOD_7,vb+COL_GROUP_OFFSETS[0]);
    TT_SFPLOAD(p_sfpu::LREG1,IM,ADDR_MOD_7,vb+COL_GROUP_OFFSETS[1]);
    TT_SFPLOAD(p_sfpu::LREG2,IM,ADDR_MOD_7,vb+COL_GROUP_OFFSETS[2]);
    TT_SFPLOAD(p_sfpu::LREG3,IM,ADDR_MOD_7,vb+COL_GROUP_OFFSETS[3]);
    TT_SFPLOAD(p_sfpu::LREG4,IM,ADDR_MOD_7,pb+COL_GROUP_OFFSETS[0]);
    TT_SFPLOAD(p_sfpu::LREG5,IM,ADDR_MOD_7,pb+COL_GROUP_OFFSETS[1]);
    TT_SFPLOAD(p_sfpu::LREG6,IM,ADDR_MOD_7,pb+COL_GROUP_OFFSETS[2]);
    TT_SFPLOAD(p_sfpu::LREG7,IM,ADDR_MOD_7,pb+COL_GROUP_OFFSETS[3]);
    TTI_SFPNOP;
    TTI_SFPTRANSP(0,0,0,0);
    TTI_SFPMOV(0,p_sfpu::LREG4,p_sfpu::LREG4,1 /* negate */);
    TTI_SFPADD(p_sfpu::LREG0,p_sfpu::LCONST_1,p_sfpu::LREG4,p_sfpu::LREG1,0);
    // Native scalar broadcast touches only LREG0/2/14, preserving delta in1.
    _broadcast_scalar_from_dest_(beta*DEST_TILE_SIZE_RAW);
    TTI_SFPMUL(p_sfpu::LREG1,p_sfpu::LREG0,p_sfpu::LCONST_0,p_sfpu::LREG1,0);
    TTI_SFPNOP;
    // As in _gdn_copy_row_: four identical row vectors, inverse transpose,
    // then the same four slots written to every four-row band.
    TTI_SFPMOV(0,p_sfpu::LREG1,p_sfpu::LREG0,0);
    TTI_SFPMOV(0,p_sfpu::LREG1,p_sfpu::LREG2,0);
    TTI_SFPMOV(0,p_sfpu::LREG1,p_sfpu::LREG3,0);
    TTI_SFPTRANSP(0,0,0,0);
    for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {
        const uint32_t base=ob+half*FACE2_BASE+band*ROW_BAND_STRIDE;
        TT_SFPSTORE(p_sfpu::LREG0,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[0]);
        TT_SFPSTORE(p_sfpu::LREG1,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[1]);
        TT_SFPSTORE(p_sfpu::LREG2,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[2]);
        TT_SFPSTORE(p_sfpu::LREG3,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[3]);
    }
}

// Preserve TWO rounding steps: product=delta*K, then updated=product+state.
// This deliberately uses SFPMUL followed by SFPADD, never SFPMAD/FMA.
inline void _gdn_col_mul_add_(uint32_t data,uint32_t column,uint32_t output,uint32_t residual) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    const uint32_t db=data*DEST_TILE_SIZE_RAW, cb=column*DEST_TILE_SIZE_RAW;
    const uint32_t ob=output*DEST_TILE_SIZE_RAW, rb=residual*DEST_TILE_SIZE_RAW;
    for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {
        const uint32_t left=half*FACE2_BASE+band*ROW_BAND_STRIDE;
        const uint32_t right=left+FACE1_BASE;
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,cb+left);
        lltt::replay(REPLAY_SLOT_BROADCAST,REPLAY_LEN_BROADCAST);
        _broadcast_stage3_with_data_prefetch_(db+left,db+left+ODD_COLS_OFFSET,db+right,db+right+ODD_COLS_OFFSET);
        TTI_SFPMUL(LREG_DATA0,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA0,0);
        TTI_SFPMUL(LREG_DATA1,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA1,0);
        TTI_SFPMUL(LREG_DATA2,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA2,0);
        TTI_SFPMUL(LREG_DATA3,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA3,0);
        // Broadcast is complete; its two scratch LREGs can carry residuals.
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,rb+left);
        TT_SFPLOAD(LREG_TMP,IM,ADDR_MOD_7,rb+left+ODD_COLS_OFFSET);
        TTI_SFPNOP;
        TTI_SFPADD(LREG_DATA0,p_sfpu::LCONST_1,LREG_BCAST,LREG_DATA0,0);
        TTI_SFPADD(LREG_DATA1,p_sfpu::LCONST_1,LREG_TMP,LREG_DATA1,0);
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,rb+right);
        TT_SFPLOAD(LREG_TMP,IM,ADDR_MOD_7,rb+right+ODD_COLS_OFFSET);
        TTI_SFPNOP;
        TTI_SFPADD(LREG_DATA2,p_sfpu::LCONST_1,LREG_BCAST,LREG_DATA2,0);
        TTI_SFPADD(LREG_DATA3,p_sfpu::LCONST_1,LREG_TMP,LREG_DATA3,0);
        TT_SFPSTORE(LREG_DATA0,IM,ADDR_MOD_7,ob+left);
        TT_SFPSTORE(LREG_DATA1,IM,ADDR_MOD_7,ob+left+ODD_COLS_OFFSET);
        TT_SFPSTORE(LREG_DATA2,IM,ADDR_MOD_7,ob+right);
        TT_SFPSTORE(LREG_DATA3,IM,ADDR_MOD_7,ob+right+ODD_COLS_OFFSET);
    }
}

// Decay product, K product and accumulation remain separate FP32 operations.
template<bool ACCUMULATE>
inline void _gdn_decay_predict_(uint32_t data,uint32_t column,uint32_t output,uint32_t scalar,uint32_t segment) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    const uint32_t slot=(segment/2)*FACE1_BASE+(segment%2)*ODD_COLS_OFFSET;
    const uint32_t db=data*DEST_TILE_SIZE_RAW, cb=column*DEST_TILE_SIZE_RAW+slot;
    const uint32_t ob=output*DEST_TILE_SIZE_RAW, rb=ob;
    // First K tile loads the decay; later K tiles retain LREG7 across copy_tile.
    // Column broadcast uses LREG0..6 and leaves this scalar untouched.
    if constexpr(!ACCUMULATE) {
        _broadcast_scalar_from_dest_(scalar*DEST_TILE_SIZE_RAW);
        TTI_SFPMOV(0,LREG_BCAST,LREG_SCALAR_INV_STD,0);
    }
    for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {
        const uint32_t left=half*FACE2_BASE+band*ROW_BAND_STRIDE;
        const uint32_t right=left+FACE1_BASE;
        // Prepared K is already replicated; all SFPU lanes contain this row's K.
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,cb+left);
        TT_SFPLOAD(LREG_DATA0,IM,ADDR_MOD_7,db+left);
        TT_SFPLOAD(LREG_DATA1,IM,ADDR_MOD_7,db+left+ODD_COLS_OFFSET);
        TT_SFPLOAD(LREG_DATA2,IM,ADDR_MOD_7,db+right);
        TT_SFPLOAD(LREG_DATA3,IM,ADDR_MOD_7,db+right+ODD_COLS_OFFSET);
        // Preserve rounded decay in state DST, then reuse those FP32 LREGs.
        TTI_SFPMUL(LREG_DATA0,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA0,0);
        TTI_SFPMUL(LREG_DATA1,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA1,0);
        TTI_SFPMUL(LREG_DATA2,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA2,0);
        TTI_SFPMUL(LREG_DATA3,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA3,0);
        TT_SFPSTORE(LREG_DATA0,IM,ADDR_MOD_7,db+left);
        TT_SFPSTORE(LREG_DATA1,IM,ADDR_MOD_7,db+left+ODD_COLS_OFFSET);
        TT_SFPSTORE(LREG_DATA2,IM,ADDR_MOD_7,db+right);
        TT_SFPSTORE(LREG_DATA3,IM,ADDR_MOD_7,db+right+ODD_COLS_OFFSET);
        TTI_SFPMUL(LREG_DATA0,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA0,0);
        TTI_SFPMUL(LREG_DATA1,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA1,0);
        TTI_SFPMUL(LREG_DATA2,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA2,0);
        TTI_SFPMUL(LREG_DATA3,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA3,0);
        if constexpr(ACCUMULATE) {
        // Broadcast is complete; its two scratch LREGs can carry residuals.
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,rb+left);
        TT_SFPLOAD(LREG_TMP,IM,ADDR_MOD_7,rb+left+ODD_COLS_OFFSET);
        TTI_SFPNOP;
        TTI_SFPADD(LREG_DATA0,p_sfpu::LCONST_1,LREG_BCAST,LREG_DATA0,0);
        TTI_SFPADD(LREG_DATA1,p_sfpu::LCONST_1,LREG_TMP,LREG_DATA1,0);
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,rb+right);
        TT_SFPLOAD(LREG_TMP,IM,ADDR_MOD_7,rb+right+ODD_COLS_OFFSET);
        TTI_SFPNOP;
        TTI_SFPADD(LREG_DATA2,p_sfpu::LCONST_1,LREG_BCAST,LREG_DATA2,0);
        TTI_SFPADD(LREG_DATA3,p_sfpu::LCONST_1,LREG_TMP,LREG_DATA3,0);
        }
        TT_SFPSTORE(LREG_DATA0,IM,ADDR_MOD_7,ob+left);
        TT_SFPSTORE(LREG_DATA1,IM,ADDR_MOD_7,ob+left+ODD_COLS_OFFSET);
        TT_SFPSTORE(LREG_DATA2,IM,ADDR_MOD_7,ob+right);
        TT_SFPSTORE(LREG_DATA3,IM,ADDR_MOD_7,ob+right+ODD_COLS_OFFSET);
    }
}


// Match native calculate_typecast_fp32_to_fp16b's integer sequence exactly.
// DATA stays unrounded for the current Q readout. TMP and MASK are dead scratch;
// packed Q/K direct loads do not use the old column mask in LREG_MASK.
template<uint32_t DATA>
inline void _gdn_store_bf16_rne_(uint32_t address) {
    TTI_SFPMOV(0,DATA,LREG_MASK,0);
    TTI_SFPSHFT((-16)&0xFFF,DATA,LREG_TMP,5);
    TTI_SFPAND(0,p_sfpu::LREG12,LREG_TMP,0);
    TTI_SFPIADD(0,p_sfpu::LREG13,LREG_MASK,sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(0,LREG_TMP,LREG_MASK,sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPAND(0,p_sfpu::LREG14,LREG_MASK,0);
    TT_SFPSTORE(LREG_MASK,InstrModLoadStore::FP32,ADDR_MOD_7,address);
}

// Preserve update=(delta*K)+decayed and readout=(update*Q)+accumulator.
// Rounded updated values remain live across Q broadcast; persistence stores
// fill that broadcast's four otherwise-idle dependent-shift latency slots.
template<bool ACCUMULATE>
inline void _gdn_update_readout_(uint32_t state,uint32_t key,uint32_t output,uint32_t delta,uint32_t query,uint32_t segment) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    constexpr bool STATE_BF16=get_compile_time_arg_val(1);
    const uint32_t slot=(segment/2)*FACE1_BASE+(segment%2)*ODD_COLS_OFFSET;
    const uint32_t db=delta*DEST_TILE_SIZE_RAW, kb=key*DEST_TILE_SIZE_RAW+slot;
    const uint32_t sb=state*DEST_TILE_SIZE_RAW, qb=query*DEST_TILE_SIZE_RAW+slot, ob=output*DEST_TILE_SIZE_RAW;
    for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {
        const uint32_t left=half*FACE2_BASE+band*ROW_BAND_STRIDE;
        const uint32_t right=left+FACE1_BASE;
        // Prepared K is already replicated; all SFPU lanes contain this row's K.
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,kb+left);
        TT_SFPLOAD(LREG_DATA0,IM,ADDR_MOD_7,db+left);
        TT_SFPLOAD(LREG_DATA1,IM,ADDR_MOD_7,db+left+ODD_COLS_OFFSET);
        TT_SFPLOAD(LREG_DATA2,IM,ADDR_MOD_7,db+right);
        TT_SFPLOAD(LREG_DATA3,IM,ADDR_MOD_7,db+right+ODD_COLS_OFFSET);
        TTI_SFPMUL(LREG_DATA0,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA0,0);
        TTI_SFPMUL(LREG_DATA1,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA1,0);
        TTI_SFPMUL(LREG_DATA2,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA2,0);
        TTI_SFPMUL(LREG_DATA3,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA3,0);
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,sb+left);
        TT_SFPLOAD(LREG_TMP,IM,ADDR_MOD_7,sb+left+ODD_COLS_OFFSET);
        TTI_SFPNOP;
        TTI_SFPADD(LREG_DATA0,p_sfpu::LCONST_1,LREG_BCAST,LREG_DATA0,0);
        TTI_SFPADD(LREG_DATA1,p_sfpu::LCONST_1,LREG_TMP,LREG_DATA1,0);
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,sb+right);
        TT_SFPLOAD(LREG_TMP,IM,ADDR_MOD_7,sb+right+ODD_COLS_OFFSET);
        TTI_SFPNOP;
        TTI_SFPADD(LREG_DATA2,p_sfpu::LCONST_1,LREG_BCAST,LREG_DATA2,0);
        TTI_SFPADD(LREG_DATA3,p_sfpu::LCONST_1,LREG_TMP,LREG_DATA3,0);
        // Q is also pre-expanded. Four independent state stores preserve
        // updated FP32 values and cover the loaded-Q dependency latency.
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,qb+left);
        if constexpr(STATE_BF16) {
            _gdn_store_bf16_rne_<LREG_DATA0>(sb+left);
            _gdn_store_bf16_rne_<LREG_DATA1>(sb+left+ODD_COLS_OFFSET);
            _gdn_store_bf16_rne_<LREG_DATA2>(sb+right);
            _gdn_store_bf16_rne_<LREG_DATA3>(sb+right+ODD_COLS_OFFSET);
        } else {
        TT_SFPSTORE(LREG_DATA0,IM,ADDR_MOD_7,sb+left);
        TT_SFPSTORE(LREG_DATA1,IM,ADDR_MOD_7,sb+left+ODD_COLS_OFFSET);
        TT_SFPSTORE(LREG_DATA2,IM,ADDR_MOD_7,sb+right);
        TT_SFPSTORE(LREG_DATA3,IM,ADDR_MOD_7,sb+right+ODD_COLS_OFFSET);
        }
        TTI_SFPMUL(LREG_DATA0,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA0,0);
        TTI_SFPMUL(LREG_DATA1,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA1,0);
        TTI_SFPMUL(LREG_DATA2,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA2,0);
        TTI_SFPMUL(LREG_DATA3,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA3,0);
        if constexpr(ACCUMULATE) {
            TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,ob+left);
            TT_SFPLOAD(LREG_TMP,IM,ADDR_MOD_7,ob+left+ODD_COLS_OFFSET);
            TTI_SFPNOP;
            TTI_SFPADD(LREG_DATA0,p_sfpu::LCONST_1,LREG_BCAST,LREG_DATA0,0);
            TTI_SFPADD(LREG_DATA1,p_sfpu::LCONST_1,LREG_TMP,LREG_DATA1,0);
            TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,ob+right);
            TT_SFPLOAD(LREG_TMP,IM,ADDR_MOD_7,ob+right+ODD_COLS_OFFSET);
            TTI_SFPNOP;
            TTI_SFPADD(LREG_DATA2,p_sfpu::LCONST_1,LREG_BCAST,LREG_DATA2,0);
            TTI_SFPADD(LREG_DATA3,p_sfpu::LCONST_1,LREG_TMP,LREG_DATA3,0);
        }
        TT_SFPSTORE(LREG_DATA0,IM,ADDR_MOD_7,ob+left);
        TT_SFPSTORE(LREG_DATA1,IM,ADDR_MOD_7,ob+left+ODD_COLS_OFFSET);
        TT_SFPSTORE(LREG_DATA2,IM,ADDR_MOD_7,ob+right);
        TT_SFPSTORE(LREG_DATA3,IM,ADDR_MOD_7,ob+right+ODD_COLS_OFFSET);
    }
}


// Replicate each prepared column-zero value across its tile with the same
// mask/rotate-add broadcast used by the incumbent's per-state-tile path.
// Each segment occupies one eight-column slot; four segments fill one tile.
inline void _gdn_expand_column_(uint32_t data,uint32_t unused,uint32_t output,uint32_t segment) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    const uint32_t slot=(segment/2)*FACE1_BASE+(segment%2)*ODD_COLS_OFFSET;
    const uint32_t db=data*DEST_TILE_SIZE_RAW,ob=output*DEST_TILE_SIZE_RAW+slot;
    for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {
        const uint32_t left=half*FACE2_BASE+band*ROW_BAND_STRIDE;
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,db+left);
        lltt::replay(REPLAY_SLOT_BROADCAST,REPLAY_LEN_BROADCAST);
        _broadcast_stage3_preserve_data_();
        TT_SFPSTORE(LREG_BCAST,IM,ADDR_MOD_7,ob+left);
    }
}

} // namespace sfpu
#endif
template<bool INIT=true>
ALWI void gdn_mul_scalar(uint32_t data,uint32_t scalar) {
    if constexpr(INIT) sfpu_normalize_bcast_scalar_init();
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_mul_scalar_,data,scalar,data,VectorMode::None)));
}

ALWI void gdn_copy_row(uint32_t data) {
    sfpu_bcast_row_init();
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_copy_row_,data,data,data,VectorMode::None)));
}


ALWI void gdn_delta_row(uint32_t value,uint32_t prediction,uint32_t output,uint32_t beta) {
    sfpu_bcast_row_init();
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_delta_row_,value,prediction,output,VectorMode::None,beta)));
}

ALWI void gdn_col_mul_add(uint32_t data,uint32_t column,uint32_t residual) {
    sfpu_bcast_col_init();
    MATH((::ckernel::_sfpu_binary_check_<DST_SYNC_MODE>(data,residual,data,VectorMode::None),
        SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_col_mul_add_,data,column,data,VectorMode::None,residual)));
}

template<bool INIT=true>
ALWI void gdn_col_mul_add_to(uint32_t data,uint32_t column,uint32_t residual,uint32_t output) {
    if constexpr(INIT) sfpu_bcast_col_init();
    MATH((::ckernel::_sfpu_binary_check_<DST_SYNC_MODE>(data,residual,output,VectorMode::None),
        SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_col_mul_add_,data,column,output,VectorMode::None,residual)));
}


// Native broadcast implementation supports distinct input/output registers;
// use that directly instead of copying a retained state before multiplying it.
template<bool INIT=true>
ALWI void gdn_col_mul_to(uint32_t data,uint32_t column,uint32_t output) {
    if constexpr(INIT) sfpu_bcast_col_init();
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,
        _calculate_sfpu_binary_bcast_full_tile_,(ckernel::BinaryOp::MUL,ckernel::BroadcastType::COL),
        data,column,output,VectorMode::None)));
}
ALWI void gdn_copy_row_to(uint32_t row,uint32_t output) {
    sfpu_bcast_row_init();
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_copy_row_,row,row,output,VectorMode::None)));
}

template<bool ACCUMULATE, bool INIT=true>
ALWI void gdn_decay_predict(uint32_t data,uint32_t column,uint32_t output,uint32_t scalar,uint32_t segment) {
    if constexpr(INIT) sfpu_bcast_col_init();
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_decay_predict_,(ACCUMULATE),data,column,output,VectorMode::None,scalar,segment)));
}


template<bool ACCUMULATE>
ALWI void gdn_update_readout(uint32_t state,uint32_t key,uint32_t output,uint32_t delta,uint32_t query,uint32_t segment) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_update_readout_,(ACCUMULATE),state,key,output,VectorMode::None,delta,query,segment)));
}


ALWI void gdn_expand_column(uint32_t tile,uint32_t output,uint32_t segment) {
    sfpu_bcast_col_init();
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_expand_column_,tile,tile,output,VectorMode::None,segment)));
}

} // namespace ckernel

void load(uint32_t cb,uint32_t tile,uint32_t dst) {
    copy_tile(cb,tile,dst);
}
void finish(uint32_t cb,uint32_t tile) {
    tile_regs_commit(); tile_regs_wait(); pack_tile<true>(0,cb,tile); tile_regs_release();
}
// Exactly the accurate FP32 reduction topology: fold the four H tiles with
// SFPU addition, then SFPU reduce their32 rows into row0. No TF32 FPU path.
void contract(uint32_t matrix,uint32_t column,uint32_t result) {
    CircularBuffer dst(result); dst.reserve_back(4);
    PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_COL,PackMode::Default>(result)));
    for(uint32_t n=0;n<4;n++) {
        tile_regs_acquire();
        for(uint32_t m=0;m<4;m++) {
            const uint32_t work=m==0 ? 0 : 2;
            load(matrix,m*4+n,work); load(column,m,1);
            sfpu_mul_bcast_col_init(); sfpu_mul_bcast_col(work,1);
            if(m) { add_binary_tile_init(); add_binary_tile(0,2,0); }
        }
        sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_COL>(0,1,1);
        finish(result,n);
    }
    PACK((llk_pack_reduce_mask_clear())); dst.push_back(4);
}

// Raw mode matches external FP32 square/sum(+epsilon)/rsqrt/multiply, then
// separate Q scaling. Normalized mode is a pure transpose control.
void prepare_vector(uint32_t input,uint32_t output,bool is_q) {
    constexpr bool NORMALIZED=get_compile_time_arg_val(0);
    CircularBuffer in(input),col(output),norm(16),rows(17);
    in.wait_front(4);
    copy_init(input);
    uint32_t transposed_source=input;
    if constexpr(!NORMALIZED) {
        norm.reserve_back(1); tile_regs_acquire();
        for(uint32_t i=0;i<4;i++) {
            const uint32_t work=i?2:0;
            load(input,i,work); load(input,i,1);
            mul_binary_tile_init(); mul_binary_tile(work,1,work);
            if(i) { add_binary_tile_init(); add_binary_tile(0,2,0); }
        }
        sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_ROW>(0,1,1);
        // Float32 encodings of1e-6 and128**-0.5. Scaling remains separate.
        binop_with_scalar_tile_init(); add_unary_tile(0,0x358637bdu);
        rsqrt_tile_init(); rsqrt_tile(0);
        finish(16,0); norm.push_back(1); norm.wait_front(1);
        rows.reserve_back(4);
        for(uint32_t i=0;i<4;i++) {
            tile_regs_acquire(); load(input,i,0); load(16,0,1);
            gdn_mul_scalar(0,1);
            if(is_q) { binop_with_scalar_tile_init(); mul_unary_tile(0,0x3db504f3u); }
            finish(17,i);
        }
        rows.push_back(4); rows.wait_front(4); norm.pop_front(1);
        transposed_source=17;
    }
    // Four independent 32-row segments fill four eight-column slots of DST1.
    // Preserve this output across transposes; each transpose writes only DST0.
    col.reserve_back(1); transpose_init(transposed_source); tile_regs_acquire();
    for(uint32_t i=0;i<4;i++) {
        transpose_tile(transposed_source,i,0); gdn_expand_column(0,1,i);
    }
    tile_regs_commit(); tile_regs_wait(); pack_tile<true>(1,output,0); tile_regs_release();
    col.push_back(1); in.pop_front(4);
    if constexpr(!NORMALIZED) rows.pop_front(4);
}


// Per value-column full-DST allocation: state0..3, work4, Q/K5,
// accumulator/delta6, decay/beta7. No retained state crosses a release.
void kernel_main() {
#ifdef TRISC_MATH
    static_assert(get_dest_max_tiles<DST_SYNC_MODE,true,DstTileShape::Tile32x32>()>=8,
                  "Retained GDN needs eight FP32 DST tiles and full synchronization");
#endif
    const uint32_t count=get_arg_val<uint32_t>(0);
    const uint32_t start_head=get_arg_val<uint32_t>(1);
    constexpr bool STATE_BF16=get_compile_time_arg_val(1);
    constexpr bool GROUP_QK=get_compile_time_arg_val(2);
    constexpr bool FUSE_ACCUMULATOR=get_compile_time_arg_val(3);
    constexpr bool HOIST_DELTA=get_compile_time_arg_val(4);
    constexpr bool HOIST_SCALAR_INIT=get_compile_time_arg_val(5);
    constexpr bool HOIST_COL_INIT=get_compile_time_arg_val(6);
    constexpr bool FUSE_DECAY_PREDICT=get_compile_time_arg_val(7);
    static_assert(!FUSE_DECAY_PREDICT || FUSE_ACCUMULATOR,
                  "Decay prediction fusion requires ordered fused accumulation");
    static_assert(FUSE_DECAY_PREDICT && FUSE_ACCUMULATOR && HOIST_DELTA,
                  "Replicated Q/K candidate requires the active fused recurrence path");
    compute_kernel_hw_startup(STATE_BF16?3:5,6,12);
    copy_init(STATE_BF16?3:5);
    CircularBuffer q(13),k(14),v(2),raw_g(3),decay(18),beta(4),state(5),zero(6),one(7);
    CircularBuffer updated(11),out(12),raw_q(0),raw_k(1);
    zero.wait_front(1); one.wait_front(1);
    for(uint32_t head=0;head<count;head++) {
        state.wait_front(16); raw_g.wait_front(1);
        decay.reserve_back(1); tile_regs_acquire(); load(3,0,0);
        exp_tile_init<false>(); exp_tile<false>(0); finish(18,0);
        decay.push_back(1); decay.wait_front(1); raw_g.pop_front(1);
        // Core boundaries may split triples. Prepare at the local boundary or
        // global group start, and retain only while the same Q/K group continues.
        if (!GROUP_QK || head==0 || (start_head+head)%3==0) {
            prepare_vector(0,13,true); prepare_vector(1,14,false);
        } else {
            // Reader's rings remain one head deep: consume every duplicate input
            // to permit it to fetch the next state/value/gate without stalling.
            raw_q.wait_front(4); raw_k.wait_front(4);
            raw_q.pop_front(4); raw_k.pop_front(4);
        }
        copy_init(STATE_BF16?3:5);
        q.wait_front(1); k.wait_front(1); v.wait_front(4); beta.wait_front(1);
        updated.reserve_back(16); out.reserve_back(4);
        for(uint32_t n=0;n<4;n++) {
            tile_regs_acquire();
            load(18,0,7);
            if constexpr(STATE_BF16) {
                reconfig_data_format_srca(5); copy_init(5);
            }
            // copy_tile does not alter SFPU config, MOD7, LREGs or replay.
            if constexpr(HOIST_SCALAR_INIT && !FUSE_DECAY_PREDICT) sfpu_normalize_bcast_scalar_init();
            for(uint32_t m=0;m<4;m++) {
                load(5,m*4+n,m);
                if constexpr(!FUSE_DECAY_PREDICT) gdn_mul_scalar<!HOIST_SCALAR_INIT>(m,7);
            }
            if constexpr(STATE_BF16) {
                reconfig_data_format_srca(3); copy_init(3);
            }
            // One packed K tile survives the four ordered prediction/update segments.
            load(14,0,5);
            // Preserve the baseline four-K-tile accumulation order exactly.
            if constexpr(HOIST_COL_INIT && FUSE_ACCUMULATOR) sfpu_bcast_col_init();
            for(uint32_t m=0;m<4;m++) {
                if constexpr(FUSE_DECAY_PREDICT) {
                    if(m) gdn_decay_predict<true,!HOIST_COL_INIT>(m,5,6,7,m);
                    else gdn_decay_predict<false,!HOIST_COL_INIT>(m,5,6,7,m);
                } else if constexpr(FUSE_ACCUMULATOR) {
                    if(m) {
                        // Exact separate FP32 product then add into accumulator.
                        // Skip the full temporary-tile store/reload/add pass.
                        gdn_col_mul_add_to<!HOIST_COL_INIT>(m,5,6,6);
                    } else { gdn_col_mul_to<!HOIST_COL_INIT>(m,5,6); }
                } else {
                    const uint32_t work=m?4:6;
                    gdn_col_mul_to(m,5,work);
                    if(m) { add_binary_tile_init(); add_binary_tile(6,4,6); }
                }
            }
            sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
            sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_COL>(6,1,1);
            load(2,n,4); load(4,0,7);
            gdn_delta_row(4,6,4,7);
            // DST7's beta is dead after delta; reuse it for Q beside K in DST5.
            // Each m keeps exactly the old update and readout arithmetic, while
            // eliminating an intervening updated-state store/reload traversal.
            static_assert(HOIST_DELTA && FUSE_ACCUMULATOR,
                          "Fused update/readout requires hoisted delta and ordered accumulation");
            load(13,0,7);
            if constexpr(STATE_BF16) {
                // Establish exactly the native RNE integer constants. Column
                // init below restores the zero-stride MOD7 without changing them.
                typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();
            }
            sfpu_bcast_col_init();
            for(uint32_t m=0;m<4;m++) {
                if(m) gdn_update_readout<true>(m,5,6,4,7,m);
                else gdn_update_readout<false>(m,5,6,4,7,m);
            }
            sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
            sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_COL>(6,1,1);
            // Persistent state was rounded from a copy of each live update.
            tile_regs_commit(); tile_regs_wait();
            PACK((llk_pack_reduce_mask_clear()));
            // Q readout is already complete in FP32 DST6. Only persistence rounds.
            if constexpr(STATE_BF16) pack_reconfig_data_format(12,11);
            for(uint32_t m=0;m<4;m++) pack_tile<true>(m,11,m*4+n);
            if constexpr(STATE_BF16) pack_reconfig_data_format(11,12);
            PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_COL,PackMode::Default>(12)));
            pack_tile<true>(6,12,n);
            PACK((llk_pack_reduce_mask_clear()));
            tile_regs_release();
        }
        updated.push_back(16); out.push_back(4);
        state.pop_front(16); decay.pop_front(1); v.pop_front(4); beta.pop_front(1);
        if (!GROUP_QK || head+1==count || (start_head+head)%3==2) {
            q.pop_front(1); k.pop_front(1);
        }
    }
    zero.pop_front(1); one.pop_front(1);
}
"""


def _fused_delta_row_head_partition(batch, available_cores, cores=None, grouped_qk=True):
    """Pure host partition: (grouped, [(global_head_start, head_count), ...])."""
    assert batch >= 1 and available_cores >= 1
    assert cores is None or cores >= 1
    grouped = bool(grouped_qk and batch >= 3)
    # Balance individual heads; compute still caches within global triples.
    unit = 1
    units = batch * 48 // unit
    ncores = min(units, available_cores, cores or available_cores)
    result = []
    start = 0
    for index in range(ncores):
        count = unit * (units // ncores + (index < units % ncores))
        result.append((start, count))
        start += count
    assert start == batch * 48
    return grouped, result


def _fused_delta_row_recurrent_sfpu_compact(
    q, k, v, g, beta, state, *, normalized=False, packed_qkv=False, output=None, cores=None, grouped_qk=True
):
    """Accept raw RM[B,1,C] directly, or the original TILE interface."""
    row_major = q.layout == ttnn.ROW_MAJOR_LAYOUT
    assert not (row_major and normalized)
    assert not packed_qkv or row_major
    assert state.dtype in (ttnn.float32, ttnn.bfloat16)
    state_bf16 = state.dtype == ttnn.bfloat16
    state_bytes = 2048 if state_bf16 else 4096
    batch = state.shape[0] // 48
    assert tuple(state.shape) == (batch * 48, 128, 128)
    assert q.dtype == k.dtype == (ttnn.float32 if normalized else ttnn.bfloat16)
    assert v.dtype == ttnn.bfloat16
    assert tuple(q.shape) == tuple(k.shape)
    if packed_qkv:
        assert tuple(q.shape) == tuple(k.shape) == tuple(v.shape) == (batch, 1, 10240)
        assert q.buffer_address() == k.buffer_address() == v.buffer_address()
    elif normalized:
        assert tuple(q.shape) == (batch, 16, 128)
    else:
        assert q.shape[0] == batch and q.shape[-1] == 2048 and q.shape[1] in (1, 32)
    if not packed_qkv:
        assert v.shape[0] == batch and v.shape[-1] == 6144 and v.shape[1] in (1, 32)
    assert tuple(g.shape) == tuple(beta.shape) == (batch, 1, 48)
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
            (1, batch * 48, 128),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=state.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    assert tuple(output.shape) == (1, batch * 48, 128)
    grid = state.device().compute_with_storage_grid_size()
    group_qk, partition = _fused_delta_row_head_partition(batch, grid.x * grid.y, cores, grouped_qk)
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
        64,
        192,
        int(row_major),
        int(packed_qkv),
        state_bytes,
    ]
    for t in inputs:
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct = [state_bytes]
    for t in (state, output):
        wct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct.append(batch * 48)

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
        kernel(_fused_delta_row_GDN_COMPACT_READER, ttnn.ReaderConfigDescriptor(), rr, rct),
        kernel(_fused_delta_row_GDN_COMPACT_WRITER, ttnn.WriterConfigDescriptor(), wr, wct),
        kernel(
            _fused_delta_row_GDN_COMPACT_COMPUTE,
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


_affinev3__GDN_AFFINE_GATE_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
void kernel_main() {
    constexpr auto na=TensorAccessorArgs<0>();
    constexpr auto wa=TensorAccessorArgs<na.next_compile_time_args_offset()>();
    constexpr auto pa=TensorAccessorArgs<wa.next_compile_time_args_offset()>();
    const auto norm=TensorAccessor(na,get_arg_val<uint32_t>(0),2048);
    const auto weight=TensorAccessor(wa,get_arg_val<uint32_t>(1),2048);
    const auto packed=TensorAccessor(pa,get_arg_val<uint32_t>(2),2048);
    const uint32_t batch=get_arg_val<uint32_t>(3),start=get_arg_val<uint32_t>(4),count=get_arg_val<uint32_t>(5);
    Noc noc; CircularBuffer n(0),w(1),gate(2),scratch(3);
    scratch.reserve_back(2);
    const uint32_t sb=(scratch.get_write_ptr()+63u)&~63u;
    for(uint32_t tile=start;tile<start+count;tile++) {
        n.reserve_back(1); w.reserve_back(1); gate.reserve_back(1);
        const uint32_t head=tile/4,column=tile%4,np=n.get_write_ptr();
        for(uint32_t user=0;user<batch;user++) {
            const uint32_t source_head=user*48+head,row=source_head%32;
            const uint32_t source_tile=(source_head/32)*4+column;
            for(uint32_t face=0;face<2;face++) {
                const uint32_t src=((row/16)*512+face*256+(row%16)*16)*2;
                const uint32_t dst=(user/16)*1024+face*512+(user%16)*32;
                // A32B DRAM read is legal when endpoint modulo64 agrees.
                // Source-head parity is constant, destination-user parity is
                // not, so only the matching half can bypass staging.
                if((src&63u)==((np+dst)&63u)) {
                    noc.async_read(norm,n,32,{.page_id=source_tile,.offset_bytes=src},{.offset_bytes=dst});
                } else {
                    noc.async_read(norm,CoreLocalMem<uint32_t>(sb+user*128+face*64),64,
                        {.page_id=source_tile,.offset_bytes=src&~63u},{});
                }
            }
        }
        for(uint32_t face=0;face<2;face++)
            noc.async_read(weight,CoreLocalMem<uint32_t>(sb+4096+face*64),64,
                {.page_id=column,.offset_bytes=face*512},{});
        noc.async_read(packed,gate,2048,{.page_id=320+tile},{.offset_bytes=0});
        noc.async_read_barrier();
        // Local NoC copies replace all baby-RISC per-element loops.
        for(uint32_t user=0;user<batch;user++) for(uint32_t face=0;face<2;face++) {
            const uint32_t row=(user*48+head)%32;
            const uint32_t src=((row/16)*512+face*256+(row%16)*16)*2;
            const uint32_t dst=(user/16)*1024+face*512+(user%16)*32;
            if((src&63u)!=((np+dst)&63u))
                tt::data_movement::common::tt_memmove<false,true,true,32>(
                    noc,np+dst,sb+user*128+face*64+(src&63u),32);
        }
        // Only gamma row0 is needed: compute broadcasts it in FP32 DST.
        for(uint32_t face=0;face<2;face++)
            tt::data_movement::common::tt_memmove<false,true,true,32>(
                noc,w.get_write_ptr()+face*512,sb+4096+face*64,32);
        for(uint32_t half=0;half<2;half++) {
            const uint32_t first=batch>half*16?batch:half*16,end=(half+1)*16;
            if(first<end) for(uint32_t face=0;face<2;face++) {
                const uint32_t dst=half*1024+face*512+(first%16)*32;
                noc.async_write_zeros(n,(end-first)*32,{.offset_bytes=dst});
                noc.async_write_zeros(gate,(end-first)*32,{.offset_bytes=dst});
            }
        }
        // BH local zero uses read-loopback; this completes copies and pad zeros.
        noc.write_zeros_l1_barrier();
        n.push_back(1); w.push_back(1); gate.push_back(1);
    }
}
"""

_affinev3__GDN_AFFINE_GATE_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto oa=TensorAccessorArgs<0>();
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(0),2048);
    const uint32_t start=get_arg_val<uint32_t>(1),count=get_arg_val<uint32_t>(2);
    Noc noc; CircularBuffer out(4);
    for(uint32_t tile=start;tile<start+count;tile++) {
        out.wait_front(1);
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),output,2048,
            {.offset_bytes=0},{.page_id=tile});
        noc.async_write_barrier(); out.pop_front(1);
    }
}
"""

_affinev3__GDN_AFFINE_GATE_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    const uint32_t count=get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0,4); copy_init(0);
    CircularBuffer norm(0),weight(1),gate(2),out(4);
    for(uint32_t tile=0;tile<count;tile++) {
        norm.wait_front(1); weight.wait_front(1); gate.wait_front(1);
        out.reserve_back(1); tile_regs_acquire();
        copy_tile(0,0,0); copy_tile(1,0,1);
        sfpu_mul_bcast_row_init(); sfpu_mul_bcast_row<true>(0,1);
        // Match the standalone BF16 norm-weight multiplication boundary.
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>(0);
        copy_tile(2,0,2);
        // Original gate is cast toFP32 before fused SILU/multiply.
        silu_tile_init(); silu_tile<true>(2);
        mul_binary_tile_init(); mul_binary_tile(2,0,0);
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>(0);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0,4); tile_regs_release();
        out.push_back(1); norm.pop_front(1); weight.pop_front(1); gate.pop_front(1);
    }
}
"""


def _affinev3_gdn_affine_gate_flat(normalized, weight, packed, *, output=None, cores=None):
    """[1,B*48,128] BF16 normalized heads -> [1,B,6144] BF16 gated output."""
    assert len(normalized.shape) == 3 and normalized.shape[0] == 1 and normalized.shape[2] == 128
    assert normalized.shape[1] % 48 == 0
    batch = normalized.shape[1] // 48
    assert 1 <= batch <= 32 and tuple(weight.shape) == (1, 1, 128)
    assert len(packed.shape) == 3 and tuple(packed.shape)[:2] == (1, batch) and packed.shape[2] >= 16384
    for tensor in (normalized, weight, packed):
        assert tensor.dtype == ttnn.bfloat16 and tensor.layout == ttnn.TILE_LAYOUT
        assert tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert tuple(normalized.padded_shape) == (1, ((batch * 48 + 31) // 32) * 32, 128)
    assert tuple(weight.padded_shape) == (1, 32, 128)
    assert packed.padded_shape[-2] == 32 and packed.padded_shape[-1] % 32 == 0
    if output is None:
        output = ttnn.empty(
            (1, batch, 6144),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=normalized.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    assert tuple(output.shape) == (1, batch, 6144) and tuple(output.padded_shape) == (1, 32, 6144)
    assert output.dtype == ttnn.bfloat16 and output.layout == ttnn.TILE_LAYOUT
    assert output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert output.buffer_address() not in [t.buffer_address() for t in (normalized, weight, packed)]
    grid = normalized.device().compute_with_storage_grid_size()
    workers = min(192, cores or grid.x * grid.y)
    assert 1 <= workers <= grid.x * grid.y
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    start = 0
    for i, c in enumerate(coords):
        count = 192 // workers + (i < 192 % workers)
        rr[c.x][c.y] = [
            normalized.buffer_address(),
            weight.buffer_address(),
            packed.buffer_address(),
            batch,
            start,
            count,
        ]
        wr[c.x][c.y] = [output.buffer_address(), start, count]
        cr[c.x][c.y] = [count]
        start += count
    rct = []
    for t in (normalized, weight, packed):
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
    kernels = [
        kernel(_affinev3__GDN_AFFINE_GATE_READER, ttnn.ReaderConfigDescriptor(), rr, rct),
        kernel(
            _affinev3__GDN_AFFINE_GATE_WRITER,
            ttnn.WriterConfigDescriptor(),
            wr,
            ttnn.TensorAccessorArgs(output).get_compile_time_args(),
        ),
        kernel(_affinev3__GDN_AFFINE_GATE_COMPUTE, config, cr, []),
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=(8192 if i == 3 else 2048),
            core_ranges=ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=i,
                    data_format=ttnn.float32 if i == 3 else ttnn.bfloat16,
                    page_size=4096 if i == 3 else 2048,
                )
            ],
        )
        for i in range(5)
    ]
    ttnn.generic_op(
        [normalized, weight, packed, output], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    )
    return output


_gate178_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto pa=TensorAccessorArgs<0>();
    constexpr auto da=TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto na=TensorAccessorArgs<da.next_compile_time_args_offset()>();
    auto packed=TensorAccessor(pa,get_arg_val<uint32_t>(0),2048);
    auto bias=TensorAccessor(da,get_arg_val<uint32_t>(1),4096);
    auto neg=TensorAccessor(na,get_arg_val<uint32_t>(2),4096);
    const uint32_t task=get_arg_val<uint32_t>(3),user=task/2,col=task%2;
    Noc noc; CircularBuffer a(0),b(1),d(2),n(3),scratch(6);
    a.reserve_back(1); b.reserve_back(1); d.reserve_back(1); n.reserve_back(1); scratch.reserve_back(1);
    noc.async_write_zeros(a,2048);noc.async_write_zeros(b,4096);
    noc.async_write_zeros(d,4096);noc.async_write_zeros(n,4096);
    noc.write_zeros_l1_barrier();
    const uint32_t sb=(scratch.get_write_ptr()+63u)&~63u;
    for(uint32_t face=0;face<2;face++) {
        const uint32_t src=(user/16)*1024+face*512+(user%16)*32;
        noc.async_read(packed,CoreLocalMem<uint32_t>(sb+face*64),64,
            {.page_id=514+col,.offset_bytes=src&~63u},{});
        noc.async_read(packed,CoreLocalMem<uint32_t>(sb+128+face*64),64,
            {.page_id=512+col,.offset_bytes=src&~63u},{});
        noc.async_read(bias,d,64,{.page_id=col,.offset_bytes=face*1024},{.offset_bytes=face*1024});
        noc.async_read(neg,n,64,{.page_id=col,.offset_bytes=face*1024},{.offset_bytes=face*1024});
    }
    noc.async_read_barrier();
    // Exact bitwise BF16 widening, only32 values per tile and no floating math.
    auto ap=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(a.get_write_ptr());
    auto bp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b.get_write_ptr());
    auto src=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sb);
    for(uint32_t face=0;face<2;face++) for(uint32_t i=0;i<16;i++) {
        ap[face*256+i]=src[face*32+(user%2)*16+i];
        bp[face*256+i]=uint32_t(src[64+face*32+(user%2)*16+i])<<16;
    }
    a.push_back(1);b.push_back(1);d.push_back(1);n.push_back(1);
}
"""
_gate178_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto ba=TensorAccessorArgs<0>();
    constexpr auto ga=TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    auto beta=TensorAccessor(ba,get_arg_val<uint32_t>(0),2048);
    auto decay=TensorAccessor(ga,get_arg_val<uint32_t>(1),4096);
    const uint32_t task=get_arg_val<uint32_t>(2);
    Noc noc; CircularBuffer b(4),g(5);b.wait_front(1);g.wait_front(1);
    noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(b),beta,2048,{}, {.page_id=task});
    noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(g),decay,4096,{}, {.page_id=task});
    noc.async_write_barrier();b.pop_front(1);g.pop_front(1);
}
"""
_gate178_COMPUTE = r"""
// Required by native softplus: otherwise it selects the BF16 polynomial.
#define INP_FLOAT32 1
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
namespace ckernel {
ALWI void native_bf16_sigmoid_in_fp32_dst(uint32_t idst) {
    // Native BF16 sigmoid math/RNE, with FP32 DST storage.
    // Each SFPU vector covers4rows x8columns. Two iterations cover16
    // face columns; VectorMode::R visits both top faces (logical32columns).
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,true,calculate_sigmoid,(false,false,2),idst,VectorMode::R));
}
ALWI void native_fp32_softplus_row(uint32_t idst) {
    // Same native polynomial and constants; Face0/1, both 8-column groups in the first 4-row band.
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,true,calculate_softplus,(false,true,2),
        idst,VectorMode::R,0x3f800000u,0x3f800000u,0x41a00000u));
}
}
void kernel_main() {
    compute_kernel_hw_startup(0,2,5);
    constexpr uint32_t stage=get_compile_time_arg_val(0);
    add_init(0,2);
    CircularBuffer a(0),b(1),d(2),n(3),beta(4),g(5);
    a.wait_front(1);b.wait_front(1);d.wait_front(1);n.wait_front(1);
    beta.reserve_back(1);g.reserve_back(1);tile_regs_acquire();
    // Native ADD defaults to FPU for mixed BF16 / FP32 input. Preserve its
    // source-register conversion, instead of silently upgrading this boundary.
    add_tiles(0,2,0,0,0);
    if constexpr(stage!=1) {
        softplus_tile_init<true>();native_fp32_softplus_row(0);
    }
    reconfig_data_format_srca(0,3);copy_init(3);
    if constexpr(stage==0) {
        copy_tile(3,0,1);mul_binary_tile_init();mul_binary_tile(1,0,0);
    }
    copy_tile(1,0,1);sigmoid_tile_init<false>();native_bf16_sigmoid_in_fp32_dst(1);
    tile_regs_commit();tile_regs_wait();
    pack_tile(0,5);pack_reconfig_data_format(5,4);pack_tile(1,4);tile_regs_release();
    g.push_back(1);beta.push_back(1);a.pop_front(1);b.pop_front(1);d.pop_front(1);n.pop_front(1);
}
"""


def _gate178_gdn_beta_decay(packed, dt_bias, neg_a, *, outputs=None, diagnostic_stage=0):
    """Return beta BF16, decay FP32 [B,1,48], preserving native decode mapping."""
    assert tuple(packed.shape)[:1] == (1,) and len(packed.shape) == 3
    assert diagnostic_stage in (0, 1, 2)
    batch = packed.shape[1]
    assert 1 <= batch <= 32 and packed.shape[2] >= 16496
    assert packed.dtype == ttnn.bfloat16 and packed.padded_shape[-2] == 32
    for tensor in (packed, dt_bias, neg_a):
        assert tensor.layout == ttnn.TILE_LAYOUT and tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    for tensor in (dt_bias, neg_a):
        assert tuple(tensor.shape) == (1, 1, 48) and tensor.dtype == ttnn.float32
    if outputs is None:
        outputs = tuple(
            ttnn.empty(
                (batch, 1, 48),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=packed.device(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for dtype in (ttnn.bfloat16, ttnn.float32)
        )
    beta, g = outputs
    for tensor, dtype in zip(outputs, (ttnn.bfloat16, ttnn.float32)):
        assert tuple(tensor.shape) == (batch, 1, 48) and tuple(tensor.padded_shape) == (batch, 32, 64)
        assert tensor.dtype == dtype and tensor.layout == ttnn.TILE_LAYOUT
        assert tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert len({t.buffer_address() for t in (packed, dt_bias, neg_a, beta, g)}) == 5
    grid = packed.device().compute_with_storage_grid_size()
    assert grid.x * grid.y >= 2 * batch
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(2 * batch)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    for task, c in enumerate(coords):
        rr[c.x][c.y] = [packed.buffer_address(), dt_bias.buffer_address(), neg_a.buffer_address(), task]
        wr[c.x][c.y] = [beta.buffer_address(), g.buffer_address(), task]
        cr[c.x][c.y] = []

    def args(tensors):
        result = []
        for t in tensors:
            result.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        return result

    def kernel(source, config, runtime, ct):
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=runtime,
            config=config,
        )

    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for i in (1, 3):
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    kernels = [
        kernel(_gate178_READER, ttnn.ReaderConfigDescriptor(), rr, args((packed, dt_bias, neg_a))),
        kernel(_gate178_WRITER, ttnn.WriterConfigDescriptor(), wr, args(outputs)),
        kernel(_gate178_COMPUTE, config, cr, [diagnostic_stage]),
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=2048 if i in (0, 4) else 4096,
            core_ranges=ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=i,
                    data_format=ttnn.bfloat16 if i in (0, 4) else ttnn.float32,
                    page_size=2048 if i in (0, 4) else 4096,
                )
            ],
        )
        for i in range(7)
    ]
    ttnn.generic_op([packed, dt_bias, neg_a, beta, g], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return beta, g


_norm_zero_meanFOLD_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto xa=TensorAccessorArgs<0>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0),2048);
    const uint32_t block=get_arg_val<uint32_t>(1);
    Noc noc; CircularBuffer data(0); data.reserve_back(5);
    for(uint32_t chunk=0;chunk<5;chunk++)
        noc.async_read(x,data,2048,{.page_id=chunk*32+block},{.offset_bytes=chunk*2048});
    noc.async_read_barrier(); data.push_back(5);
}
"""
_norm_zero_meanFOLD_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto fa=TensorAccessorArgs<0>();
    const auto folded=TensorAccessor(fa,get_arg_val<uint32_t>(0),4096);
    Noc noc; CircularBuffer data(3); data.wait_front(1);
    noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(data),folded,4096,
        {.offset_bytes=0},{.page_id=get_arg_val<uint32_t>(1)});
    noc.async_write_barrier(); data.pop_front(1);
}
"""
_norm_zero_meanFOLD_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    compute_kernel_hw_startup(0,0,3); copy_init(0);
    CircularBuffer x(0),out(3); x.wait_front(5); out.reserve_back(1);
    tile_regs_acquire();
    // Identical scalar arithmetic/order to the incumbent's five-tile fold.
    for(uint32_t i=0;i<5;i++) {
        const uint32_t work=i?2:0;
        copy_tile(0,i,work); copy_tile(0,i,1);
        mul_binary_tile_init(); mul_binary_tile(work,1,work);
        if(i) { add_binary_tile_init(); add_binary_tile(0,2,0); }
    }
    tile_regs_commit(); tile_regs_wait(); pack_tile(0,3); tile_regs_release();
    out.push_back(1); x.pop_front(5);
}
"""
_norm_zero_meanINVERSE_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto fa=TensorAccessorArgs<0>();
    const auto folded=TensorAccessor(fa,get_arg_val<uint32_t>(0),4096);
    const uint32_t user=get_arg_val<uint32_t>(1);
    Noc noc; CircularBuffer dense(0); dense.reserve_back(1);
    for(uint32_t block=0;block<32;block++) for(uint32_t face=0;face<2;face++) {
        // Folded channel d=block*32+face*16 becomes row=block,col=face*16
        // in the same dense face-major tile the old reader constructed.
        const uint32_t source=((user/16)*512+face*256+(user%16)*16)*4;
        const uint32_t target=((block/16)*512+face*256+(block%16)*16)*4;
        noc.async_read(folded,dense,64,{.page_id=block,.offset_bytes=source},{.offset_bytes=target});
    }
    noc.async_read_barrier(); dense.push_back(1);
}
"""
_norm_zero_meanINVERSE_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto ia=TensorAccessorArgs<0>();
    const auto inverse=TensorAccessor(ia,get_arg_val<uint32_t>(0),4096);
    const uint32_t user=get_arg_val<uint32_t>(1),batch=get_arg_val<uint32_t>(2);
    Noc noc; CircularBuffer data(2),zero(4);
    if(user+1==batch && batch<32) {
        zero.reserve_back(1);
        volatile tt_l1_ptr uint32_t* z=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero.get_write_ptr());
        for(uint32_t i=0;i<1024;i++) z[i]=0;
        zero.push_back(1); zero.wait_front(1);
    }
    data.wait_front(1);
    for(uint32_t face=0;face<2;face++) {
        const uint32_t target=((user/16)*512+face*256+(user%16)*16)*4;
        // Only col0 is consumed, but initialize the entire row without
        // read-modify-write or overlap with another user's output.
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(data),inverse,64,
            {.offset_bytes=face*1024},{.page_id=0,.offset_bytes=target});
    }
    if(user+1==batch && batch<32) {
        for(uint32_t row=batch;row<32;) {
            const uint32_t end=(row/16+1)*16;
            for(uint32_t face=0;face<2;face++) {
                const uint32_t target=((row/16)*512+face*256+(row%16)*16)*4;
                noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(zero),inverse,(end-row)*64,
                    {.offset_bytes=0},{.page_id=0,.offset_bytes=target});
            }
            row=end;
        }
    }
    noc.async_write_barrier(); data.pop_front(1);
    if(user+1==batch && batch<32) zero.pop_front(1);
}
"""
_norm_zero_meanINVERSE_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    constexpr uint32_t EPS_BITS=get_compile_time_arg_val(0), RECIPROCAL_BITS=get_compile_time_arg_val(1);
    compute_kernel_hw_startup(0,0,2); copy_init(0);
    CircularBuffer dense(0),inverse(2); dense.wait_front(1); inverse.reserve_back(1);
    tile_regs_acquire(); copy_tile(0,0,0);
    // The tile is bit-identical to DST0 after the incumbent's five-tile fold.
    // Keep both axis order and all scalar operations exactly as in that helper.
    sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_COL>(0,1,1);
    sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_ROW>(0,1,1);
    binop_with_scalar_tile_init(); mul_unary_tile(0,RECIPROCAL_BITS);
    binop_with_scalar_tile_init(); add_unary_tile(0,EPS_BITS);
    rsqrt_tile_init(); rsqrt_tile(0);
    tile_regs_commit(); tile_regs_wait(); pack_tile(0,2); tile_regs_release();
    inverse.push_back(1); dense.pop_front(1);
}
"""
_norm_zero_meanOUTPUT_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto xa=TensorAccessorArgs<0>();
    constexpr auto ga=TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto ia=TensorAccessorArgs<ga.next_compile_time_args_offset()>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0),2048);
    const auto gamma=TensorAccessor(ga,get_arg_val<uint32_t>(1),4096);
    const auto inverse=TensorAccessor(ia,get_arg_val<uint32_t>(2),4096);
    const uint32_t block=get_arg_val<uint32_t>(3), count=get_arg_val<uint32_t>(4);
    Noc noc; CircularBuffer data(0),weight(1),inv(2);
    data.reserve_back(count); weight.reserve_back(count); inv.reserve_back(1);
    // Only col0 of left faces0/2 participates in column normalization.
    // Keep their original offsets; the two right faces are never consumed.
    noc.async_read(inverse,inv,1024,{.page_id=0,.offset_bytes=0},{.offset_bytes=0});
    noc.async_read(inverse,inv,1024,{.page_id=0,.offset_bytes=2048},{.offset_bytes=2048});
    for(uint32_t chunk=0;chunk<count;chunk++) {
        const uint32_t page=block+chunk;
        noc.async_read(x,data,2048,{.page_id=page},{.offset_bytes=chunk*2048});
        noc.async_read(gamma,weight,4096,{.page_id=page},{.offset_bytes=chunk*4096});
    }
    noc.async_read_barrier(); inv.push_back(1); data.push_back(count); weight.push_back(count);
}
"""
_norm_zero_meanOUTPUT_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto oa=TensorAccessorArgs<0>();
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(0),2048);
    const uint32_t block=get_arg_val<uint32_t>(1),batch=get_arg_val<uint32_t>(2),count=get_arg_val<uint32_t>(3);
    Noc noc; CircularBuffer data(3); data.wait_front(count);
    volatile tt_l1_ptr uint32_t* raw=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data.get_read_ptr());
    for(uint32_t chunk=0;chunk<count;chunk++) {
        for(uint32_t row=batch;row<32;row++) for(uint32_t face=0;face<2;face++) {
            const uint32_t word=chunk*512+(row/16)*256+face*128+(row%16)*8;
            for(uint32_t j=0;j<8;j++) raw[word+j]=0;
        }
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(data),output,2048,
            {.offset_bytes=chunk*2048},{.page_id=block+chunk});
    }
    noc.async_write_barrier(); data.pop_front(count);
}
"""
_norm_zero_meanOUTPUT_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"
// Specialize a known +0 mean without changing the native centering/multiply.
namespace ckernel {
#ifdef TRISC_MATH
namespace sfpu {
inline void _hidden_normalize_zero_(uint32_t data,uint32_t inverse,uint32_t output) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    const uint32_t db=data*DEST_TILE_SIZE_RAW,ib=inverse*DEST_TILE_SIZE_RAW,ob=output*DEST_TILE_SIZE_RAW;
    // The incumbent broadcasts a +0 tile, then negates it into LREG7.
    TTI_SFPMOV(0,p_sfpu::LCONST_0,p_sfpu::LREG7,1 /* negate */);
    for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {
        const uint32_t left=half*FACE2_BASE+band*ROW_BAND_STRIDE;
        const uint32_t right=left+FACE1_BASE;
        TT_SFPLOAD(LREG_DATA0,IM,ADDR_MOD_7,db+left);
        TT_SFPLOAD(LREG_DATA1,IM,ADDR_MOD_7,db+left+ODD_COLS_OFFSET);
        TT_SFPLOAD(LREG_DATA2,IM,ADDR_MOD_7,db+right);
        TT_SFPLOAD(LREG_DATA3,IM,ADDR_MOD_7,db+right+ODD_COLS_OFFSET);
        // Identical native inverse broadcast, data+(-0), multiply and stores.
        _finish_col_normalize_row_band_(ib+left,ob+left,ob+left+ODD_COLS_OFFSET,
                                        ob+right,ob+right+ODD_COLS_OFFSET);
    }
}
} // namespace sfpu
#endif
ALWI void hidden_normalize_zero(uint32_t data,uint32_t inverse,uint32_t output) {
    sfpu_bcast_col_init();
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _hidden_normalize_zero_,data,inverse,output,VectorMode::None)));
}
} // namespace ckernel

void kernel_main() {
    const uint32_t count=get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(2,2,3);
    CircularBuffer x(0),gamma(1),inverse(2),out(3);
    x.wait_front(count); gamma.wait_front(count); inverse.wait_front(1); out.reserve_back(count);
    for(uint32_t chunk=0;chunk<count;chunk++) {
        tile_regs_acquire();
        reconfig_data_format_srca(0); copy_init(0); copy_tile(0,chunk,0);
        reconfig_data_format_srca(2); copy_init(2); copy_tile(2,0,2);
        // Preserve even the incumbent zero-subtract before inverse multiply.
        hidden_normalize_zero(0,2,0);
        copy_tile(1,chunk,1);
        sfpu_mul_bcast_row_init(); sfpu_mul_bcast_row(0,1);
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>(0);
        tile_regs_commit(); tile_regs_wait(); pack_tile<true>(0,3,chunk); tile_regs_release();
    }
    out.push_back(count); x.pop_front(count); gamma.pop_front(count); inverse.pop_front(1);
}
"""


def _norm_zero_meanhidden_rmsnorm_sfpu(x, gamma, *, epsilon=1e-6, output=None, scratch=None):
    """Optional scratch=(folded, inverse), FP32 interleaved TILE [1,32,1024]/[1,32,32].
    Default scratch lives in L1; explicitly supplied DRAM scratch remains supported.

    Supply output and both scratch buffers to avoid tensor allocation. Their
    addresses are registered and rebound in each generic-op invocation.
    """
    assert len(x.shape) == 3 and x.shape[0] == 1 and 1 <= x.shape[1] <= 32 and x.shape[2] == 5120
    assert x.dtype == ttnn.bfloat16 and tuple(gamma.shape) == (1, 1, 5120) and gamma.dtype == ttnn.float32
    assert epsilon > 0

    def allocate(shape, dtype, memory=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.empty(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=x.device(), memory_config=memory)

    if output is None:
        output = allocate(tuple(x.shape), ttnn.bfloat16)
    if scratch is None:
        scratch = (
            allocate((1, 32, 1024), ttnn.float32, ttnn.L1_MEMORY_CONFIG),
            allocate((1, 32, 32), ttnn.float32, ttnn.L1_MEMORY_CONFIG),
        )
    assert len(scratch) == 2
    folded, inverse = scratch
    assert tuple(output.shape) == tuple(x.shape) and output.dtype == ttnn.bfloat16
    assert tuple(folded.shape) == (1, 32, 1024) and folded.dtype == ttnn.float32
    assert tuple(inverse.shape) == (1, 32, 32) and inverse.dtype == ttnn.float32
    tensors = (x, gamma, output, folded, inverse)
    for tensor in tensors:
        assert tensor.layout == ttnn.TILE_LAYOUT
    for tensor in (x, gamma, output):
        assert tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    for tensor in (folded, inverse):
        assert tensor.memory_config() in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)
    assert len({t.buffer_address() for t in tensors}) == len(tensors)
    grid = x.device().compute_with_storage_grid_size()
    assert grid.x * grid.y >= 32
    output_cores = min(80, grid.x * grid.y)
    output_tiles = (160 + output_cores - 1) // output_cores

    def output_count(c):
        return 160 // output_cores + (c < 160 % output_cores)

    def output_start(c):
        return c * (160 // output_cores) + min(c, 160 % output_cores)

    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(output_cores)]
    bits = lambda v: struct.unpack("<I", struct.pack("<f", v))[0]

    def accessor_args(*tensors):
        args = []
        for t in tensors:
            args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        return args

    def launch(sources, runtime, ct, sizes, active, tensors):
        selected = coords[:active]
        ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in selected])
        config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for index, count, dtype in sizes:
            if dtype == ttnn.float32:
                modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes
        configs = [ttnn.ReaderConfigDescriptor(), ttnn.WriterConfigDescriptor(), config]
        kernels = []
        for source, args, compile_args, kernel_config in zip(sources, runtime, ct, configs):
            rt = ttnn.RuntimeArgs()
            for index, c in enumerate(selected):
                rt[c.x][c.y] = args(index)
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=source,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=ranges,
                    compile_time_args=compile_args,
                    runtime_args=rt,
                    config=kernel_config,
                )
            )
        cbs = []
        for index, count, dtype in sizes:
            page = 2048 if dtype == ttnn.bfloat16 else 4096
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=count * page,
                    core_ranges=ranges,
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
                )
            )
        ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))

    launch(
        [_norm_zero_meanFOLD_READER, _norm_zero_meanFOLD_WRITER, _norm_zero_meanFOLD_COMPUTE],
        [lambda c: [x.buffer_address(), c], lambda c: [folded.buffer_address(), c], lambda c: []],
        [accessor_args(x), accessor_args(folded), []],
        [(0, 5, ttnn.bfloat16), (3, 1, ttnn.float32)],
        32,
        [x, folded],
    )
    launch(
        [_norm_zero_meanINVERSE_READER, _norm_zero_meanINVERSE_WRITER, _norm_zero_meanINVERSE_COMPUTE],
        [lambda u: [folded.buffer_address(), u], lambda u: [inverse.buffer_address(), u, x.shape[1]], lambda u: []],
        [accessor_args(folded), accessor_args(inverse), [bits(epsilon), bits(1 / 5120)]],
        [(0, 1, ttnn.float32), (2, 1, ttnn.float32), (4, 1, ttnn.float32)],
        x.shape[1],
        [folded, inverse],
    )
    launch(
        [_norm_zero_meanOUTPUT_READER, _norm_zero_meanOUTPUT_WRITER, _norm_zero_meanOUTPUT_COMPUTE],
        [
            lambda c: [
                x.buffer_address(),
                gamma.buffer_address(),
                inverse.buffer_address(),
                output_start(c),
                output_count(c),
            ],
            lambda c: [output.buffer_address(), output_start(c), x.shape[1], output_count(c)],
            lambda c: [output_count(c)],
        ],
        [accessor_args(x, gamma, inverse), accessor_args(output), []],
        [
            (0, output_tiles, ttnn.bfloat16),
            (1, output_tiles, ttnn.float32),
            (2, 1, ttnn.float32),
            (3, output_tiles, ttnn.bfloat16),
        ],
        output_cores,
        [x, gamma, inverse, output],
    )
    return output
