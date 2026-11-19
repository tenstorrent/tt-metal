# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Role-isolated precision, split input projection and phase-specific CCL candidates."""
import math

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_optimization import Candidate, FamilyCandidate


class HybridCCL(FamilyCandidate):
    communication_dtype = ttnn.bfloat8_b

    @classmethod
    def from_state_dict(cls, sd, **kwargs):
        self = Candidate.from_state_dict.__func__(cls, sd, **kwargs)
        self.rs_index = 0
        return self

    def reduce(self, x):
        if x.shape[2] == 1:
            return Candidate.reduce(self, x)
        return FamilyCandidate.reduce(self, x)


class PrefillActivation(HybridCCL):
    low_ccl = False

    def reduce(self, x):
        return super().reduce(x) if self.low_ccl else Candidate.reduce(self, x)

    def norm(self, x, name):
        result = super().norm(x, name)
        if x.shape[2] > 1 and name in ("input_layernorm", "post_attention_layernorm"):
            result = ttnn.typecast(result, ttnn.bfloat8_b)
        return result

    def finish(self, x, mixed):
        if x.shape[2] == 1:
            return Candidate.finish(self, x, mixed)
        residual = ttnn.add(x, mixed)
        h = self.norm(residual, "post_attention_layernorm")
        h = ttnn.experimental.minimal_matmul(h, self.weights["packed_mlp.weight"], fuse_swiglu=True,
            config=self.prefill_program, compute_kernel_config=self.mlp_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        return ttnn.add(residual, self.reduce(self.linear(h, "mlp.down_proj")))


class SplitInput(Candidate):
    @classmethod
    def from_state_dict(cls, sd, **kwargs):
        self = super().from_state_dict(sd, **kwargs)
        widths = [2048, 1024, 32, 32] if self.kind == "linear_attention" else [2048, 256, 256]
        bank_count = self.device.dram_grid_size().x
        banks = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(bank_count - 1, 0))})
        offset = 0
        self.input_names = []
        for i, n in enumerate(widths):
            name = f"split_input_{i}"
            self.input_names.append(name)
            w = self.weights["packed_input.weight"][:, offset : offset + n]
            offset += n
            mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(banks, (4096, math.ceil(n / (32 * bank_count)) * 32), ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.dram_weights[name] = ttnn.to_memory_config(w, mem)
            ow = math.ceil(n / (32 * 32))
            om = ttnn.create_sharded_memory_config(
                (32, ow * 32),
                ttnn.CoreGrid(x=8, y=4),
                ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )
            self.dram_layouts[name] = (
                self.ar_mem,
                om,
                ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=2, per_core_M=1, per_core_N=ow, num_workers_per_dram_bank=2, use_multicore_pipeline=True
                ),
            )
        return self

    def linear(self, x, name):
        if x.shape[2] == 1 and name == "packed_input":
            return ttnn.concat(
                [
                    ttnn.to_memory_config(super(SplitInput, self).linear(x, n), ttnn.L1_MEMORY_CONFIG)
                    for n in self.input_names
                ],
                dim=3,
            )
        return super().linear(x, name)


class ColumnOutput(FamilyCandidate):
    """Gather local attention/MLP features, then column-parallel output projection.

    The result stays hidden-sharded through residual and distributed normalization.
    This tests collective placement as a complete consumer family.
    """
    column_mlp = False

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        self = super().from_state_dict(state_dict, **kwargs)
        prefix = f"model.language_model.layers.{kwargs['layer_idx']}."
        sd = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)} or state_dict
        self.column_names = ["linear_attn.out_proj" if self.kind == "linear_attention" else "self_attn.o_proj"]
        if self.column_mlp:
            self.column_names.append("mlp.down_proj")
        banks = self.device.dram_grid_size().x
        bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        for name in self.column_names:
            host = sd[name + ".weight"].T.contiguous()
            k, n = host.shape[0], host.shape[1] // 4
            def upload(memory):
                return ttnn.from_torch(host, device=self.device, dtype=self.weight_dtype(name),
                                      layout=ttnn.TILE_LAYOUT, memory_config=memory,
                                      mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=1))
            self.weights[name + ".weight"] = upload(ttnn.DRAM_MEMORY_CONFIG)
            self.dram_weights[name] = upload(ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
                ttnn.ShardSpec(bank_grid, (k, math.ceil(n / (32 * banks)) * 32), ttnn.ShardOrientation.ROW_MAJOR)))
            inc = 32
            im = ttnn.create_sharded_memory_config((32, k // inc), ttnn.CoreGrid(x=8, y=4),
                  ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True)
            om = ttnn.create_sharded_memory_config((32, 32), ttnn.CoreGrid(x=8, y=4),
                  ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True)
            self.dram_layouts[name] = (im, om, ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=2, per_core_M=1, per_core_N=1, num_workers_per_dram_bank=2,
                use_multicore_pipeline=True))
        return self

    def linear(self, x, name):
        if name in self.column_names:
            x = self.gather(ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG if x.shape[2] <= 32 else ttnn.DRAM_MEMORY_CONFIG))
        return super().linear(x, name)

    def reduce(self, x):
        return x if x.shape[3] == 1024 else super().reduce(x)


class RolePrecision(Candidate):
    low_roles = ()
    low_dtype = None
    low_fidelity = None

    def weight_dtype(self, name):
        if name in self.low_roles and self.low_dtype is not None:
            return self.low_dtype
        return super().weight_dtype(name)

    def matmul_compute(self, name):
        if name in self.low_roles and self.low_fidelity is not None:
            return ttnn.WormholeComputeKernelConfig(
                math_fidelity=self.low_fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
            )
        return super().matmul_compute(name)


def extra_candidate(name):
    if name in ("prefill_activation_bfp8", "prefill_activation_ccl_bfp8"):
        return type("PrefillActivation_" + name, (PrefillActivation,), {"low_ccl": name.endswith("ccl_bfp8")})
    if name == "hybrid_ccl_bfp8":
        return HybridCCL
    if name in ("column_attention", "column_all"):
        return type("ColumnOutput_" + name, (ColumnOutput,), {"column_mlp": name == "column_all"})
    if name in ("family_gather_mixed", "family_fused_mixed"):
        return type("MixedFusion_" + name, (FamilyCandidate,), dict(
            fused_gather_enabled=True, fused_scatter_enabled=name == "family_fused_mixed",
            communication_dtype=ttnn.bfloat8_b, norm_communication_dtype=ttnn.bfloat16))
    if name == "split_input":
        return SplitInput
    roles = {
        "input_lofi": (("packed_input",), None, ttnn.MathFidelity.LoFi),
        "output_lofi": (("linear_attn.out_proj", "self_attn.o_proj"), None, ttnn.MathFidelity.LoFi),
        "input_bfp4": (("packed_input",), ttnn.bfloat4_b, ttnn.MathFidelity.LoFi),
        "output_bfp4": (("linear_attn.out_proj", "self_attn.o_proj"), ttnn.bfloat4_b, ttnn.MathFidelity.LoFi),
        "gate_up_bfp4": (("mlp.gate_proj", "mlp.up_proj", "packed_mlp"), ttnn.bfloat4_b, ttnn.MathFidelity.LoFi),
        "down_bfp4": (("mlp.down_proj",), ttnn.bfloat4_b, ttnn.MathFidelity.LoFi),
    }
    selected, dtype, fidelity = roles[name]
    return type(
        "RolePrecision_" + name, (RolePrecision,), dict(low_roles=selected, low_dtype=dtype, low_fidelity=fidelity)
    )
