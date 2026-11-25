# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Nemotron TP4 attention/Mamba and EP4 ReLU² experts on P300_X2.

The optimized single-chip decoder is the numerical baseline. Weight conversion
is setup-only. Input/output use the same BF16 residual ownership selected at
construction. Caller owns paged cache, device positions and recurrent state.
"""

import math
from dataclasses import dataclass, replace

import ttnn

from .optimized_decoder import DecoderState, OptimizedDecoder
from .precision import load_precision

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
# perf/multiuser accuracy fix: dt = softplus(dt + dt_bias) with NO lower clamp, matching the
# CUDA/Triton kernels used in training and by vLLM (dt_limit=(0, inf)). The HF torch fallback
# clamps at time_step_min=0.001; on real text ~12% of (token, head) dt values fall below it,
# which shortens long-memory heads. NEMOTRON_DT_CLAMP=1 restores the bring-up behaviour.
import os as _os

DT_ACTIVATIONS = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)]
# perf/multiuser prefill: Mamba spans run norm/in_proj/conv/out_proj/CCL once per span and only the
# SSD scan in chunks of MAMBA_SCAN_CHUNK rows (was: everything per 32-row piece). 0 = legacy path.
MAMBA_SCAN_CHUNK = int(_os.environ.get("NEMOTRON_MAMBA_SCAN_CHUNK", "256"))
# perf/multiuser prefill: router logits as one bf16 x bf16 matmul (HiFi4, fp32 accumulate, fp32 out);
# both operands are exactly bf16, so products are exact and only summation order differs from the
# elementwise-multiply + reduce router (2.3 ms + 0.9 ms per 256 rows). Prefill only; decode unchanged.
ROUTER_MATMUL = _os.environ.get("NEMOTRON_ROUTER_MATMUL", "1") == "1"
# perf/multiuser decode: 2..MOE_INDEXED_MAX rows use the indexed TP sparse_matmul over all 6*B routed ids
# (fixed shape, traceable) plus a masked weighted sum, instead of the EP grouped path (built for prefill).
# perf/multiuser decode: router via the bf16 matmul for decode rows too (validated in prefill); the fp32
# elementwise router cost ~0.46 ms per MoE layer at 32 rows.
ROUTER_MATMUL_DECODE = _os.environ.get("NEMOTRON_ROUTER_MATMUL_DECODE", "1") == "1"
# perf/multiuser decode: Mamba batched decode without repeat_interleave / state transpose / update where().
MAMBA_DECODE_V2 = _os.environ.get("NEMOTRON_MAMBA_DECODE_V2", "1") == "1"
MAMBA_READOUT_DRAM = _os.environ.get("NEMOTRON_MAMBA_READOUT_DRAM", "1") == "1"
# perf/multiuser round 6: rows decode (b > 1) keeps the readout in the (1,1,b,1024) row layout (D_row skip).
MAMBA_ROW_READOUT = _os.environ.get("NEMOTRON_MAMBA_ROW_READOUT", "1") == "1"
# round 6: prefill gated RMS norm on (1,1,2L,512) rows instead of (1,L,2,512) (2 valid rows per 32-row tile: the
# reshape + norm + reshape back cost 1.65 ms per Mamba layer per 2048 tokens).
MAMBA_PREFILL_NORM_ROWS = _os.environ.get("NEMOTRON_MAMBA_PREFILL_NORM_ROWS", "1") == "1"
# round 6: prefill span SSD computes C @ B^T once per B/C group (2) instead of per head (16; the 8 heads of a group
# share B and C) and broadcasts it into the per-head decay multiply. Same products.
MAMBA_PREFILL_GROUP_CB = _os.environ.get("NEMOTRON_MAMBA_PREFILL_GROUP_CB", "1") == "1"
# round 6: prefill span head split/merge as column slices + concat instead of reshape/permute/repeat_interleave
# (tile-padded 4-D reshapes: ~1.2 ms per Mamba layer per 2048 tokens). Pure data movement.
MAMBA_PREFILL_HEAD_SLICES = _os.environ.get("NEMOTRON_MAMBA_PREFILL_HEAD_SLICES", "1") == "1"
# round 6: routed_combine zero-weight (remote-expert) slots read one of the 32 reserved zero rows (row n-32 + t % 32)
# instead of all reading row n-1: ~3/4 of every device's slots are remote, and one hot DRAM page made routed_combine
# 4x slower (2048 tokens 1.53 -> 0.36 ms standalone). Same values (weight 0 x zero row).
MOE_SPREAD_ZERO_ROWS = _os.environ.get("NEMOTRON_MOE_SPREAD_ZERO_ROWS", "1") == "1"
SPREAD_ZERO_MAX_TOKENS = 16384
# round 6: the prefill paths above that add ops (head slices, grouped CB, spread zero rows) cost host dispatch time on
# short eager prefills (128 tokens: +22 / +7 / +12 ms) and win from ~1024 tokens (2048: -5% / -3% / -8%); use them
# only for spans of at least PREFILL_FAST_MIN tokens.
PREFILL_FAST_MIN = int(_os.environ.get("NEMOTRON_PREFILL_FAST_MIN", "1024"))
# perf/multiuser round 6: EP decode (B >= MOE_INDEXED_MAX+1) dispatch as a one-hot matmul (PT^T @ x, an exact row
# copy; replaces a 120 us embedding gather + bf16->bfp8 typecast) and combine as an embedding gather of only the 6*B
# used expert rows + masked weighted sum (replaces the 180 us routed_combine and the zero-row fixups). Decode only.
# MoE layer (device 0 trace, random routing): B=8 1.22 -> 1.08 ms, B=16 1.42 -> 1.32; rel-L2 vs fp32 unchanged.
MOE_FAST_DC = _os.environ.get("NEMOTRON_MOE_FAST_DC", "1") == "1"
# round 6: the fast decode combine's weighted sum as one (rows x 6rows) selection-weights matmul.
MOE_COMBINE_MATMUL = _os.environ.get("NEMOTRON_MOE_COMBINE_MATMUL", "1") == "1"
# perf/multiuser round 6: decode grouped matmuls with per_expert_lanes (generalized in tt-metal to more experts than core
# rows: expert e on row e % rows, every core reading its own weight share instead of one reader per column). Output
# bit-identical; grouped up+down (32 local experts) B=8 0.53 -> 0.40 ms, B=16 0.81 -> 0.58, B=32 1.13 -> 0.78,
# 2048-token prefill 1.57 -> 1.13 ms.
MOE_LANES = _os.environ.get("NEMOTRON_MOE_LANES", "1") == "1"
MOE_INDEXED_MAX = int(
    _os.environ.get("NEMOTRON_MOE_INDEXED_MAX", "4")
)  # measured crossover: B=4 26.5 vs 31.3 ms, B=8 41.3 vs 36.4 ms
if _os.environ.get("NEMOTRON_DT_CLAMP", "0") == "1":
    DT_ACTIVATIONS.append(ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU_MIN, 0.001))


@dataclass(frozen=True)
class MeshPolicy:
    residual_layout: str = "sharded"
    ccl_dtype: str = "bfloat16"
    moe_decode: str = "indexed_tp"
    prefill_chunk: int = 2048
    projection_mode: str = "dram"
    row_fusion: bool = True
    persistent_ccl: bool = True
    ep_filter_remote: bool = True
    dram_readers: int = 1
    input_fusion: bool = False
    input_fusion_k: int = 7
    reduction: str = "scatter"
    packed_norm: bool = True
    sharded_norm: bool = True
    sparse_up_n: int = 1
    sparse_up_block: int = 21
    dram_roles: tuple = ("qkv",)
    packed_ep_up: bool = True
    ep_decode_activation_l1: bool = True
    ep_decode_output_l1: bool = True
    mamba_decode_readout_l1: bool = True


class MultichipDecoder(OptimizedDecoder):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, policy=None, precision=None):
        import torch

        expected = {
            "hidden_size": 2688,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "mamba_num_heads": 64,
            "mamba_head_dim": 64,
            "n_groups": 8,
            "ssm_state_size": 128,
            "conv_kernel": 4,
            "n_routed_experts": 128,
            "num_experts_per_tok": 6,
            "moe_intermediate_size": 1856,
            "moe_shared_expert_intermediate_size": 3712,
            "n_group": 1,
            "topk_group": 1,
            "max_position_embeddings": 262144,
        }
        for name, value in expected.items():
            if getattr(hf_config, name) != value:
                raise ValueError(f"Target config requires {name}={value}")
        if hf_config.moe_latent_size is not None or hf_config.mlp_hidden_act != "relu2":
            raise ValueError("Target requires non-latent relu2 experts")
        self = cls()
        self.policy = policy or MeshPolicy()
        self.precision = load_precision(precision)
        if self.precision["weight_groups"]["up"] == "bfloat16":
            # Two half-width EP up packs bound the BF16 weight ring in L1.
            self.policy = replace(self.policy, packed_ep_up=False)
        self.projection_compute = {
            role: ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=getattr(ttnn.MathFidelity, fidelity),
                math_approx_mode=self.precision["math_approx_mode"],
                fp32_dest_acc_en=self.precision["fp32_dest_acc_en"],
                packer_l1_acc=True,
            )
            for role, fidelity in self.precision["compute_fidelities"].items()
        }
        self.projection_fused_compute = {
            role: ttnn.ComputeConfigDescriptor(math_fidelity=getattr(ttnn.MathFidelity, fidelity))
            for role, fidelity in self.precision["compute_fidelities"].items()
        }
        if tuple(mesh_device.shape) != (1, 4):
            raise ValueError("This decoder targets the four-device P300_X2 mesh")
        if self.policy.residual_layout not in ("replicated", "sharded"):
            raise ValueError("Unknown residual layout")
        if self.policy.moe_decode not in ("ep", "indexed_tp"):
            raise ValueError("Unknown expert execution policy")
        if self.policy.projection_mode not in ("generic", "tuned", "dram"):
            raise ValueError("Unknown projection policy")
        if self.policy.ccl_dtype not in ("bfloat16", "bfloat8_b", "float32"):
            raise ValueError("Unsupported collective dtype")
        if self.policy.reduction not in ("scatter", "all_reduce"):
            raise ValueError("Unknown reduction policy")
        if self.policy.reduction == "all_reduce" and (
            self.policy.residual_layout != "replicated" or self.policy.row_fusion
        ):
            raise ValueError("all_reduce requires replicated residual and separate row projection")
        if self.policy.input_fusion and self.policy.residual_layout != "sharded":
            raise ValueError("Input fusion requires sharded residuals")
        if self.policy.sparse_up_n not in (1, 3) or self.policy.sparse_up_block not in (7, 21):
            raise ValueError("Unsupported indexed sparse geometry")
        if self.policy.dram_readers not in (1, 2) or not 1 <= self.policy.prefill_chunk <= 8192:
            raise ValueError("Unsupported DRAM readers or prefill chunk")
        self.config, self.layer_idx, self.device = hf_config, layer_idx, mesh_device
        self.kind = hf_config.layers_block_type[layer_idx]
        self._inside_prefill = False
        # Keep the scan partition stable across tile-aligned prefix-cache hits.
        # Larger Mamba chunks change recurrence rounding at a 32-token resume.
        self.prefill_chunk = 32 if self.kind == "mamba" else self.policy.prefill_chunk
        self.projection_dispatch_counts = {}
        self.decode_batch_chunk = 32
        self.router_chunk = 256
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.fast_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.fused_compute = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi)
        self.router_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        prefix = f"backbone.layers.{layer_idx}."
        sd = {k.removeprefix(prefix): v for k, v in state_dict.items()}

        def tensor(values, dtype=ttnn.bfloat16, dim=None, layout=ttnn.TILE_LAYOUT):
            mapper = (
                ttnn.ReplicateTensorToMesh(mesh_device) if dim is None else ttnn.ShardTensorToMesh(mesh_device, dim)
            )
            return ttnn.from_torch(
                values.contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh_device,
                mesh_mapper=mapper,
                memory_config=DRAM,
            )

        def ranks(values, dtype=ttnn.bfloat16, dim=-1):
            return tensor(torch.cat(values, dim=dim), dtype, dim=dim)

        self.dense_roles = {}

        def projection(role, values, dim=-1):
            result = ranks(values, getattr(ttnn, self.precision["weight_groups"][role]), dim)
            self.dense_roles[id(result)] = role
            return result

        if self.kind == "attention":
            weights = {n: sd[f"mixer.{n}_proj.weight"] * sd["norm.weight"] for n in ("q", "k", "v")}
            self.qkv = projection(
                "qkv",
                [
                    torch.cat(
                        [
                            weights["q"][r * 1024 : (r + 1) * 1024],
                            weights["k"][(r // 2) * 128 : (r // 2 + 1) * 128],
                            weights["v"][(r // 2) * 128 : (r // 2 + 1) * 128],
                        ]
                    ).T[None, None]
                    for r in range(4)
                ],
            )
            self.o = projection(
                "o", [sd["mixer.o_proj.weight"][:, r * 1024 : (r + 1) * 1024].T[None, None] for r in range(4)], -2
            )
            self.qkv_scalar_memory = ttnn.create_sharded_memory_config(
                (32, 128),
                ttnn.CoreGrid(x=5, y=2),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.qkv_decode_memory = self.qkv_scalar_memory
            self.batch_heads_memory, self.batch_concat_memory = {}, {}
            for batch in range(1, 33):
                self.batch_heads_memory[batch] = ttnn.create_sharded_memory_config(
                    (32, 128),
                    ttnn.num_cores_to_corerangeset(2 * batch, ttnn.CoreCoord(8, 8), row_wise=True),
                    ttnn.ShardStrategy.HEIGHT,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                self.batch_concat_memory[batch] = ttnn.create_sharded_memory_config(
                    (32, 128),
                    ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(8, 8), row_wise=True),
                    ttnn.ShardStrategy.HEIGHT,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
            self.sdpa_capacity_chunks = {}
            self.sdpa_decode_programs = {
                k: ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=k)
                for k in (128, 256)
            }
            self.batch_sdpa_decode_programs = self.sdpa_decode_programs
            self.sdpa_prefill_program = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=512, exp_approx_mode=False
            )
        elif self.kind == "mamba":
            w = sd["mixer.in_proj.weight"] * sd["norm.weight"]
            local = []
            conv_indices = []
            for r in range(4):
                ci = torch.cat(
                    [
                        torch.arange(r * 1024, (r + 1) * 1024),
                        torch.arange(4096 + r * 256, 4096 + (r + 1) * 256),
                        torch.arange(5120 + r * 256, 5120 + (r + 1) * 256),
                    ]
                )
                conv_indices.append(ci)
                wi = torch.cat(
                    [
                        torch.arange(r * 1024, (r + 1) * 1024),
                        4096 + ci,
                        torch.arange(10240 + r * 16, 10240 + (r + 1) * 16),
                    ]
                )
                local.append(torch.nn.functional.pad(w[wi].T, (0, 16))[None, None])
            self.in_proj = projection("in_proj", local)
            w = sd["mixer.out_proj.weight"] * sd["mixer.norm.weight"]
            self.out_proj = projection(
                "out_proj", [w[:, r * 1024 : (r + 1) * 1024].T[None, None] for r in range(4)], -2
            )
            self.conv_weights = [
                ranks([sd["mixer.conv1d.weight"][ci, 0, i].reshape(1, 1, 1, -1) for ci in conv_indices])
                for i in range(4)
            ]
            self.conv_bias = ranks([sd["mixer.conv1d.bias"][ci].reshape(1, 1, 1, -1) for ci in conv_indices])
            self.dt_bias = ranks(
                [sd["mixer.dt_bias"][r * 16 : (r + 1) * 16].reshape(1, 1, 1, 16) for r in range(4)], ttnn.float32
            )
            self.A = ranks(
                [-sd["mixer.A_log"][r * 16 : (r + 1) * 16].float().exp().reshape(1, 16, 1, 1) for r in range(4)],
                ttnn.float32,
                1,
            )
            self.D = ranks(
                [sd["mixer.D"][r * 16 : (r + 1) * 16].reshape(1, 16, 1, 1) for r in range(4)], ttnn.float32, 1
            )
            # D per channel (head value repeated over its 64 channels) for the row-layout readout (b > 1).
            self.D_row = ranks(
                [sd["mixer.D"][r * 16 : (r + 1) * 16].repeat_interleave(64).reshape(1, 1, 1, 1024) for r in range(4)],
                ttnn.float32,
            )
            self.causal_masks = {
                n: tensor(torch.tril(torch.ones(1, 1, n, n)), ttnn.float32) for n in range(32, 257, 32)
            }
            self.tail_masks = {}
            for n in range(1, 32):
                mask = torch.zeros(1, 1, 32, 1)
                mask[:, :, :n] = 1
                self.tail_masks[n] = tensor(mask, ttnn.float32)
        elif self.kind == "moe":
            self.norm = tensor(sd["norm.weight"].reshape(1, 1, 1, -1))
            self.norm_expand_zero = tensor(torch.zeros(1, 1, 32, 2688))
            self.route_expand_zero = tensor(torch.zeros(1, 1, 32, 128), ttnn.float32)
            mask = torch.zeros(1, 1, 32, 32)
            mask[..., :6] = 1
            self.route_score_mask = tensor(mask, ttnn.float32)
            self.router_rows = tensor(sd["mixer.gate.weight"].reshape(1, 1, 128, 2688), ttnn.float32)
            self.router_t = tensor(sd["mixer.gate.weight"].T.contiguous().reshape(1, 1, 2688, 128), ttnn.bfloat16)
            self.bias = tensor(sd["mixer.gate.e_score_correction_bias"].reshape(1, 1, 1, 128), ttnn.float32)
            self.expert_start = ranks([torch.full((1, 1, 1, 32), r * 32) for r in range(4)], ttnn.float32)
            raw = {}
            for role in ("up", "down"):
                key = f"mixer.experts.{role}_proj"
                raw[role] = (
                    sd[key]
                    if key in sd
                    else torch.stack([sd[f"mixer.experts.{e}.{role}_proj.weight"] for e in range(128)])
                )
            self.grouped_up = []
            # Pack once from host weights; the split control keeps the same total
            # BFP8 bytes and is never retained alongside the packed layout.
            up_ranges = ((0, 1856),) if self.policy.packed_ep_up else ((0, 928), (928, 1856))
            for begin, end in up_ranges:
                self.grouped_up.append(
                    ranks(
                        [
                            raw["up"][r * 32 : (r + 1) * 32, begin:end]
                            .transpose(-1, -2)
                            .permute(1, 0, 2)
                            .reshape(1, 1, 2688, -1)
                            for r in range(4)
                        ],
                        getattr(ttnn, self.precision["weight_groups"]["up"]),
                    )
                )
            down_ranges = (
                ((0, 1408), (1408, 2816)) if self.precision["weight_groups"]["down"] == "bfloat16" else ((0, 2816),)
            )
            self.grouped_down_parts = [
                ranks(
                    [
                        torch.nn.functional.pad(raw["down"][r * 32 : (r + 1) * 32], (0, 0, 0, 128))[:, begin:end]
                        .transpose(-1, -2)
                        .permute(1, 0, 2)
                        .reshape(1, 1, 1856, -1)
                        for r in range(4)
                    ],
                    getattr(ttnn, self.precision["weight_groups"]["down"]),
                )
                for begin, end in down_ranges
            ]
            self.grouped_down = self.grouped_down_parts[0]
            self.combine_zero = tensor(torch.zeros(1, 1, 1, 2816))
            self.zero_expert_row = tensor(torch.zeros(1, 1, 1, 2816), layout=ttnn.ROW_MAJOR_LAYOUT)
            self.zero_expert_rows32 = tensor(torch.zeros(1, 1, 32, 2816), layout=ttnn.ROW_MAJOR_LAYOUT)
            self.zero_row_spread = tensor(
                (torch.arange(SPREAD_ZERO_MAX_TOKENS) % 32).float().repeat(6, 1), ttnn.float32
            )  # (6, T_max): slot (k, t) -> t % 32
            self.shared_up = projection(
                "shared_up",
                [sd["mixer.shared_experts.up_proj.weight"][r * 928 : (r + 1) * 928].T[None, None] for r in range(4)],
            )
            self.shared_down = projection(
                "shared_down",
                [
                    sd["mixer.shared_experts.down_proj.weight"][:, r * 928 : (r + 1) * 928].T[None, None]
                    for r in range(4)
                ],
                -2,
            )
            if self.policy.moe_decode == "indexed_tp":
                up = torch.nn.functional.pad(raw["up"], (0, 0, 0, 64))
                down = torch.nn.functional.pad(raw["down"], (0, 64))
                self.up = projection("up", [up[:, r * 480 : (r + 1) * 480].transpose(-1, -2)[None] for r in range(4)])
                self.down = projection(
                    "down", [down[:, :, r * 480 : (r + 1) * 480].transpose(-1, -2)[None] for r in range(4)], -2
                )
                self.dummy_sparsity = tensor(torch.ones(1, 1, 1, 128), layout=ttnn.ROW_MAJOR_LAYOUT)
                self.reduce_mapping = tensor(
                    torch.zeros(1, 128, dtype=torch.int32), ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT
                )
        else:
            raise ValueError(self.kind)
        self.dense_programs = {}
        geometries = {
            "qkv": (10, 4, 7, 1),
            "in_proj": (9, 9, 7, 1),
            "out_proj": (7, 6, 8, 2),
            "o": (7, 6, 8, 2),
            "shared_up": (10, 3, 7, 1),
            "shared_down": (7, 6, 29, 2),
        }
        for role, (gx, gy, block, pn) in geometries.items():
            self.dense_programs[role] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=block,
                out_subblock_h=1,
                out_subblock_w=pn,
                per_core_M=1,
                per_core_N=pn,
                fuse_batch=True,
                mcast_in0=True,
                fused_activation=None,
            )
        self.sparse_programs = {}
        for width, grid, block, pn in (
            (480, (5, 3 if self.policy.sparse_up_n == 1 else 1), self.policy.sparse_up_block, self.policy.sparse_up_n),
            (2688, (7, 6), 15, 2),
        ):
            self.sparse_programs[width] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=grid,
                in0_block_w=block,
                out_subblock_h=1,
                out_subblock_w=pn,
                per_core_M=1,
                per_core_N=pn,
                fuse_batch=False,
                mcast_in0=True,
                fused_activation=None,
            )
        self.row_weights = {
            id(getattr(self, role)): ttnn.reshape(getattr(self, role), tuple(getattr(self, role).shape)[-2:])
            for role in ("o", "out_proj")
            if hasattr(self, role)
        }
        self.dram_weights, self.dram_programs, self.dram_inputs = {}, {}, {}
        if self.policy.projection_mode == "dram":
            banks = mesh_device.dram_grid_size().x
            bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
            for role in self.policy.dram_roles:
                if not hasattr(self, role):
                    continue
                weight = getattr(self, role)
                k, n = tuple(weight.shape)[-2:]
                memory = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(bank_grid, (k, math.ceil(n / (32 * banks)) * 32), ttnn.ShardOrientation.ROW_MAJOR),
                )
                self.dram_weights[role] = ttnn.to_memory_config(weight, memory)
                # Use complete K tiles over four activation workers where possible.
                workers = 4 if k % 128 == 0 else 1
                self.dram_inputs[role] = ttnn.create_sharded_memory_config(
                    (32, k // workers),
                    ttnn.CoreGrid(x=workers, y=1),
                    ttnn.ShardStrategy.WIDTH,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                self.dram_programs[role] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=7 if k == 2688 else 1,
                    per_core_M=1,
                    per_core_N=math.ceil(n / (32 * workers)),
                    num_workers_per_dram_bank=self.policy.dram_readers,
                    use_multicore_pipeline=self.policy.dram_readers > 1,
                )
        self.concat_subgrid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
        self.norm_memory = ttnn.create_sharded_memory_config(
            (32, 672),
            ttnn.CoreGrid(x=4, y=1),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(4, 1),
            subblock_w=3,
            block_h=1,
            block_w=21,
            inplace=False,
        )
        self.ccl_outputs = {}
        if self.policy.persistent_ccl:
            for rows in range(1, 33):
                for role, width in (("scatter", 672), ("gather", 2688)):
                    self.ccl_outputs[role, rows] = tensor(
                        torch.zeros(1, 1, rows, width), getattr(ttnn, self.policy.ccl_dtype)
                    )
        if self.policy.input_fusion and self.kind != "moe":
            from models.common.tt_ccl import get_tt_ccl

            self.ccl_manager = get_tt_ccl(mesh_device)
            self.input_semaphores = self.ccl_manager.get_and_cycle_ag_semaphore_handles(cluster_axis=1)
            self.input_barrier = self.ccl_manager.get_and_cycle_barrier_semaphore_handle(cluster_axis=1)
            self.input_gather_buffer = tensor(torch.zeros(1, 1, 32, 2688))
            self.input_program = ttnn.MinimalMatmulConfig(
                M_block_size=1,
                K_block_size=self.policy.input_fusion_k,
                N_block_size=4,
                subblock_h=1,
                subblock_w=4,
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 3),
            )
        return self

    def allocate_state(self, *, context_length=262144, block_size=32, num_slots=1):
        import torch

        if not 1 <= context_length <= self.config.max_position_embeddings or block_size != 32:
            raise ValueError("Invalid cache context or block size")

        def zero(shape, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                torch.zeros(shape),
                device=self.device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                memory_config=DRAM,
            )

        if self.kind == "attention":
            shape = (math.ceil(context_length / 32) * num_slots, 1, 32, 128)
            return DecoderState(key=zero(shape, ttnn.bfloat8_b), value=zero(shape, ttnn.bfloat8_b))
        if self.kind == "mamba":
            return DecoderState(
                conv=[zero((1, 1, 1, 1536)) for _ in range(3)], recurrent=zero((1, 16, 64, 128), ttnn.float32)
            )
        return DecoderState()

    def _gather(self, x):
        x = ttnn.typecast(x, getattr(ttnn, self.policy.ccl_dtype))
        return ttnn.all_gather(
            x,
            dim=3,
            cluster_axis=1,
            memory_config=DRAM,
            output_tensor=(
                self.ccl_outputs.get(("gather", x.shape[2]))
                if self.policy.persistent_ccl and tuple(x.shape)[:2] == (1, 1)
                else None
            ),
        )

    def _finish(self, partial, residual=None):
        partial = ttnn.to_memory_config(ttnn.typecast(partial, getattr(ttnn, self.policy.ccl_dtype)), DRAM)
        if self.policy.reduction == "all_reduce":
            if self.policy.residual_layout != "replicated" or self.policy.row_fusion:
                raise ValueError("all_reduce requires replicated residual and separate row projection")
            out = ttnn.all_reduce(partial, cluster_axis=1, memory_config=DRAM)
            out = ttnn.typecast(out, ttnn.bfloat16)
            return out if residual is None else ttnn.add(out, residual)
        # Larger prefills use an ordinary full-width row projection. The
        # actual tensor width identifies whether this call already scattered;
        # a policy flag alone must not skip their required reduction.
        if self.policy.row_fusion and self.kind != "moe" and partial.shape[-1] == 672:
            out = partial
        else:
            out = ttnn.reduce_scatter(
                partial,
                dim=3,
                cluster_axis=1,
                memory_config=DRAM,
                output_tensor=(
                    self.ccl_outputs.get(("scatter", partial.shape[2]))
                    if self.policy.persistent_ccl and tuple(partial.shape)[:2] == (1, 1)
                    else None
                ),
            )
        if self.policy.residual_layout == "replicated":
            out = self._gather(out)
        out = ttnn.typecast(out, ttnn.bfloat16)
        return out if residual is None else ttnn.add(out, residual)

    def _norm(self, x):
        if self.policy.packed_norm and x.shape[0] > 1:
            batch, width = x.shape[0], x.shape[-1]
            # Keep one physical row tile through the input projection. Unpacking
            # an L1 norm output here would expand B logical rows into B tiles.
            return self._norm(ttnn.reshape(x, (1, 1, batch, width)))
        if (
            self.policy.input_fusion
            and self.policy.residual_layout == "sharded"
            and self.kind != "moe"
            and not self._inside_prefill
        ):
            batch = x.shape[0]
            if batch > 1:
                x = ttnn.reshape(x, (1, 1, batch, 672))
            stats = ttnn.rms_norm_pre_all_gather(x, compute_kernel_config=self.compute, dtype=ttnn.float32)
            stats = ttnn.all_gather(stats, dim=3, cluster_axis=1, memory_config=DRAM)
            x = ttnn.rms_norm_post_all_gather(
                x,
                stats,
                epsilon=self.config.layer_norm_epsilon,
                compute_kernel_config=self.compute,
                dtype=ttnn.bfloat16,
                memory_config=DRAM,
            )
            return ttnn.reshape(x, (batch, 1, 1, 672)) if batch > 1 else x
        if self.policy.residual_layout == "sharded":
            x = ttnn.typecast(self._gather(x), ttnn.bfloat16)
        if (
            self.policy.sharded_norm
            and self.kind != "moe"
            and x.shape[0] == 1
            and x.shape[-2] <= 32
            and not self._inside_prefill
        ):
            return ttnn.rms_norm(
                ttnn.to_memory_config(x, self.norm_memory),
                epsilon=self.config.layer_norm_epsilon,
                program_config=self.norm_program,
                compute_kernel_config=self.compute,
                memory_config=self.norm_memory,
            )
        return self._norm_unsharded(x)

    def _linear(self, x, w, dtype=ttnn.bfloat16, **kwargs):
        kwargs.pop("decode_qkv", None)
        bias = kwargs.pop("bias", None)
        requested = kwargs.pop("memory_config", DRAM)
        role = self.dense_roles.get(id(w))
        if role is not None and self.precision["activation_dtype"] != "bfloat16":
            x = ttnn.typecast(x, getattr(ttnn, self.precision["activation_dtype"]))
        batch = x.shape[0]
        packed_batch = role is not None and batch > 1
        if packed_batch:
            x = ttnn.reshape(x, (1, 1, batch, x.shape[-1]))
        if self.policy.input_fusion and role in ("qkv", "in_proj") and x.shape[-1] == 672:
            dispatch = "fused_input"
            out = ttnn.experimental.all_gather_minimal_matmul_async(
                x,
                w,
                config=self.input_program,
                multi_device_global_semaphore=self.input_semaphores,
                barrier_semaphore=self.input_barrier,
                topology=ttnn.Topology.Linear,
                cluster_axis=1,
                num_links=1,
                num_workers_per_link=3,
                num_buffers_per_channel=2,
                persistent_output_buffer=self.input_gather_buffer,
                memory_config=DRAM,
                dtype=dtype,
                compute_kernel_config=self.projection_compute[role],
                force_transpose=False,
                transpose_core_grid=False,
            )[0]
            out = out[:, :, : x.shape[-2], :]
        # On this fixed mesh, larger resident fused plans may assign reader
        # traffic to the writer's NoC in dedicated-accounting mode. Keep the
        # validated <=128 physical-row envelope; larger prefills use linear
        # below and _finish performs their reduce-scatter (AUTODEBUG_watcher).
        elif not self._inside_prefill and self.policy.row_fusion and role in ("o", "out_proj") and x.shape[-2] <= 128:
            dispatch = "fused_row"
            out = ttnn.experimental.matmul_reduce_scatter_prefill(
                (
                    ttnn.pad(
                        ttnn.to_memory_config(x, DRAM), [(0, 0), (0, 0), (0, (-x.shape[-2]) % 32), (0, 0)], value=0.0
                    )
                    if x.shape[-2] % 32
                    else ttnn.to_memory_config(x, DRAM)
                ),
                self.row_weights[id(w)],
                cluster_axis=1,
                scatter_dim=-1,
                topology=ttnn.Topology.Linear,
                compute_kernel_config=self.projection_fused_compute[role],
                memory_config=DRAM,
            )
            out = out[:, :, : x.shape[-2], :]
        elif not self._inside_prefill and role in self.dram_weights and x.shape[-2] <= 32:
            dispatch = "dram"
            out = ttnn.linear(
                ttnn.to_memory_config(x, self.dram_inputs[role]),
                self.dram_weights[role],
                dtype=dtype,
                compute_kernel_config=self.projection_compute[role],
                program_config=self.dram_programs[role],
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
        else:
            # A short resumed prefill must use the same projection algorithm as
            # its rows in a longer fresh prefill. Decode keeps its tuned plans.
            dispatch = "generic"
            if (
                not self._inside_prefill
                and self.policy.projection_mode != "generic"
                and role in self.dense_programs
                and x.shape[-2] <= 32
            ):
                dispatch = "tuned_dense"
                kwargs.setdefault("program_config", self.dense_programs[role])
            out = ttnn.linear(
                ttnn.to_memory_config(x, DRAM),
                w,
                dtype=dtype,
                compute_kernel_config=self.compute if role is None else self.projection_compute[role],
                **kwargs,
            )
        key = f"{'prefill' if self._inside_prefill else 'decode'}:{role}:{dispatch}"
        self.projection_dispatch_counts[key] = self.projection_dispatch_counts.get(key, 0) + 1
        out = ttnn.to_memory_config(out, requested)
        if packed_batch:
            out = ttnn.reshape(out, (batch, 1, 1, out.shape[-1]))
        return ttnn.add(out, bias) if bias is not None else out

    def _router_logits(self, x):
        use = (ROUTER_MATMUL and self._inside_prefill and x.shape[2] > 32) or (
            ROUTER_MATMUL_DECODE and not self._inside_prefill
        )
        if not use:
            return super()._router_logits(x)
        return ttnn.linear(
            ttnn.typecast(x, ttnn.bfloat16) if x.dtype != ttnn.bfloat16 else x,
            self.router_t,
            dtype=ttnn.float32,
            memory_config=DRAM,
            compute_kernel_config=self.router_compute,
        )

    def _fast_dispatch(self, x, gather):
        """Rows of x in routing_metadata's expert-grouped order: one-hot PT (rows x n, PT[t, r] = gather[r] == t) as a
        transposed matmul. Every product is 1.0 * x or 0.0 (HiFi4, fp32 dest), so the copy is exact."""
        rows, n = x.shape[2], gather.shape[-1]
        if not hasattr(self, "_fast_dc_rows"):
            self._fast_dc_rows = {}
        if rows not in self._fast_dc_rows:
            import torch

            self._fast_dc_rows[rows] = ttnn.from_torch(
                torch.arange(rows, dtype=torch.float32).reshape(1, 1, rows, 1),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
        grow = ttnn.to_layout(ttnn.typecast(ttnn.reshape(gather, (1, 1, 1, n)), ttnn.float32), ttnn.TILE_LAYOUT)
        pt = ttnn.eq(grow, self._fast_dc_rows[rows], dtype=ttnn.bfloat16)
        # bf16 output then the same bf16->bfp8 typecast as the embedding path: packing the matmul result straight to
        # bfp8 rounds differently (measured +8% MoE rel-L2 vs fp32); this way the dispatched tensor is bit-identical.
        out = ttnn.matmul(
            pt,
            x if x.dtype == ttnn.bfloat16 else ttnn.typecast(x, ttnn.bfloat16),
            transpose_a=True,
            dtype=ttnn.bfloat16,
            memory_config=DRAM,
            compute_kernel_config=self.compute,
        )
        return ttnn.typecast(out, ttnn.bfloat8_b) if self.policy.packed_ep_up else out

    def _fast_combine(self, down, combine, weights, rows):
        """sum_k w[t, k] * down[combine[k, t]] over only the 6*rows used expert rows. Rows the grouped matmul never
        wrote may hold garbage, and zero weights point at arbitrary rows, so non-finite gathered values are zeroed."""
        width = down.shape[-1]
        table = ttnn.reshape(down, (down.shape[-2], width))
        g = ttnn.embedding(
            ttnn.reshape(combine[:, :rows], (1, 6 * rows)), table, layout=ttnn.TILE_LAYOUT, memory_config=DRAM
        )
        g = ttnn.reshape(g, (1, 1, 6 * rows, width))
        g = ttnn.where(ttnn.isfinite(g), g, 0.0)
        if MOE_COMBINE_MATMUL:
            # One matmul (rows x 6rows) @ (6rows x W): row t of the selection matrix holds slot k's routing weight at
            # column k*rows + t. bf16 rows x fp32 (TF32 srcB) weights, products exact in the fp32 DST, as routed_combine.
            if not hasattr(self, "_combine_sel"):
                self._combine_sel = {}
            if rows not in self._combine_sel:
                import torch

                sel = torch.zeros(1, 1, rows, 6 * rows)
                for k in range(6):
                    sel[0, 0, torch.arange(rows), k * rows + torch.arange(rows)] = 1.0
                self._combine_sel[rows] = ttnn.from_torch(
                    sel,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=DRAM,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                )
            w_row = ttnn.reshape(weights[:, :, :rows, :], (1, 1, 1, 6 * rows))
            out = ttnn.matmul(
                ttnn.multiply(self._combine_sel[rows], w_row),
                g,
                dtype=ttnn.bfloat16,
                memory_config=DRAM,
                compute_kernel_config=self.compute,
            )
            return out[..., :2688]
        # fp32 weights and accumulation, as in routed_combine.
        w = ttnn.reshape(weights[:, :, :rows, :], (1, 1, 6 * rows, 1))
        gw = ttnn.multiply(ttnn.typecast(g, ttnn.float32), w)
        out = ttnn.sum(ttnn.reshape(gw, (1, 6, rows, width)), dim=1, keepdim=True)
        return ttnn.typecast(out, ttnn.bfloat16)[..., :2688]

    def _moe_indexed_rows(self, x, indices, selected):
        rows = x.shape[2]
        n = 6 * rows
        if not hasattr(self, "indexed_masks"):
            self.indexed_masks = {}
        if rows not in self.indexed_masks:
            import torch

            mask = torch.zeros(1, n, rows, 1)
            for k in range(n):
                mask[0, k, k // 6, 0] = 1.0
            self.indexed_masks[rows] = ttnn.from_torch(
                mask,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
        ids = ttnn.to_layout(ttnn.typecast(indices, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT)
        ids = ttnn.reshape(ids, (1, 1, 1, n))
        up = ttnn.sparse_matmul(
            x,
            self.up,
            sparsity=self.dummy_sparsity,
            indices=ids,
            dtype=ttnn.bfloat16,
            program_config=self.sparse_programs[480],
            compute_kernel_config=self.projection_compute["up"],
        )
        act = ttnn.unary_chain(up, [ttnn.UnaryOpType.RELU, ttnn.UnaryOpType.SQUARE])
        act = ttnn.reshape(act, (1, n, rows, act.shape[-1]))
        down = ttnn.sparse_matmul(
            act,
            self.down,
            sparsity=self.dummy_sparsity,
            indices=ids,
            dtype=ttnn.bfloat16,
            is_input_a_sparse=True,
            program_config=self.sparse_programs[2688],
            compute_kernel_config=self.projection_compute["down"],
        )
        weights = ttnn.reshape(ttnn.typecast(selected, ttnn.bfloat16), (1, n, 1, 1))
        weights = ttnn.multiply(self.indexed_masks[rows], weights)
        routed = ttnn.sum(ttnn.multiply(ttnn.reshape(down, (1, n, rows, 2688)), weights), dim=1, keepdim=True)
        return routed

    def _moe(self, x):
        indices, selected, _ = self._route(x)
        if 1 < x.shape[2] <= MOE_INDEXED_MAX and not self._inside_prefill and self.policy.moe_decode == "indexed_tp":
            routed = self._moe_indexed_rows(x, indices, selected)
            shared = self._linear(
                ttnn.unary_chain(self._linear(x, self.shared_up), [ttnn.UnaryOpType.RELU, ttnn.UnaryOpType.SQUARE]),
                self.shared_down,
            )
            return ttnn.add(routed, shared)
        if x.shape[2] == 1 and self.policy.moe_decode == "indexed_tp":
            if self.precision["activation_dtype"] != "bfloat16":
                x = ttnn.typecast(x, getattr(ttnn, self.precision["activation_dtype"]))
            ids = ttnn.to_layout(ttnn.typecast(indices, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT)
            up = ttnn.sparse_matmul(
                x,
                self.up,
                sparsity=self.dummy_sparsity,
                indices=ids,
                dtype=ttnn.bfloat16,
                program_config=self.sparse_programs[480],
                compute_kernel_config=self.projection_compute["up"],
            )
            act = ttnn.unary_chain(up, [ttnn.UnaryOpType.RELU, ttnn.UnaryOpType.SQUARE])
            if self.precision["activation_dtype"] != "bfloat16":
                act = ttnn.typecast(act, getattr(ttnn, self.precision["activation_dtype"]))
            down = ttnn.sparse_matmul(
                act,
                self.down,
                sparsity=self.dummy_sparsity,
                indices=ids,
                dtype=ttnn.bfloat16,
                is_input_a_sparse=True,
                program_config=self.sparse_programs[2688],
                compute_kernel_config=self.projection_compute["down"],
            )
            routed = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
                ttnn.reshape(down, (6, 1, 1, 2688)),
                ids,
                self.reduce_mapping,
                0,
                num_shared_experts=0,
                split_size=2688,
                # Every column owns an intermediate TP slice of each expert.
                # The fused reducer masks by owner ROW for axis 1; all rows
                # are zero on our 1x4 mesh, so all four partials survive.
                cluster_axis=1,
                scores_tensor=ttnn.to_layout(ttnn.typecast(selected, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT),
                compute_kernel_config=self.compute,
            )[0]
        else:
            start = self.expert_start[..., :6]
            ids_float = ttnn.typecast(indices, ttnn.float32)
            valid = ttnn.logical_and(ttnn.ge(ids_float, start), ttnn.lt(ids_float, ttnn.add(start, 32)))
            local_ids = ttnn.typecast(
                ttnn.where(valid, ttnn.subtract(ids_float, start), 32.0 if self.policy.ep_filter_remote else 31.0),
                ttnn.uint32,
            )
            gather, combine, weights, counts = ttnn.experimental.nomic_embed_text_v2_moe.routing_metadata(
                local_ids, ttnn.multiply(selected, valid), num_experts=33 if self.policy.ep_filter_remote else 32
            )
            if self.policy.ep_filter_remote:
                # The last metadata expert owns remote slots. Removing that final
                # count preserves every real expert's tile-rounded row offset.
                counts = counts[:, :32]
            fast_dc = MOE_FAST_DC and not self._inside_prefill and self.policy.ep_filter_remote and x.shape[2] <= 32
            if fast_dc:
                dispatched = self._fast_dispatch(x, gather)
            else:
                table = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
                dispatched = ttnn.embedding(gather, table, layout=ttnn.TILE_LAYOUT, memory_config=DRAM)
                dispatched = ttnn.reshape(dispatched, (1, 1, dispatched.shape[-2], 2688))
                # Full-width up needs the smaller BFP8 input ring in the native
                # grouped kernel. Router/norm and the down input remain unchanged.
                if self.policy.packed_ep_up:
                    dispatched = ttnn.typecast(dispatched, ttnn.bfloat8_b)
            # Bit-identical to the shared schedule; faster at every measured size (decode 8-32 rows and prefill).
            lanes = {"per_expert_lanes": True} if MOE_LANES else {}
            use_l1_activation = self.policy.ep_decode_activation_l1 and not self._inside_prefill
            activation_memory = L1 if use_l1_activation else DRAM
            # Multi-row DOWN's full-K circular buffers overlap the live L1
            # activation plus output at B32. Keep its output in DRAM; the
            # indexed single-row path above and all prefill placement are unchanged.
            down_memory = (
                L1 if self.policy.ep_decode_output_l1 and not self._inside_prefill and x.shape[2] == 1 else DRAM
            )
            outputs = [
                ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul(
                    dispatched,
                    w,
                    counts,
                    dtype=ttnn.bfloat16,
                    memory_config=activation_memory,
                    compute_kernel_config=self.projection_compute["up"],
                    **lanes,
                )
                for w in self.grouped_up
            ]
            if use_l1_activation:
                up = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=3, memory_config=activation_memory)
                if len(outputs) > 1:
                    for output in outputs:
                        ttnn.deallocate(output)
                act = ttnn.unary_chain(
                    up,
                    [ttnn.UnaryOpType.RELU, ttnn.UnaryOpType.SQUARE],
                    memory_config=activation_memory,
                )
                ttnn.deallocate(up)
            else:
                up = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=3)
                act = ttnn.unary_chain(up, [ttnn.UnaryOpType.RELU, ttnn.UnaryOpType.SQUARE])
            if self.precision["activation_dtype"] != "bfloat16":
                act = ttnn.typecast(act, getattr(ttnn, self.precision["activation_dtype"]))
            down_parts = [
                ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul(
                    act,
                    weight,
                    counts,
                    dtype=ttnn.bfloat16,
                    memory_config=down_memory,
                    compute_kernel_config=self.projection_compute["down"],
                    output_layout=ttnn.ROW_MAJOR_LAYOUT,
                    **lanes,
                )
                for weight in self.grouped_down_parts
            ]
            down = down_parts[0] if len(down_parts) == 1 else ttnn.concat(down_parts, dim=3, memory_config=down_memory)
            if len(down_parts) > 1:
                for part in down_parts:
                    ttnn.deallocate(part)
            if use_l1_activation:
                ttnn.deallocate(act)
            if fast_dc:
                routed = self._fast_combine(down, combine, weights, x.shape[2])
            elif self.policy.ep_filter_remote:
                # Metadata for 33 experts reserves at least one full unused tile
                # after all 32 real regions. Combine must never read skipped rows:
                # zero * uninitialized NaN is still NaN. Point every zero-weight
                # slot to this explicitly initialized row instead.
                rows = down.shape[2]
                tokens = weights.shape[2]
                spread = MOE_SPREAD_ZERO_ROWS and PREFILL_FAST_MIN <= tokens <= SPREAD_ZERO_MAX_TOKENS and rows >= 64
                ttnn.experimental.slice_write(
                    self.zero_expert_rows32 if spread else self.zero_expert_row,
                    down,
                    [0, 0, rows - (32 if spread else 1), 0],
                    [1, 1, rows, 2816],
                    [1, 1, 1, 1],
                )
                weights_2d = ttnn.reshape(weights, (6, tokens))
                combine_float = ttnn.typecast(ttnn.to_layout(combine, ttnn.TILE_LAYOUT), ttnn.float32)
                zero_rows = ttnn.add(self.zero_row_spread[:, :tokens], float(rows - 32)) if spread else float(rows - 1)
                combine = ttnn.to_layout(
                    ttnn.typecast(ttnn.where(ttnn.ne(weights_2d, 0.0), combine_float, zero_rows), ttnn.uint32),
                    ttnn.ROW_MAJOR_LAYOUT,
                )
            if not fast_dc:
                routed = ttnn.experimental.nomic_embed_text_v2_moe.routed_combine(
                    down,
                    combine,
                    weights,
                    self.combine_zero,
                    num_tokens=x.shape[2],
                    memory_config=DRAM,
                    compute_kernel_config=self.compute,
                )[..., :2688]
        shared = self._linear(
            ttnn.unary_chain(self._linear(x, self.shared_up), [ttnn.UnaryOpType.RELU, ttnn.UnaryOpType.SQUARE]),
            self.shared_down,
        )
        return ttnn.add(routed, shared)

    def _attention_decode(self, x, *, state, page_table, current_pos, residual):
        batch = x.shape[2]
        qkv = self._linear(
            x,
            self.qkv,
            dtype=ttnn.bfloat8_b if self._inside_prefill else ttnn.bfloat16,
            memory_config=self.qkv_scalar_memory if batch == 1 else self.qkv_decode_memory,
            decode_qkv=True,
        )
        if self._inside_prefill:
            # Match bulk-prefill QKV rounding. The fused decode-head/cache API
            # requires BF16 storage, so promote the already-quantized values.
            qkv = ttnn.typecast(qkv, ttnn.bfloat16)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=8, num_kv_heads=1, overlap_qk_coregrid=False, memory_config=self.batch_heads_memory[batch]
        )
        ttnn.experimental.paged_fused_update_cache(
            state.key, k, state.value, v, update_idxs_tensor=current_pos, page_table=page_table
        )
        # Capacity is host-known metadata; current logical positions remain on device.
        pages = page_table.shape[-1]
        key_chunk = self.sdpa_capacity_chunks.get(pages, 128 if pages <= 256 else 256)
        # SDPA reads full chunks before masking. Initialize every lookahead
        # mapping as an alias of this request's last valid physical page.
        # Cache updates above retain the original logical table and capacity.
        read_table = page_table
        padding = -pages % (key_chunk // 32)
        if padding:
            read_table = ttnn.concat([page_table, ttnn.repeat(page_table[:, -1:], (1, padding))], dim=1)
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            state.key,
            state.value,
            read_table,
            cur_pos_tensor=current_pos,
            program_config=(self.sdpa_decode_programs if batch == 1 else self.batch_sdpa_decode_programs)[key_chunk],
            compute_kernel_config=self.compute,
        )
        out = ttnn.to_memory_config(out, self.batch_concat_memory[batch])
        out = ttnn.experimental.nlp_concat_heads_decode(
            out, num_heads=8, **({"sub_core_grids": self.concat_subgrid} if batch > 8 and batch % 8 else {})
        )
        out = out[:, :, :batch, :]
        return self._finish(self._linear(out, self.o), residual)

    def _mamba_decode(self, x, *, state, residual, use_mac=True, active_mask=None):
        if MAMBA_DECODE_V2 and use_mac and not self._inside_prefill and residual.shape[0] > 1:
            return self._mamba_decode_v2(x, state=state, residual=residual, active_mask=active_mask)
        return self._mamba_decode_v1(x, state=state, residual=residual, use_mac=use_mac, active_mask=active_mask)

    def _mamba_decode_v2(self, x, *, state, residual, active_mask=None):
        """Batched decode; same math as v1 with cheaper data movement:
        * B/C stay per group: x*dt viewed as [b, 2 groups, 8 heads*64, 1] broadcasts against B [b, 2, 1, 128]
          (no repeat_interleave); the readout is state[b,g,(h,p),n] @ C[b,g,n] (no transpose of the state).
        * Inactive rows mask dt (decay = exp(0) = 1, update = x*0*B = 0) instead of where() on the update.
        """
        batch = residual.shape[0]
        projected = self._linear(x, self.in_proj)
        if projected.shape[0] != batch:
            projected = ttnn.reshape(projected, (batch, 1, 1, 2592))
        gate = projected[:, :, :, :1024]
        conv_in = projected[:, :, :, 1024:2560]
        dt = ttnn.typecast(projected[:, :, :, 2560:2576], ttnn.float32)
        conv = ttnn.multiply(conv_in, self.conv_weights[3])
        for history, weight in zip(state.conv, self.conv_weights[:3]):
            conv = ttnn.mac(history, weight, conv)
        for i in range(2):
            if active_mask is None:
                ttnn.copy(state.conv[i + 1], state.conv[i])
            else:
                ttnn.where(active_mask, state.conv[i + 1], state.conv[i], output_tensor=state.conv[i])
        if active_mask is None:
            ttnn.copy(conv_in, state.conv[2])
        else:
            ttnn.where(active_mask, conv_in, state.conv[2], output_tensor=state.conv[2])
        conv = ttnn.add(conv, self.conv_bias, activations=[ttnn.UnaryOpType.SILU])
        x_flat = ttnn.typecast(conv[:, :, :, :1024], ttnn.float32)
        x_ssm = ttnn.reshape(x_flat, (batch, 16, 64, 1))
        vector_view = ttnn.experimental.view
        B = vector_view(ttnn.typecast(conv[:, :, :, 1024:1280], ttnn.float32), (batch, 2, 1, 128))
        C = vector_view(ttnn.typecast(conv[:, :, :, 1280:1536], ttnn.float32), (batch, 2, 1, 128))
        dt = ttnn.add(dt, self.dt_bias, activations=DT_ACTIVATIONS)
        dt = ttnn.reshape(dt, (batch, 16, 1, 1))
        if active_mask is not None:
            dt = ttnn.where(ttnn.typecast(active_mask, ttnn.float32), dt, 0.0)
        decay = ttnn.multiply(dt, self.A, activations=[ttnn.UnaryOpType.EXP])
        xdt = ttnn.reshape(ttnn.multiply(x_ssm, dt), (batch, 2, 512, 1))
        update = ttnn.reshape(ttnn.multiply(xdt, B), (batch, 16, 64, 128))
        new_state = ttnn.mac(state.recurrent, decay, update, output_tensor=state.recurrent)
        del update, xdt, decay, dt, B, conv, conv_in, projected
        if MAMBA_READOUT_DRAM:
            # perf/multiuser: matmul against a pre-transposed C with a DRAM output runs on the full grid
            # (B=32: 77 us vs 166 us for linear(transpose_b) into L1; identical values).
            y = ttnn.matmul(
                ttnn.reshape(new_state, (batch, 2, 512, 128)),
                ttnn.transpose(C, -2, -1),
                dtype=ttnn.float32,
                memory_config=DRAM,
                compute_kernel_config=self.compute,
            )
        else:
            y = ttnn.linear(
                ttnn.reshape(new_state, (batch, 2, 512, 128)),
                C,
                dtype=ttnn.float32,
                memory_config=L1,
                compute_kernel_config=self.compute,
                transpose_b=True,
            )
        y = ttnn.transpose(ttnn.reshape(y, (batch, 16, 64, 1)), -2, -1)
        x_out = ttnn.experimental.view(x_flat, (batch, 16, 1, 64))
        y = ttnn.mac(x_out, self.D, y)
        y = ttnn.experimental.view(y, (batch, 1, 1, 1024))
        gated = ttnn.multiply(y, ttnn.typecast(gate, ttnn.float32), input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
        grouped = ttnn.experimental.view(gated, (batch, 2, 1, 512))
        grouped = ttnn.rms_norm(grouped, epsilon=self.config.layer_norm_epsilon, compute_kernel_config=self.compute)
        normalized = ttnn.typecast(ttnn.experimental.view(grouped, (batch, 1, 1, 1024)), ttnn.bfloat16)
        return self._finish(self._linear(normalized, self.out_proj), residual)

    def _mamba_decode_rows(self, x, *, state, residual, active_mask, active_rows):
        """Batched decode on the (1,1,b,W) row layout (perf/multiuser round 3, NEMOTRON_MAMBA_ROWS=1).

        Same math as _mamba_decode_v2. The conv state is (1,1,S,1536) rows (all S slots in one tile row group, so the
        conv taps/shift cost one tile row instead of b padded tiles); rows >= b and inactive rows keep their state via
        active_rows (1,1,S,1). The SSM state stays slot-major (b,16,64,128) and is updated in place as in v2.
        """
        batch = residual.shape[2]
        slots = state.conv[0].shape[2]
        projected = self._linear(x, self.in_proj)  # (1,1,b,2592)
        gate = projected[:, :, :, :1024]
        conv_in = projected[:, :, :, 1024:2560]
        dt = ttnn.typecast(projected[:, :, :, 2560:2576], ttnn.float32)
        if batch != slots:
            conv_in = ttnn.pad(conv_in, [(0, 0), (0, 0), (0, slots - batch), (0, 0)], 0.0)
        conv = ttnn.multiply(conv_in, self.conv_weights[3])
        for history, weight in zip(state.conv, self.conv_weights[:3]):
            conv = ttnn.mac(history, weight, conv)
        for i in range(2):
            ttnn.where(active_rows, state.conv[i + 1], state.conv[i], output_tensor=state.conv[i])
        ttnn.where(active_rows, conv_in, state.conv[2], output_tensor=state.conv[2])
        conv = ttnn.add(conv, self.conv_bias, activations=[ttnn.UnaryOpType.SILU])
        # (1,1,1,X) == (b,1,1,X) at b=1, so the free views of v2 apply there; b>1 needs real reshapes
        rs = ttnn.experimental.view if batch == 1 else ttnn.reshape
        x_flat = ttnn.typecast(conv[:, :, :batch, :1024], ttnn.float32)  # (1,1,b,1024)
        x_ssm = ttnn.reshape(x_flat, (batch, 16, 64, 1))
        B = rs(ttnn.typecast(conv[:, :, :batch, 1024:1280], ttnn.float32), (batch, 2, 1, 128))
        C = rs(ttnn.typecast(conv[:, :, :batch, 1280:1536], ttnn.float32), (batch, 2, 1, 128))
        dt = ttnn.add(dt, self.dt_bias, activations=DT_ACTIVATIONS)
        dt = ttnn.reshape(dt, (batch, 16, 1, 1))
        dt = ttnn.where(ttnn.typecast(active_mask, ttnn.float32), dt, 0.0)
        decay = ttnn.multiply(dt, self.A, activations=[ttnn.UnaryOpType.EXP])
        xdt = ttnn.reshape(ttnn.multiply(x_ssm, dt), (batch, 2, 512, 1))
        update = ttnn.reshape(ttnn.multiply(xdt, B), (batch, 16, 64, 128))
        new_state = ttnn.mac(state.recurrent, decay, update, output_tensor=state.recurrent)
        del update, xdt, decay, dt, B, conv, conv_in, projected
        if batch > 1 and MAMBA_ROW_READOUT:
            # Readout as (b,8,128,128) @ (b,8,128,1): the same dot products as (b,2,512,128) @ (b,2,128,1), but 8b
            # batches for the cores instead of 2b (41.7 us on 32 cores at b=16). Then (b,8,128,1) -> (b,8,1,128) ->
            # rows (1,1,b,1024) and the D skip in the row layout: the same fp32 mac as below, without the (b,16,1,64)
            # one-row tiles (reshape + transpose + mac + reshape 71 us at b=16).
            y = ttnn.matmul(
                ttnn.reshape(new_state, (batch, 8, 128, 128)),
                ttnn.repeat_interleave(ttnn.transpose(C, -2, -1), 4, dim=1),
                dtype=ttnn.float32,
                memory_config=DRAM,
                compute_kernel_config=self.compute,
            )
            y = ttnn.reshape(ttnn.transpose(y, -2, -1), (1, 1, batch, 1024))
            y = ttnn.mac(x_flat, self.D_row, y)
        else:
            y = ttnn.matmul(
                ttnn.reshape(new_state, (batch, 2, 512, 128)),
                ttnn.transpose(C, -2, -1),
                dtype=ttnn.float32,
                memory_config=DRAM,
                compute_kernel_config=self.compute,
            )
            y = ttnn.transpose(ttnn.reshape(y, (batch, 16, 64, 1)), -2, -1)  # (b,16,1,64)
            y = ttnn.mac(rs(x_flat, (batch, 16, 1, 64)), self.D, y)
            y = rs(y, (1, 1, batch, 1024))
        gated = ttnn.multiply(y, ttnn.typecast(gate, ttnn.float32), input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
        grouped = rs(gated, (1, 2, 1, 512)) if batch == 1 else ttnn.reshape(gated, (1, 1, 2 * batch, 512))
        grouped = ttnn.rms_norm(grouped, epsilon=self.config.layer_norm_epsilon, compute_kernel_config=self.compute)
        normalized = ttnn.typecast(rs(grouped, (1, 1, batch, 1024)), ttnn.bfloat16)
        return self._finish(self._linear(normalized, self.out_proj), residual)

    def _mamba_decode_v1(self, x, *, state, residual, use_mac=True, active_mask=None):
        batch = residual.shape[0]
        projected = self._linear(x, self.in_proj)
        if projected.shape[0] != batch:
            projected = ttnn.reshape(projected, (batch, 1, 1, 2592))
        gate = projected[:, :, :, :1024]
        conv_in = projected[:, :, :, 1024:2560]
        dt = ttnn.typecast(projected[:, :, :, 2560:2576], ttnn.float32)
        conv = ttnn.multiply(conv_in, self.conv_weights[3])
        for history, weight in zip(state.conv, self.conv_weights[:3]):
            conv = ttnn.mac(history, weight, conv) if use_mac else ttnn.add(conv, ttnn.multiply(history, weight))
        for i in range(2):
            if active_mask is None:
                ttnn.copy(state.conv[i + 1], state.conv[i])
            else:
                ttnn.where(active_mask, state.conv[i + 1], state.conv[i], output_tensor=state.conv[i])
        if active_mask is None:
            ttnn.copy(conv_in, state.conv[2])
        else:
            ttnn.where(active_mask, conv_in, state.conv[2], output_tensor=state.conv[2])
        conv = ttnn.add(conv, self.conv_bias, activations=[ttnn.UnaryOpType.SILU])
        x_flat = ttnn.typecast(conv[:, :, :, :1024], ttnn.float32)
        x_ssm = ttnn.reshape(x_flat, (batch, 16, 64, 1))
        vector_view = ttnn.experimental.view
        B = vector_view(ttnn.typecast(conv[:, :, :, 1024:1280], ttnn.float32), (batch, 2, 1, 128))
        C = vector_view(ttnn.typecast(conv[:, :, :, 1280:1536], ttnn.float32), (batch, 2, 1, 128))
        B = ttnn.repeat_interleave(B, 8, dim=1)
        C = ttnn.repeat_interleave(C, 8, dim=1)
        dt = ttnn.add(
            dt,
            self.dt_bias,
            activations=DT_ACTIVATIONS,
        )
        dt = ttnn.reshape(dt, (batch, 16, 1, 1))
        decay = ttnn.multiply(dt, self.A, activations=[ttnn.UnaryOpType.EXP])
        update = ttnn.multiply(ttnn.multiply(x_ssm, dt), B)
        if active_mask is not None:
            recurrent_mask = ttnn.typecast(active_mask, ttnn.float32)
            decay = ttnn.where(recurrent_mask, decay, 1.0)
            update = ttnn.where(recurrent_mask, update, 0.0)
        new_state = (
            ttnn.mac(state.recurrent, decay, update, output_tensor=state.recurrent)
            if use_mac
            else ttnn.add(ttnn.multiply(state.recurrent, decay), update, output_tensor=state.recurrent)
        )
        if batch > 1:
            del update, decay, dt, B, conv, conv_in, projected, x_ssm
        if batch > 1 and self.policy.mamba_decode_readout_l1:
            y = ttnn.linear(
                C,
                new_state,
                dtype=ttnn.float32,
                memory_config=L1,
                compute_kernel_config=self.compute,
                transpose_b=True,
            )
        else:
            y = self._linear(C, new_state, ttnn.float32, transpose_b=True)
        x_out = ttnn.experimental.view(x_flat, (batch, 16, 1, 64))
        y = ttnn.mac(x_out, self.D, y) if use_mac else ttnn.add(y, ttnn.multiply(x_out, self.D))
        y = ttnn.experimental.view(y, (batch, 1, 1, 1024))
        gated = ttnn.multiply(y, ttnn.typecast(gate, ttnn.float32), input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
        view = ttnn.experimental.view
        grouped = view(gated, (batch, 2, 1, 512))
        grouped = ttnn.rms_norm(grouped, epsilon=self.config.layer_norm_epsilon, compute_kernel_config=self.compute)
        normalized = ttnn.typecast(view(grouped, (batch, 1, 1, 1024)), ttnn.bfloat16)
        return self._finish(self._linear(normalized, self.out_proj), residual)

    def _mamba_prefill_chunk(self, x, *, state, valid_length=None):
        length = x.shape[2]
        projected = self._linear(x, self.in_proj)
        gate = projected[:, :, :, :1024]
        conv_in = projected[:, :, :, 1024:2560]
        dt = ttnn.typecast(projected[:, :, :, 2560:2576], ttnn.float32)
        history = ttnn.concat([*state.conv, conv_in], dim=2)
        conv = ttnn.multiply(history[:, :, :length, :], self.conv_weights[0])
        for i in range(1, 4):
            conv = ttnn.add(conv, ttnn.multiply(history[:, :, i : i + length, :], self.conv_weights[i]))
        for i in range(3):
            if valid_length is None:
                ttnn.copy(conv_in[:, :, length - 3 + i : length - 2 + i, :], state.conv[i])
            else:
                ttnn.copy(history[:, :, valid_length + i : valid_length + i + 1, :], state.conv[i])
        conv = ttnn.add(conv, self.conv_bias, activations=[ttnn.UnaryOpType.SILU])
        xs = ttnn.permute(
            ttnn.reshape(ttnn.typecast(conv[:, :, :, :1024], ttnn.float32), (1, length, 16, 64)), (0, 2, 1, 3)
        )
        B = ttnn.permute(
            ttnn.reshape(ttnn.typecast(conv[:, :, :, 1024:1280], ttnn.float32), (1, length, 2, 128)), (0, 2, 1, 3)
        )
        C = ttnn.permute(
            ttnn.reshape(ttnn.typecast(conv[:, :, :, 1280:1536], ttnn.float32), (1, length, 2, 128)), (0, 2, 1, 3)
        )
        B = ttnn.repeat_interleave(B, 8, dim=1)
        C = ttnn.repeat_interleave(C, 8, dim=1)
        dt = ttnn.add(
            dt,
            self.dt_bias,
            activations=DT_ACTIVATIONS,
        )
        dt = ttnn.permute(ttnn.reshape(dt, (1, length, 16, 1)), (0, 2, 1, 3))
        if valid_length is not None:
            dt = ttnn.multiply(dt, self.tail_masks[valid_length])
        adt = ttnn.multiply(dt, self.A)
        accumulated = ttnn.cumsum(adt, dim=2)
        differences = ttnn.subtract(accumulated, ttnn.transpose(accumulated, 2, 3))
        decay = ttnn.where(self.causal_masks[length], differences, -1e30)
        xd = ttnn.multiply(xs, dt)
        CB = self._linear(C, B, ttnn.float32, transpose_b=True)
        local = self._linear(
            ttnn.multiply(CB, decay, input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, 0.0)]),
            xd,
            ttnn.float32,
        )
        from_prefix = self._linear(C, state.recurrent, ttnn.float32, transpose_b=True)
        y = ttnn.add(
            local,
            ttnn.multiply(
                from_prefix, accumulated, input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, 0.0)]
            ),
        )
        y = ttnn.add(y, ttnn.multiply(xs, self.D))
        last = accumulated[:, :, length - 1 : length, :]
        Bdecay = ttnn.multiply(B, ttnn.subtract(last, accumulated, activations=[ttnn.UnaryOpType.EXP]))
        update = self._linear(xd, Bdecay, ttnn.float32, transpose_a=True)
        ttnn.add(
            ttnn.multiply(
                state.recurrent, last, input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, 0.0)]
            ),
            update,
            output_tensor=state.recurrent,
        )
        y = ttnn.reshape(ttnn.permute(y, (0, 2, 1, 3)), (1, 1, length, 1024))
        gated = ttnn.multiply(y, ttnn.typecast(gate, ttnn.float32), input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
        grouped = ttnn.reshape(gated, (1, 1, 2 * length, 512) if MAMBA_PREFILL_NORM_ROWS else (1, length, 2, 512))
        grouped = ttnn.rms_norm(grouped, epsilon=self.config.layer_norm_epsilon, compute_kernel_config=self.compute)
        normalized = ttnn.typecast(ttnn.reshape(grouped, (1, 1, length, 1024)), ttnn.bfloat16)
        out = self._linear(normalized, self.out_proj)
        return out if valid_length is None else out[:, :, :valid_length, :]

    def _mamba_prefill_span(self, inp, *, state):
        """Whole span: one norm/in_proj/conv/gated-norm/out_proj/reduce-scatter; SSD scanned in chunks.

        Same SSD algebra as _mamba_prefill_chunk (fp32), with chunk length MAMBA_SCAN_CHUNK.
        Padded rows get dt = 0 (no decay, no input), so they never touch the recurrent state.
        """
        n = inp.shape[2]
        P = math.ceil(n / 32) * 32
        work = inp if P == n else ttnn.pad(inp, [(0, 0), (0, 0), (0, P - n), (0, 0)], value=0.0)
        x = self._norm(work)
        projected = self._linear(x, self.in_proj)
        gate = projected[:, :, :, :1024]
        conv_in = projected[:, :, :, 1024:2560]
        dt = ttnn.typecast(projected[:, :, :, 2560:2576], ttnn.float32)
        history = ttnn.concat([*state.conv, conv_in], dim=2)
        conv = ttnn.multiply(history[:, :, :P, :], self.conv_weights[0])
        for i in range(1, 4):
            conv = ttnn.add(conv, ttnn.multiply(history[:, :, i : i + P, :], self.conv_weights[i]))
        for i in range(3):
            ttnn.copy(history[:, :, n + i : n + i + 1, :], state.conv[i])
        del history
        conv = ttnn.add(conv, self.conv_bias, activations=[ttnn.UnaryOpType.SILU])
        head_slices = MAMBA_PREFILL_HEAD_SLICES and P >= PREFILL_FAST_MIN
        group_cb = MAMBA_PREFILL_GROUP_CB and P >= PREFILL_FAST_MIN
        if head_slices:
            # (1,1,P,1024) -> (1,16,P,64): head h = columns [64h, 64h+64); B/C (1,1,P,256) -> groups of 128.
            xf = ttnn.typecast(conv[:, :, :, :1536], ttnn.float32)
            xs = ttnn.concat([xf[:, :, :, 64 * h : 64 * h + 64] for h in range(16)], dim=1)
            bg = [xf[:, :, :, 1024 + 128 * g : 1152 + 128 * g] for g in range(2)]
            cg = [xf[:, :, :, 1280 + 128 * g : 1408 + 128 * g] for g in range(2)]
            B2, C2 = ttnn.concat(bg, dim=1), ttnn.concat(cg, dim=1)  # (1,2,P,128)
            B = ttnn.concat([bg[h // 8] for h in range(16)], dim=1)  # (1,16,P,128) = repeat_interleave(B2, 8)
            C = ttnn.concat([cg[h // 8] for h in range(16)], dim=1)
            del xf, bg, cg
        else:
            xs = ttnn.permute(
                ttnn.reshape(ttnn.typecast(conv[:, :, :, :1024], ttnn.float32), (1, P, 16, 64)), (0, 2, 1, 3)
            )
            B = ttnn.permute(
                ttnn.reshape(ttnn.typecast(conv[:, :, :, 1024:1280], ttnn.float32), (1, P, 2, 128)), (0, 2, 1, 3)
            )
            C = ttnn.permute(
                ttnn.reshape(ttnn.typecast(conv[:, :, :, 1280:1536], ttnn.float32), (1, P, 2, 128)), (0, 2, 1, 3)
            )
            B2, C2 = B, C  # (1,2,P,128): one B/C per group of 8 heads
            B = ttnn.repeat_interleave(B, 8, dim=1)
            C = ttnn.repeat_interleave(C, 8, dim=1)
        dt = ttnn.add(dt, self.dt_bias, activations=DT_ACTIVATIONS)
        if P != n:
            dt = ttnn.pad(dt[:, :, :n, :], [(0, 0), (0, 0), (0, P - n), (0, 0)], value=0.0)
        dt = ttnn.permute(ttnn.reshape(dt, (1, P, 16, 1)), (0, 2, 1, 3))
        ys = []
        for c0 in range(0, P, MAMBA_SCAN_CHUNK):
            q = min(MAMBA_SCAN_CHUNK, P - c0)
            whole = q == P
            xs_c = xs if whole else xs[:, :, c0 : c0 + q, :]
            dt_c = dt if whole else dt[:, :, c0 : c0 + q, :]
            B_c = B if whole else B[:, :, c0 : c0 + q, :]
            C_c = C if whole else C[:, :, c0 : c0 + q, :]
            adt = ttnn.multiply(dt_c, self.A)
            accumulated = ttnn.cumsum(adt, dim=2)
            differences = ttnn.subtract(accumulated, ttnn.transpose(accumulated, 2, 3))
            decay = ttnn.where(self.causal_masks[q], differences, -1e30)
            xd = ttnn.multiply(xs_c, dt_c)
            if group_cb:
                C2_c = C2 if whole else C2[:, :, c0 : c0 + q, :]
                B2_c = B2 if whole else B2[:, :, c0 : c0 + q, :]
                CB = ttnn.reshape(self._linear(C2_c, B2_c, ttnn.float32, transpose_b=True), (2, 1, q, q))
                weighted = ttnn.multiply(
                    CB,
                    ttnn.reshape(decay, (2, 8, q, q)),
                    input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, 0.0)],
                )
                weighted = ttnn.reshape(weighted, (1, 16, q, q))
            else:
                CB = self._linear(C_c, B_c, ttnn.float32, transpose_b=True)
                weighted = ttnn.multiply(
                    CB, decay, input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, 0.0)]
                )
            local = self._linear(weighted, xd, ttnn.float32)
            del weighted
            from_prefix = self._linear(C_c, state.recurrent, ttnn.float32, transpose_b=True)
            y = ttnn.add(
                local,
                ttnn.multiply(
                    from_prefix,
                    accumulated,
                    input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, 0.0)],
                ),
            )
            ys.append(ttnn.add(y, ttnn.multiply(xs_c, self.D)))
            last = accumulated[:, :, q - 1 : q, :]
            Bdecay = ttnn.multiply(B_c, ttnn.subtract(last, accumulated, activations=[ttnn.UnaryOpType.EXP]))
            update = self._linear(xd, Bdecay, ttnn.float32, transpose_a=True)
            ttnn.add(
                ttnn.multiply(
                    state.recurrent, last, input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, 0.0)]
                ),
                update,
                output_tensor=state.recurrent,
            )
            del adt, accumulated, differences, decay, xd, CB, local, from_prefix, y, last, Bdecay, update
        y = ys[0] if len(ys) == 1 else ttnn.concat(ys, dim=2)
        del ys, xs, B, C, B2, C2, dt, conv
        if head_slices:
            y = ttnn.concat([y[:, h : h + 1, :, :] for h in range(16)], dim=3)  # (1,16,P,64) -> (1,1,P,1024)
        else:
            y = ttnn.reshape(ttnn.permute(y, (0, 2, 1, 3)), (1, 1, P, 1024))
        gated = ttnn.multiply(y, ttnn.typecast(gate, ttnn.float32), input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
        grouped = ttnn.reshape(gated, (1, 1, 2 * P, 512) if MAMBA_PREFILL_NORM_ROWS else (1, P, 2, 512))
        grouped = ttnn.rms_norm(grouped, epsilon=self.config.layer_norm_epsilon, compute_kernel_config=self.compute)
        normalized = ttnn.typecast(ttnn.reshape(grouped, (1, 1, P, 1024)), ttnn.bfloat16)
        out = self._finish(self._linear(normalized, self.out_proj), work)
        return out if P == n else out[:, :, :n, :]

    def _attention_prefill_chunk(self, x, *, state, page_table, start_pos, residual=None, metadata=None):
        length = x.shape[2]
        qkv = self._linear(x, self.qkv, dtype=ttnn.bfloat8_b)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn.reshape(qkv, (1, length, 1280)), num_heads=8, num_kv_heads=1, transpose_key=False
        )
        if state.key.dtype != k.dtype:
            k = ttnn.typecast(k, state.key.dtype)
            v = ttnn.typecast(v, state.value.dtype)
        block = state.key.shape[2]
        fill_table = (
            page_table[:, start_pos // block : (start_pos + length) // block]
            if metadata is None
            else metadata.fill_table
        )
        ttnn.experimental.paged_fill_cache(state.key, k, fill_table, batch_idx=0)
        ttnn.experimental.paged_fill_cache(state.value, v, fill_table, batch_idx=0)
        read_table = page_table
        padding_pages = -page_table.shape[-1] % 16
        if padding_pages:
            read_table = ttnn.concat([page_table, ttnn.repeat(page_table[:, -1:], (1, padding_pages))], dim=1)
        out = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            state.key,
            state.value,
            read_table,
            **({"chunk_start_idx": start_pos} if metadata is None else {"chunk_start_idx_tensor": metadata.start}),
            program_config=self.sdpa_prefill_program,
            compute_kernel_config=self.compute,
        )
        out = ttnn.reshape(ttnn.transformer.concatenate_heads(out), (1, 1, length, 1024))
        return self._finish(self._linear(out, self.o), residual)

    @staticmethod
    def _prefill_unaligned_input(hidden_states):
        """A fixed source shape bounds scalar-slice signatures for resume tails."""
        length = hidden_states.shape[2]
        return ttnn.pad(hidden_states, [(0, 0), (0, 0), (0, 512 - length), (0, 0)], value=0.0)

    @staticmethod
    def _prefill_unaligned_output(metadata, length):
        return ttnn.to_layout(metadata.unaligned_output, ttnn.TILE_LAYOUT)[:, :, :length, :]

    def prefill_prepared(self, hidden_states, *, state, page_table, start_pos, metadata):
        """Bounded full-model chunk with runtime attention metadata.

        The legacy decoder API remains available for its independent tests. The
        full model calls this entrypoint only with the canonical bounded spans.
        A sub-tile attention prefix uses the established paged decode path;
        fixed scratch geometry avoids a program for every (length, offset) pair.
        """
        length = hidden_states.shape[2]
        if not 0 <= start_pos < start_pos + length <= self.config.max_position_embeddings:
            raise ValueError("Prefill span must lie within advertised context")
        if self.kind != "attention":
            return self.prefill_forward(hidden_states, state=state, page_table=page_table, start_pos=start_pos)
        if start_pos % 512 and length > 512 - start_pos % 512:
            raise ValueError("An unaligned prepared span must end at or before the next 512-token boundary")
        previous = self._inside_prefill
        self._inside_prefill = True
        try:
            # Tensor-offset SDPA supports tile-aligned starts independently of
            # its 512-token K chunk. Only sub-tile starts need scalar updates.
            if start_pos % 32:
                work = self._prefill_unaligned_input(hidden_states)
                for offset in range(length):
                    inp = work[:, :, offset : offset + 1, :]
                    current = ttnn.reshape(metadata.positions[offset : offset + 1, :], (1,))
                    out = self.decode_forward(inp, state=state, page_table=page_table, current_pos=current)
                    ttnn.experimental.slice_write(
                        ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT),
                        metadata.unaligned_output,
                        [0, 0, offset, 0],
                        [1, 1, offset + 1, out.shape[-1]],
                        [1, 1, 1, 1],
                    )
                return self._prefill_unaligned_output(metadata, length)
            physical = math.ceil(length / 32) * 32
            work = (
                hidden_states
                if physical == length
                else ttnn.pad(hidden_states, [(0, 0), (0, 0), (0, physical - length), (0, 0)], value=0.0)
            )
            out = self._attention_prefill_chunk(
                self._norm(work),
                state=state,
                page_table=page_table,
                start_pos=start_pos,
                residual=work,
                metadata=metadata,
            )
            return out[:, :, :length, :]
        finally:
            self._inside_prefill = previous

    def prefill_moe_prepared(self, hidden_states, *, state, page_table):
        """Pack equal physical buckets; every split starts at a fixed boundary.

        MoE has no temporal state. Padding therefore changes neither cache nor
        valid-token semantics. Binary groups preserve the packed EP/TP path
        without arbitrary mixed-length sums and split offsets in program keys.
        """
        buckets = {}
        for index, hidden in enumerate(hidden_states):
            physical = math.ceil(hidden.shape[2] / 32) * 32
            buckets.setdefault(physical, []).append((index, hidden))
        outputs = [None] * len(hidden_states)
        for physical, requests in buckets.items():
            offset = 0
            while offset < len(requests):
                limit = min(32, 8192 // physical, len(requests) - offset)
                count = 1 << (limit.bit_length() - 1)
                group = requests[offset : offset + count]
                padded = [
                    (
                        x
                        if x.shape[2] == physical
                        else ttnn.pad(x, [(0, 0), (0, 0), (0, physical - x.shape[2]), (0, 0)], value=0.0)
                    )
                    for _, x in group
                ]
                packed = padded[0] if count == 1 else ttnn.concat(padded, dim=2)
                result = self.prefill_forward(packed, state=state, page_table=page_table, start_pos=0)
                for row, (index, hidden) in enumerate(group):
                    block = result[:, :, row * physical : (row + 1) * physical, :]
                    outputs[index] = block[:, :, : hidden.shape[2], :]
                offset += count
        return outputs

    def decode_forward(self, hidden_states, *, state, page_table=None, current_pos=None, mask_inactive=False):
        """Decode independent request rows; position and route values stay on device."""
        if isinstance(hidden_states, (tuple, list)):
            requests = list(zip(hidden_states, state, page_table, current_pos, strict=True))
            if not requests:
                return []
            native = True
            if self.kind == "attention":
                native = all(
                    st.key is state[0].key
                    and st.value is state[0].value
                    and tuple(p.shape) == tuple(page_table[0].shape)
                    for _, st, p, _ in requests
                )
            elif self.kind == "mamba":
                buffers = [id(t) for st in state for t in (*st.conv, st.recurrent)]
                native = len(set(buffers)) == len(buffers)
            if not native:
                return [
                    self.decode_forward(x, state=st, page_table=p, current_pos=c, mask_inactive=mask_inactive)
                    for x, st, p, c in requests
                ]
            outputs = []
            for start in range(0, len(requests), self.decode_batch_chunk):
                group = requests[start : start + self.decode_batch_chunk]
                if len(group) == 1:
                    x, st, p, c = group[0]
                    outputs.append(
                        self.decode_forward(x, state=st, page_table=p, current_pos=c, mask_inactive=mask_inactive)
                    )
                    continue
                if self.kind == "mamba":
                    inp = ttnn.concat([r[0] for r in group], dim=0)
                    combined = DecoderState(
                        conv=[
                            ttnn.concat([r[1].conv[i] for r in group], dim=0, memory_config=ttnn.L1_MEMORY_CONFIG)
                            for i in range(3)
                        ],
                        recurrent=ttnn.concat(
                            [r[1].recurrent for r in group], dim=0, memory_config=ttnn.L1_MEMORY_CONFIG
                        ),
                    )
                    active_mask = None
                    if mask_inactive:
                        positions = ttnn.concat([r[3] for r in group], dim=0)
                        active_mask = ttnn.reshape(
                            ttnn.typecast(ttnn.ge(ttnn.to_layout(positions, ttnn.TILE_LAYOUT), 0), ttnn.bfloat16),
                            (len(group), 1, 1, 1),
                        )
                    out = self._mamba_decode(self._norm(inp), state=combined, residual=inp, active_mask=active_mask)
                    for i, (_, destination, _, _) in enumerate(group):
                        for new, old in zip(combined.conv, destination.conv):
                            ttnn.copy(new[i : i + 1], old)
                        ttnn.copy(combined.recurrent[i : i + 1], destination.recurrent)
                        outputs.append(out[i : i + 1])
                    del combined, inp, out
                else:
                    # Convert the complete batch once instead of untilizing each request.
                    inp = ttnn.reshape(
                        ttnn.concat([r[0] for r in group], dim=0), (1, 1, len(group), hidden_states[0].shape[-1])
                    )
                    if self.kind == "moe":
                        out = self._finish(self._moe(self._norm(inp)), inp)
                    else:
                        pages = ttnn.concat([r[2] for r in group], dim=0)
                        positions = ttnn.concat([r[3] for r in group], dim=0)
                        out = self._attention_decode(
                            self._norm(inp), state=group[0][1], page_table=pages, current_pos=positions, residual=inp
                        )
                    # Batch-axis slices preserve the request API without per-row tilization.
                    packed = ttnn.reshape(out, (len(group), 1, 1, hidden_states[0].shape[-1]))
                    outputs.extend(packed[i : i + 1] for i in range(len(group)))
            return outputs
        x = self._norm(hidden_states)
        if self.kind == "attention":
            return self._attention_decode(
                x, state=state, page_table=page_table, current_pos=current_pos, residual=hidden_states
            )
        if self.kind == "mamba":
            active_mask = None
            if mask_inactive:
                active_mask = ttnn.reshape(
                    ttnn.typecast(ttnn.ge(ttnn.to_layout(current_pos, ttnn.TILE_LAYOUT), 0), ttnn.bfloat16),
                    (1, 1, 1, 1),
                )
            return self._mamba_decode(x, state=state, residual=hidden_states, active_mask=active_mask)
        return self._finish(self._moe(x), hidden_states)

    def prefill_forward(self, hidden_states, *, state, page_table=None, start_pos=0, positions=None):
        if isinstance(hidden_states, (tuple, list)):
            if self.kind == "moe":
                return self._moe_prefill_requests(
                    hidden_states, state=state, page_table=page_table, start_pos=start_pos, positions=positions
                )
            return [
                self.prefill_forward(x, state=st, page_table=p, start_pos=a, positions=positions)
                for x, st, p, a in zip(hidden_states, state, page_table, start_pos, strict=True)
            ]
        length = hidden_states.shape[2]
        if not 0 <= start_pos < start_pos + length <= self.config.max_position_embeddings:
            raise ValueError("Prefill span must lie within advertised context")
        previous = self._inside_prefill
        self._inside_prefill = True
        outputs = []
        offset = 0
        try:
            if self.kind == "mamba" and MAMBA_SCAN_CHUNK and length > 1:
                return self._mamba_prefill_span(hidden_states, state=state)
            while offset < length:
                absolute, remaining = start_pos + offset, length - offset
                count = min(self.prefill_chunk, remaining)
                if self.kind == "attention" and absolute % 512:
                    count = 1
                    inp = hidden_states[:, :, offset : offset + 1, :]
                    out = self.decode_forward(
                        inp, state=state, page_table=page_table, current_pos=positions[absolute : absolute + 1, :]
                    )
                elif self.kind == "mamba" and count == 1:
                    inp = hidden_states[:, :, offset : offset + 1, :]
                    out = self._mamba_decode(self._norm(inp), state=state, residual=inp, use_mac=False)
                else:
                    if self.kind == "mamba" and count >= 32:
                        count = count // 32 * 32
                    inp = hidden_states[:, :, offset : offset + count, :]
                    physical = math.ceil(count / 32) * 32
                    work = (
                        inp
                        if physical == count
                        else ttnn.pad(inp, [(0, 0), (0, 0), (0, physical - count), (0, 0)], value=0.0)
                    )
                    x = self._norm(work)
                    if self.kind == "attention":
                        out = self._attention_prefill_chunk(
                            x, state=state, page_table=page_table, start_pos=absolute, residual=work
                        )
                    elif self.kind == "mamba":
                        partial = self._mamba_prefill_chunk(x, state=state, valid_length=count if count < 32 else None)
                        out = self._finish(partial, inp)
                    else:
                        out = self._finish(self._moe(x), work)
                    out = out[:, :, :count, :]
                outputs.append(out)
                offset += count
            while len(outputs) > 32:
                outputs = [ttnn.concat(outputs[i : i + 32], dim=2) for i in range(0, len(outputs), 32)]
            return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)
        finally:
            self._inside_prefill = previous
