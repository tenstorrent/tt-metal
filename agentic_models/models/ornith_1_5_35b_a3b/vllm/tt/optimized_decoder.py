# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Optimized single-chip Ornith Qwen3.5 MoE decoder. Setup owns all host conversions.

Inputs are device BF16 [1,1,T,2048]. A request owns its cache/state explicitly.
Prefill accepts logical lengths and processes bounded chunks. Decode consumes one
row, device current position, device page table and device rotary cos/sin. Caller
refreshes these stable tensors before trace replay. State updates are in-place.
"""

from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule


@dataclass
class DecoderState:
    conv: object = None
    recurrent: object = None
    key: object = None
    value: object = None


@dataclass(frozen=True)
class SharedCacheBatchPlan:
    """Owned, immutable page bindings; retain the plan until all its traces end."""

    key: object
    value: object
    page_table: object
    row_page_tables: tuple
    owner_token: object


class OptimizedDecoder(LightweightModule):
    # Named construction policy. Router, normalization and recurrence retain
    # their accuracy-sensitive FP32 policy; stage 8 chooses any BFP4 default.
    ATTENTION_WEIGHT_DTYPE = "bfloat8_b"
    LINEAR_INPUT_WEIGHT_DTYPE = None
    LINEAR_OUTPUT_WEIGHT_DTYPE = None
    EXPERT_WEIGHT_DTYPE = "bfloat8_b"
    ATTENTION_FIDELITY = "HiFi2"
    PREFILL_ATTENTION_FIDELITY = "HiFi4"
    PREFILL_LINEAR_CORRECTION = ("input", "output")
    PREFILL_CORRECTION_MODE = "prepared"
    SHARED_FIDELITY = "LoFi"
    ATTENTION_FP32_DEST = True
    ATTENTION_PACKER_L1_ACC = True
    PREFILL_ATTENTION_FP32_DEST = True
    PREFILL_ATTENTION_PACKER_L1_ACC = True
    PREFILL_BF16_ACC_MIN_TOKENS = {"full_attention": 512}
    SHARED_FP32_DEST = False
    SHARED_PACKER_L1_ACC = True
    ATTN_INPUT_CORES = 8
    OUTPUT_INPUT_CORES = 8
    PREFILL_ATTN_BLOCK = 16
    PREFILL_OUTPUT_BLOCK = 16
    PREFILL_OUT_BLOCK_H = 4
    PREFILL_PROJECTION_INTERLEAVED = True
    PREFILL_PROJECTION_INPUT_L1 = False
    PREFILL_PROJECTION_GRID = (11, 8)
    PREFILL_OUT_BLOCK_W = 8
    SPLIT_HEADS_PROJECTION = False
    SPLIT_LINEAR_PROJECTION = 0  # 3: qkv/z/ab; 4: qkv/z/a/b, same packed consumers
    KV_CACHE_DTYPE = "bfloat8_b"
    SDPA_FIDELITY = "HiFi2"
    SDPA_FP32_DEST = True
    ROUTER_FIDELITY = "HiFi4"
    ROUTER_PROGRAM = False
    ROUTER_PROGRAM_GRID = (8, 4)
    ROUTER_K_BLOCK = 8
    ROUTER_INPUT_L1 = False
    SHARED_PROGRAM = False
    SHARED_PROGRAM_GRID = (11, 4)
    SHARED_K_BLOCKS = (16, 8)
    SHARED_INPUT_L1 = False
    DIRECT_DOT_FIDELITY = "HiFi4"
    DIRECT_DOT_INPUT_L1 = False
    ATTENTION_DECODE_INPUT_DTYPE = None
    SPARSE_DOWN_WEIGHT_DTYPE = None
    BATCH_DIRECT_GATHER = False
    BATCH_DIRECT_POST_COMBINE_MAX = 9
    GENERALIZED_SCALAR_GATE = True
    GENERALIZED_BATCH_GATE = False
    BATCH_DIRECT_SLICE_FROM_L1 = False
    PREFILL_SDPA_GRID = (8, 8)
    PREFILL_SDPA_Q_CHUNK = 256
    PREFILL_SDPA_K_CHUNK = 256
    DECODE_SDPA_K_CHUNK = 0
    DRAM_OUTPUT_PIPELINE = False
    DRAM_SPLIT_PROJECTIONS = False
    SPARSE_K_BLOCKS = (1, 1)
    SPARSE_OUT_TILES = (1, 1)
    SPARSE_SUBBLOCK = 1
    SPARSE_TILE_HEIGHT = 32
    SPARSE_OUTPUT_L1 = False
    SPARSE_INPUT_L1 = False
    SPARSE_GATE_WEIGHT_DTYPE = None
    BATCH_RECURRENCE_L1 = True
    BATCH_RECURRENCE_INPLACE = True
    BATCH_RECURRENCE_DRAM_UPDATE = True
    BATCH_RECURRENCE_INPUT_L1 = False
    BATCH_RECURRENCE_L1_INPUTS = ()
    BATCH_RECURRENCE_SHARD = (8, 8)
    BATCH_RECURRENCE_FALLBACK_GRID = (11, 10)
    FUSE_BATCH_SHARED_CACHE = True
    FUSE_BATCH_MOE = True
    FUSE_BATCH_DIRECT = True
    BATCH_DIRECT_MIN_SIZE = 2
    FUSE_BATCH_PROJECTION = True
    FUSE_BATCH_OUTPUT = True
    FUSE_BATCH_BIAS = True
    FUSE_BATCH_HEADS = True
    FUSE_BATCH_RECURRENCE = True
    FUSE_BATCH_CONV = True
    FUSE_BATCH_PACKED = True
    FUSE_BATCH_STATE_COPY = True
    FUSE_BATCH_HISTORY_COPY = True
    BATCH_MOE_GRID = (11, 8)
    FUSE_BATCH_CONV_CARRY = False
    FUSE_BATCH_CONV_PAD = True
    BATCH_CONV_BUCKETS = (4, 8, 16, 32)
    BATCH_CONV_GROUP_SIZE = 8
    BATCH_CONV_GROUP_ADAPTIVE = False
    BATCH_CONV_GROUP_BATCHES = (9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
    BATCH_MIN_SIZE = 3
    FUSE_DISJOINT_HEADS = True
    FUSE_HEAD_CAST = True
    FUSE_STATE_WRITE = True
    FUSE_LLAMA_ROPE = True
    ROPE_PREP_L1 = False
    FUSE_INVERSE_META = True
    FUSE_OUTPUT_DRAM = True
    FUSE_OUTPUT_L1 = True
    KEEP_INTERLEAVED_QK = False
    DRAM_OUTPUT_READERS = 2
    DRAM_OUTPUT_BLOCK = 8
    FUSE_CONV_CARRY = True
    FUSE_QKV_L1 = True
    FUSE_PACKED_DRAM = True
    DRAM_ATTN_READERS = 3
    DRAM_ATTN_BLOCK = 1
    DRAM_ATTN_OUT_TILES = 12
    LINEAR_PROJECTION_WIDTH = 0
    FUSE_NORM_TO_PROJ = True
    FUSE_DECAY_EXP = True
    FUSE_Q_SCALE = True
    FUSE_PREDICTION_DOT = True
    FUSE_OUTPUT_DOT = True
    TOPK_GROUPS = 1
    TOPK_PAD_WIDTH = 256
    FUSE_NATIVE_GATE = False
    PREFILL_CHUNK_SIZES = {"linear_attention": 4096, "full_attention": 8192}
    FUSE_UNIFIED_DIRECT = False
    NATIVE_FIDELITY = "LoFi"
    EXPERT_ND_WIDTHS = (2, 6)
    FUSE_DIRECT_META = True
    NATIVE_DIRECT_RM = False
    NATIVE_INPUT_PREFIX = True
    BATCH_NATIVE_INPUT_RM = False
    FUSE_PAD_CAST = True
    PACK_ROUTER_BF16 = False
    FUSE_ROUTER_GATHER_L1 = True
    DIRECT_DOT_WORKERS = 0
    DIRECT_DOT_SUBBLOCK = 0
    DIRECT_DOT_L1 = False
    FUSE_DIRECT_RESIDUAL = True
    ROOT_NORM_CORES = 8
    FUSE_ROOT_NATIVE_NORM = True
    FUSE_RM_HISTORY = True
    FUSE_LINEAR_HEAD_NORM = False
    FUSE_LINEAR_GATE = True
    PRECISE_LINEAR_PREFILL = True
    FUSE_SCALAR_CASTS = True
    PRECISE_GDN_CHUNK = 32
    PRECISE_GDN_MIN_SEQ = 4096
    PRECISE_GDN_INVERSE = "AUTO"
    FUSE_GATED_NORM = True
    FUSE_DECODE_GATED_NORM = False
    FUSE_BATCH_GATED_NORM = True
    BATCH_GATED_NORM_MIN = 16
    GATED_NORM_MIN_LENGTH = 256
    FUSE_DECODE_RESHAPE = True
    FUSE_REPEAT_HEADS = True
    FUSE_DIRECT_DOT = True
    NATIVE_DIRECT_COMBINE = False
    PACK_ROUTER = True
    FUSE_SHARED_NATIVE = True
    NATIVE_DECODE_GRID = (11, 8)
    FUSE_NATIVE_DIRECT = True
    FUSE_NATIVE_DECODE = True
    FUSE_RESIDUAL = True
    ROUTE_TILED = True
    FUSE_ROUTE_INDEX_COMBINE = True
    FUSE_ROUTE_REDUCE = True
    FUSE_ROUTE_PREFILL = True
    FUSE_DECODE_COMBINE = False
    FUSE_COMBINE = True
    FUSE_RECURRENCE_PREDECAY = False
    FUSE_SOFTPLUS = True
    FUSE_GDN_FLAT = True
    FUSE_GDN_OUTPUT = True
    FUSE_CONV_INPUT = True
    FUSE_NORM_ROPE = False
    FUSE_HEAD_NORM = False
    FUSE_CACHE = True
    MOE_GRID = (8, 8)
    CONV_CHANNEL = 128
    ANCHOR_ACT = False
    CONCAT_DECODE = False
    FUSE_FMA = True
    FUSE_L2 = True
    SHARDED_HEADS = True
    FUSE_MOE = True
    FUSE_CONV = True
    FUSE_HEADS = True
    PACK_ATTENTION = True
    PACK_SHARED = True
    PACK_EXPERTS = True

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs):
        import torch

        self = cls()
        # A later construction policy must not change an existing decoder's
        # cache dtype or execution choices after its weights/configs are built.
        for name in dir(cls):
            if name.isupper():
                value = getattr(cls, name)
                setattr(self, name, value.copy() if isinstance(value, dict) else value)
        self.config = getattr(hf_config, "text_config", hf_config)
        c = self.config
        if (c.hidden_size, c.num_experts, c.num_experts_per_tok, c.moe_intermediate_size) != (2048, 256, 8, 512):
            raise ValueError("Expected Ornith-1.5-35B-A3B text config")
        if mesh_device.get_num_devices() != 1:
            raise ValueError("Optimized decoder requires a 1x1 mesh")
        if self.FUSE_BATCH_DIRECT and not all(
            (self.FUSE_BATCH_MOE, self.FUSE_MOE, self.FUSE_NATIVE_DIRECT, self.FUSE_SHARED_NATIVE, self.PACK_ROUTER)
        ):
            raise ValueError("Direct batch MoE requires native shared experts, packed routing and batch MoE")
        if self.FUSE_ROUTER_GATHER_L1 and (
            self.ROOT_NORM_CORES != 8
            or self.PACK_ROUTER_BF16
            or not all(
                (
                    self.FUSE_MOE,
                    self.FUSE_NATIVE_DECODE,
                    self.FUSE_NATIVE_DIRECT,
                    self.FUSE_SHARED_NATIVE,
                    self.PACK_ROUTER,
                )
            )
        ):
            raise ValueError("Router gather requires eight norm shards, FP32 packed router and native shared decode")
        if self.DIRECT_DOT_WORKERS not in (0, 8, 16) or self.DIRECT_DOT_SUBBLOCK not in (0, 2, 4):
            raise ValueError("Direct dot requires 0/8/16 workers and 0/2/4 subblock width")
        if self.DIRECT_DOT_WORKERS and (
            self.NATIVE_DIRECT_COMBINE
            or not all((self.FUSE_DIRECT_DOT, self.FUSE_NATIVE_DIRECT, self.FUSE_NATIVE_DECODE, self.FUSE_MOE))
        ):
            raise ValueError("Explicit direct dot requires the native decode score-dot path")
        self.kind = c.layer_types[layer_idx]
        if self.kind == "full_attention" and self.FUSE_LLAMA_ROPE:
            if self.FUSE_NORM_ROPE or not (self.FUSE_HEADS and self.SHARDED_HEADS):
                raise ValueError("Fused Q/K RoPE requires native sharded heads and excludes DiT norm/RoPE")
        self.device = mesh_device
        self._shared_cache_owner = object()
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.sdpa_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, self.SDPA_FIDELITY),
            math_approx_mode=False,
            fp32_dest_acc_en=self.SDPA_FP32_DEST,
            packer_l1_acc=False,
        )
        self.router_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, self.ROUTER_FIDELITY),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.dot_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, self.DIRECT_DOT_FIDELITY),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.attention_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, self.ATTENTION_FIDELITY),
            math_approx_mode=False,
            fp32_dest_acc_en=self.ATTENTION_FP32_DEST,
            packer_l1_acc=self.ATTENTION_PACKER_L1_ACC,
        )
        self.prefill_attention_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, self.PREFILL_ATTENTION_FIDELITY),
            math_approx_mode=False,
            fp32_dest_acc_en=self.PREFILL_ATTENTION_FP32_DEST,
            packer_l1_acc=self.PREFILL_ATTENTION_PACKER_L1_ACC,
        )
        self.prefill_attention_bf16_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, self.PREFILL_ATTENTION_FIDELITY),
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=self.PREFILL_ATTENTION_PACKER_L1_ACC,
        )
        self.shared_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, self.SHARED_FIDELITY),
            math_approx_mode=False,
            fp32_dest_acc_en=self.SHARED_FP32_DEST,
            packer_l1_acc=self.SHARED_PACKER_L1_ACC,
        )
        if self.FUSE_ROUTER_GATHER_L1:
            router_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
            self.router_gather_memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(router_grid, [32, 64], ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.router_gather_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 1),
                # The legacy gather factory fixes K-block width to one input
                # shard (8 tiles), regardless of the caller's requested value.
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=2,
                per_core_M=1,
                per_core_N=2,
                fuse_batch=True,
                mcast_in0=False,
                gather_in0=True,
                num_global_cb_receivers=1,
            )
        if self.DIRECT_DOT_WORKERS:
            self.direct_dot_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, self.DIRECT_DOT_WORKERS // 8),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=self.DIRECT_DOT_SUBBLOCK or 4,
                per_core_M=1,
                per_core_N=64 // self.DIRECT_DOT_WORKERS,
                fuse_batch=True,
                mcast_in0=True,
            )
        self.gate_zero = None
        self.softplus_activation = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)]
        self.weights = {}
        if self.NATIVE_INPUT_PREFIX:
            self.native_input_zero = ttnn.from_torch(
                torch.zeros(1, dtype=torch.int32),
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        need_sparse = not (self.FUSE_MOE and self.FUSE_NATIVE_DECODE and self.FUSE_ROUTE_PREFILL)

        def upload(x, dtype=ttnn.bfloat16, dram_sharded=False):
            memory = ttnn.DRAM_MEMORY_CONFIG
            if dram_sharded:
                banks = mesh_device.dram_grid_size().x
                grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
                memory = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(
                        grid,
                        [x.shape[-2], ((x.shape[-1] + banks * 32 - 1) // (banks * 32)) * 32],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                )
            return ttnn.from_torch(
                x.contiguous(),
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory,
            )

        if self.TOPK_GROUPS > 1:
            self.topk_group_offsets = upload(
                (torch.arange(self.TOPK_GROUPS, dtype=torch.int32) * (256 // self.TOPK_GROUPS)).reshape(
                    1, 1, self.TOPK_GROUPS, 1
                ),
                ttnn.uint32,
            )
        if self.FUSE_NATIVE_GATE:
            self.gate_zero = upload(torch.zeros(1, 1, 1, 256), ttnn.float32)
        self.generalized_gate_constants = {}
        for gate_batch in range(1, 33 if self.GENERALIZED_BATCH_GATE else 2 if self.GENERALIZED_SCALAR_GATE else 1):
            grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, row), ttnn.CoreCoord(min(8, gate_batch - row * 8) - 1, row))
                    for row in range((gate_batch + 7) // 8)
                }
            )
            gate_memory = ttnn.create_sharded_memory_config(
                (32, 32),
                grid,
                ttnn.ShardStrategy.HEIGHT,
                use_height_and_width_as_shard_shape=True,
            )
            # Do not reserve every batch's constants in L1 concurrently with
            # the large head-sharded recurrence. Scalar constants stay resident.
            storage = gate_memory if gate_batch == 1 else ttnn.DRAM_MEMORY_CONFIG
            gate_bias = ttnn.from_torch(
                torch.zeros(gate_batch, 16, 16),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=storage,
            )
            gate_indices = ttnn.from_torch(
                torch.arange(256).reshape(1, 16, 16).transpose(-1, -2).repeat(gate_batch, 1, 1).contiguous(),
                device=mesh_device,
                dtype=ttnn.uint16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=storage,
            )
            self.generalized_gate_constants[gate_batch] = (gate_memory, gate_bias, gate_indices)
        for name, value in state_dict.items():
            if not need_sparse and ".experts." in name:
                continue
            if name.endswith("experts.gate_up_proj"):
                gate, up = value.chunk(2, dim=1)
                self.weights["expert_gate"] = upload(
                    gate.transpose(-1, -2).unsqueeze(0),
                    getattr(ttnn, self.SPARSE_GATE_WEIGHT_DTYPE or self.EXPERT_WEIGHT_DTYPE),
                )
                self.weights["expert_up"] = upload(
                    up.transpose(-1, -2).unsqueeze(0),
                    getattr(ttnn, self.SPARSE_GATE_WEIGHT_DTYPE or self.EXPERT_WEIGHT_DTYPE),
                )
            elif name.endswith("experts.down_proj"):
                self.weights["expert_down"] = upload(
                    value.transpose(-1, -2).unsqueeze(0),
                    getattr(ttnn, self.SPARSE_DOWN_WEIGHT_DTYPE or self.EXPERT_WEIGHT_DTYPE),
                )
            elif name.endswith("conv1d.weight"):
                self.taps = [upload(value[:, 0, i].float().reshape(1, 1, 1, -1), ttnn.float32) for i in range(4)]
            elif name.endswith("A_log"):
                self.decay = upload(-value.float().exp().reshape(1, 1, 32), ttnn.float32)
            elif name.endswith("dt_bias"):
                self.dt = upload(value.float().reshape(1, 1, 32), ttnn.float32)
            elif name.endswith("norm.weight"):
                if name == "linear_attn.norm.weight":
                    self.gated_norm_weight = upload(value.reshape(-1), ttnn.bfloat16)
                zero_centered = name != "linear_attn.norm.weight"
                self.weights[name] = upload((value.float() + int(zero_centered)).reshape(1, 1, 1, -1), ttnn.float32)
            else:
                self.weights[name] = upload(
                    value.T,
                    ttnn.float32
                    if name in ("mlp.gate.weight", "mlp.shared_expert_gate.weight")
                    else getattr(
                        ttnn,
                        self.EXPERT_WEIGHT_DTYPE
                        if "shared_expert." in name
                        else (self.LINEAR_OUTPUT_WEIGHT_DTYPE or self.ATTENTION_WEIGHT_DTYPE)
                        if name == "linear_attn.out_proj.weight"
                        else (self.LINEAR_INPUT_WEIGHT_DTYPE or self.ATTENTION_WEIGHT_DTYPE)
                        if name.startswith("linear_attn.in_proj_")
                        else self.ATTENTION_WEIGHT_DTYPE,
                    ),
                    dram_sharded=(
                        self.FUSE_OUTPUT_DRAM and name in ("self_attn.o_proj.weight", "linear_attn.out_proj.weight")
                    )
                    or (
                        (self.DRAM_SPLIT_PROJECTIONS or self.SPLIT_LINEAR_PROJECTION)
                        and (
                            name.startswith("linear_attn.in_proj_")
                            or name in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight")
                        )
                    ),
                )
        if self.PACK_ATTENTION:
            names = (
                ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                if self.kind == "full_attention"
                else [
                    "linear_attn.in_proj_qkv",
                    "linear_attn.in_proj_z",
                    "linear_attn.in_proj_a",
                    "linear_attn.in_proj_b",
                ]
            )
            packed_weight = torch.cat([state_dict[n + ".weight"].T for n in names], dim=-1)
            if self.kind == "linear_attention" and self.LINEAR_PROJECTION_WIDTH:
                if self.LINEAR_PROJECTION_WIDTH < packed_weight.shape[-1]:
                    raise ValueError("Projection padding cannot discard live columns")
                packed_weight = torch.nn.functional.pad(
                    packed_weight, (0, self.LINEAR_PROJECTION_WIDTH - packed_weight.shape[-1])
                )
            self.weights["packed_attn.weight"] = upload(
                packed_weight,
                dtype=getattr(
                    ttnn,
                    (self.LINEAR_INPUT_WEIGHT_DTYPE or self.ATTENTION_WEIGHT_DTYPE)
                    if self.kind == "linear_attention"
                    else self.ATTENTION_WEIGHT_DTYPE,
                ),
                dram_sharded=self.FUSE_PACKED_DRAM,
            )
        if self.SPLIT_LINEAR_PROJECTION:
            if self.SPLIT_LINEAR_PROJECTION not in (3, 4) or self.kind != "linear_attention":
                raise ValueError("Split linear projections require a linear layer and 3 or 4 groups")
            if self.SPLIT_LINEAR_PROJECTION == 3:
                self.weights["linear_attn.in_proj_ab.weight"] = upload(
                    torch.cat(
                        [state_dict["linear_attn.in_proj_a.weight"].T, state_dict["linear_attn.in_proj_b.weight"].T],
                        dim=-1,
                    ),
                    getattr(ttnn, self.LINEAR_INPUT_WEIGHT_DTYPE or self.ATTENTION_WEIGHT_DTYPE),
                    dram_sharded=True,
                )
        if self.FUSE_HEADS and self.kind == "full_attention":
            qg = state_dict["self_attn.q_proj.weight"].reshape(16, 512, 2048)
            norm_rope_order = torch.cat(
                [torch.stack([torch.arange(32), torch.arange(32, 64)], dim=-1).flatten(), torch.arange(64, 256)]
            )
            q_weights = qg[:, :256]
            k_weights = state_dict["self_attn.k_proj.weight"].reshape(2, 256, 2048)
            if self.FUSE_NORM_ROPE or self.FUSE_LLAMA_ROPE:
                q_weights = q_weights[:, norm_rope_order]
                k_weights = k_weights[:, norm_rope_order]
                transform = torch.zeros(1, 1, 32, 32)
                transform[..., torch.arange(0, 32, 2), torch.arange(1, 32, 2)] = 1
                transform[..., torch.arange(1, 32, 2), torch.arange(0, 32, 2)] = -1
                self.rope_transform = upload(transform)
                if self.FUSE_LLAMA_ROPE:
                    for name in ("q", "k"):
                        gamma = state_dict["self_attn." + name + "_norm.weight"].float() + 1
                        self.weights["self_attn." + name + "_norm.weight"] = upload(
                            gamma[norm_rope_order].reshape(1, 1, 1, 256), ttnn.float32
                        )
                    self.rope_compute = ttnn.init_device_compute_kernel_config(
                        mesh_device.arch(),
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        math_approx_mode=False,
                        fp32_dest_acc_en=False,
                        packer_l1_acc=False,
                    )
                    self.rope_transform_decode = ttnn.from_torch(
                        transform.repeat(1, 2, 1, 1),
                        device=mesh_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.create_sharded_memory_config(
                            (32, 32),
                            ttnn.CoreGrid(x=2, y=1),
                            ttnn.ShardStrategy.HEIGHT,
                            use_height_and_width_as_shard_shape=True,
                        ),
                    )
                    if self.FUSE_BATCH_HEADS:
                        self.batch_rope_transform = ttnn.from_torch(
                            transform.repeat(1, 64, 1, 1),
                            device=mesh_device,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            memory_config=ttnn.create_sharded_memory_config(
                                (32, 32),
                                ttnn.CoreGrid(x=8, y=8),
                                ttnn.ShardStrategy.HEIGHT,
                                use_height_and_width_as_shard_shape=True,
                            ),
                        )
            self.weights["heads_attn.weight"] = upload(
                torch.cat(
                    [
                        q_weights.reshape(4096, 2048).T,
                        k_weights.reshape(512, 2048).T,
                        state_dict["self_attn.v_proj.weight"].T,
                        qg[:, 256:].reshape(4096, 2048).T,
                    ],
                    dim=-1,
                ),
                dtype=getattr(ttnn, self.ATTENTION_WEIGHT_DTYPE),
                dram_sharded=self.FUSE_PACKED_DRAM,
            )
        if self.SPLIT_HEADS_PROJECTION and self.kind == "full_attention":
            for role, matrix in {
                "qg": torch.cat([q_weights.reshape(4096, 2048).T, qg[:, 256:].reshape(4096, 2048).T], dim=-1),
                "k": k_weights.reshape(512, 2048).T,
                "v": state_dict["self_attn.v_proj.weight"].T,
            }.items():
                self.weights["heads_split_" + role + ".weight"] = upload(
                    matrix, getattr(ttnn, self.ATTENTION_WEIGHT_DTYPE), dram_sharded=True
                )
            del self.weights["heads_attn.weight"]
        if (self.FUSE_HEAD_NORM or self.FUSE_NORM_ROPE) and self.kind == "full_attention":
            for name, heads in (("q", 16), ("k", 2)):
                gamma = state_dict["self_attn." + name + "_norm.weight"].float() + 1
                if self.FUSE_NORM_ROPE:
                    gamma = gamma[norm_rope_order]
                self.weights[name + "_head_norm"] = upload(gamma.repeat(heads).reshape(1, 1, 1, -1), ttnn.float32)
        if self.PACK_SHARED:
            self.weights["packed_shared.weight"] = upload(
                torch.cat(
                    [
                        state_dict["mlp.shared_expert.gate_proj.weight"].T,
                        state_dict["mlp.shared_expert.up_proj.weight"].T,
                        torch.nn.functional.pad(state_dict["mlp.shared_expert_gate.weight"].T, (0, 31)),
                    ],
                    dim=-1,
                ),
                dtype=getattr(ttnn, self.EXPERT_WEIGHT_DTYPE),
            )
        if self.PACK_EXPERTS and need_sparse:
            self.weights["expert_gate_up"] = upload(
                state_dict["mlp.experts.gate_up_proj"].transpose(-1, -2).unsqueeze(0),
                getattr(ttnn, self.SPARSE_GATE_WEIGHT_DTYPE or self.EXPERT_WEIGHT_DTYPE),
            )
        if self.FUSE_CONV and self.kind == "linear_attention":
            self.conv_taps = [
                upload(state_dict["linear_attn.conv1d.weight"][:, 0, i].reshape(1, 1, -1)) for i in range(4)
            ]
            self.conv_start = ttnn.from_torch(
                torch.tensor([0], dtype=torch.int32),
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            if self.FUSE_CONV_CARRY:
                core = ttnn.CoreCoord(mesh_device.compute_with_storage_grid_size().x - 1, 0)
                grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})
                self.conv_semaphores = [ttnn.create_global_semaphore(mesh_device, grid, 0) for _ in range(3)]
                self.conv_exchange = ttnn.from_torch(
                    torch.zeros(1, 12, 8192),
                    device=mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                self.conv_ends = {
                    1: ttnn.from_torch(
                        torch.tensor([1], dtype=torch.int32),
                        device=mesh_device,
                        dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                }

        if self.FUSE_BATCH_CONV and self.kind == "linear_attention":
            self.batch_conv = {}
            for bucket in self.BATCH_CONV_BUCKETS:
                taps = [
                    upload(state_dict["linear_attn.conv1d.weight"][:, 0, i].repeat(bucket).reshape(1, 1, -1))
                    for i in range(4)
                ]
                exchange = (
                    ttnn.from_torch(
                        torch.zeros(1, 12, bucket * 8192),
                        device=mesh_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                    if self.FUSE_BATCH_CONV_CARRY
                    else None
                )
                self.batch_conv[bucket] = (taps, exchange)

        def upload_expert(weight):
            if self.EXPERT_ND_WIDTHS is None:
                return upload(weight, getattr(ttnn, self.EXPERT_WEIGHT_DTYPE))
            grid = mesh_device.dram_grid_size()
            width = self.EXPERT_ND_WIDTHS[0 if weight.shape[-1] == 512 else 1]
            mem = ttnn.MemoryConfig(
                buffer_type=ttnn.BufferType.DRAM,
                nd_shard_spec=ttnn.NdShardSpec(
                    shard_shape=ttnn.Shape([32, 32 * width]),
                    grid=ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
                    ),
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            return ttnn.from_torch(
                weight.contiguous(),
                device=mesh_device,
                dtype=getattr(ttnn, self.EXPERT_WEIGHT_DTYPE),
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
            )

        if self.FUSE_MOE:
            self.expert_lists = [
                [upload_expert(w.T) for w in values]
                for values in (
                    state_dict["mlp.experts.gate_up_proj"][:, :512],
                    state_dict["mlp.experts.gate_up_proj"][:, 512:],
                    state_dict["mlp.experts.down_proj"],
                )
            ]
            self.expert_ids = ttnn.from_torch(
                torch.arange(256, dtype=torch.int32),
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            self.route_mask = (
                ttnn.from_torch(
                    torch.zeros(1, 256, dtype=torch.int32),
                    device=mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                if self.FUSE_ROUTE_PREFILL
                else None
            )
            if self.FUSE_NATIVE_DIRECT:
                self.direct_n = 9 if self.FUSE_SHARED_NATIVE else 8
                self.direct_e = 257 if self.FUSE_SHARED_NATIVE else 256
                self.direct_zero, self.direct_one, self.direct_offsets = [
                    ttnn.from_torch(a, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
                    for a in (
                        torch.zeros(1, 1, 1, self.direct_e, dtype=torch.int32),
                        torch.ones(1, 1, 1, self.direct_n, dtype=torch.int32),
                        torch.arange(self.direct_n, dtype=torch.int32).reshape(1, 1, 1, self.direct_n) * 32,
                    )
                ]
                if self.FUSE_DIRECT_META:
                    count_base = torch.zeros(1, 1, 1, self.direct_e, dtype=torch.int32)
                    offset_base = count_base.clone()
                    if self.FUSE_SHARED_NATIVE:
                        count_base[..., 256] = 1
                        offset_base[..., 256] = 8 * 32
                    self.direct_count_base, self.direct_offset_base, self.direct_one, self.direct_offsets = [
                        ttnn.from_torch(a, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
                        for a in (
                            count_base,
                            offset_base,
                            torch.ones(1, 1, 1, 8, dtype=torch.int32),
                            torch.arange(8, dtype=torch.int32).reshape(1, 1, 1, 8) * 32,
                        )
                    ]
                if self.NATIVE_DIRECT_COMBINE:
                    index = torch.zeros(self.direct_n, 32, dtype=torch.int32)
                    index[:, 0] = torch.arange(self.direct_n) * 32
                    self.direct_combine_index = ttnn.from_torch(
                        index, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
                    )
                if self.FUSE_INVERSE_META and self.FUSE_DIRECT_META and not self.FUSE_UNIFIED_DIRECT:
                    # Native idx[e] selects the metadata slot; weights still
                    # belong to local expert e. Inactive experts map to a
                    # valid zero-count sentinel, not an out-of-bounds index.
                    index = torch.full((1, 1, 1, self.direct_e), self.direct_n, dtype=torch.int32)
                    if self.FUSE_SHARED_NATIVE:
                        index[..., 256] = 8
                    (
                        self.direct_index_base,
                        self.direct_router_slots,
                        self.direct_slot_counts,
                        self.direct_slot_offsets,
                    ) = [
                        ttnn.from_torch(a, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
                        for a in (
                            index,
                            torch.arange(8, dtype=torch.int32).reshape(1, 1, 1, 8),
                            torch.tensor([[1] * self.direct_n + [0]], dtype=torch.int32),
                            torch.tensor([[32 * i for i in range(self.direct_n)] + [0]], dtype=torch.int32),
                        )
                    ]
                self.direct_expert_lists = self.expert_lists
                self.direct_expert_ids = self.expert_ids
                if self.FUSE_SHARED_NATIVE:
                    self.direct_expert_lists = [
                        v + [upload_expert(state_dict["mlp.shared_expert." + name + "_proj.weight"].T)]
                        for v, name in zip(self.expert_lists, ("gate", "up", "down"))
                    ]
                    self.direct_expert_ids = ttnn.from_torch(
                        torch.arange(257, dtype=torch.int32),
                        device=mesh_device,
                        dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                    self.direct_shared_id = upload(torch.tensor([[[[256]]]], dtype=torch.int32), ttnn.uint32)
                    self.weights["native_shared_gate.weight"] = upload(state_dict["mlp.shared_expert_gate.weight"].T)
                    if self.PACK_ROUTER:
                        self.weights["native_router.weight"] = upload(
                            torch.cat(
                                [
                                    state_dict["mlp.gate.weight"].T.float(),
                                    torch.nn.functional.pad(
                                        state_dict["mlp.shared_expert_gate.weight"].T.float(), (0, 31)
                                    ),
                                ],
                                dim=-1,
                            ),
                            ttnn.bfloat16 if self.PACK_ROUTER_BF16 else ttnn.float32,
                        )
                        if self.FUSE_ROUTER_GATHER_L1:
                            # Gather's interleaved reader reads two N tiles on
                            # EVERY ring core, without masking past logical N.
                            # Eight cores therefore require physical N=512.
                            # Keep the normal 288-column router for other paths.
                            self.weights["native_router_gather.weight"] = upload(
                                torch.nn.functional.pad(
                                    torch.cat(
                                        [
                                            state_dict["mlp.gate.weight"].T.float(),
                                            state_dict["mlp.shared_expert_gate.weight"].T.float(),
                                        ],
                                        dim=-1,
                                    ),
                                    (0, 255),
                                ),
                                ttnn.float32,
                            )
            self.route_index_payloads = {}
            self.expert_meta = {}
            self.combine_rows = {}
            self.combine_zero = (
                upload(torch.zeros(1, 1, 1, 2048))
                if self.FUSE_COMBINE
                or self.FUSE_DECODE_COMBINE
                or self.FUSE_ROUTE_INDEX_COMBINE
                or self.FUSE_BATCH_DIRECT
                else None
            )
            if self.FUSE_BATCH_DIRECT:
                if self.BATCH_DIRECT_GATHER:
                    self.batch_gather_index = ttnn.from_torch(
                        torch.arange(32 * 9, dtype=torch.int32).reshape(32, 9).T.contiguous(),
                        device=mesh_device,
                        dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                self.batch_direct_offsets = ttnn.from_torch(
                    (torch.arange(257, dtype=torch.int32) * 32).reshape(1, 257),
                    device=mesh_device,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                self.batch_direct_meta = {}
                for batch in range(2, 33):
                    base = torch.zeros(1, 1, 1, 257, dtype=torch.int32)
                    base[..., 256] = batch
                    count_base, count_source = [
                        ttnn.from_torch(a, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
                        for a in (base, torch.full((1, 1, 1, 8 * batch), batch, dtype=torch.int32))
                    ]
                    rows = torch.arange(batch).float().reshape(1, 1, batch, 1)
                    self.batch_direct_meta[batch] = (
                        count_base,
                        count_source,
                        upload(rows, ttnn.float32),
                        upload(rows + 256 * 32, ttnn.float32),
                    )
            if self.FUSE_DECODE_COMBINE:
                combine_index = torch.zeros(8, 32, dtype=torch.int32)
                combine_index[:, 0] = torch.arange(8, dtype=torch.int32)
                self.decode_combine_index = ttnn.from_torch(
                    combine_index, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
                )
            self.unified_compute = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
            self.expert_compute = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=getattr(ttnn.MathFidelity, self.NATIVE_FIDELITY),
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
        if self.FUSE_L2:
            self.weights["l2.weight"] = upload(torch.full((1, 1, 1, 128), 128**-0.5), ttnn.float32)
            self.weights["q_l2.weight"] = upload(torch.full((1, 1, 1, 128), 1.0 / 128), ttnn.float32)
            self.weights["linear_q_head_norm"] = upload(
                torch.full((1, 1, 1, 2048), 1.0 / 128 if self.FUSE_Q_SCALE else 128**-0.5), ttnn.float32
            )
            self.weights["linear_k_head_norm"] = upload(torch.full((1, 1, 1, 2048), 128**-0.5), ttnn.float32)
        self.zero_g = upload(torch.zeros(1, 1, 32), ttnn.float32) if self.FUSE_RECURRENCE_PREDECAY else None
        self.sparse_dummy = (
            ttnn.from_torch(
                torch.ones(1, 1, 1, 256), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            if need_sparse
            else None
        )
        if self.kind == "linear_attention":
            i, j = torch.arange(32)[:, None], torch.arange(32)[None, :]
            masks = torch.cat(
                [((i < 16) & (j < 16)).float(), ((i >= 16) & (j >= 16)).float(), ((i >= 16) & (j < 16)).float()], dim=1
            )
            self.gdn_constants = {
                name: upload(value.reshape(1, 1, *value.shape), ttnn.float32)
                for name, value in dict(
                    eye=torch.eye(64), tril=torch.ones(64, 64).tril(), ones=torch.ones(64, 64), masks=masks
                ).items()
            }
        if (self.FUSE_GDN_FLAT or self.PRECISE_GDN_CHUNK == 32) and self.kind == "linear_attention":
            self.gdn_constants32 = {
                name: upload(value.reshape(1, 1, *value.shape), ttnn.float32)
                for name, value in dict(
                    eye=torch.eye(32), tril=torch.ones(32, 32).tril(), ones=torch.ones(32, 32), masks=masks
                ).items()
            }
        # Drop setup-only duplicate projections; packing preserves total logical weights.
        if self.PACK_EXPERTS and need_sparse:
            for name in ("expert_gate", "expert_up"):
                del self.weights[name]
        if self.PACK_SHARED:
            for name in (
                "mlp.shared_expert.gate_proj.weight",
                "mlp.shared_expert.up_proj.weight",
                "mlp.shared_expert_gate.weight",
            ):
                del self.weights[name]
        if self.PACK_ATTENTION and not self.SPLIT_LINEAR_PROJECTION:
            for name in names:
                del self.weights[name + ".weight"]
        if self.FUSE_HEADS and self.kind == "full_attention":
            self.weights.pop("packed_attn.weight", None)
        # Setup-only quantization residual. Both components remain BFP8. The
        # extra projection is used only in accurate prefill; decode reads one.
        self.prefill_corrections = {}
        if self.kind == "linear_attention" and self.PREFILL_LINEAR_CORRECTION:
            for weight_name, weight in self.weights.items():
                # The fair split control changes decode only. Prefill still
                # consumes packed_attn, including its prepared correction.
                if self.SPLIT_LINEAR_PROJECTION and weight_name.startswith("linear_attn.in_proj_"):
                    continue
                role = (
                    "input"
                    if weight_name == "packed_attn.weight" or weight_name.startswith("linear_attn.in_proj_")
                    else "output"
                    if weight_name == "linear_attn.out_proj.weight"
                    else None
                )
                if role not in self.PREFILL_LINEAR_CORRECTION:
                    continue
                original = packed_weight if weight_name == "packed_attn.weight" else state_dict[weight_name].T
                residual = original.float() - ttnn.to_torch(weight).float().reshape_as(original)
                self.prefill_corrections[weight_name] = upload(
                    residual,
                    ttnn.bfloat8_b,
                    dram_sharded=self.FUSE_PACKED_DRAM
                    if weight_name == "packed_attn.weight"
                    else (self.FUSE_OUTPUT_DRAM and role == "output")
                    or (self.DRAM_SPLIT_PROJECTIONS and role == "input"),
                )
        if self.PREFILL_CORRECTION_MODE == "prepared":
            # Preparation is outside measured execution. Keep the accuracy-
            # required BF16 working matrices for prefill, and BFP8 for decode.
            # Reconstruct once; repeated prefill must not reshard whole weights.
            for name, low in self.prefill_corrections.items():
                weight = self.weights[name]
                high = ttnn.to_memory_config(weight, ttnn.DRAM_MEMORY_CONFIG)
                low = ttnn.to_memory_config(low, ttnn.DRAM_MEMORY_CONFIG)
                self.prefill_corrections[name] = ttnn.to_memory_config(
                    ttnn.add(high, low, dtype=ttnn.bfloat16), weight.memory_config()
                )
        if self.PREFILL_PROJECTION_INTERLEAVED:
            if self.PREFILL_CORRECTION_MODE != "prepared":
                raise ValueError("Interleaved prefill weights require prepared precision policy")
            for name, weight in self.weights.items():
                if name in (
                    "heads_attn.weight",
                    "packed_attn.weight",
                    "self_attn.o_proj.weight",
                    "linear_attn.out_proj.weight",
                ):
                    self.prefill_corrections[name] = ttnn.to_memory_config(
                        self.prefill_corrections.get(name, weight), ttnn.DRAM_MEMORY_CONFIG
                    )
        if self.FUSE_BATCH_MOE and self.FUSE_MOE and self.FUSE_ROUTE_PREFILL:
            # Each active expert needs at most one 32-row region for B<=32.
            # Prepare exact metadata-sized index payloads outside trace/runtime.
            for batch in range(2, 33):
                capacity = 256 * batch
                rows = torch.arange(capacity, dtype=torch.int64)
                encoded = torch.zeros(1, 1, capacity, 32)
                for byte in range(3):
                    encoded[0, 0, :, byte] = ((rows >> (8 * byte)) & 255).float()
                self.route_index_payloads[capacity] = upload(encoded.bfloat16())
        if self.FUSE_BATCH_PROJECTION and self.kind == "full_attention":
            self.batch_row_ids = [
                ttnn.from_torch(
                    torch.tensor([i], dtype=torch.int32),
                    device=self.device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                for i in range(32)
            ]
        return self

    def create_state(self, *, context=262144, page_size=32):
        """Setup boundary: one state per request, retain buffers through live traces."""
        import torch

        def zeros(shape, dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(torch.zeros(shape), device=self.device, dtype=dtype, layout=layout)

        if self.kind == "linear_attention":
            return DecoderState(
                conv=zeros(
                    (1, 1, 3, 8192),
                    ttnn.bfloat16,
                    ttnn.ROW_MAJOR_LAYOUT if self.FUSE_RM_HISTORY and self.FUSE_CONV else ttnn.TILE_LAYOUT,
                ),
                recurrent=zeros((1, 32, 128, 128), ttnn.float32),
            )
        if page_size != 32 or context < 256 or context > self.config.max_position_embeddings or context % 256:
            raise ValueError(
                "Cache extent must be a positive multiple of 256 within advertised context; pages are 32 tokens"
            )
        shape = (context // page_size, 2, page_size, 256)
        return DecoderState(
            key=zeros(shape, getattr(ttnn, self.KV_CACHE_DTYPE)), value=zeros(shape, getattr(ttnn, self.KV_CACHE_DTYPE))
        )

    def _prefill_output_block_height(self, per_core_m):
        return max(
            height for height in range(1, min(per_core_m, self.PREFILL_OUT_BLOCK_H) + 1) if per_core_m % height == 0
        )

    def linear(self, x, name, dtype=ttnn.bfloat16, bias=None, *, decode_rows=False, prefill=False, _weight=None):
        if name == "packed_attn" and self.SPLIT_LINEAR_PROJECTION and not prefill and _weight is None:
            if x.shape[-2] == 1 or decode_rows:
                x = self._decode_projection_input(x)
            roles = ("qkv", "z", "ab") if self.SPLIT_LINEAR_PROJECTION == 3 else ("qkv", "z", "a", "b")
            return ttnn.concat(
                [self.linear(x, "linear_attn.in_proj_" + role, dtype=dtype, decode_rows=decode_rows) for role in roles],
                dim=-1,
            )
        if (
            not prefill
            and _weight is None
            and self.ATTENTION_DECODE_INPUT_DTYPE
            and name.startswith(("self_attn.", "linear_attn.", "heads_", "packed_attn"))
        ):
            x = ttnn.typecast(x, getattr(ttnn, self.ATTENTION_DECODE_INPUT_DTYPE))
        if name == "heads_attn" and self.SPLIT_HEADS_PROJECTION:
            if x.shape[-2] == 1 or decode_rows:
                x = self._decode_projection_input(x)
            qg, k, v = [
                self.linear(x, "heads_split_" + role, dtype=dtype, decode_rows=decode_rows, prefill=prefill)
                for role in ("qg", "k", "v")
            ]
            packed = ttnn.concat([qg[..., :4096], k, v, qg[..., 4096:]], dim=-1)
            if (x.shape[-2] == 1 or decode_rows) and self.FUSE_QKV_L1:
                packed = ttnn.to_memory_config(
                    packed,
                    ttnn.create_sharded_memory_config(
                        (32, 256),
                        ttnn.CoreGrid(x=9, y=4),
                        ttnn.ShardStrategy.WIDTH,
                        use_height_and_width_as_shard_shape=True,
                    ),
                )
            return packed
        weight_name = name + ".weight"
        if prefill and _weight is None and weight_name in self.prefill_corrections:
            if self.PREFILL_CORRECTION_MODE == "prepared":
                return self.linear(
                    x,
                    name,
                    dtype=dtype,
                    bias=bias,
                    decode_rows=decode_rows,
                    prefill=True,
                    _weight=self.prefill_corrections[weight_name],
                )
            if self.PREFILL_CORRECTION_MODE == "weights":
                original = self.weights[weight_name]
                high = ttnn.to_memory_config(original, ttnn.DRAM_MEMORY_CONFIG)
                low = ttnn.to_memory_config(self.prefill_corrections[weight_name], ttnn.DRAM_MEMORY_CONFIG)
                restored = ttnn.to_memory_config(ttnn.add(high, low, dtype=ttnn.bfloat16), original.memory_config())
                return self.linear(
                    x, name, dtype=dtype, bias=bias, decode_rows=decode_rows, prefill=True, _weight=restored
                )
            high = self.linear(
                x,
                name,
                dtype=ttnn.float32,
                bias=bias,
                decode_rows=decode_rows,
                prefill=True,
                _weight=self.weights[weight_name],
            )
            low = self.linear(
                x,
                name,
                dtype=ttnn.float32,
                decode_rows=decode_rows,
                prefill=True,
                _weight=self.prefill_corrections[weight_name],
            )
            return ttnn.typecast(ttnn.add(high, low), dtype)
        weight = self.weights[weight_name] if _weight is None else _weight
        compute = (
            self.router_compute
            if name in ("mlp.gate", "native_router", "native_shared_gate", "mlp.shared_expert_gate")
            else (self.prefill_attention_compute if prefill else self.attention_compute)
            if name.startswith(("self_attn.", "linear_attn.", "heads_", "packed_attn"))
            else self.shared_compute
            if name.startswith(("mlp.shared_expert.", "packed_shared"))
            else self.compute
        )
        bf16_min = self.PREFILL_BF16_ACC_MIN_TOKENS.get(self.kind)
        if prefill and compute is self.prefill_attention_compute and bf16_min and x.shape[-2] >= bf16_min:
            # Large-prefill accumulation is independently measured. Short
            # chunks and continuation tails retain the base precision policy.
            compute = self.prefill_attention_bf16_compute
        if (
            prefill
            and self.PREFILL_PROJECTION_INTERLEAVED
            and name in ("heads_attn", "packed_attn", "self_attn.o_proj", "linear_attn.out_proj")
        ):
            mt, nt = (x.shape[-2] + 31) // 32, (weight.shape[-1] + 31) // 32
            gx, gy = self.PREFILL_PROJECTION_GRID
            gy = min(gy, mt)
            pm, pn = (mt + gy - 1) // gy, (nt + gx - 1) // gx
            bh = self._prefill_output_block_height(pm)
            bw = max(v for v in range(1, min(pn, self.PREFILL_OUT_BLOCK_W) + 1) if pn % v == 0)
            sw = max(v for v in range(1, min(bw, 4) + 1) if bw % v == 0)
            config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=self.PREFILL_OUTPUT_BLOCK
                if name.endswith(("o_proj", "out_proj"))
                else self.PREFILL_ATTN_BLOCK,
                out_subblock_h=1,
                out_subblock_w=sw,
                out_block_h=bh,
                out_block_w=bw,
                per_core_M=pm,
                per_core_N=pn,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            )
            return ttnn.linear(
                ttnn.to_memory_config(
                    x, ttnn.L1_MEMORY_CONFIG if self.PREFILL_PROJECTION_INPUT_L1 else ttnn.DRAM_MEMORY_CONFIG
                ),
                weight,
                bias=bias,
                dtype=dtype,
                compute_kernel_config=compute,
                program_config=config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if self.FUSE_OUTPUT_DRAM and name in ("self_attn.o_proj", "linear_attn.out_proj"):
            if x.shape[-2] == 1 or decode_rows:
                mem = ttnn.create_sharded_memory_config(
                    (32, 4096 // self.OUTPUT_INPUT_CORES),
                    ttnn.CoreGrid(x=self.OUTPUT_INPUT_CORES, y=1),
                    ttnn.ShardStrategy.WIDTH,
                    use_height_and_width_as_shard_shape=True,
                )
                x = ttnn.to_memory_config(x, mem)
                if bias is not None:
                    spec = weight.memory_config().shard_spec
                    bias = ttnn.to_memory_config(
                        bias,
                        ttnn.MemoryConfig(
                            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                            ttnn.BufferType.DRAM,
                            ttnn.ShardSpec(spec.grid, [32, 256], ttnn.ShardOrientation.ROW_MAJOR),
                        ),
                    )
                out = ttnn.linear(
                    x,
                    weight,
                    bias=bias,
                    dtype=dtype,
                    compute_kernel_config=compute,
                    program_config=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                        in0_block_w=self.DRAM_OUTPUT_BLOCK,
                        per_core_M=1,
                        per_core_N=8,
                        num_workers_per_dram_bank=self.DRAM_OUTPUT_READERS,
                        use_multicore_pipeline=self.DRAM_OUTPUT_PIPELINE,
                    ),
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                )
                return out if self.FUSE_OUTPUT_L1 else ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
            m_tiles = (x.shape[-2] + 31) // 32
            return ttnn.linear(
                x,
                weight,
                dtype=dtype,
                compute_kernel_config=compute,
                program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(8, min(8, m_tiles)),
                    in0_block_w=self.PREFILL_OUTPUT_BLOCK,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    out_block_h=self._prefill_output_block_height((m_tiles + 7) // 8),
                    out_block_w=8,
                    per_core_M=(m_tiles + 7) // 8,
                    per_core_N=8,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=True,
                ),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        split_projection = (
            name.startswith("heads_split_")
            or self.SPLIT_LINEAR_PROJECTION
            and name.startswith("linear_attn.in_proj_")
            or self.DRAM_SPLIT_PROJECTIONS
            and (
                name.startswith("linear_attn.in_proj_")
                or name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
            )
        )
        if (self.FUSE_PACKED_DRAM and name in ("heads_attn", "packed_attn")) or split_projection:
            n_tiles = (weight.shape[-1] + 31) // 32
            banks = self.device.dram_grid_size().x
            per_n = (n_tiles + banks - 1) // banks
            if x.shape[-2] == 1 or decode_rows:
                x = self._decode_projection_input(x)
                readers = min(self.DRAM_ATTN_READERS, per_n) if split_projection else self.DRAM_ATTN_READERS
                config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=self.DRAM_ATTN_BLOCK,
                    per_core_M=1,
                    per_core_N=per_n
                    if split_projection
                    else 8
                    if self.FUSE_QKV_L1 and name == "heads_attn"
                    else self.DRAM_ATTN_OUT_TILES or per_n,
                    num_workers_per_dram_bank=readers,
                    use_multicore_pipeline=readers > 1,
                )
                out = ttnn.linear(
                    x,
                    weight,
                    dtype=dtype,
                    compute_kernel_config=compute,
                    program_config=config,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                )
                if self.FUSE_QKV_L1 and name == "heads_attn":
                    return out
                return ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
            m_tiles = (x.shape[-2] + 31) // 32
            per_m = (m_tiles + 7) // 8
            # The WIDTH-DRAM reader walks a whole bank shard per K block;
            # narrower out_block_w is not implemented despite passing validation.
            sub_n = max(v for v in range(1, 5) if per_n % v == 0)
            config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, min(8, m_tiles)),
                # Interleaved prefill tuning uses narrow output blocks. This
                # direct-helper path reads an entire DRAM bank shard, so its
                # double-buffered B tiles need the smaller established K block.
                in0_block_w=min(self.PREFILL_ATTN_BLOCK, 4)
                if self.PREFILL_PROJECTION_INTERLEAVED
                else self.PREFILL_ATTN_BLOCK,
                out_subblock_h=1,
                out_subblock_w=sub_n,
                out_block_h=self._prefill_output_block_height(per_m),
                out_block_w=per_n,
                per_core_M=per_m,
                per_core_N=per_n,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            )
            return ttnn.linear(
                x,
                weight,
                dtype=dtype,
                compute_kernel_config=compute,
                program_config=config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        router_program = self.ROUTER_PROGRAM and name in (
            "mlp.gate",
            "native_router",
            "native_shared_gate",
            "mlp.shared_expert_gate",
        )
        shared_program = self.SHARED_PROGRAM and name.startswith(("mlp.shared_expert.", "packed_shared"))
        if router_program or shared_program:
            gx, gy = self.ROUTER_PROGRAM_GRID if router_program else self.SHARED_PROGRAM_GRID
            mt, nt = (x.shape[-2] + 31) // 32, (weight.shape[-1] + 31) // 32
            gy = min(gy, mt)
            pm, pn = (mt + gy - 1) // gy, (nt + gx - 1) // gx
            block_h = self._prefill_output_block_height(pm)
            sub_w = max(v for v in range(1, min(pn, 4) + 1) if pn % v == 0)
            input_l1 = self.ROUTER_INPUT_L1 if router_program else self.SHARED_INPUT_L1
            program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=self.ROUTER_K_BLOCK
                if router_program
                else self.SHARED_K_BLOCKS[int(name.endswith("down_proj"))],
                out_subblock_h=1,
                out_subblock_w=sub_w,
                out_block_h=block_h,
                out_block_w=pn,
                per_core_M=pm,
                per_core_N=pn,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            )
            return ttnn.linear(
                ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG if input_l1 else ttnn.DRAM_MEMORY_CONFIG),
                weight,
                bias=bias,
                dtype=dtype,
                compute_kernel_config=compute,
                program_config=program,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                activation="silu" if self.ANCHOR_ACT and name == "mlp.shared_expert.gate_proj" else None,
            )
        return ttnn.linear(
            x,
            weight,
            bias=bias,
            dtype=dtype,
            activation="silu" if self.ANCHOR_ACT and name == "mlp.shared_expert.gate_proj" else None,
            compute_kernel_config=compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _decode_projection_input(self, x):
        # Share this allocation across same-input split projections. Calls on
        # an already matching shard are no-ops, including the packed path.
        memory = ttnn.create_sharded_memory_config(
            (32, x.shape[-1] // self.ATTN_INPUT_CORES),
            ttnn.CoreGrid(x=self.ATTN_INPUT_CORES, y=1),
            ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        return ttnn.to_memory_config(x, memory)

    def _accurate_sum_squares(self, value):
        """Local SFPU statistics only: no collective or host tensor boundary."""
        return ttnn.rms_norm_pre_all_gather(
            value,
            dtype=ttnn.float32,
            compute_kernel_config=self.compute,
            fast_and_approximate_mode=False,
        )[..., :1]

    def norm(self, x, name, *, decode_rows=False, router_carry=False, precise=False):
        if precise:
            if x.is_sharded():
                x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            if x.layout != ttnn.TILE_LAYOUT:
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            xf = x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)
            stats = self._accurate_sum_squares(xf)
            normalized = ttnn.multiply(
                xf,
                stats,
                input_tensor_b_activations=[
                    ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 1.0 / x.shape[-1]),
                    ttnn.UnaryWithParam(ttnn.UnaryOpType.ADD_UNARY_SFPU, self.config.rms_norm_eps),
                    ttnn.UnaryOpType.RSQRT,
                ],
            )
            return ttnn.multiply(normalized, self.weights[name + ".weight"], dtype=ttnn.bfloat16)
        if self.ROOT_NORM_CORES and x.shape[-1] == 2048 and (x.shape[-2] == 1 or decode_rows):
            cores = self.ROOT_NORM_CORES
            grid = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(min(8, cores) - 1, (cores + 7) // 8 - 1))}
            )
            mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, [32, 2048 // cores], ttnn.ShardOrientation.ROW_MAJOR),
            )
            value = ttnn.to_memory_config(x, mem)
            value = ttnn.rms_norm(
                value,
                epsilon=self.config.rms_norm_eps,
                weight=self.weights[name + ".weight"],
                compute_kernel_config=self.compute,
                memory_config=mem,
                program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=[min(8, cores), (cores + 7) // 8],
                    subblock_w=min(4, 64 // cores),
                    block_h=1,
                    block_w=64 // cores,
                    inplace=False,
                ),
            )
            if self.FUSE_PACKED_DRAM and self.FUSE_NORM_TO_PROJ and name == "input_layernorm":
                return value
            if router_carry and self.FUSE_ROUTER_GATHER_L1 and name == "post_attention_layernorm" and x.shape[-2] == 1:
                return value
            return ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)
        if self.FUSE_ROOT_NATIVE_NORM and x.shape[-1] == 2048 and x.shape[-2] == 1:
            return ttnn.experimental.dit_fused_distributed_rmsnorm(
                x,
                None,
                self.device,
                [],
                epsilon=self.config.rms_norm_eps,
                weight=self.weights[name + ".weight"],
                compute_kernel_config=self.compute,
            )
        if x.layout == ttnn.ROW_MAJOR_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        original_mem = x.memory_config()
        kwargs = {}
        if x.is_sharded():
            x = ttnn.to_memory_config(
                x, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, original_mem.shard_spec)
            )
            kwargs["program_config"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[1, 1],
                subblock_w=4,
                block_h=1,
                block_w=8,
                inplace=False,
            )
        result = ttnn.rms_norm(
            x,
            epsilon=self.config.rms_norm_eps,
            weight=self.weights[name + ".weight"],
            compute_kernel_config=self.compute,
            memory_config=x.memory_config() if x.is_sharded() else ttnn.DRAM_MEMORY_CONFIG,
            **kwargs,
        )

        return ttnn.to_memory_config(result, original_mem) if x.is_sharded() else result

    def moe(self, x):
        """Device router and all eight selected experts, plus gated shared expert."""
        # Each token has independent expert indices. Indexed sparse matmul keeps
        # expert weights resident, gathering only the eight routed expert groups.
        if x.shape[-2] >= 32 or self.FUSE_NATIVE_DECODE:
            return self.moe_prefill(x)
        rows = []
        for i in range(x.shape[-2]):
            row = ttnn.to_memory_config(
                x[:, :, i : i + 1, :],
                ttnn.L1_MEMORY_CONFIG
                if self.SPARSE_INPUT_L1 and self.SPARSE_TILE_HEIGHT == 32
                else ttnn.DRAM_MEMORY_CONFIG,
            )
            logits = self.linear(ttnn.typecast(row, ttnn.float32), "mlp.gate", dtype=ttnn.float32)
            scores, indices = ttnn.topk(ttnn.reshape(logits, (1, 256)), k=8, dim=-1)
            scores = ttnn.softmax(scores, dim=-1, numeric_stable=True, compute_kernel_config=self.compute)
            ids = ttnn.to_layout(ttnn.reshape(ttnn.typecast(indices, ttnn.uint16), (1, 1, 1, 8)), ttnn.ROW_MAJOR_LAYOUT)
            expert_row = row
            if self.SPARSE_TILE_HEIGHT != 32:
                if self.SPARSE_TILE_HEIGHT > 1:
                    expert_row = ttnn.pad(row, ((0, 0), (0, 0), (0, self.SPARSE_TILE_HEIGHT - 1), (0, 0)), value=0.0)
                expert_row = ttnn.tilize(
                    expert_row,
                    tile=ttnn.Tile((self.SPARSE_TILE_HEIGHT, 32)),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

            def sparse(a, w, sparse_a=False, activate=False):
                n = w.shape[-1]
                role = int(sparse_a)
                tiles = self.SPARSE_OUT_TILES[role]
                cores = (n // 32 + tiles - 1) // tiles
                pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(min(8, cores), (cores + 7) // 8),
                    in0_block_w=self.SPARSE_K_BLOCKS[role],
                    out_subblock_h=1,
                    out_subblock_w=min(tiles, self.SPARSE_SUBBLOCK),
                    out_block_h=1,
                    out_block_w=tiles,
                    per_core_M=1,
                    per_core_N=tiles,
                    fuse_batch=False,
                    fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if activate else None,
                    mcast_in0=True,
                )
                return ttnn.sparse_matmul(
                    a,
                    w,
                    sparsity=self.sparse_dummy,
                    indices=ids,
                    is_input_a_sparse=sparse_a,
                    is_input_b_sparse=True,
                    program_config=pc,
                    compute_kernel_config=self.shared_compute,
                    dtype=ttnn.bfloat16,
                    output_tile=ttnn.Tile((self.SPARSE_TILE_HEIGHT, 32)),
                    memory_config=ttnn.L1_MEMORY_CONFIG if self.SPARSE_OUTPUT_L1 else ttnn.DRAM_MEMORY_CONFIG,
                )

            def standard_active(value):
                # Each expert owns one complete tiny tile. Folding those rows
                # avoids ND reshape's fixed-32 padding and retile's clamping
                # mismatch for a batch of individually padded one-row inputs.
                flat = ttnn.experimental.view(value, (1, 1, 8 * self.SPARSE_TILE_HEIGHT, value.shape[-1]))
                if flat.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                    # Generic to_memory_config constructs a standard-tile copy.
                    # An explicit destination preserves the custom page shape.
                    buffer = ttnn.allocate_tensor_on_device(
                        ttnn.TensorSpec(
                            flat.shape,
                            flat.dtype,
                            ttnn.TILE_LAYOUT,
                            ttnn.DRAM_MEMORY_CONFIG,
                            tile=ttnn.Tile((self.SPARSE_TILE_HEIGHT, 32)),
                        ),
                        self.device,
                    )
                    ttnn.copy(flat, buffer)
                    flat = buffer
                return ttnn.tilize(
                    flat,
                    tile=ttnn.Tile((32, 32)),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

            if self.PACK_EXPERTS:
                gu = sparse(expert_row, self.weights["expert_gate_up"])
                if self.SPARSE_TILE_HEIGHT != 32:
                    gu = standard_active(gu)
                g, u = gu[..., :512], gu[..., 512:]
            else:
                g = sparse(expert_row, self.weights["expert_gate"], activate=self.ANCHOR_ACT)
                if self.SPARSE_TILE_HEIGHT != 32:
                    # Preserve each tiny L1 result before the next sparse call:
                    # the tiny-tile sparse writer can overlap live L1 storage.
                    g = standard_active(g)
                u = sparse(expert_row, self.weights["expert_up"])
                if self.SPARSE_TILE_HEIGHT != 32:
                    u = standard_active(u)
            product = ttnn.multiply(g, u, input_tensor_a_activations=[] if self.ANCHOR_ACT else [ttnn.UnaryOpType.SILU])
            if self.SPARSE_TILE_HEIGHT != 32:
                product = ttnn.tilize(
                    product,
                    tile=ttnn.Tile((self.SPARSE_TILE_HEIGHT, 32)),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                product = ttnn.experimental.view(product, (1, 8, self.SPARSE_TILE_HEIGHT, 512))
            y = sparse(product, self.weights["expert_down"], True)
            small_rows = None
            if self.SPARSE_TILE_HEIGHT != 32:
                small_rows = ttnn.to_layout(standard_active(y), ttnn.ROW_MAJOR_LAYOUT)
                if self.SPARSE_TILE_HEIGHT > 1:
                    small_rows = ttnn.slice(
                        small_rows,
                        (0, 0, 0, 0),
                        (1, 1, 8 * self.SPARSE_TILE_HEIGHT, 2048),
                        (1, 1, self.SPARSE_TILE_HEIGHT, 1),
                    )
            if self.FUSE_DECODE_COMBINE:
                expert_rows = (
                    small_rows
                    if small_rows is not None
                    else ttnn.reshape(ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT), (1, 1, 8, 2048))
                )
                expert_rows = ttnn.pad(expert_rows, ((0, 0), (0, 0), (0, 24), (0, 0)), value=0.0)
                combine_weights = ttnn.pad(
                    ttnn.reshape(scores, (1, 8, 1, 1)), ((0, 0), (0, 0), (0, 31), (0, 0)), value=0.0
                )
                y = ttnn.experimental.nomic_embed_text_v2_moe.routed_combine(
                    expert_rows,
                    self.decode_combine_index,
                    combine_weights,
                    self.combine_zero,
                    num_tokens=1,
                    compute_kernel_config=self.compute,
                )
            else:
                if small_rows is not None:
                    y = ttnn.to_layout(ttnn.reshape(small_rows, (1, 8, 1, 2048)), ttnn.TILE_LAYOUT)
                y = ttnn.multiply(y, ttnn.reshape(scores, (1, 8, 1, 1)))
                y = ttnn.sum(y, dim=1, keepdim=True)
            shared = self.shared_expert(row)
            rows.append(ttnn.add(ttnn.typecast(y, ttnn.bfloat16), shared))
        return rows[0] if len(rows) == 1 else ttnn.concat(rows, dim=2)

    def linear_attention(
        self,
        x,
        state,
        residual=None,
        projection=None,
        return_projection_input=False,
        return_recurrence_inputs=False,
        *,
        precise=None,
        prefill=False,
    ):
        t = x.shape[-2]
        if precise is None:
            precise = self.PRECISE_LINEAR_PREFILL and t > 1
        gated_norm = (
            (self.FUSE_GATED_NORM and t >= self.GATED_NORM_MIN_LENGTH if t > 1 else self.FUSE_DECODE_GATED_NORM)
            and self.PACK_ATTENTION
            and self.FUSE_GDN_OUTPUT
            and not precise
        )
        head_major_y = None
        flat_gdn = self.FUSE_GDN_FLAT and self.FUSE_CONV and t > 1 and not precise
        native_linear_norm = (
            self.FUSE_LINEAR_HEAD_NORM
            and self.FUSE_CONV
            and self.FUSE_L2
            and t == 1
            and not self.FUSE_RECURRENCE_PREDECAY
            and not precise
        )
        projections = projection
        if projections is None:
            projections = self.linear(x, "packed_attn", prefill=prefill) if self.PACK_ATTENTION else None
        if self.PACK_ATTENTION:
            qkv = projections if self.FUSE_CONV_INPUT and self.FUSE_CONV and t >= 3 else projections[..., :8192]
        else:
            qkv = self.linear(x, "linear_attn.in_proj_qkv", prefill=prefill)
        history = None if self.FUSE_CONV else ttnn.concat([state.conv, qkv], dim=2)
        if self.FUSE_CONV:
            physical = (t + 31) // 32 * 32
            inp = ttnn.pad(qkv, ((0, 0), (0, 0), (0, physical - t), (0, 0)), value=0.0) if physical != t else qkv
            hist = ttnn.to_layout(ttnn.reshape(state.conv, (1, 3, 8192)), ttnn.ROW_MAJOR_LAYOUT)
            conv = (
                ttnn.experimental.kda.qkv_causal_conv1d_silu_with_carry
                if self.FUSE_CONV_CARRY
                else ttnn.experimental.kda.qkv_causal_conv1d_silu
            )
            extra = (
                dict(
                    actual_end=self.conv_ends[t],
                    exchange_buffer=self.conv_exchange,
                    exchange_semaphores=self.conv_semaphores,
                )
                if self.FUSE_CONV_CARRY
                else dict(predecessor_carry=hist)
            )
            conv_outputs = conv(
                ttnn.reshape(inp, (1, physical, inp.shape[-1])),
                hist,
                *self.conv_taps,
                2048,
                2048,
                4096,
                actual_start=self.conv_start,
                program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=self.CONV_CHANNEL),
                compute_kernel_config=self.compute,
                **extra,
            )
            q, k, v = conv_outputs[:3]
            if self.FUSE_CONV_CARRY:
                tail = ttnn.reshape(conv_outputs[3], tuple(state.conv.shape))
            elif self.FUSE_RM_HISTORY:
                tail = ttnn.to_layout(qkv[:, :, -3:, :8192] if t >= 3 else qkv, ttnn.ROW_MAJOR_LAYOUT)
                if t < 3:
                    tail = ttnn.concat([state.conv, tail], dim=2)[:, :, -3:, :]
            else:
                tail = qkv[:, :, -3:, :8192] if t >= 3 else ttnn.concat([state.conv, qkv], dim=2)[:, :, -3:, :]
            ttnn.copy(tail, state.conv)
            if not flat_gdn:

                def heads(value, count):
                    value = value[:, :t, :]
                    if t == 1 and self.FUSE_DECODE_RESHAPE:
                        return ttnn.to_layout(
                            ttnn.reshape(ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT), (1, t, count, 128)),
                            ttnn.TILE_LAYOUT,
                            **({"dtype": ttnn.float32} if self.FUSE_HEAD_CAST else {}),
                        )
                    return ttnn.reshape(value, (1, t, count, 128))

                if native_linear_norm:
                    q, k = [
                        ttnn.experimental.dit_fused_distributed_rmsnorm(
                            ttnn.reshape(value[:, :t, :], (1, 1, t, 2048)),
                            None,
                            self.device,
                            [],
                            epsilon=1e-6 / 128,
                            num_heads_per_device=16,
                            per_head_norm=True,
                            weight=self.weights[f"linear_{name}_head_norm"],
                            dtype=ttnn.float32,
                            compute_kernel_config=self.compute,
                        )
                        for value, name in [(q, "q"), (k, "k")]
                    ]
                    # Normalize/split returns [1,16,T,128], already in head-major order.
                    v = heads(v, 32)
                else:
                    q, k, v = heads(q, 16), heads(k, 16), heads(v, 32)
        else:
            out = None
            for i in range(4):
                term = ttnn.multiply(ttnn.typecast(history[:, :, i : i + t, :], ttnn.float32), self.taps[i])
                out = term if out is None else ttnn.add(out, term)
            ttnn.copy(history[:, :, -3:, :], state.conv)
            qkv = ttnn.silu(ttnn.typecast(out, ttnn.bfloat16))
            q = ttnn.reshape(qkv[..., :2048], (1, t, 16, 128))
            k = ttnn.reshape(qkv[..., 2048:4096], (1, t, 16, 128))
            v = ttnn.reshape(qkv[..., 4096:], (1, t, 32, 128))

        if return_recurrence_inputs:
            return q, k, v

        def l2(y, query=False):
            if y.dtype != ttnn.float32:
                y = ttnn.typecast(y, ttnn.float32)
            if precise:
                return ttnn.multiply(
                    y,
                    self._accurate_sum_squares(y),
                    input_tensor_b_activations=[
                        ttnn.UnaryWithParam(ttnn.UnaryOpType.ADD_UNARY_SFPU, 1e-6),
                        ttnn.UnaryOpType.RSQRT,
                    ],
                )
            if self.FUSE_L2:
                y = ttnn.to_layout(y, ttnn.TILE_LAYOUT)
                return ttnn.rms_norm(
                    y,
                    epsilon=1e-6 / 128,
                    weight=self.weights[
                        (
                            "q_l2.weight"
                            if query and self.FUSE_Q_SCALE and t == 1 and not self.FUSE_RECURRENCE_PREDECAY
                            else "l2.weight"
                        )
                    ],
                    compute_kernel_config=self.compute,
                )
            return ttnn.multiply(y, ttnn.rsqrt(ttnn.add(ttnn.sum(ttnn.multiply(y, y), dim=-1, keepdim=True), 1e-6)))

        a = ttnn.reshape(
            (
                projections[..., 12288:12320]
                if self.PACK_ATTENTION
                else self.linear(x, "linear_attn.in_proj_a", prefill=prefill)
            ),
            (1, t, 32),
        )
        b = ttnn.reshape(
            (
                projections[..., 12320:12352]
                if self.PACK_ATTENTION
                else self.linear(x, "linear_attn.in_proj_b", prefill=prefill)
            ),
            (1, t, 32),
        )
        scalar_casts = self.FUSE_SCALAR_CASTS and t == 1 and not precise and not self.FUSE_RECURRENCE_PREDECAY
        beta = ttnn.sigmoid(b)
        if not scalar_casts:
            beta = ttnn.typecast(beta, ttnn.float32)
        # FP32-first MAC widens a inside SFPU; ordinary mixed ADD uses FPU.
        biased = (
            ttnn.mac(self.dt, 1.0, a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if scalar_casts
            else ttnn.add(ttnn.typecast(a, ttnn.float32), self.dt)
        )
        g = (
            ttnn.multiply(self.decay, biased, input_tensor_b_activations=self.softplus_activation)
            if self.FUSE_SOFTPLUS
            else ttnn.multiply(self.decay, ttnn.softplus(biased))
        )
        if not flat_gdn and not native_linear_norm:
            q, k = l2(q, query=True), l2(k)
        if t == 1 and self.FUSE_RECURRENCE_PREDECAY:
            decayed = ttnn.multiply(state.recurrent, ttnn.reshape(ttnn.exp(g), (1, 32, 1, 1)))
            y, s = ttnn.transformer.fused_recurrent_gated_delta_rule(
                q,
                k,
                v,
                self.zero_g,
                beta,
                initial_state=decayed,
                output_final_state=True,
                compute_kernel_config=self.compute,
            )
        elif t == 1:
            # Keep the state/decay update on FP32 SFPU. The native recurrent
            # kernel's FPU scalar multiply truncates near-unit decay on this
            # checkpoint (see AUTODEBUG/AUTOFIX evidence).
            if native_linear_norm:
                q = ttnn.repeat_interleave(q, 2, dim=1)
                if not self.FUSE_OUTPUT_DOT:
                    q = ttnn.transpose(q, -1, -2)
                k_row = ttnn.repeat_interleave(k, 2, dim=1)
                k = ttnn.transpose(k_row, -1, -2)
            elif self.FUSE_REPEAT_HEADS:
                q = ttnn.repeat_interleave(
                    ttnn.permute(q, (0, 2, 1, 3) if self.FUSE_OUTPUT_DOT else (0, 2, 3, 1)), 2, dim=1
                )
                k_row = ttnn.repeat_interleave(ttnn.permute(k, (0, 2, 1, 3)), 2, dim=1)
                k = ttnn.transpose(k_row, -1, -2)
            else:
                q = ttnn.permute(
                    ttnn.repeat_interleave(q, 2, dim=2), (0, 2, 1, 3) if self.FUSE_OUTPUT_DOT else (0, 2, 3, 1)
                )
                repeated_k = ttnn.repeat_interleave(k, 2, dim=2)
                k_row = ttnn.permute(repeated_k, (0, 2, 1, 3)) if self.FUSE_PREDICTION_DOT else None
                k = ttnn.permute(repeated_k, (0, 2, 3, 1))
            value = ttnn.permute(v if v.dtype == ttnn.float32 else ttnn.typecast(v, ttnn.float32), (0, 2, 1, 3))
            rate = ttnn.reshape(beta, (1, 32, 1, 1))
            if self.FUSE_DECAY_EXP:
                s = ttnn.multiply(
                    state.recurrent, ttnn.reshape(g, (1, 32, 1, 1)), input_tensor_b_activations=[ttnn.UnaryOpType.EXP]
                )
            else:
                decay = ttnn.reshape(ttnn.exp(g), (1, 32, 1, 1))
                s = ttnn.multiply(state.recurrent, decay)
            prediction = (
                ttnn.matmul(k_row, s, compute_kernel_config=self.compute, dtype=ttnn.float32)
                if self.FUSE_PREDICTION_DOT
                else ttnn.sum(ttnn.multiply(s, k), dim=-2, keepdim=True)
            )
            delta = ttnn.multiply(
                ttnn.subtract(value, prediction),
                rate,
                **({"dtype": ttnn.float32, "fast_and_approximate_mode": False} if scalar_casts else {}),
            )
            s = (
                ttnn.addcmul(
                    s,
                    k,
                    delta,
                    value=1.0,
                    **({"output_tensor": state.recurrent} if self.FUSE_STATE_WRITE else {}),
                )
                if self.FUSE_FMA
                else ttnn.add(s, ttnn.multiply(k, delta))
            )
            scaled_q = q if self.FUSE_Q_SCALE and self.FUSE_L2 and not precise else ttnn.multiply(q, 128**-0.5)
            y = (
                ttnn.matmul(scaled_q, s, compute_kernel_config=self.compute, dtype=ttnn.float32)
                if self.FUSE_OUTPUT_DOT
                else ttnn.sum(ttnn.multiply(s, scaled_q), dim=-2, keepdim=True)
            )
            y = ttnn.permute(y, (0, 2, 1, 3))
        else:
            if flat_gdn and physical != t:
                g = ttnn.pad(g, ((0, 0), (0, physical - t), (0, 0)), value=0.0)
                beta = ttnn.pad(beta, ((0, 0), (0, physical - t), (0, 0)), value=0.0)
            # C32 AUTO wins on long prefill; retain C64 for short requests.
            precise_c32 = precise and self.PRECISE_GDN_CHUNK == 32 and t >= self.PRECISE_GDN_MIN_SEQ
            y, s = ttnn.transformer.chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                initial_state=state.recurrent,
                output_final_state=True,
                output_head_major=self.FUSE_GDN_OUTPUT,
                chunk_size=32 if flat_gdn or precise_c32 else 64,
                wy_inverse=getattr(ttnn.ChunkGdnWyInverse, self.PRECISE_GDN_INVERSE)
                if precise
                else ttnn.ChunkGdnWyInverse.AUTO,
                **(self.gdn_constants32 if flat_gdn or precise_c32 else self.gdn_constants),
            )
            if gated_norm:
                head_major_y = ttnn.reshape(y, (32, physical if flat_gdn else t, 128))
            if self.FUSE_GDN_OUTPUT and not gated_norm:
                y = ttnn.permute(ttnn.reshape(y, (1, 32, physical if flat_gdn else t, 128)), (0, 2, 1, 3))
            if flat_gdn and not gated_norm:
                y = y[:, :t, :, :]
        if not (t == 1 and self.FUSE_STATE_WRITE and self.FUSE_FMA and not self.FUSE_RECURRENCE_PREDECAY):
            ttnn.copy(s, state.recurrent)
        if gated_norm:
            if head_major_y is None:
                head_major_y = ttnn.reshape(ttnn.permute(y, (0, 2, 1, 3)), (32, t, 128))
            y = self._gated_head_norm(head_major_y, projections, t)
            return (
                y if return_projection_input else self.linear(y, "linear_attn.out_proj", bias=residual, prefill=prefill)
            )
        y = ttnn.reshape(ttnn.typecast(y, ttnn.bfloat16), (1, 1, t * 32, 128))
        z = (
            projections[..., 8192:12288]
            if self.PACK_ATTENTION
            else self.linear(x, "linear_attn.in_proj_z", prefill=prefill)
        )
        if t == 1 and self.FUSE_DECODE_RESHAPE:
            z = ttnn.to_layout(
                ttnn.reshape(ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT), (1, 1, 32, 128)), ttnn.TILE_LAYOUT
            )
        else:
            z = ttnn.reshape(z, (1, 1, t * 32, 128))
        y = (
            ttnn.multiply(
                self.norm(y, "linear_attn.norm", precise=precise), z, input_tensor_b_activations=[ttnn.UnaryOpType.SILU]
            )
            if self.FUSE_LINEAR_GATE
            else ttnn.multiply(self.norm(y, "linear_attn.norm", precise=precise), ttnn.silu(z))
        )
        y = ttnn.reshape(y, (1, 1, t, 4096))
        return y if return_projection_input else self.linear(y, "linear_attn.out_proj", bias=residual, prefill=prefill)

    def _gated_head_norm(self, head_major, packed, length):
        """Native sigmoid-gated normalization plus Z gives RMSNorm(Y)*SiLU(Z)."""
        physical = (length + 31) // 32 * 32
        head_major = ttnn.typecast(head_major, ttnn.bfloat16)
        if head_major.shape[1] != physical:
            head_major = ttnn.pad(head_major, ((0, 0), (0, physical - head_major.shape[1]), (0, 0)), value=0.0)
        gate = ttnn.reshape(packed, (1, length, packed.shape[-1]))
        if length != physical:
            gate = ttnn.pad(gate, ((0, 0), (0, physical - length), (0, 0)), value=0.0)
        y = ttnn.experimental.kda.sigmoid_gated_rms_norm(
            head_major,
            gate,
            self.gated_norm_weight,
            32,
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=self.compute,
            output_dtype=ttnn.bfloat16,
            gate_column_offset=8192,
        )
        y = ttnn.multiply(y, gate[..., 8192:12288])
        return ttnn.reshape(y[:, :length, :], (1, 1, length, 4096))

    def _decode_attention_residual(
        self,
        x,
        *,
        state,
        page_table=None,
        current_pos=None,
        cos=None,
        sin=None,
        projection=None,
        return_projection_input=False,
        prefill=False,
    ):
        """Request-local attention/cache work, shared by scalar and batch decode."""
        if x.shape[-2] != 1:
            raise ValueError("decode_forward requires one token per request")
        h = x if projection is not None else self.norm(x, "input_layernorm")
        a = (
            self.linear_attention(
                h,
                state,
                residual=x if self.FUSE_RESIDUAL else None,
                projection=projection,
                return_projection_input=return_projection_input,
                prefill=prefill,
            )
            if self.kind == "linear_attention"
            else self.full_attention(
                h,
                state,
                page_table,
                current_pos,
                cos,
                sin,
                residual=x if self.FUSE_RESIDUAL else None,
                projection=projection,
                return_projection_input=return_projection_input,
                prefill=prefill,
            )
        )
        return a if return_projection_input or self.FUSE_RESIDUAL else ttnn.add(x, a)

    def decode_forward(self, x, *, state, page_table=None, current_pos=None, cos=None, sin=None, prefill=False):
        """One token [1,1,1,2048]; all mutable execution operands are device tensors."""
        h = self._decode_attention_residual(
            x, state=state, page_table=page_table, current_pos=current_pos, cos=cos, sin=sin, prefill=prefill
        )
        normalized = self.norm(h, "post_attention_layernorm", router_carry=self.FUSE_ROUTER_GATHER_L1)
        if self.FUSE_OUTPUT_DRAM and self.FUSE_OUTPUT_L1:
            # The next norm consumes the projection directly in L1. Export
            # the residual only for the later generic score-dot bias reader.
            h = ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG)
        if (
            self.FUSE_DIRECT_RESIDUAL
            and not self.NATIVE_DIRECT_COMBINE
            and self.FUSE_DIRECT_DOT
            and self.FUSE_NATIVE_DECODE
            and self.FUSE_NATIVE_DIRECT
            and self.FUSE_SHARED_NATIVE
        ):
            return self.moe_prefill(normalized, residual=h, scalar_decode=True)
        if self.FUSE_ROUTER_GATHER_L1 or self.DIRECT_DOT_WORKERS:
            return ttnn.add(h, self.moe_prefill(normalized, scalar_decode=True))
        return ttnn.add(h, self.moe(normalized))

    def moe_prefill(self, x, residual=None, *, batch_decode=False, scalar_decode=False):
        """Batched expert baseline: evaluate experts, apply the actual top-eight gate."""
        batch_direct = batch_decode and self.FUSE_BATCH_DIRECT and x.shape[-2] >= self.BATCH_DIRECT_MIN_SIZE
        shared_score = None
        if self.FUSE_SHARED_NATIVE and self.PACK_ROUTER and (x.shape[-2] == 1 or batch_direct):
            if scalar_decode and self.FUSE_ROUTER_GATHER_L1:
                # Preserve the norm's eight BF16 width shards through the FP32
                # router cast. The FFN receives a separate BF16 boundary below.
                router_input = ttnn.typecast(x, ttnn.float32, memory_config=x.memory_config())
                packed = ttnn.linear(
                    router_input,
                    self.weights["native_router_gather.weight"],
                    dtype=ttnn.float32,
                    compute_kernel_config=self.router_compute,
                    program_config=self.router_gather_config,
                    memory_config=self.router_gather_memory,
                )
                packed = ttnn.to_memory_config(packed, ttnn.L1_MEMORY_CONFIG)
            else:
                packed = self.linear(ttnn.typecast(x, ttnn.float32), "native_router", dtype=ttnn.float32)
            logits, shared_score = packed[..., :256], ttnn.typecast(packed[..., 256:257], ttnn.bfloat16)
        else:
            logits = self.linear(ttnn.typecast(x, ttnn.float32), "mlp.gate", dtype=ttnn.float32)
        if (self.GENERALIZED_SCALAR_GATE and x.shape[-2] == 1) or (self.GENERALIZED_BATCH_GATE and batch_decode):
            gate_batch = x.shape[-2]
            gate_memory, gate_bias, gate_indices = self.generalized_gate_constants[gate_batch]
            if gate_batch > 1:
                gate_bias = ttnn.to_memory_config(gate_bias, gate_memory)
                gate_indices = ttnn.to_memory_config(gate_indices, gate_memory)
            gate_input = ttnn.to_layout(
                ttnn.reshape(
                    ttnn.to_layout(ttnn.typecast(logits, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT), (gate_batch, 16, 16)
                ),
                ttnn.TILE_LAYOUT,
                memory_config=gate_memory,
            )
            outputs = [
                ttnn.empty(
                    (gate_batch, 1, 16),
                    device=self.device,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=gate_memory,
                )
                for dtype in (ttnn.bfloat16, ttnn.uint16)
            ]
            scores, ids = ttnn.experimental.deepseek.moe.generalized_moe_gate(
                gate_input,
                bias_tensor=gate_bias,
                input_indices_tensor=gate_indices,
                output_tensor=outputs[0],
                output_indices_tensor=outputs[1],
                eps=0.0,
                scaling_factor=1.0,
                enable_sigmoid=False,
                topk=8,
                output_softmax=True,
            )
            routed = []
            for tensor, dtype in ((scores, ttnn.float32), (ids, ttnn.uint32)):
                tensor = ttnn.to_memory_config(ttnn.typecast(tensor, dtype), ttnn.DRAM_MEMORY_CONFIG)
                if gate_batch > 1:
                    tensor = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
                tensor = ttnn.reshape(tensor, (1, 1, gate_batch, 16))
                if gate_batch > 1:
                    tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
                routed.append(tensor[..., :8])
            scores, ids = routed
        elif self.FUSE_NATIVE_GATE and x.shape[-2] == 1:
            shifted = ttnn.subtract(logits, ttnn.add(ttnn.max(logits, dim=-1, keepdim=True), 20.0))
            scores, ids = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
                shifted,
                self.gate_zero,
                n_groups=1,
                summed_experts_per_group=1,
                topk_groups=1,
                n_activated_experts=8,
                route_scale=1.0,
                epsilon=0.0,
                stable_sort=True,
                score_func="sigmoid",
            )
            scores = ttnn.typecast(scores, ttnn.float32)
            ids = ttnn.typecast(ids, ttnn.uint32)
        else:
            if self.TOPK_PAD_WIDTH > 256 and x.shape[-2] == 1:
                logits = ttnn.pad(logits, ((0, 0), (0, 0), (0, 0), (0, self.TOPK_PAD_WIDTH - 256)), value=-float("inf"))
            if self.TOPK_GROUPS > 1 and x.shape[-2] == 1:
                groups = self.TOPK_GROUPS
                values, local_ids = ttnn.topk(ttnn.reshape(logits, (1, 1, groups, 256 // groups)), k=8, dim=-1)
                candidates = ttnn.reshape(ttnn.add(local_ids, self.topk_group_offsets), (1, 1, 1, 8 * groups))
                scores, selected = ttnn.topk(ttnn.reshape(values, (1, 1, 1, 8 * groups)), k=8, dim=-1)
                ids = ttnn.gather(candidates, -1, selected)
            else:
                scores, ids = ttnn.topk(logits, k=8, dim=-1)
            scores = ttnn.softmax(scores, dim=-1, numeric_stable=True, compute_kernel_config=self.compute)
        if batch_direct:
            return self.moe_native_batch_direct(x, scores, ids, shared_score)
        if self.FUSE_NATIVE_DIRECT and x.shape[-2] == 1:
            if scalar_decode and self.FUSE_ROUTER_GATHER_L1:
                x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            return self.moe_native_direct(x, scores, ids, shared_score, residual, scalar_decode=scalar_decode)
        if self.FUSE_ROUTE_PREFILL:
            return self.moe_routed_prefill(x, scores, ids)
        weights = None
        if not (self.FUSE_COMBINE and self.FUSE_MOE):
            weights = ttnn.scatter(
                ttnn.zeros_like(ttnn.typecast(logits, ttnn.bfloat16)),
                dim=-1,
                index=ids,
                src=ttnn.typecast(scores, ttnn.bfloat16),
            )

        def mm(a, name):
            return ttnn.matmul(
                a,
                self.weights[name],
                activation="silu" if self.ANCHOR_ACT and name == "expert_gate" else None,
                compute_kernel_config=self.compute,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        if self.FUSE_MOE:
            t = x.shape[-2]
            physical = (t + 31) // 32 * 32
            inp = ttnn.pad(x, ((0, 0), (0, 0), (0, physical - t), (0, 0)), value=0.0) if physical != t else x
            # Native direct-write reader currently checks destination offsets
            # against input capacity even with read_x_at_offset=False. Padding
            # enlarges only allocation capacity; input_m_tiles remains physical/32.
            inp = ttnn.typecast(inp, ttnn.bfloat8_b)
            inp = ttnn.pad(inp, ((0, 0), (0, 0), (0, 255 * physical), (0, 0)), value=0.0)
            out = ttnn.empty(
                (1, 1, 256 * physical, 2048), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            counts, offsets = self.expert_meta[physical]
            out = ttnn.bringup.moe_fused_swiglu(
                inp,
                *self.expert_lists,
                counts,
                self.expert_ids,
                input_m_tiles=physical // 32,
                output=out,
                expert_region_offsets=offsets,
                read_x_at_offset=False,
                compute_kernel_config=self.expert_compute,
                core_grid=ttnn.CoreCoord(*self.MOE_GRID),
            )
            down = None if self.FUSE_COMBINE else ttnn.reshape(out, (1, 256, physical, 2048))[:, :, :t, :]
        else:
            if self.PACK_EXPERTS:
                gu = mm(x, "expert_gate_up")
                gate, up = gu[..., :512], gu[..., 512:]
            else:
                gate, up = mm(x, "expert_gate"), mm(x, "expert_up")
            down = mm(
                ttnn.multiply(gate, up, input_tensor_a_activations=[] if self.ANCHOR_ACT else [ttnn.UnaryOpType.SILU]),
                "expert_down",
            )
        if self.FUSE_COMBINE and self.FUSE_MOE:
            index = ttnn.add(
                ttnn.multiply(ttnn.typecast(ids, ttnn.float32), float(physical)),
                self.combine_rows[physical][:, :, :t, :],
            )
            combine_weights = ttnn.typecast(ttnn.typecast(scores, ttnn.bfloat16), ttnn.float32)
            if physical != t:
                padding = ((0, 0), (0, 0), (0, physical - t), (0, 0))
                index = ttnn.pad(index, padding, value=0.0)
                combine_weights = ttnn.pad(combine_weights, padding, value=0.0)
            index = ttnn.to_layout(ttnn.typecast(ttnn.permute(index, (0, 3, 2, 1)), ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
            combine_weights = ttnn.permute(combine_weights, (0, 3, 2, 1))
            routed = ttnn.experimental.nomic_embed_text_v2_moe.routed_combine(
                ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT),
                ttnn.reshape(index, (8, physical)),
                combine_weights,
                self.combine_zero,
                num_tokens=t,
                compute_kernel_config=self.compute,
            )
            return ttnn.add(routed, self.shared_expert(x))
        weights = ttnn.permute(weights, (0, 3, 2, 1))
        routed = ttnn.sum(ttnn.multiply(down, weights), dim=1, keepdim=True)
        shared = self.shared_expert(x)
        return ttnn.add(ttnn.typecast(routed, ttnn.bfloat16), shared)

    def _native_input_prefix(self, x, capacity, *, row_major):
        """Reserve the native ABI capacity; initialize only its one input tile.

        read_x_at_offset=False and input_m_tiles=1 make all active experts read
        this tile. Indexed copy supports a single source entry into a larger
        destination. Unused capacity is never consumed by this native mode.
        """
        layout = ttnn.ROW_MAJOR_LAYOUT if row_major else ttnn.TILE_LAYOUT
        dtype = ttnn.bfloat16 if row_major else ttnn.bfloat8_b
        source = ttnn.to_layout(x, layout)
        if x.shape[-2] < 32:
            source = ttnn.pad(source, ((0, 0), (0, 0), (0, 32 - x.shape[-2]), (0, 0)), value=0.0)
        if not row_major:
            source = ttnn.typecast(source, dtype)
        storage = ttnn.empty(
            (capacity // 32, 1, 32, 2048),
            dtype=dtype,
            layout=layout,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.experimental.kda.indexed_entry_copy(source, storage, self.native_input_zero, index_scale=1, index_offset=0)
        return ttnn.experimental.view(storage, (1, 1, capacity, 2048))

    def moe_native_direct(self, x, scores, ids, shared_score=None, residual=None, *, scalar_decode=False):
        """One-token FFN: all active experts read the same row, in router order."""
        if self.FUSE_SHARED_NATIVE:
            if not self.FUSE_DIRECT_META:
                ids = ttnn.concat([ids, self.direct_shared_id], dim=-1)
            shared_score = ttnn.typecast(
                ttnn.sigmoid(self.linear(x, "native_shared_gate") if shared_score is None else shared_score),
                ttnn.float32,
            )
            scores = ttnn.concat([scores, shared_score], dim=-1)
        ids = ttnn.to_layout(ids, ttnn.ROW_MAJOR_LAYOUT)
        inverse = self.FUSE_INVERSE_META and self.FUSE_DIRECT_META and not self.FUSE_UNIFIED_DIRECT
        if inverse:
            expert_ids = ttnn.reshape(
                ttnn.scatter(self.direct_index_base, dim=-1, index=ids, src=self.direct_router_slots),
                (1, self.direct_e),
            )
            counts, offsets = self.direct_slot_counts, self.direct_slot_offsets
        else:
            expert_ids = self.direct_expert_ids
            counts = ttnn.scatter(
                self.direct_count_base if self.FUSE_DIRECT_META else self.direct_zero,
                dim=-1,
                index=ids,
                src=self.direct_one,
            )
            offsets = ttnn.scatter(
                self.direct_offset_base if self.FUSE_DIRECT_META else self.direct_zero,
                dim=-1,
                index=ids,
                src=self.direct_offsets,
            )
        capacity = self.direct_n * 32
        if self.NATIVE_INPUT_PREFIX:
            inp = self._native_input_prefix(x, capacity, row_major=self.NATIVE_DIRECT_RM)
        elif self.NATIVE_DIRECT_RM:
            inp = ttnn.pad(
                ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), ((0, 0), (0, 0), (0, capacity - 1), (0, 0)), value=0.0
            )
        elif self.FUSE_PAD_CAST:
            inp = ttnn.typecast(ttnn.pad(x, ((0, 0), (0, 0), (0, capacity - 1), (0, 0)), value=0.0), ttnn.bfloat8_b)
        else:
            inp = ttnn.pad(ttnn.typecast(x, ttnn.bfloat8_b), ((0, 0), (0, 0), (0, capacity - 1), (0, 0)), value=0.0)
        out = ttnn.empty((1, 1, capacity, 2048), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        if self.FUSE_UNIFIED_DIRECT:
            inp = ttnn.repeat(inp[:, :, :32, :], (1, 1, self.direct_n, 1))
            out = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
                inp,
                ttnn.reshape(offsets, (self.direct_e,)),
                ttnn.reshape(counts, (self.direct_e,)),
                self.direct_expert_ids,
                *self.direct_expert_lists,
                32,
                compute_kernel_config=self.unified_compute,
            )
        else:
            out = ttnn.bringup.moe_fused_swiglu(
                inp,
                *self.direct_expert_lists,
                counts if inverse else ttnn.reshape(counts, (1, self.direct_e)),
                expert_ids,
                input_m_tiles=1,
                output=out,
                expert_region_offsets=offsets if inverse else ttnn.reshape(offsets, (1, self.direct_e)),
                read_x_at_offset=False,
                compute_kernel_config=self.expert_compute,
                core_grid=ttnn.CoreCoord(*self.NATIVE_DECODE_GRID),
            )
        if self.NATIVE_DIRECT_COMBINE:
            combine_weights = ttnn.pad(
                ttnn.reshape(scores, (1, self.direct_n, 1, 1)), ((0, 0), (0, 0), (0, 31), (0, 0)), value=0.0
            )
            out = ttnn.experimental.nomic_embed_text_v2_moe.routed_combine(
                ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT),
                self.direct_combine_index,
                combine_weights,
                self.combine_zero,
                num_tokens=1,
                compute_kernel_config=self.compute,
            )
        else:
            out = ttnn.reshape(out, (1, self.direct_n, 32, 2048))[:, :, :1, :]
            if self.FUSE_DIRECT_DOT:
                dot_options = {}
                if scalar_decode and self.DIRECT_DOT_WORKERS:
                    dot_options = {
                        "program_config": self.direct_dot_config,
                        "memory_config": ttnn.L1_MEMORY_CONFIG if self.DIRECT_DOT_L1 else ttnn.DRAM_MEMORY_CONFIG,
                    }
                dot_input = ttnn.reshape(out, (1, 1, self.direct_n, 2048))
                if self.DIRECT_DOT_INPUT_L1:
                    scores, dot_input = [ttnn.to_memory_config(v, ttnn.L1_MEMORY_CONFIG) for v in (scores, dot_input)]
                out = ttnn.linear(
                    scores,
                    dot_input,
                    bias=residual,
                    compute_kernel_config=self.dot_compute,
                    dtype=ttnn.float32 if residual is None else ttnn.bfloat16,
                    **dot_options,
                )
            else:
                out = ttnn.sum(ttnn.multiply(out, ttnn.reshape(scores, (1, self.direct_n, 1, 1))), dim=1, keepdim=True)
        if out.dtype != ttnn.bfloat16:
            out = ttnn.typecast(out, ttnn.bfloat16)
        return out if self.FUSE_SHARED_NATIVE else ttnn.add(out, self.shared_expert(x))

    def moe_native_batch_direct(self, x, scores, ids, shared_score):
        """One tile of distinct decode rows per active expert; no token dispatch."""
        batch = x.shape[-2]
        if not 2 <= batch <= 32:
            raise ValueError("Direct batch MoE requires between 2 and 32 decode rows")
        count_base, count_source, token_rows, shared_rows = self.batch_direct_meta[batch]
        flat_ids = ttnn.reshape(ttnn.to_layout(ids, ttnn.ROW_MAJOR_LAYOUT), (1, 1, 1, 8 * batch))
        # Repeated IDs always store the same B, so every active expert owns
        # one tile containing ALL request rows. Unselected expert/token pairs
        # are computed but never gathered; selected pairs retain row identity.
        counts = ttnn.scatter(count_base, dim=-1, index=flat_ids, src=count_source)
        capacity = 257 * 32
        row_major = not self.ROUTE_TILED if self.BATCH_NATIVE_INPUT_RM is None else self.BATCH_NATIVE_INPUT_RM
        if self.NATIVE_INPUT_PREFIX:
            inp = self._native_input_prefix(x, capacity, row_major=row_major)
        elif row_major:
            inp = ttnn.pad(
                ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), ((0, 0), (0, 0), (0, capacity - batch), (0, 0)), value=0.0
            )
        else:
            inp = ttnn.typecast(ttnn.pad(x, ((0, 0), (0, 0), (0, capacity - batch), (0, 0)), value=0.0), ttnn.bfloat8_b)
        out = ttnn.empty((1, 1, capacity, 2048), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        out = ttnn.bringup.moe_fused_swiglu(
            inp,
            *self.direct_expert_lists,
            ttnn.reshape(counts, (1, 257)),
            self.direct_expert_ids,
            input_m_tiles=1,
            output=out,
            expert_region_offsets=self.batch_direct_offsets,
            read_x_at_offset=False,
            compute_kernel_config=self.expert_compute,
            core_grid=ttnn.CoreCoord(*self.BATCH_MOE_GRID),
        )
        # Native reader currently bounds destination offsets by input capacity,
        # hence the padded allocation above despite reading only its first tile.
        index = ttnn.add(ttnn.multiply(ttnn.typecast(ids, ttnn.float32), 32.0), token_rows)
        index = ttnn.concat([index, shared_rows], dim=-1)
        weights = ttnn.concat(
            [
                ttnn.typecast(ttnn.typecast(scores, ttnn.bfloat16), ttnn.float32),
                ttnn.typecast(ttnn.sigmoid(shared_score), ttnn.float32),
            ],
            dim=-1,
        )
        if batch < 32:
            padding = ((0, 0), (0, 0), (0, 32 - batch), (0, 0))
            # Zero-weight padded slots still reference a written, finite row.
            index = ttnn.pad(index, padding, value=256 * 32)
            weights = ttnn.pad(weights, padding, value=0.0)
        if self.BATCH_DIRECT_GATHER:
            flat = ttnn.to_layout(
                ttnn.reshape(ttnn.to_layout(ttnn.typecast(index, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), (1, 32 * 9)),
                ttnn.TILE_LAYOUT,
            )
            selected = ttnn.tosa_gather(ttnn.reshape(out, (1, capacity, 2048)), flat)
            return ttnn.experimental.nomic_embed_text_v2_moe.routed_combine(
                ttnn.to_layout(ttnn.reshape(selected, (1, 1, 32 * 9, 2048)), ttnn.ROW_MAJOR_LAYOUT),
                self.batch_gather_index,
                ttnn.permute(weights, (0, 3, 2, 1)),
                self.combine_zero,
                num_tokens=batch,
                compute_kernel_config=self.compute,
            )
        if batch <= self.BATCH_DIRECT_POST_COMBINE_MAX:
            table = ttnn.reshape(ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT), (capacity, 2048))
            row_index = ttnn.to_layout(ttnn.typecast(index, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
            row_weights = ttnn.to_layout(ttnn.typecast(weights, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
            combined = ttnn.bringup.post_combine_reduce(
                table,
                row_weights,
                combine_row_index=row_index,
                output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            return combined if batch == 32 else combined[..., :batch, :]
        index = ttnn.transpose(
            ttnn.reshape(ttnn.to_layout(ttnn.typecast(index, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), (32, 9)),
            0,
            1,
        )
        return ttnn.experimental.nomic_embed_text_v2_moe.routed_combine(
            ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT),
            index,
            ttnn.permute(weights, (0, 3, 2, 1)),
            self.combine_zero,
            num_tokens=batch,
            compute_kernel_config=self.compute,
        )

    def moe_routed_prefill(self, x, scores, ids):
        """Local-only native dispatch, actual expert counts, fused SwiGLU/reduce."""
        t = x.shape[-2]
        physical = (t + 31) // 32 * 32
        # At most eight assignments per token, plus 31 alignment rows per expert.
        capacity = (8 * t + 31 * min(256, 8 * t) + 31) // 32 * 32
        ids16 = ttnn.typecast(ids, ttnn.uint16)
        histogram_ids = ids16
        # The pinned masked_bincount splits rows over a fixed 64-core grid (rows % 64 == 0). Pad rows carry
        # expert id 256, which the op skips (expert_idx < n_routed_experts), so the counts are unchanged.
        hist_rows = (t + 63) // 64 * 64
        if hist_rows != t:
            histogram_ids = ttnn.pad(ids16, ((0, 0), (0, 0), (0, hist_rows - t), (0, 0)), value=256)
        hist = ttnn.experimental.deepseek_prefill.masked_bincount(
            ttnn.reshape(histogram_ids, (hist_rows, 8)),
            self.route_mask,
            256,
            8,
        )
        counts = ttnn.reshape(hist, (1, 256))
        sizes = ttnn.typecast(ttnn.to_layout(ttnn.reshape(hist, (1, 1, 1, 256)), ttnn.TILE_LAYOUT), ttnn.float32)
        sizes = ttnn.multiply(ttnn.ceil(ttnn.multiply(sizes, 1.0 / 32)), 32.0)
        regions = ttnn.subtract(ttnn.cumsum(sizes, dim=3), sizes)
        regions = ttnn.reshape(ttnn.to_layout(ttnn.typecast(regions, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), (1, 256))
        offsets = regions
        dispatched, metadata = ttnn.experimental.deepseek_prefill.dispatch(
            ttnn.reshape(x, (1, t, 2048)),
            ttnn.reshape(ttnn.to_layout(ids16, ttnn.ROW_MAJOR_LAYOUT), (1, t, 8)),
            offsets,
            self.route_mask,
            dispatch_group_size=1,
            experts_per_chip=256,
            num_routed_experts=256,
            num_experts_per_tok=8,
            metadata_len=3,
            max_dispatch_buffer_token_size=capacity,
            cluster_axis=0,
            num_links=0,
        )
        if self.ROUTE_TILED:
            dispatched = ttnn.to_layout(dispatched, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        out = ttnn.empty((1, 1, capacity, 2048), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        out = ttnn.bringup.moe_fused_swiglu(
            dispatched,
            *self.expert_lists,
            counts,
            self.expert_ids,
            input_m_tiles=physical // 32,
            output=out,
            expert_region_offsets=regions,
            read_x_at_offset=True,
            compute_kernel_config=self.expert_compute,
            core_grid=ttnn.CoreCoord(*(self.BATCH_MOE_GRID if 1 < t <= 32 else self.MOE_GRID)),
        )
        if self.FUSE_ROUTE_INDEX_COMBINE:
            # Route a three-byte row index through the native metadata consumer;
            # gather/reduce the wide expert output in the dedicated local op.
            routed_index = ttnn.experimental.deepseek_prefill.combine(
                self.route_index_payloads[capacity],
                metadata,
                counts,
                regions,
                dispatch_group_size=1,
                experts_per_chip=256,
                num_experts_per_tok=8,
                seq_len_per_chip=physical,
                cluster_axis=0,
                num_links=0,
            )
            encoded = ttnn.typecast(
                ttnn.to_layout(ttnn.reshape(routed_index, (1, 1, physical * 8, 32)), ttnn.TILE_LAYOUT), ttnn.float32
            )
            index = ttnn.add(
                encoded[..., :1],
                ttnn.add(ttnn.multiply(encoded[..., 1:2], 256.0), ttnn.multiply(encoded[..., 2:3], 65536.0)),
            )
            index = ttnn.transpose(
                ttnn.reshape(ttnn.to_layout(ttnn.typecast(index, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), (physical, 8)),
                0,
                1,
            )
            weights = ttnn.typecast(ttnn.typecast(scores, ttnn.bfloat16), ttnn.float32)
            if physical != t:
                weights = ttnn.pad(weights, ((0, 0), (0, 0), (0, physical - t), (0, 0)), value=0.0)
            weights = ttnn.permute(weights, (0, 3, 2, 1))
            routed = ttnn.experimental.nomic_embed_text_v2_moe.routed_combine(
                ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT),
                index,
                weights,
                self.combine_zero,
                num_tokens=t,
                compute_kernel_config=self.compute,
            )
            return ttnn.add(routed, self.shared_expert(x))
        combined = ttnn.experimental.deepseek_prefill.combine(
            out,
            metadata,
            counts,
            regions,
            dispatch_group_size=1,
            experts_per_chip=256,
            num_experts_per_tok=8,
            seq_len_per_chip=physical,
            cluster_axis=0,
            num_links=0,
        )
        if physical != t:
            scores = ttnn.pad(scores, ((0, 0), (0, 0), (0, physical - t), (0, 0)), value=0.0)
        if self.FUSE_ROUTE_REDUCE:
            routed = ttnn.bringup.post_combine_reduce(
                combined,
                ttnn.to_layout(ttnn.typecast(scores, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT),
                output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            combined = ttnn.to_layout(ttnn.reshape(combined, (1, physical, 8, 2048)), ttnn.TILE_LAYOUT)
            weights = ttnn.reshape(ttnn.typecast(scores, ttnn.bfloat16), (1, physical, 8, 1))
            routed = ttnn.reshape(
                ttnn.sum(ttnn.multiply(combined, weights), dim=2, keepdim=True), (1, 1, physical, 2048)
            )
        return ttnn.add(routed[:, :, :t, :], self.shared_expert(x))

    def shared_expert(self, x):
        if self.PACK_SHARED:
            packed = self.linear(x, "packed_shared")
            gate, up, score = packed[..., :512], packed[..., 512:1024], packed[..., 1024:1025]
        else:
            gate = self.linear(x, "mlp.shared_expert.gate_proj")
            up = self.linear(x, "mlp.shared_expert.up_proj")
            score = self.linear(x, "mlp.shared_expert_gate")
        y = ttnn.multiply(gate, up, input_tensor_a_activations=[] if self.ANCHOR_ACT else [ttnn.UnaryOpType.SILU])
        y = self.linear(y, "mlp.shared_expert.down_proj")
        return ttnn.multiply(y, score, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])

    def qkv(self, x, cos, sin, projection=None, *, prefill=False):
        tokens = x.shape[-2]
        if projection is not None and len(projection) == 5:
            q, k, v, gate, slot = projection
            rows = []
            for value, heads, column in ((q, 16, 0), (k, 2, 1), (v, 2, 0)):
                core = ttnn.CoreCoord(column, 0)
                memory = ttnn.create_sharded_memory_config(
                    (32, 256),
                    ttnn.CoreRangeSet({ttnn.CoreRange(core, core)}),
                    ttnn.ShardStrategy.HEIGHT,
                    use_height_and_width_as_shard_shape=True,
                )
                rows.append(ttnn.slice(value, (0, slot, 0, 0), (1, slot + 1, heads, 256), memory_config=memory))
            return *rows, gate
        if self.FUSE_HEAD_NORM or self.FUSE_NORM_ROPE:
            rope_args = {}
            if self.FUSE_NORM_ROPE:

                def expand_rope(value, identity):
                    if value.shape[-1] == 256:
                        return value
                    paired = ttnn.repeat_interleave(value[..., :32], 2, dim=-1)
                    return ttnn.pad(paired, ((0, 0), (0, 0), (0, 0), (0, 192)), value=identity)

                rope_args = dict(
                    transformation_mat=self.rope_transform,
                    rope_cos=expand_rope(cos, 1.0),
                    rope_sin=expand_rope(sin, 0.0),
                )
            projections = self.linear(x, "heads_attn", prefill=prefill)
            if tokens == 1 and self.FUSE_QKV_L1 and self.FUSE_PACKED_DRAM:

                def part(start, end, grid=None):
                    memory = (
                        ttnn.DRAM_MEMORY_CONFIG
                        if grid is None
                        else ttnn.create_sharded_memory_config(
                            (32, 256),
                            grid,
                            ttnn.ShardStrategy.WIDTH,
                            use_height_and_width_as_shard_shape=True,
                        )
                    )
                    return ttnn.slice(projections, (0, 0, 0, start), (1, 1, 1, end), memory_config=memory)

                q_part = part(0, 4096, ttnn.CoreGrid(x=8, y=2))
                k_part = part(4096, 4608, ttnn.CoreGrid(x=2, y=1))
                v_part, gate = part(4608, 5120), part(5120, 9216)
            else:
                q_part, k_part = projections[..., :4096], projections[..., 4096:4608]
                v_part, gate = projections[..., 4608:5120], projections[..., 5120:]
            q, k = [
                ttnn.experimental.dit_fused_distributed_rmsnorm(
                    y,
                    None,
                    self.device,
                    [],
                    epsilon=self.config.rms_norm_eps,
                    num_heads_per_device=heads,
                    per_head_norm=True,
                    weight=self.weights[name + "_head_norm"],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    **rope_args,
                    compute_kernel_config=self.compute,
                )
                for y, heads, name in (
                    (q_part, 16, "q"),
                    (k_part, 2, "k"),
                )
            ]
            if self.FUSE_NORM_ROPE and not self.KEEP_INTERLEAVED_QK:

                def original_order(y):
                    b, h, t, _ = y.shape
                    rot = ttnn.reshape(y[..., :64], (b, h, t, 32, 2))
                    rot = ttnn.reshape(ttnn.permute(rot, (0, 1, 2, 4, 3)), (b, h, t, 64))
                    return ttnn.concat([rot, y[..., 64:]], dim=-1)

                q, k = original_order(q), original_order(k)
            v = ttnn.permute(ttnn.reshape(v_part, (1, tokens, 2, 256)), (0, 2, 1, 3))
            if tokens == 1 and self.SHARDED_HEADS:
                q, k, v = [ttnn.permute(y, (0, 2, 1, 3)) for y in (q, k, v)]
        elif self.FUSE_HEADS:
            if projection is None:
                projections = self.linear(x, "heads_attn", prefill=prefill)
                if tokens == 1 and self.FUSE_QKV_L1 and self.FUSE_PACKED_DRAM:
                    gate = ttnn.slice(
                        projections,
                        (0, 0, 0, 5120),
                        (1, 1, 1, 9216),
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    qkv_mem = ttnn.create_sharded_memory_config(
                        (32, 256),
                        ttnn.CoreGrid(x=10, y=2),
                        ttnn.ShardStrategy.WIDTH,
                        use_height_and_width_as_shard_shape=True,
                    )
                    qkv = ttnn.slice(
                        projections,
                        (0, 0, 0, 0),
                        (1, 1, 1, 5120),
                        memory_config=qkv_mem,
                    )
                else:
                    gate = projections[..., 5120:]
                    qkv = projections[..., :5120]
            else:
                qkv, gate, batch_offset = projection
            if tokens == 1:
                mem = ttnn.create_sharded_memory_config(
                    (32, 256),
                    ttnn.CoreGrid(y=1, x=2 if self.FUSE_DISJOINT_HEADS else 1),
                    ttnn.ShardStrategy.HEIGHT,
                    use_height_and_width_as_shard_shape=True,
                )
                q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                    qkv,
                    num_heads=16,
                    num_kv_heads=2,
                    memory_config=mem,
                    overlap_qk_coregrid=not self.FUSE_DISJOINT_HEADS,
                    **({"batch_offset": batch_offset, "slice_size": 1} if projection is not None else {}),
                )
                if not self.SHARDED_HEADS:
                    q, k, v = [
                        ttnn.permute(ttnn.to_memory_config(a, ttnn.DRAM_MEMORY_CONFIG), (0, 2, 1, 3)) for a in (q, k, v)
                    ]
            else:
                q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                    ttnn.reshape(qkv, (1, tokens, 5120)),
                    num_heads=16,
                    num_kv_heads=2,
                    transpose_key=False,
                )
            q, k = self.norm(q, "self_attn.q_norm"), self.norm(k, "self_attn.k_norm")
        else:
            projections = self.linear(x, "packed_attn", prefill=prefill) if self.PACK_ATTENTION else None
            packed = ttnn.reshape(
                projections[..., :8192] if self.PACK_ATTENTION else self.linear(x, "self_attn.q_proj", prefill=prefill),
                (1, tokens, 16, 512),
            )
            q, gate = packed[..., :256], packed[..., 256:]
            q = ttnn.permute(self.norm(q, "self_attn.q_norm"), (0, 2, 1, 3))
            k = ttnn.reshape(
                (
                    projections[..., 8192:8704]
                    if self.PACK_ATTENTION
                    else self.linear(x, "self_attn.k_proj", prefill=prefill)
                ),
                (1, tokens, 2, 256),
            )
            k = ttnn.permute(self.norm(k, "self_attn.k_norm"), (0, 2, 1, 3))
            v = ttnn.permute(
                ttnn.reshape(
                    (
                        projections[..., 8704:9216]
                        if self.PACK_ATTENTION
                        else self.linear(x, "self_attn.v_proj", prefill=prefill)
                    ),
                    (1, tokens, 2, 256),
                ),
                (0, 2, 1, 3),
            )

        if self.FUSE_LLAMA_ROPE:
            tables = []
            for value, identity in ((cos, 1.0), (sin, 0.0)):
                if value.shape[-1] == 64:
                    value = ttnn.pad(
                        ttnn.repeat_interleave(value[..., :32], 2, dim=-1),
                        ((0, 0), (0, 0), (0, 0), (0, 192)),
                        value=identity,
                    )
                if tokens == 1:
                    if value.shape[1] != 2:
                        value = ttnn.repeat(value, (1, 2, 32 // value.shape[-2], 1))
                    value = ttnn.to_memory_config(
                        value,
                        ttnn.create_sharded_memory_config(
                            (32, 256),
                            ttnn.CoreGrid(x=2, y=1),
                            ttnn.ShardStrategy.HEIGHT,
                            use_height_and_width_as_shard_shape=True,
                        ),
                    )
                tables.append(value)
            if tokens == 1:

                def head_memory(column):
                    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(column, 0), ttnn.CoreCoord(column, 0))})
                    return ttnn.create_sharded_memory_config(
                        (32, 256),
                        grid,
                        ttnn.ShardStrategy.HEIGHT,
                        use_height_and_width_as_shard_shape=True,
                    )

                q, k = ttnn.experimental.rotary_embedding_llama_fused_qk(
                    ttnn.to_memory_config(q, head_memory(0)),
                    ttnn.to_memory_config(k, head_memory(1)),
                    *tables,
                    self.rope_transform_decode,
                    compute_kernel_config=self.rope_compute,
                )
            else:
                q, k = [
                    ttnn.experimental.rotary_embedding_llama(
                        value,
                        *tables,
                        self.rope_transform,
                        is_decode_mode=False,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        compute_kernel_config=self.rope_compute,
                    )
                    for value in (q, k)
                ]
            return q, k, v, ttnn.reshape(gate, (1, 1, tokens, 4096))

        def rope(y):
            rot = y[..., :64]
            rotated = ttnn.experimental.rotary_embedding(
                rot,
                cos,
                sin,
                token_index=0 if tokens == 1 else None,
                compute_kernel_config=self.compute,
            )
            rows = y.shape[-2]
            return ttnn.concat([rotated[..., :rows, :], y[..., 64:]], dim=-1)

        return (
            (q if self.FUSE_NORM_ROPE else rope(q)),
            (k if self.FUSE_NORM_ROPE else rope(k)),
            v,
            ttnn.reshape(gate, (1, 1, tokens, 4096)),
        )

    def _decode_sdpa_k_chunk(self, key, page_table):
        # Causal SDPA reads a whole K chunk before applying its mask. Bound
        # that read by each request's virtual extent, not the shared pool.
        extent = page_table.shape[-1] * key.shape[-2]
        requested = self.DECODE_SDPA_K_CHUNK
        if extent <= 0 or extent % 32:
            raise ValueError("SDPA virtual cache extent must be a positive multiple of 32")
        if requested != 0 and (requested < 32 or requested & (requested - 1)):
            raise ValueError("SDPA K chunk must be zero or a power of two >=32")
        divisor = extent & -extent
        if requested == 0:
            # These are the kernel's dynamic caps. Keep dynamic dispatch only
            # when every possible chunk divides this request's virtual extent.
            cap = 128 if self.SDPA_FP32_DEST else 256
            maximum = min(cap, 1 << (extent - 1).bit_length())
            return 0 if extent % maximum == 0 else min(cap, divisor)
        return min(requested, divisor)

    def full_attention(
        self,
        x,
        state,
        page_table,
        current_pos,
        cos,
        sin,
        residual=None,
        projection=None,
        return_projection_input=False,
        *,
        prefill=False,
    ):
        q, k, v, gate = self.qkv(x, cos, sin, projection=projection, prefill=prefill)

        def cache_mem(column):
            grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(column, 0), ttnn.CoreCoord(column, 0))})
            return ttnn.create_sharded_memory_config(
                (32, 256), grid, ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
            )

        mem, vmem = cache_mem(1 if self.FUSE_CACHE else 0), cache_mem(0)
        if not (self.FUSE_HEADS and self.SHARDED_HEADS):
            k = ttnn.to_memory_config(ttnn.permute(k, (0, 2, 1, 3)), mem)
            v = ttnn.to_memory_config(ttnn.permute(v, (0, 2, 1, 3)), vmem)
        if self.FUSE_HEADS and self.SHARDED_HEADS:
            k = ttnn.to_memory_config(k, mem)
            v = ttnn.to_memory_config(v, vmem)
        if self.FUSE_CACHE:
            ttnn.experimental.paged_fused_update_cache(
                state.key, k, state.value, v, update_idxs_tensor=current_pos, page_table=page_table
            )
        else:
            ttnn.experimental.paged_update_cache(state.key, k, update_idxs_tensor=current_pos, page_table=page_table)
            ttnn.experimental.paged_update_cache(state.value, v, update_idxs_tensor=current_pos, page_table=page_table)
        if not (self.FUSE_HEADS and self.SHARDED_HEADS):
            q = ttnn.permute(q, (0, 2, 1, 3))
        y = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            state.key,
            state.value,
            page_table,
            cur_pos_tensor=current_pos,
            compute_kernel_config=self.sdpa_compute,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                q_chunk_size=32,
                k_chunk_size=self._decode_sdpa_k_chunk(state.key, page_table),
                exp_approx_mode=False,
            ),
        )
        if self.CONCAT_DECODE:
            y = ttnn.experimental.nlp_concat_heads_decode(ttnn.to_memory_config(y, mem), num_heads=16)
            y = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)[:, :, :1, :]
        y = ttnn.multiply(
            ttnn.reshape(y, (1, 1, 1, 4096)),
            gate,
            input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
            memory_config=(
                ttnn.create_sharded_memory_config(
                    (32, 512),
                    ttnn.CoreGrid(x=8, y=1),
                    ttnn.ShardStrategy.WIDTH,
                    use_height_and_width_as_shard_shape=True,
                )
                if self.FUSE_OUTPUT_DRAM
                else ttnn.DRAM_MEMORY_CONFIG
            ),
        )
        return y if return_projection_input else self.linear(y, "self_attn.o_proj", bias=residual, prefill=prefill)

    def prefill_chunk_forward(self, x, *, state, page_table, chunk_page_table, chunk_start, cos, sin):
        q, k, v, gate = self.qkv(self.norm(x, "input_layernorm"), cos, sin, prefill=True)
        ttnn.experimental.paged_fill_cache(
            state.key, ttnn.typecast(k, state.key.dtype) if k.dtype != state.key.dtype else k, chunk_page_table
        )
        ttnn.experimental.paged_fill_cache(
            state.value, ttnn.typecast(v, state.value.dtype) if v.dtype != state.value.dtype else v, chunk_page_table
        )
        y = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            state.key,
            state.value,
            page_table,
            chunk_start_idx_tensor=chunk_start,
            compute_kernel_config=self.sdpa_compute,
            program_config=ttnn.SDPAProgramConfig(
                # The running SDPA output is BF16 between K blocks. 32-key
                # blocks accumulate excessive rounding at 262K context.
                compute_with_storage_grid_size=self.PREFILL_SDPA_GRID,
                q_chunk_size=min(self.PREFILL_SDPA_Q_CHUNK, x.shape[-2] & -x.shape[-2]),
                k_chunk_size=self.PREFILL_SDPA_K_CHUNK,
                exp_approx_mode=False,
            ),
        )
        y = ttnn.reshape(ttnn.transformer.concatenate_heads(y), (1, 1, x.shape[-2], 4096))
        h = ttnn.add(
            x,
            self.linear(
                ttnn.multiply(y, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]),
                "self_attn.o_proj",
                prefill=True,
            ),
        )
        return ttnn.add(h, self.moe(self.norm(h, "post_attention_layernorm")))

    def prepare_rotary(self, cos_host, sin_host, *, decode=False):
        """Host setup: prepare stable tables; decode_forward also accepts width64.

        Full-width interleaved RoPE avoids partial-channel adapters. Keep each
        request's prepared tables in DRAM so32 live requests do not exhaust L1.
        The fused decode rotary kernel requires transient HEIGHT-L1 copies.
        """
        import torch

        result = {}
        for name, value, identity in (("cos", cos_host, 1), ("sin", sin_host, 0)):
            memory = ttnn.DRAM_MEMORY_CONFIG
            if self.FUSE_LLAMA_ROPE:
                if value.shape[-1] != 64:
                    raise ValueError("prepare_rotary expects canonical64-wide host tables")
                value = torch.cat(
                    [
                        value[..., :32].repeat_interleave(2, dim=-1),
                        torch.full((*value.shape[:-1], 192), identity, dtype=value.dtype),
                    ],
                    dim=-1,
                )
                if decode:
                    if tuple(value.shape[:-1]) != (1, 1, 1):
                        raise ValueError("Decode rotary setup requires one position")
                    value = value.repeat(1, 2, 32, 1)
                    if self.ROPE_PREP_L1:
                        memory = ttnn.create_sharded_memory_config(
                            (32, 256),
                            ttnn.CoreGrid(x=2, y=1),
                            ttnn.ShardStrategy.HEIGHT,
                            use_height_and_width_as_shard_shape=True,
                        )
            result[name] = ttnn.from_torch(
                value.contiguous(),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory,
            )
        return result

    def prepare_prefill(self, *, seq_len, start_pos=0, page_table_host=None, chunk_size=None):
        """Host setup only. Bind all device positions, RoPE and chunk page maps.

        An unaligned prefix is advanced by single-token updates until the next
        page boundary, preserving all earlier cache rows. Tail padding is causal
        and occupies future rows in this request's own pages only.
        """
        import torch

        if seq_len < 1 or start_pos < 0 or start_pos + seq_len > self.config.max_position_embeddings:
            raise ValueError("Prefill extent exceeds advertised context")
        if chunk_size is None:
            chunk_size = self.PREFILL_CHUNK_SIZES[self.kind]
        if chunk_size <= 0 or chunk_size % 32:
            raise ValueError("Physical chunk size must be a positive multiple of 32")

        def upload(t, dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t.contiguous(), device=self.device, dtype=dtype, layout=layout)

        chunks = []
        offset = 0
        if self.kind == "full_attention":
            if page_table_host is None or page_table_host.ndim != 2 or page_table_host.shape[0] != 1:
                raise ValueError("One page-table row required per request")
            if page_table_host.shape[1] % 8:
                raise ValueError("Page-table extent must cover a multiple of 256 tokens")
            if page_table_host.shape[1] * 32 < start_pos + seq_len:
                raise ValueError("Page table does not cover request")
            table = upload(page_table_host, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        while offset < seq_len:
            pos = start_pos + offset
            decode = self.kind == "full_attention" and pos % 32 != 0
            count = 1 if decode else min(chunk_size, seq_len - offset)
            physical = (count + 31) // 32 * 32 if self.kind == "full_attention" and not decode else count
            kw = {}
            if self.kind == "full_attention":
                # Match HF's FP32 evaluation order at large positions.
                inv = 1.0 / (10000000.0 ** (torch.arange(0, 64, 2).float() / 64))
                phase = torch.arange(pos, pos + physical).float()[:, None] * inv[None]
                phase = torch.cat([phase, phase], -1)[None, None]
                kw = dict(
                    page_table=table,
                    **self.prepare_rotary(phase.cos().bfloat16(), phase.sin().bfloat16(), decode=decode),
                )
                position = upload(torch.tensor([pos], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                if decode:
                    kw["current_pos"] = position
                else:
                    kw["chunk_start"] = position
                    kw["chunk_page_table"] = upload(
                        page_table_host[:, pos // 32 : (pos + physical) // 32], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
                    )
            chunks.append((offset, count, physical, decode, kw))
            offset += count
        if self.FUSE_MOE:
            for _, count, physical, _, _ in chunks:
                if self.FUSE_ROUTE_INDEX_COMBINE and physical > 1:
                    capacity = (8 * physical + 31 * min(256, 8 * physical) + 31) // 32 * 32
                    if capacity not in self.route_index_payloads:
                        rows = torch.arange(capacity, dtype=torch.int64)
                        encoded = torch.zeros(1, 1, capacity, 32)
                        for byte in range(3):
                            encoded[0, 0, :, byte] = ((rows >> (8 * byte)) & 255).float()
                        self.route_index_payloads[capacity] = upload(encoded.bfloat16(), ttnn.bfloat16)
                extent = (physical + 31) // 32 * 32
                if extent not in self.expert_meta:
                    self.expert_meta[extent] = (
                        upload(torch.full((256,), extent, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                        upload(torch.arange(256, dtype=torch.int32) * extent, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                    )
                if self.FUSE_COMBINE and extent not in self.combine_rows:
                    self.combine_rows[extent] = upload(
                        torch.arange(extent).float().reshape(1, 1, extent, 1), ttnn.float32
                    )
        if self.FUSE_CONV_CARRY and self.kind == "linear_attention":
            for _, count, _, _, _ in chunks:
                if count not in self.conv_ends:
                    self.conv_ends[count] = upload(
                        torch.tensor([count], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
                    )
        return dict(seq_len=seq_len, start_pos=start_pos, chunks=chunks)

    def prefill_forward(self, x, *, state, plan):
        """Logical [1,1,S,2048] -> same shape; explicit fresh/continued state.

        Several requests may be supplied as parallel lists of tensors, states and
        plans. Each request can have a different length, start and page ownership.
        """
        if isinstance(x, (list, tuple)):
            return [self.prefill_forward(a, state=b, plan=c) for a, b, c in zip(x, state, plan, strict=True)]
        if tuple(x.shape) != (1, 1, plan["seq_len"], 2048):
            raise ValueError("Input shape must match the prepared logical length")
        outputs = []
        for offset, count, physical, decode, kw in plan["chunks"]:
            row = x[:, :, offset : offset + count, :]
            if physical != count:
                row = ttnn.pad(row, ((0, 0), (0, 0), (0, physical - count), (0, 0)), value=0.0)
            if self.kind == "linear_attention":
                precise = self.PRECISE_LINEAR_PREFILL
                h = ttnn.add(
                    row,
                    self.linear_attention(
                        self.norm(row, "input_layernorm", precise=precise), state, precise=precise, prefill=True
                    ),
                )
                y = ttnn.add(h, self.moe(self.norm(h, "post_attention_layernorm", precise=precise)))
            else:
                # Non-aligned prefill may dispatch a decode kernel; retain prefill precision.
                y = (
                    self.decode_forward(row, state=state, prefill=True, **kw)
                    if decode
                    else self.prefill_chunk_forward(row, state=state, **kw)
                )
            outputs.append(y[:, :, :count, :])
        if len(outputs) == 1:
            return outputs[0]
        # A final one-token chunk can retain the decode projection's WIDTH-L1
        # layout. Concat requires the same placement as the earlier chunks.
        outputs = [ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG) if y.is_sharded() else y for y in outputs]
        return ttnn.concat(outputs, dim=2)

    def _batch_linear_conv(self, packed, states):
        """Independent users occupy independent channels in one depthwise conv."""
        batch = len(states)
        group = self.BATCH_CONV_GROUP_SIZE
        if self.BATCH_CONV_GROUP_ADAPTIVE and batch > 1:
            group = min(group, 1 << (batch.bit_length() - 1))
        if batch in self.BATCH_CONV_GROUP_BATCHES and 0 < group < batch:
            outputs = []
            start = 0
            while start < batch:
                remaining = batch - start
                count = min(group, remaining)
                if self.BATCH_CONV_GROUP_ADAPTIVE and remaining in self.BATCH_CONV_GROUP_BATCHES:
                    count = min(count, 1 << (remaining.bit_length() - 1))
                end = start + count
                # Exact logical bounds avoid padding a short final group to a
                # larger user bucket. Each history tensor keeps its allocation.
                outputs.append(self._batch_linear_conv(packed[..., start:end, :], states[start:end]))
                start = end
            return tuple(ttnn.concat([output[i] for output in outputs], dim=0) for i in range(3))
        bucket = min(n for n in self.BATCH_CONV_BUCKETS if n >= batch)
        width = bucket * 8192
        taps, exchange = self.batch_conv[bucket]
        inp = ttnn.reshape(ttnn.to_layout(packed[..., :8192], ttnn.ROW_MAJOR_LAYOUT), (1, 1, batch * 8192))
        if bucket != batch:
            inp = ttnn.pad(inp, ((0, 0), (0, 0), (0, width - batch * 8192)), value=0.0)
        current = inp
        inp = (
            ttnn.reshape(
                ttnn.tilize_with_val_padding(inp, (1, 32, width), 0.0, use_multicore=True),
                ttnn.Shape((1, 32, width)),
                ttnn.Shape((1, 32, width)),
            )
            if self.FUSE_BATCH_CONV_PAD
            else ttnn.to_layout(ttnn.pad(inp, ((0, 0), (0, 31), (0, 0)), value=0.0), ttnn.TILE_LAYOUT)
        )
        history = ttnn.reshape(ttnn.concat([state.conv for state in states], dim=3), (1, 3, batch * 8192))
        if bucket != batch:
            history = ttnn.pad(history, ((0, 0), (0, 0), (0, width - batch * 8192)), value=0.0)
        op = (
            ttnn.experimental.kda.qkv_causal_conv1d_silu_with_carry
            if self.FUSE_BATCH_CONV_CARRY
            else ttnn.experimental.kda.qkv_causal_conv1d_silu
        )
        extra = (
            dict(actual_end=self.conv_ends[1], exchange_buffer=exchange, exchange_semaphores=self.conv_semaphores)
            if self.FUSE_BATCH_CONV_CARRY
            else dict(predecessor_carry=history)
        )
        conv = op(
            inp,
            history,
            *taps,
            bucket * 2048,
            bucket * 2048,
            bucket * 4096,
            actual_start=self.conv_start,
            **extra,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=self.CONV_CHANNEL),
            compute_kernel_config=self.compute,
        )
        carry = ttnn.reshape(
            conv[3] if self.FUSE_BATCH_CONV_CARRY else ttnn.concat([history[:, 1:, :], current], dim=1),
            (1, 1, 3, width),
        )
        if self.FUSE_BATCH_HISTORY_COPY:
            carry = ttnn.reshape(ttnn.permute(ttnn.reshape(carry, (3, bucket, 8192)), (1, 0, 2)), (bucket, 1, 3, 8192))
        for i, state in enumerate(states):
            if self.FUSE_BATCH_HISTORY_COPY:
                ttnn.experimental.kda.indexed_entry_copy(
                    carry, state.conv, self.conv_start, index_scale=1, index_offset=i
                )
            else:
                ttnn.slice(carry, (0, 0, 0, i * 8192), (1, 1, 3, (i + 1) * 8192), output_tensor=state.conv)
        # Output partitions are arbitrary channel ranges. Restore actual user
        # Q/K/V boundaries only after discarding physical time padding.
        output = ttnn.to_layout(ttnn.concat(conv[:3], dim=2)[:, :1, :], ttnn.ROW_MAJOR_LAYOUT)
        output = ttnn.reshape(output, (bucket, 1, 8192))[:batch]
        return tuple(
            ttnn.to_layout(
                ttnn.reshape(output[..., start:end], (batch, 1, heads, 128)), ttnn.TILE_LAYOUT, dtype=ttnn.float32
            )
            for start, end, heads in ((0, 2048, 16), (2048, 4096, 16), (4096, 8192, 32))
        )

    def _batch_linear_attention(self, inputs, states, projections):
        """Peer-merge the existing FP32 recurrence; persistent states stay stable."""
        batch = len(inputs)
        packed = projections if isinstance(projections, ttnn.Tensor) else ttnn.concat(projections, dim=2)
        if self.FUSE_BATCH_CONV:
            q, k, value = self._batch_linear_conv(packed, states)
        else:
            heads = [
                self.linear_attention(x, state, projection=projection, return_recurrence_inputs=True)
                for x, state, projection in zip(inputs, states, projections, strict=True)
            ]
            q, k, value = [ttnn.concat([row[i] for row in heads], dim=0) for i in range(3)]
        q, k = [
            ttnn.rms_norm(
                value,
                epsilon=1e-6 / 128,
                weight=self.weights[name],
                compute_kernel_config=self.compute,
            )
            for value, name in ((q, "q_l2.weight"), (k, "l2.weight"))
        ]
        q = ttnn.repeat_interleave(ttnn.permute(q, (0, 2, 1, 3)), 2, dim=1)
        k_row = ttnn.repeat_interleave(ttnn.permute(k, (0, 2, 1, 3)), 2, dim=1)
        k = ttnn.transpose(k_row, -1, -2)
        value = ttnn.permute(value, (0, 2, 1, 3))
        a = ttnn.reshape(packed[..., 12288:12320], (1, batch, 32))
        b = ttnn.reshape(packed[..., 12320:12352], (1, batch, 32))
        beta = ttnn.sigmoid(b)
        if not self.FUSE_SCALAR_CASTS:
            beta = ttnn.typecast(beta, ttnn.float32)
        biased = (
            ttnn.mac(self.dt, 1.0, a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if self.FUSE_SCALAR_CASTS
            else ttnn.add(ttnn.typecast(a, ttnn.float32), self.dt)
        )
        g = ttnn.multiply(
            self.decay,
            biased,
            input_tensor_b_activations=self.softplus_activation,
        )

        def scalars(value):
            return ttnn.to_layout(
                ttnn.reshape(ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT), (batch, 32, 1, 1)),
                ttnn.TILE_LAYOUT,
            )

        recurrence_memory = ttnn.L1_MEMORY_CONFIG if self.BATCH_RECURRENCE_L1 else ttnn.DRAM_MEMORY_CONFIG
        recurrence_config = None
        head_count = batch * 32
        state_grid = None
        if self.BATCH_RECURRENCE_SHARD:
            gx, gy = self.BATCH_RECURRENCE_SHARD
            if head_count % (gx * gy) == 0:
                heads_per_core = head_count // (gx * gy)
                state_grid = ttnn.CoreGrid(x=gx, y=gy)
        if state_grid is None and self.BATCH_RECURRENCE_FALLBACK_GRID:
            gx, gy = self.BATCH_RECURRENCE_FALLBACK_GRID
            # Exact subsets keep whole heads without padding logical requests.
            # The reuse factory reads the input shard's actual CoreRangeSet.
            heads_per_core = next(
                h for h in range(1, head_count + 1) if head_count % h == 0 and head_count // h <= gx * gy
            )
            cores = head_count // heads_per_core
            state_grid = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(i % gx, i // gx), ttnn.CoreCoord(i % gx, i // gx)) for i in range(cores)]
            )
        if state_grid is not None:
            recurrence_memory = ttnn.create_sharded_memory_config(
                (heads_per_core * 128, 128),
                state_grid,
                ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            # This block must equal the physical whole-head shard. A smaller
            # independent M block underallocates the borrowed B circular buffer.
            recurrence_config = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=heads_per_core,
                per_core_N=4,
            )

        def recurrent_matmul(a, b):
            return ttnn.matmul(
                a,
                b,
                compute_kernel_config=self.compute,
                dtype=ttnn.float32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                **({"program_config": recurrence_config} if recurrence_config and b.is_sharded() else {}),
            )

        state = ttnn.concat([s.recurrent for s in states], dim=0, memory_config=recurrence_memory)
        if self.BATCH_RECURRENCE_INPUT_L1:
            q, k_row, k, value = [ttnn.to_memory_config(a, ttnn.L1_MEMORY_CONFIG) for a in (q, k_row, k, value)]
        if self.BATCH_RECURRENCE_L1_INPUTS:
            q, k_row, k, value = [
                ttnn.to_memory_config(a, ttnn.L1_MEMORY_CONFIG) if n in self.BATCH_RECURRENCE_L1_INPUTS else a
                for n, a in (("q", q), ("k_row", k_row), ("k", k), ("value", value))
            ]
        decayed = ttnn.multiply(
            state,
            scalars(g),
            input_tensor_b_activations=[ttnn.UnaryOpType.EXP],
            memory_config=recurrence_memory,
            **({"output_tensor": state} if self.BATCH_RECURRENCE_INPLACE else {}),
        )
        prediction = recurrent_matmul(k_row, decayed)
        delta = ttnn.multiply(
            ttnn.subtract(value, prediction),
            scalars(beta),
            **({"dtype": ttnn.float32, "fast_and_approximate_mode": False} if self.FUSE_SCALAR_CASTS else {}),
        )
        updated = ttnn.addcmul(
            decayed,
            k,
            delta,
            value=1.0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if self.BATCH_RECURRENCE_DRAM_UPDATE else recurrence_memory,
            **(
                {"output_tensor": decayed}
                if self.BATCH_RECURRENCE_INPLACE and not self.BATCH_RECURRENCE_DRAM_UPDATE
                else {}
            ),
        )
        if self.BATCH_RECURRENCE_DRAM_UPDATE:
            ttnn.deallocate(decayed)
            if not self.BATCH_RECURRENCE_INPLACE:
                ttnn.deallocate(state)
        copy_source = (
            ttnn.to_memory_config(updated, ttnn.DRAM_MEMORY_CONFIG)
            if (self.BATCH_RECURRENCE_L1 or recurrence_config) and not self.BATCH_DIRECT_SLICE_FROM_L1
            else updated
        )
        for i, state in enumerate(states):
            if self.FUSE_BATCH_STATE_COPY and not self.BATCH_DIRECT_SLICE_FROM_L1:
                ttnn.experimental.kda.indexed_entry_copy(
                    copy_source, state.recurrent, self.conv_start, index_scale=1, index_offset=i
                )
            else:
                ttnn.slice(copy_source, (i, 0, 0, 0), (i + 1, 32, 128, 128), output_tensor=state.recurrent)
        y = recurrent_matmul(q, updated)
        if self.FUSE_BATCH_GATED_NORM and batch >= self.BATCH_GATED_NORM_MIN:
            head_major_y = ttnn.reshape(ttnn.permute(y, (2, 1, 0, 3)), (32, batch, 128))
            return self._gated_head_norm(head_major_y, packed, batch)
        y = ttnn.reshape(ttnn.permute(ttnn.typecast(y, ttnn.bfloat16), (0, 2, 1, 3)), (1, 1, batch * 32, 128))
        z = ttnn.to_layout(
            ttnn.reshape(ttnn.to_layout(packed[..., 8192:12288], ttnn.ROW_MAJOR_LAYOUT), (1, 1, batch * 32, 128)),
            ttnn.TILE_LAYOUT,
        )
        y = ttnn.multiply(self.norm(y, "linear_attn.norm"), z, input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
        return ttnn.reshape(y, (1, 1, batch, 4096))

    def _batch_heads(self, packed, operands):
        """Merge head split/norm/rotary; keep cache consumers request-local."""
        batch = packed.shape[-2]

        def grid(indices):
            # Lists preserve user order. The set overload sorts ranges x-first,
            # which disagrees with native Q/V splitting and rotary table pairing.
            return ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(i % 8, i // 8), ttnn.CoreCoord(i % 8, i // 8)) for i in indices]
            )

        def height(indices):
            return ttnn.create_sharded_memory_config(
                (32, 256),
                grid(indices),
                ttnn.ShardStrategy.HEIGHT,
                use_height_and_width_as_shard_shape=True,
            )

        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            packed,
            num_heads=16,
            num_kv_heads=2,
            memory_config=height(range(2 * batch)),
            overlap_qk_coregrid=False,
        )
        rows = max(r for r in (1, 2, 4) if batch % r == 0)
        normalized = []
        for value, name, start in ((q, "q", 0), (k, "k", rows)):
            cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, start), ttnn.CoreCoord(7, start + rows - 1))})
            memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED if rows > 1 else ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(cores, [32 * (batch // rows), 32], ttnn.ShardOrientation.ROW_MAJOR),
            )
            normalized.append(
                ttnn.rms_norm(
                    ttnn.to_memory_config(value, memory),
                    epsilon=self.config.rms_norm_eps,
                    weight=self.weights["self_attn." + name + "_norm.weight"],
                    compute_kernel_config=self.compute,
                    memory_config=memory,
                    program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
                        compute_with_storage_grid_size=[8, rows],
                        subblock_w=1,
                        block_h=batch // rows,
                        block_w=1,
                        inplace=False,
                    ),
                )
            )
        tables = []
        for name, identity in (("cos", 1.0), ("sin", 0.0)):
            parts = []
            for kw in operands:
                value = kw[name]
                if value.shape[-1] == 64:
                    value = ttnn.pad(
                        ttnn.repeat_interleave(value[..., :32], 2, dim=-1),
                        ((0, 0), (0, 0), (0, 0), (0, 192)),
                        value=identity,
                    )
                if value.shape[1] != 2:
                    value = ttnn.repeat(value, (1, 2, 32 // value.shape[-2], 1))
                parts.append(ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG))
            tables.append(ttnn.to_memory_config(ttnn.concat(parts, dim=1), height(range(2 * batch))))
        q, k = ttnn.experimental.rotary_embedding_llama_fused_qk(
            ttnn.to_memory_config(normalized[0], height(range(0, 2 * batch, 2))),
            ttnn.to_memory_config(normalized[1], height(range(1, 2 * batch, 2))),
            *tables,
            self.batch_rope_transform,
            compute_kernel_config=self.rope_compute,
        )
        return q, k, v

    def _batch_projections(self, x, operands, *, retain_packed=False, retain_shared_heads=False):
        """Peer-merge one physical M tile, retaining independent request rows."""
        batch = x.shape[-2]
        h = self.norm(x, "input_layernorm", decode_rows=True)
        packed = self.linear(h, "heads_attn" if self.kind == "full_attention" else "packed_attn", decode_rows=True)
        if self.kind == "linear_attention":
            if retain_packed:
                return packed
            rows = ttnn.to_layout(packed, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return [ttnn.to_layout(rows[:, :, i : i + 1, :], ttnn.TILE_LAYOUT) for i in range(batch)]
        qkv_memory = ttnn.create_sharded_memory_config(
            (32, 256),
            ttnn.CoreGrid(x=10, y=2),
            ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        qkv = ttnn.slice(packed, (0, 0, 0, 0), (1, 1, batch, 5120), memory_config=qkv_memory)
        gates = ttnn.slice(packed, (0, 0, 0, 5120), (1, 1, batch, 9216), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if retain_shared_heads:
            return (*self._batch_heads(qkv, operands), gates)
        gates = ttnn.to_layout(gates, ttnn.ROW_MAJOR_LAYOUT)
        if self.FUSE_BATCH_HEADS:
            q, k, v = self._batch_heads(qkv, operands)
            return [(q, k, v, ttnn.to_layout(gates[:, :, i : i + 1, :], ttnn.TILE_LAYOUT), i) for i in range(batch)]
        return [
            (qkv, ttnn.to_layout(gates[:, :, i : i + 1, :], ttnn.TILE_LAYOUT), self.batch_row_ids[i])
            for i in range(batch)
        ]

    def _shared_cache_pool_signature(self, key, value):
        """Check pool metadata without synchronizing or reading device contents."""
        for tensor in (key, value):
            if (
                not isinstance(tensor, ttnn.Tensor)
                or not ttnn.is_tensor_storage_on_device(tensor)
                or not tensor.is_allocated()
            ):
                raise ValueError("Shared-cache pools must be allocated device tensors")
            shape = tuple(tensor.shape)
            if (
                tensor.device() != self.device
                or len(shape) != 4
                or shape[0] < 1
                or shape[1:] != (2, 32, 256)
                or tuple(tensor.padded_shape) != shape
                or tensor.dtype != getattr(ttnn, self.KV_CACHE_DTYPE)
                or tensor.layout != ttnn.TILE_LAYOUT
                or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            ):
                raise ValueError(
                    "Shared-cache pools require configured KV dtype, TILE layout and DRAM placement [blocks,2,32,256] on this decoder's device"
                )
        if tuple(key.shape) != tuple(value.shape):
            raise ValueError("Shared-cache key and value pool geometry must match")
        if key.buffer_address() == value.buffer_address():
            raise ValueError("Shared-cache key and value pools must use distinct buffers")
        return tuple(key.shape), key.buffer_address(), value.buffer_address()

    def prepare_shared_cache_batch(self, *, states, page_tables_host):
        """Bind exclusive pages for an optional native batch cache/SDPA graph.

        This setup-only API clones the mappings. With this plan, decode operands
        omit page_table; current positions and rotary tables remain mutable.
        The plan belongs to this decoder. Do not write or deallocate its page
        tensors; retain the plan and its pools until every using trace ends.
        Existing calls without a plan retain sequential cache-update semantics.
        """
        import torch

        if self.kind != "full_attention" or not 1 <= len(states) <= 32:
            raise ValueError("Shared-cache plans require1..32 full-attention requests")
        key, value = states[0].key, states[0].value
        signature = self._shared_cache_pool_signature(key, value)
        if any(self._shared_cache_pool_signature(s.key, s.value) != signature for s in states):
            raise ValueError("Every request must use the same key and value pools")
        if page_tables_host.dtype != torch.int32 or page_tables_host.ndim != 2:
            raise ValueError("Page tables must be a two-dimensional host INT32 tensor")
        pages = page_tables_host.detach().cpu().clone().contiguous()
        if pages.shape[0] != len(states) or pages.shape[1] < 1:
            raise ValueError("One nonempty page-table row is required per request")
        if int(pages.min()) < 0 or int(pages.max()) >= key.shape[0]:
            raise ValueError("Page index is outside the shared pool")
        if torch.unique(pages).numel() != pages.numel():
            raise ValueError("Shared-cache batch plans require disjoint pages, including writable tiles")
        table = ttnn.from_torch(pages, device=self.device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        rows = tuple(table[i : i + 1, :] for i in range(len(states)))
        return SharedCacheBatchPlan(key, value, table, rows, self._shared_cache_owner)

    def _shared_cache_attention(self, heads, operands, plan):
        q, k, v, gates = heads
        batch = q.shape[1]
        v = ttnn.to_memory_config(v, q.memory_config())
        positions = ttnn.concat([kw["current_pos"] for kw in operands], dim=0)
        ttnn.experimental.paged_fused_update_cache(
            plan.key, k, plan.value, v, update_idxs_tensor=positions, page_table=plan.page_table
        )

        def grid(indices):
            return ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(i % 8, i // 8), ttnn.CoreCoord(i % 8, i // 8)) for i in indices]
            )

        query_cores = list(range(0, 2 * batch, 2))
        reduction_order = query_cores + [i for i in range(64) if i not in query_cores]
        y = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            plan.key,
            plan.value,
            plan.page_table,
            cur_pos_tensor=positions,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.sdpa_compute,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                q_chunk_size=32,
                k_chunk_size=self._decode_sdpa_k_chunk(plan.key, plan.page_table),
                exp_approx_mode=False,
                sub_core_grids=grid(reduction_order),
            ),
        )
        memory = ttnn.create_sharded_memory_config(
            (32, 256),
            grid(range(batch)),
            ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        y = ttnn.experimental.nlp_concat_heads_decode(
            ttnn.to_memory_config(y, memory), num_heads=16, sub_core_grids=grid(range(64))
        )
        y = ttnn.reshape(y, ttnn.Shape((1, 1, batch, 4096)), ttnn.Shape((1, 1, 32, 4096)))
        output_memory = ttnn.create_sharded_memory_config(
            (32, 512),
            ttnn.CoreGrid(x=8, y=1),
            ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        return ttnn.multiply(
            y, gates, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID], memory_config=output_memory
        )

    def decode_batch(self, inputs, *, states, operands, shared_cache_plan=None):
        """Independent request state with optional shared-weight FFN peer merge."""
        if not 1 <= len(inputs) <= 32:
            raise ValueError("Batch must be between 1 and 32")
        if len(states) != len(inputs) or len(operands) != len(inputs):
            raise ValueError("One state and operand mapping are required per request")
        if shared_cache_plan is not None:
            plan = shared_cache_plan
            if self.kind != "full_attention":
                raise ValueError("Shared-cache plans require a full-attention decoder")
            if not isinstance(plan, SharedCacheBatchPlan) or plan.owner_token is not self._shared_cache_owner:
                raise ValueError("Shared-cache plan belongs to a different decoder")
            signature = self._shared_cache_pool_signature(plan.key, plan.value)
            if len(plan.row_page_tables) != len(inputs) or any(
                self._shared_cache_pool_signature(s.key, s.value) != signature for s in states
            ):
                raise ValueError("Shared-cache plan does not match this batch and pool")
            if any("page_table" in kw for kw in operands):
                raise ValueError("A shared-cache plan owns page bindings; omit per-request page_table operands")
            operands = [dict(kw, page_table=page) for kw, page in zip(operands, plan.row_page_tables, strict=True)]
        if not (
            self.FUSE_BATCH_MOE
            and self.FUSE_MOE
            and (self.FUSE_ROUTE_PREFILL or self.FUSE_BATCH_DIRECT)
            and len(inputs) >= self.BATCH_MIN_SIZE
        ):
            return [self.decode_forward(x, state=s, **kw) for x, s, kw in zip(inputs, states, operands, strict=True)]
        batch_input = ttnn.concat(inputs, dim=2) if self.FUSE_BATCH_PROJECTION or self.FUSE_BATCH_OUTPUT else None
        merge_shared = (
            shared_cache_plan is not None
            and self.kind == "full_attention"
            and self.FUSE_BATCH_SHARED_CACHE
            and self.FUSE_BATCH_PROJECTION
            and self.FUSE_BATCH_OUTPUT
            and self.FUSE_BATCH_HEADS
            and self.FUSE_LLAMA_ROPE
            and self.FUSE_HEADS
            and self.SHARDED_HEADS
        )
        merge_recurrence = (
            self.kind == "linear_attention"
            and self.FUSE_BATCH_RECURRENCE
            and self.FUSE_BATCH_PROJECTION
            and self.FUSE_BATCH_OUTPUT
            and all(
                (
                    self.FUSE_CONV,
                    self.FUSE_HEAD_CAST,
                    self.FUSE_L2,
                    self.FUSE_Q_SCALE,
                    self.FUSE_REPEAT_HEADS,
                    self.FUSE_FMA,
                    self.FUSE_DECAY_EXP,
                    self.FUSE_SOFTPLUS,
                    self.FUSE_PREDICTION_DOT,
                    self.FUSE_OUTPUT_DOT,
                )
            )
            and not (self.FUSE_RECURRENCE_PREDECAY or self.FUSE_LINEAR_HEAD_NORM)
            and len({s.recurrent.buffer_address() for s in states}) == len(states)
            and len({s.conv.buffer_address() for s in states}) == len(states)
        )
        projections = (
            self._batch_projections(
                batch_input,
                operands,
                retain_packed=merge_recurrence and self.FUSE_BATCH_CONV and self.FUSE_BATCH_PACKED,
                retain_shared_heads=merge_shared,
            )
            if self.FUSE_BATCH_PROJECTION
            else [None] * len(inputs)
        )
        if merge_shared:
            h = self._shared_cache_attention(projections, operands, shared_cache_plan)
        elif merge_recurrence:
            h = self._batch_linear_attention(inputs, states, projections)
        else:
            residuals = [
                ttnn.to_memory_config(
                    self._decode_attention_residual(
                        x, state=s, projection=p, return_projection_input=self.FUSE_BATCH_OUTPUT, **kw
                    ),
                    ttnn.DRAM_MEMORY_CONFIG,
                )
                for x, s, kw, p in zip(inputs, states, operands, projections, strict=True)
            ]
            h = ttnn.concat(residuals, dim=2)
        if self.FUSE_BATCH_OUTPUT:
            projected = self.linear(
                h,
                "self_attn.o_proj" if self.kind == "full_attention" else "linear_attn.out_proj",
                decode_rows=True,
                bias=batch_input if self.FUSE_BATCH_BIAS else None,
            )
            h = (
                projected
                if self.FUSE_BATCH_BIAS
                else ttnn.add(projected, batch_input, memory_config=projected.memory_config())
            )
        normalized = self.norm(h, "post_attention_layernorm", decode_rows=self.FUSE_BATCH_OUTPUT)
        out = ttnn.add(
            h,
            self.moe_prefill(normalized, batch_decode=self.FUSE_BATCH_DIRECT),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Untilize the merged output once; unaligned TILE row slices would
        # otherwise convert the entire merged tensor repeatedly.
        rows = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        return [ttnn.to_layout(rows[:, :, i : i + 1, :], ttnn.TILE_LAYOUT) for i in range(len(inputs))]
