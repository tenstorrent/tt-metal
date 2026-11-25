# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Ornith TP4 attention / EP4 active-expert decoder on 1x4 Blackhole.

Setup owns host conversions. Both layer kinds preserve the selected residual
layout across stack boundaries. Request state and collective workspace must
outlive every trace which references them; execution is serialized on CQ0.
"""
from dataclasses import dataclass

import ttnn
from models.demos.gpt_oss.tt.ccl import CCLManager

from .optimized_decoder import DecoderState, OptimizedDecoder, SharedCacheBatchPlan
from .precision import layer_precision

DRAM = ttnn.DRAM_MEMORY_CONFIG


@dataclass(frozen=True)
class MeshPolicy:
    residual_layout: str = "replicated"
    num_links: int = 2
    prefill_chunk: int = 2048
    batch1_path: str = "direct"
    ccl_mode: str = "direct"
    dram_projection: bool = True
    packed_router: bool = True
    small_matmul: bool = True
    batch_heads: bool = True
    batch_shared_cache: bool = True
    residual_l1: bool = True
    attention_fidelity: str = "HiFi2"
    attention_readers: int = 2
    output_readers: int = 2
    projection_input_cores: int = 8
    attention_block: int = 0
    output_block: int = 0
    router_fidelity: str = "HiFi4"
    router_block: int = 16
    router_input_l1: bool = False
    recurrence_dram_update: bool = True
    expert_grid: tuple = (11, 8)
    expert_gate_dtype: str = "bfloat8_b"
    expert_down_dtype: str = "bfloat8_b"
    routed_row_major: bool = True
    indexed_combine: bool = False
    ccl_dtype: str = "bfloat16"


class MeshCCLManager(CCLManager):
    def _init_subdevice(self):
        g = self.mesh_device.compute_with_storage_grid_size()
        self.ccl_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
        self.ccl_sub_device_id = ttnn.SubDeviceId(0)


class CollectiveWorkspace:
    """Scratch shared only by serial layers; prepare every capacity before capture."""

    def __init__(self, device, policy):
        self.device = device
        self.policy = policy
        self.ccl = MeshCCLManager(device, policy.num_links, ttnn.Topology.Linear)
        self.expert_outputs = {}
        self.route_payloads = {}
        self.direct_output = ttnn.zeros((1, 1, 256, 2048), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.ar_index = 0
        if policy.ccl_mode == "direct":
            mem = ttnn.create_sharded_memory_config(
                (32, 1024), ttnn.CoreGrid(x=8, y=1), ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
            )
            self.ar_buffers = [
                ttnn.empty(
                    (1, 1, 32, 8192),
                    dtype=getattr(ttnn, policy.ccl_dtype),
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=mem,
                )
                for _ in range(2)
            ]
            self.ar_semaphores = [ttnn.create_global_semaphore(device, self.ccl.ccl_cores, 0) for _ in range(2)]

    def prepare(self, tokens):
        capacity = ((8 * tokens + 31 * min(64, 8 * tokens) + 31) // 32) * 32
        if capacity not in self.expert_outputs:
            self.expert_outputs[capacity] = ttnn.zeros(
                (capacity, 2048), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            if self.policy.indexed_combine:
                import torch

                rows = torch.arange(capacity, dtype=torch.int64)
                payload = torch.zeros(1, 1, capacity, 32, dtype=torch.bfloat16)
                for byte in range(3):
                    payload[0, 0, :, byte] = ((rows >> (8 * byte)) & 255).to(torch.bfloat16)
                self.route_payloads[capacity] = ttnn.from_torch(
                    payload,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                )
        return capacity


class MultichipDecoder(OptimizedDecoder):
    FUSE_PACKED_DRAM = False
    FUSE_OUTPUT_DRAM = False
    FUSE_QKV_L1 = False
    FUSE_NORM_TO_PROJ = False
    FUSE_ROUTER_GATHER_L1 = False
    FUSE_DIRECT_RESIDUAL = False
    FUSE_GATED_NORM = False
    FUSE_BATCH_GATED_NORM = False
    FUSE_CONV_CARRY = False
    FUSE_MOE = False  # prepare_prefill metadata is owned by this subclass
    FUSE_RESIDUAL = False
    PREFILL_CHUNK_SIZES = {"linear_attention": 2048, "full_attention": 2048}
    # Q256 native SDPA circular buffers overlap the persistent direct-CCL
    # L1 buffers. Q64 retains K256/FP32 accumulation and wins the measured
    # 32K prefill comparison over Q128 by 23.8%; padding remains internal.
    PREFILL_SDPA_Q_CHUNK = 64

    @classmethod
    def from_state_dict(
        cls, state_dict, *, hf_config, layer_idx, mesh_device, policy=None, collective_workspace=None, precision=None
    ):
        import torch

        self = cls()
        self.config = getattr(hf_config, "text_config", hf_config)
        c = self.config
        if tuple(mesh_device.shape) != (1, 4):
            raise ValueError("Ornith multichip requires the 1x4 Blackhole mesh")
        if (c.hidden_size, c.num_experts, c.num_experts_per_tok, c.moe_intermediate_size) != (2048, 256, 8, 512):
            raise ValueError("Expected Ornith-1.5-35B-A3B")
        self.device = mesh_device
        self.kind = c.layer_types[layer_idx]
        self.policy = policy or MeshPolicy()
        self.precision = layer_precision(precision, layer_idx) if precision is not None else None
        self.KV_CACHE_DTYPE = precision["kv_cache_dtype"] if precision is not None else "bfloat8_b"
        if self.policy.residual_layout not in ("sharded", "replicated"):
            raise ValueError("Unknown residual layout")
        self.hidden_width = 512 if self.policy.residual_layout == "sharded" else 2048
        self.workspace = collective_workspace or CollectiveWorkspace(mesh_device, self.policy)
        if self.workspace.device != mesh_device or self.workspace.policy != self.policy:
            raise ValueError("Collective workspace device/policy mismatch")
        self.ccl = self.workspace.ccl
        self._shared_cache_owner = object()
        self.weights = {}
        self.prefill_corrections = {}
        self.shard_norms = {}
        self.PREFILL_CHUNK_SIZES = {k: self.policy.prefill_chunk for k in self.PREFILL_CHUNK_SIZES}

        def compute(fidelity, fp32=True, packer=False):
            return ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=getattr(ttnn.MathFidelity, fidelity),
                math_approx_mode=False,
                fp32_dest_acc_en=fp32,
                packer_l1_acc=packer,
            )

        fidelities = self.precision["compute_fidelities"] if self.precision is not None else {}
        self.compute = compute(fidelities.get("norm_recurrence", "HiFi4"))
        self.router_compute = compute(fidelities.get("router", self.policy.router_fidelity))
        self.attention_compute = compute(fidelities.get("attention_decode", self.policy.attention_fidelity), True, True)
        self.BATCH_RECURRENCE_DRAM_UPDATE = self.policy.recurrence_dram_update
        self.prefill_attention_compute = compute(fidelities.get("attention_prefill", "HiFi4"), True, True)
        self.prefill_attention_bf16_compute = compute(fidelities.get("attention_prefill", "HiFi4"), False, True)
        self.sdpa_compute = compute(fidelities.get("sdpa", "HiFi2"))
        self.expert_compute = compute(fidelities.get("experts", "LoFi"), False)
        self.shared_compute = compute(fidelities.get("shared", "LoFi"), False, True)
        self.rope_compute = compute("HiFi4", False)
        self.softplus_activation = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)]

        def upload(value, dtype=ttnn.bfloat16, dim=None, layout=ttnn.TILE_LAYOUT, memory=DRAM):
            return ttnn.from_torch(
                value.contiguous(),
                device=mesh_device,
                dtype=dtype,
                layout=layout,
                memory_config=memory,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
                if dim is None
                else ttnn.ShardTensorToMesh(mesh_device, dim=dim),
            )

        def matrix(name, parts, accurate=False):
            packed = torch.cat(parts, dim=0)
            group = "shared_gate_up" if name == "shared_gu" else "shared_down" if name == "shared_down" else "attention"
            dtype = self.precision["weight_groups"][group] if self.precision is not None else "bfloat8_b"
            self.weights[name + ".weight"] = upload(packed, getattr(ttnn, dtype), dim=0)
            if accurate:
                # Same prepared BF16 policy as baseline, with rank-local BFP8 residual.
                high = self.weights[name + ".weight"]
                host = ttnn.to_torch(high, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).reshape_as(packed)
                low = upload(packed.float() - host.float(), ttnn.bfloat8_b, dim=0)
                self.prefill_corrections[name + ".weight"] = ttnn.add(high, low, dtype=ttnn.bfloat16)

        fixed_weights = self.precision["weight_groups"] if self.precision is not None else {}
        norm_dtype = getattr(ttnn, fixed_weights.get("norm", "float32"))
        router_dtype = getattr(ttnn, fixed_weights.get("router", "float32"))
        for name in ("input_layernorm", "post_attention_layernorm"):
            gamma = (state_dict[name + ".weight"].float() + 1).reshape(1, 1, 1, 2048)
            self.weights[name + ".weight"] = upload(gamma, norm_dtype)
            self.shard_norms[name] = upload(gamma, norm_dtype, dim=3)
        self.weights["mlp.gate.weight"] = upload(state_dict["mlp.gate.weight"].T.float(), router_dtype)
        self.weights["mlp.shared_expert_gate.weight"] = upload(
            state_dict["mlp.shared_expert_gate.weight"].T.float(), router_dtype
        )
        self.weights["mesh_router.weight"] = upload(
            torch.nn.functional.pad(
                torch.cat(
                    [state_dict["mlp.gate.weight"].T.float(), state_dict["mlp.shared_expert_gate.weight"].T.float()],
                    dim=1,
                ),
                (0, 255),
            ),
            router_dtype,
        )
        matrix(
            "shared_gu",
            [
                torch.cat(
                    [
                        state_dict["mlp.shared_expert." + role + "_proj.weight"][r * 128 : (r + 1) * 128].T
                        for role in ("gate", "up")
                    ],
                    dim=1,
                )
                for r in range(4)
            ],
        )
        matrix(
            "shared_down",
            [state_dict["mlp.shared_expert.down_proj.weight"][:, r * 128 : (r + 1) * 128].T for r in range(4)],
        )
        if self.kind == "linear_attention":
            qkv = state_dict["linear_attn.in_proj_qkv.weight"]
            parts = []
            taps = []
            for r in range(4):
                indices = torch.cat(
                    [
                        torch.arange(r * 512, (r + 1) * 512),
                        torch.arange(2048 + r * 512, 2048 + (r + 1) * 512),
                        torch.arange(4096 + r * 1024, 4096 + (r + 1) * 1024),
                    ]
                )
                parts.append(
                    torch.cat(
                        [
                            qkv[indices].T,
                            state_dict["linear_attn.in_proj_z.weight"][r * 1024 : (r + 1) * 1024].T,
                            *[
                                torch.nn.functional.pad(
                                    state_dict["linear_attn.in_proj_" + role + ".weight"][r * 8 : (r + 1) * 8].T,
                                    (0, 24),
                                )
                                for role in ("a", "b")
                            ],
                        ],
                        dim=1,
                    )
                )
                taps.append(state_dict["linear_attn.conv1d.weight"][indices, 0, :])
            matrix("packed_attn", parts, accurate=True)
            matrix(
                "linear_attn.out_proj",
                [state_dict["linear_attn.out_proj.weight"][:, r * 1024 : (r + 1) * 1024].T for r in range(4)],
                accurate=True,
            )
            self.conv_taps = [upload(torch.cat([p[:, i] for p in taps]).reshape(1, 1, -1), dim=2) for i in range(4)]
            self.batch_conv = {}
            for bucket in self.BATCH_CONV_BUCKETS:
                self.batch_conv[bucket] = (
                    [
                        upload(torch.cat([part[:, i].repeat(bucket) for part in taps]).reshape(1, 1, -1), dim=2)
                        for i in range(4)
                    ],
                    None,
                )
            self.conv_start = upload(torch.zeros(1, dtype=torch.int32), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            self.dt = upload(state_dict["linear_attn.dt_bias"].float().reshape(1, 1, 32), ttnn.float32, dim=2)
            self.decay = upload(-state_dict["linear_attn.A_log"].float().exp().reshape(1, 1, 32), ttnn.float32, dim=2)
            self.weights["linear_attn.norm.weight"] = upload(
                state_dict["linear_attn.norm.weight"].float().reshape(1, 1, 1, 128), ttnn.float32
            )
            self.weights["l2.weight"] = upload(torch.full((1, 1, 1, 128), 128**-0.5), ttnn.float32)
            self.weights["q_l2.weight"] = upload(torch.full((1, 1, 1, 128), 1 / 128), ttnn.float32)
            i, j = torch.arange(32)[:, None], torch.arange(32)[None, :]
            masks = torch.cat(
                [((i < 16) & (j < 16)).float(), ((i >= 16) & (j >= 16)).float(), ((i >= 16) & (j < 16)).float()], dim=1
            )
            for chunk, attr in ((64, "gdn_constants"), (32, "gdn_constants32")):
                setattr(
                    self,
                    attr,
                    {
                        n: upload(v.reshape(1, 1, *v.shape), ttnn.float32)
                        for n, v in dict(
                            eye=torch.eye(chunk),
                            tril=torch.ones(chunk, chunk).tril(),
                            ones=torch.ones(chunk, chunk),
                            masks=masks,
                        ).items()
                    },
                )
        else:
            order = torch.cat(
                [torch.stack([torch.arange(32), torch.arange(32, 64)], dim=-1).flatten(), torch.arange(64, 256)]
            )
            qg = state_dict["self_attn.q_proj.weight"].reshape(16, 512, 2048)
            kw = state_dict["self_attn.k_proj.weight"].reshape(2, 256, 2048)
            vw = state_dict["self_attn.v_proj.weight"].reshape(2, 256, 2048)
            matrix(
                "heads_attn",
                [
                    torch.cat(
                        [
                            qg[r * 4 : (r + 1) * 4, :256][:, order].reshape(1024, 2048).T,
                            kw[r // 2, order].T,
                            vw[r // 2].T,
                            qg[r * 4 : (r + 1) * 4, 256:].reshape(1024, 2048).T,
                        ],
                        dim=1,
                    )
                    for r in range(4)
                ],
            )
            matrix(
                "self_attn.o_proj",
                [state_dict["self_attn.o_proj.weight"][:, r * 1024 : (r + 1) * 1024].T for r in range(4)],
            )
            for name in ("q", "k"):
                self.weights["self_attn." + name + "_norm.weight"] = upload(
                    (state_dict["self_attn." + name + "_norm.weight"].float() + 1)[order].reshape(1, 1, 1, 256),
                    ttnn.float32,
                )
            transform = torch.zeros(1, 1, 32, 32)
            transform[..., torch.arange(0, 32, 2), torch.arange(1, 32, 2)] = 1
            transform[..., torch.arange(1, 32, 2), torch.arange(0, 32, 2)] = -1
            self.batch_row_ids = [
                upload(torch.tensor([i], dtype=torch.int32), ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
                for i in range(32)
            ]
            self.rope_transform = upload(transform)
            self.batch_rope_transform = upload(
                transform.repeat(1, 64, 1, 1),
                memory=ttnn.create_sharded_memory_config(
                    (32, 32),
                    ttnn.CoreGrid(x=8, y=8),
                    ttnn.ShardStrategy.HEIGHT,
                    use_height_and_width_as_shard_shape=True,
                ),
            )
            self.rope_transform_decode = upload(
                transform.repeat(1, 2, 1, 1),
                memory=ttnn.create_sharded_memory_config(
                    (32, 32),
                    ttnn.CoreGrid(x=2, y=1),
                    ttnn.ShardStrategy.HEIGHT,
                    use_height_and_width_as_shard_shape=True,
                ),
            )
        grid = mesh_device.dram_grid_size()

        def expert_mem(width):
            return ttnn.MemoryConfig(
                buffer_type=ttnn.BufferType.DRAM,
                nd_shard_spec=ttnn.NdShardSpec(
                    shard_shape=ttnn.Shape([32, width]),
                    grid=ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
                    ),
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )

        gu = state_dict["mlp.experts.gate_up_proj"]
        down = state_dict["mlp.experts.down_proj"]
        self.expert_lists = [
            [
                upload(
                    torch.cat([values[64 * r + j].T for r in range(4)], dim=0),
                    getattr(ttnn, dtype),
                    dim=0,
                    memory=expert_mem(width),
                )
                for j in range(64)
            ]
            for values, width, dtype in (
                (
                    gu[:, :512],
                    64,
                    self.precision["weight_groups"]["expert_gate_up"]
                    if self.precision
                    else self.policy.expert_gate_dtype,
                ),
                (
                    gu[:, 512:],
                    64,
                    self.precision["weight_groups"]["expert_gate_up"]
                    if self.precision
                    else self.policy.expert_gate_dtype,
                ),
                (
                    down,
                    192,
                    self.precision["weight_groups"]["expert_down"] if self.precision else self.policy.expert_down_dtype,
                ),
            )
        ]
        if self.policy.dram_projection:
            for name, weight in list(self.weights.items()):
                if name in (
                    "heads_attn.weight",
                    "packed_attn.weight",
                    "linear_attn.out_proj.weight",
                    "self_attn.o_proj.weight",
                ):
                    if name not in self.prefill_corrections:
                        self.prefill_corrections[name] = weight
                    banks = mesh_device.dram_grid_size().x
                    width = ((weight.shape[-1] + 32 * banks - 1) // (32 * banks)) * 32
                    mem = ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                        ttnn.BufferType.DRAM,
                        ttnn.ShardSpec(
                            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
                            [weight.shape[-2], width],
                            ttnn.ShardOrientation.ROW_MAJOR,
                        ),
                    )
                    self.weights[name] = ttnn.to_memory_config(weight, mem)
        self.direct_counts_zero = upload(
            torch.zeros(1, 1, 1, 256, dtype=torch.int32), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.direct_count_one = upload(
            torch.ones(1, 1, 1, 8, dtype=torch.int32), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.direct_slot_offsets = upload(
            (32 * torch.arange(8, dtype=torch.int32)).reshape(1, 1, 1, 8), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        ownership = torch.zeros(4, 1, 1, 256)
        for rank in range(4):
            ownership[rank, ..., rank * 64 : (rank + 1) * 64] = 1
        self.direct_ownership = upload(ownership, ttnn.float32, dim=0)
        self.native_input_zero = upload(torch.zeros(1, dtype=torch.int32), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.expert_ids = upload(torch.arange(256, dtype=torch.int32), ttnn.uint32, dim=0, layout=ttnn.ROW_MAJOR_LAYOUT)
        table = torch.full((4, 256), -1, dtype=torch.int32)
        for r in range(4):
            table[r, r * 64 : (r + 1) * 64] = 0
        self.ep_reduce_table = upload(table, ttnn.int32, dim=0, layout=ttnn.ROW_MAJOR_LAYOUT)
        # Native dispatch skips -1 experts; counts/offsets remain indexed by
        # global ID, but only this rank's 64 experts occupy packed regions.
        self.route_mask = self.ep_reduce_table
        for b in range(1, 33):
            self.workspace.prepare(b)
        return self

    def create_state(self, *, context=262144, page_size=32):
        if not 256 <= context <= 262144 or context % 256 or page_size != 32:
            raise ValueError("Cache allocation extent must be a multiple of256 within context")

        def zeros(shape, dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.zeros(shape, device=self.device, dtype=dtype, layout=layout)

        if self.kind == "linear_attention":
            return DecoderState(
                conv=zeros((1, 1, 3, 2048), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                recurrent=zeros((1, 8, 128, 128), ttnn.float32),
            )
        shape = (context // 32, 1, 32, 256)
        return DecoderState(
            key=zeros(shape, getattr(ttnn, self.KV_CACHE_DTYPE)), value=zeros(shape, getattr(ttnn, self.KV_CACHE_DTYPE))
        )

    def _gather(self, x):
        return ttnn.experimental.all_gather_async(
            x,
            dim=3,
            cluster_axis=1,
            mesh_device=self.device,
            topology=self.ccl.topology,
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            num_links=self.policy.num_links,
            memory_config=x.memory_config(),
        )

    def _reduce(self, x):
        original_dtype = x.dtype
        ccl_dtype = getattr(ttnn, self.policy.ccl_dtype)
        if x.dtype != ccl_dtype:
            x = ttnn.typecast(x, ccl_dtype)
        if self.hidden_width == 2048 and self.policy.ccl_mode == "direct" and x.shape[-2] <= 32:
            mem = ttnn.create_sharded_memory_config(
                (32, 256), ttnn.CoreGrid(x=8, y=1), ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
            )
            i = self.workspace.ar_index
            self.workspace.ar_index = 1 - i
            result = ttnn.experimental.all_reduce_async(
                ttnn.to_memory_config(x, mem),
                self.workspace.ar_buffers[i],
                cluster_axis=1,
                mesh_device=self.device,
                multi_device_global_semaphore=self.workspace.ar_semaphores[i],
                memory_config=mem,
                dtype=ccl_dtype,
                topology=self.ccl.topology,
                num_links=self.policy.num_links,
            )
            return result if result.dtype == original_dtype else ttnn.typecast(result, original_dtype)
        x = ttnn.to_memory_config(x, DRAM)
        y = ttnn.experimental.reduce_scatter_minimal_async(
            x,
            dim=3,
            cluster_axis=1,
            topology=self.ccl.topology,
            multi_device_global_semaphore=self.ccl.get_rs_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            num_links=self.policy.num_links,
            memory_config=DRAM,
        )
        result = y if self.hidden_width == 512 else self._gather(y)
        return result if result.dtype == original_dtype else ttnn.typecast(result, original_dtype)

    def _root_norm(self, x, name, precise=False):
        if self.hidden_width == 2048:
            if self.policy.residual_l1 and not precise and x.shape[-2] <= 32:
                mem = ttnn.create_sharded_memory_config(
                    (32, 256),
                    ttnn.CoreGrid(x=8, y=1),
                    ttnn.ShardStrategy.WIDTH,
                    use_height_and_width_as_shard_shape=True,
                )
                return ttnn.rms_norm(
                    ttnn.to_memory_config(x, mem),
                    epsilon=self.config.rms_norm_eps,
                    weight=self.weights[name + ".weight"],
                    compute_kernel_config=self.compute,
                    memory_config=mem,
                    program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
                        compute_with_storage_grid_size=[8, 1],
                        subblock_w=4,
                        block_h=1,
                        block_w=8,
                        inplace=False,
                    ),
                )
            # Direct all-reduce uses eight width shards. The baseline's batched
            # sharded norm config describes its single-core head layout only.
            if x.shape[-2] > 1 and x.is_sharded():
                x = ttnn.to_memory_config(x, DRAM)
            return self.norm(x, name, precise=precise)
        x = ttnn.to_memory_config(x, DRAM)
        if precise:
            xf = ttnn.typecast(x, ttnn.float32)
            stats = ttnn.rms_norm_pre_all_gather(
                xf, compute_kernel_config=self.compute, dtype=ttnn.float32, fast_and_approximate_mode=False
            )
            gathered = self._gather(stats)
            total = gathered[..., :1]
            for rank in range(1, 4):
                total = ttnn.add(total, gathered[..., rank * 32 : rank * 32 + 1])
            normalized = ttnn.multiply(
                xf,
                total,
                input_tensor_b_activations=[
                    ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 1 / 2048),
                    ttnn.UnaryWithParam(ttnn.UnaryOpType.ADD_UNARY_SFPU, self.config.rms_norm_eps),
                    ttnn.UnaryOpType.RSQRT,
                ],
            )
            return self._gather(ttnn.multiply(normalized, self.shard_norms[name], dtype=ttnn.bfloat16))
        stats = ttnn.rms_norm_pre_all_gather(x, compute_kernel_config=self.compute, dtype=ttnn.bfloat16)
        n = ttnn.rms_norm_post_all_gather(
            x,
            self._gather(stats),
            epsilon=self.config.rms_norm_eps,
            weight=self.shard_norms[name],
            compute_kernel_config=self.compute,
            memory_config=DRAM,
        )
        return self._gather(n)

    def linear(self, x, name, dtype=ttnn.bfloat16, bias=None, *, decode_rows=False, prefill=False, _weight=None):
        if self.precision is not None and name not in ("mlp.gate", "mlp.shared_expert_gate"):
            activation_dtype = getattr(ttnn, self.precision["activation_dtype"])
            if x.dtype != activation_dtype:
                x = ttnn.typecast(x, activation_dtype)
        weight = (
            _weight
            if _weight is not None
            else self.prefill_corrections.get(name + ".weight", self.weights[name + ".weight"])
            if prefill
            else self.weights[name + ".weight"]
        )
        compute = (
            self.router_compute
            if name in ("mlp.gate", "mlp.shared_expert_gate")
            else self.shared_compute
            if name.startswith("shared_")
            else self.prefill_attention_compute
            if prefill
            else self.attention_compute
        )
        if (
            self.policy.dram_projection
            and not prefill
            and name in ("heads_attn", "packed_attn", "self_attn.o_proj", "linear_attn.out_proj")
        ):
            cores = self.policy.projection_input_cores
            is_output = name.endswith(("o_proj", "out_proj"))
            block = self.policy.output_block if is_output else self.policy.attention_block
            mem = ttnn.create_sharded_memory_config(
                (32, x.shape[-1] // cores),
                ttnn.CoreGrid(x=cores, y=1),
                ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )
            y = ttnn.linear(
                ttnn.to_memory_config(x, mem),
                weight,
                dtype=dtype,
                compute_kernel_config=compute,
                program_config=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=block or x.shape[-1] // (32 * cores),
                    per_core_M=1,
                    per_core_N=weight.memory_config().shard_spec.shape[1] // 32,
                    num_workers_per_dram_bank=self.policy.output_readers
                    if is_output
                    else self.policy.attention_readers,
                    use_multicore_pipeline=True,
                ),
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            y = ttnn.to_memory_config(y, DRAM)
        elif self.policy.small_matmul and name in ("shared_gu", "shared_down") and x.shape[-2] <= 32:
            ntiles = (weight.shape[-1] + 31) // 32
            cores = min(8, ntiles)
            per_n = (ntiles + cores - 1) // cores
            y = ttnn.linear(
                ttnn.to_memory_config(x, DRAM),
                weight,
                dtype=dtype,
                compute_kernel_config=compute,
                program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(cores, 1),
                    in0_block_w=x.shape[-1] // 32,
                    out_subblock_h=1,
                    out_subblock_w=min(4, per_n),
                    per_core_M=1,
                    per_core_N=per_n,
                    fuse_batch=True,
                    mcast_in0=True,
                ),
                memory_config=DRAM,
            )
        else:
            y = ttnn.linear(
                ttnn.to_memory_config(x, DRAM), weight, dtype=dtype, compute_kernel_config=compute, memory_config=DRAM
            )
        return ttnn.add(y, bias) if bias is not None else y

    def _router(self, x):
        if not self.policy.packed_router:
            return (
                self.linear(ttnn.typecast(x, ttnn.float32), "mlp.gate", dtype=ttnn.float32),
                self.linear(ttnn.typecast(x, ttnn.float32), "mlp.shared_expert_gate", dtype=ttnn.float32),
            )
        options = {}
        if x.shape[-2] <= 32:
            options["program_config"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 1),
                in0_block_w=self.policy.router_block,
                out_subblock_h=1,
                out_subblock_w=2,
                per_core_M=1,
                per_core_N=2,
                fuse_batch=True,
                mcast_in0=True,
            )
        logits = ttnn.linear(
            ttnn.to_memory_config(
                x, ttnn.L1_MEMORY_CONFIG if self.policy.router_input_l1 and x.shape[-2] <= 32 else DRAM
            ),
            self.weights["mesh_router.weight"],
            dtype=ttnn.float32,
            compute_kernel_config=self.router_compute,
            memory_config=DRAM,
            **options,
        )
        return logits[..., :256], logits[..., 256:257]

    def _router_topk(self, logits, *, fast_router=False):
        # The small-width FP32 parallel engine is measured for batch-one
        # decode. Keep scalar prefill and fixed batches on their original
        # single-core route; larger prefill retains normal row parallelism.
        options = {}
        if logits.shape[-2] <= 32:
            end = ttnn.CoreCoord(4, 2) if fast_router and logits.shape[-2] == 1 else ttnn.CoreCoord(0, 0)
            options["sub_core_grids"] = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), end)})
        return ttnn.topk(logits, k=8, dim=-1, **options)

    def moe(self, x, *, fast_router=False):
        # Residual storage may be BFP8; dispatch's row-major input contract is
        # BF16. Conversion preserves the quantized values at that boundary.
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        if x.is_sharded():
            x = ttnn.to_memory_config(x, DRAM)
        t = x.shape[-2]
        if t == 1 and self.policy.batch1_path == "direct":
            return self._moe_direct(x, fast_router=fast_router)
        physical = (t + 31) // 32 * 32
        logits, shared_gate = self._router(x)
        scores, ids = self._router_topk(logits, fast_router=fast_router)
        scores = ttnn.typecast(
            ttnn.softmax(scores, dim=-1, numeric_stable=True, compute_kernel_config=self.compute), ttnn.bfloat16
        )
        ids = ttnn.typecast(ids, ttnn.uint16)
        # The pinned masked_bincount splits rows over a fixed 64-core grid (rows % 64 == 0). Pad rows carry
        # expert id 256, which the op skips (expert_idx < n_routed_experts), so the counts are unchanged.
        hist_rows = (t + 63) // 64 * 64
        hist_ids = ttnn.pad(ids, ((0, 0), (0, 0), (0, hist_rows - t), (0, 0)), value=256) if hist_rows != t else ids
        counts = ttnn.reshape(
            ttnn.experimental.deepseek_prefill.masked_bincount(
                ttnn.reshape(hist_ids, (hist_rows, 8)), self.route_mask, 256, 8
            ),
            (1, 256),
        )
        sizes = ttnn.typecast(ttnn.to_layout(ttnn.reshape(counts, (1, 1, 1, 256)), ttnn.TILE_LAYOUT), ttnn.float32)
        sizes = ttnn.multiply(ttnn.ceil(ttnn.multiply(sizes, 1 / 32)), 32)
        offsets = ttnn.reshape(
            ttnn.to_layout(
                ttnn.typecast(ttnn.subtract(ttnn.cumsum(sizes, dim=3), sizes), ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT
            ),
            (1, 256),
        )
        capacity = ((8 * t + 31 * min(64, 8 * t) + 31) // 32) * 32
        dispatch_ids = ttnn.to_layout(ids, ttnn.ROW_MAJOR_LAYOUT)
        routed, metadata = ttnn.experimental.deepseek_prefill.dispatch(
            ttnn.reshape(x, (1, t, 2048)),
            ttnn.reshape(dispatch_ids, (1, t, 8)),
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
        if not self.policy.routed_row_major:
            routed = ttnn.to_layout(routed, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        routed = ttnn.reshape(routed, (capacity, 2048))
        out = ttnn.bringup.moe_fused_swiglu(
            routed,
            *self.expert_lists,
            counts,
            self.expert_ids,
            input_m_tiles=physical // 32,
            output=self.workspace.expert_outputs[capacity],
            expert_region_offsets=offsets,
            read_x_at_offset=True,
            compute_kernel_config=self.expert_compute,
            core_grid=ttnn.CoreCoord(*self.policy.expert_grid),
        )
        combined = ttnn.experimental.deepseek_prefill.combine(
            self.workspace.route_payloads[capacity]
            if self.policy.indexed_combine
            else ttnn.reshape(out, (1, 1, capacity, 2048)),
            metadata,
            counts,
            offsets,
            dispatch_group_size=1,
            experts_per_chip=256,
            num_experts_per_tok=8,
            seq_len_per_chip=physical,
            cluster_axis=0,
            num_links=0,
        )
        combine_options = {}
        if self.policy.indexed_combine:
            # Transport only exact byte-encoded row IDs. Combine zeroes remote
            # slots; the dispatch-table mask below suppresses their contribution.
            encoded = ttnn.typecast(
                ttnn.to_layout(ttnn.reshape(combined, (1, 1, physical * 8, 32)), ttnn.TILE_LAYOUT),
                ttnn.float32,
            )
            index = ttnn.add(
                encoded[..., :1],
                ttnn.add(ttnn.multiply(encoded[..., 1:2], 256.0), ttnn.multiply(encoded[..., 2:3], 65536.0)),
            )
            combine_options["combine_row_index"] = ttnn.reshape(
                ttnn.to_layout(ttnn.typecast(index, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), (1, 1, physical, 8)
            )
            combined = ttnn.reshape(ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT), (capacity, 2048))
        if physical != t:
            scores = ttnn.pad(scores, ((0, 0), (0, 0), (0, physical - t), (0, 0)), value=0)
            dispatch_ids = ttnn.pad(dispatch_ids, ((0, 0), (0, 0), (0, physical - t), (0, 0)), value=0)
        partial = ttnn.bringup.post_combine_reduce(
            combined,
            ttnn.reshape(ttnn.to_layout(scores, ttnn.ROW_MAJOR_LAYOUT), (1, 1, physical, 8, 1)),
            dispatch_ids,
            self.ep_reduce_table,
            expert_dim=3,
            output_memory_config=DRAM,
            **combine_options,
        )[..., :t, :]
        gu = self.linear(x, "shared_gu")
        product = ttnn.multiply(gu[..., :128], gu[..., 128:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        shared = self.linear(product, "shared_down")
        gate = shared_gate
        return self._reduce(ttnn.add(partial, ttnn.multiply(shared, ttnn.typecast(ttnn.sigmoid(gate), ttnn.bfloat16))))

    def _moe_direct(self, x, *, fast_router=False):
        """EP batch1: only selected local experts read the input tile.

        Output slots belong to router order. Other ranks' slots may retain
        finite old data; the rank ownership mask gives those slots zero score.
        The zero-initialized workspace is shared only by serialized forwards.
        """
        logits, shared_gate = self._router(x)
        scores, ids = self._router_topk(logits, fast_router=fast_router)
        scores = ttnn.typecast(
            ttnn.softmax(scores, dim=-1, numeric_stable=True, compute_kernel_config=self.compute), ttnn.bfloat16
        )
        mask = ttnn.gather(self.direct_ownership, dim=-1, index=ids)
        scores = ttnn.multiply(ttnn.typecast(scores, ttnn.float32), mask)
        ids = ttnn.to_layout(ttnn.typecast(ids, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        counts = ttnn.reshape(
            ttnn.scatter(self.direct_counts_zero, dim=-1, index=ids, src=self.direct_count_one), (1, 256)
        )
        offsets = ttnn.reshape(
            ttnn.scatter(self.direct_counts_zero, dim=-1, index=ids, src=self.direct_slot_offsets), (1, 256)
        )
        inp = self._native_input_prefix(x, 256, row_major=False)
        output = ttnn.bringup.moe_fused_swiglu(
            inp,
            *self.expert_lists,
            counts,
            self.expert_ids,
            input_m_tiles=1,
            output=self.workspace.direct_output,
            expert_region_offsets=offsets,
            read_x_at_offset=False,
            compute_kernel_config=self.expert_compute,
            core_grid=ttnn.CoreCoord(*self.policy.expert_grid),
        )
        selected = ttnn.reshape(ttnn.reshape(output, (1, 8, 32, 2048))[:, :, :1, :], (1, 1, 8, 2048))
        partial = ttnn.matmul(scores, selected, compute_kernel_config=self.compute, dtype=ttnn.bfloat16)
        gu = self.linear(x, "shared_gu")
        shared = self.linear(
            ttnn.multiply(gu[..., :128], gu[..., 128:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            "shared_down",
        )
        gate = shared_gate
        return self._reduce(ttnn.add(partial, ttnn.multiply(shared, ttnn.typecast(ttnn.sigmoid(gate), ttnn.bfloat16))))

    def prepare_prefill(self, **kwargs):
        plan = super().prepare_prefill(**kwargs)
        for _, _, physical, _, _ in plan["chunks"]:
            self.workspace.prepare(physical)
        return plan

    def decode_forward(self, x, *, state, page_table=None, current_pos=None, cos=None, sin=None, prefill=False):
        n = self._root_norm(x, "input_layernorm")
        a = (
            self.linear_attention(n, state, prefill=prefill)
            if self.kind == "linear_attention"
            else self.full_attention(n, state, page_table, current_pos, cos, sin, prefill=prefill)
        )
        h = ttnn.add(x, self._reduce(a))
        return ttnn.add(h, self.moe(self._root_norm(h, "post_attention_layernorm"), fast_router=not prefill))

    def prefill_forward(self, x, *, state, plan):
        if isinstance(x, (list, tuple)):
            return [self.prefill_forward(a, state=b, plan=c) for a, b, c in zip(x, state, plan, strict=True)]
        if tuple(x.shape) != (1, 1, plan["seq_len"], self.hidden_width):
            raise ValueError("Input must match logical length and selected mesh residual layout")
        outputs = []
        for offset, count, physical, decode, kw in plan["chunks"]:
            row = x[:, :, offset : offset + count, :]
            if physical != count:
                row = ttnn.pad(row, ((0, 0), (0, 0), (0, physical - count), (0, 0)), value=0)
            if decode:
                y = self.decode_forward(row, state=state, prefill=True, **kw)
            else:
                n = self._root_norm(row, "input_layernorm", precise=self.kind == "linear_attention")
                a = (
                    self.linear_attention(n, state, precise=True, prefill=True)
                    if self.kind == "linear_attention"
                    else self._prefill_attention(n, state=state, **kw)
                )
                h = ttnn.add(row, self._reduce(a))
                y = ttnn.add(
                    h, self.moe(self._root_norm(h, "post_attention_layernorm", precise=self.kind == "linear_attention"))
                )
            # Long chunks use RS/AG in DRAM; <=32-token tails can use the
            # width-sharded direct collective. Concat requires one local layout.
            outputs.append(ttnn.to_memory_config(y, DRAM)[:, :, :count, :])
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)

    def prefill_chunk_forward(self, x, *, state, **operands):
        n = self._root_norm(x, "input_layernorm")
        h = ttnn.add(x, self._reduce(self._prefill_attention(n, state=state, **operands)))
        return ttnn.add(h, self.moe(self._root_norm(h, "post_attention_layernorm")))

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
            num_heads=4,
            num_kv_heads=1,
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
            ttnn.to_memory_config(y, memory), num_heads=4, sub_core_grids=grid(range(64))
        )
        y = ttnn.reshape(y, ttnn.Shape((1, 1, batch, 1024)), ttnn.Shape((1, 1, 32, 1024)))
        output_memory = ttnn.create_sharded_memory_config(
            (32, 128),
            ttnn.CoreGrid(x=8, y=1),
            ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        return ttnn.multiply(
            y, gates, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID], memory_config=output_memory
        )

    def decode_batch(
        self, inputs, *, states, operands, shared_cache_plan=None, state_write_indices=None, batch_active_mask=None
    ):
        if state_write_indices is not None and (
            self.kind != "linear_attention" or len(inputs) < 2 or len(state_write_indices) != len(inputs)
        ):
            raise ValueError("Conditional writes require one index per batched linear-attention request")
        if shared_cache_plan is not None:
            plan = shared_cache_plan
            if (
                self.kind != "full_attention"
                or not isinstance(plan, SharedCacheBatchPlan)
                or plan.owner_token is not self._shared_cache_owner
            ):
                raise ValueError("Shared cache plan must belong to this full-attention decoder")
            signature = self._shared_cache_pool_signature(plan.key, plan.value)
            if len(plan.row_page_tables) != len(inputs) or any(
                self._shared_cache_pool_signature(s.key, s.value) != signature for s in states
            ):
                raise ValueError("Shared cache plan does not match states")
            if any("page_table" in kw for kw in operands):
                raise ValueError("Shared cache plan owns page tables")
            operands = [dict(kw, page_table=table) for kw, table in zip(operands, plan.row_page_tables, strict=True)]
        if not 1 <= len(inputs) <= 32 or len(inputs) != len(states) or len(inputs) != len(operands):
            raise ValueError("Provide1..32 request tensors with matching state and operands")
        batch = len(inputs)
        if batch == 1:
            return [self.decode_forward(inputs[0], state=states[0], **operands[0])]
        batch_input = ttnn.concat(inputs, dim=2)
        n = self._root_norm(batch_input, "input_layernorm")
        packed = self.linear(n, "packed_attn" if self.kind == "linear_attention" else "heads_attn", decode_rows=True)
        if self.kind == "linear_attention":
            local = self._batch_linear_attention(inputs, states, packed, state_write_indices=state_write_indices)
        else:
            # Packed input projection and output projection are shared across
            # rows; native cache ownership remains request-local.
            gates = packed[..., 1536:2560]
            qkv = ttnn.to_memory_config(
                packed[..., :1536],
                ttnn.create_sharded_memory_config(
                    (32, 256),
                    ttnn.CoreGrid(x=6, y=1),
                    ttnn.ShardStrategy.WIDTH,
                    use_height_and_width_as_shard_shape=True,
                ),
            )
            heads = self._batch_heads(qkv, operands) if self.policy.batch_heads else None
            if heads is not None and shared_cache_plan is not None and self.policy.batch_shared_cache:
                local = self._shared_cache_attention((*heads, gates), operands, shared_cache_plan)
            else:
                gates = ttnn.to_layout(gates, ttnn.ROW_MAJOR_LAYOUT)
                local = ttnn.concat(
                    [
                        self.full_attention(
                            x,
                            state,
                            projection=(
                                (*heads, ttnn.to_layout(gates[:, :, i : i + 1, :], ttnn.TILE_LAYOUT), i)
                                if heads is not None
                                else (
                                    qkv,
                                    ttnn.to_layout(gates[:, :, i : i + 1, :], ttnn.TILE_LAYOUT),
                                    self.batch_row_ids[i],
                                )
                            ),
                            return_projection_input=True,
                            **kw,
                        )
                        for i, (x, state, kw) in enumerate(zip(inputs, states, operands, strict=True))
                    ],
                    dim=2,
                )
        if batch_active_mask is not None:
            # SDPA skips writes for current_pos=-1. Do not let stale inactive
            # scratch values reach projection/MoE and affect active-row packing.
            local = ttnn.where(batch_active_mask, local, 0.0, memory_config=local.memory_config())
        projected = self.linear(
            local, "linear_attn.out_proj" if self.kind == "linear_attention" else "self_attn.o_proj", decode_rows=True
        )
        h = ttnn.add(batch_input, self._reduce(projected))
        y = ttnn.add(h, self.moe(self._root_norm(h, "post_attention_layernorm")))
        # Request rows are independent tiled tensors. Splitting a width-sharded
        # ROW_MAJOR tensor would retain a one-row physical shard, which cannot
        # be retiled. Keep mesh ownership and normalize only the on-chip layout.
        rows = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        return [ttnn.to_layout(rows[:, :, i : i + 1, :], ttnn.TILE_LAYOUT) for i in range(len(inputs))]

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
            qkv = projections if self.FUSE_CONV_INPUT and self.FUSE_CONV and t >= 3 else projections[..., :2048]
        else:
            qkv = self.linear(x, "linear_attn.in_proj_qkv", prefill=prefill)
        history = None if self.FUSE_CONV else ttnn.concat([state.conv, qkv], dim=2)
        if self.FUSE_CONV:
            physical = (t + 31) // 32 * 32
            inp = ttnn.pad(qkv, ((0, 0), (0, 0), (0, physical - t), (0, 0)), value=0.0) if physical != t else qkv
            hist = ttnn.to_layout(ttnn.reshape(state.conv, (1, 3, 2048)), ttnn.ROW_MAJOR_LAYOUT)
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
                512,
                512,
                1024,
                actual_start=self.conv_start,
                program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=self.CONV_CHANNEL),
                compute_kernel_config=self.compute,
                **extra,
            )
            q, k, v = conv_outputs[:3]
            if self.FUSE_CONV_CARRY:
                tail = ttnn.reshape(conv_outputs[3], tuple(state.conv.shape))
            elif self.FUSE_RM_HISTORY:
                tail = ttnn.to_layout(qkv[:, :, -3:, :2048] if t >= 3 else qkv, ttnn.ROW_MAJOR_LAYOUT)
                if t < 3:
                    tail = ttnn.concat([state.conv, tail], dim=2)[:, :, -3:, :]
            else:
                tail = qkv[:, :, -3:, :2048] if t >= 3 else ttnn.concat([state.conv, qkv], dim=2)[:, :, -3:, :]
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
                            ttnn.reshape(value[:, :t, :], (1, 1, t, 512)),
                            None,
                            self.device,
                            [],
                            epsilon=1e-6 / 128,
                            num_heads_per_device=4,
                            per_head_norm=True,
                            weight=self.weights[f"linear_{name}_head_norm"],
                            dtype=ttnn.float32,
                            compute_kernel_config=self.compute,
                        )
                        for value, name in [(q, "q"), (k, "k")]
                    ]
                    # Normalize/split returns [1,16,T,128], already in head-major order.
                    v = heads(v, 8)
                else:
                    q, k, v = heads(q, 4), heads(k, 4), heads(v, 8)
        else:
            out = None
            for i in range(4):
                term = ttnn.multiply(ttnn.typecast(history[:, :, i : i + t, :], ttnn.float32), self.taps[i])
                out = term if out is None else ttnn.add(out, term)
            ttnn.copy(history[:, :, -3:, :], state.conv)
            qkv = ttnn.silu(ttnn.typecast(out, ttnn.bfloat16))
            q = ttnn.reshape(qkv[..., :512], (1, t, 4, 128))
            k = ttnn.reshape(qkv[..., 512:1024], (1, t, 4, 128))
            v = ttnn.reshape(qkv[..., 1024:], (1, t, 8, 128))

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
                projections[..., 3072:3080]
                if self.PACK_ATTENTION
                else self.linear(x, "linear_attn.in_proj_a", prefill=prefill)
            ),
            (1, t, 8),
        )
        b = ttnn.reshape(
            (
                projections[..., 3104:3112]
                if self.PACK_ATTENTION
                else self.linear(x, "linear_attn.in_proj_b", prefill=prefill)
            ),
            (1, t, 8),
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
            decayed = ttnn.multiply(state.recurrent, ttnn.reshape(ttnn.exp(g), (1, 8, 1, 1)))
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
            rate = ttnn.reshape(beta, (1, 8, 1, 1))
            if self.FUSE_DECAY_EXP:
                s = ttnn.multiply(
                    state.recurrent, ttnn.reshape(g, (1, 8, 1, 1)), input_tensor_b_activations=[ttnn.UnaryOpType.EXP]
                )
            else:
                decay = ttnn.reshape(ttnn.exp(g), (1, 8, 1, 1))
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
                head_major_y = ttnn.reshape(y, (8, physical if flat_gdn else t, 128))
            if self.FUSE_GDN_OUTPUT and not gated_norm:
                y = ttnn.permute(ttnn.reshape(y, (1, 8, physical if flat_gdn else t, 128)), (0, 2, 1, 3))
            if flat_gdn and not gated_norm:
                y = y[:, :t, :, :]
        if not (t == 1 and self.FUSE_STATE_WRITE and self.FUSE_FMA and not self.FUSE_RECURRENCE_PREDECAY):
            ttnn.copy(s, state.recurrent)
        if gated_norm:
            if head_major_y is None:
                head_major_y = ttnn.reshape(ttnn.permute(y, (0, 2, 1, 3)), (8, t, 128))
            y = self._gated_head_norm(head_major_y, projections, t)
            return (
                y if return_projection_input else self.linear(y, "linear_attn.out_proj", bias=residual, prefill=prefill)
            )
        y = ttnn.reshape(ttnn.typecast(y, ttnn.bfloat16), (1, 1, t * 8, 128))
        z = (
            projections[..., 2048:3072]
            if self.PACK_ATTENTION
            else self.linear(x, "linear_attn.in_proj_z", prefill=prefill)
        )
        if t == 1 and self.FUSE_DECODE_RESHAPE:
            z = ttnn.to_layout(ttnn.reshape(ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT), (1, 1, 8, 128)), ttnn.TILE_LAYOUT)
        else:
            z = ttnn.reshape(z, (1, 1, t * 8, 128))
        y = (
            ttnn.multiply(
                self.norm(y, "linear_attn.norm", precise=precise), z, input_tensor_b_activations=[ttnn.UnaryOpType.SILU]
            )
            if self.FUSE_LINEAR_GATE
            else ttnn.multiply(self.norm(y, "linear_attn.norm", precise=precise), ttnn.silu(z))
        )
        y = ttnn.reshape(y, (1, 1, t, 1024))
        return y if return_projection_input else self.linear(y, "linear_attn.out_proj", bias=residual, prefill=prefill)

    def qkv(self, x, cos, sin, projection=None, *, prefill=False):
        tokens = x.shape[-2]
        if projection is not None and len(projection) == 5:
            q, k, v, gate, slot = projection
            rows = []
            for value, heads, column in ((q, 4, 0), (k, 1, 1), (v, 1, 0)):
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

                q_part = part(0, 1024, ttnn.CoreGrid(x=8, y=2))
                k_part = part(1024, 1280, ttnn.CoreGrid(x=2, y=1))
                v_part, gate = part(1280, 1536), part(1536, 2560)
            else:
                q_part, k_part = projections[..., :1024], projections[..., 1024:1280]
                v_part, gate = projections[..., 1280:1536], projections[..., 1536:]
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
                    (q_part, 4, "q"),
                    (k_part, 1, "k"),
                )
            ]
            if self.FUSE_NORM_ROPE and not self.KEEP_INTERLEAVED_QK:

                def original_order(y):
                    b, h, t, _ = y.shape
                    rot = ttnn.reshape(y[..., :64], (b, h, t, 32, 2))
                    rot = ttnn.reshape(ttnn.permute(rot, (0, 1, 2, 4, 3)), (b, h, t, 64))
                    return ttnn.concat([rot, y[..., 64:]], dim=-1)

                q, k = original_order(q), original_order(k)
            v = ttnn.permute(ttnn.reshape(v_part, (1, tokens, 1, 256)), (0, 2, 1, 3))
            if tokens == 1 and self.SHARDED_HEADS:
                q, k, v = [ttnn.permute(y, (0, 2, 1, 3)) for y in (q, k, v)]
        elif self.FUSE_HEADS:
            if projection is None:
                projections = self.linear(x, "heads_attn", prefill=prefill)
                if tokens == 1 and self.FUSE_QKV_L1 and self.FUSE_PACKED_DRAM:
                    gate = ttnn.slice(
                        projections,
                        (0, 0, 0, 1536),
                        (1, 1, 1, 2560),
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    qkv_mem = ttnn.create_sharded_memory_config(
                        (32, 256),
                        ttnn.CoreGrid(x=6, y=1),
                        ttnn.ShardStrategy.WIDTH,
                        use_height_and_width_as_shard_shape=True,
                    )
                    qkv = ttnn.slice(
                        projections,
                        (0, 0, 0, 0),
                        (1, 1, 1, 1536),
                        memory_config=qkv_mem,
                    )
                else:
                    gate = projections[..., 1536:]
                    qkv = projections[..., :1536]
            else:
                qkv, gate, batch_offset = projection
            if tokens == 1:
                qkv = ttnn.to_memory_config(
                    qkv,
                    ttnn.create_sharded_memory_config(
                        (32, 256),
                        ttnn.CoreGrid(x=6, y=1),
                        ttnn.ShardStrategy.WIDTH,
                        use_height_and_width_as_shard_shape=True,
                    ),
                )
                mem = ttnn.create_sharded_memory_config(
                    (32, 256),
                    ttnn.CoreGrid(y=1, x=2 if self.FUSE_DISJOINT_HEADS else 1),
                    ttnn.ShardStrategy.HEIGHT,
                    use_height_and_width_as_shard_shape=True,
                )
                q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                    qkv,
                    num_heads=4,
                    num_kv_heads=1,
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
                    ttnn.reshape(qkv, (1, tokens, 1536)),
                    num_heads=4,
                    num_kv_heads=1,
                    transpose_key=False,
                )
            q, k = self.norm(q, "self_attn.q_norm"), self.norm(k, "self_attn.k_norm")
        else:
            projections = self.linear(x, "packed_attn", prefill=prefill) if self.PACK_ATTENTION else None
            packed = ttnn.reshape(
                projections[..., :2048] if self.PACK_ATTENTION else self.linear(x, "self_attn.q_proj", prefill=prefill),
                (1, tokens, 4, 512),
            )
            q, gate = packed[..., :256], packed[..., 256:]
            q = ttnn.permute(self.norm(q, "self_attn.q_norm"), (0, 2, 1, 3))
            k = ttnn.reshape(
                (
                    projections[..., 2048:2304]
                    if self.PACK_ATTENTION
                    else self.linear(x, "self_attn.k_proj", prefill=prefill)
                ),
                (1, tokens, 1, 256),
            )
            k = ttnn.permute(self.norm(k, "self_attn.k_norm"), (0, 2, 1, 3))
            v = ttnn.permute(
                ttnn.reshape(
                    (
                        projections[..., 2304:2560]
                        if self.PACK_ATTENTION
                        else self.linear(x, "self_attn.v_proj", prefill=prefill)
                    ),
                    (1, tokens, 1, 256),
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
            return q, k, v, ttnn.reshape(gate, (1, 1, tokens, 1024))

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
            ttnn.reshape(gate, (1, 1, tokens, 1024)),
        )

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
            y = ttnn.experimental.nlp_concat_heads_decode(ttnn.to_memory_config(y, mem), num_heads=4)
            y = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)[:, :, :1, :]
        y = ttnn.multiply(
            ttnn.reshape(y, (1, 1, 1, 1024)),
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

    def _prefill_attention(self, x, *, state, page_table, chunk_page_table, chunk_start, cos, sin):
        q, k, v, gate = self.qkv(x, cos, sin, prefill=True)
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
        y = ttnn.reshape(ttnn.transformer.concatenate_heads(y), (1, 1, x.shape[-2], 1024))
        return self.linear(
            ttnn.multiply(y, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]),
            "self_attn.o_proj",
            prefill=True,
        )

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
                or shape[1:] != (1, 32, 256)
                or tuple(tensor.padded_shape) != shape
                or tensor.dtype != getattr(ttnn, self.KV_CACHE_DTYPE)
                or tensor.layout != ttnn.TILE_LAYOUT
                or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            ):
                raise ValueError(
                    "Shared-cache pools require configured KV dtype, TILE layout and DRAM placement [blocks,1,32,256] on this decoder's device"
                )
        if tuple(key.shape) != tuple(value.shape):
            raise ValueError("Shared-cache key and value pool geometry must match")
        if key.buffer_address() == value.buffer_address():
            raise ValueError("Shared-cache key and value pools must use distinct buffers")
        return tuple(key.shape), key.buffer_address(), value.buffer_address()

    def _batch_linear_conv(self, packed, states, *, state_write_indices=None):
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
                outputs.append(
                    self._batch_linear_conv(
                        packed[..., start:end, :],
                        states[start:end],
                        state_write_indices=None if state_write_indices is None else state_write_indices[start:end],
                    )
                )
                start = end
            return tuple(ttnn.concat([output[i] for output in outputs], dim=0) for i in range(3))
        bucket = min(n for n in self.BATCH_CONV_BUCKETS if n >= batch)
        width = bucket * 2048
        taps, exchange = self.batch_conv[bucket]
        inp = ttnn.reshape(ttnn.to_layout(packed[..., :2048], ttnn.ROW_MAJOR_LAYOUT), (1, 1, batch * 2048))
        if bucket != batch:
            inp = ttnn.pad(inp, ((0, 0), (0, 0), (0, width - batch * 2048)), value=0.0)
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
        history = ttnn.reshape(ttnn.concat([state.conv for state in states], dim=3), (1, 3, batch * 2048))
        if bucket != batch:
            history = ttnn.pad(history, ((0, 0), (0, 0), (0, width - batch * 2048)), value=0.0)
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
            bucket * 512,
            bucket * 512,
            bucket * 1024,
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
            carry = ttnn.reshape(ttnn.permute(ttnn.reshape(carry, (3, bucket, 2048)), (1, 0, 2)), (bucket, 1, 3, 2048))
        for i, state in enumerate(states):
            if self.FUSE_BATCH_HISTORY_COPY:
                ttnn.experimental.kda.indexed_entry_copy(
                    carry,
                    state.conv,
                    self.conv_start if state_write_indices is None else state_write_indices[i],
                    index_scale=1,
                    index_offset=i,
                )
            else:
                ttnn.slice(carry, (0, 0, 0, i * 2048), (1, 1, 3, (i + 1) * 2048), output_tensor=state.conv)
        # Output partitions are arbitrary channel ranges. Restore actual user
        # Q/K/V boundaries only after discarding physical time padding.
        output = ttnn.to_layout(ttnn.concat(conv[:3], dim=2)[:, :1, :], ttnn.ROW_MAJOR_LAYOUT)
        output = ttnn.reshape(output, (bucket, 1, 2048))[:batch]
        return tuple(
            ttnn.to_layout(
                ttnn.reshape(output[..., start:end], (batch, 1, heads, 128)), ttnn.TILE_LAYOUT, dtype=ttnn.float32
            )
            for start, end, heads in ((0, 512, 4), (512, 1024, 4), (1024, 2048, 8))
        )

    def _batch_linear_attention(self, inputs, states, projections, *, state_write_indices=None):
        """Peer-merge the existing FP32 recurrence; persistent states stay stable."""
        batch = len(inputs)
        if state_write_indices is not None and not (
            self.FUSE_BATCH_CONV
            and self.FUSE_BATCH_HISTORY_COPY
            and self.FUSE_BATCH_STATE_COPY
            and not self.BATCH_DIRECT_SLICE_FROM_L1
        ):
            raise ValueError("Conditional writes require the selected indexed convolution/state copy path")
        packed = projections if isinstance(projections, ttnn.Tensor) else ttnn.concat(projections, dim=2)
        if self.FUSE_BATCH_CONV:
            q, k, value = self._batch_linear_conv(packed, states, state_write_indices=state_write_indices)
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
        a = ttnn.reshape(packed[..., 3072:3080], (1, batch, 8))
        b = ttnn.reshape(packed[..., 3104:3112], (1, batch, 8))
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
                ttnn.reshape(ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT), (batch, 8, 1, 1)),
                ttnn.TILE_LAYOUT,
            )

        recurrence_memory = ttnn.L1_MEMORY_CONFIG if self.BATCH_RECURRENCE_L1 else ttnn.DRAM_MEMORY_CONFIG
        recurrence_config = None
        head_count = batch * 8
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
                    copy_source,
                    state.recurrent,
                    self.conv_start if state_write_indices is None else state_write_indices[i],
                    index_scale=1,
                    index_offset=i,
                )
            else:
                ttnn.slice(copy_source, (i, 0, 0, 0), (i + 1, 8, 128, 128), output_tensor=state.recurrent)
        y = recurrent_matmul(q, updated)
        if self.FUSE_BATCH_GATED_NORM and batch >= self.BATCH_GATED_NORM_MIN:
            head_major_y = ttnn.reshape(ttnn.permute(y, (2, 1, 0, 3)), (8, batch, 128))
            return self._gated_head_norm(head_major_y, packed, batch)
        y = ttnn.reshape(ttnn.permute(ttnn.typecast(y, ttnn.bfloat16), (0, 2, 1, 3)), (1, 1, batch * 8, 128))
        z = ttnn.to_layout(
            ttnn.reshape(ttnn.to_layout(packed[..., 2048:3072], ttnn.ROW_MAJOR_LAYOUT), (1, 1, batch * 8, 128)),
            ttnn.TILE_LAYOUT,
        )
        y = ttnn.multiply(self.norm(y, "linear_attn.norm"), z, input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
        return ttnn.reshape(y, (1, 1, batch, 1024))
