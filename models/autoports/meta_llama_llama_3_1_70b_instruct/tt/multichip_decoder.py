# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Four-chip tensor-parallel Llama 3.1 70B decoder.

The optimized single-chip decoder is the numerical and operation-policy
baseline.  The selected path presents the four physical chips as a logical
1x4 ring: QKV and packed gate/up are column parallel, output and down are row
parallel, and only the latter two projections all-reduce.  Residuals and norms
are replicated and each device owns two KV heads.

The compiler-captured 2x2 mapping is retained as ``Provenance2DDecoder`` for
comparison.  It shards residual hidden on axis 0 and heads/features on axis 1.

The selected public stacked-layer tensor is ``[1, batch, seq, 8192]`` on every
device.  Both contiguous and genuinely paged KV caches are supported.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

import ttnn
from models.autoports.meta_llama_llama_3_1_70b_instruct.tt.functional_decoder import (
    EMITTED_BATCH,
    EMITTED_CACHE_LENGTH,
    _config_value,
    _state_tensor,
)
from models.autoports.meta_llama_llama_3_1_70b_instruct.tt.optimized_decoder import (
    TILE_SIZE,
    OptimizationConfig,
    OptimizedDecoder,
    _dram_weight_memory_config,
    _width_sharded_l1,
)
from models.common.modules.tt_ccl import TT_CCL, get_tt_ccl

PROVENANCE_MESH_SHAPE = (2, 2)
TARGET_MESH_SHAPE = (1, 4)
HIDDEN_AXIS = 0
HEAD_AXIS = 1
TARGET_TP_DEGREE = 4
PAGED_BLOCK_SIZE = 64


@dataclass(frozen=True)
class MultiChipConfig:
    """Fixed four-Blackhole flat-TP4 policy plus bounded profiling controls."""

    optimized: OptimizationConfig = field(
        default_factory=lambda: OptimizationConfig(
            decode_matmul_strategy="dram_sharded",
            packed_gate_up=True,
            qkv_cores=16,
            prefill_grid=(8, 8),
        )
    )
    topology: object = ttnn.Topology.Ring
    num_links: tuple[int, int] = (2, 2)
    norm_num_links: int = 1
    collective_dtype: object = ttnn.bfloat16
    collective_implementation: str = "decomposed_persistent"
    # Limit the peak-live packed gate/up activation at the advertised 131072
    # context without changing the logical prefill contract.
    prefill_mlp_chunk_size: int = 4096
    # DRAM-sharded decode geometry is deliberately local to the multichip
    # stage: OptimizationConfig remains the immutable single-chip baseline.
    qkv_input_cores: int = 16
    qkv_in0_block_w: int = 16
    qkv_per_core_n: int = 5
    output_input_cores: int = 8
    output_in0_block_w: int = 8
    output_per_core_n: int = 8
    gate_up_input_cores: int = 32
    gate_up_in0_block_w: int = 8
    gate_up_per_core_n: int = 14
    down_input_cores: int = 32
    down_in0_block_w: int = 7
    down_per_core_n: int = 8

    def __post_init__(self):
        if self.prefill_mlp_chunk_size < TILE_SIZE or self.prefill_mlp_chunk_size % TILE_SIZE:
            raise ValueError(f"prefill_mlp_chunk_size must be a positive multiple of {TILE_SIZE}")
        if self.collective_implementation not in {"composite", "decomposed_persistent"}:
            raise ValueError("collective_implementation must be 'composite' or 'decomposed_persistent'")


@dataclass(frozen=True)
class SharedDecoderTensors:
    """Large immutable constants that a future decoder stack must share."""

    rotary_cos: object
    rotary_sin: object
    position_indices: object
    mask_positions: object
    mask_zero: object
    mask_negative_infinity: object


def _mesh_tensor(
    tensor: torch.Tensor,
    mesh_device,
    *,
    mesh_mapper,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        memory_config=memory_config,
        mesh_mapper=mesh_mapper,
    )


def _mapper_2d(mesh_device, dims):
    return ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=list(PROVENANCE_MESH_SHAPE))


def _pack_axis1_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Arrange Q/K/V so an axis-1 shard contains a complete local Q-K-V pack."""

    q_t = q.transpose(0, 1)
    k_t = k.transpose(0, 1)
    v_t = v.transpose(0, 1)
    local_packs = []
    for axis1_rank in range(PROVENANCE_MESH_SHAPE[HEAD_AXIS]):
        local_packs.append(
            torch.cat(
                (
                    q_t.tensor_split(PROVENANCE_MESH_SHAPE[HEAD_AXIS], dim=-1)[axis1_rank],
                    k_t.tensor_split(PROVENANCE_MESH_SHAPE[HEAD_AXIS], dim=-1)[axis1_rank],
                    v_t.tensor_split(PROVENANCE_MESH_SHAPE[HEAD_AXIS], dim=-1)[axis1_rank],
                ),
                dim=-1,
            )
        )
    return torch.cat(local_packs, dim=-1)


def _pack_axis1_gate_up(gate_t: torch.Tensor, up_t: torch.Tensor) -> torch.Tensor:
    """Arrange fused gate/up so each axis-1 shard retains both local operands."""

    local_packs = []
    for axis1_rank in range(PROVENANCE_MESH_SHAPE[HEAD_AXIS]):
        local_packs.append(
            torch.cat(
                (
                    gate_t.tensor_split(PROVENANCE_MESH_SHAPE[HEAD_AXIS], dim=-1)[axis1_rank],
                    up_t.tensor_split(PROVENANCE_MESH_SHAPE[HEAD_AXIS], dim=-1)[axis1_rank],
                ),
                dim=-1,
            )
        )
    return torch.cat(local_packs, dim=-1)


def _check_target_mesh(mesh_device) -> None:
    shape = tuple(mesh_device.shape)
    if mesh_device.get_num_devices() != TARGET_TP_DEGREE or shape != TARGET_MESH_SHAPE:
        raise ValueError(
            f"MultiChipDecoder requires mesh {TARGET_MESH_SHAPE}; "
            f"got shape={shape}, devices={mesh_device.get_num_devices()}"
        )


def _check_provenance_mesh(mesh_device) -> None:
    shape = tuple(mesh_device.shape)
    if mesh_device.get_num_devices() != TARGET_TP_DEGREE or shape != PROVENANCE_MESH_SHAPE:
        raise ValueError(
            f"Provenance2DDecoder requires mesh {PROVENANCE_MESH_SHAPE}; "
            f"got shape={shape}, devices={mesh_device.get_num_devices()}"
        )


def _flat_dram_matmul_program_config(
    *, role: str, m: int, k: int, n: int, input_cores: int, in0_block_w: int, per_core_n: int
):
    """Build a legal DRAM-sharded matmul without coupling A and output shards."""

    if m < 1 or k % TILE_SIZE or n % TILE_SIZE:
        raise ValueError(f"{role}: M must be positive and K={k}, N={n} must be tile aligned")
    k_tiles = k // TILE_SIZE
    n_tiles = n // TILE_SIZE
    if input_cores < 1 or k_tiles % input_cores:
        raise ValueError(f"{role}: input_cores={input_cores} must divide K tiles={k_tiles}")
    input_shard_k_tiles = k_tiles // input_cores
    if in0_block_w < 1 or input_shard_k_tiles % in0_block_w:
        raise ValueError(f"{role}: in0_block_w={in0_block_w} must divide input shard K tiles={input_shard_k_tiles}")
    if per_core_n < 1 or math.ceil(n_tiles / per_core_n) > 110:
        raise ValueError(f"{role}: per_core_n={per_core_n} creates an invalid output grid")
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=math.ceil(m / TILE_SIZE),
        per_core_N=per_core_n,
        fused_activation=None,
    )


def _logical_chunk_ranges(seq_len: int, chunk_size: int) -> tuple[tuple[int, int], ...]:
    """Return aligned internal starts while preserving an arbitrary final tail."""

    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    if chunk_size < TILE_SIZE or chunk_size % TILE_SIZE:
        raise ValueError(f"chunk_size must be a positive multiple of {TILE_SIZE}")
    return tuple((start, min(start + chunk_size, seq_len)) for start in range(0, seq_len, chunk_size))


def prepare_shared_tensors(*, hf_config, mesh_device, batch: int, max_cache_len: int) -> SharedDecoderTensors:
    """Create stack-shared RoPE and position constants on the target mesh."""

    if tuple(mesh_device.shape) == TARGET_MESH_SHAPE:
        _check_target_mesh(mesh_device)
    else:
        _check_provenance_mesh(mesh_device)
    if batch < 1 or batch > EMITTED_BATCH:
        raise ValueError(f"batch must be in [1, {EMITTED_BATCH}], got {batch}")
    if max_cache_len < 1:
        raise ValueError(f"max_cache_len must be positive, got {max_cache_len}")

    hidden_size = int(_config_value(hf_config, "hidden_size"))
    num_heads = int(_config_value(hf_config, "num_attention_heads"))
    head_dim = int(_config_value(hf_config, "head_dim") or hidden_size // num_heads)
    rotary = LlamaRotaryEmbedding(hf_config)
    positions = torch.arange(max_cache_len, dtype=torch.long).unsqueeze(0)
    rope_probe = torch.zeros((1, 1, max_cache_len, head_dim), dtype=torch.bfloat16)
    cos, sin = rotary(rope_probe, positions)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    return SharedDecoderTensors(
        rotary_cos=_mesh_tensor(cos.to(torch.bfloat16).unsqueeze(1), mesh_device, mesh_mapper=replicate),
        rotary_sin=_mesh_tensor(sin.to(torch.bfloat16).unsqueeze(1), mesh_device, mesh_mapper=replicate),
        position_indices=_mesh_tensor(
            torch.arange(max_cache_len, dtype=torch.int32).unsqueeze(1).expand(-1, batch),
            mesh_device,
            mesh_mapper=replicate,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
        ),
        # The multichip decode path uses cur_pos_tensor directly.  Keep these
        # tiny baseline fields because OptimizedDecoder's constructor contract
        # owns them, but do not build a context-sized replicated mask.
        mask_positions=_mesh_tensor(
            torch.zeros((1, 1, 1, 1), dtype=torch.int32),
            mesh_device,
            mesh_mapper=replicate,
            dtype=ttnn.int32,
        ),
        mask_zero=_mesh_tensor(torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16), mesh_device, mesh_mapper=replicate),
        mask_negative_infinity=_mesh_tensor(
            torch.full((1, 1, 1, 1), float("-inf"), dtype=torch.bfloat16),
            mesh_device,
            mesh_mapper=replicate,
        ),
    )


class Provenance2DDecoder(OptimizedDecoder):
    """Compiler-provenance 2D TP implementation derived from OptimizedDecoder."""

    single_chip_baseline = OptimizedDecoder

    def __init__(
        self,
        *,
        multichip_config: MultiChipConfig,
        global_hidden_size: int,
        global_num_heads: int,
        global_num_kv_heads: int,
        global_intermediate_size: int,
        tt_ccl: TT_CCL | None = None,
        **kwargs,
    ):
        self.multichip_config = multichip_config
        self.global_hidden_size = global_hidden_size
        self.global_num_heads = global_num_heads
        self.global_num_kv_heads = global_num_kv_heads
        self.global_intermediate_size = global_intermediate_size
        super().__init__(optimization_config=multichip_config.optimized, **kwargs)

        _check_provenance_mesh(self.mesh_device)
        if self.multichip_config.topology != ttnn.Topology.Linear:
            raise ValueError("The target 2x2 path requires linear per-axis collectives")
        if len(self.multichip_config.num_links) != 2 or any(x < 1 for x in self.multichip_config.num_links):
            raise ValueError("num_links must contain one positive link count per mesh axis")
        if self.global_hidden_size != self.hidden_size * PROVENANCE_MESH_SHAPE[HIDDEN_AXIS]:
            raise ValueError("local/global hidden ownership is inconsistent")
        if self.global_num_heads != self.num_heads * PROVENANCE_MESH_SHAPE[HEAD_AXIS]:
            raise ValueError("local/global query-head ownership is inconsistent")
        if self.global_num_kv_heads != self.num_kv_heads * PROVENANCE_MESH_SHAPE[HEAD_AXIS]:
            raise ValueError("local/global KV-head ownership is inconsistent")
        if self.global_intermediate_size != self.intermediate_size * PROVENANCE_MESH_SHAPE[HEAD_AXIS]:
            raise ValueError("local/global MLP ownership is inconsistent")

        # Persistent semaphore ownership is mesh-wide and must be shared by all
        # decoder layers.  These handles are safe to capture in TTNN traces.
        self.tt_ccl = tt_ccl if tt_ccl is not None else get_tt_ccl(self.mesh_device)
        if self.tt_ccl.mesh_device is not self.mesh_device:
            raise ValueError("tt_ccl must belong to the decoder's mesh_device")

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        batch: int = EMITTED_BATCH,
        max_cache_len: int = EMITTED_CACHE_LENGTH,
        multichip_config: MultiChipConfig | None = None,
        tt_ccl: TT_CCL | None = None,
        shared_tensors: SharedDecoderTensors | None = None,
        **_kwargs,
    ) -> "Provenance2DDecoder":
        _check_provenance_mesh(mesh_device)
        if batch < 1 or batch > EMITTED_BATCH:
            raise ValueError(f"batch must be in [1, {EMITTED_BATCH}], got {batch}")
        if max_cache_len < 1:
            raise ValueError(f"max_cache_len must be positive, got {max_cache_len}")

        policy = multichip_config or MultiChipConfig(topology=ttnn.Topology.Linear)
        if policy.optimized.decode_matmul_strategy != "dram_sharded":
            raise ValueError("The 2D path currently requires DRAM-sharded decode matmuls")

        global_hidden_size = int(_config_value(hf_config, "hidden_size"))
        global_num_heads = int(_config_value(hf_config, "num_attention_heads"))
        global_num_kv_heads = int(_config_value(hf_config, "num_key_value_heads"))
        head_dim = int(_config_value(hf_config, "head_dim") or global_hidden_size // global_num_heads)
        global_intermediate_size = int(_config_value(hf_config, "intermediate_size"))
        rms_norm_eps = float(_config_value(hf_config, "rms_norm_eps"))

        divisors = (
            ("hidden_size", global_hidden_size, PROVENANCE_MESH_SHAPE[HIDDEN_AXIS]),
            ("num_attention_heads", global_num_heads, PROVENANCE_MESH_SHAPE[HEAD_AXIS]),
            ("num_key_value_heads", global_num_kv_heads, PROVENANCE_MESH_SHAPE[HEAD_AXIS]),
            ("intermediate_size", global_intermediate_size, PROVENANCE_MESH_SHAPE[HEAD_AXIS]),
        )
        for name, value, divisor in divisors:
            if value % divisor:
                raise ValueError(f"{name}={value} must divide mesh factor {divisor}")
        if global_hidden_size != global_num_heads * head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")

        q = _state_tensor(state_dict, layer_idx, "self_attn.q_proj.weight").to(torch.bfloat16)
        k = _state_tensor(state_dict, layer_idx, "self_attn.k_proj.weight").to(torch.bfloat16)
        v = _state_tensor(state_dict, layer_idx, "self_attn.v_proj.weight").to(torch.bfloat16)
        output = _state_tensor(state_dict, layer_idx, "self_attn.o_proj.weight").to(torch.bfloat16)
        gate = _state_tensor(state_dict, layer_idx, "mlp.gate_proj.weight").to(torch.bfloat16)
        up = _state_tensor(state_dict, layer_idx, "mlp.up_proj.weight").to(torch.bfloat16)
        down = _state_tensor(state_dict, layer_idx, "mlp.down_proj.weight").to(torch.bfloat16)
        input_norm = _state_tensor(state_dict, layer_idx, "input_layernorm.weight").to(torch.bfloat16)
        post_attention_norm = _state_tensor(state_dict, layer_idx, "post_attention_layernorm.weight").to(torch.bfloat16)

        packed_qkv = _pack_axis1_qkv(q, k, v)
        output_t = output.transpose(0, 1)
        gate_t = gate.transpose(0, 1)
        up_t = up.transpose(0, 1)
        down_t = down.transpose(0, 1)
        decode_packed_gate_up = policy.optimized.packed_gate_up or policy.optimized.packed_decode_gate_up
        packed_gate_up = _pack_axis1_gate_up(gate_t, up_t) if decode_packed_gate_up else None

        local_hidden_size = global_hidden_size // PROVENANCE_MESH_SHAPE[HIDDEN_AXIS]
        local_num_heads = global_num_heads // PROVENANCE_MESH_SHAPE[HEAD_AXIS]
        local_num_kv_heads = global_num_kv_heads // PROVENANCE_MESH_SHAPE[HEAD_AXIS]
        local_intermediate_size = global_intermediate_size // PROVENANCE_MESH_SHAPE[HEAD_AXIS]
        local_qkv_width = (local_num_heads + 2 * local_num_kv_heads) * head_dim
        local_gate_up_width = local_intermediate_size * (2 if decode_packed_gate_up else 1)

        qkv_mapper = _mapper_2d(mesh_device, (0, 1))
        output_mapper = _mapper_2d(mesh_device, (1, 0))
        gate_up_mapper = _mapper_2d(mesh_device, (0, 1))
        down_mapper = _mapper_2d(mesh_device, (1, 0))
        norm_mapper = _mapper_2d(mesh_device, (0, None))
        attention_dtype = policy.optimized.attention_weight_dtype
        gate_up_dtype = policy.optimized.gate_up_weight_dtype
        down_dtype = policy.optimized.down_weight_dtype
        shared = shared_tensors or prepare_shared_tensors(
            hf_config=hf_config,
            mesh_device=mesh_device,
            batch=batch,
            max_cache_len=max_cache_len,
        )

        decode_qkv = _mesh_tensor(
            packed_qkv,
            mesh_device,
            mesh_mapper=qkv_mapper,
            dtype=attention_dtype,
            memory_config=_dram_weight_memory_config(mesh_device, k=local_hidden_size, n=local_qkv_width),
        )
        decode_output = _mesh_tensor(
            output_t,
            mesh_device,
            mesh_mapper=output_mapper,
            dtype=attention_dtype,
            memory_config=_dram_weight_memory_config(mesh_device, k=local_hidden_size, n=local_hidden_size),
        )
        decode_gate = _mesh_tensor(
            packed_gate_up if packed_gate_up is not None else gate_t,
            mesh_device,
            mesh_mapper=gate_up_mapper,
            dtype=gate_up_dtype,
            memory_config=_dram_weight_memory_config(mesh_device, k=local_hidden_size, n=local_gate_up_width),
        )
        decode_up = (
            None
            if packed_gate_up is not None
            else _mesh_tensor(
                up_t,
                mesh_device,
                mesh_mapper=gate_up_mapper,
                dtype=gate_up_dtype,
                memory_config=_dram_weight_memory_config(mesh_device, k=local_hidden_size, n=local_intermediate_size),
            )
        )
        decode_down = _mesh_tensor(
            down_t,
            mesh_device,
            mesh_mapper=down_mapper,
            dtype=down_dtype,
            memory_config=_dram_weight_memory_config(mesh_device, k=local_intermediate_size, n=local_hidden_size),
        )

        # Decode weights are DRAM-sharded.  Independent interleaved copies avoid
        # the known Blackhole corruption when 2-D prefill per_core_N disagrees
        # with an eight-bank DRAM shard width.
        prefill_qkv = _mesh_tensor(packed_qkv, mesh_device, mesh_mapper=qkv_mapper, dtype=attention_dtype)
        prefill_output = _mesh_tensor(output_t, mesh_device, mesh_mapper=output_mapper, dtype=attention_dtype)
        prefill_gate = _mesh_tensor(
            _pack_axis1_gate_up(gate_t, up_t) if policy.optimized.packed_gate_up else gate_t,
            mesh_device,
            mesh_mapper=gate_up_mapper,
            dtype=gate_up_dtype,
        )
        prefill_up = (
            None
            if policy.optimized.packed_gate_up
            else _mesh_tensor(up_t, mesh_device, mesh_mapper=gate_up_mapper, dtype=gate_up_dtype)
        )
        prefill_down = _mesh_tensor(down_t, mesh_device, mesh_mapper=down_mapper, dtype=down_dtype)

        return cls(
            multichip_config=policy,
            prefill_qkv_weight=prefill_qkv,
            prefill_output_weight=prefill_output,
            prefill_gate_weight=prefill_gate,
            prefill_up_weight=prefill_up,
            prefill_down_weight=prefill_down,
            mesh_device=mesh_device,
            layer_idx=layer_idx,
            batch=batch,
            max_cache_len=max_cache_len,
            hidden_size=local_hidden_size,
            num_heads=local_num_heads,
            num_kv_heads=local_num_kv_heads,
            head_dim=head_dim,
            intermediate_size=local_intermediate_size,
            rms_norm_eps=rms_norm_eps,
            global_hidden_size=global_hidden_size,
            global_num_heads=global_num_heads,
            global_num_kv_heads=global_num_kv_heads,
            global_intermediate_size=global_intermediate_size,
            tt_ccl=tt_ccl,
            input_norm=_mesh_tensor(input_norm, mesh_device, mesh_mapper=norm_mapper),
            post_attention_norm=_mesh_tensor(post_attention_norm, mesh_device, mesh_mapper=norm_mapper),
            qkv_weight=decode_qkv,
            output_weight=decode_output,
            gate_weight=decode_gate,
            up_weight=decode_up,
            down_weight=decode_down,
            rotary_cos=shared.rotary_cos,
            rotary_sin=shared.rotary_sin,
            position_indices=shared.position_indices,
            mask_positions=shared.mask_positions,
            mask_zero=shared.mask_zero,
            mask_negative_infinity=shared.mask_negative_infinity,
        )

    @staticmethod
    def mesh_mapper_for_input(mesh_device):
        """Shard hidden on axis 0 and replicate the stacked residual on axis 1."""

        _check_provenance_mesh(mesh_device)
        return _mapper_2d(mesh_device, (3, None))

    @staticmethod
    def mesh_mapper_for_cache(mesh_device):
        """Shard global KV heads on axis 1 and replicate them on axis 0."""

        _check_provenance_mesh(mesh_device)
        return _mapper_2d(mesh_device, (None, 1))

    @staticmethod
    def mesh_mapper_for_page_table(mesh_device):
        """Page tables and current-position metadata are replicated."""

        _check_provenance_mesh(mesh_device)
        return ttnn.ReplicateTensorToMesh(mesh_device)

    def _validate_caches(self, key_cache, value_cache, page_table=None) -> None:
        key_shape = tuple(key_cache.shape)
        value_shape = tuple(value_cache.shape)
        if key_shape != value_shape:
            raise ValueError(f"key/value cache shapes differ: {key_shape} versus {value_shape}")
        if page_table is None:
            expected = (self.batch, self.num_kv_heads, self.max_cache_len, self.head_dim)
            if key_shape != expected:
                raise ValueError(f"contiguous caches must have local shape {expected}, got {key_shape}")
            return
        if len(key_shape) != 4 or key_shape[1] != self.num_kv_heads or key_shape[3] != self.head_dim:
            raise ValueError(
                "paged caches must have local shape "
                f"[blocks, {self.num_kv_heads}, block_size, {self.head_dim}], got {key_shape}"
            )
        if key_shape[2] != PAGED_BLOCK_SIZE:
            raise ValueError(f"paged block size must be {PAGED_BLOCK_SIZE}, got {key_shape[2]}")
        page_shape = tuple(page_table.shape)
        if len(page_shape) != 2 or page_shape[0] != self.batch:
            raise ValueError(f"page_table must have local shape [{self.batch}, pages], got {page_shape}")

    def _all_reduce_partial(self, partial, *, axis: int, memory_config):
        payload = partial
        if self.multichip_config.collective_dtype != ttnn.bfloat16:
            payload = ttnn.typecast(payload, self.multichip_config.collective_dtype, memory_config=memory_config)
        reduced = ttnn.experimental.all_reduce_async(
            payload,
            cluster_axis=axis,
            mesh_device=self.mesh_device,
            barrier_semaphores=self.tt_ccl.barrier_semaphore_handles[axis],
            rs_global_semaphores=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=axis),
            ag_global_semaphores=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=axis),
            num_links=self.multichip_config.num_links[axis],
            math_op=ttnn.ReduceType.Sum,
            memory_config=memory_config,
            topology=self.multichip_config.topology,
        )
        if self.multichip_config.collective_dtype != ttnn.bfloat16:
            reduced = ttnn.typecast(reduced, ttnn.bfloat16, memory_config=memory_config)
        return reduced

    def _distributed_norm(self, hidden_states, weight):
        # Interleaved stats keep one correctness path for arbitrary prefill
        # lengths and decode; later conversions are local and traceable.
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        stats = ttnn.rms_norm_pre_all_gather(
            hidden_states,
            compute_kernel_config=self.norm_compute_kernel,
            dtype=ttnn.bfloat16,
        )
        batch = hidden_states.shape[0]
        stats = ttnn.reshape(stats, ttnn.Shape((batch, 1, hidden_states.shape[-2], 32)))
        gathered = ttnn.experimental.all_gather_async(
            stats,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(HIDDEN_AXIS),
            num_links=self.multichip_config.norm_num_links,
            cluster_axis=HIDDEN_AXIS,
            topology=self.multichip_config.topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(HIDDEN_AXIS),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        stats.deallocate(True)
        result = ttnn.rms_norm_post_all_gather(
            hidden_states,
            gathered,
            epsilon=self.rms_norm_eps,
            weight=weight,
            compute_kernel_config=self.norm_compute_kernel,
        )
        gathered.deallocate(True)
        return result

    def _fill_cache(self, cache, values, *, page_table, seq_len: int) -> None:
        cache_values = values if values.dtype == cache.dtype else ttnn.typecast(values, cache.dtype)
        for user_id in range(self.batch):
            user = ttnn.slice(
                cache_values,
                [user_id, 0, 0, 0],
                [user_id + 1, self.num_kv_heads, seq_len, self.head_dim],
                [1, 1, 1, 1],
            )
            if page_table is None:
                ttnn.fill_cache(cache, user, user_id)
            else:
                ttnn.experimental.paged_fill_cache(cache, user, page_table, batch_idx=user_id)

    def _mlp_prefill_2d(self, hidden_states, seq_len: int):
        if self.optimization_config.packed_gate_up:
            gate_up = self._prefill_linear(
                hidden_states,
                self.prefill_gate_weight,
                role="gate_up_packed",
                seq_len=seq_len,
                k=self.hidden_size,
                n=2 * self.intermediate_size,
                compute_kernel=self.gate_up_compute_kernel,
            )
            gate_up = self._all_reduce_partial(gate_up, axis=HIDDEN_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            gate = ttnn.slice(
                gate_up,
                [0, 0, 0, 0],
                [1, self.batch, seq_len, self.intermediate_size],
                [1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            up = ttnn.slice(
                gate_up,
                [0, 0, 0, self.intermediate_size],
                [1, self.batch, seq_len, 2 * self.intermediate_size],
                [1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            gate = self._prefill_linear(
                hidden_states,
                self.prefill_gate_weight,
                role="gate",
                seq_len=seq_len,
                k=self.hidden_size,
                n=self.intermediate_size,
                compute_kernel=self.gate_up_compute_kernel,
            )
            up = self._prefill_linear(
                hidden_states,
                self.prefill_up_weight,
                role="up",
                seq_len=seq_len,
                k=self.hidden_size,
                n=self.intermediate_size,
                compute_kernel=self.gate_up_compute_kernel,
            )
            gate = self._all_reduce_partial(gate, axis=HIDDEN_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            up = self._all_reduce_partial(up, axis=HIDDEN_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gated = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        down = self._prefill_linear(
            gated,
            self.prefill_down_weight,
            role="down",
            seq_len=seq_len,
            k=self.intermediate_size,
            n=self.hidden_size,
            compute_kernel=self.down_compute_kernel,
        )
        return self._all_reduce_partial(down, axis=HEAD_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _mlp_decode_2d(self, hidden_states):
        mlp_input = ttnn.to_memory_config(hidden_states, self.gate_up_input_mem_config)
        if self.decode_packed_gate_up:
            gate_up = ttnn.linear(
                mlp_input,
                self.gate_weight,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=self.gate_up_decode_program_config,
                compute_kernel_config=self.gate_up_compute_kernel,
            )
            gate_up = ttnn.to_memory_config(gate_up, ttnn.L1_MEMORY_CONFIG)
            gate_up = self._all_reduce_partial(gate_up, axis=HIDDEN_AXIS, memory_config=ttnn.L1_MEMORY_CONFIG)
            gate = ttnn.slice(
                gate_up,
                [0, 0, 0, 0],
                [1, 1, self.batch, self.intermediate_size],
                [1, 1, 1, 1],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            up = ttnn.slice(
                gate_up,
                [0, 0, 0, self.intermediate_size],
                [1, 1, self.batch, 2 * self.intermediate_size],
                [1, 1, 1, 1],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        else:
            gate = ttnn.linear(
                mlp_input,
                self.gate_weight,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=self.gate_up_decode_program_config,
                compute_kernel_config=self.gate_up_compute_kernel,
            )
            up = ttnn.linear(
                mlp_input,
                self.up_weight,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=self.gate_up_decode_program_config,
                compute_kernel_config=self.gate_up_compute_kernel,
            )
            gate = self._all_reduce_partial(
                ttnn.to_memory_config(gate, ttnn.L1_MEMORY_CONFIG),
                axis=HIDDEN_AXIS,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            up = self._all_reduce_partial(
                ttnn.to_memory_config(up, ttnn.L1_MEMORY_CONFIG),
                axis=HIDDEN_AXIS,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        gated = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        gated = ttnn.to_memory_config(gated, self.down_input_mem_config)
        down = ttnn.linear(
            gated,
            self.down_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.down_decode_program_config,
            compute_kernel_config=self.down_compute_kernel,
        )
        down = ttnn.to_memory_config(down, self.residual_mem_config)
        return self._all_reduce_partial(down, axis=HEAD_AXIS, memory_config=self.residual_mem_config)

    def prefill_forward(self, hidden_states, key_cache, value_cache, *, page_table=None):
        seq_len = self._validate_hidden_states(hidden_states)
        self._validate_caches(key_cache, value_cache, page_table)
        if page_table is not None and math.ceil(seq_len / PAGED_BLOCK_SIZE) > page_table.shape[1]:
            raise ValueError("page_table does not cover the logical prefill length")

        residual = hidden_states
        normed = self._distributed_norm(hidden_states, self.input_norm)
        fused_qkv = self._prefill_linear(
            normed,
            self.prefill_qkv_weight,
            role="qkv",
            seq_len=seq_len,
            k=self.hidden_size,
            n=(self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            compute_kernel=self.attention_compute_kernel,
        )
        fused_qkv = self._all_reduce_partial(fused_qkv, axis=HIDDEN_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        fused_qkv = ttnn.reshape(
            fused_qkv,
            [self.batch, seq_len, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim],
        )
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused_qkv,
            None,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cos = ttnn.slice(self.rotary_cos, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim], [1, 1, 1, 1])
        sin = ttnn.slice(self.rotary_sin, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim], [1, 1, 1, 1])
        query = ttnn.experimental.rotary_embedding(query, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.experimental.rotary_embedding(key, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        query = ttnn.slice(
            query,
            [0, 0, 0, 0],
            [self.batch, self.num_heads, seq_len, self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.slice(
            key,
            [0, 0, 0, 0],
            [self.batch, self.num_kv_heads, seq_len, self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._fill_cache(key_cache, key, page_table=page_table, seq_len=seq_len)
        self._fill_cache(value_cache, value, page_table=page_table, seq_len=seq_len)

        attention = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.prefill_sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel,
        )
        attention = ttnn.transformer.concatenate_heads(attention, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.reshape(attention, [1, self.batch, seq_len, self.hidden_size])
        attention = self._prefill_linear(
            attention,
            self.prefill_output_weight,
            role="output",
            seq_len=seq_len,
            k=self.hidden_size,
            n=self.hidden_size,
            compute_kernel=self.attention_compute_kernel,
        )
        attention = self._all_reduce_partial(attention, axis=HEAD_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.add(
            residual,
            attention,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        residual = hidden_states
        hidden_states = self._distributed_norm(hidden_states, self.post_attention_norm)
        hidden_states = self._mlp_prefill_2d(hidden_states, seq_len)
        return ttnn.add(
            residual,
            hidden_states,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def decode_forward(self, hidden_states, key_cache, value_cache, *, current_pos: int, page_table=None):
        self._validate_hidden_states(hidden_states, expected_seq_len=1)
        self._validate_caches(key_cache, value_cache, page_table)
        if current_pos < 0 or current_pos >= self.max_cache_len:
            raise ValueError(f"current_pos must be in [0, {self.max_cache_len}), got {current_pos}")
        if page_table is not None and current_pos // PAGED_BLOCK_SIZE >= page_table.shape[1]:
            raise ValueError("page_table does not cover current_pos")

        hidden_states = ttnn.reshape(hidden_states, [1, 1, self.batch, self.hidden_size])
        hidden_states = ttnn.to_memory_config(hidden_states, self.residual_mem_config)
        residual = hidden_states
        normed = self._distributed_norm(hidden_states, self.input_norm)
        normed = ttnn.to_memory_config(normed, self.qkv_input_mem_config)
        fused_qkv = ttnn.linear(
            normed,
            self.qkv_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.qkv_decode_program_config,
            compute_kernel_config=self.attention_compute_kernel,
        )
        fused_qkv = ttnn.sharded_to_interleaved(fused_qkv, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
        fused_qkv = self._all_reduce_partial(fused_qkv, axis=HIDDEN_AXIS, memory_config=ttnn.L1_MEMORY_CONFIG)
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused_qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            overlap_qk_coregrid=None,
            memory_config=self.decode_attention_mem_config,
        )
        cos = ttnn.slice(
            self.rotary_cos,
            [0, 0, current_pos, 0],
            [1, 1, current_pos + 1, self.head_dim],
            [1, 1, 1, 1],
        )
        sin = ttnn.slice(
            self.rotary_sin,
            [0, 0, current_pos, 0],
            [1, 1, current_pos + 1, self.head_dim],
            [1, 1, 1, 1],
        )
        query = ttnn.experimental.rotary_embedding(query, cos, sin, 0, memory_config=self.decode_attention_mem_config)
        key = ttnn.experimental.rotary_embedding(key, cos, sin, 0, memory_config=self.decode_cache_mem_config)
        update_indices = ttnn.slice(
            self.position_indices,
            [current_pos, 0],
            [current_pos + 1, self.batch],
            [1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        update_indices = ttnn.reshape(update_indices, [self.batch])
        ttnn.experimental.paged_update_cache(
            key_cache,
            key,
            update_idxs_tensor=update_indices,
            share_cache=False,
            page_table=page_table,
        )
        ttnn.experimental.paged_update_cache(
            value_cache,
            value,
            update_idxs_tensor=update_indices,
            share_cache=False,
            page_table=page_table,
        )
        query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
        if page_table is None:
            attention = ttnn.transformer.scaled_dot_product_attention_decode(
                query,
                key_cache,
                value_cache,
                is_causal=True,
                cur_pos_tensor=update_indices,
                scale=self.scale,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=self.decode_sdpa_program_config,
                compute_kernel_config=self.decode_sdpa_compute_kernel,
            )
        else:
            attention = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                query,
                key_cache,
                value_cache,
                page_table_tensor=page_table,
                cur_pos_tensor=update_indices,
                scale=self.scale,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=self.decode_sdpa_program_config,
                compute_kernel_config=self.decode_sdpa_compute_kernel,
            )
        attention = ttnn.to_memory_config(attention, self.decode_attention_mem_config)
        attention = ttnn.experimental.nlp_concat_heads_decode(
            attention,
            num_heads=self.num_heads,
            sub_core_grids=self.decode_compute_core_grid,
        )
        attention = ttnn.slice(
            attention,
            [0, 0, 0, 0],
            [1, 1, self.batch, self.hidden_size],
            [1, 1, 1, 1],
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        attention = ttnn.to_memory_config(attention, self.output_input_mem_config)
        attention = ttnn.linear(
            attention,
            self.output_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.output_decode_program_config,
            compute_kernel_config=self.attention_compute_kernel,
        )
        attention = ttnn.to_memory_config(attention, self.residual_mem_config)
        attention = self._all_reduce_partial(attention, axis=HEAD_AXIS, memory_config=self.residual_mem_config)
        hidden_states = ttnn.add(
            residual,
            attention,
            dtype=ttnn.bfloat16,
            memory_config=self.residual_mem_config,
        )
        residual = hidden_states
        hidden_states = self._distributed_norm(hidden_states, self.post_attention_norm)
        hidden_states = self._mlp_decode_2d(hidden_states)
        hidden_states = ttnn.add(
            residual,
            hidden_states,
            dtype=ttnn.bfloat16,
            memory_config=self.residual_mem_config,
        )
        return ttnn.reshape(hidden_states, [1, self.batch, 1, self.hidden_size])

    def forward(
        self,
        hidden_states,
        key_cache,
        value_cache,
        *,
        mode: str,
        current_pos: int | None = None,
        page_table=None,
    ):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, key_cache, value_cache, page_table=page_table)
        if mode == "decode":
            if current_pos is None:
                raise ValueError("decode mode requires current_pos")
            return self.decode_forward(
                hidden_states,
                key_cache,
                value_cache,
                current_pos=current_pos,
                page_table=page_table,
            )
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")


def _pack_flat_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Arrange a complete Q/K/V group for each flattened mesh rank."""

    q_t = q.transpose(0, 1)
    k_t = k.transpose(0, 1)
    v_t = v.transpose(0, 1)
    return torch.cat(
        [
            torch.cat(
                (
                    q_t.tensor_split(TARGET_TP_DEGREE, dim=-1)[rank],
                    k_t.tensor_split(TARGET_TP_DEGREE, dim=-1)[rank],
                    v_t.tensor_split(TARGET_TP_DEGREE, dim=-1)[rank],
                ),
                dim=-1,
            )
            for rank in range(TARGET_TP_DEGREE)
        ],
        dim=-1,
    )


def _pack_flat_gate_up(gate_t: torch.Tensor, up_t: torch.Tensor) -> torch.Tensor:
    """Arrange local gate/up pairs for a flattened TP4 column projection."""

    return torch.cat(
        [
            torch.cat(
                (
                    gate_t.tensor_split(TARGET_TP_DEGREE, dim=-1)[rank],
                    up_t.tensor_split(TARGET_TP_DEGREE, dim=-1)[rank],
                ),
                dim=-1,
            )
            for rank in range(TARGET_TP_DEGREE)
        ],
        dim=-1,
    )


class MultiChipDecoder(OptimizedDecoder):
    """Selected flat TP4 layer on the physical 2x2 four-Blackhole mesh.

    The 2D compiler mapping remains available as :class:`Provenance2DDecoder`
    for provenance and profiler comparisons.  Measurement showed that decode
    is communication-launch bound, so the selected stack contract flattens the
    same four devices: residuals and norms are replicated; QKV and gate/up are
    column parallel; output and down are row parallel.  That reduces a packed
    layer from four projection all-reduces plus two norm gathers to two
    full-mesh all-reduces and gives every device two KV heads.
    """

    single_chip_baseline = OptimizedDecoder

    def __init__(
        self,
        *,
        multichip_config: MultiChipConfig,
        global_num_heads: int,
        global_num_kv_heads: int,
        global_intermediate_size: int,
        tt_ccl: TT_CCL | None = None,
        **kwargs,
    ):
        self.multichip_config = multichip_config
        self.global_num_heads = global_num_heads
        self.global_num_kv_heads = global_num_kv_heads
        self.global_intermediate_size = global_intermediate_size
        super().__init__(optimization_config=multichip_config.optimized, **kwargs)
        _check_target_mesh(self.mesh_device)
        if self.global_num_heads != self.num_heads * TARGET_TP_DEGREE:
            raise ValueError("local/global query-head ownership is inconsistent")
        if self.global_num_kv_heads != self.num_kv_heads * TARGET_TP_DEGREE:
            raise ValueError("local/global KV-head ownership is inconsistent")
        if self.global_intermediate_size != self.intermediate_size * TARGET_TP_DEGREE:
            raise ValueError("local/global MLP ownership is inconsistent")
        if self.multichip_config.topology not in (ttnn.Topology.Linear, ttnn.Topology.Ring):
            raise ValueError("flat TP4 requires linear or ring topology")
        if (
            self.multichip_config.prefill_mlp_chunk_size < TILE_SIZE
            or self.multichip_config.prefill_mlp_chunk_size % TILE_SIZE
        ):
            raise ValueError(f"prefill_mlp_chunk_size must be a positive multiple of {TILE_SIZE}")

        self.local_attention_width = self.num_heads * self.head_dim
        padded_rows = TILE_SIZE * math.ceil(self.batch / TILE_SIZE)
        device_grid = self.mesh_device.compute_with_storage_grid_size()
        # The base class supplies the operation and numerical policy.  These
        # TP4-local overrides separate activation and output shard counts,
        # which the DRAM-sharded kernel supports but the baseline helper does
        # not express.
        self.qkv_input_mem_config = _width_sharded_l1(
            rows=padded_rows,
            width=self.hidden_size,
            cores=self.multichip_config.qkv_input_cores,
            device_grid=device_grid,
        )
        self.qkv_decode_program_config = _flat_dram_matmul_program_config(
            role="flat_tp4_qkv",
            m=padded_rows,
            k=self.hidden_size,
            n=(self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            input_cores=self.multichip_config.qkv_input_cores,
            in0_block_w=self.multichip_config.qkv_in0_block_w,
            per_core_n=self.multichip_config.qkv_per_core_n,
        )
        self.output_input_mem_config = _width_sharded_l1(
            rows=padded_rows,
            width=self.local_attention_width,
            cores=self.multichip_config.output_input_cores,
            device_grid=device_grid,
        )
        self.output_decode_program_config = _flat_dram_matmul_program_config(
            role="flat_tp4_output",
            m=padded_rows,
            k=self.local_attention_width,
            n=self.hidden_size,
            input_cores=self.multichip_config.output_input_cores,
            in0_block_w=self.multichip_config.output_in0_block_w,
            per_core_n=self.multichip_config.output_per_core_n,
        )
        self.gate_up_input_mem_config = _width_sharded_l1(
            rows=padded_rows,
            width=self.hidden_size,
            cores=self.multichip_config.gate_up_input_cores,
            device_grid=device_grid,
        )
        self.gate_up_decode_program_config = _flat_dram_matmul_program_config(
            role="flat_tp4_gate_up",
            m=padded_rows,
            k=self.hidden_size,
            n=2 * self.intermediate_size,
            input_cores=self.multichip_config.gate_up_input_cores,
            in0_block_w=self.multichip_config.gate_up_in0_block_w,
            per_core_n=self.multichip_config.gate_up_per_core_n,
        )
        self.down_input_mem_config = _width_sharded_l1(
            rows=padded_rows,
            width=self.intermediate_size,
            cores=self.multichip_config.down_input_cores,
            device_grid=device_grid,
        )
        self.down_decode_program_config = _flat_dram_matmul_program_config(
            role="flat_tp4_down",
            m=padded_rows,
            k=self.intermediate_size,
            n=self.hidden_size,
            input_cores=self.multichip_config.down_input_cores,
            in0_block_w=self.multichip_config.down_in0_block_w,
            per_core_n=self.multichip_config.down_per_core_n,
        )
        # Metadata-only 2-D views make decode RoPE lookup depend on a
        # refreshable device position tensor instead of a Python slice baked
        # into the trace.
        self.decode_rotary_cos = ttnn.reshape(self.rotary_cos, [self.max_cache_len, self.head_dim])
        self.decode_rotary_sin = ttnn.reshape(self.rotary_sin, [self.max_cache_len, self.head_dim])
        self.tt_ccl = tt_ccl if tt_ccl is not None else get_tt_ccl(self.mesh_device)
        if self.tt_ccl.mesh_device is not self.mesh_device:
            raise ValueError("tt_ccl must belong to the decoder's mesh_device")
        self._barrier_semaphores = self.tt_ccl.barrier_semaphore_handles[HEAD_AXIS]
        self._decode_collective_buffers = None
        if self.multichip_config.collective_implementation == "decomposed_persistent":
            collective_dtype = self.multichip_config.collective_dtype
            full_shape = ttnn.Shape([1, 1, self.batch, self.hidden_size])
            shard_shape = ttnn.Shape([1, 1, self.batch, self.hidden_size // TARGET_TP_DEGREE])
            self._decode_collective_buffers = (
                ttnn.allocate_tensor_on_device(
                    full_shape,
                    collective_dtype,
                    ttnn.TILE_LAYOUT,
                    self.mesh_device,
                    ttnn.L1_MEMORY_CONFIG,
                ),
                ttnn.allocate_tensor_on_device(
                    shard_shape,
                    collective_dtype,
                    ttnn.TILE_LAYOUT,
                    self.mesh_device,
                    ttnn.L1_MEMORY_CONFIG,
                ),
                ttnn.allocate_tensor_on_device(
                    full_shape,
                    collective_dtype,
                    ttnn.TILE_LAYOUT,
                    self.mesh_device,
                    ttnn.L1_MEMORY_CONFIG,
                ),
            )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        batch: int = EMITTED_BATCH,
        max_cache_len: int = EMITTED_CACHE_LENGTH,
        multichip_config: MultiChipConfig | None = None,
        tt_ccl: TT_CCL | None = None,
        shared_tensors: SharedDecoderTensors | None = None,
        **_kwargs,
    ) -> "MultiChipDecoder":
        _check_target_mesh(mesh_device)
        if batch < 1 or batch > EMITTED_BATCH:
            raise ValueError(f"batch must be in [1, {EMITTED_BATCH}], got {batch}")
        if max_cache_len < 1:
            raise ValueError(f"max_cache_len must be positive, got {max_cache_len}")
        policy = multichip_config or MultiChipConfig()
        if policy.optimized.decode_matmul_strategy != "dram_sharded":
            raise ValueError("flat TP4 currently requires DRAM-sharded decode matmuls")
        if not (policy.optimized.packed_gate_up or policy.optimized.packed_decode_gate_up):
            raise ValueError("flat TP4 requires packed gate/up to keep the two-collective decode graph")

        hidden_size = int(_config_value(hf_config, "hidden_size"))
        global_num_heads = int(_config_value(hf_config, "num_attention_heads"))
        global_num_kv_heads = int(_config_value(hf_config, "num_key_value_heads"))
        head_dim = int(_config_value(hf_config, "head_dim") or hidden_size // global_num_heads)
        global_intermediate_size = int(_config_value(hf_config, "intermediate_size"))
        rms_norm_eps = float(_config_value(hf_config, "rms_norm_eps"))
        for name, value in (
            ("num_attention_heads", global_num_heads),
            ("num_key_value_heads", global_num_kv_heads),
            ("intermediate_size", global_intermediate_size),
        ):
            if value % TARGET_TP_DEGREE:
                raise ValueError(f"{name}={value} must divide flat TP={TARGET_TP_DEGREE}")
        if hidden_size != global_num_heads * head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")

        q = _state_tensor(state_dict, layer_idx, "self_attn.q_proj.weight").to(torch.bfloat16)
        k = _state_tensor(state_dict, layer_idx, "self_attn.k_proj.weight").to(torch.bfloat16)
        v = _state_tensor(state_dict, layer_idx, "self_attn.v_proj.weight").to(torch.bfloat16)
        output_t = _state_tensor(state_dict, layer_idx, "self_attn.o_proj.weight").to(torch.bfloat16).transpose(0, 1)
        gate_t = _state_tensor(state_dict, layer_idx, "mlp.gate_proj.weight").to(torch.bfloat16).transpose(0, 1)
        up_t = _state_tensor(state_dict, layer_idx, "mlp.up_proj.weight").to(torch.bfloat16).transpose(0, 1)
        down_t = _state_tensor(state_dict, layer_idx, "mlp.down_proj.weight").to(torch.bfloat16).transpose(0, 1)
        input_norm = _state_tensor(state_dict, layer_idx, "input_layernorm.weight").to(torch.bfloat16)
        post_attention_norm = _state_tensor(state_dict, layer_idx, "post_attention_layernorm.weight").to(torch.bfloat16)

        packed_qkv = _pack_flat_qkv(q, k, v)
        packed_gate_up = _pack_flat_gate_up(gate_t, up_t)
        local_num_heads = global_num_heads // TARGET_TP_DEGREE
        local_num_kv_heads = global_num_kv_heads // TARGET_TP_DEGREE
        local_intermediate_size = global_intermediate_size // TARGET_TP_DEGREE
        local_attention_width = local_num_heads * head_dim
        local_qkv_width = (local_num_heads + 2 * local_num_kv_heads) * head_dim
        local_gate_up_width = 2 * local_intermediate_size
        shard_last = ttnn.ShardTensorToMesh(mesh_device, dim=1)
        shard_first = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        shared = shared_tensors or prepare_shared_tensors(
            hf_config=hf_config,
            mesh_device=mesh_device,
            batch=batch,
            max_cache_len=max_cache_len,
        )
        attention_dtype = policy.optimized.attention_weight_dtype
        gate_up_dtype = policy.optimized.gate_up_weight_dtype
        down_dtype = policy.optimized.down_weight_dtype

        qkv_weight = _mesh_tensor(
            packed_qkv,
            mesh_device,
            mesh_mapper=shard_last,
            dtype=attention_dtype,
            memory_config=_dram_weight_memory_config(mesh_device, k=hidden_size, n=local_qkv_width),
        )
        output_weight = _mesh_tensor(
            output_t,
            mesh_device,
            mesh_mapper=shard_first,
            dtype=attention_dtype,
            memory_config=_dram_weight_memory_config(mesh_device, k=local_attention_width, n=hidden_size),
        )
        gate_weight = _mesh_tensor(
            packed_gate_up,
            mesh_device,
            mesh_mapper=shard_last,
            dtype=gate_up_dtype,
            memory_config=_dram_weight_memory_config(mesh_device, k=hidden_size, n=local_gate_up_width),
        )
        down_weight = _mesh_tensor(
            down_t,
            mesh_device,
            mesh_mapper=shard_first,
            dtype=down_dtype,
            memory_config=_dram_weight_memory_config(mesh_device, k=local_intermediate_size, n=hidden_size),
        )

        # The selected 8x8 prefill grid makes each per_core_N equal the eight-
        # bank DRAM shard width for every local projection.  The same physical
        # BFP4 weights can therefore serve prefill and decode without the
        # baseline's duplicate interleaved copies.  This saves 9.63 GB/device
        # in an 80-layer stack and is required for full-context KV headroom.
        prefill_qkv = qkv_weight
        prefill_output = output_weight
        prefill_gate = gate_weight
        prefill_down = down_weight

        return cls(
            multichip_config=policy,
            prefill_qkv_weight=prefill_qkv,
            prefill_output_weight=prefill_output,
            prefill_gate_weight=prefill_gate,
            prefill_up_weight=None,
            prefill_down_weight=prefill_down,
            mesh_device=mesh_device,
            layer_idx=layer_idx,
            batch=batch,
            max_cache_len=max_cache_len,
            hidden_size=hidden_size,
            num_heads=local_num_heads,
            num_kv_heads=local_num_kv_heads,
            head_dim=head_dim,
            intermediate_size=local_intermediate_size,
            rms_norm_eps=rms_norm_eps,
            global_num_heads=global_num_heads,
            global_num_kv_heads=global_num_kv_heads,
            global_intermediate_size=global_intermediate_size,
            tt_ccl=tt_ccl,
            input_norm=_mesh_tensor(input_norm, mesh_device, mesh_mapper=replicate),
            post_attention_norm=_mesh_tensor(post_attention_norm, mesh_device, mesh_mapper=replicate),
            qkv_weight=qkv_weight,
            output_weight=output_weight,
            gate_weight=gate_weight,
            up_weight=None,
            down_weight=down_weight,
            rotary_cos=shared.rotary_cos,
            rotary_sin=shared.rotary_sin,
            position_indices=shared.position_indices,
            mask_positions=shared.mask_positions,
            mask_zero=shared.mask_zero,
            mask_negative_infinity=shared.mask_negative_infinity,
        )

    @staticmethod
    def mesh_mapper_for_input(mesh_device):
        """Replicate the full stacked residual across flattened TP4."""

        _check_target_mesh(mesh_device)
        return ttnn.ReplicateTensorToMesh(mesh_device)

    @staticmethod
    def mesh_mapper_for_cache(mesh_device):
        """Shard eight global KV heads into two heads on each device."""

        _check_target_mesh(mesh_device)
        return ttnn.ShardTensorToMesh(mesh_device, dim=1)

    @staticmethod
    def mesh_mapper_for_page_table(mesh_device):
        _check_target_mesh(mesh_device)
        return ttnn.ReplicateTensorToMesh(mesh_device)

    def _validate_caches(self, key_cache, value_cache, page_table=None) -> None:
        key_shape = tuple(key_cache.shape)
        if key_shape != tuple(value_cache.shape):
            raise ValueError("key/value cache shapes differ")
        if page_table is None:
            expected = (self.batch, self.num_kv_heads, self.max_cache_len, self.head_dim)
            if key_shape != expected:
                raise ValueError(f"contiguous caches must have local shape {expected}, got {key_shape}")
            return
        if len(key_shape) != 4 or key_shape[1] != self.num_kv_heads or key_shape[3] != self.head_dim:
            raise ValueError(
                f"paged caches must have local shape [blocks, {self.num_kv_heads}, block, {self.head_dim}]"
            )
        if key_shape[2] != PAGED_BLOCK_SIZE:
            raise ValueError(f"paged block size must be {PAGED_BLOCK_SIZE}, got {key_shape[2]}")
        if len(page_table.shape) != 2 or page_table.shape[0] != self.batch:
            raise ValueError(f"page_table must have local shape [{self.batch}, pages]")

    def _all_reduce_flat(self, partial, *, memory_config):
        payload = partial
        if self.multichip_config.collective_dtype != ttnn.bfloat16:
            payload = ttnn.typecast(payload, self.multichip_config.collective_dtype, memory_config=memory_config)
        decode_shape = (1, 1, self.batch, self.hidden_size)
        if (
            self.multichip_config.collective_implementation == "decomposed_persistent"
            and tuple(payload.shape) == decode_shape
        ):
            persistent_intermediate, persistent_scattered, persistent_gathered = self._decode_collective_buffers
            interleaved = ttnn.to_memory_config(payload, ttnn.L1_MEMORY_CONFIG)
            scattered = ttnn.experimental.reduce_scatter_minimal_async(
                interleaved,
                persistent_output_buffers=[persistent_intermediate, persistent_scattered],
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=HEAD_AXIS),
                barrier_semaphore=self._barrier_semaphores[0],
                num_links=self.multichip_config.num_links[HEAD_AXIS],
                memory_config=ttnn.L1_MEMORY_CONFIG,
                intermediate_memory_config=ttnn.L1_MEMORY_CONFIG,
                topology=self.multichip_config.topology,
                cluster_axis=HEAD_AXIS,
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
            reduced = ttnn.experimental.all_gather_async(
                scattered,
                persistent_output_buffer=persistent_gathered,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=HEAD_AXIS),
                barrier_semaphore=self._barrier_semaphores[1],
                num_links=self.multichip_config.num_links[HEAD_AXIS],
                memory_config=ttnn.L1_MEMORY_CONFIG,
                topology=self.multichip_config.topology,
                cluster_axis=HEAD_AXIS,
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
            reduced = ttnn.to_memory_config(reduced, memory_config)
            if self.multichip_config.collective_dtype != ttnn.bfloat16:
                reduced = ttnn.typecast(reduced, ttnn.bfloat16, memory_config=memory_config)
            return reduced
        reduced = ttnn.experimental.all_reduce_async(
            payload,
            cluster_axis=HEAD_AXIS,
            mesh_device=self.mesh_device,
            barrier_semaphores=self._barrier_semaphores,
            rs_global_semaphores=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=HEAD_AXIS),
            ag_global_semaphores=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=HEAD_AXIS),
            math_op=ttnn.ReduceType.Sum,
            num_links=self.multichip_config.num_links[HEAD_AXIS],
            memory_config=memory_config,
            topology=self.multichip_config.topology,
        )
        if self.multichip_config.collective_dtype != ttnn.bfloat16:
            reduced = ttnn.typecast(reduced, ttnn.bfloat16, memory_config=memory_config)
        return reduced

    def _fill_cache(self, cache, values, *, page_table, seq_len: int) -> None:
        cache_values = values if values.dtype == cache.dtype else ttnn.typecast(values, cache.dtype)
        for user_id in range(self.batch):
            user = ttnn.slice(
                cache_values,
                [user_id, 0, 0, 0],
                [user_id + 1, self.num_kv_heads, seq_len, self.head_dim],
                [1, 1, 1, 1],
            )
            if page_table is None:
                ttnn.fill_cache(cache, user, user_id)
            else:
                ttnn.experimental.paged_fill_cache(cache, user, page_table, batch_idx=user_id)

    def _mlp_prefill_flat(self, hidden_states, seq_len: int):
        chunk_size = self.multichip_config.prefill_mlp_chunk_size
        if seq_len <= chunk_size:
            partial = super()._mlp_prefill(hidden_states, seq_len)
            reduced = self._all_reduce_flat(partial, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(partial)
            return reduced

        # Packed gate/up is the largest live activation at long context.  Run
        # it in logical (not padded-public) chunks and concatenate the reduced
        # hidden outputs.  The last chunk may be any positive length.
        reduced_chunks = []
        for start, end in _logical_chunk_ranges(seq_len, chunk_size):
            chunk = ttnn.slice(
                hidden_states,
                [0, 0, start, 0],
                [1, self.batch, end, self.hidden_size],
                [1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            partial = super()._mlp_prefill(chunk, end - start)
            reduced_chunks.append(self._all_reduce_flat(partial, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            ttnn.deallocate(chunk)
            ttnn.deallocate(partial)
        reduced = ttnn.concat(reduced_chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for chunk in reduced_chunks:
            ttnn.deallocate(chunk)
        return reduced

    def _mlp_decode_flat(self, hidden_states):
        return self._all_reduce_flat(super()._mlp_decode(hidden_states), memory_config=self.residual_mem_config)

    def prefill_forward(self, hidden_states, key_cache, value_cache, *, page_table=None):
        seq_len = self._validate_hidden_states(hidden_states)
        self._validate_caches(key_cache, value_cache, page_table)
        if page_table is not None and math.ceil(seq_len / PAGED_BLOCK_SIZE) > page_table.shape[1]:
            raise ValueError("page_table does not cover the logical prefill length")
        residual = hidden_states
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.input_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.norm_compute_kernel,
        )
        fused_qkv = self._prefill_linear(
            normed,
            self.prefill_qkv_weight,
            role="flat_qkv",
            seq_len=seq_len,
            k=self.hidden_size,
            n=(self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            compute_kernel=self.attention_compute_kernel,
        )
        fused_qkv = ttnn.reshape(
            fused_qkv,
            [self.batch, seq_len, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim],
        )
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused_qkv,
            None,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cos = ttnn.slice(self.rotary_cos, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim], [1, 1, 1, 1])
        sin = ttnn.slice(self.rotary_sin, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim], [1, 1, 1, 1])
        query = ttnn.experimental.rotary_embedding(query, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.experimental.rotary_embedding(key, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        query = ttnn.slice(
            query,
            [0, 0, 0, 0],
            [self.batch, self.num_heads, seq_len, self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.slice(
            key,
            [0, 0, 0, 0],
            [self.batch, self.num_kv_heads, seq_len, self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._fill_cache(key_cache, key, page_table=page_table, seq_len=seq_len)
        self._fill_cache(value_cache, value, page_table=page_table, seq_len=seq_len)
        attention = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.prefill_sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel,
        )
        attention = ttnn.transformer.concatenate_heads(attention, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.reshape(attention, [1, self.batch, seq_len, self.local_attention_width])
        attention = self._prefill_linear(
            attention,
            self.prefill_output_weight,
            role="flat_output",
            seq_len=seq_len,
            k=self.local_attention_width,
            n=self.hidden_size,
            compute_kernel=self.attention_compute_kernel,
        )
        attention = self._all_reduce_flat(attention, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.add(residual, attention, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # TTNN operations are enqueued in order, so these buffers can be
        # returned to the allocator after the residual add has consumed them.
        # Keeping the Python references alive until frame return would make a
        # full-context prefill physically impossible despite TP4 KV sharding.
        for temporary in (normed, fused_qkv, query, key, value, cos, sin, attention):
            ttnn.deallocate(temporary)
        residual = hidden_states
        hidden_states = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.post_attention_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.norm_compute_kernel,
        )
        mlp_input = hidden_states
        hidden_states = self._mlp_prefill_flat(mlp_input, seq_len)
        output = ttnn.add(residual, hidden_states, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mlp_input)
        ttnn.deallocate(hidden_states)
        ttnn.deallocate(residual)
        return output

    def _decode_forward_device_position(
        self,
        hidden_states,
        key_cache,
        value_cache,
        *,
        current_pos_tensor,
        rope_idx_tensor,
        page_table,
        advance_position: bool,
    ):
        """Decode core whose position and RoPE lookup remain device dynamic."""

        hidden_states = ttnn.reshape(hidden_states, [1, 1, self.batch, self.hidden_size])
        hidden_states = ttnn.to_memory_config(hidden_states, self.residual_mem_config)
        residual = hidden_states
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.input_norm,
            program_config=self.norm_program_config,
            memory_config=self.residual_mem_config,
            compute_kernel_config=self.norm_compute_kernel,
        )
        normed = ttnn.to_memory_config(normed, self.qkv_input_mem_config)
        fused_qkv = ttnn.linear(
            normed,
            self.qkv_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.qkv_decode_program_config,
            compute_kernel_config=self.attention_compute_kernel,
        )
        fused_qkv = ttnn.sharded_to_interleaved(fused_qkv, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused_qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            overlap_qk_coregrid=None,
            memory_config=self.decode_attention_mem_config,
        )
        cos = ttnn.embedding(rope_idx_tensor, self.decode_rotary_cos, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rope_idx_tensor, self.decode_rotary_sin, layout=ttnn.TILE_LAYOUT)
        cos = ttnn.unsqueeze_to_4D(cos)
        sin = ttnn.unsqueeze_to_4D(sin)
        query = ttnn.experimental.rotary_embedding(query, cos, sin, 0, memory_config=self.decode_attention_mem_config)
        key = ttnn.experimental.rotary_embedding(key, cos, sin, 0, memory_config=self.decode_cache_mem_config)
        ttnn.experimental.paged_update_cache(
            key_cache, key, update_idxs_tensor=current_pos_tensor, share_cache=False, page_table=page_table
        )
        ttnn.experimental.paged_update_cache(
            value_cache, value, update_idxs_tensor=current_pos_tensor, share_cache=False, page_table=page_table
        )
        query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
        if page_table is None:
            attention = ttnn.transformer.scaled_dot_product_attention_decode(
                query,
                key_cache,
                value_cache,
                is_causal=True,
                cur_pos_tensor=current_pos_tensor,
                scale=self.scale,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=self.decode_sdpa_program_config,
                compute_kernel_config=self.decode_sdpa_compute_kernel,
            )
        else:
            attention = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                query,
                key_cache,
                value_cache,
                page_table_tensor=page_table,
                cur_pos_tensor=current_pos_tensor,
                scale=self.scale,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=self.decode_sdpa_program_config,
                compute_kernel_config=self.decode_sdpa_compute_kernel,
            )
        attention = ttnn.to_memory_config(attention, self.decode_attention_mem_config)
        attention = ttnn.experimental.nlp_concat_heads_decode(
            attention, num_heads=self.num_heads, sub_core_grids=self.decode_compute_core_grid
        )
        attention = ttnn.slice(
            attention,
            [0, 0, 0, 0],
            [1, 1, self.batch, self.local_attention_width],
            [1, 1, 1, 1],
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        attention = ttnn.to_memory_config(attention, self.output_input_mem_config)
        attention = ttnn.linear(
            attention,
            self.output_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.output_decode_program_config,
            compute_kernel_config=self.attention_compute_kernel,
        )
        attention = ttnn.to_memory_config(attention, self.residual_mem_config)
        attention = self._all_reduce_flat(attention, memory_config=self.residual_mem_config)
        hidden_states = ttnn.add(residual, attention, dtype=ttnn.bfloat16, memory_config=self.residual_mem_config)
        residual = hidden_states
        hidden_states = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.post_attention_norm,
            program_config=self.norm_program_config,
            memory_config=self.residual_mem_config,
            compute_kernel_config=self.norm_compute_kernel,
        )
        hidden_states = self._mlp_decode_flat(hidden_states)
        hidden_states = ttnn.add(residual, hidden_states, dtype=ttnn.bfloat16, memory_config=self.residual_mem_config)
        if advance_position:
            # A stack owner captures this once after all decoder layers.  The
            # one-layer harness uses it here to exercise the same lifecycle.
            ttnn.plus_one(current_pos_tensor, skip_negative_entries=True)
            ttnn.plus_one(rope_idx_tensor)
        return ttnn.reshape(hidden_states, [1, self.batch, 1, self.hidden_size])

    def decode_forward_from_position_tensor(
        self,
        hidden_states,
        key_cache,
        value_cache,
        *,
        current_pos_tensor,
        rope_idx_tensor,
        page_table=None,
        advance_position: bool = False,
    ):
        """Trace-safe decode using a persistent replicated ``[1, batch]`` position."""

        self._validate_hidden_states(hidden_states, expected_seq_len=1)
        self._validate_caches(key_cache, value_cache, page_table)
        if tuple(current_pos_tensor.shape) != (self.batch,):
            raise ValueError(f"current_pos_tensor must have shape [{self.batch}]")
        if tuple(rope_idx_tensor.shape) != (1, self.batch):
            raise ValueError(f"rope_idx_tensor must have shape [1, {self.batch}]")
        return self._decode_forward_device_position(
            hidden_states,
            key_cache,
            value_cache,
            current_pos_tensor=current_pos_tensor,
            rope_idx_tensor=rope_idx_tensor,
            page_table=page_table,
            advance_position=advance_position,
        )

    def decode_forward(self, hidden_states, key_cache, value_cache, *, current_pos: int, page_table=None):
        self._validate_hidden_states(hidden_states, expected_seq_len=1)
        self._validate_caches(key_cache, value_cache, page_table)
        if current_pos < 0 or current_pos >= self.max_cache_len:
            raise ValueError(f"current_pos must be in [0, {self.max_cache_len}), got {current_pos}")
        if page_table is not None and current_pos // PAGED_BLOCK_SIZE >= page_table.shape[1]:
            raise ValueError("page_table does not cover current_pos")
        position_2d = ttnn.slice(
            self.position_indices,
            [current_pos, 0],
            [current_pos + 1, self.batch],
            [1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        current_pos_tensor = ttnn.reshape(position_2d, [self.batch])
        rope_idx_tensor = ttnn.typecast(position_2d, ttnn.uint32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self._decode_forward_device_position(
            hidden_states,
            key_cache,
            value_cache,
            current_pos_tensor=current_pos_tensor,
            rope_idx_tensor=rope_idx_tensor,
            page_table=page_table,
            advance_position=False,
        )

    def forward(
        self,
        hidden_states,
        key_cache,
        value_cache,
        *,
        mode: str,
        current_pos: int | None = None,
        page_table=None,
    ):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, key_cache, value_cache, page_table=page_table)
        if mode == "decode":
            if current_pos is None:
                raise ValueError("decode mode requires current_pos")
            return self.decode_forward(
                hidden_states,
                key_cache,
                value_cache,
                current_pos=current_pos,
                page_table=page_table,
            )
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")


__all__ = [
    "HEAD_AXIS",
    "HIDDEN_AXIS",
    "MultiChipConfig",
    "MultiChipDecoder",
    "PAGED_BLOCK_SIZE",
    "PROVENANCE_MESH_SHAPE",
    "SharedDecoderTensors",
    "Provenance2DDecoder",
    "TARGET_MESH_SHAPE",
    "TARGET_TP_DEGREE",
    "prepare_shared_tensors",
]
