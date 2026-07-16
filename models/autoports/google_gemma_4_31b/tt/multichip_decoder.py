# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Four-P150 tensor-parallel decoder layer for ``google/gemma-4-31B``.

The layer boundary is a replicated BF16 residual tensor.  Attention and MLP
weights are tensor-parallel: QKV/gate/up are column parallel, O/down are row
parallel, and every device owns only its local attention and KV-cache heads.
The public prefill/decode and logical-length contracts are inherited from the
optimized single-chip decoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

import ttnn
from models.autoports.google_gemma_4_31b.tt.functional_decoder import (
    FULL_ATTN_Q_CHUNK,
    MLP_CHUNK,
    FunctionalDecoder,
    _validate_target_config,
)
from models.autoports.google_gemma_4_31b.tt.optimized_decoder import (
    DEFAULT_OPTIMIZATION_POLICY,
    OptimizedDecoder,
    _dram_weight_memory_config,
)
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
from models.demos.gemma4.tt.attention.operations import (
    PREFILL_SDPA_MAX_SEQ,
    PREFILL_SLIDING_CHUNK_SIZE,
    apply_per_head_norm,
    apply_qkv_projection,
    apply_rope,
    apply_rope_decode_peruser,
    effective_block_size,
    prefill_sdpa_program_config,
    split_qkv_heads_decode,
    split_qkv_heads_prefill,
)
from models.demos.gemma4.tt.ccl import CCLManager, ccl_allreduce
from models.demos.gemma4.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.tt_transformers.tt.common import PagedAttentionConfig

TARGET_MESH_SHAPE = (1, 4)
TP_SIZE = 4
PAGE_BLOCK_SIZE = 64
QKV_DECODE_OUTPUT_CORES = 32
MLP_DECODE_CORES = 14
MLP_BFP8_PACKED_GATE_UP_BLOCK_W_MAX = 6
MLP_PREFILL_CORES = 24
MLP_PREFILL_1D_MAX_ROWS = 128
PERSISTENT_CCL_CORES = 24
DEFAULT_MULTICHIP_OPTIMIZATION_POLICY = replace(
    DEFAULT_OPTIMIZATION_POLICY,
    name=f"{DEFAULT_OPTIMIZATION_POLICY.name}_tp4_packed_mlp_bfp8",
    qkv_in0_block_w=7,
    mlp_gate_up_topology="packed",
    mlp_packed_output_dtype=ttnn.bfloat8_b,
)
_PERSISTENT_CCL_POOLS: dict[int, dict] = {}


def release_multichip_decoder_resources(mesh_device) -> bool:
    """Release mesh-scoped decoder resources before the owning mesh closes.

    Traces capture the persistent scratch addresses, so this is a terminal
    mesh-owner operation: callers must release every trace and stop using all
    decoders on the mesh before invoking it.  Returning ``False`` makes an
    already-clean teardown idempotent.
    """
    key = id(mesh_device)
    pool = _PERSISTENT_CCL_POOLS.get(key)
    if pool is None:
        return False
    if pool.get("mesh_device") is not mesh_device:
        raise RuntimeError("persistent CCL pool identity does not match the mesh being released")

    ttnn.synchronize_device(mesh_device)
    for buffer in tuple(pool["buffers"].values()):
        buffer.deallocate(True)
    pool["buffers"].clear()
    pool["semaphores"].clear()
    pool["mesh_device"] = None
    pool["released"] = True
    del _PERSISTENT_CCL_POOLS[key]
    return True


@dataclass(frozen=True)
class MultichipDecoderTimings:
    prefill_ms: float | None = None
    decode_ms: float | None = None
    traced_decode_ms: float | None = None


def _layer_state(state_dict: dict[str, torch.Tensor], layer_idx: int) -> dict[str, torch.Tensor]:
    for prefix in (f"model.language_model.layers.{layer_idx}.", f"model.layers.{layer_idx}."):
        local = {key.removeprefix(prefix): value for key, value in state_dict.items() if key.startswith(prefix)}
        if local:
            return local
    raise KeyError(f"no Gemma 4 layer {layer_idx} weights found")


def _tp_tensor(
    source: torch.Tensor,
    mesh_device,
    *,
    mesh_dim: int,
    dtype: ttnn.DataType,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """Load a global TT-layout weight and fracture one dimension over TP=4."""
    return ttnn.from_torch(
        source.detach().contiguous(),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=mesh_dim),
    )


def _persistent_ccl_memory_config(mesh_device, shard_height: int, width: int) -> ttnn.MemoryConfig:
    """Place stable async-CCL storage on the tail of the Blackhole worker grid.

    Prefill RMSNorm, SDPA, and the tuned MLP occupy the low rectangular grids.
    Keeping the persistent buffer on the final 24 row-major cores lets those
    programs compile after a decode trace without reclaiming trace-owned
    addresses.  The row projections remain on their tuned grids; only their
    reduced partial is resharded at the collective boundary.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    total_cores = grid.x * grid.y
    if total_cores < PERSISTENT_CCL_CORES:
        raise ValueError(f"persistent TP4 all-reduce requires {PERSISTENT_CCL_CORES} workers, got {grid.x}x{grid.y}")
    if width % (ttnn.TILE_SIZE * PERSISTENT_CCL_CORES):
        raise ValueError(f"persistent TP4 width {width} is not tile-divisible over {PERSISTENT_CCL_CORES} cores")

    first = total_cores - PERSISTENT_CCL_CORES
    first_x, first_y = first % grid.x, first // grid.x
    ranges = {
        ttnn.CoreRange(
            ttnn.CoreCoord(first_x, first_y),
            ttnn.CoreCoord(grid.x - 1, first_y),
        )
    }
    if first_y + 1 < grid.y:
        ranges.add(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, first_y + 1),
                ttnn.CoreCoord(grid.x - 1, grid.y - 1),
            )
        )
    return ttnn.create_sharded_memory_config(
        shape=(shard_height, width // PERSISTENT_CCL_CORES),
        core_grid=ttnn.CoreRangeSet(ranges),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _tp_allreduce(
    partial,
    mesh_config,
    ccl_manager,
    *,
    communication_dtype: ttnn.DataType,
    use_persistent_async: bool,
    persistent_role: str,
):
    """Reduce a row-parallel result while keeping BF16 layer boundaries.

    ``communication_dtype`` controls the async reduction output.  Fabric input
    pages retain the partial tensor's dtype: attention sends BF16 pages, while
    the packed MLP sends its native BFP8 pages and accumulates to BF16.  An
    explicit BFP8-to-BF16 cast would not recover projection precision and would
    add a conversion plus larger wire pages.  The optional BFP8 policy instead
    quantizes BF16 partials before the collective and restores BF16 afterward.
    """
    communicated = partial
    if partial.dtype != communication_dtype:
        communicated = ttnn.typecast(
            partial,
            communication_dtype,
            memory_config=partial.memory_config(),
        )
        partial.deallocate(True)
    if use_persistent_async:
        pool = ccl_manager._persistent_allreduce_pool
        if pool.get("released") or pool.get("mesh_device") is not ccl_manager.mesh_device:
            raise RuntimeError("persistent CCL pool was released or belongs to a different mesh")
        input_memory_config = communicated.memory_config()
        if (
            input_memory_config.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED
            or input_memory_config.buffer_type != ttnn.BufferType.L1
        ):
            raise ValueError(
                "persistent Blackhole all-reduce requires an L1 WIDTH_SHARDED partial, " f"got {input_memory_config}"
            )
        shard_spec = input_memory_config.shard_spec
        if shard_spec is None:
            raise ValueError("persistent Blackhole all-reduce requires a shard spec")
        rows, width = communicated.shape[-2], communicated.shape[-1]
        persistent_memory_config = _persistent_ccl_memory_config(
            ccl_manager.mesh_device,
            shard_spec.shape[0],
            width,
        )
        persistent_partial = ttnn.to_memory_config(communicated, persistent_memory_config)
        communicated.deallocate(True)
        communicated = persistent_partial
        input_memory_config = persistent_memory_config
        shard_spec = input_memory_config.shard_spec
        if shard_spec is None:
            raise ValueError("persistent Blackhole all-reduce reshard lost its shard spec")
        slot = pool["slot"]
        pool["slot"] = (slot + 1) % len(pool["semaphores"])
        grid_key = tuple(
            (core_range.start.x, core_range.start.y, core_range.end.x, core_range.end.y)
            for core_range in shard_spec.grid.ranges()
        )
        key = (
            communication_dtype,
            tuple(shard_spec.shape),
            grid_key,
            shard_spec.orientation,
        )
        buffer = pool["buffers"].get(key)
        if buffer is None:
            # The minimal all-reduce consumes a stable intermediate whose
            # per-core L1 shard is TP times the reduced output shard.  It must
            # use the output grid because the device op installs the scratch
            # and result as globally allocated circular buffers.
            buffer_memory_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    shard_spec.grid,
                    [shard_spec.shape[0], shard_spec.shape[1] * TP_SIZE],
                    shard_spec.orientation,
                ),
            )
            # Both row projections have the same physical TP4 shape and are
            # serialized on the command queue.  Sharing one physical-capacity
            # buffer avoids role/slot and logical-M duplication while the two
            # semaphores still maintain independent collective epochs.
            buffer_source = torch.zeros(
                (1, TP_SIZE, shard_spec.shape[0], width * TP_SIZE),
                dtype=torch.bfloat16,
            )
            buffer = ttnn.from_torch(
                buffer_source,
                device=ccl_manager.mesh_device,
                dtype=communication_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=buffer_memory_config,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    ccl_manager.mesh_device,
                    dims=(0, 1),
                    mesh_shape=ccl_manager.mesh_device.shape,
                ),
            )
            pool["buffers"][key] = buffer
        reduced = ttnn.experimental.all_reduce_async(
            communicated,
            buffer,
            mesh_config.tp_axis,
            ccl_manager.mesh_device,
            pool["semaphores"][slot],
            dtype=communication_dtype,
            memory_config=input_memory_config,
            topology=ccl_manager.topology,
            num_links=ccl_manager.num_links,
        )
        communicated.deallocate(True)
        reduced_l1 = reduced
        reduced = ttnn.sharded_to_interleaved(reduced_l1, ttnn.DRAM_MEMORY_CONFIG)
        reduced_l1.deallocate(True)
    else:
        reduced = ccl_allreduce(communicated, mesh_config, ccl_manager)
    if communication_dtype == ttnn.bfloat16:
        return reduced
    restored = ttnn.typecast(reduced, ttnn.bfloat16)
    reduced.deallocate(True)
    return restored


class _TPOptimizedSharedMLP:
    """GeGLU with BFP4 TP weights and optimized local decode matmuls."""

    def __init__(
        self,
        *,
        mesh_device,
        mesh_config,
        ccl_manager,
        state,
        policy,
        communication_dtype,
        use_persistent_async_ccl,
    ):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.policy = policy
        self.communication_dtype = communication_dtype
        self.use_persistent_async_ccl = use_persistent_async_ccl
        self.is_decode = False
        hidden = int(state["gate_proj.weight"].shape[1])
        intermediate = int(state["gate_proj.weight"].shape[0])
        self.local_intermediate = intermediate // TP_SIZE
        if hidden != self.local_intermediate:
            raise ValueError(
                f"Gemma 4 31B TP=4 expects local intermediate == hidden, got " f"{self.local_intermediate} and {hidden}"
            )

        gate = state["gate_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        up = state["up_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        down = state["down_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)

        # Large-M prefill and M=1 decode use separate physical placements, as
        # in the optimized single-chip baseline.  Both retain BFP4/LoFi.
        self.gate_prefill = _tp_tensor(gate, mesh_device, mesh_dim=3, dtype=policy.mlp_gate_up_weight_dtype)
        self.up_prefill = _tp_tensor(up, mesh_device, mesh_dim=3, dtype=policy.mlp_gate_up_weight_dtype)
        self.down_prefill = _tp_tensor(down, mesh_device, mesh_dim=2, dtype=policy.mlp_down_weight_dtype)
        local_mem = _dram_weight_memory_config(mesh_device, k=hidden, n=hidden)
        self.gate_decode = _tp_tensor(
            gate,
            mesh_device,
            mesh_dim=3,
            dtype=policy.mlp_gate_up_weight_dtype,
            memory_config=local_mem,
        )
        self.up_decode = _tp_tensor(
            up,
            mesh_device,
            mesh_dim=3,
            dtype=policy.mlp_gate_up_weight_dtype,
            memory_config=local_mem,
        )
        self.down_decode = _tp_tensor(
            down,
            mesh_device,
            mesh_dim=2,
            dtype=policy.mlp_down_weight_dtype,
            memory_config=local_mem,
        )
        self.packed_gate_up_decode = None
        if policy.mlp_gate_up_topology == "packed":
            packed_per_device = []
            gate_chunks = torch.chunk(gate, TP_SIZE, dim=3)
            up_chunks = torch.chunk(up, TP_SIZE, dim=3)
            for device_idx in range(TP_SIZE):
                packed_per_device.append(torch.cat((gate_chunks[device_idx], up_chunks[device_idx]), dim=3))
            packed_source = torch.cat(packed_per_device, dim=3)
            self.packed_gate_up_decode = _tp_tensor(
                packed_source,
                mesh_device,
                mesh_dim=3,
                dtype=policy.mlp_gate_up_weight_dtype,
                memory_config=_dram_weight_memory_config(mesh_device, k=hidden, n=2 * hidden),
            )
        self.gate_up_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=policy.mlp_gate_up_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.down_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=policy.mlp_down_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def _reduce(self, partial):
        use_persistent_async = self.use_persistent_async_ccl and self.is_decode
        if partial.is_sharded() and not use_persistent_async:
            interleaved = ttnn.sharded_to_interleaved(partial, ttnn.DRAM_MEMORY_CONFIG)
            partial.deallocate(True)
            partial = interleaved
        return _tp_allreduce(
            partial,
            self.mesh_config,
            self.ccl_manager,
            communication_dtype=self.communication_dtype if self.is_decode else ttnn.bfloat16,
            use_persistent_async=use_persistent_async,
            persistent_role="mlp_down",
        )

    def _decode_memory_config(self, num_cores: int, width: int) -> ttnn.MemoryConfig:
        if width % (ttnn.TILE_SIZE * num_cores):
            raise ValueError(f"decode width {width} is not tile-divisible across {num_cores} cores")
        core_grid = ttnn.num_cores_to_corerangeset(
            num_cores,
            self.mesh_device.compute_with_storage_grid_size(),
            row_wise=True,
        )
        return ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, width // num_cores),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    @staticmethod
    def _decode_program_config(*, k, n, num_cores, in0_block_w, fused_activation=None):
        k_tiles_per_core = k // (ttnn.TILE_SIZE * num_cores)
        if k_tiles_per_core % in0_block_w:
            raise ValueError(f"in0_block_w={in0_block_w} does not divide {k_tiles_per_core} " f"K tiles/core for K={k}")
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=in0_block_w,
            per_core_M=1,
            per_core_N=math.ceil(n / (ttnn.TILE_SIZE * num_cores)),
            fused_activation=fused_activation,
        )

    @property
    def packed_gate_up_in0_block_w(self) -> int:
        """Largest L1-safe packed gate/up K block for the active weight dtype."""
        requested = self.policy.gate_up_in0_block_w
        if self.policy.mlp_gate_up_weight_dtype == ttnn.bfloat8_b:
            return min(requested, MLP_BFP8_PACKED_GATE_UP_BLOCK_W_MAX)
        return requested

    def __call__(self, hidden_states):
        if not self.is_decode:
            rows = hidden_states.shape[-2]
            if rows <= MLP_PREFILL_1D_MAX_ROWS:
                program_args = dict(
                    compute_with_storage_grid_size=(8, 3),
                    in0_block_w=7,
                    out_subblock_h=1,
                    out_subblock_w=7,
                    per_core_M=rows // ttnn.TILE_SIZE,
                    per_core_N=self.local_intermediate // (ttnn.TILE_SIZE * MLP_PREFILL_CORES),
                    fuse_batch=True,
                    mcast_in0=True,
                )
                program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    **program_args,
                    fused_activation=None,
                )
                gate_program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    **program_args,
                    fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0),
                )
                up = ttnn.linear(
                    hidden_states,
                    self.up_prefill,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=program,
                    compute_kernel_config=self.gate_up_compute,
                )
                gate = ttnn.linear(
                    hidden_states,
                    self.gate_prefill,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=gate_program,
                    compute_kernel_config=self.gate_up_compute,
                )
                activated = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                gate.deallocate(True)
                up.deallocate(True)
                partial = ttnn.linear(
                    activated,
                    self.down_prefill,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=program,
                    compute_kernel_config=self.down_compute,
                )
                activated.deallocate(True)
                return self._reduce(partial)
            gate = ttnn.linear(hidden_states, self.gate_prefill)
            gate = ttnn.gelu(gate, fast_and_approximate_mode=True)
            up = ttnn.linear(hidden_states, self.up_prefill)
            activated = ttnn.mul(gate, up)
            gate.deallocate(True)
            up.deallocate(True)
            partial = ttnn.linear(activated, self.down_prefill)
            activated.deallocate(True)
            return self._reduce(partial)

        hidden = hidden_states.shape[-1]
        input_mem = self._decode_memory_config(self.policy.decode_num_cores, hidden)
        local_mem = self._decode_memory_config(self.policy.decode_num_cores, self.local_intermediate)
        sharded_input = ttnn.to_memory_config(hidden_states, input_mem)
        gate_program = self._decode_program_config(
            k=hidden,
            n=self.local_intermediate,
            num_cores=self.policy.decode_num_cores,
            in0_block_w=self.policy.gate_up_in0_block_w,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0),
        )
        up_program = self._decode_program_config(
            k=hidden,
            n=self.local_intermediate,
            num_cores=self.policy.decode_num_cores,
            in0_block_w=self.policy.gate_up_in0_block_w,
        )
        if self.policy.mlp_gate_up_topology == "packed" and hidden_states.shape[-2] == 1:
            packed_mem = self._decode_memory_config(self.policy.decode_num_cores, 2 * self.local_intermediate)
            packed_program = self._decode_program_config(
                k=hidden,
                n=2 * self.local_intermediate,
                num_cores=self.policy.decode_num_cores,
                in0_block_w=self.packed_gate_up_in0_block_w,
            )
            packed_sharded = ttnn.linear(
                sharded_input,
                self.packed_gate_up_decode,
                dtype=self.policy.mlp_packed_output_dtype,
                memory_config=packed_mem,
                program_config=packed_program,
                compute_kernel_config=self.gate_up_compute,
            )
            sharded_input.deallocate(True)
            packed = ttnn.sharded_to_interleaved(packed_sharded, ttnn.DRAM_MEMORY_CONFIG)
            packed_sharded.deallocate(True)
            gate = ttnn.slice(packed, [0, 0, 0, 0], [1, 1, packed.shape[2], self.local_intermediate])
            up = ttnn.slice(
                packed,
                [0, 0, 0, self.local_intermediate],
                [1, 1, packed.shape[2], 2 * self.local_intermediate],
            )
            packed.deallocate(True)
            gate = ttnn.gelu(gate, fast_and_approximate_mode=True)
            gate_sharded = ttnn.to_memory_config(gate, local_mem)
            gate.deallocate(True)
            up_sharded = ttnn.to_memory_config(up, local_mem)
            up.deallocate(True)
            gate, up = gate_sharded, up_sharded
        elif self.policy.mlp_gate_up_topology == "packed":
            # The packed N=2*local_intermediate matmul is the fastest M=1
            # graph, but its triple-buffered BFP4 weight CB cannot coexist
            # with persistent collectives when a distinct logical-M program
            # is compiled.  For M>1, keep the same BFP8 projection boundary
            # and device-only math while halving N.  Spill each projection to
            # DRAM before launching the next so their L1 outputs never overlap.
            up_sharded = ttnn.linear(
                sharded_input,
                self.up_decode,
                dtype=self.policy.mlp_packed_output_dtype,
                memory_config=local_mem,
                program_config=up_program,
                compute_kernel_config=self.gate_up_compute,
            )
            up = ttnn.sharded_to_interleaved(up_sharded, ttnn.DRAM_MEMORY_CONFIG)
            up_sharded.deallocate(True)
            gate_sharded = ttnn.linear(
                sharded_input,
                self.gate_decode,
                dtype=self.policy.mlp_packed_output_dtype,
                memory_config=local_mem,
                program_config=up_program,
                compute_kernel_config=self.gate_up_compute,
            )
            sharded_input.deallocate(True)
            gate = ttnn.sharded_to_interleaved(gate_sharded, ttnn.DRAM_MEMORY_CONFIG)
            gate_sharded.deallocate(True)
            gate = ttnn.gelu(gate, fast_and_approximate_mode=True)
            gate_sharded = ttnn.to_memory_config(gate, local_mem)
            gate.deallocate(True)
            up_sharded = ttnn.to_memory_config(up, local_mem)
            up.deallocate(True)
            gate, up = gate_sharded, up_sharded
        else:
            up = ttnn.linear(
                sharded_input,
                self.up_decode,
                memory_config=local_mem,
                program_config=up_program,
                compute_kernel_config=self.gate_up_compute,
            )
            gate = ttnn.linear(
                sharded_input,
                self.gate_decode,
                memory_config=local_mem,
                program_config=gate_program,
                compute_kernel_config=self.gate_up_compute,
            )
            sharded_input.deallocate(True)
        activated = ttnn.mul(gate, up, memory_config=local_mem)
        gate.deallocate(True)
        up.deallocate(True)
        down_program = self._decode_program_config(
            k=self.local_intermediate,
            n=hidden,
            num_cores=self.policy.decode_num_cores,
            in0_block_w=self.policy.down_in0_block_w,
        )
        partial = ttnn.linear(
            activated,
            self.down_decode,
            memory_config=input_mem,
            program_config=down_program,
            compute_kernel_config=self.down_compute,
        )
        activated.deallocate(True)
        return self._reduce(partial)


class MultichipDecoder(OptimizedDecoder):
    """Real Gemma 4 31B decoder layer specialized for the local 1x4 mesh."""

    baseline_cls = OptimizedDecoder
    mesh_profile = {
        "name": "gemma4_31b_p150x4_tp4_replicated_residual_v1",
        "single_chip_baseline": DEFAULT_OPTIMIZATION_POLICY.name,
        "target_mesh": "1x4 Blackhole P150b",
        "tp": TP_SIZE,
        "activation_contract": "replicated BF16 residual at layer input/output",
        "attention": "column-parallel local-head QKV/SDPA; row-parallel O plus TP sum",
        "mlp": "BFP4 column-parallel gate/up; BFP4 row-parallel down plus persistent TP sum; 14-core local decode",
        "collective_dtype": "BF16 prefill; BFP8 decode input/output restored to BF16 boundary",
        "kv_cache": "BFP8 paged local KV heads; replicated page table and positions",
        "moe": "not applicable: dense target",
    }

    def __init__(self, **kwargs):
        # Bypass FusedDecoder/OptimizedDecoder construction-time module
        # rewrites: this class already installs its TP-aware attention and MLP.
        FunctionalDecoder.__init__(self, **kwargs)
        self.policy = DEFAULT_MULTICHIP_OPTIMIZATION_POLICY
        self.attention_compute = None
        self.attention_qkv_compute = None
        self.attention_o_compute = None
        self.decode_wqkv = None
        self.decode_wq = None
        self.decode_wk = None
        self.decode_wv = None
        self.decode_o_proj = None
        self.mesh_config = None
        self.ccl_manager = None
        self.qkv_decode_output_cores = QKV_DECODE_OUTPUT_CORES
        self.communication_dtype = ttnn.bfloat8_b
        self.use_persistent_async_ccl = True
        self.sdpa_q_chunk_size = 32
        self.sdpa_k_chunk_size = 64
        self.sdpa_exp_approx_mode = False
        self.sdpa_force_full_grid = False
        self.timings = MultichipDecoderTimings()

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        tensor_cache_path=None,
        optimization_policy=DEFAULT_MULTICHIP_OPTIMIZATION_POLICY,
        bounded_sliding_kv_cache=True,
        num_links=2,
        qkv_decode_output_cores=QKV_DECODE_OUTPUT_CORES,
        communication_dtype=ttnn.bfloat8_b,
        prefill_communication_dtype=ttnn.bfloat16,
        residual_dtype=ttnn.bfloat16,
        use_persistent_async_ccl=True,
        topology=ttnn.Topology.Linear,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"unsupported MultichipDecoder kwargs: {sorted(kwargs)}")
        if tuple(mesh_device.shape) != TARGET_MESH_SHAPE or mesh_device.get_num_devices() != TP_SIZE:
            raise ValueError(
                f"MultichipDecoder requires MeshShape{TARGET_MESH_SHAPE}, got "
                f"shape={tuple(mesh_device.shape)} devices={mesh_device.get_num_devices()}"
            )
        if qkv_decode_output_cores not in (8, 16, QKV_DECODE_OUTPUT_CORES):
            raise ValueError("qkv_decode_output_cores must be one of the validated 8/16/32-core geometries")
        contract = _validate_target_config(hf_config, layer_idx)
        model_args = Gemma4ModelArgs.from_hf_config(hf_config)
        mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=TP_SIZE), prefill=ModeConfig(tp=TP_SIZE))
        ccl_manager = CCLManager(mesh_device, num_links=num_links, topology=topology)
        if use_persistent_async_ccl:
            pool_key = id(mesh_device)
            pool = _PERSISTENT_CCL_POOLS.get(pool_key)
            if pool is None or pool["mesh_device"] is not mesh_device:
                grid = mesh_device.compute_with_storage_grid_size()
                cores = ttnn.num_cores_to_corerangeset(grid.x * grid.y, grid, row_wise=True)
                pool = {
                    "mesh_device": mesh_device,
                    "semaphores": [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(2)],
                    "buffers": {},
                    "slot": 0,
                }
                _PERSISTENT_CCL_POOLS[pool_key] = pool
                ttnn.synchronize_device(mesh_device)
            ccl_manager._persistent_allreduce_pool = pool
        layer = Gemma4DecoderLayer(
            mesh_device=mesh_device,
            hf_config=model_args,
            state_dict=state_dict,
            layer_idx=layer_idx,
            ccl_manager=ccl_manager,
            dtype=ttnn.bfloat8_b,
            attention_dtype=optimization_policy.attention_weight_dtype,
            shared_mlp_dtype=ttnn.bfloat8_b,
            tensor_cache_path=tensor_cache_path,
            mesh_config=mesh_config,
            max_seq_len=contract.max_position_embeddings,
            max_local_batch_size=32,
            bounded_sliding_kv_cache=bounded_sliding_kv_cache,
        )
        old_mlp = layer.shared_mlp
        local = _layer_state(state_dict, layer_idx)
        mlp_state = {key.removeprefix("mlp."): value for key, value in local.items() if key.startswith("mlp.")}
        mlp_policy = replace(
            optimization_policy,
            name=f"{optimization_policy.name}_tp4_square_mlp_14c",
            decode_num_cores=MLP_DECODE_CORES,
            gate_up_in0_block_w=12,
            down_in0_block_w=12,
        )
        layer.shared_mlp = _TPOptimizedSharedMLP(
            mesh_device=mesh_device,
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            state=mlp_state,
            policy=mlp_policy,
            communication_dtype=communication_dtype,
            use_persistent_async_ccl=use_persistent_async_ccl,
        )
        for weight in (old_mlp.gate_proj, old_mlp.up_proj, old_mlp.down_proj):
            weight.deallocate(True)
        decoder = cls(layer=layer, contract=contract, layer_idx=layer_idx, mesh_device=mesh_device)
        decoder.policy = optimization_policy
        decoder.mesh_config = mesh_config
        decoder.ccl_manager = ccl_manager
        decoder.qkv_decode_output_cores = qkv_decode_output_cores
        decoder.communication_dtype = communication_dtype
        decoder.prefill_communication_dtype = prefill_communication_dtype
        decoder.residual_dtype = residual_dtype
        decoder.use_persistent_async_ccl = use_persistent_async_ccl
        decoder.attention_qkv_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=optimization_policy.resolved_attention_qkv_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        decoder.attention_o_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=optimization_policy.resolved_attention_o_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        decoder.attention_compute = decoder.attention_qkv_compute

        # Preserve the optimized baseline's decode dataflow with TP-local
        # packed QKV and row-local O copies, each width-sharded over all DRAM
        # banks.  The demo weights remain interleaved for large-M prefill.
        config = layer.self_attn.config
        attn_state = {
            key.removeprefix("self_attn."): value for key, value in local.items() if key.startswith("self_attn.")
        }
        q_weight = attn_state["q_proj.weight"]
        k_weight = attn_state["k_proj.weight"]
        v_weight = k_weight if config.use_kv_tying else attn_state["v_proj.weight"]
        packed_per_device = []
        q_chunks = torch.chunk(q_weight, TP_SIZE, dim=0)
        k_chunks = torch.chunk(k_weight, TP_SIZE, dim=0)
        v_chunks = torch.chunk(v_weight, TP_SIZE, dim=0)
        for device_idx in range(TP_SIZE):
            packed_per_device.append(
                torch.cat(
                    (
                        q_chunks[device_idx].transpose(-2, -1),
                        k_chunks[device_idx].transpose(-2, -1),
                        v_chunks[device_idx].transpose(-2, -1),
                    ),
                    dim=-1,
                )
            )
        packed_qkv = torch.cat(packed_per_device, dim=-1).unsqueeze(0).unsqueeze(0)
        local_qkv_width = packed_per_device[0].shape[-1]
        decoder.decode_wqkv = _tp_tensor(
            packed_qkv,
            mesh_device,
            mesh_dim=3,
            dtype=optimization_policy.resolved_attention_qkv_weight_dtype,
            memory_config=_dram_weight_memory_config(mesh_device, k=config.hidden_size, n=local_qkv_width),
        )
        if optimization_policy.attention_projection_topology == "split":
            split_sources = []
            for chunks in (q_chunks, k_chunks, v_chunks):
                split_sources.append(
                    torch.cat([chunk.transpose(-2, -1) for chunk in chunks], dim=-1).unsqueeze(0).unsqueeze(0)
                )
            decoder.decode_wq, decoder.decode_wk, decoder.decode_wv = (
                _tp_tensor(
                    source,
                    mesh_device,
                    mesh_dim=3,
                    dtype=optimization_policy.resolved_attention_qkv_weight_dtype,
                    memory_config=_dram_weight_memory_config(
                        mesh_device,
                        k=config.hidden_size,
                        n=source.shape[-1] // TP_SIZE,
                    ),
                )
                for source in split_sources
            )
        o_source = attn_state["o_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        local_o_k = config.num_attention_heads * config.head_dim // TP_SIZE
        decoder.decode_o_proj = _tp_tensor(
            o_source,
            mesh_device,
            mesh_dim=2,
            dtype=optimization_policy.resolved_attention_o_weight_dtype,
            memory_config=_dram_weight_memory_config(mesh_device, k=local_o_k, n=config.hidden_size),
        )
        return decoder

    def _decode_attention_tp(
        self,
        hidden_states,
        *,
        rope_mats,
        page_table,
        kv_cache,
        current_position,
        current_position_cache,
        batch_size,
    ):
        """TP-local decode with DRAM-sharded packed QKV/O projections."""
        attention = self.layer.self_attn
        config, weights = attention.config, attention.weights
        local_heads = config.num_attention_heads // TP_SIZE
        local_kv_heads = config.num_key_value_heads // TP_SIZE
        input_mem = self._decode_memory_config(self.mesh_device, self.policy.decode_num_cores, hidden_states.shape[-1])
        sharded_input = ttnn.to_memory_config(hidden_states, input_mem)
        decode_weights = (
            (self.decode_wqkv,)
            if self.policy.attention_projection_topology == "packed"
            else (self.decode_wq, self.decode_wk, self.decode_wv)
        )
        qkv_parts = []
        for decode_weight in decode_weights:
            qkv_n = decode_weight.shape[-1]
            output_cores = self.qkv_decode_output_cores if len(decode_weights) == 1 else self.policy.decode_num_cores
            qkv_grid = ttnn.num_cores_to_corerangeset(
                output_cores,
                self.mesh_device.compute_with_storage_grid_size(),
                row_wise=True,
            )
            qkv_mem = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, qkv_n // output_cores),
                core_grid=qkv_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            qkv_program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=self.policy.qkv_in0_block_w,
                per_core_M=1,
                per_core_N=qkv_n // (ttnn.TILE_SIZE * output_cores),
            )
            qkv_sharded = ttnn.linear(
                sharded_input,
                decode_weight,
                memory_config=qkv_mem,
                program_config=qkv_program,
                compute_kernel_config=self.attention_qkv_compute,
                dtype=self.qkv_split_input_dtype,
            )
            qkv_parts.append(ttnn.sharded_to_interleaved(qkv_sharded, ttnn.L1_MEMORY_CONFIG))
            qkv_sharded.deallocate(True)
        sharded_input.deallocate(True)
        if len(qkv_parts) == 1:
            qkv = qkv_parts[0]
        else:
            qkv = ttnn.concat(qkv_parts, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
            for part in qkv_parts:
                part.deallocate(True)
        q, k, v = split_qkv_heads_decode(qkv, config, weights.is_global, tp=TP_SIZE, kv_replicated=False)
        qkv.deallocate(True)

        q_sharded_mem = q.memory_config()
        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        q = apply_per_head_norm(q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)
        k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.to_memory_config(v, ttnn.DRAM_MEMORY_CONFIG)
        k = apply_per_head_norm(k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
        v = apply_per_head_norm(v, None, config.rms_norm_eps, with_scale=False)

        cos_cache, sin_cache = rope_mats
        cos_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(current_position, cos_cache, layout=ttnn.TILE_LAYOUT))
        sin_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(current_position, sin_cache, layout=ttnn.TILE_LAYOUT))
        if batch_size == 1:
            q = apply_rope(q, cos_pos, sin_pos, token_index=0)
            k = apply_rope(k, cos_pos, sin_pos, token_index=0)
        else:
            cos_b = ttnn.transpose(cos_pos, 1, 2)[:, :batch_size, :, :]
            sin_b = ttnn.transpose(sin_pos, 1, 2)[:, :batch_size, :, :]
            q = apply_rope_decode_peruser(q, cos_b, sin_b)
            k = apply_rope_decode_peruser(k, cos_b, sin_b)

        k = self._prepare_cache_update_input(k)
        v = self._prepare_cache_update_input(v)

        cache_position = current_position_cache if current_position_cache is not None else current_position
        k_cache, v_cache = kv_cache
        block_size = effective_block_size(k_cache, config.head_dim, local_kv_heads)
        cache_geometry_matches = (
            int(k_cache.padded_shape[1]) == local_kv_heads and int(k_cache.padded_shape[-1]) == config.head_dim
        )
        if config.cache_position_modulo is None and cache_geometry_matches:
            device_grid = self.mesh_device.compute_with_storage_grid_size()
            grid_x = min(batch_size, device_grid.x)
            while batch_size % grid_x:
                grid_x -= 1
            grid_h = batch_size // grid_x
            k_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(grid_x - 1, grid_h - 1),
                    )
                }
            )
            v_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, grid_h),
                        ttnn.CoreCoord(grid_x - 1, 2 * grid_h - 1),
                    )
                }
            )
            k_memory_config = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, config.head_dim),
                core_grid=k_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            v_memory_config = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, config.head_dim),
                core_grid=v_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            k = ttnn.to_memory_config(k, k_memory_config)
            v = ttnn.to_memory_config(v, v_memory_config)
            ttnn.experimental.paged_fused_update_cache(
                k_cache,
                k,
                v_cache,
                v,
                update_idxs_tensor=cache_position,
                page_table=page_table,
            )
        else:
            k = ttnn.to_memory_config(k, q_sharded_mem)
            v = ttnn.to_memory_config(v, q_sharded_mem)
            update_args = dict(
                update_idxs_tensor=cache_position,
                page_table=page_table,
                block_size=block_size,
                num_kv_heads=local_kv_heads,
            )
            if config.cache_position_modulo is not None:
                update_args["cache_position_modulo"] = config.cache_position_modulo
            ttnn.experimental.paged_update_cache(k_cache, k, **update_args)
            ttnn.experimental.paged_update_cache(v_cache, v, **update_args)
        k.deallocate(True)
        v.deallocate(True)

        sdpa_grid = (
            self.mesh_device.compute_with_storage_grid_size()
            if self.sdpa_force_full_grid or config.head_dim < 512
            else ttnn.CoreCoord(8, 4)
        )
        sdpa_program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=sdpa_grid,
            q_chunk_size=self.sdpa_q_chunk_size,
            k_chunk_size=self.sdpa_k_chunk_size,
            exp_approx_mode=self.sdpa_exp_approx_mode,
        )
        sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            k_cache,
            v_cache,
            cur_pos_tensor=cache_position,
            page_table_tensor=page_table,
            scale=1.0,
            sliding_window_size=(config.sliding_window if config.is_sliding else None),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=sdpa_program,
            block_size=block_size,
            num_kv_heads=local_kv_heads,
            **(
                {"cache_position_modulo": config.cache_position_modulo}
                if config.cache_position_modulo is not None
                else {}
            ),
        )
        q.deallocate(True)
        from models.tt_transformers.tt.model_config import num_to_corerange

        grid = self.mesh_device.compute_with_storage_grid_size()
        grid_x = min(batch_size, grid.x)
        if batch_size >= grid_x and batch_size % grid_x:
            grid_x = max(x for x in range(grid_x, 0, -1) if batch_size % x == 0 and batch_size // x <= grid.y)
        core_grid = ttnn.CoreRangeSet({num_to_corerange(batch_size, grid_x=grid_x, grid_y=grid.y)})
        head_mem = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, config.head_dim),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        sdpa_sharded = ttnn.to_memory_config(sdpa, head_mem)
        sdpa.deallocate(True)
        concatenated = ttnn.experimental.nlp_concat_heads_decode(sdpa_sharded, num_heads=local_heads)
        sdpa_sharded.deallocate(True)
        output = ttnn.sharded_to_interleaved(concatenated, ttnn.DRAM_MEMORY_CONFIG)
        concatenated.deallocate(True)
        if output.shape[2] != batch_size:
            padded = output
            output = padded[:, :, :batch_size, :]
            padded.deallocate(True)

        o_k = local_heads * config.head_dim
        o_input_mem = self._decode_memory_config(self.mesh_device, self.policy.decode_num_cores, o_k)
        o_output_mem = self._decode_memory_config(self.mesh_device, self.policy.decode_num_cores, config.hidden_size)
        o_input = ttnn.to_memory_config(output, o_input_mem)
        output.deallocate(True)
        o_program = self._decode_matmul_program_config(
            k=o_k,
            n=config.hidden_size,
            num_cores=self.policy.decode_num_cores,
            in0_block_w=self.policy.o_proj_in0_block_w,
        )
        projected_sharded = ttnn.linear(
            o_input,
            self.decode_o_proj,
            memory_config=o_output_mem,
            program_config=o_program,
            compute_kernel_config=self.attention_o_compute,
        )
        o_input.deallocate(True)
        if self.use_persistent_async_ccl:
            projected = projected_sharded
        else:
            projected = ttnn.sharded_to_interleaved(projected_sharded, ttnn.DRAM_MEMORY_CONFIG)
            projected_sharded.deallocate(True)
        return _tp_allreduce(
            projected,
            self.mesh_config,
            self.ccl_manager,
            communication_dtype=self.communication_dtype,
            use_persistent_async=self.use_persistent_async_ccl,
            persistent_role="attention_o",
        )

    def init_paged_kv_cache(self, *, max_context=262_144, batch_size=1):
        config = self.layer.self_attn.config
        physical_context = config.sliding_window if config.is_sliding else max_context
        num_blocks_per_user = (physical_context + PAGE_BLOCK_SIZE - 1) // PAGE_BLOCK_SIZE
        paged = PagedAttentionConfig(
            block_size=PAGE_BLOCK_SIZE,
            max_num_blocks=num_blocks_per_user * batch_size,
        )
        cache = init_kv_cache(
            self.mesh_device,
            config,
            paged_attention_config=paged,
            cache_dtype=self.policy.kv_cache_dtype,
            max_num_blocks_override=num_blocks_per_user * batch_size,
        )
        rows = torch.arange(num_blocks_per_user * batch_size, dtype=torch.int32).reshape(
            batch_size, num_blocks_per_user
        )
        page_table = ttnn.from_torch(
            rows,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return cache, page_table

    def _prefill_attention_tp(
        self,
        hidden_states,
        *,
        source_hidden_states=None,
        rope_mats,
        page_table,
        kv_cache,
        user_id,
        valid_seq_len,
        residual=None,
    ):
        """TP-local prefill with the optimized BFP8 cache-fill contract."""
        attention = self.layer.self_attn
        config, weights = attention.config, attention.weights
        qkv = apply_qkv_projection(hidden_states, weights)
        # Prefill never reads the normalized attention input after QKV
        # projection.  Releasing this prompt-sized DRAM buffer here is
        # required before head concat and the output all-reduce allocate their
        # prompt-sized results.  The caller retains the separate residual.
        hidden_states.deallocate(True)
        if source_hidden_states is not None and source_hidden_states.is_allocated():
            # A batched reshape is normally an alias, but non-tile-aligned
            # per-user lengths can materialize a distinct tiled tensor.  Free
            # that original normalization result as well when it still owns
            # storage.
            source_hidden_states.deallocate(True)
        q, k, v = split_qkv_heads_prefill(
            qkv,
            config,
            weights.is_global,
            tp=TP_SIZE,
            kv_replicated=weights.kv_replicated,
        )
        qkv.deallocate(True)
        q = apply_per_head_norm(q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)
        k = apply_per_head_norm(k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
        v = apply_per_head_norm(v, None, config.rms_norm_eps, with_scale=False)
        cos_cache, sin_cache = rope_mats
        q = apply_rope(q, cos_cache, sin_cache)
        k = apply_rope(k, cos_cache, sin_cache)

        local_kv_heads = config.num_key_value_heads // TP_SIZE
        k_cache, v_cache = kv_cache
        block_size = effective_block_size(k_cache, config.head_dim, local_kv_heads)
        modulo = {"cache_position_modulo": config.cache_position_modulo} if config.cache_position_modulo else {}
        if config.cache_position_modulo is not None and valid_seq_len < k.shape[-2]:
            self._fill_bounded_sliding_cache_exact(
                k_cache,
                v_cache,
                k,
                v,
                page_table,
                user_id=user_id,
                valid_seq_len=valid_seq_len,
                block_size=block_size,
                num_kv_heads=local_kv_heads,
                cache_position_modulo=config.cache_position_modulo,
            )
        else:
            k_fill = ttnn.typecast(k, k_cache.dtype) if k.dtype != k_cache.dtype else k
            v_fill = ttnn.typecast(v, v_cache.dtype) if v.dtype != v_cache.dtype else v
            ttnn.experimental.paged_fill_cache(
                k_cache, k_fill, page_table, batch_idx=user_id, block_size=block_size, **modulo
            )
            ttnn.experimental.paged_fill_cache(
                v_cache, v_fill, page_table, batch_idx=user_id, block_size=block_size, **modulo
            )
            if k_fill is not k:
                k_fill.deallocate(True)
            if v_fill is not v:
                v_fill.deallocate(True)

        seq_len = q.shape[-2]
        concatenated = None
        if seq_len > PREFILL_SDPA_MAX_SEQ and config.is_sliding:
            # Sliding attention must retain the prompt K/V sources, but its
            # bounded SDPA chunks can still be written directly into the final
            # TP-local head layout.  Avoid materializing both the accumulated
            # [heads, seq, dim] result and a full-sequence head permutation.
            concatenated = self._chunked_sliding_attention_concatenated(
                q,
                k,
                v,
                config.sliding_window,
                config.head_dim,
            )
            q = None
            k = None
            v = None
            sdpa = None
        elif seq_len > PREFILL_SDPA_MAX_SEQ:
            # Full attention consumes K/V through the cache after the fill.
            # Release the prompt-sized source tensors, then stream each bounded
            # SDPA result directly into its final head-concatenated output.
            # This avoids both the accumulated-output concat and a giant final
            # permute that requires a late contiguous DRAM allocation.
            k.deallocate(True)
            v.deallocate(True)
            k = None
            v = None
            read_k_cache = self._paged_cache_read_view(k_cache, local_kv_heads, config.head_dim)
            read_v_cache = self._paged_cache_read_view(v_cache, local_kv_heads, config.head_dim)
            concatenated = self._chunked_full_attention_concatenated(
                q, read_k_cache, read_v_cache, page_table, user_id, config.head_dim
            )
            q = None
            sdpa = None
        else:
            sdpa = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=1.0,
                sliding_window_size=config.sliding_window if config.is_sliding else None,
                program_config=prefill_sdpa_program_config(config.head_dim, seq_len),
            )
        if q is not None:
            q.deallocate(True)
        if k is not None:
            k.deallocate(True)
        if v is not None:
            v.deallocate(True)
        local_heads = config.num_attention_heads // TP_SIZE
        if concatenated is None:
            concatenated = self._concatenate_heads(sdpa, num_heads=local_heads, head_dim=config.head_dim)
            sdpa.deallocate(True)
        if seq_len > PREFILL_SDPA_MAX_SEQ:
            output = self._chunked_attention_output_projection(concatenated, weights.o_proj, residual=residual)
            concatenated.deallocate(True)
            return output, residual is not None
        partial = ttnn.linear(concatenated, weights.o_proj)
        concatenated.deallocate(True)
        return (
            _tp_allreduce(
                partial,
                self.mesh_config,
                self.ccl_manager,
                # BFP8 async communication wins at M=1, while prefill pays for the
                # explicit casts and is faster with its native BF16 reduction.
                communication_dtype=self.prefill_communication_dtype,
                use_persistent_async=False,
                persistent_role="attention_o_prefill",
            ),
            False,
        )

    @staticmethod
    def _paged_cache_read_view(cache, num_kv_heads, head_dim):
        """Return a zero-copy HMA layer view for paged attention reads.

        vLLM may share one physical hybrid-cache tensor between sliding and
        global layers with different heads, block size, and head dimension.
        Paged fill writes raw tiles using the layer's effective block geometry;
        chunked SDPA must read those same tiles through that geometry rather
        than the allocation owner's shape.
        """
        block_size = effective_block_size(cache, head_dim, num_kv_heads)
        desired_shape = (cache.padded_shape[0], num_kv_heads, block_size, head_dim)
        if tuple(cache.padded_shape) == desired_shape:
            return cache
        if cache.dtype != ttnn.bfloat8_b:
            raise ValueError(f"HMA cache read views require BFP8 storage, got {cache.dtype}")
        if cache.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"HMA cache read views require TILE layout, got {cache.layout}")
        if block_size % ttnn.TILE_SIZE or head_dim % ttnn.TILE_SIZE:
            raise ValueError(f"HMA cache read view is not tile aligned: block_size={block_size}, head_dim={head_dim}")
        if math.prod(desired_shape) != math.prod(cache.padded_shape):
            raise ValueError(
                f"HMA cache view volume mismatch: allocation={tuple(cache.padded_shape)}, desired={desired_shape}"
            )
        # Every dimension is tile aligned and the paged-fill kernel stores the
        # layer view as a raw contiguous tile stream.  experimental.view only
        # changes the tensor spec; it neither copies nor changes vLLM ownership.
        return ttnn.experimental.view(cache, desired_shape)

    def _chunked_attention_output_projection(self, concatenated, output_weight, *, residual=None):
        """Project and reduce long attention outputs without a giant CCL temporary.

        The inter-device all-reduce needs a reduce-scatter workspace larger
        than its logical BF16 result.  Running it on the complete prompt can
        therefore fail even after SDPA/head concatenation has been streamed.
        Reduce bounded chunks and write their sharded TILE results directly
        into one final TILE tensor so slice-write does not perform a repeated
        full-output TILE/row-major conversion.
        """
        seq_len, hidden = concatenated.shape[-2], output_weight.shape[-1]
        output = ttnn.allocate_tensor_on_device(
            shape=(1, 1, seq_len, hidden),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        width_cores = 7
        if hidden % (width_cores * ttnn.TILE_SIZE):
            raise ValueError(f"attention output width {hidden} is not tile-divisible over {width_cores} cores")
        for start in range(0, seq_len, MLP_CHUNK):
            end = min(start + MLP_CHUNK, seq_len)
            chunk_rows = end - start
            chunk = ttnn.slice(concatenated, [0, 0, start, 0], [1, 1, end, concatenated.shape[-1]])
            partial = ttnn.linear(chunk, output_weight)
            chunk.deallocate(True)
            reduced = _tp_allreduce(
                partial,
                self.mesh_config,
                self.ccl_manager,
                communication_dtype=self.prefill_communication_dtype,
                use_persistent_async=False,
                persistent_role="attention_o_prefill_chunk",
            )
            chunk_output = reduced
            if residual is not None:
                # RMSNorm and the residual add are independent for every row.
                # Fuse them into this bounded chunk so long prefill never has
                # three full hidden-width prompt tensors alive together.
                normalized = self.layer.post_attention_layernorm.forward(reduced)
                reduced.deallocate(True)
                residual_chunk = ttnn.slice(residual, [0, 0, start, 0], [1, 1, end, hidden])
                chunk_output = ttnn.add(residual_chunk, normalized)
                residual_chunk.deallocate(True)
                normalized.deallocate(True)
            padded_chunk_rows = math.ceil(chunk_rows / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
            tile_rows = padded_chunk_rows // ttnn.TILE_SIZE
            height_cores = math.gcd(tile_rows, 8)
            shard_memory = ttnn.create_sharded_memory_config(
                shape=(padded_chunk_rows // height_cores, hidden // width_cores),
                core_grid=ttnn.CoreGrid(x=width_cores, y=height_cores),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            write_chunk = ttnn.to_memory_config(chunk_output, shard_memory)
            chunk_output.deallocate(True)
            ttnn.experimental.slice_write(
                write_chunk,
                output,
                [0, 0, start, 0],
                [1, 1, end, hidden],
                [1, 1, 1, 1],
            )
            write_chunk.deallocate(True)
        return output

    def _chunked_full_attention_concatenated(self, q, k_cache, v_cache, page_table, user_id, head_dim):
        """Stream full-attention chunks into the final TP-local head layout."""
        num_heads, seq_len = q.shape[1], q.shape[2]
        user_page_table = page_table
        owns_page_table = False
        if page_table.shape[0] > 1:
            user_page_table = ttnn.slice(page_table, [user_id, 0], [user_id + 1, page_table.shape[1]])
            owns_page_table = True
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        output_rm = ttnn.allocate_tensor_on_device(
            shape=(1, 1, seq_len, num_heads * head_dim),
            dtype=q.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for start in range(0, seq_len, FULL_ATTN_Q_CHUNK):
            end = min(start + FULL_ATTN_Q_CHUNK, seq_len)
            q_chunk = ttnn.slice(q, [0, 0, start, 0], [1, num_heads, end, head_dim])
            sdpa_chunk = ttnn.transformer.chunked_scaled_dot_product_attention(
                q_chunk,
                k_cache,
                v_cache,
                user_page_table,
                chunk_start_idx=start,
                scale=1.0,
                program_config=program_config,
            )
            q_chunk.deallocate(True)
            transposed = ttnn.permute(sdpa_chunk, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            sdpa_chunk.deallocate(True)
            chunk = ttnn.reshape(transposed, [1, 1, end - start, num_heads * head_dim])
            chunk_rm = ttnn.to_layout(chunk, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.experimental.slice_write(
                chunk_rm,
                output_rm,
                [0, 0, start, 0],
                [1, 1, end, num_heads * head_dim],
                [1, 1, 1, 1],
            )
            chunk_rm.deallocate(True)
            chunk.deallocate(True)
            if transposed.is_allocated():
                transposed.deallocate(True)
        if owns_page_table:
            user_page_table.deallocate(True)
        q.deallocate(True)
        output = ttnn.to_layout(output_rm, ttnn.TILE_LAYOUT)
        output_rm.deallocate(True)
        return output

    def _chunked_sliding_attention_concatenated(self, q, k, v, sliding_window, head_dim):
        """Stream sliding-attention chunks into the final TP-local head layout."""
        num_heads, num_kv_heads, seq_len = q.shape[1], k.shape[1], q.shape[2]
        history = math.ceil(sliding_window / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        output_rm = ttnn.allocate_tensor_on_device(
            shape=(1, 1, seq_len, num_heads * head_dim),
            dtype=q.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for start in range(0, seq_len, PREFILL_SLIDING_CHUNK_SIZE):
            end = min(start + PREFILL_SLIDING_CHUNK_SIZE, seq_len)
            slice_start = max(0, start - history)
            q_slice = ttnn.slice(q, [0, 0, slice_start, 0], [1, num_heads, end, head_dim])
            k_slice = ttnn.slice(k, [0, 0, slice_start, 0], [1, num_kv_heads, end, head_dim])
            v_slice = ttnn.slice(v, [0, 0, slice_start, 0], [1, num_kv_heads, end, head_dim])
            sdpa_chunk = ttnn.transformer.scaled_dot_product_attention(
                q_slice,
                k_slice,
                v_slice,
                is_causal=True,
                scale=1.0,
                sliding_window_size=sliding_window,
            )
            q_slice.deallocate(True)
            k_slice.deallocate(True)
            v_slice.deallocate(True)
            drop = start - slice_start
            if drop:
                padded_chunk = sdpa_chunk
                sdpa_chunk = ttnn.slice(
                    padded_chunk,
                    [0, 0, drop, 0],
                    [1, num_heads, end - slice_start, head_dim],
                )
                padded_chunk.deallocate(True)
            transposed = ttnn.permute(sdpa_chunk, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            sdpa_chunk.deallocate(True)
            chunk = ttnn.reshape(transposed, [1, 1, end - start, num_heads * head_dim])
            chunk_rm = ttnn.to_layout(chunk, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.experimental.slice_write(
                chunk_rm,
                output_rm,
                [0, 0, start, 0],
                [1, 1, end, num_heads * head_dim],
                [1, 1, 1, 1],
            )
            chunk_rm.deallocate(True)
            chunk.deallocate(True)
            if transposed.is_allocated():
                transposed.deallocate(True)
        q.deallocate(True)
        k.deallocate(True)
        v.deallocate(True)
        output = ttnn.to_layout(output_rm, ttnn.TILE_LAYOUT)
        output_rm.deallocate(True)
        return output

    def _chunked_mlp_residual(self, residual):
        """Stream the complete long-prefill MLP residual branch by row chunks."""
        seq_len, hidden = residual.shape[-2], residual.shape[-1]
        output = ttnn.allocate_tensor_on_device(
            shape=(1, 1, seq_len, hidden),
            dtype=self.residual_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        width_cores = 7
        if hidden % (width_cores * ttnn.TILE_SIZE):
            raise ValueError(f"MLP residual width {hidden} is not tile-divisible over {width_cores} cores")
        for start in range(0, seq_len, MLP_CHUNK):
            end = min(start + MLP_CHUNK, seq_len)
            chunk_rows = end - start
            residual_chunk = ttnn.slice(residual, [0, 0, start, 0], [1, 1, end, hidden])
            normed = self.layer.pre_feedforward_layernorm.forward(residual_chunk)
            residual_chunk.deallocate(True)
            mlp_output = self.layer.shared_mlp(normed)
            normed.deallocate(True)
            post_norm = self.layer.post_feedforward_layernorm.forward(mlp_output)
            mlp_output.deallocate(True)
            residual_chunk = ttnn.slice(residual, [0, 0, start, 0], [1, 1, end, hidden])
            combined = ttnn.add(residual_chunk, post_norm)
            residual_chunk.deallocate(True)
            post_norm.deallocate(True)
            if combined.dtype != self.residual_dtype:
                converted = ttnn.typecast(combined, self.residual_dtype)
                combined.deallocate(True)
                combined = converted
            if self.layer.layer_scalar != 1.0:
                scaled = ttnn.mul(combined, self.layer.layer_scalar)
                combined.deallocate(True)
                combined = scaled

            padded_chunk_rows = math.ceil(chunk_rows / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
            tile_rows = padded_chunk_rows // ttnn.TILE_SIZE
            height_cores = math.gcd(tile_rows, 8)
            shard_memory = ttnn.create_sharded_memory_config(
                shape=(padded_chunk_rows // height_cores, hidden // width_cores),
                core_grid=ttnn.CoreGrid(x=width_cores, y=height_cores),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            write_chunk = ttnn.to_memory_config(combined, shard_memory)
            combined.deallocate(True)
            ttnn.experimental.slice_write(
                write_chunk,
                output,
                [0, 0, start, 0],
                [1, 1, end, hidden],
                [1, 1, 1, 1],
            )
            write_chunk.deallocate(True)
        residual.deallocate(True)
        return output

    def _forward_device(
        self,
        hidden_states,
        *,
        rope_mats,
        page_table,
        kv_cache,
        is_decode,
        current_position=None,
        current_position_cache=None,
        token_index=None,
        batch_size=1,
        user_id=0,
        valid_seq_len=None,
    ):
        """Host-free replicated-residual composition for both layer kinds."""
        self.layer.shared_mlp.is_decode = is_decode
        residual = hidden_states
        normed = self.layer.input_layernorm.forward(hidden_states)
        attn_input = normed
        if not is_decode and batch_size > 1:
            attn_input = ttnn.reshape(normed, [batch_size, 1, normed.shape[-2] // batch_size, -1])
        if is_decode:
            attn_output = self._decode_attention_tp(
                attn_input,
                rope_mats=rope_mats,
                page_table=page_table,
                kv_cache=kv_cache,
                current_position=current_position,
                current_position_cache=current_position_cache,
                batch_size=batch_size,
            )
            attention_residual_fused = False
        else:
            attn_output, attention_residual_fused = self._prefill_attention_tp(
                attn_input,
                source_hidden_states=normed,
                rope_mats=rope_mats,
                page_table=page_table,
                kv_cache=kv_cache,
                user_id=user_id,
                valid_seq_len=valid_seq_len,
                residual=residual if batch_size == 1 else None,
            )
        if is_decode:
            normed.deallocate(True)
        if attention_residual_fused:
            residual.deallocate(True)
            hidden_states = attn_output
        else:
            attn_output = self.layer.post_attention_layernorm.forward(attn_output)
            if not is_decode and batch_size > 1:
                residual = ttnn.reshape(residual, [1, 1, residual.shape[-2] * residual.shape[-3], -1])
            hidden_states = ttnn.add(residual, attn_output)
            attn_output.deallocate(True)

        if not is_decode and batch_size == 1 and hidden_states.shape[-2] > MLP_CHUNK:
            return self._chunked_mlp_residual(hidden_states)

        residual = hidden_states
        normed = self.layer.pre_feedforward_layernorm.forward(hidden_states)
        if not is_decode and normed.shape[-2] > MLP_CHUNK:
            outputs = []
            for start in range(0, normed.shape[-2], MLP_CHUNK):
                end = min(start + MLP_CHUNK, normed.shape[-2])
                chunk = ttnn.slice(normed, [0, 0, start, 0], [1, 1, end, normed.shape[-1]])
                outputs.append(self.layer.shared_mlp(chunk))
                chunk.deallocate(True)
            # Every chunk has consumed the full normalization result.  Release
            # it before concat allocates another prompt-sized BF16 tensor.
            normed.deallocate(True)
            mlp_output = ttnn.concat(outputs, dim=2)
            for output in outputs:
                output.deallocate(True)
        else:
            mlp_output = self.layer.shared_mlp(normed)
            normed.deallocate(True)
        hidden_states = self.layer.post_feedforward_layernorm.forward(mlp_output)
        mlp_output.deallocate(True)
        combined = ttnn.add(residual, hidden_states)
        residual.deallocate(True)
        hidden_states.deallocate(True)
        if combined.dtype != self.residual_dtype:
            converted = ttnn.typecast(combined, self.residual_dtype)
            combined.deallocate(True)
            combined = converted
        if self.layer.layer_scalar != 1.0:
            scaled = ttnn.mul(combined, self.layer.layer_scalar)
            combined.deallocate(True)
            combined = scaled
        return combined


__all__ = [
    "MultichipDecoder",
    "MultichipDecoderTimings",
    "PAGE_BLOCK_SIZE",
    "QKV_DECODE_OUTPUT_CORES",
    "release_multichip_decoder_resources",
    "TARGET_MESH_SHAPE",
    "TP_SIZE",
]
