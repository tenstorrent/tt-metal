# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exact TP4 terminal hyper-connection mixer for backbone and MTP."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.checkpoint import INDEX_SHA256, Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.config import Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.tt.model import Qwen38FinalMixerWeights
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    MESH_SHAPE,
    Qwen38MeshContract,
    TensorPlacement,
    replicate_tensor_2d_mesh_mapper,
    require_lane_count,
    same_buffer,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.decode_matmul import (
    dense_dtype_tag,
    dense_math_fidelity_name,
    dram_sharded_matmul_configs,
    dram_sharded_weight_memory_config,
)

TP_AXIS = 1
TP_SIZE = 4
HIDDEN_SIZE = 2560
LOCAL_HIDDEN_SIZE = HIDDEN_SIZE // TP_SIZE
RESIDUAL_BRANCHES = 4
RESIDUAL_RANK = 320
RMS_NORM_EPS = 1.0e-6
RESIDUAL_LOCAL_SHAPE = (1, RESIDUAL_BRANCHES, 1, LOCAL_HIDDEN_SIZE)
# The branch-major residual read as one (branch, local hidden) row; the tile
# pages coincide, so the view moves no data (see ttnn/gr.py).
FLAT_LOCAL_WIDTH = RESIDUAL_BRANCHES * LOCAL_HIDDEN_SIZE
FLAT_LOCAL_SHAPE = (1, 1, 1, FLAT_LOCAL_WIDTH)
OUTPUT_LOCAL_SHAPE = (1, 1, 1, LOCAL_HIDDEN_SIZE)
# Resident weights: local shape, dtype, mesh shard dim.  The two matmul weights
# are DRAM width-sharded across the banks for the decode program.
DEVICE_WEIGHTS = {
    "norm_scale": (RESIDUAL_LOCAL_SHAPE, ttnn.float32, 3),
    "down": ((1, 1, FLAT_LOCAL_WIDTH, RESIDUAL_RANK), ttnn.bfloat16, 2),
    "up": ((1, 1, RESIDUAL_RANK, FLAT_LOCAL_WIDTH), ttnn.bfloat16, 3),
}
MATMUL_WEIGHTS = ("down", "up")


def _shape(tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def _padded_shape(tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.padded_shape)


def _tensor_key(tensor) -> tuple[str, int]:
    return (str(tensor.device()), int(tensor.tensor_id))


def _deallocate(*tensors) -> None:
    for tensor in tensors:
        if tensor is not None:
            ttnn.deallocate(tensor)


def _padded_shape(tensor) -> tuple[int, ...]:
    return tuple(int(item) for item in tensor.padded_shape)


def _tensor_key(tensor) -> tuple[str, int]:
    tensor_id = getattr(tensor, "tensor_id", None)
    if callable(tensor_id):
        tensor_id = tensor_id()
    return ("ttnn", int(tensor_id)) if tensor_id is not None else ("python", id(tensor))


def _cache_directory(
    root: str | Path,
    checkpoint: Qwen38Checkpoint,
    mesh_contract: Qwen38MeshContract,
    namespace: str,
    tt_metal_sha: str,
) -> Path:
    if namespace not in {"backbone", "mtp"}:
        raise ValueError(f"unsupported final-mixer namespace {namespace!r}")
    if len(tt_metal_sha) != 40 or any(character not in "0123456789abcdef" for character in tt_metal_sha):
        raise ValueError(f"tt_metal_sha must be a lowercase 40-hex revision, got {tt_metal_sha!r}")
    physical = "-".join(str(value) for value in mesh_contract.physical_ids)
    path = (
        Path(root).resolve()
        / "final-mixer"
        / namespace
        / f"index-{INDEX_SHA256}"
        / f"config-{checkpoint.config.config_sha256}"
        / f"tt-metal-{tt_metal_sha}"
        / f"mesh-1x4-physical-{physical}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prepare(source: Qwen38FinalMixerWeights) -> dict[str, torch.Tensor]:
    """Put every weight in its device-local decode layout before mesh sharding.

    As in ttnn/gr.py, the sharded axis of each matmul weight is ordered
    (device, branch, local hidden): ``down`` stacks its input-branch blocks
    along K, ``up`` its output-branch blocks along N, so each projection is one
    matmul on the flat residual row.
    """

    config = source.config
    exact = {
        "hidden_size": HIDDEN_SIZE,
        "residual_branches": RESIDUAL_BRANCHES,
        "residual_rank": RESIDUAL_RANK,
        "rms_norm_eps": RMS_NORM_EPS,
    }
    for name, expected in exact.items():
        actual = getattr(config, name)
        if actual != expected:
            raise ValueError(f"pinned final mixer requires {name}={expected!r}, got {actual!r}")
    mesh_ranges = tuple((index * LOCAL_HIDDEN_SIZE, (index + 1) * LOCAL_HIDDEN_SIZE) for index in range(TP_SIZE))
    if tuple(source.placement.hidden_ranges) != mesh_ranges:
        raise ValueError(
            f"pinned final mixer requires hidden ranges {mesh_ranges}, got {source.placement.hidden_ranges}"
        )
    norm = source.norm.reshape(RESIDUAL_BRANCHES, HIDDEN_SIZE)
    shards = [source.device_shard(device) for device in range(TP_SIZE)]
    return {
        # Branch-major to match the persistent residual layout; the 1/4 branch
        # mean is folded in (exact exponent shift), as in ttnn/gr.py.
        "norm_scale": ((1.0 + norm.float()) / RESIDUAL_BRANCHES)
        .reshape(1, RESIDUAL_BRANCHES, 1, HIDDEN_SIZE)
        .contiguous(),
        # [rank, input_branch, local hidden] -> [(input_branch, local hidden), rank]
        "down": torch.cat(
            [shard.down.permute(1, 2, 0).reshape(FLAT_LOCAL_WIDTH, RESIDUAL_RANK) for shard in shards],
            dim=0,
        ).reshape(1, 1, TP_SIZE * FLAT_LOCAL_WIDTH, RESIDUAL_RANK),
        # [output_branch, local hidden, rank] -> [rank, (output_branch, local hidden)]
        "up": torch.cat(
            [shard.up.permute(2, 0, 1).reshape(RESIDUAL_RANK, FLAT_LOCAL_WIDTH) for shard in shards],
            dim=1,
        ).reshape(1, 1, RESIDUAL_RANK, TP_SIZE * FLAT_LOCAL_WIDTH),
    }


@dataclass(frozen=True)
class Qwen38TTNNFinalMixerWeights:
    norm_scale: Any
    down: Any
    up: Any
    replicated_anchor: Any
    namespace: Literal["backbone", "mtp"]
    weight_dtype: Any = ttnn.bfloat16  # the down / up matmul weights' dtype (QWEN38_DENSE_WEIGHT_DTYPE)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Qwen38Checkpoint,
        placement: Qwen38Placement,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        cache_root: str | Path,
        *,
        namespace: Literal["backbone", "mtp"] = "backbone",
        tt_metal_sha: str,
        weight_dtype=None,
    ) -> "Qwen38TTNNFinalMixerWeights":
        mesh_contract.validate_mesh(mesh_device)
        if weight_dtype is None:
            weight_dtype = ttnn.bfloat16  # the production path
        weight_tag = dense_dtype_tag(weight_dtype)
        source = (
            Qwen38FinalMixerWeights.from_checkpoint(checkpoint, placement)
            if namespace == "backbone"
            else Qwen38FinalMixerWeights.from_mtp_checkpoint(checkpoint, placement)
        )
        prepared = _prepare(source)
        cache = _cache_directory(cache_root, checkpoint, mesh_contract, namespace, tt_metal_sha)
        hidden_output_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3))
        hidden_input_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 2))

        def upload(value: torch.Tensor, name: str, dtype, mapper, memory_config):
            return ttnn.as_tensor(
                value.contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=memory_config,
                mesh_mapper=mapper,
                cache_file_name=cache / name,
            )

        result = cls(
            # "-q-bm": the quarter-scaled branch-major gamma must never load a
            # stale cache written by the unscaled or [1,1,4,H] row layouts.
            norm_scale=upload(
                prepared["norm_scale"],
                "norm-scale-q-bm.fp32",
                ttnn.float32,
                hidden_output_mapper,
                ttnn.DRAM_MEMORY_CONFIG,
            ),
            # The matmul weights are DRAM width-sharded for the decode program;
            # the renamed tensorbins deliberately orphan the batched
            # interleaved caches.
            down=upload(
                prepared["down"],
                "down-dram-sharded.bf16" if weight_dtype == ttnn.bfloat16 else f"down-dram-sharded.{weight_tag}",
                weight_dtype,
                hidden_input_mapper,
                dram_sharded_weight_memory_config(mesh_device, FLAT_LOCAL_WIDTH, RESIDUAL_RANK),
            ),
            up=upload(
                prepared["up"],
                "up-dram-sharded.bf16" if weight_dtype == ttnn.bfloat16 else f"up-dram-sharded.{weight_tag}",
                weight_dtype,
                hidden_output_mapper,
                dram_sharded_weight_memory_config(mesh_device, RESIDUAL_RANK, FLAT_LOCAL_WIDTH),
            ),
            replicated_anchor=ttnn.from_torch(
                torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh_device),
            ),
            namespace=namespace,
            weight_dtype=weight_dtype,
        )
        result.validate(mesh_contract)
        return result

    def validate(self, mesh_contract: Qwen38MeshContract) -> None:
        for name, (shape, dtype, shard_dim) in DEVICE_WEIGHTS.items():
            tensor = getattr(self, name)
            if name in MATMUL_WEIGHTS:
                dtype = self.weight_dtype
            if _shape(tensor) != shape or tensor.dtype != dtype or tensor.layout != ttnn.TILE_LAYOUT:
                raise RuntimeError(
                    f"final mixer {name} must be TILE {dtype} {shape}, got {tensor.layout} {tensor.dtype} {_shape(tensor)}"
                )
            if (name in MATMUL_WEIGHTS) != (
                tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
            ):
                raise RuntimeError(f"final mixer {name} has memory layout {tensor.memory_config().memory_layout}")
            mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=shard_dim)
        mesh_contract.validate_tensor(self.replicated_anchor, placement=TensorPlacement.REPLICATED)

    def deallocate(self) -> None:
        _deallocate(self.norm_scale, self.down, self.up, self.replicated_anchor)


class Qwen38TTNNFinalMixer:
    """Read the final four-branch residual into one hidden-width shard."""

    _fused_forward = None  # QWEN38_FUSED=final_mixer binds ttnn/fused/final_mixer per instance; the body is the chain

    def __init__(
        self,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        weights: Qwen38TTNNFinalMixerWeights,
        *,
        collective_topology=None,
    ) -> None:
        mesh_contract.validate_mesh(mesh_device)
        weights.validate(mesh_contract)
        self.mesh_device = mesh_device
        self.mesh_contract = mesh_contract
        self.weights = weights
        self.collective_topology = collective_topology or ttnn.Topology.Linear
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # The down / up linears run the fidelity of their weight format (decode_matmul: HiFi4 for bf16, HiFi2 for
        # bf8, LoFi for bf4); the norms keep compute_config.
        self.weight_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, dense_math_fidelity_name(weights.weight_dtype)),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # Decode matmul configs, the GR read pattern: K=2560 over five cores
        # (eight-tile K blocks), K=320 over two cores (five-tile K blocks).
        self.down_act_memory_config, self.down_program_config = dram_sharded_matmul_configs(
            mesh_device, FLAT_LOCAL_WIDTH, RESIDUAL_RANK, num_cores=5
        )
        self.up_act_memory_config, self.up_program_config = dram_sharded_matmul_configs(
            mesh_device, RESIDUAL_RANK, FLAT_LOCAL_WIDTH, num_cores=2
        )
        # The gamma multiply runs on the flat row (the GR pattern); this view
        # shares the resident weight's buffer.
        self.norm_scale_flat = ttnn.experimental.view(weights.norm_scale, FLAT_LOCAL_SHAPE)
        # QWEN38_FUSED=final_mixer: the fused programs (F2's stats/normalize/down/gate and the composite-order low-rank
        # sum) bind at construction; the pinned chain below stays the class body.
        from models.demos.blackhole.qwen38_flash_next.ttnn import fused as fused_kernels

        if fused_kernels.enabled("final_mixer") and weights.weight_dtype != ttnn.bfloat16:
            raise ValueError(
                "QWEN38_FUSED=final_mixer streams BF16 mixer weights through BF16 circular buffers; "
                f"these final mixer weights are {weights.weight_dtype}"
            )
        if fused_kernels.enabled("final_mixer"):
            self._fused_forward = functools.partial(fused_kernels.kernel("final_mixer").fused, self)

    def _normalize(self, residual):
        stats = ttnn.rms_norm_pre_all_gather(
            residual,
            # Avoid the standard all-gather's one-page Fabric scatter-state
            # initializer defect. BF16 produces the same two-page statistics
            # transfer already exercised by the GR and PLE paths.
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        stats = ttnn.reshape(stats, (1, RESIDUAL_BRANCHES, 1, 32))
        self.mesh_contract.validate_tensor(stats, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        gathered = ttnn.all_gather(
            stats,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _deallocate(stats)
        self.mesh_contract.validate_tensor(gathered, placement=TensorPlacement.REPLICATED)
        # The fused post kernel broadcasts one gamma tile row-block over every
        # branch block of the branch-major residual, so apply the per-branch
        # scale elementwise after the unscaled RMS unit (the GR pattern).  The
        # scale already carries the 1/4 branch mean.  The multiply reads both
        # operands through their flat views and writes the down matmul's
        # five-core activation shard directly.
        unit = ttnn.rms_norm_post_all_gather(
            residual,
            gathered,
            epsilon=RMS_NORM_EPS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
            dtype=ttnn.bfloat16,
        )
        _deallocate(gathered)
        normalized_ws = ttnn.multiply(
            ttnn.experimental.view(unit, FLAT_LOCAL_SHAPE),
            self.norm_scale_flat,
            dtype=ttnn.bfloat16,
            memory_config=self.down_act_memory_config,
        )
        _deallocate(unit)
        self.mesh_contract.validate_tensor(normalized_ws, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if _shape(normalized_ws) != FLAT_LOCAL_SHAPE or normalized_ws.memory_config() != self.down_act_memory_config:
            raise RuntimeError(
                f"final mixer normalization produced {_shape(normalized_ws)} {normalized_ws.memory_config()}"
            )
        return normalized_ws

    def __call__(self, residual):
        if _shape(residual) != RESIDUAL_LOCAL_SHAPE or residual.dtype != ttnn.bfloat16:
            raise ValueError(f"final mixer input must be TILE BF16 {RESIDUAL_LOCAL_SHAPE}")
        self.mesh_contract.validate_tensor(residual, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if self._fused_forward is not None:
            return self._fused_forward(residual)
        # The flat row in the down matmul's activation layout; the gate
        # multiply reads the same shard at the end.
        normalized_ws = self._normalize(residual)
        # One K=2560 matmul on the flat row sums the branches.
        partial_ws = ttnn.linear(
            normalized_ws,
            self.weights.down,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.down_program_config,
            dtype=ttnn.float32,
            compute_kernel_config=self.weight_compute_config,
        )
        partial = ttnn.to_memory_config(partial_ws, ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(partial_ws)
        self.mesh_contract.mark_local_partial(
            partial,
            replicated_reference=self.weights.replicated_anchor,
            expected_shape=(1, 1, 1, RESIDUAL_RANK),
        )
        down = ttnn.all_reduce(
            partial,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.collective_topology,
        )
        _deallocate(partial)
        self.mesh_contract.validate_tensor(down, placement=TensorPlacement.REPLICATED)
        # One BF16 rounding of the TP sum; the row already carries the 1/4
        # branch mean through the folded gamma.
        down_bf16 = ttnn.typecast(down, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(down)
        # The SiLU writes the up matmul's two-core activation shard directly.
        low_rank_ws = ttnn.silu(down_bf16, memory_config=self.up_act_memory_config)
        _deallocate(down_bf16)
        if low_rank_ws.memory_config() != self.up_act_memory_config:
            raise RuntimeError(f"final mixer low-rank row has memory config {low_rank_ws.memory_config()}")
        self.mesh_contract.validate_tensor(low_rank_ws, placement=TensorPlacement.REPLICATED)
        # The up weight stacks its output branches along N: one matmul yields
        # the flat branch-major gate.
        up_ws = ttnn.linear(
            low_rank_ws,
            self.weights.up,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.up_program_config,
            compute_kernel_config=self.weight_compute_config,
        )
        _deallocate(low_rank_ws)
        up_flat = ttnn.to_memory_config(up_ws, ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(up_ws)
        self.mesh_contract.validate_tensor(up_flat, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if _shape(up_flat) != FLAT_LOCAL_SHAPE:
            raise RuntimeError(f"final mixer up projection has unexpected shape {_shape(up_flat)}")
        # The gate multiplies the normalized shard in place: sharded lhs,
        # interleaved rhs, interleaved product.  The branch-major view owns
        # gated_flat's buffer from here on.
        gate_flat = ttnn.sigmoid(up_flat, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(up_flat)
        gated_flat = ttnn.multiply(normalized_ws, gate_flat, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(gate_flat, normalized_ws)
        gated = ttnn.experimental.view(gated_flat, RESIDUAL_LOCAL_SHAPE)
        # Branch mean without a permute: fast_reduce_nc sums the four branch
        # tile row-blocks (generic dim-1 reduce would transpose); the 1/4 is
        # already in normalized_ws.  fast_reduce_nc reports the 32-row tile
        # padding as its logical row count; restore the logical shape without
        # moving data.
        gated_sum = ttnn.experimental.fast_reduce_nc(
            gated,
            dims=[1],
            output=None,
            compute_kernel_config=self.compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        output = ttnn.reshape(gated_sum, OUTPUT_LOCAL_SHAPE, gated_sum.padded_shape)
        _deallocate(gated)
        self.mesh_contract.validate_tensor(output, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if _shape(output) != OUTPUT_LOCAL_SHAPE or output.dtype != ttnn.bfloat16:
            raise RuntimeError(f"final mixer output must be BF16 {OUTPUT_LOCAL_SHAPE}, got {_shape(output)}")
        return output

    # ------------------------------------------------------------------ rows path (batched lanes)

    def _validate_rows(self, tensor, expected: tuple[int, ...], *, label: str) -> None:
        if _shape(tensor) != expected or tensor.layout != ttnn.TILE_LAYOUT or tensor.dtype != ttnn.bfloat16:
            raise ValueError(
                f"{label} must be TILE BF16 {expected}, got {tensor.layout} {tensor.dtype} {_shape(tensor)}"
            )
        self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)

    def rows(self, residual_rows, *, flat_views: bool = False):
        """``__call__`` over ``rows`` branch-major residual rows ``[1,4,rows,640]`` (1..32, row u = lane u) ->
        ``[1,1,rows,640]``: the same ops row for row (every op is per row or per element; the DRAM-sharded matmuls
        read one tile row either way), the two zero-copy views replaced by real permutes (the GR rows pattern) unless
        ``flat_views`` (the same pages under ``ttnn.experimental.view``, as the 1-row body and GR ``read_rows``).  The
        1-row body above is untouched."""

        shape = _shape(residual_rows)
        if len(shape) != 4:
            raise ValueError(f"final mixer residual rows must be rank 4 [1,4,rows,640], got {list(shape)}")
        rows = require_lane_count(shape[2], label="final mixer residual rows")
        residual_shape = (1, RESIDUAL_BRANCHES, rows, LOCAL_HIDDEN_SIZE)
        flat_shape = (1, 1, rows, FLAT_LOCAL_WIDTH)
        flat_padded = (1, 1, ttnn.TILE_SIZE, FLAT_LOCAL_WIDTH)
        block_shape = (1, 1, rows, LOCAL_HIDDEN_SIZE)
        self._validate_rows(residual_rows, residual_shape, label="final mixer residual rows")
        dram = ttnn.DRAM_MEMORY_CONFIG
        stats = ttnn.rms_norm_pre_all_gather(
            residual_rows, dtype=ttnn.bfloat16, memory_config=dram, compute_kernel_config=self.compute_config
        )
        stats = ttnn.reshape(stats, (1, RESIDUAL_BRANCHES, rows, 32))
        self.mesh_contract.validate_tensor(stats, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        gathered = ttnn.all_gather(stats, dim=3, cluster_axis=TP_AXIS, memory_config=dram)
        _deallocate(stats)
        self.mesh_contract.validate_tensor(gathered, placement=TensorPlacement.REPLICATED)
        unit = ttnn.rms_norm_post_all_gather(
            residual_rows,
            gathered,
            epsilon=RMS_NORM_EPS,
            memory_config=dram,
            compute_kernel_config=self.compute_config,
            dtype=ttnn.bfloat16,
        )
        _deallocate(gathered)
        self._validate_rows(unit, residual_shape, label="final mixer rows RMS unit")
        # Branch-major -> token-major -> one flat (branch, local hidden) row per lane.
        if flat_views:
            unit_flat = ttnn.experimental.view(unit, flat_shape)  # the view owns unit's buffer
        else:
            unit_tokens = ttnn.permute(unit, (0, 2, 1, 3), memory_config=dram)
            _deallocate(unit)
            unit_flat = ttnn.reshape(unit_tokens, flat_shape)
            if not same_buffer(
                unit_flat, unit_tokens
            ):  # a copy today (the last dim changes); never free a view's source
                _deallocate(unit_tokens)
        if _shape(unit_flat) != flat_shape or _padded_shape(unit_flat) != flat_padded:
            raise RuntimeError(
                f"final mixer rows flat unit must be {flat_shape} backed by {flat_padded}, "
                f"got {_shape(unit_flat)}/{_padded_shape(unit_flat)}"
            )
        normalized_ws = ttnn.multiply(
            unit_flat, self.norm_scale_flat, dtype=ttnn.bfloat16, memory_config=self.down_act_memory_config
        )
        _deallocate(unit_flat)
        self.mesh_contract.validate_tensor(normalized_ws, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if _shape(normalized_ws) != flat_shape or normalized_ws.memory_config() != self.down_act_memory_config:
            raise RuntimeError(
                f"final mixer rows normalization produced {_shape(normalized_ws)} {normalized_ws.memory_config()}"
            )
        partial_ws = ttnn.linear(
            normalized_ws,
            self.weights.down,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.down_program_config,
            dtype=ttnn.float32,
            compute_kernel_config=self.weight_compute_config,
        )
        partial = ttnn.to_memory_config(partial_ws, dram)
        _deallocate(partial_ws)
        self.mesh_contract.mark_local_partial(
            partial, replicated_reference=self.weights.replicated_anchor, expected_shape=(1, 1, rows, RESIDUAL_RANK)
        )
        down = ttnn.all_reduce(partial, cluster_axis=TP_AXIS, memory_config=dram, topology=self.collective_topology)
        _deallocate(partial)
        self.mesh_contract.validate_tensor(down, placement=TensorPlacement.REPLICATED)
        down_bf16 = ttnn.typecast(down, ttnn.bfloat16, memory_config=dram)
        _deallocate(down)
        low_rank_ws = ttnn.silu(down_bf16, memory_config=self.up_act_memory_config)
        _deallocate(down_bf16)
        if low_rank_ws.memory_config() != self.up_act_memory_config:
            raise RuntimeError(f"final mixer rows low-rank rows have memory config {low_rank_ws.memory_config()}")
        self.mesh_contract.validate_tensor(low_rank_ws, placement=TensorPlacement.REPLICATED)
        up_ws = ttnn.linear(
            low_rank_ws,
            self.weights.up,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.up_program_config,
            compute_kernel_config=self.weight_compute_config,
        )
        _deallocate(low_rank_ws)
        up_flat = ttnn.to_memory_config(up_ws, dram)
        _deallocate(up_ws)
        self.mesh_contract.validate_tensor(up_flat, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if _shape(up_flat) != flat_shape:
            raise RuntimeError(f"final mixer rows up projection has shape {_shape(up_flat)}, expected {flat_shape}")
        gate_flat = ttnn.sigmoid(up_flat, memory_config=dram)
        _deallocate(up_flat)
        gated_flat = ttnn.multiply(normalized_ws, gate_flat, memory_config=dram)
        _deallocate(gate_flat, normalized_ws)
        # Flat rows -> token-major -> branch-major, then the branch mean (the 1/4 is in norm_scale).
        if flat_views:
            gated = ttnn.experimental.view(gated_flat, residual_shape)  # owns gated_flat's buffer
        else:
            gated_tokens = ttnn.reshape(gated_flat, (1, rows, RESIDUAL_BRANCHES, LOCAL_HIDDEN_SIZE))
            if not same_buffer(gated_tokens, gated_flat):  # a copy today (the last dim changes)
                _deallocate(gated_flat)
            gated = ttnn.permute(gated_tokens, (0, 2, 1, 3), memory_config=dram)
            _deallocate(gated_tokens)
        self._validate_rows(gated, residual_shape, label="final mixer rows gated unit")
        gated_sum = ttnn.experimental.fast_reduce_nc(
            gated, dims=[1], output=None, compute_kernel_config=self.compute_config, memory_config=dram
        )
        _deallocate(gated)
        # fast_reduce_nc may report the tile padding as its row count (the 1-row body's reshape); restore the rows.
        output = (
            gated_sum
            if _shape(gated_sum) == block_shape
            else ttnn.reshape(gated_sum, block_shape, gated_sum.padded_shape)
        )
        self._validate_rows(output, block_shape, label="final mixer rows output")
        return output


__all__ = ["Qwen38TTNNFinalMixer", "Qwen38TTNNFinalMixerWeights"]
