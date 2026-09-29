# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel Qwen3.8-Flash-Next Gated DeltaNet.

This is the device implementation of the pinned ``Qwen4Exp`` linear-attention
block, not the CPU oracle.  It deliberately has a narrow boundary:

* input and output are hidden-width shards on the columns of an exact 1x4 mesh;
* Q/K heads (16) and value heads (48) are split 4/12 per device;
* convolution and recurrent state are genuinely head sharded, never represented
  by four replicated tensors that happen to contain different values;
* the recurrent state remains FP32 and is updated in preallocated buffers;
* the Qwen3.8 output gate is ``sigmoid(z)`` (Qwen3.6 uses ``silu(z)``);
* the row-parallel output is reduced and scattered back to H/4.

The ordinary-decode path is the primary bring-up path.  ``forward_prefill`` is
an exact device-only serial baseline over that same transition.  A future
chunk-prefill optimization may replace it only after matching this state and
output contract.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.checkpoint import INDEX_SHA256, Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.tt.gdn import Qwen38GDNWeights
from models.demos.blackhole.qwen38_flash_next.ttnn import fused, prefill_glue
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    LONG_CHUNK_ROWS,
    MESH_SHAPE,
    Qwen38MeshContract,
    TensorPlacement,
    is_slab_rows,
    replicate_tensor_2d_mesh_mapper,
    require_lane_count,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.decode_matmul import (
    DENSE_DTYPE_TAGS,
    TWO_READER_QUALIFIED_DTYPES,
    dense_dtype_tag,
    dense_math_fidelity_name,
    dram_sharded_matmul_configs,
    dram_sharded_row_tiles,
    dram_sharded_weight_memory_config,
    prefill_matmul_program_config,
    validate_decode_dram_workers,
    validate_dram_sharded_weight,
    weight_layout_tag,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.prefill_dense import Qwen38TTNNPrefillDense, prefill_linear

TP_SIZE = 4
TP_AXIS = 1
HIDDEN_SIZE = 2560
HIDDEN_SIZE_PER_DEVICE = HIDDEN_SIZE // TP_SIZE
QK_HEADS = 16
QK_HEADS_PER_DEVICE = QK_HEADS // TP_SIZE
VALUE_HEADS = 48
VALUE_HEADS_PER_DEVICE = VALUE_HEADS // TP_SIZE
HEAD_DIM = 128
QK_WIDTH = QK_HEADS * HEAD_DIM
QK_WIDTH_PER_DEVICE = QK_HEADS_PER_DEVICE * HEAD_DIM
VALUE_WIDTH = VALUE_HEADS * HEAD_DIM
VALUE_WIDTH_PER_DEVICE = VALUE_HEADS_PER_DEVICE * HEAD_DIM
QKV_WIDTH = 2 * QK_WIDTH + VALUE_WIDTH
QKV_WIDTH_PER_DEVICE = 2 * QK_WIDTH_PER_DEVICE + VALUE_WIDTH_PER_DEVICE
CONV_KERNEL_SIZE = 4
QK_REPEAT_FACTOR = VALUE_HEADS // QK_HEADS
# Checkpoint order [q|k|v|z|a|b] packs to 4120 columns.  On device the packed
# weight keeps that order but starts ``a`` and ``b`` on tile boundaries, with
# zero weight columns between, so the decode split is four tile-aligned slices.
QKVZAB_WIDTH_PER_DEVICE = QKV_WIDTH_PER_DEVICE + VALUE_WIDTH_PER_DEVICE + 2 * VALUE_HEADS_PER_DEVICE
A_COLUMN = QKV_WIDTH_PER_DEVICE + VALUE_WIDTH_PER_DEVICE
B_COLUMN = A_COLUMN + ttnn.TILE_SIZE
PROJECTION_WIDTH_PER_DEVICE = B_COLUMN + ttnn.TILE_SIZE
RMS_NORM_EPS = 1.0e-6
# FLA l2 normalization of q/k: rms_norm with eps / head_dim, then head_dim ** -0.5.
QK_L2_NORM_EPS = 1.0e-6
# Multi-row (MTP verify) path: the chunk kernel is handed exactly one full chunk of
# CHUNK_SIZE rows; the rows past the real ones are zeroed on device.  The FIR
# history is the CONV_KERNEL_SIZE - 1 rows before the first new row.
CHUNK_SIZE = ttnn.TILE_SIZE
CONV_HISTORY_ROWS = CONV_KERNEL_SIZE - 1
CONV_WINDOW_ROWS = CONV_HISTORY_ROWS + CHUNK_SIZE
# The rows path keeps the FIR history in a full 32-row tile (rows 0..2 hold the history, rows 3..31
# are never read) so ``[history | qkv]`` is a tile-aligned concat of two tiles; every row read from
# that 64-row window (the FIR taps, the next history) is an exact 0/1 selection matmul against a
# constant, never a row-unaligned slice.  Logical window row m (m = 0..34: the three history rows,
# then the CHUNK_SIZE new rows) sits at buffer row ``_window_buffer_row(m)``.
CONV_WINDOW_TILE_ROWS = 2 * CHUNK_SIZE
CHUNK_QUADRANT_MASK_WIDTH = 3 * ttnn.TILE_SIZE
# The long prefill chunk hands the kernel LONG_CHUNK_ROWS rows (four 32-row chunks in one call, the state
# carried between them inside the scan); its window is [history tile | 128 qkv rows] = 160 rows.
LONG_CONV_WINDOW_TILE_ROWS = CHUNK_SIZE + LONG_CHUNK_ROWS


def rows_tile_count(rows: int) -> int:
    """Kernel rows of the rows path for ``rows`` real rows: one tile up to 32, the long chunk's 128."""

    if 1 <= rows <= CHUNK_SIZE:
        return CHUNK_SIZE
    if rows == LONG_CHUNK_ROWS or is_slab_rows(rows):
        return rows
    raise ValueError(f"GDN rows path admits 1..{CHUNK_SIZE} rows, {LONG_CHUNK_ROWS} or a slab row count, got {rows}")


def _window_buffer_row(logical_row: int) -> int:
    if logical_row < CONV_HISTORY_ROWS:
        return logical_row
    return CHUNK_SIZE + logical_row - CONV_HISTORY_ROWS


def rows_window_select_tiles(rows: int) -> dict[str, torch.Tensor]:
    """Host images of the rows path's exact 0/1 selection matrices (one nonzero per selected element).

    ``conv_taps[t]`` ``[T, 32 + T]`` (T the kernel rows): row j selects logical window row t + j (FIR tap
    t, t = 0..2; tap 3 is the new rows themselves).  ``history_select_stack`` ``[32, 2048]``: row a is the
    flattened ``[32, 64]`` select whose rows 0..2 pick logical window rows a + 1 .. a + 3 (the next history
    after committing a + 1 rows; rows 3..31 are zero); the long chunk commits every row, so it carries the
    one constant ``history_select_full`` ``[32, 32 + T]`` (rows 0..2 pick logical window rows T .. T + 2)
    instead.  ``qk_expand`` ``[512, 1536]``: the GQA expansion, key head h to value heads 3h .. 3h + 2 (one
    1.0 per output column), so the chunk kernel sees H = HV = 12 heads and skips its own repeat_interleave.
    """

    kernel_rows = rows_tile_count(rows)
    window_rows = CHUNK_SIZE + kernel_rows
    taps = torch.zeros(CONV_HISTORY_ROWS, kernel_rows, window_rows)
    for tap in range(CONV_HISTORY_ROWS):
        for row in range(kernel_rows):
            taps[tap, row, _window_buffer_row(tap + row)] = 1.0
    stack = torch.zeros(CHUNK_SIZE, CHUNK_SIZE * CONV_WINDOW_TILE_ROWS)
    for accepted in range(min(rows, CHUNK_SIZE)):
        select = torch.zeros(CHUNK_SIZE, CONV_WINDOW_TILE_ROWS)
        for index in range(CONV_HISTORY_ROWS):
            select[index, _window_buffer_row(accepted + 1 + index)] = 1.0
        stack[accepted] = select.reshape(-1)
    history_select_full = torch.zeros(CHUNK_SIZE, window_rows)
    for index in range(CONV_HISTORY_ROWS):
        history_select_full[index, _window_buffer_row(kernel_rows + index)] = 1.0
    expand = torch.zeros(QK_WIDTH_PER_DEVICE, VALUE_WIDTH_PER_DEVICE)
    for head in range(QK_HEADS_PER_DEVICE):
        for repeat in range(QK_REPEAT_FACTOR):
            value_head = head * QK_REPEAT_FACTOR + repeat
            expand[
                head * HEAD_DIM : (head + 1) * HEAD_DIM, value_head * HEAD_DIM : (value_head + 1) * HEAD_DIM
            ] = torch.eye(HEAD_DIM)
    return {
        "conv_taps": taps,
        "history_select_stack": stack,
        "history_select_full": history_select_full,
        "qk_expand": expand,
    }


def pack_projection_columns(shards) -> torch.Tensor:
    """Pack the four device-local ``[q|k|v|z|a|b]`` streams into one ``[1,1,K,4*N]`` linear weight.

    Each device block is ``[HIDDEN_SIZE, PROJECTION_WIDTH_PER_DEVICE]``: columns up to
    ``A_COLUMN + 12`` are the checkpoint order, ``b`` moves to ``B_COLUMN`` and the
    two 20-column gaps are zero.  Zero weight columns produce zero outputs that
    the decode split never reads.
    """

    a_end = A_COLUMN + VALUE_HEADS_PER_DEVICE
    blocks = []
    for shard in shards:
        fused = shard.fused_qkvzab.transpose(0, 1)
        expected = (HIDDEN_SIZE, QKVZAB_WIDTH_PER_DEVICE)
        if tuple(fused.shape) != expected:
            raise RuntimeError(f"fused qkvzab shard shape {tuple(fused.shape)} != {expected}")
        block = torch.zeros(HIDDEN_SIZE, PROJECTION_WIDTH_PER_DEVICE, dtype=fused.dtype)
        block[:, :a_end] = fused[:, :a_end]
        block[:, B_COLUMN : B_COLUMN + VALUE_HEADS_PER_DEVICE] = fused[:, a_end:QKVZAB_WIDTH_PER_DEVICE]
        blocks.append(block)
    return torch.cat(blocks, dim=1).reshape(1, 1, HIDDEN_SIZE, len(blocks) * PROJECTION_WIDTH_PER_DEVICE)


def _shape(tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def _tensor_key(tensor) -> tuple[str, int]:
    tensor_id = getattr(tensor, "tensor_id", None)
    if callable(tensor_id):
        tensor_id = tensor_id()
    return ("ttnn", int(tensor_id)) if tensor_id is not None else ("python", id(tensor))


def _deallocate(*tensors) -> None:
    """Deallocate each distinct backing tensor at most once."""

    seen: set[tuple[str, int]] = set()
    for tensor in tensors:
        if tensor is None:
            continue
        key = _tensor_key(tensor)
        if key in seen:
            continue
        seen.add(key)
        ttnn.deallocate(tensor)


def _copy_inplace(source, target, *, label: str) -> None:
    """Copy into a persistent target and reject an address-changing result."""

    target_id = _tensor_key(target)
    copied = ttnn.copy(source, target)
    if _tensor_key(target) != target_id:
        raise RuntimeError(f"{label} target address changed during ttnn.copy")
    if copied is not None and _tensor_key(copied) != target_id:
        raise RuntimeError(f"{label} ttnn.copy returned a different tensor")


def _require_shape(tensor, expected: tuple[int, ...], *, label: str) -> None:
    actual = _shape(tensor)
    if actual != expected:
        raise RuntimeError(f"{label} local shape must be {expected}, got {actual}")


def softplus_gate(a_fp32, dt_bias, *, memory_config):
    """``softplus(a + dt_bias)`` in FP32 as two programs: the add, then ``ttnn.softplus``.

    Not the fused ``SOFTPLUS`` activation of ``ttnn.add``: it returns exactly 0 below about -5 and
    is 1.5e-3 off elsewhere; the standalone op never flushes and is 4.7e-4-accurate.
    """
    shifted = ttnn.add(a_fp32, dt_bias, memory_config=memory_config)
    softplus = ttnn.softplus(shifted, beta=1.0, threshold=20.0, memory_config=memory_config)
    _deallocate(shifted)
    return softplus


def _exact_lane(lane, lanes: int) -> int:
    if isinstance(lane, bool) or type(lane) is not int or not 0 <= lane < lanes:
        raise ValueError(f"lane must be an int in [0,{lanes}), got {lane!r}")
    return lane


def _retag_head_shard_after_reshape(tensor, *, reference, shard_dim: int) -> None:
    """Record the new logical head axis after a local-only reshape.

    TTNN reshape/repeat currently preserve the input ``PlacementShard`` axis.
    GDN splits the fused last-axis shard into ``[..., heads, head_dim]`` without
    moving bytes between devices, so the same physical shard becomes the head
    axis in the reshaped tensor.
    """

    topology = reference.tensor_topology()
    tensor.update_tensor_topology(
        ttnn.TensorTopology(
            topology.distribution_shape(),
            [ttnn.PlacementReplicate(), ttnn.PlacementShard(shard_dim)],
            topology.mesh_coords(),
        )
    )


def _allocate_head_sharded_zero(
    mesh_device,
    mesh_contract: Qwen38MeshContract,
    *,
    local_shape: tuple[int, ...],
    dtype,
    shard_dim: int,
    label: str,
):
    """Allocate one zero-filled local tensor per device and record its TP shard.

    ``moreh_full`` avoids the host-to-device zero upload while preserving the
    exact local storage that the former ``ShardTensor2dMesh`` path produced.
    The fill initially reports replicated placement metadata; changing that
    metadata to the known head shard is local-only and moves no bytes.
    """

    tensor = ttnn.moreh_full(
        list(local_shape),
        0.0,
        mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    try:
        topology = tensor.tensor_topology()
        distribution_shape = tuple(int(value) for value in topology.distribution_shape())
        mesh_coords = tuple(tuple(int(value) for value in coord) for coord in topology.mesh_coords())
        placement_names = tuple(type(value).__name__ for value in topology.placements())
        expected_coords = tuple((0, column) for column in range(TP_SIZE))
        if distribution_shape != MESH_SHAPE or mesh_coords != expected_coords:
            raise RuntimeError(
                f"{label} device allocation has unexpected topology: "
                f"distribution={distribution_shape} coordinates={mesh_coords}"
            )
        if placement_names != ("PlacementReplicate", "PlacementReplicate"):
            raise RuntimeError(f"{label} device allocation is not initially replicated: {placement_names}")

        tensor.update_tensor_topology(
            ttnn.TensorTopology(
                topology.distribution_shape(),
                [ttnn.PlacementReplicate(), ttnn.PlacementShard(shard_dim)],
                topology.mesh_coords(),
            )
        )
        mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HEAD_SHARDED, shard_dim=shard_dim)
        local_tensors = tuple(ttnn.get_device_tensors(tensor))
        if len(local_tensors) != TP_SIZE:
            raise RuntimeError(f"{label} device allocation has {len(local_tensors)} locals, expected {TP_SIZE}")
        for index, local in enumerate(local_tensors):
            if _shape(local) != local_shape:
                raise RuntimeError(f"{label} local {index} shape must be {local_shape}, got {_shape(local)}")
            if local.dtype != dtype or local.layout != ttnn.TILE_LAYOUT:
                raise RuntimeError(f"{label} local {index} has unexpected dtype/layout")
            if local.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                raise RuntimeError(f"{label} local {index} must be DRAM resident")
        return tensor
    except BaseException:
        _deallocate(tensor)
        raise


def _require_pinned_config(checkpoint: Qwen38Checkpoint, layer_index: int) -> None:
    config = checkpoint.config
    exact = {
        "hidden_size": HIDDEN_SIZE,
        "gdn_qk_heads": QK_HEADS,
        "gdn_value_heads": VALUE_HEADS,
        "gdn_key_head_dim": HEAD_DIM,
        "gdn_value_head_dim": HEAD_DIM,
        "gdn_conv_kernel": CONV_KERNEL_SIZE,
        "gdn_output_gate": "sigmoid",
        "rms_norm_eps": RMS_NORM_EPS,
    }
    for name, expected in exact.items():
        actual = getattr(config, name)
        if actual != expected:
            raise ValueError(f"pinned GDN field {name} must be {expected!r}, got {actual!r}")
    if not 0 <= layer_index < config.num_hidden_layers:
        raise ValueError(f"GDN layer index is outside [0,{config.num_hidden_layers}): {layer_index}")
    if config.layer_types[layer_index] != "linear_attention":
        raise ValueError(f"layer {layer_index} is {config.layer_types[layer_index]!r}, not linear_attention")


def _cache_directory(
    cache_root: str | Path,
    checkpoint: Qwen38Checkpoint,
    mesh_contract: Qwen38MeshContract,
    layer_index: int,
    tt_metal_sha: str,
) -> Path:
    if len(tt_metal_sha) != 40 or any(character not in "0123456789abcdef" for character in tt_metal_sha):
        raise ValueError(f"tt_metal_sha must be a lowercase 40-hex commit, got {tt_metal_sha!r}")
    physical = "-".join(str(device_id) for device_id in mesh_contract.physical_ids)
    path = (
        Path(cache_root).resolve()
        / "gdn"
        / f"index-{INDEX_SHA256}"
        / checkpoint.config.config_sha256
        / f"tt-metal-{tt_metal_sha}"
        / f"mesh-1x4-physical-{physical}"
        / f"layer-{layer_index:02d}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class Qwen38TTNNGDNWeights:
    """Resident TP4 weights for one Qwen3.8 GDN layer."""

    layer_index: int
    qkvzab: Any
    out: Any
    conv_taps: tuple[Any, Any, Any, Any]
    dt_bias: Any
    neg_exp_A: Any
    norm: Any
    projection_dtype: Any
    # The DRAM readers per bank the projection weights are laid out for (decode_matmul); the module runs them so.
    decode_dram_workers_per_bank: int = 1

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Qwen38Checkpoint,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        cache_root: str | Path,
        *,
        layer_index: int,
        tt_metal_sha: str,
        projection_dtype=None,
        decode_dram_workers_per_bank: int = 1,
    ) -> "Qwen38TTNNGDNWeights":
        """Pack and upload one exact checkpoint layer.

        The host packing order is ``[device0 local streams | ... | device3]``.
        Sharding that reordered dimension gives each coordinate exactly its four
        Q/K heads and twelve value heads.  No projection tensor is replicated.
        Cache paths bind the checkpoint config, exact tt-metal commit, physical
        mesh order, layer, and dtype so a stale cross-runtime tensorbin cannot
        be selected silently.
        """

        validate_decode_dram_workers(decode_dram_workers_per_bank)
        mesh_contract.validate_mesh(mesh_device)
        _require_pinned_config(checkpoint, layer_index)
        if projection_dtype is None:
            # The continuation's correctness path keeps non-expert projection
            # weights in BF16.  BF8_B is an explicit later experiment, never
            # an implicit substitute during ordinary-decode bring-up.
            projection_dtype = ttnn.bfloat16
        if projection_dtype not in DENSE_DTYPE_TAGS:
            raise ValueError("GDN projection weights must use BFLOAT16, BFLOAT8_B or BFLOAT4_B")

        source = Qwen38GDNWeights.from_checkpoint(checkpoint, layer_index)
        shards = tuple(source.device_shard(device_index) for device_index in range(TP_SIZE))
        cache_dir = _cache_directory(cache_root, checkpoint, mesh_contract, layer_index, tt_metal_sha)
        dtype_tag = dense_dtype_tag(projection_dtype)

        # Each device-local TTNN linear weight is [K,N].  Concatenating the
        # transposed local blocks along N makes Shard(dim=3) select one whole
        # [q|k|v|z|a|b] block, rather than slicing through the stream boundary.
        qkvzab_host = pack_projection_columns(shards)
        expected_qkvzab = (1, 1, HIDDEN_SIZE, TP_SIZE * PROJECTION_WIDTH_PER_DEVICE)
        if tuple(qkvzab_host.shape) != expected_qkvzab:
            raise RuntimeError(f"packed qkvzab shape {tuple(qkvzab_host.shape)} != {expected_qkvzab}")
        # Projection weights are DRAM width-sharded for the decode matmul
        # program; the renamed tensorbins deliberately orphan interleaved
        # caches and the unpadded 4120-column packing.
        if decode_dram_workers_per_bank != 1 and projection_dtype not in TWO_READER_QUALIFIED_DTYPES:
            raise ValueError(
                "two DRAM readers per bank are qualified for "
                f"{sorted(dense_dtype_tag(d) for d in TWO_READER_QUALIFIED_DTYPES)} GDN projection weights, "
                f"not {dtype_tag}"
            )
        # Two readers per bank pad this weight's bank shard to 18 tiles (576 columns; 17 with one reader): the file
        # name carries the wider layout (weight_layout_tag), the loaded members are checked against it below.
        layout_suffix = weight_layout_tag(
            mesh_device,
            HIDDEN_SIZE,
            PROJECTION_WIDTH_PER_DEVICE,
            num_workers_per_dram_bank=decode_dram_workers_per_bank,
        )
        qkvzab = ttnn.as_tensor(
            qkvzab_host,
            dtype=projection_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=dram_sharded_weight_memory_config(
                mesh_device,
                HIDDEN_SIZE,
                PROJECTION_WIDTH_PER_DEVICE,
                num_workers_per_dram_bank=decode_dram_workers_per_bank,
            ),
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3)),
            cache_file_name=cache_dir / f"qkvzab_tile_aligned_ab_dram_sharded{layout_suffix}.{dtype_tag}",
        )
        validate_dram_sharded_weight(
            qkvzab,
            mesh_device,
            HIDDEN_SIZE,
            PROJECTION_WIDTH_PER_DEVICE,
            num_workers_per_dram_bank=decode_dram_workers_per_bank,
        )

        # Row-parallel output projection: each coordinate owns the checkpoint
        # columns for its twelve value heads, transposed to local [1536,2560].
        out_host = torch.cat([shard.out.transpose(0, 1).contiguous() for shard in shards], dim=0).reshape(
            1, 1, VALUE_WIDTH, HIDDEN_SIZE
        )
        out = ttnn.as_tensor(
            out_host,
            dtype=projection_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=dram_sharded_weight_memory_config(mesh_device, VALUE_WIDTH_PER_DEVICE, HIDDEN_SIZE),
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 2)),
            cache_file_name=cache_dir / f"out_dram_sharded.{dtype_tag}",
        )

        def upload_head_vector(value: torch.Tensor, name: str, *, dtype):
            value = value.reshape(1, 1, 1, -1).contiguous()
            return ttnn.as_tensor(
                value,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3)),
                cache_file_name=cache_dir / name,
            )

        # Reorder every convolution tap with the same per-device stream packing
        # as qkvzab.  The activation remains BF16; parameters driving the decay
        # are uploaded as FP32 to match the pinned SSM arithmetic.
        conv_taps = tuple(
            upload_head_vector(
                torch.cat([shard.conv[:, 0, tap] for shard in shards], dim=0).to(torch.bfloat16),
                f"conv-tap-{tap}.bf16",
                dtype=ttnn.bfloat16,
            )
            for tap in range(CONV_KERNEL_SIZE)
        )
        dt_bias = upload_head_vector(
            torch.cat([shard.dt_bias for shard in shards], dim=0).float(),
            "dt-bias.fp32",
            dtype=ttnn.float32,
        )
        neg_exp_A = upload_head_vector(
            -torch.exp(torch.cat([shard.A_log for shard in shards], dim=0).float()),
            "neg-exp-A.fp32",
            dtype=ttnn.float32,
        )
        norm = ttnn.as_tensor(
            source.norm.reshape(1, 1, 1, HEAD_DIM).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh_device),
            cache_file_name=cache_dir / "norm.bf16",
        )

        result = cls(
            layer_index,
            qkvzab,
            out,
            conv_taps,
            dt_bias,
            neg_exp_A,
            norm,
            projection_dtype,
            decode_dram_workers_per_bank=decode_dram_workers_per_bank,
        )
        result.validate(mesh_contract)
        return result

    def validate(self, mesh_contract: Qwen38MeshContract) -> None:
        mesh_contract.validate_tensor(self.qkvzab, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        mesh_contract.validate_tensor(self.out, placement=TensorPlacement.HEAD_SHARDED, shard_dim=2)
        _require_shape(self.qkvzab, (1, 1, HIDDEN_SIZE, PROJECTION_WIDTH_PER_DEVICE), label="GDN qkvzab")
        _require_shape(self.out, (1, 1, VALUE_WIDTH_PER_DEVICE, HIDDEN_SIZE), label="GDN out")
        for tap_index, tap in enumerate(self.conv_taps):
            mesh_contract.validate_tensor(tap, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
            _require_shape(tap, (1, 1, 1, QKV_WIDTH_PER_DEVICE), label=f"GDN conv tap {tap_index}")
            if tap.dtype != ttnn.bfloat16:
                raise RuntimeError(f"GDN conv tap {tap_index} must be BF16, got {tap.dtype}")
        for name, tensor in (("dt_bias", self.dt_bias), ("neg_exp_A", self.neg_exp_A)):
            mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
            _require_shape(tensor, (1, 1, 1, VALUE_HEADS_PER_DEVICE), label=f"GDN {name}")
            if tensor.dtype != ttnn.float32:
                raise RuntimeError(f"GDN {name} must be FP32, got {tensor.dtype}")
        mesh_contract.validate_tensor(self.norm, placement=TensorPlacement.REPLICATED)
        _require_shape(self.norm, (1, 1, 1, HEAD_DIM), label="GDN gated RMSNorm weight")
        if self.norm.dtype != ttnn.bfloat16:
            raise RuntimeError(f"GDN gated RMSNorm weight must be BF16, got {self.norm.dtype}")

    def deallocate(self) -> None:
        _deallocate(self.qkvzab, self.out, *self.conv_taps, self.dt_bias, self.neg_exp_A, self.norm)


@dataclass
class Qwen38TTNNGDNSnapshot:
    """Preallocated rollback image for one layer's mutable state."""

    layer_index: int
    recurrent: Any
    conv: tuple[Any, Any, Any, Any]
    captured: bool = False
    conv_phase: int = 0
    batch_size: int = 1

    def validate(self, mesh_contract: Qwen38MeshContract) -> None:
        batch = require_lane_count(self.batch_size, label="GDN snapshot batch size", error_type=RuntimeError)
        mesh_contract.validate_tensor(self.recurrent, placement=TensorPlacement.HEAD_SHARDED, shard_dim=1)
        _require_shape(
            self.recurrent,
            (batch, VALUE_HEADS_PER_DEVICE, HEAD_DIM, HEAD_DIM),
            label="GDN snapshot recurrent state",
        )
        if self.recurrent.dtype != ttnn.float32:
            raise RuntimeError(f"GDN snapshot recurrent state must be FP32, got {self.recurrent.dtype}")
        for index, tensor in enumerate(self.conv):
            mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
            _require_shape(tensor, (1, 1, batch, QKV_WIDTH_PER_DEVICE), label=f"GDN snapshot conv[{index}]")
        if not 0 <= self.conv_phase < CONV_KERNEL_SIZE:
            raise RuntimeError(f"GDN snapshot conv phase must be in [0,{CONV_KERNEL_SIZE}), got {self.conv_phase}")

    def deallocate(self) -> None:
        _deallocate(self.recurrent, *self.conv)


@dataclass
class Qwen38TTNNGDNState:
    """Fixed-address mutable state owned by exactly one GDN layer.

    The four convolution buffers form a ring: the token at step ``n`` is
    written straight into slot ``n % 4`` and the FIR pairs tap ``i`` with slot
    ``(n + 1 + i) % 4``, so no shift copies run.  ``conv_phase`` is the slot
    the next token lands in; snapshots carry it.  A captured graph binds one
    phase, so a trace is valid at positions congruent mod 4 (the decoder's
    graphs are already per position mod 32 through the QSA layers).
    """

    layer_index: int
    recurrent: Any
    conv: tuple[Any, Any, Any, Any]
    zero_recurrent: Any
    zero_conv: Any
    mesh_contract: Qwen38MeshContract
    conv_phase: int = 0
    # Lanes of the batched layout: recurrent [B,12,128,128], conv slots [1,1,B,2560] (lane u = batch index u /
    # row u).  The 1-row decode body (forward_decode) admits batch 1 only; the batched step is a separate path.
    batch_size: int = 1

    @classmethod
    def allocate(
        cls,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        *,
        layer_index: int,
        batch_size: int = 1,
    ) -> "Qwen38TTNNGDNState":
        mesh_contract.validate_mesh(mesh_device)
        batch_size = require_lane_count(batch_size, label="GDN state batch size")
        if layer_index < 0:
            raise ValueError("layer_index must be nonnegative")

        allocated: list[Any] = []

        def allocate_recurrent(label: str):
            tensor = _allocate_head_sharded_zero(
                mesh_device,
                mesh_contract,
                local_shape=(batch_size, VALUE_HEADS_PER_DEVICE, HEAD_DIM, HEAD_DIM),
                dtype=ttnn.float32,
                shard_dim=1,
                label=label,
            )
            allocated.append(tensor)
            return tensor

        def allocate_conv(label: str):
            tensor = _allocate_head_sharded_zero(
                mesh_device,
                mesh_contract,
                local_shape=(1, 1, batch_size, QKV_WIDTH_PER_DEVICE),
                dtype=ttnn.bfloat16,
                shard_dim=3,
                label=label,
            )
            allocated.append(tensor)
            return tensor

        try:
            result = cls(
                layer_index=layer_index,
                recurrent=allocate_recurrent("GDN recurrent state"),
                conv=tuple(allocate_conv(f"GDN conv[{index}]") for index in range(CONV_KERNEL_SIZE)),
                zero_recurrent=allocate_recurrent("GDN zero recurrent source"),
                zero_conv=allocate_conv("GDN zero conv source"),
                mesh_contract=mesh_contract,
                batch_size=batch_size,
            )
            result.validate()
            return result
        except BaseException:
            _deallocate(*allocated)
            raise

    def validate(self) -> None:
        batch = require_lane_count(self.batch_size, label="GDN state batch size", error_type=RuntimeError)
        if len(self.conv) != CONV_KERNEL_SIZE:
            raise RuntimeError(f"GDN state requires {CONV_KERNEL_SIZE} conv buffers, got {len(self.conv)}")
        owned = (self.recurrent, *self.conv, self.zero_recurrent, self.zero_conv)
        if len({_tensor_key(tensor) for tensor in owned}) != CONV_KERNEL_SIZE + 3:
            raise RuntimeError("GDN state requires seven distinct backing tensors")
        self.mesh_contract.validate_tensor(self.recurrent, placement=TensorPlacement.HEAD_SHARDED, shard_dim=1)
        self.mesh_contract.validate_tensor(self.zero_recurrent, placement=TensorPlacement.HEAD_SHARDED, shard_dim=1)
        recurrent_shape = (batch, VALUE_HEADS_PER_DEVICE, HEAD_DIM, HEAD_DIM)
        _require_shape(self.recurrent, recurrent_shape, label=f"GDN recurrent state (batch {batch})")
        _require_shape(self.zero_recurrent, recurrent_shape, label=f"GDN zero recurrent source (batch {batch})")
        if self.recurrent.dtype != ttnn.float32 or self.zero_recurrent.dtype != ttnn.float32:
            raise RuntimeError("GDN recurrent state and zero source must both be FP32")

        for name, tensor in (("zero_conv", self.zero_conv), *[(f"conv[{i}]", t) for i, t in enumerate(self.conv)]):
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
            _require_shape(tensor, (1, 1, batch, QKV_WIDTH_PER_DEVICE), label=f"GDN {name} (batch {batch})")
            if tensor.dtype != ttnn.bfloat16:
                raise RuntimeError(f"GDN {name} must be BF16, got {tensor.dtype}")
        if not 0 <= self.conv_phase < CONV_KERNEL_SIZE:
            raise RuntimeError(f"GDN conv phase must be in [0,{CONV_KERNEL_SIZE}), got {self.conv_phase}")

    def conv_window(self) -> tuple[Any, Any, Any, Any]:
        """FIR operands for the next token, oldest first; the last slot receives that token."""

        return tuple(self.conv[(self.conv_phase + 1 + index) % CONV_KERNEL_SIZE] for index in range(CONV_KERNEL_SIZE))

    def advance_conv_window(self) -> None:
        self.conv_phase = (self.conv_phase + 1) % CONV_KERNEL_SIZE

    def reset_inplace(self) -> None:
        """Zero all state without changing any address (trace safe)."""

        self.validate()
        _copy_inplace(self.zero_recurrent, self.recurrent, label="GDN recurrent reset")
        for index, tensor in enumerate(self.conv):
            _copy_inplace(self.zero_conv, tensor, label=f"GDN conv[{index}] reset")
        self.conv_phase = 0

    def allocate_snapshot(self) -> Qwen38TTNNGDNSnapshot:
        """Allocate one persistent capture target for commit/rollback."""

        recurrent = ttnn.empty_like(
            self.recurrent,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        conv = tuple(
            ttnn.empty_like(
                tensor,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for tensor in self.conv
        )
        snapshot = Qwen38TTNNGDNSnapshot(self.layer_index, recurrent, conv, batch_size=self.batch_size)
        snapshot.validate(self.mesh_contract)
        return snapshot

    def reset_lane_inplace(self, lane: int) -> None:
        """Zero lane ``lane`` of the recurrent state and of every ring slot, every other lane unchanged, no address
        change: a keep-mask multiply (lane 1.0 everywhere but 0.0 at ``lane``; x * 1.0 is x, x * 0.0 is the zero
        source's +0.0) landed by an in-place copy.  Eager host uploads: a lifecycle op between steps, never traced.
        The ring phase is shared, so the lane joins the resident residue class (the caller admits it at a step of
        its position's residue)."""

        self.validate()
        lane = _exact_lane(lane, self.batch_size)
        keep = torch.ones(self.batch_size, dtype=torch.float32)
        keep[lane] = 0.0
        device = self.recurrent.device()
        mapper = replicate_tensor_2d_mesh_mapper(device)

        def upload(host: torch.Tensor, dtype):
            return ttnn.from_torch(
                host,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        recurrent_keep = upload(keep.reshape(self.batch_size, 1, 1, 1), ttnn.float32)
        conv_keep = upload(keep.reshape(1, 1, self.batch_size, 1).to(torch.bfloat16), ttnn.bfloat16)
        try:
            kept = ttnn.multiply(self.recurrent, recurrent_keep, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            _copy_inplace(kept, self.recurrent, label=f"GDN lane {lane} recurrent reset")
            _deallocate(kept)
            for index, slot in enumerate(self.conv):
                kept = ttnn.multiply(slot, conv_keep, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                _copy_inplace(kept, slot, label=f"GDN lane {lane} conv[{index}] reset")
                _deallocate(kept)
        finally:
            _deallocate(recurrent_keep, conv_keep)

    def capture_into(self, snapshot: Qwen38TTNNGDNSnapshot) -> None:
        if snapshot.layer_index != self.layer_index:
            raise ValueError(
                f"cannot capture layer {self.layer_index} state into layer {snapshot.layer_index} snapshot"
            )
        snapshot.validate(self.mesh_contract)
        _copy_inplace(self.recurrent, snapshot.recurrent, label="GDN recurrent snapshot capture")
        for index, (source, target) in enumerate(zip(self.conv, snapshot.conv)):
            _copy_inplace(source, target, label=f"GDN conv[{index}] snapshot capture")
        snapshot.conv_phase = self.conv_phase
        snapshot.captured = True

    def restore_from(self, snapshot: Qwen38TTNNGDNSnapshot) -> None:
        if snapshot.layer_index != self.layer_index:
            raise ValueError(
                f"cannot restore layer {snapshot.layer_index} snapshot into layer {self.layer_index} state"
            )
        if not snapshot.captured:
            raise RuntimeError("cannot restore an uncaptured GDN snapshot")
        snapshot.validate(self.mesh_contract)
        _copy_inplace(snapshot.recurrent, self.recurrent, label="GDN recurrent snapshot restore")
        for index, (source, target) in enumerate(zip(snapshot.conv, self.conv)):
            _copy_inplace(source, target, label=f"GDN conv[{index}] snapshot restore")
        self.conv_phase = snapshot.conv_phase

    def deallocate(self) -> None:
        _deallocate(self.recurrent, *self.conv, self.zero_recurrent, self.zero_conv)


@dataclass(frozen=True)
class Qwen38TTNNGDNResult:
    hidden_sharded: Any
    state: Qwen38TTNNGDNState


def _upload_replicated_constant(
    mesh_device, mesh_contract: Qwen38MeshContract, host: torch.Tensor, dtype, *, label: str
):
    tensor = ttnn.from_torch(
        host.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh_device),
    )
    mesh_contract.validate_tensor(tensor, placement=TensorPlacement.REPLICATED)
    _require_shape(tensor, tuple(host.shape), label=label)
    return tensor


def _require_landed(result, target, *, label: str) -> None:
    """An ``output_tensor=`` op must return the persistent target itself."""

    if result is not None and _tensor_key(result) != _tensor_key(target):
        raise RuntimeError(f"{label} did not land in its persistent buffer")


def chunk_constant_tiles() -> dict[str, torch.Tensor]:
    """Host images of the chunk kernel's constant tiles: eye / tril / ones ``[1,1,32,32]`` and the three
    32x32 quadrant masks packed into ``[1,1,32,96]`` (top-left, bottom-right, bottom-left)."""

    rows = torch.arange(CHUNK_SIZE).unsqueeze(1) < CHUNK_SIZE // 2
    columns = torch.arange(CHUNK_SIZE).unsqueeze(0) < CHUNK_SIZE // 2
    masks = torch.cat([(rows & columns).float(), (~rows & ~columns).float(), (~rows & columns).float()], dim=1)
    return {
        "eye": torch.eye(CHUNK_SIZE).reshape(1, 1, CHUNK_SIZE, CHUNK_SIZE),
        "tril": torch.tril(torch.ones(CHUNK_SIZE, CHUNK_SIZE)).reshape(1, 1, CHUNK_SIZE, CHUNK_SIZE),
        "ones": torch.ones(1, 1, CHUNK_SIZE, CHUNK_SIZE),
        "masks": masks.reshape(1, 1, CHUNK_SIZE, CHUNK_QUADRANT_MASK_WIDTH),
    }


@dataclass(frozen=True)
class Qwen38TTNNGDNRowsConstants:
    """Read-only device tensors shared by every GDN layer's ``rows``-row path (one set per row count).

    ``row_mask_*`` is 1.0 for rows below ``rows`` and 0.0 for the padding rows up to the chunk; ``arange``
    (a column) and ``arange_row`` (a row) hold the row index (fp32, exact); ``one`` is the fp32 scalar 1.0;
    ``conv_taps``, ``history_select_stack`` and ``qk_expand`` are the 0/1 selection matrices of
    :func:`rows_window_select_tiles` (bf16: exactly representable); ``eye``/``tril``/``ones``/``masks`` are
    the chunk kernel's constant tiles.  Everything is uploaded here so nothing is uploaded inside a trace.
    ``select_compute_config`` (HiFi4, fp32 accumulation, no approximations) is the matmul configuration
    under which a 0/1 selection is exact: every output element is one bf16 value times 1.0 plus exact zeros.
    """

    rows: int
    tile_rows: int  # the kernel rows T: CHUNK_SIZE (rows <= 32) or LONG_CHUNK_ROWS
    row_mask_bf16: Any  # [1, T, 1, 1] BF16, multiplies the [1, T, heads, 128] q/k
    row_mask_bf16_col: Any  # [1, 1, T, 1] BF16, multiplies the token-major flat [1, 1, T, 1536] v
    row_mask_fp32: Any  # [1, 1, T, 1] FP32, multiplies the [1, 1, T, heads] beta / log decay
    arange: Any  # [1, 1, CHUNK_SIZE, 1] FP32 row index
    arange_row: Any  # [1, 1, 1, CHUNK_SIZE] FP32 row index
    one: Any  # [1, 1, 1, 1] FP32
    conv_taps: tuple[Any, Any, Any]  # 3 x [1, 1, T, 32 + T] BF16
    history_select_stack: Any  # [1, 1, CHUNK_SIZE, CHUNK_SIZE * CONV_WINDOW_TILE_ROWS] BF16 (rows <= 32)
    history_select_full: Any  # [1, 1, CHUNK_SIZE, 32 + T] BF16: the full-commit select (every row committed)
    qk_expand: Any  # [1, 1, QK_WIDTH_PER_DEVICE, VALUE_WIDTH_PER_DEVICE] BF16
    eye: Any
    tril: Any
    ones: Any
    masks: Any
    select_compute_config: Any
    mesh_contract: Qwen38MeshContract
    # ``[12, 1, 32, 1]`` FP32 row index in the chunk prims' own layout, uploaded for the 32-row form when the
    # verify-rows wrap runs (``fused.gdn_rows_wrap``, a default; None under QWEN38_FUSED_OFF=gdn_rows_wrap): its
    # commit masks ``beta_c`` / ``g_c`` in that layout, so the mask is built there too and the multiply is element
    # for element.
    arange_c: Any = None

    @property
    def window_rows(self) -> int:
        return CHUNK_SIZE + self.tile_rows

    @classmethod
    def allocate(cls, mesh_device, mesh_contract: Qwen38MeshContract, *, rows: int) -> "Qwen38TTNNGDNRowsConstants":
        mesh_contract.validate_mesh(mesh_device)
        tile_rows = rows_tile_count(rows)
        window_rows = CHUNK_SIZE + tile_rows
        keep = (torch.arange(tile_rows) < rows).float()
        tiles = chunk_constant_tiles()
        # A slab reads its FIR taps and next history by row shifts (a [T, 32 + T] select per tap would be 22 GFLOP
        # at 2048 rows): only the GQA expand of the 32-row selects is uploaded for it.
        slab = is_slab_rows(rows)
        selects = rows_window_select_tiles(CHUNK_SIZE if slab else rows)
        uploaded: list[Any] = []

        def upload(host: torch.Tensor, dtype, label: str):
            tensor = _upload_replicated_constant(mesh_device, mesh_contract, host, dtype, label=label)
            uploaded.append(tensor)
            return tensor

        try:
            return cls(
                rows=rows,
                tile_rows=tile_rows,
                row_mask_bf16=upload(keep.reshape(1, tile_rows, 1, 1).to(torch.bfloat16), ttnn.bfloat16, "row mask"),
                row_mask_bf16_col=upload(
                    keep.reshape(1, 1, tile_rows, 1).to(torch.bfloat16), ttnn.bfloat16, "row mask column"
                ),
                row_mask_fp32=upload(keep.reshape(1, 1, tile_rows, 1), ttnn.float32, "row mask fp32"),
                arange=upload(torch.arange(CHUNK_SIZE).float().reshape(1, 1, CHUNK_SIZE, 1), ttnn.float32, "arange"),
                arange_row=upload(
                    torch.arange(CHUNK_SIZE).float().reshape(1, 1, 1, CHUNK_SIZE), ttnn.float32, "arange row"
                ),
                one=upload(torch.ones(1, 1, 1, 1), ttnn.float32, "one"),
                conv_taps=(
                    ()
                    if slab
                    else tuple(
                        upload(
                            selects["conv_taps"][tap].reshape(1, 1, tile_rows, window_rows),
                            ttnn.bfloat16,
                            f"conv tap {tap} select",
                        )
                        for tap in range(CONV_HISTORY_ROWS)
                    )
                ),
                history_select_stack=upload(
                    selects["history_select_stack"].reshape(1, 1, CHUNK_SIZE, CHUNK_SIZE * CONV_WINDOW_TILE_ROWS),
                    ttnn.bfloat16,
                    "history select stack",
                ),
                history_select_full=(
                    None
                    if slab
                    else upload(
                        selects["history_select_full"].reshape(1, 1, CHUNK_SIZE, window_rows),
                        ttnn.bfloat16,
                        "history select full",
                    )
                ),
                qk_expand=upload(
                    selects["qk_expand"].reshape(1, 1, QK_WIDTH_PER_DEVICE, VALUE_WIDTH_PER_DEVICE),
                    ttnn.bfloat16,
                    "q/k GQA expand",
                ),
                arange_c=(
                    upload(
                        torch.arange(CHUNK_SIZE)
                        .float()
                        .reshape(1, 1, CHUNK_SIZE, 1)
                        .expand(VALUE_HEADS_PER_DEVICE, 1, CHUNK_SIZE, 1)
                        .contiguous(),
                        ttnn.float32,
                        "arange in the chunk prims' layout",
                    )
                    if fused.enabled(fused.gdn_rows_wrap.NAME) and tile_rows == CHUNK_SIZE
                    else None
                ),
                eye=upload(tiles["eye"], ttnn.float32, "chunk eye"),
                tril=upload(tiles["tril"], ttnn.float32, "chunk tril"),
                ones=upload(tiles["ones"], ttnn.float32, "chunk ones"),
                masks=upload(tiles["masks"], ttnn.float32, "chunk quadrant masks"),
                select_compute_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                ),
                mesh_contract=mesh_contract,
            )
        except BaseException:
            _deallocate(*uploaded)
            raise

    def deallocate(self) -> None:
        _deallocate(
            self.row_mask_bf16,
            self.row_mask_bf16_col,
            self.row_mask_fp32,
            self.arange,
            self.arange_row,
            self.one,
            *self.conv_taps,
            self.history_select_stack,
            *(() if self.history_select_full is None else (self.history_select_full,)),
            self.qk_expand,
            self.eye,
            self.tril,
            self.ones,
            self.masks,
            *(() if self.arange_c is None else (self.arange_c,)),
        )


@dataclass(frozen=True)
class Qwen38TTNNRowsSelectors:
    """Per-pass device selects derived once from the accept count and shared by every layer's commit.

    ``accepted`` is the number of accepted drafts (fp32 scalar on device, 0 <= accepted < rows); row 0 is
    the base token and is always committed, so ``committed_mask`` is ``arange <= accepted`` (accepted + 1
    rows).  ``history_select`` is the ``[32, 64]`` 0/1 matrix that reads the next GDN FIR history (window
    rows accepted + 1 .. accepted + 3) out of the ``[history | qkv]`` window with one exact matmul per
    layer; ``onehot_bf16[j]`` (``accepted == j``) drives the PLE commit's exact multiply/add select.
    """

    rows: int
    committed_mask: Any  # [1, 1, CHUNK_SIZE, 1] FP32
    onehot_bf16: tuple[Any, ...]  # rows x [1, 1, 1, 1] BF16
    history_select: Any  # [1, 1, CHUNK_SIZE, CONV_WINDOW_TILE_ROWS] BF16
    # The same mask in the chunk prims' layout, built only when the constants carry ``arange_c`` (the verify-rows
    # wrap): its commit multiplies ``beta_c`` / ``g_c`` ``[12, 1, 32, 1]`` element for element.
    committed_mask_c: Any = None
    # The pass's device accept count itself (fp32 ``[1, 1, 1, 1]``, the caller's tensor: not deallocated here); the
    # verify-rows fold's commit reads it to pick the committed prefix state.
    accepted: Any = None

    def validate(self, rows: int) -> None:
        if self.rows != rows or len(self.onehot_bf16) != rows:
            raise ValueError(f"rows selectors were built for {self.rows} rows, the path runs {rows}")
        _require_shape(self.committed_mask, (1, 1, CHUNK_SIZE, 1), label="committed rows mask")
        if self.committed_mask_c is not None:
            _require_shape(
                self.committed_mask_c,
                (VALUE_HEADS_PER_DEVICE, 1, CHUNK_SIZE, 1),
                label="committed rows mask (prim layout)",
            )
            if self.committed_mask_c.dtype != ttnn.float32:
                raise RuntimeError(f"prim-layout committed mask dtype is {self.committed_mask_c.dtype}")
        _require_shape(self.history_select, (1, 1, CHUNK_SIZE, CONV_WINDOW_TILE_ROWS), label="rows history select")
        if self.committed_mask.dtype != ttnn.float32 or self.history_select.dtype != ttnn.bfloat16:
            raise RuntimeError(
                f"rows selector dtypes are {self.committed_mask.dtype} (mask) / {self.history_select.dtype} (select)"
            )
        for index, bf16 in enumerate(self.onehot_bf16):
            _require_shape(bf16, (1, 1, 1, 1), label=f"rows selector onehot_bf16[{index}]")
            if bf16.dtype != ttnn.bfloat16:
                raise RuntimeError(f"rows selector onehot_bf16[{index}] dtype is {bf16.dtype}")

    def deallocate(self) -> None:
        _deallocate(self.committed_mask, *self.onehot_bf16, self.history_select, self.committed_mask_c)


def build_rows_selectors(accepted, constants: Qwen38TTNNGDNRowsConstants) -> Qwen38TTNNRowsSelectors:
    """Derive the commit selects from the device accept count (fp32 ``[1,1,1,1]``, replicated).

    rows + 4 tiny ops per pass, not per layer (one more with the verify-rows wrap's prim-layout mask): the
    mask, the one-hot row (``arange_row == accepted``,
    written as bf16 by the comparison), the history select as ``onehot_row @ history_select_stack``
    (exact: one row of the stack) reshaped to ``[32, 64]``, and the rows one-hot scalars for the PLE.
    Comparisons on fp32 integers are exact.
    """

    _require_shape(accepted, (1, 1, 1, 1), label="accept count")
    if accepted.dtype != ttnn.float32:
        raise RuntimeError(f"accept count must be FP32, got {accepted.dtype}")
    dram = ttnn.DRAM_MEMORY_CONFIG
    committed_mask = ttnn.le(constants.arange, accepted, memory_config=dram)
    onehot_row = ttnn.eq(constants.arange_row, accepted, dtype=ttnn.bfloat16, memory_config=dram)
    _require_shape(onehot_row, (1, 1, 1, CHUNK_SIZE), label="rows one-hot row")
    select_flat = ttnn.matmul(
        onehot_row,
        constants.history_select_stack,
        memory_config=dram,
        compute_kernel_config=constants.select_compute_config,
    )
    _deallocate(onehot_row)
    _require_shape(select_flat, (1, 1, 1, CHUNK_SIZE * CONV_WINDOW_TILE_ROWS), label="rows history select row")
    # The last dim changes, so this is a real relayout into a new buffer, not a view of select_flat.
    history_select = ttnn.reshape(select_flat, (1, 1, CHUNK_SIZE, CONV_WINDOW_TILE_ROWS))
    _deallocate(select_flat)
    onehot_bf16 = tuple(
        ttnn.eq(accepted, float(index), dtype=ttnn.bfloat16, memory_config=dram) for index in range(constants.rows)
    )
    # One more tiny op per pass (not per layer) when the wrap runs: the same comparison in the prims' layout.
    committed_mask_c = None if constants.arange_c is None else ttnn.le(constants.arange_c, accepted, memory_config=dram)
    selectors = Qwen38TTNNRowsSelectors(
        constants.rows, committed_mask, onehot_bf16, history_select, committed_mask_c, accepted=accepted
    )
    selectors.validate(constants.rows)
    return selectors


# The rows-state fields the fused prefill rows form adds (None on every other form).
_FUSED_ROWS_BUFFERS = ("sig", "o16", "gated", "history_next")


def rows_qk_layout(tile_rows: int, flat_qk: bool, fused_prefill: bool = False) -> tuple[tuple[int, ...], int]:
    """The rows state's q/k local shape and mesh shard dim: token-major heads ``[1, T, 12, 128]`` (shard dim 2), or
    under the slab's gdn_qk_flat form the raw conv rows ``[1, 1, T, 512]`` (shard dim 3), the kernel's flat form, or
    under the slab's fused prefill rows form the chunk prims' own pad-free pages ``[12, NC, 32, 128]`` (shard dim 0:
    the value heads lead), which ``gdn_pre_rows`` writes and ``ttnn.prim.chunk_gdn_prep`` reads without a relayout."""

    if fused_prefill:
        return fused.gdn_prefill_rows.buffer_layouts(tile_rows)["q"][0], 0
    if flat_qk:
        return (1, 1, tile_rows, QK_WIDTH_PER_DEVICE), 3
    return (1, tile_rows, VALUE_HEADS_PER_DEVICE, HEAD_DIM), 2


@dataclass
class Qwen38TTNNGDNRowsState:
    """Per-layer persistent buffers of the ``rows``-row path (fixed addresses across passes).

    ``history`` is the FIR history (rows 0..2: the CONV_HISTORY_ROWS q/k/v rows before the first new
    row, in a full 32-row tile whose rows 3..31 are never read; it replaces the phase-indexed ring of
    the 1-row path, which a variable-advance trace cannot bind).  ``qkv`` holds this pass's projected
    q|k|v rows (CHUNK_SIZE rows, rows past ``rows`` zero) and ``q``/``k``/``v``/``beta``/``g`` the chunk
    kernel's inputs (q/k already expanded to the 12 value heads; v token-major flat, the composite's
    ``flat v`` form that the prep reader addresses by tile without a head-split relayout), kept so the
    commit can rerun the kernel over the committed prefix from the committed recurrent state.
    """

    layer_index: int
    constants: Qwen38TTNNGDNRowsConstants
    history: Any  # [1, 1, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE] BF16, rows 0..CONV_HISTORY_ROWS-1 valid
    qkv: Any  # [1, 1, T, QKV_WIDTH_PER_DEVICE] BF16
    # [1, T, VALUE_HEADS_PER_DEVICE, HEAD_DIM] BF16, l2-normalized, GQA-expanded; under flat_qk (the slab's
    # gdn_qk_flat form) [1, 1, T, QK_WIDTH_PER_DEVICE] BF16, the raw conv q/k the kernel normalizes itself
    q: Any
    k: Any
    v: Any  # [1, 1, T, VALUE_WIDTH_PER_DEVICE] BF16, token-major flat (head h at columns 128h..)
    beta: Any  # [1, 1, T, VALUE_HEADS_PER_DEVICE] FP32
    g: Any  # same, the log decay
    output: Any  # [1, 1, rows, HIDDEN_SIZE_PER_DEVICE] BF16: the pass's output rows (valid until the next pass)
    mesh_contract: Qwen38MeshContract
    owns_history: bool = (
        True  # False: ``history`` is another rows state's buffer (the long chunk shares the 32-row one)
    )
    # False: the pass buffers (qkv, q, k, v, beta, g, output) are another layer's slab rows state's; the slab's
    # layers run one after another, so the 35 GDN layers share one set (52 MB at 2048 rows) and keep their histories.
    owns_body: bool = True
    # The slab's gdn_qk_flat form (prefill_glue, tolerance): q/k are the raw conv rows in the kernel's flat form.
    flat_qk: bool = False
    # The slab's fused prefill rows form (QWEN38_FUSED=gdn_prefill_rows, bitwise): q/k/beta/g take the chunk prims'
    # pad-free per-chunk pages and the four buffers below carry the pair's hand-offs.  None off the form.
    fused_prefill: bool = False
    sig: Any = None  # [1, 1, T, VALUE_WIDTH_PER_DEVICE] BF16: bf16(sigmoid(fp32 z)), the pre program's gate input
    o16: Any = None  # [VALUE_HEADS_PER_DEVICE, T, HEAD_DIM] BF16: the scan's output cast, post_cast -> post_norm
    gated: Any = None  # [1, 1, T, VALUE_WIDTH_PER_DEVICE] BF16: the gated rows the out-projection reads
    # [1, 1, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE] BF16: the next pass's FIR history, built by the gated epilogue and
    # copied into the shared ``history`` by ``commit_rows_full`` (``forward_rows`` must not touch that buffer).
    history_next: Any = None
    # The verify-rows wrap's per-layer buffers (``fused.gdn_rows_wrap.attach`` puts them here when that kernel runs
    # and this is the 32-row form), freed with the state.  While they are here the wrap owns the recurrence: the
    # prim layouts live in them and ``q`` / ``k`` / ``beta`` / ``g`` are not written.
    wrap_buffers: Any = None
    # The verify-rows fold's per-layer buffers (``fused.gdn_rows_scan.attach``: the prefix states), freed with the
    # state; while they are here the fold owns the recurrence and the commit, and the wrap is not attached.
    scan_buffers: Any = None

    @classmethod
    def allocate(
        cls,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        constants: Qwen38TTNNGDNRowsConstants,
        *,
        layer_index: int,
        history=None,
        body: "Qwen38TTNNGDNRowsState | None" = None,
        flat_qk: bool = False,
        fused_prefill: bool = False,
    ) -> "Qwen38TTNNGDNRowsState":
        """``history`` hands over another rows state's history buffer (same layer): the two row forms then carry
        one FIR history and need no sync between them.  ``body`` hands over another layer's slab rows state whose
        pass buffers this layer reuses (slab rows only).  ``flat_qk`` allocates q/k in the kernel's flat form and
        ``fused_prefill`` in the chunk prims' pad-free form with the fused pair's four extra buffers (both slab
        rows only, exclusive of each other; a shared body carries its own setting)."""

        mesh_contract.validate_mesh(mesh_device)
        if constants.mesh_contract != mesh_contract:
            raise ValueError("GDN rows constants belong to a different physical mesh contract")
        if body is not None and (not is_slab_rows(constants.rows) or body.constants is not constants):
            raise ValueError("a shared GDN rows body is a slab option over the same constants")
        if flat_qk and not is_slab_rows(constants.rows):
            raise ValueError("flat q/k rows buffers are a slab option (gdn_qk_flat)")
        if fused_prefill and not is_slab_rows(constants.rows):
            raise ValueError("the fused prefill rows buffers are a slab option (gdn_prefill_rows)")
        if fused_prefill and flat_qk:
            raise ValueError("gdn_qk_flat and gdn_prefill_rows are exclusive slab forms")
        if body is not None and body.flat_qk != flat_qk:
            raise ValueError("a shared GDN rows body and its layer disagree on flat q/k")
        if body is not None and getattr(body, "fused_prefill", False) != fused_prefill:
            raise ValueError("a shared GDN rows body and its layer disagree on the fused prefill rows form")
        allocated: list[Any] = []

        def zero(local_shape: tuple[int, ...], dtype, shard_dim: int, label: str):
            tensor = _allocate_head_sharded_zero(
                mesh_device, mesh_contract, local_shape=local_shape, dtype=dtype, shard_dim=shard_dim, label=label
            )
            allocated.append(tensor)
            return tensor

        tile_rows = constants.tile_rows
        qk_shape, qk_shard_dim = rows_qk_layout(tile_rows, flat_qk, fused_prefill)
        # Under the fused prefill rows form the pass buffers q / k / beta / g take the chunk prims' pad-free pages
        # and four more buffers carry the pair's hand-offs; off the form they are not allocated at all.
        extra = fused.gdn_prefill_rows.buffer_layouts(tile_rows) if fused_prefill else {}

        def fused_buffer(name: str):
            if not fused_prefill:
                return None
            if body is not None:
                return getattr(body, name)
            shape, dtype, shard_dim = extra[name]
            return zero(shape, dtype, shard_dim, f"GDN rows fused {name}")

        try:
            result = cls(
                layer_index=layer_index,
                constants=constants,
                history=(
                    zero((1, 1, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE), ttnn.bfloat16, 3, "GDN rows history")
                    if history is None
                    else history
                ),
                qkv=(
                    body.qkv
                    if body is not None
                    else zero((1, 1, tile_rows, QKV_WIDTH_PER_DEVICE), ttnn.bfloat16, 3, "GDN rows qkv")
                ),
                q=body.q if body is not None else zero(qk_shape, ttnn.bfloat16, qk_shard_dim, "GDN rows q"),
                k=body.k if body is not None else zero(qk_shape, ttnn.bfloat16, qk_shard_dim, "GDN rows k"),
                v=(
                    body.v
                    if body is not None
                    else zero((1, 1, tile_rows, VALUE_WIDTH_PER_DEVICE), ttnn.bfloat16, 3, "GDN rows v")
                ),
                beta=(
                    fused_buffer("beta")
                    if fused_prefill
                    else (
                        body.beta
                        if body is not None
                        else zero((1, 1, tile_rows, VALUE_HEADS_PER_DEVICE), ttnn.float32, 3, "GDN rows beta")
                    )
                ),
                g=(
                    fused_buffer("g")
                    if fused_prefill
                    else (
                        body.g
                        if body is not None
                        else zero((1, 1, tile_rows, VALUE_HEADS_PER_DEVICE), ttnn.float32, 3, "GDN rows log decay")
                    )
                ),
                output=(
                    body.output
                    if body is not None
                    else zero((1, 1, constants.rows, HIDDEN_SIZE_PER_DEVICE), ttnn.bfloat16, 3, "GDN rows output")
                ),
                mesh_contract=mesh_contract,
                owns_history=history is None,
                owns_body=body is None,
                flat_qk=flat_qk,
                fused_prefill=fused_prefill,
                sig=fused_buffer("sig"),
                o16=fused_buffer("o16"),
                gated=fused_buffer("gated"),
                history_next=fused_buffer("history_next"),
            )
            result.validate()
            return result
        except BaseException:
            _deallocate(*allocated)
            raise

    @property
    def fused_fields(self) -> tuple[str, ...]:
        """The pass buffers the fused prefill rows form adds beside the eight (empty off the form)."""

        return _FUSED_ROWS_BUFFERS if self.fused_prefill else ()

    def validate(self) -> None:
        owned = (self.history, self.qkv, self.q, self.k, self.v, self.beta, self.g, self.output)
        owned += tuple(getattr(self, name) for name in self.fused_fields)
        if len({_tensor_key(tensor) for tensor in owned}) != len(owned):
            raise RuntimeError(f"GDN rows state requires {len(owned)} distinct backing tensors")
        tile_rows = self.constants.tile_rows
        qk_shape, qk_shard_dim = rows_qk_layout(tile_rows, self.flat_qk, self.fused_prefill)
        extra = fused.gdn_prefill_rows.buffer_layouts(tile_rows) if self.fused_prefill else {}
        expected = {
            "history": ((1, 1, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE), ttnn.bfloat16, 3),
            "qkv": ((1, 1, tile_rows, QKV_WIDTH_PER_DEVICE), ttnn.bfloat16, 3),
            "q": (qk_shape, ttnn.bfloat16, qk_shard_dim),
            "k": (qk_shape, ttnn.bfloat16, qk_shard_dim),
            "v": ((1, 1, tile_rows, VALUE_WIDTH_PER_DEVICE), ttnn.bfloat16, 3),
            "beta": extra.get("beta", ((1, 1, tile_rows, VALUE_HEADS_PER_DEVICE), ttnn.float32, 3)),
            "g": extra.get("g", ((1, 1, tile_rows, VALUE_HEADS_PER_DEVICE), ttnn.float32, 3)),
            "output": ((1, 1, self.constants.rows, HIDDEN_SIZE_PER_DEVICE), ttnn.bfloat16, 3),
            **{name: extra[name] for name in self.fused_fields},
        }
        for name, (shape, dtype, shard_dim) in expected.items():
            tensor = getattr(self, name)
            _require_shape(tensor, shape, label=f"GDN rows {name}")
            if tensor.dtype != dtype:
                raise RuntimeError(f"GDN rows {name} must be {dtype}, got {tensor.dtype}")
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HEAD_SHARDED, shard_dim=shard_dim)
        if not self.fused_prefill and any(getattr(self, name) is not None for name in _FUSED_ROWS_BUFFERS):
            raise RuntimeError("GDN rows buffers of the fused prefill form are set on a state that is not on it")

    def deallocate(self) -> None:
        _deallocate(
            *((self.history,) if self.owns_history else ()),
            *((self.qkv, self.q, self.k, self.v, self.beta, self.g, self.output) if self.owns_body else ()),
            *(tuple(getattr(self, name) for name in self.fused_fields) if self.owns_body else ()),
        )
        if self.wrap_buffers is not None:
            self.wrap_buffers.deallocate()
            self.wrap_buffers = None
        if self.scan_buffers is not None:
            self.scan_buffers.deallocate()
            self.scan_buffers = None


# --------------------------------------------------------------------------- lane rows path (the MTP lanes verify)
# B lanes verify R = k + 1 rows each in ONE 32-row tile, lane-major (row u*R + j = lane u's row j; rows B*R .. 31 pad).
# The dense work (all-gather, projection, output projection, reduce-scatter) runs on that tile exactly as the
# rows path does; the per-lane work (FIR history, chunk kernel from lane u's recurrent state, commit) runs on a
# batch axis: one exact 0/1 ``expand_select`` copies lane u's rows into its own 32-row tile (rows R .. 31 zero),
# one batched kernel call takes ``[B, 32, 12, 128]`` from ``recurrent [B, 12, 128, 128]`` (bitwise per lane: the
# single-chip microbench of stage 0b), and one ``fold_select`` brings the per-lane outputs back to the lane-major
# tile before the gate.  Every select is one nonzero term per output element under HiFi4 / fp32 accumulation.


def lane_rows_select_tiles(lanes: int, rows: int) -> dict[str, torch.Tensor]:
    """Host images of the lane rows path's exact 0/1 selects.

    ``expand_select`` ``[B*32, 32]``: row ``u*32 + j`` takes lane-major row ``u*R + j`` for ``j < R``;
    ``fold_select`` ``[32, B*32]`` is its transpose; ``history_select_stack`` ``[32, 2048]``: row ``c`` (``c = 0 ..
    R``) is the flattened ``[32, 64]`` select whose rows 0..2 pick logical window rows ``c .. c + 2`` -- the FIR
    history after committing ``c`` rows (row 0 = the KEEP row: an inactive or freshly admitted lane commits nothing
    and its history is copied unchanged).
    """

    lanes = require_lane_count(lanes, label="lane rows lanes")
    if not 1 <= rows <= CHUNK_SIZE or lanes * rows > CHUNK_SIZE:
        raise ValueError(f"lane rows admit B x R <= {CHUNK_SIZE} rows in one tile, got {lanes} x {rows}")
    expand = torch.zeros(lanes * CHUNK_SIZE, CHUNK_SIZE)
    for lane in range(lanes):
        for row in range(rows):
            expand[lane * CHUNK_SIZE + row, lane * rows + row] = 1.0
    stack = torch.zeros(CHUNK_SIZE, CHUNK_SIZE * CONV_WINDOW_TILE_ROWS)
    for committed in range(rows + 1):
        select = torch.zeros(CHUNK_SIZE, CONV_WINDOW_TILE_ROWS)
        for index in range(CONV_HISTORY_ROWS):
            select[index, _window_buffer_row(committed + index)] = 1.0
        stack[committed] = select.reshape(-1)
    return {"expand_select": expand, "fold_select": expand.t().contiguous(), "history_select_stack": stack}


@dataclass(frozen=True)
class Qwen38TTNNGDNLaneRowsConstants:
    """Read-only device tensors of the lane rows path for one ``(lanes, rows)``, beside the ``rows``-row
    :class:`Qwen38TTNNGDNRowsConstants` (whose ``qk_expand`` and chunk tiles the lane path shares).

    ``expand_select`` ``[1,1,B*32,32]`` / ``fold_select`` ``[1,1,32,B*32]`` BF16; ``history_select_stack``
    ``[1,1,32,2048]`` BF16 indexed by committed rows (the KEEP row at 0); ``conv_taps`` 3 x ``[1,B,32,64]`` BF16 (the
    rows path's tap selects, one copy per lane: the A operand of an equal-batch matmul carries the batch);
    ``row_mask_bf16`` ``[B,32,1,1]``, ``row_mask_bf16_col`` / ``row_mask_fp32`` ``[1,B,32,1]`` (1.0 below ``rows``);
    ``arange`` ``[1,B,32,1]`` and ``arange_row`` ``[1,B,1,32]`` FP32 row indices per lane (the per-lane comparisons
    against a ``[1,B,1,1]`` count).
    """

    lanes: int
    rows: int
    expand_select: Any
    fold_select: Any
    history_select_stack: Any
    conv_taps: tuple[Any, Any, Any]
    row_mask_bf16: Any
    row_mask_bf16_col: Any
    row_mask_fp32: Any
    arange: Any
    arange_row: Any
    select_compute_config: Any
    mesh_contract: Qwen38MeshContract

    @classmethod
    def allocate(
        cls, mesh_device, mesh_contract: Qwen38MeshContract, *, lanes: int, rows: int
    ) -> "Qwen38TTNNGDNLaneRowsConstants":
        mesh_contract.validate_mesh(mesh_device)
        selects = lane_rows_select_tiles(lanes, rows)
        taps = rows_window_select_tiles(rows)["conv_taps"]  # [3, 32, 64]
        keep = (torch.arange(CHUNK_SIZE) < rows).float()
        arange = torch.arange(CHUNK_SIZE).float()
        uploaded: list[Any] = []

        def upload(host: torch.Tensor, dtype, label: str):
            tensor = _upload_replicated_constant(mesh_device, mesh_contract, host, dtype, label=label)
            uploaded.append(tensor)
            return tensor

        try:
            return cls(
                lanes=lanes,
                rows=rows,
                expand_select=upload(
                    selects["expand_select"].reshape(1, 1, lanes * CHUNK_SIZE, CHUNK_SIZE), ttnn.bfloat16, "lane expand"
                ),
                fold_select=upload(
                    selects["fold_select"].reshape(1, 1, CHUNK_SIZE, lanes * CHUNK_SIZE), ttnn.bfloat16, "lane fold"
                ),
                history_select_stack=upload(
                    selects["history_select_stack"].reshape(1, 1, CHUNK_SIZE, CHUNK_SIZE * CONV_WINDOW_TILE_ROWS),
                    ttnn.bfloat16,
                    "lane history select stack",
                ),
                conv_taps=tuple(
                    upload(
                        taps[tap]
                        .reshape(1, 1, CHUNK_SIZE, CONV_WINDOW_TILE_ROWS)
                        .expand(1, lanes, -1, -1)
                        .contiguous(),
                        ttnn.bfloat16,
                        f"lane conv tap {tap} select",
                    )
                    for tap in range(CONV_HISTORY_ROWS)
                ),
                row_mask_bf16=upload(
                    keep.reshape(1, CHUNK_SIZE, 1, 1).expand(lanes, -1, -1, -1).contiguous().to(torch.bfloat16),
                    ttnn.bfloat16,
                    "lane row mask",
                ),
                row_mask_bf16_col=upload(
                    keep.reshape(1, 1, CHUNK_SIZE, 1).expand(1, lanes, -1, -1).contiguous().to(torch.bfloat16),
                    ttnn.bfloat16,
                    "lane row mask column",
                ),
                row_mask_fp32=upload(
                    keep.reshape(1, 1, CHUNK_SIZE, 1).expand(1, lanes, -1, -1).contiguous(),
                    ttnn.float32,
                    "lane row mask fp32",
                ),
                arange=upload(
                    arange.reshape(1, 1, CHUNK_SIZE, 1).expand(1, lanes, -1, -1).contiguous(),
                    ttnn.float32,
                    "lane arange",
                ),
                arange_row=upload(
                    arange.reshape(1, 1, 1, CHUNK_SIZE).expand(1, lanes, -1, -1).contiguous(),
                    ttnn.float32,
                    "lane arange row",
                ),
                select_compute_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                ),
                mesh_contract=mesh_contract,
            )
        except BaseException:
            _deallocate(*uploaded)
            raise

    def deallocate(self) -> None:
        _deallocate(
            self.expand_select,
            self.fold_select,
            self.history_select_stack,
            *self.conv_taps,
            self.row_mask_bf16,
            self.row_mask_bf16_col,
            self.row_mask_fp32,
            self.arange,
            self.arange_row,
        )


@dataclass
class Qwen38TTNNGDNLaneRowsState:
    """Per-layer persistent buffers of the lane rows path (fixed addresses across passes): the rows state of
    :class:`Qwen38TTNNGDNRowsState` with the lane on a batch axis.

    ``qkv`` ``[1,1,32,2560]`` is the lane-major projected tile; ``qkv_lanes`` ``[1,1,B*32,2560]`` its expand (lane
    u's rows at ``u*32 .. u*32 + R - 1``, viewed ``[1,B,32,2560]`` by the batched ops); ``history`` ``[1,B,32,2560]``
    (rows 0..2 of every lane valid); ``q`` / ``k`` ``[B,32,12,128]``, ``v`` ``[1,B,32,1536]``, ``beta`` / ``g``
    ``[1,B,32,12]`` FP32 the chunk kernel's inputs, kept for the commit's masked rerun.
    """

    layer_index: int
    lanes: int
    constants: Qwen38TTNNGDNRowsConstants
    lane_constants: Qwen38TTNNGDNLaneRowsConstants
    history: Any
    qkv: Any
    qkv_lanes: Any
    q: Any
    k: Any
    v: Any
    beta: Any
    g: Any
    mesh_contract: Qwen38MeshContract
    # The verify-rows fold's lanes buffers (``fused.gdn_rows_scan.attach_lanes``: the ``[B * R, 12, 128, 128]`` prefix
    # states), freed with the state; while they are here the fold owns the lane recurrence and the lane commit.
    scan_buffers: Any = None

    @classmethod
    def allocate(
        cls,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        constants: Qwen38TTNNGDNRowsConstants,
        lane_constants: Qwen38TTNNGDNLaneRowsConstants,
        *,
        layer_index: int,
    ) -> "Qwen38TTNNGDNLaneRowsState":
        mesh_contract.validate_mesh(mesh_device)
        if constants.mesh_contract != mesh_contract or lane_constants.mesh_contract != mesh_contract:
            raise ValueError("GDN lane rows constants belong to a different physical mesh contract")
        if constants.rows != lane_constants.rows or constants.tile_rows != CHUNK_SIZE:
            raise ValueError(
                f"GDN lane rows need the {lane_constants.rows}-row one-tile constants, got rows {constants.rows} "
                f"on {constants.tile_rows} kernel rows"
            )
        lanes = lane_constants.lanes
        allocated: list[Any] = []

        def zero(local_shape: tuple[int, ...], dtype, shard_dim: int, label: str):
            tensor = _allocate_head_sharded_zero(
                mesh_device, mesh_contract, local_shape=local_shape, dtype=dtype, shard_dim=shard_dim, label=label
            )
            allocated.append(tensor)
            return tensor

        qk_shape = (lanes, CHUNK_SIZE, VALUE_HEADS_PER_DEVICE, HEAD_DIM)
        try:
            result = cls(
                layer_index=layer_index,
                lanes=lanes,
                constants=constants,
                lane_constants=lane_constants,
                history=zero((1, lanes, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE), ttnn.bfloat16, 3, "GDN lane rows history"),
                qkv=zero((1, 1, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE), ttnn.bfloat16, 3, "GDN lane rows qkv"),
                qkv_lanes=zero(
                    (1, 1, lanes * CHUNK_SIZE, QKV_WIDTH_PER_DEVICE), ttnn.bfloat16, 3, "GDN lane rows qkv lanes"
                ),
                q=zero(qk_shape, ttnn.bfloat16, 2, "GDN lane rows q"),
                k=zero(qk_shape, ttnn.bfloat16, 2, "GDN lane rows k"),
                v=zero((1, lanes, CHUNK_SIZE, VALUE_WIDTH_PER_DEVICE), ttnn.bfloat16, 3, "GDN lane rows v"),
                beta=zero((1, lanes, CHUNK_SIZE, VALUE_HEADS_PER_DEVICE), ttnn.float32, 3, "GDN lane rows beta"),
                g=zero((1, lanes, CHUNK_SIZE, VALUE_HEADS_PER_DEVICE), ttnn.float32, 3, "GDN lane rows log decay"),
                mesh_contract=mesh_contract,
            )
            result.validate()
            return result
        except BaseException:
            _deallocate(*allocated)
            raise

    def validate(self) -> None:
        owned = (self.history, self.qkv, self.qkv_lanes, self.q, self.k, self.v, self.beta, self.g)
        if len({_tensor_key(tensor) for tensor in owned}) != len(owned):
            raise RuntimeError("GDN lane rows state requires eight distinct backing tensors")
        lanes = self.lanes
        expected = {
            "history": ((1, lanes, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE), ttnn.bfloat16, 3),
            "qkv": ((1, 1, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE), ttnn.bfloat16, 3),
            "qkv_lanes": ((1, 1, lanes * CHUNK_SIZE, QKV_WIDTH_PER_DEVICE), ttnn.bfloat16, 3),
            "q": ((lanes, CHUNK_SIZE, VALUE_HEADS_PER_DEVICE, HEAD_DIM), ttnn.bfloat16, 2),
            "k": ((lanes, CHUNK_SIZE, VALUE_HEADS_PER_DEVICE, HEAD_DIM), ttnn.bfloat16, 2),
            "v": ((1, lanes, CHUNK_SIZE, VALUE_WIDTH_PER_DEVICE), ttnn.bfloat16, 3),
            "beta": ((1, lanes, CHUNK_SIZE, VALUE_HEADS_PER_DEVICE), ttnn.float32, 3),
            "g": ((1, lanes, CHUNK_SIZE, VALUE_HEADS_PER_DEVICE), ttnn.float32, 3),
        }
        for name, (shape, dtype, shard_dim) in expected.items():
            tensor = getattr(self, name)
            _require_shape(tensor, shape, label=f"GDN lane rows {name}")
            if tensor.dtype != dtype:
                raise RuntimeError(f"GDN lane rows {name} must be {dtype}, got {tensor.dtype}")
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HEAD_SHARDED, shard_dim=shard_dim)

    def deallocate(self) -> None:
        _deallocate(self.history, self.qkv, self.qkv_lanes, self.q, self.k, self.v, self.beta, self.g)
        if self.scan_buffers is not None:
            self.scan_buffers.deallocate()
            self.scan_buffers = None


@dataclass(frozen=True)
class Qwen38TTNNRowsSelectorsLanes:
    """Per-pass device selects of the lane commit, derived once from the per-lane accept counts and the active mask
    and shared by every layer's commit.

    ``c_u = (a_u + 1) * active_u`` committed rows per lane (``a_u = -1`` for a seeded, not yet verified lane, 0 for
    an inactive one: the KEEP row).  ``committed_mask`` ``[1,B,32,1]`` FP32 = ``arange < c_u``; ``history_select``
    ``[1,B,32,64]`` BF16 the stack row ``c_u`` per lane; ``commit_col`` / ``keep_col`` ``[B,1,1,1]`` FP32 = ``[c_u >= 1]``
    and its complement (the exact per-lane state select of the commit: an inactive or freshly seeded lane keeps its
    recurrent state bitwise); ``onehot_bf16[c]`` ``[B,1,1,1]`` BF16 = ``c_u == c`` for ``c = 0 .. R``
    (the PLE lane commit's multiply/add select, the KEEP candidate at 0).
    """

    lanes: int
    rows: int
    committed_mask: Any
    history_select: Any
    commit_col: Any
    keep_col: Any
    onehot_bf16: tuple[Any, ...]
    # ``c_u`` itself, ``[1,B,1,1]`` FP32 TILE: the verify-rows fold's lane commit reads it on the device (the pick of
    # prefix slot ``u * R + c_u - 1``; ``c_u == 0`` keeps the lane's state).  None on selectors built without it.
    committed_counts: Any = None

    def validate(self, lanes: int, rows: int) -> None:
        if (self.lanes, self.rows) != (lanes, rows) or len(self.onehot_bf16) != rows + 1:
            raise ValueError(
                f"lane selectors were built for {self.lanes} x {self.rows}, the path runs {lanes} x {rows}"
            )
        if self.committed_counts is not None:
            _require_shape(self.committed_counts, (1, lanes, 1, 1), label="lane committed counts")
            if self.committed_counts.dtype != ttnn.float32:
                raise RuntimeError(f"lane committed counts must be FP32, got {self.committed_counts.dtype}")
        _require_shape(self.committed_mask, (1, lanes, CHUNK_SIZE, 1), label="lane committed rows mask")
        _require_shape(self.history_select, (1, lanes, CHUNK_SIZE, CONV_WINDOW_TILE_ROWS), label="lane history select")
        for name, tensor in (("commit_col", self.commit_col), ("keep_col", self.keep_col)):
            _require_shape(tensor, (lanes, 1, 1, 1), label=f"lane selector {name}")
            if tensor.dtype != ttnn.float32:
                raise RuntimeError(f"lane selector {name} must be FP32, got {tensor.dtype}")
        if self.committed_mask.dtype != ttnn.float32 or self.history_select.dtype != ttnn.bfloat16:
            raise RuntimeError(
                f"lane selector dtypes are {self.committed_mask.dtype} (mask) / {self.history_select.dtype} (select)"
            )
        for index, bf16 in enumerate(self.onehot_bf16):
            _require_shape(bf16, (lanes, 1, 1, 1), label=f"lane selector onehot_bf16[{index}]")
            if bf16.dtype != ttnn.bfloat16:
                raise RuntimeError(f"lane selector onehot_bf16[{index}] dtype is {bf16.dtype}")

    def deallocate(self) -> None:
        _deallocate(
            self.committed_mask,
            self.history_select,
            self.commit_col,
            self.keep_col,
            *self.onehot_bf16,
            self.committed_counts,
        )


def build_rows_selectors_lanes(
    accepted_lanes, active_lanes, constants: Qwen38TTNNGDNLaneRowsConstants
) -> Qwen38TTNNRowsSelectorsLanes:
    """Derive the lane commit selects from the per-lane accept counts and the active mask (both FP32 TILE
    ``[1,B,1,1]``, replicated): ``rows + 8`` tiny ops per pass, not per layer.  Comparisons on fp32 integers are
    exact; the history select is one row of the stack per lane (``onehot_row @ stack``, one nonzero term)."""

    lanes, rows = constants.lanes, constants.rows
    for name, tensor in (("accept counts", accepted_lanes), ("active mask", active_lanes)):
        _require_shape(tensor, (1, lanes, 1, 1), label=f"lane {name}")
        if tensor.dtype != ttnn.float32:
            raise RuntimeError(f"lane {name} must be FP32, got {tensor.dtype}")
    dram = ttnn.DRAM_MEMORY_CONFIG
    plus_one = ttnn.add(accepted_lanes, 1.0, memory_config=dram)
    committed = ttnn.multiply(plus_one, active_lanes, memory_config=dram)
    committed_mask = ttnn.lt(constants.arange, committed, memory_config=dram)
    onehot_row = ttnn.eq(constants.arange_row, committed, dtype=ttnn.bfloat16, memory_config=dram)
    _require_shape(onehot_row, (1, lanes, 1, CHUNK_SIZE), label="lane committed one-hot row")
    select_flat = ttnn.matmul(
        onehot_row,
        constants.history_select_stack,
        memory_config=dram,
        compute_kernel_config=constants.select_compute_config,
    )
    _require_shape(select_flat, (1, lanes, 1, CHUNK_SIZE * CONV_WINDOW_TILE_ROWS), label="lane history select row")
    history_select = ttnn.reshape(select_flat, (1, lanes, CHUNK_SIZE, CONV_WINDOW_TILE_ROWS))
    # [1, B, 1, 1] -> [B, 1, 1, 1]: the same B one-element tiles, a metadata view.
    committed_col = ttnn.experimental.view(committed, (lanes, 1, 1, 1))
    commit_col = ttnn.ge(committed_col, 1.0, memory_config=dram)
    keep_col = ttnn.rsub(commit_col, 1.0, memory_config=dram)
    onehot_bf16 = tuple(
        ttnn.eq(committed_col, float(count), dtype=ttnn.bfloat16, memory_config=dram) for count in range(rows + 1)
    )
    # ``committed`` (``committed_col`` is its view) stays: the fold's lane commit reads the counts on the device
    _deallocate(plus_one, onehot_row, select_flat)
    selectors = Qwen38TTNNRowsSelectorsLanes(
        lanes, rows, committed_mask, history_select, commit_col, keep_col, onehot_bf16, committed_counts=committed
    )
    selectors.validate(lanes, rows)
    return selectors


@dataclass(frozen=True)
class Qwen38TTNNGDNRowsResult:
    """``hidden_rows`` is ``rows_state.output`` (``[1,1,rows,640]``, persistent: read it before the next pass,
    never deallocate it) or, for ``forward_rows(full_tile=True)``, the whole ``[1,1,32,640]`` output tile in a new
    buffer the caller deallocates; ``final_state`` is the chunk kernel's state after all ``rows`` rows in a new FP32
    buffer (the committed state is untouched until ``commit_rows``); the caller owns and deallocates it -- or ``None``
    under the verify-rows fold (``QWEN38_FUSED=gdn_rows_scan``), whose state after all rows is the last of the prefix
    states the rows state owns (``commit_rows_full`` reads it from there)."""

    hidden_rows: Any
    final_state: Any
    state: Qwen38TTNNGDNState
    rows_state: Qwen38TTNNGDNRowsState


class Qwen38TTNNGDN:
    """One exact TP4 Qwen3.8 Gated DeltaNet layer."""

    # The prefill slab's dense-linear policy (ttnn/prefill_dense: the QWEN38_PREFILL_DENSE_* switches); the decode
    # linears never read it.
    prefill_dense: Qwen38TTNNPrefillDense | None = None

    def __init__(
        self,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        weights: Qwen38TTNNGDNWeights,
        *,
        collective_topology=None,
        prefill_dense: Qwen38TTNNPrefillDense | None = None,
    ) -> None:
        mesh_contract.validate_mesh(mesh_device)
        weights.validate(mesh_contract)
        self.prefill_dense = Qwen38TTNNPrefillDense.resolve(prefill_dense, mesh_device)
        workers = validate_decode_dram_workers(weights.decode_dram_workers_per_bank)
        validate_dram_sharded_weight(
            weights.qkvzab,
            mesh_device,
            HIDDEN_SIZE,
            PROJECTION_WIDTH_PER_DEVICE,
            num_workers_per_dram_bank=workers,
        )
        validate_dram_sharded_weight(
            weights.out,
            mesh_device,
            VALUE_WIDTH_PER_DEVICE,
            HIDDEN_SIZE,
            num_workers_per_dram_bank=workers,
        )
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
        # The projection linears (qkvzab, out) run the fidelity of their weight format (decode_matmul: HiFi4 for bf16,
        # HiFi2 for bf8, LoFi for bf4); every other program keeps compute_config.
        self.projection_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, dense_math_fidelity_name(weights.projection_dtype)),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.in_proj_act_memory_config, self.in_proj_program_config = dram_sharded_matmul_configs(
            mesh_device, HIDDEN_SIZE, PROJECTION_WIDTH_PER_DEVICE, num_cores=8, num_workers_per_dram_bank=workers
        )
        self.out_proj_act_memory_config, self.out_proj_program_config = dram_sharded_matmul_configs(
            mesh_device, VALUE_WIDTH_PER_DEVICE, HIDDEN_SIZE, num_cores=16, num_workers_per_dram_bank=workers
        )
        # Recurrent step kernels: the qualified configuration of the shared FLA
        # decode step for K = V = 128 (HiFi2, FP32 accumulation).  Changing any
        # field moves a rounding point.
        head_tiles = HEAD_DIM // ttnn.TILE_SIZE
        self.recurrent_matmul_program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            in0_block_w=head_tiles,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=1,
            per_core_N=head_tiles,
        )
        self.recurrent_read_compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.recurrent_write_compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # projection -> gated output: the composed chain, or the fused kernel when QWEN38_FUSED names gdn_step
        self._step = fused.resolve("gdn_step")
        # The prefill slab's GDN body: the two rows programs around the chunk prims when QWEN38_FUSED names
        # gdn_prefill_rows and the pass's rows state carries their buffers, today's chain otherwise.  Resolved once,
        # so a trace capture keeps the choice; the 32-row, 128-row, lane and decode bodies never read it.
        self._prefill_rows = fused.resolve_admitted("gdn_prefill_rows")
        self._prefill_rows_on = fused.enabled("gdn_prefill_rows")
        # the rows recurrence (forward_rows, commit_rows): the composite, or the two chunk prims called directly under
        # QWEN38_FUSED=gdn_rows_prims_direct where the call's tensors meet the form's contract (shapes only, so the
        # warm pass and the capture take one branch)
        self._rows_chunk = fused.resolve_admitted("gdn_rows_prims_direct")
        # the rows body between the projection and the out-projection: the wrap's two programs around the two prims
        # (a default) where the rows state carries its buffers, the chain elsewhere and under QWEN38_FUSED_OFF=gdn_rows_wrap
        self._rows_body_call = fused.resolve_admitted("gdn_rows_wrap")
        # the verify rows' fold (opt-in QWEN38_FUSED=gdn_rows_scan): one serial-recurrence program where the rows state
        # carries its prefix states, today's stream (the wrap or the chain, through _rows_body_wrap) elsewhere
        self._rows_scan_call = fused.resolve_admitted("gdn_rows_scan")
        # The prefill glue policy (QWEN38_PREFILL_GLUE), resolved once; read when a slab rows state is allocated.
        self.glue = prefill_glue.policy()
        # The lane body runs the same program on its B rows (one item per (lane, value head), one state slot per
        # lane); with the kernel off it stays the lane chain, the class attribute.
        if fused.enabled("gdn_step"):
            self.forward_decode_lanes = functools.partial(type(self)._forward_decode_lanes_fused, self)

    def allocate_state(self) -> Qwen38TTNNGDNState:
        return Qwen38TTNNGDNState.allocate(
            self.mesh_device,
            self.mesh_contract,
            layer_index=self.weights.layer_index,
            batch_size=1,
        )

    def _gdn_step(self):
        """The projection -> gated-output step: the fused kernel when it serves (``gdn_step`` is a default;
        QWEN38_FUSED_OFF=gdn_step restores the chain) and the call's tensors meet its input contract, the composed
        chain otherwise; resolved once (fakes that skip ``__init__`` resolve here)."""

        step = self.__dict__.get("_step")
        if step is None:
            step = self._step = fused.resolve_admitted("gdn_step")
        return step

    def _gdn_prefill_rows(self):
        """The slab body: the fused pair when ``QWEN38_FUSED`` names ``gdn_prefill_rows`` and the call's rows state
        meets its input contract, today's chain otherwise; resolved once (fakes that skip ``__init__`` here)."""

        body = self.__dict__.get("_prefill_rows")
        if body is None:
            body = self._prefill_rows = fused.resolve_admitted("gdn_prefill_rows")
        return body

    def _fused_prefill_rows_buffers(self) -> bool:
        """Whether a slab rows state allocated now carries the fused prefill rows buffers: the switch alone, read
        once per module, so the buffers a capture binds match the callable that was resolved with it."""

        on = self.__dict__.get("_prefill_rows_on")
        if on is None:
            on = self._prefill_rows_on = fused.enabled("gdn_prefill_rows")
        return on

    def _chunk_rows_kernel(self):
        """The rows recurrence over one chunk: the composite ``ttnn.transformer.chunk_gated_delta_rule`` (the default)
        or, under QWEN38_FUSED=gdn_rows_prims_direct, its two phase prims called directly where the call's tensors meet
        the form's contract (ttnn/fused/gdn_rows_prims_direct); resolved once (fakes that skip ``__init__`` resolve
        here)."""

        kernel = self.__dict__.get("_rows_chunk")
        if kernel is None:
            kernel = self._rows_chunk = fused.resolve_admitted("gdn_rows_prims_direct")
        return kernel

    def _rows_body_wrap(self):
        """Today's rows body from the projection to the out-projection: on a rows state that carries the wrap's
        buffers (the 32-row verify tile, by default) ``gdn_pre_rows`` + the two chunk prims + ``gdn_post_rows``
        (ttnn/fused/gdn_rows_wrap), the chain elsewhere and under QWEN38_FUSED_OFF=gdn_rows_wrap; resolved once (fakes
        that skip ``__init__`` resolve here).  The verify-rows fold's composed callable."""

        body = self.__dict__.get("_rows_body_call")
        if body is None:
            body = self._rows_body_call = fused.resolve_admitted("gdn_rows_wrap")
        return body

    def _rows_body(self):
        """The rows body ``forward_rows`` dispatches to: on a rows state that carries the verify-rows fold's buffers
        (opt-in QWEN38_FUSED=gdn_rows_scan) the one-program serial recurrence (ttnn/fused/gdn_rows_scan), else today's
        stream through ``_rows_body_wrap``; resolved once (fakes that skip ``__init__`` resolve here)."""

        body = self.__dict__.get("_rows_scan_call")
        if body is None:
            body = self._rows_scan_call = fused.resolve_admitted("gdn_rows_scan")
        return body

    def _validate_state(self, state: Qwen38TTNNGDNState) -> None:
        if state.layer_index != self.weights.layer_index:
            raise ValueError(f"GDN layer {self.weights.layer_index} received state owned by layer {state.layer_index}")
        if state.mesh_contract != self.mesh_contract:
            raise ValueError("GDN state belongs to a different physical mesh contract")
        state.validate()
        if state.batch_size != 1:
            raise ValueError(f"the 1-row GDN decode body admits a batch-1 state, got batch {state.batch_size}")

    def _all_gather_hidden(self, hidden_sharded):
        _require_shape(hidden_sharded, (1, 1, 1, HIDDEN_SIZE_PER_DEVICE), label="GDN decode input")
        self.mesh_contract.validate_tensor(hidden_sharded, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        # The gather writes the in-projection's eight-core activation layout;
        # the tensor stays allocated through the layer (20 KB per core).
        full_hidden = ttnn.all_gather(
            hidden_sharded,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=self.in_proj_act_memory_config,
        )
        self.mesh_contract.validate_tensor(full_hidden, placement=TensorPlacement.REPLICATED)
        _require_shape(full_hidden, (1, 1, 1, HIDDEN_SIZE), label="GDN gathered hidden")
        return full_hidden

    def _project(self, full_hidden):
        """Project the gathered hidden row into one L1 row tile ``[1,1,1,4160]`` = q|k|v|z|a|b, a and b tile-aligned."""

        projected_ws = ttnn.linear(
            full_hidden,
            self.weights.qkvzab,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.in_proj_program_config,
            compute_kernel_config=self.projection_compute_config,
        )
        projected = ttnn.to_memory_config(projected_ws, ttnn.L1_MEMORY_CONFIG)
        _deallocate(projected_ws)
        self.mesh_contract.validate_tensor(projected, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        _require_shape(projected, (1, 1, 1, PROJECTION_WIDTH_PER_DEVICE), label="GDN fused projection")
        return projected

    def _split_projection(self, projected, newest):
        """The q/k/v columns land in ``newest``, the ring slot for this token; z, a, b are returned."""

        # Every slice starts on a tile boundary, so each is one device op; the
        # newest-token slice writes the persistent slot (no staging copy).
        ttnn.slice(projected, (0, 0, 0, 0), (1, 1, 1, QKV_WIDTH_PER_DEVICE), output_tensor=newest)
        z = ttnn.slice(
            projected,
            (0, 0, 0, QKV_WIDTH_PER_DEVICE),
            (1, 1, 1, A_COLUMN),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        a = ttnn.slice(
            projected,
            (0, 0, 0, A_COLUMN),
            (1, 1, 1, A_COLUMN + VALUE_HEADS_PER_DEVICE),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        b = ttnn.slice(
            projected,
            (0, 0, 0, B_COLUMN),
            (1, 1, 1, B_COLUMN + VALUE_HEADS_PER_DEVICE),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _deallocate(projected)
        for name, tensor, expected in (
            ("z", z, (1, 1, 1, VALUE_WIDTH_PER_DEVICE)),
            ("a", a, (1, 1, 1, VALUE_HEADS_PER_DEVICE)),
            ("b", b, (1, 1, 1, VALUE_HEADS_PER_DEVICE)),
        ):
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
            _require_shape(tensor, expected, label=f"GDN projected {name}")
        return z, a, b

    def _causal_conv_decode(self, window):
        # ``window`` is oldest -> newest; tap i pairs with window[i].
        conv = ttnn.multiply(window[0], self.weights.conv_taps[0], memory_config=ttnn.L1_MEMORY_CONFIG)
        for index in range(1, CONV_KERNEL_SIZE):
            previous = conv
            conv = ttnn.mac(window[index], self.weights.conv_taps[index], previous)
            if _tensor_key(conv) != _tensor_key(previous):
                _deallocate(previous)
        conv = ttnn.silu(conv, memory_config=ttnn.L1_MEMORY_CONFIG)
        self.mesh_contract.validate_tensor(conv, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        _require_shape(conv, (1, 1, 1, QKV_WIDTH_PER_DEVICE), label="GDN causal convolution")
        return conv

    def _make_recurrent_inputs(self, conv, a, b):
        q_slice = ttnn.slice(
            conv,
            (0, 0, 0, 0),
            (1, 1, 1, QK_WIDTH_PER_DEVICE),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        k_slice = ttnn.slice(
            conv,
            (0, 0, 0, QK_WIDTH_PER_DEVICE),
            (1, 1, 1, 2 * QK_WIDTH_PER_DEVICE),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        v_slice = ttnn.slice(
            conv,
            (0, 0, 0, 2 * QK_WIDTH_PER_DEVICE),
            (1, 1, 1, QKV_WIDTH_PER_DEVICE),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _deallocate(conv)

        # Heads go on dim 1 so every recurrent operand is already in matmul
        # form [1,H,1,128]: the head split is one repack per operand and the
        # GQA repeat over dim 1 is one page-copy op.  k's tile padding is
        # zeroed because k^T's padding lanes enter the outer-product
        # contraction (0 * finite = 0); q's and v's padding never does.
        q_heads = ttnn.reshape(q_slice, (1, QK_HEADS_PER_DEVICE, 1, HEAD_DIM))
        k_heads = ttnn.reshape(k_slice, (1, QK_HEADS_PER_DEVICE, 1, HEAD_DIM), pad_value=0.0)
        q = ttnn.repeat_interleave(q_heads, QK_REPEAT_FACTOR, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        k = ttnn.repeat_interleave(k_heads, QK_REPEAT_FACTOR, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        v = ttnn.reshape(v_slice, (1, VALUE_HEADS_PER_DEVICE, 1, HEAD_DIM))
        for tensor, reference in ((q, q_slice), (k, k_slice), (v, v_slice)):
            _retag_head_shard_after_reshape(tensor, reference=reference, shard_dim=1)
        _deallocate(q_slice, k_slice, v_slice, q_heads, k_heads)
        for name, tensor in (("query", q), ("key", k), ("value", v)):
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HEAD_SHARDED, shard_dim=1)
            _require_shape(
                tensor,
                (1, VALUE_HEADS_PER_DEVICE, 1, HEAD_DIM),
                label=f"GDN recurrent {name}",
            )

        # The authenticated token17 layer-0 boundary requires the explicit
        # FP32 sigmoid path; beta stays FP32 straight into the FP32 recurrent
        # step. Retain every producer until the recurrent consumer is
        # enqueued: reshape/topology metadata does not transfer the underlying
        # buffer lifetime.  [1,H,1,1] is the broadcast form the step consumes.
        b_fp32 = ttnn.typecast(b, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
        beta_fp32 = ttnn.sigmoid(b_fp32, memory_config=ttnn.L1_MEMORY_CONFIG)
        beta = ttnn.reshape(beta_fp32, (1, VALUE_HEADS_PER_DEVICE, 1, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        _retag_head_shard_after_reshape(beta, reference=beta_fp32, shard_dim=1)
        beta_producers = (b, b_fp32, beta_fp32)

        a_fp32 = ttnn.typecast(a, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
        _deallocate(a)
        softplus = softplus_gate(a_fp32, self.weights.dt_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
        _deallocate(a_fp32)
        log_decay_raw = ttnn.multiply(
            self.weights.neg_exp_A,
            softplus,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _deallocate(softplus)
        log_decay = ttnn.reshape(log_decay_raw, (1, VALUE_HEADS_PER_DEVICE, 1, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        _retag_head_shard_after_reshape(log_decay, reference=log_decay_raw, shard_dim=1)
        beta_producers = (*beta_producers, log_decay_raw)
        for name, tensor in (("beta", beta), ("log_decay", log_decay)):
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HEAD_SHARDED, shard_dim=1)
            _require_shape(tensor, (1, VALUE_HEADS_PER_DEVICE, 1, 1), label=f"GDN recurrent {name}")
        if log_decay.dtype != ttnn.float32:
            raise RuntimeError(f"GDN log decay must be FP32, got {log_decay.dtype}")
        return q, k, v, beta, log_decay, beta_producers

    def _recurrent_decode(self, q, k, v, beta, log_decay, beta_producers, state: Qwen38TTNNGDNState):
        """One FP32 gated delta-rule step on [1,H,1,128] operands, written into ``state.recurrent``.

        q and k are l2-normalized in BF16 before the FP32 promotion (the pinned
        Qwen4Exp order); beta and the log decay arrive FP32 and are used as is.
        v is promoted by the subtract: BF16 -> FP32 widening is exact wherever
        it happens.
        """

        l1 = ttnn.L1_MEMORY_CONFIG
        q_normed = ttnn.rms_norm(q, epsilon=QK_L2_NORM_EPS / HEAD_DIM)
        q_unit = ttnn.multiply(q_normed, HEAD_DIM**-0.5, memory_config=l1)
        k_normed = ttnn.rms_norm(k, epsilon=QK_L2_NORM_EPS / HEAD_DIM)
        k_unit = ttnn.multiply(k_normed, HEAD_DIM**-0.5, memory_config=l1)
        _deallocate(q, k, q_normed, k_normed)
        q_fp32 = ttnn.typecast(q_unit, ttnn.float32, memory_config=l1)
        k_row = ttnn.typecast(k_unit, ttnn.float32, memory_config=l1)
        _deallocate(q_unit, k_unit)
        q_row = ttnn.multiply(q_fp32, HEAD_DIM**-0.5, memory_config=l1)
        _deallocate(q_fp32)

        # Decay the state straight from its DRAM home; exp(log_decay) is fused
        # into the multiply.
        decayed = ttnn.multiply(
            state.recurrent,
            log_decay,
            input_tensor_b_activations=[ttnn.UnaryOpType.EXP],
            memory_config=l1,
        )
        v_read = ttnn.matmul(
            k_row,
            decayed,
            memory_config=l1,
            program_config=self.recurrent_matmul_program_config,
            compute_kernel_config=self.recurrent_read_compute_config,
        )
        delta = ttnn.subtract(v, v_read, dtype=ttnn.float32, memory_config=l1)
        _deallocate(v, v_read)
        k_col = ttnn.transpose(k_row, 2, 3, memory_config=l1)
        _deallocate(k_row)
        outer = ttnn.matmul(
            k_col,
            delta,
            memory_config=l1,
            compute_kernel_config=self.recurrent_write_compute_config,
        )
        _deallocate(k_col, delta)
        update = ttnn.multiply(outer, beta, memory_config=l1)
        _deallocate(outer)
        # The state add lands in the persistent buffer: fixed address, no copy.
        new_recurrent = ttnn.add(decayed, update, output_tensor=state.recurrent)
        _deallocate(decayed, update, beta, log_decay, *beta_producers)
        if new_recurrent is not None and _tensor_key(new_recurrent) != _tensor_key(state.recurrent):
            raise RuntimeError("GDN recurrent update did not land in the persistent state")
        _require_shape(
            state.recurrent,
            (1, VALUE_HEADS_PER_DEVICE, HEAD_DIM, HEAD_DIM),
            label="GDN next recurrent state",
        )
        if state.recurrent.dtype != ttnn.float32:
            raise RuntimeError(f"GDN next recurrent state must be FP32, got {state.recurrent.dtype}")

        output = ttnn.matmul(
            q_row,
            state.recurrent,
            memory_config=l1,
            program_config=self.recurrent_matmul_program_config,
            compute_kernel_config=self.recurrent_read_compute_config,
        )
        _deallocate(q_row)
        _require_shape(output, (1, VALUE_HEADS_PER_DEVICE, 1, HEAD_DIM), label="GDN recurrent output")
        return output

    def _gate(self, recurrent_output, z):
        # The serial reference returns a BF16 per-token recurrent output while
        # retaining the carried state in FP32.  Match that boundary explicitly.
        # The per-head RMSNorm runs on the [1,H,1,128] step output directly.
        output_bf16 = (
            recurrent_output
            if recurrent_output.dtype == ttnn.bfloat16
            else ttnn.typecast(recurrent_output, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
        )
        if output_bf16 is not recurrent_output:
            _deallocate(recurrent_output)
        normalized_heads = ttnn.rms_norm(
            output_bf16,
            weight=self.weights.norm,
            epsilon=RMS_NORM_EPS,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _deallocate(output_bf16)
        normalized = ttnn.reshape(normalized_heads, (1, 1, 1, VALUE_WIDTH_PER_DEVICE))
        _retag_head_shard_after_reshape(normalized, reference=normalized_heads, shard_dim=3)
        _deallocate(normalized_heads)
        self.mesh_contract.validate_tensor(normalized, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        _require_shape(normalized, (1, 1, 1, VALUE_WIDTH_PER_DEVICE), label="GDN normalized output")

        # Exact Qwen4Exp difference from the Qwen3.6 jumping-off point.
        z_fp32 = ttnn.typecast(z, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
        _deallocate(z)
        sigmoid_fp32 = ttnn.sigmoid(z_fp32, memory_config=ttnn.L1_MEMORY_CONFIG)
        _deallocate(z_fp32)
        sigmoid_bf16 = ttnn.typecast(sigmoid_fp32, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
        _deallocate(sigmoid_fp32)
        # The gate product is written straight into the out-projection's
        # sixteen-core activation layout.
        gated = ttnn.multiply(normalized, sigmoid_bf16, memory_config=self.out_proj_act_memory_config)
        _deallocate(normalized, sigmoid_bf16)
        self.mesh_contract.validate_tensor(gated, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        _require_shape(gated, (1, 1, 1, VALUE_WIDTH_PER_DEVICE), label="GDN sigmoid-gated output")
        return gated

    def _out_project(self, gated, full_hidden):
        partial_ws = ttnn.linear(
            gated,
            self.weights.out,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.out_proj_program_config,
            compute_kernel_config=self.projection_compute_config,
        )
        _deallocate(gated)
        # The line reduce-scatter reads the sixteen-core partial in place.
        self.mesh_contract.mark_local_partial(
            partial_ws,
            replicated_reference=full_hidden,
            expected_shape=(1, 1, 1, HIDDEN_SIZE),
        )
        output = ttnn.reduce_scatter(
            partial_ws,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.collective_topology,
        )
        _deallocate(partial_ws)
        self.mesh_contract.mark_collective_shard(
            output,
            replicated_reference=full_hidden,
            shard_dim=3,
            expected_local_shape=(1, 1, 1, HIDDEN_SIZE_PER_DEVICE),
        )
        _deallocate(full_hidden)
        return output

    def forward_decode(self, hidden_sharded, state: Qwen38TTNNGDNState) -> Qwen38TTNNGDNResult:
        """Advance one true-global-B1 token and mutate ``state`` in place."""

        self._validate_state(state)
        full_hidden = self._all_gather_hidden(hidden_sharded)
        window = state.conv_window()
        projected = self._project(full_hidden)
        step = self._gdn_step()
        gated = step(self, projected, window, state)
        state.advance_conv_window()
        output = self._out_project(gated, full_hidden)
        return Qwen38TTNNGDNResult(output, state)

    def forward_prefill(self, hidden_sharded, state: Qwen38TTNNGDNState) -> Qwen38TTNNGDNResult:
        """Exact device-only serial prefill over the ordinary decode transition.

        ``hidden_sharded`` has local shape ``[1,1,T,640]`` and must carry the
        same hidden-sharded mesh topology as decode.  This path intentionally
        establishes the state-transition golden before the chunk kernel is
        admitted; it performs no host inference and does not replace GDN with a
        dense or CPU fallback.
        """

        self._validate_state(state)
        shape = _shape(hidden_sharded)
        if len(shape) != 4 or shape[:2] != (1, 1) or shape[3] != HIDDEN_SIZE_PER_DEVICE or shape[2] <= 0:
            raise ValueError(f"GDN prefill input must have local shape [1,1,T,640], got {shape}")
        self.mesh_contract.validate_tensor(hidden_sharded, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)

        outputs = []
        for position in range(shape[2]):
            token = ttnn.slice(
                hidden_sharded,
                (0, 0, position, 0),
                (1, 1, position + 1, HIDDEN_SIZE_PER_DEVICE),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            result = self.forward_decode(token, state)
            _deallocate(token)
            outputs.append(result.hidden_sharded)
        if len(outputs) == 1:
            output = outputs[0]
        else:
            output = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            _deallocate(*outputs)
        self.mesh_contract.validate_tensor(output, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        _require_shape(output, (1, 1, shape[2], HIDDEN_SIZE_PER_DEVICE), label="GDN prefill output")
        return Qwen38TTNNGDNResult(output, state)

    # ------------------------------------------------------------------ lanes path (batched decode)
    #
    # ``forward_decode_lanes`` is ``forward_decode`` on B lanes: the same ops on ``[1,1,B,*]`` rows (row u = lane
    # u), the recurrent operands ``[B,12,1,128]`` (lane u's heads at batch index u) against the ``[B,12,128,128]``
    # state, the ring slots ``[1,1,B,2560]`` under one shared phase (the residue rule: every resident lane's
    # position is congruent to the slot mod 4).  Every op is per row / per (lane, head) tile or a batched matmul
    # over (lane, head) with the 1-row program config (per_core_M = 1), so lane u is the 1-row path on user u's
    # stream.  Lanes without a session replay harmlessly in their own rows; a lane joins through
    # ``Qwen38TTNNGDNState.reset_lane_inplace`` at a step of its position's residue class.

    def allocate_lane_state(self, lanes: int) -> Qwen38TTNNGDNState:
        return Qwen38TTNNGDNState.allocate(
            self.mesh_device, self.mesh_contract, layer_index=self.weights.layer_index, batch_size=lanes
        )

    def _validate_lane_state(self, state: Qwen38TTNNGDNState) -> int:
        if state.layer_index != self.weights.layer_index:
            raise ValueError(f"GDN layer {self.weights.layer_index} received state owned by layer {state.layer_index}")
        if state.mesh_contract != self.mesh_contract:
            raise ValueError("GDN state belongs to a different physical mesh contract")
        state.validate()
        return state.batch_size

    def _all_gather_hidden_lanes(self, hidden_lanes, lanes: int):
        _require_shape(hidden_lanes, (1, 1, lanes, HIDDEN_SIZE_PER_DEVICE), label="GDN lanes input")
        self.mesh_contract.validate_tensor(hidden_lanes, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        full_hidden = ttnn.all_gather(
            hidden_lanes,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=self.in_proj_act_memory_config,
        )
        self.mesh_contract.validate_tensor(full_hidden, placement=TensorPlacement.REPLICATED)
        _require_shape(full_hidden, (1, 1, lanes, HIDDEN_SIZE), label="GDN lanes gathered hidden")
        return full_hidden

    def _project_lanes_unsplit(self, full_hidden, lanes: int):
        """``_project`` on B rows: the ``[1,1,B,4160]`` L1 row tile, row u = lane u's q|k|v|z|a|b, before the ring-slot
        write and the z/a/b split (the fused ``gdn_step`` program takes this tile whole and does both itself)."""

        projected_ws = ttnn.linear(
            full_hidden,
            self.weights.qkvzab,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.in_proj_program_config,
            compute_kernel_config=self.projection_compute_config,
        )
        projected = ttnn.to_memory_config(projected_ws, ttnn.L1_MEMORY_CONFIG)
        _deallocate(projected_ws)
        self.mesh_contract.validate_tensor(projected, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        _require_shape(projected, (1, 1, lanes, PROJECTION_WIDTH_PER_DEVICE), label="GDN lanes fused projection")
        return projected

    def _project_lanes(self, full_hidden, newest, lanes: int):
        """``_project`` on B rows; row u's q/k/v columns land in row u of ``newest``, the ring slot of this step."""

        projected = self._project_lanes_unsplit(full_hidden, lanes)
        landed = ttnn.slice(projected, (0, 0, 0, 0), (1, 1, lanes, QKV_WIDTH_PER_DEVICE), output_tensor=newest)
        _require_landed(landed, newest, label="GDN lanes ring slot write")
        pieces = {}
        for name, (start, end) in (
            ("z", (QKV_WIDTH_PER_DEVICE, A_COLUMN)),
            ("a", (A_COLUMN, A_COLUMN + VALUE_HEADS_PER_DEVICE)),
            ("b", (B_COLUMN, B_COLUMN + VALUE_HEADS_PER_DEVICE)),
        ):
            piece = ttnn.slice(projected, (0, 0, 0, start), (1, 1, lanes, end), memory_config=ttnn.L1_MEMORY_CONFIG)
            self.mesh_contract.validate_tensor(piece, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
            _require_shape(piece, (1, 1, lanes, end - start), label=f"GDN lanes projected {name}")
            pieces[name] = piece
        _deallocate(projected)
        return pieces["z"], pieces["a"], pieces["b"]

    def _causal_conv_lanes(self, window, lanes: int):
        # ``_causal_conv_decode`` row for row: the [1,1,1,2560] taps broadcast over the lane rows.
        conv = ttnn.multiply(window[0], self.weights.conv_taps[0], memory_config=ttnn.L1_MEMORY_CONFIG)
        for index in range(1, CONV_KERNEL_SIZE):
            previous = conv
            conv = ttnn.mac(window[index], self.weights.conv_taps[index], previous)
            if _tensor_key(conv) != _tensor_key(previous):
                _deallocate(previous)
        conv = ttnn.silu(conv, memory_config=ttnn.L1_MEMORY_CONFIG)
        self.mesh_contract.validate_tensor(conv, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        _require_shape(conv, (1, 1, lanes, QKV_WIDTH_PER_DEVICE), label="GDN lanes causal convolution")
        return conv

    def _make_recurrent_inputs_lanes(self, conv, a, b, lanes: int):
        """``_make_recurrent_inputs`` on B rows: row u's heads become batch index u of the ``[B,H,1,128]`` operands
        (the same head split per row, so lane u's tiles hold what the 1-row path's would)."""

        l1 = ttnn.L1_MEMORY_CONFIG
        q_slice = ttnn.slice(conv, (0, 0, 0, 0), (1, 1, lanes, QK_WIDTH_PER_DEVICE), memory_config=l1)
        k_slice = ttnn.slice(
            conv, (0, 0, 0, QK_WIDTH_PER_DEVICE), (1, 1, lanes, 2 * QK_WIDTH_PER_DEVICE), memory_config=l1
        )
        v_slice = ttnn.slice(
            conv, (0, 0, 0, 2 * QK_WIDTH_PER_DEVICE), (1, 1, lanes, QKV_WIDTH_PER_DEVICE), memory_config=l1
        )
        _deallocate(conv)
        q_heads = ttnn.reshape(q_slice, (lanes, QK_HEADS_PER_DEVICE, 1, HEAD_DIM))
        k_heads = ttnn.reshape(k_slice, (lanes, QK_HEADS_PER_DEVICE, 1, HEAD_DIM), pad_value=0.0)
        q = ttnn.repeat_interleave(q_heads, QK_REPEAT_FACTOR, dim=1, memory_config=l1)
        k = ttnn.repeat_interleave(k_heads, QK_REPEAT_FACTOR, dim=1, memory_config=l1)
        v = ttnn.reshape(v_slice, (lanes, VALUE_HEADS_PER_DEVICE, 1, HEAD_DIM))
        for tensor, reference in ((q, q_slice), (k, k_slice), (v, v_slice)):
            _retag_head_shard_after_reshape(tensor, reference=reference, shard_dim=1)
        _deallocate(q_slice, k_slice, v_slice, q_heads, k_heads)
        for name, tensor in (("query", q), ("key", k), ("value", v)):
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HEAD_SHARDED, shard_dim=1)
            _require_shape(tensor, (lanes, VALUE_HEADS_PER_DEVICE, 1, HEAD_DIM), label=f"GDN lanes recurrent {name}")

        b_fp32 = ttnn.typecast(b, ttnn.float32, memory_config=l1)
        beta_fp32 = ttnn.sigmoid(b_fp32, memory_config=l1)
        beta = ttnn.reshape(beta_fp32, (lanes, VALUE_HEADS_PER_DEVICE, 1, 1), memory_config=l1)
        _retag_head_shard_after_reshape(beta, reference=beta_fp32, shard_dim=1)
        beta_producers = (b, b_fp32, beta_fp32)

        a_fp32 = ttnn.typecast(a, ttnn.float32, memory_config=l1)
        _deallocate(a)
        softplus = softplus_gate(a_fp32, self.weights.dt_bias, memory_config=l1)
        _deallocate(a_fp32)
        log_decay_raw = ttnn.multiply(self.weights.neg_exp_A, softplus, memory_config=l1)
        _deallocate(softplus)
        log_decay = ttnn.reshape(log_decay_raw, (lanes, VALUE_HEADS_PER_DEVICE, 1, 1), memory_config=l1)
        _retag_head_shard_after_reshape(log_decay, reference=log_decay_raw, shard_dim=1)
        beta_producers = (*beta_producers, log_decay_raw)
        for name, tensor in (("beta", beta), ("log_decay", log_decay)):
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HEAD_SHARDED, shard_dim=1)
            _require_shape(tensor, (lanes, VALUE_HEADS_PER_DEVICE, 1, 1), label=f"GDN lanes recurrent {name}")
        if log_decay.dtype != ttnn.float32:
            raise RuntimeError(f"GDN lanes log decay must be FP32, got {log_decay.dtype}")
        return q, k, v, beta, log_decay, beta_producers

    def _recurrent_decode_lanes(self, q, k, v, beta, log_decay, beta_producers, state: Qwen38TTNNGDNState, lanes: int):
        """``_recurrent_decode`` with (lane, head) as the batch: the same FP32 step per (u, h) tile into
        ``state.recurrent[u]``; the matmuls keep the 1-row program config (one output tile per core)."""

        l1 = ttnn.L1_MEMORY_CONFIG
        q_normed = ttnn.rms_norm(q, epsilon=QK_L2_NORM_EPS / HEAD_DIM)
        q_unit = ttnn.multiply(q_normed, HEAD_DIM**-0.5, memory_config=l1)
        k_normed = ttnn.rms_norm(k, epsilon=QK_L2_NORM_EPS / HEAD_DIM)
        k_unit = ttnn.multiply(k_normed, HEAD_DIM**-0.5, memory_config=l1)
        _deallocate(q, k, q_normed, k_normed)
        q_fp32 = ttnn.typecast(q_unit, ttnn.float32, memory_config=l1)
        k_row = ttnn.typecast(k_unit, ttnn.float32, memory_config=l1)
        _deallocate(q_unit, k_unit)
        q_row = ttnn.multiply(q_fp32, HEAD_DIM**-0.5, memory_config=l1)
        _deallocate(q_fp32)

        decayed = ttnn.multiply(
            state.recurrent,
            log_decay,
            input_tensor_b_activations=[ttnn.UnaryOpType.EXP],
            memory_config=l1,
        )
        v_read = ttnn.matmul(
            k_row,
            decayed,
            memory_config=l1,
            program_config=self.recurrent_matmul_program_config,
            compute_kernel_config=self.recurrent_read_compute_config,
        )
        delta = ttnn.subtract(v, v_read, dtype=ttnn.float32, memory_config=l1)
        _deallocate(v, v_read)
        k_col = ttnn.transpose(k_row, 2, 3, memory_config=l1)
        _deallocate(k_row)
        outer = ttnn.matmul(
            k_col,
            delta,
            memory_config=l1,
            compute_kernel_config=self.recurrent_write_compute_config,
        )
        _deallocate(k_col, delta)
        update = ttnn.multiply(outer, beta, memory_config=l1)
        _deallocate(outer)
        new_recurrent = ttnn.add(decayed, update, output_tensor=state.recurrent)
        _deallocate(decayed, update, beta, log_decay, *beta_producers)
        _require_landed(new_recurrent, state.recurrent, label="GDN lanes recurrent update")
        _require_shape(
            state.recurrent, (lanes, VALUE_HEADS_PER_DEVICE, HEAD_DIM, HEAD_DIM), label="GDN lanes next recurrent state"
        )
        if state.recurrent.dtype != ttnn.float32:
            raise RuntimeError(f"GDN lanes next recurrent state must be FP32, got {state.recurrent.dtype}")

        output = ttnn.matmul(
            q_row,
            state.recurrent,
            memory_config=l1,
            program_config=self.recurrent_matmul_program_config,
            compute_kernel_config=self.recurrent_read_compute_config,
        )
        _deallocate(q_row)
        _require_shape(output, (lanes, VALUE_HEADS_PER_DEVICE, 1, HEAD_DIM), label="GDN lanes recurrent output")
        return output

    def _gate_and_project_lanes(self, recurrent_output, z, full_hidden, lanes: int):
        """``_gate_and_project`` on B rows: lane u's heads fold back to row u of the ``[1,1,B,1536]`` gate input."""

        l1 = ttnn.L1_MEMORY_CONFIG
        output_bf16 = (
            recurrent_output
            if recurrent_output.dtype == ttnn.bfloat16
            else ttnn.typecast(recurrent_output, ttnn.bfloat16, memory_config=l1)
        )
        if output_bf16 is not recurrent_output:
            _deallocate(recurrent_output)
        normalized_heads = ttnn.rms_norm(output_bf16, weight=self.weights.norm, epsilon=RMS_NORM_EPS, memory_config=l1)
        _deallocate(output_bf16)
        normalized = ttnn.reshape(normalized_heads, (1, 1, lanes, VALUE_WIDTH_PER_DEVICE))
        _retag_head_shard_after_reshape(normalized, reference=normalized_heads, shard_dim=3)
        _deallocate(normalized_heads)
        self.mesh_contract.validate_tensor(normalized, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        _require_shape(normalized, (1, 1, lanes, VALUE_WIDTH_PER_DEVICE), label="GDN lanes normalized output")

        z_fp32 = ttnn.typecast(z, ttnn.float32, memory_config=l1)
        _deallocate(z)
        sigmoid_fp32 = ttnn.sigmoid(z_fp32, memory_config=l1)
        _deallocate(z_fp32)
        sigmoid_bf16 = ttnn.typecast(sigmoid_fp32, ttnn.bfloat16, memory_config=l1)
        _deallocate(sigmoid_fp32)
        gated = ttnn.multiply(normalized, sigmoid_bf16, memory_config=self.out_proj_act_memory_config)
        _deallocate(normalized, sigmoid_bf16)
        self.mesh_contract.validate_tensor(gated, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        _require_shape(gated, (1, 1, lanes, VALUE_WIDTH_PER_DEVICE), label="GDN lanes sigmoid-gated output")
        return self._out_project_lanes(gated, full_hidden, lanes)

    def _out_project_lanes(self, gated, full_hidden, lanes: int):
        """``_out_project`` on B rows: the out linear on the gated ``[1,1,B,1536]`` rows and the line reduce-scatter
        into the ``[1,1,B,640]`` hidden-sharded rows; frees ``gated``, the partial and ``full_hidden``."""

        partial_ws = ttnn.linear(
            gated,
            self.weights.out,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.out_proj_program_config,
            compute_kernel_config=self.projection_compute_config,
        )
        _deallocate(gated)
        self.mesh_contract.mark_local_partial(
            partial_ws, replicated_reference=full_hidden, expected_shape=(1, 1, lanes, HIDDEN_SIZE)
        )
        output = ttnn.reduce_scatter(
            partial_ws,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.collective_topology,
        )
        _deallocate(partial_ws)
        self.mesh_contract.mark_collective_shard(
            output,
            replicated_reference=full_hidden,
            shard_dim=3,
            expected_local_shape=(1, 1, lanes, HIDDEN_SIZE_PER_DEVICE),
        )
        _deallocate(full_hidden)
        return output

    def forward_decode_lanes(self, hidden_lanes, state: Qwen38TTNNGDNState) -> Qwen38TTNNGDNResult:
        """Advance one token per lane (``hidden_lanes`` ``[1,1,B,640]``, row u = lane u) and mutate ``state`` in
        place; the ring phase advances once for every lane.  Returns the ``[1,1,B,640]`` hidden-sharded rows."""

        lanes = self._validate_lane_state(state)
        full_hidden = self._all_gather_hidden_lanes(hidden_lanes, lanes)
        window = state.conv_window()
        z, a, b = self._project_lanes(full_hidden, window[-1], lanes)
        state.advance_conv_window()
        conv = self._causal_conv_lanes(window, lanes)
        q, k, v, beta, log_decay, beta_producers = self._make_recurrent_inputs_lanes(conv, a, b, lanes)
        recurrent_output = self._recurrent_decode_lanes(q, k, v, beta, log_decay, beta_producers, state, lanes)
        output = self._gate_and_project_lanes(recurrent_output, z, full_hidden, lanes)
        return Qwen38TTNNGDNResult(output, state)

    def _forward_decode_lanes_fused(self, hidden_lanes, state: Qwen38TTNNGDNState) -> Qwen38TTNNGDNResult:
        """:meth:`forward_decode_lanes` with the fused ``gdn_step`` program on the B rows in place of the lane chain
        from the projection to the gated output (ttnn/fused/gdn_step: one core per (lane, value head) item, lane u's
        state slot updated in place); bound to ``forward_decode_lanes`` at construction when ``QWEN38_FUSED`` names
        gdn_step (the class body stays the fallback).  The ring order is the chain's: the window is read before the
        step, the program lands row u's q/k/v in row u of the window's last slot, and the phase advances once after.
        """

        lanes = self._validate_lane_state(state)
        full_hidden = self._all_gather_hidden_lanes(hidden_lanes, lanes)
        window = state.conv_window()
        projected = self._project_lanes_unsplit(full_hidden, lanes)
        gated = fused.gdn_step.gdn_step(self, projected, window, state)
        state.advance_conv_window()
        output = self._out_project_lanes(gated, full_hidden, lanes)
        return Qwen38TTNNGDNResult(output, state)

    # ------------------------------------------------------------------ rows path (MTP v2 verify)
    #
    # ``forward_rows`` runs R = rows consecutive positions in one pass: the same projection, FIR,
    # normalization, gate and output ops as ``forward_decode`` on CHUNK_SIZE-row tiles (the input
    # is zero padded to one tile, so every matmul keeps per_core_M = 1), and the recurrence through
    # ``ttnn.transformer.chunk_gated_delta_rule`` over exactly one full chunk with the rows past R
    # zeroed on device (k = v = 0, beta = 0, g = 0: an identity update).  It reads the committed
    # recurrent state and never writes it; ``commit_rows`` reruns the kernel over the committed
    # prefix (beta and g masked by the per-pass selectors) and lands the state in place, then moves
    # the FIR history forward by accepted + 1 rows with an exact 0/1 selection matmul.  Row-count and row
    # index never appear as per-pass host arguments: one program set per R, fixed buffers.

    def allocate_rows_constants(self, rows: int) -> Qwen38TTNNGDNRowsConstants:
        return Qwen38TTNNGDNRowsConstants.allocate(self.mesh_device, self.mesh_contract, rows=rows)

    def allocate_rows_state(
        self, constants: Qwen38TTNNGDNRowsConstants, *, history=None, body: Qwen38TTNNGDNRowsState | None = None
    ) -> Qwen38TTNNGDNRowsState:
        # gdn_qk_flat (prefill_glue, tolerance) is a slab form: the chunk rows states keep the head-major q/k.
        # gdn_prefill_rows (QWEN38_FUSED, bitwise) is the other slab form, and takes precedence over it: its
        # programs produce the chain's own normalized and scaled q/k, so the flat kernel form has nothing to do.
        fused_prefill = is_slab_rows(constants.rows) and self._fused_prefill_rows_buffers()
        flat_qk = is_slab_rows(constants.rows) and self.glue.enabled("gdn_qk_flat") and not fused_prefill
        rows_state = Qwen38TTNNGDNRowsState.allocate(
            self.mesh_device,
            self.mesh_contract,
            constants,
            layer_index=self.weights.layer_index,
            history=history,
            body=body,
            flat_qk=flat_qk,
            fused_prefill=fused_prefill,
        )
        # Whether this state runs the verify-rows wrap is decided here, before the warm pass: its buffers are the
        # admission, so the warm rounds, the capture and the commit that reads them cannot disagree.
        # The fold first (its prefix states are the admission); the wrap only where the fold did not attach.
        if fused.gdn_rows_scan.attach(self, rows_state) is None:
            fused.gdn_rows_wrap.attach(self, rows_state)
        return rows_state

    def _validate_rows_state(self, rows_state: Qwen38TTNNGDNRowsState) -> int:
        if rows_state.layer_index != self.weights.layer_index:
            raise ValueError(
                f"GDN layer {self.weights.layer_index} received rows state owned by layer {rows_state.layer_index}"
            )
        if rows_state.mesh_contract != self.mesh_contract:
            raise ValueError("GDN rows state belongs to a different physical mesh contract")
        rows_state.validate()
        return rows_state.constants.rows

    def sync_rows_history_from_state(self, state: Qwen38TTNNGDNState, rows_state: Qwen38TTNNGDNRowsState) -> None:
        """Eager mode switch (1-row -> rows): copy the ring's three history slots into ``history``."""

        self._validate_state(state)
        self._validate_rows_state(rows_state)
        window = state.conv_window()
        combined = ttnn.concat(list(window[:CONV_HISTORY_ROWS]), dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Zero rows 3..31 of the tile: ``ttnn.pad`` inside the input's own tile returns a view of
        # ``combined`` on this runtime, so only ``combined`` is deallocated.
        padded = ttnn.pad(
            combined,
            [(0, 0), (0, 0), (0, CHUNK_SIZE - CONV_HISTORY_ROWS), (0, 0)],
            0.0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _require_shape(padded, (1, 1, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE), label="GDN rows history tile")
        _copy_inplace(padded, rows_state.history, label="GDN rows history load")
        _deallocate(combined)

    def sync_state_from_rows_history(self, rows_state: Qwen38TTNNGDNRowsState, state: Qwen38TTNNGDNState) -> None:
        """Eager mode switch (rows -> 1-row): the ring's three history slots take ``history``; the phase is kept."""

        self._validate_state(state)
        self._validate_rows_state(rows_state)
        window = state.conv_window()
        for index in range(CONV_HISTORY_ROWS):
            row = ttnn.slice(
                rows_state.history,
                (0, 0, index, 0),
                (1, 1, index + 1, QKV_WIDTH_PER_DEVICE),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            _copy_inplace(row, window[index], label=f"GDN rows history store {index}")
            _deallocate(row)

    def _all_gather_rows(self, hidden_rows, rows: int):
        """The caller hands either the ``rows`` rows ``[1,1,rows,640]`` or a full 32-row tile ``[1,1,32,640]``
        whose rows past ``rows`` are already zero (a persistent input tile written in place): the second form
        skips the pad."""

        tile_rows = rows_tile_count(rows)
        if _shape(hidden_rows) != (1, 1, tile_rows, HIDDEN_SIZE_PER_DEVICE):
            _require_shape(hidden_rows, (1, 1, rows, HIDDEN_SIZE_PER_DEVICE), label="GDN rows input")
        self.mesh_contract.validate_tensor(hidden_rows, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        # Zero rows up to one tile on device: the projection of a zero row is exactly zero.  The padding
        # stays inside the input's own tile, so ``ttnn.pad`` returns a view of the input on this runtime
        # (measured on 4x p150: deallocating it freed the caller's rows); it is never deallocated here.
        if _shape(hidden_rows)[2] < tile_rows:
            padded = ttnn.pad(
                hidden_rows,
                [(0, 0), (0, 0), (0, CHUNK_SIZE - rows), (0, 0)],
                0.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            _retag_head_shard_after_reshape(padded, reference=hidden_rows, shard_dim=3)
        else:
            padded = hidden_rows
        _require_shape(padded, (1, 1, tile_rows, HIDDEN_SIZE_PER_DEVICE), label="GDN rows padded input")
        # One tile lands in the in-proj activation shard; the long chunk's four tiles are gathered
        # interleaved and moved into that shard one tile at a time by the projection.
        full_hidden = ttnn.all_gather(
            padded,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=self.in_proj_act_memory_config if tile_rows == CHUNK_SIZE else ttnn.DRAM_MEMORY_CONFIG,
        )
        self.mesh_contract.validate_tensor(full_hidden, placement=TensorPlacement.REPLICATED)
        _require_shape(full_hidden, (1, 1, tile_rows, HIDDEN_SIZE), label="GDN rows gathered hidden")
        return full_hidden

    def _project_rows_linear(self, full_hidden, rows_state: Qwen38TTNNGDNRowsState):
        """The rows projection itself, with the mesh tags and the shape the rows path checks.

        The DRAM-sharded matmul admits one row tile per call on the pinned runtime: at 32 rows the gathered
        shard is the call's input; at 128 rows each row tile is moved into the in-proj activation shard and
        the four projections are concatenated (the same program on the same rows, so row j is bitwise).

        Returns the ``[1, 1, T, 4160]`` projection; the caller owns it -- ``_project_rows`` lands the q|k|v columns
        and slices z / a / b out of it, the fused rows body hands the whole tile to ``gdn_pre_rows``.
        """

        tile_rows = rows_state.constants.tile_rows
        if tile_rows == CHUNK_SIZE:
            projected_ws = ttnn.linear(
                full_hidden,
                self.weights.qkvzab,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=self.in_proj_program_config,
                compute_kernel_config=self.projection_compute_config,
            )
            projected = ttnn.to_memory_config(projected_ws, ttnn.L1_MEMORY_CONFIG)
            _deallocate(projected_ws)
        elif is_slab_rows(tile_rows):
            projected = self._slab_projection(full_hidden, rows_state)
        else:
            projected_tiles = []
            for tile in dram_sharded_row_tiles(full_hidden, self.in_proj_act_memory_config):
                projected_ws = ttnn.linear(
                    tile,
                    self.weights.qkvzab,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                    program_config=self.in_proj_program_config,
                    compute_kernel_config=self.projection_compute_config,
                )
                projected_tiles.append(ttnn.to_memory_config(projected_ws, ttnn.DRAM_MEMORY_CONFIG))
                _deallocate(tile, projected_ws)
            projected = ttnn.concat(projected_tiles, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            _deallocate(*projected_tiles)
        _retag_head_shard_after_reshape(projected, reference=rows_state.qkv, shard_dim=3)
        self.mesh_contract.validate_tensor(projected, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        _require_shape(projected, (1, 1, tile_rows, PROJECTION_WIDTH_PER_DEVICE), label="GDN rows projection")
        return projected

    def _land_rows_qkv(self, projected, rows_state: Qwen38TTNNGDNRowsState) -> None:
        """The projection's q|k|v columns into the persistent ``qkv`` tile: the FIR window and the commit's history
        advance read it, so both rows bodies land it."""

        tile_rows = rows_state.constants.tile_rows
        landed = ttnn.slice(
            projected, (0, 0, 0, 0), (1, 1, tile_rows, QKV_WIDTH_PER_DEVICE), output_tensor=rows_state.qkv
        )
        _require_landed(landed, rows_state.qkv, label="GDN rows qkv slice")

    def _project_rows(self, full_hidden, rows_state: Qwen38TTNNGDNRowsState):
        """The chain's rows projection: the linear, the q|k|v landing into the persistent ``qkv``, and the z / a / b
        column slices the conv, the gates and the epilogue take."""

        tile_rows = rows_state.constants.tile_rows
        projected = self._project_rows_linear(full_hidden, rows_state)
        self._land_rows_qkv(projected, rows_state)
        columns = {
            "z": (QKV_WIDTH_PER_DEVICE, A_COLUMN),
            "a": (A_COLUMN, A_COLUMN + VALUE_HEADS_PER_DEVICE),
            "b": (B_COLUMN, B_COLUMN + VALUE_HEADS_PER_DEVICE),
        }
        pieces = {}
        for name, (start, end) in columns.items():
            piece = ttnn.slice(projected, (0, 0, 0, start), (1, 1, tile_rows, end), memory_config=ttnn.L1_MEMORY_CONFIG)
            self.mesh_contract.validate_tensor(piece, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
            _require_shape(piece, (1, 1, tile_rows, end - start), label=f"GDN rows projected {name}")
            pieces[name] = piece
        _deallocate(projected)
        return pieces["z"], pieces["a"], pieces["b"]

    def _slab_projection(self, full_hidden, rows_state: Qwen38TTNNGDNRowsState):
        """The slab's whole projection ``[1, 1, T, 4160]``: one 2D-multicast matmul over every row on an interleaved
        copy of the weight, or on the resident prefill copy and with the prefill dense policy's fidelity when its
        switches say so, and the head-shard retag of the result.

        ``_project_rows``' slab branch calls it and then slices; the fused prefill rows body (``fused/
        gdn_prefill_rows.py``) calls it and reads the q|k|v, z and a/b columns out of the tensor itself.  The retag
        and the two checks here are metadata and shape reads, so ``_project_rows`` repeating them costs nothing.
        """

        tile_rows = rows_state.constants.tile_rows
        projected = prefill_linear(
            full_hidden,
            self.weights.qkvzab,
            self._slab_program_config(tile_rows, HIDDEN_SIZE, PROJECTION_WIDTH_PER_DEVICE),
            compute_kernel_config=self.prefill_dense.compute_config(self.projection_compute_config),
            resident_weight=self.prefill_dense.resident("qkvzab"),
        )
        _retag_head_shard_after_reshape(projected, reference=rows_state.qkv, shard_dim=3)
        self.mesh_contract.validate_tensor(projected, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        _require_shape(projected, (1, 1, tile_rows, PROJECTION_WIDTH_PER_DEVICE), label="GDN rows projection")
        return projected

    def _conv_window_rows(self, rows_state: Qwen38TTNNGDNRowsState):
        """``[history tile | qkv tiles]``: whole tiles on dim 2 (no row padding, so a plain tile concat).

        Logical window row m (history rows 0..2, then the new rows) is buffer row ``_window_buffer_row(m)``;
        the FIR taps and the next history are read out of it with the constant 0/1 selection matmuls.
        """

        window = ttnn.concat([rows_state.history, rows_state.qkv], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _retag_head_shard_after_reshape(window, reference=rows_state.qkv, shard_dim=3)
        _require_shape(
            window, (1, 1, rows_state.constants.window_rows, QKV_WIDTH_PER_DEVICE), label="GDN rows conv window"
        )
        return window

    def _select_rows(self, select, window, *, memory_config, label: str, output_tensor=None):
        """``select @ window``: an exact 0/1 row selection (HiFi4, fp32 accumulation, one nonzero term per
        output element: x * 1.0 plus exact zeros is x, and the bf16 result is the selected bf16 row).  With
        ``output_tensor`` the matmul writes the persistent buffer itself."""

        selected = ttnn.matmul(
            select,
            window,
            memory_config=memory_config,
            compute_kernel_config=self.compute_config,
            optional_output_tensor=output_tensor,
        )
        if output_tensor is not None:
            _require_landed(selected, output_tensor, label=label)
            selected = output_tensor
        _retag_head_shard_after_reshape(selected, reference=window, shard_dim=3)
        _require_shape(selected, (1, 1, _shape(select)[2], QKV_WIDTH_PER_DEVICE), label=label)
        return selected

    def _slab_program_config(self, rows: int, k: int, n: int):
        """The slab's 2D-multicast matmul config for one linear, built once per (rows, k, n): today's config, or the
        prefill dense policy's wide grid (QWEN38_PREFILL_DENSE_GRID=wide)."""

        configs = self.__dict__.setdefault("_slab_program_configs", {})
        key = (rows, k, n)
        if key not in configs:
            if self.prefill_dense.policy.grid == "today":
                configs[key] = prefill_matmul_program_config(self.mesh_device, rows, k, n)
            else:
                configs[key] = self.prefill_dense.program_config(rows, k, n)
        return configs[key]

    def _shifted_rows_slab(self, rows_state: Qwen38TTNNGDNRowsState):
        """The slab's FIR taps 0..2 by row shifts: tap t is ``[history rows t..2 | new rows 0..T-4+t]``, built in
        ROW_MAJOR (a row slice at any offset) and tilized; the same rows the 0/1 selects pick at 32 / 128 rows."""

        dram = ttnn.DRAM_MEMORY_CONFIG
        tile_rows = rows_state.constants.tile_rows
        history_rm = ttnn.to_layout(rows_state.history, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
        qkv_rm = ttnn.to_layout(rows_state.qkv, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
        pieces = []
        for tap in range(CONV_HISTORY_ROWS):
            kept = ttnn.slice(
                history_rm, (0, 0, tap, 0), (1, 1, CONV_HISTORY_ROWS, QKV_WIDTH_PER_DEVICE), memory_config=dram
            )
            new = ttnn.slice(
                qkv_rm,
                (0, 0, 0, 0),
                (1, 1, tile_rows - CONV_HISTORY_ROWS + tap, QKV_WIDTH_PER_DEVICE),
                memory_config=dram,
            )
            shifted_rm = ttnn.concat([kept, new], dim=2, memory_config=dram)
            _deallocate(kept, new)
            shifted = ttnn.to_layout(shifted_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
            _deallocate(shifted_rm)
            _retag_head_shard_after_reshape(shifted, reference=rows_state.qkv, shard_dim=3)
            _require_shape(shifted, (1, 1, tile_rows, QKV_WIDTH_PER_DEVICE), label=f"GDN slab FIR tap {tap}")
            pieces.append(shifted)
        _deallocate(history_rm, qkv_rm)
        return pieces

    def _causal_conv_rows(self, rows_state: Qwen38TTNNGDNRowsState):
        # Tap t reads logical window rows t .. t + T - 1; tap 3 is the new rows themselves (the persistent qkv).
        if is_slab_rows(rows_state.constants.tile_rows):
            pieces = self._shifted_rows_slab(rows_state)
        else:
            window = self._conv_window_rows(rows_state)
            pieces = [
                self._select_rows(select, window, memory_config=ttnn.L1_MEMORY_CONFIG, label=f"GDN rows FIR tap {tap}")
                for tap, select in enumerate(rows_state.constants.conv_taps)
            ]
            _deallocate(window)
        pieces.append(rows_state.qkv)
        # Same tap arithmetic as _causal_conv_decode; the [1,1,1,2560] taps broadcast over the rows.
        conv = ttnn.multiply(pieces[0], self.weights.conv_taps[0], memory_config=ttnn.L1_MEMORY_CONFIG)
        for index in range(1, CONV_KERNEL_SIZE):
            previous = conv
            conv = ttnn.mac(pieces[index], self.weights.conv_taps[index], previous)
            if _tensor_key(conv) != _tensor_key(previous):
                _deallocate(previous)
        _deallocate(*pieces[:CONV_HISTORY_ROWS])
        conv = ttnn.silu(conv, memory_config=ttnn.L1_MEMORY_CONFIG)
        _retag_head_shard_after_reshape(conv, reference=rows_state.qkv, shard_dim=3)
        _require_shape(
            conv, (1, 1, rows_state.constants.tile_rows, QKV_WIDTH_PER_DEVICE), label="GDN rows causal convolution"
        )
        return conv

    def _make_chunk_inputs(self, conv, a, b, rows_state: Qwen38TTNNGDNRowsState) -> None:
        """Write the chunk kernel's inputs into the persistent rows buffers.

        q and k are expanded to the 12 value heads first (``qk_expand``: an exact 0/1 matmul, so the chunk
        kernel sees H = HV and skips its GQA repeat), then l2-normalized per row in BF16 exactly as
        ``_recurrent_decode`` does it (rms_norm with eps / head_dim, then head_dim ** -0.5): a copied head
        normalizes to the same bits, so the kernel reads what the 1-row path's repeat_interleave would
        have given it.  beta and the log decay use the 1-row arithmetic.  The row masks zero rows >= R
        (x * 1.0 and x * 0.0 are exact) and are the ops that land in the persistent buffers; the long chunk
        has no padding rows (every mask is all ones), so its producing ops land in the buffers themselves.
        """

        l1 = ttnn.L1_MEMORY_CONFIG
        constants = rows_state.constants
        tile_rows = constants.tile_rows
        full_rows = constants.rows == tile_rows and tile_rows != CHUNK_SIZE  # the long chunk and the slab
        q_slice = ttnn.slice(conv, (0, 0, 0, 0), (1, 1, tile_rows, QK_WIDTH_PER_DEVICE), memory_config=l1)
        k_slice = ttnn.slice(
            conv, (0, 0, 0, QK_WIDTH_PER_DEVICE), (1, 1, tile_rows, 2 * QK_WIDTH_PER_DEVICE), memory_config=l1
        )
        v_slice = ttnn.slice(
            conv, (0, 0, 0, 2 * QK_WIDTH_PER_DEVICE), (1, 1, tile_rows, QKV_WIDTH_PER_DEVICE), memory_config=l1
        )
        _deallocate(conv)
        # Heads on dim 2 ([B, T, H, K], the kernel's token-major form); the tile padding of the head
        # axis is zeroed so no padding value reaches the kernel's head split.  Under the slab's gdn_qk_flat form
        # (prefill_glue, tolerance) the raw conv q/k land token-major [1, 1, T, 512] instead: the kernel's flat
        # form maps value head hv to key head hv // 3 at read time and l2-normalizes q/k over K in fp32 with the
        # scale folded in, in place of the chain's 0/1 expand, BF16 rms_norm and BF16 scale below.
        for name, source, target in (("query", q_slice, rows_state.q), ("key", k_slice, rows_state.k)):
            if rows_state.flat_qk:
                landed = ttnn.multiply(source, constants.row_mask_bf16_col, output_tensor=target)
                _require_landed(landed, target, label=f"GDN rows flat {name}")
                _deallocate(source)
                continue
            expanded = ttnn.matmul(
                source, constants.qk_expand, memory_config=l1, compute_kernel_config=self.compute_config
            )
            _retag_head_shard_after_reshape(expanded, reference=source, shard_dim=3)
            _require_shape(expanded, (1, 1, tile_rows, VALUE_WIDTH_PER_DEVICE), label=f"GDN rows expanded {name}")
            _deallocate(source)
            heads_tensor = ttnn.reshape(expanded, (1, tile_rows, VALUE_HEADS_PER_DEVICE, HEAD_DIM), pad_value=0.0)
            _retag_head_shard_after_reshape(heads_tensor, reference=expanded, shard_dim=2)
            _deallocate(expanded)
            normed = ttnn.rms_norm(heads_tensor, epsilon=QK_L2_NORM_EPS / HEAD_DIM)
            _deallocate(heads_tensor)
            if full_rows:
                landed = ttnn.multiply(normed, HEAD_DIM**-0.5, output_tensor=target)
                _require_landed(landed, target, label=f"GDN rows {name}")
            else:
                unit = ttnn.multiply(normed, HEAD_DIM**-0.5, memory_config=l1)
                landed = ttnn.multiply(unit, constants.row_mask_bf16, output_tensor=target)
                _require_landed(landed, target, label=f"GDN rows {name}")
                _deallocate(unit)
            _deallocate(normed)
        # v stays token-major flat [1, 1, T, 1536]: the composite's flat-v form (no head split, no fill, no
        # transpose; the prep reader addresses head h's tiles at columns 128h.. of the same tile row).
        landed = ttnn.multiply(v_slice, constants.row_mask_bf16_col, output_tensor=rows_state.v)
        _require_landed(landed, rows_state.v, label="GDN rows value")
        _deallocate(v_slice)

        b_fp32 = ttnn.typecast(b, ttnn.float32, memory_config=l1)
        _deallocate(b)
        if full_rows:
            landed = ttnn.sigmoid(b_fp32, output_tensor=rows_state.beta)
        else:
            beta_fp32 = ttnn.sigmoid(b_fp32, memory_config=l1)
            landed = ttnn.multiply(beta_fp32, constants.row_mask_fp32, output_tensor=rows_state.beta)
            _deallocate(beta_fp32)
        _require_landed(landed, rows_state.beta, label="GDN rows beta")
        _deallocate(b_fp32)

        a_fp32 = ttnn.typecast(a, ttnn.float32, memory_config=l1)
        _deallocate(a)
        softplus = softplus_gate(a_fp32, self.weights.dt_bias, memory_config=l1)
        _deallocate(a_fp32)
        if full_rows:
            landed = ttnn.multiply(self.weights.neg_exp_A, softplus, output_tensor=rows_state.g)
        else:
            log_decay = ttnn.multiply(self.weights.neg_exp_A, softplus, memory_config=l1)
            landed = ttnn.multiply(log_decay, constants.row_mask_fp32, output_tensor=rows_state.g)
            _deallocate(log_decay)
        _require_landed(landed, rows_state.g, label="GDN rows log decay")
        _deallocate(softplus)
        rows_state.validate()

    def _chunk_rows(
        self, rows_state: Qwen38TTNNGDNRowsState, initial_state, committed_mask=None, committed_mask_c=None
    ):
        """One full-chunk run of the kernel from ``initial_state`` (read only).

        Returns the head-major output ``[VALUE_HEADS_PER_DEVICE, CHUNK_SIZE, HEAD_DIM]`` (TILE, the kernel's
        own layout and output dtype: FP32 on the pinned runtime, measured on 4x p150; ``output_head_major``
        skips the composite's untilize / row-major permute) and the FP32 final state
        ``[1, VALUE_HEADS_PER_DEVICE, HEAD_DIM, HEAD_DIM]`` in a new buffer.  With ``committed_mask`` beta
        and g of the rows past the committed prefix are zeroed first (the catch-up), which makes those
        rows an identity update.
        """

        constants = rows_state.constants
        tile_rows = constants.tile_rows
        wrap = fused.gdn_rows_wrap.buffers_of(rows_state)
        if wrap is not None:
            # The verify-rows wrap owns the prims' layouts: the recurrence reads its buffers, and a commit masks
            # ``beta_c`` / ``g_c`` with the same 0/1 mask in that layout (one multiply each, no broadcast).
            if committed_mask is not None and committed_mask_c is None:
                raise RuntimeError(
                    "the verify-rows wrap's commit needs the prim-layout committed mask, but this pass's selectors "
                    "carry none (the rows constants were built without arange_c)"
                )
            output, final_state = fused.gdn_rows_wrap.chunk(
                self, rows_state, wrap, initial_state, None if committed_mask is None else committed_mask_c
            )
            return self._validate_chunk_rows(output, final_state, rows_state, initial_state, tile_rows)
        if committed_mask is None:
            beta, g, masked = rows_state.beta, rows_state.g, ()
        else:
            beta = ttnn.multiply(rows_state.beta, committed_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            g = ttnn.multiply(rows_state.g, committed_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            masked = (beta, g)
        # [1, 1, T, HV] -> [1, T, HV] and [1, 1, T, HV * V] -> [1, T, HV * V]: the tile grid is unchanged, so
        # these are views of the buffers.  The rank-3 v is the composite's token-major flat form.  T = 128
        # runs the kernel's four 32-row chunks in one call (the state carried inside the scan in fp32).
        beta_rows = ttnn.reshape(beta, (1, tile_rows, VALUE_HEADS_PER_DEVICE))
        g_rows = ttnn.reshape(g, (1, tile_rows, VALUE_HEADS_PER_DEVICE))
        v_rows = ttnn.reshape(rows_state.v, (1, tile_rows, VALUE_WIDTH_PER_DEVICE))
        # gdn_qk_flat: the rank-3 views [1, T, 512] select the composite's flat q/k form (its own reader mapping
        # and in-kernel l2 norm with ``scale`` folded in); otherwise the head-major buffers as they are.
        q_rows = ttnn.reshape(rows_state.q, (1, tile_rows, QK_WIDTH_PER_DEVICE)) if rows_state.flat_qk else rows_state.q
        k_rows = ttnn.reshape(rows_state.k, (1, tile_rows, QK_WIDTH_PER_DEVICE)) if rows_state.flat_qk else rows_state.k
        # The recurrence itself: the composite, or (opt-in, admitted by shape) its two prims called directly; one
        # resolved callable serves the warm pass and the capture.
        kernel = self._chunk_rows_kernel()
        output, final_state = kernel(self, q_rows, k_rows, v_rows, g_rows, beta_rows, initial_state, constants)
        _deallocate(*masked)
        return self._validate_chunk_rows(output, final_state, rows_state, initial_state, tile_rows)

    def _validate_chunk_rows(self, output, final_state, rows_state, initial_state, tile_rows: int):
        """The head-shard tags, shapes and dtypes every rows recurrence returns, whichever form produced them."""

        if final_state is None:
            raise RuntimeError("the GDN rows recurrence returned no final state")
        _retag_head_shard_after_reshape(final_state, reference=initial_state, shard_dim=1)
        _require_shape(final_state, (1, VALUE_HEADS_PER_DEVICE, HEAD_DIM, HEAD_DIM), label="GDN rows final state")
        if final_state.dtype != ttnn.float32:
            raise RuntimeError(f"GDN rows final state must be FP32, got {final_state.dtype}")
        _retag_head_shard_after_reshape(output, reference=rows_state.v, shard_dim=0)
        _require_shape(output, (VALUE_HEADS_PER_DEVICE, tile_rows, HEAD_DIM), label="GDN rows recurrent output")
        if output.dtype not in (ttnn.bfloat16, ttnn.float32) or output.layout != ttnn.TILE_LAYOUT:
            raise RuntimeError(
                f"GDN rows recurrent output must be BF16 or FP32 TILE, got {output.dtype} {output.layout}"
            )
        return output, final_state

    def _chunk_rows_composite(self, q_rows, k_rows, v_rows, g_rows, beta_rows, initial_state, constants):
        """Today's recurrence call: the composite over the one chunk, its relayout and its two phase prims inside it
        (``output_head_major`` keeps the kernel's TILE layout: no untilize, no row-major permute).  The composed chain
        of ``gdn_rows_prims_direct``, whose fused form calls the same two prims from Python."""

        return ttnn.transformer.chunk_gated_delta_rule(
            q_rows,
            k_rows,
            v_rows,
            g_rows,
            beta_rows,
            scale=HEAD_DIM**-0.5,
            initial_state=initial_state,
            output_final_state=True,
            chunk_size=CHUNK_SIZE,
            output_head_major=True,
            eye=constants.eye,
            tril=constants.tril,
            ones=constants.ones,
            masks=constants.masks,
        )

    def _gate_and_project_rows(
        self, recurrent_output, z, full_hidden, rows_state: Qwen38TTNNGDNRowsState, *, full_tile: bool = False
    ):
        """``_gate_and_project`` on CHUNK_SIZE rows; the first ``rows`` rows of the reduce-scatter land in
        ``rows_state.output``, or with ``full_tile`` the whole 32-row reduce-scatter output is returned (a new buffer
        the caller owns; its rows past ``rows`` are exact zeros: the zero q rows give a zero recurrent output, whose
        norm, gate and projection stay zero)."""

        rows = rows_state.constants.rows
        tile_rows = rows_state.constants.tile_rows

        l1 = ttnn.L1_MEMORY_CONFIG
        # Head-major [HV, T, V] TILE -> [1, HV, T, V] is a view (last two dims unchanged): one (head, token)
        # row per tile row, so the per-head RMSNorm is the same [.., 128] row op as the 1-row path's, on the
        # same values.  The view shares the kernel output's buffer; only the original is deallocated.
        head_rows = ttnn.reshape(recurrent_output, (1, VALUE_HEADS_PER_DEVICE, tile_rows, HEAD_DIM))
        _retag_head_shard_after_reshape(head_rows, reference=z, shard_dim=1)
        # The 1-row path's boundary: the FP32 recurrent output becomes BF16 before the per-head RMSNorm.
        head_rows_bf16 = ttnn.typecast(head_rows, ttnn.bfloat16, memory_config=l1)
        _deallocate(recurrent_output)
        _retag_head_shard_after_reshape(head_rows_bf16, reference=z, shard_dim=1)
        normalized_heads = ttnn.rms_norm(
            head_rows_bf16, weight=self.weights.norm, epsilon=RMS_NORM_EPS, memory_config=l1
        )
        _deallocate(head_rows_bf16)
        _require_shape(
            normalized_heads, (1, VALUE_HEADS_PER_DEVICE, tile_rows, HEAD_DIM), label="GDN rows normalized heads"
        )
        z_fp32 = ttnn.typecast(z, ttnn.float32, memory_config=l1)
        _deallocate(z)
        sigmoid_fp32 = ttnn.sigmoid(z_fp32, memory_config=l1)
        _deallocate(z_fp32)
        sigmoid_bf16 = ttnn.typecast(sigmoid_fp32, ttnn.bfloat16, memory_config=l1)
        _deallocate(sigmoid_fp32)
        if tile_rows == CHUNK_SIZE:
            # Head-major [1, HV, 32, 128] and token-major [1, 1, 32, HV * 128] TILE tensors store their tiles in
            # the same order (tile (h, c) at 4h + c), so the fold to the gate's row form is a metadata view of
            # the same buffer: no permute, no relayout.  The view owns the buffer from here on.  The tile is
            # gated straight into the out-proj activation shard.
            normalized = ttnn.experimental.view(normalized_heads, (1, 1, CHUNK_SIZE, VALUE_WIDTH_PER_DEVICE))
            _retag_head_shard_after_reshape(normalized, reference=z, shard_dim=3)
            self.mesh_contract.validate_tensor(normalized, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
            _require_shape(normalized, (1, 1, tile_rows, VALUE_WIDTH_PER_DEVICE), label="GDN rows normalized output")
            gated = ttnn.multiply(normalized, sigmoid_bf16, memory_config=self.out_proj_act_memory_config)
            _deallocate(normalized, sigmoid_bf16)
            self.mesh_contract.validate_tensor(gated, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
            _require_shape(gated, (1, 1, tile_rows, VALUE_WIDTH_PER_DEVICE), label="GDN rows sigmoid-gated output")
            output = self._out_proj_tile(gated, full_hidden)
        elif is_slab_rows(tile_rows):
            # The slab: the head-major output folded to token-major rows as twelve whole-tile head slices
            # concatenated on the width (no padded token-major intermediate), gated interleaved, one 2D-multicast
            # out-proj over every row, one reduce-scatter, the rows copied into the persistent output.
            heads = [
                ttnn.slice(normalized_heads, (0, head, 0, 0), (1, head + 1, tile_rows, HEAD_DIM), memory_config=l1)
                for head in range(VALUE_HEADS_PER_DEVICE)
            ]
            _deallocate(normalized_heads)
            normalized = ttnn.concat(heads, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            _deallocate(*heads)
            _retag_head_shard_after_reshape(normalized, reference=z, shard_dim=3)
            _require_shape(normalized, (1, 1, tile_rows, VALUE_WIDTH_PER_DEVICE), label="GDN slab normalized output")
            gated = ttnn.multiply(normalized, sigmoid_bf16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            _deallocate(normalized, sigmoid_bf16)
            output = self._slab_out_projection(gated, full_hidden, rows_state)
        else:
            # The long chunk folds one 32-row tile at a time: the head-major rows of tile c are a whole-tile
            # slice whose fold is the same metadata view, gated by the gate's rows of that tile straight into
            # the out-proj activation shard; the out-proj and its reduce-scatter run per tile (the 32-row
            # programs on the same rows) and the tiles are copied into the persistent output (concat admits no
            # output tensor on this runtime).
            output_tiles = []
            for start in range(0, tile_rows, CHUNK_SIZE):
                heads_tile = ttnn.slice(
                    normalized_heads,
                    (0, 0, start, 0),
                    (1, VALUE_HEADS_PER_DEVICE, start + CHUNK_SIZE, HEAD_DIM),
                    memory_config=l1,
                )
                normalized = ttnn.experimental.view(heads_tile, (1, 1, CHUNK_SIZE, VALUE_WIDTH_PER_DEVICE))
                _retag_head_shard_after_reshape(normalized, reference=z, shard_dim=3)
                _require_shape(normalized, (1, 1, CHUNK_SIZE, VALUE_WIDTH_PER_DEVICE), label="GDN rows folded tile")
                gate_tile = ttnn.slice(
                    sigmoid_bf16, (0, 0, start, 0), (1, 1, start + CHUNK_SIZE, VALUE_WIDTH_PER_DEVICE), memory_config=l1
                )
                gated = ttnn.multiply(normalized, gate_tile, memory_config=self.out_proj_act_memory_config)
                _deallocate(normalized, gate_tile)
                self.mesh_contract.validate_tensor(gated, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
                _require_shape(gated, (1, 1, CHUNK_SIZE, VALUE_WIDTH_PER_DEVICE), label="GDN rows gated tile")
                output_tiles.append(self._out_proj_tile(gated, full_hidden))
            _deallocate(normalized_heads, sigmoid_bf16)
            output = ttnn.concat(output_tiles, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            _deallocate(*output_tiles)
            _copy_inplace(output, rows_state.output, label="GDN rows output")
            _deallocate(output)
            output = rows_state.output
        _deallocate(full_hidden)
        return self._rows_output_tile(output, rows_state, full_tile=full_tile)

    def _rows_output_tile(self, output, rows_state: Qwen38TTNNGDNRowsState, *, full_tile: bool):
        """The rows body's tail, shared by the chain and the fused wrap: the whole 32-row reduce-scatter tile
        (``full_tile``) or its ``rows``-row slice written into the persistent ``rows_state.output``."""

        rows = rows_state.constants.rows
        tile_rows = rows_state.constants.tile_rows
        if full_tile:
            if tile_rows != CHUNK_SIZE:
                raise ValueError(f"full_tile is the 32-row form's option, got tile rows {tile_rows}")
            _require_shape(output, (1, 1, CHUNK_SIZE, HIDDEN_SIZE_PER_DEVICE), label="GDN rows output tile")
            self.mesh_contract.validate_tensor(output, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
            return output
        if tile_rows == CHUNK_SIZE:
            # The row slice of the first tile may alias its input on this runtime; writing it into the persistent
            # output buffer (the 1-row path's slice form) keeps a fixed address and lets the 32-row tensor go.
            landed = ttnn.slice(
                output, (0, 0, 0, 0), (1, 1, rows, HIDDEN_SIZE_PER_DEVICE), output_tensor=rows_state.output
            )
            _require_landed(landed, rows_state.output, label="GDN rows output slice")
            _deallocate(output)
        self.mesh_contract.validate_tensor(rows_state.output, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        _require_shape(rows_state.output, (1, 1, rows, HIDDEN_SIZE_PER_DEVICE), label="GDN rows output")
        return rows_state.output

    def _slab_out_projection(self, gated, full_hidden, rows_state: Qwen38TTNNGDNRowsState):
        """The slab's tail: one 2D-multicast out-projection over every gated row, one reduce-scatter, the rows
        copied into the persistent output.  ``gated`` is consumed.

        ``_gate_and_project_rows``' slab branch builds ``gated`` with the chain's ops and calls this; the fused
        prefill rows body's ``gdn_post_rows`` writes the same rows into its persistent buffer and calls it too.
        """

        tile_rows = rows_state.constants.tile_rows
        partial = prefill_linear(
            gated,
            self.weights.out,
            self._slab_program_config(tile_rows, VALUE_WIDTH_PER_DEVICE, HIDDEN_SIZE),
            compute_kernel_config=self.prefill_dense.compute_config(self.projection_compute_config),
            resident_weight=self.prefill_dense.resident("out"),
        )
        if rows_state.fused_prefill:
            # the fused body hands over its persistent pass buffer, which outlives the call
            if _tensor_key(gated) != _tensor_key(rows_state.gated):
                raise RuntimeError("the fused slab body must hand its own gated buffer to the out-projection")
        else:
            _deallocate(gated)
        self.mesh_contract.mark_local_partial(
            partial, replicated_reference=full_hidden, expected_shape=(1, 1, tile_rows, HIDDEN_SIZE)
        )
        output = ttnn.reduce_scatter(
            partial,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.collective_topology,
        )
        _deallocate(partial)
        self.mesh_contract.mark_collective_shard(
            output,
            replicated_reference=full_hidden,
            shard_dim=3,
            expected_local_shape=(1, 1, tile_rows, HIDDEN_SIZE_PER_DEVICE),
        )
        _copy_inplace(output, rows_state.output, label="GDN slab output")
        _deallocate(output)
        return rows_state.output

    def _out_proj_tile(self, gated_tile, full_hidden):
        """The 1-row out-proj on one gated 32-row tile in the out-proj activation shard, reduce-scattered."""

        partial_ws = ttnn.linear(
            gated_tile,
            self.weights.out,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.out_proj_program_config,
            compute_kernel_config=self.projection_compute_config,
        )
        _deallocate(gated_tile)
        self.mesh_contract.mark_local_partial(
            partial_ws,
            replicated_reference=full_hidden,
            expected_shape=(1, 1, CHUNK_SIZE, HIDDEN_SIZE),
        )
        output = ttnn.reduce_scatter(
            partial_ws,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.collective_topology,
        )
        _deallocate(partial_ws)
        self.mesh_contract.mark_collective_shard(
            output,
            replicated_reference=full_hidden,
            shard_dim=3,
            expected_local_shape=(1, 1, CHUNK_SIZE, HIDDEN_SIZE_PER_DEVICE),
        )
        return output

    def forward_rows(
        self, hidden_rows, state: Qwen38TTNNGDNState, rows_state: Qwen38TTNNGDNRowsState, *, full_tile: bool = False
    ) -> Qwen38TTNNGDNRowsResult:
        """Run ``rows`` consecutive positions from the committed state without committing anything.

        ``hidden_rows`` has local shape ``[1,1,rows,640]`` (hidden sharded).  ``state.recurrent`` is read
        only; the ring and its phase are not touched (the rows path keeps its own ``history``).  The
        result's ``final_state`` is the state after all rows in a new buffer.  With ``full_tile`` the result's
        ``hidden_rows`` is the whole 32-row output tile in a new buffer the caller deallocates (rows past ``rows``
        exact zeros), not the persistent ``rows_state.output`` slice: the caller that pads the rows back to the
        tile saves the slice and the pad.
        """

        self._validate_state(state)
        rows = self._validate_rows_state(rows_state)
        full_hidden = self._all_gather_rows(hidden_rows, rows)
        if is_slab_rows(rows_state.constants.tile_rows):
            # The slab body only: the two rows programs around the chunk prims when the fused kernel serves this
            # call, today's chain (the four calls below, in the same order) otherwise.  The 32-row and 128-row
            # bodies never resolve it.
            body = self._gdn_prefill_rows()
            output, final_state = body(self, full_hidden, rows_state, state.recurrent, full_tile=full_tile)
            return Qwen38TTNNGDNRowsResult(output, final_state, state, rows_state)
        # The body between the gather and the result: today's chain (``rows_body_composed``: the same calls in the
        # same order) or, on a rows state the verify-rows wrap owns, its two programs around the two chunk prims.
        body = self._rows_body()
        output, final_state = body(self, full_hidden, rows_state, state, full_tile=full_tile)
        return Qwen38TTNNGDNRowsResult(output, final_state, state, rows_state)

    def _advance_history_rows(self, rows_state: Qwen38TTNNGDNRowsState, selectors: Qwen38TTNNRowsSelectors) -> None:
        """``history <- window[c : c + 3]`` with ``c = accepted + 1``: one exact 0/1 selection matmul against
        the per-pass ``history_select`` (rows 0..2 of the result are the three rows, rows 3..31 exact zeros),
        written by the matmul into the persistent history tile (the window is a separate buffer)."""

        window = self._conv_window_rows(rows_state)
        self._select_rows(
            selectors.history_select,
            window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            label="GDN rows history select",
            output_tensor=rows_state.history,
        )
        _deallocate(window)

    def _step_row_state(self, rows_state: Qwen38TTNNGDNRowsState, initial_state, row: int):
        """Row ``row`` of the pass through the 1-row FP32 step arithmetic of ``_recurrent_decode``, into a new buffer.

        The rows buffers already hold the l2-normalized, GQA-expanded BF16 q/k (the 12 head rows the 1-row
        path's repeat_interleave produces), so this starts at the step's FP32 promotion: the same ops,
        program and compute configs from there on (bitwise the 1-row path's state update).  A flat q/k rows
        state (the slab's gdn_qk_flat form) holds raw conv rows and has no 1-row step.
        """

        if rows_state.flat_qk:
            raise ValueError("the 1-row step reads head-major normalized q/k: not a flat q/k (gdn_qk_flat) rows state")
        l1 = ttnn.L1_MEMORY_CONFIG
        # Only the state update is needed (no q read): k, v, beta and the log decay of the row.
        k_row_tile = ttnn.slice(
            rows_state.k, (0, row, 0, 0), (1, row + 1, VALUE_HEADS_PER_DEVICE, HEAD_DIM), memory_config=l1
        )
        k_unit = ttnn.reshape(k_row_tile, (1, VALUE_HEADS_PER_DEVICE, 1, HEAD_DIM))
        _deallocate(k_row_tile)
        v_row = ttnn.slice(rows_state.v, (0, 0, row, 0), (1, 1, row + 1, VALUE_WIDTH_PER_DEVICE), memory_config=l1)
        v = ttnn.reshape(v_row, (1, VALUE_HEADS_PER_DEVICE, 1, HEAD_DIM))
        _deallocate(v_row)
        scalars = []
        for source in (rows_state.beta, rows_state.g):
            scalar_row = ttnn.slice(source, (0, 0, row, 0), (1, 1, row + 1, VALUE_HEADS_PER_DEVICE), memory_config=l1)
            scalars.append(ttnn.reshape(scalar_row, (1, VALUE_HEADS_PER_DEVICE, 1, 1), memory_config=l1))
            _deallocate(scalar_row)
        beta, log_decay = scalars
        for name, tensor, expected in (
            ("key", k_unit, (1, VALUE_HEADS_PER_DEVICE, 1, HEAD_DIM)),
            ("value", v, (1, VALUE_HEADS_PER_DEVICE, 1, HEAD_DIM)),
            ("beta", beta, (1, VALUE_HEADS_PER_DEVICE, 1, 1)),
            ("log_decay", log_decay, (1, VALUE_HEADS_PER_DEVICE, 1, 1)),
        ):
            _require_shape(tensor, expected, label=f"GDN rows row-{row} step {name}")

        k_row = ttnn.typecast(k_unit, ttnn.float32, memory_config=l1)
        _deallocate(k_unit)
        decayed = ttnn.multiply(
            initial_state,
            log_decay,
            input_tensor_b_activations=[ttnn.UnaryOpType.EXP],
            memory_config=l1,
        )
        v_read = ttnn.matmul(
            k_row,
            decayed,
            memory_config=l1,
            program_config=self.recurrent_matmul_program_config,
            compute_kernel_config=self.recurrent_read_compute_config,
        )
        delta = ttnn.subtract(v, v_read, dtype=ttnn.float32, memory_config=l1)
        _deallocate(v, v_read)
        k_col = ttnn.transpose(k_row, 2, 3, memory_config=l1)
        _deallocate(k_row)
        outer = ttnn.matmul(
            k_col,
            delta,
            memory_config=l1,
            compute_kernel_config=self.recurrent_write_compute_config,
        )
        _deallocate(k_col, delta)
        update = ttnn.multiply(outer, beta, memory_config=l1)
        _deallocate(outer)
        stepped = ttnn.add(decayed, update, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(decayed, update, beta, log_decay)
        _retag_head_shard_after_reshape(stepped, reference=initial_state, shard_dim=1)
        _require_shape(stepped, (1, VALUE_HEADS_PER_DEVICE, HEAD_DIM, HEAD_DIM), label=f"GDN rows row-{row} step state")
        return stepped

    def _step_committed_rows_state(
        self, rows_state: Qwen38TTNNGDNRowsState, initial_state, selectors: Qwen38TTNNRowsSelectors
    ):
        """The state after the committed rows through the 1-row FP32 step arithmetic, into a new buffer.

        All ``rows`` rows are stepped in sequence from ``initial_state`` (a trace cannot bind the accept count)
        and the state after row ``accepted`` is selected with the exact one-hot multiply/add (``x * 1.0``,
        ``x * 0.0`` and ``+ 0.0`` are exact in FP32).  ``rows`` x (the step's ops) + 3 ``rows`` select ops.
        """

        dram = ttnn.DRAM_MEMORY_CONFIG
        state = initial_state
        selected = None
        for row in range(rows_state.constants.rows):
            stepped = self._step_row_state(rows_state, state, row)
            if state is not initial_state:
                _deallocate(state)
            state = stepped
            weight = ttnn.typecast(selectors.onehot_bf16[row], ttnn.float32, memory_config=dram)
            weighted = ttnn.multiply(stepped, weight, memory_config=dram)
            _deallocate(weight)
            if selected is None:
                selected = weighted
            else:
                summed = ttnn.add(selected, weighted, memory_config=dram)
                _deallocate(selected, weighted)
                selected = summed
        _deallocate(state)
        _retag_head_shard_after_reshape(selected, reference=initial_state, shard_dim=1)
        _require_shape(
            selected, (1, VALUE_HEADS_PER_DEVICE, HEAD_DIM, HEAD_DIM), label="GDN rows committed rows step state"
        )
        if selected.dtype != ttnn.float32:
            raise RuntimeError(f"GDN rows committed rows step state must be FP32, got {selected.dtype}")
        return selected

    def commit_rows(
        self,
        state: Qwen38TTNNGDNState,
        rows_state: Qwen38TTNNGDNRowsState,
        selectors: Qwen38TTNNRowsSelectors,
        *,
        step_on_full_rejection: bool = False,
        step_committed_rows: bool = False,
    ) -> None:
        """Commit ``accepted + 1`` rows of the last ``forward_rows`` pass into ``state`` in place.

        The catch-up reruns the chunk kernel from the committed state with beta and g of the rows past the
        committed prefix zeroed (``selectors.committed_mask``); the resulting state lands in
        ``state.recurrent`` and the FIR history moves forward by ``accepted + 1`` rows.

        Row 0 (the base token) is always committed, so there is no identity pass: the design's earlier
        "restore the old state when nothing was accepted" switch (written for the ``arange < a`` mask) has no
        case left.  The default therefore accepts the kernel's TF32-class state rounding on every commit (the
        design's decision: 4.9e-4 on a passthrough, below the kernel's own 2.1e-3 state error).  The switch
        kept in its place, ``step_on_full_rejection``, commits a full rejection (one row) through the 1-row
        FP32 step arithmetic instead, selected exactly against the kernel's result
        (``step * [a == 0] + chunk * [a != 0]``); it is a diagnostic for a state divergence that tracks full
        rejections and costs about 20 more ops per layer.  ``step_committed_rows`` (the state re-anchor)
        replaces the kernel's committed state altogether: every committed row goes through the 1-row FP32 step
        arithmetic (``_step_committed_rows_state``), so the committed state is the 1-row path's; the chunk
        kernel is not rerun.  About ``rows`` x 16 + 3 ``rows`` ops per layer.
        """

        self._validate_state(state)
        rows = self._validate_rows_state(rows_state)
        selectors.validate(rows)
        if (step_committed_rows or step_on_full_rejection) and fused.gdn_rows_wrap.buffers_of(rows_state) is not None:
            # Both anchor forms read the chain's q / k / beta / g layouts, which the wrap does not write.
            raise ValueError(
                "the 1-row step anchors of commit_rows need the chain's rows buffers; this layer runs the "
                "verify-rows wrap (a default), which keeps the chunk prims' layouts instead: set "
                "QWEN38_FUSED_OFF=gdn_rows_wrap to use the anchors"
            )
        scan = fused.gdn_rows_scan.buffers_of(rows_state)
        if scan is not None:
            if step_committed_rows or step_on_full_rejection:
                raise ValueError(
                    "the 1-row step anchors of commit_rows need the chain's rows buffers; this layer runs the "
                    "verify-rows fold (QWEN38_FUSED=gdn_rows_scan), whose committed state is the forward pass's own "
                    "prefix state"
                )
            # The state after accepted + 1 rows is the forward pass's prefix slot `accepted`: one pick, no re-run.
            fused.gdn_rows_scan.commit(self, rows_state, scan, state, selectors)
            self._advance_history_rows(rows_state, selectors)
            state.validate()
            return
        if step_committed_rows:
            stepped = self._step_committed_rows_state(rows_state, state.recurrent, selectors)
            _copy_inplace(stepped, state.recurrent, label="GDN rows committed rows step state")
            _deallocate(stepped)
            self._advance_history_rows(rows_state, selectors)
            state.validate()
            return
        output, final_state = self._chunk_rows(
            rows_state,
            initial_state=state.recurrent,
            committed_mask=selectors.committed_mask,
            committed_mask_c=selectors.committed_mask_c,
        )
        _deallocate(output)
        if step_on_full_rejection:
            dram = ttnn.DRAM_MEMORY_CONFIG
            stepped = self._step_row_state(rows_state, state.recurrent, 0)
            # ``accepted == 0`` and its complement as fp32 scalars (exact 0/1 selects of the state source).
            full_rejection = ttnn.typecast(selectors.onehot_bf16[0], ttnn.float32, memory_config=dram)
            partial_acceptance = ttnn.subtract(rows_state.constants.one, full_rejection, memory_config=dram)
            from_step = ttnn.multiply(stepped, full_rejection, memory_config=dram)
            from_chunk = ttnn.multiply(final_state, partial_acceptance, memory_config=dram)
            landed = ttnn.add(from_step, from_chunk, output_tensor=state.recurrent)
            _require_landed(landed, state.recurrent, label="GDN rows committed state select")
            _deallocate(stepped, from_step, from_chunk, final_state, full_rejection, partial_acceptance)
        else:
            _copy_inplace(final_state, state.recurrent, label="GDN rows committed state")
            _deallocate(final_state)
        self._advance_history_rows(rows_state, selectors)
        state.validate()

    def commit_rows_full(self, state: Qwen38TTNNGDNState, rows_state: Qwen38TTNNGDNRowsState, final_state) -> None:
        """Commit every row of the last ``forward_rows`` pass (the long chunk: no accept scalar, no re-run).

        The forward pass's own ``final_state`` (consumed here) lands in ``state.recurrent`` and the FIR history
        becomes the last three new rows through the constant ``history_select_full`` (three ops per layer).
        """

        self._validate_state(state)
        self._validate_rows_state(rows_state)
        scan = fused.gdn_rows_scan.buffers_of(rows_state)
        if final_state is None and scan is not None:
            # the fold keeps the state after all rows in its last prefix slot
            fused.gdn_rows_scan.commit_all_rows(scan, state)
        else:
            _require_shape(final_state, (1, VALUE_HEADS_PER_DEVICE, HEAD_DIM, HEAD_DIM), label="GDN rows final state")
            _copy_inplace(final_state, state.recurrent, label="GDN rows committed state")
            _deallocate(final_state)
        if rows_state.fused_prefill:
            # The fused slab body's gated epilogue already built the next pass's history tile from the projection's
            # last three rows (gdn_post_rows), in place of the seven programs of the branch below; the commit is the
            # one copy, which is also what keeps ``forward_rows`` side-effect free on the shared history buffer.
            _copy_inplace(rows_state.history_next, rows_state.history, label="GDN slab history")
        elif is_slab_rows(rows_state.constants.tile_rows):
            # The last three new rows into history rows 0..2 (rows 3..31 zero) by row shifts: the last row tile
            # untilized, its rows 29..31 sliced, padded to a tile, tilized, copied into the persistent history.
            dram = ttnn.DRAM_MEMORY_CONFIG
            tile_rows = rows_state.constants.tile_rows
            last_tile = ttnn.slice(
                rows_state.qkv,
                (0, 0, tile_rows - CHUNK_SIZE, 0),
                (1, 1, tile_rows, QKV_WIDTH_PER_DEVICE),
                memory_config=dram,
            )
            last_rm = ttnn.to_layout(last_tile, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
            _deallocate(last_tile)
            tail = ttnn.slice(
                last_rm,
                (0, 0, CHUNK_SIZE - CONV_HISTORY_ROWS, 0),
                (1, 1, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE),
                memory_config=dram,
            )
            _deallocate(last_rm)
            padded = ttnn.pad(
                tail, [(0, 0), (0, 0), (0, CHUNK_SIZE - CONV_HISTORY_ROWS), (0, 0)], 0.0, memory_config=dram
            )
            history = ttnn.to_layout(padded, ttnn.TILE_LAYOUT, memory_config=dram)
            _require_shape(history, (1, 1, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE), label="GDN slab history tile")
            _copy_inplace(history, rows_state.history, label="GDN slab history")
            _deallocate(history, padded, tail)
        else:
            window = self._conv_window_rows(rows_state)
            self._select_rows(
                rows_state.constants.history_select_full,
                window,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                label="GDN rows history select",
                output_tensor=rows_state.history,
            )
            _deallocate(window)
        state.validate()

    # ------------------------------------------------------------------ lane rows path (MTP lanes verify)

    def allocate_lane_rows_constants(self, lanes: int, rows: int) -> Qwen38TTNNGDNLaneRowsConstants:
        return Qwen38TTNNGDNLaneRowsConstants.allocate(self.mesh_device, self.mesh_contract, lanes=lanes, rows=rows)

    def allocate_lane_rows_state(
        self, constants: Qwen38TTNNGDNRowsConstants, lane_constants: Qwen38TTNNGDNLaneRowsConstants
    ) -> Qwen38TTNNGDNLaneRowsState:
        rows_state = Qwen38TTNNGDNLaneRowsState.allocate(
            self.mesh_device, self.mesh_contract, constants, lane_constants, layer_index=self.weights.layer_index
        )
        # Whether this state runs the verify-rows fold's lanes form is decided here, before the warm pass: its prefix
        # states are the admission, so the warm rounds, the capture and the commit that picks them cannot disagree.
        fused.gdn_rows_scan.attach_lanes(self, rows_state)
        return rows_state

    def _validate_lane_rows_state(self, rows_state: Qwen38TTNNGDNLaneRowsState, state: Qwen38TTNNGDNState) -> int:
        if rows_state.layer_index != self.weights.layer_index:
            raise ValueError(
                f"GDN layer {self.weights.layer_index} received lane rows state owned by layer {rows_state.layer_index}"
            )
        if rows_state.mesh_contract != self.mesh_contract:
            raise ValueError("GDN lane rows state belongs to a different physical mesh contract")
        rows_state.validate()
        if self._validate_lane_state(state) != rows_state.lanes:
            raise ValueError(f"GDN lane state has {state.batch_size} lanes, the lane rows state {rows_state.lanes}")
        return rows_state.constants.rows

    def _qkv_lanes_view(self, rows_state: Qwen38TTNNGDNLaneRowsState):
        """``qkv_lanes`` ``[1,1,B*32,2560]`` as the batched ops read it, ``[1,B,32,2560]``: whole tiles, the same
        pages (never released on its own: the state owns the buffer)."""

        view = ttnn.experimental.view(rows_state.qkv_lanes, (1, rows_state.lanes, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE))
        _retag_head_shard_after_reshape(view, reference=rows_state.qkv, shard_dim=3)
        return view

    def _expand_lane_rows(self, piece, rows_state: Qwen38TTNNGDNLaneRowsState):
        """``expand_select @ piece``: the lane-major ``[1,1,32,W]`` rows as ``[1,B,32,W]`` per-lane tiles (rows
        ``R .. 31`` of every lane exact zeros); ``piece`` is consumed."""

        width = _shape(piece)[3]
        expanded = ttnn.matmul(
            rows_state.lane_constants.expand_select,
            piece,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        _deallocate(piece)
        _require_shape(expanded, (1, 1, rows_state.lanes * CHUNK_SIZE, width), label="GDN lane rows expand")
        lanes = ttnn.experimental.view(expanded, (1, rows_state.lanes, CHUNK_SIZE, width))
        _retag_head_shard_after_reshape(lanes, reference=rows_state.qkv, shard_dim=3)
        return lanes

    def _lane_conv_window(self, rows_state: Qwen38TTNNGDNLaneRowsState):
        """``[history | qkv_lanes]`` per lane: ``[1,B,64,2560]``, whole tiles on dim 2."""

        window = ttnn.concat(
            [rows_state.history, self._qkv_lanes_view(rows_state)], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        _retag_head_shard_after_reshape(window, reference=rows_state.qkv, shard_dim=3)
        _require_shape(
            window,
            (1, rows_state.lanes, CONV_WINDOW_TILE_ROWS, QKV_WIDTH_PER_DEVICE),
            label="GDN lane rows conv window",
        )
        return window

    def _causal_conv_rows_lanes(self, rows_state: Qwen38TTNNGDNLaneRowsState):
        """The rows path's FIR on every lane's own tile: the three tap selects as equal-batch matmuls over the
        per-lane windows, tap 3 the lane's new rows; the same tap arithmetic as ``_causal_conv_rows``."""

        l1 = ttnn.L1_MEMORY_CONFIG
        window = self._lane_conv_window(rows_state)
        pieces = [
            ttnn.matmul(select, window, memory_config=l1, compute_kernel_config=self.compute_config)
            for select in rows_state.lane_constants.conv_taps
        ]
        _deallocate(window)
        pieces.append(self._qkv_lanes_view(rows_state))
        conv = ttnn.multiply(pieces[0], self.weights.conv_taps[0], memory_config=l1)
        for index in range(1, CONV_KERNEL_SIZE):
            previous = conv
            conv = ttnn.mac(pieces[index], self.weights.conv_taps[index], previous)
            if _tensor_key(conv) != _tensor_key(previous):
                _deallocate(previous)
        _deallocate(*pieces[:CONV_HISTORY_ROWS])
        conv = ttnn.silu(conv, memory_config=l1)
        _retag_head_shard_after_reshape(conv, reference=rows_state.qkv, shard_dim=3)
        _require_shape(
            conv, (1, rows_state.lanes, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE), label="GDN lane rows causal convolution"
        )
        return conv

    def _make_chunk_inputs_lanes(self, conv, a_lanes, b_lanes, rows_state: Qwen38TTNNGDNLaneRowsState) -> None:
        """``_make_chunk_inputs`` on the per-lane tiles: the kernel's q / k ``[B,32,12,128]``, flat v, beta and log
        decay land in the persistent lane buffers with every lane's rows past ``R`` zeroed by the lane row masks."""

        l1 = ttnn.L1_MEMORY_CONFIG
        constants, lane_constants, lanes = rows_state.constants, rows_state.lane_constants, rows_state.lanes
        q_slice = ttnn.slice(conv, (0, 0, 0, 0), (1, lanes, CHUNK_SIZE, QK_WIDTH_PER_DEVICE), memory_config=l1)
        k_slice = ttnn.slice(
            conv, (0, 0, 0, QK_WIDTH_PER_DEVICE), (1, lanes, CHUNK_SIZE, 2 * QK_WIDTH_PER_DEVICE), memory_config=l1
        )
        v_slice = ttnn.slice(
            conv, (0, 0, 0, 2 * QK_WIDTH_PER_DEVICE), (1, lanes, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE), memory_config=l1
        )
        _deallocate(conv)
        for name, source, target in (("query", q_slice, rows_state.q), ("key", k_slice, rows_state.k)):
            expanded = ttnn.matmul(
                source, constants.qk_expand, memory_config=l1, compute_kernel_config=self.compute_config
            )
            _retag_head_shard_after_reshape(expanded, reference=source, shard_dim=3)
            _require_shape(expanded, (1, lanes, CHUNK_SIZE, VALUE_WIDTH_PER_DEVICE), label=f"GDN lane rows {name}")
            _deallocate(source)
            heads_tensor = ttnn.reshape(expanded, (lanes, CHUNK_SIZE, VALUE_HEADS_PER_DEVICE, HEAD_DIM), pad_value=0.0)
            _retag_head_shard_after_reshape(heads_tensor, reference=expanded, shard_dim=2)
            _deallocate(expanded)
            normed = ttnn.rms_norm(heads_tensor, epsilon=QK_L2_NORM_EPS / HEAD_DIM)
            _deallocate(heads_tensor)
            unit = ttnn.multiply(normed, HEAD_DIM**-0.5, memory_config=l1)
            landed = ttnn.multiply(unit, lane_constants.row_mask_bf16, output_tensor=target)
            _require_landed(landed, target, label=f"GDN lane rows {name}")
            _deallocate(unit, normed)
        landed = ttnn.multiply(v_slice, lane_constants.row_mask_bf16_col, output_tensor=rows_state.v)
        _require_landed(landed, rows_state.v, label="GDN lane rows value")
        _deallocate(v_slice)

        b_fp32 = ttnn.typecast(b_lanes, ttnn.float32, memory_config=l1)
        _deallocate(b_lanes)
        beta_fp32 = ttnn.sigmoid(b_fp32, memory_config=l1)
        landed = ttnn.multiply(beta_fp32, lane_constants.row_mask_fp32, output_tensor=rows_state.beta)
        _require_landed(landed, rows_state.beta, label="GDN lane rows beta")
        _deallocate(b_fp32, beta_fp32)

        a_fp32 = ttnn.typecast(a_lanes, ttnn.float32, memory_config=l1)
        _deallocate(a_lanes)
        softplus = softplus_gate(a_fp32, self.weights.dt_bias, memory_config=l1)
        _deallocate(a_fp32)
        log_decay = ttnn.multiply(self.weights.neg_exp_A, softplus, memory_config=l1)
        landed = ttnn.multiply(log_decay, lane_constants.row_mask_fp32, output_tensor=rows_state.g)
        _require_landed(landed, rows_state.g, label="GDN lane rows log decay")
        _deallocate(softplus, log_decay)
        rows_state.validate()

    def _chunk_rows_lanes(self, rows_state: Qwen38TTNNGDNLaneRowsState, initial_state, committed_mask=None):
        """ONE ``chunk_gated_delta_rule`` call over the B lanes from ``initial_state`` ``[B,12,128,128]`` (read only):
        head-major output ``[B*12,32,128]`` and the FP32 final states ``[B,12,128,128]`` in a new buffer; with
        ``committed_mask`` ``[1,B,32,1]`` beta and g of the rows past each lane's committed prefix are zeroed first."""

        lanes, constants = rows_state.lanes, rows_state.constants
        if committed_mask is None:
            beta, g, masked = rows_state.beta, rows_state.g, ()
        else:
            beta = ttnn.multiply(rows_state.beta, committed_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            g = ttnn.multiply(rows_state.g, committed_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            masked = (beta, g)
        # [1, B, T, HV] -> [B, T, HV] and [1, B, T, HV * V] -> [B, T, HV * V]: the tile grid is unchanged, views.
        beta_rows = ttnn.reshape(beta, (lanes, CHUNK_SIZE, VALUE_HEADS_PER_DEVICE))
        g_rows = ttnn.reshape(g, (lanes, CHUNK_SIZE, VALUE_HEADS_PER_DEVICE))
        v_rows = ttnn.reshape(rows_state.v, (lanes, CHUNK_SIZE, VALUE_WIDTH_PER_DEVICE))
        output, final_state = ttnn.transformer.chunk_gated_delta_rule(
            rows_state.q,
            rows_state.k,
            v_rows,
            g_rows,
            beta_rows,
            scale=HEAD_DIM**-0.5,
            initial_state=initial_state,
            output_final_state=True,
            chunk_size=CHUNK_SIZE,
            output_head_major=True,
            eye=constants.eye,
            tril=constants.tril,
            ones=constants.ones,
            masks=constants.masks,
        )
        _deallocate(*masked)
        if final_state is None:
            raise RuntimeError("chunk_gated_delta_rule returned no final state")
        _retag_head_shard_after_reshape(final_state, reference=initial_state, shard_dim=1)
        _require_shape(
            final_state, (lanes, VALUE_HEADS_PER_DEVICE, HEAD_DIM, HEAD_DIM), label="GDN lane rows final state"
        )
        if final_state.dtype != ttnn.float32:
            raise RuntimeError(f"GDN lane rows final state must be FP32, got {final_state.dtype}")
        _retag_head_shard_after_reshape(output, reference=rows_state.v, shard_dim=0)
        _require_shape(
            output, (lanes * VALUE_HEADS_PER_DEVICE, CHUNK_SIZE, HEAD_DIM), label="GDN lane rows recurrent output"
        )
        if output.dtype not in (ttnn.bfloat16, ttnn.float32) or output.layout != ttnn.TILE_LAYOUT:
            raise RuntimeError(
                f"GDN lane rows recurrent output must be BF16 or FP32 TILE, got {output.dtype} {output.layout}"
            )
        return output, final_state

    def _gate_and_project_rows_lanes(self, recurrent_output, z, full_hidden, rows_state: Qwen38TTNNGDNLaneRowsState):
        """``_gate_and_project_rows`` on the per-lane head-major output: the per-head RMS norm on ``[B,12,32,128]``,
        the fold view ``[1,1,B*32,1536]`` (tile ``(b, h, c)`` sits at ``b*48 + 4h + c`` in both layouts), the
        ``fold_select`` back to the lane-major tile, the gate and the one-tile output projection.  Returns the
        ``[1,1,32,640]`` reduce-scatter output (rows past ``B*R`` exact zeros) in a new buffer."""

        lanes = rows_state.lanes
        l1 = ttnn.L1_MEMORY_CONFIG
        head_rows = ttnn.reshape(recurrent_output, (lanes, VALUE_HEADS_PER_DEVICE, CHUNK_SIZE, HEAD_DIM))
        _retag_head_shard_after_reshape(head_rows, reference=z, shard_dim=1)
        head_rows_bf16 = ttnn.typecast(head_rows, ttnn.bfloat16, memory_config=l1)
        _deallocate(recurrent_output)
        _retag_head_shard_after_reshape(head_rows_bf16, reference=z, shard_dim=1)
        normalized_heads = ttnn.rms_norm(
            head_rows_bf16, weight=self.weights.norm, epsilon=RMS_NORM_EPS, memory_config=l1
        )
        _deallocate(head_rows_bf16)
        _require_shape(
            normalized_heads,
            (lanes, VALUE_HEADS_PER_DEVICE, CHUNK_SIZE, HEAD_DIM),
            label="GDN lane rows normalized heads",
        )
        z_fp32 = ttnn.typecast(z, ttnn.float32, memory_config=l1)
        _deallocate(z)
        sigmoid_fp32 = ttnn.sigmoid(z_fp32, memory_config=l1)
        _deallocate(z_fp32)
        sigmoid_bf16 = ttnn.typecast(sigmoid_fp32, ttnn.bfloat16, memory_config=l1)
        _deallocate(sigmoid_fp32)
        flat = ttnn.experimental.view(normalized_heads, (1, 1, lanes * CHUNK_SIZE, VALUE_WIDTH_PER_DEVICE))
        _retag_head_shard_after_reshape(flat, reference=z, shard_dim=3)
        normalized = ttnn.matmul(
            rows_state.lane_constants.fold_select, flat, memory_config=l1, compute_kernel_config=self.compute_config
        )
        _deallocate(normalized_heads)  # the fold view shares its buffer
        _retag_head_shard_after_reshape(normalized, reference=z, shard_dim=3)
        self.mesh_contract.validate_tensor(normalized, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        _require_shape(normalized, (1, 1, CHUNK_SIZE, VALUE_WIDTH_PER_DEVICE), label="GDN lane rows folded output")
        gated = ttnn.multiply(normalized, sigmoid_bf16, memory_config=self.out_proj_act_memory_config)
        _deallocate(normalized, sigmoid_bf16)
        self.mesh_contract.validate_tensor(gated, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        _require_shape(gated, (1, 1, CHUNK_SIZE, VALUE_WIDTH_PER_DEVICE), label="GDN lane rows gated output")
        output = self._out_proj_tile(gated, full_hidden)
        _deallocate(full_hidden)
        _require_shape(output, (1, 1, CHUNK_SIZE, HIDDEN_SIZE_PER_DEVICE), label="GDN lane rows output tile")
        self.mesh_contract.validate_tensor(output, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        return output

    def forward_rows_lanes(
        self, hidden_rows, state: Qwen38TTNNGDNState, rows_state: Qwen38TTNNGDNLaneRowsState
    ) -> Qwen38TTNNGDNRowsResult:
        """Run R consecutive positions of every lane from the lanes' committed states without committing anything.

        ``hidden_rows`` is the lane-major ``[1,1,32,640]`` tile (row ``u*R + j`` = lane u's row j, rows past ``B*R``
        zero); ``state.recurrent`` ``[B,12,128,128]`` is read only.  The result's ``hidden_rows`` is the whole
        ``[1,1,32,640]`` output tile in a new buffer (rows past ``B*R`` exact zeros) and ``final_state`` the lanes'
        states after all rows, a new buffer the caller deallocates.
        """

        rows = self._validate_lane_rows_state(rows_state, state)
        full_hidden = self._all_gather_rows(hidden_rows, rows)
        # The body between the gather and the result, dispatched exactly as ``forward_rows`` dispatches the single
        # stream's: the verify-rows fold's registry step (``fused.resolve_admitted("gdn_rows_scan")``, resolved once)
        # runs the fold's lanes form on a lane rows state that carries its buffers and the lanes chain
        # (``_rows_body_lanes_chain``, the fold's composed fallback for a lane rows state) otherwise --
        # QWEN38_FUSED_OFF=gdn_rows_scan restores the chain for both streams.
        body = self._rows_body()
        output, final_state = body(self, full_hidden, rows_state, state)
        _require_shape(output, (1, 1, CHUNK_SIZE, HIDDEN_SIZE_PER_DEVICE), label="GDN lane rows output tile")
        self.mesh_contract.validate_tensor(output, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        return Qwen38TTNNGDNRowsResult(output, final_state, state, rows_state)

    def _rows_body_lanes_chain(self, full_hidden, rows_state: Qwen38TTNNGDNLaneRowsState, state: Qwen38TTNNGDNState):
        """The lanes chain between the gather and the result: the projection, the per-lane expands, the FIR, the
        chunk inputs, ONE batched chunk call from the lanes' states and the gate + out-projection.  Returns
        ``(output, final_state)`` with the lanes' states after all rows in a new buffer the caller deallocates."""

        z, a, b = self._project_rows(full_hidden, rows_state)
        self._select_rows(
            rows_state.lane_constants.expand_select,
            rows_state.qkv,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            label="GDN lane rows expand",
            output_tensor=rows_state.qkv_lanes,
        )
        a_lanes = self._expand_lane_rows(a, rows_state)
        b_lanes = self._expand_lane_rows(b, rows_state)
        conv = self._causal_conv_rows_lanes(rows_state)
        self._make_chunk_inputs_lanes(conv, a_lanes, b_lanes, rows_state)
        recurrent_output, final_state = self._chunk_rows_lanes(rows_state, initial_state=state.recurrent)
        output = self._gate_and_project_rows_lanes(recurrent_output, z, full_hidden, rows_state)
        return output, final_state

    def commit_rows_lanes(
        self,
        state: Qwen38TTNNGDNState,
        rows_state: Qwen38TTNNGDNLaneRowsState,
        selectors: Qwen38TTNNRowsSelectorsLanes,
    ) -> None:
        """Commit ``c_u`` rows of the last ``forward_rows_lanes`` pass into lane u's state in place.

        The catch-up reruns the batched kernel from the committed states with every lane's beta and g masked past
        its committed prefix; the final states land in ``state.recurrent`` through the exact fp32 0/1 select
        ``recurrent * [c_u == 0] + final * [c_u >= 1]`` (a lane committing nothing -- inactive, or seeded and not yet
        verified -- keeps its recurrent bitwise: the masked rerun is an identity only to TF32 precision), and every
        lane's FIR history moves forward by ``c_u``
        rows with one equal-batch 0/1 selection matmul (the KEEP row copies an inactive lane's history unchanged).
        """

        rows = self._validate_lane_rows_state(rows_state, state)
        selectors.validate(rows_state.lanes, rows)
        dram = ttnn.DRAM_MEMORY_CONFIG
        scan = fused.gdn_rows_scan.lane_buffers_of(rows_state)
        if scan is not None:
            # Lane u's state after c_u rows is the forward pass's prefix slot u * R + c_u - 1: one pick per lane, no
            # re-run; a lane committing nothing keeps its state (the pick writes nothing for it).
            fused.gdn_rows_scan.commit_lanes(self, rows_state, scan, state, selectors)
        else:
            output, final_state = self._chunk_rows_lanes(
                rows_state, initial_state=state.recurrent, committed_mask=selectors.committed_mask
            )
            _deallocate(output)
            kept = ttnn.multiply(state.recurrent, selectors.keep_col, memory_config=dram)
            taken = ttnn.multiply(final_state, selectors.commit_col, memory_config=dram)
            landed = ttnn.add(kept, taken, output_tensor=state.recurrent)
            _require_landed(landed, state.recurrent, label="GDN lane rows committed states")
            _deallocate(kept, taken, final_state)
        window = self._lane_conv_window(rows_state)
        landed = ttnn.matmul(
            selectors.history_select,
            window,
            memory_config=dram,
            compute_kernel_config=self.compute_config,
            optional_output_tensor=rows_state.history,
        )
        _require_landed(landed, rows_state.history, label="GDN lane rows history select")
        _deallocate(window)
        state.validate()


def validate_gdn_static_contract() -> None:
    """No-device invariant gate used by static bring-up tests."""

    exact = {
        "TP_SIZE": TP_SIZE,
        "HIDDEN_SIZE": HIDDEN_SIZE,
        "HIDDEN_SIZE_PER_DEVICE": HIDDEN_SIZE_PER_DEVICE,
        "QK_HEADS": QK_HEADS,
        "QK_HEADS_PER_DEVICE": QK_HEADS_PER_DEVICE,
        "VALUE_HEADS": VALUE_HEADS,
        "VALUE_HEADS_PER_DEVICE": VALUE_HEADS_PER_DEVICE,
        "HEAD_DIM": HEAD_DIM,
        "QKV_WIDTH": QKV_WIDTH,
        "QKV_WIDTH_PER_DEVICE": QKV_WIDTH_PER_DEVICE,
        "VALUE_WIDTH_PER_DEVICE": VALUE_WIDTH_PER_DEVICE,
        "QKVZAB_WIDTH_PER_DEVICE": QKVZAB_WIDTH_PER_DEVICE,
        "A_COLUMN": A_COLUMN,
        "B_COLUMN": B_COLUMN,
        "PROJECTION_WIDTH_PER_DEVICE": PROJECTION_WIDTH_PER_DEVICE,
        "CONV_KERNEL_SIZE": CONV_KERNEL_SIZE,
        "QK_REPEAT_FACTOR": QK_REPEAT_FACTOR,
    }
    expected = {
        "TP_SIZE": 4,
        "HIDDEN_SIZE": 2560,
        "HIDDEN_SIZE_PER_DEVICE": 640,
        "QK_HEADS": 16,
        "QK_HEADS_PER_DEVICE": 4,
        "VALUE_HEADS": 48,
        "VALUE_HEADS_PER_DEVICE": 12,
        "HEAD_DIM": 128,
        "QKV_WIDTH": 10240,
        "QKV_WIDTH_PER_DEVICE": 2560,
        "VALUE_WIDTH_PER_DEVICE": 1536,
        "QKVZAB_WIDTH_PER_DEVICE": 4120,
        "A_COLUMN": 4096,
        "B_COLUMN": 4128,
        "PROJECTION_WIDTH_PER_DEVICE": 4160,
        "CONV_KERNEL_SIZE": 4,
        "QK_REPEAT_FACTOR": 3,
    }
    if exact != expected:
        raise RuntimeError(f"Qwen3.8 GDN static contract drifted: {exact} != {expected}")


__all__ = [
    "Qwen38TTNNGDN",
    "Qwen38TTNNGDNResult",
    "Qwen38TTNNGDNRowsConstants",
    "Qwen38TTNNGDNRowsResult",
    "Qwen38TTNNGDNRowsState",
    "Qwen38TTNNGDNWeights",
    "Qwen38TTNNGDNState",
    "Qwen38TTNNGDNSnapshot",
    "Qwen38TTNNRowsSelectors",
    "build_rows_selectors",
    "chunk_constant_tiles",
    "rows_window_select_tiles",
    "validate_gdn_static_contract",
]
