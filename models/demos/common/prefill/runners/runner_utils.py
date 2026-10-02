# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


import os
from pathlib import Path

from loguru import logger

import ttnn


def _create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def open_mesh_device(
    mesh_shape: tuple, model_cfg: type, l1_small_size: int = 0, trace_region_size: int = 0
) -> ttnn.MeshDevice:
    sp = mesh_shape[0]
    fabric_mode = os.environ.get("PREFILL_FABRIC_MODE", "").strip().lower()
    fabric_mode_map = {
        "1d": ttnn.FabricConfig.FABRIC_1D,
        "2d": ttnn.FabricConfig.FABRIC_2D,
        "1d_ring": ttnn.FabricConfig.FABRIC_1D_RING,
        "2d_torus_x": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
        "2d_torus_y": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
        "2d_torus_xy": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    }
    if fabric_mode in fabric_mode_map:
        fabric_config = fabric_mode_map[fabric_mode]
    elif fabric_mode:
        raise ValueError(f"PREFILL_FABRIC_MODE must be one of {sorted(fabric_mode_map)}, got {fabric_mode!r}")
    else:
        # The torus modes need a descriptor declaring RING on both axes; a LINE one hangs at bring-up.
        fabric_config = ttnn.FabricConfig.FABRIC_2D_TORUS_XY
    logger.info(f"Fabric config: {fabric_config} (sp={sp}, PREFILL_FABRIC_MODE={fabric_mode or 'unset'})")

    fabric_router_config = _create_fabric_router_config(
        max_payload_size=model_cfg.FABRIC_PAYLOAD_SIZE,
    )

    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.RELAXED_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        fabric_router_config,
    )
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*mesh_shape), l1_small_size=l1_small_size, trace_region_size=trace_region_size
    )


H2D_PAGE_ALIGNMENT_BYTES = 64

_H2D_ID_BYTES = 4


def h2d_row_len(chunk_size: int, sp_factor: int) -> int:
    assert chunk_size % sp_factor == 0, f"chunk_size={chunk_size} must be divisible by sp_factor={sp_factor}"
    return chunk_size // sp_factor


TILE_HEIGHT = 32

MTP_PAD_TOKEN_ID = 0xFFFFFFFF

MTP_TOKEN_ALIGN = TILE_HEIGHT


def num_mtp_tokens(mtp_levels: int) -> int:
    assert mtp_levels >= 0, f"mtp_levels must be non-negative, got {mtp_levels}"
    if not mtp_levels:
        return 0
    return -(-mtp_levels // MTP_TOKEN_ALIGN) * MTP_TOKEN_ALIGN


def mtp_union_rows(chunk_size: int, sp_factor: int, mtp_levels: int) -> int:
    rows = h2d_row_len(chunk_size, sp_factor) + num_mtp_tokens(mtp_levels)
    assert rows % TILE_HEIGHT == 0, (
        f"union embedding is {rows} rows, not a whole number of {TILE_HEIGHT}-row tiles; "
        f"chunk_size/sp_factor = {chunk_size // sp_factor} must itself be tile-aligned"
    )
    return rows


def make_token_spec(mesh_shape: tuple, row_len: int) -> ttnn.TensorSpec:
    return ttnn.TensorSpec(
        shape=ttnn.Shape([mesh_shape[0], 1, row_len]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


def make_h2d_spec(mesh_shape: tuple, chunk_size: int, mtp_levels: int = 0) -> ttnn.TensorSpec:
    if mtp_levels:
        return make_token_spec(mesh_shape, mtp_union_rows(chunk_size, mesh_shape[0], mtp_levels))
    return make_token_spec(mesh_shape, h2d_row_len(chunk_size, mesh_shape[0]))


def build_h2d_service(
    mesh_device: ttnn.MeshDevice,
    *,
    global_spec: ttnn.TensorSpec,
    mapper_config: ttnn.MeshMapperConfig,
    worker_cores: ttnn.CoreRange,
    metadata_size_bytes: int,
) -> ttnn.H2DStreamService:
    row_len = int(global_spec.shape[-1])
    per_chip_bytes = row_len * _H2D_ID_BYTES
    assert per_chip_bytes % H2D_PAGE_ALIGNMENT_BYTES == 0, (
        f"per-chip page is {per_chip_bytes}B for a {row_len}-id row, not a multiple of "
        f"{H2D_PAGE_ALIGNMENT_BYTES}B; the socket rejects a non-PCIe-aligned page size outright"
    )
    mapper = ttnn.create_mesh_mapper(mesh_device, mapper_config)
    service = ttnn.H2DStreamService(
        mesh_device=mesh_device,
        global_spec=global_spec,
        fifo_size_bytes=8 * per_chip_bytes,
        max_socket_page_size_bytes=per_chip_bytes,
        mapper=mapper,
        worker_cores=worker_cores,
        metadata_size_bytes=metadata_size_bytes,
    )
    logger.info(
        f"[h2d] H2DStreamService built: global_shape=({mesh_device.shape[0]},1,{row_len}) "
        f"uint32 ROW_MAJOR DRAM, per_chip_bytes={per_chip_bytes}, worker_cores={worker_cores}"
    )
    return service


def activation_global_spec(rows: int, hidden_size: int, planes: int = 1) -> ttnn.TensorSpec:
    return ttnn.TensorSpec(
        shape=ttnn.Shape([1, planes, rows, hidden_size]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


def d2d_activation_rows(chunk_size: int, *, sp_factor: int, mtp_levels: int = 0) -> int:
    if not mtp_levels:
        return chunk_size
    return chunk_size + sp_factor * mtp_union_rows(chunk_size, sp_factor, mtp_levels)


def d2d_activation_width(hidden_size: int, *, dflash: bool = False) -> int:
    return hidden_size * (2 if dflash else 1)


def resolve_trace_dir(path) -> Path:
    path = Path(path)
    if (path / "metadata.json").exists():
        return path
    subs = [d for d in sorted(path.iterdir()) if d.is_dir() and (d / "metadata.json").exists()]
    if len(subs) != 1:
        raise FileNotFoundError(f"no metadata.json in {path} or a unique subdir (found {len(subs)} candidates)")
    return subs[0]


def load_trace_token_ids(trace_dir, total_len=None) -> list:
    import json

    with open(Path(trace_dir) / "metadata.json") as f:
        md = json.load(f)
    tids = list(md["token_ids"])
    return tids[:total_len] if total_len is not None else tids


def _snap_counts_to_starts(counts, valid_starts, num_layers):
    valid = sorted(valid_starts)
    boundaries, s = [], 0
    for c in counts[:-1]:
        s += c
        boundaries.append(s)
    snapped, prev = [], 0
    for b in boundaries:
        cand = min(
            (v for v in valid if prev < v < num_layers and v not in snapped),
            key=lambda v: (abs(v - b), v),
            default=None,
        )
        if cand is None:
            raise ValueError(f"cannot place {len(counts)} pipeline ranks on valid layer boundaries {valid}")
        snapped.append(cand)
        prev = cand
    out, prev = [], 0
    for b in [*snapped, num_layers]:
        out.append(b - prev)
        prev = b
    return out


def compute_layer_split(
    num_layers: int, num_ranks: int, valid_starts=None, mtp_levels: int = 0
) -> list[tuple[int, int]]:
    override = os.environ.get("PREFILL_PP_LAYER_COUNTS")
    if override:
        counts = [int(x) for x in override.split(",")]
        if len(counts) != num_ranks or sum(counts) != num_layers:
            raise ValueError(
                f"PREFILL_PP_LAYER_COUNTS={override!r} must list {num_ranks} counts summing to "
                f"{num_layers} (got {len(counts)} counts summing to {sum(counts)})"
            )
    else:
        base, rem = divmod(num_layers + mtp_levels, num_ranks)
        counts = [base + (1 if r < rem else 0) for r in range(num_ranks)]
        counts[-1] -= mtp_levels
        if counts[-1] < 1:
            raise ValueError(
                f"{mtp_levels} MTP levels leave the last of {num_ranks} ranks {counts[-1]} trunk layers "
                f"out of {num_layers}: the tail would hold no trunk stage. Use fewer ranks, or set "
                f"PREFILL_PP_LAYER_COUNTS (TRUNK counts, summing to {num_layers})."
            )
        if valid_starts is not None:
            counts = _snap_counts_to_starts(counts, valid_starts, num_layers)

    ranges = []
    start = 0
    for count in counts:
        ranges.append((start, count))
        start += count

    if valid_starts is not None:
        for first_idx, _ in ranges:
            if first_idx not in valid_starts:
                near = sorted(b for b in valid_starts if abs(b - first_idx) <= 4)
                raise ValueError(
                    f"pipeline rank starts at layer {first_idx}, not a valid boundary for this model "
                    f"(nearest valid: {near}). Set PREFILL_PP_LAYER_COUNTS so every cumulative boundary "
                    f"is a valid start."
                )
    return ranges
