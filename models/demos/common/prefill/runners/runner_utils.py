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
    mesh_shape: tuple, max_payload_size: int, l1_small_size: int = 0, trace_region_size: int = 0
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
        max_payload_size=max_payload_size,
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


# Rows per chip the MTP union's token ids ride in below the hidden on the D2D activation (one tile row of
# byte-encoded ids; the last rank gathers the union embedding from them, see mtp_prefill/id_transport.py).
MTP_ID_ROWS = TILE_HEIGHT


def d2d_activation_rows(chunk_size: int, *, sp_factor: int, mtp_levels: int = 0) -> int:
    if not mtp_levels:
        return chunk_size
    return chunk_size + sp_factor * MTP_ID_ROWS


def d2d_activation_width(hidden_size: int, *, dflash: bool = False) -> int:
    return hidden_size * (2 if dflash else 1)


def indexer_indices_global_spec(per_chip_shape, mesh_shape) -> ttnn.TensorSpec:
    _, _, rows, k = per_chip_shape
    sp, tp = mesh_shape
    return ttnn.TensorSpec(
        shape=ttnn.Shape([1, 1, sp * rows, tp * k]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


def resolve_trace_dir(path) -> Path:
    path = Path(path)
    if (path / "metadata.json").exists():
        return path
    subs = [d for d in sorted(path.iterdir()) if d.is_dir() and (d / "metadata.json").exists()]
    if len(subs) != 1:
        raise FileNotFoundError(f"no metadata.json in {path} or a unique subdir (found {len(subs)} candidates)")
    return subs[0]


def load_trace_golden_span(trace_dir) -> tuple:
    import json

    with open(Path(trace_dir) / "metadata.json") as f:
        md = json.load(f)
    rows = md.get("capture_rows")
    if rows is not None:
        return int(rows[0]), int(rows[1])
    return 0, int(md.get("n_prompt_tokens") or len(md["token_ids"]))


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


def _balanced_counts(layer_costs, num_ranks: int, tail_cost: float, valid_starts=None) -> list[int]:
    """Contiguous partition of ``layer_costs`` into ``num_ranks`` non-empty runs minimizing the costliest rank,
    the last rank also paying ``tail_cost``; among those, the most even one (least sum of squared rank costs).
    Exact: one DP for the bound, a second under it. ``valid_starts`` restricts every rank start but the first."""
    n = len(layer_costs)
    prefix = [0.0]
    for c in layer_costs:
        prefix.append(prefix[-1] + c)
    allowed = (lambda i: True) if valid_starts is None else (lambda i: i in valid_starts)
    inf = float("inf")

    def solve(score, bound):
        # best[r][i]: best score over splits of layers [0, i) into r ranks; cut[r][i]: that split's last start.
        best = [[inf] * (n + 1) for _ in range(num_ranks + 1)]
        cut = [[0] * (n + 1) for _ in range(num_ranks + 1)]
        best[0][0] = 0.0
        for r in range(1, num_ranks + 1):
            for i in range(r, n + 1):
                extra = tail_cost if (r == num_ranks and i == n) else 0.0
                for j in range(r - 1, i):
                    if (j > 0 and not allowed(j)) or best[r - 1][j] == inf:
                        continue
                    rank = prefix[i] - prefix[j] + extra
                    if rank > bound:
                        continue
                    v = score(best[r - 1][j], rank)
                    if v < best[r][i]:
                        best[r][i], cut[r][i] = v, j
        return best[num_ranks][n], cut

    worst, _ = solve(max, inf)
    if worst == inf:
        raise ValueError(f"cannot split {n} layers into {num_ranks} ranks on valid starts {sorted(valid_starts)}")
    _, cut = solve(lambda acc, rank: acc + rank * rank, worst + 1e-9)
    counts, i = [], n
    for r in range(num_ranks, 0, -1):
        j = cut[r][i]
        counts.append(i - j)
        i = j
    return counts[::-1]


def compute_layer_split(
    num_layers: int, num_ranks: int, valid_starts=None, mtp_levels: int = 0, layer_costs=None
) -> list[tuple[int, int]]:
    """Each rank's ``(first_layer, count)``. ``PREFILL_PP_LAYER_COUNTS`` overrides. With ``layer_costs`` --
    ``(per-trunk-layer costs, last-rank tail cost)``, the tail being the MTP levels -- the split balances
    that cost; otherwise layers and MTP levels are dealt out evenly."""
    override = os.environ.get("PREFILL_PP_LAYER_COUNTS")
    if override:
        counts = [int(x) for x in override.split(",")]
        if len(counts) != num_ranks or sum(counts) != num_layers:
            raise ValueError(
                f"PREFILL_PP_LAYER_COUNTS={override!r} must list {num_ranks} counts summing to "
                f"{num_layers} (got {len(counts)} counts summing to {sum(counts)})"
            )
    elif layer_costs is not None:
        costs, tail_cost = layer_costs
        assert len(costs) == num_layers, f"{len(costs)} layer costs for {num_layers} layers"
        counts = _balanced_counts(costs, num_ranks, tail_cost, valid_starts)
    else:
        base, rem = divmod(num_layers + mtp_levels, num_ranks)
        counts = [base + (1 if r >= num_ranks - rem else 0) for r in range(num_ranks)]
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
