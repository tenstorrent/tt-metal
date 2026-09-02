# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Model-agnostic utilities for the prefill runner.

Only metal-native helpers live here — pure ttnn, no upward dependencies
on blaze (`_migration`, `_mpi_test_helpers`) and no dependency on any specific
model package. Migration-coupled diagnostics live in blaze at
`disaggregation/migration/python/prefill_runner_util.py`.

Per-model plumbing (which model to build, where its weights/config/trace live,
how to allocate its KV cache, how to call chunked prefill) lives behind the
PrefillModelAdapter (../adapter.py), NOT here. Model-specific KV diagnostics /
PCC live in the model package's own runner_utils.
"""

import os
from pathlib import Path

from loguru import logger

import ttnn


def _create_fabric_router_config(max_payload_size):
    """FabricRouterConfig with a custom max payload size. Inlined here (a 3-line
    ttnn wrapper) so this common module needs no model-package import."""
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


# ---------------------------------------------------------------------------
# Device / H2D-service setup
# ---------------------------------------------------------------------------
def open_mesh_device(
    mesh_shape: tuple, model_cfg: type, l1_small_size: int = 0, trace_region_size: int = 0
) -> ttnn.MeshDevice:
    """Configure fabric and open the mesh device.

    Default fabric is 1D for sp<=8, else 2D. PREFILL_FABRIC_MODE overrides this: the D2D-socket
    pipeline needs 2D even at sp=8 because a MeshSocket routes over 2D fabric, and set_fabric_config
    is one global config for the whole run. Accepted modes: 1d, 2d, 1d_ring, 2d_torus_x,
    2d_torus_y, 2d_torus_xy. A torus mode physically wraps the named axis (x = cols = tp_axis,
    y = rows = sp_axis) into a ring and MUST match the mesh-graph descriptor's dim_types (a Ring
    collective on an axis the fabric does not wrap hangs). The per-axis CCL topology is derived from
    the opened fabric via tt_ccl.per_axis_topology().

    `l1_small_size` > 0 carves an L1_SMALL region (needed when an op routes its
    semaphores there, e.g. the Kimi MoE routing all-gather with use_l1_small_for_semaphores).

    `trace_region_size` > 0 reserves device DRAM for ttnn trace capture — needed when the runtime
    replays a captured forward (TtPrefillRuntime use_trace). 0 = no trace region (default)."""
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
        fabric_config = ttnn.FabricConfig.FABRIC_1D if sp <= 8 else ttnn.FabricConfig.FABRIC_2D
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
"""PCIe alignment the H2D socket requires of its per-chip page size (``H2DSocket::set_page_size``:
``page_size % pcie_alignment_ == 0``, tt_metal/distributed/h2d_socket.cpp:799).

Hardcoded because ttnn exposes ``get_dram_alignment`` / ``get_l1_alignment`` to python but not the
HOST/PCIe one. 64 is Blackhole's ``PCIE_ALIGNMENT``; Wormhole's is 32, so 64 satisfies both -- this
only ever rounds a row UP, and the extra ids are never read.
"""

_H2D_ID_BYTES = 4  # uint32 token ids


def h2d_row_len(chunk_size: int, sp_factor: int) -> int:
    """Per-chip TRUNK row length: this chip's ``chunk_size // sp_factor`` shard of the chunk.

    Deliberately NOT rounded to the PCIe page -- every caller gets exactly the width it always got,
    and a ``chunk_size / sp_factor`` that is not already page-aligned is a pre-existing condition of
    that configuration. MTP widens the SOCKET row past this (:func:`mtp_union_rows`) but not this
    number: it is where the runner cuts the arriving row, so an MTP run and a plain run hand the
    model byte-identical trunk ids.
    """
    assert chunk_size % sp_factor == 0, f"chunk_size={chunk_size} must be divisible by sp_factor={sp_factor}"
    return chunk_size // sp_factor


# ---------------------------------------------------------------------------
# MTP transport (GLM-5.2, issue #53533)
# ---------------------------------------------------------------------------

TILE_HEIGHT = 32

MTP_TOKEN_ALIGN = TILE_HEIGHT
"""Granularity of the MTP token block, in ids. Three constraints meet here, and 32 is the smallest
number satisfying all of them:

* the H2D socket's page must be ``H2D_PAGE_ALIGNMENT_BYTES``-aligned, so ``L + num_mtp_tokens`` is a
  multiple of 16 ids (16 * 4 B = 64 B);
* the runner cuts the arriving row at ``L`` into the trunk ids and the MTP ids, and both cuts must
  land on a page boundary too;
* the first rank embeds the two halves separately and stacks them, and a TILE row-concat needs BOTH
  operands to be a whole number of 32-row tiles -- so ``num_mtp_tokens`` itself must be tile-aligned,
  not just the sum.

With the production ``L = 640``: the trunk is 40 pages / 20 tiles, the MTP block is 2 pages / 1 tile,
the socket row is 672 ids = 42 pages, and the union embedding is 21 tiles. No pad anywhere.

That third constraint is what one widened row alone would not have given: 644 ids satisfies the page
rules once rounded but leaves a 4-row MTP block that cannot be tile-concatenated onto the trunk
embedding without a pad. Rounding the block to a tile is what lets the union be built as two
gathers stacked rather than one gather over a re-joined id row -- see
:meth:`~models.demos.deepseek_v3_d_p.tt.mtp_prefill.device_windows.MTPUnionEmbedding.from_ids`.
"""


def num_mtp_tokens(mtp_levels: int) -> int:
    """MTP lookahead ids the H2D row carries past this chip's trunk shard: ``mtp_levels`` rounded up
    to ``MTP_TOKEN_ALIGN``. 0 when MTP is off.

    THE number every side of the MTP transport builds to -- the producer's rows, the H2D socket's
    spec, the runner's cut point, the union embedding's height and the D2D activation's height. Chip
    ``c``'s MTP ids are ``stream[(c+1)*L : (c+1)*L + num_mtp_tokens]`` -- they hang past the end of
    ``c``'s own shard into ``c+1``'s territory, so ``chunk_row ++ mtp_row`` is exactly the contiguous
    ``stream[c*L : c*L + L + num_mtp_tokens]`` and MTP level ``k``'s window is the SAME local slice
    ``[k, k+L)`` on every chip -- one uniform slice, no cross-chip rotation.

    Note they are PER CHIP: only the last chip's ids reach into the next chunk, the other ``sp-1``
    take theirs from inside this one. That is what makes the windows uniform.
    """
    assert mtp_levels >= 0, f"mtp_levels must be non-negative, got {mtp_levels}"
    if not mtp_levels:
        return 0
    return -(-mtp_levels // MTP_TOKEN_ALIGN) * MTP_TOKEN_ALIGN


def mtp_union_rows(chunk_size: int, sp_factor: int, mtp_levels: int) -> int:
    """Rows of ONE chip's union embedding: its ``L`` chunk rows plus the ``num_mtp_tokens`` lookahead
    rows.

    Level ``k`` (1..K) reads rows ``[k, k+L)`` of it, so the deepest level touches row ``K + L - 1``
    and the rest of the lookahead is transport padding no level reads.
    """
    rows = h2d_row_len(chunk_size, sp_factor) + num_mtp_tokens(mtp_levels)
    assert rows % TILE_HEIGHT == 0, (
        f"union embedding is {rows} rows, not a whole number of {TILE_HEIGHT}-row tiles; "
        f"chunk_size/sp_factor = {chunk_size // sp_factor} must itself be tile-aligned"
    )
    return rows


# ---------------------------------------------------------------------------
# H2D token sockets
# ---------------------------------------------------------------------------


def make_token_spec(mesh_shape: tuple, row_len: int) -> ttnn.TensorSpec:
    """``[sp_factor, 1, row_len]`` uint32 ROW_MAJOR DRAM -- the shape of any per-chip token push.
    The mapper shards axis 0 across the SP rows, so each chip receives ``[1, 1, row_len]``."""
    return ttnn.TensorSpec(
        shape=ttnn.Shape([mesh_shape[0], 1, row_len]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


def make_h2d_spec(mesh_shape: tuple, chunk_size: int, mtp_levels: int = 0) -> ttnn.TensorSpec:
    """Per-push spec of THE H2D token socket -- there is exactly one, MTP or not.

    Plain: ``chunk_size // sp`` ids per chip. MTP: those plus ``num_mtp_tokens(mtp_levels)``
    lookahead ids, so chip ``c``'s row is the contiguous ``stream[c*L : c*L + L + num_mtp_tokens]``.
    The runner cuts the row back at ``L`` on arrival (``prefill_runner._socket_next``), and hands the model the same
    ``[1, 1, chunk_size // sp]`` trunk it gets with MTP off.
    """
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
    """Construct an H2DStreamService delivering `global_spec` once per push.

    Build the spec with :func:`make_h2d_spec`, which maps ``[Shard(0), Replicate]`` on a
    ``(sp, tp)`` mesh: the first tensor axis splits across the mesh rows and nothing else is split.
    """
    row_len = int(global_spec.shape[-1])
    per_chip_bytes = row_len * _H2D_ID_BYTES
    assert per_chip_bytes % H2D_PAGE_ALIGNMENT_BYTES == 0, (
        f"per-chip page is {per_chip_bytes}B for a {row_len}-id row, not a multiple of "
        f"{H2D_PAGE_ALIGNMENT_BYTES}B; the socket rejects a non-PCIe-aligned page size outright"
    )
    mapper = ttnn.create_mesh_mapper(mesh_device, mapper_config)
    # worker_cores set so the service-core kernel multicasts a data-ready inc
    # after each transfer; inbound_socket_service_sync() waits on that on-device, which
    # avoids the host-side barrier() round-trip per iteration.
    # metadata_size_bytes set so the producer can ship per-iter control bytes
    # (slot_id, actual_start, actual_end) inline with the token push.
    service = ttnn.H2DStreamService(
        mesh_device=mesh_device,
        global_spec=global_spec,
        fifo_size_bytes=8 * per_chip_bytes,  # 8 in-flight pages of headroom (0 would auto-size)
        max_socket_page_size_bytes=per_chip_bytes,  # cap socket page at one tensor page (0 = auto/coalesced)
        mapper=mapper,
        worker_cores=worker_cores,
        metadata_size_bytes=metadata_size_bytes,
    )
    logger.info(
        f"[h2d] H2DStreamService built: global_shape=({mesh_device.shape[0]},1,{row_len}) "
        f"uint32 ROW_MAJOR DRAM, per_chip_bytes={per_chip_bytes}, worker_cores={worker_cores}"
    )
    return service


# ---------------------------------------------------------------------------
# D2D pipeline activation
# ---------------------------------------------------------------------------


def activation_global_spec(rows: int, hidden_size: int) -> ttnn.TensorSpec:
    """Global spec of the inter-rank activation carried over the D2D pipeline socket:
    ``[1, 1, rows, hidden_size]`` bf16 TILE DRAM. The caller's mesh mapper shards it (rows across SP,
    emb across TP) to match the embedding output layout the downstream model consumes.

    Size it with :func:`d2d_activation_rows` and :func:`d2d_activation_width`, never with
    ``chunk_size`` directly -- MTP makes the two differ."""
    return ttnn.TensorSpec(
        shape=ttnn.Shape([1, 1, rows, hidden_size]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


def d2d_activation_rows(chunk_size: int, *, sp_factor: int, mtp_levels: int = 0) -> int:
    """GLOBAL row count of the D2D pipeline activation for this configuration.

    Plain prefill and DFlash ship one row per token: ``chunk_size``. MTP stacks the chunk's union
    EMBEDDING under the hidden, so each chip sends its ``L`` hidden rows followed by its
    ``L + num_mtp_tokens`` embedding rows. Both are multiples of 32, so the receiver's split is a
    tile-aligned row slice.

    Sending the embedding rather than the ids is what lets the LAST rank run its levels without an
    embedding table: it slices windows out of what arrived instead of gathering them itself.
    """
    if not mtp_levels:
        return chunk_size
    return chunk_size + sp_factor * mtp_union_rows(chunk_size, sp_factor, mtp_levels)


def d2d_activation_width(hidden_size: int, *, dflash: bool = False) -> int:
    """GLOBAL width of the D2D pipeline activation. Plain hidden is ``hidden_size``; DFlash packs the
    drafter FC partial beside it (2H). MTP adds ROWS, not columns -- see :func:`d2d_activation_rows`.
    """
    return hidden_size * (2 if dflash else 1)


def resolve_trace_dir(path) -> Path:
    """Resolve a trace dir to the one holding metadata.json. vllm traces nest metadata.json + kv_cache
    under a single run-hash subdir, so if `path` itself has no metadata.json, descend into the sole
    subdir that does."""
    path = Path(path)
    if (path / "metadata.json").exists():
        return path
    subs = [d for d in sorted(path.iterdir()) if d.is_dir() and (d / "metadata.json").exists()]
    if len(subs) != 1:
        raise FileNotFoundError(f"no metadata.json in {path} or a unique subdir (found {len(subs)} candidates)")
    return subs[0]


def load_trace_token_ids(trace_dir, total_len=None) -> list:
    """Input token_ids from a resolved trace's metadata.json (optionally truncated to total_len)."""
    import json

    with open(Path(trace_dir) / "metadata.json") as f:
        md = json.load(f)
    tids = list(md["token_ids"])
    return tids[:total_len] if total_len is not None else tids


# ---------------------------------------------------------------------------
# Layer assignment
# ---------------------------------------------------------------------------


def _snap_counts_to_starts(counts, valid_starts, num_layers):
    """Nudge an even split's interior rank boundaries onto the nearest valid start (preserving
    sum == num_layers), for models that constrain where a rank may begin (layer_split_boundaries).
    Nearest by |distance| then lower index; each boundary is used at most once and stays increasing."""
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
    """Contiguous (first_layer_idx, count) per rank. PREFILL_PP_LAYER_COUNTS, a
    comma-separated count list summing to num_layers, overrides the default even
    split (remainder handed to the earlier ranks).

    ``valid_starts`` (from the adapter's ``layer_split_boundaries``): layer indices at which a rank may
    begin. None => unconstrained. When set, the default even split is auto-snapped onto valid
    boundaries, and any split (explicit or snapped) whose rank starts fall off them is rejected early.

    ``mtp_levels`` (K): the LAST rank also runs K MTP levels after its trunk layers, so an even split
    of the trunk alone leaves it K blocks late with every other rank idle behind it. Balance
    num_layers + K layer-EQUIVALENTS and hand the tail back its K: GLM-5.2 on 4 ranks goes 18/20/20/20
    to trunk 22/20/20/16 (equivalents 22/20/20/20), the #53533 re-split; K=0 is identical to the old
    split. One level == one layer is a MODEL, not a measurement -- a level is a full MoE block plus
    eh_proj and two norms, so it under-counts slightly; PREFILL_PP_LAYER_COUNTS (TRUNK counts only)
    stays the hand-tuning escape hatch."""
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
