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


def h2d_row_len(chunk_size: int, sp_factor: int, lookahead: int = 0) -> int:
    """Per-chip H2D row length: this chip's `chunk_size // sp_factor` trunk tokens plus at least
    `lookahead` tokens of OVERLAP with the next chip's slice, rounded UP so the row is a whole
    number of PCIe pages.

    With `lookahead == 0` this is the plain SP shard. With `lookahead == K` (MTP with K levels) chip
    `c` carries `stream[c*L : c*L + L + K]`, so its row holds every token any MTP level needs from
    it: level `k`'s window is the SAME local slice `[k, k+L)` on every chip, which makes the shift a
    single uniform on-device slice with no cross-chip ring rotation. See
    ``mtp_prefill/device_windows.py`` for the id algebra this mirrors.

    The rounding is not cosmetic: the socket rejects a page size that is not PCIe-aligned outright
    (`Page size must be PCIE-aligned`), and `L + K` generally is not one. At the production shape
    L=640, K=4 gives 644 ids = 2576 B, which is not a multiple of 64, so the row is padded to 656.
    Only the row WIDTH grows -- the trunk still reads `[0, L)` and level `k` still reads `[k, k+L)`,
    both far inside it, so the pad ids are never looked at. Use `h2d_lookahead` to get the padded
    overlap the producer must actually supply.
    """
    assert chunk_size % sp_factor == 0, f"chunk_size={chunk_size} must be divisible by sp_factor={sp_factor}"
    raw = chunk_size // sp_factor + lookahead
    if lookahead == 0:
        # Deliberately NOT rounded: every non-MTP caller gets exactly the width it always got. A
        # chunk_size/sp_factor that is not already page-aligned is a pre-existing condition of that
        # configuration, and silently widening its rows here would change the trunk's own shard.
        return raw
    ids_per_page = H2D_PAGE_ALIGNMENT_BYTES // _H2D_ID_BYTES
    return ((raw + ids_per_page - 1) // ids_per_page) * ids_per_page


def h2d_lookahead(chunk_size: int, sp_factor: int, mtp_levels: int) -> int:
    """Ids past this chip's own trunk shard that its H2D row carries: `mtp_levels`, rounded up to
    the page alignment by `h2d_row_len`.

    THE number every side of the transport must build to -- producer rows, runner socket spec, and
    the runtime's local warm-up input. Deriving all three from one function is the point: a producer
    that sends `L + K` while the socket is sized `L + pad` is a shape error at best and a silent
    misread at worst. 0 when MTP is off, so the non-MTP path is untouched.
    """
    return h2d_row_len(chunk_size, sp_factor, mtp_levels) - chunk_size // sp_factor


def make_global_spec(mesh_shape: tuple, chunk_size: int, lookahead: int = 0) -> ttnn.TensorSpec:
    """Per-push input spec used by `build_h2d_service` to set the service's
    global tensor shape (the producer matches it on the host side). One push carries one
    chunk_size-token chunk (+ `lookahead` overlapping tokens per chip, see `h2d_row_len`).
    Shape `(sp_factor, 1, h2d_row_len(chunk_size, sp_factor, lookahead))` uint32 ROW_MAJOR DRAM --
    NOT `chunk_size // sp_factor + lookahead`: a non-zero `lookahead` is rounded up to a PCIe page."""
    sp_factor = mesh_shape[0]
    return ttnn.TensorSpec(
        shape=ttnn.Shape([sp_factor, 1, h2d_row_len(chunk_size, sp_factor, lookahead)]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


def build_h2d_service(
    mesh_device: ttnn.MeshDevice,
    *,
    mesh_shape: tuple,
    chunk_size: int,
    mapper_config: ttnn.MeshMapperConfig,
    worker_cores: ttnn.CoreRange,
    metadata_size_bytes: int,
    lookahead: int = 0,
) -> ttnn.H2DStreamService:
    """Construct an H2DStreamService whose per-shard backing tensor matches
    what `prepare_prefill_input_tensor` would have produced. Each push carries one chunk_size-token
    chunk (chunked prefill streams one chunk per push), not the full sequence.

    Per-shard target: `(1, 1, h2d_row_len(chunk_size, sp_factor, lookahead))` uint32 ROW_MAJOR DRAM.
    Achieved by setting global_spec.shape = `(sp_factor, 1, h2d_row_len(...))` and
    mapping `[Shard(0), Replicate]` on a `(sp, tp)` mesh — first axis of the
    tensor is sharded across mesh rows (sp), nothing else is split.

    `lookahead` > 0 (MTP) widens every chip's row by K overlapping tokens; the producer must build
    the same overlapping rows. See `h2d_row_len`.
    """
    sp_factor, tp_factor = mesh_shape
    isl_per_chip = h2d_row_len(chunk_size, sp_factor, lookahead)
    per_chip_bytes = isl_per_chip * 4  # uint32

    global_spec = make_global_spec(mesh_shape, chunk_size, lookahead)
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
        f"[h2d] H2DStreamService built: global_shape=({sp_factor},1,{isl_per_chip}) "
        f"uint32 ROW_MAJOR DRAM, per_chip_bytes={per_chip_bytes}, worker_cores={worker_cores}"
    )
    return service


def activation_global_spec(chunk_size: int, hidden_size: int) -> ttnn.TensorSpec:
    """Global spec of the inter-rank hidden state carried over the D2D pipeline socket:
    [1, 1, chunk_size, hidden_size] bf16 TILE DRAM. The caller's mesh mapper shards it (seq across SP
    rows, emb across TP cols) to match the embedding output layout the downstream model consumes."""
    return ttnn.TensorSpec(
        shape=ttnn.Shape([1, 1, chunk_size, hidden_size]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


MTP_TOKEN_COLS = 32
"""Columns ONE MTP level's token ids occupy in the packed D2D activation, per TP shard. One tile
width, so every group boundary is tile-aligned and the unpack is a plain slice. The ids ride as
base-256 digits (bf16 holds integers exactly only to 256); the encoding lives with the model, in
``deepseek_v3_d_p/tt/mtp_prefill/device_windows.py``. Here only the WIDTH matters -- the runner
sizes the socket, it never looks inside."""


def mtp_token_block_cols(mtp_levels: int) -> int:
    """Per-TP-shard columns the ``mtp_levels`` token groups add to the D2D activation."""
    return MTP_TOKEN_COLS * mtp_levels if mtp_levels else 0


def d2d_activation_width(hidden_size: int, *, mtp_levels: int = 0, tp_factor: int = 1, dflash: bool = False) -> int:
    """GLOBAL width of the D2D pipeline activation for this configuration.

    Plain hidden is ``hidden_size``. DFlash packs the drafter FC partial beside it (2H). MTP packs
    ``mtp_levels`` token groups beside it -- a per-chip ``+32K``, and the socket's mapper shards the
    last dim across the ``tp_factor`` TP columns, so globally ``+32K*tp``.

    DFlash and MTP are different models and the runner asserts they are not both on, but the widths
    compose, so express that rather than branch on it.
    """
    width = hidden_size * (2 if dflash else 1)
    return width + mtp_token_block_cols(mtp_levels) * tp_factor


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
