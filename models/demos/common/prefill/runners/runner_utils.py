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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from loguru import logger

import ttnn
from models.demos.common.prefill.topology import assert_torus_xy_topology, per_axis_topology

_FABRIC_MODE_MAP = {
    "1d": ttnn.FabricConfig.FABRIC_1D,
    "2d": ttnn.FabricConfig.FABRIC_2D,
    "1d_ring": ttnn.FabricConfig.FABRIC_1D_RING,
    "2d_torus_x": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
    "2d_torus_y": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
    "2d_torus_xy": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
}

_FABRIC_MODE_DIM_TYPES = {
    "2d": ("LINE", "LINE"),
    "2d_torus_x": ("LINE", "RING"),
    "2d_torus_y": ("RING", "LINE"),
    "2d_torus_xy": ("RING", "RING"),
}

_FABRIC2D_SCOPED_MODELS = frozenset({"deepseek_v3_d_p", "deepseek_v32", "kimi_k2_6", "kimi_k2_7", "kimi_k3", "glm_5_2"})


@dataclass(frozen=True)
class MeshDescriptorTopology:
    """The topology-bearing fields needed to validate a mesh-graph descriptor before device open."""

    name: str
    dims: tuple[int, ...]
    dim_types: tuple[str, ...]


@dataclass(frozen=True)
class PrefillTopologyProfile:
    """Resolved, fail-closed fabric contract shared by prefill runner ranks."""

    name: str
    fabric_mode: str
    fabric_config: ttnn.FabricConfig
    reliability_mode: ttnn.FabricReliabilityMode
    per_axis_topology: tuple[ttnn.Topology, ttnn.Topology]
    descriptor_path: Path | None
    descriptors: tuple[MeshDescriptorTopology, ...]
    channel_policies: tuple[str, ...]
    production: bool


def _strip_textproto_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"(?m)(//|#).*?$", "", text)


def _extract_braced_blocks(text: str, field: str) -> list[str]:
    """Return balanced ``field { ... }`` bodies from a protobuf text-format document."""
    blocks = []
    pattern = re.compile(rf"\b{re.escape(field)}\s*\{{")
    cursor = 0
    while match := pattern.search(text, cursor):
        body_start = match.end()
        depth = 1
        index = body_start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth:
            raise ValueError(f"unterminated {field} block in mesh graph descriptor")
        blocks.append(text[body_start : index - 1])
        cursor = index
    return blocks


def parse_mesh_descriptor_topologies(path: str | Path) -> tuple[MeshDescriptorTopology, ...]:
    """Parse the dimensions and LINE/RING types from every mesh descriptor in ``path``.

    This intentionally parses only the small topology subset needed for pre-open safety. The C++
    control plane remains authoritative for the complete protobuf. Missing ``dim_types`` entries
    have protobuf's documented LINE default.
    """
    descriptor_path = Path(path).expanduser().resolve(strict=True)
    text = _strip_textproto_comments(descriptor_path.read_text())
    mesh_blocks = _extract_braced_blocks(text, "mesh_descriptors")
    if not mesh_blocks:
        raise ValueError(f"{descriptor_path} contains no mesh_descriptors blocks")

    parsed = []
    for index, mesh_block in enumerate(mesh_blocks):
        topology_blocks = _extract_braced_blocks(mesh_block, "device_topology")
        if len(topology_blocks) != 1:
            raise ValueError(f"{descriptor_path}: mesh_descriptors[{index}] must contain exactly one device_topology")
        topology = topology_blocks[0]
        dims_match = re.search(r"\bdims\s*:\s*\[([^]]*)\]", topology)
        if not dims_match:
            raise ValueError(f"{descriptor_path}: mesh_descriptors[{index}] has no device_topology.dims")
        dims = tuple(int(value) for value in re.findall(r"\d+", dims_match.group(1)))
        if not dims or any(dim <= 0 for dim in dims):
            raise ValueError(f"{descriptor_path}: mesh_descriptors[{index}] has invalid dims {dims}")

        types_match = re.search(r"\bdim_types\s*:\s*\[([^]]*)\]", topology)
        dim_types = tuple(re.findall(r"\b(?:LINE|RING)\b", types_match.group(1))) if types_match else ()
        if len(dim_types) > len(dims):
            raise ValueError(
                f"{descriptor_path}: mesh_descriptors[{index}] has {len(dim_types)} dim_types for {len(dims)} dims"
            )
        dim_types += ("LINE",) * (len(dims) - len(dim_types))

        name_match = re.search(r'\bname\s*:\s*"([^"]+)"', mesh_block)
        parsed.append(MeshDescriptorTopology(name_match.group(1) if name_match else f"mesh[{index}]", dims, dim_types))
    return tuple(parsed)


def parse_mesh_descriptor_channel_policies(path: str | Path) -> tuple[str, ...]:
    """Return every mesh/connection channel policy, applying protobuf's STRICT default."""
    descriptor_path = Path(path).expanduser().resolve(strict=True)
    text = _strip_textproto_comments(descriptor_path.read_text())
    channel_blocks = _extract_braced_blocks(text, "channels")
    if not channel_blocks:
        raise ValueError(f"{descriptor_path} contains no channels blocks")

    policies = []
    for index, block in enumerate(channel_blocks):
        matches = re.findall(r"\bpolicy\s*:\s*(STRICT|RELAXED)\b", block)
        if len(matches) > 1:
            raise ValueError(f"{descriptor_path}: channels[{index}] declares more than one policy")
        policies.append(matches[0] if matches else "STRICT")
    return tuple(policies)


def resolve_prefill_topology_profile(
    mesh_shape: tuple[int, int], model_name: str, environ: Mapping[str, str] | None = None
) -> PrefillTopologyProfile:
    """Resolve and validate the prefill fabric contract without opening hardware."""
    environ = os.environ if environ is None else environ
    fabric_mode = environ.get("PREFILL_FABRIC_MODE", "").strip().lower()
    if not fabric_mode:
        raise ValueError(
            "PREFILL_FABRIC_MODE is required; production must not infer fabric from SP size. "
            "Use 2d for an unwrapped LoudBox or 2d_torus_xy for production Galaxy."
        )
    if fabric_mode not in _FABRIC_MODE_MAP:
        raise ValueError(f"PREFILL_FABRIC_MODE must be one of {sorted(_FABRIC_MODE_MAP)}, got {fabric_mode!r}")
    if len(mesh_shape) != 2 or any(int(dim) <= 0 for dim in mesh_shape):
        raise ValueError(f"prefill requires a positive two-dimensional mesh shape, got {mesh_shape!r}")

    if model_name in _FABRIC2D_SCOPED_MODELS and fabric_mode not in {"2d", "2d_torus_xy"}:
        raise ValueError(
            f"{model_name} prefill permits only local Fabric2D ('2d') or production TorusXY "
            f"('2d_torus_xy'), got {fabric_mode!r}"
        )

    descriptor_raw = environ.get("TT_MESH_GRAPH_DESC_PATH", "").strip()
    descriptor_path = Path(descriptor_raw).expanduser().resolve() if descriptor_raw else None
    needs_descriptor = fabric_mode.startswith("2d_torus_")
    if needs_descriptor and descriptor_path is None:
        raise ValueError(f"PREFILL_FABRIC_MODE={fabric_mode} requires explicit TT_MESH_GRAPH_DESC_PATH")

    descriptors: tuple[MeshDescriptorTopology, ...] = ()
    channel_policies: tuple[str, ...] = ()
    expected_dim_types = _FABRIC_MODE_DIM_TYPES.get(fabric_mode)
    if descriptor_path is not None and expected_dim_types is not None:
        descriptors = parse_mesh_descriptor_topologies(descriptor_path)
        channel_policies = parse_mesh_descriptor_channel_policies(descriptor_path)
        mismatches = [descriptor for descriptor in descriptors if descriptor.dim_types != expected_dim_types]
        if mismatches:
            details = ", ".join(f"{d.name}:{d.dims}/{d.dim_types}" for d in mismatches)
            raise ValueError(
                f"PREFILL_FABRIC_MODE={fabric_mode} requires descriptor dim_types={expected_dim_types}; "
                f"{descriptor_path} has incompatible meshes: {details}"
            )
        dimension_mismatches = [descriptor for descriptor in descriptors if descriptor.dims != tuple(mesh_shape)]
        if dimension_mismatches:
            details = ", ".join(f"{d.name}:{d.dims}" for d in dimension_mismatches)
            raise ValueError(
                f"PREFILL_FABRIC_MODE={fabric_mode} mesh_shape={tuple(mesh_shape)} does not match descriptor "
                f"{descriptor_path}: {details}"
            )

    profile_name = {
        "2d": "bh_loudbox_fabric2d",
        "2d_torus_xy": "bh_galaxy_torus_xy",
    }.get(fabric_mode, f"compat_{fabric_mode}")
    return PrefillTopologyProfile(
        name=profile_name,
        fabric_mode=fabric_mode,
        fabric_config=_FABRIC_MODE_MAP[fabric_mode],
        reliability_mode=(
            ttnn.FabricReliabilityMode.STRICT_INIT
            if channel_policies and all(policy == "STRICT" for policy in channel_policies)
            else ttnn.FabricReliabilityMode.RELAXED_INIT
        ),
        per_axis_topology=per_axis_topology(_FABRIC_MODE_MAP[fabric_mode]),
        descriptor_path=descriptor_path,
        descriptors=descriptors,
        channel_policies=channel_policies,
        production=model_name in _FABRIC2D_SCOPED_MODELS and fabric_mode == "2d_torus_xy",
    )


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
    mesh_shape: tuple,
    model_cfg: type,
    l1_small_size: int = 0,
    trace_region_size: int = 0,
    *,
    model_name: str | None = None,
) -> ttnn.MeshDevice:
    """Validate the explicit topology contract, configure fabric, and open the mesh device.

    ``PREFILL_FABRIC_MODE`` is mandatory: the runner never infers fabric from SP size. Scoped Kimi
    K2.6/K2.7/K3 and GLM-5.2 runs accept unwrapped Fabric2D for local LoudBox validation and
    Fabric2D TorusXY for production. Torus modes require a matching explicit mesh descriptor before
    any device is opened.

    `l1_small_size` > 0 carves an L1_SMALL region (needed when an op routes its
    semaphores there, e.g. the Kimi MoE routing all-gather with use_l1_small_for_semaphores).

    `trace_region_size` > 0 reserves device DRAM for ttnn trace capture — needed when the runtime
    replays a captured forward (TtPrefillRuntime use_trace). 0 = no trace region (default)."""
    model_name = (model_name or os.environ.get("PREFILL_MODEL", "")).strip()
    if not model_name:
        raise ValueError("PREFILL_MODEL is required before opening a prefill mesh; no default model is permitted")
    profile = resolve_prefill_topology_profile(mesh_shape, model_name)
    descriptor_summary = [f"{d.name}:{d.dims}/{d.dim_types}" for d in profile.descriptors]
    logger.info(
        f"Prefill topology profile: name={profile.name} model={model_name} mesh={tuple(mesh_shape)} "
        f"fabric={profile.fabric_config} mode={profile.fabric_mode} topology={profile.per_axis_topology} "
        f"reliability={profile.reliability_mode} descriptor={profile.descriptor_path} "
        f"descriptor_meshes={descriptor_summary or '<auto/local>'} "
        f"channel_policies={profile.channel_policies or '<auto/local>'}"
    )

    fabric_router_config = _create_fabric_router_config(
        max_payload_size=model_cfg.FABRIC_PAYLOAD_SIZE,
    )

    ttnn.set_fabric_config(
        profile.fabric_config,
        profile.reliability_mode,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        fabric_router_config,
    )
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*mesh_shape), l1_small_size=l1_small_size, trace_region_size=trace_region_size
    )
    active_fabric = ttnn.get_fabric_config()
    if active_fabric != profile.fabric_config:
        ttnn.close_mesh_device(mesh_device)
        raise RuntimeError(
            f"active fabric {active_fabric} does not match resolved prefill profile {profile.fabric_config}"
        )
    if profile.production:
        try:
            assert_torus_xy_topology(active_fabric)
        except Exception:
            ttnn.close_mesh_device(mesh_device)
            raise
    logger.info(
        f"Active prefill fabric verified: {active_fabric}; "
        f"SP/TP topology={profile.per_axis_topology}; profile={profile.name}"
    )
    return mesh_device


def make_global_spec(mesh_shape: tuple, chunk_size: int) -> ttnn.TensorSpec:
    """Per-push input spec used by `build_h2d_service` to set the service's
    global tensor shape (the producer matches it on the host side). One push carries one
    chunk_size-token chunk. Shape `(sp_factor, 1, chunk_size // sp_factor)` uint32 ROW_MAJOR DRAM."""
    sp_factor = mesh_shape[0]
    isl_per_chip = chunk_size // sp_factor
    return ttnn.TensorSpec(
        shape=ttnn.Shape([sp_factor, 1, isl_per_chip]),
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
) -> ttnn.H2DStreamService:
    """Construct an H2DStreamService whose per-shard backing tensor matches
    what `prepare_prefill_input_tensor` would have produced. Each push carries one chunk_size-token
    chunk (chunked prefill streams one chunk per push), not the full sequence.

    Per-shard target: `(1, 1, chunk_size // sp_factor)` uint32 ROW_MAJOR DRAM.
    Achieved by setting global_spec.shape = `(sp_factor, 1, chunk_size // sp_factor)` and
    mapping `[Shard(0), Replicate]` on a `(sp, tp)` mesh — first axis of the
    tensor is sharded across mesh rows (sp), nothing else is split.
    """
    sp_factor, tp_factor = mesh_shape
    assert chunk_size % sp_factor == 0, f"chunk_size={chunk_size} must be divisible by sp_factor={sp_factor}"
    isl_per_chip = chunk_size // sp_factor
    per_chip_bytes = isl_per_chip * 4  # uint32

    global_spec = make_global_spec(mesh_shape, chunk_size)
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


def compute_layer_split(num_layers: int, num_ranks: int, valid_starts=None) -> list[tuple[int, int]]:
    """Contiguous (first_layer_idx, count) per rank. PREFILL_PP_LAYER_COUNTS, a
    comma-separated count list summing to num_layers, overrides the default even
    split (remainder handed to the earlier ranks).

    ``valid_starts`` (from the adapter's ``layer_split_boundaries``): layer indices at which a rank may
    begin. None => unconstrained. When set, the default even split is auto-snapped onto valid
    boundaries, and any split (explicit or snapped) whose rank starts fall off them is rejected early."""
    override = os.environ.get("PREFILL_PP_LAYER_COUNTS")
    if override:
        counts = [int(x) for x in override.split(",")]
        if len(counts) != num_ranks or sum(counts) != num_layers:
            raise ValueError(
                f"PREFILL_PP_LAYER_COUNTS={override!r} must list {num_ranks} counts summing to "
                f"{num_layers} (got {len(counts)} counts summing to {sum(counts)})"
            )
    else:
        base, rem = divmod(num_layers, num_ranks)
        counts = [base + (1 if r < rem else 0) for r in range(num_ranks)]
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
