# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint for the HostIoDecoderStage multi-turn sweep harness.

Thin argparse wrapper around :func:`host_io_decoder_harness.run_sweep`. Every
knob in :class:`HostIoDecoderSweepConfig` is exposed as a flag; boolean knobs
use ``argparse.BooleanOptionalAction`` so each is toggleable with
``--knob`` / ``--no-knob``.

The same entrypoint supports one local decoder-layer sweep and rank-parallel
layer verification under ``tt-run`` /
``mpirun`` / ``srun``. When multiple ``--decoder-layer-indices`` values are supplied
rank ``i`` verifies the i-th layer id, and the launcher world size must match
the number of layer ids.

Environment variables (read only as fallback defaults; flags always win):
    DEEPSEEK_V3_TRACE_ROOT          -> --trace-root
    DEEPSEEK_V3_KV_CACHE_DUMP_DIR   -> --dump-dir

Slow dispatch is required for this sweep to function. Set this in
the calling environment before invoking::

    TT_METAL_SLOW_DISPATCH_MODE=1 python -m models.demos.deepseek_v3_b1.tests.unit_tests.run_host_io_decoder_sweep ...

Usage examples
--------------

Mode A (single slot), one prompt, dump everything to ``./dumps``::

    TT_METAL_SLOW_DISPATCH_MODE=1 \\
    python -m models.demos.deepseek_v3_b1.tests.unit_tests.run_host_io_decoder_sweep \\
        --decoder-layer-indices 4 \\
        --trace-root /mnt/models/ref_data/debug_traces \\
        --model-id DeepSeek_R1_0528 \\
        --prompt vllm-97854ebb-128000tok \\
        --num-decode-steps 1024 \\
        --num-replication-slots 1 \\
        --dump-dir ./dumps

Mode B (8 replication slots), validate metadata and hidden states from a
chunked trace, no dumps::

    TT_METAL_SLOW_DISPATCH_MODE=1 \\
    python -m models.demos.deepseek_v3_b1.tests.unit_tests.run_host_io_decoder_sweep \\
        --decoder-layer-indices 4 \\
        --trace-root /mnt/models/ref_data/debug_traces \\
        --model-id DeepSeek_R1_0528 \\
        --prompt vllm-97854ebb-128000tok \\
        --num-replication-slots 8 \\
        --validate-hidden-states-cross-trace \\
        --pcc-threshold 0.97 \\
        --no-dump-hidden-states --no-dump-kv-cache

Note: ``--validate-kv-cache-cross-trace`` is not implemented for chunked trace
input yet; keep it disabled.

Dry-run (print resolved config and exit, no device opened)::

    python -m models.demos.deepseek_v3_b1.tests.unit_tests.run_host_io_decoder_sweep \\
        --decoder-layer-indices 4 \\
        --trace-root /mnt/models/ref_data/debug_traces \\
        --model-id DeepSeek_R1_0528 \\
        --prompt vllm-97854ebb-128000tok \\
        --dry-run

Four-layer rank-parallel verification, one rank per host::

    export TRACE_ROOT=/mnt/models/ref_data/debug_traces
    export PROMPT=vllm-97854ebb-128000tok

    TT_METAL_SLOW_DISPATCH_MODE=1 python -m ttnn.distributed.ttrun \\
      --tcp-interface ens5f0np0 \\
      --rank-bindings-mapping decoder_verify_4x_rank_bindings_mapping.yaml \\
      --mpi-args "--map-by rankfile:file=decoder_verify_4x_rank_file_current --bind-to none --tag-output" \\
      python -m models.demos.deepseek_v3_b1.tests.unit_tests.run_host_io_decoder_sweep \\
        --decoder-layer-indices 4 5 6 7 \\
        --trace-root "${TRACE_ROOT}" \\
        --model-id DeepSeek_R1_0528 \\
        --prompt "${PROMPT}" \\
        --validate-hidden-states-cross-trace \\
        --pcc-threshold 0.97

"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.tests.unit_tests.ccl_test_utils import create_fabric_router_config
from models.demos.deepseek_v3_b1.tests.unit_tests.host_io_decoder_harness import (
    DEFAULT_CACHE_PATH,
    DEFAULT_HF_MODEL_PATH,
    KV_CACHE_DUMP_DIR_ENV,
    TRACE_ROOT_ENV,
    HostIoDecoderSweepConfig,
    open_mesh_device,
    run_sweep,
)

# Production device_params constants ported from the original pytest's indirect
# ``device_params`` fixture. The fabric config, max packet payload, and worker
# L1 size are not currently exposed as CLI knobs because no caller has needed
# to vary them; promote any of these to flags later if that changes.
_FABRIC_CONFIG = ttnn.FabricConfig.FABRIC_2D_TORUS_X
_LAYER_PARALLEL_FABRIC_CONFIG = ttnn.FabricConfig.FABRIC_2D_TORUS_Y
_FABRIC_ROUTER_MAX_PAYLOAD_BYTES = 15232
_WORKER_L1_SIZE = 1431568


# ---------------------------------------------------------------------------
# Argparse setup
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multi-turn HostIoDecoderStage sweep against one or more reference "
            "hidden-state traces. Validates metadata round-trip, cross-slot determinism "
            "for hidden states and KV cache, and (optionally) cross-trace PCC; dumps "
            "per-(slot, prompt) hidden states and KV-cache slices to disk."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- required: prompt source + identifying knobs ---
    required = parser.add_argument_group("required")
    required.add_argument(
        "--decoder-layer-indices",
        dest="decoder_layer_indices",
        type=int,
        nargs="+",
        required=True,
        metavar="LAYER",
        help=(
            "DeepSeek V3 decoder layer index or indices to instantiate. A single value runs the normal "
            "local sweep; multiple values require one launcher rank per layer."
        ),
    )
    required.add_argument(
        "--trace-root",
        type=Path,
        default=None,
        help=(
            f"Chunked trace registry root, e.g. /mnt/models/ref_data/debug_traces. Falls back to "
            f"${TRACE_ROOT_ENV} if omitted; required if neither this flag nor the env var is set."
        ),
    )
    required.add_argument(
        "--model-id",
        default="DeepSeek_R1_0528",
        help="Trace-registry model id resolved by debug_trace_io.py, e.g. DeepSeek_R1_0528.",
    )
    required.add_argument(
        "--prompt",
        dest="prompt_names",
        nargs="+",
        required=True,
        metavar="NAME",
        help=(
            "Prompt id under the resolved model trace directory, e.g. vllm-97854ebb-128000tok. "
            "Chunked trace input currently supports exactly one prompt id."
        ),
    )
    required.add_argument(
        "--num-decode-steps",
        type=int,
        default=None,
        help=(
            "Optional decode-step limit. If omitted, the full chunked trace is used. When set, the loader "
            "converts this to the Group A row slice using trace metadata."
        ),
    )

    # --- model shape ---
    model = parser.add_argument_group("model shape")
    model.add_argument("--max-seq-len", type=int, default=128 * 1024)
    model.add_argument("--num-slots", type=int, default=64)
    model.add_argument("--mesh-rows", type=int, default=4)
    model.add_argument("--mesh-cols", type=int, default=2)

    # --- replication / mode ---
    parser.add_argument(
        "--num-replication-slots",
        type=int,
        default=None,
        help=(
            "Replication slots (1 = Mode A; >1 = Mode B); must be < --num-slots. Defaults to 8 for a "
            "single local layer and 1 for rank-parallel layer verification."
        ),
    )

    # --- validation knobs ---
    val = parser.add_argument_group("validation")
    val.add_argument(
        "--validate-metadata-roundtrip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assert per-iteration that slot_id / position_id / token_id / token_0_type round-trip through D2H.",
    )
    val.add_argument(
        "--validate-hidden-states-cross-slot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Mode B only: per-prompt torch.equal of every replicated slot's collected output against slot 0. "
            "Defaults to on for a single local layer and off for rank-parallel layer verification."
        ),
    )
    val.add_argument(
        "--validate-kv-cache-cross-slot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Mode B only: per-prompt torch.equal of every replicated slot's KV-cache slice against slot 0. "
            "Defaults to on for a single local layer and off for rank-parallel layer verification."
        ),
    )
    val.add_argument(
        "--validate-hidden-states-cross-trace",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Per-(prompt, slot) PCC of collected output against the prompt's reference 'output' trace. "
            "Defaults to off for a single local layer and on for rank-parallel verification."
        ),
    )
    val.add_argument(
        "--pcc-threshold",
        type=float,
        default=0.97,
        help="PCC threshold used only when --validate-hidden-states-cross-trace is set.",
    )
    val.add_argument(
        "--validate-kv-cache-cross-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Per-(prompt, slot) PCC of the on-device KV-cache slice against the chunked trace KV reference. "
            "Forces the host KV-cache pull even when no KV-cache dump is requested."
        ),
    )
    val.add_argument(
        "--kv-cache-pcc-threshold",
        type=float,
        default=0.97,
        help="PCC threshold used only when --validate-kv-cache-cross-trace is set.",
    )

    # --- dump knobs ---
    dump = parser.add_argument_group("dumps")
    dump.add_argument(
        "--dump-hidden-states",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Write one (seq_len, HIDDEN_SIZE) bf16 file per (slot, prompt) to --dump-dir. "
            "Opt-in (default off) so validation-only runs don't require --dump-dir."
        ),
    )
    dump.add_argument(
        "--dump-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Write one (1, seq_len, kvpe_dim) bf16 file per (slot, prompt) to --dump-dir. "
            "Opt-in (default off) so the harness can skip the ~9.6 GB device-to-host KV "
            "cache pull entirely on validation-only / pure-sweep runs."
        ),
    )
    dump.add_argument(
        "--dump-dir",
        type=Path,
        default=None,
        help=(
            f"Output directory for dumps. Falls back to ${KV_CACHE_DUMP_DIR_ENV} if omitted; required if any "
            f"dump knob is on. May contain {{layer}}, {{layer_idx}}, and/or {{rank}} format fields; in "
            f"rank-parallel mode, a plain directory gets per-layer subdirectories."
        ),
    )

    # --- weights ---
    weights = parser.add_argument_group("weights")
    weights.add_argument(
        "--hf-model-path",
        type=Path,
        default=DEFAULT_HF_MODEL_PATH,
        help="HuggingFace model checkpoint root (must contain config.json + safetensors).",
    )
    weights.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="CacheWeightProvider persistent cache directory.",
    )

    # --- misc ---
    misc = parser.add_argument_group("misc")
    misc.add_argument("--seed", type=int, default=0)
    misc.add_argument(
        "--log-per-iteration",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, log every inner sweep iteration; otherwise log every 1000 outer positions per prompt.",
    )
    misc.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config and exit before opening any device.",
    )

    return parser


# ---------------------------------------------------------------------------
# Config + device_params resolution
# ---------------------------------------------------------------------------


def _layer_ids_from_args(args: argparse.Namespace) -> list[int]:
    """Return decoder layer ids as a list."""
    if isinstance(args.decoder_layer_indices, int):
        return [args.decoder_layer_indices]
    return list(args.decoder_layer_indices)


def _format_path(raw: str | Path, *, layer_idx: int, rank: int) -> Path:
    text = str(raw)
    if "{" not in text:
        return Path(text)
    try:
        return Path(text.format(layer=layer_idx, layer_idx=layer_idx, rank=rank))
    except KeyError as e:
        raise ValueError(
            f"Unsupported path format field {e!s}; supported fields are {{layer}}, {{layer_idx}}, and {{rank}}"
        ) from e


def _resolve_trace_root(args: argparse.Namespace) -> Path:
    """Resolve the chunked trace registry root."""
    if args.trace_root is not None:
        return Path(args.trace_root)
    raw = os.environ.get(TRACE_ROOT_ENV)
    if not raw:
        raise ValueError(f"--trace-root is required (or set ${TRACE_ROOT_ENV})")
    return Path(raw)


def _validate_num_decode_steps(num_decode_steps: int | None) -> int | None:
    if num_decode_steps is not None and num_decode_steps <= 0:
        raise ValueError(f"--num-decode-steps must be > 0 when provided, got {num_decode_steps}")
    return num_decode_steps


def _resolve_dump_dir(
    args: argparse.Namespace,
    *,
    layer_idx: int,
    rank: int,
    layer_parallel: bool,
) -> Path | None:
    """Resolve dump dir, adding layer subdirs only for plain paths in parallel mode."""
    raw = args.dump_dir if args.dump_dir is not None else os.environ.get(KV_CACHE_DUMP_DIR_ENV)
    if raw is None:
        return None

    resolved = _format_path(raw, layer_idx=layer_idx, rank=rank)
    if "{" in str(raw) or not layer_parallel:
        return resolved
    return resolved / f"layer_{layer_idx:02d}"


def _config_from_args_for_rank(
    args: argparse.Namespace,
    *,
    rank: int,
) -> HostIoDecoderSweepConfig:
    """Build a frozen config for either local single-layer or rank-selected layer mode."""
    layer_ids = _layer_ids_from_args(args)
    layer_parallel = len(layer_ids) > 1
    layer_idx = _layer_for_rank(layer_ids, rank=rank)

    num_replication_slots = args.num_replication_slots
    if num_replication_slots is None:
        num_replication_slots = 1 if layer_parallel else 8

    validate_hidden_states_cross_slot = args.validate_hidden_states_cross_slot
    if validate_hidden_states_cross_slot is None:
        validate_hidden_states_cross_slot = not layer_parallel

    validate_kv_cache_cross_slot = args.validate_kv_cache_cross_slot
    if validate_kv_cache_cross_slot is None:
        validate_kv_cache_cross_slot = not layer_parallel

    validate_hidden_states_cross_trace = args.validate_hidden_states_cross_trace
    if validate_hidden_states_cross_trace is None:
        validate_hidden_states_cross_trace = layer_parallel

    return HostIoDecoderSweepConfig(
        decoder_layer_index=layer_idx,
        trace_root=_resolve_trace_root(args),
        model_id=args.model_id,
        prompt_names=tuple(args.prompt_names),
        num_decode_steps=_validate_num_decode_steps(args.num_decode_steps),
        max_seq_len=args.max_seq_len,
        num_slots=args.num_slots,
        mesh_rows=args.mesh_rows,
        mesh_cols=args.mesh_cols,
        num_replication_slots=num_replication_slots,
        validate_metadata_roundtrip=args.validate_metadata_roundtrip,
        validate_hidden_states_cross_slot=validate_hidden_states_cross_slot,
        validate_kv_cache_cross_slot=validate_kv_cache_cross_slot,
        validate_hidden_states_cross_trace=validate_hidden_states_cross_trace,
        pcc_threshold=args.pcc_threshold,
        validate_kv_cache_cross_trace=args.validate_kv_cache_cross_trace,
        kv_cache_pcc_threshold=args.kv_cache_pcc_threshold,
        dump_hidden_states=args.dump_hidden_states,
        dump_kv_cache=args.dump_kv_cache,
        dump_dir=_resolve_dump_dir(args, layer_idx=layer_idx, rank=rank, layer_parallel=layer_parallel),
        hf_model_path=args.hf_model_path,
        cache_path=args.cache_path,
        seed=args.seed,
        log_per_iteration=args.log_per_iteration,
    )


def _config_from_args(args: argparse.Namespace) -> HostIoDecoderSweepConfig:
    """Build a single local-layer config; retained for callers/tests that import this helper."""
    layer_ids = _layer_ids_from_args(args)
    if len(layer_ids) != 1:
        raise ValueError("_config_from_args expects exactly one --decoder-layer-indices value")
    return _config_from_args_for_rank(args, rank=0)


def _build_device_params(config: HostIoDecoderSweepConfig, *, layer_parallel: bool = False) -> dict:
    """Return the device_params dict consumed by :func:`open_mesh_device`.

    Production fabric / worker-l1 constants ported from the original pytest's
    indirect ``device_params`` fixture. These are intentionally not CLI flags
    today; promote if a caller needs to vary them.
    """
    return {
        "fabric_config": _LAYER_PARALLEL_FABRIC_CONFIG if layer_parallel else _FABRIC_CONFIG,
        "fabric_router_config": create_fabric_router_config(_FABRIC_ROUTER_MAX_PAYLOAD_BYTES),
        "worker_l1_size": _WORKER_L1_SIZE,
    }


def _init_rank_context_or_error() -> tuple[int, int]:
    """Return launcher rank/size without initializing TTNN distributed context.

    Rank-parallel verification runs independent single-stage decoder sweeps. If
    we initialize TTNN distributed context here, PipelineBlock sees the launcher
    world as a multi-stage pipeline and selects the embedding first-stage path
    instead of the single-stage injected-hidden-states path.
    """

    rank = os.environ.get("OMPI_COMM_WORLD_RANK")
    world_size = os.environ.get("OMPI_COMM_WORLD_SIZE")
    return int(rank), int(world_size)


def _validate_layer_world_size(layer_ids: list[int], *, world_size: int) -> None:
    if len(layer_ids) != world_size:
        raise ValueError(
            f"Number of --decoder-layer-indices values ({len(layer_ids)}) must match launcher world size "
            f"({world_size}). Launch {len(layer_ids)} ranks or pass exactly {world_size} layers."
        )


def _layer_for_rank(layer_ids: list[int], *, rank: int, world_size: int | None = None) -> int:
    if world_size is None:
        world_size = len(layer_ids)
    _validate_layer_world_size(layer_ids, world_size=world_size)
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Rank {rank} is outside launcher world size {world_size}.")
    return layer_ids[rank]


def _log_resolved_config(config: HostIoDecoderSweepConfig) -> None:
    """Render the resolved config in a human-readable block, before the run."""
    logger.info("Resolved HostIoDecoderSweepConfig:")
    for field, value in (
        ("decoder_layer_index", config.decoder_layer_index),
        ("trace_root", config.trace_root),
        ("model_id", config.model_id),
        ("prompt_names", list(config.prompt_names)),
        ("num_decode_steps", config.num_decode_steps),
        ("max_seq_len", config.max_seq_len),
        ("num_slots", config.num_slots),
        ("mesh_rows x mesh_cols", f"{config.mesh_rows} x {config.mesh_cols}"),
        ("num_replication_slots", config.num_replication_slots),
        ("validate_metadata_roundtrip", config.validate_metadata_roundtrip),
        ("validate_hidden_states_cross_slot", config.validate_hidden_states_cross_slot),
        ("validate_kv_cache_cross_slot", config.validate_kv_cache_cross_slot),
        ("validate_hidden_states_cross_trace", config.validate_hidden_states_cross_trace),
        ("pcc_threshold", config.pcc_threshold),
        ("validate_kv_cache_cross_trace", config.validate_kv_cache_cross_trace),
        ("kv_cache_pcc_threshold", config.kv_cache_pcc_threshold),
        ("dump_hidden_states", config.dump_hidden_states),
        ("dump_kv_cache", config.dump_kv_cache),
        ("dump_dir", config.dump_dir),
        ("hf_model_path", config.hf_model_path),
        ("cache_path", config.cache_path),
        ("seed", config.seed),
        ("log_per_iteration", config.log_per_iteration),
    ):
        logger.info(f"  {field:38s} = {value}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    layer_ids = _layer_ids_from_args(args)
    multi_layer_requested = len(layer_ids) > 1
    rank: int | None = None
    world_size: int | None = None
    layer_parallel = False

    try:
        if multi_layer_requested:
            rank, world_size = _init_rank_context_or_error()
            _validate_layer_world_size(layer_ids, world_size=world_size)
            layer_parallel = True
            config = _config_from_args_for_rank(
                args,
                rank=rank,
            )
        else:
            config = _config_from_args_for_rank(
                args,
                rank=0,
            )
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        parser.error(str(e))
        return 2  # pragma: no cover (parser.error sys.exits 2)

    if rank is not None and world_size is not None:
        logger.info(f"rank={rank}/{world_size}: verifying decoder layer {config.decoder_layer_index}")
    _log_resolved_config(config)

    if args.dry_run:
        logger.info("--dry-run set; exiting before opening device")
        return 0

    device_params = _build_device_params(config, layer_parallel=layer_parallel)
    with open_mesh_device(device_params) as parent_mesh:
        result = run_sweep(config, parent_mesh)

    # Result summary so the CLI ends with an audit-friendly digest of what ran.
    logger.info("=" * 80)
    if rank is not None and world_size is not None:
        logger.info(f"SUMMARY rank={rank}/{world_size} layer={config.decoder_layer_index}")
    else:
        logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"prompts run: {list(result.collected.keys())}")
    logger.info(f"prompt_lengths: {result.schedule.prompt_lengths}")
    logger.info(f"total positions: {result.schedule.total_length()}")
    if result.kv_cache is not None:
        logger.info(f"kv_cache shape: {tuple(result.kv_cache.shape)} dtype={result.kv_cache.dtype}")
    else:
        logger.info("kv_cache: not pulled (no KV-cache validation, no KV-cache dump)")
    # Per-validation-gate summary so the CLI tail makes it obvious which gates
    # actually ran and (where applicable) what threshold they used.
    hidden_states_cross_slot_status = (
        "disabled"
        if not config.validate_hidden_states_cross_slot
        else ("enabled" if config.num_replication_slots > 1 else "skipped/no-op (num_replication_slots=1)")
    )
    hidden_states_cross_trace_status = (
        f"enabled (threshold={config.pcc_threshold})" if config.validate_hidden_states_cross_trace else "disabled"
    )
    kv_cache_cross_slot_status = (
        "disabled"
        if not config.validate_kv_cache_cross_slot
        else ("enabled" if config.num_replication_slots > 1 else "skipped/no-op (num_replication_slots=1)")
    )
    kv_cache_cross_trace_status = (
        f"enabled (threshold={config.kv_cache_pcc_threshold})" if config.validate_kv_cache_cross_trace else "disabled"
    )
    logger.info(
        f"validation gates: "
        f"metadata_roundtrip={'enabled' if config.validate_metadata_roundtrip else 'disabled'}, "
        f"hidden_states_cross_slot={hidden_states_cross_slot_status}, "
        f"hidden_states_cross_trace={hidden_states_cross_trace_status}, "
        f"kv_cache_cross_slot={kv_cache_cross_slot_status}, "
        f"kv_cache_cross_trace={kv_cache_cross_trace_status}"
    )
    if result.kv_cache_references is not None:
        logger.info(
            f"kv_cache_references loaded for {len(result.kv_cache_references)} prompt(s): "
            f"{sorted(result.kv_cache_references.keys())}"
        )
    if config.dump_dir is not None and (config.dump_hidden_states or config.dump_kv_cache):
        logger.info(f"dumps written under: {config.dump_dir}")
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
