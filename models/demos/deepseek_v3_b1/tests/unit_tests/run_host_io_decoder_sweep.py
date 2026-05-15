# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint for the HostIoDecoderStage multi-turn sweep harness.

Thin argparse wrapper around :func:`host_io_decoder_harness.run_sweep`. Every
knob in :class:`HostIoDecoderSweepConfig` is exposed as a flag; boolean knobs
use ``argparse.BooleanOptionalAction`` so each is toggleable with
``--knob`` / ``--no-knob``.

Environment variables (read only as fallback defaults; flags always win):
    DEEPSEEK_V3_HIDDEN_STATES_DIR   -> --hidden-states-dir
    DEEPSEEK_V3_KV_CACHE_DUMP_DIR   -> --dump-dir

Slow dispatch is required for this sweep to function. Set this in
the calling environment before invoking::

    TT_METAL_SLOW_DISPATCH_MODE=1 python -m models.demos.deepseek_v3_b1.tests.unit_tests.run_host_io_decoder_sweep ...

Usage examples
--------------

Mode A (single slot), one prompt, dump everything to ``./dumps``::

    TT_METAL_SLOW_DISPATCH_MODE=1 \\
    python -m models.demos.deepseek_v3_b1.tests.unit_tests.run_host_io_decoder_sweep \\
        --decoder-layer-idx 4 \\
        --hidden-states-dir /data/asaigal/pipeclean_traces \\
        --prompt pipeclean_seq_8192 \\
        --num-replication-slots 1 \\
        --dump-dir ./dumps

Mode B (8 replication slots), two-prompt multi-turn, validate everything,
no dumps, use real GPU reference traces for cross-trace PCC::

    TT_METAL_SLOW_DISPATCH_MODE=1 \\
    DEEPSEEK_V3_HIDDEN_STATES_DIR=/data/gpu_reference \\
    python -m models.demos.deepseek_v3_b1.tests.unit_tests.run_host_io_decoder_sweep \\
        --decoder-layer-idx 4 \\
        --prompt q_what_is_python q_quick_brown_fox \\
        --num-replication-slots 8 \\
        --validate-hidden-states-cross-trace \\
        --no-dump-hidden-states --no-dump-kv-cache

Dry-run (print resolved config and exit, no device opened)::

    python -m models.demos.deepseek_v3_b1.tests.unit_tests.run_host_io_decoder_sweep \\
        --decoder-layer-idx 4 \\
        --hidden-states-dir /tmp/x --prompt foo --dry-run
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
    HIDDEN_STATES_DIR_ENV,
    KV_CACHE_DUMP_DIR_ENV,
    HostIoDecoderSweepConfig,
    open_mesh_device,
    run_sweep,
)

# Production device_params constants ported from the original pytest's indirect
# ``device_params`` fixture. The fabric config, max packet payload, and worker
# L1 size are not currently exposed as CLI knobs because no caller has needed
# to vary them; promote any of these to flags later if that changes.
_FABRIC_CONFIG = ttnn.FabricConfig.FABRIC_2D_TORUS_X
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
        "--decoder-layer-idx",
        type=int,
        required=True,
        help="DeepSeek V3 decoder layer index to instantiate (e.g. 4 for the first MoE layer).",
    )
    required.add_argument(
        "--hidden-states-dir",
        type=Path,
        default=None,
        help=(
            f"Directory containing per-prompt reference traces (*.pt files with bf16 "
            f"{{'input', 'output'}} tensors). Falls back to ${HIDDEN_STATES_DIR_ENV} if omitted; "
            f"required if neither flag nor env is set."
        ),
    )
    required.add_argument(
        "--prompt",
        dest="prompt_names",
        nargs="+",
        required=True,
        metavar="NAME",
        help=(
            "One or more prompt names (file stems under --hidden-states-dir). Multiple "
            "prompts are run back-to-back in a single decoder launch; positions are "
            "disjoint and accumulating across prompts (multi-turn semantics)."
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
        default=8,
        help="Replication slots (1 = Mode A; >1 = Mode B); must be < --num-slots.",
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
        default=True,
        help="Mode B only: per-prompt torch.equal of every replicated slot's collected output against slot 0.",
    )
    val.add_argument(
        "--validate-kv-cache-cross-slot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mode B only: per-prompt torch.equal of every replicated slot's KV-cache slice against slot 0.",
    )
    val.add_argument(
        "--validate-hidden-states-cross-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Per-(prompt, slot) PCC of collected output against the prompt's reference 'output' trace.",
    )
    val.add_argument(
        "--pcc-threshold",
        type=float,
        default=0.97,
        help="PCC threshold used only when --validate-hidden-states-cross-trace is set.",
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
            f"Output directory for dumps. Falls back to ${KV_CACHE_DUMP_DIR_ENV} if omitted; "
            f"required if any dump knob is on."
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


def _resolve_hidden_states_dir(args: argparse.Namespace) -> Path:
    """Flag wins; env is fallback; error if neither set."""
    if args.hidden_states_dir is not None:
        return args.hidden_states_dir
    raw = os.environ.get(HIDDEN_STATES_DIR_ENV)
    if raw:
        return Path(raw)
    raise ValueError(f"--hidden-states-dir is required (or set ${HIDDEN_STATES_DIR_ENV})")


def _resolve_dump_dir(args: argparse.Namespace) -> Path | None:
    """Flag wins; env is fallback; None permitted iff both dump knobs are off."""
    if args.dump_dir is not None:
        return args.dump_dir
    raw = os.environ.get(KV_CACHE_DUMP_DIR_ENV)
    if raw:
        return Path(raw)
    return None


def _config_from_args(args: argparse.Namespace) -> HostIoDecoderSweepConfig:
    """Build a frozen :class:`HostIoDecoderSweepConfig` from parsed argparse args."""
    hidden_states_dir = _resolve_hidden_states_dir(args)
    dump_dir = _resolve_dump_dir(args)
    return HostIoDecoderSweepConfig(
        decoder_layer_idx=args.decoder_layer_idx,
        hidden_states_dir=hidden_states_dir,
        prompt_names=tuple(args.prompt_names),
        max_seq_len=args.max_seq_len,
        num_slots=args.num_slots,
        mesh_rows=args.mesh_rows,
        mesh_cols=args.mesh_cols,
        num_replication_slots=args.num_replication_slots,
        validate_metadata_roundtrip=args.validate_metadata_roundtrip,
        validate_hidden_states_cross_slot=args.validate_hidden_states_cross_slot,
        validate_kv_cache_cross_slot=args.validate_kv_cache_cross_slot,
        validate_hidden_states_cross_trace=args.validate_hidden_states_cross_trace,
        pcc_threshold=args.pcc_threshold,
        dump_hidden_states=args.dump_hidden_states,
        dump_kv_cache=args.dump_kv_cache,
        dump_dir=dump_dir,
        hf_model_path=args.hf_model_path,
        cache_path=args.cache_path,
        seed=args.seed,
        log_per_iteration=args.log_per_iteration,
    )


def _build_device_params(config: HostIoDecoderSweepConfig) -> dict:
    """Return the device_params dict consumed by :func:`open_mesh_device`.

    Production fabric / worker-l1 constants ported from the original pytest's
    indirect ``device_params`` fixture. These are intentionally not CLI flags
    today; promote if a caller needs to vary them.
    """
    return {
        "fabric_config": _FABRIC_CONFIG,
        "fabric_router_config": create_fabric_router_config(_FABRIC_ROUTER_MAX_PAYLOAD_BYTES),
        "worker_l1_size": _WORKER_L1_SIZE,
    }


def _log_resolved_config(config: HostIoDecoderSweepConfig) -> None:
    """Render the resolved config in a human-readable block, before the run."""
    logger.info("Resolved HostIoDecoderSweepConfig:")
    for field, value in (
        ("decoder_layer_idx", config.decoder_layer_idx),
        ("hidden_states_dir", config.hidden_states_dir),
        ("prompt_names", list(config.prompt_names)),
        ("max_seq_len", config.max_seq_len),
        ("num_slots", config.num_slots),
        ("mesh_rows x mesh_cols", f"{config.mesh_rows} x {config.mesh_cols}"),
        ("num_replication_slots", config.num_replication_slots),
        ("validate_metadata_roundtrip", config.validate_metadata_roundtrip),
        ("validate_hidden_states_cross_slot", config.validate_hidden_states_cross_slot),
        ("validate_kv_cache_cross_slot", config.validate_kv_cache_cross_slot),
        ("validate_hidden_states_cross_trace", config.validate_hidden_states_cross_trace),
        ("pcc_threshold", config.pcc_threshold),
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

    try:
        config = _config_from_args(args)
    except (ValueError, FileNotFoundError) as e:
        parser.error(str(e))
        return 2  # pragma: no cover (parser.error sys.exits 2)

    _log_resolved_config(config)

    if args.dry_run:
        logger.info("--dry-run set; exiting before opening device")
        return 0

    device_params = _build_device_params(config)
    with open_mesh_device(device_params) as parent_mesh:
        result = run_sweep(config, parent_mesh)

    # Result summary so the CLI ends with an audit-friendly digest of what ran.
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"prompts run: {list(result.collected.keys())}")
    logger.info(f"prompt_lengths: {result.schedule.prompt_lengths}")
    logger.info(f"total positions: {result.schedule.total_length()}")
    if result.kv_cache is not None:
        logger.info(f"kv_cache shape: {tuple(result.kv_cache.shape)} dtype={result.kv_cache.dtype}")
    else:
        logger.info("kv_cache: not pulled (no KV-cache validation, no KV-cache dump)")
    if config.dump_dir is not None and (config.dump_hidden_states or config.dump_kv_cache):
        logger.info(f"dumps written under: {config.dump_dir}")
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
