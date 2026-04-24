# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import gc
import json
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from models.demos.deepseek_v4_flash.converter import convert_hf_checkpoint
from models.demos.deepseek_v4_flash.manifest import load_tt_manifest
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint

DEMO_NAME = "deepseek_v4_flash_tiny_scaffold"
DEMO_NOTE = (
    "Tiny scaffold/demo only: synthetic checkpoint, one decoder layer, TTNN LM head; not a full model/perf result."
)
DEFAULT_TOKENS = 32
DEFAULT_LAYER = 2
DEFAULT_TOP_K = 5
DEFAULT_WARMUP_RUNS = 1
DEFAULT_MEASURE_RUNS = 1


@dataclass(frozen=True)
class PreparedCheckpoint:
    preprocessed_path: Path
    generated_synthetic_checkpoint: bool


@dataclass(frozen=True)
class DemoTimings:
    setup_s: float
    model_init_s: float
    warmup_s: float
    run_s: float
    total_s: float


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DeepSeek V4 Flash tiny scaffold demo/perf smoke.")
    parser.add_argument(
        "--preprocessed-path",
        type=Path,
        default=None,
        help="Path to an existing TT-preprocessed tiny checkpoint. If omitted, a synthetic tiny checkpoint is generated.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Optional directory for generated synthetic checkpoint artifacts when --preprocessed-path is omitted.",
    )
    parser.add_argument("--tokens", type=_positive_int, default=DEFAULT_TOKENS, help="Number of input tokens.")
    parser.add_argument("--layer", type=_nonnegative_int, default=DEFAULT_LAYER, help="Decoder layer index to run.")
    parser.add_argument(
        "--top-k",
        type=_positive_int,
        default=DEFAULT_TOP_K,
        help="Number of first-token logits to report.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=_nonnegative_int,
        default=DEFAULT_WARMUP_RUNS,
        help="Number of untimed warmup forwards before measurement.",
    )
    parser.add_argument(
        "--measure-runs",
        type=_positive_int,
        default=DEFAULT_MEASURE_RUNS,
        help="Number of measured forwards. The last logits are summarized.",
    )
    return parser


def run_tiny_model_demo(args: argparse.Namespace) -> dict:
    total_start = time.perf_counter()
    setup_start = time.perf_counter()
    with prepared_tiny_checkpoint(
        preprocessed_path=args.preprocessed_path,
        artifact_dir=args.artifact_dir,
    ) as checkpoint:
        manifest = load_tt_manifest(checkpoint.preprocessed_path)
        vocab_size = int(manifest["config"]["vocab_size"])
        input_ids = deterministic_input_ids(tokens=args.tokens, vocab_size=vocab_size)
        setup_s = time.perf_counter() - setup_start

        ttnn = _import_ttnn()
        require_t3k_available(ttnn)
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 4))
        model = None
        try:
            from models.demos.deepseek_v4_flash.ttnn_model import TtDeepSeekV4FlashTinyModel

            model_init_start = time.perf_counter()
            model = TtDeepSeekV4FlashTinyModel.from_preprocessed(
                checkpoint.preprocessed_path,
                mesh_device=mesh_device,
                layer=args.layer,
            )
            _synchronize_submeshes(ttnn, mesh_device)
            model_init_s = time.perf_counter() - model_init_start

            warmup_start = time.perf_counter()
            for _ in range(args.warmup_runs):
                model(input_ids)
                _synchronize_submeshes(ttnn, mesh_device)
            warmup_s = time.perf_counter() - warmup_start

            logits = None
            run_start = time.perf_counter()
            for _ in range(args.measure_runs):
                logits = model(input_ids)
                _synchronize_submeshes(ttnn, mesh_device)
            run_s = time.perf_counter() - run_start
        finally:
            model = None
            gc.collect()
            submeshes = list(mesh_device.get_submeshes())
            for submesh in submeshes:
                ttnn.synchronize_device(submesh)
            try:
                for submesh in submeshes:
                    ttnn.close_mesh_device(submesh)
                submeshes.clear()
                ttnn.close_mesh_device(mesh_device)
            finally:
                if hasattr(ttnn, "FabricConfig") and hasattr(ttnn, "set_fabric_config"):
                    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

        if logits is None:
            raise RuntimeError("No logits were produced; measure-runs must be positive.")

        timings = DemoTimings(
            setup_s=setup_s,
            model_init_s=model_init_s,
            warmup_s=warmup_s,
            run_s=run_s,
            total_s=time.perf_counter() - total_start,
        )
        return summarize_demo_result(
            logits=logits,
            input_ids=input_ids,
            timings=timings,
            checkpoint=checkpoint,
            layer=args.layer,
            top_k=args.top_k,
            warmup_runs=args.warmup_runs,
            measure_runs=args.measure_runs,
        )


@contextmanager
def prepared_tiny_checkpoint(
    *,
    preprocessed_path: Path | None,
    artifact_dir: Path | None,
) -> Iterator[PreparedCheckpoint]:
    if preprocessed_path is not None:
        preprocessed_path = preprocessed_path.expanduser().resolve()
        if not preprocessed_path.is_dir():
            raise FileNotFoundError(f"TT-preprocessed checkpoint path does not exist: {preprocessed_path}")
        yield PreparedCheckpoint(
            preprocessed_path=preprocessed_path,
            generated_synthetic_checkpoint=False,
        )
        return

    if artifact_dir is None:
        with tempfile.TemporaryDirectory(prefix="deepseek_v4_flash_tiny_demo_") as temp_dir:
            base = Path(temp_dir)
            yield _generate_preprocessed_checkpoint(base)
        return

    artifact_dir = artifact_dir.expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix="deepseek_v4_flash_tiny_demo_", dir=artifact_dir))
    yield _generate_preprocessed_checkpoint(base)


def deterministic_input_ids(*, tokens: int, vocab_size: int) -> torch.Tensor:
    if tokens <= 0:
        raise ValueError(f"tokens must be positive, got {tokens}")
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    return torch.zeros(1, tokens, dtype=torch.int64)


def summarize_demo_result(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    timings: DemoTimings,
    checkpoint: PreparedCheckpoint,
    layer: int,
    top_k: int,
    warmup_runs: int,
    measure_runs: int,
) -> dict:
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"logits must have shape [1, tokens, vocab], got {tuple(logits.shape)}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    first_token = logits[0, 0].float()
    top_count = min(top_k, first_token.shape[0])
    top_values, top_indices = torch.topk(first_token, k=top_count)
    checksum = float(logits.float().sum().item())

    return {
        "demo": DEMO_NAME,
        "note": DEMO_NOTE,
        "checkpoint": {
            "preprocessed_path": str(checkpoint.preprocessed_path),
            "generated_synthetic_checkpoint": checkpoint.generated_synthetic_checkpoint,
        },
        "input": {
            "token_count": int(input_ids.shape[1]),
            "input_ids_shape": list(input_ids.shape),
            "layer": int(layer),
            "warmup_runs": int(warmup_runs),
            "measure_runs": int(measure_runs),
        },
        "logits": {
            "shape": list(logits.shape),
            "checksum": _round_float(checksum),
            "first_token_top_k": [
                {"id": int(index.item()), "value": _round_float(float(value.item()))}
                for index, value in zip(top_indices, top_values)
            ],
        },
        "timing_s": {
            "setup": _round_float(timings.setup_s),
            "model_init": _round_float(timings.model_init_s),
            "warmup": _round_float(timings.warmup_s),
            "run": _round_float(timings.run_s),
            "total": _round_float(timings.total_s),
        },
    }


def require_t3k_available(ttnn_module) -> None:
    try:
        cluster_type = ttnn_module.cluster.get_cluster_type()
        num_devices = ttnn_module.get_num_devices()
    except Exception as exc:
        raise RuntimeError(f"Unable to query TT cluster for DeepSeek V4 Flash tiny demo: {exc}") from exc

    if cluster_type != ttnn_module.cluster.ClusterType.T3K or num_devices != 8:
        raise RuntimeError(f"Requires T3K with 8 devices, found cluster_type={cluster_type}, num_devices={num_devices}")


def _synchronize_submeshes(ttnn_module, mesh_device) -> None:
    for submesh in list(mesh_device.get_submeshes()):
        ttnn_module.synchronize_device(submesh)


def _generate_preprocessed_checkpoint(base: Path) -> PreparedCheckpoint:
    source = generate_tiny_hf_checkpoint(base / "hf_source", num_hidden_layers=3)
    preprocessed = convert_hf_checkpoint(source, base / "tt_preprocessed")
    return PreparedCheckpoint(
        preprocessed_path=preprocessed.resolve(),
        generated_synthetic_checkpoint=True,
    )


def _import_ttnn():
    try:
        import ttnn
    except ImportError as exc:
        raise RuntimeError("TTNN is required to run the DeepSeek V4 Flash tiny demo.") from exc
    return ttnn


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}")
    return parsed


def _round_float(value: float) -> float:
    return round(float(value), 6)


def main() -> None:
    args = create_arg_parser().parse_args()
    try:
        summary = run_tiny_model_demo(args)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
