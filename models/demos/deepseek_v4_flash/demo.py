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
    "Tiny scaffold/demo only: synthetic checkpoint, decoder layer stack, TTNN LM head, batch-1 "
    "host-owned compressed decode cache; not a full model/perf result or final optimized decode kernel."
)
DEFAULT_TOKENS = 32
DEFAULT_LAYER = 2
DEFAULT_TOP_K = 5
DEFAULT_WARMUP_RUNS = 1
DEFAULT_MEASURE_RUNS = 1
DEFAULT_DECODE_STEPS = 0
DEFAULT_GENERATE_STEPS = 0


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
    decode_s: float = 0.0


@dataclass(frozen=True)
class GenerationTimings:
    prefill_s: float
    decode_s: float
    total_s: float


@dataclass(frozen=True)
class TinyGenerationResult:
    prefill_logits: torch.Tensor
    decode_logits: torch.Tensor
    generated_token_ids: torch.Tensor
    cache_current_position: int
    timings: GenerationTimings


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
        "--layer-ids",
        type=_layer_ids_arg,
        default=None,
        help="Comma-separated decoder layer indices to run as a stack. Overrides --layer.",
    )
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
    parser.add_argument(
        "--decode-steps",
        type=_nonnegative_int,
        default=DEFAULT_DECODE_STEPS,
        help="Run this many deterministic batch-1 decode steps after the prefill input.",
    )
    parser.add_argument(
        "--generate-steps",
        type=_nonnegative_int,
        default=DEFAULT_GENERATE_STEPS,
        help="Run this many greedy batch-1 generated decode steps after the prefill input.",
    )
    return parser


def run_tiny_model_demo(args: argparse.Namespace) -> dict:
    total_start = time.perf_counter()
    setup_start = time.perf_counter()
    layer_ids = _demo_layer_ids(args)
    if args.decode_steps > 0 and args.generate_steps > 0:
        raise ValueError("Pass either --decode-steps or --generate-steps, not both")
    with prepared_tiny_checkpoint(
        preprocessed_path=args.preprocessed_path,
        artifact_dir=args.artifact_dir,
        layer_ids=layer_ids,
    ) as checkpoint:
        manifest = load_tt_manifest(checkpoint.preprocessed_path)
        vocab_size = int(manifest["config"]["vocab_size"])
        input_ids = deterministic_input_ids(tokens=args.tokens, vocab_size=vocab_size)
        decode_input_ids = deterministic_decode_input_ids(tokens=args.decode_steps, vocab_size=vocab_size)
        decode_requested = args.decode_steps > 0
        generate_requested = args.generate_steps > 0
        setup_s = time.perf_counter() - setup_start

        ttnn = _import_ttnn()
        require_t3k_available(ttnn)
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 4))
        model = None
        try:
            from models.demos.deepseek_v4_flash.ttnn_model import TtDeepSeekV4FlashTinyModel

            model_init_start = time.perf_counter()
            if args.layer_ids is None:
                model = TtDeepSeekV4FlashTinyModel.from_preprocessed(
                    checkpoint.preprocessed_path,
                    mesh_device=mesh_device,
                    layer=args.layer,
                )
            else:
                model = TtDeepSeekV4FlashTinyModel.from_preprocessed(
                    checkpoint.preprocessed_path,
                    mesh_device=mesh_device,
                    layer_ids=layer_ids,
                )
            _synchronize_submeshes(ttnn, mesh_device)
            model_init_s = time.perf_counter() - model_init_start

            warmup_start = time.perf_counter()
            for _ in range(args.warmup_runs):
                if generate_requested:
                    run_tiny_generation_loop(
                        model,
                        input_ids=input_ids,
                        generate_steps=args.generate_steps,
                        synchronize=lambda: _synchronize_submeshes(ttnn, mesh_device),
                    )
                elif decode_requested:
                    _run_prefill_decode_sequence(
                        model,
                        input_ids=input_ids,
                        decode_input_ids=decode_input_ids,
                        ttnn_module=ttnn,
                        mesh_device=mesh_device,
                    )
                else:
                    model(input_ids)
                    _synchronize_submeshes(ttnn, mesh_device)
            if args.warmup_runs == 0:
                _synchronize_submeshes(ttnn, mesh_device)
            warmup_s = time.perf_counter() - warmup_start

            logits = None
            decode_logits = None
            decode_s = 0.0
            generation_result = None
            generation_prefill_s = 0.0
            generation_decode_s = 0.0
            generation_total_s = 0.0
            run_start = time.perf_counter()
            for _ in range(args.measure_runs):
                if generate_requested:
                    generation_result = run_tiny_generation_loop(
                        model,
                        input_ids=input_ids,
                        generate_steps=args.generate_steps,
                        synchronize=lambda: _synchronize_submeshes(ttnn, mesh_device),
                    )
                    logits = generation_result.prefill_logits
                    generation_prefill_s += generation_result.timings.prefill_s
                    generation_decode_s += generation_result.timings.decode_s
                    generation_total_s += generation_result.timings.total_s
                elif decode_requested:
                    logits, decode_logits, measured_decode_s = _run_prefill_decode_sequence(
                        model,
                        input_ids=input_ids,
                        decode_input_ids=decode_input_ids,
                        ttnn_module=ttnn,
                        mesh_device=mesh_device,
                    )
                    decode_s += measured_decode_s
                else:
                    logits = model(input_ids)
                    _synchronize_submeshes(ttnn, mesh_device)
            run_s = time.perf_counter() - run_start
            if generation_result is not None:
                generation_result = TinyGenerationResult(
                    prefill_logits=generation_result.prefill_logits,
                    decode_logits=generation_result.decode_logits,
                    generated_token_ids=generation_result.generated_token_ids,
                    cache_current_position=generation_result.cache_current_position,
                    timings=GenerationTimings(
                        prefill_s=generation_prefill_s / args.measure_runs,
                        decode_s=generation_decode_s / args.measure_runs,
                        total_s=generation_total_s / args.measure_runs,
                    ),
                )
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
            decode_s=decode_s,
            total_s=time.perf_counter() - total_start,
        )
        return summarize_demo_result(
            logits=logits,
            decode_logits=decode_logits,
            input_ids=input_ids,
            timings=timings,
            checkpoint=checkpoint,
            layer_ids=layer_ids,
            top_k=args.top_k,
            warmup_runs=args.warmup_runs,
            measure_runs=args.measure_runs,
            decode_steps=args.decode_steps,
            generate_steps=args.generate_steps,
            generation_result=generation_result,
        )


@contextmanager
def prepared_tiny_checkpoint(
    *,
    preprocessed_path: Path | None,
    artifact_dir: Path | None,
    layer_ids: tuple[int, ...] = (DEFAULT_LAYER,),
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
            yield _generate_preprocessed_checkpoint(base, layer_ids=layer_ids)
        return

    artifact_dir = artifact_dir.expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix="deepseek_v4_flash_tiny_demo_", dir=artifact_dir))
    yield _generate_preprocessed_checkpoint(base, layer_ids=layer_ids)


def deterministic_input_ids(*, tokens: int, vocab_size: int) -> torch.Tensor:
    if tokens <= 0:
        raise ValueError(f"tokens must be positive, got {tokens}")
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    return torch.zeros(1, tokens, dtype=torch.int64)


def deterministic_decode_input_ids(*, tokens: int, vocab_size: int) -> torch.Tensor:
    if tokens < 0:
        raise ValueError(f"decode tokens must be non-negative, got {tokens}")
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if tokens == 0:
        return torch.empty(1, 0, dtype=torch.int64)
    return deterministic_input_ids(tokens=tokens, vocab_size=vocab_size)


def greedy_next_token_id(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"logits must have shape [1, tokens, vocab], got {tuple(logits.shape)}")
    if logits.shape[1] == 0:
        raise ValueError("logits must contain at least one token")
    if logits.shape[2] == 0:
        raise ValueError("logits must contain at least one vocab entry")
    return torch.argmax(logits[0, -1].float(), dim=-1).reshape(1, 1).to(torch.int64)


def run_tiny_generation_loop(
    model,
    *,
    input_ids: torch.Tensor,
    generate_steps: int,
    synchronize,
    timer=time.perf_counter,
) -> TinyGenerationResult:
    if generate_steps < 0:
        raise ValueError(f"generate_steps must be non-negative, got {generate_steps}")

    total_start = timer()
    prefill_start = timer()
    prefill_logits, cache = model.prefill_with_decode_cache(input_ids)
    synchronize()
    prefill_s = timer() - prefill_start

    current_logits = prefill_logits
    generated_token_ids = []
    decode_logits = []
    decode_s = 0.0
    for _ in range(generate_steps):
        next_token = greedy_next_token_id(current_logits)
        generated_token_ids.append(next_token)
        decode_start = timer()
        current_logits, cache = model.decode_step(next_token, cache=cache)
        synchronize()
        decode_s += timer() - decode_start
        decode_logits.append(current_logits)

    if generated_token_ids:
        generated_tokens = torch.cat(generated_token_ids, dim=1)
        generated_logits = torch.cat(decode_logits, dim=1)
    else:
        generated_tokens = torch.empty(1, 0, dtype=torch.int64)
        generated_logits = prefill_logits.new_empty((1, 0, prefill_logits.shape[-1]))

    return TinyGenerationResult(
        prefill_logits=prefill_logits,
        decode_logits=generated_logits,
        generated_token_ids=generated_tokens,
        cache_current_position=int(cache.current_position),
        timings=GenerationTimings(
            prefill_s=prefill_s,
            decode_s=decode_s,
            total_s=timer() - total_start,
        ),
    )


def summarize_demo_result(
    *,
    logits: torch.Tensor,
    decode_logits: torch.Tensor | None = None,
    generation_result: TinyGenerationResult | None = None,
    input_ids: torch.Tensor,
    timings: DemoTimings,
    checkpoint: PreparedCheckpoint,
    top_k: int,
    warmup_runs: int,
    measure_runs: int,
    decode_steps: int = 0,
    generate_steps: int = 0,
    layer: int | None = None,
    layer_ids: tuple[int, ...] | None = None,
) -> dict:
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"logits must have shape [1, tokens, vocab], got {tuple(logits.shape)}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if layer_ids is None:
        if layer is None:
            raise ValueError("layer or layer_ids must be provided")
        layer_ids = (int(layer),)
    if not layer_ids:
        raise ValueError("layer_ids must be non-empty")
    if decode_steps < 0:
        raise ValueError(f"decode_steps must be non-negative, got {decode_steps}")
    if generate_steps < 0:
        raise ValueError(f"generate_steps must be non-negative, got {generate_steps}")
    if decode_steps > 0 and generate_steps > 0:
        raise ValueError("decode_steps and generate_steps are mutually exclusive")

    first_token = logits[0, 0].float()
    top_count = min(top_k, first_token.shape[0])
    top_values, top_indices = torch.topk(first_token, k=top_count)
    checksum = float(logits.float().sum().item())

    summary = {
        "demo": DEMO_NAME,
        "note": DEMO_NOTE,
        "checkpoint": {
            "preprocessed_path": str(checkpoint.preprocessed_path),
            "generated_synthetic_checkpoint": checkpoint.generated_synthetic_checkpoint,
        },
        "input": {
            "token_count": int(input_ids.shape[1]),
            "input_ids_shape": list(input_ids.shape),
            "layer": int(layer_ids[-1]),
            "layer_ids": [int(layer_id) for layer_id in layer_ids],
            "warmup_runs": int(warmup_runs),
            "measure_runs": int(measure_runs),
            "decode_steps": int(decode_steps),
            "generate_steps": int(generate_steps),
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
    if decode_logits is not None:
        if decode_logits.ndim != 3 or decode_logits.shape[0] != 1:
            raise ValueError(
                f"decode_logits must have shape [1, decode_steps, vocab], got {tuple(decode_logits.shape)}"
            )
        if int(decode_logits.shape[1]) != decode_steps:
            raise ValueError(f"decode_logits token count {decode_logits.shape[1]} does not match {decode_steps}")
        final_token = decode_logits[0, -1].float()
        decode_top_count = min(top_k, final_token.shape[0])
        decode_top_values, decode_top_indices = torch.topk(final_token, k=decode_top_count)
        summary["decode_logits"] = {
            "shape": list(decode_logits.shape),
            "checksum": _round_float(float(decode_logits.float().sum().item())),
            "final_token_top_k": [
                {"id": int(index.item()), "value": _round_float(float(value.item()))}
                for index, value in zip(decode_top_indices, decode_top_values)
            ],
        }
        summary["timing_s"]["decode"] = _round_float(timings.decode_s)
    if generation_result is not None:
        summary["generation"] = summarize_generation_result(
            generation_result,
            prompt_tokens=int(input_ids.shape[1]),
            top_k=top_k,
        )
    return summary


def summarize_generation_result(
    result: TinyGenerationResult,
    *,
    prompt_tokens: int,
    top_k: int,
) -> dict:
    if result.decode_logits.ndim != 3 or result.decode_logits.shape[0] != 1:
        raise ValueError(
            f"decode_logits must have shape [1, generated_tokens, vocab], got {tuple(result.decode_logits.shape)}"
        )
    if tuple(result.generated_token_ids.shape[:1]) != (1,):
        raise ValueError(
            "generated_token_ids must have shape [1, generated_tokens], "
            f"got {tuple(result.generated_token_ids.shape)}"
        )
    generated_tokens = int(result.generated_token_ids.shape[1])
    if int(result.decode_logits.shape[1]) != generated_tokens:
        raise ValueError(
            f"decode_logits token count {result.decode_logits.shape[1]} does not match generated token count "
            f"{generated_tokens}"
        )
    metrics = summarize_generation_metrics(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        timings=result.timings,
    )
    summary = {
        "generated_token_ids": [int(token) for token in result.generated_token_ids[0].tolist()],
        "decode_cache_current_position": int(result.cache_current_position),
        "metrics": metrics,
        "decode_logits": {
            "shape": list(result.decode_logits.shape),
            "checksum": _round_float(float(result.decode_logits.float().sum().item())),
        },
    }
    if generated_tokens > 0:
        final_token = result.decode_logits[0, -1].float()
        top_count = min(top_k, final_token.shape[0])
        top_values, top_indices = torch.topk(final_token, k=top_count)
        summary["decode_logits"]["final_token_top_k"] = [
            {"id": int(index.item()), "value": _round_float(float(value.item()))}
            for index, value in zip(top_indices, top_values)
        ]
    else:
        summary["decode_logits"]["final_token_top_k"] = []
    return summary


def summarize_generation_metrics(
    *,
    prompt_tokens: int,
    generated_tokens: int,
    timings: GenerationTimings,
) -> dict:
    _validate_nonnegative_number(timings.prefill_s, "prefill_s")
    _validate_nonnegative_number(timings.decode_s, "decode_s")
    _validate_nonnegative_number(timings.total_s, "total_s")
    if prompt_tokens <= 0:
        raise ValueError(f"prompt_tokens must be positive, got {prompt_tokens}")
    if generated_tokens < 0:
        raise ValueError(f"generated_tokens must be non-negative, got {generated_tokens}")

    per_token_decode_s = timings.decode_s / generated_tokens if generated_tokens > 0 else 0.0
    decode_tokens_per_s_per_user = (
        generated_tokens / timings.decode_s if generated_tokens > 0 and timings.decode_s > 0 else 0.0
    )
    return {
        "prompt_tokens": int(prompt_tokens),
        "generated_tokens": int(generated_tokens),
        "users": 1,
        "prefill_latency_s": _round_float(timings.prefill_s),
        "total_decode_latency_s": _round_float(timings.decode_s),
        "per_token_decode_latency_s": _round_float(per_token_decode_s),
        "generation_total_latency_s": _round_float(timings.total_s),
        "effective_decode_tokens_s_per_user": _round_float(decode_tokens_per_s_per_user),
    }


def _run_prefill_decode_sequence(
    model,
    *,
    input_ids: torch.Tensor,
    decode_input_ids: torch.Tensor,
    ttnn_module,
    mesh_device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    prefill_logits, cache = model.prefill_with_decode_cache(input_ids)
    _synchronize_submeshes(ttnn_module, mesh_device)
    decode_start = time.perf_counter()
    decode_logits = []
    for token_index in range(decode_input_ids.shape[1]):
        step_logits, cache = model.decode_step(
            decode_input_ids[:, token_index : token_index + 1],
            cache=cache,
        )
        _synchronize_submeshes(ttnn_module, mesh_device)
        decode_logits.append(step_logits)
    decode_s = time.perf_counter() - decode_start
    return prefill_logits, torch.cat(decode_logits, dim=1), decode_s


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


def _generate_preprocessed_checkpoint(
    base: Path, *, layer_ids: tuple[int, ...] = (DEFAULT_LAYER,)
) -> PreparedCheckpoint:
    num_hidden_layers = max(DEFAULT_LAYER + 1, max(layer_ids) + 1)
    source = generate_tiny_hf_checkpoint(
        base / "hf_source",
        num_hidden_layers=num_hidden_layers,
        compress_ratios=_demo_compress_ratios(layer_ids, num_hidden_layers=num_hidden_layers),
    )
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


def _layer_ids_arg(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(part == "" for part in parts):
        raise argparse.ArgumentTypeError(f"expected comma-separated non-negative integers, got {value!r}")
    try:
        layer_ids = tuple(_nonnegative_int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated non-negative integers, got {value!r}") from exc
    if len(set(layer_ids)) != len(layer_ids):
        raise argparse.ArgumentTypeError(f"layer ids must not contain duplicates, got {value!r}")
    return layer_ids


def _demo_layer_ids(args: argparse.Namespace) -> tuple[int, ...]:
    if args.layer_ids is not None:
        return tuple(int(layer_id) for layer_id in args.layer_ids)
    return (int(args.layer),)


def _demo_compress_ratios(layer_ids: tuple[int, ...], *, num_hidden_layers: int) -> tuple[int, ...] | None:
    if layer_ids == (DEFAULT_LAYER,):
        return None
    ratios = [0, 0, 4]
    while len(ratios) < num_hidden_layers:
        ratios.append(4)
    for layer_id in layer_ids:
        ratios[layer_id] = 4
    return tuple(ratios[:num_hidden_layers])


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _validate_nonnegative_number(value: float, label: str) -> None:
    if value < 0:
        raise ValueError(f"{label} must be non-negative, got {value}")


def main() -> None:
    args = create_arg_parser().parse_args()
    try:
        summary = run_tiny_model_demo(args)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
