# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""JSON contract adapter for the real DSV4 Flash two-layer eval runner.

This is a stable batch-1 server/eval-shaped boundary around
``real_generation_eval_runner``. It is intentionally not an HTTP server, vLLM
engine, all-layer runtime, tokenizer-heavy generation path, or persistent
serving cache.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.demos.deepseek_v4_flash.real_decode_logits_smoke import DEFAULT_DECODE_LOGITS_TOP_K
from models.demos.deepseek_v4_flash.real_decode_stack_logits_smoke import DEFAULT_DECODE_STACK_LAYERS
from models.demos.deepseek_v4_flash.real_generation_eval_runner import (
    DEFAULT_EVAL_DECODE_STEPS,
    DEFAULT_REAL_SNAPSHOT_DIR,
    RUNNER_NAME,
    run_real_generation_eval_runner,
)
from models.demos.deepseek_v4_flash.real_input_stack_logits_smoke import DEFAULT_INPUT_STACK_MAX_EMBEDDING_ROWS
from models.demos.deepseek_v4_flash.real_multi_token_decode_smoke import (
    DEFAULT_MULTI_TOKEN_MAX_BYTES,
    DEFAULT_MULTI_TOKEN_MAX_TENSORS,
)
from models.demos.deepseek_v4_flash.real_prefill_cache_prep_smoke import DEFAULT_SEQUENCE_LENGTH

REAL_SERVER_ADAPTER_SCHEMA_VERSION = 1
REAL_SERVER_ADAPTER_NAME = "deepseek_v4_flash_real_server_adapter"
REAL_SERVER_ADAPTER_CONTRACT_VERSION = 1

REAL_SERVER_LIMITATION_FLAGS = {
    "two_layer_stepping_stone": True,
    "not_production_serving": True,
    "no_http_server": True,
    "not_vllm_engine": True,
    "not_all_61_layers": True,
    "tokenizer_text_generation_excluded": True,
    "persistent_device_cache_excluded": True,
    "generated_ids_not_fed_back": True,
}
REAL_SERVER_LIMITATIONS = (
    "two-layer carried-cache stepping stone over the real generation/eval runner",
    "batch-1 request contract only",
    "not production tt-inference-server serving",
    "not a vLLM engine",
    "not an all-61-layer model path",
    "no HTTP server in this module",
    "no tokenizer-owned text generation loop",
    "no production persistent KV/cache runtime",
    "supplied decode token IDs are fed; top-1 generated IDs are reported but not fed back",
)


@dataclass(frozen=True)
class RealServerRequest:
    """Batch-1 real-runner request accepted by the server/eval adapter."""

    request_id: str = "request-0"
    snapshot_dir: str | Path = DEFAULT_REAL_SNAPSHOT_DIR
    input_ids: Sequence[int] | None = None
    input_ids_path: str | Path | None = None
    prompt_path: str | Path | None = None
    max_tokens: int | None = None
    decode_steps: int | None = None
    prefill_seq_len: int = DEFAULT_SEQUENCE_LENGTH
    layers: Sequence[int] = DEFAULT_DECODE_STACK_LAYERS
    top_k: int = DEFAULT_DECODE_LOGITS_TOP_K
    embedding_mode: str = "slice"
    max_embedding_rows: int = DEFAULT_INPUT_STACK_MAX_EMBEDDING_ROWS
    vocab_mode: str = "slice"
    full_vocab: bool = False
    vocab_start: int = 0
    vocab_size: int | None = None
    max_tensors: int = DEFAULT_MULTI_TOKEN_MAX_TENSORS
    max_bytes: int = DEFAULT_MULTI_TOKEN_MAX_BYTES
    cpu_only: bool = False
    device_id: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")

        input_ids = None if self.input_ids is None else _normalize_int_sequence(self.input_ids, "input_ids")
        input_ids_path = _normalize_optional_path(self.input_ids_path, "input_ids_path")
        prompt_path = _normalize_optional_path(self.prompt_path, "prompt_path")
        provided_inputs = sum(value is not None for value in (input_ids, input_ids_path, prompt_path))
        if provided_inputs > 1:
            raise ValueError("Pass only one of input_ids, input_ids_path, or prompt_path")

        decode_steps = _resolve_decode_steps(self.max_tokens, self.decode_steps)
        max_tokens = decode_steps if self.max_tokens is None else _normalize_positive_int(self.max_tokens, "max_tokens")
        prefill_seq_len = _normalize_positive_int(self.prefill_seq_len, "prefill_seq_len")
        if input_ids is not None and len(input_ids) != prefill_seq_len + decode_steps:
            raise ValueError(
                f"input_ids must contain exactly prefill_seq_len + decode_steps token IDs "
                f"({prefill_seq_len + decode_steps}), got {len(input_ids)}"
            )

        layers = _normalize_layers(self.layers)
        top_k = _normalize_positive_int(self.top_k, "top_k")
        embedding_mode = _normalize_choice(self.embedding_mode, "embedding_mode", ("slice", "full"))
        max_embedding_rows = _normalize_positive_int(self.max_embedding_rows, "max_embedding_rows")
        vocab_mode = _normalize_choice(self.vocab_mode, "vocab_mode", ("slice", "full"))
        full_vocab = _normalize_bool(self.full_vocab, "full_vocab") or vocab_mode == "full"
        vocab_mode = "full" if full_vocab else "slice"
        vocab_start = _normalize_nonnegative_int(self.vocab_start, "vocab_start")
        vocab_size = None if self.vocab_size is None else _normalize_positive_int(self.vocab_size, "vocab_size")
        if full_vocab and (vocab_start != 0 or vocab_size is not None):
            raise ValueError("full-vocab requests require vocab_start=0 and vocab_size=None")

        object.__setattr__(self, "snapshot_dir", Path(self.snapshot_dir).expanduser())
        object.__setattr__(self, "input_ids", input_ids)
        object.__setattr__(self, "input_ids_path", input_ids_path)
        object.__setattr__(self, "prompt_path", prompt_path)
        object.__setattr__(self, "max_tokens", max_tokens)
        object.__setattr__(self, "decode_steps", decode_steps)
        object.__setattr__(self, "prefill_seq_len", prefill_seq_len)
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "embedding_mode", embedding_mode)
        object.__setattr__(self, "max_embedding_rows", max_embedding_rows)
        object.__setattr__(self, "vocab_mode", vocab_mode)
        object.__setattr__(self, "full_vocab", full_vocab)
        object.__setattr__(self, "vocab_start", vocab_start)
        object.__setattr__(self, "vocab_size", vocab_size)
        object.__setattr__(self, "max_tensors", _normalize_positive_int(self.max_tensors, "max_tensors"))
        object.__setattr__(self, "max_bytes", _normalize_positive_int(self.max_bytes, "max_bytes"))
        object.__setattr__(self, "cpu_only", _normalize_bool(self.cpu_only, "cpu_only"))
        object.__setattr__(self, "device_id", _normalize_nonnegative_int(self.device_id, "device_id"))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RealServerRequest":
        allowed = {
            "request_id",
            "snapshot_dir",
            "input_ids",
            "input_ids_path",
            "prompt_path",
            "max_tokens",
            "decode_steps",
            "prefill_seq_len",
            "layers",
            "top_k",
            "embedding_mode",
            "max_embedding_rows",
            "vocab_mode",
            "full_vocab",
            "vocab_start",
            "vocab_size",
            "max_tensors",
            "max_bytes",
            "cpu_only",
            "device_id",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unknown RealServerRequest field(s): {unknown}")
        return cls(**{key: data[key] for key in allowed if key in data})

    def to_mapping(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "snapshot_dir": str(self.snapshot_dir),
            "input_ids": None if self.input_ids is None else [int(value) for value in self.input_ids],
            "input_ids_path": None if self.input_ids_path is None else str(self.input_ids_path),
            "prompt_path": None if self.prompt_path is None else str(self.prompt_path),
            "max_tokens": int(self.max_tokens),
            "decode_steps": int(self.decode_steps),
            "prefill_seq_len": int(self.prefill_seq_len),
            "layers": [int(layer) for layer in self.layers],
            "top_k": int(self.top_k),
            "embedding_mode": self.embedding_mode,
            "max_embedding_rows": int(self.max_embedding_rows),
            "vocab_mode": self.vocab_mode,
            "full_vocab": bool(self.full_vocab),
            "vocab_start": int(self.vocab_start),
            "vocab_size": self.vocab_size,
            "max_tensors": int(self.max_tensors),
            "max_bytes": int(self.max_bytes),
            "cpu_only": bool(self.cpu_only),
            "device_id": int(self.device_id),
        }


def ensure_real_server_request(request: RealServerRequest | Mapping[str, Any]) -> RealServerRequest:
    if isinstance(request, RealServerRequest):
        return request
    if isinstance(request, Mapping):
        return RealServerRequest.from_mapping(request)
    raise TypeError(f"request must be a RealServerRequest or mapping, got {type(request).__name__}")


def run_real_server_request(request: RealServerRequest | Mapping[str, Any]) -> dict[str, Any]:
    request = ensure_real_server_request(request)
    runner_result = run_real_generation_eval_runner(
        request.snapshot_dir,
        layers=request.layers,
        prefill_seq_len=request.prefill_seq_len,
        decode_steps=request.decode_steps,
        input_ids=request.input_ids,
        input_ids_path=request.input_ids_path,
        prompt_path=request.prompt_path,
        embedding_mode=request.embedding_mode,
        max_embedding_rows=request.max_embedding_rows,
        vocab_mode=request.vocab_mode,
        full_vocab_smoke=request.full_vocab,
        vocab_start=request.vocab_start,
        vocab_size=request.vocab_size,
        top_k=request.top_k,
        max_tensors=request.max_tensors,
        max_bytes=request.max_bytes,
        cpu_only=request.cpu_only,
        device_id=request.device_id,
    )
    return summarize_real_server_response(request=request, runner_result=runner_result)


def summarize_real_server_response(
    *,
    request: RealServerRequest,
    runner_result: Mapping[str, Any],
) -> dict[str, Any]:
    timing = dict(runner_result["timing"])
    timing["tokens_per_sec_per_user"] = timing.get("decode_tokens_per_sec_per_user")
    generated = runner_result["generated"]
    ttnn_top1_ids = [int(value) for value in generated.get("ttnn_top1_ids", [])]
    reference_top1_ids = [int(value) for value in generated["reference_top1_ids"]]
    generated_token_ids = ttnn_top1_ids if ttnn_top1_ids else reference_top1_ids
    generated_source = "ttnn_top1_ids" if ttnn_top1_ids else "reference_top1_ids"
    limitation_flags = {
        **REAL_SERVER_LIMITATION_FLAGS,
        "default_sliced_vocab": runner_result["vocab"]["mode"] == "slice",
        "full_vocab_requested": bool(request.full_vocab),
        "cpu_reference_only": bool(request.cpu_only),
    }

    return {
        "schema_version": REAL_SERVER_ADAPTER_SCHEMA_VERSION,
        "request_id": request.request_id,
        "adapter": {
            "name": REAL_SERVER_ADAPTER_NAME,
            "contract_version": REAL_SERVER_ADAPTER_CONTRACT_VERSION,
            "runner": RUNNER_NAME,
            "batch_size": 1,
            "mode": runner_result["mode"],
            "cpu_only": bool(request.cpu_only),
            "device_id": int(request.device_id),
            "request": request.to_mapping(),
            "limitations": list(REAL_SERVER_LIMITATIONS),
        },
        "runner": {
            "name": runner_result["runner"],
            "schema_version": runner_result["schema_version"],
            "wrapped_smoke_schema_version": runner_result["wrapped_smoke_schema_version"],
        },
        "passed": bool(runner_result["passed"]),
        "mode": runner_result["mode"],
        "input": {
            "source": runner_result["input"]["source"],
            "source_path": runner_result["input"]["source_path"],
            "prompt_label": runner_result["input"]["prompt_label"],
            "token_ids": runner_result["input"]["token_ids"],
            "prefill_token_ids": runner_result["input"]["prefill_token_ids"],
            "supplied_decode_token_ids": runner_result["input"]["supplied_decode_token_ids"],
            "decode_feed_mode": runner_result["input"]["decode_feed_mode"],
        },
        "tokens": {
            "input_ids": runner_result["input"]["token_ids"],
            "prefill_token_ids": runner_result["input"]["prefill_token_ids"],
            "supplied_token_ids": runner_result["input"]["supplied_decode_token_ids"],
            "generated_token_ids": generated_token_ids,
            "generated_token_source": generated_source,
            "reference_generated_token_ids": reference_top1_ids,
            "ttnn_generated_token_ids": ttnn_top1_ids,
            "generated_ids_are_fed_back": False,
        },
        "top_k": runner_result["top_k"],
        "correctness": runner_result["correctness"],
        "timing": timing,
        "payload_bytes": runner_result["payload_bytes"],
        "host_visible_boundaries": runner_result["host_boundaries"],
        "ttnn_ops": runner_result["ttnn_ops"],
        "vocab": runner_result["vocab"],
        "positions": runner_result["positions"],
        "limitation_flags": limitation_flags,
    }


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a real DeepSeek V4 Flash two-layer server/eval adapter request.")
    parser.add_argument("--request-json", type=Path, help="Path to a JSON object matching RealServerRequest.")
    parser.add_argument("--request-id", default="request-0")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path(os.environ.get("DSV4_FLASH_REAL_SNAPSHOT_DIR", str(DEFAULT_REAL_SNAPSHOT_DIR))),
    )
    parser.add_argument("--input-ids", type=int, nargs="+")
    parser.add_argument("--input-ids-path", type=Path)
    parser.add_argument("--prompt-path", type=Path)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--decode-steps", type=int)
    parser.add_argument("--prefill-seq-len", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_DECODE_STACK_LAYERS))
    parser.add_argument("--top-k", type=int, default=DEFAULT_DECODE_LOGITS_TOP_K)
    parser.add_argument("--embedding-mode", choices=("slice", "full"), default="slice")
    parser.add_argument("--max-embedding-rows", type=int, default=DEFAULT_INPUT_STACK_MAX_EMBEDDING_ROWS)
    parser.add_argument(
        "--vocab-mode",
        choices=("full", "slice"),
        default=os.environ.get("DSV4_FLASH_REAL_SERVER_ADAPTER_VOCAB_MODE", "slice"),
    )
    parser.add_argument("--full-vocab", action="store_true")
    parser.add_argument(
        "--vocab-start",
        type=int,
        default=_optional_int_env("DSV4_FLASH_REAL_SERVER_ADAPTER_VOCAB_START", 0),
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=_optional_int_env("DSV4_FLASH_REAL_SERVER_ADAPTER_VOCAB_SIZE"),
    )
    parser.add_argument("--max-tensors", type=int, default=DEFAULT_MULTI_TOKEN_MAX_TENSORS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MULTI_TOKEN_MAX_BYTES)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--verbose-logs", action="store_true")
    return parser


def main() -> None:
    args = create_arg_parser().parse_args()
    if not args.verbose_logs:
        os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
        os.environ.setdefault("LOGGER_LEVEL", "Warning")

    request = _request_from_args(args)
    response = run_real_server_request(request)
    print(json.dumps(response, indent=2 if args.pretty else None, sort_keys=True))
    if not response["passed"]:
        raise SystemExit(1)


def _request_from_args(args: argparse.Namespace) -> RealServerRequest:
    if args.request_json is not None:
        with Path(args.request_json).expanduser().open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{args.request_json} must contain a JSON object")
        return RealServerRequest.from_mapping(payload)
    return RealServerRequest(
        request_id=args.request_id,
        snapshot_dir=args.snapshot_dir,
        input_ids=args.input_ids,
        input_ids_path=args.input_ids_path,
        prompt_path=args.prompt_path,
        max_tokens=args.max_tokens,
        decode_steps=args.decode_steps,
        prefill_seq_len=args.prefill_seq_len,
        layers=args.layers,
        top_k=args.top_k,
        embedding_mode=args.embedding_mode,
        max_embedding_rows=args.max_embedding_rows,
        vocab_mode=args.vocab_mode,
        full_vocab=args.full_vocab,
        vocab_start=args.vocab_start,
        vocab_size=args.vocab_size,
        max_tensors=args.max_tensors,
        max_bytes=args.max_bytes,
        cpu_only=args.cpu_only,
        device_id=args.device_id,
    )


def _normalize_int_sequence(values: Sequence[int], label: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of integer token IDs, got {type(values).__name__}")
    return tuple(_normalize_nonnegative_int(value, f"{label}[{index}]") for index, value in enumerate(values))


def _normalize_layers(values: Sequence[int]) -> tuple[int, ...]:
    layers = _normalize_int_sequence(values, "layers")
    if len(layers) < 2:
        raise ValueError("layers must contain at least two consecutive layer IDs")
    if list(layers) != list(range(layers[0], layers[0] + len(layers))):
        raise ValueError(f"layers must be consecutive, got {list(layers)}")
    return layers


def _normalize_optional_path(value: str | Path | None, label: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{label} must be a path-like value when provided, got {type(value).__name__}")
    return Path(value).expanduser()


def _resolve_decode_steps(max_tokens: int | None, decode_steps: int | None) -> int:
    if max_tokens is None and decode_steps is None:
        return DEFAULT_EVAL_DECODE_STEPS
    if max_tokens is not None and decode_steps is not None:
        max_tokens = _normalize_positive_int(max_tokens, "max_tokens")
        decode_steps = _normalize_positive_int(decode_steps, "decode_steps")
        if max_tokens != decode_steps:
            raise ValueError(
                f"max_tokens ({max_tokens}) and decode_steps ({decode_steps}) must match when both are set"
            )
        return decode_steps
    if decode_steps is not None:
        return _normalize_positive_int(decode_steps, "decode_steps")
    return _normalize_positive_int(max_tokens, "max_tokens")


def _normalize_choice(value: Any, label: str, choices: Sequence[str]) -> str:
    if value not in choices:
        raise ValueError(f"{label} must be one of {tuple(choices)}, got {value!r}")
    return str(value)


def _normalize_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean, got {value!r}")
    return bool(value)


def _normalize_nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{label} must be non-negative, got {value}")
    return int(value)


def _normalize_positive_int(value: Any, label: str) -> int:
    parsed = _normalize_nonnegative_int(value, label)
    if parsed <= 0:
        raise ValueError(f"{label} must be positive, got {parsed}")
    return parsed


def _optional_int_env(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


if __name__ == "__main__":
    main()
