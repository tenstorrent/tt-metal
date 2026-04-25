# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Narrow server/vLLM-facing adapter for the DeepSeek V4 Flash tiny scaffold.

This module intentionally does not implement vLLM or tt-inference-server APIs.
It provides a typed batch-1 request contract and JSON-serializable result shape
that future server glue can call while the model remains a tiny scaffold:
synthetic/preprocessed checkpoint, batch 1, host-owned compressed decode cache,
and a per-request helper rather than a production persistent runtime.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from models.demos.deepseek_v4_flash import demo as tiny_demo
from models.demos.deepseek_v4_flash.manifest import load_tt_manifest

ADAPTER_NAME = "deepseek_v4_flash_tiny_server_adapter"
ADAPTER_CONTRACT_VERSION = 1
DEFAULT_MESH_SHAPE = (2, 4)
ADAPTER_LIMITATIONS = (
    "tiny scaffold only",
    "synthetic or TT-preprocessed tiny checkpoint",
    "batch 1 only",
    "host-owned compressed decode cache",
    "no tokenizer/text prompt encoding",
    "not final vLLM or production tt-inference-server integration",
)


@dataclass(frozen=True)
class TinyServerRequest:
    """Batch-1 tiny-scaffold request accepted by the server adapter.

    ``input_ids`` are the already-tokenized prompt. ``prompt`` is optional
    caller metadata only because this scaffold does not own a tokenizer.
    ``decode_steps`` runs deterministic or caller-supplied decode token IDs;
    ``generate_steps`` runs the greedy tiny generation loop. They are mutually
    exclusive, mirroring the demo path.
    """

    input_ids: Sequence[int] | torch.Tensor
    prompt: str | None = None
    generate_steps: int = tiny_demo.DEFAULT_GENERATE_STEPS
    decode_steps: int = tiny_demo.DEFAULT_DECODE_STEPS
    decode_input_ids: Sequence[int] | torch.Tensor | None = None
    top_k: int = tiny_demo.DEFAULT_TOP_K
    layer_ids: Sequence[int] | None = None
    preprocessed_path: str | Path | None = None
    artifact_dir: str | Path | None = None

    def __post_init__(self) -> None:
        input_ids = _normalize_token_ids(self.input_ids, label="input_ids", allow_empty=False)
        decode_input_ids = None
        if self.decode_input_ids is not None:
            decode_input_ids = _normalize_token_ids(
                self.decode_input_ids,
                label="decode_input_ids",
                allow_empty=False,
            )
        generate_steps = _normalize_nonnegative_int(self.generate_steps, "generate_steps")
        decode_steps = _normalize_nonnegative_int(self.decode_steps, "decode_steps")
        top_k = _normalize_positive_int(self.top_k, "top_k")
        layer_ids = (
            (tiny_demo.DEFAULT_LAYER,)
            if self.layer_ids is None
            else _normalize_layer_ids(self.layer_ids, label="layer_ids")
        )
        preprocessed_path = _normalize_optional_path(self.preprocessed_path, "preprocessed_path")
        artifact_dir = _normalize_optional_path(self.artifact_dir, "artifact_dir")

        if self.prompt is not None and not isinstance(self.prompt, str):
            raise TypeError(f"prompt must be a string when provided, got {type(self.prompt).__name__}")
        if generate_steps > 0 and decode_steps > 0:
            raise ValueError("generate_steps and decode_steps are mutually exclusive")
        if decode_input_ids is not None and generate_steps > 0:
            raise ValueError("decode_input_ids cannot be used with generate_steps")
        if decode_input_ids is not None and len(decode_input_ids) != decode_steps:
            raise ValueError(f"decode_input_ids length {len(decode_input_ids)} must match decode_steps {decode_steps}")
        if preprocessed_path is not None and artifact_dir is not None:
            raise ValueError("preprocessed_path and artifact_dir are mutually exclusive")

        object.__setattr__(self, "input_ids", input_ids)
        object.__setattr__(self, "decode_input_ids", decode_input_ids)
        object.__setattr__(self, "generate_steps", generate_steps)
        object.__setattr__(self, "decode_steps", decode_steps)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "layer_ids", layer_ids)
        object.__setattr__(self, "preprocessed_path", preprocessed_path)
        object.__setattr__(self, "artifact_dir", artifact_dir)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TinyServerRequest":
        allowed = {
            "input_ids",
            "prompt",
            "generate_steps",
            "decode_steps",
            "decode_input_ids",
            "top_k",
            "layer_ids",
            "preprocessed_path",
            "artifact_dir",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unknown TinyServerRequest field(s): {unknown}")
        return cls(**{key: data[key] for key in allowed if key in data})

    @property
    def mode(self) -> str:
        if self.generate_steps > 0:
            return "generate"
        if self.decode_steps > 0:
            return "decode"
        return "prefill"

    def input_ids_tensor(self, *, vocab_size: int) -> torch.Tensor:
        _validate_vocab_range(self.input_ids, vocab_size=vocab_size, label="input_ids")
        return torch.tensor([list(self.input_ids)], dtype=torch.int64)

    def decode_input_ids_tensor(self, *, vocab_size: int) -> torch.Tensor:
        if self.decode_steps == 0:
            return tiny_demo.deterministic_decode_input_ids(tokens=0, vocab_size=vocab_size)
        if self.decode_input_ids is None:
            return tiny_demo.deterministic_decode_input_ids(tokens=self.decode_steps, vocab_size=vocab_size)
        _validate_vocab_range(self.decode_input_ids, vocab_size=vocab_size, label="decode_input_ids")
        return torch.tensor([list(self.decode_input_ids)], dtype=torch.int64)


def run_tiny_server_request(request: TinyServerRequest | Mapping[str, Any]) -> dict[str, Any]:
    """Run a tiny-scaffold request on an adapter-owned T3K 2x4 mesh.

    The mesh is opened for this call and synchronized/closed in a ``finally``
    block. This is intentionally a smoke/integration surface, not a persistent
    production serving runtime.
    """

    request = ensure_tiny_server_request(request)
    ttnn = tiny_demo._import_ttnn()
    tiny_demo.require_t3k_available(ttnn)

    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*DEFAULT_MESH_SHAPE))
    try:
        return run_tiny_server_request_on_mesh(
            request,
            mesh_device=mesh_device,
            ttnn_module=ttnn,
            device_ownership="adapter-owned-t3k-mesh",
        )
    finally:
        _close_owned_mesh(ttnn, mesh_device)


def run_tiny_server_request_on_mesh(
    request: TinyServerRequest | Mapping[str, Any],
    *,
    mesh_device,
    ttnn_module=None,
    device_ownership: str = "caller-owned-mesh",
) -> dict[str, Any]:
    """Run a request on a caller-owned T3K 2x4 mesh without closing it."""

    request = ensure_tiny_server_request(request)
    ttnn = ttnn_module if ttnn_module is not None else tiny_demo._import_ttnn()
    _validate_adapter_mesh(mesh_device)

    total_start = time.perf_counter()
    setup_start = time.perf_counter()
    with tiny_demo.prepared_tiny_checkpoint(
        preprocessed_path=request.preprocessed_path,
        artifact_dir=request.artifact_dir,
        layer_ids=request.layer_ids,
    ) as checkpoint:
        manifest = load_tt_manifest(checkpoint.preprocessed_path)
        vocab_size = int(manifest["config"]["vocab_size"])
        input_ids = request.input_ids_tensor(vocab_size=vocab_size)
        decode_input_ids = request.decode_input_ids_tensor(vocab_size=vocab_size)
        setup_s = time.perf_counter() - setup_start

        from models.demos.deepseek_v4_flash.ttnn_model import TtDeepSeekV4FlashTinyModel

        model = None
        try:
            model_init_start = time.perf_counter()
            model = TtDeepSeekV4FlashTinyModel.from_preprocessed(
                checkpoint.preprocessed_path,
                mesh_device=mesh_device,
                layer_ids=request.layer_ids,
            )
            tiny_demo._synchronize_submeshes(ttnn, mesh_device)
            model_init_s = time.perf_counter() - model_init_start

            run_start = time.perf_counter()
            logits = None
            decode_logits = None
            decode_s = 0.0
            generation_result = None
            if request.generate_steps > 0:
                generation_result = tiny_demo.run_tiny_generation_loop(
                    model,
                    input_ids=input_ids,
                    generate_steps=request.generate_steps,
                    synchronize=lambda: tiny_demo._synchronize_submeshes(ttnn, mesh_device),
                )
                logits = generation_result.prefill_logits
            elif request.decode_steps > 0:
                logits, decode_logits, decode_s = tiny_demo._run_prefill_decode_sequence(
                    model,
                    input_ids=input_ids,
                    decode_input_ids=decode_input_ids,
                    ttnn_module=ttnn,
                    mesh_device=mesh_device,
                )
            else:
                logits = model(input_ids)
                tiny_demo._synchronize_submeshes(ttnn, mesh_device)
            run_s = time.perf_counter() - run_start
        finally:
            model = None
            gc.collect()
            tiny_demo._synchronize_submeshes(ttnn, mesh_device)

        return summarize_tiny_server_result(
            request=request,
            logits=logits,
            decode_logits=decode_logits,
            generation_result=generation_result,
            input_ids=input_ids,
            timings=tiny_demo.DemoTimings(
                setup_s=setup_s,
                model_init_s=model_init_s,
                warmup_s=0.0,
                run_s=run_s,
                total_s=time.perf_counter() - total_start,
                decode_s=decode_s,
            ),
            checkpoint=checkpoint,
            layer_ids=request.layer_ids,
            device_ownership=device_ownership,
        )


def summarize_tiny_server_result(
    *,
    request: TinyServerRequest,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    timings: tiny_demo.DemoTimings,
    checkpoint: tiny_demo.PreparedCheckpoint,
    layer_ids: tuple[int, ...],
    decode_logits: torch.Tensor | None = None,
    generation_result: tiny_demo.TinyGenerationResult | None = None,
    device_ownership: str = "unspecified",
) -> dict[str, Any]:
    """Shape a demo summary into the adapter's JSON-serializable response."""

    summary = tiny_demo.summarize_demo_result(
        logits=logits,
        decode_logits=decode_logits,
        generation_result=generation_result,
        input_ids=input_ids,
        timings=timings,
        checkpoint=checkpoint,
        top_k=request.top_k,
        warmup_runs=0,
        measure_runs=1,
        decode_steps=request.decode_steps,
        generate_steps=request.generate_steps,
        layer_ids=layer_ids,
    )
    adapter = {
        "name": ADAPTER_NAME,
        "contract_version": ADAPTER_CONTRACT_VERSION,
        "mode": request.mode,
        "batch_size": 1,
        "top_k": int(request.top_k),
        "input_ids": [int(token_id) for token_id in request.input_ids],
        "decode_input_ids": (
            None if request.decode_input_ids is None else [int(token_id) for token_id in request.decode_input_ids]
        ),
        "device_ownership": device_ownership,
        "limitations": list(ADAPTER_LIMITATIONS),
    }
    if request.prompt is not None:
        adapter["prompt"] = request.prompt
    summary["adapter"] = adapter
    return summary


def ensure_tiny_server_request(request: TinyServerRequest | Mapping[str, Any]) -> TinyServerRequest:
    if isinstance(request, TinyServerRequest):
        return request
    if isinstance(request, Mapping):
        return TinyServerRequest.from_mapping(request)
    raise TypeError(f"request must be a TinyServerRequest or mapping, got {type(request).__name__}")


def _close_owned_mesh(ttnn_module, mesh_device) -> None:
    submeshes = list(mesh_device.get_submeshes())
    try:
        for submesh in submeshes:
            ttnn_module.synchronize_device(submesh)
    finally:
        try:
            for submesh in submeshes:
                ttnn_module.close_mesh_device(submesh)
            ttnn_module.close_mesh_device(mesh_device)
        finally:
            if hasattr(ttnn_module, "FabricConfig") and hasattr(ttnn_module, "set_fabric_config"):
                ttnn_module.set_fabric_config(ttnn_module.FabricConfig.DISABLED)


def _validate_adapter_mesh(mesh_device) -> None:
    mesh_shape = tuple(mesh_device.shape)
    if mesh_shape != DEFAULT_MESH_SHAPE:
        raise ValueError(
            f"DeepSeek V4 Flash tiny server adapter expects mesh shape {DEFAULT_MESH_SHAPE}, got {mesh_shape}"
        )
    if int(mesh_device.get_num_devices()) != 8:
        raise ValueError(
            f"DeepSeek V4 Flash tiny server adapter expects 8 mesh devices, got {mesh_device.get_num_devices()}"
        )


def _normalize_token_ids(
    values: Sequence[int] | torch.Tensor,
    *,
    label: str,
    allow_empty: bool,
) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        if values.ndim == 2 and tuple(values.shape[:1]) == (1,):
            values = values[0].tolist()
        elif values.ndim == 1:
            values = values.tolist()
        else:
            raise ValueError(f"{label} must have shape [tokens] or [1, tokens], got {tuple(values.shape)}")
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of integer token IDs, got {type(values).__name__}")

    normalized = tuple(_normalize_nonnegative_int(value, f"{label}[{index}]") for index, value in enumerate(values))
    if not normalized and not allow_empty:
        raise ValueError(f"{label} must contain at least one token")
    return normalized


def _normalize_layer_ids(values: Sequence[int], *, label: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of integer layer IDs, got {type(values).__name__}")
    normalized = tuple(_normalize_nonnegative_int(value, f"{label}[{index}]") for index, value in enumerate(values))
    if not normalized:
        raise ValueError(f"{label} must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must not contain duplicates, got {normalized}")
    return normalized


def _normalize_optional_path(value: str | Path | None, label: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{label} must be a path-like value when provided, got {type(value).__name__}")
    return Path(value).expanduser()


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


def _validate_vocab_range(values: Sequence[int], *, vocab_size: int, label: str) -> None:
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    max_token_id = max(values)
    if max_token_id >= vocab_size:
        raise ValueError(f"{label} values must be in [0, {vocab_size}), got max={max_token_id}")
