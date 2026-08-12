# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Eager program readiness and compile-output resource ownership."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

import ttnn
from models.common.llm_runtime.tensor_resources import (
    TensorResourceOrphan,
    attach_cleanup_failures,
    best_effort_deallocate_owned_tensors,
    release_orphans,
)

_PROGRAM_KEY_DOMAIN = "tttv2.llm-runtime.program"
_PROGRAM_KEY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ProgramKey:
    """Full content digest for one operation-produced program signature."""

    digest: str

    def __post_init__(self) -> None:
        validate_sha256_digest(self.digest, "program")

    @classmethod
    def from_signature(cls, signature: Any) -> "ProgramKey":
        return cls(signature_digest(_PROGRAM_KEY_DOMAIN, _PROGRAM_KEY_SCHEMA_VERSION, signature))


@dataclass(frozen=True)
class OutputSpec:
    shape: tuple[int, ...]
    dtype: Any
    layout: Any = None
    memory_config: Any = None

    @classmethod
    def from_value(cls, value: Any) -> "OutputSpec":
        if isinstance(value, tuple):
            if not value:
                raise ValueError("Cannot derive an output specification from an empty tuple")
            value = value[0]
        if isinstance(value, torch.Tensor):
            return cls(shape=tuple(value.shape), dtype=value.dtype)
        if isinstance(value, ttnn.Tensor):
            allocated = value.is_allocated() if hasattr(value, "is_allocated") else False
            return cls(
                shape=tuple(value.shape),
                dtype=value.dtype,
                layout=value.layout,
                memory_config=value.spec.memory_config if allocated else None,
            )
        raise TypeError(f"Cannot derive an output specification from {type(value).__name__}")


@dataclass(frozen=True)
class CompiledProgram:
    """Program metadata retained independently from all trace state."""

    key: ProgramKey
    signature: Any
    output_spec: OutputSpec


class ProgramCompiler:
    """Compile and retain the eager program set for one execution lane.

    ``EagerExecutor.compile_*`` derives an operation signature and calls
    `compile`. Warmup registers the complete program set before
    ``TraceCompiler`` closes the compile gate. Forward execution then uses
    `require_compiled` to reject unseen signatures after trace
    activation. Trace artifacts remain in the separate trace registry.
    """

    def __init__(self, mesh_device: Any, bound_cache_context: Callable[[], Any]):
        self.mesh_device = mesh_device
        self._bound_cache_context = bound_cache_context
        self._programs: dict[ProgramKey, CompiledProgram] = {}
        self._program_keys: dict[tuple[Any, ...], ProgramKey] = {}
        self._compile_orphans: list[TensorResourceOrphan] = []
        self._trace_capture_in_progress = False
        self._trace_active = False
        self._released = False
        self._post_activation_compile_rejections = 0

    # Public API

    @property
    def trace_capture_in_progress(self) -> bool:
        return self._trace_capture_in_progress

    @property
    def trace_active(self) -> bool:
        return self._trace_active

    @property
    def compile_orphan_count(self) -> int:
        return len(self._compile_orphans)

    @property
    def compiled_programs(self) -> tuple[CompiledProgram, ...]:
        """Return an immutable snapshot of the authoritative program registry."""

        return tuple(self._programs.values())

    @property
    def post_activation_compile_rejections(self) -> int:
        """Return unseen-program compile attempts rejected after activation."""

        return self._post_activation_compile_rejections

    def key_for(self, signature: Any) -> ProgramKey:
        """Return the stable program key for one operation signature."""

        material = _canonical_value(_signature_key_material(signature))
        key = self._program_keys.get(material)
        if key is None:
            key = ProgramKey.from_signature(signature)
            self._program_keys[material] = key
        return key

    def compile(
        self,
        signature: Any,
        invoke: Callable[[Any], Any],
        *,
        output_spec: Callable[[Any], OutputSpec] = OutputSpec.from_value,
        release_output: Callable[[Any], Any] = lambda output: output,
        expected_output_spec: OutputSpec | None = None,
    ) -> CompiledProgram:
        """Compile one signature and release its transient invocation output."""

        self._ensure_live()
        if self._compile_orphans:
            raise RuntimeError("Cannot compile while unreleased compile outputs remain; clean up this compiler")
        key = self.key_for(signature)
        existing = self._programs.get(key)
        if existing is not None:
            _ensure_matching_signature(key, existing.signature, signature)
            if expected_output_spec is not None and existing.output_spec != expected_output_spec:
                raise ValueError(f"Program key {key.digest} was compiled with a different output contract")
            return existing
        if self._trace_capture_in_progress:
            raise RuntimeError(f"Cannot compile uncompiled program key {key.digest} while trace capture is in progress")
        if self._trace_active:
            self._post_activation_compile_rejections += 1
            raise RuntimeError(f"Cannot compile uncompiled program key {key.digest} after trace activation")

        cache_context = self._bound_cache_context()
        if cache_context is None:
            raise RuntimeError("Paged KV cache must be allocated and bound before compilation")
        output = invoke(cache_context)
        owned_output = release_output(output)
        try:
            ttnn.synchronize_device(self.mesh_device)
            spec = output_spec(output)
            if expected_output_spec is not None and spec != expected_output_spec:
                raise ValueError(f"Program key {key.digest} produced an unexpected output contract")
        except BaseException as primary:
            cleanup_failures = self._release_or_retain_compile_output(owned_output)
            try:
                ttnn.synchronize_device(self.mesh_device)
            except BaseException as error:
                cleanup_failures.append(error)
            attach_cleanup_failures(primary, cleanup_failures)
            raise

        cleanup_failures = self._release_or_retain_compile_output(owned_output)
        try:
            ttnn.synchronize_device(self.mesh_device)
        except BaseException as primary:
            attach_cleanup_failures(primary, cleanup_failures)
            raise
        if cleanup_failures:
            error = RuntimeError(f"Failed to deallocate {len(cleanup_failures)} compile output resource(s)")
            attach_cleanup_failures(error, cleanup_failures)
            raise error from cleanup_failures[0]

        program = CompiledProgram(key=key, signature=signature, output_spec=spec)
        self._programs[key] = program
        return program

    def require_compiled(self, key: ProgramKey, signature: Any | None = None) -> CompiledProgram:
        """Return registered program metadata or reject an unseen signature."""

        self._ensure_live()
        program = self._programs.get(key)
        if program is None:
            suffix = " after trace activation" if self._trace_active else ""
            raise RuntimeError(f"Program key {key.digest} was not compiled{suffix}")
        if signature is not None:
            _ensure_matching_signature(key, program.signature, signature)
        return program

    def set_trace_capture_in_progress(self, value: bool) -> None:
        self._ensure_live()
        if not isinstance(value, bool):
            raise TypeError("trace capture state must be bool")
        if value and self._trace_active:
            raise RuntimeError("Cannot begin trace capture after trace activation")
        self._trace_capture_in_progress = value

    def set_trace_active(self, value: bool) -> None:
        self._ensure_live()
        if not isinstance(value, bool):
            raise TypeError("trace active state must be bool")
        if value and self._trace_capture_in_progress:
            raise RuntimeError("Trace capture must finish before trace activation")
        self._trace_active = value

    def cleanup(self) -> None:
        """Release retryable compile outputs and terminalize the registry."""

        if self._released:
            return
        failures = release_orphans(self._compile_orphans)
        if failures:
            error = RuntimeError(f"Failed to release {len(failures)} compile output resource(s)")
            attach_cleanup_failures(error, failures)
            raise error from failures[0]
        self._program_keys.clear()
        self._trace_capture_in_progress = False
        self._trace_active = False
        self._released = True

    # Private implementation

    def _release_or_retain_compile_output(self, output: Any) -> list[BaseException]:
        orphan = TensorResourceOrphan(output)
        failures = best_effort_deallocate_owned_tensors(output, orphan.deallocated_tensor_ids)
        if failures:
            self._compile_orphans.append(orphan)
        return failures

    def _ensure_live(self) -> None:
        if self._released:
            raise RuntimeError("ProgramCompiler has been released")


# Public helpers


def signature_digest(domain: str, schema_version: int, signature: Any) -> str:
    """Return a stable SHA-256 digest for explicit signature key material."""

    payload = (
        ("domain", domain),
        ("schema_version", schema_version),
        ("signature", _signature_key_material(signature)),
    )
    encoded = json.dumps(_canonical_value(payload), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_sha256_digest(digest: str, domain: str) -> None:
    """Validate the full lowercase digest representation used by registry keys."""

    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{domain} key digest must be a full lowercase SHA-256 hexadecimal digest")


# Private helpers


def _signature_key_material(signature: Any) -> tuple[Any, ...]:
    """Return the explicit tagged primitive tuple supplied by a signature."""

    try:
        material = signature.key_material
    except AttributeError as error:
        raise TypeError(f"{type(signature).__name__} must expose key_material") from error
    if callable(material):
        material = material()
    if not isinstance(material, tuple):
        raise TypeError("signature key_material must be a tuple")
    return material


def _canonical_value(value: Any) -> Any:
    if value is None:
        return ("null",)
    if isinstance(value, Enum):
        if not isinstance(value.value, str):
            raise TypeError("signature enum values must be stable strings")
        return ("enum", value.value)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", str(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("signature key material must not contain non-finite floats")
        return ("float", value.hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, tuple):
        return ("tuple", tuple(_canonical_value(item) for item in value))
    raise TypeError(
        "signature key material must contain only None, bool, int, finite float, str, stable enums, and tuples; "
        f"got {type(value).__name__}"
    )


def _ensure_matching_signature(key: ProgramKey, retained: Any, candidate: Any) -> None:
    if retained != candidate:
        raise RuntimeError(f"Program key collision for digest {key.digest}: retained signature differs")
