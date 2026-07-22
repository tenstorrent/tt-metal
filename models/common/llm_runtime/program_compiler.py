# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Eager program readiness and compile-output resource ownership."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
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

    @property
    def short(self) -> str:
        """Display-only digest prefix; registry identity always uses ``digest``."""

        return self.digest[:12]


@dataclass(frozen=True)
class OutputSpec:
    shape: tuple[int, ...]
    dtype: Any
    layout: Any = None
    memory_config: Any = None

    @classmethod
    def from_value(cls, value: Any) -> "OutputSpec":
        value = _primary_output(value)
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


@dataclass
class CompiledProgram:
    """Program metadata retained independently from all trace state."""

    key: ProgramKey
    signature: Any
    output_spec: OutputSpec
    ready: bool = True


class ProgramCompiler:
    """Own the sole compiled-program registry for one Llama executor lane."""

    def __init__(self, mesh_device: Any, bound_cache_context: Callable[[], Any]):
        self.mesh_device = mesh_device
        self._bound_cache_context = bound_cache_context
        self._programs: dict[ProgramKey, CompiledProgram] = {}
        self._program_keys: dict[tuple[Any, ...], ProgramKey] = {}
        self._compile_orphans: list[TensorResourceOrphan] = []
        self._trace_capture_in_progress = False
        self._trace_active = False
        self._released = False

    @property
    def programs(self) -> Mapping[ProgramKey, CompiledProgram]:
        return self._programs.copy()

    @property
    def trace_capture_in_progress(self) -> bool:
        return self._trace_capture_in_progress

    @property
    def trace_active(self) -> bool:
        return self._trace_active

    @property
    def compile_orphan_count(self) -> int:
        return len(self._compile_orphans)

    def key_for(self, signature: Any) -> ProgramKey:
        material = _canonical_value(signature_key_material(signature))
        key = self._program_keys.get(material)
        if key is None:
            key = ProgramKey.from_signature(signature)
            self._program_keys[material] = key
        return key

    def get(self, key: ProgramKey) -> CompiledProgram | None:
        return self._programs.get(key)

    def get_for_signature(self, signature: Any) -> CompiledProgram | None:
        key = self.key_for(signature)
        program = self._programs.get(key)
        if program is not None:
            _ensure_matching_signature(key, program.signature, signature)
        return program

    def require_bound_cache_context(self) -> Any:
        context = self._bound_cache_context()
        if context is None:
            raise RuntimeError("Paged KV cache must be allocated and bound before compilation")
        return context

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
        self._ensure_no_compile_orphans()
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
            raise RuntimeError(f"Cannot compile uncompiled program key {key.digest} after trace activation")

        cache_context = self.require_bound_cache_context()
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
        self._ensure_live()
        program = self._programs.get(key)
        if program is None or not program.ready:
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
        if self._released:
            return
        failures = release_orphans(self._compile_orphans)
        if failures:
            error = RuntimeError(f"Failed to release {len(failures)} compile output resource(s)")
            attach_cleanup_failures(error, failures)
            raise error from failures[0]
        for program in self._programs.values():
            program.ready = False
        self._program_keys.clear()
        self._trace_capture_in_progress = False
        self._trace_active = False
        self._released = True

    def _release_or_retain_compile_output(self, output: Any) -> list[BaseException]:
        orphan = TensorResourceOrphan(output)
        failures = best_effort_deallocate_owned_tensors(output, orphan.deallocated_tensor_ids)
        if failures:
            self._compile_orphans.append(orphan)
        return failures

    def _ensure_live(self) -> None:
        if self._released:
            raise RuntimeError("ProgramCompiler has been released")

    def _ensure_no_compile_orphans(self) -> None:
        if self._compile_orphans:
            raise RuntimeError("Cannot compile while unreleased compile outputs remain; clean up this compiler")


def signature_key_material(signature: Any) -> tuple[Any, ...]:
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


def signature_digest(domain: str, schema_version: int, signature: Any) -> str:
    payload = (
        ("domain", domain),
        ("schema_version", schema_version),
        ("signature", signature_key_material(signature)),
    )
    encoded = json.dumps(_canonical_value(payload), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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


def validate_sha256_digest(digest: str, domain: str) -> None:
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{domain} key digest must be a full lowercase SHA-256 hexadecimal digest")


def _primary_output(value: Any) -> Any:
    if isinstance(value, tuple):
        for item in value:
            if item is not None:
                return item
    return value
