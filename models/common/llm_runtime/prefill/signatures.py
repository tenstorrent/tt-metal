# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Prefill compiler and trace identity policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from models.common.llm_runtime.prefill.plan import PrefillRequest
from models.common.llm_runtime.prefill.sampling_helpers import _TILE_SIZE, SamplingPath
from models.common.modules.sampling.params import PreparedSamplingParams
from models.common.sampling.sampling_params import SamplingParams

PrefillVariant = Literal["regular-single", "regular-batched", "chunked"]


@dataclass(frozen=True)
class PrefillProgramSignature:
    """Material values selecting one eager prefill program variant.

    Active row count is deliberately absent: regular paged fill uses ``-1``
    skip rows, so all active counts with the same padded geometry execute the
    same compiled program. Sampling remains material because it changes the
    postprocessing program family.
    """

    operation_variant: PrefillVariant
    padded_batch_size: int
    invocation_sequence_length: int
    page_table_width: int
    chunk_page_table_width: int | None
    sampling_path: SamplingPath
    penalties_enabled: bool = False
    logprobs_enabled: bool = False
    last_token_tile_start: int | None = None

    def key_material(self) -> tuple[tuple[str, str | int | None], ...]:
        material = [
            ("operation_variant", self.operation_variant),
            ("padded_batch_size", self.padded_batch_size),
            ("invocation_sequence_length", self.invocation_sequence_length),
            ("page_table_width", self.page_table_width),
            ("chunk_page_table_width", self.chunk_page_table_width),
            ("sampling_path", self.sampling_path),
        ]
        if self.penalties_enabled:
            material.append(("penalties_enabled", True))
        if self.logprobs_enabled:
            material.append(("logprobs_enabled", True))
        material.append(("last_token_tile_start", self.last_token_tile_start))
        return tuple(material)


@dataclass(frozen=True)
class PrefillTraceSignature:
    """Identity of one static prefill body and persistent input geometry."""

    operation_variant: PrefillVariant
    padded_batch_size: int
    padded_sequence_length: int
    page_table_width: int
    chunk_page_table_width: int | None
    sampling_path: SamplingPath = "logits"
    penalties_enabled: bool = False
    logprobs_enabled: bool = False

    def key_material(self) -> tuple[tuple[str, str | int | None], ...]:
        material = [
            ("operation_variant", self.operation_variant),
            ("padded_batch_size", self.padded_batch_size),
            ("padded_sequence_length", self.padded_sequence_length),
            ("page_table_width", self.page_table_width),
            ("chunk_page_table_width", self.chunk_page_table_width),
        ]
        if self.sampling_path != "logits":
            material.append(("sampling_path", self.sampling_path))
        if self.penalties_enabled:
            material.append(("penalties_enabled", True))
        if self.logprobs_enabled:
            material.append(("logprobs_enabled", True))
        return tuple(material)


@dataclass(frozen=True)
class PreparedPrefill:
    """A request classified once for eager compilation or traced dispatch."""

    request: PrefillRequest
    sampling_params: SamplingParams | None
    sampling_path: SamplingPath
    program_signatures: tuple[PrefillProgramSignature, ...]
    trace_signature: PrefillTraceSignature | None
    prepared_sampling: PreparedSamplingParams | None = None


def build_program_signatures(
    request: PrefillRequest,
    sampling_path: SamplingPath,
    *,
    static_q128_topk_supported: bool,
    penalties_enabled: bool = False,
    logprobs_enabled: bool = False,
) -> tuple[PrefillProgramSignature, ...]:
    variant: PrefillVariant
    if request.uses_chunked_prefill:
        variant = "chunked"
    elif request.kind == "batched":
        variant = "regular-batched"
    else:
        variant = "regular-single"
    signatures = []
    for chunk in request.chunks:
        last_token_tile_start = None
        uses_static_q128_topk = (
            static_q128_topk_supported
            and sampling_path == "topk"
            and request.kind == "single"
            and not request.uses_chunked_prefill
            and request.padded_sequence_length == 128
        )
        if uses_static_q128_topk or (
            sampling_path == "argmax"
            and request.kind == "single"
            and not request.uses_chunked_prefill
            and request.padded_sequence_length == 128
        ):
            relative_last = request.last_token_indices[0] - request.cached_tokens[0]
            last_token_tile_start = (relative_last // _TILE_SIZE) * _TILE_SIZE
        signatures.append(
            PrefillProgramSignature(
                operation_variant=variant,
                padded_batch_size=request.padded_batch_size,
                invocation_sequence_length=chunk.chunk_size,
                page_table_width=request.page_table_width,
                chunk_page_table_width=(
                    int(chunk.chunk_page_table.shape[-1]) if chunk.chunk_page_table is not None else None
                ),
                sampling_path=sampling_path,
                penalties_enabled=penalties_enabled,
                logprobs_enabled=logprobs_enabled,
                last_token_tile_start=last_token_tile_start,
            )
        )
    return tuple(dict.fromkeys(signatures))


def build_trace_signature(
    request: PrefillRequest,
    *,
    trace_enabled: bool,
    sampling_path: SamplingPath = "logits",
    penalties_enabled: bool = False,
    logprobs_enabled: bool = False,
) -> PrefillTraceSignature | None:
    if not trace_enabled:
        return None
    chunk = request.chunks[0]
    return PrefillTraceSignature(
        operation_variant=(
            "chunked"
            if request.uses_chunked_prefill
            else "regular-batched"
            if request.kind == "batched"
            else "regular-single"
        ),
        padded_batch_size=request.padded_batch_size,
        padded_sequence_length=chunk.chunk_size,
        page_table_width=request.page_table_width,
        chunk_page_table_width=(int(chunk.chunk_page_table.shape[-1]) if chunk.chunk_page_table is not None else None),
        sampling_path=sampling_path,
        penalties_enabled=penalties_enabled,
        logprobs_enabled=logprobs_enabled,
    )


def capture_schema_fingerprint(prepared: PreparedPrefill) -> tuple[Any, ...]:
    """Describe the sampling-free hidden trace allocation."""

    hidden_signature = replace(
        prepared.trace_signature,
        sampling_path="logits",
        penalties_enabled=False,
        logprobs_enabled=False,
    )
    return (
        "prefill-hidden-v2",
        hidden_signature,
        tuple(field for field, value in hidden_signature.key_material() if value is not None),
    )


def workspace_fingerprint(
    prepared: PreparedPrefill,
    *,
    sampling_output_rows: int,
) -> tuple[Any, ...]:
    """Describe one program alias's separately owned postprocess state."""

    return (
        "prefill-postprocess-v1",
        prepared.sampling_path,
        sampling_output_rows,
        prepared.sampling_params is not None,
    )
