# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Content-addressed independent CPU reference cache shared by KDA acceptance and performance."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.kda import KDAReferenceState, kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.tests.kda.cases import KDATestCase, KDAWeightSource

# Covers the stored reference: kda_forward_reference (models/demos/deepseek_v3_d_p/reference/kda) and the
# payload layout below. Bump when either changes the stored tensors.
_CPU_REFERENCE_CACHE_VERSION = 5


def _tensor_sha256(tensor: torch.Tensor) -> str:
    storage = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(storage)).hexdigest()


def _state_tensors(state: KDAReferenceState) -> dict[str, torch.Tensor]:
    return {
        "recurrent": state.recurrent,
        "q_convolution": state.q_convolution,
        "k_convolution": state.k_convolution,
        "v_convolution": state.v_convolution,
    }


def cpu_reference_cache_path(
    weights: KDAWeightSource,
    hidden: torch.Tensor,
    initial_state: KDAReferenceState | None,
) -> Path:
    """Key one reference run by model, layer, head slice, weights, config, exact input and initial state."""
    fingerprint = hashlib.sha256()
    payload = {
        "version": _CPU_REFERENCE_CACHE_VERSION,
        "model": weights.model,
        "layer": weights.layer_idx,
        "head_slice": weights.head_slice,
        "weights": weights.identity,
        "config": asdict(weights.config),
        "hidden": [str(hidden.dtype), list(hidden.shape), _tensor_sha256(hidden)],
        "initial_state": None
        if initial_state is None
        else {
            name: [str(tensor.dtype), list(tensor.shape), _tensor_sha256(tensor)]
            for name, tensor in _state_tensors(initial_state).items()
        },
    }
    fingerprint.update(json.dumps(payload, sort_keys=True).encode())
    return (
        Path(ttnn.CONFIG.model_cache_path)
        / weights.model
        / weights.identity
        / "cpu_reference"
        / f"layer_{weights.layer_idx}_t{hidden.shape[1]}_{fingerprint.hexdigest()[:20]}.pt"
    )


@dataclass(frozen=True)
class KDAChunkReference:
    output: torch.Tensor
    state: KDAReferenceState
    seconds: float
    cache_hit: bool


def _validate_cached_reference(
    config: KDAConfig, sequence: int, payload: dict[str, Any]
) -> tuple[torch.Tensor, KDAReferenceState]:
    tensors = {name: payload[name] for name in payload["digests"]}
    expected_shapes = {
        "output": (1, sequence, config.hidden_size),
        "recurrent": (1, config.num_heads, config.head_k_dim, config.head_v_dim),
        "q_convolution": (1, config.conv_kernel_size - 1, config.q_dim),
        "k_convolution": (1, config.conv_kernel_size - 1, config.k_dim),
        "v_convolution": (1, config.conv_kernel_size - 1, config.v_dim),
    }
    assert set(tensors) == set(expected_shapes), f"unexpected CPU-reference cache tensors: {set(tensors)}"
    for name, tensor in tensors.items():
        assert isinstance(tensor, torch.Tensor), f"cached {name} is not a tensor"
        assert (
            tuple(tensor.shape) == expected_shapes[name]
        ), f"cached {name} shape {tuple(tensor.shape)} != {expected_shapes[name]}"
        assert _tensor_sha256(tensor) == payload["digests"][name], f"cached {name} checksum mismatch"
    return tensors["output"], KDAReferenceState(
        recurrent=tensors["recurrent"],
        q_convolution=tensors["q_convolution"],
        k_convolution=tensors["k_convolution"],
        v_convolution=tensors["v_convolution"],
    )


def _load_or_compute_chunk(case: KDATestCase, chunk: int, initial_state: KDAReferenceState | None) -> KDAChunkReference:
    hidden = case.chunk_valid_hidden(chunk)
    cache_path = cpu_reference_cache_path(case.weights, hidden, initial_state)
    label = f"KDA {case.spec.name} chunk {chunk} T={hidden.shape[1]}"
    start = time.perf_counter()
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        output, state = _validate_cached_reference(case.config, hidden.shape[1], payload)
        elapsed = time.perf_counter() - start
        logger.info(f"{label} CPU reference cache hit: {cache_path} ({elapsed:.3f} s)")
        return KDAChunkReference(output, state, elapsed, cache_hit=True)

    output, state = kda_forward_reference(hidden, case.weights.load_state_dict(), case.config, initial_state)
    tensors = {"output": output, **_state_tensors(state)}
    tensors = {name: tensor.detach().clone() for name, tensor in tensors.items()}
    payload = {**tensors, "digests": {name: _tensor_sha256(tensor) for name, tensor in tensors.items()}}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    elapsed = time.perf_counter() - start
    logger.info(f"{label} CPU reference cache miss, computed in {elapsed:.3f} s: {cache_path}")
    state = KDAReferenceState(**{name: tensors[name] for name in _state_tensors(state)})
    return KDAChunkReference(tensors["output"], state, elapsed, cache_hit=False)


def cpu_references(case: KDATestCase) -> tuple[KDAChunkReference, ...]:
    """Reference output and state after every chained chunk; chunk i starts from chunk i-1's state."""
    references = []
    state = None
    for chunk in range(case.num_chunks):
        reference = _load_or_compute_chunk(case, chunk, state)
        references.append(reference)
        state = reference.state
    return tuple(references)
