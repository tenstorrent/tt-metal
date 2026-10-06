# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Content-addressed GDN CPU references (``reference/gdn``) in the CPU oracle cache shared by every worktree."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.gdn import GDNConfig, GDNReferenceState, gdn_forward_reference
from models.demos.deepseek_v3_d_p.tests.gdn.cases import (
    GDNTestCase,
    GDNWeightSource,
    compute_on_cache_miss,
    prepared_cache_miss,
)
from models.demos.deepseek_v3_d_p.utils.oracle_cache import oracle_cache_root, publish_once

# Covers the stored reference: gdn_forward_reference (models/demos/deepseek_v3_d_p/reference/gdn) and the payload
# layout below. Bump when either changes the stored tensors; the cache is shared by every worktree
# (utils/oracle_cache.py), so an unmerged branch bumps to a value no other branch uses.
_CPU_REFERENCE_CACHE_VERSION = 1


def _tensor_sha256(tensor: torch.Tensor) -> str:
    storage = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(storage)).hexdigest()


def cpu_reference_cache_path(
    weights: GDNWeightSource, hidden: torch.Tensor, initial_state: GDNReferenceState | None
) -> Path:
    """Key one reference run by model, layer, head slice, weights, config, exact input and initial state."""
    payload = {
        "version": _CPU_REFERENCE_CACHE_VERSION,
        "model": weights.model,
        "layer": weights.layer_idx,
        "key_head_slice": weights.key_head_slice,
        "weights": weights.identity,
        "config": asdict(weights.config),
        "hidden": [str(hidden.dtype), list(hidden.shape), _tensor_sha256(hidden)],
        "initial_state": None
        if initial_state is None
        else {
            name: [str(tensor.dtype), list(tensor.shape), _tensor_sha256(tensor)]
            for name, tensor in (("conv", initial_state.conv), ("recurrent", initial_state.recurrent))
        },
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
    return (
        oracle_cache_root()
        / "gdn"
        / weights.model
        / weights.identity
        / "cpu_reference"
        / f"layer_{weights.layer_idx}_t{hidden.shape[0]}_{fingerprint}.pt"
    )


@dataclass(frozen=True)
class GDNChunkReference:
    """FP32 reference of one chunk: output ``[T, hidden]`` and the state after it."""

    output: torch.Tensor
    state: GDNReferenceState
    seconds: float
    cache_hit: bool


def _validate(config: GDNConfig, sequence: int, payload: dict[str, Any]) -> tuple[torch.Tensor, GDNReferenceState]:
    expected_shapes = {
        "output": (sequence, config.hidden_size),
        "conv": (config.conv_kernel_size - 1, config.conv_dim),
        "recurrent": (config.num_value_heads, config.head_k_dim, config.head_v_dim),
    }
    tensors = {name: payload[name] for name in payload["digests"]}
    assert set(tensors) == set(expected_shapes), f"unexpected GDN CPU-reference cache tensors: {set(tensors)}"
    for name, tensor in tensors.items():
        assert tuple(tensor.shape) == expected_shapes[name], f"cached {name} shape {tuple(tensor.shape)}"
        assert _tensor_sha256(tensor) == payload["digests"][name], f"cached {name} checksum mismatch"
    return tensors["output"], GDNReferenceState(conv=tensors["conv"], recurrent=tensors["recurrent"])


def _load_or_compute_chunk(
    case: GDNTestCase, chunk: int, initial_state: GDNReferenceState | None, compute_missing: bool
) -> GDNChunkReference:
    hidden = case.chunk_valid_hidden(chunk)
    cache_path = cpu_reference_cache_path(case.weights, hidden, initial_state)
    label = f"GDN {case.spec.name} chunk {chunk} T={hidden.shape[0]}"
    start = time.perf_counter()
    if not compute_missing and not cache_path.is_file():
        raise prepared_cache_miss(case.spec.name, f"CPU reference (chunk {chunk})", cache_path)

    def produce() -> dict[str, Any]:
        output, state = gdn_forward_reference(hidden, case.weights.load_state_dict(), case.config, initial_state)
        tensors = {"output": output.detach().clone(), "conv": state.conv.clone(), "recurrent": state.recurrent.clone()}
        return {**tensors, "digests": {name: _tensor_sha256(tensor) for name, tensor in tensors.items()}}

    payload, produced = publish_once(
        cache_path, produce, torch.save, lambda path: torch.load(path, map_location="cpu", weights_only=True)
    )
    output, state = _validate(case.config, hidden.shape[0], payload)
    elapsed = time.perf_counter() - start
    hit = "miss, computed" if produced else "hit"
    logger.info(f"{label} CPU reference cache {hit} in {elapsed:.3f} s: {cache_path}")
    return GDNChunkReference(output, state, elapsed, cache_hit=not produced)


def _chained_references(case: GDNTestCase, compute_missing: bool) -> tuple[GDNChunkReference, ...]:
    references = []
    state = None
    for chunk in range(case.num_chunks):
        reference = _load_or_compute_chunk(case, chunk, state, compute_missing)
        references.append(reference)
        state = reference.state
    return tuple(references)


def cpu_references(case: GDNTestCase) -> tuple[GDNChunkReference, ...]:
    """Reference output and state after every chained chunk; chunk i starts from chunk i-1's state.

    Loads prepared entries; a miss fails fast unless ``GDN_CACHE_MISS=compute``.
    """
    return _chained_references(case, compute_on_cache_miss())


def prepare_cpu_references(case: GDNTestCase) -> tuple[GDNChunkReference, ...]:
    """Fill the reference cache for every chained chunk (the CPU preparation step)."""
    return _chained_references(case, compute_missing=True)
