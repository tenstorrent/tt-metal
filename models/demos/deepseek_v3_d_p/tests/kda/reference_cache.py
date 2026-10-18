# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Content-addressed independent CPU reference cache shared by KDA acceptance and performance."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.kda import KDAReferenceState, kda_forward_reference
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import KIMI_K3_FIRST_KDA_LAYER
from models.demos.deepseek_v3_d_p.tests.kda.utils import KimiK3TestCase

_CPU_REFERENCE_CACHE_VERSION = 4


def _tensor_sha256(tensor: torch.Tensor) -> str:
    storage = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(storage)).hexdigest()


def _cpu_reference_cache_path(case: KimiK3TestCase) -> Path:
    reference_dir = Path(inspect.getfile(kda_forward_reference)).parent
    fingerprint = hashlib.sha256()
    fingerprint.update(f"v{_CPU_REFERENCE_CACHE_VERSION}".encode())
    fingerprint.update(str(KIMI_K3_FIRST_KDA_LAYER).encode())
    fingerprint.update(str(case.hidden.shape[1]).encode())
    fingerprint.update(case.weights_identity.encode())
    fingerprint.update(json.dumps(asdict(case.config), sort_keys=True).encode())
    fingerprint.update(_tensor_sha256(case.hidden).encode())
    for source_path in sorted(reference_dir.glob("*.py")):
        fingerprint.update(source_path.name.encode())
        fingerprint.update(source_path.read_bytes())
    return (
        Path(ttnn.CONFIG.model_cache_path)
        / "kimi_k3"
        / case.weights_identity
        / "cpu_reference"
        / f"layer_{KIMI_K3_FIRST_KDA_LAYER}_t{case.hidden.shape[1]}_{fingerprint.hexdigest()[:20]}.pt"
    )


def _reference_tensors(output: torch.Tensor, state: KDAReferenceState) -> dict[str, torch.Tensor]:
    return {
        "output": output.detach().clone(),
        "recurrent": state.recurrent.detach().clone(),
        "q_convolution": state.q_convolution.detach().clone(),
        "k_convolution": state.k_convolution.detach().clone(),
        "v_convolution": state.v_convolution.detach().clone(),
    }


def _validate_cached_reference(case: KimiK3TestCase, payload: dict[str, Any]) -> tuple[torch.Tensor, KDAReferenceState]:
    tensors = {name: payload[name] for name in payload["digests"]}
    expected_shapes = {
        "output": (1, case.hidden.shape[1], case.config.hidden_size),
        "recurrent": (1, case.config.num_heads, case.config.head_k_dim, case.config.head_v_dim),
        "q_convolution": (1, case.config.conv_kernel_size - 1, case.config.q_dim),
        "k_convolution": (1, case.config.conv_kernel_size - 1, case.config.k_dim),
        "v_convolution": (1, case.config.conv_kernel_size - 1, case.config.v_dim),
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


def load_or_compute_cpu_reference(case: KimiK3TestCase) -> tuple[torch.Tensor, KDAReferenceState, float]:
    cache_path = _cpu_reference_cache_path(case)
    start = time.perf_counter()
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        output, state = _validate_cached_reference(case, payload)
        elapsed = time.perf_counter() - start
        logger.info(f"KDA T={case.hidden.shape[1]} CPU reference cache hit: {cache_path}")
        logger.info(f"KDA T={case.hidden.shape[1]} CPU reference load completed in {elapsed:.3f} seconds")
        return output, state, elapsed

    output, state = kda_forward_reference(case.hidden, case.state_dict, case.config)
    tensors = _reference_tensors(output, state)
    payload = {**tensors, "digests": {name: _tensor_sha256(tensor) for name, tensor in tensors.items()}}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    elapsed = time.perf_counter() - start
    logger.info(f"KDA T={case.hidden.shape[1]} CPU reference cache miss: {cache_path}")
    logger.info(f"KDA T={case.hidden.shape[1]} CPU reference computation completed in {elapsed:.3f} seconds")
    return output, state, elapsed
