# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The KDA case matrix shared by the CPU preparation step and the device tests.

A case is fully described by a ``KDACaseSpec``. Device tests and the preparation command build cases from the
same registered specs with the same deterministic builders, so both derive identical cache keys. The weight
identity never requires loading or generating weights: synthetic weights are identified by their generator
version and real weights by their pinned checkpoint content digest.
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config, kimi_k3_model_config
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import (
    KIMI_K3_FIRST_KDA_LAYER,
    KIMI_K3_HF_REVISION,
    KIMI_K3_LAYER_1_SHA256,
    kda_state_dict_sha256,
    load_kda_layer_state_dict,
)
from models.demos.deepseek_v3_d_p.tests.kda.head_slice import (
    galaxy_chip_head_slice_config,
    kda_head_slice_config,
    slice_kda_heads,
)
from models.demos.deepseek_v3_d_p.tests.kda.utils import random_weights, to_sp_input
from models.demos.deepseek_v3_d_p.tt.kda.config import KDAProgramConfig, kimi_k3_program_config
from models.demos.deepseek_v3_d_p.tt.kda.kda import ttKDA
from models.demos.deepseek_v3_d_p.tt.kda.weights import KDAWeights
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.tt_transformers.tt.ccl import TT_CCL

# Identifies the output of utils.random_weights for a given config. Bump when random_weights changes the
# weights it returns (seed, distributions, shapes or key set).
SYNTHETIC_WEIGHTS_VERSION = 1
# Seed of the deterministic hidden-state input shared by every KDA case.
HIDDEN_SEED = 1607

_SYNTHETIC = "synthetic"
_REAL = "real"

# Device tests load prepared caches and fail fast on a miss ("fail", the default); "compute" builds a missing
# CPU reference / weights inside the device run, for local iteration and for CI jobs without a preparation step.
CACHE_MISS_ENV = "KDA_CACHE_MISS"
PREPARE_COMMAND = "python -m models.demos.deepseek_v3_d_p.tests.kda.prepare"


class KDAPreparedCacheMiss(FileNotFoundError):
    """A device test needed a cache entry the CPU preparation step has not produced."""


def compute_on_cache_miss() -> bool:
    policy = os.environ.get(CACHE_MISS_ENV, "fail")
    if policy not in ("fail", "compute"):
        raise ValueError(f"{CACHE_MISS_ENV} must be 'fail' or 'compute', got {policy!r}")
    return policy == "compute"


def prepared_cache_miss(case_name: str, artifact: str, path: Path) -> KDAPreparedCacheMiss:
    return KDAPreparedCacheMiss(
        f"KDA prepared-cache miss for case {case_name}: {artifact} not found at {path}. "
        f"Prepare it without a device: {PREPARE_COMMAND} --case {case_name} "
        f"(real weights need KIMI_K3_CKPT), or set {CACHE_MISS_ENV}=compute to build it inside this device run."
    )


@dataclass(frozen=True)
class KDAWeightSource:
    """Which KDA layer weights a case uses; identity is known without materializing the weights."""

    model: str
    layer_idx: int
    kind: str
    config: KDAConfig
    head_slice: tuple[int, int] | None = None
    checkpoint_dir: Path | None = None

    @property
    def identity(self) -> str:
        """Content identity of the weights: pinned checkpoint digest or synthetic generator version."""
        base = KIMI_K3_LAYER_1_SHA256 if self.kind == _REAL else f"{_SYNTHETIC}-v{SYNTHETIC_WEIGHTS_VERSION}"
        if self.head_slice is None:
            return base
        return f"{base}-heads{self.head_slice[0]}-{self.head_slice[1]}"

    def load_state_dict(self) -> dict[str, torch.Tensor]:
        """Materialize the canonical host state dict (expensive; preparation and compute-on-miss only)."""
        return _load_state_dict(self)


@functools.lru_cache(maxsize=1)
def _load_state_dict(source: KDAWeightSource) -> dict[str, torch.Tensor]:
    if source.kind == _SYNTHETIC:
        # Synthetic weights are drawn directly at the (possibly sliced) head count.
        return random_weights(source.config)
    if source.checkpoint_dir is None:
        raise ValueError("real KDA weights require checkpoint_dir (set KIMI_K3_CKPT)")
    downloaded_config = json.loads((source.checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    assert downloaded_config == kimi_k3_model_config(), "checkpoint config.json differs from the pinned in-tree copy"
    layer_config = kimi_k3_kda_config()
    state_dict = load_kda_layer_state_dict(source.checkpoint_dir, source.layer_idx, layer_config)
    checkpoint_identity = kda_state_dict_sha256(state_dict)
    assert checkpoint_identity == KIMI_K3_LAYER_1_SHA256, (
        f"Kimi-K3 layer {source.layer_idx} weights do not match pinned revision {KIMI_K3_HF_REVISION}: "
        f"{checkpoint_identity}"
    )
    if source.head_slice is None:
        return state_dict
    start, stop = source.head_slice
    return slice_kda_heads(state_dict, layer_config, num_heads=stop - start, head_start=start)


@dataclass(frozen=True)
class KDACaseSpec:
    """One registered KDA case: weights, mesh placement and chunk schedule.

    ``chunk_valid_tokens`` lists the valid tokens of each chained chunk; every chunk but the last is full, so a
    shorter last entry is a ragged tail. ``head_slice`` selects source-layer heads ``[start, stop)`` run on one
    chip (LoudBox LB-B runs heads/4 on TP1).
    """

    model: str
    weights: str
    mesh_shape: tuple[int, int]
    tensor_parallel_axis: int
    chunk_tokens: int
    chunk_valid_tokens: tuple[int, ...]
    head_slice: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.model != "kimi_k3":
            raise ValueError(f"unsupported KDA case model {self.model!r}")
        if self.weights not in (_SYNTHETIC, _REAL):
            raise ValueError(f"weights must be {_SYNTHETIC!r} or {_REAL!r}, got {self.weights!r}")
        if self.tensor_parallel_axis not in (0, 1):
            raise ValueError(f"tensor_parallel_axis must be 0 or 1, got {self.tensor_parallel_axis}")
        sp_size = self.mesh_shape[1 - self.tensor_parallel_axis]
        if self.chunk_tokens <= 0 or self.chunk_tokens % (sp_size * ttnn.TILE_SIZE):
            raise ValueError(f"chunk_tokens {self.chunk_tokens} must be a positive multiple of SP*{ttnn.TILE_SIZE}")
        if not self.chunk_valid_tokens:
            raise ValueError("chunk_valid_tokens must list at least one chunk")
        if any(valid != self.chunk_tokens for valid in self.chunk_valid_tokens[:-1]):
            raise ValueError("only the last chained chunk may be ragged")
        if not 0 < self.chunk_valid_tokens[-1] <= self.chunk_tokens:
            raise ValueError(f"last chunk valid tokens must be in (0, {self.chunk_tokens}]")

    @property
    def name(self) -> str:
        rows, columns = self.mesh_shape
        name = (
            f"{self.model}-{self.weights}-mesh{rows}x{columns}-tpaxis{self.tensor_parallel_axis}-T{self.chunk_tokens}"
        )
        if self.head_slice is not None:
            name += f"-heads{self.head_slice[0]}-{self.head_slice[1]}"
        if len(self.chunk_valid_tokens) > 1:
            name += f"-chunks{len(self.chunk_valid_tokens)}"
        if self.chunk_valid_tokens[-1] != self.chunk_tokens:
            name += f"-last{self.chunk_valid_tokens[-1]}"
        return name

    def weight_source(self, checkpoint_dir: Path | None = None) -> KDAWeightSource:
        config = kimi_k3_kda_config()
        if self.head_slice is not None:
            start, stop = self.head_slice
            if not 0 <= start < stop <= config.num_heads:
                raise ValueError(f"head_slice {self.head_slice} outside [0, {config.num_heads})")
            config = kda_head_slice_config(config, stop - start)
        return KDAWeightSource(
            model=self.model,
            layer_idx=KIMI_K3_FIRST_KDA_LAYER,
            kind=self.weights,
            config=config,
            head_slice=self.head_slice,
            checkpoint_dir=checkpoint_dir if self.weights == _REAL else None,
        )


@dataclass(frozen=True)
class KDATestCase:
    """A built case: weight source and the deterministic physical input of every chunk."""

    spec: KDACaseSpec
    weights: KDAWeightSource
    hidden: torch.Tensor

    @property
    def config(self) -> KDAConfig:
        return self.weights.config

    @property
    def num_chunks(self) -> int:
        return len(self.spec.chunk_valid_tokens)

    def chunk_hidden(self, chunk: int) -> torch.Tensor:
        """Physical (padded) input of one chunk, ``[1, chunk_tokens, hidden]``."""
        tokens = self.spec.chunk_tokens
        return self.hidden[:, chunk * tokens : (chunk + 1) * tokens]

    def chunk_valid_hidden(self, chunk: int) -> torch.Tensor:
        """Valid tokens of one chunk, the reference input."""
        return self.chunk_hidden(chunk)[:, : self.spec.chunk_valid_tokens[chunk]]


def build_kda_case(spec: KDACaseSpec, checkpoint_dir: Path | None = None) -> KDATestCase:
    """Build a case deterministically; identical in the preparation step and the device test."""
    weights = spec.weight_source(checkpoint_dir)
    hidden = torch.randn(
        1,
        spec.chunk_tokens * len(spec.chunk_valid_tokens),
        weights.config.hidden_size,
        generator=torch.Generator().manual_seed(HIDDEN_SEED),
        dtype=torch.bfloat16,
    )
    return KDATestCase(spec=spec, weights=weights, hidden=hidden)


def kda_weight_cache_dir(weights: KDAWeightSource, mesh_shape: tuple[int, int], tensor_parallel_axis: int) -> Path:
    """TTNN model-cache directory of one weight identity and mesh placement (tensorbin stems add layer/config)."""
    rows, columns = mesh_shape
    layout = f"mesh{rows}x{columns}_tpaxis{tensor_parallel_axis}"
    return Path(ttnn.CONFIG.model_cache_path) / weights.model / weights.identity / layout


def _spec(
    weights: str,
    mesh_shape: tuple[int, int],
    tensor_parallel_axis: int,
    chunk_tokens: int,
    *,
    chunk_valid_tokens: tuple[int, ...] | None = None,
    head_slice: tuple[int, int] | None = None,
) -> KDACaseSpec:
    return KDACaseSpec(
        model="kimi_k3",
        weights=weights,
        mesh_shape=mesh_shape,
        tensor_parallel_axis=tensor_parallel_axis,
        chunk_tokens=chunk_tokens,
        chunk_valid_tokens=chunk_valid_tokens or (chunk_tokens,),
        head_slice=head_slice,
    )


def _loudbox_schedules(chunk_tokens: int) -> tuple[tuple[int, ...], ...]:
    """Single chunk, three chained chunks, and a ragged tail that ends inside an SP rank."""
    ragged_tail = chunk_tokens * 3 // 4 + ttnn.TILE_SIZE
    return ((chunk_tokens,), (chunk_tokens,) * 3, (chunk_tokens, ragged_tail))


# LoudBox LB-A: 2x4 mesh, SP2 x TP4, 1280 tokens per chunk (640 per SP rank).
LB_A = ((2, 4), 1, 1280)
# LoudBox LB-B: 8x1 mesh, SP8 x TP1, 5120 tokens per chunk, one TP4 shard's heads (96 / 4).
LB_B = ((8, 1), 1, 5120)
# One Galaxy TP4 rank's heads (rank 0 of the K3 96 heads); see tests/kda/head_slice.py.
_K3_QUARTER_HEADS = (0, galaxy_chip_head_slice_config(kimi_k3_kda_config()).num_heads)

_REGISTERED_SPECS = (
    # LoudBox validation matrix (tt_metal_tracker-g1b.4.6).
    *(_spec(_SYNTHETIC, *LB_A, chunk_valid_tokens=s) for s in _loudbox_schedules(LB_A[2])),
    *(_spec(_REAL, *LB_A, chunk_valid_tokens=s) for s in _loudbox_schedules(LB_A[2])),
    *(
        _spec(_SYNTHETIC, *LB_B, chunk_valid_tokens=s, head_slice=_K3_QUARTER_HEADS)
        for s in _loudbox_schedules(LB_B[2])
    ),
    *(_spec(_REAL, *LB_B, chunk_valid_tokens=s, head_slice=_K3_QUARTER_HEADS) for s in _loudbox_schedules(LB_B[2])),
    # layer/test_acceptance.py synthetic accuracy and perf/test_layer_perf.py synthetic perf.
    _spec(_SYNTHETIC, (2, 4), 1, 5120),
    _spec(_SYNTHETIC, (2, 4), 0, 5120),
    _spec(_SYNTHETIC, (8, 4), 1, 5120),
    # layer/test_dynamic_trace.py production local trace (SP1).
    *(_spec(_SYNTHETIC, mesh, axis, rows) for mesh, axis in (((1, 8), 1), ((8, 1), 0)) for rows in (640, 2560)),
    # layer/test_acceptance.py real-weight accuracy.
    _spec(_REAL, (1, 8), 1, 128),
    _spec(_REAL, (2, 4), 1, 128),
    _spec(_REAL, (2, 4), 0, 128),
    _spec(_REAL, (8, 4), 1, 512),
    # perf/test_layer_perf.py real-weight perf.
    _spec(_REAL, (1, 8), 1, 5120),
    _spec(_REAL, (2, 4), 1, 5120),
    _spec(_REAL, (2, 4), 0, 5120),
)
KDA_CASES: dict[str, KDACaseSpec] = {spec.name: spec for spec in _REGISTERED_SPECS}
assert len(KDA_CASES) == len(_REGISTERED_SPECS), "duplicate KDA case names"


def registered_kda_case(
    weights: str,
    mesh_shape: tuple[int, int],
    tensor_parallel_axis: int,
    chunk_tokens: int,
    *,
    chunk_valid_tokens: tuple[int, ...] | None = None,
    head_slice: tuple[int, int] | None = None,
) -> KDACaseSpec:
    """Return the registered Kimi-K3 spec; unregistered cases cannot be prepared, so they are rejected."""
    spec = _spec(
        weights,
        tuple(mesh_shape),
        tensor_parallel_axis,
        chunk_tokens,
        chunk_valid_tokens=chunk_valid_tokens,
        head_slice=head_slice,
    )
    if spec.name not in KDA_CASES:
        raise KeyError(f"KDA case {spec.name} is not registered in tests/kda/cases.py::KDA_CASES")
    return KDA_CASES[spec.name]


def make_kda_device_case(
    mesh_device: ttnn.MeshDevice,
    case: KDATestCase,
    *,
    summary_group_chunks: int | None = None,
    program_config: KDAProgramConfig | None = None,
) -> tuple[ttKDA, ttnn.Tensor]:
    """Construct the case's production-dimension layer on its registered mesh and its first chunk's input.

    Weights load from the prepared cache. On a miss the test
    fails fast unless ``KDA_CACHE_MISS=compute``, which prepares the host weights in-process (real weights also
    write a missing cache).
    """
    spec = case.spec
    if tuple(mesh_device.shape) != spec.mesh_shape:
        raise ValueError(f"{spec.name} is registered for mesh {spec.mesh_shape}, got {tuple(mesh_device.shape)}")
    tensor_parallel_axis = spec.tensor_parallel_axis
    sequence_parallel_axis = 1 - tensor_parallel_axis
    selected_program_config = program_config or kimi_k3_program_config(
        active_seq_len_local=spec.chunk_tokens // spec.mesh_shape[sequence_parallel_axis],
        # Match production: ring the TP axis wherever the opened fabric wraps it (Galaxy torus).
        tp_ccl_topology=(
            ttnn.Topology.Ring
            if spec.mesh_shape[sequence_parallel_axis] == 1
            else per_axis_topology()[tensor_parallel_axis]
        ),
    )
    if summary_group_chunks is not None:
        selected_program_config = replace(
            selected_program_config,
            recurrence=replace(selected_program_config.recurrence, summary_group_chunks=summary_group_chunks),
        )
    cache_dir = kda_weight_cache_dir(case.weights, spec.mesh_shape, tensor_parallel_axis)
    cache_complete = KDAWeights.check_cache_complete(
        cache_dir,
        f"layer_{case.weights.layer_idx}.kda",
        case.config,
        spec.mesh_shape,
        tensor_parallel_axis=tensor_parallel_axis,
    )
    if cache_complete:
        state_dict, weight_cache_path = None, cache_dir
        logger.info(f"KDA {spec.name}: weights load from prepared cache {cache_dir}")
    elif compute_on_cache_miss():
        logger.info(f"KDA {spec.name}: preparing host weights in the device run")
        state_dict = case.weights.load_state_dict()
        weight_cache_path = cache_dir if case.weights.kind == _REAL else None
    else:
        raise prepared_cache_miss(spec.name, "weight cache", cache_dir)
    layer = ttKDA(
        mesh_device,
        case.config,
        state_dict,
        weight_cache_path=weight_cache_path,
        layer_idx=case.weights.layer_idx,
        tt_ccl=TT_CCL(mesh_device),
        sp_axis=sequence_parallel_axis,
        tp_axis=tensor_parallel_axis,
        program_config=selected_program_config,
        active_seq_len=spec.chunk_tokens,
    )
    return layer, to_sp_input(case.chunk_hidden(0), mesh_device, sequence_parallel_axis)
