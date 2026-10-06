# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The GDN-on-KDA case matrix shared by the CPU preparation step and the device tests.

A case is fully described by a ``GDNCaseSpec``. Device tests and the preparation command
(``python -m models.demos.deepseek_v3_d_p.tests.gdn.prepare --case <name>``) build cases from the same registered
specs with the same deterministic builders, so both derive identical cache keys. The weight identity never needs
the weights: synthetic weights are identified by their generator version, real weights by the pinned layer digest.

Layouts (tt-work artifacts/loudbox-linear-prefill-requirements.md, design gdn-on-kda §6.3):

* ``LB-A``: 2x4 mesh, SP2 x TP4, 1280 tokens per chunk (640 per SP rank), all heads.
* ``LB-B``: 8x1 mesh, SP8 x TP1, 5120 tokens per chunk, one Galaxy TP4 rank's heads: K heads ``[0, Hk / 4)`` and
  their V heads (whole K-head groups, ``reference/gdn/head_slice.py``); its oracle is that rank's partial output.
* ``SP1-1x4``: 1x4 mesh, SP1 x TP4, 640 tokens (the comparison geometry of the current GDN implementation).
* ``1x1``: one chip, 640 tokens, one Galaxy TP4 rank's heads (single-chip direct scan).

Schedules: one chunk, three chained chunks, or a full chunk followed by a ragged one ending inside an SP rank.
"""

from __future__ import annotations

import functools
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.gdn.config import GDNConfig
from models.demos.deepseek_v3_d_p.reference.gdn.head_slice import gdn_head_slice_config, slice_gdn_heads
from models.demos.deepseek_v3_d_p.reference.gdn.qwen_models import (
    QWEN_FIRST_GDN_LAYER,
    QWEN_GDN_MODELS,
    qwen_gdn_config,
)
from models.demos.deepseek_v3_d_p.tests.gdn.checkpoint_utils import (
    QWEN_LAYER_0_SHA256,
    gdn_checkpoint_dir,
    gdn_state_dict_sha256,
    load_gdn_layer_state_dict,
)
from models.demos.deepseek_v3_d_p.tt.gdn.config import gdn_program_config
from models.demos.deepseek_v3_d_p.tt.gdn.gdn import ttGDN
from models.demos.deepseek_v3_d_p.tt.gdn.weights import GDNWeights
from models.demos.deepseek_v3_d_p.tt.kda.config import KDAProgramConfig
from models.tt_transformers.tt.ccl import TT_CCL

# Identifies the output of synthetic_gdn_weights for a given config. Bump when it changes the weights it returns
# (seed, distributions, shapes or key set).
SYNTHETIC_WEIGHTS_VERSION = 1
_SYNTHETIC_SEED = 20261006
# Seed of the deterministic hidden-state input shared by every GDN case.
HIDDEN_SEED = 1609
# Galaxy TP size whose per-rank heads LB-B and single-chip cases run.
GALAXY_TENSOR_PARALLEL_SIZE = 4

# Two K heads per Galaxy rank would not fit the smallest real model, so the toy keeps the real models' structure
# (Hk = 16 split into whole K-head groups of G = 3 V heads) at small dims: 4 K heads and 12 V heads, K = V = 32.
TOY_GDN_CONFIG = GDNConfig(
    hidden_size=256,
    num_key_heads=4,
    num_value_heads=12,
    head_k_dim=32,
    head_v_dim=32,
    conv_kernel_size=4,
    norm_eps=1e-6,
    output_gate_activation="silu",
)


@dataclass(frozen=True)
class GDNModel:
    """One model's GDN layer under test; ``layer_sha256`` is None for a synthetic-only model."""

    config: Callable[[], GDNConfig]
    layer_idx: int
    layer_sha256: str | None


GDN_MODELS = {
    "toy": GDNModel(lambda: TOY_GDN_CONFIG, 0, None),
    **{
        name: GDNModel(functools.partial(qwen_gdn_config, name), QWEN_FIRST_GDN_LAYER, QWEN_LAYER_0_SHA256[name])
        for name in QWEN_GDN_MODELS
    },
}

SYNTHETIC = "synthetic"
REAL = "real"
RANDN = "randn"
TEXT = "text"

# Device tests load prepared caches and fail fast on a miss ("fail", the default); "compute" builds a missing
# CPU reference / weights inside the device run (local iteration only).
CACHE_MISS_ENV = "GDN_CACHE_MISS"
PREPARE_COMMAND = "python -m models.demos.deepseek_v3_d_p.tests.gdn.prepare"


class GDNPreparedCacheMiss(FileNotFoundError):
    """A device test needed a cache entry the CPU preparation step has not produced."""


def compute_on_cache_miss() -> bool:
    policy = os.environ.get(CACHE_MISS_ENV, "fail")
    if policy not in ("fail", "compute"):
        raise ValueError(f"{CACHE_MISS_ENV} must be 'fail' or 'compute', got {policy!r}")
    return policy == "compute"


def prepared_cache_miss(case_name: str, artifact: str, path: Path) -> GDNPreparedCacheMiss:
    return GDNPreparedCacheMiss(
        f"GDN prepared-cache miss for case {case_name}: {artifact} not found at {path}. "
        f"Prepare it without a device: {PREPARE_COMMAND} --case {case_name} (real weights need the layer fetched by "
        f"{PREPARE_COMMAND} --fetch-layer <model>), or set {CACHE_MISS_ENV}=compute to build it in this device run."
    )


def synthetic_gdn_weights(config: GDNConfig) -> dict[str, torch.Tensor]:
    """Deterministic bf16 canonical weights with Qwen-like statistics at any size.

    Projections are scaled by ``fan_in^-0.5`` so unit-variance inputs give unit-variance projections; the decay
    follows the transformers initialization (``A ~ U(1, 16)``, ``dt_bias = softplus^-1(dt)`` with
    ``dt ~ logU(1e-3, 1e-1)``): per-token log decays span about -1e-4 (long memory) to -15 (fast forgetting),
    median about -0.1, and beta = sigmoid(b) stays inside (0.01, 0.99).
    """
    generator = torch.Generator().manual_seed(_SYNTHETIC_SEED)

    def normal(*shape: int, fan_in: int) -> torch.Tensor:
        return (torch.randn(*shape, generator=generator) * fan_in**-0.5).to(torch.bfloat16)

    def uniform(*shape: int, low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand(*shape, generator=generator)

    hidden, heads = config.hidden_size, config.num_value_heads
    dt = torch.exp(uniform(heads, low=math.log(1e-3), high=math.log(1e-1)))
    return {
        "in_proj_qkv.weight": normal(config.conv_dim, hidden, fan_in=hidden),
        "in_proj_z.weight": normal(config.v_dim, hidden, fan_in=hidden),
        "in_proj_a.weight": normal(heads, hidden, fan_in=hidden),
        "in_proj_b.weight": normal(heads, hidden, fan_in=hidden),
        "out_proj.weight": normal(hidden, config.v_dim, fan_in=config.v_dim),
        "conv1d.weight": uniform(config.conv_dim, 1, config.conv_kernel_size, low=-0.5, high=0.5).to(torch.bfloat16),
        "A_log": torch.log(uniform(heads, low=1.0, high=16.0)),
        "dt_bias": dt + torch.log(-torch.expm1(-dt)),
        "norm.weight": (1.0 + 0.1 * torch.randn(config.head_v_dim, generator=generator)).to(torch.bfloat16),
    }


@dataclass(frozen=True)
class GDNWeightSource:
    """Which GDN layer weights a case uses; identity is known without materializing the weights.

    ``key_head_slice`` ``(start, stop)`` restricts the layer to K heads ``[start, stop)`` and their V heads.
    """

    model: str
    kind: str
    config: GDNConfig
    key_head_slice: tuple[int, int] | None = None

    @property
    def layer_idx(self) -> int:
        return GDN_MODELS[self.model].layer_idx

    @property
    def identity(self) -> str:
        """Content identity of the weights: pinned layer digest or synthetic generator version."""
        model = GDN_MODELS[self.model]
        base = model.layer_sha256 if self.kind == REAL else f"{SYNTHETIC}-v{SYNTHETIC_WEIGHTS_VERSION}"
        if self.key_head_slice is None:
            return base
        return f"{base}-kheads{self.key_head_slice[0]}-{self.key_head_slice[1]}"

    def load_state_dict(self) -> dict[str, torch.Tensor]:
        """Materialize the canonical host weights (expensive; preparation and compute-on-miss only)."""
        return _load_state_dict(self)


@functools.lru_cache(maxsize=1)
def _load_state_dict(source: GDNWeightSource) -> dict[str, torch.Tensor]:
    if source.kind == SYNTHETIC:
        # Synthetic weights are drawn directly at the (possibly sliced) head count.
        return synthetic_gdn_weights(source.config)
    model = GDN_MODELS[source.model]
    full_config = model.config()
    state_dict = load_gdn_layer_state_dict(
        gdn_checkpoint_dir(source.model, model.layer_idx), model.layer_idx, full_config
    )
    digest = gdn_state_dict_sha256(state_dict)
    if digest != model.layer_sha256:
        raise ValueError(f"{source.model} layer {model.layer_idx} weights {digest} != pinned {model.layer_sha256}")
    if source.key_head_slice is None:
        return state_dict
    start, stop = source.key_head_slice
    return slice_gdn_heads(state_dict, full_config, key_head_start=start, num_key_heads=stop - start)


@dataclass(frozen=True)
class GDNCaseSpec:
    """One registered GDN case: weights, inputs, mesh placement and chunk schedule.

    ``chunk_valid_tokens`` lists the valid tokens of each chained chunk; every chunk but the last is full, so a
    shorter last entry is a ragged tail.
    """

    model: str
    weights: str
    mesh_shape: tuple[int, int]
    tensor_parallel_axis: int
    chunk_tokens: int
    chunk_valid_tokens: tuple[int, ...]
    key_head_slice: tuple[int, int] | None = None
    inputs: str = RANDN

    def __post_init__(self) -> None:
        if self.model not in GDN_MODELS:
            raise ValueError(f"unsupported GDN case model {self.model!r}")
        if self.weights not in (SYNTHETIC, REAL):
            raise ValueError(f"weights must be {SYNTHETIC!r} or {REAL!r}, got {self.weights!r}")
        if self.weights == REAL and GDN_MODELS[self.model].layer_sha256 is None:
            raise ValueError(f"{self.model} has no real checkpoint")
        if self.inputs not in (RANDN, TEXT):
            raise ValueError(f"inputs must be {RANDN!r} or {TEXT!r}, got {self.inputs!r}")
        if self.inputs == TEXT and self.weights != REAL:
            raise ValueError("a real-text input is embedded by the real checkpoint; it needs real weights")
        if self.tensor_parallel_axis not in (0, 1):
            raise ValueError(f"tensor_parallel_axis must be 0 or 1, got {self.tensor_parallel_axis}")
        sp_size = self.mesh_shape[1 - self.tensor_parallel_axis]
        if self.chunk_tokens <= 0 or self.chunk_tokens % (sp_size * ttnn.TILE_SIZE):
            raise ValueError(f"chunk_tokens {self.chunk_tokens} must be a positive multiple of SP*{ttnn.TILE_SIZE}")
        if not self.chunk_valid_tokens:
            raise ValueError("chunk_valid_tokens must list at least one chunk")
        if any(valid != self.chunk_tokens for valid in self.chunk_valid_tokens[:-1]):
            raise ValueError("only the last chained chunk may be ragged")
        if not 0 < self.chunk_valid_tokens[-1] <= self.chunk_tokens or self.chunk_valid_tokens[-1] % ttnn.TILE_SIZE:
            raise ValueError(f"last chunk valid tokens must be 32-aligned in (0, {self.chunk_tokens}]")

    @property
    def name(self) -> str:
        rows, columns = self.mesh_shape
        name = (
            f"{self.model}-{self.weights}-mesh{rows}x{columns}-tpaxis{self.tensor_parallel_axis}-T{self.chunk_tokens}"
        )
        if self.key_head_slice is not None:
            name += f"-kheads{self.key_head_slice[0]}-{self.key_head_slice[1]}"
        if len(self.chunk_valid_tokens) > 1:
            name += f"-chunks{len(self.chunk_valid_tokens)}"
        if self.chunk_valid_tokens[-1] != self.chunk_tokens:
            name += f"-last{self.chunk_valid_tokens[-1]}"
        if self.inputs == TEXT:
            name += "-text"
        return name

    @property
    def sequence_parallel_axis(self) -> int:
        return 1 - self.tensor_parallel_axis

    @property
    def sequence_parallel_size(self) -> int:
        return self.mesh_shape[self.sequence_parallel_axis]

    def weight_source(self) -> GDNWeightSource:
        config = GDN_MODELS[self.model].config()
        if self.key_head_slice is not None:
            start, stop = self.key_head_slice
            if not 0 <= start < stop <= config.num_key_heads:
                raise ValueError(f"key_head_slice {self.key_head_slice} outside [0, {config.num_key_heads})")
            config = gdn_head_slice_config(config, stop - start)
        return GDNWeightSource(model=self.model, kind=self.weights, config=config, key_head_slice=self.key_head_slice)


@dataclass(frozen=True)
class GDNTestCase:
    """A built case: weight source and the deterministic physical input of every chunk."""

    spec: GDNCaseSpec
    weights: GDNWeightSource
    hidden: torch.Tensor

    @property
    def config(self) -> GDNConfig:
        return self.weights.config

    @property
    def num_chunks(self) -> int:
        return len(self.spec.chunk_valid_tokens)

    def chunk_hidden(self, chunk: int) -> torch.Tensor:
        """Physical (padded) input of one chunk, ``[1, chunk_tokens, hidden]``."""
        tokens = self.spec.chunk_tokens
        return self.hidden[:, chunk * tokens : (chunk + 1) * tokens]

    def chunk_valid_hidden(self, chunk: int) -> torch.Tensor:
        """Valid tokens of one chunk, the reference input ``[T_valid, hidden]``."""
        return self.chunk_hidden(chunk)[0, : self.spec.chunk_valid_tokens[chunk]]


def build_gdn_case(spec: GDNCaseSpec) -> GDNTestCase:
    """Build a case deterministically; identical in the preparation step and the device test."""
    weights = spec.weight_source()
    if spec.inputs == TEXT:
        # The case is expressible so the accuracy beads (tt_metal_tracker-g1b.5.6 / .7 / .8 / .11) register their
        # cells here; building the embedded real-text window (layer-0 input norm, Flash-Next hyper-connection mix,
        # as g1b.5.12's gdn_decay_range_qwen.py) is theirs to add.
        raise NotImplementedError(f"{spec.name}: real-text GDN inputs are not built yet (g1b.5.6/.7/.8/.11)")
    tokens = spec.chunk_tokens * len(spec.chunk_valid_tokens)
    hidden = torch.randn(
        1,
        tokens,
        weights.config.hidden_size,
        generator=torch.Generator().manual_seed(HIDDEN_SEED),
        dtype=torch.bfloat16,
    )
    return GDNTestCase(spec=spec, weights=weights, hidden=hidden)


def gdn_weight_cache_dir(weights: GDNWeightSource, mesh_shape: tuple[int, int], tensor_parallel_axis: int) -> Path:
    """TTNN model-cache directory of one weight identity and mesh placement (tensorbin stems add layer/config)."""
    rows, columns = mesh_shape
    return (
        Path(ttnn.CONFIG.model_cache_path)
        / "gdn"
        / weights.model
        / weights.identity
        / f"mesh{rows}x{columns}_tpaxis{tensor_parallel_axis}"
    )


def gdn_weight_cache_prefix(weights: GDNWeightSource) -> str:
    return f"layer_{weights.layer_idx}.gdn"


SCHEDULES = ("single", "chained3", "ragged")
# layout -> (mesh shape, TP axis, tokens per chunk, runs one Galaxy TP4 rank's heads)
LAYOUTS = {
    "LB-A": ((2, 4), 1, 1280, False),
    "LB-B": ((8, 1), 1, 5120, True),
    "SP1-1x4": ((1, 4), 1, 640, False),
    "1x1": ((1, 1), 1, 640, True),
}


def _schedule(chunk_tokens: int, schedule: str) -> tuple[int, ...]:
    return {
        "single": (chunk_tokens,),
        "chained3": (chunk_tokens,) * 3,
        "ragged": (chunk_tokens, chunk_tokens * 3 // 4 + ttnn.TILE_SIZE),
    }[schedule]


def gdn_case_spec(model: str, layout: str, schedule: str, weights: str = SYNTHETIC, inputs: str = RANDN) -> GDNCaseSpec:
    mesh_shape, tensor_parallel_axis, chunk_tokens, rank_heads = LAYOUTS[layout]
    key_heads = GDN_MODELS[model].config().num_key_heads
    if rank_heads and key_heads % GALAXY_TENSOR_PARALLEL_SIZE:
        raise ValueError(f"{model}: {key_heads} K heads do not split into Galaxy TP{GALAXY_TENSOR_PARALLEL_SIZE} ranks")
    return GDNCaseSpec(
        model=model,
        weights=weights,
        mesh_shape=mesh_shape,
        tensor_parallel_axis=tensor_parallel_axis,
        chunk_tokens=chunk_tokens,
        chunk_valid_tokens=_schedule(chunk_tokens, schedule),
        key_head_slice=(0, key_heads // GALAXY_TENSOR_PARALLEL_SIZE) if rank_heads else None,
        inputs=inputs,
    )


_REGISTERED_SPECS = (
    # Synthetic weights on seeded random inputs: every model, layout and schedule (tt_metal_tracker-g1b.5.4.3).
    *(gdn_case_spec(model, layout, schedule) for model in GDN_MODELS for layout in LAYOUTS for schedule in SCHEDULES),
    # Real weights on random inputs and on real text at the LoudBox layouts (accuracy beads g1b.5.6/.7/.8/.11).
    *(
        gdn_case_spec(model, layout, schedule, REAL, inputs)
        for model in QWEN_GDN_MODELS
        for layout in ("LB-A", "LB-B")
        for inputs in (RANDN, TEXT)
        for schedule in SCHEDULES
    ),
)
GDN_CASES: dict[str, GDNCaseSpec] = {spec.name: spec for spec in _REGISTERED_SPECS}
assert len(GDN_CASES) == len(_REGISTERED_SPECS), "duplicate GDN case names"


def registered_gdn_case(
    model: str, layout: str, schedule: str, weights: str = SYNTHETIC, inputs: str = RANDN
) -> GDNCaseSpec:
    """Return the registered spec; unregistered cases cannot be prepared, so they are rejected."""
    spec = gdn_case_spec(model, layout, schedule, weights, inputs)
    if spec.name not in GDN_CASES:
        raise KeyError(f"GDN case {spec.name} is not registered in tests/gdn/cases.py::GDN_CASES")
    return GDN_CASES[spec.name]


def case_program_config(spec: GDNCaseSpec) -> KDAProgramConfig:
    """The production GDN program configuration of the case's geometry (Linear TP: LoudBox does not wrap TP)."""
    return gdn_program_config(
        active_seq_len_local=spec.chunk_tokens // spec.sequence_parallel_size,
        sequence_parallel_size=spec.sequence_parallel_size,
        tp_ccl_topology=ttnn.Topology.Linear,
    )


def make_gdn_device_case(mesh_device: ttnn.MeshDevice, case: GDNTestCase) -> ttGDN:
    """Construct the case's layer on its registered mesh with weights from the prepared cache.

    A cache miss fails fast unless ``GDN_CACHE_MISS=compute``, which prepares the host weights in-process.
    """
    spec = case.spec
    if tuple(mesh_device.shape) != spec.mesh_shape:
        raise ValueError(f"{spec.name} is registered for mesh {spec.mesh_shape}, got {tuple(mesh_device.shape)}")
    cache_dir = gdn_weight_cache_dir(case.weights, spec.mesh_shape, spec.tensor_parallel_axis)
    prefix = gdn_weight_cache_prefix(case.weights)
    if GDNWeights.check_cache_complete(
        cache_dir, prefix, case.config, spec.mesh_shape, tensor_parallel_axis=spec.tensor_parallel_axis
    ):
        state_dict = None
        logger.info(f"GDN {spec.name}: weights load from prepared cache {cache_dir}")
    elif compute_on_cache_miss():
        logger.info(f"GDN {spec.name}: preparing host weights in the device run")
        state_dict = case.weights.load_state_dict()
    else:
        raise prepared_cache_miss(spec.name, "weight cache", cache_dir)
    return ttGDN(
        mesh_device,
        case.config,
        state_dict,
        program_config=case_program_config(spec),
        active_seq_len=spec.chunk_tokens,
        layer_idx=case.weights.layer_idx,
        weight_cache_path=cache_dir if state_dict is None else None,
        tt_ccl=TT_CCL(mesh_device) if spec.mesh_shape[spec.tensor_parallel_axis] > 1 else None,
        sp_axis=spec.sequence_parallel_axis,
        tp_axis=spec.tensor_parallel_axis,
    )
