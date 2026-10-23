# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The sampled token chosen on device, inside TAIL, from the candidate row: an integer-exact composite of ttnn ops.

Today's sampled step reads the candidate row (``Qwen38TTNNLMHead.sampling_candidates``) back to the host, samples
there and writes the token row before HEAD can start.  This module moves the choice of the token into TAIL's
epilogue so the sampled loop is the greedy loop plus one small host write per step (the uniform draw).  The host
reference :func:`device_sampler_reference` computes the same token from the same row bit for bit, so a run can be
re-derived on the host from the read rows and the ledger of draws.

The op sequence (:func:`sample_on_device`, 57 ttnn calls = 76 device programs on the replicated 128-lane row):

1. Split: the row ``[v_d | g_d]`` per shard becomes the value lanes ``s`` and the global-id lanes ``g``
   (``ttnn.gather`` with constant index rows: 32-bit copies).
2. Rank, value descending and global id ascending: ``rank_j = #{i: s_i > s_j} + #{i: s_i == s_j, g_i < g_j}``
   from three comparison matrices and a sum of 0/1 lanes (exact on every reduce path); the inverse permutation by
   another 0/1 sum; the sorted views by gathers.  Equal values are ordered by the lowest global id, the rule of
   the host samplers (``torch.argmax`` / the stable sort over ascending ids).
3. Weights: the temperature is a host-written table ``W(i) = round(2**18 * exp(-i / (1024 T)))`` indexed by
   ``floor(1024 * (s_max - s_j))`` (clamped to the table), so nothing transcendental runs on the device and the
   host reference needs no model of the SFPU exp.  Lanes at or beyond ``top_k`` are masked to weight 0.
4. Exact prefix sums: the weights split into an 11-bit high and an 8-bit low half through two ``[1,128] x
   [128,128]`` fp32 matmuls against a constant upper-triangular 0/1 matrix (the TF32 input truncation of the FPU
   is lossless on 11-bit integers, the fp32 accumulation exact); recombined on the SFPU.
5. Filters as prefix lengths: ``K = min(#(C_ex <= tau), #(W >= max(m, 1)))`` with ``tau = top_p * S_k`` (the
   total weight of the top-k lanes) and ``m = min_p * 2**18``: the top-k mask zeroes the lanes beyond ``top_k``,
   so the second count is ``min(top_k, K_min_p, K_positive)``; ``K >= 1`` always (lane 0 has weight ``2**18``,
   exclusive prefix 0).
6. Draw: ``theta = u * S`` (``S`` the total kept weight); the chosen sorted lane is ``min(#(C_in <= theta), K - 1)``,
   the reference's inverse CDF (a lane beyond ``K`` reaches ``theta`` only when ``theta == S``, where the clamp
   decides); its id is a gather.
7. Select: ``token_row = greedy_row * greedy_flag + splat(sampled) * (1 - greedy_flag)``; ``greedy_flag = 1``
   makes the row the greedy resolve's, bitwise (the temperature-0 short-circuit).

Exactly two fp32 roundings (``tau`` and ``theta``, single SFPU multiplies, RNE) are not integer-exact; the
reference reproduces them with fp32 torch multiplies (``_sfpu_mul``).  Everything else is integer arithmetic
below ``2**24`` on the SFPU, 0/1 or 11-bit inputs through the FPU, or 32-bit data movement.

The uniform draw ``u = n * 2**-24`` comes from a host splitmix64 stream per request (:class:`UniformStream`),
written per step into a persistent ``[1,1,1,1]`` tensor behind HEAD; the same stream feeds a host fallback so a
fallback step continues the sequence.

Ties: the per-shard ``ttnn.topk`` decides which members of a tie at a shard's 32nd value are read (unchanged from
the host sampler's relaxed gate; the read set can differ from ``torch.topk``'s only there).  From the row on the
sampler is a deterministic function of ``(row, parameters, u)`` with the lowest global id first among equal values,
so on the same read row it agrees with ``sample_candidates`` up to the quantization boundaries below; against the
full vocabulary it can also differ by exchanging tokens of equal bf16 logit at a shard's 32nd value.  Top-k is
limited to the 32 candidates the row carries per shard.

Deviations from the host sampler (:func:`~models.demos.blackhole.qwen38_flash_next.ttnn.sampling.sample_candidates`),
stated: the weights are quantized (relative error at most ``2**-10`` from the 1/1024 grid plus ``0.5 / W`` from the
rounding, far below the bf16 rounding of the logits themselves); the top-p boundary compares exact integer
prefixes to one rounded product; a token with ``p / p_max < 2**-19`` gets weight 0.  The device path serves
``0 < T <= 4``, ``1 <= top_k <= 32``, ``min_p <= 1`` and no penalties; everything else keeps the host loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from ttnn.tools.trace_allocation_tracker import acknowledge_corruptible

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    Qwen38MeshContract,
    TensorPlacement,
    replicate_tensor_2d_mesh_mapper,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import (
    SAMPLING_CANDIDATE_ROW_SHAPE,
    SAMPLING_CANDIDATES_PER_DEVICE,
    TOKEN_ROW_SHAPE,
    TP_SIZE,
    _deallocate,
    _metadata,
    _shape,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import UNIFORM_BITS, Qwen38SamplingParameters, UniformStream

LANES = TP_SIZE * SAMPLING_CANDIDATES_PER_DEVICE  # 128 candidates per row
LANE_SHAPE = (1, 1, 1, LANES)
SCALAR_SHAPE = (1, 1, 1, 1)
MATRIX_SHAPE = (1, 1, LANES, LANES)
WEIGHT_BITS = 18
WEIGHT_ONE = 1 << WEIGHT_BITS  # the weight of the maximum lane
DELTA_GRID = 1024  # table index per unit of logit difference
TABLE_SIZE = 65536
TABLE_SHAPE = (1, 1, 1, TABLE_SIZE)
WEIGHT_LOW_BITS = 8  # the low half of a weight: below 2**8; the high half at most 2**10 (TF32-exact)
MAX_TEMPERATURE = 4.0  # exp(-65535 / (1024 T)) * 2**18 < 1 for T <= 4: the table's last entry is 0
MAX_TOP_K = SAMPLING_CANDIDATES_PER_DEVICE
DEVICE_OP_COUNT = 57


def _sfpu_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """One fp32 SFPU multiply: the two rounding sites of the composite (RNE, pinned by micro-test M1)."""

    return (left.to(torch.float32) * right.to(torch.float32)).to(torch.float32)


def weight_table(temperature: float) -> torch.Tensor:
    """FP32 ``[1,1,1,65536]``: ``round(2**18 * exp(-i / (1024 T)))``, integers, non-increasing, ``table[0] = 2**18``."""

    if not 0 < temperature <= MAX_TEMPERATURE:
        raise ValueError(f"temperature must be in (0, {MAX_TEMPERATURE}], got {temperature}")
    index = torch.arange(TABLE_SIZE, dtype=torch.float64)
    table = torch.round(WEIGHT_ONE * torch.exp(-index / (DELTA_GRID * float(temperature))))
    return table.to(torch.float32).reshape(TABLE_SHAPE)


@dataclass(frozen=True)
class Qwen38DeviceSamplerPolicy:
    """The per-request scalars the host writes: what the composite reads besides the row and the draw."""

    temperature: float
    top_k: int
    top_p: float
    min_p: float
    greedy: bool = False
    # The OpenAI presence penalty over the request's output: the one-program sampler (``fused/sampler_tail``) keeps
    # the emitted tokens on the device and applies it; the composite takes no penalty (``from_parameters`` refuses).
    presence_penalty: float = 0.0

    def __post_init__(self) -> None:
        if not self.greedy and not 0 < self.temperature <= MAX_TEMPERATURE:
            raise ValueError(f"temperature must be in (0, {MAX_TEMPERATURE}], got {self.temperature}")
        if not 0 <= self.presence_penalty <= 2:
            raise ValueError(f"presence_penalty must be in [0, 2], got {self.presence_penalty}")
        if isinstance(self.top_k, bool) or type(self.top_k) is not int or not 1 <= self.top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be an integer in [1, {MAX_TOP_K}], got {self.top_k!r}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if not 0 <= self.min_p <= 1:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}")

    @classmethod
    def greedy_policy(cls) -> "Qwen38DeviceSamplerPolicy":
        """Neutral parameters with the flag set: the composite runs and the greedy row is selected."""

        return cls(temperature=1.0, top_k=MAX_TOP_K, top_p=1.0, min_p=0.0, greedy=True)

    @classmethod
    def from_parameters(
        cls, p: Qwen38SamplingParameters, *, presence_on_device: bool = False
    ) -> "Qwen38DeviceSamplerPolicy | None":
        """The device policy of a request, or ``None`` when the request needs the host loop (``refusal`` says why)."""

        if p.temperature == 0:
            return cls.greedy_policy()
        if cls.refusal(p, presence_on_device=presence_on_device) is not None:
            return None
        return cls(
            temperature=p.temperature,
            top_k=p.top_k,
            top_p=p.top_p,
            min_p=p.min_p,
            presence_penalty=p.presence_penalty if presence_on_device else 0.0,
        )

    @classmethod
    def refusal(cls, p: Qwen38SamplingParameters, *, presence_on_device: bool = False) -> str | None:
        """Why a sampled request keeps the host loop: a temperature above ``MAX_TEMPERATURE``, ``top_k`` 0 or above
        the row's 32 per shard, a frequency penalty (counts, not bits), a repetition penalty (the multiplicative
        transformers rule), a negative presence penalty (it can lift unread tokens: the host path's own fallback),
        or any presence penalty when the sampler serving cannot apply one (the composite)."""

        if p.temperature == 0:
            return None
        if not p.temperature <= MAX_TEMPERATURE:
            return f"temperature {p.temperature} above the table's {MAX_TEMPERATURE}"
        if not 1 <= p.top_k <= MAX_TOP_K:
            return f"top_k {p.top_k} outside [1, {MAX_TOP_K}]"
        if p.frequency_penalty != 0:
            return "frequency penalty (counts, not presence bits)"
        if p.repetition_penalty != 1:
            return "repetition penalty (the multiplicative rule)"
        if p.presence_penalty < 0:
            return "negative presence penalty (can lift unread tokens)"
        if p.presence_penalty != 0 and not presence_on_device:
            return "presence penalty on a sampler without the device history"
        return None

    @property
    def min_weight(self) -> float:
        """``max(fp32(min_p) * 2**18, 1)``: a lane is kept when its weight reaches it (min-p, and the zero-weight tail)."""

        return max(float(torch.tensor(self.min_p, dtype=torch.float32) * WEIGHT_ONE), 1.0)


@dataclass(frozen=True)
class Qwen38DeviceSample:
    """What the composite decides for one row: the token, the kept prefix length and the chosen sorted lane."""

    token_id: int
    kept: int
    sorted_lane: int
    uniform: float


def device_sampler_reference(
    values: torch.Tensor,
    ids: torch.Tensor,
    policy: Qwen38DeviceSamplerPolicy,
    uniform: float,
    *,
    table=None,
    seen: torch.Tensor | None = None,
) -> Qwen38DeviceSample:
    """The host reference of :func:`sample_on_device` (and of ``fused/sampler_tail``) for one row: ``values`` fp32
    ``[128]`` and ``ids`` int64 ``[128]`` in lane order (shard d's 32 lanes at ``32 d``), the policy's temperature
    table, one draw ``u``; ``seen`` (bool ``[128]``: the lanes whose id the request has emitted) takes the policy's
    presence penalty off those values first (an fp32 subtract, the host sampler's), required when it is nonzero."""

    if tuple(values.shape) != (LANES,) or values.dtype != torch.float32:
        raise ValueError(f"values must be fp32 [{LANES}], got {values.dtype} {tuple(values.shape)}")
    if tuple(ids.shape) != (LANES,) or ids.dtype != torch.int64:
        raise ValueError(f"ids must be int64 [{LANES}], got {ids.dtype} {tuple(ids.shape)}")
    if not 0 <= uniform < 1 or uniform * (1 << UNIFORM_BITS) != int(uniform * (1 << UNIFORM_BITS)):
        raise ValueError(f"uniform must be a multiple of 2**-{UNIFORM_BITS} in [0, 1), got {uniform!r}")
    if table is None:
        table = weight_table(policy.temperature)
    table = table.reshape(-1)
    if policy.presence_penalty != 0:
        if seen is None or tuple(seen.shape) != (LANES,) or seen.dtype != torch.bool:
            raise ValueError(f"a presence policy needs the row's seen mask (bool [{LANES}]), got {seen}")
        values = values.clone()
        values[seen] = values[seen] - torch.tensor(policy.presence_penalty, dtype=torch.float32)
    by_id = torch.argsort(ids)  # the ids of a row are distinct: ascending id first, then the stable sort by value
    order = by_id[torch.sort(values[by_id], descending=True, stable=True).indices]
    s_sorted = values[order]
    g_sorted = ids[order]
    delta = (s_sorted - s_sorted[0]) * -float(DELTA_GRID)  # fp32: -(s_j - s_max) * 1024, exact scaling
    index = torch.clamp(torch.floor(delta), 0.0, float(TABLE_SIZE - 1)).to(torch.int64)
    weights = table[index].to(torch.int64)
    weights[policy.top_k :] = 0
    inclusive = torch.cumsum(weights, dim=0)
    exclusive = inclusive - weights
    total_top_k = inclusive[policy.top_k - 1]
    tau = _sfpu_mul(torch.tensor(policy.top_p), total_top_k)
    k_p = int((exclusive.to(torch.float32) <= tau).sum())
    # W is 0 at and beyond top_k and min_weight >= 1: the second count is min(top_k, K_min_p, K_positive).
    k_m = int((weights.to(torch.float32) >= torch.tensor(policy.min_weight, dtype=torch.float32)).sum())
    kept = min(k_p, k_m)
    if kept < 1:
        raise AssertionError(f"kept prefix {kept}: lane 0 always has weight {WEIGHT_ONE}")
    total_kept = inclusive[kept - 1]
    theta = _sfpu_mul(torch.tensor(uniform), total_kept)
    below = int((inclusive[:kept].to(torch.float32) <= theta).sum())
    lane = min(below, kept - 1)
    return Qwen38DeviceSample(int(g_sorted[lane]), kept, lane, float(uniform))


def candidate_row_lanes(row: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The value lanes and the id lanes (lane order) of one host candidate row ``SAMPLING_CANDIDATE_ROW_SHAPE``."""

    packs = row.reshape(TP_SIZE, 2, SAMPLING_CANDIDATES_PER_DEVICE)
    return packs[:, 0].reshape(-1).to(torch.float32), packs[:, 1].reshape(-1).to(torch.int64)


def _value_index_row() -> torch.Tensor:
    return (torch.arange(LANES) // SAMPLING_CANDIDATES_PER_DEVICE * 2 * SAMPLING_CANDIDATES_PER_DEVICE) + (
        torch.arange(LANES) % SAMPLING_CANDIDATES_PER_DEVICE
    )


# --- the device side ------------------------------------------------------------------------------------------


@dataclass
class Qwen38TTNNDeviceSamplerConstants:
    """The composite's replicated device tensors: constants built once, per-request scalars, the per-step draw.

    Every tensor is replicated (the row is replicated after the all-gather).  The host writes ``weight_table``
    when a request's temperature differs from the resident one, the request scalars at request start and
    ``uniform`` once per step; all of them before the TAIL that reads them (cq 0 order).
    """

    value_index: Any  # uint32 TILE [128]: the row lanes holding the values
    id_index: Any  # uint32 TILE [128]: the row lanes holding the global ids
    zero_index: Any  # uint32 TILE [1]
    lane_row: Any  # fp32 TILE [128]: 0..127
    lane_col: Any  # fp32 TILE [128,1]: 0..127
    upper_ones: Any  # fp32 TILE [128,128]: [i <= j]
    unit_column: Any  # fp32 TILE TOKEN_ROW_SHAPE: 1 at column 0
    weight_table: Any  # fp32 ROW_MAJOR [65536]
    top_k_last: Any  # uint32 TILE [1]: top_k - 1
    top_p: Any  # fp32 TILE [1]
    min_weight: Any  # fp32 TILE [1]: max(min_p * 2**18, 1): the kept lanes reach it (min-p and the positive cut)
    greedy_flag: Any  # fp32 TILE [1]: 1 selects the greedy row
    sampled_flag: Any  # fp32 TILE [1]: 1 - greedy_flag
    top_k_mask: Any  # fp32 TILE [128]: 1 below top_k
    uniform: Any  # fp32 TILE [1]: the step's draw
    compute_config: Any
    resident_temperature: float | None = None
    resident_policy: Qwen38DeviceSamplerPolicy | None = None
    mesh_device: Any = field(default=None, repr=False)

    @classmethod
    def build(cls, mesh_device, mesh_contract: Qwen38MeshContract) -> "Qwen38TTNNDeviceSamplerConstants":
        replicate = replicate_tensor_2d_mesh_mapper(mesh_device)

        def upload(host: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                host,
                dtype=dtype,
                layout=layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate,
            )

        lanes = torch.arange(LANES, dtype=torch.float32)
        rows = lanes.reshape(LANES, 1)
        unit_column = torch.zeros(TOKEN_ROW_SHAPE, dtype=torch.float32)
        unit_column[..., 0] = 1.0
        result = cls(
            value_index=upload(_value_index_row().to(torch.int32).reshape(LANE_SHAPE), ttnn.uint32),
            id_index=upload(
                (_value_index_row() + SAMPLING_CANDIDATES_PER_DEVICE).to(torch.int32).reshape(LANE_SHAPE), ttnn.uint32
            ),
            zero_index=upload(torch.zeros(SCALAR_SHAPE, dtype=torch.int32), ttnn.uint32),
            lane_row=upload(lanes.reshape(LANE_SHAPE), ttnn.float32),
            lane_col=upload(rows.reshape(1, 1, LANES, 1), ttnn.float32),
            upper_ones=upload((rows <= lanes).to(torch.float32).reshape(MATRIX_SHAPE), ttnn.float32),
            unit_column=upload(unit_column, ttnn.float32),
            weight_table=upload(weight_table(1.0), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
            top_k_last=upload(torch.full(SCALAR_SHAPE, MAX_TOP_K - 1, dtype=torch.int32), ttnn.uint32),
            top_p=upload(torch.ones(SCALAR_SHAPE), ttnn.float32),
            min_weight=upload(torch.ones(SCALAR_SHAPE), ttnn.float32),
            greedy_flag=upload(torch.ones(SCALAR_SHAPE), ttnn.float32),
            sampled_flag=upload(torch.zeros(SCALAR_SHAPE), ttnn.float32),
            top_k_mask=upload(torch.ones(LANE_SHAPE), ttnn.float32),
            uniform=upload(torch.zeros(SCALAR_SHAPE), ttnn.float32),
            compute_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
            resident_temperature=1.0,
            resident_policy=Qwen38DeviceSamplerPolicy.greedy_policy(),
            mesh_device=mesh_device,
        )
        result.validate(mesh_contract)
        return result

    _TENSORS = (
        ("value_index", LANE_SHAPE, "uint32", "TILE_LAYOUT"),
        ("id_index", LANE_SHAPE, "uint32", "TILE_LAYOUT"),
        ("zero_index", SCALAR_SHAPE, "uint32", "TILE_LAYOUT"),
        ("lane_row", LANE_SHAPE, "float32", "TILE_LAYOUT"),
        ("lane_col", (1, 1, LANES, 1), "float32", "TILE_LAYOUT"),
        ("upper_ones", MATRIX_SHAPE, "float32", "TILE_LAYOUT"),
        ("unit_column", TOKEN_ROW_SHAPE, "float32", "TILE_LAYOUT"),
        ("weight_table", TABLE_SHAPE, "float32", "ROW_MAJOR_LAYOUT"),
        ("top_k_last", SCALAR_SHAPE, "uint32", "TILE_LAYOUT"),
        ("top_p", SCALAR_SHAPE, "float32", "TILE_LAYOUT"),
        ("min_weight", SCALAR_SHAPE, "float32", "TILE_LAYOUT"),
        ("greedy_flag", SCALAR_SHAPE, "float32", "TILE_LAYOUT"),
        ("sampled_flag", SCALAR_SHAPE, "float32", "TILE_LAYOUT"),
        ("top_k_mask", LANE_SHAPE, "float32", "TILE_LAYOUT"),
        ("uniform", SCALAR_SHAPE, "float32", "TILE_LAYOUT"),
    )
    # The host rewrites these between replays (request start, every step): marked corruptible with the token row.
    HOST_WRITTEN = (
        "weight_table",
        "top_k_last",
        "top_p",
        "min_weight",
        "greedy_flag",
        "sampled_flag",
        "top_k_mask",
        "uniform",
    )

    def tensors(self) -> tuple[Any, ...]:
        return tuple(getattr(self, name) for name, _shape_, _dtype, _layout in self._TENSORS)

    def validate(self, mesh_contract: Qwen38MeshContract) -> None:
        for name, shape, dtype, layout in self._TENSORS:
            tensor = getattr(self, name)
            expected_dtype, expected_layout = getattr(ttnn, dtype), getattr(ttnn, layout)
            if _shape(tensor) != shape or tensor.dtype != expected_dtype or tensor.layout != expected_layout:
                raise RuntimeError(f"{name} must be {dtype} {layout} {shape}, got {_metadata(tensor)}")
            mesh_contract.validate_tensor(tensor, placement=TensorPlacement.REPLICATED)

    def _write(self, name: str, host: torch.Tensor, dtype, layout=None) -> None:
        image = ttnn.from_torch(
            host,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT if layout is None else layout,
            mesh_mapper=replicate_tensor_2d_mesh_mapper(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(image, getattr(self, name))

    def write_policy(self, policy: Qwen38DeviceSamplerPolicy) -> dict[str, Any]:
        """Request start: the scalars, the top-k mask and (when the temperature changes) the table; eager writes,
        ordered before the request's first TAIL.  Returns what was written (the ledger)."""

        if not isinstance(policy, Qwen38DeviceSamplerPolicy):
            raise TypeError("policy must be a Qwen38DeviceSamplerPolicy")
        written = {"table": False}
        if not policy.greedy and policy.temperature != self.resident_temperature:
            self._write("weight_table", weight_table(policy.temperature), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
            self.resident_temperature = policy.temperature
            written["table"] = True
        if self.resident_policy is None or policy != self.resident_policy:
            mask = (torch.arange(LANES) < policy.top_k).to(torch.float32).reshape(LANE_SHAPE)
            self._write("top_k_last", torch.full(SCALAR_SHAPE, policy.top_k - 1, dtype=torch.int32), ttnn.uint32)
            self._write("top_p", torch.tensor(policy.top_p, dtype=torch.float32).reshape(SCALAR_SHAPE), ttnn.float32)
            self._write("min_weight", torch.full(SCALAR_SHAPE, policy.min_weight), ttnn.float32)
            self._write("greedy_flag", torch.full(SCALAR_SHAPE, 1.0 if policy.greedy else 0.0), ttnn.float32)
            self._write("sampled_flag", torch.full(SCALAR_SHAPE, 0.0 if policy.greedy else 1.0), ttnn.float32)
            self._write("top_k_mask", mask, ttnn.float32)
            self.resident_policy = policy
            written["scalars"] = True
        else:
            written["scalars"] = False
        return written

    def write_uniform(self, uniform: float) -> None:
        """One step's draw (``n * 2**-24``, exact in fp32) into the persistent scalar."""

        if not 0 <= uniform < 1:
            raise ValueError(f"uniform must lie in [0, 1), got {uniform!r}")
        self._write("uniform", torch.full(SCALAR_SHAPE, uniform, dtype=torch.float32), ttnn.float32)

    def mark_corruptible(self) -> None:
        for name in self.HOST_WRITTEN:
            acknowledge_corruptible(getattr(self, name))

    def release(self) -> None:
        _deallocate(*self.tensors())


def sample_on_device(row: Any, greedy_token_row: Any, constants: Qwen38TTNNDeviceSamplerConstants):
    """TAIL's last ops after the greedy resolve and the candidate row: the token row the epilogue copies out.

    ``row`` is the replicated FP32 ROW_MAJOR ``SAMPLING_CANDIDATE_ROW_SHAPE`` of ``sampling_candidates``,
    ``greedy_token_row`` the FP32 TILE ``TOKEN_ROW_SHAPE`` of ``resolve_greedy_on_device``.  Returns an FP32 TILE
    ``TOKEN_ROW_SHAPE`` whose column 0 is the chosen id (the module docstring's steps 1-7; every metadata check
    prints actual against expected).  Neither input is consumed.
    """

    dram = ttnn.DRAM_MEMORY_CONFIG
    fp32, u32, tile = ttnn.float32, ttnn.uint32, ttnn.TILE_LAYOUT

    def require(tensor, shape, dtype, layout, label) -> None:
        if _shape(tensor) != shape or tensor.dtype != dtype or tensor.layout != layout:
            raise RuntimeError(f"{label} must be {dtype} {layout} {shape}, got {_metadata(tensor)}")

    require(row, SAMPLING_CANDIDATE_ROW_SHAPE, fp32, ttnn.ROW_MAJOR_LAYOUT, "candidate row")
    require(greedy_token_row, TOKEN_ROW_SHAPE, fp32, tile, "greedy token row")

    # 1. Split the row into the value lanes and the id lanes (32-bit copies).
    row_tile = ttnn.to_layout(row, tile, memory_config=dram)
    values = ttnn.gather(row_tile, 3, constants.value_index, memory_config=dram)
    ids = ttnn.gather(row_tile, 3, constants.id_index, memory_config=dram)
    _deallocate(row_tile)
    require(values, LANE_SHAPE, fp32, tile, "value lanes")

    # 2. Rank: value descending, global id ascending (the host samplers' tie rule); the inverse permutation; the
    # sorted views.
    value_col = ttnn.transpose(values, 2, 3, memory_config=dram)
    greater = ttnn.gt(value_col, values, dtype=fp32, memory_config=dram)  # [i, j] = s_i > s_j
    equal = ttnn.eq(value_col, values, dtype=fp32, memory_config=dram)
    _deallocate(value_col)
    require(greater, MATRIX_SHAPE, fp32, tile, "comparison matrix")
    id_col = ttnn.transpose(ids, 2, 3, memory_config=dram)
    lower_id = ttnn.lt(id_col, ids, dtype=fp32, memory_config=dram)  # [i, j] = g_i < g_j
    _deallocate(id_col)
    equal_before = ttnn.multiply(equal, lower_id, memory_config=dram)  # [i, j] = s_i == s_j and g_i < g_j
    _deallocate(equal, lower_id)
    precedes = ttnn.add(greater, equal_before, memory_config=dram)
    _deallocate(greater, equal_before)
    rank = ttnn.sum(precedes, dim=2, keepdim=True, memory_config=dram)  # 0/1 sum over i: rank_j in 0..127
    _deallocate(precedes)
    require(rank, LANE_SHAPE, fp32, tile, "rank row")
    rank_col = ttnn.transpose(rank, 2, 3, memory_config=dram)
    _deallocate(rank)
    at_position = ttnn.eq(rank_col, constants.lane_row, dtype=fp32, memory_config=dram)  # [j, p] = rank_j == p
    _deallocate(rank_col)
    lane_at_position = ttnn.multiply(at_position, constants.lane_col, memory_config=dram)  # j where rank_j == p
    _deallocate(at_position)
    inverse = ttnn.sum(lane_at_position, dim=2, keepdim=True, memory_config=dram)  # inverse[p] = lane of rank p
    _deallocate(lane_at_position)
    inverse_index = ttnn.typecast(inverse, u32, memory_config=dram)
    _deallocate(inverse)
    values_sorted = ttnn.gather(values, 3, inverse_index, memory_config=dram)
    ids_sorted = ttnn.gather(ids, 3, inverse_index, memory_config=dram)
    _deallocate(values, ids, inverse_index)

    # 3. Weights from the temperature table: index floor(1024 * (s_max - s_j)), clamped; top-k mask.
    maximum = ttnn.gather(values_sorted, 3, constants.zero_index, memory_config=dram)
    below_maximum = ttnn.subtract(values_sorted, maximum, memory_config=dram)  # s_j - s_max <= 0 (fp32 RNE)
    _deallocate(maximum)
    scaled = ttnn.multiply(below_maximum, -float(DELTA_GRID), memory_config=dram)  # exact: a power of two
    _deallocate(below_maximum)
    index_real = ttnn.floor(scaled, memory_config=dram)
    _deallocate(scaled)
    index_clamped = ttnn.clip(index_real, 0.0, float(TABLE_SIZE - 1), memory_config=dram)
    _deallocate(index_real)
    table_index = ttnn.typecast(index_clamped, u32, memory_config=dram)
    _deallocate(index_clamped)
    table_index_rm = ttnn.to_layout(table_index, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    _deallocate(table_index)
    weights_rm = ttnn.gather(constants.weight_table, 3, table_index_rm, memory_config=dram)
    _deallocate(table_index_rm)
    require(weights_rm, LANE_SHAPE, fp32, ttnn.ROW_MAJOR_LAYOUT, "table weights")
    weights_raw = ttnn.to_layout(weights_rm, tile, memory_config=dram)
    _deallocate(weights_rm)
    weights = ttnn.multiply(weights_raw, constants.top_k_mask, memory_config=dram)
    _deallocate(weights_raw)

    # 4. Exact prefix sums: 11-bit high and 8-bit low halves through the FPU, recombined on the SFPU.
    high_scaled = ttnn.multiply(weights, 2.0**-WEIGHT_LOW_BITS, memory_config=dram)
    high = ttnn.floor(high_scaled, memory_config=dram)  # <= 2**10
    _deallocate(high_scaled)
    high_back = ttnn.multiply(high, float(1 << WEIGHT_LOW_BITS), memory_config=dram)
    low = ttnn.subtract(weights, high_back, memory_config=dram)  # < 2**8
    _deallocate(high_back)
    prefix_high = ttnn.matmul(
        high, constants.upper_ones, memory_config=dram, compute_kernel_config=constants.compute_config
    )
    prefix_low = ttnn.matmul(
        low, constants.upper_ones, memory_config=dram, compute_kernel_config=constants.compute_config
    )
    _deallocate(high, low)
    require(prefix_high, LANE_SHAPE, fp32, tile, "prefix sums")
    prefix_high_back = ttnn.multiply(prefix_high, float(1 << WEIGHT_LOW_BITS), memory_config=dram)
    _deallocate(prefix_high)
    inclusive = ttnn.add(prefix_high_back, prefix_low, memory_config=dram)  # <= 32 * 2**18 = 2**23
    _deallocate(prefix_high_back, prefix_low)
    exclusive = ttnn.subtract(inclusive, weights, memory_config=dram)

    # 5. Filters as prefix lengths: K = min(#(C_ex <= tau), #(W >= max(m, 1))) >= 1; the top-k mask already zeroes
    # the lanes at or beyond top_k, so the second count is at most top_k and covers the positive-weight cut too.
    total_top_k = ttnn.gather(inclusive, 3, constants.top_k_last, memory_config=dram)
    tau = ttnn.multiply(total_top_k, constants.top_p, memory_config=dram)  # RNE site 1
    _deallocate(total_top_k)
    within_top_p = ttnn.le(exclusive, tau, dtype=fp32, memory_config=dram)
    _deallocate(exclusive, tau)
    kept_top_p = ttnn.sum(within_top_p, dim=3, keepdim=True, memory_config=dram)
    _deallocate(within_top_p)
    within_min_weight = ttnn.ge(weights, constants.min_weight, dtype=fp32, memory_config=dram)
    _deallocate(weights)
    kept_min_weight = ttnn.sum(within_min_weight, dim=3, keepdim=True, memory_config=dram)
    _deallocate(within_min_weight)
    kept = ttnn.minimum(kept_top_p, kept_min_weight, memory_config=dram)
    _deallocate(kept_top_p, kept_min_weight)
    require(kept, SCALAR_SHAPE, fp32, tile, "kept prefix length")
    last_kept = ttnn.subtract(kept, 1.0, memory_config=dram)
    _deallocate(kept)
    last_kept_index = ttnn.typecast(last_kept, u32, memory_config=dram)
    total_kept = ttnn.gather(inclusive, 3, last_kept_index, memory_config=dram)
    _deallocate(last_kept_index)

    # 6. The draw: the count of inclusive prefixes at or below theta = u * S, clamped to the last kept lane (a
    # lane beyond K has C_in >= S >= theta, equal only when theta == S, when the clamp decides anyway).
    theta = ttnn.multiply(total_kept, constants.uniform, memory_config=dram)  # RNE site 2
    _deallocate(total_kept)
    below_theta = ttnn.le(inclusive, theta, dtype=fp32, memory_config=dram)
    _deallocate(inclusive, theta)
    count = ttnn.sum(below_theta, dim=3, keepdim=True, memory_config=dram)
    _deallocate(below_theta)
    chosen = ttnn.minimum(count, last_kept, memory_config=dram)
    _deallocate(count, last_kept)
    chosen_index = ttnn.typecast(chosen, u32, memory_config=dram)
    _deallocate(chosen)
    sampled = ttnn.gather(ids_sorted, 3, chosen_index, memory_config=dram)
    _deallocate(ids_sorted, values_sorted, chosen_index)
    require(sampled, SCALAR_SHAPE, fp32, tile, "sampled id")

    # 7. Select: the greedy row when the flag is set (bitwise), the splatted sampled id otherwise (0/1 products).
    sampled_row = ttnn.multiply(constants.unit_column, sampled, memory_config=dram)
    _deallocate(sampled)
    greedy_part = ttnn.multiply(greedy_token_row, constants.greedy_flag, memory_config=dram)
    sampled_part = ttnn.multiply(sampled_row, constants.sampled_flag, memory_config=dram)
    _deallocate(sampled_row)
    token_row = ttnn.add(greedy_part, sampled_part, memory_config=dram)
    _deallocate(greedy_part, sampled_part)
    require(token_row, TOKEN_ROW_SHAPE, fp32, tile, "sampled token row")
    return token_row


__all__ = [
    "DELTA_GRID",
    "DEVICE_OP_COUNT",
    "LANES",
    "MAX_TEMPERATURE",
    "MAX_TOP_K",
    "TABLE_SIZE",
    "UNIFORM_BITS",
    "WEIGHT_LOW_BITS",
    "WEIGHT_ONE",
    "Qwen38DeviceSample",
    "Qwen38DeviceSamplerPolicy",
    "Qwen38TTNNDeviceSamplerConstants",
    "UniformStream",
    "candidate_row_lanes",
    "device_sampler_reference",
    "sample_on_device",
    "weight_table",
]
