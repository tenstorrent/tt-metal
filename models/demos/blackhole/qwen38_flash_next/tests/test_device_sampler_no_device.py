# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The device sampler: its op sequence on a torch model of the device == the host reference bitwise; the argmax
short-circuit; ties; invariants; the reference against the host sampler; the RNG; the static op-list pin."""

from __future__ import annotations

import inspect
import re
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.ttnn import device_sampler as ds
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import (
    SAMPLING_CANDIDATE_ROW_SHAPE,
    SAMPLING_CANDIDATES_PER_DEVICE,
    TOKEN_ROW_SHAPE,
    TP_SIZE,
    VOCAB_SIZE,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import (
    Qwen38CandidateFallback,
    Qwen38CandidateRow,
    Qwen38SamplingParameters,
    UniformStream,
    sample_candidates,
    sample_full_vocabulary,
)

K = SAMPLING_CANDIDATES_PER_DEVICE
PEAK_IDS = (95_859, 248_044, 248_319, 7, 62_086, 131_071)
TF32_SIGNIFICANT_BITS = 11
TEMPERATURES = (0.1, 0.5, 0.7, 1.0, 1.5, 4.0)
TOP_PS = (0.5, 0.8, 0.95, 1.0)
TOP_KS = (1, 5, 20, 32)
MIN_PS = (0.0, 0.05, 1.0)
EDGE_UNIFORMS = (0.0, 2.0**-24, 1.0 - 2.0**-24)


def _logits(seed: int, *, peaks: int = 24, scale: float = 2.5) -> torch.Tensor:
    """A bf16 logit row at model scale: a wide body plus a few dozen peaks (some at the probe ids)."""

    generator = torch.Generator().manual_seed(seed)
    row = torch.randn(VOCAB_SIZE, generator=generator) * scale + 2.0
    ids = torch.randperm(VOCAB_SIZE, generator=generator)[:peaks].tolist() + list(PEAK_IDS)
    row[ids] = 16.0 + torch.rand(len(ids), generator=generator) * 8.0
    return row.to(torch.bfloat16)


def _host_row(bf16: torch.Tensor) -> torch.Tensor:
    return Qwen38CandidateRow.emulate(bf16).to_host_row()


def _tf32(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    _, exponent = torch.frexp(x)
    quantum = torch.ldexp(torch.ones_like(x), exponent - TF32_SIGNIFICANT_BITS)
    return torch.round(x / quantum) * quantum


# --- a torch model of the device ops the composite uses ------------------------------------------------------
#
# Every op is modelled by its exact semantics.  The FPU stages (sum, matmul) assert their inputs are TF32-exact
# (the design's invariant: 0/1 lanes or integers below 2**11) and their integer results below 2**24; the SFPU
# eltwise ops are fp32 torch arithmetic (RNE); gathers and layout changes are copies.


class FakeTensor:
    def __init__(self, data: torch.Tensor, dtype: str, layout: str) -> None:
        self.data = data
        self.dtype = dtype
        self.layout = layout
        self.allocated = True

    @property
    def shape(self):
        return tuple(self.data.shape)


class FakeMeshContract:
    def __init__(self) -> None:
        self.validated: list[str] = []

    def validate_tensor(self, tensor, *, placement, shard_dim=None) -> None:
        assert isinstance(tensor, FakeTensor) and tensor.allocated
        self.validated.append(str(placement))


class FakeTTNN:
    """The subset of ttnn the composite calls, on FakeTensors; records the op sequence."""

    float32 = "float32"
    uint32 = "uint32"
    bfloat16 = "bfloat16"
    TILE_LAYOUT = "TILE_LAYOUT"
    ROW_MAJOR_LAYOUT = "ROW_MAJOR_LAYOUT"
    DRAM_MEMORY_CONFIG = "DRAM_MEMORY_CONFIG"
    MathFidelity = SimpleNamespace(HiFi4="HiFi4")

    def __init__(self) -> None:
        self.ops: list[str] = []
        self.fpu_inputs: list[torch.Tensor] = []
        self.corruptible: list[FakeTensor] = []

    @staticmethod
    def WormholeComputeKernelConfig(**fields):
        return SimpleNamespace(**fields)

    def from_torch(self, host, dtype, layout, device=None, memory_config=None, mesh_mapper=None):
        if dtype == "uint32":
            assert host.dtype in (torch.int32, torch.int64) and bool((host >= 0).all())
            data = host.to(torch.int64).clone()
        else:
            assert dtype == "float32" and host.dtype == torch.float32
            data = host.clone()
        return FakeTensor(data, dtype, layout)

    def copy_host_to_device_tensor(self, host: FakeTensor, device: FakeTensor) -> None:
        assert (host.shape, host.dtype, host.layout) == (device.shape, device.dtype, device.layout)
        device.data.copy_(host.data)
        self.ops.append("host_write")

    def mark_corruptible(self, tensor: FakeTensor) -> None:
        self.corruptible.append(tensor)

    def deallocate(self, tensor: FakeTensor) -> None:
        assert tensor.allocated, "double deallocate"
        tensor.allocated = False

    # -- helpers --
    def _live(self, *tensors: FakeTensor) -> None:
        for tensor in tensors:
            assert isinstance(tensor, FakeTensor) and tensor.allocated, "op on a deallocated tensor"

    def _binary(self, name, a, b, *, dtype=None, memory_config=None, op):
        self.ops.append(name)
        self._live(a)
        if isinstance(b, FakeTensor):
            self._live(b)
            assert a.layout == b.layout and a.dtype == b.dtype == "float32", (name, a.layout, b.layout)
            right = b.data
        else:
            right = torch.tensor(float(b), dtype=torch.float32)
        result = op(a.data, right).to(torch.float32)
        return FakeTensor(result, "float32", a.layout)

    def to_layout(self, tensor, layout, memory_config=None):
        self.ops.append("to_layout")
        self._live(tensor)
        return FakeTensor(tensor.data.clone(), tensor.dtype, layout)

    def gather(self, tensor, dim, index, memory_config=None):
        self.ops.append("gather")
        self._live(tensor, index)
        assert dim == 3 and index.dtype == "uint32" and tensor.layout == index.layout, (tensor.layout, index.layout)
        assert index.shape[:3] == tensor.shape[:3] == (1, 1, 1)
        assert bool((index.data >= 0).all()) and bool((index.data < tensor.shape[3]).all()), "gather index range"
        if tensor.layout == "ROW_MAJOR_LAYOUT":
            assert index.shape[3] <= 1920, "ROW_MAJOR gather index rows above 1,920 lanes are wrong on silicon"
        return FakeTensor(torch.gather(tensor.data, 3, index.data), tensor.dtype, tensor.layout)

    def transpose(self, tensor, a, b, memory_config=None):
        self.ops.append("transpose")
        self._live(tensor)
        assert (a, b) == (2, 3)
        return FakeTensor(tensor.data.transpose(2, 3).contiguous(), tensor.dtype, tensor.layout)

    def gt(self, a, b, dtype=None, memory_config=None):
        return self._binary("gt", a, b, op=lambda x, y: (x > y))

    def eq(self, a, b, dtype=None, memory_config=None):
        return self._binary("eq", a, b, op=lambda x, y: (x == y))

    def le(self, a, b, dtype=None, memory_config=None):
        return self._binary("le", a, b, op=lambda x, y: (x <= y))

    def ge(self, a, b, dtype=None, memory_config=None):
        return self._binary("ge", a, b, op=lambda x, y: (x >= y))

    def lt(self, a, b, dtype=None, memory_config=None):
        return self._binary("lt", a, b, op=lambda x, y: (x < y))

    def multiply(self, a, b, memory_config=None):
        return self._binary("multiply", a, b, op=lambda x, y: x * y)

    def add(self, a, b, memory_config=None):
        return self._binary("add", a, b, op=lambda x, y: x + y)

    def subtract(self, a, b, memory_config=None):
        return self._binary("subtract", a, b, op=lambda x, y: x - y)

    def minimum(self, a, b, memory_config=None):
        return self._binary("minimum", a, b, op=torch.minimum)

    def floor(self, tensor, memory_config=None):
        self.ops.append("floor")
        self._live(tensor)
        return FakeTensor(torch.floor(tensor.data), tensor.dtype, tensor.layout)

    def clip(self, tensor, low, high, memory_config=None):
        self.ops.append("clip")
        self._live(tensor)
        return FakeTensor(torch.clamp(tensor.data, float(low), float(high)), tensor.dtype, tensor.layout)

    def typecast(self, tensor, dtype, memory_config=None):
        self.ops.append("typecast")
        self._live(tensor)
        assert dtype == "uint32" and tensor.dtype == "float32"
        data = tensor.data
        assert bool((data == torch.floor(data)).all()) and bool((data >= 0).all()) and bool((data < 2**32).all())
        return FakeTensor(data.to(torch.int64), "uint32", tensor.layout)

    def _fpu_inputs(self, *tensors: torch.Tensor) -> None:
        for data in tensors:
            self.fpu_inputs.append(data)
            assert torch.equal(_tf32(data), data), "an FPU stage read a value that is not TF32-exact"

    def sum(self, tensor, dim, keepdim=True, memory_config=None):
        self.ops.append("sum")
        self._live(tensor)
        assert dim in (2, 3) and keepdim and tensor.dtype == "float32" and tensor.layout == "TILE_LAYOUT"
        self._fpu_inputs(tensor.data)
        exact = tensor.data.to(torch.float64).sum(dim=dim, keepdim=True)
        assert bool((exact.abs() < 2**24).all()) and torch.equal(exact, torch.round(exact))
        return FakeTensor(exact.to(torch.float32), "float32", tensor.layout)

    def matmul(self, a, b, memory_config=None, compute_kernel_config=None):
        self.ops.append("matmul")
        self._live(a, b)
        assert compute_kernel_config is not None and compute_kernel_config.fp32_dest_acc_en
        assert compute_kernel_config.math_fidelity == "HiFi4" and not compute_kernel_config.packer_l1_acc
        self._fpu_inputs(a.data, b.data)
        exact = torch.matmul(a.data.to(torch.float64), b.data.to(torch.float64))
        assert bool((exact.abs() < 2**24).all()) and torch.equal(exact, torch.round(exact))
        return FakeTensor(exact.to(torch.float32), "float32", a.layout)


@pytest.fixture
def fake(monkeypatch):
    """The composite's module bound to the torch model; constants built through it."""

    fake_ttnn = FakeTTNN()
    monkeypatch.setattr(ds, "ttnn", fake_ttnn)
    monkeypatch.setattr(ds, "acknowledge_corruptible", fake_ttnn.mark_corruptible)
    monkeypatch.setattr(ds, "replicate_tensor_2d_mesh_mapper", lambda mesh: None)
    # The composite's releases go through the fake too: an op on a released tensor fails.
    monkeypatch.setattr(ds, "_deallocate", lambda *tensors: [fake_ttnn.deallocate(t) for t in tensors if t is not None])
    contract = FakeMeshContract()
    constants = ds.Qwen38TTNNDeviceSamplerConstants.build("mesh", contract)
    assert constants.validate(contract) is None
    return SimpleNamespace(ttnn=fake_ttnn, constants=constants, contract=contract)


def _run_composite(fake, host_row: torch.Tensor, greedy_id: int, policy, uniform: float) -> tuple[int, list[str]]:
    row = FakeTensor(host_row.reshape(SAMPLING_CANDIDATE_ROW_SHAPE).clone(), "float32", "ROW_MAJOR_LAYOUT")
    greedy_row = torch.zeros(TOKEN_ROW_SHAPE, dtype=torch.float32)
    greedy_row[..., 0] = float(greedy_id)
    fake.constants.write_policy(policy)
    fake.constants.write_uniform(uniform)
    before = len(fake.ttnn.ops)
    token_row = ds.sample_on_device(row, FakeTensor(greedy_row, "float32", "TILE_LAYOUT"), fake.constants)
    ops = [op for op in fake.ttnn.ops[before:]]
    assert token_row.shape == TOKEN_ROW_SHAPE and token_row.dtype == "float32" and token_row.layout == "TILE_LAYOUT"
    assert torch.equal(token_row.data[..., 1:], torch.zeros(1, 1, 1, 31))
    value = float(token_row.data[0, 0, 0, 0])
    assert value == int(value) and 0 <= value < VOCAB_SIZE
    return int(value), ops


def _policies():
    for temperature in TEMPERATURES:
        for top_p in TOP_PS:
            for top_k in TOP_KS:
                for min_p in MIN_PS:
                    yield ds.Qwen38DeviceSamplerPolicy(temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p)


# --- T1: emulation == reference bitwise over the parameter matrix, rows and draws -------------------------------


def test_composite_on_the_device_model_equals_the_reference_bitwise(fake) -> None:
    stream = UniformStream(20260904)
    cases = 0
    tokens = Counter()
    for seed in range(6):
        host_row = _host_row(_logits(seed, scale=2.5 + seed))
        values, ids = ds.candidate_row_lanes(host_row)
        for policy in _policies():
            table = ds.weight_table(policy.temperature)
            for uniform in (stream.next_uniform(), stream.next_uniform(), *EDGE_UNIFORMS):
                expected = ds.device_sampler_reference(values, ids, policy, uniform, table=table)
                actual, ops = _run_composite(fake, host_row, 0, policy, uniform)
                assert actual == expected.token_id, (seed, policy, uniform, actual, expected)
                assert len(ops) == ds.DEVICE_OP_COUNT, (len(ops), ops)
                cases += 1
                tokens[actual] += 1
    assert cases == 6 * len(TEMPERATURES) * len(TOP_PS) * len(TOP_KS) * len(MIN_PS) * 5
    assert len(tokens) > 20  # the draws move the token


def test_many_draws_on_one_row_follow_the_reference_and_cover_the_kept_set(fake) -> None:
    host_row = _host_row(_logits(11))
    values, ids = ds.candidate_row_lanes(host_row)
    policy = ds.Qwen38DeviceSamplerPolicy(temperature=1.0, top_k=20, top_p=0.95, min_p=0.0)
    stream = UniformStream(7)
    chosen = Counter()
    for _ in range(200):
        uniform = stream.next_uniform()
        expected = ds.device_sampler_reference(values, ids, policy, uniform)
        actual, _ops = _run_composite(fake, host_row, 0, policy, uniform)
        assert actual == expected.token_id
        chosen[actual] += 1
    assert len(chosen) >= 3 and chosen.most_common(1)[0][0] == int(ids[int(torch.argmax(values))])


# --- T2: the argmax short-circuit ------------------------------------------------------------------------------


def test_greedy_flag_selects_the_resolves_row_bitwise_whatever_the_parameters(fake) -> None:
    for seed, greedy_id in ((1, 95_859), (2, 0), (3, 248_319), (4, 62_086)):
        bf16 = _logits(seed)
        bf16[95_859] = bf16[248_044] = 40.0  # a cross-shard tie: the resolve's owner rule decides, not the sampler
        host_row = _host_row(bf16)
        for policy in list(_policies())[::37]:
            greedy = ds.Qwen38DeviceSamplerPolicy(
                temperature=policy.temperature, top_k=policy.top_k, top_p=policy.top_p, min_p=policy.min_p, greedy=True
            )
            actual, ops = _run_composite(fake, host_row, greedy_id, greedy, 0.5)
            assert actual == greedy_id and len(ops) == ds.DEVICE_OP_COUNT
    assert ds.Qwen38DeviceSamplerPolicy.greedy_policy().greedy
    assert ds.Qwen38DeviceSamplerPolicy.from_parameters(Qwen38SamplingParameters.greedy()).greedy


# --- T3: ties --------------------------------------------------------------------------------------------------


def test_ties_are_broken_by_lane_order_and_exchange_only_equal_logits(fake) -> None:
    bf16 = _logits(5, peaks=0)
    bf16[:K] = 25.0  # shard 0: a whole tie group at the top, straddling top_k and top_p
    bf16[62_080 : 62_080 + 8] = 25.0  # shard 1: the same value
    bf16[124_160 : 124_160 + 40] = 24.0  # shard 2: a tie at its 32nd value (topk picks members)
    host_row = _host_row(bf16)
    values, ids = ds.candidate_row_lanes(host_row)
    stream = UniformStream(3)
    fallbacks = 0
    for policy in (
        ds.Qwen38DeviceSamplerPolicy(temperature=1.0, top_k=20, top_p=0.95, min_p=0.0),
        ds.Qwen38DeviceSamplerPolicy(temperature=0.5, top_k=32, top_p=0.5, min_p=0.0),
        ds.Qwen38DeviceSamplerPolicy(temperature=4.0, top_k=5, top_p=1.0, min_p=0.5),
    ):
        for _ in range(40):
            uniform = stream.next_uniform()
            expected = ds.device_sampler_reference(values, ids, policy, uniform)
            actual, _ops = _run_composite(fake, host_row, 0, policy, uniform)
            assert actual == expected.token_id
            # The same tie rule as the host samplers (lowest global id): the candidate sampler on the same row
            # agrees; against the full vocabulary the token has the same bf16 logit (the read set at shard 2's
            # 32nd value is torch.topk's choice among the tie).
            parameters = Qwen38SamplingParameters(
                temperature=policy.temperature,
                top_p=policy.top_p,
                top_k=policy.top_k,
                presence_penalty=0.0,
                seed=3,
                min_p=policy.min_p,
            )
            try:
                candidate = sample_candidates(
                    Qwen38CandidateRow.from_host_row(host_row), parameters, generator=_one_shot(uniform)
                )
            except Qwen38CandidateFallback:
                fallbacks += 1  # the host guard refuses a kept set at the shard floor; the device samples the row
            else:
                assert candidate.token_id == actual
            full = sample_full_vocabulary(bf16.to(torch.float32), parameters, generator=_one_shot(uniform))
            assert float(bf16[actual]) == float(bf16[full.token_id])
    assert fallbacks > 0  # the host guard refuses these rows (kept set at the shard floor); the device samples them
    # A whole row of equal values: sorted lane n holds the n-th lowest id of the row (ascending ids: lane order).
    flat = torch.full((VOCAB_SIZE,), 3.0).to(torch.bfloat16)
    host_row = _host_row(flat)
    values, ids = ds.candidate_row_lanes(host_row)
    policy = ds.Qwen38DeviceSamplerPolicy(temperature=1.0, top_k=32, top_p=1.0, min_p=0.0)
    ascending = torch.sort(ids).values
    for uniform in (0.0, 0.5, 1.0 - 2.0**-24):
        expected = ds.device_sampler_reference(values, ids, policy, uniform)
        assert expected.token_id == int(ascending[expected.sorted_lane]) and expected.kept == 32
        assert _run_composite(fake, host_row, 0, policy, uniform)[0] == expected.token_id


class _OneShot:
    """A UniformStream stand-in that yields one fixed draw (so the host sampler consumes the same ``u``)."""

    def __init__(self, uniform: float, seed: int = 3) -> None:
        self.uniform, self.seed = uniform, seed

    def next_uniform(self) -> float:
        return self.uniform


def _one_shot(uniform: float) -> UniformStream:
    stream = UniformStream(3)
    stream.next_uniform = lambda: uniform  # type: ignore[method-assign]
    return stream


# --- T4: invariants -----------------------------------------------------------------------------------------------


def test_table_and_prefix_invariants(expect_error) -> None:
    for temperature in TEMPERATURES:
        table = ds.weight_table(temperature).reshape(-1)
        assert table.dtype == torch.float32 and table.shape == (ds.TABLE_SIZE,)
        assert float(table[0]) == ds.WEIGHT_ONE and bool((table[1:] <= table[:-1]).all())
        assert torch.equal(table, torch.round(table)) and float(table[-1]) == 0.0
        assert float(table[1]) < ds.WEIGHT_ONE  # the grid resolves one step at the largest temperature
    with expect_error(ValueError, match="temperature"):
        ds.weight_table(4.5)
    host_row = _host_row(_logits(9))
    values, ids = ds.candidate_row_lanes(host_row)
    for policy in list(_policies())[::7]:
        sample = ds.device_sampler_reference(values, ids, policy, 0.25)
        assert 1 <= sample.kept <= policy.top_k and 0 <= sample.sorted_lane < sample.kept
    # min_p 1 keeps the maximum lanes (two here: a tie at the top); top_p tiny and top_k 1 keep the first lane only.
    maximal = int((values == values.max()).sum())
    assert maximal == 2
    for policy, kept in (
        (ds.Qwen38DeviceSamplerPolicy(temperature=1.0, top_k=32, top_p=1.0, min_p=1.0), maximal),
        (ds.Qwen38DeviceSamplerPolicy(temperature=1.0, top_k=32, top_p=2.0**-20, min_p=0.0), 1),
        (ds.Qwen38DeviceSamplerPolicy(temperature=1.0, top_k=1, top_p=1.0, min_p=0.0), 1),
    ):
        sample = ds.device_sampler_reference(values, ids, policy, 1.0 - 2.0**-24)
        assert sample.kept == kept and sample.sorted_lane == kept - 1
        assert float(values[(ids == sample.token_id).nonzero()[0, 0]]) == float(values.max())


def test_policy_routing_from_parameters(expect_error) -> None:
    thinking = Qwen38SamplingParameters.official_thinking(seed=1)
    policy = ds.Qwen38DeviceSamplerPolicy.from_parameters(thinking)
    assert policy == ds.Qwen38DeviceSamplerPolicy(temperature=1.0, top_k=20, top_p=0.95, min_p=0.0)
    # Penalties, temperature above the table's range and top_k 0 keep the host loop.
    assert ds.Qwen38DeviceSamplerPolicy.from_parameters(Qwen38SamplingParameters.official_non_thinking(seed=1)) is None
    base = {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "presence_penalty": 0.0, "seed": 0}
    assert (
        ds.Qwen38DeviceSamplerPolicy.from_parameters(Qwen38SamplingParameters(**{**base, "temperature": 4.5})) is None
    )
    assert ds.Qwen38DeviceSamplerPolicy.from_parameters(Qwen38SamplingParameters(**{**base, "top_k": 0})) is None
    assert ds.Qwen38DeviceSamplerPolicy.from_parameters(Qwen38SamplingParameters(**{**base, "top_k": 33})) is None
    for penalty in ({"frequency_penalty": 0.1}, {"presence_penalty": 1.5}, {"repetition_penalty": 1.1}):
        assert ds.Qwen38DeviceSamplerPolicy.from_parameters(Qwen38SamplingParameters(**{**base, **penalty})) is None
    assert ds.Qwen38DeviceSamplerPolicy.from_parameters(Qwen38SamplingParameters(**{**base, "min_p": 0.1})).min_p == 0.1
    for bad in ({"top_k": 0}, {"top_k": 33}, {"top_p": 0.0}, {"min_p": 1.5}, {"temperature": 0.0}):
        with expect_error(ValueError):
            ds.Qwen38DeviceSamplerPolicy(**{"temperature": 1.0, "top_k": 20, "top_p": 1.0, "min_p": 0.0, **bad})
    assert ds.Qwen38DeviceSamplerPolicy(temperature=1.0, top_k=1, top_p=1.0, min_p=0.5).min_weight == 2.0**17
    assert ds.Qwen38DeviceSamplerPolicy(temperature=1.0, top_k=1, top_p=1.0, min_p=0.0).min_weight == 1.0


# --- T5: the reference against the host sampler under the same draw ----------------------------------------------


def _classify(values, ids, policy, uniform, actual: int, host: int) -> str:
    """Why the device reference and the host sampler chose differently: an equal-logit exchange, or a draw / top-p
    boundary within the weight quantization."""

    if float(values[(ids == actual).nonzero()[0, 0]]) == float(values[(ids == host).nonzero()[0, 0]]):
        return "equal_logit_exchange"
    order = torch.sort(values, descending=True, stable=True).indices
    scaled = (values[order][: policy.top_k] / policy.temperature).to(torch.float64)
    probabilities = torch.softmax(scaled, dim=0)
    cumulative = torch.cumsum(probabilities, dim=0)
    tolerance = 2.0**-9
    if bool((torch.abs(cumulative - uniform) < tolerance).any()):
        return "draw_within_quantization_of_a_boundary"
    if bool((torch.abs(cumulative - policy.top_p) < tolerance).any()):
        return "top_p_boundary_within_quantization"
    if policy.min_p > 0 and bool((torch.abs(probabilities / probabilities[0] - policy.min_p) < tolerance).any()):
        return "min_p_boundary_within_quantization"
    return "unclassified"


def test_reference_agrees_with_the_host_sampler_except_at_quantization_boundaries() -> None:
    classes = Counter()
    total = 0
    stream = UniformStream(20260906)
    for seed in range(8):
        bf16 = _logits(40 + seed, scale=2.5 + seed % 3)
        host_row = _host_row(bf16)
        row = Qwen38CandidateRow.from_host_row(host_row)
        values, ids = ds.candidate_row_lanes(host_row)
        for policy in list(_policies())[::5]:
            parameters = Qwen38SamplingParameters(
                temperature=policy.temperature,
                top_p=policy.top_p,
                top_k=policy.top_k,
                presence_penalty=0.0,
                seed=3,
                min_p=policy.min_p,
            )
            for _ in range(8):
                uniform = stream.next_uniform()
                expected = ds.device_sampler_reference(values, ids, policy, uniform)
                host = sample_candidates(row, parameters, generator=_one_shot(uniform))
                assert host.uniform == uniform
                total += 1
                if host.token_id == expected.token_id:
                    classes["agree"] += 1
                else:
                    classes[_classify(values, ids, policy, uniform, expected.token_id, host.token_id)] += 1
    # The same read row, the same tie rule (lowest global id): the two differ only at the quantization boundaries.
    assert classes["unclassified"] == 0 and classes["equal_logit_exchange"] == 0, classes
    assert classes["agree"] / total >= 0.995, classes
    print(f"DEVICE_SAMPLER_VS_HOST total={total} classes={dict(classes)}")


def test_kept_distribution_is_close_to_the_filtered_softmax() -> None:
    worst = 0.0
    for seed in range(20):
        host_row = _host_row(_logits(60 + seed, scale=2.0 + seed % 4))
        values, ids = ds.candidate_row_lanes(host_row)
        for policy in (
            ds.Qwen38DeviceSamplerPolicy(temperature=1.0, top_k=20, top_p=0.95, min_p=0.0),
            ds.Qwen38DeviceSamplerPolicy(temperature=0.7, top_k=20, top_p=0.8, min_p=0.0),
        ):
            table = ds.weight_table(policy.temperature).reshape(-1)
            order = torch.sort(values, descending=True, stable=True).indices
            sorted_values = values[order]
            index = torch.clamp(torch.floor((sorted_values[0] - sorted_values) * 1024), 0, ds.TABLE_SIZE - 1).long()
            weights = table[index].to(torch.float64)
            weights[policy.top_k :] = 0
            exact = torch.softmax((sorted_values[: policy.top_k] / policy.temperature).to(torch.float64), 0)
            kept = ds.device_sampler_reference(values, ids, policy, 0.5).kept
            device = weights[:kept] / weights[:kept].sum()
            reference = exact[:kept] / exact[:kept].sum()
            worst = max(worst, float((device - reference).abs().sum()))
    assert worst < 1e-3, worst


# --- T6: the RNG ------------------------------------------------------------------------------------------------


def test_uniform_stream_vectors_exactness_and_independence(expect_error) -> None:
    stream = UniformStream(0)
    first = [stream.next_bits() for _ in range(3)]
    # splitmix64 of seed 0: the first outputs' top 24 bits (0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F).
    assert first == [0xE220A8397B1DCDAF >> 40, 0x6E789E6AA1B965F4 >> 40, 0x06C45D188009454F >> 40]
    assert stream.draws == 3 and stream.seed == 0 and stream.initial_seed() == 0
    stream = UniformStream(20260904)
    draws = [stream.next_uniform() for _ in range(1000)]
    assert all(0 <= u < 1 and u * 2**24 == int(u * 2**24) for u in draws)
    assert all(torch.tensor(u, dtype=torch.float32).item() == u for u in draws)  # exact in fp32
    assert len(set(draws)) > 990
    again = UniformStream(20260904)
    assert [again.next_uniform() for _ in range(1000)] == draws
    assert UniformStream(20260905).next_uniform() != draws[0]
    for bad in (-1, 2**64, 1.0, True):
        with expect_error(ValueError):
            UniformStream(bad)
    # The host samplers consume the stream in place of a torch generator: one draw per sampled token.
    row = Qwen38CandidateRow.emulate(_logits(2))
    parameters = Qwen38SamplingParameters.official_thinking(seed=5)
    stream = UniformStream(5)
    sample = sample_candidates(row, parameters, generator=stream)
    assert stream.draws == 1 and sample.uniform == UniformStream(5).next_uniform()
    with expect_error(ValueError, match="stream seed"):
        sample_candidates(row, parameters, generator=UniformStream(6))


# --- T8: the static op-list pin -----------------------------------------------------------------------------------

FORBIDDEN_IN_BODY = (
    "to_torch(",
    "from_torch(",
    "synchronize",
    "topk_large_indices",
    "ttnn.exp(",
    "ttnn.softmax(",
    "ttnn.cumsum(",
    "ttnn.sampling(",
    "ttnn.argmax(",
    "ttnn.sort(",
    "ttnn.topk(",
    "ttnn.max(",
    "ttnn.where(",
)


def test_composite_op_list_is_pinned_and_every_fpu_stage_reads_small_integers() -> None:
    body = inspect.getsource(ds.sample_on_device).split('"""')[2]
    ops = re.findall(r"ttnn\.(\w+)\(", body)
    assert len(ops) == ds.DEVICE_OP_COUNT == 57, (len(ops), ops)
    counts = Counter(ops)
    assert counts == {
        "to_layout": 3,
        "gather": 9,
        "transpose": 3,
        "gt": 1,
        "eq": 2,
        "lt": 1,
        "multiply": 12,
        "add": 3,
        "sum": 5,
        "typecast": 4,
        "subtract": 4,
        "floor": 2,
        "clip": 1,
        "matmul": 2,
        "le": 2,
        "ge": 1,
        "minimum": 2,
    }, counts
    for forbidden in FORBIDDEN_IN_BODY:
        assert forbidden not in body, forbidden
    # The FPU stages: both matmuls on the split halves against the constant triangle; every sum over 0/1 lanes
    # (the rank and the filters) or the lane-index one-hot (below 128).
    assert re.findall(r"ttnn\.matmul\(\s*(\w+), constants\.upper_ones", body) == ["high", "low"]
    assert re.findall(r"ttnn\.sum\((\w+), dim=(\d)", body) == [
        ("precedes", "2"),
        ("lane_at_position", "2"),
        ("within_top_p", "3"),
        ("within_min_weight", "3"),
        ("below_theta", "3"),
    ]
    # The table lookup is the one ROW_MAJOR gather (a TILE gather costs a pad fill and a slice besides, but the
    # composite with ROW_MAJOR gathers replayed at 3.0 ms against 0.98 ms on the pinned runtime).
    assert len(re.findall(r"ttnn\.gather\(constants\.weight_table, 3, ", body)) == 1
    assert "compute_kernel_config=constants.compute_config" in body
    # The two rounding sites are single multiplies against the host-written scalars.
    assert (
        "ttnn.multiply(total_top_k, constants.top_p" in body and "ttnn.multiply(total_kept, constants.uniform" in body
    )
    # The greedy resolve and the candidate row do not know the sampler exists.
    from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import Qwen38TTNNLMHead

    for name in ("greedy_candidates", "resolve_greedy_on_device", "sampling_candidates"):
        assert "device_sampler" not in inspect.getsource(getattr(Qwen38TTNNLMHead, name))
    assert ds.LANES == TP_SIZE * K == 128 and ds.TABLE_SIZE == 65536 and ds.WEIGHT_ONE == 2**18


def test_constants_are_replicated_and_the_host_written_ones_are_marked_corruptible(expect_error, fake) -> None:
    constants = fake.constants
    assert all("REPLICATED" in placement for placement in fake.contract.validated)
    assert len(fake.contract.validated) == 2 * len(constants._TENSORS) == 2 * 15  # build validates, the fixture again
    constants.mark_corruptible()
    assert [t for t in fake.ttnn.corruptible] == [getattr(constants, name) for name in constants.HOST_WRITTEN]
    # A policy write rewrites the scalars and the table only when the temperature changes; a repeat writes nothing.
    fake.ttnn.ops.clear()
    policy = ds.Qwen38DeviceSamplerPolicy(temperature=0.7, top_k=20, top_p=0.8, min_p=0.0)
    assert constants.write_policy(policy) == {"table": True, "scalars": True}
    assert fake.ttnn.ops.count("host_write") == 7
    assert constants.write_policy(policy) == {"table": False, "scalars": False}
    assert constants.write_policy(ds.Qwen38DeviceSamplerPolicy.greedy_policy()) == {"table": False, "scalars": True}
    assert (
        float(constants.greedy_flag.data.reshape(-1)[0]) == 1.0
        and float(constants.sampled_flag.data.reshape(-1)[0]) == 0.0
    )
    assert constants.top_k_mask.data.reshape(-1).sum() == 32
    constants.write_uniform(0.75)
    assert float(constants.uniform.data.reshape(-1)[0]) == 0.75
    with expect_error(ValueError):
        constants.write_uniform(1.0)
    constants.release()
    assert not any(tensor.allocated for tensor in constants.tensors())
