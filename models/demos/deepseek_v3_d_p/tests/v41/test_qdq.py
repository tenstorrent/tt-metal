# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 device quantize-dequantize ops (tt/v41/qdq.py) vs the reference kernels, bit for bit.

Contract: given the same bf16 input, each device QDQ returns exactly the bf16 values of the reference
``kernel_cpu`` QDQ (``torch.equal``), row-locally on every device of the mesh, deterministically.

Inputs are built to hit every rounding decision: random groups over ~40 binary decades, all-zero and
tiny groups (amax floors), group amax exactly on / one bf16 ulp around power-of-two scale boundaries,
values exactly on / one ulp around every e4m3 or E2M1 half-way point, e4m3-scale ties and scale
saturation (group-16 FP4). A small hand-derived golden checks the formulas independently of the
reference. bf16 subnormal inputs are excluded (documented limitation, see qdq.py).
"""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41.kernel_cpu import act_quant, fp4_act_quant
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tt.v41 import qdq

SEQ = 2048

MESH_2X4 = pytest.param(
    (2, 4),
    fabric2d_device_params(),
    marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
    id="fabric2d-mesh-2x4",
)

E4M3_VALUES = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
E4M3_VALUES = E4M3_VALUES[torch.isfinite(E4M3_VALUES) & (E4M3_VALUES >= 0)].unique()  # 0, 2^-9 ... 448
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _reference_fp8(x: torch.Tensor) -> torch.Tensor:
    return act_quant(x.clone(), qdq.FP8_GROUP, "ue8m0", inplace=True)


def _reference_fp4_ue8m0(x: torch.Tensor) -> torch.Tensor:
    return fp4_act_quant(x.clone(), qdq.FP4_UE8M0_GROUP, inplace=True)


def _reference_fp4_e4m3(x: torch.Tensor) -> torch.Tensor:
    return fp4_act_quant(x.clone(), qdq.FP4_E4M3_GROUP, inplace=True, scale_dtype=torch.float8_e4m3fn)


# op id -> (device op, reference, group size, format grid of quantized magnitudes, format max)
OPS = {
    "fp8_e4m3_g32_ue8m0": (qdq.fp8_qdq, _reference_fp8, qdq.FP8_GROUP, E4M3_VALUES, 448.0),
    "fp4_e2m1_g32_ue8m0": (qdq.fp4_ue8m0_qdq, _reference_fp4_ue8m0, qdq.FP4_UE8M0_GROUP, E2M1_VALUES, 6.0),
    "fp4_e2m1_g16_e4m3": (qdq.fp4_e4m3_qdq, _reference_fp4_e4m3, qdq.FP4_E4M3_GROUP, E2M1_VALUES, 6.0),
}


def _bf16_neighbors(v: torch.Tensor) -> torch.Tensor:
    """Positive bf16 values v -> [v, next below, next above] (bf16 bit pattern +-1)."""
    bits = v.to(torch.bfloat16).view(torch.int16)
    return torch.cat([bits, bits - 1, bits + 1]).view(torch.bfloat16).float()


def _tie_pool(grid: torch.Tensor) -> torch.Tensor:
    """Grid values, every half-way point between neighbours and their bf16 neighbours (unit scale)."""
    midpoints = (grid[1:] + grid[:-1]) / 2
    return _bf16_neighbors(torch.cat([grid[1:], midpoints])).unique()


def _finish(gen: torch.Generator, values: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
    """Place amax in a random column, clip other magnitudes to amax, apply random signs."""
    n, group = values.shape
    values = torch.minimum(values, amax[:, None])
    values[torch.arange(n), torch.randint(group, (n,), generator=gen)] = amax
    signs = torch.where(torch.rand(n, group, generator=gen) < 0.5, -1.0, 1.0)
    return values * signs


def _random_groups(gen: torch.Generator, n: int, group: int) -> torch.Tensor:
    """Normal values with per-group magnitude 2^U(-30, 12); 1/16 all-zero, 1/16 below the amax floors."""
    scale = torch.exp2(torch.empty(n, 1).uniform_(-30, 12, generator=gen))
    values = torch.randn(n, group, generator=gen) * scale
    kind = torch.randint(16, (n,), generator=gen)
    values[kind == 0] = 0.0
    values[kind == 1] *= 2.0**-100 / scale[kind == 1]  # ~2^-100: below every floor, still normal bf16
    return values


def _tie_groups(gen: torch.Generator, op: str, n: int) -> torch.Tensor:
    """Groups whose scale is exact and known, filled with half-way points of the element format."""
    _, _, group, grid, fmt_max = OPS[op]
    if op == "fp4_e2m1_g16_e4m3":
        scales = E4M3_VALUES[1:][torch.randint(len(E4M3_VALUES) - 1, (n,), generator=gen)]
    else:
        # power-of-two scales; the fp8 exponents stay above the 2^-22 floor, the fp4 ones far from subnormals
        low, high = (-20, 12) if op == "fp8_e4m3_g32_ue8m0" else (-100, 12)
        scales = torch.exp2(torch.randint(low, high, (n,), generator=gen).float())
    # amax = fmt_max * scale sits exactly on the scale boundary (fast_round_scale does not bump; the
    # e4m3 scale amax / 6 is exact).
    amax = fmt_max * scales
    pool = _tie_pool(grid)
    picks = pool[torch.randint(len(pool), (n, group), generator=gen)] * scales[:, None]
    return _finish(gen, picks, amax)


def _scale_boundary_groups(gen: torch.Generator, op: str, n: int) -> torch.Tensor:
    """Groups whose amax is on, or one bf16 ulp around, a scale rounding boundary."""
    _, _, group, _, fmt_max = OPS[op]
    if op == "fp4_e2m1_g16_e4m3":
        # amax / 6 at every e4m3 value and half-way point, including 464 (saturates to 448) and beyond
        points = torch.cat([E4M3_VALUES[1:], (E4M3_VALUES[1:] + E4M3_VALUES[:-1]) / 2, torch.tensor([464.0, 1e4])])
        boundaries = points * 6.0
    else:
        exponents = torch.arange(-20 if op == "fp8_e4m3_g32_ue8m0" else -110, 14).float()
        boundaries = fmt_max * torch.exp2(exponents)
    candidates = _bf16_neighbors(boundaries)
    amax = candidates[torch.randint(len(candidates), (n,), generator=gen)]
    values = torch.rand(n, group, generator=gen) * amax[:, None]
    return _finish(gen, values, amax)


def _input(op: str, kind: str, width: int, seed: int = 0) -> torch.Tensor:
    group = OPS[op][2]
    n = SEQ * width // group
    gen = torch.Generator().manual_seed(seed)
    if kind == "random":
        groups = _random_groups(gen, n, group)
    elif kind == "ties":
        groups = _tie_groups(gen, op, n)
    else:
        groups = _scale_boundary_groups(gen, op, n)
    x = groups.reshape(1, 1, SEQ, width).to(torch.bfloat16)
    assert torch.equal(x.float().reshape(n, group), groups.to(torch.bfloat16).float())
    return x


def _to_mesh(x: torch.Tensor, mesh_device, shard: bool) -> ttnn.Tensor:
    mapper = ttnn.ShardTensorToMesh(mesh_device, dim=2) if shard else ttnn.ReplicateTensorToMesh(mesh_device)
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper)


def _from_mesh_sharded(t: ttnn.Tensor, mesh_device) -> torch.Tensor:
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor, x: torch.Tensor) -> None:
    assert actual.dtype == torch.bfloat16 and actual.shape == expected.shape
    mismatch = actual != expected
    if mismatch.any():
        idx = mismatch.nonzero()[:8].tolist()
        examples = [(i, x[tuple(i)].item(), actual[tuple(i)].item(), expected[tuple(i)].item()) for i in idx]
        raise AssertionError(f"{int(mismatch.sum())} mismatches; (index, input, device, reference): {examples}")


@pytest.mark.parametrize("kind", ["random", "ties", "scale_boundaries"])
@pytest.mark.parametrize("width", [512, 128])
@pytest.mark.parametrize("op", list(OPS))
@pytest.mark.parametrize("mesh_device, device_params", [MESH_2X4], indirect=True)
def test_qdq_matches_reference(mesh_device, device_params, op, width, kind):
    """Sequence-sharded over the mesh (each device quantizes its own rows); exact and deterministic."""
    device_op, reference, *_ = OPS[op]
    x = _input(op, kind, width)
    expected = reference(x)
    tt_x = _to_mesh(x, mesh_device, shard=True)
    first = _from_mesh_sharded(device_op(tt_x), mesh_device)
    _assert_exact(first, expected, x)
    second = _from_mesh_sharded(device_op(tt_x), mesh_device)
    assert torch.equal(first, second), "device QDQ is not deterministic across repeated runs"


@pytest.mark.parametrize("op", list(OPS))
@pytest.mark.parametrize("mesh_device, device_params", [MESH_2X4], indirect=True)
def test_qdq_replicated(mesh_device, device_params, op):
    """A replicated input yields the reference values on every device."""
    device_op, reference, *_ = OPS[op]
    x = _input(op, "random", 512, seed=1)
    expected = reference(x)
    out = device_op(_to_mesh(x, mesh_device, shard=False))
    for device_out in ttnn.get_device_tensors(out):
        _assert_exact(ttnn.to_torch(device_out), expected, x)


# Hand-derived goldens (one group per row, rest zero). Each entry: input group -> expected output group.
GOLDEN = {
    # amax 448 -> scale 2^ceil(log2(1)) = 1. e4m3 near 17: grid 16, 18 (quantum 2) -> tie 17 -> 16 (even
    # mantissa); 19 -> 20. 2^-10 is half of the smallest subnormal 2^-9 -> 0; 3*2^-10 -> 2^-8 (even code 2).
    # 1.0625 -> 1 (quantum 1/8 -> tie -> even).
    "fp8_e4m3_g32_ue8m0": [
        ([448.0, 17.0, 19.0, 2**-10, 3 * 2**-10, 1.0625, -17.0], [448.0, 16.0, 20.0, 0.0, 2**-8, 1.0, -16.0]),
        # amax 450 -> scale 2 (bump); 450/2 = 225 -> 224 (quantum 16 in [128, 256)) -> 448; 3/2 = 1.5 exact
        ([450.0, 3.0], [448.0, 3.0]),
        # all-zero group -> scale floor 2^-22, output 0
        ([0.0], [0.0]),
    ],
    # amax 6 -> scale 1; E2M1 ties 0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5 -> 4.
    # amax 6.0625 -> scale 2: 6.0625/2 = 3.03 -> 3 -> 6; 1.4/2 = 0.7 -> 0.5 -> 1.
    "fp4_e2m1_g32_ue8m0": [
        ([6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.75], [6.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -1.0]),
        ([6.0625, 1.40625], [6.0, 1.0]),
    ],
    # amax 7.5 -> scale e4m3(1.25) = 1.25; 0.9375 = 0.75 s -> 1 s = 1.25; 2.1875 = 1.75 s -> 2 s = 2.5.
    # amax 6.375 -> 6.375/6 = 1.0625, e4m3 tie between 1 and 1.125 -> 1 (even); 6.375/1 clamps to 6.
    # amax 3008 -> 501.3 saturates to scale 448; 3008/448 = 6.7 -> 6 -> 2688; 900/448 = 2.009 -> 2 -> 896.
    # amax bf16(0.01) -> floor 6*2^-9 -> scale 2^-9; 0.01/2^-9 = 5.125 -> 6; 2^-9 = 1 s -> 1 s.
    "fp4_e2m1_g16_e4m3": [
        ([7.5, 0.9375, 2.1875], [7.5, 1.25, 2.5]),
        ([6.375, 1.5], [6.0, 1.5]),
        ([3008.0, 900.0], [2688.0, 896.0]),
        ([0.01, 2**-9], [0.01171875, 2**-9]),
    ],
}


@pytest.mark.parametrize("op", list(OPS))
@pytest.mark.parametrize("mesh_device, device_params", [MESH_2X4], indirect=True)
def test_qdq_hand_golden(mesh_device, device_params, op):
    device_op, reference, group, *_ = OPS[op]
    width, rows = 128, 32 * mesh_device.get_num_devices()
    x = torch.zeros(rows, width)
    expected = torch.zeros(rows, width)
    for row, (inputs, outputs) in enumerate(GOLDEN[op]):
        x[row, : len(inputs)] = torch.tensor(inputs)
        expected[row, : len(outputs)] = torch.tensor(outputs)
    x = x.reshape(1, 1, rows, width).to(torch.bfloat16)
    expected = expected.reshape(1, 1, rows, width).to(torch.bfloat16)
    _assert_exact(reference(x), expected, x)  # the golden agrees with the reference
    _assert_exact(_from_mesh_sharded(device_op(_to_mesh(x, mesh_device, shard=True)), mesh_device), expected, x)


@pytest.mark.parametrize("mesh_device, device_params", [MESH_2X4], indirect=True)
def test_qdq_rejects_unsupported_input(mesh_device, device_params, expect_error):
    x = torch.zeros(1, 1, 32, 48, dtype=torch.bfloat16)
    with expect_error(AssertionError, "not a multiple of the group size"):
        qdq.fp8_qdq(ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device))
    with expect_error(AssertionError, "must be bf16"):
        qdq.fp8_qdq(
            ttnn.from_torch(x[..., :32].float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        )
