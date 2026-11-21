# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 0: the fused SiLU on the first operand, `memory_config=`, width-sharded operands (Qwen).

- the two model calls of Qwen's gated MLP, `multiply(a, b, input_tensor_a_activations=[SILU], dtype=bfloat16,
  memory_config=<the operands' config>)`: values against silu(a) * b in float64, and the result's placement;
- the `sharded_local` regime: chosen on the host for width-sharded operands, the rings borrowed from the shards
  (zero copy), b's per-image row picked by offset under a leading-dim broadcast, bit-exact without the SiLU;
- refusals: an activation other than [SILU], a memory config other than the first operand's;
- one device session: another program before, tt-metal's to_layout after.
"""

import pytest
import torch

import ttnn
from eval import memory
from eval.metrics import DEFAULT_TOLERANCES
from ttnn.operations.multiply import multiply, validate
from ttnn.operations.multiply.multiply import _dispatch

SILU = [ttnn.UnaryOpType.SILU]
WIDTH = memory.Sharded("L1", "width")


def _width_config(device, shard_shape, cores):
    """A width-sharded config as the model builds it: `shard_shape` on the first `cores` cores, row-major."""
    return ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.num_cores_to_corerangeset(cores, device.compute_with_storage_grid_size(), row_wise=True),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _put(t, device, placement=ttnn.DRAM_MEMORY_CONFIG):
    placed = memory.place(placement, tuple(t.shape), ttnn.TILE_LAYOUT, device)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=placed)


def _pair(a_shape, b_shape, seed=0):
    torch.manual_seed(seed)
    return torch.randn(a_shape).to(torch.bfloat16), torch.randn(b_shape).to(torch.bfloat16)


def _expect_silu_product(got, ta, tb):
    """Against silu(a) * b in float64, at the shared bfloat16 / fp32-DEST tolerance (spec NUMERIC modes)."""
    a = ta.double()
    expected = a * torch.sigmoid(a) * tb.double()
    got = got.double()
    assert got.shape == expected.shape, (got.shape, expected.shape)
    pcc_min, rms_max = DEFAULT_TOLERANCES[(ttnn.bfloat16, True)]
    pcc = torch.corrcoef(torch.stack([got.flatten(), expected.flatten()]))[0, 1].item()
    rms = ((got - expected).pow(2).mean().sqrt() / expected.pow(2).mean().sqrt()).item()
    assert pcc >= pcc_min and rms <= rms_max, f"PCC {pcc}, relative RMS {rms}"


def _expect_exact(got, ta, tb):
    expected = (ta.float() * tb.float()).to(torch.bfloat16)
    got = got.to(torch.bfloat16)
    assert got.shape == expected.shape, (got.shape, expected.shape)
    mismatch = (got.float() != expected.float()).sum().item()
    assert mismatch == 0, f"{mismatch} elements differ"


@pytest.mark.parametrize("shape,sharded", [((1, 1, 128, 512), False), ((1, 1, 32, 256), True)], ids=["dram", "width"])
def test_qwen_model_call(device, shape, sharded):
    """The model's call, as traced: values against silu(a) * b in float64, and the result in the operands'
    placement (DRAM interleaved; width-sharded over two cores with shard shape [32, 128])."""
    ta, tb = _pair(shape, shape)
    placement = _width_config(device, (32, 128), 2) if sharded else ttnn.DRAM_MEMORY_CONFIG
    a, b = _put(ta, device, placement), _put(tb, device, placement)
    out = multiply(a, b, input_tensor_a_activations=SILU, dtype=ttnn.bfloat16, memory_config=a.memory_config())
    assert out.dtype == ttnn.bfloat16
    assert out.memory_config() == a.memory_config()
    _expect_silu_product(ttnn.to_torch(out), ta, tb)


@pytest.mark.parametrize(
    "a_shape,b_shape,b_tiles",
    [
        ((1, 1, 32, 256), (1, 1, 32, 256), 1),  # flat: b's shard is a's (8 cores of one tile)
        ((2, 4, 128, 512), (2, 4, 1, 512), 8),  # over_h: one b row per image, 16 cores
        ((2, 4, 64, 64), (1, 4, 1, 64), 4),  # over_h, b broadcast along dim 0: offsets 0,1,2,3,0,1,2,3
        ((1, 4, 8400), (1, 1, 8400), 263),  # YOLO shape: 263 tile columns, one core
    ],
)
@pytest.mark.parametrize("silu", [False, True], ids=["product", "silu"])
def test_sharded_local(device, a_shape, b_shape, b_tiles, silu):
    ta, tb = _pair(a_shape, b_shape)
    a, b = _put(ta, device, WIDTH), _put(tb, device, WIDTH)
    act = SILU if silu else None
    axes = validate(a, b, input_tensor_a_activations=act)
    assert axes["sharding"] == "width"
    out, plan = _dispatch(a, b, axes, None)
    assert plan.regime == "sharded_local"
    assert plan.b_tiles == b_tiles
    # The shard is the per-core block: every core of the shard grid owns its whole resident shard.
    assert len(plan.counts) == a.memory_config().shard_spec.grid.num_cores()
    assert out.memory_config() == a.memory_config()
    got = ttnn.to_torch(out)
    if silu:
        _expect_silu_product(got, ta, tb)
    else:
        _expect_exact(got, ta, tb)


def test_sharded_rings_are_borrowed(device):
    """Zero copy: every ring of the sharded program is borrowed from its tensor's resident shard."""
    from ttnn.operations.multiply.multiply_program_artifacts import create_program_artifacts

    ta, tb = _pair((1, 1, 32, 256), (1, 1, 32, 256))
    a, b = _put(ta, device, WIDTH), _put(tb, device, WIDTH)
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(ttnn.Shape([1, 1, 32, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, a.memory_config()), device
    )
    spec, _, _, _ = create_program_artifacts(
        a, b, out, broadcast="none", compute_kernel_config=ttnn.ComputeConfigDescriptor(), sharding="width"
    )
    assert {d.unique_id: d.borrowed_from for d in spec.dataflow_buffers} == {"in_a": "a", "in_b": "b", "out": "out"}


@pytest.mark.parametrize(
    "a_shape,b_shape", [((1, 1, 64, 128), (1, 1, 64, 128)), ((1, 4, 400), (1, 1, 400))], ids=["flat", "over_h"]
)
def test_silu_interleaved(device, a_shape, b_shape):
    """The SiLU in the interleaved regimes (flat_stream and row_bcast_strips), in L1."""
    ta, tb = _pair(a_shape, b_shape)
    a, b = _put(ta, device, ttnn.L1_MEMORY_CONFIG), _put(tb, device, ttnn.L1_MEMORY_CONFIG)
    _expect_silu_product(ttnn.to_torch(multiply(a, b, input_tensor_a_activations=SILU)), ta, tb)


def test_memory_config_argument(device):
    """`memory_config=` the first operand's config is the same call as leaving it out (bit for bit)."""
    ta, tb = _pair((1, 4, 400), (1, 1, 400))
    a, b = _put(ta, device, ttnn.L1_MEMORY_CONFIG), _put(tb, device, ttnn.L1_MEMORY_CONFIG)
    out = multiply(a, b, memory_config=a.memory_config())
    assert out.memory_config() == ttnn.L1_MEMORY_CONFIG
    _expect_exact(ttnn.to_torch(out), ta, tb)


def test_refusals(device):
    ta, tb = _pair((1, 1, 32, 64), (1, 1, 32, 64))
    a, b = _put(ta, device), _put(tb, device)
    with pytest.raises(NotImplementedError, match="a_activation"):
        multiply(a, b, input_tensor_a_activations=[ttnn.UnaryOpType.RELU])
    with pytest.raises(NotImplementedError, match="memory_argument"):
        multiply(a, b, memory_config=ttnn.L1_MEMORY_CONFIG)  # a is in DRAM
    # Width-sharded operands whose shards do not cover the same columns.
    wa = _put(ta, device, WIDTH)
    wb = _put(tb, device, _width_config(device, (32, 64), 1))
    with pytest.raises(NotImplementedError, match="shard"):
        multiply(wa, wb)


def test_one_device_session(device):
    """Another program before multiply, tt-metal's untilize after it, for the SiLU paths and the sharded
    regime, in one session."""
    for seed, (shape_a, shape_b, placement) in enumerate(
        [
            ((1, 1, 128, 512), (1, 1, 128, 512), ttnn.DRAM_MEMORY_CONFIG),
            ((1, 4, 400), (1, 1, 400), ttnn.DRAM_MEMORY_CONFIG),
            ((1, 1, 32, 256), (1, 1, 32, 256), WIDTH),
        ]
    ):
        ta, tb = _pair(shape_a, shape_b, seed)
        # Another program first: a tilize of a on the device, then the placement the call asks for.
        a_rm = ttnn.from_torch(ta, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        a = ttnn.to_layout(a_rm, ttnn.TILE_LAYOUT)
        if placement is WIDTH:
            a = ttnn.to_memory_config(a, memory.place(WIDTH, shape_a, ttnn.TILE_LAYOUT, device))
        b = _put(tb, device, placement)
        out = multiply(a, b, input_tensor_a_activations=SILU, memory_config=a.memory_config())
        # A program of tt-metal's own after multiply: an untilize with untagged transfers.
        if placement is WIDTH:
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        got = ttnn.to_torch(ttnn.from_device(ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)))
        _expect_silu_product(got, ta, tb)
