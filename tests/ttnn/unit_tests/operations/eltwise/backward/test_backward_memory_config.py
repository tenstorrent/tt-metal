# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""A gradient the caller did not preallocate still lands where the caller asked.

The backward ops allocate their outputs with empty_like(input), which inherits the
input's memory config and ignores the memory_config argument. div_bw went further:
with the request unmet, its second gradient came back as -inf.

When no memory_config is passed, a result follows the operand. clamp_bw and lgamma_bw take a grad and
an input of the same shape and follow grad. prod_bw and repeat_bw take a grad of another shape
(reduced for prod_bw, repeated for repeat_bw) and return a tensor shaped like input.
"""

import pytest
import torch
import ttnn

SHAPE = (1, 1, 32, 32)
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG

# op name, number of tensor arguments, trailing scalar arguments
OPS = [
    ("abs_bw", 2, []),
    ("exp_bw", 2, []),
    ("neg_bw", 2, []),
    ("sqrt_bw", 2, []),
    ("silu_bw", 2, []),
    ("rsqrt_bw", 2, []),
    ("frac_bw", 2, []),
    ("pow_bw", 2, [2.0]),
    ("lgamma_bw", 2, []),
    ("prod_bw", 2, []),
    # grad has the input's shape here. The sum runs over a size-1 dim.
    ("repeat_bw", 2, [[1, 2, 1, 1]]),
    ("clamp_bw", 2, [0.5]),
    ("clamp_bw", 2, [0.5, 2.0]),
    # min and max as tensors
    ("clamp_bw", 4, []),
    ("add_bw", 3, []),
    ("sub_bw", 3, []),
    ("mul_bw", 3, []),
    ("div_bw", 3, []),
    ("rsub_bw", 3, []),
    ("assign_bw", 3, []),
    ("addalpha_bw", 3, [2.0]),
    ("subalpha_bw", 3, [2.0]),
    ("addcmul_bw", 4, [2.0]),
    ("addcdiv_bw", 4, [2.0]),
]

VALUES = (1.5, 2.5, 0.5, 3.0)

# Ops whose first gradient is grad itself, returned without a dispatch. #55613 settled that
# those keep grad's own memory config rather than the requested one, since honouring the
# request would mean materialising a copy where none is needed, so output 0 is out of scope
# here. Every other output of the same op is not.
PASSTHROUGH_FIRST_OUTPUT = {"frac_bw", "addcmul_bw", "addcdiv_bw"}


def _t(v, device, memory_config):
    return ttnn.from_torch(
        torch.full(SHAPE, v, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def _call(op, arity, scalars, device, memory_config):
    args = [_t(VALUES[i], device, ttnn.DRAM_MEMORY_CONFIG) for i in range(arity)]
    kwargs = {} if memory_config is None else {"memory_config": memory_config}
    out = getattr(ttnn, op)(*args, *scalars, **kwargs)
    out = out if isinstance(out, (list, tuple)) else [out]
    return [x for x in out if x is not None]


@pytest.mark.parametrize("op, arity, scalars", OPS)
def test_gradients_land_in_the_requested_memory(device, op, arity, scalars):
    outputs = _call(op, arity, scalars, device, ttnn.L1_MEMORY_CONFIG)
    assert outputs, f"{op} returned no gradient"
    for i, out in enumerate(outputs):
        if i == 0 and op in PASSTHROUGH_FIRST_OUTPUT:
            continue
        assert out.memory_config().buffer_type == ttnn.BufferType.L1, f"{op} output {i} is in {out.memory_config()}"


@pytest.mark.parametrize("op, arity, scalars", OPS)
def test_requesting_a_memory_config_does_not_move_any_value(device, op, arity, scalars):
    default = _call(op, arity, scalars, device, None)
    asked = _call(op, arity, scalars, device, ttnn.L1_MEMORY_CONFIG)
    for i, (a, b) in enumerate(zip(default, asked)):
        x, y = ttnn.to_torch(a).float(), ttnn.to_torch(b).float()
        moved = ((x != y) & ~(torch.isnan(x) & torch.isnan(y))).sum().item()
        assert moved == 0, f"{op} output {i}: {moved} of {x.numel()} values move when L1 is requested"


INTERLEAVED_PAIRS = [(L1, DRAM), (DRAM, L1), (L1, L1), (DRAM, DRAM)]


def _tensor(device, memory_config, shape=SHAPE, value=None):
    data = torch.rand(shape) + 0.5 if value is None else torch.full(shape, value)
    return ttnn.from_torch(
        data.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )


def _sharded(shape, strategy, cores):
    return ttnn.create_sharded_memory_config(
        shape,
        core_grid=ttnn.CoreGrid(x=cores, y=1),
        strategy=strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


CLAMP_CASES = [
    ("float min only", lambda g, x, b: ttnn.clamp_bw(g, x, min=0.8)[0]),
    ("float max only", lambda g, x, b: ttnn.clamp_bw(g, x, max=1.2)[0]),
    ("float both", lambda g, x, b: ttnn.clamp_bw(g, x, min=0.8, max=1.2)[0]),
    ("tensor min only", lambda g, x, b: ttnn.clamp_bw(g, x, min=b)[0]),
    ("tensor max only", lambda g, x, b: ttnn.clamp_bw(g, x, max=b)[0]),
    ("tensor both", lambda g, x, b: ttnn.clamp_bw(g, x, min=b, max=b)[0]),
]

# prod_bw's grad has the reduced shape. Dims 1 and 0 are size 1 and grad has input shape.
PROD_DIMS = {-1: (1, 1, 32, 1), -2: (1, 1, 1, 32), 1: SHAPE, 0: SHAPE}


@pytest.mark.parametrize("grad_config, input_config", INTERLEAVED_PAIRS)
@pytest.mark.parametrize("name, call", CLAMP_CASES)
def test_clamp_bw_unset_memory_config_follows_grad(device, grad_config, input_config, name, call):
    grad = _tensor(device, grad_config)
    inp = _tensor(device, input_config)
    bound = _tensor(device, input_config, value=1.0)
    assert call(grad, inp, bound).memory_config() == grad_config


@pytest.mark.parametrize("grad_config, input_config", INTERLEAVED_PAIRS)
def test_lgamma_bw_unset_memory_config_follows_grad(device, grad_config, input_config):
    grad = _tensor(device, grad_config)
    inp = _tensor(device, input_config)
    assert ttnn.lgamma_bw(grad, inp)[0].memory_config() == grad_config


@pytest.mark.parametrize("grad_config, input_config", INTERLEAVED_PAIRS)
@pytest.mark.parametrize("dim", [None, *PROD_DIMS], ids=["all dims", *[f"dim {d}" for d in PROD_DIMS]])
def test_prod_bw_unset_memory_config_follows_the_input(device, grad_config, input_config, dim):
    grad = _tensor(device, grad_config, shape=SHAPE if dim is None else PROD_DIMS[dim])
    inp = _tensor(device, input_config)
    result = ttnn.prod_bw(grad, inp) if dim is None else ttnn.prod_bw(grad, inp, dim=dim)
    assert result[0].memory_config() == input_config


@pytest.mark.parametrize("dim", PROD_DIMS, ids=[f"dim {d}" for d in PROD_DIMS])
def test_prod_bw_requested_memory_config_is_honoured(device, dim):
    grad = _tensor(device, DRAM, shape=PROD_DIMS[dim])
    inp = _tensor(device, DRAM)
    assert ttnn.prod_bw(grad, inp, dim=dim, memory_config=L1)[0].memory_config() == L1


@pytest.mark.parametrize("grad_config, input_config", INTERLEAVED_PAIRS)
@pytest.mark.parametrize("repeat", [[1, 2, 1, 1], [1, 0, 1, 1]], ids=["sum over dim 1", "zero repeat"])
def test_repeat_bw_unset_memory_config_follows_the_input(device, grad_config, input_config, repeat):
    # The repeat sum goes through a preallocated output, which would default to DRAM if left unset.
    grad = _tensor(device, grad_config, shape=(1, 2, 32, 32))
    inp = _tensor(device, input_config)
    assert ttnn.repeat_bw(grad, inp, repeat)[0].memory_config() == input_config


# Sharded: grad and input are 64x64 height-sharded over two cores or interleaved in DRAM.
SHARDED_SHAPE = (1, 1, 64, 64)
SHARDED_PAIRS = [("sharded", "dram"), ("dram", "sharded"), ("sharded", "sharded")]


def _sharded_or_dram(kind, shape, cores):
    return _sharded(shape, ttnn.ShardStrategy.HEIGHT, cores) if kind == "sharded" else DRAM


@pytest.mark.parametrize("grad_kind, input_kind", SHARDED_PAIRS)
@pytest.mark.parametrize("name, call", CLAMP_CASES)
def test_clamp_bw_sharded_unset_memory_config_follows_grad(device, grad_kind, input_kind, name, call):
    grad = _tensor(device, _sharded_or_dram(grad_kind, SHARDED_SHAPE, 2), shape=SHARDED_SHAPE)
    input_config = _sharded_or_dram(input_kind, SHARDED_SHAPE, 2)
    inp = _tensor(device, input_config, shape=SHARDED_SHAPE)
    bound = _tensor(device, input_config, shape=SHARDED_SHAPE, value=1.0)
    assert call(grad, inp, bound).memory_config() == grad.memory_config()


@pytest.mark.parametrize("grad_kind, input_kind", [("sharded", "dram"), ("sharded", "sharded")])
def test_lgamma_bw_sharded_grad_unset_memory_config_follows_grad(device, grad_kind, input_kind):
    grad = _tensor(device, _sharded_or_dram(grad_kind, SHARDED_SHAPE, 2), shape=SHARDED_SHAPE)
    inp = _tensor(device, _sharded_or_dram(input_kind, SHARDED_SHAPE, 2), shape=SHARDED_SHAPE)
    assert ttnn.lgamma_bw(grad, inp)[0].memory_config() == grad.memory_config()


@pytest.mark.parametrize("grad_kind, input_kind", SHARDED_PAIRS)
def test_repeat_bw_sharded_unset_memory_config_follows_the_input(device, grad_kind, input_kind):
    grad_shape, input_shape = (1, 2, 32, 32), SHAPE
    grad_data = torch.randn(grad_shape).bfloat16()
    grad = ttnn.from_torch(
        grad_data,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded_or_dram(grad_kind, grad_shape, 2),
    )
    inp = _tensor(device, _sharded_or_dram(input_kind, input_shape, 1), shape=input_shape)
    result = ttnn.repeat_bw(grad, inp, [1, 2, 1, 1])[0]
    assert result.memory_config() == inp.memory_config()
    expected = grad_data.float().sum(dim=1, keepdim=True)
    assert torch.allclose(ttnn.to_torch(result).float(), expected, rtol=0.05, atol=0.1)


def test_repeat_bw_width_sharded_grad_unset_memory_config_follows_the_input(device):
    # The grad's shard shape was built for the larger grad tensor and does not fit the summed result.
    # Placing the result by the grad's config fails here. It has to follow the input.
    grad_shape, input_shape = (1, 2, 32, 128), (1, 1, 32, 128)
    grad_data = torch.randn(grad_shape).bfloat16()
    grad = ttnn.from_torch(
        grad_data,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded(grad_shape, ttnn.ShardStrategy.WIDTH, 4),
    )
    inp = _tensor(device, DRAM, shape=input_shape)
    result = ttnn.repeat_bw(grad, inp, [1, 2, 1, 1])[0]
    assert result.memory_config() == DRAM
    expected = grad_data.float().sum(dim=1, keepdim=True)
    assert torch.allclose(ttnn.to_torch(result).float(), expected, rtol=0.05, atol=0.1)


@pytest.mark.parametrize("dim", [None, -1, -2])
def test_prod_bw_sharded_grad_follows_the_input(device, dim):
    # With dim -1 or -2, the grad has the reduced shape. Its shard spec was built for a smaller tensor than the
    # input-shaped result. A shape this small is not tile sized. The spec is built on the tile-padded shape.
    # With all dims, the grad is sliced to one value and the slice keeps the grad's shard config.
    if dim is None:
        grad_shape, grad_config = SHARDED_SHAPE, _sharded(SHARDED_SHAPE, ttnn.ShardStrategy.HEIGHT, 2)
    elif dim == -1:
        grad_shape, grad_config = (1, 1, 64, 1), _sharded((1, 1, 64, 32), ttnn.ShardStrategy.HEIGHT, 2)
    else:
        grad_shape, grad_config = (1, 1, 1, 64), _sharded((1, 1, 32, 64), ttnn.ShardStrategy.WIDTH, 2)
    grad_data = (torch.rand(grad_shape) + 0.5).bfloat16()
    # Values near 1 keep the product of 4096 elements from underflowing to zero.
    input_data = (1 + (torch.rand(SHARDED_SHAPE) - 0.5) * 0.02).bfloat16()
    inp = ttnn.from_torch(input_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)

    def prod_bw(memory_config):
        grad = ttnn.from_torch(
            grad_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )
        return ttnn.prod_bw(grad, inp, dim=dim)[0]

    result = prod_bw(grad_config)
    assert result.memory_config() == inp.memory_config()
    # Moving the grad into shards changes where it lives.
    assert torch.equal(ttnn.to_torch(result), ttnn.to_torch(prod_bw(DRAM)))
