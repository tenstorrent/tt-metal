# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Unary scalars are runtime args, not compiled into the kernel (#62051).

Each case runs one op chain with several scalar values on one device. Only the first value may compile a program;
the others must hit it. The output for every value must match, bit for bit, what that value gives on an empty
program cache, so a scalar that is not refreshed on a cache hit fails. Outputs of distinct values must differ, or the
case could not tell a stale scalar from a fresh one.

That the generated code is the same for every value is checked per op on the host by the UnaryRuntimeScalars gtests;
these cases cover the device path: one scalar, two scalars, a compiled-in param next to a runtime one, a chain, and
integer dtypes.
"""

import pytest
import torch

import ttnn

Op = ttnn.UnaryOpType


def P(op_type, *params):
    return ttnn.UnaryWithParam(op_type, *(float(p) for p in params))


def _input(dtype):
    torch.manual_seed(0)
    if dtype in (ttnn.bfloat16, ttnn.float32):
        x = torch.rand((64, 128), dtype=torch.float32) * 8 - 4
        return x.to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)
    if dtype == ttnn.int32:
        return torch.randint(-1000, 1000, (64, 128), dtype=torch.int32)
    return torch.randint(0, 65536, (64, 128), dtype=torch.int32)


def _run(device, torch_input, dtype, chain):
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(ttnn.unary_chain(tt_input, chain))
    bits = {torch.float32: torch.int32, torch.bfloat16: torch.int16}
    return out.view(bits[out.dtype]) if out.dtype in bits else out


def _check_runtime_scalars(device, dtype, chains):
    torch_input = _input(dtype)
    device.enable_program_cache()
    device.clear_program_cache()
    warm = []
    entries_after_first = None
    for i, chain in enumerate(chains):
        warm.append(_run(device, torch_input, dtype, chain))
        entries = device.num_program_cache_entries()
        if i == 0:
            entries_after_first = entries
        else:
            assert entries == entries_after_first, f"value {i} compiled a new program ({entries} entries)"

    for i, chain in enumerate(chains):
        device.clear_program_cache()
        cold = _run(device, torch_input, dtype, chain)
        assert torch.equal(warm[i], cold), f"value {i} differs between a cache hit and a fresh program"

    for i in range(1, len(chains)):
        assert not torch.equal(warm[0], warm[i]), f"values 0 and {i} give the same output; the case is not testing"


CASES = {
    "relu_max": [[P(Op.RELU_MAX, v)] for v in (0.5, 2.0, 3.0)],
    "hardtanh": [[P(Op.HARDTANH, lo, hi)] for lo, hi in ((-1.0, 1.0), (-0.5, 2.0), (0.0, 0.25))],
    # The rounding mode stays compiled in; only the value is a runtime arg.
    "rdiv": [[P(Op.RDIV, v, 0.0)] for v in (1.25, 2.75, -3.5)],
    # Two scalar ops in one chain take consecutive runtime args.
    "add_then_mul": [
        [P(Op.ADD_UNARY_SFPU, a), P(Op.MUL_UNARY_SFPU, m)] for a, m in ((1.0, 2.0), (1.0, 3.0), (-2.0, 2.0))
    ],
    "bitwise_and": [[P(Op.BITWISE_AND, v)] for v in (0x0F, 0xF0, 0x3C)],
    "add": [[P(Op.ADD_UNARY_SFPU, v)] for v in (1, 77, 5000)],
}


@pytest.mark.parametrize(
    "case,dtype",
    [
        *[
            (case, dtype)
            for case in ("relu_max", "hardtanh", "rdiv", "add_then_mul")
            for dtype in (ttnn.bfloat16, ttnn.float32)
        ],
        ("bitwise_and", ttnn.int32),
        ("bitwise_and", ttnn.uint16),
        ("add", ttnn.int32),
        ("add", ttnn.uint32),
    ],
)
def test_runtime_scalars_reuse_program(device, case, dtype):
    _check_runtime_scalars(device, dtype, CASES[case])


def test_sharded_relu_max_packer_threshold_stays_in_cache_key(device):
    """Sharded BF16 relu_max(x, 6) runs in the packer with the threshold compiled in; other values take the SFPU."""
    sharded = ttnn.create_sharded_memory_config(
        shape=(32, 256),
        core_grid=ttnn.CoreGrid(y=1, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    torch.manual_seed(0)
    torch_input = (torch.rand((256, 256), dtype=torch.float32) * 16 - 8).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded
    )
    device.enable_program_cache()
    device.clear_program_cache()
    outputs = []
    entries = []
    for threshold in (6.0, 4.0, 5.0, 6.0):
        outputs.append(ttnn.relu_max(tt_input, threshold, memory_config=sharded))
        entries.append(device.num_program_cache_entries())
    assert entries[1] == entries[0] + 1, "relu_max(x, 4) reused the packer program built for relu_max(x, 6)"
    assert entries[3] == entries[1], "relu_max(x, 5) or a repeated relu_max(x, 6) compiled a new program"
    for threshold, tt_out in zip((6.0, 4.0, 5.0, 6.0), outputs):
        expected = torch.clamp(torch_input, min=0.0, max=threshold)
        assert torch.equal(ttnn.to_torch(tt_out), expected), f"threshold {threshold}"


@pytest.mark.parametrize("ttnn_op", ["clamp", "fill"])
def test_signed_zero_scalar_shares_program(device, ttnn_op):
    """+0.0 and -0.0 run on one program, each returning its own sign of zero."""
    tt_in = ttnn.from_torch(
        torch.full((64, 64), -1.0, dtype=torch.float32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    device.clear_program_cache()
    entries_after_first = None
    for scalar in (0.0, -0.0, 0.0, -0.0):
        tt_out = ttnn.clamp(tt_in, scalar, 1.0) if ttnn_op == "clamp" else ttnn.fill(tt_in, scalar)
        if entries_after_first is None:
            entries_after_first = device.num_program_cache_entries()
        assert device.num_program_cache_entries() == entries_after_first, f"scalar {scalar} compiled a new program"
        expected = torch.full((64, 64), scalar, dtype=torch.float32)
        assert torch.equal(ttnn.to_torch(tt_out).view(torch.int32), expected.view(torch.int32)), f"scalar {scalar}"
