# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""div's thread plan and block knobs (op_design.md, Work Distribution: "Tests the plan needs").

- every NEO of the plan computes its own tiles (a correct result alone does not show it);
- the block form with full and ragged blocks, at other block sizes and plans than the default;
- the op in one device session after other programs, and before a tt-metal program (`ttnn.to_layout`).
"""

import pytest
import torch

import ttnn
from ttnn.operations.div import default_compute_kernel_config, div
from ttnn.operations.div.div_program_artifacts import THREAD_PLAN, create_program_artifacts, split_rounds


def _put(t, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=memory_config)


def _inputs(shape, seed=0):
    torch.manual_seed(seed)
    ta = torch.randn(*shape, dtype=torch.bfloat16)
    tb = torch.randn(*shape, dtype=torch.bfloat16)
    tb = torch.where(tb == 0, torch.ones_like(tb), tb)
    return ta, tb


def _run(a, b, *, divisor=None, rounding_mode=None, out_dtype=ttnn.bfloat16, **knobs):
    """div's dispatch with the host knobs (thread plan, block_tiles, buffer_depth, probes) turned."""
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(a.shape, out_dtype, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG), a.device()
    )
    spec, run_args, indices = create_program_artifacts(
        a, b, out, divisor=divisor, rounding_mode=rounding_mode,
        compute_kernel_config=default_compute_kernel_config(), **knobs
    )  # fmt: skip
    tensors = [a] + ([] if b is None else [b]) + [out]
    return ttnn.to_torch(ttnn.generic_op(tensors, spec, run_args, indices)).float()


def _check_div(got, ta, tb):
    expected = (ta.double() / tb.double()).float()
    # one rounding into bfloat16 of the quotient: relative error <= 2^-8
    assert torch.allclose(got, expected, rtol=2**-8, atol=1e-30), (got - expected).abs().max()


def test_each_neo_computes_its_own_tiles(device):
    """With tag_threads, odd NEOs multiply: tile i of a cluster must come from NEO i mod C."""
    C = THREAD_PLAN[1]
    shape = (1, 3, 320, 320)  # 300 tiles: 75 rounds of 4 -> 8 or 12 tiles per cluster on 32 clusters
    ta, tb = _inputs(shape)
    _, counts, starts = split_rounds(300, device.compute_with_storage_grid_size(), C)
    local = {starts[core] + i: i for core, n in counts.items() for i in range(n)}
    assert max(local.values()) >= C, "every NEO must get a tile on some cluster"

    got = _run(_put(ta, device), _put(tb, device), tag_threads=True)
    seen = set()
    for page in range(300):
        ch, rest = divmod(page, 100)
        th, tw = divmod(rest, 10)
        at = (0, ch, slice(th * 32, th * 32 + 32), slice(tw * 32, tw * 32 + 32))
        x, y = ta[at].double(), tb[at].double()
        neo = local[page] % C
        seen.add(neo)
        expected = (x * y if neo & 1 else x / y).float()
        assert torch.allclose(got[at], expected, rtol=2**-8, atol=1e-30), (page, neo)
    assert seen == set(range(C)), seen


@pytest.mark.parametrize("threads", [(1, 1, 1), (1, 4, 1), (2, 4, 2), (4, 4, 2)], ids=lambda p: "R%dC%dW%d" % p)
@pytest.mark.parametrize("block_tiles", [1, 3, 4, 8])
def test_blocks_and_plans(device, threads, block_tiles):
    """900 tiles: 28 or 32 per cluster, so most NEOs end on a short block for block_tiles 3 and 8."""
    ta, tb = _inputs((1, 4, 480, 480))
    _check_div(_run(_put(ta, device), _put(tb, device), threads=threads, block_tiles=block_tiles), ta, tb)


@pytest.mark.parametrize("buffer_depth", [1, 3])
def test_buffer_depth(device, buffer_depth):
    ta, tb = _inputs((1, 4, 480, 480))
    _check_div(_run(_put(ta, device), _put(tb, device), buffer_depth=buffer_depth), ta, tb)


@pytest.mark.parametrize("rounding_mode", [None, "floor", "trunc"])
def test_number_blocks(device, rounding_mode):
    """The number path over many tiles per NEO, with a ragged last block."""
    ta, _ = _inputs((1, 2, 33600))  # the traced YOLO shape: 1050 tiles
    ta = ta * 8
    got = _run(_put(ta, device), None, divisor=2.0, rounding_mode=rounding_mode, out_dtype=ttnn.float32,
               block_tiles=3)  # fmt: skip
    expected = torch.div(ta.double(), 2.0, rounding_mode=rounding_mode).float()
    assert torch.equal(got, expected), (got - expected).abs().max()


def test_int_divisor_equals_float(device):
    """R8: div(a, 2) and div(a, 2.0) are accepted and agree."""
    ta, _ = _inputs((1, 2, 400))
    a = _put(ta, device)
    got_int = ttnn.to_torch(div(a, 2)).float()
    got_float = ttnn.to_torch(div(a, 2.0)).float()
    assert torch.equal(got_int, got_float)
    assert torch.equal(got_int, (ta.float() / 2))


def test_output_in_first_operand_buffer(device):
    """R13: the output lives in a's buffer, interleaved."""
    ta, tb = _inputs((1, 1, 32, 64))
    for mc, buffer in ((ttnn.DRAM_MEMORY_CONFIG, ttnn.BufferType.DRAM), (ttnn.L1_MEMORY_CONFIG, ttnn.BufferType.L1)):
        out = div(_put(ta, device, mc), _put(tb, device, mc))
        assert out.memory_config().buffer_type == buffer
        assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


def test_programs_share_a_device_session(device):
    """div after other programs, every variant, on one open device: tensor and number divisors, each rounding
    mode, both output dtypes, a different plan, and a tt-metal program (`to_layout`) in the middle."""
    shape = (1, 3, 320, 320)
    steps = [
        dict(),
        dict(number=True),
        dict(rounding_mode="floor", dtype=ttnn.float32),
        dict(number=True, rounding_mode="trunc"),
        dict(to_layout=True),
        dict(threads=(1, 1, 1)),
        dict(number=True, to_layout=True),
        dict(),
    ]
    for seed, step in enumerate(steps):
        ta, tb = _inputs(shape, seed)
        a, b = _put(ta, device), _put(tb, device)
        divisor = 4.0 if step.get("number") else None
        mode = step.get("rounding_mode")
        if "threads" in step:
            got = _run(a, b, threads=step["threads"])
        else:
            out = div(a, divisor if divisor is not None else b, rounding_mode=mode, dtype=step.get("dtype"))
            if step.get("to_layout"):
                out = ttnn.from_device(ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT))
            got = ttnn.to_torch(out).float()
        denominator = torch.tensor(divisor).double() if divisor is not None else tb.double()
        expected = torch.div(ta.double(), denominator, rounding_mode=mode).float()
        if mode is None:
            assert torch.allclose(got, expected, rtol=2**-8, atol=1e-30), (seed, step)
        else:
            # floor / trunc of a quotient that may sit within rounding distance of an integer (NUMERIC freedoms)
            assert ((got - expected).abs() > 1).sum() == 0 and ((got != expected).float().mean() < 0.01), (seed, step)


def test_a_tt_metal_program_follows(device):
    """`ttnn.to_layout` to row-major (untagged, explicit sync) right after div, in the same session."""
    ta, tb = _inputs((1, 3, 320, 320))
    out = div(_put(ta, device), _put(tb, device))
    got = ttnn.to_torch(ttnn.from_device(ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT))).float()
    _check_div(got, ta, tb)
    out = div(_put(ta, device), 3.0)
    got = ttnn.to_torch(ttnn.from_device(ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT))).float()
    assert torch.allclose(got, (ta.double() / 3.0).float(), rtol=2**-8, atol=1e-30)
