# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""expand beyond the acceptance cells: the blocking knobs turned (multi-block runs, ring wrap, fewer cores),
the structural count of input reads, and the op in one device session with other programs (target page,
*Ring protocols of a data-movement kernel*)."""

import pytest
import torch

import ttnn
from ttnn.operations.expand import expand
from ttnn.operations.expand.expand_program_artifacts import (
    THREAD_PLAN,
    create_program_artifacts,
    geometry,
    sources_of_run,
    split_runs,
)

TRACED = ((1, 2560), (1330, -1))


def _input(device, shape, seed=0):
    torch.manual_seed(seed)
    t = torch.randn(shape, dtype=torch.bfloat16)
    return t, ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _run_with_knobs(x, result_shape, **knobs):
    """The op's one dispatch, with the blocking knobs set by the test."""
    spec = ttnn.TensorSpec(ttnn.Shape(result_shape), x.dtype, x.layout, memory_config=x.memory_config())
    out = ttnn.allocate_tensor_on_device(spec, x.device())
    program, run_args, indices = create_program_artifacts(x, out, result_shape, **knobs)
    return ttnn.generic_op([x, out], program, run_args, indices)


@pytest.mark.parametrize(
    "knobs",
    [
        {"block_src_tiles": 1},  # one source per block: many blocks, the ring wraps every 2 blocks
        {"block_src_tiles": 2},  # a short last block on the cores whose run touches 3 sources
        {"block_src_tiles": 1, "buffer_depth": 3},  # 3 blocks per cluster: a ring three blocks deep
        {"max_cores": 1},  # the whole output from one cluster: one block of 80 sources
        {"max_cores": 7, "block_src_tiles": 3},  # run boundaries at odd places in the sources' rows
    ],
    ids=lambda k: "-".join(f"{n}={v}" for n, v in k.items()),
)
def test_knobs_traced_call(device, knobs):
    (shape, asked) = TRACED
    t, x = _input(device, shape)
    got = ttnn.to_torch(_run_with_knobs(x, [1330, 2560], **knobs))
    assert torch.equal(got, t.expand(*asked))


@pytest.mark.parametrize("shape,asked", [((1, 8192), (17, -1)), ((1, 50), (100, -1)), ((1, 128), (1024, -1))])
def test_knobs_small_blocks(device, shape, asked):
    t, x = _input(device, shape)
    result = list(t.expand(*asked).shape)
    got = ttnn.to_torch(_run_with_knobs(x, result, block_src_tiles=1, max_cores=5))
    assert torch.equal(got, t.expand(*asked))


@pytest.mark.parametrize(
    "shape,asked",
    [TRACED, ((1, 512), (128, 512)), ((1, 128), (1024, -1)), ((1, 8192), (17, -1)), ((1, 17), (32, 17))],
)
def test_input_reads_structural_count(device, shape, asked):
    """Each cluster reads every source tile its run touches once: input page reads <= num_sources +
    engaged clusters - 1 (the design's data-movement budget), and the runs cover every output tile once."""
    _, x = _input(device, shape)
    out = expand(x, asked)
    result = list(out.shape)
    _, tensor_w_tiles, fan, total = geometry(out, result)
    _, counts, starts = split_runs(total, device)
    covered = sorted((starts[c], counts[c]) for c in counts)
    assert covered[0][0] == 0 and all(a[0] + a[1] == b[0] for a, b in zip(covered, covered[1:]))
    assert covered[-1][0] + covered[-1][1] == total
    reads = sum(sources_of_run(starts[c], counts[c], fan) for c in counts)
    num_sources = tensor_w_tiles
    assert num_sources <= reads <= num_sources + len(counts) - 1, (reads, num_sources, len(counts))
    assert THREAD_PLAN == (1, 0, 1)


def test_after_another_program(device):
    """Another program, then expand, in one device session."""
    t0, x0 = _input(device, (1, 1, 64, 64), seed=1)
    rm = ttnn.to_layout(x0, ttnn.ROW_MAJOR_LAYOUT)
    assert torch.equal(ttnn.to_torch(rm), t0)
    t, x = _input(device, TRACED[0])
    assert torch.equal(ttnn.to_torch(expand(x, TRACED[1])), t.expand(*TRACED[1]))


def test_before_a_tt_metal_program(device):
    """expand, then a program of tt-metal's own on its output (untilize: explicit sync, plain transfers), in one
    device session, twice over, with a second expand between."""
    for seed, (shape, asked) in enumerate([TRACED, ((1, 50), (100, -1))]):
        t, x = _input(device, shape, seed)
        out = expand(x, asked)
        got = ttnn.to_torch(ttnn.from_device(ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)))
        assert torch.equal(got, t.expand(*asked))


def test_output_shape_list_and_tuple(device):
    t, x = _input(device, (1, 64))
    for asked in ([32, -1], (32, 64), [32, 64]):
        assert torch.equal(ttnn.to_torch(expand(x, asked)), t.expand(*asked))


@pytest.mark.parametrize(
    "shape,asked,error",
    [
        ((1, 64), (64,), ValueError),  # X1
        ((32, 64), (64, 64), ValueError),  # X2
        ((1, 64), (-1, 32, 64), ValueError),  # X3
        ((1, 64), (0, 64), NotImplementedError),  # empty result: unspecified
        ((1, 64), (2, 32, 64), NotImplementedError),  # added dimension: outside the first phase
        ((32, 1), (32, 64), NotImplementedError),  # W expands: outside the first phase
    ],
)
def test_refusals(device, shape, asked, error):
    _, x = _input(device, shape)
    with pytest.raises(error):
        expand(x, asked)


def test_l1_input_refused(device):
    t = torch.randn((1, 64), dtype=torch.bfloat16)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                        memory_config=ttnn.L1_MEMORY_CONFIG)
    with pytest.raises(NotImplementedError):
        expand(x, (32, -1))


def test_output_placed_like_input(device):
    """R6: the output is allocated in the input's buffer, interleaved."""
    _, x = _input(device, TRACED[0])
    out = expand(x, TRACED[1])
    assert out.memory_config() == x.memory_config()
    assert out.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


def test_sharded_input_refused_by_name(device):
    """R7: the placement axes come from eval.memory.memory_axes, so a sharded input is refused on `sharding`."""
    t = torch.randn((1, 64), dtype=torch.bfloat16)
    mem = ttnn.create_sharded_memory_config(
        (32, 64), core_grid=ttnn.CoreGrid(y=1, x=1), strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
    )
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    with pytest.raises(NotImplementedError, match="sharding"):
        expand(x, (32, -1))


@pytest.mark.parametrize(
    "knobs,blocks",
    [({}, 1), ({"block_src_tiles": 2}, 2), ({"block_src_tiles": 1}, 3), ({"block_src_tiles": 1, "buffer_depth": 3}, 3)],
    ids=["default", "block2", "block1", "block1-depth3"],
)
def test_ring_depth_structural_count(device, knobs, blocks):
    """The ring holds min(buffer_depth, blocks of the busiest cluster) blocks: with one block per cluster (the
    default on the traced call) a second half would never be live (L1 ledger, capacity vs live set)."""
    _, x = _input(device, TRACED[0])
    result = [1330, 2560]
    spec = ttnn.TensorSpec(ttnn.Shape(result), x.dtype, x.layout, memory_config=x.memory_config())
    out = ttnn.allocate_tensor_on_device(spec, device)
    program, _, _ = create_program_artifacts(x, out, result, **knobs)
    _, _, fan, total = geometry(out, result)
    _, counts, starts = split_runs(total, device)
    max_src = max(sources_of_run(starts[c], counts[c], fan) for c in counts)
    block = min(max_src, knobs.get("block_src_tiles", max_src))
    assert -(-max_src // block) == blocks
    depth = min(knobs.get("buffer_depth", 2), blocks)
    (ring,) = program.dataflow_buffers
    assert ring.num_entries == depth * block
