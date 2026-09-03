# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Contract: the read returns the same bits every time, from inputs it leaves alone.

`test_attn_res.py` measures what one read computes. Nothing there measures what a read
leaves behind, and the two are independent — a read that corrupts state it shares with the
next read still agrees with its own oracle, and PCC against torch is computed per read.

`TtAttnRes` carries three things across reads, each documented in `tt/attn_res/attn_res.py`
as safe to reuse and none of it measured until here:

  * `_exchange_scratch` — one `[1, 2 * tp, N, 1]` buffer per token count, which every read
    writes its own statistics and its peers' planes into. Its docstring's claim is that it
    "carries nothing between reads".
  * `_exchange_semaphore` — one global arrival semaphore for the whole walk, deliberately
    not program-local, reset in kernel after the wait rather than at launch.
  * `_query_columns` — the transposed query stack, memoized on `id(q)`, so a block's second
    read takes a different host path to the same operand than its first.

A second run of an identical read is what tests all three at once, and it needs no oracle:
same inputs, same state, same program, so the only correct answer is the same bits. The
same run also gives the other two properties for free — a borrowed input read back
afterwards must be unchanged, and a warm program cache must not grow.

Random queries throughout. What is under test is repetition, which the weights cannot
influence: any query at all exercises the same scratch, semaphore and memo.
"""

import pytest
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import EPS
from models.demos.deepseek_v3_d_p.tests.attn_res.assertions import assert_bit_identical
from models.demos.deepseek_v3_d_p.tests.attn_res.model.harness import (
    FABRIC,
    HIDDEN_SIZE,
    L1_SMALL_SIZE,
    PER_CHIP_TOKENS,
    blackhole_only,
    compose,
    generator,
    place_case,
    placements,
    random_case,
    random_queries,
    read_block,
)
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    torus_x_device_params,
    torus_xy_device_params,
)
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes

TP_AXIS = 1

# One full 12-layer block of read sites against the widest sealed set the walk reaches, so
# the batch axis, the site-major permute and every candidate-axis kernel are all live.
READ_SITES = 24
NUM_SEALED = 8

# Trace capture needs its own region; the rest of the suite runs eager and does not reserve one.
TRACE_REGION_SIZE = 23887872
TRACED = {**FABRIC, "trace_region_size": TRACE_REGION_SIZE}

on_mesh = pytest.mark.parametrize("mesh_device, device_params", placements(), indirect=True)
# The traced arms mirror `placements()` rather than sampling it: a trace freezes the
# fabric routes its captured programs resolved, so a route that only goes wrong on a
# wrapped axis stays wrong for the whole replay and cannot be caught on the other arms.
_TRACED_BOX = {"trace_region_size": TRACE_REGION_SIZE, "require_exact_physical_num_devices": True}

on_traced_mesh = pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param((2, 4), TRACED, id="mesh-2x4"),
        pytest.param(
            (8, 4),
            fabric2d_device_params(l1_small_size=L1_SMALL_SIZE, **_TRACED_BOX),
            id="mesh-8x4",
        ),
        pytest.param(
            (8, 4),
            torus_x_device_params(l1_small_size=L1_SMALL_SIZE, **_TRACED_BOX),
            id="torusx-mesh-8x4",
        ),
        pytest.param(
            (8, 4),
            torus_xy_device_params(l1_small_size=L1_SMALL_SIZE, **_TRACED_BOX),
            id="torusxy-mesh-8x4",
        ),
    ],
    indirect=True,
)

pytestmark = blackhole_only


def _host_inputs(num_tokens, seed=0):
    """One block's inputs on host: the live stream, the sealed set, and a query per site."""
    rng = generator(seed)
    running_sum, block_residual = random_case(rng, num_tokens, NUM_SEALED)
    return running_sum, block_residual, random_queries(rng, READ_SITES)


@on_mesh
def test_read_repeats_exactly_without_disturbing_its_inputs(mesh_device, device_params):
    """Two identical runs of a 24-site block: same bits out, same bits left behind.

    The second run is the one carrying the signal. It reuses the scratch the first run
    wrote, waits on a semaphore the first run's kernels released, and takes the memoized
    query stack rather than building one. A stale plane in the scratch, a semaphore left
    off zero, or a freed memo entry all first become visible here.

    Inputs are borrowed by both `inter_block` and `merge`, so they are also compared before
    and after. The op reaches its operands through slices that share their parent's buffer,
    which makes an accidental in-place write land in a caller's tensor rather than a
    temporary — and every read after it would still match its own oracle.
    """
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS, tp_axis=TP_AXIS)
    num_tokens = PER_CHIP_TOKENS * op.sp_factor
    running_sum, block_residual, queries = _host_inputs(num_tokens)

    tt_prefix, tt_block = place_case(op, running_sum, block_residual)
    tt_queries = [op.to_query(q) for q in queries]
    before = (compose(op, tt_prefix), compose(op, tt_block), [compose(op, q) for q in tt_queries])

    first = list(read_block(op, tt_block, tt_prefix, tt_queries))
    ttnn.synchronize_device(mesh_device)
    warm_entries = mesh_device.num_program_cache_entries()

    second = list(read_block(op, tt_block, tt_prefix, tt_queries))
    ttnn.synchronize_device(mesh_device)

    for site, (once, twice) in enumerate(zip(first, second, strict=True)):
        assert_bit_identical(once, twice, name=f"site {site} repeat")

    assert_bit_identical(before[0], compose(op, tt_prefix), name="running_sum immutability")
    assert_bit_identical(before[1], compose(op, tt_block), name="block_residual immutability")
    for site, was in enumerate(before[2]):
        assert_bit_identical(was, compose(op, tt_queries[site]), name=f"query {site} immutability")

    entries = mesh_device.num_program_cache_entries()
    assert entries == warm_entries, (
        f"the second run compiled {entries - warm_entries} more programs than the first — "
        f"the read is not hitting the program cache it already populated"
    )

    logger.info(
        f"{READ_SITES} sites x2 at S={NUM_SEALED}, T={num_tokens} ({PER_CHIP_TOKENS}/chip): "
        f"bit-identical, inputs unchanged, {entries} cached programs"
    )

    for tensor in (tt_prefix, tt_block, *tt_queries):
        ttnn.deallocate(tensor)


@on_traced_mesh
def test_trace_replay_matches_eager(mesh_device, device_params):
    """A captured read replays to the same bits it produced eagerly.

    The walk is traced in production, and capture changes what the host does without
    changing what the device computes — so any divergence here is the op reaching for
    something that only exists on the eager path. The exchange scratch and the global
    semaphore are exactly that shape of dependency: both are host-allocated once and then
    addressed by the kernels, so a capture that recorded them wrongly would replay against
    the wrong buffer.

    One site, not the block. Capture cost scales with the call sequence and the sequence
    that matters here is a single fused read.
    """
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS, tp_axis=TP_AXIS)
    num_tokens = PER_CHIP_TOKENS * op.sp_factor
    running_sum, block_residual, queries = _host_inputs(num_tokens)

    tt_prefix, tt_block = place_case(op, running_sum, block_residual)
    tt_queries = [op.to_query(q) for q in queries]

    partials, shifts, masses = op.inter_block(tt_block, tt_queries)

    # Capture replays programs, it does not build them: the compile pass, the query stack and
    # the lazily-created scratch and semaphore all have to exist before the region opens.
    eager = op.merge(partials, shifts, masses, tt_prefix, tt_queries[0], 0)
    ttnn.synchronize_device(mesh_device)
    want = compose(op, eager)
    ttnn.deallocate(eager)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = op.merge(partials, shifts, masses, tt_prefix, tt_queries[0], 0)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    for _ in range(2):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)

    assert_bit_identical(want, compose(op, traced), name="trace replay")
    logger.info(f"trace replay bit-identical to eager at T={num_tokens} ({PER_CHIP_TOKENS}/chip)")

    ttnn.release_trace(mesh_device, trace_id)
    ttnn.deallocate(traced)
    for tensor in (partials, shifts, masses, tt_prefix, tt_block, *tt_queries):
        ttnn.deallocate(tensor)
