# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What AttnRes costs a model's L1_SMALL pool, as the sealed set grows under it.

The pool is sized once, when the mesh opens, and every component on the device shares
it. A component can only be budgeted for if its share is a number the caller can know
in advance -- and a walk's is not automatically one, because the block statistics carry
the sealed set's width, so each seal depth is a distinct program. Anything that allocated
global semaphores during program construction would take another set per depth and hold
it for as long as the program cache, and the CCL factories put those in L1_SMALL whenever
the pool is non-empty. The only share a caller can budget for is none: the op makes
each of its global semaphores once and holds it for the walk, and its gathers allocate
nothing per call.
"""

import pytest
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import EPS, attn_res_stack
from models.demos.deepseek_v3_d_p.tests.attn_res.assertions import assert_accurate
from models.demos.deepseek_v3_d_p.tests.attn_res.model.harness import (
    HIDDEN_SIZE,
    PER_CHIP_TOKENS,
    blackhole_only,
    compose,
    generator,
    place,
    placements,
    random_hidden,
    random_queries,
)
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import TtAttnResWalk

PCC_GATE = 0.9999

# The depth a K3 forward pass reaches -- 93 layers over blocks of 12. Sealing every layer
# instead of every twelfth walks the same depths at a twelfth of the device time, and it
# is the depth that shapes the statistics tensor, not the layer count that produced it.
SEALS = 8
SEAL_EVERY_LAYER = 1

MODULE_SCALE = 0.02

PLACEMENTS = placements()

pytestmark = blackhole_only


def _l1_small_per_bank(mesh_device):
    return ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL).total_bytes_allocated_per_bank


def _walk_sampling_the_pool(mesh_device):
    """Walk `SEALS` depths, sampling the pool at each, and check the walk's own numbers.

    Returns the per-depth samples and the PCC. The accuracy check is part of the
    measurement rather than a separate gate: a walk that dies early or skips its
    gather also reports a flat footprint, so an unchecked footprint proves nothing.
    """
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS)

    rng = generator(0)
    hidden_states = random_hidden(rng, PER_CHIP_TOKENS * op.sp_factor)
    q_pre, q_post = random_queries(rng, SEALS), random_queries(rng, SEALS)
    q_out = random_queries(rng, 1)[0]

    embeddings = place(op, hidden_states.unsqueeze(0).unsqueeze(0))
    tt_pre = [op.to_query(q) for q in q_pre]
    tt_post = [op.to_query(q) for q in q_post]
    tt_out = op.to_query(q_out)

    walk = TtAttnResWalk(op, embeddings, tt_pre, tt_post, tt_out, SEALS, block_size=SEAL_EVERY_LAYER)

    module = lambda hidden: ttnn.multiply(hidden, MODULE_SCALE)
    samples = []
    for layer_idx in range(SEALS):
        hidden, borrowed = walk.open_layer(layer_idx)
        walk.write(module(hidden))
        post_attention = walk.read()
        walk.write(module(post_attention))
        ttnn.deallocate(post_attention)
        # Layer 0 reads the live stream itself, which the walk still owns.
        if not borrowed:
            ttnn.deallocate(hidden)
        samples.append(
            (walk.stream.num_sealed, _l1_small_per_bank(mesh_device), mesh_device.num_program_cache_entries())
        )

    device_out = walk.finish()
    got = compose(op, device_out)
    ttnn.deallocate(device_out)
    for tensor in (tt_out, *tt_pre, *tt_post):
        ttnn.deallocate(tensor)

    torch_module = lambda hidden: hidden * MODULE_SCALE
    want = attn_res_stack(
        hidden_states,
        q_pre,
        q_post,
        q_out,
        [torch_module] * SEALS,
        [torch_module] * SEALS,
        block_size=SEAL_EVERY_LAYER,
        eps=EPS,
    )

    for depth, footprint, entries in samples:
        logger.info(f"sealed depth {depth}: L1_SMALL {footprint} B/bank, {entries} program cache entries")

    depths = [depth for depth, _, _ in samples]
    entries = [entry for _, _, entry in samples]
    assert depths == list(range(1, SEALS + 1)), (
        "sealing every layer has to walk the depth up one snapshot at a time, or the "
        f"samples are not distinct statistics shapes: got {depths}"
    )
    assert entries[-1] > entries[0], (
        "a deeper sealed set is a different statistics shape and so a different program; "
        f"a cache that did not grow means the depths collapsed onto one: got {entries}"
    )

    pcc = assert_accurate(want, got, name="walk under a tight L1_SMALL pool", pcc_threshold=PCC_GATE)
    return samples, pcc


@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=["mesh_device", "device_params"])
def test_l1_small_stays_empty_as_the_sealed_set_grows(mesh_device, device_params):
    """The op's share of the shared pool has to be nothing, at every depth.

    The mesh opens with the tightest pool a K3 config gives it, which is what makes this
    able to fail: at a non-empty pool a collective that allocates its own semaphores
    lands them here, and at an empty one it would fall back to main L1 and this would
    pass on an op that still grows without bound.
    """
    samples, pcc = _walk_sampling_the_pool(mesh_device)

    footprints = sorted({footprint for _, footprint, _ in samples})
    assert footprints == [0], (
        "AttnRes has to cost the shared pool nothing at every depth, so that a model can "
        f"size it for everything else: got {footprints} B/bank"
    )
    logger.info(f"{SEALS} seals, {2 * SEALS} reads, 0 B/bank of L1_SMALL: PCC {pcc:.7f}")
