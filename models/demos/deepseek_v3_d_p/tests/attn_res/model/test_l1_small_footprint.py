# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What AttnRes costs a model's L1_SMALL pool, as the sealed set grows under it.

The pool is sized once, when the mesh opens, and every component on the device shares
it. A component can only be budgeted for if its share is a number the caller can know
in advance -- and a walk's is not automatically one, because the statistics tensor
carries the sealed set's width, so each seal depth is a distinct program. Any collective
that allocates global semaphores during program construction therefore takes another set
per depth and holds them for as long as the program cache, and `ttnn.all_gather`'s
factory puts those in L1_SMALL whenever the pool is non-empty, with no argument to ask
otherwise. The only share a caller can budget for is none, which is what handing the
collective persistent handles buys.

The two gates here are one measurement read twice. The walk is identical; only the
collective differs, and the arm that lets the collective allocate for itself is what
makes the arm that does not vacuous-proof: it holds the claim that the pool is where
those semaphores land, so a future ttnn that stops using L1_SMALL for them fails the
control rather than silently turning the real gate into an assertion about nothing.
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
from models.demos.deepseek_v3_d_p.tt.tt_ccl import TT_CCL

PCC_GATE = 0.9999

# The depth a K3 forward pass reaches -- 93 layers over blocks of 12. Sealing every layer
# instead of every twelfth walks the same depths at a twelfth of the device time, and it
# is the depth that shapes the statistics tensor, not the layer count that produced it.
SEALS = 8
SEAL_EVERY_LAYER = 1

MODULE_SCALE = 0.02

PLACEMENTS = placements()

pytestmark = blackhole_only


class TtAttnResImplicitSemaphores(TtAttnRes):
    """The same walk with the collective left to allocate its own semaphores.

    `ttnn.all_reduce` passes every semaphore argument down as nullopt, which selects the
    reduce-scatter-then-all-gather fallback, and the all-gather factory allocates there.
    This exists to be measured against the op proper, not as a configuration anything
    ships -- the growth it shows is the cost the op is required not to have.
    """

    def _all_reduce(self, tensor):
        return ttnn.all_reduce(
            tensor,
            cluster_axis=self.tp_axis,
            num_links=self.num_links,
            topology=self.topology[self.tp_axis],
        )


def _l1_small_per_bank(mesh_device):
    return ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL).total_bytes_allocated_per_bank


def _exhausted_the_pool(error) -> bool:
    """Whether the allocator refused for want of L1_SMALL rather than for anything else.

    Both fragments are required: an out-of-memory on main L1 is a different failure and
    has to keep failing the test.
    """
    text = str(error)
    return "Not enough space to allocate" in text and "L1_SMALL" in text


def _first_line(error) -> str:
    return str(error).strip().splitlines()[0]


def _walk_sampling_the_pool(mesh_device, op_class, samples=None, **op_kwargs):
    """Walk `SEALS` depths, sampling the pool at each, and check the walk's own numbers.

    Returns the per-depth samples and the PCC. The accuracy check is part of the
    measurement rather than a separate gate: a walk that dies early or skips its
    collective also reports a flat footprint, so an unchecked footprint proves nothing.

    A caller that expects the walk to run the pool dry passes `samples` in, so that what
    was measured before the allocator gave up survives the exception.
    """
    op = op_class(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS, **op_kwargs)

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
    samples = [] if samples is None else samples
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

    pcc = assert_accurate(
        want, got, name=f"{op_class.__name__} walk under a tight L1_SMALL pool", pcc_threshold=PCC_GATE
    )
    return samples, pcc


@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=["mesh_device", "device_params"])
def test_l1_small_stays_empty_as_the_sealed_set_grows(mesh_device, device_params):
    """The op's share of the shared pool has to be nothing, at every depth.

    The mesh opens with the tightest pool a K3 config gives it, which is what makes this
    able to fail: at a non-empty pool a collective that allocates its own semaphores
    lands them here, and at an empty one it would fall back to main L1 and this would
    pass on an op that still grows without bound.
    """
    samples, pcc = _walk_sampling_the_pool(mesh_device, TtAttnRes)

    footprints = sorted({footprint for _, footprint, _ in samples})
    assert footprints == [0], (
        "AttnRes has to cost the shared pool nothing at every depth, so that a model can "
        f"size it for everything else: got {footprints} B/bank"
    )
    logger.info(f"{SEALS} seals, {2 * SEALS} reads, 0 B/bank of L1_SMALL: PCC {pcc:.7f}")


@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=["mesh_device", "device_params"])
def test_implicit_semaphores_grow_the_pool_with_the_sealed_set(mesh_device, device_params):
    """The control: the same walk, letting the collective allocate, has to cost the pool.

    This is what the pool being AttnRes's problem looks like, and it is the reason the
    gate above is worth asserting. If this ever stops costing the pool, L1_SMALL is no
    longer where those semaphores land and the gate above has to be re-derived rather
    than trusted.

    How far it gets before the pool runs out is the fabric's to decide, not the claim's:
    measured here, an unwrapped fabric climbs through all eight depths at 128 B/bank each
    while a wrapped one exhausts the same pool inside the first seal. Both are the pool
    being consumed and both pass; a walk that reaches every depth for free is the only
    outcome that does not.
    """
    samples = []
    try:
        _walk_sampling_the_pool(mesh_device, TtAttnResImplicitSemaphores, samples)
    except RuntimeError as error:
        if not _exhausted_the_pool(error):
            raise
        logger.info(f"L1_SMALL ran dry after {len(samples)} of {SEALS} seals: {_first_line(error)}")
        return

    footprints = [footprint for _, footprint, _ in samples]
    assert footprints[0] > 0, (
        "a collective allocating its own semaphores has to land them in a non-empty "
        f"L1_SMALL pool; nothing here means the pool is not the resource at stake: got {footprints}"
    )
    assert footprints[-1] > footprints[0], (
        "each seal depth hashes its own program and so takes its own semaphore set; a "
        f"flat footprint across depths means the depths collapsed onto one: got {footprints}"
    )
    logger.info(f"{SEALS} seals, {2 * SEALS} reads, L1_SMALL {footprints[0]} -> {footprints[-1]} B/bank")


@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=["mesh_device", "device_params"])
def test_a_shared_ccl_pool_costs_the_shared_l1_small_nothing_either(mesh_device, device_params):
    """The same gate for the caller that hands the op the model's own semaphore pool.

    A model that already has a `TT_CCL` should pass it rather than let the op keep a
    private set, because that pool is double-buffered and a private set is not -- which
    matters as soon as something else has a collective in flight on the same axis. That
    is only advice worth giving if it costs the pool the same nothing, and the handles
    reach a different code path to get there, so it is measured rather than assumed.
    """
    samples, pcc = _walk_sampling_the_pool(mesh_device, TtAttnRes, tt_ccl=TT_CCL(mesh_device))

    footprints = sorted({footprint for _, footprint, _ in samples})
    assert footprints == [0], (
        "borrowing the model's pool has to cost the shared L1_SMALL the same nothing a "
        f"private set does: got {footprints} B/bank"
    )
    logger.info(f"{SEALS} seals on a shared TT_CCL pool, 0 B/bank of L1_SMALL: PCC {pcc:.7f}")
