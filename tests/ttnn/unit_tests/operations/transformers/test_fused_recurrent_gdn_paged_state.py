# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""fused_recurrent_gated_delta_rule with paged_state=True: the recurrent state lives in a [P, K, V] pool, head h reads
pool page idx[h] and writes its final state back to that page.

A serving engine passes one pool page per (row, value head) taken from its block table, so the same compiled program
(and trace) follows a request to other pages when only the index tensor changes. The reference is the same op on a
contiguous [B, HV, K, V] state.
"""

import pytest
import torch

import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.transformers.gdn_decode_test_utils import (
    OP,
    K,
    V,
    assert_accuracy,
    make_inputs,
    to_device,
)

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="fused_recurrent_gated_delta_rule is Blackhole-only")

H, HV = 4, 12  # Qwen3.6-27B TP 4 per device


def _pool_pages(B, P, offset):
    """Distinct, scattered pool pages: row b, value head j -> page (offset + 7 * b) % (P // HV) * HV + j."""
    return [(offset + 7 * b) % (P // HV) * HV + j for b in range(B) for j in range(HV)]


def _index(pages, device=None):
    return ttnn.from_torch(
        torch.tensor(pages, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )


def _pool_with_states(P, pages, states, seed):
    """A [P, K, V] pool of noise with states [B * HV, K, V] placed at the given pages."""
    pool = torch.randn(P, K, V, generator=torch.Generator().manual_seed(seed))
    pool[pages] = states.reshape(-1, K, V)
    return pool


@pytest.mark.parametrize("B", [1, 8, 32])
def test_paged_state_matches_contiguous(device, B):
    """o and the pool's state pages equal the contiguous run bit for bit; every other pool page is untouched. B = 32
    puts 384 heads on the grid, several per core."""
    P = 64 * HV
    x = make_inputs(B, 1, H, HV, K, V, seed=100 + B)
    pages = _pool_pages(B, P, offset=3)
    pool_host = _pool_with_states(P, pages, x["initial_state"], seed=B)
    q, k, v, g, beta = (to_device(device, x[n]) for n in ("q", "k", "v", "g", "beta"))

    o_ref, st_ref = OP(q, k, v, g, beta, initial_state=to_device(device, x["initial_state"]), output_final_state=True)
    o_ref, st_ref = ttnn.to_torch(o_ref), ttnn.to_torch(st_ref)
    o_gold, st_gold = ttnn.get_golden_function(OP)(
        x["q"], x["k"], x["v"], x["g"], x["beta"], initial_state=x["initial_state"], output_final_state=True
    )
    assert_accuracy(o_gold, o_ref, "contiguous o vs golden")
    assert_accuracy(st_gold, st_ref, "contiguous state vs golden")

    pool = to_device(device, pool_host)
    o, state = OP(
        q,
        k,
        v,
        g,
        beta,
        initial_state=pool,
        initial_state_block_idx=_index(pages, device),
        output_final_state=True,
        paged_state=True,
    )
    assert state.buffer_address() == pool.buffer_address(), "the state output must be the pool"
    assert torch.equal(ttnn.to_torch(o), o_ref), "paged o differs from contiguous o"

    pool_after = ttnn.to_torch(pool)
    assert torch.equal(pool_after[pages], st_ref.reshape(-1, K, V)), "pool pages differ from the contiguous state"
    others = [p for p in range(P) if p not in set(pages)]
    assert torch.equal(pool_after[others], pool_host[others]), "a paged write touched a page it does not own"


@pytest.mark.parametrize("B", [8, 32])
def test_heads_beyond_core_count_match_golden(device, B):
    """Without paging: B * HV above the core count gives the same accuracy as one head per core."""
    x = make_inputs(B, 1, H, HV, K, V, seed=7 + B)
    q, k, v, g, beta = (to_device(device, x[n]) for n in ("q", "k", "v", "g", "beta"))
    o, st = OP(q, k, v, g, beta, initial_state=to_device(device, x["initial_state"]), output_final_state=True)
    o_gold, st_gold = ttnn.get_golden_function(OP)(
        x["q"], x["k"], x["v"], x["g"], x["beta"], initial_state=x["initial_state"], output_final_state=True
    )
    assert_accuracy(o_gold, ttnn.to_torch(o), "o")
    assert_accuracy(st_gold, ttnn.to_torch(st), "final state")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 4 * 1024 * 1024}], indirect=True)
def test_paged_state_trace_follows_index(device):
    """Capture once, then move every row to other pool pages by rewriting only the index tensor."""
    B, P = 4, 32 * HV
    x = make_inputs(B, 1, H, HV, K, V, seed=11)
    first = _pool_pages(B, P, offset=1)
    second = _pool_pages(B, P, offset=2)
    assert not set(first) & set(second)
    pool_host = _pool_with_states(P, first, x["initial_state"], seed=1)
    pool_host[second] = 0.5 * x["initial_state"].reshape(-1, K, V)
    pool = to_device(device, pool_host)
    idx = _index(first, device)
    q, k, v, g, beta = (to_device(device, x[n]) for n in ("q", "k", "v", "g", "beta"))

    def step():
        o, _ = OP(
            q,
            k,
            v,
            g,
            beta,
            initial_state=pool,
            initial_state_block_idx=idx,
            output_final_state=True,
            paged_state=True,
        )
        return o

    step()  # compile; updates the states in the first pages
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    o = step()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    first_before_replay = ttnn.to_torch(pool)[first]

    ttnn.copy_host_to_device_tensor(_index(second), idx)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    o_trace = ttnn.to_torch(o)
    ttnn.release_trace(device, tid)

    second_in = 0.5 * x["initial_state"]
    o_gold, st_gold = ttnn.get_golden_function(OP)(
        x["q"], x["k"], x["v"], x["g"], x["beta"], initial_state=second_in, output_final_state=True
    )
    pool_after = ttnn.to_torch(pool)
    assert_accuracy(o_gold, o_trace, "replayed o on the second pages")
    assert_accuracy(st_gold.reshape(-1, K, V), pool_after[second], "states in the second pages")
    assert torch.equal(pool_after[first], first_before_replay), "the replay touched the first pages"


def test_paged_state_rejects_per_token_output(device, expect_error):
    x = make_inputs(1, 4, H, HV, K, V, seed=3)
    q, k, v, g, beta = (to_device(device, x[n]) for n in ("q", "k", "v", "g", "beta"))
    pool = to_device(device, torch.zeros(HV, K, V))
    with expect_error(RuntimeError, "output_per_token_state"):
        OP(
            q,
            k,
            v,
            g,
            beta,
            initial_state=pool,
            initial_state_block_idx=_index(list(range(HV)), device),
            output_per_token_state=True,
            paged_state=True,
        )
