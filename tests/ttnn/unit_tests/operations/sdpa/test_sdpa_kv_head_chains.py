# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""K/V-head chains, sanity subset: 8x8 cases (multicast chains) that fit Wormhole and Blackhole grids. Every grid,
chunk, batch, head ratio and MLA's separate K and V heads run in
tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_kv_head_chains_sweep.py."""

import pytest

from tests.ttnn.unit_tests.operations.sdpa.kv_head_chains_test_utils import (
    check_kv_head_chains_gqa,
    check_kv_head_chains_mask,
    check_kv_head_chains_match_q_head_chains,
    check_kv_head_chains_ratios,
)


@pytest.mark.timeout(600)
def test_kv_head_chains_match_q_head_chains(device):
    check_kv_head_chains_match_q_head_chains(device, (8, 8), 256, 256)


@pytest.mark.timeout(600)
def test_kv_head_chains_gqa(device):
    # q_chunk 192 does not divide S: KV-head chains together with Q-head crossing
    check_kv_head_chains_gqa(device, 2, (8, 8), 192, 512)


@pytest.mark.timeout(600)
def test_kv_head_chains_mask(device):
    check_kv_head_chains_mask(device, (8, 8))


@pytest.mark.timeout(600)
def test_kv_head_chains_mha(device):
    # one Q head per KV head: KV-head and Q-head chains coincide
    check_kv_head_chains_ratios(device, 32, 32, 512, (8, 8))
