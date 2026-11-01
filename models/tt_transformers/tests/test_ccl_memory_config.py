# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

import ttnn
from models.tt_transformers.tt import ccl as tt_transformers_ccl


@pytest.mark.parametrize("configuration", ["omitted", "none", "custom"])
def test_all_reduce_resolves_default_memory_config_at_call_time(monkeypatch, configuration):
    # A module constant can outlive engine cleanup if captured in defaults.
    # Replacing it also distinguishes a fresh lookup from that retained value.
    current_dram = object()
    custom_config = object()
    result = object()
    calls = []
    deallocations = []
    monkeypatch.setattr(ttnn, "DRAM_MEMORY_CONFIG", current_dram)

    def reduce_scatter(input_tensor, **kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(ttnn.experimental, "reduce_scatter_minimal_async", reduce_scatter)
    tensor = SimpleNamespace(shape=(1, 1, 32, 64), is_sharded=lambda: False, deallocate=deallocations.append)
    mesh = SimpleNamespace(shape=(1, 2))
    collectives = SimpleNamespace(
        get_and_cycle_rs_semaphore_handles=lambda: [],
        get_and_cycle_barrier_semaphore_handle=lambda: None,
    )
    kwargs = (
        {} if configuration == "omitted" else {"rs_memory_config": None if configuration == "none" else custom_config}
    )
    output = tt_transformers_ccl.tt_all_reduce(
        tensor,
        mesh,
        collectives,
        cluster_axis=0,
        num_reduce_scatter_links=1,
        num_all_gather_links=1,
        **kwargs,
    )
    expected = {"omitted": current_dram, "none": None, "custom": custom_config}[configuration]
    assert output is result
    assert len(calls) == 1 and calls[0]["intermediate_memory_config"] is expected
    assert deallocations == [True]
