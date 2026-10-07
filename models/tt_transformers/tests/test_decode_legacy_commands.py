# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Stable branch: adapters without ``decode_input_update_contract`` call ``Generator.decode_forward``
with the legacy ``reset_batch`` shape; ``_legacy_decode_commands`` maps it onto the four explicit
decode commands the ported base requires. Host-only."""

from collections import defaultdict
from types import SimpleNamespace

import pytest

from models.tt_transformers.tt.generator import Generator


def _generator(*, traced: bool, always_refresh: bool = False):
    g = Generator.__new__(Generator)
    g.trace_ids_decode = defaultdict(lambda: None)
    if traced:
        g.trace_ids_decode[True] = {0: 1}
        g.trace_ids_decode[False] = {0: 2}
    g.model = [SimpleNamespace(_tt_vllm_always_refresh_decode_trace_inputs=always_refresh)]
    g.data_parallel = 1
    return g


def test_steady_traced_device_sampled_step_replays_the_trace():
    g = _generator(traced=True)
    g._prev_on_device_sampling = True
    cmds = g._legacy_decode_commands(reset_batch=False, enable_trace=True, on_device_sampling=True, mode_switched=False)
    assert cmds == {
        "reload_inputs": False,
        "reload_page_table": False,
        "reload_sampling_params": True,
        "reset_sampling_state": False,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(reset_batch=True, enable_trace=True, on_device_sampling=True, mode_switched=False),
        dict(reset_batch=False, enable_trace=False, on_device_sampling=True, mode_switched=False),
        dict(reset_batch=False, enable_trace=True, on_device_sampling=False, mode_switched=False),
        dict(reset_batch=False, enable_trace=True, on_device_sampling=True, mode_switched=True),
        dict(
            reset_batch=False, enable_trace=True, on_device_sampling=True, mode_switched=False, force_reload_inputs=True
        ),
    ],
)
def test_reset_untraced_host_sampling_mode_switch_or_forced_reload_reloads_inputs(kwargs):
    g = _generator(traced=True)
    g._prev_on_device_sampling = kwargs["on_device_sampling"]
    cmds = g._legacy_decode_commands(**kwargs)
    assert cmds["reload_inputs"] is True and cmds["reload_page_table"] is True
    assert cmds["reset_sampling_state"] is kwargs["reset_batch"]


def test_first_step_without_a_trace_and_a_sampling_mode_change_reload_inputs():
    g = _generator(traced=False)
    assert (
        g._legacy_decode_commands(reset_batch=False, enable_trace=True, on_device_sampling=True, mode_switched=False)[
            "reload_inputs"
        ]
        is True
    )
    g = _generator(traced=True)
    g._prev_on_device_sampling = False
    assert (
        g._legacy_decode_commands(reset_batch=False, enable_trace=True, on_device_sampling=True, mode_switched=False)[
            "reload_inputs"
        ]
        is True
    )


def test_model_that_always_refreshes_reloads_inputs():
    g = _generator(traced=True, always_refresh=True)
    g._prev_on_device_sampling = True
    assert (
        g._legacy_decode_commands(reset_batch=False, enable_trace=True, on_device_sampling=True, mode_switched=False)[
            "reload_inputs"
        ]
        is True
    )
