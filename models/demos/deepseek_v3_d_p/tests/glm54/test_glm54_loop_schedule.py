# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: which block order TtPrefillTransformer runs (``_build_schedule``), without building blocks.

A config without loop fields (GLM-5.3 and every other model) and a GLM-5.4 stage slice that holds no loop layer run
each block once, in order, as before the loop existed. A slice holding the whole loop runs the reference
``loop_schedule``, restricted to its ``loop_passes`` when it is a pipeline stage holding one pass (G1, G2). A slice
holding part of the loop's layers, passes that cannot follow its prelude/coda layers, and a loop without one gate
weight per pass are rejected. The device schedule of the reduced stack is also checked on device
(test_glm54_loop_device, test_glm54_stage_device).
"""

import pytest

from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import glm_5_3_hf_config
from models.demos.deepseek_v3_d_p.reference.glm_5_4 import LoopStep
from models.demos.deepseek_v3_d_p.reference.glm_5_4_config import glm_5_4_hf_config
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer


def _schedule(config, first_layer_idx, num_layers, state_dict=None, kv_only_last_layer=False, loop_passes=None):
    t = TtPrefillTransformer.__new__(TtPrefillTransformer)
    t.first_layer_idx, t.num_layers, t.kv_only_last_layer = first_layer_idx, num_layers, kv_only_last_layer
    t._loop_passes = loop_passes
    t.mesh_device = None  # gates are never built on these paths
    steps, gates = t._build_schedule(config, state_dict or {}, 1, None, 1)
    assert gates == {}
    return steps


def _plain(first, n):
    return [LoopStep("layer", 0, i) for i in range(first, first + n)]


def test_config_without_loop_runs_each_block_once():
    assert _schedule(glm_5_3_hf_config(), 0, 78) == _plain(0, 78)
    assert _schedule(glm_5_3_hf_config(), 19, 19) == _plain(19, 19)


@pytest.mark.parametrize("first, n", [(0, 26), (54, 24)], ids=["prelude-G0", "coda-G3"])
def test_glm54_slice_without_loop_layers_runs_each_block_once(first, n):
    assert _schedule(glm_5_4_hf_config(), first, n) == _plain(first, n)


@pytest.mark.parametrize("first, n", [(0, 40), (30, 48)], ids=["head", "tail"])
def test_glm54_slice_with_part_of_the_loop_is_rejected(first, n, expect_error):
    with expect_error(NotImplementedError, "whole loop or none"):
        _schedule(glm_5_4_hf_config(), first, n)


def test_glm54_loop_needs_one_gate_per_pass(expect_error):
    with expect_error(ValueError, "2 gate weight dicts, got 1"):
        _schedule(glm_5_4_hf_config(), 0, 78, state_dict={"loop_gates": [{}]})


def test_glm54_kv_only_last_layer_cannot_be_a_loop_layer(expect_error):
    with expect_error(ValueError, "kv-only last layer"):
        _schedule(glm_5_4_hf_config(), 0, 54, kv_only_last_layer=True)


@pytest.mark.parametrize(
    "first, n, passes, match",
    [
        (0, 54, (2,), "prelude layers 0..25 must precede loop pass 1"),
        (26, 52, (1,), "coda layers 54..77 must follow loop pass 2"),
        (26, 28, (2, 1), "not a contiguous run"),
        (26, 28, (3,), "not a contiguous run"),
        (0, 26, (1,), "hold no loop layer"),
    ],
    ids=["prelude-before-pass2", "coda-after-pass1", "unordered", "past-loop-count", "no-loop-layers"],
)
def test_glm54_stage_passes_are_checked(first, n, passes, match, expect_error):
    with expect_error(ValueError, match):
        _schedule(glm_5_4_hf_config(), first, n, loop_passes=passes)
