# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tracy profiling entry points for Mistral Small 4: the full model, ONE layer deep.

WHY ONE LAYER
-------------
For "where does the time go inside a layer", depth is pure noise: 36 layers produce 36x the
Tracy zones and a capture large enough to be awkward to load, while every layer is the same
shape. One layer keeps the whole real path -- embedding, MLA, MoE gate/dispatch/experts/combine,
the CCLs, norms -- at 1/36th the trace size, and the per-op table reads directly as the layer
budget.

TWO PARALLELISATIONS
--------------------
* ``test_profile_tp4``  -- single rank, mesh 8x4, SP=8 x TP=4, ``torus_xy``. The production
  serving config (the row `serve_mistral4_interactive.py` and the bring-up `pretrained` test
  both use). TP>1 means the MLA tp-axis CCLs (`reduce_scatter` + `high_bw_all_gather` in
  `_q_a_latent` / `_kv_stem` / the output gate) are LIVE -- these are exactly what TP=1 skips.
* PP=4 is NOT defined here. It reuses the existing
  ``test_prefill_pipeline_concurrent.py::test_mistral4_pp4_concurrent_throughput`` with
  ``PP_TOTAL_LAYERS=4`` (=> 1 layer per stage). Duplicating its 4-stage build here would have
  been ~400 lines of drift-prone copy; the env knob is one line. See ``run_mistral4_profile.sh``.

READING THE RESULT
------------------
Bound the measurement with the signposts the model already emits rather than eyeballing the
whole capture -- ``perf_utils.run_model_device_perf_test_with_merge(between_signposts=...)``
does this, and the same pairs work as Tracy zone filters:

    MLA_START / MLA_END          the attention leg
    MoE_START / MoE_END          the MoE leg
    forward_layer_<i>_start      layer boundary

Without a signpost bound the numbers include one-time weight tilize/typecast at construction,
which on a real checkpoint dwarfs the steady-state layer.

RUN
---
    /data/kmabee/disagg_logs/run_mistral4_profile.sh tp4     # or: pp4, or: both
"""

import os

import pytest

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.mistral_small_4_config import MistralSmall4Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode

# Window. 1024 is the smallest that satisfies BOTH tile alignment and the MoE masked_bincount
# 64-core grid (64 tokens/chip x sp=8 = 512 minimum, and 1024 keeps 128/chip).
PROFILE_ISL = int(os.environ.get("M4_PROFILE_ISL", 1024))
PROFILE_LAYERS = int(os.environ.get("M4_PROFILE_LAYERS", 1))
# Real weights by default: the MoE expert path and its dtypes are a large part of the layer
# budget, and synthetic weights would not exercise the same dispatch.
PROFILE_PRETRAINED = os.environ.get("M4_PROFILE_PRETRAINED", "1") not in ("0", "false", "no")


@pytest.mark.skipif(not is_blackhole(), reason="Mistral Small 4 targets Blackhole")
@pytest.mark.parametrize("tokenizer", ["right"], indirect=True, ids=["right_pad"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=MistralSmall4Config.FABRIC_PAYLOAD_SIZE),
            2,
            ttnn.Topology.Linear,
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["mistral_small_4"], indirect=True, ids=["mistral4"])
@pytest.mark.timeout(0)
def test_profile_tp4(
    variant,
    config_only,
    mesh_device,
    device_params,
    num_links,
    topology,
    weight_cache_path,
    is_ci_env,
    is_ci_v2_env,
    tokenizer,
    request,
):
    """SP=8 x TP=4 on one 8x4 mesh, `PROFILE_LAYERS` deep. Run under `python -m tracy`."""
    from models.demos.deepseek_v3_d_p.tests.test_prefill_transformer import run_model

    run_model(
        variant,
        config_only,
        mesh_device,
        device_params,
        False,  # is_balanced
        PROFILE_ISL,
        8,  # dispatch_buffer_capacity_factor
        PROFILE_LAYERS,
        MistralSmall4Config.NUM_ROUTED_EXPERTS,
        GateComputeMode.GPT_DEVICE,
        num_links,
        topology,
        False,  # pcc_validation -- host reference would dominate the wall clock and is not profiled
        False,  # determinism_check
        1,  # num_iterations
        "json_prompts",
        PROFILE_PRETRAINED,
        False,  # return_kv_cache
        0.0,  # temperature
        weight_cache_path,
        is_ci_env,
        is_ci_v2_env,
        tokenizer,
        request,
    )
