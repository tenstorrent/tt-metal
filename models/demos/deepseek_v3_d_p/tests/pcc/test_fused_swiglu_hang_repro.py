# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Repro loop for the intermittent MoeFusedSwiGlu hang at GLM-5.3 4x4 shapes (not for CI).

test_glm_moe runs under the DeepSeek-V3 variant, whose config carries no
ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD, so its routed experts never take the hybrid split and never
reach moe_fused_swiglu. The GLM model splits at 320 tokens. This module reruns test_glm_moe's 4x4
perf row with that threshold set and the forward looped, and a watchdog that writes HANG_MARKER
(path from the env) when one iteration outlasts REPRO_ITER_TIMEOUT_S, leaving the devices hung for
tt-triage.
"""

import os
import threading
import time

import pytest
import torch
import ttnn
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.tests.pcc import test_ttnn_moe as moe_test

ITERS = int(os.environ.get("REPRO_ITERS", "400"))
ITER_TIMEOUT_S = float(os.environ.get("REPRO_ITER_TIMEOUT_S", "300"))
HANG_MARKER = os.environ.get("HANG_MARKER", "/tmp/fused_swiglu_hang")


def _looping_measure(mesh_device):
    def measure(forward, tt_moe, tt_x):
        beat = {"iter": -1, "t": time.time(), "done": False}

        def watchdog():
            while not beat["done"]:
                time.sleep(5)
                stalled = time.time() - beat["t"]
                if stalled > ITER_TIMEOUT_S:
                    msg = f"HANG at iter {beat['iter'] + 1} of {ITERS}: no progress for {stalled:.0f}s"
                    logger.error(msg)
                    with open(HANG_MARKER, "w") as f:
                        f.write(msg + "\n")
                    return

        threading.Thread(target=watchdog, daemon=True).start()
        out = None
        for i in range(ITERS):
            out = forward()
            ttnn.synchronize_device(mesh_device)
            beat["iter"], beat["t"] = i, time.time()
            if i % 25 == 0:
                logger.info(f"REPRO iter {i} ok")
        beat["done"] = True
        logger.info(f"REPRO {ITERS} iterations, no hang")
        return out

    return measure


@pytest.mark.parametrize(
    "seq_len_per_chip, emb_dim, hidden_dim, num_routed_experts, num_experts_per_tok, dispatch_buffer_capacity_factor",
    [(1280, GLM53Config.EMB_SIZE, GLM53Config.MOE_INTERMEDIATE_SIZE, 256, 8, 8)],
    ids=["glm-4x4-1280"],
)
# Std of a random per-expert e_score_correction_bias. noaux_tc selects on score + bias, so a nonzero bias
# gives a heavy-tailed load like the trained gate's: experts above the 320 split (unified half), below
# it (fused half) and empty. 0 keeps test_glm_moe's near-uniform ~160 tokens per expert (fused only).
@pytest.mark.parametrize("skew", [0.0, 0.05, 0.2], ids=lambda s: f"skew{s}")
@pytest.mark.parametrize("threshold", [GLM53Config.ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD], ids=lambda t: f"hyb{t}")
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, sp_axis",
    [
        pytest.param(
            (4, 4),
            moe_test.torus_x_device_params(fabric_payload_size=moe_test.moe_fabric_payload(DeepSeekV3Config)),
            2,
            1,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 4), topology="mesh-4x4"),
            id="torus-x-4x4-sp1",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["deepseek_v3_d_p"], indirect=True, ids=["ds-ref"])
@pytest.mark.timeout(0)
def test_fused_swiglu_hang_repro(
    variant,
    config_only,
    mesh_device,
    device_params,
    seq_len_per_chip,
    emb_dim,
    hidden_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_buffer_capacity_factor,
    num_links,
    sp_axis,
    threshold,
    skew,
    request,
    monkeypatch,
):
    # The split the GLM model ships, on the variant config run_model reads it from.
    monkeypatch.setattr(variant.model_config, "ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD", threshold, raising=False)
    # run_model persists the gate (bias included) to a shape-keyed weight cache, so each skew needs its
    # own directory: a shared one would load the other skew's bias, or hand ours to other runs.
    monkeypatch.setenv("DS_MOE_CACHE_DIR", f"/tmp/fused_swiglu_repro_moe_cache_skew{skew}")
    if skew:
        make = moe_test.create_gate_weights

        def skewed(num_experts, dim, seed):
            w = make(num_experts, dim, seed=seed)
            gen = torch.Generator().manual_seed(4242)
            bias = torch.randn(num_experts, generator=gen) * skew
            w["e_score_correction_bias"] = bias.to(w["e_score_correction_bias"].dtype)
            return w

        monkeypatch.setattr(moe_test, "create_gate_weights", skewed)
    moe_test.run_model(
        variant,
        config_only,
        mesh_device,
        device_params,
        seq_len_per_chip,
        emb_dim,
        hidden_dim,
        num_routed_experts,
        num_experts_per_tok,
        dispatch_buffer_capacity_factor,
        False,  # run_pcc_check: a perf-style loop
        num_links,
        moe_test.per_axis_topology(device_params["fabric_config"]),
        moe_test.GateComputeMode.DEVICE_FP32,
        request,
        sp_axis=sp_axis,
        measure=_looping_measure(mesh_device),
    )
