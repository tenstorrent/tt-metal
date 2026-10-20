# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident multi-layer mesh model (tt/mesh_model.py): the chained layers vs the HF layer chain of the
L45_S128 dump (prompt tokens fed through the traced decode step, output after the last built layer vs that
layer's HF prefill rows), greedy ids vs the host head on the same rows, and the traced end-to-end decode
step time -> tok/s per user and aggregate.

    GLM53_MESH_SHAPE=1,4 GLM53_MODEL_LAYERS=0-5 GLM53_MODEL_B=2 pytest models/demos/glm53_flash/tests/test_mesh_model.py -s
    GLM53_MESH_SHAPE=8,4 GLM53_MODEL_LAYERS=0-44 GLM53_MODEL_B=32 ...   (the whole Galaxy, all layers)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel

DUMP = Path("bringup/references/glm53_flash/L45_S128")
MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "1,4").split(","))
B = int(os.environ.get("GLM53_MODEL_B", "2"))  # users per row
S = int(os.environ.get("GLM53_MODEL_S", "32"))  # prompt tokens through the decode step (<= 128)
N_PERF = int(os.environ.get("GLM53_MODEL_PERF_STEPS", "20"))


def _layer_range() -> list[int]:
    spec = os.environ.get("GLM53_MODEL_LAYERS", "0-5")
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in spec.split(",")]


@pytest.mark.timeout(10800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=200_000_000)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_mesh_model_chain_and_perf(mesh_device):
    layers = _layer_range()
    cfg = GLM53FlashConfig.from_hf_config_json(default_snapshot_dir() / "config.json")
    model = GLM53MeshModel(mesh_device, cfg, layers, max_batch=B, max_seq_len=256)
    U = model.total_users
    prompt = torch.load(DUMP / "prefill_input.pt")["prompt_ids"].reshape(-1)[:S]  # the dump's random prompt
    ref = torch.load(DUMP / f"layer{layers[-1]}_prefill.pt")["x_out"][0, :S]  # HF chain rows [S, 4, 4096]
    assert layers[0] == 0, "the chain must start at layer 0 (the dump's prompt embeddings)"

    # every user decodes the same prompt (rows must agree across users and with HF's prefill rows)
    ids0 = prompt[0].repeat(U)
    trace_id, (streams_tr, logits_tr, ids_tr) = model.capture(ids0, [0] * U)
    model.reset_kda_states()  # the capture pass advanced the KDA states a second time at position 0
    outs = []
    step_times = []
    for t in range(S):
        model.load_ids(prompt[t].repeat(U))
        model.set_positions([t] * U)
        t0 = time.perf_counter()
        model.replay(trace_id)
        step_times.append(time.perf_counter() - t0)
        out = model.mesh.to_host(streams_tr, rows=2 if model.mesh.rows > 1 else None).float()[0, 0]  # [U, 16384]
        outs.append(out)
    per_user = []
    for u in range(U):
        chain = torch.stack([o[u] for o in outs]).reshape(S, 4, -1)
        per_user.append(pcc(ref, chain))
    per_user.sort()
    # greedy ids of the last position vs the host head on HF's rows
    from models.demos.glm53_flash.reference.checkpoint import FINAL_NORM_KEY, LM_HEAD_KEY, load_tensors, shard_headers

    snapshot = default_snapshot_dir()
    w = load_tensors(snapshot, [FINAL_NORM_KEY, LM_HEAD_KEY], shard_headers(snapshot))
    h = ref.float().mean(1)  # [S, 4096]
    hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * w[FINAL_NORM_KEY].float()
    host_top1 = (hn @ w[LM_HEAD_KEY].float().t()).argmax(-1)  # [S]
    tt_last = model.ids_to_host(ids_tr)  # ids of the last step (position S-1) for every user
    agree_last = (tt_last == host_top1[S - 1]).float().mean().item()
    logger.info(
        f"mesh {MESH} layers {layers[0]}..{layers[-1]} ({len(layers)}), {U} users: chain output after layer {layers[-1]} vs HF "
        f"prefill rows (S={S}): per-user pcc min {per_user[0]:.6f} median {per_user[len(per_user) // 2]:.6f} max {per_user[-1]:.6f}; "
        f"greedy id at position {S - 1} == host head argmax for {agree_last * 100:.0f} % of users (host id {int(host_top1[S - 1])})"
    )
    assert per_user[0] > 0.99
    # traced e2e decode perf: steady-state replays at a fixed position
    model.set_positions([S] * U)
    for _ in range(3):
        model.replay(trace_id)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(N_PERF):
        model.replay(trace_id, blocking=False)
    ttnn.synchronize_device(mesh_device)
    ms = (time.perf_counter() - t0) / N_PERF * 1e3
    logger.info(
        f"PERF mesh {MESH} layers {len(layers)} users/row {B} (total {U}): traced decode step {ms:.2f} ms "
        f"-> {1e3 / ms:.1f} tok/s per user, {U * 1e3 / ms:.0f} tok/s aggregate "
        f"(per layer {ms / len(layers):.2f} ms; the prompt loop above averaged {sum(step_times) / len(step_times) * 1e3:.2f} ms/step blocking)"
    )
    ttnn.release_trace(mesh_device, trace_id)
