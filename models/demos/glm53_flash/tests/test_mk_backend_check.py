# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The backend's residual after a prefix of layers vs the HF goldens, token by token: StackBackend(layers=0..L) on a
mesh, one user, the golden's tokens fed through eager decode steps, the stacked residual (kv.res) read after every
step and compared with HF's x_out of layer L at that step (reference/hf_layer_reference.py --layers 0-4 --seq-len 1
--num-decode 3 --stream). Bisects the embed -> layers -> residual glue, the DSA layer and the state carry.

    GLM53_MESH_SHAPE=8,4 GLM53_FABRIC=1d GLM53_MK_EP=1 GLM53_STACK_DSA=1 GLM53_MK_LAYERS=0-3 GLM53_MK_STEPS=4 pytest ...
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import optimized_decoder as OD
from models.autoports.zai_org_glm_5_3_flash.tt.config import load_hf_text_config
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tests.test_mk_backend_probe import HFSource, _layers
from models.demos.glm53_flash.tests.test_mk_layer_check import pcc
from models.demos.glm53_flash.tt.mesh import mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
GOLDEN = os.environ.get("GLM53_MK_GOLDEN", "/home/ttuser/ashai/glm53/_logs/mesh/hf_golden_t1d3")
STEPS = int(os.environ.get("GLM53_MK_STEPS", "4"))
SEQ = int(os.environ.get("GLM53_MK_SEQ", "2048"))
TRACED = os.environ.get("GLM53_MK_TRACED", "0") == "1"  # replay the captured decode trace instead of eager steps
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "512"))
WORKER_L1 = int(os.environ.get("GLM53_MK_WORKER_L1", "1264512"))


@pytest.mark.timeout(5000)
@pytest.mark.parametrize(
    "device_params",
    [mesh_device_params(*MESH, trace_region_size=TRACE_MB << 20, worker_l1_size=WORKER_L1)],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_mk_backend_check(mesh_device, monkeypatch):
    from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import StackBackend

    snapshot = default_snapshot_dir()
    cfg = load_hf_text_config(str(snapshot))
    source = HFSource(snapshot, cfg)
    monkeypatch.setattr(OD, "FdTileCacheSource", lambda cache_dir, manifest: source)
    layers = _layers()
    L = layers[-1]
    golds = [torch.load(os.path.join(GOLDEN, f"layer{L}_prefill.pt"))] + [
        torch.load(os.path.join(GOLDEN, f"layer{L}_decode{k}.pt")) for k in range(STEPS - 1)
    ]
    first = int(torch.load(os.path.join(GOLDEN, "prefill_input.pt"))["prompt_ids"][0, 0])
    tokens = [first] + [int(g["token"][0, 0]) for g in golds[1:]]
    ep = os.environ.get("GLM53_MK_EP", "1") == "1"
    t0 = time.perf_counter()
    backend = StackBackend(mesh_device, max_batch=1, layers=layers, hf_config=cfg, topk=1, sample=False, ep_weights=ep)
    logger.info(f"StackBackend built in {time.perf_counter() - t0:.0f} s: mesh {MESH}, layers {layers}")
    kv = backend.allocate_kv_cache(max_batch=1, max_seq_len=SEQ)
    io = backend.allocate_decode_io(kv)
    backend.write(io.page_table, torch.arange(kv.blocks_per_seq, dtype=torch.int32).reshape(1, kv.blocks_per_seq))
    backend.write(io.cur_pos, torch.zeros(1, dtype=torch.int32))
    H = cfg.hidden_size
    worst = 1.0
    tr = None
    for step, (tok, gold) in enumerate(zip(tokens, golds)):
        backend.write(io.tokens, torch.tensor([tok], dtype=torch.int32))
        if TRACED and step > 0:
            if tr is None:
                tr = backend.capture_decode(io, kv, 1)
            backend.replay(tr)
        else:
            backend.decode(io, kv, 1)
        ttnn.synchronize_device(mesh_device)
        res = ttnn.to_torch(ttnn.get_device_tensors(kv.res[0])[0]).float()  # [1, 1, 4, H] at one row per chunk
        o = res.reshape(-1, H)[:4]
        g = gold["x_out"].float()[0, 0].reshape(4, H)
        gi = gold["x_in"].float()[0, 0].reshape(4, H)
        p = pcc(o, g)
        p = p if p == p else 0.0  # NaN (a NaN / constant residual) counts as a miss
        per_stream = [round(pcc(o[s_], g[s_]), 5) for s_ in range(4)]
        sampled = backend.read(io.sampled)
        stats = (
            f"nan {int(torch.isnan(o).sum())} inf {int(torch.isinf(o).sum())} zero {int((o == 0).sum())} "
            f"absmax {float(o.nan_to_num(0).abs().max()):.3g} std {float(o.nan_to_num(0).std()):.3g}"
        )
        logger.info(
            f"MK_BACKEND_CHECK: layers {layers[0]}-{L} mesh {MESH} step {step} token {tok}: residual pcc {p:.6f} per "
            f"stream {per_stream} (res shape {tuple(res.shape)}; out {stats}; golden std {g.std():.4f}, x_in(L) std "
            f"{gi.std():.4f}); sampled {int(sampled[0])}"
        )
        worst = min(worst, p)
    logger.info(
        f"MK_BACKEND_CHECK_SUMMARY: layers {layers[0]}-{L} mesh {MESH} traced {TRACED} steps {STEPS}: worst pcc {worst:.6f}"
    )
    assert worst > 0.98, f"residual pcc {worst}"
