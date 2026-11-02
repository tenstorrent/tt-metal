# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Logits-level accuracy of the resident mesh model (tt/mesh_model.py) vs HF on a natural prompt.

The chain PCC of ``test_mesh_model.py`` (raw mHC residual streams after 45 layers) is depth-chaotic for this model
and not a correctness metric (doc/mesh/README.md, chain note). This test builds the model exactly like
``test_mesh_model.py::_ttft`` (same env knobs), prefills the dump's prompt (``GLM53_DUMP``: ``L45_NAT_S128`` = 128
chat-templated natural tokens with HF logits, default ``L45_S128`` = the random prompt), pulls the last layer's
streams to host and runs the host head on them (``reference/logits_check.py``): per-position logits PCC, top-1
agreement, top-5 overlap and KL(HF || device) vs the HF logits of the dump (or vs the host head on the dump's
``x_out`` of the last built layer for a partial chain). Soft bars: ``GLM53_TOP1_BAR`` (default 0.9) on the top-1
agreement rate and ``GLM53_LOGITS_PCC_BAR`` (default 0.95) on the mean logits PCC. The CPU floor of these numbers
(bf16-rounded hidden) is ``tests/test_logits_check.py``.

    GLM53_DUMP=L45_NAT_S128 GLM53_MESH_SHAPE=2,4 GLM53_MODEL_LAYERS=0-44 GLM53_MODEL_B=1 \\
        pytest models/demos/glm53_flash/tests/test_mesh_model_logits.py -s
    GLM53_DUMP=L45_NAT_S128 GLM53_MESH_SHAPE=8,4 GLM53_MODEL_LAYERS=0-44 GLM53_MODEL_B=32 ...   (the whole Galaxy)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir  # noqa: E402
from models.demos.glm53_flash.reference.logits_check import (  # noqa: E402
    compare_logits,
    load_head_weights,
    logits_vs_dump,
)
from models.demos.glm53_flash.tests.utils import pcc  # noqa: E402
from models.demos.glm53_flash.tt.config import GLM53FlashConfig  # noqa: E402
from models.demos.glm53_flash.tt.mesh import mesh_device_params  # noqa: E402
from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel  # noqa: E402

_dump = os.environ.get("GLM53_DUMP", "L45_S128")
DUMP = Path(_dump) if "/" in _dump else Path("bringup/references/glm53_flash") / _dump
MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "1,4").split(","))
B = int(os.environ.get("GLM53_MODEL_B", "2"))  # users per row
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "256"))
_DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
EXPERT_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DTYPE", "bfp8")]  # routed gate/up
EXPERT_DOWN_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DOWN_DTYPE", "bfp8")]  # routed down
_cache_env = os.environ.get("GLM53_MESH_CACHE", "/home/ttuser/ashai/glm53/_cache/mesh")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)
PCC_BAR = float(os.environ.get("GLM53_LOGITS_PCC_BAR", "0.98"))  # mean per-position logits PCC
TOP1_BAR = float(os.environ.get("GLM53_TOP1_BAR", "0.85"))  # top-1 agreement rate
EVERY = int(os.environ.get("GLM53_LOGITS_EVERY", "1"))  # log every n-th position (mismatches always)
N_RUNS = int(os.environ.get("GLM53_LOGITS_RUNS", "2"))  # cold + warm prefills; the last one is graded
JSON_OUT = os.environ.get("GLM53_LOGITS_JSON")  # optional: the per-position report as json


def _layer_range() -> list[int]:
    spec = os.environ.get("GLM53_MODEL_LAYERS", "0-5")
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in spec.split(",")]


@pytest.mark.timeout(10800)
@pytest.mark.parametrize(
    "device_params", [mesh_device_params(*MESH, trace_region_size=TRACE_MB * 1_000_000)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_mesh_model_logits(mesh_device):
    layers = _layer_range()
    assert layers[0] == 0, "the chain must start at layer 0 (the dump's prompt embeddings)"
    assert (DUMP / "prefill_input.pt").exists(), f"dump {DUMP} missing (reference/dump_natural_prompt.py)"
    cfg = GLM53FlashConfig.from_hf_config_json(default_snapshot_dir() / "config.json")
    model = GLM53MeshModel(
        mesh_device,
        cfg,
        layers,
        max_batch=B,
        max_seq_len=512,
        weight_cache_path=CACHE_ROOT,
        expert_dtype=EXPERT_DTYPE,
        expert_down_dtype=EXPERT_DOWN_DTYPE,
    )
    logger.info(
        f"model: mesh {MESH} layers {layers[0]}..{layers[-1]} B/row {B} experts {EXPERT_DTYPE}/{EXPERT_DOWN_DTYPE}; dump {DUMP}"
    )
    prompt = torch.load(DUMP / "prefill_input.pt")["prompt_ids"].reshape(-1)
    S_p = int(prompt.numel())
    ref_streams = torch.load(DUMP / f"layer{layers[-1]}_prefill.pt")["x_out"][0, :S_p]  # [S, 4, 4096] bf16
    head_w = load_head_weights()
    report, dev_logits, prev_logits = None, None, None
    for it in range(N_RUNS):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        streams, logits, owner = model.prefill(prompt, user=0)
        tok = model.first_token(logits, owner)
        wall = time.perf_counter() - t0
        out = (
            model.mesh.to_host(streams, rows=2 if model.mesh.rows > 1 else None).float()[0, 0, :S_p].reshape(S_p, 4, -1)
        )
        report, dev_logits, ref_logits, source = logits_vs_dump(
            out, DUMP, layers[-1], eps=cfg.rms_norm_eps, head_weights=head_w
        )
        label = "cold" if it == 0 else f"warm {it}"
        logger.info(
            f"LOGITS mesh {MESH} layers {len(layers)} prompt {S_p} tokens ({label}, prefill {wall * 1e3:.0f} ms): "
            f"{report.summary('host head on device streams vs HF')} [reference: {source}]; chain pcc vs HF rows "
            f"{pcc(ref_streams, out):.6f} (not a correctness metric)"
        )
        logger.info(
            f"first token: device head {tok}, host head on device streams {int(report.dev_top1[-1])}, HF {int(report.ref_top1[-1])} "
            f"({'match' if tok == int(report.ref_top1[-1]) else 'DIFFERENT'} vs HF; HF top-5 {report.ref_topk[-1].tolist()})"
        )
        if prev_logits is not None:
            logger.info(
                f"run-to-run max |dlogits| vs the previous prefill {float((dev_logits - prev_logits).abs().max()):.4g}"
            )
        prev_logits = dev_logits
        _device_head_last(model, logits, owner, ref_logits[-1:])
        ttnn.deallocate(streams)
        ttnn.deallocate(logits)
    for line in report.rows(every=EVERY):
        logger.info("  " + line)
    if JSON_OUT:
        d = report.as_dict()
        d.update({"mesh": list(MESH), "layers": [layers[0], layers[-1]], "dump": str(DUMP), "first_token_device": tok})
        Path(JSON_OUT).write_text(json.dumps(d, indent=1))
    logger.info(
        f"LOGITS-SUMMARY mesh {MESH} layers {len(layers)} S={S_p}: logits pcc mean {report.mean_pcc:.4f} (bar {PCC_BAR}), "
        f"top-1 {report.top1_rate * 100:.1f} % (bar {TOP1_BAR * 100:.0f} %), top-5 overlap {report.topk_rate * 100:.1f} %, "
        f"KL mean {report.mean_kl:.4f} nats"
    )
    assert report.top1_rate >= TOP1_BAR, f"top-1 agreement {report.top1_rate:.3f} below {TOP1_BAR}"
    assert report.mean_pcc >= PCC_BAR, f"mean logits pcc {report.mean_pcc:.4f} below {PCC_BAR}"


def _device_head_last(model, logits, owner, ref_last: torch.Tensor):
    """The device's own head (bf16 x bfp8 lm_head) on the last position vs HF: the served logits, informational."""
    try:
        shard = model.mesh.device_tensors(logits)[owner * model.mesh.cols]
        dev_last = ttnn.to_torch(shard).float().reshape(-1)[: ref_last.shape[-1]][None]
        rep = compare_logits(dev_last, ref_last)
        logger.info(rep.summary("device head (last position) vs HF"))
    except Exception as e:  # informational only: the host-head metric above is the graded one
        logger.warning(f"device head logits comparison skipped: {e}")
