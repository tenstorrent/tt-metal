# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The row-local prefill wave (tt-blaze G53F-0250) graded against the CPU golden, side by side with the SP prefill.

All 45 layers on the seed server's 8x4 geometry (run with the server's env). Per wave, every mesh row's final streams
go through the host head at EVERY position; the per-position top-1 is compared with the golden's own top-1 (the host
head on the golden ``decoder_output_layer_44`` rows, the same rows the CPU model produced) -- the teacher-forced
agreement of GZ4 / G53F-0069, here per prompt row. The same prompts through the existing SP prefill give the bar each
wave row must meet: agreement >= SP's - ``GLM53_WAVE_GOLDEN_TOL`` (default 0.005 = 10 of 2048 positions) and final
stream PCC vs the golden >= SP's - 0.002.

* wave A: the first ``GLM53_WAVE_GOLDEN_S`` (2048) tokens of the golden ``long`` job in all 8 rows (the BS 16 serving
  case: one prompt in every slot; 8 samples of the same prompt at 8 row positions);
* wave B: 256 tokens of ``mu0`` / ``mu1`` / ``long`` cycled over the 8 rows (different prompts side by side).

    (seed server env, MAXSEQ=1048576) pytest models/demos/glm53_flash/tests/test_prefill_wave_golden.py -s
"""

from __future__ import annotations

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import (
    FINAL_NORM_KEY,
    LM_HEAD_KEY,
    default_snapshot_dir,
    load_tensors,
    shard_headers,
)
from models.demos.glm53_flash.tests.test_prefill_vs_blaze_golden import _meta, _streams_host, _trace
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
MAX_SEQ = int(os.environ.get("GLM53F_PREFILL_MAXSEQ", "8192"))
S_LONG = int(os.environ.get("GLM53_WAVE_GOLDEN_S", "2048"))
TOL = float(os.environ.get("GLM53_WAVE_GOLDEN_TOL", "0.005"))
PCC_TOL = 0.002
_DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
EXPERT_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DTYPE", "bfp8")]
EXPERT_DOWN_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DOWN_DTYPE", "bfp8")]
_cache_env = os.environ.get("GLM53_MESH_CACHE", "/mnt/tt-data/sdawle/glm53f/prefill/mesh_cache")
CACHE_ROOT = None if _cache_env.lower() == "none" else _cache_env
OUT = os.environ.get("GLM53_WAVE_GOLDEN_OUT", "/mnt/tt-data/sdawle/glm53f/prefill/out/wave_golden.json")


def _job_tokens(job: str) -> list[int]:
    m = _meta(job)
    return list(m["prompt_token_ids"]) + list(m["completion_token_ids"])


class _Head:
    """The host LM head (final norm on the unweighted stream mean, then the vocab matmul) -> per-position top-1."""

    def __init__(self, cfg):
        snap = default_snapshot_dir()
        w = load_tensors(snap, [FINAL_NORM_KEY, LM_HEAD_KEY], shard_headers(snap))
        self.norm_w, self.head_w, self.eps = (
            w[FINAL_NORM_KEY].float(),
            w[LM_HEAD_KEY].to(torch.bfloat16),
            cfg.rms_norm_eps,
        )

    def top1(self, streams: torch.Tensor) -> torch.Tensor:
        S = streams.shape[0]
        h = streams.float().reshape(S, 4, -1).mean(1)
        hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps) * self.norm_w
        out = []
        for a in range(0, S, 256):  # 256 rows x 154880 logits at a time
            out.append((hn[a : a + 256].to(torch.bfloat16) @ self.head_w.t()).float().argmax(-1))
        return torch.cat(out)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_prefill_wave_vs_golden(mesh_device):
    torch.set_num_threads(min(32, os.cpu_count() or 8))
    cfg = GLM53FlashConfig.from_hf_config_json(default_snapshot_dir() / "config.json")
    t0 = time.perf_counter()
    from pathlib import Path

    model = GLM53MeshModel(
        mesh_device,
        cfg,
        list(range(45)),
        max_batch=1,
        max_seq_len=MAX_SEQ,
        weight_cache_path=None if CACHE_ROOT is None else Path(CACHE_ROOT),
        expert_dtype=EXPERT_DTYPE,
        expert_down_dtype=EXPERT_DOWN_DTYPE,
    )
    rows = model.mesh.rows
    logger.info(f"[wave-golden] model built in {time.perf_counter() - t0:.0f} s (max_seq {MAX_SEQ})")
    head = _Head(cfg)

    gold_cache: dict = {}

    def golden(job: str, S: int):
        """The golden's final-layer rows [0, S) and their host-head top-1 (cached per job)."""
        if job not in gold_cache:
            g = _trace(job, "decoder_output_layer_44").float()
            gold_cache[job] = (g, head.top1(g[: max(S_LONG, 256)]))
        g, t = gold_cache[job]
        return g[:S], t[:S]

    def grade(streams_host: torch.Tensor, job: str, S: int) -> dict:
        g, gt = golden(job, S)
        top = head.top1(streams_host)
        return dict(
            agree=round(float((top == gt).float().mean()), 6),
            agree_n=int((top == gt).sum()),
            first_token=int(top[-1]),
            golden_first=int(gt[-1]),
            stream_pcc=round(pcc(g, streams_host.float()), 6),
        )

    def sp(job: str, S: int) -> dict:
        ids = torch.tensor(_job_tokens(job)[:S], dtype=torch.int64)
        ttnn.synchronize_device(mesh_device)
        t = time.perf_counter()
        st, lg, _ = model.prefill(ids, user=0)
        ttnn.synchronize_device(mesh_device)
        wall = time.perf_counter() - t
        host = _streams_host(model, st, S)
        ttnn.deallocate(st)
        ttnn.deallocate(lg)
        return dict(job=job, S=S, wall_s=round(wall, 3), **grade(host, job, S))

    def wave(jobs: list[str], S: int) -> list[dict]:
        prompts = [torch.tensor(_job_tokens(j)[:S], dtype=torch.int64) for j in jobs]
        walls = []
        for _ in range(2):  # the first call compiles; the second is the wave's wall
            ttnn.synchronize_device(mesh_device)
            t = time.perf_counter()
            st, lg, _ = model.prefill_wave_ids(prompts, user=0)
            ttnn.synchronize_device(mesh_device)
            walls.append(round(time.perf_counter() - t, 3))
            toks = model.wave_first_tokens(lg)
            ttnn.deallocate(lg)
            if len(walls) == 1:
                ttnn.deallocate(st)
        res = []
        for r, j in enumerate(jobs):
            host = model.mesh.to_host(st, row=r).float()[0, 0, :S]
            res.append(dict(row=r, job=j, S=S, device_first_token=int(toks[r]), walls_s=walls, **grade(host, j, S)))
        ttnn.deallocate(st)
        return res

    report: dict = {"rows": rows, "S_long": S_LONG, "tol": TOL}
    # SP references first (each prompt alone, the server's existing path)
    sp_ref = {("long", S_LONG): sp("long", S_LONG)}
    for j in ("mu0", "mu1", "long"):
        sp_ref[(j, 256)] = sp(j, 256)
    report["sp"] = [v for v in sp_ref.values()]
    for v in report["sp"]:
        logger.info(f"[wave-golden] SP {v}")
    report["wave_A"] = wave(["long"] * rows, S_LONG)
    report["wave_B"] = wave([("mu0", "mu1", "long")[r % 3] for r in range(rows)], 256)
    bad = []
    for key in ("wave_A", "wave_B"):
        for v in report[key]:
            ref = sp_ref[(v["job"], v["S"])]
            v["agree_minus_sp"] = round(v["agree"] - ref["agree"], 6)
            v["pcc_minus_sp"] = round(v["stream_pcc"] - ref["stream_pcc"], 6)
            logger.info(f"[wave-golden] {key} {v}")
            if v["agree"] < ref["agree"] - TOL or v["stream_pcc"] < ref["stream_pcc"] - PCC_TOL:
                bad.append((key, v["row"], v["job"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)
    print(f"[wave-golden] {json.dumps(report)}", flush=True)
    assert not bad, f"wave rows less accurate than the SP prefill vs the golden: {bad}"
