# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: where the whole model's run-to-run difference starts. Two runs of test_perf's config (the canonical prompt,
spec target.seq / target.chunk, all layers) give last-chunk hidden states 22-26% apart (top-1 vs text 0.64-0.67),
also with the deterministic combine. This runs the full prefill GLM_DM_RUNS times (default 2) in one process, keeps
every (chunk, layer) output on the host (bf16) and prints, per chunk, the first layer that differs from run 0 and
the rel L2 at a few depths.

    GLM_EXPERTS_MODE=ts TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 BRINGUP_FABRIC_PAYLOAD=8704 \\
    BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml \\
    scripts/run_safe_pytest.sh --mesh --run-all --no-precompile models/demos/glm53_flash_d_p_lb/tests/zz_scratch_determ_model.py -s
"""

import os

import torch

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)


def _rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@mesh_parametrize
def test_determ_model(mesh_device):
    from models.demos.common.bringup.reference.prompt import tokens as prompt_tokens

    seq = int(os.environ.get("GLM_PERF_SEQ", S.get("target.seq")))
    chunk = int(os.environ.get("GLM_PERF_CHUNK", S.get("target.chunk")))
    tokens = prompt_tokens(S, seq).to(torch.long)[:seq]
    layers = S.layers()
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=False)
    runs = []
    for r in range(int(os.environ.get("GLM_DM_RUNS", "2"))):
        out = {}
        for start in range(0, seq, chunk):
            h = model.embed(tokens[start : start + chunk])
            for i in layers:
                h2 = model.layer(i, h, start, None)
                model.free(h)
                h = h2
                out[(start, i)] = model.to_host(h).to(torch.bfloat16)
            model.free(h)
        runs.append(out)
        print(f"[dm] run {r} done", flush=True)
    base = runs[0]
    depth = [layers[0], 3, 4, 5, 8, 12, 20, 30, layers[-1]]
    for r in range(1, len(runs)):
        print(f"[dm] run {r} vs 0: per chunk, the first differing layer and rel L2 at layers {depth}", flush=True)
        for start in range(0, seq, chunk):
            first = next((i for i in layers if not torch.equal(runs[r][(start, i)], base[(start, i)])), None)
            rels = " ".join(f"{_rel(runs[r][(start, i)], base[(start, i)]):.1e}" for i in depth if i in layers)
            print(f"[dm] chunk {start:6d}: first differing layer {first}, rel {rels}", flush=True)
        cfg = model.cfg
        for start in [int(v) for v in os.environ.get("GLM_DM_FULL", "0,5120,51200").split(",")]:
            for i in layers:  # (every layer, with its block type: where the difference grows)
                bt = ("kda" if cfg.is_kda(i) else "dsa") + ("_moe" if cfg.is_moe(i) else "_dense")
                print(
                    f"[dm] full chunk {start:6d} L{i:02d} {bt:9s} rel {_rel(runs[r][(start, i)], base[(start, i)]):.2e}"
                )
