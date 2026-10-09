# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""In-model A/B check on a few layers, minutes instead of a whole-model run: build GLM_AB_LAYERS (default 2-4: one
kda_dense, dsa_moe, kda_moe), run one chunk (GLM_AB_CHUNK, default the spec's; GLM_AB_START, default 0: a later start reuses the KDA carry the previous run advanced) and
read back every layer's output. Runs:
  base, base again (determinism: must be bit-identical; a difference means a race / uninitialized read), then
  each switch in GLM_AB_SWITCHES flipped on the same tokens and state.
A switch is "module.ATTR=value" on a models.demos.glm53_flash_d_p.tt module read at call time, e.g.
  GLM_AB_SWITCHES="mla_attention.USE_BMM=1"   (comma: separate runs; "+": switches combined in one run)
Prints, per layer, rel L2 and PCC of each run vs the base run (the earlier layers locate where a change first shows).
The layers' caches are rewritten at the same positions by every run, so each run sees the same state."""

import importlib
import os

import torch

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)


def _cmp(a: torch.Tensor, b: torch.Tensor):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    rel = float((a - b).norm() / b.norm().clamp_min(1e-30))
    ac, bc = a - a.mean(), b - b.mean()
    pcc = float((ac * bc).sum() / (ac.norm() * bc.norm()).clamp_min(1e-30))
    return rel, pcc, int((a != b).sum())


@mesh_parametrize
def test_ab_layers(mesh_device):
    from models.demos.common.bringup.reference.prompt import tokens as prompt_tokens

    chunk = int(os.environ.get("GLM_AB_CHUNK", S.get("target.chunk")))
    start = int(os.environ.get("GLM_AB_START", "0"))  # 0: KDA starts from its zero state every run
    lo, hi = (int(v) for v in os.environ.get("GLM_AB_LAYERS", "2-4").split("-"))
    layers = [i for i in S.layers() if lo <= i <= hi]
    toks = prompt_tokens(S, start + chunk).to(torch.long)[start : start + chunk]
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=False)
    if os.environ.get("GLM_AB_NO_PROGRAM_CACHE") == "1":  # every op compiles fresh (isolates program-cache hits)
        mesh_device.disable_and_clear_program_cache()

    def run():
        outs = {}
        h = model.embed(toks)
        for i in layers:
            h2 = model.layer(i, h, start, None)
            model.free(h)
            h = h2
            outs[i] = model.to_host(h).float()
        model.free(h)
        return outs

    def flip(sw):
        mod, rest = sw.split(".", 1)
        attr, val = rest.split("=")
        m = importlib.import_module(f"models.demos.glm53_flash_d_p.tt.{mod}")
        old = getattr(m, attr)
        new = type(old)(int(val)) if isinstance(old, (bool, int)) else type(old)(val)
        setattr(m, attr, new)
        return m, attr, old

    base = run()
    again = run()
    print(f"[ab] layers {layers}, chunk {chunk}, start {start}", flush=True)
    for i in layers:
        rel, pcc, nd = _cmp(again[i], base[i])
        print(f"[ab] base-again  L{i:02d} rel {rel:.3e} pcc {pcc:.7f} differing {nd}", flush=True)
    for sw in [s for s in os.environ.get("GLM_AB_SWITCHES", "").split(",") if s]:
        flipped = [flip(one) for one in sw.split("+")]  # "a.X=1+b.Y=0": several switches in one run
        try:
            out = run()
        finally:
            for m, attr, old in flipped:
                setattr(m, attr, old)
        for i in layers:
            rel, pcc, nd = _cmp(out[i], base[i])
            print(f"[ab] {sw:28s} L{i:02d} rel {rel:.3e} pcc {pcc:.7f} differing {nd}", flush=True)
