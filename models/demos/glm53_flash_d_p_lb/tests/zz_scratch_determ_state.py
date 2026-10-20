# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: run-to-run vs device-to-reference, state and outputs. The full prefill (the s56320 golden's prompt,
5120 chunks) GLM_DS_RUNS times (default 2); per run: every layer's state after the prefill (MLA latent cache, indexer
pooled keys, KDA recurrent / conv), the last chunk's per-layer outputs and its routing (the joint rows' expert ids).
Prints, per layer: run 1 vs run 0 and run 0 vs the golden (fp32 CPU reference) - rel L2 of the outputs and of each
state tensor (the caches also per position range), and the fraction of the last chunk's tokens whose expert set
differs between the runs.

    GLM_EXPERTS_MODE=ts TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 BRINGUP_FABRIC_PAYLOAD=8704 \\
    BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml \\
    scripts/run_safe_pytest.sh --mesh --run-all --no-precompile models/demos/glm53_flash_d_p_lb/tests/zz_scratch_determ_state.py -s
"""

import os

import torch
from safetensors.torch import load_file

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)


def _rel(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@mesh_parametrize
def test_determ_state(mesh_device):
    import ttnn
    from models.demos.common.bringup.reference.golden import Golden
    from models.demos.glm53_flash_d_p.tt import router as router_mod

    g = Golden.for_rung(S, "s56320")
    gdir = S.golden_root / "s56320_c5120"
    tokens = g.tokens().to(torch.long)
    seq, chunk = 56320, 5120
    last = seq - chunk
    layers = S.layers()
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=False)
    cfg = model.cfg
    K = cfg.num_experts_per_tok
    o_joint = router_mod.TtRouter.joint
    cur = {"layer": None, "cap": False, "rt": {}}
    rid = {}
    for b in model.blocks.values():
        if getattr(b, "router", None) is not None:
            rid[id(b.router)] = b.i

    def w_joint(self, x):
        rows = o_joint(self, x)
        if cur["cap"]:
            per = [ttnn.to_torch(t).reshape(-1, 2 * K)[:, :K].to(torch.int32) for t in ttnn.get_device_tensors(rows)]
            cur["rt"][rid[id(self)]] = torch.cat(per).sort(-1).values
        return rows

    router_mod.TtRouter.joint = w_joint
    runs = []
    try:
        for r in range(int(os.environ.get("GLM_DS_RUNS", "2"))):
            outs, cur["rt"] = {}, {}
            for start in range(0, seq, chunk):
                cur["cap"] = start == last
                h = model.embed(tokens[start : start + chunk])
                for i in layers:
                    h2 = model.layer(i, h, start, None)
                    model.free(h)
                    h = h2
                    if start == last:
                        outs[i] = model.to_host(h).to(torch.bfloat16)
                model.free(h)
            st = {i: {k: v.clone() for k, v in model.blocks[i].state_torch(seq).items()} for i in layers}
            runs.append((outs, dict(cur["rt"]), st))
            print(f"[ds] run {r} done", flush=True)
    finally:
        router_mod.TtRouter.joint = o_joint
    (o0, r0, s0), (o1, r1, s1) = runs[0], runs[1]
    # (position ranges of the caches: every 5120 tokens; the pooled keys every 5120 / index_kpool)
    print(
        "[ds] per layer: last-chunk output rel (run1 vs run0 | run0 vs golden); routing sets differing run1 vs run0; "
        "state rel (run1 vs run0 | run0 vs golden)",
        flush=True,
    )
    for i in layers:
        gl = load_file(str(gdir / "chunk_10" / f"layer_{i:02d}.safetensors"))
        gk = load_file(str(gdir / "kv_cache" / f"layer_{i}.safetensors"))
        go = gl["out"].float()
        line = f"[ds] L{i:02d} out {_rel(o1[i], o0[i]):.2e} | {_rel(o0[i].float().reshape(go.shape), go):.2e}"
        if i in r0 and i in r1:
            line += f"  route {float((r0[i] != r1[i]).any(-1).float().mean()):.4f}"
        for k, v0 in s0[i].items():
            gv = gk.get(f"{k}_cache_layer_{i}")
            v1 = s1[i][k]
            gr = ""
            if gv is not None:
                gv = gv.float()
                n = min(gv.shape[0], v0.shape[0]) if k in ("kv_latent", "index_key") else None
                gr = f"{_rel(v0[:n], gv[:n]) if n else _rel(v0.reshape(gv.shape), gv):.2e}"
            line += f"  {k} {_rel(v1, v0):.2e} | {gr}"
            if k in ("kv_latent", "index_key"):  # (per position range: early, middle, last chunk)
                pool = 1 if k == "kv_latent" else cfg.index_kpool
                segs = [(0, chunk), (seq // 2, seq // 2 + chunk), (last, seq)]
                line += (
                    " ["
                    + " ".join(f"{_rel(v1[a // pool:b // pool], v0[a // pool:b // pool]):.1e}" for a, b in segs)
                    + "]"
                )
        print(line, flush=True)
