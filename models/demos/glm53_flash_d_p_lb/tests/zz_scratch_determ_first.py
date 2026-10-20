# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: the folded shared expert's first-call difference. test_ab_layers on layers 2-8 (ts experts, folded_sp):
the process's first call differs from every later one from L5 on (rel 1.6e-3), deterministically; separate / ag do
not. This runs the same chunk GLM_DF_CALLS times (default 3) and captures, per MoE layer and call, the block's
stages on the host: the MoE input x, the shared expert's tiled input / output, y_se (its row-major partials, the
combine's input) and the block output; prints every stage of every call vs the last call.

    GLM_EXPERTS_MODE=ts TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 BRINGUP_FABRIC_PAYLOAD=8704 \\
    BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml \\
    scripts/run_safe_pytest.sh --mesh --no-precompile models/demos/glm53_flash_d_p_lb/tests/zz_scratch_determ_first.py -s
"""

import os
from collections import defaultdict

import torch

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)


def _host(t):
    import ttnn

    return torch.stack([ttnn.to_torch(d).float().reshape(-1) for d in ttnn.get_device_tensors(t)])


def _cmp(a, b):
    a, b = a.double(), b.double()
    rel = float((a - b).norm() / b.norm().clamp_min(1e-30))
    per = [int((a[i] != b[i]).sum()) for i in range(a.shape[0])]
    return f"rel {rel:.3e} differing per chip {per}"


@mesh_parametrize
def test_determ_first(mesh_device):
    from models.demos.common.bringup.reference.prompt import tokens as prompt_tokens
    from models.demos.deepseek_v3_d_p.tt.moe import tt_k3_token_split_moe as k3
    from models.demos.glm53_flash_d_p.tt import experts_ts, shared_tp_row

    chunk = int(S.get("target.chunk"))
    lo, hi = (int(v) for v in os.environ.get("GLM_DF_LAYERS", "2-8").split("-"))
    watch = [int(v) for v in os.environ.get("GLM_DF_WATCH", "4,5").split(",")]
    layers = [i for i in S.layers() if lo <= i <= hi]
    toks = prompt_tokens(S, chunk).to(torch.long)[:chunk]
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=False)
    cap = defaultdict(dict)  # (layer, call) -> stage -> host tensor
    cur = {"layer": None, "call": 0}

    def want():
        return cur["layer"] in watch

    o_call = experts_ts.TtExpertsTs.__call__

    def w_call(self, x, *a, **k):
        cur["layer"] = self.layer
        if want():
            cap[(self.layer, cur["call"])]["x"] = _host(x)
        y = o_call(self, x, *a, **k)
        if want():
            cap[(self.layer, cur["call"])]["y"] = _host(y)
        cur["layer"] = None
        return y

    o_se = shared_tp_row.TtSharedTpRow.__call__

    def w_se(self, x, *a, **k):
        o = o_se(self, x, *a, **k)
        if want():
            c = cap[(cur["layer"], cur["call"])]
            c["se_in"], c["se_out"] = _host(x), _host(o)
        return o

    o_ec = k3.TtK3TokenSplitMoE.experts_combine

    def w_ec(self, *a, **k):
        if want() and getattr(self, "y_se", None) is not None:
            cap[(cur["layer"], cur["call"])]["y_se"] = _host(self.y_se)
        r = o_ec(self, *a, **k)
        if want():
            cap[(cur["layer"], cur["call"])]["out_rm"] = _host(self.out)
        return r

    experts_ts.TtExpertsTs.__call__ = w_call
    shared_tp_row.TtSharedTpRow.__call__ = w_se
    k3.TtK3TokenSplitMoE.experts_combine = w_ec
    outs = []
    try:
        for c in range(int(os.environ.get("GLM_DF_CALLS", "3"))):
            cur["call"] = c
            h = model.embed(toks)
            o = {}
            for i in layers:
                h2 = model.layer(i, h, 0, None)
                model.free(h)
                h = h2
                o[i] = model.to_host(h).float()
            model.free(h)
            outs.append(o)
    finally:
        experts_ts.TtExpertsTs.__call__ = o_call
        shared_tp_row.TtSharedTpRow.__call__ = o_se
        k3.TtK3TokenSplitMoE.experts_combine = o_ec
    last = len(outs) - 1
    for c in range(last):
        for i in layers:
            a, b = outs[c][i].reshape(1, -1), outs[last][i].reshape(1, -1)
            print(f"[df] call {c} vs {last} layer out L{i:02d} {_cmp(a, b)}", flush=True)
        for L in watch:
            for st, t in cap[(L, c)].items():
                ref = cap[(L, last)].get(st)
                if ref is not None and ref.shape == t.shape:
                    print(f"[df] call {c} vs {last} L{L:02d} {st:7s} {_cmp(t, ref)}", flush=True)
