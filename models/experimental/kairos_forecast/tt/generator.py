# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""generate(): the reference's KairosModel.generate on TtKairos, with the host front-end and RouterState.

Order (modeling_kairos.py generate / _autoregressive_generate at 16eb4b98): the flip wraps the whole rollout, so
all x steps run first, then all -x steps, and every forward routes on the bias the previous one left. When the
horizon fits one forward (H <= 128) the two halves are routed on the host in that order and then run on the
device as one batch of 2S rows: the device output does not feed the routing, so the bias sequence is unchanged.
"""
from __future__ import annotations

import torch

from models.experimental.kairos_forecast.reference.dataflow import head_to_quantiles
from models.experimental.kairos_forecast.tt.host_frontend import front_end

STEP = 128  # horizon per forward (2 decoder tokens x 64)
MEDIAN = 4


def _pad_front(f, n):
    """Pad a front to n tokens: masked tokens on a null expert with zero weight (zero embedding input)."""
    B, N = f["attn_mask"].shape
    g = dict(f)
    g["reg_pos"] = f.get("reg_pos", torch.full((B,), N))
    if N == n:
        return g
    g["x_final"] = torch.cat([f["x_final"], torch.zeros(3, B, n - N, f["x_final"].shape[-1])], 2)
    g["weights"] = torch.cat([f["weights"], torch.zeros(B, n - N, 3)], 1)
    g["indices"] = torch.cat([f["indices"], torch.full((B, n - N, 3), 3, dtype=f["indices"].dtype)], 1)
    g["attn_mask"] = torch.cat([f["attn_mask"], torch.zeros(B, n - N, dtype=torch.bool)], 1)
    return g


def _cat_fronts(a, b):
    n = max(a["attn_mask"].shape[1], b["attn_mask"].shape[1])
    a, b = _pad_front(a, n), _pad_front(b, n)
    out = {}
    for k in a:
        if k == "x_final":
            out[k] = torch.cat([a[k], b[k]], 1)
        elif k == "loc_scale":
            out[k] = tuple(torch.cat([x, y]) for x, y in zip(a[k], b[k]))
        else:
            out[k] = torch.cat([a[k], b[k]])
    return out


class KairosGenerator:
    def __init__(self, model, router):
        self.model, self.router, self.W = model, router, model.W
        self.log = []  # per forward: {"head", "routing", "rows"}

    def _forward(self, front):
        return self.model.forward(front)

    def _rollout(self, past, horizon):
        preds = []
        for _ in range(-(-horizon // STEP)):
            front, routing = front_end(past, self.router, self.W)
            head = self._forward(front)
            self.log.append({"head": head, "routing": routing, "context": past})
            q = head_to_quantiles(head, *front["loc_scale"])
            preds.append(q)
            past = torch.cat([past, q[:, MEDIAN]], -1)
        return torch.cat(preds, -1)[..., :horizon]

    def generate(self, past_target, horizon, force_flip_invariance=True, infer_is_positive=True):
        """past_target [S, T] -> quantiles [S, 9, horizon]. Uses (and advances) self.router's bias."""
        past_target = past_target.float()
        self.log = []
        S = past_target.shape[0]
        if not force_flip_invariance:
            pred = self._rollout(past_target, horizon)
        elif horizon <= STEP:
            fa, ra = front_end(past_target, self.router, self.W)
            fb, rb = front_end(-past_target, self.router, self.W)
            front = _cat_fronts(fa, fb)
            head = self._forward(front)
            self.log += [{"head": head[:S], "routing": ra, "context": past_target},
                         {"head": head[S:], "routing": rb, "context": -past_target}]
            q = head_to_quantiles(head, *front["loc_scale"])[..., :horizon]
            pred = (q[:S] - torch.flip(q[S:], dims=[1])) / 2
        else:
            a = self._rollout(past_target, horizon)
            b = self._rollout(-past_target, horizon)
            pred = (a - torch.flip(b, dims=[1])) / 2
        if infer_is_positive:
            pos = (~torch.any(past_target < 0, dim=1)).view(-1, 1, 1)
            pred = torch.where(pos, torch.maximum(pred, torch.tensor(0.0)), pred)
        return pred
