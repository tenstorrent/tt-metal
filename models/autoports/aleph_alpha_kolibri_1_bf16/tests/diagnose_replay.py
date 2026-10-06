# SPDX-License-Identifier: Apache-2.0
"""Diagnostic comparisons only; does not produce acceptance evidence."""

import json

import torch.nn.functional as F

import ttnn

from . import diagnostic_baseline  # Select the archived implementation only in this diagnostic process.
from . import run_coverage as coverage
from .reference import norm
from .run_decoder import pcc

Base = coverage.Harness


class Diagnostic(Base):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.saved = {}
        finish = self.model._finish
        moe = self.model._moe
        linear = self.model._linear

        def save_finish(x, y):
            if x.shape[-2] == 1:
                self.saved["residual"] = x
                self.saved["attention"] = y
            return finish(x, y)

        def save_moe(x):
            result = moe(x)
            if x.shape[-2] == 1:
                self.saved["ffn_input"] = x
                self.saved["moe"] = result
            return result

        def save_linear(x, w, **kwargs):
            result = linear(x, w, **kwargs)
            if w is self.model.router and x.shape[-2] == 1:
                self.saved["logits"] = result
            return result

        self.model._finish = save_finish
        self.model._moe = save_moe
        self.model._linear = save_linear

    def check(self, name, expected, actual, **metadata):
        # Diagnostics deliberately continue through failed PCC, never acceptance.
        print("DIAG_CASE", name, pcc(expected, actual), metadata, flush=True)
        if name != "changed_input_page_position_replay" or metadata["position"] != 33:
            return
        host = {k: ttnn.to_torch(v) for k, v in self.saved.items()}
        x = host["residual"].reshape(1, 1, 2560)
        w = self.weights
        attention = self.ref.attention(norm(x, w["input_layernorm.weight"]), 33)
        y = x + norm(attention, w["post_attn_norm.weight"])
        ff = norm(y, w["post_attention_layernorm.weight"])
        logits = F.linear(ff.float(), w["mlp.gate.weight"].float()).reshape(1, 384)
        same = host["ffn_input"].reshape_as(ff)
        same_logits = F.linear(same.float(), w["mlp.gate.weight"].float()).reshape(1, 384)
        bias = w["moe.router.expert_bias"].float()
        result = {
            "output_pcc": pcc(expected, actual),
            "attention_pcc": pcc(attention, host["attention"]),
            "ffn_input_pcc": pcc(ff, same),
            "moe_pcc": pcc(self.ref.moe(ff), host["moe"]),
            "same_input_moe_pcc": pcc(self.ref.moe(same), host["moe"]),
            "ref_top8": (logits + bias).topk(8).indices.tolist(),
            "ref_top8_scores": (logits + bias).topk(8).values.tolist(),
            "same_input_top8": (same_logits + bias).topk(8).indices.tolist(),
            "tt_top8": (host["logits"].reshape(1, 384) + bias).topk(8).indices.tolist(),
            "tt_top8_scores": (host["logits"].reshape(1, 384) + bias).topk(8).values.tolist(),
        }
        print("REPLAY_DIAGNOSTIC", json.dumps(result), flush=True)
        (coverage.ROOT / "doc/functional_decoder/synthetic_replay_diagnostic.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        raise SystemExit(0)


coverage.Harness = Diagnostic
coverage.run(4, synthetic=True)
