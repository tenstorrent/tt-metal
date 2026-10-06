# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: host intermediate comparisons outside audited forwards."""

import json
import os

import torch.nn.functional as F

import ttnn

from . import diagnostic_baseline  # Select the archived implementation only in this diagnostic process.
from . import run_coverage as coverage
from .reference import norm
from .run_decoder import pcc

Base = coverage.Harness


class DiagnosticHarness(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture = False
        if os.environ.get("DIAG_SPLIT_NORM"):
            old_norm = self.model._norm

            def split_norm(x, name):
                if os.environ.get("DIAG_SPLIT_NORM") != "all" and name not in os.environ["DIAG_SPLIT_NORM"].split(","):
                    return old_norm(x, name)
                unit = ttnn.rms_norm(
                    x,
                    epsilon=self.model.config.rms_norm_eps,
                    compute_kernel_config=self.model.compute,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                return ttnn.multiply(unit, self.model.norms[name])

            self.model._norm = split_norm
        original = self.model._finish

        def finish(residual, attention):
            if self.capture:
                self.saved = (residual, attention)
            return original(residual, attention)

        self.model._finish = finish

    def check(self, name, expected, actual, **metadata):
        if name == "nonaligned_prefix_continuation":
            self.capture = True
        if name != "decode_after_nonaligned_continuation":
            return super().check(name, expected, actual, **metadata)
        residual, attention = self.saved
        x = ttnn.to_torch(residual).reshape(1, 1, 2560)
        w = self.weights
        ref_att = self.ref.attention(norm(x, w["input_layernorm.weight"]), 302)
        tt_att = ttnn.to_torch(attention).reshape_as(ref_att)
        tt_post = self.model._norm(attention, "post_attn_norm")
        tt_y = ttnn.add(residual, tt_post)
        ref_y = x + norm(ref_att, w["post_attn_norm.weight"])
        tt_ffn_in = self.model._norm(tt_y, "post_attention_layernorm")
        host_ffn_in = ttnn.to_torch(tt_ffn_in).reshape_as(x)
        ref_ffn_in = norm(ref_y, w["post_attention_layernorm.weight"])
        tt_moe = ttnn.to_torch(self.model._moe(tt_ffn_in)).reshape_as(x)
        ref_moe = self.ref.moe(ref_ffn_in)
        same_input_moe = self.ref.moe(host_ffn_in)
        tt_logits_dev = self.model._linear(
            ttnn.typecast(tt_ffn_in, ttnn.float32), self.model.router, dtype=ttnn.float32
        )
        tt_logits = ttnn.to_torch(tt_logits_dev).reshape(1, 384)
        ref_logits = F.linear(ref_ffn_in.float(), w["mlp.gate.weight"].float()).reshape(1, 384)
        same_logits = F.linear(host_ffn_in.float(), w["mlp.gate.weight"].float()).reshape(1, 384)
        bias = w["moe.router.expert_bias"].float()
        tt_ids = (
            ttnn.to_torch(ttnn.topk(ttnn.add(tt_logits_dev, self.model.expert_bias), k=6, dim=-1)[1])
            .reshape(-1)
            .tolist()
        )
        result = {
            "output_pcc": pcc(expected, actual),
            "attention_pcc": pcc(ref_att, tt_att),
            "post_attention_residual_pcc": pcc(ref_y, ttnn.to_torch(tt_y)),
            "ffn_input_pcc": pcc(ref_ffn_in, host_ffn_in),
            "moe_pcc": pcc(ref_moe, tt_moe),
            "moe_same_input_pcc": pcc(same_input_moe, tt_moe),
            "router_pcc": pcc(ref_logits, tt_logits),
            "router_same_input_pcc": pcc(same_logits, tt_logits),
            "ref_top6": (ref_logits + bias).topk(6).indices.tolist(),
            "tt_top6": tt_ids,
            "same_input_top6": (same_logits + bias).topk(6).indices.tolist(),
        }
        print("DIAGNOSTIC", json.dumps(result), flush=True)
        (
            coverage.ROOT
            / "doc/functional_decoder"
            / ("synthetic_diagnostic_" + os.environ.get("DIAG_SPLIT_NORM", "baseline") + ".json")
        ).write_text(json.dumps(result, indent=2) + "\n")
        raise SystemExit(0)


coverage.Harness = DiagnosticHarness
coverage.run(4, synthetic=True)
