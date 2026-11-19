# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: capture continuation routing outside the runtime audit."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests import stress_fused_decoder as stress
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import pcc
from models.autoports.qwen_qwen3_6_35b_a3b.tt.optimized_decoder import OptimizedDecoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    args.layer, args.seed, args.repeats = 34, 154, 1
    args.prompt_length, args.decode_length, args.continuation_length = 128, 128, 33
    args.oracle_cache = None
    captured = {}
    original_layer, original_topk = stress.layer, ttnn.topk

    def layer(*a, **kw):
        hf = original_layer(*a, **kw)
        captured["hf"] = hf

        def hook(module, inputs, output):
            if inputs[0].shape[1] == 33 and "hf_input" not in captured:
                captured["hf_input"] = inputs[0].clone()
                captured["hf_output"] = output.clone()

        hf.mlp.register_forward_hook(hook)
        return hf

    def topk(x, *a, **kw):
        result = original_topk(x, *a, **kw)
        if tuple(x.shape) == (33, 256):
            captured["tt_logits"] = ttnn.to_torch(x).clone()
            captured["tt_ids"] = ttnn.to_torch(result[1]).clone()
        return result

    class DiagnosticDecoder(OptimizedDecoder):
        def _moe(self, x, **kw):
            out = super()._moe(x, **kw)
            if x.shape[1] == 33:
                captured["tt_input"] = ttnn.to_torch(x).clone()
                captured["tt_output"] = ttnn.to_torch(out).clone()
            return out

    with (
        patch.object(stress, "layer", layer),
        patch.object(stress, "FusedDecoder", DiagnosticDecoder),
        patch.object(stress, "device_only", nullcontext),
        patch.object(ttnn, "topk", topk),
    ):
        try:
            stress.run(args)
        except AssertionError:
            pass
    required = {"hf_input", "hf_output", "tt_input", "tt_output", "tt_logits", "tt_ids"}
    assert required <= captured.keys(), captured.keys()
    hf = captured.pop("hf")
    with torch.no_grad():
        same_input = hf.mlp(captured["tt_input"])
        logits_ttinput = torch.nn.functional.linear(captured["tt_input"], hf.mlp.gate.weight)
        logits_hfinput = torch.nn.functional.linear(captured["hf_input"], hf.mlp.gate.weight)

    def ids(logits):
        return torch.topk(torch.softmax(logits.float(), dim=-1), 8, dim=-1).indices

    rows = []
    for i in (1, 19, 20, 21, 32):
        rows.append(
            dict(
                row=i,
                position=256 + i,
                input_pcc=pcc(captured["hf_input"][:, i], captured["tt_input"][:, i]),
                moe_output_pcc=pcc(captured["hf_output"][:, i], captured["tt_output"][:, i]),
                same_input_moe_pcc=pcc(same_input[:, i], captured["tt_output"][:, i]),
                tt_ids=captured["tt_ids"][i].tolist(),
                hf_ids=ids(logits_hfinput)[0, i].tolist(),
                hf_on_tt_input_ids=ids(logits_ttinput)[0, i].tolist(),
                rounded_tt_logit_ids=ids(captured["tt_logits"].bfloat16())[i].tolist(),
            )
        )
    out = Path(args.output)
    result = json.loads(out.read_text())
    result.update(
        acceptance_eligible=False,
        diagnostic_only=True,
        runtime_fallback_audit="disabled explicitly for intermediate capture",
        routing_diagnostics=rows,
    )
    out.write_text(json.dumps(result, indent=2) + "\n")
    torch.save(captured, str(out) + ".intermediates.pt")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
