# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only GDN numerical diagnosis; no TTNN imports or device execution.

Controls literal HF chunk against literal HF recurrence on identical real-weight
projections. Separately emulate the model's BF16 convolution expression and an
explicitly hypothetical TF32-truncated recurrent-state decay. Neither emulation
is asserted to reproduce hardware: a device core-only probe is needed to select
between upstream projection/conv error and fused-kernel accumulation error.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextGatedDeltaNet,
    l2norm,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights


def metrics(expected, actual):
    a, b = expected.float().reshape(-1, expected.shape[-1]), actual.float().reshape(-1, actual.shape[-1])
    raw_b = b
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    values = (a * b).sum(-1) / (a.square().sum(-1) * b.square().sum(-1)).sqrt()
    finite_pcc = values[torch.isfinite(values)]
    worst = int(torch.where(torch.isfinite(values), values, torch.inf).argmin())
    raw_a = expected.float().reshape(-1, expected.shape[-1])
    return {
        "minimum_row_pcc": finite_pcc.min().item() if finite_pcc.numel() else None,
        "mean_row_pcc": finite_pcc.mean().item() if finite_pcc.numel() else None,
        "final_row_pcc": values[-1].item() if torch.isfinite(values[-1]) else None,
        "invalid_pcc_rows": int((~torch.isfinite(values)).sum()),
        "actual_nonfinite_elements": int((~torch.isfinite(raw_b)).sum()),
        "actual_nonzero_elements": int(torch.count_nonzero(raw_b)),
        "actual_zero_rows": int((raw_b == 0).all(-1).sum()),
        "actual_constant_rows": int((b == 0).all(-1).sum()),
        "actual_maximum_absolute_value": raw_b.abs().max().item(),
        "maximum_absolute_error": (expected.float() - actual.float()).abs().max().item(),
        "worst_row": {
            "flat_index": worst,
            "pcc": values[worst].item() if torch.isfinite(values[worst]) else None,
            "expected_max_abs": raw_a[worst].abs().max().item(),
            "actual_max_abs": raw_b[worst].abs().max().item(),
            "expected_centered_l2": a[worst].norm().item(),
            "actual_centered_l2": b[worst].norm().item(),
            "max_abs_error": (raw_a[worst] - raw_b[worst]).abs().max().item(),
        },
    }


def truncate_tf32(x):
    return (x.contiguous().view(torch.int32) & -8192).view(torch.float32)


@torch.inference_mode()
def run(args):
    torch.set_num_threads(8)
    torch.manual_seed(2194 + args.layer)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    weights = load_weights(args.snapshot, args.layer, "linear_attn.")
    with torch.device("meta"):
        model = Qwen4ExpTextGatedDeltaNet(config, args.layer)
    model.load_state_dict({k.removeprefix("linear_attn."): v for k, v in weights.items()}, assign=True)
    model.eval()
    hidden = (torch.randn(1, args.length, 2560) * 0.2).bfloat16()
    projected = model.in_proj_qkv(hidden)
    history = F.pad(projected, (0, 0, 3, 0))
    taps = model.conv1d.weight[:, 0, :]
    products = [(history[:, i : i + args.length].float() * taps[:, i].float()).bfloat16() for i in range(4)]
    bf16_conv = F.silu((products[0] + products[1]) + (products[2] + products[3]))
    hf_conv = F.silu(model.conv1d(projected.transpose(1, 2))[..., : args.length]).transpose(1, 2)
    beta = model.in_proj_b(hidden).sigmoid()
    g = -model.A_log.float().exp() * F.softplus(model.in_proj_a(hidden).float() + model.dt_bias)
    gate = model.in_proj_z(hidden).reshape(1, args.length, 48, 128)

    def qkv(conv):
        q, k, v = torch.split(conv, [2048, 2048, 6144], -1)
        q = q.reshape(1, args.length, 16, 128).repeat_interleave(3, 2)
        k = k.reshape(1, args.length, 16, 128).repeat_interleave(3, 2)
        return q, k, v.reshape(1, args.length, 48, 128)

    def finish(core):
        return model.out_proj(model.norm(core.reshape(-1, 128), gate.reshape(-1, 128)).reshape(1, args.length, 6144))

    q, k, v = qkv(hf_conv)
    chunk, state = torch_chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
    recurrent, rstate = torch_recurrent_gated_delta_rule(
        q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    expected = finish(chunk)
    literal_cache = DynamicCache(config=config)
    literal = torch.cat(
        [model(hidden[:, :-1], cache_params=literal_cache), model(hidden[:, -1:], cache_params=literal_cache)], 1
    )
    report = {
        "layer": args.layer,
        "length": args.length,
        "input_scale": 0.2,
        "cpu_only": True,
        "literal_vs_manual_chunk": metrics(literal, expected),
        "chunk_vs_recurrent_output": metrics(expected, finish(recurrent)),
        "chunk_vs_recurrent_state": metrics(state, rstate),
        "conv_hf_vs_bf16_expression": metrics(hf_conv, bf16_conv),
        "decay_exp_quantiles": torch.quantile(g.exp().flatten(), torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0])).tolist(),
    }
    bq, bk, bv = qkv(bf16_conv)
    conv_core, _ = torch_recurrent_gated_delta_rule(
        bq, bk, bv, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    report["bf16_conv_expression_output"] = metrics(expected, finish(conv_core))
    # Isolate possible loss of exp(g) mantissa in the FPU broadcast multiply.
    normalized_q, normalized_k = l2norm(q.float()), l2norm(k.float())
    for mode in ("decay_only", "state_and_decay"):
        st = torch.zeros_like(state)
        outputs = []
        for index in range(args.length):
            decay = truncate_tf32(g[:, index].exp())
            old = truncate_tf32(st) if mode == "state_and_decay" else st
            st = old * decay[..., None, None]
            kt = normalized_k[:, index]
            delta = (v[:, index].float() - (st * kt[..., None]).sum(-2)) * beta[:, index, :, None].float()
            st = st + kt[..., None] * delta[..., None, :]
            outputs.append((st * (normalized_q[:, index] * (128**-0.5))[..., None]).sum(-2))
        simulated = torch.stack(outputs, 1).bfloat16()
        report[f"hypothetical_tf32_{mode}"] = metrics(expected, finish(simulated))
    if args.core_fixture:
        torch.save(
            {
                "q": normalized_q[:, :, ::3].contiguous(),
                "k": normalized_k[:, :, ::3].contiguous(),
                "v": v.float(),
                "g": g,
                "beta": beta.float(),
                "output": recurrent.float(),
                "state": rstate,
                "hidden": hidden,
                "hf_full_output": literal,
                "report": report,
            },
            args.core_fixture,
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--length", type=int, default=1025)
    parser.add_argument("--layer", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core-fixture", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
