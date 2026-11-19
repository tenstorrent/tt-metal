# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-only recurrence isolation with CPU-generated real-model operands.

Run probe_gdn_numerics.py first to produce --fixture. Upload is explicit setup.
Forward contains only TTNN, and bypasses TT projections, convolution, q/k norm,
softplus and sigmoid. Failure here localizes error to native recurrence/relayout
or its internal exp(g), not those upstream stages. Compare one long launch to
chunked launches with an identical carried FP32 state. No device is imported
unless this command is actually run.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_gdn_numerics import metrics


@torch.inference_mode()
def prefix_fixture(fixture, length):
    """Recompute the prefix final-state oracle on CPU before any TTNN import."""
    available = fixture["q"].shape[1]
    if length is None or length == available:
        return fixture
    if not 1 <= length <= available:
        raise ValueError(f"--length must be in [1,{available}]")
    from transformers.models.qwen4_exp.modeling_qwen4_exp import torch_recurrent_gated_delta_rule

    result = {name: fixture[name][:, :length].contiguous() for name in ("q", "k", "v", "g", "beta")}
    result["output"], result["state"] = torch_recurrent_gated_delta_rule(
        result["q"].repeat_interleave(3, 2),
        result["k"].repeat_interleave(3, 2),
        result["v"],
        result["g"],
        result["beta"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks", default="1025,32")
    parser.add_argument("--backend", choices=("native", "model", "model-sfpu", "model-fpu"), default="native")
    parser.add_argument("--length", type=int, help="Isolate an actual prefix and recompute its CPU final-state oracle")
    args = parser.parse_args()
    torch.set_num_threads(8)
    fixture = prefix_fixture(torch.load(args.fixture, weights_only=True), args.length)
    import ttnn

    if args.backend in ("model", "model-sfpu", "model-fpu"):
        from models.autoports.qwen_qwen3_8_flash_next.tt.gdn_recurrence import fp32_recurrent_gated_delta_rule

        # "model" follows the functional default; explicit suffixes preserve
        # reproducibility of both numerical implementations in diagnostics.
        recurrence = (
            fp32_recurrent_gated_delta_rule
            if args.backend == "model"
            else partial(fp32_recurrent_gated_delta_rule, precise_dot_products=args.backend == "model-sfpu")
        )
    else:
        recurrence = ttnn.transformer.fused_recurrent_gated_delta_rule

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    report = {
        "scope": "recurrence only; real HF normalized q/k, value, log-decay and beta",
        "runs": [],
        "backend": args.backend,
        "recurrence_arithmetic": "SFPU FP32" if args.backend in ("model", "model-sfpu") else "FPU dot products",
        "length": fixture["q"].shape[1],
    }
    try:
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        tensors = {
            name: ttnn.from_torch(fixture[name].contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
            for name in ("q", "k", "v", "g", "beta")
        }
        decay = ttnn.exp(tensors["g"])
        actual_decay = ttnn.to_torch(decay)
        expected_decay = fixture["g"].exp()
        report["exp_max_abs_error"] = (actual_decay - expected_decay).abs().max().item()
        report["exp_max_relative_error"] = (
            ((actual_decay - expected_decay).abs() / expected_decay.clamp_min(torch.finfo(torch.float32).tiny))
            .max()
            .item()
        )
        length = fixture["q"].shape[1]
        for chunk in map(int, args.chunks.split(",")):
            state = ttnn.zeros((1, 48, 128, 128), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
            results = []
            for start in range(0, length, chunk):
                end = min(start + chunk, length)
                output, new_state = recurrence(
                    tensors["q"][:, start:end],
                    tensors["k"][:, start:end],
                    tensors["v"][:, start:end],
                    tensors["g"][:, start:end],
                    tensors["beta"][:, start:end],
                    initial_state=state,
                    output_final_state=True,
                    compute_kernel_config=compute,
                )
                ttnn.copy(new_state, state)
                results.append(output)
            output = ttnn.concat(results, dim=1) if len(results) > 1 else results[0]
            ttnn.synchronize_device(mesh)
            report["runs"].append(
                {
                    "chunk": chunk,
                    "core": metrics(fixture["output"], ttnn.to_torch(output)),
                    "state": metrics(fixture["state"], ttnn.to_torch(state)),
                }
            )
    except Exception as exc:
        report["error"] = str(exc).split("backtrace:")[0]
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        ttnn.close_mesh_device(mesh)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
