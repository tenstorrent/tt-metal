# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise TTNN DiT primitives against real, same-input CUDA captures."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch


def _pcc(reference: torch.Tensor, candidate: torch.Tensor) -> float | None:
    ref = reference.double().flatten()
    got = candidate.double().flatten()
    ref = ref - ref.mean()
    got = got - got.mean()
    norm = ref.norm() * got.norm()
    return float((ref @ got) / norm) if norm > 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--device-bdf", required=True, help="PCI BDF of the reserved physical card")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)

    prefix = Path("step_000/transformer")
    block_name = "transformer_blocks.0"

    def relative_path(name: str) -> Path:
        if name.startswith("input/"):
            return prefix / block_name / f"{name}.pt"
        if name in {"attn/input", "img_mlp/input"}:
            module_name = name.split("/", 1)[0]
            return prefix / f"{block_name}.{module_name}" / "input.pt"
        return prefix / f"{block_name}.{name}.pt"

    state = torch.load(args.cuda_dir / "weights/transformer_block_00.pt", map_location="cpu", weights_only=True)

    def reference(name: str) -> torch.Tensor:
        return torch.load(args.cuda_dir / relative_path(name), map_location="cpu", weights_only=True)

    # UMD renumbers the BDF-filtered device to zero. The filter must precede
    # importing TTNN, which initializes native runtime state during import.
    os.environ["TT_VISIBLE_DEVICES"] = args.device_bdf
    os.environ.pop("TT_METAL_VISIBLE_DEVICES", None)
    import ttnn

    from models.experimental.qwen_image_2_1.tt.tt_dit_components import (
        layer_norm,
        linear,
        multiply,
        rms_norm,
        silu,
        to_device,
        to_host,
    )

    rows = []
    started = time.monotonic()
    device = None
    try:
        device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), physical_device_ids=[0])
        compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        hidden = to_device(reference("input/hidden_states"), device)
        norm = layer_norm(hidden)

        def record(name: str, result: ttnn.Tensor) -> None:
            expected = reference(name)
            actual = to_host(result, tuple(expected.shape))
            output = args.output_dir / relative_path(name)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(actual, output)
            error = (actual.float() - expected.float()).abs()
            row = {
                "component": name,
                "shape": list(expected.shape),
                "pcc": _pcc(expected, actual),
                "max_abs_error": float(error.max()),
                "mean_abs_error": float(error.mean()),
                "fraction_within_atol_0_01": float((error <= 0.01).float().mean()),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

        record("img_norm1", norm)

        attention_reference = reference("attn/input")
        attention_input = to_device(attention_reference, device)
        roundtrip = to_host(attention_input, tuple(attention_reference.shape))
        print(
            json.dumps(
                {
                    "diagnostic": "attention_input_roundtrip",
                    "equal": bool(torch.equal(roundtrip, attention_reference)),
                    "max_abs_error": float((roundtrip.float() - attention_reference.float()).abs().max()),
                }
            ),
            flush=True,
        )
        q_weight = state["attn.to_q.weight"].T.contiguous()
        q_weight_roundtrip = to_host(to_device(q_weight, device), tuple(q_weight.shape))
        print(
            json.dumps(
                {
                    "diagnostic": "q_weight_roundtrip",
                    "equal": bool(torch.equal(q_weight_roundtrip, q_weight)),
                    "max_abs_error": float((q_weight_roundtrip.float() - q_weight.float()).abs().max()),
                }
            ),
            flush=True,
        )
        for projection in ("to_q", "to_k", "to_v"):
            result = linear(attention_input, state[f"attn.{projection}.weight"], device, compute)
            record(f"attn.{projection}", result)

        for projection in ("q", "k"):
            input_name = f"attn.to_{projection}"
            projection_input = reference(input_name).reshape(1, -1, 32, 128)
            normalized = rms_norm(to_device(projection_input, device), state[f"attn.norm_{projection}.weight"], device)
            record(f"attn.norm_{projection}", normalized)

        mlp_input = to_device(reference("img_mlp/input"), device)
        gate = linear(mlp_input, state["img_mlp.gate_layer.weight"], device, compute)
        projection = linear(mlp_input, state["img_mlp.proj.weight"], device, compute)
        record("img_mlp.gate_layer", gate)
        record("img_mlp.proj", projection)
        activated = multiply(silu(gate), projection)
        mlp_output = linear(activated, state["img_mlp.out.weight"], device, compute)
        record("img_mlp.out", mlp_output)
    finally:
        if device is not None:
            ttnn.close_mesh_device(device)

    report = {
        "device_id": args.device_id,
        "device_bdf": args.device_bdf,
        "elapsed_seconds": time.monotonic() - started,
        "components": rows,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
