# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU simulator projection accuracy: quantization versus execution error.

Samples aligned output columns from the actual checkpoint, preserving full K
(one TP4 input shard for down projection). Inputs are seeded BF16 stimuli, not
captured model activations. This is not a full-model or GPQA qualification.
"""

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path


def save(path, report):
    report["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def errors(actual, expected):
    import torch

    actual, expected = actual.double(), expected.double()
    delta = actual - expected
    a, e = actual.flatten(), expected.flatten()
    ac, ec = a - a.mean(), e - e.mean()
    return dict(
        finite=bool(torch.isfinite(actual).all()),
        relative_rms=float(delta.norm() / expected.norm().clamp_min(1e-30)),
        worst_row_relative_rms=float((delta.norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-30)).max()),
        pcc=float((ac @ ec) / (ac.norm() * ec.norm()).clamp_min(1e-30)),
        max_abs=float(delta.abs().max()),
        mean_error=float(delta.mean()),
    )


def run(args):
    simulator = Path(os.environ["TT_METAL_SIMULATOR"]).resolve()
    if hashlib.sha256(simulator.read_bytes()).hexdigest() != args.simulator_sha256:
        raise ValueError("Simulator binary does not match the pinned digest")
    if os.getenv("TT_METAL_SLOW_DISPATCH_MODE") != "1":
        raise ValueError("This bounded numerical probe requires simulator slow dispatch")
    if args.output.exists():
        raise FileExistsError("Use a new simulator receipt path")

    import torch
    from safetensors import safe_open

    import ttnn

    torch.set_num_threads(1)
    index_path = args.weights / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    names = (
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.3.self_attn.q_proj.weight",
        "model.language_model.layers.31.mlp.down_proj.weight",
        "model.language_model.layers.63.mlp.gate_proj.weight",
        "lm_head.weight",
    )
    if any(name not in index for name in names):
        raise ValueError("Checkpoint does not contain the expected pinned Qwen projection names")
    report = dict(
        state="opening",
        model_accuracy_qualified=False,
        simulator=str(simulator),
        simulator_sha256=args.simulator_sha256,
        native_revision=args.native_revision,
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoint_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        checkpoint_index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
        precision_changed_in_model=False,
        scope="Single virtual Blackhole; real weight submatrices with synthetic BF16 inputs; no full-model accuracy or timing claim",
        methodology="Full K except one quarter of down-projection K; two aligned output-column samples; 32 independent rows; LoFi/HiFi4 with FP32 destination accumulation and BF16 output",
        cases=[],
        cleanup_completed=False,
    )
    save(args.output, report)
    count = ttnn.GetNumAvailableDevices()
    if count != 1:
        raise RuntimeError(f"Expected exactly one virtual chip, got {count}; refusing to open devices")
    report["visible_devices"] = count
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))

    def upload(value, dtype):
        return ttnn.from_torch(
            value.contiguous(),
            device=mesh,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def host(tensor):
        parts = ttnn.get_device_tensors(tensor)
        if len(parts) != 1:
            raise RuntimeError("Probe unexpectedly accessed more than one virtual chip")
        return ttnn.to_torch(parts[0]).float()

    try:
        for name_index, name in enumerate(names):
            with safe_open(args.weights / index[name], framework="pt", device="cpu") as checkpoint:
                source = checkpoint.get_slice(name)
                n, full_k = source.get_shape()
                k = full_k // 4 if ".down_proj." in name else full_k
                if k % 32 or n < 256:
                    raise ValueError("Unexpected checkpoint projection geometry")
                starts = (0, (n // 2 // 32) * 32)
                for start in starts:
                    # Entire 16-value exponent-sharing groups remain intact.
                    weight = source[start : start + 128, :k].T.contiguous().bfloat16()
                    seed = 20261008 + name_index * 100 + start
                    rng = torch.Generator().manual_seed(seed)
                    inputs = torch.randn(32, k, generator=rng).bfloat16()
                    reference = inputs.float() @ weight.float()
                    a = upload(inputs, ttnn.bfloat16)
                    case = dict(
                        name=name,
                        source_shape=[n, full_k],
                        source_output_rows=[start, start + 128],
                        source_input_columns=[0, k],
                        stimulus_seed=seed,
                        m=32,
                        k=k,
                        n=128,
                        weight_bf16_sha256=hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest(),
                        input_bf16_sha256=hashlib.sha256(inputs.view(torch.uint8).numpy().tobytes()).hexdigest(),
                        variants=[],
                    )
                    report["cases"].append(case)
                    report.update(state="running", active_weight=name, active_output_row=start)
                    save(args.output, report)
                    for dtype_name in ("bfloat4_b", "bfloat8_b", "bfloat16"):
                        w = upload(weight, getattr(ttnn, dtype_name))
                        dequantized = host(w)
                        quantized_reference = inputs.float() @ dequantized
                        for fidelity in ("LoFi", "HiFi4"):
                            config = ttnn.WormholeComputeKernelConfig(
                                math_fidelity=getattr(ttnn.MathFidelity, fidelity),
                                fp32_dest_acc_en=True,
                                math_approx_mode=False,
                                packer_l1_acc=True,
                            )
                            out = ttnn.matmul(a, w, compute_kernel_config=config, dtype=ttnn.bfloat16)
                            actual = host(out)
                            variant = dict(
                                weight_dtype=dtype_name,
                                fidelity=fidelity,
                                weight_quantization=errors(dequantized, weight.float()),
                                output_quantization_only=errors(quantized_reference, reference),
                                execution_on_quantized_operands=errors(actual, quantized_reference),
                                total_against_checkpoint=errors(actual, reference),
                            )
                            if not all(variant[key]["finite"] for key in variant if isinstance(variant[key], dict)):
                                raise AssertionError("Nonfinite simulator projection result")
                            case["variants"].append(variant)
                            save(args.output, report)
                            print("BFP4_SIM_CASE", json.dumps(dict(name=name, start=start, **variant)), flush=True)
                            ttnn.deallocate(out)
                        ttnn.deallocate(w)
                    ttnn.deallocate(a)
                    gc.collect()
        report["state"] = "completed"
    except BaseException as error:
        report.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:4000]))
        raise
    finally:
        try:
            ttnn.close_mesh_device(mesh)
            report["cleanup_completed"] = True
        finally:
            save(args.output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--simulator-sha256", required=True)
    parser.add_argument("--native-revision", required=True)
    run(parser.parse_args())
