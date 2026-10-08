# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated CPU ttsim paged-decode diagnostic; no model qualification."""

import argparse
import gc
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, report):
    report["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def tensor_sha(tensor):
    import torch

    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def errors(actual, expected):
    import torch

    if actual.shape != expected.shape:
        raise ValueError(f"Output shape mismatch: {actual.shape} versus {expected.shape}")
    actual, expected = actual.double(), expected.double()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite:
        return dict(finite=False)
    delta = actual - expected

    def pcc(a, b):
        a, b = a.flatten(), b.flatten()
        a, b = a - a.mean(), b - b.mean()
        return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))

    per_head = []
    per_user = []
    for user in range(actual.shape[1]):
        d, e = delta[0, user], expected[0, user]
        per_user.append(
            dict(user=user, relative_rms=float(d.norm() / e.norm().clamp_min(1e-30)), pcc=pcc(actual[0, user], e))
        )
        for head in range(actual.shape[2]):
            per_head.append(
                dict(
                    user=user,
                    head=head,
                    relative_rms=float(d[head].norm() / e[head].norm().clamp_min(1e-30)),
                    pcc=pcc(actual[0, user, head], e[head]),
                    max_abs=float(d[head].abs().max()),
                )
            )
    return dict(
        finite=True,
        relative_rms=float(delta.norm() / expected.norm().clamp_min(1e-30)),
        pcc=pcc(actual, expected),
        max_abs=float(delta.abs().max()),
        mean_error=float(delta.mean()),
        per_user=per_user,
        per_query_head=per_head,
        worst_query_head_relative_rms=max(per_head, key=lambda x: x["relative_rms"]),
        worst_query_head_max_abs=max(per_head, key=lambda x: x["max_abs"]),
    )


def kernel_gate(metrics):
    # Unchanged existing attention_tuning.accuracy per-user criteria.
    return metrics["finite"] and all(p["relative_rms"] <= 0.02 and p["pcc"] >= 0.999 for p in metrics["per_user"])


def reference(q, key, value, table, active_tokens):
    pages = table[0, : (active_tokens + 31) // 32].long()
    keys = key[pages, 0].reshape(-1, 256)[:active_tokens].float()
    scores = q[0, 0].float() @ keys.T * (256**-0.5)
    del keys
    probabilities = scores.softmax(dim=-1)
    del scores
    values = value[pages, 0].reshape(-1, 256)[:active_tokens].float()
    return (probabilities @ values).reshape(1, 1, 6, 256)


def run(args):
    if args.output.exists():
        raise FileExistsError("Preserve attempts: receipt already exists")
    sim = Path(os.environ["TT_METAL_SIMULATOR"]).resolve(strict=True)
    if sha(sim) != args.simulator_sha256:
        raise ValueError("Pinned simulator SHA256 mismatch")
    if os.environ.get("TT_METAL_SLOW_DISPATCH_MODE") != "1":
        raise ValueError("Simulator slow dispatch is required")
    if os.environ.get("TT_METAL_INSPECTOR_RPC") != "0":
        raise ValueError("Inspector RPC must remain disabled for isolated CPU execution")
    allowed_compile_fallback = os.environ.get("TT_METAL_DISABLE_SFPLOADMACRO") == "1" and args.allow_plain_sfpu_fallback
    unsafe = {
        k: v
        for k, v in os.environ.items()
        if ("TT" in k or "SIM" in k)
        and not (k == "TT_METAL_DISABLE_SFPLOADMACRO" and allowed_compile_fallback)
        and ("DISABLE" in k or "SKIP" in k)
        and v not in ("", "0", "false", "False")
    }
    if unsafe:
        raise ValueError(f"Refusing disabled checks: {unsafe}")
    native = Path(os.environ["TT_METAL_HOME"])
    revision = subprocess.check_output(["git", "-C", str(native), "rev-parse", "HEAD"], text=True).strip()
    if revision != args.native_revision:
        raise ValueError("Pinned native revision mismatch")
    sources = sorted((native / "ttnn/cpp/ttnn/operations/transformer/sdpa_decode").rglob("*.cpp"))
    sources.extend(
        native / name
        for name in (
            "tt_metal/jit_build/build.cpp",
            "tt_metal/llrt/rtoptions.cpp",
            "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h",
            "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp",
        )
    )
    report = dict(
        state="importing",
        model_accuracy_qualified=False,
        promoted_to_model=False,
        precision_changed_in_model=False,
        physical_devices_accessed=False,
        scope="Synthetic BF16 Q/K/V; CPU ttsim eager execution on one virtual Blackhole; no silicon timing claim",
        simulator=str(sim),
        simulator_sha256=sha(sim),
        native_revision=revision,
        probe_sha256=sha(__file__),
        source_sha256={str(p.relative_to(native)): sha(p) for p in sources},
        compiler_arithmetic_fallback=dict(
            enabled=allowed_compile_fallback,
            flag="TT_METAL_DISABLE_SFPLOADMACRO=1" if allowed_compile_fallback else None,
            meaning="Native LLK plain-instruction arithmetic implementation; no simulator error or instruction checks disabled",
            hardware_instruction_parity_qualified=False,
        ),
        native_library_sha256={
            str(p): sha(p)
            for p in (native.parent / "metal-install/lib").glob("*.so")
            if p.name in ("libtt_metal.so", "_ttnn.so", "_ttnncpp.so", "libtt-umd.so")
        },
        fixed_configuration=dict(
            batch=1,
            live_query_heads=6,
            device_query_heads=32,
            kv_heads=1,
            head_dim=256,
            page_size=32,
            chunk=256,
            max_cores_per_head_batch=16,
            math_fidelity="HiFi4",
            fp32_dest_acc_en=True,
            math_approx_mode=False,
            packer_l1_acc=True,
            exp_approx_mode=False,
            query_dtype="bfloat16",
        ),
        quantization_gate=None,
        kernel_gate=dict(
            reference="CPU FP32 using roundtripped device-quantized K/V",
            per_user_relative_rms_max=0.02,
            per_user_pcc_min=0.999,
        ),
        cases=[],
        cleanup_completed=False,
    )
    save(args.output, report)
    import torch

    import ttnn

    torch.set_num_threads(1)
    report["torch_version"] = torch.__version__
    report["ttnn_module"] = ttnn.__file__
    count = ttnn.GetNumAvailableDevices()
    report["visible_virtual_devices"] = count
    if count != 1:
        save(args.output, report)
        raise RuntimeError(f"Expected exactly one virtual chip, got {count}; refusing device open")
    mesh = None
    started = time.monotonic()
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
        grid = mesh.compute_with_storage_grid_size()
        report["worker_grid"] = [grid.x, grid.y]

        def upload(tensor, dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                tensor.contiguous(),
                device=mesh,
                dtype=dtype,
                layout=layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        def host(tensor):
            parts = ttnn.get_device_tensors(tensor)
            if len(parts) != 1:
                raise RuntimeError("Unexpected multi-chip tensor")
            return ttnn.to_torch(parts[0]).float()

        config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
        )
        program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=[grid.x, grid.y],
            q_chunk_size=32,
            k_chunk_size=256,
            exp_approx_mode=False,
            max_cores_per_head_batch=16,
        )

        for length in args.contexts:
            capacity = (length + 127 + 511) // 512 * 512
            seed = args.seed + length
            rng = torch.Generator().manual_seed(seed)
            pages = capacity // 32
            table = torch.randperm(pages, generator=rng, dtype=torch.int32).reshape(1, pages)
            query = torch.randn((1, 1, 6, 256), generator=rng).bfloat16()
            padded_query = torch.cat((query, torch.zeros((1, 1, 26, 256), dtype=query.dtype)), dim=2)
            original_key = torch.randn((pages, 1, 32, 256), generator=rng).bfloat16()
            original_value = torch.randn((pages, 1, 32, 256), generator=rng).bfloat16()
            q = upload(padded_query, ttnn.bfloat16)
            tt_table = upload(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            case = dict(
                context=length,
                aligned_capacity=capacity,
                seed=seed,
                query_sha256=tensor_sha(query),
                key_sha256=tensor_sha(original_key),
                original_value_sha256=tensor_sha(original_value),
                page_table_sha256=tensor_sha(table),
                causal_queries=[],
            )
            report["cases"].append(case)
            for active_tokens in (length, length - 37):
                value = original_value.clone()
                for virtual_page in range(active_tokens // 32, pages):
                    begin = max(0, active_tokens - virtual_page * 32)
                    value[int(table[0, virtual_page]), 0, begin:, :] = 32
                causal = dict(
                    active_tokens=active_tokens,
                    position=active_tokens - 1,
                    tail_sentinel=32,
                    value_with_sentinel_sha256=tensor_sha(value),
                    variants=[],
                )
                case["causal_queries"].append(causal)
                report.update(state="reference_original", active_context=length, active_tokens=active_tokens)
                save(args.output, report)
                original_reference = reference(query, original_key, value, table, active_tokens)
                causal["reference_original_sha256"] = tensor_sha(original_reference)
                pos = upload(torch.tensor([active_tokens - 1], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                baseline_actual, baseline_reference, baseline_total_error = None, None, None
                for label, k_dtype, v_dtype in (
                    ("K8_V8", "bfloat8_b", "bfloat8_b"),
                    ("K4_V4", "bfloat4_b", "bfloat4_b"),
                    ("K8_V4", "bfloat8_b", "bfloat4_b"),
                ):
                    report.update(state="uploading", active_variant=label)
                    save(args.output, report)
                    k = upload(original_key, getattr(ttnn, k_dtype))
                    v = upload(value, getattr(ttnn, v_dtype))
                    report["state"] = "reference_quantized"
                    save(args.output, report)
                    key_quantized, value_quantized = host(k), host(v)
                    quantized_reference = reference(query, key_quantized, value_quantized, table, active_tokens)
                    variant = dict(
                        name=label,
                        key_dtype=k_dtype,
                        value_dtype=v_dtype,
                        roundtrip_key_sha256=tensor_sha(key_quantized),
                        roundtrip_value_sha256=tensor_sha(value_quantized),
                        reference_quantized_sha256=tensor_sha(quantized_reference),
                    )
                    del key_quantized, value_quantized
                    gc.collect()
                    variant["quantization_only"] = errors(quantized_reference, original_reference)
                    variant["state"] = "executing"
                    causal["variants"].append(variant)
                    report["state"] = "executing"
                    save(args.output, report)
                    output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                        q,
                        k,
                        v,
                        cur_pos_tensor=pos,
                        page_table_tensor=tt_table,
                        scale=256**-0.5,
                        program_config=program,
                        compute_kernel_config=config,
                    )
                    actual = host(output)[:, :, :6, :]
                    variant.update(
                        output_sha256=tensor_sha(actual),
                        quantization_only=errors(quantized_reference, original_reference),
                        execution_on_quantized=errors(actual, quantized_reference),
                        total=errors(actual, original_reference),
                    )
                    variant["kernel_gate_passed"] = kernel_gate(variant["execution_on_quantized"])
                    if label == "K8_V8":
                        baseline_actual, baseline_reference = actual.clone(), quantized_reference.clone()
                        baseline_total_error = variant["total"].get("relative_rms")
                    variant["actual_vs_bfp8_baseline"] = errors(actual, baseline_actual)
                    variant["quantized_reference_vs_bfp8_baseline"] = errors(quantized_reference, baseline_reference)
                    variant["total_rms_ratio_to_bfp8"] = (
                        variant["total"]["relative_rms"] / max(baseline_total_error, 1e-30)
                        if variant["total"]["finite"] and baseline_total_error is not None
                        else None
                    )
                    variant["state"] = "completed"
                    save(args.output, report)
                    print(
                        "BFP4_KV_SIM_CASE",
                        json.dumps(
                            dict(
                                context=length,
                                active_tokens=active_tokens,
                                variant=label,
                                quantization_rms=variant["quantization_only"].get("relative_rms"),
                                execution_rms=variant["execution_on_quantized"].get("relative_rms"),
                                total_rms=variant["total"].get("relative_rms"),
                                kernel_gate_passed=variant["kernel_gate_passed"],
                            )
                        ),
                        flush=True,
                    )
                    ttnn.deallocate(output)
                    ttnn.deallocate(k)
                    ttnn.deallocate(v)
                    del output, k, v, actual, quantized_reference
                    gc.collect()
                ttnn.deallocate(pos)
                del value, original_reference, baseline_actual, baseline_reference
            ttnn.deallocate(q)
            ttnn.deallocate(tt_table)
            del original_key, original_value
            gc.collect()
        report["state"] = "completed"
    except BaseException as error:
        report.update(
            state="failed",
            error=dict(type=type(error).__name__, message=str(error)[:4000], traceback=traceback.format_exc()[-8000:]),
        )
        raise
    finally:
        report["elapsed_wall_seconds_not_silicon_performance"] = time.monotonic() - started
        try:
            if mesh is not None:
                ttnn.close_mesh_device(mesh)
            report["cleanup_completed"] = True
        finally:
            save(args.output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contexts", required=True, nargs="+", type=int)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--simulator-sha256", required=True)
    parser.add_argument("--native-revision", required=True)
    parser.add_argument("--allow-plain-sfpu-fallback", action="store_true")
    run(parser.parse_args())
