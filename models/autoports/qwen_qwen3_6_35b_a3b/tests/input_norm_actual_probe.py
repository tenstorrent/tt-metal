# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic actual FusedDecoder.norm on real single-token stream rows.

Uses a minimal namespace and real gamma; no full decoder/experts are allocated.
--oracle-only validates CPU inputs/references without importing TTNN. Device mode
calls the actual unbound norm method, including its allocations and layout path.
"""
import argparse
import hashlib
import json
import os
import runpy
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

ROOT = Path(__file__).parents[1]
DOC = ROOT / "doc/fused_decoder"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"
POSITIONS = [0, 127, 128, 129, 130, 430, 3979, 5990, 5991, 5992, 7551, 7668, 7703, 8157, 8319, 8320, 8352]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path, positions, all_stream):
    oracle = torch.load(path, weights_only=False)
    params = oracle["key"]["parameters"]
    assert params["layer"] == 0, "This probe uses layer0 input norm gamma"
    for key, length in (
        ("prefix", "prompt_length"),
        ("tokens", "decode_length"),
        ("continuation", "continuation_length"),
    ):
        assert oracle[key].shape == (1, params[length], 2048), (key, oracle[key].shape)
    stream = torch.cat([oracle[k] for k in ("prefix", "tokens", "continuation")], dim=1)
    chosen = list(range(stream.shape[1])) if all_stream else sorted(set(positions))
    assert chosen and min(chosen) >= 0 and max(chosen) < stream.shape[1], "Position outside oracle stream"
    return params, stream, chosen


def main(args):
    torch.set_num_threads(4)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record(args.output)
    source = Path(args.snapshot) if args.snapshot else ROOT / "tt/fused_decoder.py"
    source_hash = digest(source)
    snapshot = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / (
        "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/" + REVISION
    )
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    key = "model.language_model.layers.0.input_layernorm.weight"
    with safe_open(snapshot / index[key], framework="pt", device="cpu") as f:
        gamma = (1 + f.get_tensor(key).float()).reshape(1, 1, 2048)
    eps = json.loads((snapshot / "config.json").read_text())["text_config"]["rms_norm_eps"]
    paths = (
        [Path(p) for p in args.oracle_cache]
        if args.oracle_cache
        else [
            DOC / "oracle_cache" / name
            for name in ("stress_8192_0.pt", "stress_8192_0_seed42.pt", "stress_8192_0_seed154.pt")
        ]
    )
    result = dict(
        diagnostic_only=True,
        acceptance_eligible=False,
        source=str(source),
        source_sha256=source_hash,
        checkpoint_revision=REVISION,
        single_token_calls=True,
        all_stream=args.all_stream,
        execution="CPU only" if args.oracle_only else "serialized physical chip0",
        seeds=[],
    )
    mesh = None
    start = time.monotonic()
    try:
        if not args.oracle_only:
            import ttnn

            cls = runpy.run_path(str(source))["FusedDecoder"]
            mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
            compute = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            round_op = ttnn.UnaryWithParam(
                ttnn.UnaryOpType.TYPECAST, float(ttnn.float32.value), float(ttnn.bfloat16.value)
            )
            widen_op = ttnn.UnaryWithParam(
                ttnn.UnaryOpType.TYPECAST, float(ttnn.bfloat16.value), float(ttnn.float32.value)
            )
            obj = SimpleNamespace(
                kind="linear_attention",
                config=SimpleNamespace(rms_norm_eps=eps),
                mesh_device=mesh,
                compute=compute,
                round_widen_chain=[round_op, widen_op],
                w={
                    "input_layernorm.weight": ttnn.from_torch(
                        gamma,
                        device=mesh,
                        dtype=ttnn.float32,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                },
            )
        for path in paths:
            params, stream, positions = load_rows(path, args.positions, args.all_stream)
            summary = dict(
                parameters=params,
                oracle=str(path),
                oracle_sha256=digest(path),
                rows=0,
                norm_code_differences=0,
                affected_rows=0,
                min_pcc=1.0,
                mean_captured_rows=0,
                mean_code_differences=0,
                details=[],
            )
            for pos in positions:
                row = stream[:, pos : pos + 1].contiguous()
                wide = row.float()
                mean = wide.square().mean(-1, keepdim=True)
                expected = (wide * torch.rsqrt(mean + eps) * gamma).bfloat16()
                assert torch.isfinite(expected).all()
                summary["rows"] += 1
                if args.oracle_only:
                    continue
                x = ttnn.from_torch(
                    row,
                    device=mesh,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                original_multiply = ttnn.multiply
                observed = []

                def multiply(*a, **kw):
                    out = original_multiply(*a, **kw)
                    if len(a) >= 2 and isinstance(a[1], (int, float)) and a[1] == 1 / 2048:
                        observed.append(out)
                    return out

                try:
                    ttnn.multiply = multiply
                    actual_device = cls.norm(obj, x, "input_layernorm.weight")
                finally:
                    ttnn.multiply = original_multiply
                actual = ttnn.to_torch(actual_device).reshape(expected.shape).bfloat16()
                differences = int((expected.view(torch.int16) != actual.contiguous().view(torch.int16)).sum())
                a, b = expected.double().flatten(), actual.double().flatten()
                a, b = a - a.mean(), b - b.mean()
                pcc = float(a @ b / (a.norm() * b.norm()))
                summary["norm_code_differences"] += differences
                summary["affected_rows"] += int(differences != 0)
                summary["min_pcc"] = min(summary["min_pcc"], pcc)
                detail = dict(position=pos, norm_code_differences=differences, pcc=pcc)
                if observed:
                    assert len(observed) == 1, "Ambiguous norm mean-scale capture"
                    got_mean = ttnn.to_torch(observed[0]).float().reshape(mean.shape)
                    different = int(
                        (got_mean.contiguous().view(torch.int32) != mean.contiguous().view(torch.int32)).sum()
                    )
                    summary["mean_captured_rows"] += 1
                    summary["mean_code_differences"] += different
                    detail.update(
                        mean_code_differences=different,
                        mean_reference=float(mean.item()),
                        mean_actual=float(got_mean.item()),
                    )
                if not args.all_stream or differences or detail.get("mean_code_differences", 0):
                    if len(summary["details"]) < 64:
                        summary["details"].append(detail)
                if summary["rows"] % 512 == 0:
                    print(
                        "ACTUAL_NORM_PROGRESS",
                        params["seed"],
                        summary["rows"],
                        summary["norm_code_differences"],
                        flush=True,
                    )
            if args.oracle_only:
                for metric in ("norm_code_differences", "affected_rows", "min_pcc", "mean_code_differences"):
                    summary[metric] = None
            summary["device_comparison_performed"] = not args.oracle_only
            result["seeds"].append(summary)
            result["diagnostic_wall_seconds"] = time.monotonic() - start
            output.write_text(json.dumps(result, indent=2) + "\n")
            print("ACTUAL_NORM_SEED", json.dumps({k: v for k, v in summary.items() if k != "details"}), flush=True)
        assert digest(source) == source_hash, "Selected source changed while probing"
        result["source_unchanged"] = True
        result["all_requested_rows_completed"] = True
        output.write_text(json.dumps(result, indent=2) + "\n")
        if args.oracle_only:
            import sys

            assert not any(k == "ttnn" or k.startswith("ttnn.") for k in sys.modules)
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--oracle-cache", action="append", help="Repeat for multiple cached streams; default seeds913,42,154"
    )
    p.add_argument("--positions", type=int, nargs="+", default=POSITIONS)
    p.add_argument(
        "--all-stream", action="store_true", help="Every prefix/decode/continuation row, still one-token calls"
    )
    p.add_argument("--snapshot", help="Explicit F source or .snapshot; actual method is loaded unchanged")
    p.add_argument("--oracle-only", action="store_true", help="CPU input/reference validation, no TTNN import/device")
    p.add_argument("--output", required=True)
    main(p.parse_args())
