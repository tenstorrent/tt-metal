# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bitwise old/fused cut and complete-adapter checks on actual norm/recurrence operands.

No gather, timing, or decoder acceptance claim. --oracle-only validates fixtures
without importing TTNN. Device execution belongs to the serialized hardware owner.
"""

import argparse
import hashlib
import json
import os
import runpy
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/fused_decoder"
BASE_SHA = "61ac211a397a4ab24737581b931ea318a4ed83d2e32d8eff67a4f26acda0a178"
NAMES = ("recurrent_q_scale", "l2_sum_boundary", "norm_square_input", "norm_mean_scale")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(old, new):
    a, b = old.float().contiguous(), new.float().contiguous()
    assert a.shape == b.shape, (a.shape, b.shape)
    different = a.view(torch.int32) != b.view(torch.int32)
    first = different.flatten().nonzero().flatten()[:8]
    return dict(
        elements=a.numel(),
        fp32_code_differences=int(different.sum()),
        signed_zero_differences=int((different & (a == 0) & (b == 0)).sum()),
        passed=not bool(different.any()),
        first_differences=[
            dict(
                index=int(i),
                old_bits=int(a.view(torch.int32).flatten()[i]),
                new_bits=int(b.view(torch.int32).flatten()[i]),
            )
            for i in first
        ],
    )


def fixtures(args):
    checkpoint = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / (
        "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/345d467b117d1fd5eda9b879a5988de9ae5f47a7"
    )
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for suffix in ("input_layernorm.weight", "linear_attn.conv1d.weight"):
        key = "model.language_model.layers.0." + suffix
        with safe_open(checkpoint / index[key], framework="pt", device="cpu") as f:
            weights[suffix] = f.get_tensor(key)
    eps = json.loads((checkpoint / "config.json").read_text())["text_config"]["rms_norm_eps"]
    oracle = torch.load(args.oracle, map_location="cpu", weights_only=False)
    stream = torch.cat([oracle[k] for k in ("prefix", "tokens", "continuation")], dim=1)
    rows = [(f"position_{p}", stream[:, p : p + 1].contiguous()) for p in args.positions]
    rows.append(("prefill_first32", stream[:, :32].contiguous()))
    captures = []
    for pos in args.capture_positions:
        path = Path(f"{args.capture_base}.pos{pos}.pt")
        values = torch.load(path, map_location="cpu", weights_only=False)["tt_values"]
        history = torch.cat((values["conv_input_state"], values["projection_packed"][:, :, :8192]), dim=1).float()
        taps = weights["linear_attn.conv1d.weight"][:, 0, :].float()
        accum = history[:, 0:1] * taps[:, 0]
        for j in range(1, 4):
            accum = accum + history[:, j : j + 1] * taps[:, j]
        conv = torch.nn.functional.silu(accum.bfloat16().float()).bfloat16()
        qk = torch.cat((conv[:, :, :2048], conv[:, :, 2048:4096]), dim=0).reshape(2, 1, 16, 128)
        captures.append((pos, path, values, qk))
    assert all(torch.isfinite(x).all() for _, x in rows)
    assert all(torch.isfinite(qk).all() for _, _, _, qk in captures)
    return rows, captures, (1 + weights["input_layernorm.weight"].float()).reshape(1, 1, 2048), eps


def run(args):
    torch.set_num_threads(4)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record(output)
    base = DOC / "selected_fused_decoder.py.snapshot"
    assert digest(base) == BASE_SHA
    sources = {name: DOC / "endurance" / f"exact_next_{name}.py.snapshot" for name in NAMES}
    result = dict(
        status="running",
        diagnostic_only=True,
        acceptance_eligible=False,
        execution="CPU only" if args.oracle_only else "serialized physical chip0",
        probe_sha256=digest(__file__),
        base_sha256=BASE_SHA,
        candidates={n: dict(path=str(p), sha256=digest(p)) for n, p in sources.items()},
        oracle=str(args.oracle),
        oracle_sha256=digest(args.oracle),
        checks=[],
    )

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    mesh = None
    stage = "fixtures"
    save()
    try:
        rows, captures, gamma, eps = fixtures(args)
        result["fixtures"] = dict(
            norm_rows=[dict(name=n, shape=list(x.shape)) for n, x in rows],
            captures=[dict(position=p, path=str(path), sha256=digest(path)) for p, path, _, _ in captures],
            epsilon=eps,
            qk_origin="CPU four-tap reconstruction from captured TT history/projection and checkpoint weights",
        )
        if args.oracle_only:
            result["status"] = "cpu_fixtures_passed"
            return True
        import ttnn

        base_cls = runpy.run_path(str(base))["FusedDecoder"]
        classes = {n: runpy.run_path(str(p))["FusedDecoder"] for n, p in sources.items()}
        stage = "open_device"
        params = dict(physical_device_ids=[0], trace_region_size=0, l1_small_size=32768)
        result["device_open_parameters"] = params
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), **params)
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        U = ttnn.UnaryOpType
        rw = [
            ttnn.UnaryWithParam(U.TYPECAST, float(ttnn.float32.value), float(ttnn.bfloat16.value)),
            ttnn.UnaryWithParam(U.TYPECAST, float(ttnn.bfloat16.value), float(ttnn.float32.value)),
        ]
        scale = [ttnn.UnaryWithParam(U.MUL_UNARY_SFPU, 128**-0.5)]
        mean_scale = [ttnn.UnaryWithParam(U.MUL_UNARY_SFPU, 1 / 2048)]
        rsqrt = [ttnn.UnaryWithParam(U.RSQRT, 0.0)] + rw

        def upload(x, dtype=ttnn.float32):
            return ttnn.from_torch(
                x.contiguous(), device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        def check(label, old, new):
            stats = compare(ttnn.to_torch(old), ttnn.to_torch(new))
            result["checks"].append(dict(name=label, **stats))
            save()
            print("EXACT_FUSION", label, stats["fp32_code_differences"], flush=True)

        obj = SimpleNamespace(
            kind="linear_attention",
            config=SimpleNamespace(rms_norm_eps=eps),
            mesh_device=mesh,
            compute=compute,
            round_widen_chain=rw,
            w={"input_layernorm.weight": upload(gamma)},
        )
        obj._recurrent_sum = lambda p: base_cls._recurrent_sum(obj, p)
        for pos, _, values, qk in captures:
            stage = f"recurrence_{pos}"
            new_state = upload(values["new_state"].float())
            q32 = upload(values["delta_q"].float().repeat_interleave(2, dim=2))
            old_q = ttnn.permute(ttnn.multiply(q32, 128**-0.5), (0, 2, 3, 1))
            raw_q = ttnn.permute(q32, (0, 2, 3, 1))
            old = ttnn.multiply(new_state, old_q)
            new = ttnn.multiply(new_state, raw_q, input_tensor_b_activations=scale)
            check(f"q_scale_primitive_{pos}", old, new)
            del old, new, old_q, raw_q, q32, new_state
            inputs = [
                upload(values[key].float())
                for key in ("delta_q", "delta_k", "delta_v", "delta_g", "delta_beta", "delta_state_input")
            ]
            old = base_cls._recurrent_step(obj, *inputs, round_state=True)
            new = classes["recurrent_q_scale"]._recurrent_step(obj, *inputs, round_state=True)
            check(f"q_scale_complete_core_{pos}", old[0], new[0])
            check(f"q_scale_complete_state_{pos}", old[1], new[1])
            del old, new, inputs

            stage = f"l2_{pos}"
            wide = upload(qk.float())
            squares = ttnn.multiply(wide, wide, activations=rw)
            sums = ttnn.sum(squares, dim=-1, keepdim=True, compute_kernel_config=compute)
            old = ttnn.add(ttnn.unary_chain(sums, rw), 1e-6, activations=rw)
            new = ttnn.add(sums, 1e-6, input_tensor_a_activations=rw, activations=rw)
            check(f"l2_sum_boundary_primitive_{pos}", old, new)
            old_output = ttnn.multiply(wide, ttnn.unary_chain(old, rsqrt), activations=rw)
            new_output = ttnn.multiply(wide, ttnn.unary_chain(new, rsqrt), activations=rw)
            check(f"l2_sum_boundary_complete_{pos}", old_output, new_output)
            del wide, squares, sums, old, new, old_output, new_output

        # Capture real intermediate tensors from the frozen norm; compare before downstream rounding.
        for label, row in rows:
            stage = "norm_" + label
            x = upload(row, ttnn.bfloat16)
            original_mul, original_sub, original_add = ttnn.multiply, ttnn.subtract, ttnn.add
            squares, totals, cuts = {}, [], []

            def multiply(a, b, *aargs, **kwargs):
                out = original_mul(a, b, *aargs, **kwargs)
                if a is b:
                    squares[id(out)] = (out, a)
                if isinstance(b, (int, float)) and b == 1 / 2048:
                    totals.append(a)
                return out

            def subtract(a, b, *aargs, **kwargs):
                out = original_sub(a, b, *aargs, **kwargs)
                if id(b) in squares:
                    fused = original_sub(a, squares[id(b)][1], input_tensor_b_activations=[U.SQUARE])
                    cuts.append(("square_input", out, fused))
                return out

            def add(a, b, *aargs, **kwargs):
                out = original_add(a, b, *aargs, **kwargs)
                if isinstance(b, (int, float)) and b == eps and totals:
                    fused = ttnn.reshape(
                        original_add(totals[-1], eps, input_tensor_a_activations=mean_scale), tuple(out.shape)
                    )
                    cuts.append(("mean_scale", out, fused))
                return out

            try:
                ttnn.multiply, ttnn.subtract, ttnn.add = multiply, subtract, add
                old_norm = base_cls.norm(obj, x, "input_layernorm.weight")
            finally:
                ttnn.multiply, ttnn.subtract, ttnn.add = original_mul, original_sub, original_add
            assert [n for n, _, _ in cuts].count("square_input") == 2
            assert [n for n, _, _ in cuts].count("mean_scale") == 1
            for i, (name, old, new) in enumerate(cuts):
                check(f"norm_{name}_primitive_{label}_{i}", old, new)
            for name in ("norm_square_input", "norm_mean_scale"):
                new_norm = classes[name].norm(obj, x, "input_layernorm.weight")
                check(f"{name}_complete_{label}", old_norm, new_norm)
                del new_norm
            del old_norm, x, cuts, squares, totals
        result["status"] = "passed" if all(c["passed"] for c in result["checks"]) else "numerical_failure"
        return result["status"] == "passed"
    except Exception:
        result.update(status="failed", failure_stage=stage, error=traceback.format_exc())
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.close_mesh_device(mesh)
                result["device_closed"] = True
        finally:
            save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--oracle", default=str(DOC / "oracle_cache/stress_8192_0.pt"))
    parser.add_argument("--capture-base", default=str(DOC / "endurance_precise_probe.json"))
    parser.add_argument("--positions", nargs="+", type=int, default=[128, 130, 3979, 5991, 7703])
    parser.add_argument("--capture-positions", nargs="+", type=int, default=[128, 130, 7703])
    parser.add_argument("--oracle-only", action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if run(args) else 1)
