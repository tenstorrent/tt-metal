# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Same-operand ordered router projection, all 256 columns; diagnostic only."""

import argparse
import ast
import copy
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import torch
from safetensors import safe_open

from models.autoports.qwen_qwen3_6_35b_a3b.tests.v5_ordered_qz_probe import load_snapshot, metrics, source_methods

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "doc/fused_decoder/endurance"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scope_proof(base, candidate):
    bt, bm = source_methods(base)
    ct, cm = source_methods(candidate)
    assert {k for k in bm if ast.dump(bm[k]) != ast.dump(cm[k])} == {"from_state_dict", "linear", "_moe", "_forward"}
    setup = cm["from_state_dict"].body
    i = next(
        i for i, n in enumerate(setup) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "router_order"
    )
    assert [ast.unparse(n.targets[0]) for n in setup[i : i + 4]] == [
        "router_order",
        "self.scan_router_weight",
        "router_pair32",
        "self.router_pair32_indices",
    ]
    del setup[i : i + 4]
    method = cm["linear"]
    assert method.args.kwonlyargs[-1].arg == "ordered_router"
    method.args.kwonlyargs.pop()
    method.args.kw_defaults.pop()
    assert ast.unparse(method.body[0].test) == "ordered_router and x.shape[-2] == 1 and (key == 'mlp.gate.weight')"
    method.body.pop(0)
    method = cm["_moe"]
    assert method.args.kwonlyargs[-1].arg == "ordered_router"
    method.args.kwonlyargs.pop()
    method.args.kw_defaults.pop()
    call = method.body[1].value.args[0]
    assert call.keywords[-1].arg == "ordered_router"
    call.keywords.pop()
    call = cm["_forward"].body[-1].value.args[1]
    assert call.keywords[-1].arg == "ordered_router"
    assert ast.unparse(call.keywords[-1].value) == "decode and cache_boundary"
    call.keywords.pop()
    assert ast.dump(bt) == ast.dump(ct)
    return {
        "full_module_ast_restoration": True,
        "public_decode_only": True,
        "prefill_including_singleton_tails_unchanged": True,
        "projection_k": 512,
    }


def setup_ordered(obj, source, weight, upload, native, mesh):
    _, methods = source_methods(source)
    setup = methods["from_state_dict"].body
    i = next(
        i for i, n in enumerate(setup) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "router_order"
    )
    module = ast.fix_missing_locations(ast.Module(copy.deepcopy(setup[i : i + 4]), []))
    exec(
        compile(module, str(source) + ":ordered_setup", "exec"),
        dict(
            self=obj,
            torch=torch,
            ttnn=native,
            mesh_device=mesh,
            upload=upload,
            DRAM=native.DRAM_MEMORY_CONFIG,
            state_dict={"mlp.gate.weight": weight},
        ),
    )


def ordered_cpu(x, weight):
    order = torch.arange(2048).reshape(1024, 2).flip(1).flatten()
    products = x.reshape(2048)[order].float().reshape(4, 512, 1) * weight.T[order].float().reshape(4, 512, 256)
    sums = torch.zeros_like(products[:, 0])
    for i in range(512):
        sums = sums + products[:, i]
    total = sums[0].clone()
    for i in range(1, 4):
        total = total + sums[i]
    return total.reshape(1, 1, 256)


def run(args):
    torch.set_num_threads(8)
    assert not args.output.exists(), "Preserve previous evidence"
    scope = scope_proof(args.baseline, args.snapshot)
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    key = f"model.language_model.layers.{args.layer}.mlp.gate.weight"
    with safe_open(args.checkpoint / index[key], framework="pt", device="cpu") as f:
        weight = f.get_tensor(key)
    assert weight.dtype == torch.bfloat16 and tuple(weight.shape) == (256, 2048)
    cases = []
    payload = {}
    for position in args.positions:
        path = Path(str(args.capture_prefix) + f".pos{position}.pt")
        cap = torch.load(path, map_location="cpu", weights_only=False)
        assert cap["layer"] == args.layer and cap["position"] == position
        assert cap["measured_decoder_source_sha256"] == sha(args.baseline)
        x = cap["tt_values"]["post_norm"].reshape(1, 1, 2048).bfloat16()
        canonical = torch.nn.functional.linear(x, weight)
        ordered = ordered_cpu(x, weight)
        row = dict(
            position=position,
            capture=str(path),
            capture_sha256=sha(path),
            cpu_ordered_vs_hf_all_columns=metrics(canonical, ordered.bfloat16()),
            cpu_canonical_vs_captured_native=metrics(canonical, cap["tt_values"]["router"]),
            cpu_canonical_top8=canonical.float().softmax(-1).topk(8).indices.flatten().tolist(),
            outputs={},
        )
        assert row["cpu_ordered_vs_hf_all_columns"]["exact"], "K512 must reproduce CPU BF16 projection on all columns"
        cases.append(row)
        payload[position] = dict(x=x, canonical=canonical, ordered_fp32=ordered, outputs={})
    result = dict(
        status="cpu_proof_pass",
        classification="diagnostic_only",
        acceptance_eligible=False,
        baseline=str(args.baseline),
        baseline_sha256=sha(args.baseline),
        snapshot=str(args.snapshot),
        snapshot_sha256=sha(args.snapshot),
        probe_sha256=sha(__file__),
        scope_proof=scope,
        layer=args.layer,
        checkpoint_revision=REVISION,
        cases=cases,
        compared_columns=256,
        extra_weight_bytes=1048576,
        product_or_prefix_bytes=2097152,
        timing="not measured",
        device_closed=None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

    record(args.output)
    save()
    if args.oracle_only:
        assert "ttnn" not in sys.modules
        print("CPU_ORDERED_ROUTER_PASS", [r["position"] for r in cases])
        return
    for key in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"):
        assert os.environ.get(key) in (None, "", "0"), key
    import ttnn

    mesh = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768)
        result.update(status="running", device_closed=False)

        def upload(v, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                v.contiguous(), device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        for label, source in [("native", args.baseline), ("ordered", args.snapshot)]:
            obj = load_snapshot(source, "router_probe_" + label).FusedDecoder()
            obj.kind = "linear_attention"
            obj.mesh_device = mesh
            obj.decode_linear = {}
            obj.compute = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            obj.w = {"mlp.gate.weight": upload(weight.T, ttnn.float32)}
            if label == "ordered":
                setup_ordered(obj, source, weight, upload, ttnn, mesh)
            for row in cases:
                item = payload[row["position"]]
                dx = upload(item["x"])
                kw = {"ordered_router": True} if label == "ordered" else {}
                got = ttnn.to_torch(obj.linear(dx, "mlp.gate.weight", ttnn.bfloat16, **kw)).clone()
                assert tuple(got.shape) == (1, 1, 256) and torch.isfinite(got).all()
                check = metrics(item["canonical"], got)
                row["outputs"][label] = dict(
                    vs_cpu_all_columns=check, top8=got.float().softmax(-1).topk(8).indices.flatten().tolist()
                )
                item["outputs"][label] = got
                save()
                if label == "ordered":
                    assert check["exact"], "Ordered adapter must match all 256 CPU BF16 outputs"
        tensors = Path(str(args.output) + ".tensors.pt")
        torch.save(payload, tensors)
        result.update(status="completed_diagnostic", tensors=str(tensors), tensors_sha256=sha(tensors))
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            result["device_closed"] = True
        save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, default=E / "v6_ordered_precision_inline_selector_v2.py.snapshot")
    p.add_argument("--snapshot", type=Path, default=E / "v6_inline_ordered_router.py.snapshot")
    p.add_argument("--capture-prefix", type=Path, default=E / "v6_inline_precision_stream_probe.json")
    p.add_argument("--positions", type=int, nargs="+", default=[227, 738])
    p.add_argument("--layer", type=int, default=34)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION,
    )
    p.add_argument("--oracle-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
