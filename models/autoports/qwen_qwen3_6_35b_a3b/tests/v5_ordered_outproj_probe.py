# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Actual layer34 gated-output operands: native versus ordered output projection.

CPU-only setup/sum proof with --oracle-only; device execution belongs to parent.
All 2048 output columns compared, with FP32 ordered sums proved on selected columns.
"""
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
COLUMNS = [0, 1, 31, 32, 127, 511, 1023, 1024, 1535, 2047]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scope_proof(base, candidate):
    bt, bm = source_methods(base)
    ct, cm = source_methods(candidate)
    assert {k for k in bm if ast.dump(bm[k]) != ast.dump(cm[k])} == {"from_state_dict", "linear", "_linear_attention"}
    # Restore sole decode call argument, added linear guard/keyword, and setup assignments.
    tail = cm["_linear_attention"].body[-1].value
    assert isinstance(tail, ast.Call) and tail.keywords[-1].arg == "ordered_out"
    assert ast.unparse(tail.keywords[-1].value) == "cache_boundary"
    tail.keywords.pop()
    assert cm["linear"].args.kwonlyargs[-1].arg == "ordered_out"
    cm["linear"].args.kwonlyargs.pop()
    cm["linear"].args.kw_defaults.pop()
    cm["linear"].body.pop(0)
    setup = cm["from_state_dict"].body[-2]
    assert isinstance(setup, ast.If) and ast.unparse(setup.test) == "self.kind == 'linear_attention'"
    names = [ast.unparse(n.targets[0]) for n in setup.body]
    assert names == [
        "self.output_norm_bf16",
        "output_order",
        "self.scan_attention_out_weight",
        "output_pair32",
        "self.output_pair32_indices",
    ], names
    setup.body = setup.body[:1]
    assert ast.dump(bt) == ast.dump(ct)
    return {
        "full_module_ast_restoration": True,
        "public_decode_only": True,
        "prefill_including_singleton_tails_unchanged": True,
    }


def setup_native(obj, source, native, mesh):
    _, methods = source_methods(source)
    body = methods["from_state_dict"].body
    start = next(
        i for i, n in enumerate(body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "self.decode_linear"
    )
    end = next(
        i
        for i in range(start, len(body))
        if isinstance(body[i], ast.For) and ast.unparse(body[i].iter) == "projection_keys"
    )
    nodes = [
        n
        for n in body[start : end + 1]
        if not (isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "projection_keys")
    ]
    module = ast.fix_missing_locations(ast.Module(copy.deepcopy(nodes), []))
    exec(
        compile(module, str(source) + ":native_setup", "exec"),
        dict(self=obj, ttnn=native, mesh_device=mesh, projection_keys=("linear_attn.out_proj.weight",)),
    )


def setup_ordered(obj, source, weight, upload, native, mesh):
    _, methods = source_methods(source)
    block = methods["from_state_dict"].body[-2]
    nodes = copy.deepcopy(block.body[1:])
    module = ast.fix_missing_locations(ast.Module(nodes, []))
    exec(
        compile(module, str(source) + ":ordered_setup", "exec"),
        dict(
            self=obj,
            torch=torch,
            ttnn=native,
            mesh_device=mesh,
            upload=upload,
            DRAM=native.DRAM_MEMORY_CONFIG,
            state_dict={"linear_attn.out_proj.weight": weight},
        ),
    )


def ordered_cpu(x, weight, k):
    order = torch.arange(4096).reshape(2048, 2).flip(1).flatten()
    blocks = 4096 // k
    products = x.reshape(4096)[order].float().reshape(blocks, k, 1) * weight[COLUMNS, :].T[order].float().reshape(
        blocks, k, len(COLUMNS)
    )
    accum = torch.zeros_like(products[:, 0])
    for i in range(k):
        accum = accum + products[:, i]
    total = accum[0].clone()
    for i in range(1, blocks):
        total = total + accum[i]
    return total.reshape(1, 1, -1)


def run(args):
    torch.set_num_threads(8)
    assert not args.output.exists(), "Preserve previous evidence"
    sources = {str(k): E / f"v5_ordered_outproj_k{k}.py.snapshot" for k in (512, 1024)}
    scope = {k: scope_proof(args.baseline, p) for k, p in sources.items()}
    cap = torch.load(args.capture, map_location="cpu", weights_only=False)
    assert cap["layer"] == args.layer
    assert sha(args.baseline) in cap["source_sha256"].values()
    x = cap["tt_values"]["gated_core"].reshape(1, 1, 4096).bfloat16()
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    key = f"model.language_model.layers.{args.layer}.linear_attn.out_proj.weight"
    with safe_open(args.checkpoint / index[key], framework="pt", device="cpu") as f:
        weight = f.get_tensor(key)
    assert weight.dtype == torch.bfloat16 and tuple(weight.shape) == (2048, 4096)
    canonical = torch.nn.functional.linear(x, weight)
    canonical_fp32 = torch.nn.functional.linear(x.float(), weight.float())
    order = torch.arange(4096).reshape(2048, 2).flip(1).flatten()
    pair = torch.arange(32).reshape(16, 2).flip(1).reshape(1, 1, 32).expand(1, 128, 32)
    assert torch.equal(torch.gather(x.reshape(1, 128, 32), -1, pair).reshape(-1), x.reshape(-1)[order])
    assert torch.equal(weight.T[order][order], weight.T)
    expected = {k: ordered_cpu(x, weight, int(k)) for k in sources}
    result = dict(
        status="cpu_proof_pass",
        classification="diagnostic_only",
        acceptance_eligible=False,
        probe_sha256=sha(__file__),
        baseline=str(args.baseline),
        baseline_sha256=sha(args.baseline),
        candidates={k: {"path": str(p), "sha256": sha(p), "scope_proof": scope[k]} for k, p in sources.items()},
        capture=str(args.capture),
        capture_sha256=sha(args.capture),
        layer=args.layer,
        position=cap["position"],
        checkpoint_revision=REVISION,
        input_description="Actual captured TT gated BF16 tensor; all paths use identical operands",
        selected_columns=COLUMNS,
        cpu_selected_ordered_vs_hf={k: metrics(canonical[..., COLUMNS], v.bfloat16()) for k, v in expected.items()},
        cpu_canonical_vs_captured_native_attention=metrics(canonical, cap["tt_values"]["attention"]),
        product_or_prefix_bytes=33554432,
        extra_ordered_weight_bytes=16777216,
        timing="not measured",
        device_closed=None,
        cases=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

    record(args.output)
    save()
    if args.oracle_only:
        assert "ttnn" not in sys.modules
        print("CPU_ORDERED_OUTPROJ_PASS", result["cpu_selected_ordered_vs_hf"])
        return
    for key in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"):
        assert os.environ.get(key) in (None, "", "0"), key
    import ttnn

    mesh = None
    payload = {
        "canonical_bf16": canonical,
        "canonical_fp32": canonical_fp32,
        "selected_ordered_fp32": expected,
        "outputs": {},
    }
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768)
        result.update(status="running", device_closed=False)

        def upload(v, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                v.contiguous(), device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        dx = upload(x)
        for label, source in [("native", args.baseline)] + list(sources.items()):
            obj = load_snapshot(source, "outproj_probe_" + label).FusedDecoder()
            obj.kind = "linear_attention"
            obj.mesh_device = mesh
            obj.compute = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            obj.w = {"linear_attn.out_proj.weight": upload(weight.T)}
            setup_native(obj, args.baseline, ttnn, mesh)
            if label != "native":
                setup_ordered(obj, source, weight, upload, ttnn, mesh)
                assert torch.equal(ttnn.to_torch(obj.scan_attention_out_weight).reshape(4096, 2048), weight.T[order])
                assert torch.equal(ttnn.to_torch(obj.output_pair32_indices).long(), pair)
            payload["outputs"][label] = {}
            row = {"variant": label, "outputs": {}}
            result["cases"].append(row)
            for dtype_name, dtype in [("bf16", ttnn.bfloat16), ("fp32", ttnn.float32)]:
                kw = {} if label == "native" else {"ordered_out": True}
                output = obj.linear(dx, "linear_attn.out_proj.weight", dtype, **kw)
                assert tuple(output.shape) == (1, 1, 2048) and output.dtype == dtype
                assert output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
                got = ttnn.to_torch(output).clone()
                assert torch.isfinite(got).all()
                row["outputs"][dtype_name] = {
                    "vs_hf_all_columns": metrics(canonical if dtype_name == "bf16" else canonical_fp32, got)
                }
                if label != "native":
                    target = expected[label] if dtype_name == "fp32" else expected[label].bfloat16()
                    check = metrics(target, got[..., COLUMNS])
                    row["outputs"][dtype_name]["vs_ordered_selected"] = check
                    assert check["exact"], "Actual ordered adapter must match explicit sequential selected-column sum"
                else:
                    row["outputs"][dtype_name]["vs_captured_attention"] = metrics(cap["tt_values"]["attention"], got)
                payload["outputs"][label][dtype_name] = got
                save()
                del output
            del obj
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
    p.add_argument("--baseline", type=Path, default=E / "v5_combined_accurate_residual_explicit_gate_norm.py.snapshot")
    p.add_argument("--capture", type=Path, default=E / "v5_explicit_gate_norm_layer34_181_probe.json.pos181.pt")
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
