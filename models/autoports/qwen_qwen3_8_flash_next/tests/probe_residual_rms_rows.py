# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded native RMS row-geometry diagnostic; never an acceptance gate.

Restore the accepted mesh descriptor after lane-env; serialize hardware use::

 timeout 600 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_residual_rms_rows \
   --capture-fixture models/autoports/qwen_qwen3_8_flash_next/doc/full_model/rms_layer0_fixture.pt \
   --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/rms_rows_v1.json

Subsequent runs use --fixture instead of --capture-fixture and load no model.
Without either flag, use seeded BF16 inputs and actual checkpoint norm weights.
Capture loads only layer0 (no PLE), replays the original prompt through position64,
and saves the exact layer0 MLP residual-normalization input at positions64..95.
Native decoder_components.rms is called unchanged, with scoped operation taps.
No precision/fidelity override, alternative RMS implementation or device reset.
"""

import argparse
import hashlib
import json
import os
import time
import traceback
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from .diagnose_resume_chunk_split import ROOT, compare

DEFAULT_CHECKPOINT = "/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa"


def validate_fixture(fixture):
    assert len(fixture["inputs"]) == len(fixture["weights"]) == 4
    for x, w in zip(fixture["inputs"], fixture["weights"], strict=True):
        assert x.shape == (1, 1, 32, 2560) and x.dtype == torch.bfloat16
        assert w.shape == (1, 1, 1, 2560) and w.dtype == torch.float32
        assert torch.isfinite(x).all() and torch.isfinite(w).all()
    assert 0 < float(fixture["eps"]) < 1


def random_fixture(checkpoint, seed):
    from safetensors import safe_open

    key = "model.language_model.layers.0.mlp_hyper_connection.hc_norm.weight"
    checkpoint = Path(checkpoint)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(checkpoint / index[key], framework="pt") as source:
        weight = source.get_tensor(key).float().reshape(4, 2560) + 1
    config = json.loads((checkpoint / "config.json").read_text())
    config = config.get("text_config", config)
    rng = torch.Generator().manual_seed(seed)
    return {
        "inputs": [torch.randn(1, 1, 32, 2560, generator=rng).to(torch.bfloat16) for _ in range(4)],
        "weights": [w.reshape(1, 1, 1, 2560) for w in weight],
        "eps": config["rms_norm_eps"],
        "provenance": {"kind": "seeded_inputs_actual_layer0_mlp_norm", "seed": seed, "checkpoint": str(checkpoint)},
    }


def capture_fixture(tt, mesh, args):
    from models.autoports.qwen_qwen3_8_flash_next.tt.model import QwenModel

    prompt = json.loads(args.prompt_artifact.read_text())["prompt_token_ids"]
    model = QwenModel(args.checkpoint, mesh, packed_cache=args.packed_cache, layer_indices=(0,))
    assert model.packed_embedding is None, "Layer0 fixture capture must not load PLE"
    cache = model.create_cache(capacity=384)
    layer = model.layers[0]
    for start in (0, 32):
        tokens = tt.from_torch(
            torch.tensor([prompt[start : start + 32]], dtype=torch.int32),
            dtype=tt.uint32,
            layout=tt.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=tt.ReplicateTensorToMesh(mesh),
        )
        layer.prefill_forward(model.embed(tokens), state=cache.layers[0])
    fixture = {
        "weights": [tt.to_torch(v).detach().clone() for v in tt.get_device_tensors(layer.mlp_mix.weight)],
        "eps": layer.mlp_mix.eps,
        "provenance": {
            "kind": "exact_layer0_mlp_norm_input",
            "positions": [64, 96],
            "checkpoint": args.checkpoint,
            "prompt_artifact": str(args.prompt_artifact),
            "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
        },
    }
    original = layer.mlp_mix.read

    def capture(residual):
        fixture["inputs"] = [tt.to_torch(v).detach().clone() for v in tt.get_device_tensors(residual)]
        return original(residual)

    tokens = tt.from_torch(
        torch.tensor([prompt[64:96]], dtype=torch.int32),
        dtype=tt.uint32,
        layout=tt.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=tt.ReplicateTensorToMesh(mesh),
    )
    with patch.object(layer.mlp_mix, "read", capture):
        layer.prefill_forward(model.embed(tokens), state=cache.layers[0])
    validate_fixture(fixture)
    args.capture_fixture.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, args.capture_fixture)
    return fixture


def run_tapped(tt, rms, x, weight, eps):
    stages, counts = {}, {}
    labels = {
        "typecast": ("value_fp32", "output_bf16"),
        "multiply": ("square", "normalized", "weighted"),
        "mean": ("mean",),
        "add": ("mean_plus_eps",),
        "rsqrt": ("scale",),
    }

    def snapshot(label, value):
        ranks = [tt.to_torch(v).detach().clone() for v in tt.get_device_tensors(value)]
        stages[label] = torch.cat([v.movedim(-2, 0).reshape(v.shape[-2], -1) for v in ranks], dim=1)

    snapshot("input", x)
    with ExitStack() as stack:
        for name, names in labels.items():
            original = getattr(tt, name)

            def wrapped(*operands, _name=name, _names=names, _original=original, **kwargs):
                index = counts.get(_name, 0)
                counts[_name] = index + 1
                result = _original(*operands, **kwargs)
                assert index < len(_names), f"Unexpected native RMS call: {_name}[{index}]"
                snapshot(_names[index], result)
                return result

            stack.enter_context(patch.object(tt, name, wrapped))
        rms(x, weight, eps)
    assert counts == {name: len(names) for name, names in labels.items()}, counts
    return stages


def cpu_stages(fixture):
    per_rank = []
    for x, weight in zip(fixture["inputs"], fixture["weights"], strict=True):
        values = {"input": x, "value_fp32": x.float()}
        values["square"] = values["value_fp32"] * values["value_fp32"]
        values["mean"] = values["square"].mean(dim=-1, keepdim=True)
        values["mean_plus_eps"] = values["mean"] + fixture["eps"]
        values["scale"] = values["mean_plus_eps"].rsqrt()
        values["normalized"] = values["value_fp32"] * values["scale"]
        values["weighted"] = values["normalized"] * weight
        values["output_bf16"] = values["weighted"].to(torch.bfloat16)
        per_rank.append(values)
    return {
        name: torch.cat([rank[name].movedim(-2, 0).reshape(32, -1) for rank in per_rank], dim=1) for name in per_rank[0]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    fixture_args = parser.add_mutually_exclusive_group()
    fixture_args.add_argument("--fixture", type=Path)
    fixture_args.add_argument("--capture-fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", default=os.environ.get("QWEN_CHECKPOINT", DEFAULT_CHECKPOINT))
    parser.add_argument("--packed-cache", default=os.environ.get("QWEN_PLE_CACHE", "unused_layer0_no_ple"))
    parser.add_argument("--prompt-artifact", type=Path, default=ROOT / "doc/full_model/reduced_resume_aligned.json")
    parser.add_argument("--seed", type=int, default=162)
    args = parser.parse_args()
    report = {"completed": False, "scope": "native RMS producer diagnostic, not acceptance"}
    started, mesh = time.monotonic(), None

    def progress(phase):
        report["last_phase"] = phase
        print("RMS_ROWS_PHASE", phase, flush=True)

    try:
        assert os.environ.get("TT_MESH_GRAPH_DESC_PATH"), "Restore accepted mesh descriptor after lane-env"
        assert not os.environ.get("TT_METAL_DEVICE_PROFILER"), "Functional diagnostic, not profiling"
        import ttnn as tt
        from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import rms

        torch.set_num_threads(8)
        tt.set_fabric_config(tt.FabricConfig.FABRIC_1D_RING)
        mesh = tt.open_mesh_device(tt.MeshShape(1, 4), l1_small_size=32768)
        progress("fixture")
        if args.capture_fixture:
            fixture = capture_fixture(tt, mesh, args)
        elif args.fixture:
            fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
        else:
            fixture = random_fixture(args.checkpoint, args.seed)
        validate_fixture(fixture)
        report["fixture"] = fixture["provenance"]
        fixture_path = args.fixture or args.capture_fixture
        if fixture_path:
            report["fixture_path"] = str(fixture_path)
            report["fixture_sha256"] = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
        report["source_sha256"] = hashlib.sha256((ROOT / "tt/decoder_components.py").read_bytes()).hexdigest()
        report["eps"] = fixture["eps"]

        def upload(parts, dtype):
            return tt.from_torch(
                torch.cat(parts, dim=-1).contiguous(),
                dtype=dtype,
                layout=tt.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=tt.ShardTensorToMesh(mesh, dim=3),
                memory_config=tt.DRAM_MEMORY_CONFIG,
            )

        x, weight = upload(fixture["inputs"], tt.bfloat16), upload(fixture["weights"], tt.float32)
        progress("native_full")
        full = run_tapped(tt, rms, x, weight, fixture["eps"])
        progress("native_repeat")
        repeat = run_tapped(tt, rms, x, weight, fixture["eps"])
        progress("native_split")
        pieces = [
            run_tapped(tt, rms, x[:, :, begin:end, :], weight, fixture["eps"]) for begin, end in ((0, 4), (4, 32))
        ]
        split = {name: torch.cat([piece[name] for piece in pieces], dim=0) for name in full}
        report["stage_order"] = list(full)
        report["repeat"] = {name: compare(full[name], repeat[name]) for name in full}
        report["full_vs_split"] = {name: compare(full[name], split[name]) for name in full}
        cpu = cpu_stages(fixture)
        report["full_vs_cpu_fp32"] = {name: compare(cpu[name], full[name]) for name in full}
        report["split_vs_cpu_fp32"] = {name: compare(cpu[name], split[name]) for name in full}
        report[
            "cpu_reference_note"
        ] = "Same algebra and FP32/BF16 boundaries; CPU reduction and SFPU arithmetic need not be bitwise equivalent."
        assert report["full_vs_split"]["input"]["exact"], "Fixture slicing must preserve every input bit"
        report["first_nonexact_stage"] = next(
            (name for name in full if not report["full_vs_split"][name]["exact"]), None
        )
        report["completed"] = True
        progress("complete")
    except BaseException as exc:
        report["error"], report["traceback"] = f"{type(exc).__name__}: {exc}", traceback.format_exc()
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if mesh is not None:
            tt.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
