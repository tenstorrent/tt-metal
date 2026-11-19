# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TP4 request reuse, logical boundaries, list rows and prefix-state HF gates."""

import argparse
import hashlib
import os
import subprocess
import traceback
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import MultichipDecoder
from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import MODEL, REVISION, load_layer, pcc


def row_pcc(a, b):
    a, b = a.double().reshape(-1, a.shape[-1]), b.double().reshape(-1, b.shape[-1])
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    values = (a * b).sum(-1) / (a.square().sum(-1) * b.square().sum(-1)).sqrt()
    assert torch.isfinite(values).all(), "Non-finite per-row PCC"
    return values.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    stage = root / "doc/optimized_multichip_decoder"
    if not args.output.resolve().is_relative_to(stage.resolve()):
        raise ValueError("Contract artifacts must remain in optimized_multichip_decoder")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(183)
    c = AutoConfig.from_pretrained(MODEL, revision=REVISION, local_files_only=True).text_config
    c._attn_implementation = "sdpa"
    weights = load_layer(args.layer)
    hf = Qwen3_5DecoderLayer(c, args.layer).eval().to(torch.bfloat16)
    hf.load_state_dict(weights)
    rot = Qwen3_5TextRotaryEmbedding(c)
    results = []

    report = {
        "status": "running",
        "pass": False,
        "layer": args.layer,
        "mesh": [1, 4],
        "hf_revision": REVISION,
        "contract_schema_version": 2,
        "pcc_threshold": 0.995,
        "reference_types": {
            "hf": "independent BF16 HuggingFace Qwen3_5DecoderLayer full prefill",
            "tp4_full_prefill": "same TP4 decoder uninterrupted prefill; self-consistency only",
        },
        "checks": results,
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__).resolve(),
                root / "tt/multichip_decoder.py",
                root / "tt/optimized_decoder.py",
                root / "tt/flat_prefill.py",
                root / "doc/hf_config.json",
            )
        },
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {
            k: os.environ.get(k)
            for k in (
                "TT_METAL_WATCHER",
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_METAL_DEVICE_PROFILER",
            )
        },
        "runtime_host_conversion_guard": True,
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    try:
        layer = MultichipDecoder.from_state_dict(weights, hf_config=c, layer_idx=args.layer, mesh_device=mesh)
        report["selected_policy"] = dict(layer.policy)
        linear = layer.kind == "linear_attention"
        pages = torch.randperm(64, dtype=torch.int32).reshape(1, 64)
        state = layer.create_recurrent_state() if linear else layer.create_paged_state(pages)
        buffers = [state.conv, state.recurrent] if linear else [state.key, state.value]
        zero = [layer._upload(torch.zeros(tuple(t.shape)), dtype=t.dtype, layout=t.layout) for t in buffers]

        def reset():
            for z, t in zip(zero, buffers):
                ttnn.copy(z, t)

        def read(t):
            # Readback is outside the forward guard and only at test boundaries.
            return torch.cat([ttnn.to_torch(v) for v in ttnn.get_device_tensors(t)], dim=-1)

        def inputs(x, start=0):
            r = (
                None
                if linear
                else tuple(
                    layer._upload(t.unsqueeze(1))
                    for t in rot(x, torch.arange(start, start + x.shape[1]).reshape(1, -1))
                )
            )
            return layer.upload_residual(x), r

        def forward(x, r, start=0, target_state=None):
            with (
                patch.object(ttnn, "from_torch", side_effect=AssertionError("runtime from_torch")),
                patch.object(ttnn, "to_torch", side_effect=AssertionError("runtime to_torch")),
            ):
                return layer.prefill_forward(
                    x, state=state if target_state is None else target_state, start_pos=start, rope=r
                )

        for length in [1, 31, 32, 33, 127, 128, 129, 511, 512, 513, 1025, 17]:
            reset()
            if not linear:
                pages = torch.randperm(64, dtype=torch.int32).reshape(1, 64)
                ttnn.copy(layer._upload(pages, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), state.page_table)
            x = torch.randn(1, length, 5120, dtype=torch.bfloat16) * 0.1
            with torch.no_grad():
                expected = hf(
                    x,
                    position_embeddings=rot(x, torch.arange(length).reshape(1, -1)),
                    past_key_values=DynamicCache(config=c),
                )
            tx, r = inputs(x)
            actual = read(forward(tx, r))
            rows = row_pcc(expected.float() - x.float(), actual.float() - x.float())
            entry = {
                "case": "length",
                "reference_type": "hf",
                "length": length,
                "pcc": pcc(expected, actual),
                "min_contribution_pcc": min(rows),
                "worst_row": min(range(len(rows)), key=rows.__getitem__),
                "contribution_pcc": rows,
                "all_finite": bool(torch.isfinite(actual).all()),
            }
            results.append(entry)
            save()
            print({k: v for k, v in entry.items() if k != "contribution_pcc"}, flush=True)
            assert entry["pcc"] >= 0.995 and min(rows) >= 0.995 and entry["all_finite"]
        # Keep the TP4 self-consistency oracle and add an independent HF oracle.
        length = 545
        x = torch.randn(1, length, 5120, dtype=torch.bfloat16) * 0.1
        with torch.no_grad():
            hf_full = hf(
                x,
                position_embeddings=rot(x, torch.arange(length).reshape(1, -1)),
                past_key_values=DynamicCache(config=c),
            )
        reset()
        tx, r = inputs(x)
        full = read(forward(tx, r))
        for start in [1, 32, 33, 128, 511, 512, 544]:
            reset()
            prefix, pr = inputs(x[:, :start])
            read(forward(prefix, pr))
            n = min(8, length - start)
            suffix, sr = inputs(x[:, start : start + n], start)
            resumed = read(forward(suffix, sr, start))
            residual = x[:, start : start + n].float()
            expected = full[:, start : start + n].float() - residual
            positive = row_pcc(expected, resumed.float() - residual)
            hf_expected = hf_full[:, start : start + n]
            hf_output_pcc = pcc(hf_expected, resumed)
            hf_output_rows = row_pcc(hf_expected, resumed)
            hf_contribution = row_pcc(hf_expected.float() - residual, resumed.float() - residual)
            reset()
            negative = read(forward(suffix, sr, start))
            control = row_pcc(expected, negative.float() - residual)
            entry = {
                "case": "resume",
                "reference_types": ["tp4_full_prefill", "hf"],
                "start": start,
                "rows": n,
                "positive": positive,
                "dropped_history": control,
                "hf_output_pcc": hf_output_pcc,
                "hf_output_row_pcc": hf_output_rows,
                "hf_contribution_pcc": hf_contribution,
                "all_finite": bool(torch.isfinite(resumed).all() and torch.isfinite(hf_expected).all()),
            }
            results.append(entry)
            save()
            print(entry, flush=True)
            assert min(positive) >= 0.995, entry
            assert min(control) < 0.995, entry
            assert entry["all_finite"] and min(hf_output_pcc, *hf_output_rows, *hf_contribution) >= 0.995, entry
        mixed_x, mixed_state, mixed_rope, mixed_expected, mixed_residual = [], [], [], [], []
        starts = [32, 33]
        for start, tail in zip(starts, [33, 17]):
            xx = torch.randn(1, start + tail, 5120, dtype=torch.bfloat16) * 0.1
            with torch.no_grad():
                yy = hf(
                    xx,
                    position_embeddings=rot(xx, torch.arange(start + tail).reshape(1, -1)),
                    past_key_values=DynamicCache(config=c),
                )
            ss = (
                layer.create_recurrent_state()
                if linear
                else layer.create_paged_state(torch.randperm(64).reshape(1, 64))
            )
            prefix, pr = inputs(xx[:, :start])
            out = forward(prefix, pr, target_state=ss)
            del out
            suffix, sr = inputs(xx[:, start:], start)
            mixed_x.append(suffix)
            mixed_state.append(ss)
            mixed_rope.append(sr)
            mixed_expected.append(yy[:, start:])
            mixed_residual.append(xx[:, start:].float())
        mixed = forward(mixed_x, mixed_rope, starts, target_state=mixed_state)
        mixed_actual = [read(a) for a in mixed]
        mixed_pcc = [pcc(e, a) for e, a in zip(mixed_expected, mixed_actual)]
        mixed_output_rows = [row_pcc(e, a) for e, a in zip(mixed_expected, mixed_actual)]
        mixed_contribution = [
            row_pcc(e.float() - r, a.float() - r) for e, a, r in zip(mixed_expected, mixed_actual, mixed_residual)
        ]
        mixed_finite = all(bool(torch.isfinite(t).all()) for t in (*mixed_expected, *mixed_actual))
        results.append(
            {
                "case": "mixed_rows",
                "reference_type": "hf",
                "starts": starts,
                "lengths": [33, 17],
                "pcc": mixed_pcc,
                "output_row_pcc": mixed_output_rows,
                "contribution_pcc": mixed_contribution,
                "all_finite": mixed_finite,
            }
        )
        save()
        assert all(math.isfinite(p) and p >= 0.995 for p in mixed_pcc)
        assert mixed_finite and min(p for rows in (*mixed_output_rows, *mixed_contribution) for p in rows) >= 0.995
        report["status"] = "pass"
        report["pass"] = True
        save()
        print("TP4_CONTRACT_PASS", flush=True)
    except BaseException:
        report["status"] = "failed"
        report["exception"] = traceback.format_exc()
        save()
        raise
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
