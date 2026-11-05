# SPDX-License-Identifier: Apache-2.0
"""Correctness-only probe separating physical-row numerics from continuation state.

Uses real reduced layers 0/5 by default. Host readback is explicit probe work;
this is not a serving/performance harness and does not enable profiling.
"""

import argparse
import bz2
import json
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn

from ..tt.generator import Gemma4Generator


def compare(reference, candidate):
    assert reference.shape == candidate.shape, (reference.shape, candidate.shape)
    a, b = reference.reshape(reference.shape[0], -1).double(), candidate.reshape(candidate.shape[0], -1).double()
    ac, bc = a - a.mean(dim=1, keepdim=True), b - b.mean(dim=1, keepdim=True)
    denom = ac.norm(dim=1) * bc.norm(dim=1)
    pcc = (ac * bc).sum(dim=1) / denom.clamp_min(1e-30)
    equal_rows = (a == b).all(dim=1)
    pcc = torch.where(denom == 0, equal_rows.double(), pcc)
    return {
        "shape": list(reference.shape),
        "exact_equal": torch.equal(reference, candidate),
        "max_abs_difference": float((a - b).abs().max()),
        "mean_abs_difference": float((a - b).abs().mean()),
        "pcc_mean": float(pcc.mean()),
        "pcc_min": float(pcc.min()),
        "worst_row_offset": int(pcc.argmin()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--qkv-stages", action="store_true", help="Read layer0 QKV intermediates for fresh-shape comparisons"
    )
    parser.add_argument("--sizes", default="32,64,128,256")
    parser.add_argument("--continuations", default="32,64,128,256")
    parser.add_argument(
        "--text-file", type=Path, default=Path("models/tt_transformers/tests/tale-of-two-cities.txt.bz2")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/autoports/google_gemma_4_31b_it/readiness_vllm/prefill_geometry.json"),
    )
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    starts = [int(value) for value in args.continuations.split(",") if value.strip()]
    assert all(0 < value < 1024 and value % 32 == 0 for value in sizes + starts)
    torch.set_num_threads(8)
    text = bz2.open(args.text_file, "rt").read() if args.text_file.suffix == ".bz2" else args.text_file.read_text()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        gen = Gemma4Generator(mesh, layer_indices=None if args.full else (0, 5), batch_size=1, max_seq_len=4096)
        gen.prepare_decode()
        token_ids = gen.tokenizer.encode(text, add_special_tokens=True)[:1024]
        assert len(token_ids) == 1024, "Need at least 1024 nonrepeating natural-text tokens"
        tokens = torch.tensor(token_ids, dtype=torch.int32)
        captures = {}
        current_run = None
        observed_geometry = {}
        cache_before = mesh.num_program_cache_entries()

        def capture_layer(original, layer, layer_index, cache_index):
            def wrapped(x, **kwargs):
                output = original(x, **kwargs)
                start = int(kwargs["chunk_start"])
                rows = int(kwargs["valid_tokens"])
                key = str(layer_index)
                hidden = ttnn.to_torch(ttnn.get_device_tensors(output)[0]).float().reshape(-1, output.shape[-1])[:rows]
                cache = kwargs["kv_cache"]
                table = gen.prefill.host_pages[cache_index][0]
                positions = torch.arange(start, start + rows)
                page_ids, offsets = table[positions // 32].long(), positions % 32
                kv = []
                for tensor in cache:
                    chips = []
                    for chip in ttnn.get_device_tensors(tensor):
                        host = ttnn.to_torch(chip)
                        chips.append(host[page_ids, :, offsets, :].float())
                    kv.append(torch.cat(chips, dim=1))
                captures.setdefault(current_run, {})[key] = {
                    "start": start,
                    "hidden": hidden.clone(),
                    "k": kv[0],
                    "v": kv[1],
                }
                attention_start = min(start, layer.sliding_window) if layer.sliding_window else start
                observed_geometry.setdefault(current_run, {})[key] = {
                    "layer_type": layer.layer_type,
                    "input_shape": list(x.shape),
                    "input_padded_shape": list(x.padded_shape),
                    "chunk_start": start,
                    "valid_tokens": rows,
                    "attention_start": attention_start,
                    "q_chunk_size": 32 if attention_start % 128 else layer.geometry["prefill_q"],
                    "k_chunk_size": 32 if attention_start % 128 else layer.geometry["prefill_k"],
                    "prefill_l1": layer.prefill_l1,
                    "hidden_readback": "chip0, replicated residual",
                    "kv_readback": "all four chips, head-axis concatenation",
                }
                return output

            return wrapped

        stage_captures = {}
        in_qkv_probe = False
        stage_order = 0

        def snapshot_stage(name, tensor):
            nonlocal stage_order
            # All probed prefill tensors are [batch, heads-or-1, tokens, width].
            # Read all chips to catch a shape-dependent discrepancy on any shard.
            assert len(tensor.shape) == 4
            host_rows = []
            for chip in ttnn.get_device_tensors(tensor):
                host = ttnn.to_torch(chip).float()
                host_rows.append(host.movedim(2, 0).reshape(host.shape[2], -1)[:32].clone())
            key = f"{stage_order:02d}:{name}"
            stage_captures.setdefault(current_run, {})[key] = {
                "rows": torch.cat(host_rows, dim=1),
                "shape": list(tensor.shape),
                "padded_shape": list(tensor.padded_shape),
                "dtype": str(tensor.dtype),
                "token_axis": 2,
            }
            stage_order += 1

        stage_patches = []
        if args.qkv_stages:
            layer0 = gen.model.layers[0]
            original_norm = layer0._norm
            original_rope = layer0._rope
            original_qkv = layer0._qkv
            original_linear = layer0._linear

            def norm_probe(x, name=None):
                if in_qkv_probe:
                    snapshot_stage(f"norm[{name}].input", x)
                output = original_norm(x, name)
                if in_qkv_probe:
                    snapshot_stage(f"norm[{name}].output", output)
                return output

            def linear_probe(x, name):
                output = original_linear(x, name)
                if in_qkv_probe:
                    snapshot_stage(f"linear[{name}].output", output)
                return output

            def rope_probe(x, cos, sin, *, decode):
                if in_qkv_probe:
                    snapshot_stage("rope.input", x)
                    snapshot_stage("rope.cos", cos)
                    snapshot_stage("rope.sin", sin)
                output = original_rope(x, cos, sin, decode=decode)
                if in_qkv_probe:
                    snapshot_stage("rope.output", output)
                return output

            def qkv_probe(x, cos, sin, *, decode):
                nonlocal in_qkv_probe, stage_order
                in_qkv_probe = not decode and current_run.startswith("fresh_")
                stage_order = 0
                try:
                    if in_qkv_probe:
                        snapshot_stage("qkv.input", x)
                    result = original_qkv(x, cos, sin, decode=decode)
                    if in_qkv_probe:
                        for name, tensor in zip(("q", "k_after_cache_cast", "v_after_cache_cast"), result):
                            snapshot_stage(f"qkv.output.{name}", tensor)
                    return result
                finally:
                    in_qkv_probe = False

            stage_patches = [
                patch.object(layer0, "_norm", norm_probe),
                patch.object(layer0, "_linear", linear_probe),
                patch.object(layer0, "_rope", rope_probe),
                patch.object(layer0, "_qkv", qkv_probe),
            ]

        patches = [
            patch.object(
                layer, "prefill_chunk_forward", capture_layer(layer.prefill_chunk_forward, layer, index, ordinal)
            )
            for ordinal, (index, layer) in enumerate(zip(gen.model.layer_indices, gen.model.layers))
        ]
        patches.extend(stage_patches)
        for context in patches:
            context.start()
        try:
            current_run = "fresh_1024"
            gen.reset()
            for _, _, hidden in gen._prefill_hidden_chunks(tokens, slot=0, start=0, end=1024):
                del hidden
            for size in sizes:
                current_run = f"fresh_{size}"
                gen.reset()
                for _, _, hidden in gen._prefill_hidden_chunks(tokens, slot=0, start=0, end=size):
                    del hidden
            for start in starts:
                current_run = f"resume_{start}_prefix"
                gen.reset()
                for _, _, hidden in gen._prefill_hidden_chunks(tokens, slot=0, start=0, end=start):
                    del hidden
                current_run = f"resume_{start}_suffix"
                for _, _, hidden in gen._prefill_hidden_chunks(tokens, slot=0, start=start, end=1024):
                    del hidden
        finally:
            for context in reversed(patches):
                context.stop()

        comparisons = []
        reference = captures["fresh_1024"]
        for name, layers in captures.items():
            if name == "fresh_1024":
                continue
            for index, capture in layers.items():
                start = capture["start"]
                count = capture["hidden"].shape[0]
                comparisons.append(
                    {
                        "case": name,
                        "layer_index": int(index),
                        "layer_type": observed_geometry[name][index]["layer_type"],
                        "absolute_range": [start, start + count],
                        "reference": "fresh_1024",
                        "geometry": observed_geometry[name][index],
                        "reference_geometry": observed_geometry["fresh_1024"][index],
                        "hidden": compare(reference[index]["hidden"][start : start + count], capture["hidden"]),
                        "k": compare(reference[index]["k"][start : start + count], capture["k"]),
                        "v": compare(reference[index]["v"][start : start + count], capture["v"]),
                    }
                )
        qkv_comparisons = []
        if args.qkv_stages:
            baseline_stages = stage_captures["fresh_1024"]
            for name, stages in stage_captures.items():
                if name == "fresh_1024":
                    continue
                assert list(stages) == list(baseline_stages), "QKV call sequence changed between shapes"
                for stage, capture in stages.items():
                    baseline = baseline_stages[stage]
                    qkv_comparisons.append(
                        {
                            "case": name,
                            "layer_index": int(gen.model.layer_indices[0]),
                            "stage": stage,
                            "absolute_range": [0, 32],
                            "source_shape": capture["shape"],
                            "source_padded_shape": capture["padded_shape"],
                            "reference_shape": baseline["shape"],
                            "dtype": capture["dtype"],
                            "token_axis": capture["token_axis"],
                            **compare(baseline["rows"], capture["rows"]),
                        }
                    )
        report = {
            "scope": "full_model_probe" if args.full else "reduced_real_layers_0_5",
            "purpose": "Discriminate fresh physical-row kernel drift from continuation/cache drift",
            "source_text": str(args.text_file),
            "prompt_tokens": 1024,
            "context": 4096,
            "precision_config": gen.model.precision_config,
            "layer_indices": list(gen.model.layer_indices),
            "program_cache_before": cache_before,
            "program_cache_after": mesh.num_program_cache_entries(),
            "comparisons": comparisons,
            "qkv_stage_comparisons": qkv_comparisons,
            "interpretation": [
                "Fresh-prefix differences occur without continuation and implicate row-count-dependent numerics.",
                "Compare the earliest layer's K/V before attention to separate projection/RoPE drift from attention drift.",
                "Resume-only drift with stable fresh-prefix results implicates cache history, addressing or SDPA offset geometry.",
                "Later-layer K/V differences can propagate earlier hidden-output differences.",
            ],
            "limitations": ["Explicit host readback correctness probe; no performance or profiler measurement."],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"output": str(args.output), "comparisons": len(comparisons)}))
    finally:
        if gen is not None:
            gen.release_persistent_capture()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
