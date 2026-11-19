# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Observe existing prefill operations without changing math or adding TT buffers.

Diagnostic only: CPU readbacks invalidate performance claims. The original
fresh/resumed logit gate is recorded unchanged; it is not relaxed here.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=2559)
    parser.add_argument("--diagnostic", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    diagnostic_path = (
        args.diagnostic or ROOT / f"doc/datatype_sweep/baseline_resume_diagnostics/L2560_s{args.start}.json"
    )
    diagnostic = json.loads(diagnostic_path.read_text())
    tokens = diagnostic["tokens"]
    assert args.start == diagnostic["gate_result"]["start"] and len(tokens) == 2560
    assert diagnostic["gate_modified"] is False
    target, prefix_last = len(tokens) - 1, args.start - 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "statistics.json"
    assert not output.exists(), "Use a fresh localization output directory"
    assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "1"
    assert not os.environ.get("TT_METAL_WATCHER"), "Watcher requires a separate run"
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
    assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
    assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"

    import torch

    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import build_generator

    torch.set_num_threads(8)
    helper_path = ROOT / "doc/full_model/qualitative/hf_resume_control/run_control.py"
    spec = importlib.util.spec_from_file_location("hf_resume_helpers", helper_path)
    helpers_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers_module)
    runner_path = Path(os.environ["TT_MODEL_BRINGUP_ROOT"]) / "runtime/readiness_check/run_prefill_resume_check.py"
    helpers, helper_sha = helpers_module.checker_functions(runner_path)
    thresholds = dict(pcc_mean_min=0.99, pcc_min_min=0.98, top1_min=0.98, control_min_window=64)
    positions = helpers["window_positions"](args.start, len(tokens), 256)
    assert positions == diagnostic["positions"]
    result = dict(
        status="running",
        command=sys.argv,
        time_unix=time.time(),
        context=4096,
        layers=list(range(40)),
        mesh=[1, 4],
        precision_config="baseline",
        start=args.start,
        target=target,
        prefix_last=prefix_last,
        positions=positions,
        tokens=tokens,
        diagnostic=str(diagnostic_path),
        thresholds=thresholds,
        gate_modified=False,
        scope="Same existing TT math and operations; synchronous host observations only; no performance claim",
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), runner_path, helper_path, diagnostic_path, *sorted((ROOT / "tt").glob("*.py"))]
        },
        numeric_helpers_ast_sha256=helper_sha,
        calls=[],
        hidden_comparisons=[],
        state_comparisons=[],
    )
    captures, metadata, lookup = {}, {}, {}
    mesh = generator = None
    current = dict(phase=None, call=None, read=None, operation=None)

    def save():
        (args.output_dir / "capture_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    def tensor_meta(tensor):
        return dict(
            shape=list(tensor.shape),
            padded_shape=list(tensor.padded_shape),
            dtype=str(tensor.dtype),
            layout=str(tensor.layout),
            memory_config=str(tensor.memory_config()),
        )

    def selected_positions(index, absolute, valid):
        wanted = {target, prefix_last}
        if index == 0:
            wanted.update(range(min(64, args.start)))
        return sorted(p for p in wanted if absolute <= p < absolute + valid)

    def hidden(event, index, tensor, context):
        absolute, valid = context["absolute"], context["valid_rows"]
        wanted = selected_positions(index, absolute, valid)
        if not wanted:
            return
        phase = current["phase"]
        key = f"{phase}/l{index:02}/{event}/a{absolute}_n{valid}"
        assert key not in captures, key
        # Residual/normalized input is replicated across mesh ranks. Read only
        # rank0, then select host rows; no device slice/conversion is introduced.
        host = ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).reshape(-1, tensor.shape[-1])
        assert host.shape[-1] == 2048, (event, host.shape)
        value = host[[p - absolute for p in wanted]].detach().clone()
        assert torch.isfinite(value).all(), key
        captures[key] = value
        metadata[key] = dict(
            event=event,
            layer=index,
            phase=phase,
            positions=wanted,
            tensor=tensor_meta(tensor),
            context=context,
            mesh_read="first replica rank0",
        )
        for offset, p in enumerate(wanted):
            lookup[(phase, index, event, p)] = (key, offset)

    def states(event, index, state, context):
        if "recurrent" not in state:
            return
        for name in ("recurrent", "conv"):
            key = f"{current['phase']}/l{index:02}/state.{event}/{name}"
            assert key not in captures, key
            parts = [ttnn.to_torch(part).detach().clone() for part in ttnn.get_device_tensors(state[name])]
            assert len(parts) == 4 and all(torch.isfinite(part).all() for part in parts)
            captures[key] = parts
            metadata[key] = dict(
                event=f"state.{event}",
                layer=index,
                phase=current["phase"],
                tensor=tensor_meta(state[name]),
                context=context,
                mesh_read="all ranks, no concatenation",
            )

    def stats(left, right):
        delta = left.float() - right.float()
        return dict(
            bit_equal=torch.equal(left, right),
            pcc=helpers["_pcc"](left, right),
            max_abs=float(delta.abs().max()),
            rms_difference=float(delta.square().mean().sqrt()),
        )

    save()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
        generator = build_generator(
            ROOT, mesh, max_seq_len=4096, max_batch_size=1, host_sampling=False, precision_config="baseline"
        )
        model, executor = generator.model, generator.prefill_executor
        assert tuple(model.layer_indices) == tuple(range(40))
        result["runtime_precision"] = model.precision_summary()
        # Warm original logit diagnostics before any trace capture. State
        # readbacks are direct buffer reads and receive the same preparation.
        generator.prefill_logits([1] * 33)
        model.host_logits(generator.sampler_logits)
        for group in generator.cache:
            for name in ("recurrent", "conv"):
                if name in group[0]:
                    for part in ttnn.get_device_tensors(group[0][name]):
                        ttnn.to_torch(part)
        generator.prepare(prefill_cases=((32, 0),))
        generator.reset()
        result["trace_ids"] = {k: int(v) for k, v in generator.traces.items()}
        result["program_cache_entries_after_prepare"] = int(mesh.num_program_cache_entries())
        initial = dict(generator.counters)
        indices = {id(layer): index for index, layer in zip(model.layer_indices, model.layers)}
        original_prefill, original_read = executor.prefill, executor._read
        original_attention, original_moe, original_begin = executor._attention, executor._moe, executor._begin_layer

        def prefill(token_ids, **kwargs):
            start = int(kwargs.get("start", 0))
            length = int(token_ids.numel()) if isinstance(token_ids, torch.Tensor) else len(token_ids)
            phase = "fresh" if start == 0 and length == len(tokens) else "prefix" if start == 0 else "suffix"
            current.update(
                phase=phase, call=dict(start=start, length=length, end=start + length), read=None, operation=None
            )
            result["calls"].append(dict(phase=phase, **current["call"]))
            save()
            print("LOCALIZE_CALL", phase, start, length, flush=True)
            return original_prefill(token_ids, **kwargs)

        def read(source, offset, rows, *, valid_rows=None):
            current["read"] = dict(
                offset=offset,
                physical_rows=rows,
                valid_rows=rows if valid_rows is None else valid_rows,
                absolute=current["call"]["start"] + offset,
            )
            return original_read(source, offset, rows, valid_rows=valid_rows)

        def begin(layer, state):
            index = indices[id(layer)]
            if current["phase"] == "suffix":
                states("public_begin.before_bf16_cast", index, state, dict(current["call"]))
            out = original_begin(layer, state)
            if current["phase"] == "suffix":
                states("public_begin.after_bf16_cast", index, state, dict(current["call"]))
            return out

        def attention(layer, state, source, offset, rows, absolute, public_length):
            index = indices[id(layer)]
            context = dict(
                absolute=absolute,
                valid_rows=rows,
                physical_rows=rows,
                offset=offset,
                public_length=public_length,
                layer_kind=layer.kind,
                decode=rows == 1 and (layer.kind == "full_attention" or public_length == 1),
                cache_boundary=False,
                projection_k=1024 if public_length > 1 else 512,
            )
            current["operation"] = (index, "attention", context)
            last = absolute + rows == current["call"]["end"]
            if last:
                states("last_attention.before", index, state, context)
            out = original_attention(layer, state, source, offset, rows, absolute, public_length)
            if last:
                states("last_attention.after", index, state, context)
            current["operation"] = None
            return out

        def moe(layer, x):
            index = indices[id(layer)]
            context = dict(
                current["read"],
                layer_kind=layer.kind,
                public_length=current["call"]["length"],
                ordered_router=False,
                batched=False,
                expected_branch="indexed_sparse" if x.shape[-2] == 1 else "grouped_EP",
            )
            current["operation"] = (index, "moe", context)
            hidden("moe_input", index, x, context)
            out = original_moe(layer, x)
            hidden("moe_output", index, out, context)
            current["operation"] = None
            return out

        with ExitStack() as patches:
            for name, hook in (
                ("prefill", prefill),
                ("_read", read),
                ("_begin_layer", begin),
                ("_attention", attention),
                ("_moe", moe),
            ):
                patches.enter_context(patch.object(executor, name, hook))
            for layer in model.layers:
                index = indices[id(layer)]
                original_residual, original_norm = layer._attention_residual, layer._normalized_input

                def residual(x, _index=index, _original=original_residual, **kwargs):
                    context = current["operation"][2]
                    observed = dict(
                        context,
                        actual_decode=kwargs["decode"],
                        actual_start_pos=kwargs.get("start_pos"),
                        actual_projection_k=kwargs.get("projection_k"),
                        actual_cache_boundary=kwargs.get("cache_boundary"),
                        position_fields=sorted((kwargs.get("positions") or {}).keys()),
                    )
                    hidden("attention_input", _index, x, observed)
                    out = _original(x, **kwargs)
                    hidden("attention_output", _index, out, observed)
                    return out

                def norm(x, key, _index=index, _original=original_norm):
                    out = _original(x, key)
                    operation = current["operation"]
                    if operation is not None:
                        hidden("normalized." + key, _index, out, operation[2])
                    return out

                patches.enter_context(patch.object(layer, "_attention_residual", residual))
                patches.enter_context(patch.object(layer, "_normalized_input", norm))
            fresh = generator.prefill_logits_continuation(tokens, 0, positions=positions, prefill_prefix=True)
            resumed = generator.prefill_logits_continuation(
                tokens, args.start, positions=positions, prefill_prefix=True
            )
        assert result["calls"] == [
            dict(phase="fresh", start=0, length=len(tokens), end=len(tokens)),
            dict(phase="prefix", start=0, length=args.start, end=args.start),
            dict(phase="suffix", start=args.start, length=len(tokens) - args.start, end=len(tokens)),
        ], "Unexpected extra/missing prefill call while observing"
        ttnn.synchronize_device(mesh)
        captures["logits/fresh"] = fresh.detach().cpu().clone()
        captures["logits/resumed"] = resumed.detach().cpu().clone()
        metrics = helpers["compare_logits"](fresh, resumed)
        reasons = helpers["evaluate_case"](metrics, thresholds)
        result["gate"] = dict(metrics=metrics, reasons=reasons, passed=not reasons)
        prior = torch.load(diagnostic["tensor_path"], map_location="cpu", weights_only=True)
        result["instrumentation_vs_original_logits"] = {
            key: stats(captures["logits/" + key], prior[key + "_logits"]) for key in ("fresh", "resumed")
        }
        events = (
            "attention_input",
            "normalized.input_layernorm.weight",
            "attention_output",
            "moe_input",
            "normalized.post_attention_layernorm.weight",
            "moe_output",
        )
        for index in model.layer_indices:
            comparisons = [("fresh", "suffix", target), ("fresh", "prefix", prefix_last)]
            if index == 0:
                comparisons.extend(("fresh", "prefix", position) for position in range(min(64, args.start)))
            for left_phase, right_phase, position in comparisons:
                for event in events:
                    left_key, right_key = (left_phase, index, event, position), (right_phase, index, event, position)
                    if left_key not in lookup or right_key not in lookup:
                        continue
                    lk, li = lookup[left_key]
                    rk, ri = lookup[right_key]
                    result["hidden_comparisons"].append(
                        dict(
                            layer=index,
                            event=event,
                            position=position,
                            left=lk,
                            right=rk,
                            **stats(captures[lk][li], captures[rk][ri]),
                        )
                    )
            for name in ("recurrent", "conv"):
                key = lambda phase, event: f"{phase}/l{index:02}/state.{event}/{name}"
                pairs = [
                    (key("prefix", "last_attention.after"), key("suffix", "public_begin.before_bf16_cast")),
                    (key("suffix", "public_begin.before_bf16_cast"), key("suffix", "public_begin.after_bf16_cast")),
                    (key("fresh", "last_attention.before"), key("prefix", "last_attention.before")),
                    (key("fresh", "last_attention.after"), key("suffix", "last_attention.after")),
                ]
                for left, right in pairs:
                    if left not in captures or right not in captures:
                        continue
                    result["state_comparisons"].append(
                        dict(
                            layer=index,
                            state=name,
                            left=left,
                            right=right,
                            left_boundary=metadata[left]["context"],
                            right_boundary=metadata[right]["context"],
                            ranks=[stats(a, b) for a, b in zip(captures[left], captures[right])],
                        )
                    )
        result["capture_count"] = len(captures)
        result["program_cache_entries_after_checks"] = int(mesh.num_program_cache_entries())
        result["counter_delta"] = {key: generator.counters[key] - value for key, value in initial.items()}
        assert result["counter_delta"]["captures"] == 0
        assert {k: int(v) for k, v in generator.traces.items()} == result["trace_ids"]
        result["status"] = "complete"
    except BaseException:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
        raise
    finally:
        # Flush relevant host observations even on a failure. Never retain
        # device tensors in captures or allocate replacement device state.
        torch.save(captures, args.output_dir / "captures.pt")
        result["captures_artifact"] = str(args.output_dir / "captures.pt")
        result["elapsed_seconds"] = time.time() - result["time_unix"]
        save()
        if generator is not None and not generator.closed:
            generator.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        result["device_closed"] = True
        save()
    print("LOCALIZE_COMPLETE", json.dumps(result["gate"]), str(output), flush=True)


if __name__ == "__main__":
    main()
