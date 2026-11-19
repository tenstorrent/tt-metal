# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Read-only production-op taps for reduced prefill geometry, never acceptance.

After lane-env, restore the accepted mesh descriptor. Serialize hardware use::

 timeout 1800 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.diagnose_resume_chunk_split \
   --capacity 384 --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/resume_chunk_split.json

Uses real layers0,1,3, accepted weights/policies, the failed probe's exact prompt,
common state64, and [64,96) versus [64,68)+[68,96). Runs eagerly to locate the
first numerical producer; no sampler, trace capture, profiler, or precision
replacement. --capacity 262144 is a separate large-cache diagnostic; state reads
are bounded to the consumed region even there. State copies remain on device.
All TTNN/model imports are inside main, so --help and CPU metric checks are safe.
"""

import argparse
import hashlib
import importlib
import json
import os
import time
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


def compare(reference, candidate):
    """Rows are absolute token positions, columns are rank-concatenated values."""
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    a, b = reference.double(), candidate.double()
    delta = b - a
    ac, bc = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    numerator = (ac * bc).sum(-1)
    denominator = ac.square().sum(-1).sqrt() * bc.square().sum(-1).sqrt()
    exact_rows = (a == b).all(-1)
    pcc = torch.where(denominator > 0, numerator / denominator.clamp_min(1e-300), exact_rows.double())
    return {
        "shape_match": True,
        "shape": list(a.shape),
        "exact": bool(exact_rows.all()),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
        "rms": float(delta.square().mean().sqrt()) if delta.numel() else 0.0,
        "pcc_min": float(pcc.min()) if pcc.numel() else 1.0,
        "pcc_mean": float(pcc.mean()) if pcc.numel() else 1.0,
        "first_changed_row": next((i for i, equal in enumerate(exact_rows.tolist()) if not equal), None),
        "rows": [
            {
                "offset": i,
                "pcc": float(pcc[i]),
                "max_abs": float(delta[i].abs().max()),
                "rms": float(delta[i].square().mean().sqrt()),
                "exact": bool(exact_rows[i]),
            }
            for i in range(a.shape[0])
        ],
    }


class CallableTap:
    def __init__(self, target, callback):
        self.target, self.callback = target, callback

    def __getattr__(self, name):
        return getattr(self.target, name)

    def __call__(self, *args, **kwargs):
        result = self.target(*args, **kwargs)
        self.callback(result, args)
        return result


class Collector:
    def __init__(self, tt):
        self.tt, self.values, self.shapes = tt, {}, {}
        self.device_values = {}

    def take(self, label, value, axis=2):
        ranks = []
        for local in self.tt.get_device_tensors(value):
            host = self.tt.to_torch(local).detach().clone()
            self.shapes.setdefault(label, []).append(list(host.shape))
            ranks.append(host.movedim(axis, 0).reshape(host.shape[axis], -1))
        self.values.setdefault(label, []).append(torch.cat(ranks, dim=1))

    def take_expert_rows(self, label, value, ids, count):
        # Nonowned combine lanes are unspecified and never consumed by this
        # rank's reduction. Mask them while retaining token/expert-slot order.
        ranks = []
        for rank, (local, local_ids) in enumerate(
            zip(self.tt.get_device_tensors(value), self.tt.get_device_tensors(ids), strict=True)
        ):
            host = self.tt.to_torch(local).detach().clone().reshape(count, 10, -1)
            expert_ids = self.tt.to_torch(local_ids).long().reshape(count, 10)
            owned = (expert_ids >= rank * 128) & (expert_ids < (rank + 1) * 128)
            host = torch.where(owned.unsqueeze(-1), host, torch.zeros_like(host))
            ranks.append(host.flatten(1))
        self.values.setdefault(label, []).append(torch.cat(ranks, dim=1))

    def joined(self):
        return {label: torch.cat(parts, dim=0) for label, parts in self.values.items()}


@contextmanager
def layer_taps(layer, collector, tt):
    """Wrap call boundaries only; every production op executes unchanged."""
    mesh_module = importlib.import_module(layer.__class__.__module__)
    router_module = importlib.import_module("models.autoports.qwen_qwen3_8_flash_next.tt.router_topk")
    with ExitStack() as stack:

        def method(obj, name, callback):
            original = getattr(obj, name)

            def wrapped(*args, **kwargs):
                result = original(*args, **kwargs)
                callback(result, args, kwargs)
                return result

            stack.enter_context(patch.object(obj, name, wrapped))

        for name in ("attn_mix", "mlp_mix"):
            mixer = getattr(layer, name)
            original_read = mixer.read

            def read_with_taps(*args, _mixer=mixer, _name=name, _read=original_read, **kwargs):
                # CCL and rms helpers are shared by other layer operations.
                # Patch only while this read executes, and preserve every
                # original call/operand/order. Collector snapshots are eager
                # diagnostics; this wrapper is never used for acceptance timing.
                with ExitStack() as read_stack:

                    def boundary(label):
                        def record(result, operands):
                            collector.take(_name + "." + label + ".input", operands[0])
                            collector.take(_name + "." + label + ".output", result)

                        return record

                    read_stack.enter_context(
                        patch.object(mesh_module, "rms", CallableTap(mesh_module.rms, boundary("norm")))
                    )
                    for projection in ("down", "up"):
                        read_stack.enter_context(
                            patch.object(
                                _mixer,
                                projection,
                                CallableTap(getattr(_mixer, projection), boundary(projection)),
                            )
                        )
                    read_stack.enter_context(
                        patch.object(_mixer.ccl, "sum", CallableTap(_mixer.ccl.sum, boundary("ccl_sum")))
                    )
                    result = _read(*args, **kwargs)
                collector.take(_name + ".mixed", result[0])
                collector.take(_name + ".injection", result[1])
                return result

            stack.enter_context(patch.object(mixer, "read", read_with_taps))
            method(mixer, "write", lambda result, args, kw, name=name: collector.take(name + ".output", result))
        stack.enter_context(
            patch.object(
                layer.mlp,
                "router",
                CallableTap(layer.mlp.router, lambda result, args: collector.take("moe.router_logits", result)),
            )
        )
        method(
            router_module,
            "aten_router_topk",
            lambda result, args, kw: (
                collector.take("moe.router_scores", result[0]),
                collector.take("moe.router_ids", result[1]),
            ),
        )

        def reduce_tap(result, args, kw):
            combined, weights, ids, count = args
            collector.take_expert_rows("moe.combined_owned_expert_lanes", combined, ids, count)
            collector.take("moe.reduction_weights", weights, axis=-2)
            collector.take("moe.reduction_ids", ids, axis=-2)
            collector.take("moe.routed_reduced", result)

        method(layer.mlp, "_reduce", reduce_tap)
        method(layer.mlp, "shared_forward", lambda result, args, kw: collector.take("moe.shared_output", result))

        def moe_output(result, args):
            # Keep the unchanged device input for a separate local MoE control.
            collector.device_values.setdefault("moe.input", []).append(args[0])
            collector.take("moe.output", result)

        stack.enter_context(patch.object(layer, "mlp", CallableTap(layer.mlp, moe_output)))
        if layer.has_ple:
            stack.enter_context(
                patch.object(
                    layer,
                    "embedding",
                    CallableTap(
                        layer.embedding,
                        lambda result, args: (
                            collector.take("ple.hash_ids", args[0]),
                            collector.take("ple.embedding", result),
                        ),
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    layer, "ple", CallableTap(layer.ple, lambda result, args: collector.take("ple.output", result))
                )
            )
        if layer.chunked_attn is None:
            gdn = layer.attn

            def projection(result, args, kw):
                if args[1] is gdn.qkv:
                    collector.take("gdn.qkv", result)
                elif args[1] is gdn.zab:
                    collector.take("gdn.zab", result)

            method(mesh_module, "linear", projection)
            method(mesh_module, "mesh_exact_conv_silu", lambda result, args, kw: collector.take("gdn.conv", result))

            def recurrence(result, args, kw):
                for label, value in zip(("q", "k", "v", "decay", "beta"), args):
                    collector.take("gdn.recurrence_input." + label, value, axis=1)
                collector.take("gdn.recurrence_output", result[0], axis=1)

            method(mesh_module, "mesh_recurrent_gated_delta_rule", recurrence)
            stack.enter_context(
                patch.object(layer, "attn", CallableTap(gdn, lambda result, args: collector.take("gdn.output", result)))
            )
        else:
            qsa = layer.chunked_attn
            method(
                qsa.attention,
                "linear",
                lambda result, args, kw: (
                    collector.take("qsa.packed_qkv", result) if args[1] is qsa.packed_qkv else None
                ),
            )
            method(
                qsa,
                "_selected_ids",
                lambda result, args, kw: (
                    collector.take("qsa.index_query", args[0]),
                    collector.take("qsa.positions", kw["positions"]),
                    collector.take("qsa.selected_ids", result, axis=0),
                ),
            )
            method(
                tt.transformer,
                "scaled_dot_product_attention",
                lambda result, args, kw: (
                    collector.take("qsa.sdpa_query", args[0], axis=0),
                    collector.take("qsa.sdpa_output", result, axis=0),
                ),
            )
            method(qsa, "forward_chunk", lambda result, args, kw: collector.take("qsa.output", result))
        yield


def compare_collectors(full, split):
    a, b = full.joined(), split.joined()
    result = {"stage_order": list(a), "missing_stages": sorted(set(a) ^ set(b)), "stages": {}}
    for name in a.keys() & b.keys():
        result["stages"][name] = compare(a[name], b[name])
        if name == "qsa.selected_ids":
            result["stages"][name]["sorted_membership"] = compare(
                a[name].reshape(a[name].shape[0], 4, -1).sort(-1).values.flatten(1),
                b[name].reshape(b[name].shape[0], 4, -1).sort(-1).values.flatten(1),
            )
        if name == "output":
            result["stages"][name]["prefix64_68"] = compare(a[name][:4], b[name][:4])
            result["stages"][name]["continuation68_96"] = compare(a[name][4:], b[name][4:])
    result["first_nonexact_stage"] = next(
        (name for name in result["stage_order"] if not result["stages"].get(name, {}).get("exact", False)), None
    )
    result["first_nonexact_note"] = "Observed tap order, not a proof that an untapped intermediate is exact"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=384)
    parser.add_argument("--prompt-artifact", type=Path, default=ROOT / "doc/full_model/reduced_resume_aligned.json")
    parser.add_argument("--checkpoint", default=os.environ.get("QWEN_CHECKPOINT"))
    parser.add_argument("--packed-cache", default=os.environ.get("QWEN_PLE_CACHE"))
    parser.add_argument(
        "--repeat-token-rows",
        action="store_true",
        help="Also compare identical mixed-token inputs across packed expert rows and chunk geometries",
    )
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / "doc/full_model"):
        parser.error("Use a diagnostic artifact under doc/full_model")
    if args.capacity < 128:
        parser.error("Capacity must cover token96 and use the supported QSA allocation")
    prompt = json.loads(args.prompt_artifact.read_text())["prompt_token_ids"]
    assert len(prompt) >= 96
    report = {
        "completed": False,
        "scope": "eager reduced same-state same-input numerical diagnostic; not readiness",
        "layers": [0, 1, 3],
        "capacity": args.capacity,
        "repeat_token_rows": args.repeat_token_rows,
        "prompt_artifact": str(args.prompt_artifact),
        "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
        "intervals": {"prefix": [[0, 32], [32, 64]], "full": [[64, 96]], "split": [[64, 68], [68, 96]]},
        "source_sha256": {
            name: hashlib.sha256((ROOT / "tt" / name).read_bytes()).hexdigest()
            for name in ("model.py", "generator.py", "multichip_decoder.py", "fused_decoder.py")
        },
        "native_moe_compute_sha256": hashlib.sha256(
            (
                ROOT.parents[2]
                / "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/kernels/moe_fused_swiglu_compute.cpp"
            ).read_bytes()
        ).hexdigest(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "TT_MESH_GRAPH_DESC_PATH",
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_DEVICE_PROFILER",
                "TT_METAL_WATCHER",
            )
        },
    }
    import ttnn as tt
    from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator
    from models.autoports.qwen_qwen3_8_flash_next.tt.slot_state import _state_tensors

    mesh = gen = None
    started = time.monotonic()

    def progress(phase):
        print("RESUME_SPLIT_PHASE", phase, flush=True)
        report["last_phase"] = phase
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        assert os.environ.get("TT_MESH_GRAPH_DESC_PATH"), "Restore the accepted mesh descriptor after lane-env"
        assert not os.environ.get("TT_METAL_DEVICE_PROFILER"), "This is a bounded functional diagnostic, not profiling"
        torch.set_num_threads(8)
        tt.set_fabric_config(tt.FabricConfig.FABRIC_1D_RING)
        mesh = tt.open_mesh_device(tt.MeshShape(1, 4), trace_region_size=256 * 1024 * 1024, l1_small_size=32768)
        options = dict(layer_indices=(0, 1, 3), max_seq_len=args.capacity, max_batch_size=1)
        if args.checkpoint:
            options["checkpoint"] = args.checkpoint
        if args.packed_cache:
            options["packed_cache"] = args.packed_cache
        progress("load")
        gen = build_generator(ROOT, mesh, **options)
        model, seed = gen.model, gen.caches[0]
        full, split = model.create_cache(capacity=args.capacity), model.create_cache(capacity=args.capacity)

        def inputs(start, end):
            ids = torch.tensor([prompt[start:end]], dtype=torch.int32)
            words = torch.zeros(1, end - start, 16, dtype=torch.int32)
            words[..., 0], words[..., 1] = ids, 1
            return gen._upload(ids), gen._upload(words)

        supplied = {interval: inputs(*interval) for interval in ((0, 32), (32, 64), (64, 96), (64, 68), (68, 96))}

        def copy_cache(source, destination):
            for (name, src), (other_name, dst) in zip(_state_tensors(source), _state_tensors(destination), strict=True):
                assert name == other_name and src.shape == dst.shape and src.dtype == dst.dtype
                tt.copy(src, dst)
            destination.position = source.position

        def state_host(cache, layer_offset):
            values = {}
            prefix = f"layers.{layer_offset}."
            for name, tensor in _state_tensors(cache):
                if not name.startswith(prefix):
                    continue
                short = name.removeprefix(prefix)
                # Only consumed KV/groups matter; avoid reading the configured
                # 262144-token pool in the optional full-capacity diagnostic.
                if short in ("attention.k_cache", "attention.v_cache"):
                    tensor = tensor[:3, ...]
                elif short == "attention.compressed_keys":
                    tensor = tensor[:, :, :24, :]
                elif short == "attention.page_table":
                    tensor = tensor[:, :3]
                values[short] = torch.cat(
                    [tt.to_torch(local).flatten().clone() for local in tt.get_device_tensors(tensor)]
                ).reshape(1, -1)
            return values

        def compare_states(a, b):
            return {name: compare(a[name], b[name]) for name in a}

        progress("prefix64")
        for interval in ((0, 32), (32, 64)):
            model.forward_chunk(*supplied[interval], cache=seed)
        seed.position = 64
        copy_cache(seed, full)
        copy_cache(seed, split)
        report["initial_state64_copy_checks"] = {
            str(layer): compare_states(state_host(seed, offset), state_host(split, offset))
            for offset, layer in enumerate(model.layer_indices)
        }
        assert all(v["exact"] for item in report["initial_state64_copy_checks"].values() for v in item.values())
        residual = model.embed(supplied[(64, 96)][0])
        full_collectors, full_states, full_inputs = [], [], []
        report["local_same_input_layers"] = {}
        # Full path supplies the identical immutable residual to each local
        # split control, preventing inherited upstream drift from being blamed
        # on the next layer's arithmetic.
        for offset, (index, layer) in enumerate(zip(model.layer_indices, model.layers, strict=True)):
            progress(f"local_layer_{index}")
            full_inputs.append(residual)
            a, b = Collector(tt), Collector(tt)
            a.take("input", residual)
            with layer_taps(layer, a, tt):
                output = layer.prefill_forward(
                    residual, state=full.layers[offset], token_ids=supplied[(64, 96)][1] if layer.has_ple else None
                )
            a.take("output", output)
            full_states.append(state_host(full, offset))
            split_at68 = None
            for begin, end in ((0, 4), (4, 32)):
                piece = residual[:, :, begin:end, :]
                b.take("input", piece)
                with layer_taps(layer, b, tt):
                    candidate = layer.prefill_forward(
                        piece,
                        state=split.layers[offset],
                        token_ids=supplied[(64, 68) if begin == 0 else (68, 96)][1] if layer.has_ple else None,
                    )
                b.take("output", candidate)
                if begin == 0:
                    split_at68 = state_host(split, offset)
            item = compare_collectors(a, b)
            item["state96"] = compare_states(full_states[-1], state_host(split, offset))
            item["state68_shapes"] = {name: list(value.shape) for name, value in split_at68.items()}
            item["kind"], item["has_ple"] = layer.kind, layer.has_ple
            report["local_same_input_layers"][str(index)] = item
            assert item["stages"]["input"]["exact"], "Layer-local inputs must match exactly"
            full_collectors.append(a)
            residual = output
        full_residual = residual
        report["isolated_same_mixed_input_moe"] = {}
        for index, layer, baseline in zip(model.layer_indices, model.layers, full_collectors, strict=True):
            progress(f"isolated_moe_{index}")
            mixed = baseline.device_values["moe.input"][0]
            a, b = Collector(tt), Collector(tt)
            a.take("input", mixed)
            with layer_taps(layer, a, tt):
                result = layer.mlp(mixed)
            a.take("output", result)
            for begin, end in ((0, 4), (4, 32)):
                piece = mixed[:, :, begin:end, :]
                b.take("input", piece)
                with layer_taps(layer, b, tt):
                    result = layer.mlp(piece)
                b.take("output", result)
            comparison = compare_collectors(a, b)
            assert comparison["stages"]["input"]["exact"], "Isolated MoE inputs must match exactly"
            report["isolated_same_mixed_input_moe"][str(index)] = comparison
        if args.repeat_token_rows:
            report["repeated_token_moe"] = {}
            for index, layer, baseline in zip(model.layer_indices, model.layers, full_collectors, strict=True):
                progress(f"repeated_token_moe_{index}")
                # Identical tokens choose identical ten experts. Thus each
                # owned expert gets the same input in every row of one tile;
                # comparison within a launch isolates packed row/face effects.
                mixed = tt.repeat(baseline.device_values["moe.input"][0][:, :, :1, :], (1, 1, 32, 1))
                captures = []
                for intervals in (((0, 32),), ((0, 32),), ((0, 4), (4, 32))):
                    recorder = Collector(tt)
                    for begin, end in intervals:
                        piece = mixed[:, :, begin:end, :]
                        recorder.take("input", piece)
                        with layer_taps(layer, recorder, tt):
                            result = layer.mlp(piece)
                        recorder.take("output", result)
                    captures.append(recorder)
                a, again, split_repeated = captures
                values = a.joined()
                assert torch.equal(values["input"], values["input"][:1].expand_as(values["input"]))
                ids = values["moe.router_ids"]
                assert torch.equal(ids, ids[:1].expand_as(ids)), "Repeated inputs must select identical expert IDs"
                report["repeated_token_moe"][str(index)] = {
                    "same_launch_row0_vs_rows": {
                        name: compare(value[:1].expand_as(value), value) for name, value in values.items()
                    },
                    "repeat_launch": compare_collectors(a, again),
                    "full_vs_split": compare_collectors(a, split_repeated),
                    "interpretation": (
                        "Owned combined lanes varying within one launch despite identical inputs and routing "
                        "demonstrate row-position sensitivity inside dispatch/expert/combine. Exact repeat "
                        "launches separate deterministic lane mapping from launch nondeterminism. This uses "
                        "the real selected ten experts, not an artificial single-expert kernel."
                    ),
                }
        progress("propagated_split_stack")
        copy_cache(seed, split)
        propagated = [Collector(tt) for _ in model.layers]
        split_outputs = []
        for interval in ((64, 68), (68, 96)):
            residual = model.embed(supplied[interval][0])
            for offset, layer in enumerate(model.layers):
                recorder = propagated[offset]
                recorder.take("input", residual)
                with layer_taps(layer, recorder, tt):
                    residual = layer.prefill_forward(
                        residual, state=split.layers[offset], token_ids=supplied[interval][1] if layer.has_ple else None
                    )
                recorder.take("output", residual)
            split_outputs.append(residual)
        report["propagated_layers"] = {
            str(index): {
                **compare_collectors(a, b),
                "state96": compare_states(full_states[offset], state_host(split, offset)),
            }
            for offset, (index, a, b) in enumerate(zip(model.layer_indices, full_collectors, propagated, strict=True))
        }
        progress("terminal_same_input_and_propagated")
        terminal_full, terminal_fixed, terminal_propagated = Collector(tt), Collector(tt), Collector(tt)

        def terminal(residual, collector):
            with patch.object(
                model,
                "final_mixer",
                CallableTap(model.final_mixer, lambda result, values: collector.take("final_mixer", result)),
            ):
                logits = model.logits(residual)
            collector.take("logits", logits)

        terminal(full_residual, terminal_full)
        for begin, end in ((0, 4), (4, 32)):
            terminal(full_residual[:, :, begin:end, :], terminal_fixed)
        for residual in split_outputs:
            terminal(residual, terminal_propagated)
        report["terminal_fixed_residual"] = compare_collectors(terminal_full, terminal_fixed)
        report["terminal_propagated"] = compare_collectors(terminal_full, terminal_propagated)
        full_logits = terminal_full.joined()["logits"]
        for name, collector in (
            ("terminal_fixed_residual", terminal_fixed),
            ("terminal_propagated", terminal_propagated),
        ):
            logits = collector.joined()["logits"]
            report[name]["top1_agreement"] = float((full_logits.argmax(-1) == logits.argmax(-1)).float().mean())
            report[name]["top1_by_position"] = [
                {"position": 64 + i, "full": int(a), "split": int(b), "equal": int(a) == int(b)}
                for i, (a, b) in enumerate(zip(full_logits.argmax(-1), logits.argmax(-1), strict=True))
            ]
        report["limitations"] = [
            "Eager instrumentation does not certify trace behavior or timing.",
            "Reduced stack distributions differ from all48 layers; these are producer diagnostics, not a threshold waiver.",
            "State68 is available only on the split path; state96 comparisons and identical-input local controls isolate causal drift.",
            "A changed ID order with matching sorted membership is finite-order attention evidence, not omitted causal keys.",
        ]
        report["completed"] = True
        progress("complete")
    except BaseException as exc:
        report["error"], report["traceback"] = f"{type(exc).__name__}: {exc}", traceback.format_exc()
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if gen is not None:
            gen.close()
        if mesh is not None:
            tt.close_mesh_device(mesh)
        print(json.dumps({"completed": report["completed"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
