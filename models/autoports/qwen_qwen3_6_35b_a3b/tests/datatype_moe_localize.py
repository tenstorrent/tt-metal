# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only, same-operand indexed TP4 / grouped EP4 MoE comparison.

Run serially under an external timeout after datatype_resume_localize. Captures
are already normalized; no norm, shared expert, residual or collective belongs
in this contrast. Rank-local TP4 and EP4 contributions have different meanings:
only their CPU FP64 sums are compared. Host readbacks invalidate performance.
"""

import argparse
import hashlib
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
EVENT = "normalized.post_attention_layernorm.weight"


def load_inputs(directory, layer, phase, positions):
    import torch

    metadata = json.loads((directory / "capture_metadata.json").read_text())
    capture = torch.load(directory / "captures.pt", map_location="cpu", weights_only=True)
    selected, provenance = [], []
    for position in positions:
        matches = [
            (key, info)
            for key, info in metadata.items()
            if info.get("event") == EVENT
            and info.get("layer") == layer
            and info.get("phase") == phase
            and position in info.get("positions", [])
        ]
        assert len(matches) == 1, (position, [key for key, _ in matches])
        key, info = matches[0]
        index = info["positions"].index(position)
        value = capture[key][index].detach().clone()
        assert tuple(value.shape) == (2048,) and value.dtype == torch.bfloat16
        assert torch.isfinite(value).all()
        selected.append(value)
        provenance.append(dict(position=position, key=key, row=index, metadata=info))
    return torch.stack(selected), provenance


def logical_host_rows(value, rows, width):
    """Select valid rows from an explicit padded-row carrier, preserving callers' raw copy."""
    import torch

    assert value.shape[-1] == width and value.numel() % width == 0, (value.shape, width)
    carrier = value.reshape(-1, width)
    assert carrier.shape[0] >= rows, (value.shape, rows)
    logical = carrier[:rows].clone()
    assert torch.isfinite(logical.float()).all()
    return logical


def metrics(left, right):
    import torch

    a, b = left.double().reshape(-1), right.double().reshape(-1)
    assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    delta = a - b
    ac, bc = a - a.mean(), b - b.mean()
    denominator = ac.norm() * bc.norm()
    return dict(
        bit_equal=torch.equal(left, right),
        pcc=float((ac @ bc) / denominator) if denominator else float(torch.equal(left, right)),
        max_abs=float(delta.abs().max()),
        rms_difference=float(delta.square().mean().sqrt()),
        relative_l2=float(delta.norm() / b.norm()) if b.norm() else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, default=ROOT / "doc/datatype_sweep/resume_localize/L2560_s2559")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--phase", choices=("fresh", "prefix", "suffix"), default="suffix")
    parser.add_argument("--positions", type=int, nargs="+", default=[2559])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--validate-inputs-only", action="store_true", help="CPU only; do not import TTNN or open devices"
    )
    parser.add_argument("--skip-cpu-reference", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    assert 0 <= args.layer < 40 and 1 <= len(args.positions) <= 32
    assert len(set(args.positions)) == len(args.positions) and args.repeats >= 1
    import torch

    torch.set_num_threads(8)
    host_x, input_provenance = load_inputs(args.capture_dir, args.layer, args.phase, args.positions)
    capture_result = json.loads((args.capture_dir / "statistics.json").read_text())
    assert capture_result["precision_config"] == "baseline"
    assert capture_result["runtime_precision"]["selected_policy"]["config_id"] == "baseline_bfp8"
    source_changes = []
    for filename, expected in capture_result["source_sha256"].items():
        if "/tt/" in filename and filename.endswith(".py"):
            source = Path(filename)
            if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                source_changes.append(filename)
    assert not source_changes, f"Captured runtime sources changed: {source_changes}"
    if args.validate_inputs_only:
        print(json.dumps(dict(shape=list(host_x.shape), event=EVENT, input_provenance=input_provenance), indent=2))
        return
    assert args.output_dir is not None, "--output-dir is required for a device diagnostic"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "result.json"
    operands_path = args.output_dir / "operands.pt"
    assert not output.exists() and not operands_path.exists(), "Use a new diagnostic output directory"
    assert not os.environ.get("TT_METAL_WATCHER")
    assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "1"
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")

    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.fused_decoder import reshape
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import HF_REVISION, Checkpoint
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import CollectiveWorkspace, MultichipDecoder
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.precision import baseline_policy, layer_precision

    result = dict(
        status="running",
        command=sys.argv,
        time_unix=time.time(),
        layer=args.layer,
        phase=args.phase,
        positions=args.positions,
        input_event=EVENT,
        input_provenance=input_provenance,
        capture_dir=str(args.capture_dir),
        hardware="4 Blackhole p300c",
        mesh=[1, 4],
        precision_config=baseline_policy(),
        checkpoint_revision=HF_REVISION,
        scope="Diagnostic eager expert arithmetic only; CPU readbacks; no performance or full-model gate claim",
        comparison="Four rank partials summed in CPU FP64; TP4 and EP4 rank partials are NOT compared per rank",
        routing_policy="Real rows use frozen native singleton IDs and BF16 scores; padding retains native zero-input routing",
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), *sorted((ROOT / "tt").glob("*.py"))]
        },
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        cases=[],
        comparisons=[],
        limitations=[
            "Captured normalized input is rank0 of the full model's replicated boundary; this probe replicates those exact BF16 bytes.",
            "Zero-padding has real routing IDs and changes expert counts; it is intentionally not masked out.",
            "Checkpoint FP64 reference excludes BFP8 weight quantization, internal activation quantization, BF16 boundaries and device accumulation order.",
            "The two padding shapes are diagnostic expert launches, not a replay of the original 2560-token full-model schedule.",
        ],
    )
    operands = dict(input=host_x, input_provenance=input_provenance, cases={})
    mesh = None

    def save():
        torch.save(operands, operands_path)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    try:
        checkpoint = Checkpoint()
        assert checkpoint.path.name == HF_REVISION
        result["checkpoint_path"] = str(checkpoint.path)
        result["checkpoint_metadata_sha256"] = {
            name: hashlib.sha256((checkpoint.path / name).read_bytes()).hexdigest()
            for name in ("config.json", "model.safetensors.index.json")
        }
        state_dict = checkpoint.select(f"model.language_model.layers.{args.layer}.")
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
        decoder = MultichipDecoder.from_state_dict(
            state_dict,
            hf_config=checkpoint.config,
            layer_idx=args.layer,
            mesh_device=mesh,
            workspace=CollectiveWorkspace(mesh),
            full_model_optimized=True,
            precision_config=layer_precision(baseline_policy(), args.layer),
        )
        result["runtime_precision"] = decoder.precision_summary()
        assert decoder.residual_layout == "replicated"
        for rows in (32, 512):
            decoder.prepare_moe(rows)

        def upload(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                value.contiguous(),
                device=mesh,
                dtype=dtype,
                layout=layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        def parts(tensor, *, require_finite=True):
            values = [ttnn.to_torch(part).detach().clone() for part in ttnn.get_device_tensors(tensor)]
            assert len(values) == 4
            if require_finite:
                assert all(torch.isfinite(value.float()).all() for value in values)
            return values

        def tensor_metadata(tensor):
            return dict(
                shape=list(tensor.shape),
                padded_shape=list(tensor.padded_shape),
                dtype=str(tensor.dtype),
                layout=str(tensor.layout),
                memory_config=str(tensor.memory_config()),
            )

        def replica(tensor):
            values = parts(tensor)
            assert all(torch.equal(values[0], value) for value in values[1:]), "Routing replicas disagree"
            return values[0]

        def route(x):
            rows = x.shape[-2]
            logits = reshape(
                decoder.linear(x, "mlp.gate.weight", ttnn.float32, ordered_router=False, terminal_scan=False),
                (1, 1, rows, 256),
            )
            values, ids = (
                decoder.mesh_fp32_selector(logits)
                if rows == 1 and decoder.decode_fp32_selector
                else ttnn.topk(logits, k=8, dim=-1, sorted=True)
            )
            scores = ttnn.typecast(
                ttnn.softmax(values, dim=-1, numeric_stable=True, compute_kernel_config=decoder.compute), ttnn.bfloat16
            )
            return logits, values, ids, scores

        real_routes, singleton_partials = [], []
        original_linear = decoder.linear
        original_reduce = ttnn.experimental.fast_reduce_nc
        for index, row in enumerate(host_x):
            x = upload(row.reshape(1, 1, 2048))
            logits, values, ids, scores = route(x)
            real_routes.append(
                dict(
                    logits=replica(logits),
                    ids=replica(ids),
                    scores=replica(scores),
                    tensor_metadata={
                        name: tensor_metadata(value)
                        for name, value in (("logits", logits), ("ids", ids), ("scores", scores))
                    },
                )
            )
            case = dict(routing=real_routes[-1])
            operands["cases"][f"indexed_row{index}"] = case
            captured = {}

            def linear(value, key, *positional, **kwargs):
                return logits if key == "mlp.gate.weight" else original_linear(value, key, *positional, **kwargs)

            def reduce(value, *positional, **kwargs):
                out = original_reduce(value, *positional, **kwargs)
                if kwargs.get("dims") == [1]:
                    assert "routed" not in captured
                    captured["weighted_experts"] = value
                    captured["routed"] = out
                return out

            def frozen_softmax(value, *positional, **kwargs):
                # Only the router calls softmax inside this diagnostic _moe invocation.
                assert tuple(value.shape) == tuple(values.shape)
                return scores

            with ExitStack() as patches:
                patches.enter_context(patch.object(decoder, "linear", linear))
                if decoder.decode_fp32_selector:
                    patches.enter_context(patch.object(decoder, "mesh_fp32_selector", lambda _: (values, ids)))
                else:
                    patches.enter_context(patch.object(ttnn, "topk", lambda *_, **__: (values, ids)))
                patches.enter_context(patch.object(ttnn, "softmax", frozen_softmax))
                patches.enter_context(patch.object(ttnn.experimental, "fast_reduce_nc", reduce))
                # Keep the original indexed expert code. Its final add is irrelevant;
                # capture routed before shared addition, and exclude shared projection work.
                patches.enter_context(patch.object(decoder, "_shared", lambda value, **_: ttnn.zeros_like(value)))
                ignored = decoder._moe(x, ordered_router=False, batched=False)
            assert set(captured) == {"weighted_experts", "routed"}
            # fast_reduce_nc returns an explicit physical32-row carrier here.
            # Production immediately declares logical1/padded32 with reshape;
            # retain the carrier for evidence and select exactly its first row.
            physical = parts(captured["routed"], require_finite=False)
            case.update(
                routed_physical_rank_outputs=physical,
                routed_tensor_metadata=tensor_metadata(captured["routed"]),
                valid_row_selection="reshape(-1,2048)[:1]; matches production logical1/padded32 reshape",
            )
            local = [logical_host_rows(value, 1, 2048) for value in physical]
            singleton_partials.append(local)
            case.update(
                rank_partials=local,
                weighted_experts=parts(captured["weighted_experts"]),
                weighted_experts_tensor_metadata=tensor_metadata(captured["weighted_experts"]),
            )
            del ignored, captured, x, logits, values, ids, scores
        indexed_parts = [torch.cat([row[rank] for row in singleton_partials]) for rank in range(4)]
        indexed_sum = torch.stack([value.double() for value in indexed_parts]).sum(0)
        operands["indexed_rank_partials"] = indexed_parts
        operands["indexed_rank_sum_fp64"] = indexed_sum
        real_ids = torch.cat([logical_host_rows(row["ids"], 1, 8) for row in real_routes]).long()
        real_scores = torch.cat([logical_host_rows(row["scores"], 1, 8) for row in real_routes])
        operands["real_ids"], operands["real_scores"] = real_ids, real_scores
        assert all(len(set(ids.tolist())) == 8 for ids in real_ids)

        if not args.skip_cpu_reference:
            gate, up = state_dict["mlp.experts.gate_up_proj"].chunk(2, dim=1)
            down = state_dict["mlp.experts.down_proj"]
            reference = torch.zeros_like(host_x, dtype=torch.float64)
            with torch.no_grad():
                for expert in sorted(set(real_ids.flatten().tolist())):
                    rows, lanes = torch.where(real_ids == expert)
                    x = host_x[rows].double()
                    gu = torch.nn.functional.silu(x @ gate[expert].double().T) * (x @ up[expert].double().T)
                    values = (gu @ down[expert].double().T) * real_scores[rows, lanes].double().unsqueeze(-1)
                    reference.index_add_(0, rows, values)
            operands["cpu_checkpoint_fp64_reference"] = reference
            result["indexed_vs_checkpoint_fp64"] = metrics(indexed_sum, reference)

        deepseek = ttnn.experimental.deepseek_prefill
        original_dispatch, original_fused = deepseek.dispatch, deepseek.moe_fused_swiglu
        summaries = {}
        for rows in (32, 512):
            padded_x = torch.zeros((1, rows, 2048), dtype=torch.bfloat16)
            padded_x[0, : len(host_x)] = host_x
            x = upload(padded_x)
            logits, values, native_ids, native_scores = route(x)
            host_ids = replica(native_ids).reshape(1, 1, rows, 8).long()
            host_scores = replica(native_scores).reshape(1, 1, rows, 8)
            operands[f"native_routing_{rows}"] = dict(
                ids=host_ids.clone(), scores=host_scores.clone(), logits=replica(logits)
            )
            host_ids[0, 0, : len(host_x)] = real_ids
            host_scores[0, 0, : len(host_x)] = real_scores
            ids = upload(host_ids, ttnn.uint32)
            scores = upload(host_scores)
            operands[f"grouped_inputs_{rows}"] = dict(x=padded_x, ids=host_ids, scores=host_scores)
            del logits, values, native_ids, native_scores
            for sr in (True, False):
                for repeat in range(args.repeats):
                    label = f"grouped_n{rows}_sr{int(sr)}_repeat{repeat}"
                    observed = {}

                    def dispatch(*positional, **kwargs):
                        observed["dispatch_ids"] = parts(positional[1])
                        observed["dispatch_offsets"] = parts(positional[2])
                        out = original_dispatch(*positional, **kwargs)
                        observed["dispatch_metadata"] = parts(out[1])
                        return out

                    def fused(*positional, **kwargs):
                        observed["expert_counts"] = parts(positional[4])
                        observed["expert_offsets"] = parts(kwargs["expert_region_offsets"])
                        observed["native_stochastic_rounding_argument"] = kwargs.get("stochastic_rounding")
                        return original_fused(*positional, **kwargs)

                    with (
                        patch.object(deepseek, "dispatch", dispatch),
                        patch.object(deepseek, "moe_fused_swiglu", fused),
                    ):
                        out = decoder._grouped_moe(x, ids, scores, stochastic_rounding=sr)
                    local = [value.reshape(rows, 2048) for value in parts(out)]
                    total = torch.stack([value[: len(host_x)].double() for value in local]).sum(0)
                    summaries[label] = total
                    observed.update(rank_partials=local, real_rank_sum_fp64=total)
                    operands["cases"][label] = observed
                    counts_expected = torch.bincount(host_ids.flatten(), minlength=256)
                    for rank, actual in enumerate(observed["expert_counts"]):
                        expected = counts_expected.clone()
                        if decoder.local_expert_dispatch:
                            mask = torch.zeros(256, dtype=torch.bool)
                            mask[rank * 64 : (rank + 1) * 64] = True
                            expected[~mask] = 0
                        assert torch.equal(
                            actual.reshape(-1).long(), expected
                        ), "Native dispatch counts differ from frozen IDs"
                    assert observed["native_stochastic_rounding_argument"] is sr
                    record = dict(
                        case=label,
                        rows=rows,
                        real_rows=len(host_x),
                        stochastic_rounding=sr,
                        repeat=repeat,
                        vs_indexed_rank_sum=metrics(total, indexed_sum),
                        per_row_vs_indexed=[
                            dict(position=position, **metrics(total[index], indexed_sum[index]))
                            for index, position in enumerate(args.positions)
                        ],
                        padding_output_max_abs=max(
                            float(value[len(host_x) :].float().abs().max()) if rows > len(host_x) else 0.0
                            for value in local
                        ),
                    )
                    if "cpu_checkpoint_fp64_reference" in operands:
                        record["vs_checkpoint_fp64"] = metrics(total, operands["cpu_checkpoint_fp64_reference"])
                    result["cases"].append(record)
                    print("MOE_CASE", json.dumps(record), flush=True)
                    save()
                    del out
            del x, ids, scores
        for rows in (32, 512):
            for sr in (True, False):
                base = f"grouped_n{rows}_sr{int(sr)}_repeat0"
                for repeat in range(1, args.repeats):
                    other = f"grouped_n{rows}_sr{int(sr)}_repeat{repeat}"
                    result["comparisons"].append(
                        dict(left=base, right=other, contrast="repeat", **metrics(summaries[base], summaries[other]))
                    )
            left, right = f"grouped_n{rows}_sr1_repeat0", f"grouped_n{rows}_sr0_repeat0"
            result["comparisons"].append(
                dict(left=left, right=right, contrast="SR_only", **metrics(summaries[left], summaries[right]))
            )
        for sr in (True, False):
            left, right = f"grouped_n32_sr{int(sr)}_repeat0", f"grouped_n512_sr{int(sr)}_repeat0"
            result["comparisons"].append(
                dict(
                    left=left,
                    right=right,
                    contrast="trailing_zero_padding",
                    **metrics(summaries[left], summaries[right]),
                )
            )
        result["status"] = "complete"
    except BaseException:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
        raise
    finally:
        result["elapsed_seconds"] = time.time() - result["time_unix"]
        save()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            result["device_closed"] = True
            save()
    print("MOE_LOCALIZE_COMPLETE", str(output), flush=True)


if __name__ == "__main__":
    main()
