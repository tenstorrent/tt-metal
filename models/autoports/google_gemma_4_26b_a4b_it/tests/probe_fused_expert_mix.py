# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare live expert multiply/reduction with native matmul on captured inputs."""

import argparse
import json
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder import REVISION, load_layer
from models.autoports.google_gemma_4_26b_a4b_it.tests.runtime_audit import device_only
from models.autoports.google_gemma_4_26b_a4b_it.tt.fused_decoder import FusedDecoder, PackedExperts
from models.common.utility_functions import comp_pcc


class CaptureMix(PackedExperts):
    """Run the live sparse projections and retain their actual mixing boundary."""

    def __init__(self, source):
        self.__dict__.update(source.__dict__)
        # Preserve the multiply/reduce control after the faster decode suffix
        # becomes the selected default; sparse projections and inputs stay live.
        self.matmul_mix = False
        self.records = []

    def _chunk(self, x, routing, decode):
        sparse_matmul = ttnn.sparse_matmul
        captured_down = []

        def capture_down(*args, **kwargs):
            output = sparse_matmul(*args, **kwargs)
            if args[1] is self.down:
                captured_down.append(ttnn.clone(output, memory_config=output.memory_config()))
            return output

        with patch.object(ttnn, "sparse_matmul", capture_down):
            output = super()._chunk(x, routing, decode)
        assert len(captured_down) == 1, "Expected exactly one sparse down projection per expert chunk"
        self.records.append(
            dict(
                decode=decode,
                length=x.shape[-2],
                down=captured_down[0],
                routing=ttnn.clone(routing, memory_config=routing.memory_config()),
                mixed=ttnn.clone(output, memory_config=output.memory_config()),
            )
        )
        return output


def existing_mix(down, routing, *, decode, length, experts, hidden_size):
    """Mirror the live mixing suffix; its output is checked against the capture."""
    if decode:
        states = ttnn.reshape(ttnn.permute(down, (0, 2, 1, 3)), (1, experts, hidden_size))
        states = ttnn.mul(states, ttnn.reshape(routing, (1, experts, 1)))
        states = ttnn.unsqueeze_to_4D(ttnn.sum(states, dim=1))
        return ttnn.reshape(states, (1, 1, 1, hidden_size), (1, 1, 32, hidden_size))
    down = ttnn.reshape(down, (1, experts, length, hidden_size))
    weighted = ttnn.mul(down, ttnn.permute(routing, (0, 3, 2, 1)))
    return ttnn.reshape(ttnn.experimental.fast_reduce_nc(weighted, dims=[1]), (1, 1, length, hidden_size))


def matmul_mix(down, routing, *, decode, length, experts, hidden_size, compute, output_memory):
    # Both paths start from the captured sparse output, so these movements are
    # included in the measured graph. Permute preserves tile padding when S
    # moves from the matrix height into a batch dimension during prefill.
    down = ttnn.reshape(down, (1, experts, length, hidden_size))
    down = ttnn.permute(down, (0, 2, 1, 3))
    routes = routing if decode else ttnn.permute(routing, (0, 2, 1, 3))
    mixed = ttnn.matmul(
        routes,
        down,
        dtype=ttnn.bfloat16,
        memory_config=output_memory,
        compute_kernel_config=compute,
    )
    return mixed if decode else ttnn.permute(mixed, (0, 2, 1, 3), memory_config=output_memory)


def measure(mesh, fn, args):
    with device_only():
        output = fn()
    actual = ttnn.to_torch(output).float()
    output.deallocate(True)
    ttnn.synchronize_device(mesh)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    with device_only():
        output = fn()
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    try:
        for _ in range(args.warmup_replays):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        samples = []
        for _ in range(args.timing_samples):
            start = time.perf_counter_ns()
            for _ in range(args.repeats):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            samples.append((time.perf_counter_ns() - start) / (args.repeats * 1000))
        replay_equal = torch.equal(actual, ttnn.to_torch(output).float())
    finally:
        ttnn.release_trace(mesh, trace)
        output.deallocate(True)
    return actual, replay_equal, samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path(__file__).parents[1] / "doc/functional_decoder/headline_inputs.pt",
    )
    parser.add_argument("--candidate", choices=("all", "matmul_bf16_acc", "matmul_fp32_acc"), default="all")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--timing-samples", type=int, default=3)
    parser.add_argument("--warmup-replays", type=int, default=30)
    args = parser.parse_args()
    if min(args.repeats, args.timing_samples, args.warmup_replays) < 1:
        parser.error("repeats, timing-samples and warmup-replays must be positive")

    torch.set_num_threads(8)
    config = AutoConfig.from_pretrained(Path(__file__).parent).text_config
    config._attn_implementation = "eager"
    recorded = torch.load(args.inputs, map_location="cpu", weights_only=True)
    prefill_x = recorded["x"][:, :64].contiguous()
    decode_x = recorded["decode_inputs"][:, :1].contiguous()
    assert prefill_x.shape == (1, 64, config.hidden_size)
    assert decode_x.shape == (1, 1, config.hidden_size)
    hf = load_layer(config, args.layer, True)
    layer_type = config.layer_types[args.layer]
    rope = Gemma4TextRotaryEmbedding(config)
    extent = 1024
    with torch.no_grad():
        cos, sin = rope(prefill_x, torch.arange(extent)[None], layer_type=layer_type)

    report = dict(
        layer=args.layer,
        layer_type=layer_type,
        revision=REVISION,
        real_weights=True,
        recorded_inputs=str(args.inputs),
        input_source="live selected PackedExperts sparse-down outputs and routes from real decoder execution",
        prefill_length=64,
        decode_position=64,
        pcc_threshold=0.995,
        repeats=args.repeats,
        timing_samples=args.timing_samples,
        warmup_replays=args.warmup_replays,
        timing_scope="mixing suffix including all operand and output movement; excludes sparse projections and capture",
        cases=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        decoder = FusedDecoder.from_state_dict(
            hf.state_dict(), hf_config=config, layer_idx=args.layer, mesh_device=mesh
        )
        selected = decoder.layer.moe.experts
        assert isinstance(selected, PackedExperts)
        assert selected.prefill_batch_tokens == 64, "This probe captures the selected 64-token expert batch"
        capture = CaptureMix(selected)
        decoder.layer.moe.experts = capture
        report["capture_fusion"] = decoder.fusion
        report["capture_override"] = "multiply/reduce suffix only; live sparse projections and routing unchanged"

        def upload(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(value, dtype=dtype, layout=layout, device=mesh)

        x = upload(prefill_x[None])
        tables = tuple(upload(value[None]) for value in (cos, sin))
        pages = extent // 32
        page_table = upload(torch.arange(pages - 1, -1, -1, dtype=torch.int32)[None], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        attention = decoder.layer.self_attn.config
        cache_shape = (pages, attention.num_key_value_heads, 32, attention.head_dim)
        caches = tuple(upload(torch.zeros(cache_shape, dtype=torch.bfloat16)) for _ in range(2))
        with device_only():
            output = decoder.prefill_forward(x, rope_mats=tables, page_table=page_table, kv_cache=caches)
        output.deallocate(True)
        decode_input = upload(decode_x[None])
        decode_tables = tuple(upload(value.squeeze(0)) for value in (cos, sin))
        current_pos = torch.zeros(1, 32, dtype=torch.int32)
        current_pos[0, 0] = 64
        position = upload(current_pos, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        cache_position = upload(torch.tensor([64], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        with device_only():
            output = decoder.decode_forward(
                decode_input,
                rope_mats=decode_tables,
                current_pos=position,
                cache_pos=cache_position,
                page_table=page_table,
                kv_cache=caches,
            )
        output.deallocate(True)
        decoder.layer.moe.experts = selected
        assert [(item["decode"], item["length"]) for item in capture.records] == [(False, 64), (True, 1)]
        print("REAL_EXPERT_MIX_BOUNDARIES_READY", flush=True)

        configs = {
            name: ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32_acc,
                packer_l1_acc=False,
            )
            for name, fp32_acc in (("matmul_bf16_acc", False), ("matmul_fp32_acc", True))
            if args.candidate in ("all", name)
        }
        for captured in capture.records:
            down, routes, mixed = (captured[key] for key in ("down", "routing", "mixed"))
            kwargs = dict(
                decode=captured["decode"],
                length=captured["length"],
                experts=selected.config.num_experts,
                hidden_size=selected.config.hidden_size,
            )
            reference = ttnn.to_torch(mixed).float()
            output_memory = mixed.memory_config()
            case = dict(
                phase="decode" if captured["decode"] else "prefill",
                length=captured["length"],
                down_shape=list(down.shape),
                down_dtype=str(down.dtype),
                down_memory=str(down.memory_config()),
                routing_shape=list(routes.shape),
                routing_dtype=str(routes.dtype),
                output_shape=list(mixed.shape),
                output_dtype=str(mixed.dtype),
                output_memory=str(output_memory),
                results=[],
            )
            report["cases"].append(case)

            def baseline():
                return existing_mix(down, routes, **kwargs)

            with device_only():
                control = baseline()
            case["baseline_matches_live_capture"] = torch.equal(reference, ttnn.to_torch(control).float())
            control.deallocate(True)
            save()
            assert case["baseline_matches_live_capture"], "Probe baseline differs from live PackedExperts mixing"
            candidates = [("multiply_reduce", baseline)] + [
                (
                    name,
                    lambda compute=compute: matmul_mix(
                        down, routes, **kwargs, compute=compute, output_memory=output_memory
                    ),
                )
                for name, compute in configs.items()
            ]
            baseline_us = None
            for name, fn in candidates:
                row = dict(
                    candidate=name,
                    math_fidelity="HiFi4" if name != "multiply_reduce" else None,
                    fp32_dest_acc_en=name == "matmul_fp32_acc" if name != "multiply_reduce" else None,
                    output_dtype="BFLOAT16" if name != "multiply_reduce" else str(mixed.dtype),
                )
                try:
                    actual, replay_equal, samples = measure(mesh, fn, args)
                    passing, pcc = comp_pcc(reference, actual, 0.995)
                    finite = bool(torch.isfinite(actual).all())
                    median = statistics.median(samples)
                    if name == "multiply_reduce":
                        baseline_us = median
                    row.update(
                        pcc=float(pcc),
                        passed=bool(passing) and finite and replay_equal,
                        exact_equal=torch.equal(reference, actual),
                        max_abs=float((reference - actual).abs().max()),
                        finite=finite,
                        trace_replay_equal=replay_equal,
                        traced_host_us=median,
                        traced_host_us_samples=samples,
                        speedup_vs_multiply_reduce=baseline_us / median,
                        component_candidate=bool(passing) and finite and replay_equal and median < baseline_us,
                    )
                except Exception as error:
                    row.update(passed=False, error=f"{type(error).__name__}: {error}")
                    case["results"].append(row)
                    save()
                    raise
                case["results"].append(row)
                save()
                print(dict(phase=case["phase"], **row), flush=True)
        assert all(row["passed"] for case in report["cases"] for row in case["results"]), report["cases"]
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
