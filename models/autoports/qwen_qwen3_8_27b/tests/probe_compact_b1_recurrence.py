# SPDX-License-Identifier: Apache-2.0
"""Compare compact B1 recurrence against actual selected-policy layer0 inputs.

Exact output and FP32 state equality on every ASIC is required before timing.
This two-layer probe cannot establish full64 TSU or generation quality.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt import multichip_decoder
from models.autoports.qwen_qwen3_8_27b.tt.compact_b1_recurrence import fp32_recurrence as compact
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--parallel-v-module", type=Path, help="Experimental four-way V32 candidate module; reduced probe only"
    )
    parser.add_argument("--candidate-value-parts", type=int, choices=(1, 2, 4), default=4)
    args = parser.parse_args()
    candidate = compact
    if args.parallel_v_module:
        spec = importlib.util.spec_from_file_location("qwen38_parallel_v_candidate", args.parallel_v_module)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        candidate = module.fp32_recurrence
    # Observe the original model recurrence to obtain the actual selected inputs.
    os.environ["QWEN38_COMPACT_B1_RECURRENCE"] = "0"
    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback = True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=64 * 1024**2)
    original = multichip_decoder.fp32_recurrence
    captured = []
    traces = {}
    result = dict(
        status="running",
        scope="Real layer0 B1 recurrence output/state; reduced two-layer fixture, not full64 TSU/quality",
        tracking=os.getenv("TT_METAL_TRACE_ALLOC_TRACKING"),
        skip_program_cache=os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"),
        candidate_value_parts=args.candidate_value_parts,
    )
    if args.parallel_v_module:
        sources = [args.parallel_v_module, *args.parallel_v_module.parent.glob("*.cpp")]
        result["candidate_source_sha256"] = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}

    def observe(*inputs, **kwargs):
        frozen = dict(kwargs)
        frozen["initial_state"] = ttnn.clone(kwargs["initial_state"])
        frozen["inplace_state"] = False
        captured.append((inputs, frozen))
        return original(*inputs, **kwargs)

    try:
        gen = build_generator(
            Path(__file__).resolve().parents[1],
            mesh,
            checkpoint=args.checkpoint,
            layer_indices=[0, 3],
            batch_size=1,
            context_length=512,
            prefill_chunk_size=128,
            prepare=False,
        )
        logits = gen.prefill_forward(
            torch.full((1, 33), 100, dtype=torch.int64),
            page_table=gen.page_table,
            kv_cache=gen.kv_cache,
            prompt_lens=[33],
            return_device=True,
        )
        token = gen.host_logits(logits).reshape(-1).argmax().to(torch.int32).reshape(1, 1, 1, 1)
        gen.refresh(token, gen.tokens)
        gen.refresh(torch.tensor([33], dtype=torch.int32), gen.positions)
        gen.refresh(torch.tensor([[33]], dtype=torch.int32), gen.rope_indices)
        multichip_decoder.fp32_recurrence = observe
        try:
            gen._decode()
        finally:
            multichip_decoder.fp32_recurrence = original
        assert len(captured) == 1, len(captured)
        inputs, kwargs = captured[0]
        result["inputs"] = [
            dict(shape=list(x.shape), padded=list(x.padded_shape), memory=str(x.memory_config())) for x in inputs
        ]
        # Establish dataflow identity independently of the unchanged compute.
        q, k, v, g, beta = inputs

        def split(x, heads):
            x = ttnn.reshape(ttnn.permute(x, (0, 2, 1, 3)), [heads, 1, 128])
            return ttnn.repeat_interleave(x, 12 // heads, dim=0) if heads != 12 else x

        def scalar(x):
            return multichip_decoder.reshape(ttnn.reshape(ttnn.permute(x, (0, 2, 1)), [12, 1]), [12, 1, 1])

        expected = [
            split(ttnn.multiply(q, 128**-0.5), 4),
            split(k, 4),
            split(v, 12),
            scalar(ttnn.exp(g)),
            scalar(beta),
            ttnn.reshape(kwargs["initial_state"], [12, 128, 128]),
        ]
        actual = candidate(*inputs, **kwargs, debug_operands=True)
        operand_checks = []
        for operand, (left, right) in enumerate(zip(expected, actual)):
            for device, (l, r) in enumerate(zip(ttnn.get_device_tensors(left), ttnn.get_device_tensors(right))):
                l, r = ttnn.to_torch(l), ttnn.to_torch(r)
                if args.candidate_value_parts > 1:
                    parts = args.candidate_value_parts
                    if operand in (0, 1, 3, 4):
                        l = l.repeat_interleave(parts, dim=0)
                    elif operand == 2:
                        l = (
                            l.reshape(12, 1, parts, 128 // parts)
                            .permute(0, 2, 1, 3)
                            .reshape(12 * parts, 1, 128 // parts)
                        )
                    else:
                        l = (
                            l.reshape(12, 128, parts, 128 // parts)
                            .permute(0, 2, 1, 3)
                            .reshape(12 * parts, 128, 128 // parts)
                        )
                operand_checks.append(
                    dict(
                        operand=operand,
                        device=device,
                        exact=bool(torch.equal(l, r)),
                        max_abs=float((l - r).abs().max()),
                        expected_head0=l[0].reshape(-1)[:8].tolist(),
                        actual_head0=r[0].reshape(-1)[:8].tolist(),
                        expected_scalars=l.reshape(-1).tolist() if operand in (3, 4) else None,
                        actual_scalars=r.reshape(-1).tolist() if operand in (3, 4) else None,
                    )
                )
        result["reader_operands"] = operand_checks
        assert len(operand_checks) == 24 and all(row["exact"] for row in operand_checks), operand_checks
        functions = dict(baseline=original)
        if args.parallel_v_module:
            functions["compact"] = compact
        functions["candidate"] = candidate
        outputs = {}
        for name, function in functions.items():
            print(f"WARMUP_START {name}", flush=True)
            y, state = function(*inputs, **kwargs)
            outputs[name] = (ttnn.zeros_like(y), ttnn.zeros_like(state))
            ttnn.synchronize_device(mesh)
            print(f"WARMUP_DONE {name}", flush=True)

        def run(name):
            y, state = functions[name](*inputs, **kwargs)
            ttnn.copy(y, outputs[name][0])
            ttnn.copy(state, outputs[name][1])

        def compare():
            rows = []
            for arm in (name for name in functions if name != "baseline"):
                for kind, index in (("output", 0), ("state", 1)):
                    for device, (left, right) in enumerate(
                        zip(
                            ttnn.get_device_tensors(outputs["baseline"][index]),
                            ttnn.get_device_tensors(outputs[arm][index]),
                        )
                    ):
                        left, right = ttnn.to_torch(left), ttnn.to_torch(right)
                        rows.append(
                            dict(
                                variant=arm,
                                kind=kind,
                                device=device,
                                exact=bool(torch.equal(left, right)),
                                max_abs=float((left - right).abs().max()),
                            )
                        )
            assert len(rows) == 8 * (len(functions) - 1) and all(row["exact"] for row in rows), rows
            return rows

        for name in functions:
            run(name)
        ttnn.synchronize_device(mesh)
        result["before_replay"] = compare()
        for name in functions:
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            run(name)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            traces[name] = trace
        timing = {name: [] for name in functions}
        for order in (list(functions), list(reversed(functions)), list(functions)):
            for name in order:
                ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=True)
                started = time.monotonic()
                for _ in range(args.replays):
                    ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                timing[name].append((time.monotonic() - started) * 1e6 / args.replays)
        result.update(
            status="pass",
            trace_us=timing,
            after_replay=compare(),
            speed_ratio=statistics.median(timing["baseline"]) / statistics.median(timing["candidate"]),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
    except BaseException as exc:
        result.update(status="fail", error=repr(exc))
        raise
    finally:
        multichip_decoder.fp32_recurrence = original
        for trace in traces.values():
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
