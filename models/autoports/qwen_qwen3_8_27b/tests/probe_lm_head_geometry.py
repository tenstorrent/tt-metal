# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Trace LM-head geometry on a recorded real-weight decoder activation."""

import argparse
import faulthandler
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--block", type=int, default=10)
    parser.add_argument("--readers", type=int, default=2)
    parser.add_argument("--fidelity", choices=("HiFi2", "LoFi"), default="HiFi2")
    parser.add_argument(
        "--report-rejected", action="store_true", help="Record failed PCC candidates without selecting them"
    )
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.chunk % 64 or args.chunk < 64 or args.iterations < 1:
        parser.error("Chunk must be a positive multiple of 64; iterations must be positive")
    if args.readers not in (1, 2, 3):
        parser.error("DRAM matmul supports one to three readers per bank")
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    gen = None
    trace = None
    try:
        faulthandler.dump_traceback_later(60, repeat=True)
        gen = build_generator("models/autoports/qwen_qwen3_8_27b", mesh, layer_indices=[0, 3])
        faulthandler.cancel_dump_traceback_later()
        prompt = gen.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Explain why the sky is blue."}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt = gen.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        gen.generate(prompt, 1)
        model = gen.model
        original = model._dram_logits
        activation = []

        def record(hidden):
            activation.append(ttnn.clone(hidden))
            return original(hidden)

        model._dram_logits = record
        gen._model_step()
        model._dram_logits = original
        hidden = activation.pop()
        reference = original(hidden)
        host_reference = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(reference)]
        del reference
        banks = mesh.dram_grid_size().x
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        weights = []
        for start in range(0, model.config.vocab_size // 4, args.chunk):
            weight = model.head_weight[:, start : min(start + args.chunk, model.config.vocab_size // 4)]
            quantum = args.readers * 32
            width = ((weight.shape[-1] + banks * quantum - 1) // (banks * quantum)) * quantum
            memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(grid, [model.config.hidden_size, width], ttnn.ShardOrientation.ROW_MAJOR),
            )
            weights.append(ttnn.to_memory_config(weight, memory))
        configs = [
            ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=args.block,
                per_core_M=1,
                per_core_N=w.memory_config().shard_spec.shape[1] // 32,
                num_workers_per_dram_bank=args.readers,
                fused_activation=None,
            )
            for w in weights
        ]
        compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, args.fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        def candidate():
            parts = [
                ttnn.to_memory_config(
                    ttnn.linear(
                        hidden,
                        w,
                        dtype=ttnn.bfloat16,
                        compute_kernel_config=compute,
                        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                        program_config=c,
                    ),
                    ttnn.DRAM_MEMORY_CONFIG,
                )
                for w, c in zip(weights, configs)
            ]
            return ttnn.concat(parts, dim=-1)

        output = candidate()
        correctness = []
        for expected, shard in zip(host_reference, ttnn.get_device_tensors(output)):
            actual = ttnn.to_torch(shard).float()
            # Only the first row is active; inactive tile rows are not semantic outputs.
            active_actual, active_expected = actual[..., 0, :], expected[..., 0, :]
            pcc = torch.corrcoef(torch.stack([active_actual.flatten(), active_expected.flatten()]))[0, 1].item()
            equal = torch.equal(active_actual.argmax(-1), active_expected.argmax(-1))
            correctness.append(
                dict(
                    logit_pcc=pcc,
                    local_greedy_equal=equal,
                    logical_shape=list(actual.shape),
                    active_rows=1,
                    accepted=pcc >= 0.999 and equal,
                )
            )
        del output
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        output = candidate()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        begin = time.perf_counter()
        for _ in range(args.iterations):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        elapsed = time.perf_counter() - begin
        report = dict(
            chunk=args.chunk,
            block=args.block,
            readers=args.readers,
            iterations=args.iterations,
            head_us=elapsed * 1e6 / args.iterations,
            weights="bfloat8_b",
            activations="bfloat16",
            fidelity=f"{args.fidelity}, FP32 accumulation",
            local_weight_shapes=[list(w.shape) for w in weights],
            per_core_N=[w.memory_config().shard_spec.shape[1] // 32 for w in weights],
            correctness=correctness,
            accepted=all(row["accepted"] for row in correctness),
            scope="Head-only traced component: fixed real-weight layers 0/3 activation, includes concat and output layouts",
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if not args.report_rejected:
            assert report["accepted"], correctness
    finally:
        faulthandler.cancel_dump_traceback_later()
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
