# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision-locked TP4 terminal projection probe; never full-model timing.

Capture normalized terminal input from the real two-layer wrapper, then compare
complete LM-head paths, including input reshard, per-chunk output conversions,
padding removal and concatenation. All vocabulary ownership stays unchanged.
"""

import argparse
import json
import statistics
import time
from itertools import product
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_dram_readers import digest, geometry, pcc
from models.autoports.google_gemma_4_26b_a4b_it.tt.model import MODEL_ID, REVISION, Checkpoint
from models.demos.gemma4.tt.model import _get_lm_head_program_config


def capture(args, mesh):
    from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import build_generator

    gen = build_generator(None, mesh, max_seq_len=8192, layer_indices=(0, 5))
    retained = []
    original = ttnn.linear

    def linear(x, w, **kwargs):
        result = original(x, w, **kwargs)
        if w is gen.model.head and not retained:
            # Keep an eager prefill terminal tensor; never perform host work in capture.
            retained.append(x)
        return result

    try:
        prompt = gen.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Explain why the sum of two odd integers is even."}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        with patch.object(ttnn, "linear", linear):
            gen.prefill_logits(prompt)
        value = ttnn.to_torch(ttnn.get_device_tensors(retained[0])[0]).float()
        # prefill_logits requests all physical prompt rows. Preserve the last
        # logical prompt row, excluding internal tile padding, for M=1 decode.
        value = value[..., len(prompt) - 1 : len(prompt), :].contiguous()
        assert value.shape == (1, 1, 1, 2816), value.shape
        torch.save(
            dict(
                input=value,
                model=MODEL_ID,
                revision=REVISION,
                source="Actual normalized terminal input, real reduced layers 0/5, chat prompt",
                prompt_ids=prompt,
                model_sha256=digest(Path(__file__).parents[1] / "tt/model.py"),
            ),
            args.fixture,
        )
    finally:
        gen.teardown()


def upload(mesh, value, memory=ttnn.DRAM_MEMORY_CONFIG, sharded=False, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        value,
        device=mesh,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1) if sharded else ttnn.ReplicateTensorToMesh(mesh),
    )


def benchmark(args, mesh):
    from tracy import signpost

    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    assert fixture["model"] == MODEL_ID and fixture["revision"] == REVISION
    xhost = fixture["input"].bfloat16()
    embedding = Checkpoint().load("model.language_model.embed_tokens.")["weight"].bfloat16()
    k, local_n = embedding.shape[1], embedding.shape[0] // 4
    assert k == 2816 and local_n == 65536
    weight_host = embedding.T.contiguous()[None, None]
    del embedding
    x = upload(mesh, xhost)
    compute = ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, args.fidelity),
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    baseline_weight = upload(mesh, weight_host, sharded=True, dtype=getattr(ttnn, args.weight_dtype))
    baseline_pc = _get_lm_head_program_config(mesh, xhost.shape[-2], k, local_n)
    if args.baseline_block is not None:
        baseline_pc.in0_block_w = args.baseline_block

    def baseline():
        return ttnn.linear(
            x,
            baseline_weight,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute,
            program_config=baseline_pc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    expected_tensor = baseline()
    expected = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(expected_tensor)]
    ttnn.deallocate(expected_tensor)
    report = dict(
        model=MODEL_ID,
        revision=REVISION,
        fixture=str(args.fixture),
        fixture_sha256=digest(args.fixture),
        scope="Isolated complete TP4 LM head, actual reduced-model terminal input; no sampler or stack",
        dtype=args.weight_dtype,
        fidelity=args.fidelity,
        fp32_dest_acc_en=True,
        mesh=[1, 4],
        cases=[],
    )

    def measure(name, forward, metadata):
        row = dict(name=name, **metadata)
        report["cases"].append(row)
        out = forward()
        ttnn.synchronize_device(mesh)
        actual = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(out)]
        row["pcc_baseline_per_device"] = [pcc(a, b) for a, b in zip(actual, expected)]
        row["top1_equal_per_device"] = [bool(torch.equal(a.argmax(-1), b.argmax(-1))) for a, b in zip(actual, expected)]
        row["max_abs_diff_per_device"] = [float((a - b).abs().max()) for a, b in zip(actual, expected)]
        assert min(row["pcc_baseline_per_device"]) >= 0.999, row
        ttnn.deallocate(out)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = forward()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        try:
            for _ in range(3):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            samples = []
            signpost("LM_HEAD_" + name)
            for _ in range(args.rounds):
                started = time.perf_counter()
                for _ in range(args.replays):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                samples.append((time.perf_counter() - started) * 1e6 / args.replays)
            signpost("LM_HEAD_" + name + "_END")
            row.update(host_replay_us=samples, median_host_replay_us=statistics.median(samples), status="passed")
        finally:
            ttnn.release_trace(mesh, trace)
            ttnn.deallocate(out)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    measure("baseline", baseline, dict(program=str(baseline_pc)))
    if args.interleaved_only:
        for grid_name in args.grids:
            grid = ttnn.CoreCoord(*map(int, grid_name.split("x")))
            minimum_n = (local_n // 32 + grid.x * grid.y - 1) // (grid.x * grid.y)
            widths = args.per_core_n or [minimum_n]
            for block, per_core_n in product(args.blocks, widths):
                if per_core_n < minimum_n:
                    raise ValueError(f"per-core N{per_core_n} cannot cover the vocabulary on grid {grid_name}")
                subblock = min(per_core_n, 4)
                while per_core_n % subblock:
                    subblock -= 1
                pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
                    in0_block_w=block,
                    out_subblock_h=1,
                    out_subblock_w=subblock,
                    per_core_M=1,
                    per_core_N=per_core_n,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )

                def forward():
                    hidden = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG) if args.input_memory == "l1" else x
                    return ttnn.linear(
                        hidden,
                        baseline_weight,
                        dtype=ttnn.bfloat16,
                        compute_kernel_config=compute,
                        program_config=pc,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )

                try:
                    measure(
                        f"interleaved_g{grid_name}_b{block}_n{per_core_n}_{args.input_memory}",
                        forward,
                        dict(program=str(pc), input_memory=args.input_memory),
                    )
                except RuntimeError as error:
                    if not any(
                        term in str(error)
                        for term in (
                            "Statically allocated circular buffers",
                            "overlap with L1",
                            "L1 buffer",
                            "L1 size",
                        )
                    ):
                        raise
                    report["cases"][-1].update(status="allocation_rejected", error=str(error))
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
        return report
    banks = mesh.dram_grid_size()
    for chunk in args.chunks:
        for block in args.blocks:
            geo = geometry(
                k, chunk, block, banks.x * banks.y, mesh.compute_with_storage_grid_size().x, args.weight_dtype
            )
            cores, n = geo["input_storage_cores"], geo["physical_MKN"][-1]
            memory = lambda shape: ttnn.create_sharded_memory_config(
                shape,
                ttnn.CoreGrid(x=cores, y=1),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            weight_mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks.x - 1, banks.y - 1))}),
                    geo["weight_shard_shape"],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            weights, widths = [], []
            try:
                for start in range(0, local_n, chunk):
                    width = min(chunk, local_n - start)
                    host = torch.cat(
                        [
                            torch.nn.functional.pad(
                                weight_host[..., d * local_n + start : d * local_n + start + width], (0, n - width)
                            )
                            for d in range(4)
                        ],
                        -1,
                    )
                    weights.append(upload(mesh, host, weight_mem, True, dtype=getattr(ttnn, args.weight_dtype)))
                    widths.append(width)
                for readers in args.readers:
                    pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                        in0_block_w=block, per_core_M=1, per_core_N=n // 32 // cores, num_workers_per_dram_bank=readers
                    )

                    def forward():
                        working = ttnn.to_memory_config(x, memory(geo["input_shard_shape"]))
                        outputs = []
                        for w, width in zip(weights, widths):
                            out = ttnn.linear(
                                working,
                                w,
                                dtype=ttnn.bfloat16,
                                compute_kernel_config=compute,
                                program_config=pc,
                                memory_config=memory(geo["output_shard_shape"]),
                            )
                            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
                            outputs.append(out[..., :width])
                        return ttnn.concat(outputs, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

                    name = f"c{chunk}_b{block}_r{readers}"
                    try:
                        measure(name, forward, dict(geometry=geo, readers=readers, program=str(pc)))
                    except RuntimeError as error:
                        # Only compile/allocation legality failures may skip a candidate.
                        # Device execution, assertion, and timeout failures abort the run.
                        if not any(
                            term in str(error)
                            for term in (
                                "Statically allocated circular buffers",
                                "overlap with L1",
                                "Not enough worker",
                                "L1 buffer",
                                "L1 size",
                            )
                        ):
                            raise
                        report["cases"][-1].update(status="allocation_rejected", error=str(error))
                        args.output.write_text(json.dumps(report, indent=2) + "\n")
                        print(f"REJECT {name}: {error}", flush=True)
            finally:
                for w in weights:
                    ttnn.deallocate(w)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--weight-dtype", choices=["bfloat16", "bfloat8_b", "bfloat4_b"], default="bfloat16")
    parser.add_argument("--fidelity", choices=["LoFi", "HiFi2", "HiFi4"], default="HiFi4")
    parser.add_argument("--interleaved-only", action="store_true")
    parser.add_argument("--grids", nargs="+", default=["11x10"])
    parser.add_argument("--per-core-n", type=int, nargs="+", help="Interleaved output tile counts per core")
    parser.add_argument("--input-memory", choices=["dram", "l1"], default="dram")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks", type=int, nargs="+", default=[8192, 16384])
    parser.add_argument("--blocks", type=int, nargs="+", default=[1, 2, 4, 11, 22])
    parser.add_argument("--baseline-block", type=int, help="Match the selected production head K block")
    parser.add_argument("--readers", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--replays", type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    try:
        if args.capture:
            capture(args, mesh)
        else:
            benchmark(args, mesh)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
