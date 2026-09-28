# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-stack adapter experiment isolating expert width and page-table copies."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import AutoportGemma4ForCausalLM
from models.autoports.google_gemma_4_26b_a4b_it.tt.model import MODEL_ID, REVISION
from models.common.sampling.generator import SamplingParams
from vllm.benchmarks.datasets import RandomDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument(
        "--dataset-requests", type=int, default=5, help="Native benchmark num-prompts used to generate inputs"
    )
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument(
        "--cache-context", type=int, default=262144, help="Physical KV capacity; table width stays 8192"
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=(
            "32cached",
            "32forced",
            "64cached",
            "128cached",
            "32k22",
            "32k44",
            "32k88",
            "32fullk22",
            "gate22cores",
            "gate11cores",
            "down44cores",
            "down22cores",
            "pair22_44",
            "shared_gate22",
            "shared_gate44",
            "shared_gate88",
            "shared_down44",
            "shared_pair44",
        ),
        default=["32cached", "32forced", "64cached", "128cached"],
    )
    args = parser.parse_args()
    if not 1 <= args.length <= 1024 or min(args.requests, args.warmups, args.dataset_requests) < 1:
        parser.error("length must be 1..1024 and request/warmup counts positive")
    if not args.length + 1 <= args.cache_context <= 262144:
        parser.error("cache-context must cover the prompt plus decode and be at most 262144")
    candidates = list(dict.fromkeys(["32cached"] + args.candidates))
    if any(name.startswith("shared_") for name in candidates) and args.length != 128:
        parser.error("Shared prefill program candidates support only logical length128")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        scope="Full model, direct adapter with synchronized prefill and first decode; excludes HTTP/scheduler and later decode",
        model=MODEL_ID,
        revision=REVISION,
        seed=args.seed,
        requested_length=args.length,
        max_seq_len=262144,
        physical_cache_context=args.cache_context,
        page_table_shape=[32, 8192],
        trace_region_size=1024**3,
        requests=args.requests,
        warmups=args.warmups,
        dataset_requests=args.dataset_requests,
        candidate_order=candidates,
        prompt_source="Native RandomDataset.sample(range_ratio=0,prefix_len=0,output_len=16), pinned generator tokenizer; encode with special tokens as completion endpoint does",
        prompts=[],
        candidates=[],
        passed=False,
    )

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    # Build every prospective row config/index buffer once, before any traces.
    os.environ["GEMMA4_PREFILL_EXPERT_BATCH"] = "128"
    os.environ["GEMMA4_PREFILL_TRACE"] = "1"
    torch.set_num_threads(4)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1024**3)
    gen = None
    experts, original_configs, original_widths = [], [], []
    shared_layers, original_shared, shared_weights = [], [], []

    def timed(function):
        ttnn.synchronize_device(mesh)
        start = time.perf_counter_ns()
        value = function()
        ttnn.synchronize_device(mesh)
        return value, (time.perf_counter_ns() - start) / 1e6

    try:
        gen = Gemma4Generator(mesh, max_seq_len=262144)
        assert len(gen.model.layers) == 30
        gen.prefill_trace_enabled = True
        adapter = AutoportGemma4ForCausalLM(gen, 32)
        experts = [layer.layer.moe.experts.prefill for layer in gen.model.layers]
        assert all(
            all(rows in expert.prefill_configs and rows in expert.route_indices for rows in (32, 64, 96, 128))
            for expert in experts
        )
        original_configs = [dict(expert.prefill_configs) for expert in experts]
        original_widths = [expert.short_prefill_batch_tokens for expert in experts]
        gate_configs = {
            block: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(11, 4),
                in0_block_w=block,
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=1,
                out_block_w=1,
                per_core_M=1,
                per_core_N=1,
                fuse_batch=False,
                mcast_in0=True,
            )
            for block in (22, 44, 88)
        }
        # Coherent N geometry: preserve total output tiles while changing
        # core count, per-core N, and output block/subblock width together.
        # These are program objects only; weights and compute configs stay live.
        geometry_specs = {
            "gate22cores": ((11, 2), 2),
            "gate11cores": ((11, 1), 4),
            "down44cores": ((11, 4), 2),
            "down22cores": ((11, 2), 4),
        }
        geometry_configs = {
            name: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=grid,
                in0_block_w=22,
                out_subblock_h=1,
                out_subblock_w=columns,
                out_block_h=1,
                out_block_w=columns,
                per_core_M=1,
                per_core_N=columns,
                fuse_batch=False,
                mcast_in0=True,
            )
            for name, (grid, columns) in geometry_specs.items()
        }
        geometry_candidates = {
            "gate22cores": ("gate22cores", None),
            "gate11cores": ("gate11cores", None),
            "down44cores": (None, "down44cores"),
            "down22cores": (None, "down22cores"),
            "pair22_44": ("gate22cores", "down44cores"),
        }
        if any(name in geometry_candidates for name in candidates):
            assert all(
                configs[32][0].in0_block_w == configs[32][1].in0_block_w == 22 for configs in original_configs
            ), "Geometry sweep requires selected gate/down K22 baseline; remove conflicting GEMMA4_PREFILL_GATE_K"
        shared_candidates = {
            "shared_gate22": (22, False),
            "shared_gate44": (44, False),
            "shared_gate88": (88, False),
            "shared_down44": (None, True),
            "shared_pair44": (44, True),
        }
        shared_programs = {}
        shared_compute = None
        if any(name in shared_candidates for name in candidates):
            shared_layers = [layer.layer.shared_mlp for layer in gen.model.layers]
            original_shared = [(layer.gate_up, layer.down) for layer in shared_layers]

            def closure_weight(project):
                weights = [
                    cell.cell_contents
                    for cell in (getattr(project, "__closure__", None) or ())
                    if isinstance(cell.cell_contents, ttnn.Tensor)
                ]
                assert len(weights) == 1, "Expected one existing TT weight in original shared projection closure"
                return weights[0]

            shared_weights = [tuple(closure_weight(project) for project in pair) for pair in original_shared]
            for layer, pair in zip(shared_layers, shared_weights):
                assert layer.width == 544 and layer.decode_weights is not None
                for weight, expected_shape in zip(pair, ((2816, 1088), (544, 2816))):
                    assert weight.dtype == ttnn.bfloat16, "Shared probe must retain original BF16 weights"
                    assert all(
                        tuple(shard.shape)[-2:] == expected_shape for shard in ttnn.get_device_tensors(weight)
                    ), "Shared weight shard dimensions differ from this bounded S128 program"

            # matmul_device_operation.cpp:create_matmul_attributes chooses
            # HiFi2 for generic BF16 but LoFi when a program is supplied with
            # compute=None. Explicitly preserve the generic resolved defaults:
            # HiFi2, approximate=False, BF16 output => fp32=False, packer=True.
            shared_compute = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )

            def shared_program(grid, per_n, block_k):
                return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=grid,
                    in0_block_w=block_k,
                    out_subblock_h=1,
                    out_subblock_w=per_n,
                    out_block_h=4,
                    out_block_w=per_n,
                    per_core_M=4,
                    per_core_N=per_n,
                    fuse_batch=True,
                    mcast_in0=True,
                )

            shared_programs = {block: shared_program((9, 2), 2, block) for block in (22, 44, 88)}
            shared_programs["down44"] = shared_program((11, 4), 2, 17)
            report["shared_compute_resolution"] = {
                "source": "ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:create_matmul_attributes",
                "baseline": "Original generic closures, compute=None, resolves to HiFi2 for BF16",
                "candidate": str(shared_compute),
                "reason": "Explicit program with compute=None would silently select LoFi; preserve baseline fidelity",
                "output_defaults": "dtype omitted => input BF16; memory omitted => DRAM; no activation/bias overrides",
            }

        def shared_projection(weight, program):
            def project(x):
                assert x.dtype == ttnn.bfloat16 and x.shape[-2] == 128
                # Preserve dtype/memory/bias/activation defaults exactly.
                return ttnn.linear(x, weight, program_config=program, compute_kernel_config=shared_compute)

            return project

        report["precision"] = gen.model.precision_summary()
        report["expert_gate_k_tiles"] = [
            {str(rows): expert.prefill_configs[rows][0].in0_block_w for rows in (32, 64, 96, 128)} for expert in experts
        ]
        # Default to the proven full-context allocation. Smaller explicit
        # controls still preserve all 8192 logical columns at the API boundary.
        cache, allocated_table = gen.model.allocate_cache(slots=1, context=args.cache_context)
        table = torch.zeros(32, 8192, dtype=torch.int32)
        table[0, : allocated_table.shape[1]] = allocated_table[0]
        tables = [table] * len(gen.model.layers)
        report["allocated_pages_per_layer"] = [pair[0].shape[0] for pair in cache]
        requests = RandomDataset(random_seed=args.seed).sample(
            tokenizer=gen.tokenizer,
            num_requests=args.dataset_requests,
            prefix_len=0,
            range_ratio=0.0,
            input_len=args.length,
            output_len=16,
        )
        tokens = []
        for index, request in enumerate(requests):
            ids = gen.tokenizer.encode(request.prompt, add_special_tokens=True)
            report["prompts"].append(
                dict(
                    index=index,
                    text=request.prompt,
                    dataset_prompt_len=request.prompt_len,
                    token_ids=ids,
                    length=len(ids),
                )
            )
            assert len(ids) == args.length, "Native prompt retokenization changed length; inspect saved prompt"
            tokens.append(torch.tensor([ids], dtype=torch.long))
        prefill_params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
        decode_params = SamplingParams(temperature=[0.0] * 32, top_k=[1] * 32, top_p=[1.0] * 32)
        expected = {}
        for name in candidates:
            gen._release_trace()
            gen.prefill_prepared = None
            adapter._sampling_signature = None
            adapter._decode_batch = None
            for layer, original in zip(shared_layers, original_shared):
                layer.gate_up, layer.down = original
            shared_gate, shared_down = shared_candidates.get(name, (None, False))
            for layer, (gate_weight, down_weight) in zip(shared_layers, shared_weights):
                if shared_gate is not None:
                    layer.gate_up = shared_projection(gate_weight, shared_programs[shared_gate])
                if shared_down:
                    layer.down = shared_projection(down_weight, shared_programs["down44"])
            width = {"64cached": 64, "128cached": 128}.get(name, 32)
            gate_block = {"32k22": 22, "32k44": 44, "32k88": 88, "32fullk22": 22}.get(name)
            for layer_index, expert, configs in zip(gen.model.layer_indices, experts, original_configs):
                expert.prefill_configs = dict(configs)
                expert.short_prefill_batch_tokens = width
                if gate_block is not None and (
                    name != "32fullk22" or gen.model.config.layer_types[layer_index] == "full_attention"
                ):
                    expert.prefill_configs[32] = (gate_configs[gate_block], configs[32][1])
                if name in geometry_candidates:
                    gate_name, down_name = geometry_candidates[name]
                    expert.prefill_configs[32] = (
                        configs[32][0] if gate_name is None else geometry_configs[gate_name],
                        configs[32][1] if down_name is None else geometry_configs[down_name],
                    )
            record = dict(name=name, expert_batch=width, forced_page_copies=name.endswith("forced"), requests=[])
            if shared_layers:
                record["shared_configs"] = [
                    dict(
                        layer=layer_index,
                        gate_weight_shape=list(pair[0].shape),
                        down_weight_shape=list(pair[1].shape),
                        gate_shard_shape=list(ttnn.get_device_tensors(pair[0])[0].shape),
                        down_shard_shape=list(ttnn.get_device_tensors(pair[1])[0].shape),
                        gate_dtype=str(pair[0].dtype),
                        down_dtype=str(pair[1].dtype),
                        gate_weight_memory=str(pair[0].memory_config()),
                        down_weight_memory=str(pair[1].memory_config()),
                        gate_program=None if shared_gate is None else str(shared_programs[shared_gate]),
                        down_program=str(shared_programs["down44"]) if shared_down else None,
                        gate_compute=None if shared_gate is None else str(shared_compute),
                        down_compute=str(shared_compute) if shared_down else None,
                        decode="Original separate decode weights/programs unchanged",
                    )
                    for layer_index, pair in zip(gen.model.layer_indices, shared_weights)
                ]
            record["expert_configs"] = [
                dict(
                    layer=layer_index,
                    layer_type=gen.model.config.layer_types[layer_index],
                    gate_k_tiles={str(rows): config[0].in0_block_w for rows, config in expert.prefill_configs.items()},
                    gate32=str(expert.prefill_configs[32][0]),
                    down32=str(expert.prefill_configs[32][1]),
                    gate_dtype=str(expert.prefill_gate.dtype),
                    down_dtype=str(expert.prefill_down.dtype),
                    compute=str(expert.prefill_compute),
                )
                for layer_index, expert in zip(gen.model.layer_indices, experts)
            ]
            report["candidates"].append(record)
            sequence = [(True, 0)] * args.warmups + [(False, index % len(tokens)) for index in range(args.requests)]
            for index, (warmup, prompt_index) in enumerate(sequence):
                if record["forced_page_copies"] and gen.prefill_prepared is not None:
                    # Invalidate host comparison snapshots only. Device table
                    # buffers and actual scheduler page IDs remain untouched.
                    for snapshot in gen._tables(gen.prefill_prepared["table_host"]):
                        snapshot.fill_(-1)
                before = dict(gen.counters)
                first, prefill_ms = timed(
                    lambda: adapter.prefill_forward(
                        tokens[prompt_index],
                        table,
                        cache,
                        [args.length],
                        sampling_params=prefill_params,
                        page_tables_per_layer=tables,
                        empty_slots=[0],
                    )
                )
                second, decode_ms = timed(
                    lambda: adapter.decode_forward(
                        first,
                        torch.tensor([args.length], dtype=torch.int32),
                        table,
                        cache,
                        sampling_params=decode_params,
                        page_tables_per_layer=tables,
                        reset_batch=True,
                    )
                )
                observed = [int(first.item()), int(second.item())]
                if name == "32cached" and prompt_index not in expected:
                    expected[prompt_index] = observed
                result = dict(
                    index=index,
                    warmup=warmup,
                    prompt_index=prompt_index,
                    output_tokens=observed,
                    matches_baseline=observed == expected[prompt_index],
                    prefill_ms=prefill_ms,
                    first_decode_ms=decode_ms,
                    counters={key: value - before.get(key, 0) for key, value in gen.counters.items()},
                )
                record["requests"].append(result)
                save()
                assert gen.cache is cache, "Candidate changed the external cache binding"
                assert result["matches_baseline"], result
            measured = [result for result in record["requests"] if not result["warmup"]]
            record["median_prefill_ms"] = statistics.median(result["prefill_ms"] for result in measured)
            record["median_first_decode_ms"] = statistics.median(result["first_decode_ms"] for result in measured)
            save()
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        save()
        if gen is not None:
            gen.teardown()
        for expert, configs, width in zip(experts, original_configs, original_widths):
            expert.prefill_configs = configs
            expert.short_prefill_batch_tokens = width
        for layer, original in zip(shared_layers, original_shared):
            layer.gate_up, layer.down = original
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
