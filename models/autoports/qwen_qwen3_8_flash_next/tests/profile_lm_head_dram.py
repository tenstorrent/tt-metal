# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated real-weight LM-head DRAM-sharded candidate; no model policy edits.

Device owner only, no automatic launch. Clean environment:
  TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto \
  timeout 600 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.profile_lm_head_dram \
  --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/lm_head_dram_v1.json

Only real LM-head weights load. Inputs are seeded, replicated normalized BF16
rows, not full-model activations. All dtypes/fidelity/DEST/packer flags match the
current LM head. Reuse accepted DRAM-projection layout/program machinery with
three readers/bank; its default packer flag is explicitly replaced by the
existing LM-head compute config. Default block1 limits wide-N L1 demand.
--blocks1,2 is deliberately NOT a fallback: unsupported candidate aborts.
Separate --tracked-correctness from timing. Sixteen warmed traced iterations
include input reshard, projection, output restore/slice, physical32 padding and
persistent output copy; a second window adds common split-greedy sampling.
"""
import argparse
import json
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.profile_terminal_row_assembly import preflight


def metrics(reference, actual):
    assert reference.shape == actual.shape
    assert bool(torch.isfinite(reference).all() and torch.isfinite(actual).all())
    a, b = reference.double(), actual.double()
    ac, bc = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    pcc = (ac * bc).sum(-1) / (ac.norm(dim=-1) * bc.norm(dim=-1)).clamp_min(1e-100)
    chosen = b.argmax(-1, keepdim=True)
    result = {
        "pcc_min": float(pcc.min()),
        "pcc_mean": float(pcc.mean()),
        "max_abs": float((a - b).abs().max()),
        "exact": bool(torch.equal(reference, actual)),
    }
    for k in (1, 5, 100):
        result[f"top{k}"] = float((a.topk(k, dim=-1).indices == chosen).any(-1).double().mean())
    result["passed"] = result["pcc_min"] >= 0.999 and result["top1"] >= 0.98 and result["top100"] == 1.0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            os.environ.get(
                "QWEN_CHECKPOINT",
                "/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa",
            )
        ),
    )
    parser.add_argument("--blocks", default="1")
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--tracked-correctness", action="store_true")
    args = parser.parse_args()
    blocks = tuple(int(value) for value in args.blocks.split(","))
    assert blocks and all(value in (1, 2) for value in blocks) and len(set(blocks)) == len(blocks)
    assert args.iterations > 0 and not args.output.exists()
    preflight(args.tracked_correctness)
    from transformers import AutoConfig
    from ttnn.tools import trace_allocation_tracker as tracker

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt.dram_projection import (
        DramProjectionConfig,
        _PreparedGroup,
        weight_memory_plan,
    )
    from models.autoports.qwen_qwen3_8_flash_next.tt.model import Checkpoint
    from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import Collectives, MeshLoader
    from models.common.sampling.generator import SamplingGenerator, SamplingParams, format_sampling_params

    if not args.tracked_correctness:
        assert not tracker.TRACE_ALLOC_TRACKING and not tracker.TRACE_ALLOC_DIAGNOSTICS
    report = {
        "passed": False,
        "scope": "isolated_real_weight_LM_head_synthetic_BF16_input",
        "acceptance_eligible": False,
        "checkpoint": str(args.checkpoint),
        "tracked_correctness": args.tracked_correctness,
        "checks": {},
        "timings": {},
        "preserved_compute": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "activation": "BF16",
            "weight": "BF16",
            "output": "BF16",
        },
    }
    mesh, traces, samplers = None, {}, {}
    started = time.monotonic()
    phase = "checkpoint"

    def mark(value):
        nonlocal phase
        phase = value
        print(json.dumps({"phase": value, "wall_seconds": time.monotonic() - started}), flush=True)

    try:
        torch.set_num_threads(8)
        source = Checkpoint(args.checkpoint)
        config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True).text_config
        key = "model.language_model.embed_tokens.weight" if config.tie_word_embeddings else "lm_head.weight"
        head = source.tensor(key)
        assert tuple(head.shape) == (248320, 2560)
        report["weight"] = {"key": key, "shape": list(head.shape), "file": source.index[key], "dtype": str(head.dtype)}
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
        loader = MeshLoader({}, mesh)
        parts = list(head.chunk(4, dim=0))
        native_weight = loader.shards([part.T.contiguous() for part in parts], dim=1)
        projections = {
            "native": lambda x: ttnn.matmul(
                x, native_weight, compute_kernel_config=loader.compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        }
        weight_cache = {}
        report["candidates"] = {}
        mark("prepare_dram_layout")
        for block in blocks:
            policy = DramProjectionConfig(
                cores=8,
                readers=3,
                block=block,
                dtype="bfloat16",
                fidelity="HiFi4",
                packed=False,
                activation_dtype="bfloat16",
            )
            group = _PreparedGroup(loader, [[part] for part in parts], policy, weight_cache)
            # Preserve full-model flags; accepted decoder helper defaults packer=True.
            group.compute = loader.compute
            name = f"dram_readers3_block{block}"
            projections[name] = lambda x, group=group: group(x)[0]
            report["candidates"][name] = {
                "policy": asdict(policy),
                "compute_override": report["preserved_compute"],
                "weight_memory": weight_memory_plan(
                    2560, [62080], policy, native_dtype="bfloat16", native_packed=False
                ),
            }
        del head, parts
        generator = torch.Generator().manual_seed(127)
        host_input = torch.randn(1, 1, 32, 2560, generator=generator)
        host_input = (host_input / host_input.square().mean(-1, keepdim=True).sqrt()).to(torch.bfloat16)
        inputs = {
            count: ttnn.from_torch(
                host_input[:, :, :count].contiguous(),
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            for count in (1, 32)
        }
        outputs = {
            name: ttnn.empty(
                (1, 1, 32, 62080),
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for name in projections
        }
        tokens = {
            name: ttnn.from_torch(
                torch.zeros(1, 1, 1, 32, dtype=torch.int32),
                device=mesh,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            for name in projections
        }
        ccl = Collectives(mesh)
        ccl.prepare_sampling_collectives()
        sampler_config = SimpleNamespace(
            vocab_size=248320,
            padded_vocab_size=248320,
            max_batch_size=32,
            max_top_k=32,
            cluster_shape=(1, 4),
            sampling_all_gather_axis=1,
            model_config={
                "SAMPLING_AG_CONFIG": {"allow_force_argmax": False, "num_links": 1, "topology": ttnn.Topology.Ring}
            },
        )
        for name in projections:
            sampler = SamplingGenerator(args=sampler_config, mesh_device=mesh, tt_ccl=ccl)
            sampler.tt_sampling._allow_penalties_sampling = False
            sampler.reset_sampling_params(
                format_sampling_params(SamplingParams(temperature=1.0, top_k=1, top_p=0.0), 32)
            )
            samplers[name] = sampler

        def project(name, count):
            value = projections[name](inputs[count])
            if count != 32:
                value = ttnn.pad(value, [(0, 0), (0, 0), (0, 32 - count), (0, 0)], value=-100.0)
            ttnn.copy(value, outputs[name])

        def sample(name):
            samplers[name]._run_sampling(outputs[name], penalties_on=False, grammar_on=False, tt_out_tok=tokens[name])

        mark("warm_all_variants")
        for name in projections:
            for count in inputs:
                project(name, count)
            samplers[name].precompile(
                outputs[name], tt_out_tok=tokens[name], all_configs=False, compile_token_update=True
            )
            sample(name)
        ttnn.synchronize_device(mesh)
        cache_count = mesh.num_program_cache_entries()

        def capture(key, body):
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                body()
            except BaseException:
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                ttnn.release_trace(mesh, trace)
                raise
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            traces[key] = trace

        mark("capture")
        for name in projections:
            for count in inputs:
                capture((name, count), lambda name=name, count=count: project(name, count))
            capture((name, "sample"), lambda name=name: sample(name))
        assert mesh.num_program_cache_entries() == cache_count

        def replay(key):
            ttnn.execute_trace(mesh, traces[key], cq_id=0, blocking=False)

        mark("correctness")
        passed = True
        for count in inputs:
            values, observed = {}, {}
            for name in projections:
                replay((name, count))
                values[name] = ttnn.to_torch(outputs[name], mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3))[
                    0, 0
                ].float()
                assert bool(torch.isfinite(values[name]).all())
                if count < 32:
                    assert (values[name][count:] == -100.0).all()
                replay((name, "sample"))
                shards = [ttnn.to_torch(t).flatten().long() for t in ttnn.get_device_tensors(tokens[name])]
                assert all(torch.equal(shards[0], shard) for shard in shards[1:])
                assert torch.equal(
                    values[name][torch.arange(count), shards[0][:count]], values[name][:count].max(-1).values
                )
                observed[name] = shards[0][:count].tolist()
            report["checks"][str(count)] = {}
            for name in projections:
                if name == "native":
                    continue
                stats = metrics(values["native"][:count], values[name][:count])
                stats["common_greedy_tokens_equal"] = observed[name] == observed["native"]
                stats["tokens"] = {"native": observed["native"], "candidate": observed[name]}
                report["checks"][str(count)][name] = stats
                passed &= stats["passed"]
        if not args.tracked_correctness:
            mark("clean_traced_timing")

            def timing(operation):
                for _ in range(2):
                    operation()
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                for _ in range(args.iterations):
                    operation()
                ttnn.synchronize_device(mesh)
                return (time.perf_counter() - start) * 1000 / args.iterations

            for count in inputs:
                report["timings"][str(count)] = {}
                for name in projections:

                    def end_to_end(name=name, count=count):
                        replay((name, count))
                        replay((name, "sample"))

                    report["timings"][str(count)][name] = {
                        "projection_with_layout_and_output_copy_ms": timing(
                            lambda name=name, count=count: replay((name, count))
                        ),
                        "projection_plus_split_greedy_ms": timing(end_to_end),
                    }
        assert mesh.num_program_cache_entries() == cache_count
        report["program_cache_entries"] = cache_count
        report["limitations"] = [
            "Synthetic normalized final-mixer input; no full-model accuracy or throughput claim",
            "Q32 covers prefill head geometry; current batched decode invokes individual Q1 heads",
            "Native TT LM head is the numerical reference; these are not HF accuracy metrics",
            "All layout conversions and output slicing are inside measured projection traces",
            "Host windows include submission and boundary synchronization; no token read",
        ]
        replay((next(reversed(projections)), 1))
        replay((next(reversed(projections)), "sample"))
        ttnn.synchronize_device(mesh)
        report["passed"] = bool(passed)
        assert passed, report["checks"]
    except BaseException as error:
        report.update(failed_phase=phase, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if mesh is not None:
            for trace in traces.values():
                ttnn.release_trace(mesh, trace)
            for sampler in samplers.values():
                sampler.reset_trace()
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
