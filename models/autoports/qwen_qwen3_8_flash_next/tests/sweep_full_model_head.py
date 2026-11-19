# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-policy LM-head geometry sweep, Q1/Q32, with real checkpoint weights.

Device owner only; run serially under timeout, never alongside another device job::

    timeout 1800 python -m models.autoports.qwen_qwen3_8_flash_next.tests.sweep_full_model_head \
      --output models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/head_sweep.json

``--plan-only`` emits the host-derived geometry table without importing TTNN.
``--candidates 8:3:1,8:3:4,16:3:4`` selects cores:readers:block triples; the
current 8:3:1 baseline is always included. All candidates retain BF16 weights,
BF16 inputs/outputs, HiFi4, FP32 DEST, and packer-off. Real final-mixer inputs
can be provided as a BF16 [32,2560] tensor with --activations and a JSON
--activation-provenance file. Otherwise this is explicitly synthetic geometry
smoke, never full-model quality evidence.

All candidate setup and warmup precedes capture; immutable DRAM weights are
shared across geometries. Unsupported host constraints are recorded as rejects.
Any device/runtime error records the exact phase/exception and aborts; remaining
candidates are not silently skipped. The device owner must recover the mesh
before retrying a selected subset. This program releases only its own traces.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
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


CURRENT = "c8_r3_b1"
DEFAULT_CANDIDATES = [(8, 1, 1), (8, 2, 1)] + [
    (cores, 3, block) for cores in (8, 16, 20) for block in (1, 2, 4, 5, 8, 10)
]


def geometry(cores, readers, block):
    """Mirror _PreparedGroup host constraints; leave dynamic L1 checks to TTNN."""
    k_tiles, n_tiles, banks = 80, 1940, 8
    reasons = []
    if cores < 1 or k_tiles % cores:
        reasons.append("K=2560 must tile-divide over activation storage cores (80 % cores == 0)")
    if readers not in (1, 2, 3) or block < 1:
        reasons.append("DramProjectionConfig requires readers in1..3 and block>=1")
    shard = k_tiles // cores if cores > 0 and k_tiles % cores == 0 else None
    if shard and readers == 1 and block > 0 and (k_tiles % block or (shard % block and block % shard)):
        reasons.append("Single-reader K block and activation shard must divide one another; block must divide K tiles")
    width = math.ceil(n_tiles / cores) if cores > 0 else None
    row = {
        "name": f"c{cores}_r{readers}_b{block}",
        "cores": cores,
        "readers": readers,
        "block": block,
        "K": 2560,
        "local_N": 62080,
        "physical_rows": 32,
        "activation_shard_tiles": shard,
        "output_shard_tiles": width,
        "weight_bank_tiles": math.ceil(n_tiles / banks),
        "host_constraint_errors": reasons,
        "status": "host_rejected" if reasons else "not_run",
        "source": "tt/dram_projection.py::_PreparedGroup; matmul_device_operation.cpp::validate_matmul_dram_sharded_config",
    }
    if readers in (2, 3) and block > 0:
        # Blackhole multi-core path, one M tile, FP32 destination, CK=1.
        # These are exact DFB-size arithmetic terms from the factory, not a
        # claim that dynamic L1 allocation will fit a particular core placement.
        bank_tiles = math.ceil(n_tiles / banks)
        own = math.ceil(bank_tiles / readers)
        passes = math.ceil(own / 8)
        columns_per_pass = math.ceil(own / passes)
        kb = min(block, k_tiles)
        row["multicore_DFB_bytes_per_worker"] = {
            "activation": math.ceil(k_tiles / kb) * kb * 2048,
            "two_weight_streams": 2 * 3 * kb * columns_per_pass * 2048,
            "output": own * 2048,
        }
        row["storage_bytes_per_core"] = None if shard is None else (shard + width) * 2048
        row[
            "l1_note"
        ] = "DFBs plus storage may overlap on worker cores; runtime allocation is authoritative, no heuristic rejection"
    return row


def write_report(output, report):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    with output.with_suffix(".csv").open("w", newline="") as handle:
        fields = [
            "candidate",
            "cores",
            "readers",
            "block",
            "status",
            "rows",
            "projection_ms",
            "split_greedy_ms",
            "pcc_min",
            "greedy_tokens_equal",
            "failure",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, candidate in report["candidate_table"].items():
            for count in (1, 32):
                timing = report.get("timings", {}).get(str(count), {}).get(name, {})
                check = report.get("checks", {}).get(str(count), {}).get(name, {})
                writer.writerow(
                    {
                        "candidate": name,
                        **{k: candidate[k] for k in ("cores", "readers", "block", "status")},
                        "rows": count,
                        "projection_ms": timing.get("projection_with_layout_and_output_copy_ms"),
                        "split_greedy_ms": timing.get("projection_plus_split_greedy_ms"),
                        "pcc_min": check.get("pcc_min"),
                        "greedy_tokens_equal": check.get("common_greedy_tokens_equal"),
                        "failure": candidate.get("error", "; ".join(candidate["host_constraint_errors"])),
                    }
                )


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
    parser.add_argument(
        "--candidates", help="Comma-separated cores:readers:block triples; default compact20-row matrix"
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--activations", type=Path)
    parser.add_argument("--activation-provenance", type=Path)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--tracked-correctness", action="store_true")
    args = parser.parse_args()
    try:
        triples = (
            DEFAULT_CANDIDATES
            if args.candidates is None
            else [tuple(map(int, item.split(":"))) for item in args.candidates.split(",")]
        )
    except ValueError:
        parser.error("Expected integer cores:readers:block triples")
    if not all(len(item) == 3 for item in triples):
        parser.error("Expected cores:readers:block triples")
    triples = list(dict.fromkeys([(8, 3, 1), *triples]))
    candidate_table = {row["name"]: row for row in [geometry(*item) for item in triples]}
    assert args.iterations > 0 and not args.output.exists() and not args.output.with_suffix(".csv").exists()
    if bool(args.activations) != bool(args.activation_provenance):
        parser.error("--activations and --activation-provenance are required together")
    if args.plan_only:
        write_report(args.output, {"passed": False, "plan_only": True, "candidate_table": candidate_table})
        return
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
        "scope": "isolated_real_weight_LM_head_geometry",
        "candidate_table": candidate_table,
        "iterations": args.iterations,
        "command": sys.argv,
        "regime": "serial",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True).splitlines(),
        "acceptance_eligible": False,
        "policy_selection_note": "Synthetic disagreement needs a real-activation/full-model control before rejecting an optimization",
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
    active_candidate = None

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
        for name, row in candidate_table.items():
            active_candidate = name
            if row["host_constraint_errors"]:
                continue
            mark(f"prepare:{name}")
            policy = DramProjectionConfig(
                cores=row["cores"],
                readers=row["readers"],
                block=row["block"],
                dtype="bfloat16",
                fidelity="HiFi4",
                packed=False,
                activation_dtype="bfloat16",
            )
            group = _PreparedGroup(loader, [[part] for part in parts], policy, weight_cache)
            # Preserve full-model flags; accepted decoder helper defaults packer=True.
            group.compute = loader.compute
            row["status"] = "prepared"
            row["shared_weight_identity"] = [id(weight) for weight in group.weights]
            projections[name] = lambda x, group=group: group(x)[0]
            report["candidates"][name] = {
                "policy": asdict(policy),
                "compute_override": report["preserved_compute"],
                "weight_memory": weight_memory_plan(
                    2560, [62080], policy, native_dtype="bfloat16", native_packed=False
                ),
            }
        active_candidate = None
        identities = [row["shared_weight_identity"] for row in candidate_table.values() if row["status"] == "prepared"]
        assert identities and all(value == identities[0] for value in identities), "Immutable weight reuse failed"
        del head, parts
        if args.activations:
            host_input = torch.load(args.activations, map_location="cpu", weights_only=True)
            assert isinstance(host_input, torch.Tensor) and host_input.numel() == 32 * 2560
            assert host_input.dtype == torch.bfloat16, "Supply recorded BF16 final-mixer output without renormalization"
            host_input = host_input.reshape(1, 1, 32, 2560).contiguous()
            report["input_provenance"] = json.loads(args.activation_provenance.read_text())
            report["input_source"] = str(args.activations)
            report["synthetic_input"] = False
        else:
            generator = torch.Generator().manual_seed(127)
            host_input = torch.randn(1, 1, 32, 2560, generator=generator)
            host_input = (host_input / host_input.square().mean(-1, keepdim=True).sqrt()).to(torch.bfloat16)
            report["input_source"] = "Seed127 normalized synthetic BF16 rows; --activations was not supplied"
            report["synthetic_input"] = True
        assert bool(torch.isfinite(host_input).all())
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
            active_candidate = name
            mark(f"warm:{name}")
            for count in inputs:
                project(name, count)
            samplers[name].precompile(
                outputs[name], tt_out_tok=tokens[name], all_configs=False, compile_token_update=True
            )
            sample(name)
        ttnn.synchronize_device(mesh)
        active_candidate = None
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
            active_candidate = name
            for count in inputs:
                capture((name, count), lambda name=name, count=count: project(name, count))
            capture((name, "sample"), lambda name=name: sample(name))
        assert mesh.num_program_cache_entries() == cache_count
        active_candidate = None

        def replay(key):
            ttnn.execute_trace(mesh, traces[key], cq_id=0, blocking=False)

        mark("correctness")
        passed = True
        for count in inputs:
            values, observed = {}, {}
            for name in projections:
                active_candidate = name
                mark(f"correctness:{name}:q{count}")
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
                stats = metrics(values[CURRENT][:count], values[name][:count])
                stats["reference"] = CURRENT
                stats["native_reference"] = metrics(values["native"][:count], values[name][:count])
                stats["common_greedy_tokens_equal"] = observed[name] == observed[CURRENT]
                stats["native_greedy_tokens_equal"] = observed[name] == observed["native"]
                stats["tokens"] = {
                    "current": observed[CURRENT],
                    "native": observed["native"],
                    "candidate": observed[name],
                }
                stats["passed"] &= stats["common_greedy_tokens_equal"]
                report["checks"][str(count)][name] = stats
                if name == CURRENT:
                    passed &= stats["native_reference"]["passed"] and stats["native_greedy_tokens_equal"]
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
                    active_candidate = name
                    mark(f"timing:{name}:q{count}")

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
        for name, row in candidate_table.items():
            if row["status"] == "prepared":
                row["status"] = (
                    "passed"
                    if all(report["checks"][str(count)][name]["passed"] for count in inputs)
                    else "requires_real_input_validation"
                    if report["synthetic_input"]
                    else "numerically_rejected"
                )
        report["limitations"] = [
            (
                "Synthetic normalized final-mixer input; geometry smoke only"
                if report["synthetic_input"]
                else "Recorded final-mixer inputs; isolated head correctness only"
            ),
            "No full-model accuracy or throughput claim",
            "Q32 covers prefill head geometry; current batched decode invokes individual Q1 heads",
            "Current DRAM head is the primary numerical reference, with a native TT control; these are not HF accuracy metrics",
            "All layout conversions and output slicing are inside measured projection traces",
            "Host windows include submission and boundary synchronization; no token read",
        ]
        replay((next(reversed(projections)), 1))
        replay((next(reversed(projections)), "sample"))
        ttnn.synchronize_device(mesh)
        report["passed"] = bool(passed)
        assert passed, report["checks"]
    except BaseException as error:
        if active_candidate in candidate_table:
            candidate_table[active_candidate].update(
                status="runtime_error", failed_phase=phase, error=f"{type(error).__name__}: {error}"
            )
        report[
            "retry_instruction"
        ] = "Runtime failure aborts this run. Device owner must inspect/recover mesh before a new --candidates subset; no silent continuation."
        report.update(failed_phase=phase, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        write_report(args.output, report)
        if mesh is not None:
            for trace in traces.values():
                ttnn.release_trace(mesh, trace)
            for sampler in samplers.values():
                sampler.reset_trace()
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
