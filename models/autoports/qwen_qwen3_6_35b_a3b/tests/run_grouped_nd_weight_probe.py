# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Eager, synthetic-input exactness microgate for grouped EP weight placement.

Uses the real layer0 EP weights and selected precision. This is not a model
quality, trace, async or performance test. Run with an external timeout after
sourcing lane-env.sh and applying the opt-in ep_weight_nd_sharding implementation.
Both owners use the unchanged legal 8x8 compute grid. Diagnostic ND-to-interleaved
weight conversions happen before the case loop and are deallocated immediately.
"""

import argparse
import gc
import hashlib
import json
import sys
import traceback
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import HF_REVISION, Checkpoint
from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import (
    CollectiveWorkspace,
    MultichipDecoder,
    dispatch_capacity,
)
from models.autoports.qwen_qwen3_6_35b_a3b.tt.precision import layer_precision, resolve_precision

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
NATIVE = REPO / "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu"
SLOTS = tuple(range(32))


def bits(tensor):
    assert tensor.dtype == torch.bfloat16, tensor.dtype
    assert torch.isfinite(tensor).all(), "Nonfinite native output"
    return tensor.contiguous().view(torch.int16)


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def exact(reference, candidate, label):
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise AssertionError(f"{label}: metadata differs")
    left, right = bits(reference), bits(candidate)
    if not torch.equal(left, right):
        unequal = left != right
        first = int(unequal.reshape(-1).nonzero()[0])
        raise AssertionError(
            f"{label}: {int(unequal.sum())} unequal BF16 bit patterns; first_flat_index={first}; "
            f"reference={reference.reshape(-1)[first].item()}, candidate={candidate.reshape(-1)[first].item()}"
        )


def exact_weight(reference, candidate, label):
    """Compare all decoded weight values at their original host precision."""
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise AssertionError(f"{label}: host weight metadata differs")
    assert torch.isfinite(reference).all() and torch.isfinite(candidate).all(), label
    left = reference.contiguous().view(torch.uint8)
    right = candidate.contiguous().view(torch.uint8)
    if not torch.equal(left, right):
        unequal = left != right
        first = int(unequal.reshape(-1).nonzero()[0])
        raise AssertionError(f"{label}: {int(unequal.sum())} differing decoded weight bytes; first_byte={first}")


def cases():
    activity_sets = ((), (31,), (0, 31), (1, 7, 17, 31), tuple(range(0, 32, 2)), SLOTS)
    shared = torch.tensor([0, 63, 64, 127, 128, 191, 192, 255], dtype=torch.int32).repeat(32, 1)
    spread = torch.arange(256, dtype=torch.int32).reshape(32, 8)
    for name, ids in (("shared_early_late", shared), ("spread", spread)):
        for active in activity_sets:
            yield f"{name}/active_{len(active)}", ids, active
    for rank in (0, 3):
        ids = (torch.arange(8, dtype=torch.int32) + rank * 64).repeat(32, 1)
        for active in ((31,), SLOTS):
            yield f"rank_{rank}_only/active_{len(active)}", ids, active


def regions(ids, active, rank):
    counts = torch.zeros(256, dtype=torch.int64)
    for row in active:
        assert len(set(ids[row].tolist())) == 8, "Routes must be unique within a token"
        for expert in ids[row].tolist():
            if expert // 64 == rank:
                counts[expert] += 1
    offset, result = 0, []
    for expert, count in enumerate(counts.tolist()):
        if count:
            result.append(dict(expert=expert, count=count, offset=offset))
            offset += ((count + 31) // 32) * 32
    assert offset <= dispatch_capacity(32)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, help="Optional local checkpoint directory")
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--output", type=Path, default=ROOT / "readiness_vllm/grouped_nd_weight_probe/result.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing probe evidence; choose a fresh --output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_paths = [*sorted((ROOT / "tt").glob("*.py")), Path(__file__).resolve()]
    source_paths += sorted(path for path in NATIVE.rglob("*") if path.suffix in (".cpp", ".hpp"))
    result = dict(
        status="RUNNING",
        command=sys.argv,
        seed=args.seed,
        checkpoint_revision=HF_REVISION,
        layer=0,
        mesh=[1, 4],
        execution="eager correctness-only; diagnostic host readbacks after each complete call",
        source_sha256={
            str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths
        },
        geometry=dict(
            reference_grid=[8, 8],
            candidate_grid=[8, 8],
            reference_gu_placement="DRAM interleaved",
            candidate_gu_placement="DRAM ND shards [32,64]",
            down_placement="DRAM interleaved for both owners",
            observed_compiled_blocking=None,
            row_mode_verified=False,
            source="native geometry.cpp and common.hpp::h_row_plan; final resource fallback is not observed here",
        ),
        comparison="Exact complete GU weight values, live native expert rows and all combined rows on all four chips",
        limitations=[
            "Seeded synthetic activations and explicit unique routes; not actual router outputs or model quality.",
            "No attention, recurrent state, KV state, logits, sampling, trace or vLLM execution is tested.",
            "Eager snapshots synchronize. No latency, async, throughput or compiled row-mode claim is made.",
            "Uninitialized expert-region padding is excluded; every live expert row is compared.",
            "Reduced bucket/state/trace and full40 exactness gates must follow any passing microgate.",
        ],
        weight_checks=[],
        cases=[],
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    mesh = None
    try:
        torch.set_num_threads(8)
        rng = torch.Generator().manual_seed(args.seed)
        activation = torch.randn((1, 32, 2048), generator=rng).to(torch.bfloat16)
        scores = torch.softmax(torch.randn((1, 1, 32, 8), generator=rng), dim=-1).to(torch.bfloat16)
        result.update(input_sha256=digest(activation), scores_sha256=digest(scores), torch_version=torch.__version__)
        selected = resolve_precision()
        result["selected_precision"] = selected
        checkpoint = Checkpoint(args.model_dir)
        result["checkpoint_path"] = str(checkpoint.path)
        weights = checkpoint.select("model.language_model.layers.0.")
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
        layers = {}
        for owner, nd_sharding in (("reference", False), ("candidate", True)):
            layers[owner] = MultichipDecoder.from_state_dict(
                weights,
                hf_config=checkpoint.config,
                layer_idx=0,
                mesh_device=mesh,
                workspace=CollectiveWorkspace(mesh),
                full_model_optimized=True,
                precision_config=layer_precision(selected, 0),
                ep_weight_nd_sharding=nd_sharding,
            )
            layers[owner].prepare_moe(32)
            assert layers[owner].ep_weight_nd_sharding is nd_sharding
        del weights
        result["layer_precision"] = {owner: layer.precision_summary() for owner, layer in layers.items()}
        assert result["layer_precision"]["reference"] == result["layer_precision"]["candidate"]
        result["fused_route_plan"] = False

        # ND host reads are supported by the generic accessor path (see the
        # ND roundtrip tests), but explicitly convert to the same interleaved
        # layout for this complete value comparison. This is diagnostic-only.
        # No dtype conversion is requested; discard every temporary immediately.
        for role in ("gate", "up"):
            for expert in range(64):
                control = layers["reference"].ep_weights[role][expert]
                nd_weight = layers["candidate"].ep_weights[role][expert]
                assert control.dtype == nd_weight.dtype
                assert tuple(control.shape) == tuple(nd_weight.shape) == (2048, 512)
                assert control.memory_config() == ttnn.DRAM_MEMORY_CONFIG
                assert nd_weight.memory_config() != ttnn.DRAM_MEMORY_CONFIG
                row = dict(role=role, local_expert=expert, status="RUNNING", device_dtype=str(control.dtype))
                result["weight_checks"].append(row)
                save()
                converted = None
                try:
                    converted = ttnn.to_memory_config(nd_weight, ttnn.DRAM_MEMORY_CONFIG)
                    assert converted.dtype == control.dtype and converted.layout == control.layout
                    assert tuple(converted.shape) == tuple(control.shape)
                    assert converted.memory_config() == ttnn.DRAM_MEMORY_CONFIG
                    controls = ttnn.get_device_tensors(control)
                    candidates = ttnn.get_device_tensors(converted)
                    assert len(controls) == len(candidates) == 4
                    hashes = []
                    for rank, (left, right) in enumerate(zip(controls, candidates)):
                        reference_host = ttnn.to_torch(left)
                        candidate_host = ttnn.to_torch(right)
                        exact_weight(reference_host, candidate_host, f"weight/{role}/{expert}/rank{rank}")
                        hashes.append(digest(reference_host))
                    row.update(status="PASS", exact_all_chips=True, sha256=hashes)
                finally:
                    if converted is not None:
                        ttnn.deallocate(converted)
                save()
            print("GROUPED_ND_WEIGHT_ROLE_PASS", role, flush=True)
        result["weight_comparison"] = dict(
            status="PASS",
            matrices_per_chip=128,
            chips=4,
            compared_elements=128 * 4 * 2048 * 512,
            method="Complete decoded host floating-point bit patterns after dtype-preserving ND-to-interleaved copy",
            packed_bfp_exponent_bytes_compared=False,
            diagnostic_conversion_temps_deallocated=True,
        )

        def upload(value, dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                value.contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        device_x = upload(activation, ttnn.bfloat16)
        device_scores = upload(scores, ttnn.bfloat16)
        active_mask = upload(torch.zeros(32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        result["persistent_activity_address"] = int(active_mask.buffer_address())

        def snapshot(owner, device_ids, by_rank):
            layer = layers[owner]
            raw_output = layer.ep_outputs[dispatch_capacity(32)]
            combined = layer._grouped_moe(
                device_x,
                device_ids,
                device_scores,
                stochastic_rounding=False,
                route_activity_mask=active_mask,
                route_slots=SLOTS,
            )
            ttnn.synchronize_device(mesh)
            combined_parts = ttnn.get_device_tensors(combined)
            expert_parts = ttnn.get_device_tensors(raw_output)
            assert len(combined_parts) == len(expert_parts) == 4
            output = []
            for rank, (combined_part, expert_part) in enumerate(zip(combined_parts, expert_parts)):
                combined_host = ttnn.to_torch(combined_part).clone()
                experts_host = ttnn.to_torch(expert_part).reshape(dispatch_capacity(32), 2048)
                live = [experts_host[row["offset"] : row["offset"] + row["count"]] for row in by_rank[rank]]
                live_host = torch.cat(live).clone() if live else torch.empty((0, 2048), dtype=torch.bfloat16)
                bits(combined_host)
                bits(live_host)
                output.append(dict(combined=combined_host, experts=live_host))
                del experts_host
            ttnn.deallocate(combined)
            return output

        for name, ids, active in cases():
            by_rank = [regions(ids, active, rank) for rank in range(4)]
            row = dict(name=name, status="RUNNING", active_slots=list(active), route_ids=ids.tolist(), regions=by_rank)
            result["cases"].append(row)
            save()
            mask = torch.zeros(32, dtype=torch.int32)
            mask[list(active)] = 1
            host_mask = ttnn.from_torch(
                mask,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            ttnn.copy_host_to_device_tensor(host_mask, active_mask)
            device_ids = upload(ids.reshape(1, 1, 32, 8), ttnn.uint32)
            reference = snapshot("reference", device_ids, by_rank)
            candidate = snapshot("candidate", device_ids, by_rank)
            # Owners have independent workspaces; re-running the control checks
            # that candidate execution has not corrupted its persistent weights.
            repeated = snapshot("reference", device_ids, by_rank)
            for rank in range(4):
                for kind in ("combined", "experts"):
                    exact(reference[rank][kind], candidate[rank][kind], f"{name}/rank{rank}/{kind}/candidate")
                    exact(reference[rank][kind], repeated[rank][kind], f"{name}/rank{rank}/{kind}/repeat")
            row.update(
                status="PASS",
                exact_all_chips=True,
                reference_repeat_exact=True,
                sha256=[{kind: digest(part[kind]) for kind in ("combined", "experts")} for part in reference],
            )
            ttnn.deallocate(device_ids)
            del reference, candidate, repeated
            gc.collect()
            save()
            print("GROUPED_ND_WEIGHT_CASE_PASS", name, flush=True)
        result.update(
            status="PASS", case_count=len(result["cases"]), program_entries=int(mesh.num_program_cache_entries())
        )
    except BaseException as error:
        if result["weight_checks"] and result["weight_checks"][-1]["status"] == "RUNNING":
            result["weight_checks"][-1].update(status="FAIL", error=repr(error))
        if result["cases"] and result["cases"][-1]["status"] == "RUNNING":
            result["cases"][-1].update(status="FAIL", error=repr(error))
        result.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.close_mesh_device(mesh)
                result["mesh_closed"] = True
        except BaseException as error:
            result.update(status="FAIL", cleanup_error=repr(error), cleanup_traceback=traceback.format_exc())
            raise
        finally:
            save()
    print("GROUPED_ND_WEIGHT_PROBE_PASS", args.output, flush=True)


if __name__ == "__main__":
    main()
