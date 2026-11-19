# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real-weight TP4 hidden-shard norm + activation-AG + canonical-input probe.

Two norm phases per trace preserve native CCL ping-pong ownership. This is an
isolated boundary experiment, not a substitute for whole-decoder quality gates.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--norm", choices=("input", "post"), default="input")
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--input-pt", type=Path)
    p.add_argument("--input-key", default="d")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--queued", type=int, default=50)
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--quality-only", action="store_true")
    args = p.parse_args()
    from models.autoports.qwen_qwen3_6_27b.tt.fused_distributed_norm_candidate import geometry

    policy = json.loads(args.policy.read_text())
    assert policy.get("residual_layout", "sharded") == "sharded"
    assert args.queued >= 1
    report = {
        "status": "prepared",
        "geometry": geometry(args.batch),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "policy": policy,
        "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "allocation_tracking": os.getenv("TT_METAL_TRACE_ALLOC_TRACKING", "0") == "1",
        "skip_program_cache": os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "1",
        "watcher": "TT_METAL_WATCHER" in os.environ,
        "profiler": os.getenv("TT_METAL_DEVICE_PROFILER", "0") != "0",
        "quality": [],
        "replays": [],
        "boundary": "hidden-sharded BF16 residual -> norm -> full-hidden activation AG -> canonical matmul input; optional complete real-weight projection included separately",
        "minimal_family": "measured separately by probe_fused_minimal_norm after native AutoFix",
        "verification_slot_memory": "DRAM; actual norm-to-projection canonical input remains L1",
        "source_sha256": {
            str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [Path(__file__), *sorted((ROOT / "tt").glob("*.py"))]
        },
    }
    native_root = ROOT.parents[2] / "ttnn/cpp/ttnn/operations/experimental/ccl"
    report["native_source_sha256"] = {
        str(f.relative_to(ROOT.parents[2])): hashlib.sha256(f.read_bytes()).hexdigest()
        for family in ("dit_fused_distributed_rmsnorm", "rms_allgather")
        for f in sorted((native_root / family).rglob("*"))
        if f.is_file() and f.suffix in (".cpp", ".hpp")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    if args.cpu_only:
        report.update(status="cpu-plan-pass", hardware_validated=False)
        save()
        return

    import torch
    import ttnn
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import load_layer
    from models.autoports.qwen_qwen3_6_27b.tests.multichip_assertions import pcc
    from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import MultichipDecoder
    from models.autoports.qwen_qwen3_6_27b.tt.fused_distributed_norm_candidate import DistributedNormBoundary

    @contextmanager
    def device_only():
        def forbidden(*_a, **_k):
            raise AssertionError("Host conversion/synchronization inside measured device boundary")

        with ExitStack() as stack:
            for name in (
                "from_torch",
                "to_torch",
                "as_tensor",
                "to_device",
                "from_device",
                "copy_host_to_device_tensor",
                "synchronize_device",
            ):
                if hasattr(ttnn, name):
                    stack.enter_context(patch.object(ttnn, name, forbidden))
            yield

    mesh, traces = None, {}
    try:
        assert not report["skip_program_cache"], "Skipping program-cache tracking is not permitted"
        assert not (report["watcher"] and report["profiler"])
        torch.set_num_threads(8)
        torch.manual_seed(17)
        config = Qwen3_5TextConfig(**json.loads((ROOT / "doc/hf_config.json").read_text())["text_config"])
        weights = load_layer(args.layer)
        key = "input_layernorm.weight" if args.norm == "input" else "post_attention_layernorm.weight"
        gamma_host = (1 + weights[key].float()).reshape(1, 1, 1, 5120)
        if args.input_pt:
            data = torch.load(args.input_pt, weights_only=True, map_location="cpu")
            raw = data[args.input_key] if isinstance(data, dict) else data
            assert raw.numel() == args.batch * 5120, "Supply one actual decode hidden residual per user"
            x = raw.reshape(args.batch, 1, 5120).to(torch.bfloat16)
            report["input_origin"] = {
                "file": str(args.input_pt),
                "key": args.input_key,
                "sha256": hashlib.sha256(args.input_pt.read_bytes()).hexdigest(),
            }
        else:
            x = torch.randn(args.batch, 1, 5120, dtype=torch.bfloat16) * 0.1
            report["input_origin"] = "seed17 synthetic BF16 residual with actual pinned model norm/projection weights"
        # Nonuniform additive changes avoid RMSNorm's sign/scale symmetry.
        changed = (x.float() * 0.75 + torch.randn(x.shape) * 0.03).to(torch.bfloat16)
        hosts = [x, changed]
        report["input_hashes"] = [
            hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for t in hosts
        ]
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=64 * 1024 * 1024)
        model = MultichipDecoder.from_state_dict(
            weights, hf_config=config, layer_idx=args.layer, mesh_device=mesh, policy=policy
        )
        assert model.hidden_width == 1280
        report["resolved_policy"], report["kind"] = model.policy, model.kind
        projection = (
            "mlp.gate_up.weight"
            if args.norm == "post"
            else (
                "packed_gdn"
                if model.kind == "linear_attention"
                else "packed_qkvg" if model.policy.get("packed_qkvg") else "packed_qkv"
            )
        )
        report["projection"] = projection
        banks = [model.upload_decode_residual(v) for v in hosts]
        active = [ttnn.clone(banks[0]), ttnn.clone(banks[0])]
        gamma = [
            ttnn.from_torch(
                gamma_host,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
            )
            for _ in range(2)
        ]
        owners = {
            "control": DistributedNormBoundary(
                model, active[0], gamma[0], key=key, projection_key=projection, family="control"
            ),
            "dit": DistributedNormBoundary(
                model, active[0], gamma[0], key=key, projection_key=projection, family="dit"
            ),
            "dit_rebound": DistributedNormBoundary(
                model, active[1], gamma[1], key=key, projection_key=projection, family="dit"
            ),
        }
        slots = {}

        def sync():
            ttnn.synchronize_device(mesh)

        # Finish every mesh-wide semaphore zero before the first fabric launch.
        sync()
        for name, owner in owners.items():
            inp = active[1 if name == "dit_rebound" else 0]
            for phase in range(2):
                temporary = owner.prepare(inp, phase)
                projected = owner.project(temporary)
                # Verification owns six norm/projection pairs simultaneously.
                # Keep those test-only snapshots in DRAM so their resident L1
                # does not crowd the native norm's CBs. The actual canonical
                # norm -> projection boundary above remains unchanged.
                slots[name, phase, "norm"] = ttnn.empty_like(temporary, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                slots[name, phase, "projection"] = ttnn.empty_like(projected, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                del temporary, projected
        sync()

        def set_bank(index):
            # Keep simultaneously live inputs distinct: stale cached input
            # bindings must not pass because both owners contain equal values.
            ttnn.copy(banks[index], active[0])
            ttnn.copy(banks[1 - index], active[1])

        def pair(name, project=False):
            owner = owners[name]
            inp = active[1 if name == "dit_rebound" else 0]
            with device_only():
                for phase in range(2):
                    norm = owner.prepare(inp, phase)
                    ttnn.copy(norm, slots[name, phase, "norm"])
                    if project:
                        out = owner.project(norm)
                        ttnn.copy(out, slots[name, phase, "projection"])
                        del out
                    del norm

        def read(tensor):
            parts = [ttnn.to_torch(t).float().reshape(args.batch, -1) for t in ttnn.get_device_tensors(tensor)]
            assert len(parts) == 4
            return parts

        def snapshot(name, project=True):
            return {
                kind: [read(slots[name, phase, kind]) for phase in range(2)]
                for kind in (("norm", "projection") if project else ("norm",))
            }

        eager = {}
        for bank in range(2):
            set_bank(bank)
            for name in owners:
                pair(name, True)
                sync()
                eager[name, bank] = snapshot(name)
                pair(name, False)  # Warm the exact projection-free signature too.
                sync()
        # Both rebound data and result owners exist before ANY capture.
        report["owners"] = {name: o.ownership() for name, o in owners.items()}
        report["active_addresses"] = [int(t.buffer_address()) for t in active]
        report["slots"] = {str(k): int(v.buffer_address()) for k, v in slots.items()}
        bad = []
        for bank in range(2):
            for name in owners:
                actual_bank = 1 - bank if name == "dit_rebound" else bank
                v = hosts[actual_bank].double().reshape(args.batch, 5120)
                reference = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + config.rms_norm_eps)
                reference *= gamma_host.double().reshape(1, 5120)
                for kind in ("norm", "projection"):
                    for phase in range(2):
                        for rank in range(4):
                            actual = eager[name, bank][kind][phase][rank]
                            control = eager["control", actual_bank][kind][phase][rank]
                            scores = [pcc(control[u], actual[u]) for u in range(args.batch)]
                            row = {
                                "bank": actual_bank,
                                "input_selector": bank,
                                "path": name,
                                "phase": phase,
                                "rank": rank,
                                "kind": kind,
                                "per_user_vs_control": scores,
                            }
                            if kind == "norm":
                                row["per_user_vs_fp64_rms"] = [pcc(reference[u], actual[u]) for u in range(args.batch)]
                                scores = scores + row["per_user_vs_fp64_rms"]
                            row["pass"] = min(scores) >= 0.995 and bool(torch.isfinite(actual).all())
                            report["quality"].append(row)
                            if not row["pass"]:
                                bad.append(row)
        save()
        assert not bad, bad
        assert any(
            not torch.equal(a, b) for a, b in zip(eager["control", 0]["norm"][0], eager["control", 1]["norm"][0])
        )
        # The old gamma owner is deliberately poisoned before rebinding. Its
        # values differ, while the measured/new owner retains the actual FP32
        # model weight. This is setup-only and restored before every capture.
        gamma_zero = ttnn.from_torch(
            torch.zeros_like(gamma_host),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        )
        set_bank(0)
        ttnn.copy(gamma_zero, gamma[0])
        pair("dit", True)
        sync()
        old_slots = snapshot("dit")
        old_stats = [[ttnn.to_torch(t).clone() for t in ttnn.get_device_tensors(s)] for s in owners["dit"].stats]
        assert all(torch.count_nonzero(t) == 0 for parts in old_slots["norm"] for t in parts)
        pair("dit_rebound", True)
        sync()
        rebound = snapshot("dit_rebound")
        old_after = snapshot("dit")
        rebound_ok = all(
            torch.equal(t, eager["dit_rebound", 0][kind][phase][rank])
            for kind, phases in rebound.items()
            for phase, parts in enumerate(phases)
            for rank, t in enumerate(parts)
        )
        old_output_untouched = all(
            torch.equal(t, old_slots[kind][phase][rank])
            for kind, phases in old_after.items()
            for phase, parts in enumerate(phases)
            for rank, t in enumerate(parts)
        )
        old_stats_untouched = all(
            torch.equal(
                ttnn.to_torch(t).contiguous().view(torch.uint8), old_stats[phase][rank].contiguous().view(torch.uint8)
            )
            for phase, s in enumerate(owners["dit"].stats)
            for rank, t in enumerate(ttnn.get_device_tensors(s))
        )
        report["rebind_poison"] = {
            "old_gamma_zero_new_gamma_actual_fp32": True,
            "simultaneously_distinct_inputs": True,
            "new_owner_exact": rebound_ok,
            "old_output_untouched": old_output_untouched,
            "old_stats_untouched": old_stats_untouched,
        }
        save()
        assert rebound_ok and old_output_untouched and old_stats_untouched, report["rebind_poison"]
        ttnn.copy(gamma[1], gamma[0])
        for name in owners:
            pair(name, True)
            pair(name, False)
        sync()
        mesh.set_program_cache_misses_allowed(False)
        for name in owners:
            for project in (False, True):
                set_bank(0)
                sync()
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                pair(name, project)
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                traces[name, project] = tid

        def allocations():
            if report["allocation_tracking"]:
                from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker

                for tid in traces.values():
                    TraceAllocationTracker.verify_before_replay(mesh, tid)

        allocations()
        if report["allocation_tracking"]:
            from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker

            extra = ttnn.empty_like(active[0])
            uid = extra.buffer_unique_id()
            caught = False
            try:
                TraceAllocationTracker.verify_before_replay(mesh, next(iter(traces.values())))
            except RuntimeError as error:
                assert "device buffer(s) still alive before trace replay" in str(error)
                assert f"Buffer {uid}" in str(error)
                caught = True
            assert caught, "Trace allocation positive control failed"
            ttnn.deallocate(extra)
            del extra
            allocations()
            report["allocation_positive_control"] = "rejected live allocation; all six traces clean after release"

        def replay(name, project):
            allocations()
            ttnn.execute_trace(mesh, traces[name, project], cq_id=0, blocking=False)

        def exact(name, bank, project, label):
            actual = snapshot(name, project)
            checks = [
                torch.equal(t, eager[name, bank][kind][phase][rank])
                for kind, phases in actual.items()
                for phase, parts in enumerate(phases)
                for rank, t in enumerate(parts)
            ]
            row = {
                "path": name,
                "input_selector": bank,
                "bank": 1 - bank if name == "dit_rebound" else bank,
                "projection": project,
                "label": label,
                "all_ranks_phases_exact_to_own_eager": all(checks),
            }
            report["replays"].append(row)
            save()
            assert all(checks), row

        for name in owners:
            for project in (False, True):
                for bank in (1, 0, 1, 0):
                    set_bank(bank)
                    replay(name, project)
                    sync()
                    exact(name, bank, project, "changed_input")
                for i in range(args.queued):
                    set_bank(i % 2)
                    replay(name, project)
                sync()
                exact(name, (args.queued - 1) % 2, project, "queued_changed_final")
        instrumented = report["allocation_tracking"] or report["watcher"] or report["profiler"]
        report["timing"] = {
            "enabled": not (instrumented or args.quality_only),
            "units": "us/call",
            "calls_per_sample": 2,
            "samples": 7,
            "projection_free_boundary_includes_canonical_slot_copy": True,
        }
        if report["timing"]["enabled"]:
            set_bank(0)
            measurements = {}
            for name in ("control", "dit"):
                for project in (False, True):
                    for traced in (False, True):
                        samples = []
                        for _ in range(7):
                            sync()
                            start = time.perf_counter_ns()
                            replay(name, project) if traced else pair(name, project)
                            sync()
                            samples.append((time.perf_counter_ns() - start) / 2000)
                        measurements[str((name, project, traced))] = {
                            "samples": samples,
                            "median": statistics.median(samples),
                        }
            report["timing"]["measurements"] = measurements
        # Eager interleaving must not produce live late cache allocations or stale CCL state.
        for name in owners:
            set_bank(1)
            pair(name, True)
        sync()
        for name in owners:
            replay(name, True)
            sync()
            exact(name, 1, True, "final_after_eager_interleave")
        allocations()
        report.update(
            status="pass",
            hardware_validated=True,
            queued_pairs=args.queued,
            queued_checks="final output plus four individual changed-bank replays",
            stage_default_changed=False,
            whole_decoder_gate_substitute=False,
        )
    except BaseException:
        report.update(status="fail", traceback=traceback.format_exc())
        save()
        raise
    finally:
        if mesh is not None:
            for tid in traces.values():
                ttnn.release_trace(mesh, tid)
            ttnn.close_mesh_device(mesh)
            report["mesh_closed"] = True
        save()


if __name__ == "__main__":
    main()
