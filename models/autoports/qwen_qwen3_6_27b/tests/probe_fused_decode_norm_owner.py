# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Actual two-phase fused decode norm owner: bounded batches, rebinding and traces.

CPU preparation imports neither torch nor TTNN. Hardware execution is a focused
owner/projection check, not a full-decoder gate or a performance benchmark.
"""

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import traceback
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
KEYS = ("input_layernorm.weight", "post_attention_layernorm.weight")


def parse_batches(value):
    batches = tuple(int(v) for v in value.split(","))
    if not batches or len(set(batches)) != len(batches) or any(not 1 <= b <= 32 for b in batches):
        raise argparse.ArgumentTypeError("Use distinct logical batches in1..32")
    return batches


def cpu_plan(batches, family, layer):
    assert 0 <= layer < 64
    assert family in ("dit", "minimal", "both")
    assert all(1 <= b <= 32 for b in batches)
    families = ("dit", "minimal") if family == "both" else (family,)
    # Two phase aliases for each legal batch must cover exactly one physical row tile.
    shapes = []
    for b in range(1, 33):
        for width in (1280, 5120):
            assert ((b + 31) // 32) * 32 * width == 32 * width
            shapes.append({"logical": [1, 1, b, width], "padded": [1, 1, 32, width]})
    return {
        "families": list(families),
        "owners_per_family": 2,
        "phase_keys": list(KEYS),
        "real_layer_index": layer,
        "minimal_worker_row": layer % 10,
        "trace_batches": list(batches),
        "metadata_batches": list(range(1, 33)),
        "metadata_geometries": shapes,
        "device_storage_signatures_per_owner": 1,
        "calls_per_trace": 2,
        "verification_memory": "DRAM",
        "reference": "FP64 global RMS with actual FP32 gamma; BF16 reference through unchanged native real-weight projection",
        "minimum_per_rank_user_pcc": 0.995,
    }


def projection_groups(kind, phase, packed_qkvg):
    if phase == 1:
        return [("gate", 0, 4352), ("up", 4352, 8704)]
    if kind == "linear_attention":
        result = [
            (f"{name}_head{head}", start + head * 128, start + (head + 1) * 128)
            for name, start, heads in (("q", 0, 4), ("k", 512, 4), ("v", 1024, 12), ("z", 2560, 12))
            for head in range(heads)
        ]
        return result + [("beta", 4096, 4108), ("decay", 4128, 4140)]
    assert kind == "full_attention"
    result = [(f"q_head{h}", h * 256, (h + 1) * 256) for h in range(6)]
    result += [("k", 1536, 1792), ("v", 1792, 2048)]
    if packed_qkvg:
        result += [(f"gate_head{h}", 2048 + h * 256, 2048 + (h + 1) * 256) for h in range(6)]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--family", choices=("dit", "minimal", "both"), default="minimal")
    p.add_argument("--batches", type=parse_batches, default=parse_batches("1,3,17,32"))
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--queued", type=int, default=50)
    p.add_argument("--trace-mib", type=int, default=64)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpu-only", action="store_true")
    args = p.parse_args()
    assert args.queued >= 1 and args.trace_mib > 0
    plan = cpu_plan(args.batches, args.family, args.layer)
    policy = json.loads(args.policy.read_text())
    assert (
        isinstance(policy, dict) and policy.get("minimal_swiglu", "prefill") == "prefill"
    ), "Provide the complete selected per-layer policy with minimal_swiglu='prefill'"
    assert policy.get("residual_layout", "sharded") == "sharded"
    report = {
        "status": "prepared",
        "plan": plan,
        "command": shlex.join([sys.executable, "-m", __spec__.name, *sys.argv[1:]]),
        "policy": policy,
        "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "quality": [],
        "replay_checks": [],
        "rebind_checks": [],
        "tracker_enabled": os.getenv("TT_METAL_TRACE_ALLOC_TRACKING", "0") == "1",
        "skip_program_cache": os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "1",
        "watcher": "TT_METAL_WATCHER" in os.environ,
        "profiler": os.getenv("TT_METAL_DEVICE_PROFILER", "0") != "0",
        "timing": {
            "enabled": False,
            "reason": "Focused correctness/resource probe; measure unchanged whole decoder separately",
        },
        "source_sha256": {
            str(f.relative_to(REPO)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [
                Path(__file__),
                ROOT / "tests/run_decoder.py",
                ROOT / "tests/multichip_assertions.py",
                ROOT / "doc/hf_config.json",
                *sorted((ROOT / "tt").glob("*.py")),
            ]
        },
    }
    native_root = REPO / "ttnn/cpp/ttnn/operations/experimental/ccl"
    report["native_source_sha256"] = {
        str(f.relative_to(REPO)): hashlib.sha256(f.read_bytes()).hexdigest()
        for family in ("dit_fused_distributed_rmsnorm", "dit_fused_norm_common", "rms_allgather")
        for f in sorted((native_root / family).rglob("*"))
        if f.is_file() and f.suffix in (".cpp", ".hpp")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    if args.cpu_only:
        helper = ROOT / "tt/fused_decode_norm.py"
        if not helper.exists():
            helper = ROOT / "doc/optimized_multichip_decoder/fused_decode_norm_runtime_candidate.py"
        tree = ast.parse(helper.read_text())
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecodeNormOwner")
        assert {"normalize", "ownership", "_views"} <= {n.name for n in owner.body if isinstance(n, ast.FunctionDef)}
        report.update(
            status="cpu-plan-pass",
            hardware_validated=False,
            inspected_helper=str(helper.relative_to(REPO)),
            inspected_helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        )
        save()
        return

    import torch
    import ttnn
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from .run_decoder import load_layer, MODEL, REVISION
    from .multichip_assertions import pcc
    from .check_optimized_multichip_evidence import loaded_native_snapshot
    from ..tt.multichip_decoder import MultichipDecoder, _BlackholeCCLManager
    from ..tt.fused_decode_norm import install

    @contextmanager
    def device_only():
        def forbidden(*_a, **_kw):
            raise AssertionError("Host conversion/synchronization in norm/projection boundary")

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

    def digest_host(t):
        return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    mesh, traces = None, {}
    try:
        assert report["tracker_enabled"], "This owner validation requires effective allocation tracking"
        assert not report["skip_program_cache"], "Do not skip program-cache allocation tracking"
        assert not report["profiler"], "Profiler collection belongs to the separate performance job"
        torch.set_num_threads(8)
        torch.manual_seed(17)
        cfg = Qwen3_5TextConfig(**json.loads((ROOT / "doc/hf_config.json").read_text())["text_config"])
        weights = load_layer(args.layer)
        gamma_host = [(1 + weights[k].float()).reshape(1, 5120) for k in KEYS]
        report["weight_provenance"] = {
            "model": MODEL,
            "revision": REVISION,
            "layer": args.layer,
            "actual_fp32_affine_sha256": {k: digest_host(g) for k, g in zip(KEYS, gamma_host)},
        }
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_mib * 1024 * 1024)
        # Base is a weight/projection owner, never a normalization oracle.
        base_policy = {**policy, "minimal_swiglu": "prefill", "distributed_decode_norm": "off"}
        base = MultichipDecoder.from_state_dict(
            weights, hf_config=cfg, layer_idx=args.layer, mesh_device=mesh, policy=base_policy
        )
        report["resolved_base_policy"], report["kind"] = base.policy, base.kind
        semantic_keys = (
            (
                "packed_gdn"
                if base.kind == "linear_attention"
                else "packed_qkvg" if base.policy.get("packed_qkvg") else "packed_qkv"
            ),
            "mlp.gate_up.weight",
        )
        models = {}
        for family in plan["families"]:
            for rebound in (False, True):
                name = family + ("_rebound" if rebound else "")
                model = copy.copy(base)
                model.policy = {**base.policy, "distributed_decode_norm": family}
                model.layer_idx = args.layer
                model.shard_norms = {k: ttnn.clone(base.shard_norms[k]) for k in KEYS}
                model._ccl_buffers = {}
                model._prefill_batch_indices = {}
                model.ccl = _BlackholeCCLManager(mesh, base.ccl.num_links, base.ccl.topology)
                install(model, layer_idx=args.layer)
                models[name] = model

        def sync():
            ttnn.synchronize_device(mesh)

        def rank_addresses(tensor):
            addresses = [int(t.buffer_address()) for t in ttnn.get_device_tensors(tensor)]
            assert len(addresses) == 4
            return addresses

        # Independently verify all32 metadata views, not just traced batches.
        report["backing_identity"] = []
        for name, model in models.items():
            owner = model._fused_decode_norm
            for label, bases, views, width in (
                ("activation_ag", owner.ag_outputs, owner.ag_views, 5120),
                ("native_norm", owner.norm_outputs, owner.norm_views, 1280),
            ):
                if not bases:
                    continue
                assert set(views) == set(range(1, 33))
                for b in range(1, 33):
                    for phase in (0, 1):
                        view, backing = views[b][phase], bases[phase]
                        assert tuple(view.shape) == (1, 1, b, width)
                        assert tuple(view.padded_shape) == (1, 1, 32, width)
                        assert rank_addresses(view) == rank_addresses(backing)
                        assert view.memory_config() == backing.memory_config()
                        report["backing_identity"].append(
                            {
                                "owner": name,
                                "kind": label,
                                "batch": b,
                                "phase": phase,
                                "rank_addresses": rank_addresses(view),
                                "same_storage": True,
                            }
                        )
        report["owners_before"] = {n: m._fused_decode_norm.ownership() for n, m in models.items()}
        for rank in range(4):
            addresses = [
                t["rank_addresses"][rank] for o in report["owners_before"].values() for t in o["persistent_tensors"]
            ]
            assert len(addresses) == len(set(addresses)), (rank, "persistent owner alias")
        semaphores = [a for o in report["owners_before"].values() for a in o["semaphores"]]
        assert len(semaphores) == len(set(semaphores))

        hosts, banks, active, slots, references = {}, {}, {}, {}, {}
        for b in args.batches:
            for phase in (0, 1):
                original = torch.randn(b, 1, 5120, dtype=torch.bfloat16) * (0.1 + phase * 0.03)
                changed = (original.float() * 0.75 + torch.randn(original.shape) * 0.04).to(torch.bfloat16)
                for bank, host in enumerate((original, changed)):
                    hosts[b, phase, bank] = host
                    banks[b, phase, bank] = base.upload_decode_residual(host)
                    x = host.double().reshape(b, 5120)
                    ref = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
                    ref = ref * gamma_host[phase].double()
                    references[b, phase, bank, "norm"] = [ref.clone() for _ in range(4)]
                    # Independent normalized activation, unchanged real-weight native projection.
                    uploaded = base._upload(ref.to(torch.bfloat16).reshape(1, b, 5120))
                    canonical_key = next(iter(models.values()))._fused_decode_norm.projections[phase]
                    canonical = ttnn.to_memory_config(uploaded, base.projection_info[canonical_key]["input_mem"])
                    projected = base._linear(canonical, semantic_keys[phase])
                    sync()
                    references[b, phase, bank, "projection"] = [
                        ttnn.to_torch(t).float().reshape(b, -1) for t in ttnn.get_device_tensors(projected)
                    ]
                    del uploaded, canonical, projected
                for name in models:
                    active[b, name, phase] = ttnn.clone(banks[b, phase, 0])
        report["input_sha256"] = {str(k): digest_host(v) for k, v in hosts.items()}
        report["active_rank_addresses"] = {str(k): rank_addresses(v) for k, v in active.items()}
        sync()

        def set_bank(b, selector):
            for name in models:
                bank = 1 - selector if name.endswith("_rebound") else selector
                for phase in (0, 1):
                    ttnn.copy(banks[b, phase, bank], active[b, name, phase])

        def pair(name, b, *, allocate_slots=False):
            model = models[name]
            owner = model._fused_decode_norm
            assert owner.next_phase == 0 and owner.open_batch is None
            with device_only():
                for phase, key in enumerate(KEYS):
                    assert owner.next_phase == phase
                    norm = model._norm(active[b, name, phase], key, decode=True)
                    assert norm.memory_config() == owner.canonical[phase]
                    projected = model._linear(norm, semantic_keys[phase])
                    if allocate_slots:
                        slots[name, b, phase, "norm"] = ttnn.empty_like(norm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                        slots[name, b, phase, "projection"] = ttnn.empty_like(
                            projected, memory_config=ttnn.DRAM_MEMORY_CONFIG
                        )
                    ttnn.copy(norm, slots[name, b, phase, "norm"])
                    ttnn.copy(projected, slots[name, b, phase, "projection"])
                    del norm, projected
            assert owner.next_phase == 0 and owner.open_batch is None

        def snapshot(name, b):
            return {
                (phase, kind): [
                    ttnn.to_torch(t).float().reshape(b, -1)
                    for t in ttnn.get_device_tensors(slots[name, b, phase, kind])
                ]
                for phase in (0, 1)
                for kind in ("norm", "projection")
            }

        def storage_hashes(owner):
            return [
                digest_host(ttnn.to_torch(t))
                for s in owner.stats + owner.ag_outputs + owner.norm_outputs
                for t in ttnn.get_device_tensors(s)
            ]

        eager = {}
        failures = []
        for b in args.batches:
            for name in models:
                pair(name, b, allocate_slots=True)
            for selector in (0, 1):
                set_bank(b, selector)
                for name in models:
                    pair(name, b)
                    sync()
                    actual = snapshot(name, b)
                    eager[name, b, selector] = actual
                    bank = 1 - selector if name.endswith("_rebound") else selector
                    for (phase, kind), parts in actual.items():
                        for rank, got in enumerate(parts):
                            expected = references[b, phase, bank, kind][rank]
                            finite = bool(torch.isfinite(got).all())
                            scores = [pcc(expected[u], got[u]) for u in range(b)] if finite else []
                            group_scores = {}
                            if finite and kind == "projection":
                                for label, begin, end in projection_groups(
                                    base.kind, phase, bool(base.policy.get("packed_qkvg"))
                                ):
                                    assert end <= got.shape[-1] and expected.shape == got.shape
                                    group_scores[label] = [
                                        pcc(expected[u, begin:end], got[u, begin:end]) for u in range(b)
                                    ]
                            all_scores = scores + [value for values in group_scores.values() for value in values]
                            row = {
                                "owner": name,
                                "batch": b,
                                "phase": phase,
                                "input_bank": bank,
                                "rank": rank,
                                "kind": kind,
                                "finite": finite,
                                "per_user_pcc": scores,
                                "per_group_user_pcc": group_scores,
                                "pass": finite and min(all_scores) >= 0.995,
                            }
                            report["quality"].append(row)
                            if not row["pass"]:
                                failures.append(row)
        save()
        assert not failures, failures
        for name in models:
            for b in args.batches:
                assert any(
                    not torch.equal(a, c) for a, c in zip(eager[name, b, 0][0, "norm"], eager[name, b, 1][0, "norm"])
                )

        # Rebind test poisons the original gamma while a second owner keeps actual
        # gamma. Distinct inputs/stats/output/gamma addresses remain simultaneously live.
        b = args.batches[0]
        for family in plan["families"]:
            old, new = models[family]._fused_decode_norm, models[family + "_rebound"]._fused_decode_norm
            assert all(rank_addresses(a) != rank_addresses(c) for a, c in zip(old.gamma, new.gamma))
            zero = [ttnn.full_like(g, 0) for g in old.gamma]
            set_bank(b, 0)
            for z, g in zip(zero, old.gamma):
                ttnn.copy(z, g)
            pair(family, b)
            sync()
            old_snapshot, before = snapshot(family, b), storage_hashes(old)
            assert all(torch.count_nonzero(t) == 0 for phase in (0, 1) for t in old_snapshot[phase, "norm"])
            pair(family + "_rebound", b)
            sync()
            rebound = snapshot(family + "_rebound", b)
            exact = all(
                torch.equal(t, eager[family + "_rebound", b, 0][k][rank])
                for k, parts in rebound.items()
                for rank, t in enumerate(parts)
            )
            untouched = before == storage_hashes(old)
            old_untouched = all(
                torch.equal(t, old_snapshot[k][rank])
                for k, parts in snapshot(family, b).items()
                for rank, t in enumerate(parts)
            )
            row = {
                "family": family,
                "old_gamma_zero": True,
                "new_owner_exact": exact,
                "old_storage_untouched": untouched,
                "old_snapshots_untouched": old_untouched,
            }
            report["rebind_checks"].append(row)
            save()
            assert exact and untouched and old_untouched, row
            for source, target in zip(new.gamma, old.gamma):
                ttnn.copy(source, target)
            for z in zero:
                ttnn.deallocate(z)
            del zero, z
        # All retained allocations and every signature warmup precede first capture.
        for b in args.batches:
            set_bank(b, 0)
            for name in models:
                pair(name, b)
        sync()
        assert all(t.memory_config() == ttnn.DRAM_MEMORY_CONFIG for t in slots.values())
        report["verification_slots"] = {
            str(k): {"rank_addresses": rank_addresses(t), "memory": str(t.memory_config())} for k, t in slots.items()
        }
        report["owners_warmed"] = {n: m._fused_decode_norm.ownership() for n, m in models.items()}
        mesh.set_program_cache_misses_allowed(False)
        for b in args.batches:
            for name in models:
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                pair(name, b)
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                traces[name, b] = tid

        from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker

        def verify_allocations():
            for tid in traces.values():
                TraceAllocationTracker.verify_before_replay(mesh, tid)

        verify_allocations()
        extra = ttnn.empty_like(next(iter(banks.values())), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        uid = extra.buffer_unique_id()
        caught = False
        try:
            verify_allocations()
        except RuntimeError as error:
            assert "device buffer(s) still alive before trace replay" in str(error) and f"Buffer {uid}" in str(error)
            caught = True
        assert caught, "Allocation tracker positive control was ineffective"
        ttnn.deallocate(extra)
        del extra
        verify_allocations()
        report["allocation_positive_control"] = {
            "rejected_live_buffer": int(uid),
            "all_traces_clean_after_release": True,
        }

        def replay(name, b):
            verify_allocations()
            ttnn.execute_trace(mesh, traces[name, b], cq_id=0, blocking=False)

        def exact(name, b, selector, label):
            actual = snapshot(name, b)
            checks = [
                {
                    "phase": phase,
                    "kind": kind,
                    "rank": rank,
                    "user": u,
                    "exact": torch.equal(t[u], eager[name, b, selector][phase, kind][rank][u]),
                }
                for (phase, kind), parts in actual.items()
                for rank, t in enumerate(parts)
                for u in range(b)
            ]
            row = {
                "owner": name,
                "batch": b,
                "selector": selector,
                "label": label,
                "per_rank_user": checks,
                "pass": all(c["exact"] for c in checks),
            }
            report["replay_checks"].append(row)
            save()
            assert row["pass"], row

        # Interleave different-B captures using the SAME owner's backing allocation.
        for selector in (1, 0, 1, 0):
            for b in reversed(args.batches):
                set_bank(b, selector)
                for name in models:
                    replay(name, b)
                    sync()
                    exact(name, b, selector, "changed_input_cross_batch")
        for b in args.batches:
            for name in models:
                for i in range(args.queued):
                    set_bank(b, i % 2)
                    replay(name, b)
                sync()
                exact(name, b, (args.queued - 1) % 2, "queued_final")
        # Cached eager calls and traced reuse must coexist without reallocations.
        for b in args.batches:
            set_bank(b, 1)
            for name in models:
                pair(name, b)
                replay(name, b)
                sync()
                exact(name, b, 1, "after_eager_interleave")
        verify_allocations()
        report["owners_final"] = {n: m._fused_decode_norm.ownership() for n, m in models.items()}
        for name, final in report["owners_final"].items():
            before = report["owners_before"][name]
            assert final["persistent_addresses"] == before["persistent_addresses"]
            assert final["persistent_tensors"] == before["persistent_tensors"], "Per-rank backing changed"
            assert final["semaphores"] == before["semaphores"]
            assert final["batch_signature_device_allocations"] == 1
            assert set(args.batches) <= set(final["warmed_batches"]) <= set(range(1, 33))
        report.update(
            status="pass",
            hardware_validated=True,
            queued_pairs_per_owner_batch=args.queued,
            full_queue_coverage=args.queued >= 50,
            queued_coverage="final output plus four individually checked changed-bank replays",
            multiple_batch_traces=len(args.batches) > 1,
            stage_default_changed=False,
            whole_decoder_gate_substitute=False,
            native_loaded=loaded_native_snapshot(REPO),
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
