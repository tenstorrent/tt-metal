# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TP4 prefill norm families, independent FP64 oracle, unchanged real projections.

One layer/phase/public workload per process. No torch/TTNN import on --cpu-only.
This is a focused boundary probe, not a whole-decoder correctness/perf result.
"""

import argparse
import ast
import copy
from contextlib import ExitStack, contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
KEYS = ("input_layernorm.weight", "post_attention_layernorm.weight")
FAMILIES = ("distributed", "gather_explicit", "gather_sfpu")


def post_projection_ast(flat):
    """Extract the existing selected prefill projection, not a copied matmul policy.

    Only the already-normalized h -> act statements are compiled. No decoder
    forward call, residual update, collectives, or policy mutations are added.
    """
    source = ROOT / ("tt/flat_prefill.py" if flat else "tt/multichip_decoder.py")
    tree = ast.parse(source.read_text())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == ("_finish_flat" if flat else "_finish")
    )
    if flat:
        first = next(
            i
            for i, n in enumerate(fn.body)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "layer.swiglu_weight is not None"
        )
        body = [copy.deepcopy(fn.body[first])]
        prefix = ast.parse("layer = self\nprefill = True").body
    else:
        first = next(
            i
            for i, n in enumerate(fn.body)
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "swiglu_rows" for t in n.targets)
        )
        last = next(
            i
            for i, n in enumerate(fn.body)
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "result" for t in n.targets)
        )
        body = copy.deepcopy(fn.body[first:last])
        prefix = ast.parse("prefill = True").body
    out = ast.parse("def project_post(self, h, batch, seq):\n    pass")
    out.body[0].body = prefix + body + [ast.Return(ast.Name("act", ast.Load()))]
    ast.fix_missing_locations(out)
    assert not any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in ("_norm", "_reduce", "_collective")
        for n in ast.walk(out)
    )
    compile(out, str(source) + ":projection-only", "exec")
    return out, source


def workload_plan(batch, length, policy):
    assert 1 <= batch <= 32 and 1 <= length <= 513
    assert policy.get("flat_prefill", False), "This bounded probe requires current flat-prefill policy"
    assert policy.get("prefill_chunk", 512) == 512
    windows, start = [], 0
    while start < length:
        seq = min(512, length - start)
        flat = batch > 1 and seq > 1
        shape = (1, batch * seq, 1280) if flat else (batch, seq, 1280)
        windows.append(
            {
                "start": start,
                "seq": seq,
                "flat": flat,
                "logical_shape": list(shape),
                "padded_shape": [shape[0], (shape[1] + 31) // 32 * 32, 1280],
            }
        )
        start += seq
    return windows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--phase", choices=("input", "post"), required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--activation-file", type=Path)
    p.add_argument(
        "--supplemental", action="store_true", help="Nearzero, scaled, and NaN dead rows; <=2048 logical rows"
    )
    p.add_argument("--queued", type=int, default=50)
    p.add_argument("--samples", type=int, default=7)
    p.add_argument("--trace-mib", type=int, default=128)
    p.add_argument("--skip-timing", action="store_true")
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert 0 <= args.layer < 64 and args.queued >= 1 and args.samples >= 7
    assert not args.supplemental or args.batch * args.length <= 2048
    policy = json.loads(args.policy.read_text())
    assert isinstance(policy, dict) and policy.get("residual_layout", "sharded") == "sharded"
    assert policy.get("ccl_dtype") in (None, "bfloat16"), "FP64 oracle contract requires BF16 activation CCL"
    windows = workload_plan(args.batch, args.length, policy)
    phase = KEYS.index("input_layernorm.weight" if args.phase == "input" else "post_attention_layernorm.weight")
    key = KEYS[phase]
    report = {
        "status": "prepared",
        "command": shlex.join([sys.executable, "-m", __spec__.name, *sys.argv[1:]]),
        "layer": args.layer,
        "phase": args.phase,
        "batch": args.batch,
        "length": args.length,
        "windows": windows,
        "policy": policy,
        "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "families": list(FAMILIES),
        "quality": [],
        "replay_checks": [],
        "rebind_checks": [],
        "minimum_per_row_rank_pcc": 0.995,
        "whole_decoder_gate_substitute": False,
        "allocation_tracking": os.getenv("TT_METAL_TRACE_ALLOC_TRACKING", "0") == "1",
        "skip_program_cache": os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "1",
        "watcher": "TT_METAL_WATCHER" in os.environ,
        "profiler": os.getenv("TT_METAL_DEVICE_PROFILER", "0") != "0",
        "source_sha256": {
            str(f.relative_to(REPO)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [
                Path(__file__),
                ROOT / "tests/probe_fused_decode_norm_owner.py",
                ROOT / "tests/run_decoder.py",
                ROOT / "doc/hf_config.json",
                *sorted((ROOT / "tt").glob("*.py")),
            ]
        },
    }
    report["timing"] = {
        "enabled": not (args.skip_timing or report["allocation_tracking"] or report["watcher"] or report["profiler"]),
        "boundary": "hidden-shard -> norm including all AGs -> selected actual next projection; excludes verification copies",
        "samples": args.samples,
        "single_replay_per_sample": True,
        "not_whole_layer_latency": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    # Parse/prove source contracts without importing model/device modules.
    asts = {flat: post_projection_ast(flat)[0] for flat in (False, True)}
    norm_src = (ROOT / "tt/optimized_decoder.py").read_text()
    for anchor in (
        "def prefill_hidden_rmsnorm_sfpu(",
        'self.policy.get("sfpu_prefill_norm", False)',
        'self.policy.get("explicit_sum_norm", False)',
        "self.weights[key], dtype=ttnn.bfloat16",
    ):
        assert anchor in norm_src, anchor
    report["projection_ast_sha256"] = {
        str(k): hashlib.sha256(ast.dump(v).encode()).hexdigest() for k, v in asts.items()
    }
    if args.cpu_only:
        coverage = []
        for b, s in ((3, 33), (32, 513), (1, 512)):
            plan = workload_plan(b, s, policy)
            covered = [p["start"] + j for p in plan for j in range(p["seq"])]
            assert covered == list(range(s))
            assert all(p["padded_shape"][1] >= p["logical_shape"][1] for p in plan)
            coverage.append({"batch": b, "length": s, "windows": plan})
        report.update(
            status="cpu-plan-pass",
            hardware_validated=False,
            geometry_proof=coverage,
            frozen_kernel_duplicated=False,
            production_sources_edited=False,
        )
        save()
        return

    import torch
    import ttnn
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from .run_decoder import load_layer, MODEL, REVISION
    from .probe_fused_decode_norm_owner import projection_groups
    from .check_optimized_multichip_evidence import loaded_native_snapshot
    from ..tt.multichip_decoder import MultichipDecoder, _BlackholeCCLManager
    from ..tt.flat_prefill import _can_flatten

    @contextmanager
    def device_only():
        def forbidden(*_a, **_kw):
            raise AssertionError("Host conversion/synchronization inside measured norm/projection boundary")

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

    def digest(t):
        return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    def read(t):
        return [ttnn.to_torch(s).contiguous() for s in ttnn.get_device_tensors(t)]

    def addresses(t):
        out = [int(s.buffer_address()) for s in ttnn.get_device_tensors(t)]
        assert len(out) == 4
        return out

    mesh, traces = None, {}
    try:
        assert not report["skip_program_cache"]
        if args.skip_timing or report["watcher"]:
            assert report["allocation_tracking"], "Quality/watcher run requires strict allocation tracker"
        assert not report["profiler"], "Use a separate whole-decoder profile for op attribution"
        torch.set_num_threads(8)
        torch.manual_seed(4721)
        cfg = Qwen3_5TextConfig(**json.loads((ROOT / "doc/hf_config.json").read_text())["text_config"])
        weights = load_layer(args.layer)
        gamma_host = (1 + weights[key].float()).reshape(1, 5120)
        report["weight_provenance"] = {
            "model": MODEL,
            "revision": REVISION,
            "actual_fp32_gamma_sha256": digest(gamma_host),
        }
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_mib * 1024 * 1024)
        base = MultichipDecoder.from_state_dict(
            weights,
            hf_config=cfg,
            layer_idx=args.layer,
            mesh_device=mesh,
            policy={**policy, "distributed_decode_norm": "off"},
        )
        assert base.hidden_width == 1280 and base.policy.get("flat_norm") and base.policy.get("flat_ccl")
        assert base.kind != "full_attention" or base.policy.get(
            "packed_qkvg"
        ), "Bounded full-attention probe requires selected packed QKVG"
        for w in windows:
            assert (
                _can_flatten(base, args.batch, w["seq"]) == w["flat"]
            ), "Input policy changes selected public row layout"
        report.update(kind=base.kind, resolved_base_policy=base.policy, native_loaded=loaded_native_snapshot(REPO))
        models = {}
        for family in FAMILIES:
            for rebound in (False, True):
                name = family + ("_rebound" if rebound else "")
                model = copy.copy(base)
                model.policy = {
                    **base.policy,
                    "gather_norm": family != "distributed",
                    "flat_gather_prefill": True,
                    "sfpu_hidden_norm": False,
                    "sfpu_prefill_norm": family == "gather_sfpu",
                    "explicit_sum_norm": family == "gather_explicit",
                    "distributed_decode_norm": "off",
                }
                model.weights = {**base.weights, key: ttnn.clone(base.weights[key])}
                model.shard_norms = {**base.shard_norms, key: ttnn.clone(base.shard_norms[key])}
                model._ccl_buffers, model._prefill_batch_indices = {}, {}
                model.ccl = _BlackholeCCLManager(mesh, base.ccl.num_links, base.ccl.topology)
                models[name] = model
        report["family_policy"] = {n: m.policy for n, m in models.items()}
        report["gamma_bindings"] = {
            n: {"full": addresses(m.weights[key]), "shard": addresses(m.shard_norms[key])} for n, m in models.items()
        }
        report["ccl_semaphores"] = {
            n: [
                int(ttnn.get_global_semaphore_address(s))
                for s in m.ccl.ag_ping_pong_semaphores + m.ccl.barrier_semaphore
            ]
            for n, m in models.items()
        }
        all_sems = [a for values in report["ccl_semaphores"].values() for a in values]
        assert len(all_sems) == len(set(all_sems)), "Normalization owners share CCL semaphores"
        for rank in range(4):
            for kind in ("full", "shard"):
                addrs = [b[kind][rank] for b in report["gamma_bindings"].values()]
                assert len(addrs) == len(set(addrs)), "Normalization owners share gamma backing"
        gamma_actual = read(base.weights[key])
        assert all(torch.equal(g.reshape(1, 5120), gamma_host) for g in gamma_actual)
        report["gamma_device_reconstruction_exact"] = True
        project_post = {}
        for flat, tree in asts.items():
            scope = {"ttnn": ttnn, "__package__": "models.autoports.qwen_qwen3_6_27b.tt"}
            exec(compile(tree, "<selected prefill projection AST>", "exec"), scope)
            project_post[flat] = scope["project_post"]

        def projection(model, h, wi):
            w = windows[wi]
            if phase == 1:
                return project_post[w["flat"]](model, h, args.batch, w["seq"])
            if base.kind == "linear_attention":
                return model._prefill_packed_gdn(h) if w["flat"] else model._linear(h, "packed_gdn")
            return model._linear(h, "packed_qkvg")

        def sync():
            ttnn.synchronize_device(mesh)

        activation = args.activation_file
        if activation is None:
            candidates = [
                ROOT / f"doc/optimized_decoder/text_activations/text_l{args.layer}_b{args.batch}_s{args.length}.pt"
            ]
            if args.length == 33 and args.batch <= 32:
                candidates.append(ROOT / f"doc/optimized_decoder/text_activations/text_l{args.layer}_b32_s33.pt")
            activation = next((f for f in candidates if f.exists()), None)
        if activation is not None:
            recorded = torch.load(activation, map_location="cpu", weights_only=True)
            original = recorded["x"][: args.batch].to(torch.bfloat16).contiguous()
            assert tuple(original.shape) == (args.batch, args.length, 5120)
            report["input_provenance"] = {
                "kind": "recorded layer-input activation",
                "file": str(activation),
                "sha256": hashlib.sha256(activation.read_bytes()).hexdigest(),
                "selected_first_users": args.batch,
                "post_phase_scope": (
                    "Recorded layer input reused as post-gamma fixture; not claimed actual post-attention residual"
                    if phase
                    else None
                ),
            }
        else:
            original = torch.randn(args.batch, args.length, 5120).to(torch.bfloat16)
            report["input_provenance"] = {
                "kind": "seeded synthetic",
                "seed": 4721,
                "recorded_fixture_unavailable": True,
            }
        host_banks = {
            "normal": original,
            "changed": (original.float() * 0.73 + torch.randn(original.shape) * 0.19).to(torch.bfloat16),
        }
        if args.supplemental:
            host_banks.update(
                nearzero=(original.float() * 1e-6).to(torch.bfloat16),
                scaled=(original.float() * 4096).to(torch.bfloat16),
                poison_dead=original.clone(),
            )
        banks, backing, actual_hosts, active, slots, refs, eager = {}, {}, {}, {}, {}, {}, {}
        report["input_reconstruction"], report["physical_padding"] = [], []
        for wi, w in enumerate(windows):
            b, s, _ = w["logical_shape"]
            padded = w["padded_shape"][1]
            for bank, host in host_banks.items():
                logical = host[:, w["start"] : w["start"] + w["seq"]].contiguous().reshape(b, s, 5120)
                physical = torch.zeros(b, padded, 5120, dtype=torch.bfloat16)
                physical[:, :s] = logical
                if bank == "poison_dead":
                    physical[:, s:] = float("nan")
                tensor = ttnn.from_torch(
                    physical,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                    preserve_nan_values=True,
                )
                backing[wi, bank] = tensor
                banks[wi, bank] = ttnn.reshape(tensor, (b, s, 1280), (b, padded, 1280))
                actual = torch.cat(read(banks[wi, bank]), -1)
                assert torch.equal(actual, logical), (wi, bank, "actual BF16 reconstruction differs")
                actual_hosts[wi, bank] = actual
                report["input_reconstruction"].append(
                    {"window": wi, "bank": bank, "sha256": digest(actual), "exact": True}
                )
                if bank == "poison_dead":
                    counts = [
                        int(torch.isnan(t.cpu().to_torch_with_padded_shape()[:, s:]).sum())
                        for t in ttnn.get_device_tensors(tensor)
                    ]
                    assert counts == [b * (padded - s) * 1280] * 4
                    report["physical_padding"].append(
                        {
                            "window": wi,
                            "dead_rows_per_batch": padded - s,
                            "nan_count_by_rank": counts,
                            "applicable": padded > s,
                        }
                    )
                # FP64 normalization of reconstructed uploaded BF16, actual FP32 gamma.
                xf = actual.double()
                reference = (
                    xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + cfg.rms_norm_eps) * gamma_host.double()
                )
                normalized = base._upload(reference.to(torch.bfloat16))
                projected = projection(base, normalized, wi)
                sync()
                refs[wi, bank] = [r.reshape(args.batch * w["seq"], -1) for r in read(projected)]
                del projected, normalized, reference, xf
            for name in models:
                # Full physical backing view owns dead-row padding; copies use physical aliases.
                active[name, wi] = ttnn.clone(backing[wi, "normal"])
        views = {
            (n, wi): ttnn.reshape(t, tuple(windows[wi]["logical_shape"]), tuple(windows[wi]["padded_shape"]))
            for (n, wi), t in active.items()
        }

        def set_bank(name, wi, bank):
            ttnn.copy(backing[wi, bank], active[name, wi])

        def boundary(name, wi, *, allocate=False, verify=True, guard=True):
            model = models[name]
            with device_only() if guard else nullcontext():
                norm = model._norm(views[name, wi], key, logical_batch=args.batch)
                projected = projection(model, norm, wi)
                if allocate:
                    slots[name, wi, "norm"] = ttnn.empty_like(norm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    slots[name, wi, "projection"] = ttnn.empty_like(projected, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if verify:
                    ttnn.copy(norm, slots[name, wi, "norm"])
                    ttnn.copy(projected, slots[name, wi, "projection"])
                del norm, projected

        def hashes(name, wi):
            return {kind: [digest(t) for t in read(slots[name, wi, kind])] for kind in ("norm", "projection")}

        def row_scores(expected, got):
            # Vectorized FP64 PCC in bounded host blocks; undefined constant rows require exact equality.
            scores = []
            for start in range(0, got.shape[0], 512):
                a, b = expected[start : start + 512].double(), got[start : start + 512].double()
                ac, bc = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
                denom = ac.square().sum(-1).sqrt() * bc.square().sum(-1).sqrt()
                value = (ac * bc).sum(-1) / denom.clamp_min(1e-300)
                value = torch.where(denom == 0, (a == b).all(-1).double(), value).clamp(-1, 1)
                finite = torch.isfinite(a).all(-1) & torch.isfinite(b).all(-1)
                scores.append(torch.where(finite, value, torch.full_like(value, -1)))
            values = torch.cat(scores)
            failed = torch.nonzero(values < 0.995).reshape(-1).tolist()
            return {
                "minimum": float(values.min()),
                "worst_row": int(values.argmin()),
                "per_user_minimum": [float(v.min()) for v in values.reshape(args.batch, -1)],
                "failed_count": len(failed),
                "failed_rows_first256": failed[:256],
            }

        failures = []

        def quality(name, wi, bank, label):
            w = windows[wi]
            x = actual_hosts[wi, bank].double().reshape(args.batch * w["seq"], 5120)
            expected_norm = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + cfg.rms_norm_eps) * gamma_host.double()
            for kind in ("norm", "projection"):
                parts = read(slots[name, wi, kind])
                groups = [("all", 0, parts[0].shape[-1])]
                if kind == "projection":
                    groups += [("swiglu", 0, 4352)] if phase else projection_groups(base.kind, 0, True)
                for rank, part in enumerate(parts):
                    got = part.reshape(args.batch * w["seq"], -1)
                    expected = expected_norm if kind == "norm" else refs[wi, bank][rank]
                    assert expected.shape == got.shape
                    metrics = {label: row_scores(expected[:, lo:hi], got[:, lo:hi]) for label, lo, hi in groups}
                    row = {
                        "owner": name,
                        "window": wi,
                        "bank": bank,
                        "label": label,
                        "kind": kind,
                        "rank": rank,
                        "groups": metrics,
                        "pass": all(v["failed_count"] == 0 for v in metrics.values()),
                    }
                    report["quality"].append(row)
                    if not row["pass"]:
                        failures.append(row)
            eager[name, wi, bank] = hashes(name, wi)

        # Allocate every owner/window output before any capture; collect all family failures before asserting.
        for name in models:
            for wi in range(len(windows)):
                boundary(name, wi, allocate=True)
                for bank in host_banks:
                    set_bank(name, wi, bank)
                    boundary(name, wi)
                    sync()
                    quality(name, wi, bank, "eager_independent")
        save()
        assert not failures, f"{len(failures)} independent gate failures; see quality rows"
        for name in models:
            for wi in range(len(windows)):
                assert eager[name, wi, "normal"] != eager[name, wi, "changed"], "Changed input did not change outputs"
        # Distinct gamma/input/output bindings with poisoned original gamma and original outputs protected.
        for family in FAMILIES:
            old, new = models[family], models[family + "_rebound"]
            zeros = [ttnn.full_like(t, 0) for t in (old.weights[key], old.shard_norms[key])]
            for z, target in zip(zeros, (old.weights[key], old.shard_norms[key])):
                ttnn.copy(z, target)
            for wi in range(len(windows)):
                set_bank(family, wi, "normal")
                boundary(family, wi)
                sync()
                assert all(torch.count_nonzero(t) == 0 for t in read(slots[family, wi, "norm"]))
                before = hashes(family, wi)
                set_bank(family + "_rebound", wi, "changed")
                boundary(family + "_rebound", wi)
                sync()
                exact = hashes(family + "_rebound", wi) == eager[family + "_rebound", wi, "changed"]
                untouched = before == hashes(family, wi)
                report["rebind_checks"].append(
                    {
                        "family": family,
                        "window": wi,
                        "new_owner_exact": exact,
                        "old_zero_gamma_outputs_untouched": untouched,
                        "distinct_gamma_addresses": addresses(old.weights[key]) != addresses(new.weights[key]),
                    }
                )
                assert exact and untouched
            for source, target in ((new.weights[key], old.weights[key]), (new.shard_norms[key], old.shard_norms[key])):
                ttnn.copy(source, target)
            for z in zeros:
                ttnn.deallocate(z)
            del zeros, z
        # All persistent branches and ping-pong CCL variants warm before any capture.
        for _ in range(4):
            for name in models:
                for wi in range(len(windows)):
                    set_bank(name, wi, "normal")
                    boundary(name, wi)
                    if name in FAMILIES and report["timing"]["enabled"]:
                        boundary(name, wi, verify=False, guard=False)
        sync()
        stable = {
            "inputs": {str(k): addresses(v) for k, v in active.items()},
            "outputs": {str(k): addresses(v) for k, v in slots.items()},
            "ccl_keys": {n: sorted(str(k) for k in m._ccl_buffers) for n, m in models.items()},
            "ccl_buffers": {
                n: {str(k): [addresses(t) if t is not None else None for t in v] for k, v in m._ccl_buffers.items()}
                for n, m in models.items()
            },
            "policies": copy.deepcopy({n: m.policy for n, m in models.items()}),
        }
        report["stable_before_capture"] = stable
        mesh.set_program_cache_misses_allowed(False)
        for name in models:
            for wi in range(len(windows)):
                for mode in (
                    ("quality", "timing") if name in FAMILIES and report["timing"]["enabled"] else ("quality",)
                ):
                    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                    boundary(name, wi, verify=mode == "quality")
                    ttnn.end_trace_capture(mesh, tid, cq_id=0)
                    traces[name, wi, mode] = tid
        from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker

        def verify_allocations():
            if report["allocation_tracking"]:
                for tid in traces.values():
                    TraceAllocationTracker.verify_before_replay(mesh, tid)

        verify_allocations()
        if report["allocation_tracking"]:
            extra = ttnn.empty_like(next(iter(active.values())), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            uid = extra.buffer_unique_id()
            caught = False
            try:
                verify_allocations()
            except RuntimeError as error:
                assert "device buffer(s) still alive before trace replay" in str(error) and f"Buffer {uid}" in str(
                    error
                )
                caught = True
            assert caught, "Allocation tracker positive control ineffective"
            ttnn.deallocate(extra)
            del extra
            verify_allocations()
            report["allocation_positive_control"] = {"rejected_live_buffer": int(uid), "clean_after_release": True}

        def replay(name, wi, mode="quality"):
            verify_allocations()
            ttnn.execute_trace(mesh, traces[name, wi, mode], cq_id=0, blocking=False)

        def exact(name, wi, bank, label):
            passed = hashes(name, wi) == eager[name, wi, bank]
            report["replay_checks"].append({"owner": name, "window": wi, "bank": bank, "label": label, "exact": passed})
            save()
            assert passed, (name, wi, bank, label)

        for name in models:
            for wi in range(len(windows)):
                for bank in (
                    "changed",
                    "normal",
                    *(["nearzero", "scaled", "poison_dead"] if args.supplemental else []),
                ):
                    set_bank(name, wi, bank)
                    replay(name, wi)
                    sync()
                    exact(name, wi, bank, "changed_input_trace")
                for i in range(args.queued):
                    bank = "normal" if i % 2 == 0 else "changed"
                    set_bank(name, wi, bank)
                    replay(name, wi)
                sync()
                exact(name, wi, bank, "queued_final")
        if report["timing"]["enabled"]:
            timings = {}
            for name in FAMILIES:
                eager_samples, trace_samples = [], []
                for wi in range(len(windows)):
                    set_bank(name, wi, "normal")
                for _ in range(2):
                    for wi in range(len(windows)):
                        boundary(name, wi, verify=False, guard=False)
                        replay(name, wi, "timing")
                sync()
                for _ in range(args.samples):
                    sync()
                    start = time.perf_counter_ns()
                    for wi in range(len(windows)):
                        boundary(name, wi, verify=False, guard=False)
                    sync()
                    eager_samples.append((time.perf_counter_ns() - start) / 1000)
                    sync()
                    start = time.perf_counter_ns()
                    for wi in range(len(windows)):
                        replay(name, wi, "timing")
                    sync()
                    trace_samples.append((time.perf_counter_ns() - start) / 1000)
                timings[name] = {
                    "warmed_eager_us": eager_samples,
                    "traced_warmed_us": trace_samples,
                    "median_eager_us": statistics.median(eager_samples),
                    "median_trace_us": statistics.median(trace_samples),
                }
            report["timing"]["families"] = timings
        # Eager rebinding after capture must not allocate persistent state or contaminate other owners' traces.
        for family in FAMILIES:
            for wi in range(len(windows)):
                set_bank(family + "_rebound", wi, "changed")
                boundary(family + "_rebound", wi)
                set_bank(family, wi, "changed")
                replay(family, wi)
                sync()
                exact(family, wi, "changed", "after_rebound_eager_interleave")
        verify_allocations()
        assert stable == {
            "inputs": {str(k): addresses(v) for k, v in active.items()},
            "outputs": {str(k): addresses(v) for k, v in slots.items()},
            "ccl_keys": {n: sorted(str(k) for k in m._ccl_buffers) for n, m in models.items()},
            "ccl_buffers": {
                n: {str(k): [addresses(t) if t is not None else None for t in v] for k, v in m._ccl_buffers.items()}
                for n, m in models.items()
            },
            "policies": {n: m.policy for n, m in models.items()},
        }
        report.update(
            status="pass",
            hardware_validated=True,
            queued_per_owner_window=args.queued,
            full_queue_coverage=args.queued >= 50,
            exact_trace_equals_independently_gated_eager=True,
            native_loaded=loaded_native_snapshot(REPO),
            stage_default_changed=False,
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
