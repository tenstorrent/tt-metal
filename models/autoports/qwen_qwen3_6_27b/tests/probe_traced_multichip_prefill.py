# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated TP4 real-weight eager/traced prefill boundary and continuity probe.

No accelerator or torch imports occur until main enters the hardware branch.
The parent/device operator runs this module; --cpu-only checks the launch plan.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
import traceback
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def launch_plan(batch, length):
    assert 1 <= batch <= 32 and length >= 1
    pages = (length + 33 + 31) // 32  # Exact stress=0 reference harness allocation.
    assert pages * 32 >= length + 33
    return {
        "mesh": [1, 4],
        "batch": batch,
        "logical_length": length,
        "page_rows": [batch, pages],
        "internal_tile_length": ((length + 31) // 32) * 32,
        "public_alignment_required": False,
        "input_banks": 2,
        "trace_signatures": ["prefill", "decode"],
        "captures_per_signature": 1,
        "prefill_timing_samples_each": 7,
        "order": [
            "allocate_all_persistent_buffers",
            "warm_all_branches_and_copies",
            "capture_prefill_once",
            "capture_decode_once",
            "allocation_positive_control",
            "changed_input_replays",
            "reset_per_timing_sample",
            "eager_interleave",
            "final_prefill_replay",
            "final_decode_replay",
            "teardown",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--policy", type=Path, required=True, help="Explicit candidate policy JSON file")
    parser.add_argument("--reference", type=Path, required=True, help="Frozen accepted single-chip TTNN .pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument("--cpu-only", action="store_true", help="Validate CLI/geometry without importing torch or TTNN")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    policy = json.loads(args.policy.read_text())
    assert isinstance(policy, dict)
    assert (
        policy.get("residual_layout", "sharded") == "sharded"
    ), "This probe requires the TP4 sharded residual contract"
    policy = {"residual_layout": "sharded", **policy}
    report = {
        "status": "prepared",
        "plan": launch_plan(args.batch, args.length),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "requested_policy": policy,
        "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "reference_kind": "frozen accepted single-chip TTNN, not HF",
        "allocation_tracking": os.getenv("TT_METAL_TRACE_ALLOC_TRACKING", "0") == "1",
        "skip_program_cache": os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "1",
        "watcher": "TT_METAL_WATCHER" in os.environ,
        "profiler": os.getenv("TT_METAL_DEVICE_PROFILER", "0") != "0",
        "boundary": "one full prefill forward plus copy into a stable output slot; state reset/input copy and synchronization before the interval are excluded; completion synchronization is included",
        "captures": [],
        "quality": [],
        "exact_replay": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    if args.cpu_only:
        report["status"] = "cpu-plan-pass"
        report["hardware_validated"] = False
        save()
        print(json.dumps(report["plan"]), flush=True)
        return

    # Deliberately deferred: source preparation and --cpu-only never import TTNN.
    import torch
    import ttnn
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    from models.autoports.qwen_qwen3_6_27b.tests.multichip_assertions import pcc
    from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import load_layer
    from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import MultichipDecoder
    from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import PagedState
    from models.autoports.qwen_qwen3_6_27b.tt.page_table_ownership import page_table_certified, page_table_for_attention

    @contextmanager
    def device_only():
        def forbidden(*_args, **_kwargs):
            raise AssertionError("Host conversion or synchronization inside forward/capture")

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
    report["status"] = "running"
    save()
    try:
        assert not (report["watcher"] and report["profiler"]), "Watcher and profiler must run separately"
        assert not report["skip_program_cache"], "Program-cache allocation tracking must not be skipped"
        torch.set_num_threads(8)
        torch.manual_seed(17)
        config = Qwen3_5TextConfig(**json.loads((ROOT / "doc/hf_config.json").read_text())["text_config"])
        weights = load_layer(args.layer)
        x = torch.randn(args.batch, args.length, 5120, dtype=torch.bfloat16) * 0.1
        d = torch.randn(args.batch, 1, 5120, dtype=torch.bfloat16) * 0.1
        pages = report["plan"]["page_rows"][1]
        table = torch.randperm(args.batch * pages).reshape(args.batch, pages)

        def tensor_hash(tensor):
            return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

        metadata = {
            "layer": args.layer,
            "batch": args.batch,
            "length": args.length,
            "input_sha256": tensor_hash(x),
            "decode_input_sha256": tensor_hash(d),
            "page_table_sha256": tensor_hash(table),
            "optimized_source_sha256": hashlib.sha256((ROOT / "tt/optimized_decoder.py").read_bytes()).hexdigest(),
            "hf_revision": "8f33b98937c6031bb49e713dee3e8e185aafd26f",
        }
        ref = torch.load(args.reference, weights_only=True, map_location="cpu")
        assert (
            ref.get("metadata") == metadata
        ), "Frozen reference metadata mismatch; this probe never regenerates or substitutes the reference"
        report["reference_metadata"] = metadata
        report["reference_sha256"] = hashlib.sha256(args.reference.read_bytes()).hexdigest()
        report["source_sha256"] = {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), *sorted((ROOT / "tt").rglob("*.py"))]
        }
        host_banks = [(x, d), (-x, -d)]
        report["bank_hashes"] = [[tensor_hash(t) for t in bank] for bank in host_banks]
        rope = Qwen3_5TextRotaryEmbedding(config)
        rotations = [
            rope(x, torch.arange(args.length).expand(args.batch, -1)),
            rope(d, torch.full((args.batch, 1), args.length)),
        ]
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=64 * 1024 * 1024)
        model = MultichipDecoder.from_state_dict(
            weights, hf_config=config, layer_idx=args.layer, mesh_device=mesh, policy=policy
        )
        assert model.hidden_width == 1280
        report["resolved_policy"], report["layer_kind"] = model.policy, model.kind
        state = (
            model.create_recurrent_state(args.batch)
            if model.kind == "linear_attention"
            else model.create_paged_state(table)
        )
        tensors = [state.conv, state.recurrent] if model.kind == "linear_attention" else [state.key, state.value]
        banks = [(model.upload_residual(a), model.upload_decode_residual(b)) for a, b in host_banks]
        active = [ttnn.clone(t) for t in banks[0]]
        empty = [ttnn.clone(t) for t in tensors]
        prefilled = [[ttnn.empty_like(t) for t in tensors] for _ in banks]
        pk, dk = {}, {}
        ordered_state = state
        if model.kind == "full_attention":
            pk = {"rope": tuple(model._upload(t.unsqueeze(1)) for t in rotations[0])}
            dk = {
                "rope": tuple(model._upload(t.unsqueeze(1)) for t in rotations[1]),
                "current_pos": ttnn.from_torch(
                    torch.full((args.batch,), args.length, dtype=torch.int32),
                    device=mesh,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                ),
            }
            # A legitimate uncertified view warms the ordered branch. The helper
            # only reads the private table; this probe never mutates/exposes it.
            ordered_state = PagedState(state.key, state.value, page_table_for_attention(state))
        certified = page_table_certified(state)
        report["page_lease_certified_before_warmup"] = certified

        def sync():
            ttnn.synchronize_device(mesh)

        def copy_state(source):
            for a, b in zip(source, tensors):
                ttnn.copy(a, b)

        def set_inputs(index):
            for a, b in zip(banks[index], active):
                ttnn.copy(a, b)

        def dispatch():
            return [model._prefill_cache_bulk_calls, model._prefill_cache_ordered_calls]

        def forward(phase, owner=state):
            with device_only():
                return (
                    model.prefill_forward(active[0], state=owner, **pk)
                    if phase == "prefill"
                    else model.decode_forward(active[1], state=owner, **dk)
                )

        # Discover output specs before either capture. No output or state owner
        # is allocated once the first trace has been captured.
        set_inputs(0)
        copy_state(empty)
        discovered = forward("prefill")
        slots = {"prefill": ttnn.empty_like(discovered)}
        del discovered
        discovered = forward("decode")
        slots["decode"] = ttnn.empty_like(discovered)
        del discovered
        sync()

        def invoke(phase, owner=state):
            with device_only():
                temporary = forward(phase, owner)
                ttnn.copy(temporary, slots[phase])
                # Destroy the returned temporary before capture ends. Only the
                # pre-capture slot is retained by Python between signatures.
                del temporary

        def read(tensor, residual=False):
            values = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(tensor)]
            assert len(values) == 4, "Measured path must be TP4"
            if residual:
                return torch.cat(values, -1).reshape(args.batch, -1, 5120)
            if model.kind == "linear_attention" and len(tensor.shape) == 3:
                return torch.cat(
                    [torch.cat([v[..., a:z] for v in values], -1) for a, z in ((0, 512), (512, 1024), (1024, 2560))], -1
                )
            return torch.cat(values, 1)

        def snapshot(phase):
            return {"output": read(slots[phase], True), "state": [read(t) for t in tensors]}

        def quality(label, phase, actual):
            expected = ref[phase]
            inp = host_banks[0][0 if phase == "prefill" else 1]
            scores = {
                "output": pcc(expected, actual["output"]),
                "contribution": pcc(expected - inp, actual["output"] - inp),
                "per_user": [pcc(expected[i], actual["output"][i]) for i in range(args.batch)],
                "cache": [pcc(a, b) for a, b in zip(ref[phase + "_state"], actual["state"])],
            }
            good = min(scores["output"], scores["contribution"], *scores["per_user"], *scores["cache"]) >= 0.995
            report["quality"].append({"label": label, "phase": phase, "scores": scores, "pass": good})
            save()
            assert good, report["quality"][-1]

        def exact(label, phase, actual, expected):
            pairs = [(actual["output"], expected["output"]), *zip(actual["state"], expected["state"])]
            checks = [bool(torch.equal(a, b)) for a, b in pairs]
            report["exact_replay"].append({"label": label, "phase": phase, "output_and_cache_equal": checks})
            save()
            assert all(checks), report["exact_replay"][-1]

        # Every bank, both fill branches, reset/snapshot copies, and both forward
        # signatures are warmed before either trace. Host snapshots stay on CPU.
        eager = []
        report["eager_prefill_dispatch"] = []
        for index in range(len(banks)):
            set_inputs(index)
            copy_state(empty)
            before = dispatch()
            invoke("prefill")
            sync()
            report["eager_prefill_dispatch"].append([a - b for a, b in zip(dispatch(), before)])
            p = snapshot("prefill")
            for source, dest in zip(tensors, prefilled[index]):
                ttnn.copy(source, dest)
            invoke("decode")
            sync()
            q = snapshot("decode")
            eager.append({"prefill": p, "decode": q})
            if index == 0:
                quality("eager", "prefill", p)
                quality("eager_continuity", "decode", q)
            for _ in range(2):
                copy_state(empty)
                invoke("prefill", ordered_state)
                sync()
                exact(f"warm_ordered_bank{index}", "prefill", snapshot("prefill"), p)
                invoke("decode")
                sync()
                exact(f"warm_ordered_continuity_bank{index}", "decode", snapshot("decode"), q)
                copy_state(prefilled[index])
                invoke("decode")
                sync()
                exact(f"warm_snapshot_restore_bank{index}", "decode", snapshot("decode"), q)
                copy_state(empty)
                invoke("prefill")
                invoke("decode")
                sync()
        assert page_table_certified(state) == certified, "Warmup must not revoke the page-table lease"
        assert not torch.equal(
            eager[0]["prefill"]["output"], eager[1]["prefill"]["output"]
        ), "Changed-input negative control did not change output"
        report["persistent_addresses"] = {
            "active": [int(t.buffer_address()) for t in active],
            "state": [int(t.buffer_address()) for t in tensors],
            "slots": {k: int(t.buffer_address()) for k, t in slots.items()},
        }
        report["program_cache_entries_before_capture"] = mesh.num_program_cache_entries()
        mesh.set_program_cache_misses_allowed(False)
        for phase in ("prefill", "decode"):
            set_inputs(0)
            copy_state(empty if phase == "prefill" else prefilled[0])
            sync()
            before = dispatch()
            native_fill_calls = [0]
            original_fill = ttnn.experimental.paged_fill_cache

            def counted_ordered_fill(*arguments, **keywords):
                native_fill_calls[0] += 1
                return original_fill(*arguments, **keywords)

            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            # B1 uses the inherited ordered fill directly, so the model's
            # batched-dispatch counters do not cover that valid branch.
            # Count the actual native API only while building this trace;
            # eager timing remains uninstrumented by this host-side counter.
            fill_counter = (
                patch.object(ttnn.experimental, "paged_fill_cache", counted_ordered_fill)
                if model.kind == "full_attention" and phase == "prefill"
                else nullcontext()
            )
            with device_only(), fill_counter:
                invoke(phase)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            traces[phase] = trace
            counts = [a - b for a, b in zip(dispatch(), before)]
            report["captures"].append(
                {"phase": phase, "cache_fill_bulk_ordered": counts, "native_ordered_fill_calls": native_fill_calls[0]}
            )
            if model.kind == "full_attention" and phase == "prefill":
                assert counts[0] == 0 and native_fill_calls[0] > 0, (counts, native_fill_calls)
                if args.batch > 1:
                    assert counts[1] > 0, counts
            assert page_table_certified(state) == certified
            save()

        def verify_allocations():
            if report["allocation_tracking"]:
                from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker

                for trace in traces.values():
                    TraceAllocationTracker.verify_before_replay(mesh, trace)

        verify_allocations()
        if report["allocation_tracking"]:
            from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker
            from ttnn._ttnn.operations.trace import get_unsafe_tracked_ids

            negative = ttnn.empty_like(active[0])
            negative_id = negative.buffer_unique_id()
            assert negative_id in get_unsafe_tracked_ids(mesh, traces["prefill"])
            rejected = False
            try:
                TraceAllocationTracker.verify_before_replay(mesh, traces["prefill"])
            except RuntimeError as error:
                assert "device buffer(s) still alive before trace replay" in str(error), str(error)
                assert f"Buffer {negative_id}" in str(error), str(error)
                rejected = True
            assert rejected, "Tracker failed to reject a deliberate live allocation"
            ttnn.deallocate(negative)
            del negative
            verify_allocations()
            report["allocation_positive_control"] = {
                "rejected_before_replay": True,
                "both_traces_clean_after_release": True,
                "acknowledgements_or_corruptible_scopes": False,
            }
        else:
            report["allocation_positive_control"] = {
                "executed": False,
                "reason": "tracking disabled; performance evidence only",
            }

        def replay(phase):
            verify_allocations()
            ttnn.execute_trace(mesh, traces[phase], cq_id=0, blocking=False)
            sync()

        def replay_pair(index, label):
            set_inputs(index)
            copy_state(empty)
            replay("prefill")
            p = snapshot("prefill")
            exact(label, "prefill", p, eager[index]["prefill"])
            # No restore between these replays: decode consumes traced prefill's
            # real cache/recurrent state at the next logical position.
            replay("decode")
            q = snapshot("decode")
            exact(label, "decode", q, eager[index]["decode"])
            if index == 0:
                quality(label, "prefill", p)
                quality(label, "decode", q)

        for repeat in range(2):
            for index in range(len(banks)):
                replay_pair(index, f"changed_bank{index}_repeat{repeat}")

        instrumented = report["watcher"] or report["profiler"] or report["allocation_tracking"]
        report["timing"] = {
            "enabled": not (instrumented or args.quality_only),
            "instrumented": instrumented,
            "samples_per_path": 7,
            "state_reset_outside_boundary": True,
            "units": "us",
        }
        if report["timing"]["enabled"]:
            samples = {"eager": [], "trace": []}
            for sample in range(7):
                for mode in (("eager", "trace") if sample % 2 == 0 else ("trace", "eager")):
                    set_inputs(0)
                    copy_state(empty)
                    sync()
                    start = time.perf_counter_ns()
                    if mode == "eager":
                        invoke("prefill")
                    else:
                        ttnn.execute_trace(mesh, traces["prefill"], cq_id=0, blocking=False)
                    sync()
                    samples[mode].append((time.perf_counter_ns() - start) / 1000)
                    exact(f"timed_{mode}_{sample}", "prefill", snapshot("prefill"), eager[0]["prefill"])
            report["timing"].update(
                {
                    "samples_us": samples,
                    "median_us": {k: statistics.median(v) for k, v in samples.items()},
                    "mean_us": {k: statistics.mean(v) for k, v in samples.items()},
                }
            )
            report["timing"]["eager_over_trace"] = statistics.median(samples["eager"]) / statistics.median(
                samples["trace"]
            )

        # Exercise eager allocations with both traces alive, then finish device
        # work with decode replay. Tracker and the frozen cache reject any late
        # persistent allocation or compilation instead of masking it.
        for index in range(len(banks)):
            set_inputs(index)
            copy_state(empty)
            invoke("prefill")
            invoke("decode")
            sync()
            exact(f"eager_interleave_bank{index}", "decode", snapshot("decode"), eager[index]["decode"])
        replay_pair(0, "final_after_eager_interleave")
        verify_allocations()
        report["program_cache_entries_after_replay"] = mesh.num_program_cache_entries()
        assert report["program_cache_entries_after_replay"] == report["program_cache_entries_before_capture"]
        report["page_lease_certified_after_replay"] = page_table_certified(state)
        assert report["page_lease_certified_after_replay"] == certified
        report["final_device_forward"] = "decode_trace_replay_after_eager_interleaving"
        report["status"] = "pass"
        save()
    except BaseException as error:
        report.update(status="failed", exception=repr(error), traceback=traceback.format_exc())
        save()
        raise
    finally:
        if mesh is not None:
            for trace in traces.values():
                ttnn.release_trace(mesh, trace)
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
