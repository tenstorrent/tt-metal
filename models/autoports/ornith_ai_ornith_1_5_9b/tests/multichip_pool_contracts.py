# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP4 pooled-state isolation against private-state, serial TTNN controls.

Run under an external device lock and timeout. This is a state/trace correctness
test, not a performance measurement or an independent numerical model reference.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.run_decoder import MODEL, REVISION, ROOT, config, pcc, weights
from models.autoports.ornith_ai_ornith_1_5_9b.tests.validate_optimized import buffers, device_only
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--capacity-extra-pages", type=int, default=0)
    args = ap.parse_args()
    assert args.steps >= 1
    torch.set_num_threads(8)
    torch.manual_seed(619)
    records, trace_ids = [], []
    result = {
        "model": MODEL,
        "revision": REVISION,
        "layer": args.layer,
        "command": sys.argv,
        "mesh": [1, 4],
        "steps": args.steps,
        "tracking": os.getenv("TT_METAL_TRACE_ALLOC_TRACKING", "0"),
        "skip_program_cache_tracking": os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
        "watcher": os.getenv("TT_METAL_WATCHER"),
        "source_sha256": hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "control": "Same production TP4 layer, independent private states, serial decode_forward",
        "records": records,
        "pass": False,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output_path.write_text(json.dumps(result, indent=2) + "\n")

    save()
    c, sd = config(), weights(args.layer)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)

    def host(value, integer=False, device=True):
        return ttnn.from_torch(
            value.contiguous(),
            device=mesh if device else None,
            dtype=ttnn.int32 if integer else ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def ranks(value):
        return [ttnn.to_torch(t).clone() for t in ttnn.get_device_tensors(value)]

    def record_pcc(label, actual, expected, residual=None):
        rec = {"label": label}
        try:
            assert len(actual) == len(expected) == 4, (label, "rank count mismatch")
            assert all(a.shape == b.shape for a, b in zip(actual, expected)), (label, "shape mismatch")
            scores = [pcc(a, b) for a, b in zip(actual, expected)]
            rec["pcc"] = min(scores)
            if residual is not None:
                contributions = [pcc(a - residual, b - residual) for a, b in zip(actual, expected)]
                rec["contribution_pcc"] = min(contributions)
                scores += contributions
            assert all(math.isfinite(score) and score >= 0.995 for score in scores), rec
        except Exception as error:

            def summary(value):
                finite = torch.isfinite(value)
                selected = value[finite].double()
                return {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "elements": value.numel(),
                    "finite_count": int(finite.sum()),
                    "nan_count": int(torch.isnan(value).sum()),
                    "positive_inf_count": int(torch.isposinf(value).sum()),
                    "negative_inf_count": int(torch.isneginf(value).sum()),
                    "maxabs_finite": float(selected.abs().max()) if selected.numel() else None,
                }

            safe_label = "".join(char if char.isalnum() or char in "_-" else "_" for char in label)
            tensor_path = output_path.with_name(output_path.stem + ".failure_" + safe_label + ".pt")
            torch.save({"actual": actual, "expected": expected, "residual": residual}, tensor_path)
            rec.update(
                {
                    "error": repr(error),
                    "actual": [summary(value) for value in actual],
                    "expected": [summary(value) for value in expected],
                    "residual": summary(residual) if residual is not None else None,
                    "failure_tensors": str(tensor_path),
                }
            )
            records.append(rec)
            print(json.dumps(rec), flush=True)
            save()
            raise
        records.append(rec)
        print(json.dumps(rec), flush=True)

    try:
        layer = MultichipDecoder.from_state_dict(sd, hf_config=c, layer_idx=args.layer, mesh_device=mesh)
        assert layer.native_local and layer.residual_layout == "replicated"
        linear = layer.kind == "linear_attention"
        result["kind"] = layer.kind
        # Distinct physical capacities also give heterogeneous logical page-table widths.
        base_context = max(96, ((40 + 4 * args.steps + 31) // 32) * 32)
        assert args.capacity_extra_pages >= 0
        base_context += 32 * args.capacity_extra_pages
        contexts = [base_context] * 32 + [base_context + 64] * 3 + [base_context + 32]
        states = layer.allocate_batch_state(batch=32, context=contexts[0])
        states += layer.allocate_batch_state(batch=3, context=contexts[32])
        states += [layer.allocate_state(context=contexts[35])]
        controls = [layer.allocate_state(context=capacity) for capacity in contexts]
        lengths = [31 + i % 3 for i in range(len(states))]
        tables, state_probe_pages = [], []
        for capacity in contexts:
            local_table = torch.randperm(capacity // 32, dtype=torch.int32)[None]
            state_probe_pages.append(sorted({0, capacity // 32 - 1, *local_table[0, :2].tolist()}))
            tables.append(host(local_table, True))
        result["contexts"] = contexts
        result["page_table_widths"] = [table.shape[-1] for table in tables]
        result["capacity_extra_pages"] = args.capacity_extra_pages
        result["state_scope"] = (
            "Sampled physical pages: first, last and pages covering valid prefix/decode; complete state in ordinary pool tests"
            if args.capacity_extra_pages and not linear
            else "Complete state"
        )
        result["prefix_lengths"] = lengths
        for index, (state, control) in enumerate(zip(states, controls)):
            x = host((torch.randn(1, 1, lengths[index], 4096) * 0.1).bfloat16())
            with device_only():
                actual = layer.prefill_forward(x, state=state, start_position=0, page_table=tables[index])
                expected = layer.prefill_forward(x, state=control, start_position=0, page_table=tables[index])
            record_pcc(f"prefill_{index}", ranks(actual), ranks(expected))
            del x, actual, expected

        def state_values(collection):
            """Pure host copies; no temporary device slices survive a replay."""
            downloaded = {}
            values = []
            for index, state in enumerate(collection):
                item = {}
                for name in ("conv", "recurrent") if linear else ("key", "value"):
                    value = getattr(state, name)
                    if args.capacity_extra_pages and not linear:
                        # Transition tests sample sparse writes and sentinels instead
                        # of downloading every unused page. Ordinary tests stay full.
                        pieces = [
                            ranks(value[state.page_offset + page : state.page_offset + page + 1])
                            for page in state_probe_pages[index]
                        ]
                        item[name] = [torch.cat([piece[rank] for piece in pieces], dim=0) for rank in range(4)]
                        continue
                    if id(value) not in downloaded:
                        downloaded[id(value)] = ranks(value)
                    tensors = downloaded[id(value)]
                    if linear and state.state_row is not None:
                        if name == "conv" and value.layout == ttnn.TILE_LAYOUT:
                            tensors = [tensor.permute(2, 1, 0, 3) for tensor in tensors]
                        begin, end = state.state_row, state.state_row + 1
                    elif not linear and state.page_count is not None:
                        begin, end = state.page_offset, state.page_offset + state.page_count
                    else:
                        begin, end = 0, value.shape[0]
                    item[name] = [tensor[begin:end].clone() for tensor in tensors]
                values.append(item)
            return values

        def unchanged(label, before, after, active):
            count = 0
            for index in set(range(len(states))) - set(active):
                for name in before[index]:
                    assert all(torch.equal(a, b) for a, b in zip(before[index][name], after[index][name])), (
                        label,
                        "untouched state changed",
                        index,
                        name,
                    )
                count += 1
            field = (
                "untouched_requests_sampled_pages_bitwise_equal"
                if args.capacity_extra_pages and not linear
                else "untouched_requests_bitwise_equal"
            )
            records.append({"label": label + "_isolation", field: count})

        def check_states(label, actual, expected, active):
            for index in active:
                for name in actual[index]:
                    record_pcc(f"{label}_{name}_{index}", actual[index][name], expected[index][name])

        unique = {id(t): t for state in states + controls for t in buffers(state)}
        snapshots = [(value, ttnn.clone(value)) for value in unique.values()]

        def restore():
            with device_only():
                for value, snapshot in snapshots:
                    ttnn.copy(snapshot, value)

        cases = {
            "full": list(range(32)),
            "subset": [31, 0, 7],
            "reversed": list(reversed(range(32))),
            "single": [19],
            "mixed": [31, 34, 0, 32, 35],
        }
        # Allocate all persistent trace I/O before capture. Only outputs are acknowledged.
        inputs = {
            name: host(torch.zeros(1, 1, len(indices), 4096, dtype=torch.bfloat16)) for name, indices in cases.items()
        }
        positions = {
            name: [host(torch.tensor([lengths[index]], dtype=torch.int32), True) for index in indices]
            for name, indices in cases.items()
        }
        control_input = host(torch.zeros(1, 1, 1, 4096, dtype=torch.bfloat16))
        control_position = host(torch.zeros(1, dtype=torch.int32), True)
        reset_values = {}
        if linear:
            reset_values = {name: ttnn.zeros_like(getattr(controls[7], name)) for name in ("conv", "recurrent")}

        def forward(name):
            indices = cases[name]
            with device_only():
                return layer.decode_batch_packed(
                    inputs[name],
                    states=[states[i] for i in indices],
                    current_positions=positions[name],
                    page_tables=[tables[i] for i in indices],
                )

        def run_step(name, label, counters, trace=None, output=None):
            indices = cases[name]
            before = state_values(states)
            x = (torch.randn(1, 1, len(indices), 4096) * 0.1).bfloat16()
            ttnn.copy_host_to_device_tensor(host(x, device=False), inputs[name])
            for row, index in enumerate(indices):
                position = torch.tensor([lengths[index] + counters[index]], dtype=torch.int32)
                ttnn.copy_host_to_device_tensor(host(position, True, device=False), positions[name][row])
            if trace is None:
                output = forward(name)
            else:
                with device_only():
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            actual = ranks(output)
            for row, index in enumerate(indices):
                rx = x[:, :, row : row + 1]
                ttnn.copy_host_to_device_tensor(host(rx, device=False), control_input)
                position = torch.tensor([lengths[index] + counters[index]], dtype=torch.int32)
                ttnn.copy_host_to_device_tensor(host(position, True, device=False), control_position)
                with device_only():
                    expected = layer.decode_forward(
                        control_input,
                        state=controls[index],
                        current_position=control_position,
                        page_table=tables[index],
                    )
                record_pcc(f"{label}_output_{index}", [v[:, :, row : row + 1] for v in actual], ranks(expected), rx)
                del expected
                counters[index] += 1
            after = state_values(states)
            unchanged(label, before, after, indices)
            check_states(label, after, state_values(controls), indices)
            save()

        for name in cases:
            restore()
            counters = [0] * len(states)
            for step in range(args.steps):
                run_step(name, f"eager_{name}_{step}", counters)

        from ttnn.tools import trace_allocation_tracker

        def allocation_churn():
            with device_only():
                noise = ttnn.zeros((1, 1, 32, 4096), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                del noise

        def reset_request():
            with device_only():
                for name, zero in reset_values.items():
                    layer.write_linear_state(states[7], name, zero)
                    layer.write_linear_state(controls[7], name, zero)

        # Exact post-capture helper programs must create immutable cached constants
        # before any trace reserves temporary addresses. Keep tracking enabled;
        # only the returned trace outputs below are acknowledged as corruptible.
        allocation_churn()
        if linear:
            reset_request()
        restore()
        ttnn.synchronize_device(mesh)
        result["prewarmed_post_capture_helpers"] = ["allocation_churn"] + (["reset_request"] if linear else [])

        captured = {}
        for name in ("full", "subset"):
            restore()
            warm = forward(name)
            del warm
            restore()
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            value = forward(name)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            trace_ids.append(trace)
            trace_allocation_tracker.acknowledge_corruptible(value)
            captured[name] = (trace, value)

        # Two live traces with different sizes, refreshed inputs, evolving state,
        # and restored backing allocations between rounds.
        for step in range(args.steps):
            restore()
            counters = [0] * len(states)
            for turn, name in enumerate(("full", "subset", "full", "subset")):
                allocation_churn()
                trace, value = captured[name]
                run_step(name, f"interleaved_{step}_{turn}_{name}", counters, trace, value)

        if linear:
            restore()
            before = state_values(states)
            reset_request()
            after = state_values(states)
            unchanged("linear_reset_7", before, after, [7])
            for name in reset_values:
                assert all(torch.count_nonzero(t) == 0 for t in after[7][name]), ("reset failed", name)
            trace, value = captured["subset"]
            run_step("subset", "reset_then_existing_trace", [0] * len(states), trace, value)
        result["trace_replays"] = args.steps * 4 + int(linear)
        result["pass"] = True
        save()
        print("POOL_CONTRACTS_PASS", args.layer, flush=True)
    except Exception as error:
        result["error"] = repr(error)
        save()
        raise
    finally:
        for trace in trace_ids:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
