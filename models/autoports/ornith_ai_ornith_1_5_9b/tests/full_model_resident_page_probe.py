# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tracked sparse-page admission and mapped-boundary replay on real layers 0/3.

Unsafe sparse-page requests must reject before any trace. Compare successful
changed-table and resident continuations with explicit token/position feedback.
"""


def main():
    import argparse
    import json
    from contextlib import ExitStack
    from pathlib import Path
    from unittest.mock import patch

    import torch
    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
    from models.common.sampling.generator import SamplingParams

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    report = {
        "layers": [0, 3],
        "batch": 3,
        "context": 128,
        "trace_allocation_tracking": trace_allocation_tracker.TRACE_ALLOC_TRACKING,
        "cases": [],
    }
    assert report["trace_allocation_tracking"] == 1
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        gen = build_generator(None, mesh, max_batch_size=3, max_seq_len=128, layer_indices=[0, 3])
        params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
        lengths = [31, 32, 33]
        prompts = torch.stack([torch.arange(33) + 100 + 100 * slot for slot in range(3)])
        positions = [31, 32, -1]
        full = gen.page_table.clone()
        sparse = full.clone()
        sparse[0, 1:] = -1
        sparse[2, :] = -1
        mapped = full.clone()
        mapped[2, :] = -1
        state = gen.prepare_decode(kv_cache=gen.cache, page_table=full, slots=(0, 1, 2))
        gen.capture(state)

        def start(table):
            first = gen.prefill_forward(
                prompts,
                prompt_lens=lengths,
                empty_slots=[0, 1, 2],
                page_table=full,
                kv_cache=gen.cache,
                sampling_params=params,
            )
            return gen.decode_forward(
                first,
                positions,
                page_table=table,
                kv_cache=gen.cache,
                sampling_params=params,
                reload_inputs=True,
                reload_sampling_params=True,
                reset_sampling_state=True,
                prompt_tokens=[prompts[s, : lengths[s]].tolist() for s in range(3)],
            )

        def forward(token=None, pos=None, table=None, reload=False, refresh=False):
            return gen.decode_forward(
                token,
                pos,
                page_table=table,
                kv_cache=gen.cache,
                sampling_params=params,
                reload_inputs=reload,
                reload_page_table=refresh,
            )

        def record(name, passed, **details):
            report["cases"].append({"case": name, "passed": bool(passed), **details})
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            assert passed, name

        # Warm the explicit mixed/inactive command shape, then freeze programs.
        start(mapped)
        programs = mesh.num_program_cache_entries()
        traces = (dict(state.model_traces), dict(state.sample_traces))
        mesh.set_program_cache_misses_allowed(False)
        control = start(mapped)
        for step in range(1, 4):
            control = forward(control, [31 + step, 32 + step, -1], mapped, reload=True)
        report["explicit_final_tokens"] = control.tolist()

        def resident(steps):
            def forbidden(*_args, **_kwargs):
                raise AssertionError("Host boundary inside resident submission")

            with ExitStack() as stack:
                for name in ("to_torch", "from_torch", "copy_host_to_device_tensor", "synchronize_device"):
                    stack.enter_context(patch.object(ttnn, name, forbidden))
                tensor = gen.decode_resident(steps)
            ttnn.synchronize_device(mesh)
            return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).reshape(-1)[:3]

        start(mapped)
        record("mapped_page_boundary_resident_matches_explicit", torch.equal(resident(3)[:2].long(), control[:2, 0]))
        start(sparse)  # Legal final mapped position 31; slot 2 remains inactive.
        counters, advances = dict(gen.counters), state.position_advances
        for name, call in (
            ("bulk_first_unmapped", lambda: gen.decode_resident(1)),
            ("bulk_window_crossing_hole", lambda: gen.decode_resident(34)),
            ("ordinary_first_unmapped", forward),
        ):
            try:
                call()
            except ValueError as exc:
                record(
                    name,
                    "causal page" in str(exc) and gen.counters == counters and state.position_advances == advances,
                    message=str(exc),
                )
            else:
                raise AssertionError(f"{name} submitted invalid cache address")
        refresh_before = gen.counters["page_table_refreshes"]
        forward(table=mapped, refresh=True)
        record(
            "commanded_page_table_refresh",
            # Only slot0 grows from one causal page to its complete map.
            # Slot1 is equal and the inactive slot2 remains entirely unmapped.
            gen.counters["page_table_refreshes"] - refresh_before == 1,
            refreshed_rows=gen.counters["page_table_refreshes"] - refresh_before,
            expected_changed_rows=1,
        )
        record("changed_table_continuation_matches_explicit", torch.equal(resident(2)[:2].long(), control[:2, 0]))
        record("inactive_position_preserved", state.host_positions[2] == -1)
        record(
            "trace_and_program_identity",
            programs == mesh.num_program_cache_entries() and traces == (state.model_traces, state.sample_traces),
            programs=programs,
        )
        report["status"] = "passed"
        report["final_counters"] = dict(gen.counters)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
