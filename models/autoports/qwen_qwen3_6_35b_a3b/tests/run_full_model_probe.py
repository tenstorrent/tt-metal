# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bounded real-weight reduced full-model probe. Run with external timeout."""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import QwenModel
from models.autoports.qwen_qwen3_6_35b_a3b.tt.serving_capabilities import QwenServingCapabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--layers", default="0,3")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--positive-control", action="store_true")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--resident-l1-bytes", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-probe", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source_root = Path(__file__).parents[1]
    provenance = {
        "command": sys.argv,
        "time_unix": time.time(),
        "source_sha256": {
            str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                source_root / "tt/model.py",
                source_root / "tt/generator.py",
                source_root / "tt/multichip_decoder.py",
            ]
        },
        "environment": {
            k: os.environ.get(k) for k in ["TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"]
        },
    }
    Path(args.output + ".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    trace_region_size = QwenServingCapabilities.recommended_trace_region_size(args.batch_size)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=trace_region_size)
    try:
        model = QwenModel(
            mesh,
            layer_indices=None if args.all_layers else [int(x) for x in args.layers.split(",")],
            max_seq_len=args.context,
        )
        result = {"memory": model.memory_views, "layers": model.layer_indices, "trace_region_size": trace_region_size}
        if args.resident_l1_bytes:
            allocated = model.memory_views[-1]["l1"]["total_bytes_allocated_per_bank"]
            ballast_bytes = args.resident_l1_bytes - allocated
            assert ballast_bytes > 0 and ballast_bytes % 4 == 0
            core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
            memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(core, [1, ballast_bytes // 4], ttnn.ShardOrientation.ROW_MAJOR),
            )
            ballast = ttnn.from_torch(
                torch.zeros(1, ballast_bytes // 4),
                dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=memory,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            result["ballast_bytes"] = ballast_bytes
            model.census("ballast_resident")
        if args.generate:
            from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import QwenGenerator

            generator = QwenGenerator(model, max_batch_size=args.batch_size)
            try:

                def host(t):
                    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).clone()

                if args.batch_probe:
                    # Compile diagnostic readback layouts before generate()'s
                    # sole prepare/capture. The probe checks every inactive slot
                    # on mesh chip 0, whose local state must remain bit-exact.
                    for tensor in (
                        generator.tokens,
                        generator.sampler.tt_sampling.seeds_tt_tensor,
                        *generator.positions,
                    ):
                        host(tensor)
                    for states in generator.cache:
                        for key in ("conv", "recurrent"):
                            if key in states[0]:
                                host(states[0][key])
                if args.positive_control:
                    assert args.batch_size == 1, "The allocation positive control selects the B1 model trace"
                    generator.prepare(prefill_cases=((4, 0),))
                    unsafe = model.upload(
                        torch.ones(1, 32, dtype=torch.bfloat16), ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
                    )
                    rejected = False
                    try:
                        ttnn.execute_trace(mesh, generator.traces["model"], blocking=True)
                    except RuntimeError as exc:
                        rejected = "device buffer(s) still alive before" in str(exc)
                        print("TRACKER_POSITIVE_CONTROL", str(exc)[:1000], flush=True)
                        result["positive_control_error"] = str(exc)[:1000]
                    finally:
                        ttnn.deallocate(unsafe)
                    assert rejected, "Allocation tracker failed its positive control"
                    result["positive_control_rejected"] = True
                result["tokens"] = generator.generate([1, 2, 3, 4], 4)
                if args.batch_probe:
                    assert args.batch_size > 1
                    prepared_programs = int(mesh.num_program_cache_entries())
                    prepared_captures = generator.counters["captures"]
                    generator.reset()
                    slots = tuple(range(args.batch_size))
                    width = next(
                        states[0]["page_table"].shape[-1] for states in generator.cache if "page_table" in states[0]
                    )
                    assert width >= args.batch_size
                    pages = torch.stack([torch.arange(width).roll(-s) for s in slots]).to(torch.int32)
                    generator.prefill_forward(
                        torch.tensor([[1, 2, 3, 4]] * args.batch_size),
                        page_table=pages,
                        prompt_lens=[4] * args.batch_size,
                        slots=slots,
                        device_output=True,
                    )
                    first = generator.decode_forward(
                        torch.tensor([5] * args.batch_size), [4] * args.batch_size, slots=slots
                    )

                    saved = [
                        (slot, state[slot][key], host(state[slot][key]))
                        for state in generator.cache
                        for slot in slots[1:]
                        for key in ("conv", "recurrent")
                        if key in state[slot]
                    ]
                    old_tokens = host(generator.tokens).flatten().clone()
                    old_seeds = host(generator.sampler.tt_sampling.seeds_tt_tensor).flatten().clone()
                    old_positions = [host(position) for position in generator.positions]
                    assert [int(position.flatten()[0]) for position in old_positions] == [5] * args.batch_size
                    # Resident decode ignores host start_pos entirely. Activity
                    # changes are explicit admission operations, not inferred
                    # from the stale -1 values passed with refresh_inputs=False.
                    generator._set_active_slots((0,))
                    assert generator.active_slots == (0,)
                    resident_baseline = dict(generator.counters)
                    generator.decode_forward(
                        None, [5] + [-1] * (args.batch_size - 1), slots=slots, refresh_inputs=False
                    )
                    for slot, value, snapshot in saved:
                        assert torch.equal(host(value), snapshot), f"Inactive slot {slot} recurrent/conv state changed"
                    assert torch.equal(
                        host(generator.tokens).flatten()[1 : args.batch_size], old_tokens[1 : args.batch_size]
                    )
                    assert torch.equal(
                        host(generator.sampler.tt_sampling.seeds_tt_tensor).flatten()[1 : args.batch_size],
                        old_seeds[1 : args.batch_size],
                    )
                    for slot in slots[1:]:
                        assert torch.equal(host(generator.positions[slot]), old_positions[slot])
                    assert int(host(generator.positions[0]).flatten()[0]) == 6
                    generator._set_active_slots(slots)
                    assert generator.active_slots == slots
                    reordered = generator.decode_forward(
                        None, [5] * args.batch_size, slots=slots[::-1], refresh_inputs=False
                    )
                    final_positions = [int(host(position).flatten()[0]) for position in generator.positions]
                    assert final_positions == [7] + [6] * (args.batch_size - 1)
                    final_tokens = host(generator.tokens).flatten()
                    assert reordered.tolist() == [int(final_tokens[slot]) for slot in slots[::-1]]
                    resident_delta = {key: generator.counters[key] - value for key, value in resident_baseline.items()}
                    assert all(
                        resident_delta[key] == 0
                        for key in ("token_refreshes", "position_refreshes", "page_table_refreshes", "captures")
                    )
                    assert generator.counters["captures"] == prepared_captures
                    assert int(mesh.num_program_cache_entries()) == prepared_programs, "Post-prepare program growth"
                    result["batch"] = {
                        "size": args.batch_size,
                        "first_tokens": first.tolist(),
                        "reordered_tokens": reordered.tolist(),
                        "inactive_state_unchanged": True,
                        "inactive_slots_checked": list(slots[1:]),
                        "state_readback_mesh_chip": 0,
                        "final_positions": final_positions,
                        "resident_counter_delta": resident_delta,
                        "program_cache_entries": prepared_programs,
                        "program_cache_growth": 0,
                    }
                result["counters"] = generator.counters
            finally:
                generator.close()
        elif not args.load_only:
            cache = model.allocate_cache(context_length=args.context)
            tokens = model.upload(
                torch.tensor([[1, 2, 3, 4]], dtype=torch.int32), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            logits = model.prefill(tokens, cache=cache)
            host = model.host_logits(logits)
            assert torch.isfinite(host).all()
            result["prefill_shape"] = list(host.shape)
            decode_tokens = model.upload(
                torch.tensor([[5]], dtype=torch.int32), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            pos = model.upload(torch.tensor([4], dtype=torch.int32), ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            logits = model.decode(decode_tokens, cache=cache, positions=[pos], slots=[0])
            host = model.host_logits(logits)
            assert torch.isfinite(host).all()
            result["decode_shape"] = list(host.shape)
            result["advanced_position"] = ttnn.to_torch(ttnn.get_device_tensors(pos)[0]).tolist()
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print("FULL_MODEL_PROBE_OK", flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
