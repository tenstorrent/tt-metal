# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real layers0/3 fixed-slot, inactive-gap and lifecycle probe.

Externally serialize and timeout this job. Exercises the production model path.
No sampler or whole-model quality/performance claim is made by this probe.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, choices=[3, 4], default=4)
    args = parser.parse_args()
    import torch

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.batch_state import BatchState
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel

    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    trace_id = None
    try:
        model = OrnithModel(mesh, layer_indices=[0, 3])
        batch, context, inactive = args.batch, 256, 1
        cache = model.allocate_cache(batch=batch, context=context)
        guard = BatchState(model, cache)
        table = model.upload(
            torch.arange(context // 32, dtype=torch.int32)[None], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        tables = [table] * batch
        for slot in range(batch):
            tokens = model.upload(
                torch.arange(33)[None] + 100 + slot * 41, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            model.prefill(tokens, cache=cache, slot=slot, start=0, page_table=table)
        buffers = [(cache[0][0].conv, 3), (cache[0][0].recurrent, 1), (cache[1][0].key, 1), (cache[1][0].value, 1)]
        names = ["conv", "recurrent", "key", "value"]
        initial = [ttnn.zeros_like(value) for value, _ in buffers]
        allactive = [ttnn.zeros_like(value) for value, _ in buffers]

        def save(destination):
            for (value, _), saved in zip(buffers, destination):
                ttnn.copy(value, saved)

        def restore(saved):
            for (value, _), source in zip(buffers, saved):
                ttnn.copy(source, value)

        def state_host():
            return {
                name: ttnn.to_torch(value, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=dim))
                for name, (value, dim) in zip(names, buffers)
            }

        def row(value, name, slot):
            if name == "conv":
                return value[:, :, slot : slot + 1, :]
            if name == "recurrent":
                return value[slot : slot + 1]
            pages = context // 32
            return value[slot * pages : (slot + 1) * pages]

        def digest(value):
            return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

        def hashes(snapshot):
            return {name: [digest(row(value, name, slot)) for slot in range(batch)] for name, value in snapshot.items()}

        def assert_equal_states(a, b, slots):
            for name in names:
                for slot in slots:
                    assert torch.equal(row(a[name], name, slot), row(b[name], name, slot)), (name, slot)

        before_warmup = state_host()
        guard.warmup_slot_ops()
        assert_equal_states(state_host(), before_warmup, range(batch))

        decode_tokens = model.upload(
            torch.tensor([[[[300 + i if i < batch else 0 for i in range(32)]]]]),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        positions = [
            model.upload(torch.tensor([33], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            for _ in range(batch)
        ]

        def set_positions(values):
            for target, value in zip(positions, values):
                host = model.upload(
                    torch.tensor([value], dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=False,
                )
                ttnn.copy_host_to_device_tensor(host, target)

        def decode(masked):
            return model.decode(
                decode_tokens,
                cache=cache,
                slots=tuple(range(batch)),
                positions=positions,
                page_tables=tables,
                batch_state=guard,
                masked=masked,
            )

        def logits_host(masked):
            return model.host_logits(decode(masked))

        save(initial)
        before = state_host()
        logits_host(False)
        save(allactive)
        active_after = state_host()
        restore(initial)
        active_mask = [slot != inactive for slot in range(batch)]
        guard.set_active(active_mask)
        set_positions([33 if active else -1 for active in active_mask])
        logits_host(True)
        masked_after = state_host()
        assert_equal_states(masked_after, before, [inactive])
        assert_equal_states(masked_after, active_after, [slot for slot in range(batch) if slot != inactive])
        assert any(not torch.equal(row(before[name], name, 0), row(masked_after[name], name, 0)) for name in names)

        # Re-admit the frozen slot at its unchanged position. The independent
        # oracle starts from native all-active decode, then restores only that
        # slot from initial storage (without the mask/where helper).
        next_positions = [33 if slot == inactive else 34 for slot in range(batch)]
        guard.set_active([True] * batch)
        set_positions(next_positions)
        next_logits = logits_host(False)
        next_state = state_host()
        restore(allactive)
        old_linear = SimpleNamespace(conv=initial[0], recurrent=initial[1], state_row=inactive)
        for name in ("conv", "recurrent"):
            value = model.layers[0].read_linear_state(old_linear, name)
            model.layers[0].write_linear_state(cache[0][inactive], name, value)
        pages = context // 32
        for index in (2, 3):
            current = buffers[index][0]
            start, end = inactive * pages, (inactive + 1) * pages
            pieces = [current[:start], initial[index][start:end], current[end:]]
            ttnn.copy(ttnn.concat(pieces, dim=0), current)
        set_positions(next_positions)
        oracle_logits = logits_host(False)
        oracle_state = state_host()
        assert torch.equal(next_logits, oracle_logits), "Re-admitted next decode differs from independent state oracle"
        assert_equal_states(next_state, oracle_state, range(batch))

        # Include the last fixed slot even when a future decode bucket contains
        # only the first three rows: remap is a whole-capacity operation.
        remap = list(range(1, batch)) + [0]
        premove = state_host()
        guard.permute_slots(remap)
        postmove = state_host()
        for name in ("conv", "recurrent"):
            for destination, source_slot in enumerate(remap):
                assert torch.equal(row(postmove[name], name, destination), row(premove[name], name, source_slot))
        for name in ("key", "value"):
            assert torch.equal(postmove[name], premove[name]), "Slot permutation must not move scheduler-owned KV"
        guard.reset_slots([inactive])
        postreset = state_host()
        assert_equal_states(postreset, postmove, [slot for slot in range(batch) if slot != inactive])
        for name in ("conv", "recurrent"):
            assert torch.count_nonzero(row(postreset[name], name, inactive)) == 0
        for name in ("key", "value"):
            assert torch.equal(postreset[name], postmove[name])

        # Final trace-only test: exact scratch/masks already exist and all
        # programs were used above. Warm sentinel-safe position advance too.
        restore(initial)
        guard.set_active(active_mask)
        set_positions([33 if active else -1 for active in active_mask])
        logits_host(True)
        for position in positions:
            ttnn.plus_one(position, skip_negative_entries=True)
        restore(initial)
        set_positions([33 if active else -1 for active in active_mask])
        ttnn.synchronize_device(mesh)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        trace_output = decode(True)
        for position in positions:
            ttnn.plus_one(position, skip_negative_entries=True)
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        from ttnn.tools import trace_allocation_tracker

        # Only final logits are retained from capture; this state-focused probe
        # does not consume them between replays. Scratch predates capture.
        trace_allocation_tracker.acknowledge_corruptible(trace_output)
        restore(initial)
        set_positions([33 if active else -1 for active in active_mask])
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        replay1 = state_host()
        assert_equal_states(replay1, masked_after, range(batch))
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        replay2 = state_host()
        assert_equal_states(replay2, before, [inactive])
        final_positions = [int(ttnn.to_torch(ttnn.get_device_tensors(value)[0]).item()) for value in positions]
        assert final_positions == [35 if active else -1 for active in active_mask]
        result = {
            "status": "pass",
            "layers": [0, 3],
            "batch": batch,
            "inactive_slot": inactive,
            "production_model_path": True,
            "initial_hashes": hashes(before),
            "masked_hashes": hashes(masked_after),
            "active_matches_native": True,
            "inactive_unchanged": True,
            "next_decode_matches_oracle": True,
            "whole_pool_permutation": remap,
            "reset_preserves_peers": True,
            "warmup_preserves_state": True,
            "traced_replays": 2,
            "final_positions": final_positions,
            "mask_refreshes": guard.mask_refreshes,
            "limitation": "Reduced real layers; no full-model accuracy or performance claim.",
        }
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print("BATCH_STATE_PROBE_PASS", args.output, flush=True)
    finally:
        if trace_id is not None:
            ttnn.release_trace(mesh, trace_id)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
