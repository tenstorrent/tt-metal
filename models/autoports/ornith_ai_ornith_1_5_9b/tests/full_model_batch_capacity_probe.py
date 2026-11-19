# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialized real-device capacity check for fixed-slot public serving calls.

Default B32/context512 uses real layers0/3; --all-layers selects the full stack.
All preparation precedes capture. This diagnostic reads tokens/state to verify
results; resident decode receives no host token, position, or page-table input.
"""


def main():
    import argparse
    import hashlib
    import json
    import time
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--all-active", action="store_true")
    parser.add_argument("--full-pool-restore", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.batch <= 32 or args.context < 96 or args.context % 32:
        parser.error("Require batch1..32 and page-aligned context>=96")

    import torch

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
    from models.common.sampling.generator import SamplingParams

    torch.set_num_threads(8)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    batch = args.batch
    slots = tuple(range(batch))
    selected = sorted({0, batch // 2, batch - 1})
    inactive = 1 if batch > 1 and not args.all_active else None
    lengths = [33 + slot % 32 for slot in slots]
    report = {
        "batch": batch,
        "context": args.context,
        "layers": "all" if args.all_layers else [0, 3],
        "prompt_lens": lengths,
        "inactive_slot": inactive,
        "cases": [],
        "selective_inactive_restore": not args.full_pool_restore,
    }
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        gen = build_generator(
            None,
            mesh,
            max_batch_size=batch,
            max_seq_len=args.context,
            layer_indices=None if args.all_layers else [0, 3],
            selective_inactive_restore=not args.full_pool_restore,
        )
        params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
        prompts = torch.zeros((batch, max(lengths)), dtype=torch.int64)
        for slot, length in enumerate(lengths):
            prompts[slot, :length] = torch.arange(length) + 100 + 100 * slot

        def record(case, passed, **details):
            item = {"case": case, "passed": bool(passed), **details}
            report["cases"].append(item)
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(item), flush=True)
            if not passed:
                raise AssertionError(case)

        def read(tensor):
            return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])

        def cache_hash(slot):
            result = {}
            for index, (layer, states) in enumerate(zip(gen.model.layers, gen.cache, strict=True)):
                state = states[slot]
                for name in ("conv", "recurrent") if layer.kind == "linear_attention" else ("key", "value"):
                    tensor = getattr(state, name)
                    for rank, shard in enumerate(ttnn.get_device_tensors(tensor)):
                        value = ttnn.to_torch(shard)
                        if name == "conv":
                            value = value[:, :, slot : slot + 1]
                        elif name == "recurrent":
                            value = value[slot : slot + 1]
                        else:
                            value = value[state.page_offset : state.page_offset + state.page_count]
                        result[f"{index}.{name}.{rank}"] = hashlib.sha256(
                            value.float().contiguous().numpy().tobytes()
                        ).hexdigest()
            return result

        def prefill(order=slots, sampling=None):
            return gen.prefill_forward(
                prompts[list(order)],
                prompt_lens=[lengths[slot] for slot in order],
                empty_slots=list(order),
                page_table=gen.page_table[list(order)],
                kv_cache=gen.cache,
                sampling_params=sampling,
            )

        # Structural predicates exactly match the native decoder pool branches.
        for layer, states in zip(gen.model.layers, gen.cache, strict=True):
            assert layer.native_local and len(states) == batch
            if layer.kind == "linear_attention":
                assert all(
                    st.conv is states[0].conv and st.recurrent is states[0].recurrent and st.state_row == slot
                    for slot, st in enumerate(states)
                )
                assert tuple(states[0].conv.shape) == (3, 1, batch, 2048)
                assert states[0].conv.layout == ttnn.TILE_LAYOUT
                assert states[0].recurrent.shape[0] == batch
            else:
                assert states[0].page_offsets is not None
                assert all(
                    st.key is states[0].key and st.value is states[0].value and st.page_offset == slot * st.page_count
                    for slot, st in enumerate(states)
                )
                assert states[0].key.shape[0] == batch * states[0].page_count
        state = gen.prepare_decode(kv_cache=gen.cache, page_table=gen.page_table, slots=slots)
        reference = prefill()
        record("public_host_prefill_shape", tuple(reference.shape) == (batch, 1, gen.model.vocab_size))
        original_host_logits = gen.model.host_logits

        def forbid_logits(*_args, **_kwargs):
            raise AssertionError("Device sampling crossed the host-logits boundary")

        gen.model.host_logits = forbid_logits
        try:
            first = prefill(sampling=params)
        finally:
            gen.model.host_logits = original_host_logits
        record(
            "device_prefill_semantic_greedy",
            tuple(first.shape) == (batch, 1) and torch.equal(first, reference.argmax(dim=-1)),
        )
        gen.capture(state)
        trace_ids = (dict(state.model_traces), dict(state.sample_traces))
        captures = gen.counters["captures"]
        baseline = mesh.num_program_cache_entries()
        report["prepared_programs"] = baseline
        mesh.set_program_cache_misses_allowed(False)

        for slot in selected:
            peers = {peer: cache_hash(peer) for peer in selected if peer != slot}
            solo = prefill((slot,))
            record(
                "selected_slot_matches_batch",
                torch.equal(solo, reference[slot : slot + 1]) and peers == {peer: cache_hash(peer) for peer in peers},
                slot=slot,
            )
        order = tuple(reversed(slots))
        reordered = prefill(order)
        record("request_row_order_determinism", torch.equal(reordered, reference[list(order)]), order=list(order))
        first = prefill(sampling=params)
        positions = [length if slot != inactive else -1 for slot, length in enumerate(lengths)]
        active = [slot for slot in slots if slot != inactive]
        inactive_before = cache_hash(inactive) if inactive is not None else None

        def decode(tokens, pos, *, reload, reset=False):
            return gen.decode_forward(
                tokens,
                pos,
                page_table=gen.page_table if reload else None,
                kv_cache=gen.cache,
                sampling_params=params,
                reload_inputs=reload,
                reload_sampling_params=reset,
                reset_sampling_state=reset,
                prompt_tokens=[prompts[s, : lengths[s]].tolist() for s in slots] if reset else None,
            )

        outputs = []
        gen.model.host_logits = forbid_logits
        try:
            outputs.append(decode(first, positions, reload=True, reset=True))
            before = dict(gen.counters)
            masks_before = state.batch_state.mask_refreshes
            original_copy = ttnn.copy_host_to_device_tensor

            def forbid_upload(*_args, **_kwargs):
                raise AssertionError("Resident decode performed a host-to-device input upload")

            ttnn.copy_host_to_device_tensor = forbid_upload
            try:
                for _ in range(2):
                    previous = outputs[-1]
                    resident = read(state.tokens).reshape(-1)[:batch].long()
                    assert torch.equal(resident[active], previous[:, 0][active]), "Sampled feedback buffer mismatch"
                    outputs.append(decode(None, None, reload=False))
            finally:
                ttnn.copy_host_to_device_tensor = original_copy
        finally:
            gen.model.host_logits = original_host_logits
        actual_positions = [int(read(pos).item()) for pos in state.positions]
        expected_positions = [pos + 3 if pos >= 0 else -1 for pos in positions]
        record(
            "resident_token_feedback_positions_and_inactive_guard",
            all(tuple(output.shape) == (batch, 1) for output in outputs)
            and actual_positions == expected_positions
            and state.batch_state.mask_refreshes == masks_before
            and all(
                gen.counters[key] == before[key]
                for key in ("token_refreshes", "position_refreshes", "page_table_refreshes")
            )
            and gen.counters["model_replays"] - before["model_replays"] == 2
            and gen.counters["sampling_replays"] - before["sampling_replays"] == 2
            and (inactive is None or cache_hash(inactive) == inactive_before),
            positions=actual_positions,
            tokens=[output.reshape(-1).tolist() for output in outputs],
        )
        # Independent explicit-feedback control starts from a clean re-prefill.
        # Diagnostic host feedback is confined to this control, never the tested
        # resident path above; both use the same public traced decode API.
        token = prefill(sampling=params)
        control = []
        for step in range(3):
            pos = [position + step if position >= 0 else -1 for position in positions]
            token = decode(token, pos, reload=True, reset=step == 0)
            control.append(token)
        record(
            "resident_matches_explicit_feedback_control",
            all(
                torch.equal(actual[active], expected[active]) for actual, expected in zip(outputs, control, strict=True)
            ),
        )
        if args.benchmark_steps:
            before = dict(gen.counters)
            inactive_snapshot = cache_hash(inactive) if inactive is not None else None
            ttnn.synchronize_device(mesh)
            begin = time.perf_counter()
            resident = gen.decode_resident(args.benchmark_steps)
            ttnn.synchronize_device(mesh)
            elapsed = time.perf_counter() - begin
            delta = {key: value - before[key] for key, value in gen.counters.items()}
            record(
                "batched_resident_no_readback",
                delta["model_replays"] == delta["sampling_replays"] == args.benchmark_steps
                and all(value == 0 for key, value in delta.items() if key not in ("model_replays", "sampling_replays"))
                and all(
                    int(read(pos).item()) == (start + 3 + args.benchmark_steps if start >= 0 else -1)
                    for start, pos in zip(positions, state.positions)
                )
                and (inactive is None or cache_hash(inactive) == inactive_snapshot),
                steps=args.benchmark_steps,
                active_users=len(active),
                milliseconds_per_token=elapsed * 1000 / args.benchmark_steps,
                tokens_per_second_per_user=args.benchmark_steps / elapsed,
                counter_delta=delta,
                final_tokens=read(resident).reshape(-1)[:batch].tolist(),
            )
        record(
            "live_requests_reuse_programs_and_traces",
            mesh.num_program_cache_entries() == baseline
            and gen.counters["captures"] == captures
            and state.model_traces == trace_ids[0]
            and state.sample_traces == trace_ids[1],
            final_programs=mesh.num_program_cache_entries(),
        )
        report["passed"] = True
    except BaseException as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
        mesh.set_program_cache_misses_allowed(True)
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
