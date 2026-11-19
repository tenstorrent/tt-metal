# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Public prefill/decode with caller-owned arbitrary global KV page pools.

Run serialized under an external timeout. Default covers three fixed slots at
context256, a permuted24-block pool and an arbitrary11-block pool whose unused
virtual pages are -1. All caches, reset helpers, decode bindings, and restoration
buffers are prepared before any capture. Host-sampling decode is explicit via
sampling_params=None; this probe never substitutes a private model path.

Invalid-page checks intercept the dispatch boundary. A missing host rejection
therefore fails safely instead of sending an out-of-range table to hardware.
"""


def main():
    import argparse
    import hashlib
    import inspect
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--skip-arbitrary-pool", action="store_true")
    args = parser.parse_args()

    import torch

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.prefill_contract import PrefillContract

    torch.set_num_threads(8)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"context": 256, "batch": 3, "layers": "all" if args.all_layers else [0, 3], "cases": []}
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        gen = build_generator(
            None, mesh, max_batch_size=3, max_seq_len=256, layer_indices=None if args.all_layers else [0, 3]
        )
        local_table = gen.page_table.clone()
        requests = [33, 35, 37]
        prompts = torch.zeros(3, max(requests), dtype=torch.int64)
        for slot, length in enumerate(requests):
            prompts[slot, :length] = torch.arange(length) + 100 + 100 * slot
        pools = {"local_control": (gen.cache, local_table)}
        global24 = gen.model.allocate_cache(batch=3, context=256, global_pages=True)
        pools["global24"] = (
            global24,
            torch.tensor([list(range(16, 24)), list(range(8)), list(range(8, 16))], dtype=torch.int32),
        )
        if not args.skip_arbitrary_pool:
            if "num_blocks" not in inspect.signature(gen.model.allocate_cache).parameters:
                raise RuntimeError("Arbitrary pool test requires model.allocate_cache(..., num_blocks=11)")
            global11 = gen.model.allocate_cache(batch=3, context=256, global_pages=True, num_blocks=11)
            sparse = torch.full((3, 8), -1, dtype=torch.int32)
            sparse[:, :2] = torch.tensor([[9, 3], [1, 7], [5, 0]], dtype=torch.int32)
            pools["global11"] = (global11, sparse)
        report["page_tables"] = {name: table.tolist() for name, (_, table) in pools.items()}
        helper = PrefillContract.prepare(
            gen,
            bindings=[
                {"cache": cache, "slots": range(3), "context": 256}
                for name, (cache, _) in pools.items()
                if name != "local_control"
            ],
            sampling_state=gen.sampling_state,
        )

        def record(item):
            report["cases"].append(item)
            path.write_text(json.dumps(report, indent=2, default=str) + "\n")
            print(json.dumps(item, default=str), flush=True)
            if not item["passed"]:
                raise AssertionError(item["case"])

        def buffers(cache):
            result, seen = [], set()
            for layer_index, states in enumerate(cache):
                for state in states:
                    for name in ("conv", "recurrent", "key", "value"):
                        tensor = getattr(state, name)
                        if tensor is not None and id(tensor) not in seen:
                            result.append((f"{layer_index}.{name}", tensor))
                            seen.add(id(tensor))
            return result

        def identity(cache):
            return {
                name: (id(tensor), tuple(int(part.buffer_address()) for part in ttnn.get_device_tensors(tensor)))
                for name, tensor in buffers(cache)
            }

        def shards(tensor):
            return [ttnn.to_torch(part).float() for part in ttnn.get_device_tensors(tensor)]

        def digest(tensor):
            return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()

        def whole_hash(cache):
            return {
                f"{name}.{rank}": digest(value)
                for name, tensor in buffers(cache)
                for rank, value in enumerate(shards(tensor))
            }

        def slot_hash(cache, table, slot, *, valid_only=False, length=None):
            result = {}
            for index, (layer, states) in enumerate(zip(gen.model.layers, cache, strict=True)):
                state = states[slot]
                if layer.kind == "linear_attention":
                    for name in ("conv", "recurrent"):
                        for rank, value in enumerate(shards(layer.read_linear_state(state, name))):
                            result[f"{index}.{name}.{rank}"] = digest(value)
                else:
                    mapping = table[slot].to(torch.int64)
                    mapping = mapping[mapping >= 0]
                    if not getattr(state, "global_pages", False):
                        mapping = mapping + state.page_offset
                    for name in ("key", "value"):
                        for rank, value in enumerate(shards(getattr(state, name))):
                            selected = value[mapping, 0].reshape(-1, 256)
                            if valid_only:
                                selected = selected[:length]
                            result[f"{index}.{name}.{rank}"] = digest(selected)
            return result

        originals = {name: identity(cache) for name, (cache, _) in pools.items()}
        snapshots = {
            name: [(tensor, ttnn.zeros_like(tensor)) for _, tensor in buffers(cache)]
            for name, (cache, _) in pools.items()
        }
        states = {}
        for name, (cache, table) in pools.items():
            # Warm only in-range addresses. Sparse -1 attention-map handling is
            # exercised by the public call after preparation, not by mock setup.
            safe = table.clone()
            for slot in range(3):
                safe[slot, safe[slot] < 0] = safe[slot, 0]
            states[name] = gen.prepare_decode(kv_cache=cache, page_table=safe, slots=(0, 1, 2))
        prefill = {}
        for name, (cache, table) in pools.items():
            prefill[name] = gen.prefill_forward(
                prompts, page_table=table, kv_cache=cache, prompt_lens=requests, sampling_params=None
            )
            for destination, saved in snapshots[name]:
                ttnn.copy(destination, saved)
            # Read/access every prepared slot before capturing.
            for slot in range(3):
                slot_hash(cache, table, slot)
        for name, (cache, table) in pools.items():
            if name == "local_control":
                continue
            equal_state = [
                slot_hash(cache, table, slot, valid_only=True, length=requests[slot])
                == slot_hash(gen.cache, local_table, slot, valid_only=True, length=requests[slot])
                for slot in range(3)
            ]
            record(
                {
                    "case": "external_prefill_equals_local",
                    "pool": name,
                    "logits_bit_equal": torch.equal(prefill[name], prefill["local_control"]),
                    "valid_state_bit_equal": equal_state,
                    "passed": torch.equal(prefill[name], prefill["local_control"]) and all(equal_state),
                }
            )
        for state in states.values():
            gen.capture(state)
        trace_ids = {
            name: {"model": dict(state.model_traces), "sampling": dict(state.sample_traces)}
            for name, state in states.items()
        }
        initial_captures = gen.counters["captures"]
        prepared_programs = mesh.num_program_cache_entries()
        mesh.set_program_cache_misses_allowed(False)

        def restore(name):
            for destination, saved in snapshots[name]:
                ttnn.copy(saved, destination)

        tokens = torch.tensor([[500], [501], [502]], dtype=torch.int64)
        positions = [33, -1, 37]

        def decode(name, *, table=None, positions_override=None):
            cache, default = pools[name]
            return gen.decode_forward(
                tokens,
                positions if positions_override is None else positions_override,
                page_table=default if table is None else table,
                kv_cache=cache,
                sampling_params=None,
                reload_inputs=True,
            )

        restore("local_control")
        control = decode("local_control")
        for name, (cache, table) in pools.items():
            if name == "local_control":
                continue
            restore(name)
            peer_before = slot_hash(cache, table, 1)
            got = decode(name)
            record(
                {
                    "case": "external_decode_equals_local_and_preserves_inactive",
                    "pool": name,
                    "active_logits_bit_equal": torch.equal(got[[0, 2]], control[[0, 2]]),
                    "inactive_state_unchanged": slot_hash(cache, table, 1) == peer_before,
                    "passed": torch.equal(got[[0, 2]], control[[0, 2]]) and slot_hash(cache, table, 1) == peer_before,
                }
            )
            restore(name)
            before = gen.counters["page_table_refreshes"]
            unchanged = decode(name)
            unchanged_copies = gen.counters["page_table_refreshes"] - before
            changed = table.clone()
            if name == "global11":
                # Map a previously unallocated future page to an unused safe
                # physical block; no valid-prefix migration is needed.
                changed[0, -1] = 10
            else:
                changed[0, -2:] = changed[0, [-1, -2]]
            restore(name)
            before = gen.counters["page_table_refreshes"]
            changed_output = decode(name, table=changed)
            changed_copies = gen.counters["page_table_refreshes"] - before
            restore(name)
            before = gen.counters["page_table_refreshes"]
            repeated = decode(name, table=changed)
            repeated_copies = gen.counters["page_table_refreshes"] - before
            record(
                {
                    "case": "authoritative_reload_changed_and_unchanged_page_table",
                    "pool": name,
                    "unchanged_copies": unchanged_copies,
                    "changed_copies": changed_copies,
                    "repeated_copies": repeated_copies,
                    "outputs_bit_equal": torch.equal(unchanged[[0, 2]], changed_output[[0, 2]])
                    and torch.equal(changed_output[[0, 2]], repeated[[0, 2]]),
                    # Input reload remains authoritative for token/position;
                    # equal tables reuse their persistent device rows. Only
                    # slot0 changes in the modified future-page control.
                    "passed": unchanged_copies == 0
                    and changed_copies == 1
                    and repeated_copies == 0
                    and torch.equal(unchanged[[0, 2]], changed_output[[0, 2]])
                    and torch.equal(changed_output[[0, 2]], repeated[[0, 2]]),
                }
            )
            restore(name)
            peers_before = {slot: slot_hash(cache, table, slot) for slot in (1, 2)}
            gen.prefill_forward(
                prompts[:1, : requests[0]],
                page_table=table[:1],
                kv_cache=cache,
                prompt_lens=[requests[0]],
                empty_slots=[0],
                sampling_params=None,
            )
            record(
                {
                    "case": "external_single_slot_prefill_preserves_peers",
                    "pool": name,
                    "passed": peers_before == {slot: slot_hash(cache, table, slot) for slot in (1, 2)},
                }
            )
            # A future SDPA-safe alias must never confer ownership of a peer's
            # page when explicitly clearing the populated request prefix.
            clear_table = table[0].clone()
            clear_table[2:] = table[1, 0]
            duplicate = clear_table.clone()
            duplicate[1] = duplicate[0]
            before_clear = whole_hash(cache)
            duplicate_rejected = False
            try:
                helper.reset_slots(
                    cache=cache,
                    slots=[0],
                    clear_kv=True,
                    page_tables={0: duplicate},
                    reset_sampling_state=False,
                )
            except ValueError:
                duplicate_rejected = True
            record(
                {
                    "case": "reject_duplicate_causal_pages_before_clear",
                    "pool": name,
                    "passed": duplicate_rejected and whole_hash(cache) == before_clear,
                }
            )
            helper.reset_slots(
                cache=cache, slots=[0], clear_kv=True, page_tables={0: clear_table}, reset_sampling_state=False
            )
            causal_zero = True
            for layer, layer_states in zip(gen.model.layers, cache, strict=True):
                if layer.kind == "full_attention":
                    for field in ("key", "value"):
                        for value in shards(getattr(layer_states[0], field)):
                            causal_zero &= bool((value[table[0, :2].to(torch.int64)] == 0).all())
            record(
                {
                    "case": "global_clear_ignores_future_peer_aliases",
                    "pool": name,
                    "causal_pages_zero": causal_zero,
                    "passed": causal_zero and peers_before == {slot: slot_hash(cache, table, slot) for slot in (1, 2)},
                }
            )
            after_clear = whole_hash(cache)
            helper.reset_slots(cache=cache, slots=[0], clear_kv=True, reset_sampling_state=False)
            record(
                {
                    "case": "unknown_prefix_clear_has_no_page_ownership",
                    "pool": name,
                    "passed": whole_hash(cache) == after_clear,
                }
            )

        class UnexpectedDispatch(RuntimeError):
            pass

        def rejected_atomically(name, operation, label, table):
            cache, _ = pools[name]
            before, counters = whole_hash(cache), dict(gen.counters)
            runtime_events = len(gen.prefill_runtime.events)
            owner, attribute = (gen, "_execute") if operation == "decode" else (gen.prefill_runtime, "_execute")
            original = getattr(owner, attribute)

            def stop_dispatch(*_args, **_kwargs):
                raise UnexpectedDispatch("Malformed page table reached device model dispatch")

            setattr(owner, attribute, stop_dispatch)
            error = None
            try:
                if operation == "decode":
                    decode(name, table=table)
                else:
                    gen.prefill_forward(
                        prompts, page_table=table, kv_cache=cache, prompt_lens=requests, sampling_params=None
                    )
            except (ValueError, TypeError) as caught:
                error = type(caught).__name__
            except UnexpectedDispatch as caught:
                error = str(caught)
            finally:
                setattr(owner, attribute, original)
            cache_same = before == whole_hash(cache)
            counters_same = counters == gen.counters
            record(
                {
                    "case": "reject_invalid_pages_before_mutation",
                    "pool": name,
                    "operation": operation,
                    "invalid": label,
                    "error": error,
                    "cache_unchanged": cache_same,
                    "counters_unchanged": counters_same,
                    "passed": error in ("ValueError", "TypeError")
                    and cache_same
                    and counters_same
                    and runtime_events == len(gen.prefill_runtime.events),
                }
            )

        for name, (cache, table) in pools.items():
            if name == "local_control":
                continue
            full = next(
                states[0]
                for layer, states in zip(gen.model.layers, cache, strict=True)
                if layer.kind == "full_attention"
            )
            invalid_tables = {"wrong_width": table[:, :-1].clone()}
            for label, value in (
                ("out_of_pool", full.key.shape[0]),
                ("unallocated_causal_page", -1),
                ("invalid_negative", -2),
            ):
                malformed = table.clone()
                malformed[0, 0] = value
                invalid_tables[label] = malformed
            for label, malformed in invalid_tables.items():
                rejected_atomically(name, "prefill", label, malformed)
                rejected_atomically(name, "decode", label, malformed)
        for name, (cache, _) in pools.items():
            record(
                {
                    "case": "caller_backing_identity_preserved",
                    "pool": name,
                    "passed": identity(cache) == originals[name],
                }
            )
        # End with a valid replay; tracker validation must follow every request-
        # time path, including rejected admissions and hash readback helpers.
        restore("local_control")
        decode("local_control")
        report["prepared_programs"] = prepared_programs
        report["final_programs"] = mesh.num_program_cache_entries()
        report["trace_ids"] = {
            name: {
                "model": {str(k): str(v) for k, v in value["model"].items()},
                "sampling": {str(k): str(v) for k, v in value["sampling"].items()},
            }
            for name, value in trace_ids.items()
        }
        record(
            {
                "case": "trace_and_program_reuse",
                "passed": gen.counters["captures"] == initial_captures
                and mesh.num_program_cache_entries() == prepared_programs
                and all(
                    state.model_traces == trace_ids[name]["model"]
                    and state.sample_traces == trace_ids[name]["sampling"]
                    for name, state in states.items()
                ),
            }
        )
        report["passed"] = True
    except BaseException as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        mesh.set_program_cache_misses_allowed(True)
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
