# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced real-layer cache constructor/release/shutdown acceptance probe.

Run serialized under the parent's device lease and external timeout. This writes
an exact acceptance artifact; it makes no full-model accuracy/performance claim.
"""


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--all-layers", action="store_true")
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "layers": list(range(32)) if args.all_layers else [0, 3],
        "context": 256,
        "batch": 3,
        "physical_blocks": 29,
        "cases": [],
    }
    model_root = Path(__file__).resolve().parents[1]
    report["source_sha256"] = {
        name: hashlib.sha256((model_root / name).read_bytes()).hexdigest()
        for name in ("tt/generator.py", "tt/cache_contract.py", "tt/model.py", "tt/multichip_decoder.py")
    }

    import torch

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.cache_contract import PERSISTENT_CATEGORIES, DramBudget
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import OrnithGenerator
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel

    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = injected = None

    def record(case, passed, **details):
        item = {"case": case, "passed": bool(passed), **details}
        report["cases"].append(item)
        path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps(item, default=str), flush=True)
        if not passed:
            raise AssertionError(case)

    def tensors(cache):
        seen = set()
        for index, states in enumerate(cache):
            for state in states:
                for name in ("conv", "recurrent", "key", "value"):
                    tensor = getattr(state, name)
                    if tensor is not None and id(tensor) not in seen:
                        seen.add(id(tensor))
                        yield f"{index}.{name}", tensor

    def host_shards(tensor):
        return [ttnn.to_torch(part).float() for part in ttnn.get_device_tensors(tensor)]

    def digest(value):
        return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()

    def whole_hash(cache):
        return {
            f"{name}.{rank}": digest(value)
            for name, tensor in tensors(cache)
            for rank, value in enumerate(host_shards(tensor))
        }

    def identity(cache):
        return {
            name: [id(tensor), [int(part.buffer_address()) for part in ttnn.get_device_tensors(tensor)]]
            for name, tensor in tensors(cache)
        }

    try:
        model = OrnithModel(mesh, layer_indices=None if args.all_layers else [0, 3])
        gen = OrnithGenerator(model, max_batch_size=3, max_seq_len=256)
        owned = gen.cache
        external = model.allocate_cache(batch=3, context=256, global_pages=True, num_blocks=29)
        table = torch.tensor([list(range(20, 28)), list(range(8)), list(range(10, 18))], dtype=torch.int32)
        before_hash, before_identity = whole_hash(external), identity(external)
        allocator = model.allocate_cache
        allocation_calls = []

        def forbidden_allocation(**kwargs):
            allocation_calls.append(kwargs)
            raise AssertionError("External-cache constructor allocated another model cache")

        model.allocate_cache = forbidden_allocation
        try:
            injected = OrnithGenerator(model, max_batch_size=3, max_seq_len=256, kv_cache=external, page_table=table)
        finally:
            model.allocate_cache = allocator
        record(
            "external_constructor_has_no_model_cache_allocation",
            not allocation_calls
            and injected.cache is external
            and injected.cache_is_external
            and identity(external) == before_identity
            and whole_hash(external) == before_hash,
            allocation_calls=allocation_calls,
        )
        injected.close()
        injected.close()
        injected = None
        gen.bind_serving_cache(external, page_table=table)
        record(
            "binding_retains_standalone_storage",
            gen.cache is owned
            and gen.cache_ownership.serving.cache is external
            and identity(external) == before_identity,
        )

        lengths = [33, 35, 37]
        prompts = torch.zeros((3, max(lengths)), dtype=torch.int64)
        for slot, length in enumerate(lengths):
            prompts[slot, :length] = torch.arange(length) + 100 + 50 * slot
        gen.prefill_forward(prompts, page_table=gen.page_table, kv_cache=owned, prompt_lens=lengths)
        gen.prefill_forward(prompts, page_table=table, kv_cache=external, prompt_lens=lengths)
        gen.prepare_decode(kv_cache=external, page_table=table, slots=(0, 1, 2))

        def slot_hash(slot):
            result = {}
            for index, (layer, states) in enumerate(zip(model.layers, external)):
                state = states[slot]
                if layer.kind == "linear_attention":
                    for name in ("conv", "recurrent"):
                        for rank, value in enumerate(host_shards(layer.read_linear_state(state, name))):
                            result[f"{index}.{name}.{rank}"] = digest(value)
                else:
                    for name in ("key", "value"):
                        for rank, value in enumerate(host_shards(getattr(state, name))):
                            result[f"{index}.{name}.{rank}"] = digest(value[table[slot].long()])
            return result

        # Invalid geometry is rejected during host admission, before reset/warmup.
        protected = {"owned": whole_hash(owned), "external": whole_hash(external)}
        invalid_cache = [list(states) for states in external]
        invalid_cache[0] = invalid_cache[0][:-1]
        invalids = [("unequal_slot_capacity", invalid_cache, table), ("wrong_table_width", external, table[:, :-1])]
        out_of_pool = table.clone()
        out_of_pool[0, 0] = 29
        invalids.append(("out_of_pool_page", external, out_of_pool))
        for label, cache, pages in invalids:
            error = None
            try:
                gen.bind_serving_cache(cache, page_table=pages)
            except (ValueError, TypeError) as caught:
                error = type(caught).__name__
            record(
                "invalid_binding_is_atomic",
                error is not None
                and gen.cache_ownership.serving.cache is external
                and protected == {"owned": whole_hash(owned), "external": whole_hash(external)},
                invalid=label,
                error=error,
            )

        peers = {slot: slot_hash(slot) for slot in (1, 2)}
        owned_before = whole_hash(owned)
        kv_before = {
            name: value for name, value in whole_hash(external).items() if ".key." in name or ".value." in name
        }
        gen.release_request(0)
        zeros = True
        for layer, states in zip(model.layers, external):
            if layer.kind == "linear_attention":
                for name in ("conv", "recurrent"):
                    zeros &= all(
                        bool((value == 0).all()) for value in host_shards(layer.read_linear_state(states[0], name))
                    )
        record(
            "default_release_targets_serving_cache_and_preserves_peers",
            zeros
            and whole_hash(owned) == owned_before
            and peers == {slot: slot_hash(slot) for slot in (1, 2)}
            and kv_before
            == {name: value for name, value in whole_hash(external).items() if ".key." in name or ".value." in name},
        )
        once = whole_hash(external)
        gen.release_request(0)
        record("release_is_idempotent", whole_hash(external) == once and whole_hash(owned) == owned_before)

        gen.prefill_forward(
            prompts[:1, : lengths[0]],
            page_table=table[:1],
            kv_cache=external,
            prompt_lens=[lengths[0]],
            empty_slots=[0],
        )
        handle = gen.decode_forward(
            torch.tensor([[500], [501], [502]]),
            [33, -1, 37],
            page_table=table,
            kv_cache=external,
            sampling_params=None,
            reload_inputs=True,
            read_from_device=False,
        )
        record("submission_pending_before_close", handle.completion is None and handle.submission_id in gen.submissions)
        # MeshDevice uses one lockstep allocator view; GetMemoryView queries
        # device->allocator()->get_statistics (memory_reporter.cpp:172), and
        # MeshDevice::allocator delegates to that shared allocator view. Our TP4
        # allocations are symmetric. Repeat this measured per-device value for
        # four ranks, explicitly recording that this is not four independent
        # IDevice memory measurements.
        ttnn.synchronize_device(mesh)
        view = model.memory_view()
        free_per_device = int(view["num_banks"]) * int(view["total_bytes_free_per_bank"])
        binding = gen.cache_ownership.serving
        budget = DramBudget(
            measured_free_bytes_per_device=(free_per_device,) * 4,
            measurement_artifact=str(path),
            included_categories=tuple(sorted(PERSISTENT_CATEGORIES)),
            pending_bytes_per_device={},
            reserve_bytes_per_device=256 * 1024 * 1024,
            existing_target_pool_bytes_per_device=(binding.kv_bytes_per_device,) * 4,
        )
        capacity = gen.get_max_tokens_all_users(
            budget=budget,
            validated_num_blocks=29,
            validation_artifact=str(path),
            max_model_len=256,
            max_num_seqs=3,
        )
        report["capacity"] = {
            "scope": "full32_configured_context256_batch3" if args.all_layers else "reduced_layers_0_3_diagnostic_only",
            "memory_view": view,
            "measurement_semantics": "One symmetric MeshDevice lockstep allocator view, repeated for four ranks.",
            "allocator_source": "tt_metal/detail/reports/memory_reporter.cpp:172; tt_metal/distributed/mesh_device.cpp:2140,2237",
            "included_categories": list(budget.included_categories),
            "pending_bytes_per_device": dict(budget.pending_bytes_per_device),
            "reserve_bytes_per_device": budget.reserve_bytes_per_device,
            "retained_target_kv_bytes_per_device": binding.kv_bytes_per_device,
            "arithmetic_num_blocks_upper_bound": budget.pool_limit().num_blocks_upper_bound,
            "allocation_validated_num_blocks": 29,
            "get_max_tokens_all_users": capacity,
            "maximum_context_claim": False,
        }
        if args.all_layers:
            from dataclasses import asdict

            report["actual_cache_specs"] = {str(i): asdict(spec) for i, spec in gen.get_kv_cache_spec().items()}
        else:
            excluded = False
            try:
                gen.get_kv_cache_spec()
            except ValueError:
                excluded = True
            record("reduced_model_cannot_advertise_full_cache_specs", excluded)
        record("measured_configured_capacity", capacity == 3 * 256, scope=report["capacity"]["scope"], tokens=capacity)
        ids_before_close = identity(external)
        gen.close()  # Drains pending transfers and releases traces, not caller cache.
        counters = dict(gen.counters)
        output = gen.process_decode_output_host(handle, is_tokens=False)
        gen.close()
        again = gen.process_decode_output_host(handle, is_tokens=False)
        record(
            "close_drains_read_is_observational_and_close_is_idempotent",
            gen.closed
            and tuple(output.shape) == (3, 1, model.vocab_size)
            and again is output
            and gen.counters == counters
            and identity(external) == ids_before_close
            and not gen.submissions,
            output_shape=list(output.shape),
            trace_releases=gen.counters["releases"],
        )
        record(
            "caller_owned_backing_survives_shutdown",
            identity(external) == before_identity and all(tensor.is_allocated() for _, tensor in tensors(external)),
        )
        report["passed"] = True
    except BaseException as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        if injected is not None:
            injected.close()
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
