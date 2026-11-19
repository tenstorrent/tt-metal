# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight B3 mixed-prefill/slot probe; parent runs under bounded timeout.

Checks bit-equal batched-versus-solo last logits and valid K/V/recurrent state,
row-order changes, persistent device-logit assembly and common greedy sampling,
inactive-peer isolation, invalid-batch atomic admission, and implicit-horizon
33 -> 68 -> 513 prefill against a canonical one-shot reference. No acceptance
result is claimed merely by this script existing; JSON retains every result.
"""


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--all-layers", action="store_true")
    args = parser.parse_args()
    if args.context < 544 or args.context % 32:
        parser.error("Context must be page aligned and at least 544")

    import torch

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.prefill_contract import PrefillContract

    torch.set_num_threads(8)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"cases": [], "layers": "all" if args.all_layers else [0, 3], "context": args.context}
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        gen = build_generator(
            None, mesh, max_batch_size=3, max_seq_len=args.context, layer_indices=None if args.all_layers else [0, 3]
        )
        helper = PrefillContract.prepare(gen)
        output_tokens = gen._integer(torch.zeros((1, 1, 1, 32), dtype=torch.int32))
        gen.sampler.precompile(helper.logits, tt_out_tok=output_tokens)
        gen._sample_logits(helper.logits, output_tokens)
        helper.sampling_state.reset_history(slots=[0, 1, 2], prompt_tokens=None, output_tokens=None)
        requests = {
            slot: list(range(200 + 300 * slot, 200 + 300 * slot + count)) for slot, count in enumerate((33, 65, 129))
        }

        def record(item):
            report["cases"].append(item)
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(item), flush=True)
            if not item["passed"]:
                raise AssertionError(item["case"])

        def digest(value):
            return hashlib.sha256(value.float().contiguous().numpy().tobytes()).hexdigest()

        def shards(value):
            return [ttnn.to_torch(part).float() for part in ttnn.get_device_tensors(value)]

        def snapshot(slot, length=None):
            result = {}
            for index, (layer, states) in enumerate(zip(gen.model.layers, gen.cache, strict=True)):
                state = states[slot]
                if layer.kind == "linear_attention":
                    for name in ("conv", "recurrent"):
                        for rank, value in enumerate(shards(layer.read_linear_state(state, name))):
                            result[f"{index}.{name}.{rank}"] = digest(value)
                else:
                    pages = state.page_count or state.key.shape[0]
                    mapping = gen.page_table[slot].to(torch.int64) + state.page_offset
                    for name in ("key", "value"):
                        for rank, value in enumerate(shards(getattr(state, name))):
                            if length is None:
                                value = value[state.page_offset : state.page_offset + pages]
                            else:
                                value = value[mapping, 0].reshape(-1, 256)[:length]
                            result[f"{index}.{name}.{rank}"] = digest(value)
            return result

        def sampler_peer(slot):
            result = {}
            for name in ("prompt_mask", "output_mask", "output_counts", "output_counts_gathered"):
                for rank, value in enumerate(shards(getattr(gen.sampler.tt_penalties, name))):
                    result[f"{name}.{rank}"] = digest(value.reshape(32, -1)[slot])
            for rank, value in enumerate(shards(gen.sampler.tt_sampling.seeds_tt_tensor)):
                result[f"seed.{rank}"] = digest(value.reshape(32, -1)[slot])
            return result

        def pack(order, content=requests):
            lengths = [len(content[slot]) for slot in order]
            tokens = torch.zeros(len(order), max(lengths), dtype=torch.int64)
            for row, slot in enumerate(order):
                tokens[row, : lengths[row]] = torch.tensor(content[slot])
            return tokens, lengths, gen.page_table[list(order)]

        # These reads exercise all preprepared slot accessors before the seal.
        for slot in range(3):
            snapshot(slot)
        baseline = mesh.num_program_cache_entries()
        report["prepared_programs"] = baseline
        report["warm_program_counts"] = helper.warm_program_counts
        mesh.set_program_cache_misses_allowed(False)
        solo, solo_cache = {}, {}
        for slot in range(3):
            tokens, ends, table = pack([slot])
            solo[slot] = helper.forward(
                tokens, prompt_lens=ends, empty_slots=[slot], page_table=table, kv_cache=gen.cache
            )
            solo_cache[slot] = snapshot(slot, ends[0])
        for order in ([2, 0, 1], [1, 2, 0]):
            tokens, ends, table = pack(order)
            got = helper.forward(tokens, prompt_lens=ends, empty_slots=order, page_table=table, kv_cache=gen.cache)
            equal_logits = [torch.equal(got[row : row + 1], solo[slot]) for row, slot in enumerate(order)]
            equal_state = [snapshot(slot, len(requests[slot])) == solo_cache[slot] for slot in order]
            record(
                {
                    "case": "mixed_batched_equals_solo",
                    "slot_order": order,
                    "logits_bit_equal": equal_logits,
                    "state_bit_equal": equal_state,
                    "passed": all(equal_logits + equal_state),
                }
            )

        resumed_solo, resumed_cache = {0: solo[0]}, {0: solo_cache[0]}
        starts_by_slot = {0: 0, 1: 32, 2: 68}
        for slot in (1, 2):
            tokens, ends, table = pack([slot])
            start = starts_by_slot[slot]
            helper.forward(
                tokens, prompt_lens=[start], request_lens=ends, empty_slots=[slot], page_table=table, kv_cache=gen.cache
            )
            resumed_solo[slot] = helper.forward(
                tokens,
                prompt_lens=ends,
                start_pos=[start],
                request_lens=ends,
                empty_slots=[slot],
                page_table=table,
                kv_cache=gen.cache,
            )
            resumed_cache[slot] = snapshot(slot, ends[0])
        tokens, ends, table = pack([1, 2])
        helper.forward(
            tokens, prompt_lens=[32, 68], request_lens=ends, empty_slots=[1, 2], page_table=table, kv_cache=gen.cache
        )
        order = [2, 0, 1]
        tokens, ends, table = pack(order)
        mixed_starts = [starts_by_slot[slot] for slot in order]
        got = helper.forward(
            tokens,
            prompt_lens=ends,
            start_pos=mixed_starts,
            request_lens=ends,
            empty_slots=order,
            page_table=table,
            kv_cache=gen.cache,
        )
        equal_logits = [torch.equal(got[row : row + 1], resumed_solo[slot]) for row, slot in enumerate(order)]
        equal_state = [snapshot(slot, len(requests[slot])) == resumed_cache[slot] for slot in order]
        record(
            {
                "case": "mixed_starts_batched_equals_solo",
                "slot_order": order,
                "start_pos": mixed_starts,
                "logits_bit_equal": equal_logits,
                "state_bit_equal": equal_state,
                "passed": all(equal_logits + equal_state),
            }
        )

        tokens, ends, table = pack([2, 0, 1])
        original_host_logits = gen.model.host_logits

        def forbidden_host_logits(*_args, **_kwargs):
            raise AssertionError("Device prefill crossed the host-logits boundary")

        gen.model.host_logits = forbidden_host_logits
        try:
            batch = helper.forward(
                tokens,
                prompt_lens=ends,
                empty_slots=[2, 0, 1],
                page_table=table,
                kv_cache=gen.cache,
                sample_on_device=True,
            )
            gen._sample_logits(batch.logits, output_tokens)
        finally:
            gen.model.host_logits = original_host_logits
        assembled = gen.model.host_logits(batch.logits).reshape(32, gen.model.vocab_size)
        sampled = ttnn.to_torch(ttnn.get_device_tensors(output_tokens)[0]).reshape(-1).to(torch.int64)
        equal_rows = [torch.equal(assembled[slot], solo[slot][0, 0]) for slot in range(3)]
        equal_tokens = [int(sampled[slot]) == int(solo[slot][0, 0].argmax()) for slot in range(3)]
        record(
            {
                "case": "device_fixed_slot_logits_and_common_greedy",
                "active_slots": list(batch.active_slots),
                "logits_bit_equal": equal_rows,
                "tokens_equal": equal_tokens,
                "inactive_logits_zero": bool((assembled[3:] == 0).all()),
                "passed": all(equal_rows + equal_tokens) and bool((assembled[3:] == 0).all()),
            }
        )

        peers_before = {slot: (snapshot(slot), sampler_peer(slot)) for slot in (0, 2)}
        mixed = torch.zeros(3, 65, dtype=torch.int64)
        mixed[0] = torch.tensor(requests[1])
        inactive_result = helper.forward(
            mixed,
            prompt_lens=[65, 0, 0],
            start_pos=[0, 0, 0],
            empty_slots=[1, 0, 2],
            page_table=gen.page_table[[1, 0, 2]],
            kv_cache=gen.cache,
        )
        peers_after = {slot: (snapshot(slot), sampler_peer(slot)) for slot in (0, 2)}
        record(
            {
                "case": "inactive_rows_preserve_peers",
                "peers_bit_equal": peers_before == peers_after,
                "inactive_logits_negative_infinity": bool(torch.isneginf(inactive_result[1:]).all()),
                "passed": peers_before == peers_after and bool(torch.isneginf(inactive_result[1:]).all()),
            }
        )
        helper.reset_slots(cache=gen.cache, slots=[1], clear_kv=True)
        record(
            {
                "case": "clear_one_slot_preserves_peers",
                "passed": peers_before == {slot: (snapshot(slot), sampler_peer(slot)) for slot in (0, 2)},
            }
        )

        before_invalid = {slot: snapshot(slot) for slot in range(3)}
        tokens, ends, table = pack([0, 1, 2])
        rejected = False
        try:
            helper.forward(
                tokens,
                prompt_lens=[ends[0], ends[1], tokens.shape[1] + 1],
                empty_slots=[0, 1, 2],
                page_table=table,
                kv_cache=gen.cache,
            )
        except ValueError:
            rejected = True
        record(
            {
                "case": "invalid_later_row_atomic_admission",
                "rejected": rejected,
                "passed": rejected and before_invalid == {slot: snapshot(slot) for slot in range(3)},
            }
        )
        rejected_unknown = False
        try:
            helper.forward(tokens, prompt_lens=ends, page_table=table, kv_cache=gen.cache, invented_option=True)
        except TypeError:
            rejected_unknown = True
        record({"case": "unknown_options_rejected", "passed": rejected_unknown})

        whole = torch.arange(200, 713, dtype=torch.int64)[None]
        canonical = helper.forward(
            whole, prompt_lens=[513], empty_slots=[1], page_table=gen.page_table[1:2], kv_cache=gen.cache
        )
        previous = 0
        for end in (33, 68, 513):
            resumed = helper.forward(
                whole,
                prompt_lens=[end],
                start_pos=[previous],
                empty_slots=[1],
                page_table=gen.page_table[1:2],
                kv_cache=gen.cache,
            )
            previous = end
        expected, actual = canonical.flatten(), resumed.flatten()
        pcc = 1.0 if torch.equal(expected, actual) else float(torch.corrcoef(torch.stack((expected, actual)))[0, 1])
        record(
            {
                "case": "implicit_horizon_33_68_513",
                "pcc": pcc,
                "max_abs": float((expected - actual).abs().max()),
                "bit_equal": torch.equal(expected, actual),
                "greedy_equal": int(expected.argmax()) == int(actual.argmax()),
                "rebuild_events": helper.events[-3:],
                "passed": pcc >= 0.995 and int(expected.argmax()) == int(actual.argmax()),
            }
        )
        report["final_programs"] = mesh.num_program_cache_entries()
        record({"case": "no_late_compilation", "passed": report["final_programs"] == baseline})
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
