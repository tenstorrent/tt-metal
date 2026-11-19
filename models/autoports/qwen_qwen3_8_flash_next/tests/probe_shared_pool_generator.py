# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real reduced decoder shared-pool write/read/lifecycle test. NOT run by author.

Run only serialized by the device owner, with the accepted mesh descriptor:

timeout 1800 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
 PYTHONPATH=/home/ttuser/qwen38-flash/tt-metal python_env/bin/python \
 -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_shared_pool_generator \
 --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/shared_pool_generator.json

Real layers0/1/3, original TP4 policies, context128, batch2, global16-block
QSA pool. Compares mixed prompts and migrated continuation to solo controls;
tests real paged writes, private recurrent/indexer state, release isolation,
reused slots, dormant alias rejection, unchanged-table replay and pool addresses.
An earlier private-cache B2 generator uses the same loaded model as a matched
decode-geometry control. An uninterrupted B2 run also checks migration against
the same decode-width history, without invoking migration in the reference.
Its traces and storage are released before pool setup.
Readbacks here are correctness diagnostics, not token-out performance evidence.
"""

import argparse
import gc
import json
import os
import time
import traceback
import weakref
from pathlib import Path


def tensor_comparison(expected, actual, *, logits=False):
    """Compact CPU diagnostics; exact equality remains the acceptance criterion."""
    import torch

    result = {"expected_shape": list(expected.shape), "actual_shape": list(actual.shape)}
    if expected.shape != actual.shape:
        return {**result, "exact": False, "reason": "shape_mismatch"}
    left, right = expected.detach().reshape(-1).double(), actual.detach().reshape(-1).double()
    result["exact"] = bool(torch.equal(expected, actual))
    result["different_elements"] = int((left != right).sum())
    finite = torch.isfinite(left) & torch.isfinite(right)
    result["nonfinite_expected"] = int((~torch.isfinite(left)).sum())
    result["nonfinite_actual"] = int((~torch.isfinite(right)).sum())
    if finite.any():
        a, b = left[finite], right[finite]
        diff = (a - b).abs()
        result["max_abs"] = float(diff.max())
        result["mean_abs"] = float(diff.mean())
        ac, bc = a - a.mean(), b - b.mean()
        denominator = ac.norm() * bc.norm()
        result["pcc"] = float((ac * bc).sum() / denominator) if denominator > 0 else None
        flat_indices = finite.nonzero().flatten()
        worst = diff.topk(min(5, diff.numel())).indices
        result["largest_differences"] = (
            [
                {"flat_index": int(flat_indices[index]), "expected": float(a[index]), "actual": float(b[index])}
                for index in worst
            ]
            if not result["exact"]
            else []
        )
    if logits and left.numel() and finite.all():
        k = min(5, left.numel())
        expected_top = left.topk(k).indices.tolist()
        actual_top = right.topk(k).indices.tolist()
        result.update(
            expected_top1=expected_top[0],
            actual_top1=actual_top[0],
            top1_equal=expected_top[0] == actual_top[0],
            expected_top5=expected_top,
            actual_top5=actual_top,
            expected_top1_in_actual_top5=expected_top[0] in actual_top,
            actual_top1_in_expected_top5=actual_top[0] in expected_top,
        )
    return result


def decode_geometry_comparison(reference, actual):
    """Frozen probe_full_model_batch thresholds for n=1 versus n>1 decode.

    This gate applies to logits across the accepted reduction geometries only.
    Matching-geometry cache, position, feedback and output invariants stay exact.
    """
    import torch

    reference, actual = reference.float(), actual.float()
    assert reference.ndim == 2 and reference.shape == actual.shape
    row_pcc = []
    for left, right in zip(reference, actual, strict=True):
        a, b = left - left.mean(), right - right.mean()
        denominator = float(a.norm() * b.norm())
        row_pcc.append(float(a @ b) / denominator if denominator else float(torch.equal(left, right)))
    a, b = reference.flatten(), actual.flatten()
    a, b = a - a.mean(), b - b.mean()
    denominator = float(a.norm() * b.norm())
    whole_pcc = float(a @ b) / denominator if denominator else float(torch.equal(reference, actual))
    # topk does not promise argmax tie ordering. Compare both greedy IDs
    # with the same lowest-index argmax contract; retain the frozen thresholds.
    ref_top1 = reference.argmax(-1)
    candidates = actual.topk(100, dim=-1).indices
    result = {
        "finite": bool(torch.isfinite(reference).all() and torch.isfinite(actual).all()),
        "exact": bool(torch.equal(reference, actual)),
        "whole_pcc": whole_pcc,
        "per_row_pcc": row_pcc,
        "minimum_row_pcc": min(row_pcc),
        "top1_agreement": float((actual.argmax(-1) == ref_top1).float().mean()),
        "top5_agreement": float((candidates[:, :5] == ref_top1[:, None]).any(-1).float().mean()),
        "top100_agreement": float((candidates == ref_top1[:, None]).any(-1).float().mean()),
        "max_absolute_difference": float((reference - actual).abs().max()),
    }
    result["passed"] = (
        result["finite"]
        and result["whole_pcc"] >= 0.995
        and result["minimum_row_pcc"] >= 0.99
        and result["top1_agreement"] == 1
        and result["top5_agreement"] >= 0.98
        and result["top100_agreement"] == 1
    )
    result["threshold_source"] = "tests/probe_full_model_batch.py"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", default=os.environ.get("QWEN_CHECKPOINT"))
    parser.add_argument("--packed-cache", default=os.environ.get("QWEN_PLE_CACHE"))
    args = parser.parse_args()
    if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1":
        parser.error("Allocation tracking must be enabled before imports")
    if os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0":
        parser.error("Program-cache allocation tracking must remain enabled")
    import torch

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_full_model_serving import packed, read_ranks
    from models.autoports.qwen_qwen3_8_flash_next.tt.generator import QwenGenerator, build_generator

    torch.set_num_threads(4)
    report = dict(
        passed=False, scope="real_layers_0_1_3_shared_pool", checks=[], comparisons=[], failures=[], tracking=True
    )
    mesh = gen = None
    started = time.monotonic()
    try:
        assert os.environ.get("TT_MESH_GRAPH_DESC_PATH"), "Restore the accepted mesh descriptor after lane-env"
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=256 * 1024 * 1024, l1_small_size=32768)
        options = dict(layer_indices=(0, 1, 3), max_seq_len=128, max_batch_size=2)
        if args.checkpoint:
            options["checkpoint"] = args.checkpoint
        if args.packed_cache:
            options["packed_cache"] = args.packed_cache
        gen = build_generator("models/autoports/qwen_qwen3_8_flash_next", mesh, **options)
        tables = torch.tensor([[15, 14, 13, 12], [11, 10, 9, 8]], dtype=torch.int32)
        prompts = ([785, 13458, 13], [785, 13, 13458, 13, 785] * 6 + [785, 13, 13458])

        def prefill(rows, slots, table):
            return gen.prefill_forward(
                packed(rows),
                page_table=table,
                page_tables_per_layer=[table],
                kv_cache=pool,
                prompt_lens=list(map(len, rows)),
                slots=slots,
            )

        def decode(tokens, positions, slots, table=None, **kwargs):
            return gen.decode_forward(
                torch.tensor(tokens),
                positions,
                slots=slots,
                page_table=table,
                page_tables_per_layer=None if table is None else [table],
                kv_cache=pool,
                **kwargs,
            )

        def snapshot(cache):
            result = {"host_position": [torch.tensor([cache.position], dtype=torch.int64)]}
            for index, state in enumerate(cache.layers):
                for name in ("conv", "recurrent", "ple_conv", "token_history"):
                    value = getattr(state, name, None)
                    if value is not None:
                        result[f"{index}.{name}"] = read_ranks(value)
                if state.attention is not None:
                    for name in ("position", "page_table", "raw_history", "compressed_keys"):
                        result[f"{index}.{name}"] = read_ranks(getattr(state.attention, name))
                    pages = result[f"{index}.page_table"]
                    # Compare only logically filled KV, in causal token order.
                    # Other requests and stale unused pool blocks are intentionally
                    # outside this request's state; reading them is not meaningful.
                    for name in ("k_cache", "v_cache"):
                        logical = []
                        for storage, table in zip(read_ranks(getattr(state.attention, name)), pages, strict=True):
                            ids = table.flatten()[: (cache.position + 31) // 32].long()
                            values = storage[ids].permute(1, 0, 2, 3).reshape(1, -1, 256)
                            logical.append(values[:, : cache.position].clone())
                        result[f"{index}.{name}.logical"] = logical
            return result

        def compare(label, expected, actual, *, logits=False):
            result = {"label": label, **tensor_comparison(expected, actual, logits=logits)}
            report["comparisons"].append(result)
            if not result["exact"]:
                print("POOL_DIFFERENCE " + json.dumps(result), flush=True)
            return result["exact"]

        def equal_state(expected, actual, label):
            if expected.keys() != actual.keys():
                report["comparisons"].append({"label": label, "exact": False, "reason": "state_fields_changed"})
                return False
            exact = True
            for key in expected:
                for rank, (a, b) in enumerate(zip(expected[key], actual[key], strict=True)):
                    exact = compare(f"{label}/{key}/rank{rank}", a, b) and exact
            return exact

        def logical_state(state):
            # Private caches use IDs0..3; the pool deliberately uses IDs8..15.
            # All recurrence, positions, indexer state and logically addressed
            # KV remain comparable. Only physical page IDs are different by design.
            return {key: value for key, value in state.items() if not key.endswith(".page_table")}

        def check_pool_tables(expected_tables, label):
            # Different physical IDs are excluded only from private-versus-pool
            # comparisons. The scheduler's actual pooled mappings stay exact.
            for slot, expected in enumerate(expected_tables):
                for layer_index, state in enumerate(gen.caches[slot].layers):
                    if state.attention is not None:
                        for rank, actual in enumerate(read_ranks(state.attention.page_table)):
                            assert compare(
                                f"{label}/slot{slot}/layer{layer_index}/rank{rank}", expected.reshape(1, -1), actual
                            ), "Pooled physical page IDs differ from authoritative scheduler mapping"

        # Load the model once, but give the private control its own generator
        # buffers and caches. No pooled-generator allocation is allowed to
        # appear after this control captures its first trace.
        model, tokenizer = gen.model, gen.tokenizer
        private_tables = torch.arange(4, dtype=torch.int32).reshape(1, 4).repeat(2, 1)
        gen.warmup_model_prefill(kv_cache=gen.caches)
        gen.warmup_model_decode(kv_cache=gen.caches)

        def private_prefix(*, remap, include44=False):
            """Same B2 geometry throughout; reset reuses already prepared storage."""
            gen.reset()
            pref = gen.prefill_forward(
                packed(prompts),
                page_table=private_tables,
                kv_cache=gen.caches,
                prompt_lens=list(map(len, prompts)),
                slots=(0, 1),
            ).clone()
            pref_states = [snapshot(cache) for cache in gen.caches]
            outputs, states = [], []
            order = (1, 0) if remap else (0, 1)
            for step, token in enumerate((42, 43, 44) if include44 else (42, 43)):
                row_order = order if step else (0, 1)
                output = gen.decode_forward(
                    torch.tensor([token, token]),
                    [len(prompts[row]) + step for row in row_order],
                    slots=(0, 1),
                    page_table=private_tables,
                    kv_cache=gen.caches,
                    **({"slot_remap": (1, 0)} if remap and step == 1 else {}),
                ).clone()
                outputs.append(output)
                states.append([snapshot(cache) for cache in gen.caches])
            return pref, pref_states, outputs, states

        # The reference never calls slot_remap: it consumes42/43/44 in the
        # original row order, with identical B2 reduction geometry at each step.
        # The migrated control consumes the same per-request token history and
        # must equal the flipped reference, including every private state field.
        uninterrupted = private_prefix(remap=False, include44=True)
        migration_probe = private_prefix(remap=True, include44=True)
        migration_exact = True
        migration_by_step = []
        for step, token in enumerate((42, 43, 44)):
            order = (0, 1) if step == 0 else (1, 0)
            exact = True
            for slot, original in enumerate(order):
                exact = (
                    compare(
                        f"private_migration_vs_uninterrupted/token{token}/slot{slot}",
                        uninterrupted[2][step][original],
                        migration_probe[2][step][slot],
                        logits=True,
                    )
                    and exact
                )
                exact = (
                    equal_state(
                        uninterrupted[3][step][original],
                        migration_probe[3][step][slot],
                        f"private_migration_vs_uninterrupted_state/token{token}/slot{slot}",
                    )
                    and exact
                )
            migration_by_step.append({"token": token, "exact": exact, "reference_row_order": list(order)})
            migration_exact = exact and migration_exact
        report["private_migration_vs_uninterrupted_b2"] = {
            "exact": migration_exact,
            "steps": migration_by_step,
            "reference_uses_migration": False,
            "decode_width_sequence": [2, 2, 2],
            "tokens_per_request": [42, 43, 44],
        }
        if not migration_exact:
            report["failures"].append("Private migrated B2 output/state differs from uninterrupted B2")

        # Rebuild the original private lifecycle baseline after the diagnostic;
        # no new generator/cache storage and no trace retirement between runs.
        # This preserves the later B2,B2,B1,B1 lifecycle comparison with the pool.
        private_prefill, private_prefill_states, private_outputs, private_states = private_prefix(remap=True)
        private_decode, private_migrated = private_outputs
        private_decode_states, private_migrated_states = private_states
        repeat_exact = compare("private_baseline_rebuild/prefill", migration_probe[0], private_prefill, logits=True)
        for step, token in enumerate((42, 43)):
            repeat_exact = (
                compare(
                    f"private_baseline_rebuild/token{token}",
                    migration_probe[2][step],
                    private_outputs[step],
                    logits=True,
                )
                and repeat_exact
            )
            for slot in (0, 1):
                repeat_exact = (
                    equal_state(
                        migration_probe[3][step][slot],
                        private_states[step][slot],
                        f"private_baseline_rebuild_state/token{token}/slot{slot}",
                    )
                    and repeat_exact
                )
        report["private_baseline_rebuild_exact"] = repeat_exact
        if not repeat_exact:
            report["failures"].append("Private B2 baseline changed after reset/replay")
        gen.release_request(0)
        private_released_states = [snapshot(cache) for cache in gen.caches]
        private_continued = gen.decode_forward(
            torch.tensor([44]),
            [len(prompts[0]) + 2],
            slots=(1,),
            page_table=private_tables[:1],
            kv_cache=[gen.caches[1]],
        ).clone()
        private_continued_state = snapshot(gen.caches[1])
        private_refreshed = gen.prefill_forward(
            packed([prompts[1]]),
            page_table=private_tables[:1],
            kv_cache=[gen.caches[0]],
            prompt_lens=[len(prompts[1])],
            slots=(0,),
        ).clone()
        private_refreshed_states = [snapshot(cache) for cache in gen.caches]
        private_unchanged = gen.decode_forward(
            torch.tensor([0]),
            [97],
            slots=(1,),
            page_table=None,
            kv_cache=[gen.caches[1]],
            reload_inputs=False,
        ).clone()
        private_unchanged_states = [snapshot(cache) for cache in gen.caches]
        report["private_b2_baseline"] = {
            "same_loaded_model": True,
            "cache_geometry": "two independent four-block BF16 K/V allocations per attention layer",
            "slots": [0, 1],
            "prompt_lengths": list(map(len, prompts)),
            "decode_tokens": [42, 42],
            "page_tables": private_tables.tolist(),
            "decode_width_sequence": [2, 2, 1, 1],
            "stages": [
                "prefill",
                "decode",
                "migration",
                "release",
                "survivor_continuation",
                "reuse",
                "unchanged_table",
            ],
            "counters": dict(gen.counters),
        }
        private_refs = [weakref.ref(gen), *(weakref.ref(cache) for cache in gen.caches)]
        if gen.slot_mover.scratch is not None:
            private_refs.append(weakref.ref(gen.slot_mover.scratch))
        gen.close()
        assert not gen.traces and not gen.prefill_traces and not gen.sampling_traces
        ttnn.synchronize_device(mesh)
        gen = None
        gc.collect()
        assert all(reference() is None for reference in private_refs), "Private generator/cache ownership leaked"
        report["private_b2_baseline"]["released_before_pooled_setup"] = True
        print("PRIVATE_B2_BASELINE_COMPLETE_AND_RELEASED", flush=True)

        gen = QwenGenerator(model, tokenizer, max_seq_len=128, max_batch_size=2, defer_kv_cache=True)
        assert gen.model is model, "Pooled control replaced the loaded model"
        try:
            gen.prepare()
        except RuntimeError as error:
            assert "Deferred" in str(error)
        else:
            raise AssertionError("Unbound deferred cache prepared")
        pool = gen.allocate_kv_cache_per_layer([((16, 1, 32, 256), torch.bfloat16, 0)])
        gen.warmup_model_prefill(kv_cache=pool)
        gen.warmup_model_decode(kv_cache=pool)
        addresses = [tensor.buffer_address() for pair in pool.pairs.values() for tensor in pair]

        solo_prefill, solo_prefill_states, solo_decode, solo_states = [], [], [], []
        for row, prompt in enumerate(prompts):
            gen.reset()
            table = tables[row : row + 1]
            solo_prefill.append(prefill([prompt], (0,), table)[0].clone())
            solo_prefill_states.append(snapshot(gen.caches[0]))
            outputs, states = [], []
            for step, token in enumerate((42, 43, 44)):
                outputs.append(decode([token], [len(prompt) + step], (0,), table)[0].clone())
                states.append(snapshot(gen.caches[0]))
            solo_decode.append(outputs)
            solo_states.append(states)
        # Control first-capture versus already captured execution and possible
        # setup-history dependence before diagnosing batch-shape math.
        repeats_exact = True
        for row, prompt in enumerate(prompts):
            gen.reset()
            repeated_prefill = prefill([prompt], (0,), tables[row : row + 1])
            repeats_exact = (
                compare(f"repeat_solo_prefill/row{row}", solo_prefill[row], repeated_prefill[0], logits=True)
                and repeats_exact
            )
            repeated_decode = decode([42], [len(prompt)], (0,), tables[row : row + 1])
            repeats_exact = (
                compare(f"repeat_solo_decode/row{row}", solo_decode[row][0], repeated_decode[0], logits=True)
                and repeats_exact
            )
            repeats_exact = (
                equal_state(solo_states[row][0], snapshot(gen.caches[0]), f"repeat_solo_state/row{row}")
                and repeats_exact
            )
        report["solo_repeat_exact"] = repeats_exact
        gen.reset()
        batch = prefill(prompts, (0, 1), tables)
        prefill_exact = private_prefill_exact = True
        for row in range(2):
            prefill_exact = (
                compare(f"mixed_prefill/row{row}", solo_prefill[row], batch[row], logits=True) and prefill_exact
            )
            prefill_exact = (
                equal_state(solo_prefill_states[row], snapshot(gen.caches[row]), f"mixed_prefill_state/row{row}")
                and prefill_exact
            )
            private_prefill_exact = (
                compare(f"pooled_vs_private_b2_prefill/row{row}", private_prefill[row], batch[row], logits=True)
                and private_prefill_exact
            )
            private_prefill_exact = (
                equal_state(
                    logical_state(private_prefill_states[row]),
                    logical_state(snapshot(gen.caches[row])),
                    f"pooled_vs_private_b2_prefill_state/row{row}",
                )
                and private_prefill_exact
            )
        report["pooled_vs_private_b2_prefill_exact"] = private_prefill_exact
        assert private_prefill_exact, "Pooled prefill differs from private B2; inspect comparisons"
        assert prefill_exact, "Mixed prefill or resident state differs; inspect comparisons"
        report["checks"].append("mixed-prefill-real-global-KV-writes-match-solo")
        first = decode([42, 42], list(map(len, prompts)), (0, 1), tables)
        decode_exact = private_decode_exact = True
        for row in range(2):
            decode_exact = (
                compare(f"mixed_decode/row{row}", solo_decode[row][0], first[row], logits=True) and decode_exact
            )
            decode_exact = (
                equal_state(solo_states[row][0], snapshot(gen.caches[row]), f"mixed_decode_state/row{row}")
                and decode_exact
            )
            private_decode_exact = (
                compare(f"pooled_vs_private_b2_decode/row{row}", private_decode[row], first[row], logits=True)
                and private_decode_exact
            )
            private_decode_exact = (
                equal_state(
                    logical_state(private_decode_states[row]),
                    logical_state(snapshot(gen.caches[row])),
                    f"pooled_vs_private_b2_decode_state/row{row}",
                )
                and private_decode_exact
            )
        report["pooled_vs_private_b2_decode_exact"] = private_decode_exact
        check_pool_tables(tables, "first_decode_page_table")
        report["first_decode_feedback"] = [rank.flatten().tolist() for rank in read_ranks(gen.decode_tokens)]
        report["counters_at_first_decode"] = dict(gen.counters)
        assert private_decode_exact, "Pooled decode differs from private B2; inspect comparisons"
        report["checks"].append("pooled-b2-prefill-decode-and-logical-state-match-private-b2-exactly")
        report["solo_vs_batched_decode_exact_diagnostic"] = decode_exact
        report["solo_vs_batched_decode"] = decode_geometry_comparison(
            torch.stack([row[0] for row in solo_decode]), first
        )
        assert repeats_exact, "Same-geometry solo replay differs; inspect comparisons"
        if report["solo_vs_batched_decode"]["passed"]:
            report["checks"].append("solo-repeat-exact-and-batched-decode-meets-frozen-accuracy-gate")
        else:
            report["failures"].append("Solo versus batch decode failed frozen batch accuracy gate")
        migrated = decode(
            [43, 43], [len(prompts[1]) + 1, len(prompts[0]) + 1], (0, 1), tables.flip(0), slot_remap=(1, 0)
        )
        check_pool_tables(tables.flip(0), "migrated_page_table")
        for slot, original in enumerate((1, 0)):
            compare(f"migrated_decode_vs_solo/slot{slot}", solo_decode[original][1], migrated[slot], logits=True)
            equal_state(solo_states[original][1], snapshot(gen.caches[slot]), f"migrated_state_vs_solo/slot{slot}")
            exact = compare(
                f"migrated_decode_vs_private_b2/slot{slot}", private_migrated[slot], migrated[slot], logits=True
            )
            exact = (
                equal_state(
                    logical_state(private_migrated_states[slot]),
                    logical_state(snapshot(gen.caches[slot])),
                    f"migrated_state_vs_private_b2/slot{slot}",
                )
                and exact
            )
            assert exact, f"Migrated slot{slot} differs from same-geometry private control; inspect comparisons"
        pooled_uninterrupted_exact = True
        for slot, original in enumerate((1, 0)):
            pooled_uninterrupted_exact = (
                compare(
                    f"pooled_migration_vs_uninterrupted_b2/slot{slot}",
                    uninterrupted[2][1][original],
                    migrated[slot],
                    logits=True,
                )
                and pooled_uninterrupted_exact
            )
            pooled_uninterrupted_exact = (
                equal_state(
                    logical_state(uninterrupted[3][1][original]),
                    logical_state(snapshot(gen.caches[slot])),
                    f"pooled_migration_vs_uninterrupted_b2_state/slot{slot}",
                )
                and pooled_uninterrupted_exact
            )
        report["pooled_migration_vs_uninterrupted_b2_exact"] = pooled_uninterrupted_exact
        if not pooled_uninterrupted_exact:
            report["failures"].append("Pooled migrated B2 output/state differs from uninterrupted private B2")
        report["solo_vs_migrated_decode"] = decode_geometry_comparison(
            torch.stack([solo_decode[row][1] for row in (1, 0)]), migrated
        )
        if not report["solo_vs_migrated_decode"]["passed"]:
            report["failures"].append("Migrated decode failed frozen batch accuracy gate")
        if migration_exact and pooled_uninterrupted_exact:
            report["checks"].append("slot-remap-retains-global-KV-and-private-state")
        pool_before = [[read_ranks(t) for t in pair] for pair in pool.pairs.values()]
        survivor = snapshot(gen.caches[1])
        gen.release_request(0)
        check_pool_tables(tables.flip(0), "released_page_table")
        assert equal_state(survivor, snapshot(gen.caches[1]), "release_survivor")
        for slot in (0, 1):
            assert equal_state(
                logical_state(private_released_states[slot]),
                logical_state(snapshot(gen.caches[slot])),
                f"release_vs_private/slot{slot}",
            ), f"Release slot{slot} differs from private control"
        for before, pair in zip(pool_before, pool.pairs.values(), strict=True):
            for old, tensor in zip(before, pair, strict=True):
                assert all(torch.equal(a, b) for a, b in zip(old, read_ranks(tensor), strict=True))
        report["checks"].append("released-slot-does-not-clear-shared-pool-or-survivor")
        before = dict(gen.counters)
        try:
            prefill([prompts[1]], (0,), tables[:1])  # Conflicts with dormant survivor's ID15.
        except ValueError as error:
            assert "Overlapping" in str(error)
        else:
            raise AssertionError("Dormant request's pool pages were aliased")
        assert gen.counters == before
        report["checks"].append("dormant-page-alias-rejects-before-device-mutation")
        continued = decode([44], [len(prompts[0]) + 2], (1,), tables[:1])
        check_pool_tables(tables.flip(0), "continued_page_table")
        compare("continued_survivor_vs_solo", solo_decode[0][2], continued[0], logits=True)
        equal_state(solo_states[0][2], snapshot(gen.caches[1]), "continued_survivor_state_vs_solo")
        assert compare("continued_survivor_vs_private", private_continued, continued, logits=True)
        assert equal_state(
            logical_state(private_continued_state),
            logical_state(snapshot(gen.caches[1])),
            "continued_survivor_vs_private",
        )
        report["solo_vs_survivor_decode"] = decode_geometry_comparison(solo_decode[0][2].unsqueeze(0), continued)
        if not report["solo_vs_survivor_decode"]["passed"]:
            report["failures"].append("Survivor decode failed frozen batch accuracy gate")
        refreshed = prefill([prompts[1]], (0,), tables[1:])
        check_pool_tables(tables.flip(0), "reused_page_table")
        assert torch.equal(refreshed[0], solo_prefill[1])
        assert compare("reused_prefill_vs_private", private_refreshed, refreshed, logits=True)
        for slot in (0, 1):
            assert equal_state(
                logical_state(private_refreshed_states[slot]),
                logical_state(snapshot(gen.caches[slot])),
                f"reuse_vs_private/slot{slot}",
            )
        report["checks"].append("survivor-continues-and-released-slot-reuses")
        updates = gen.counters["page_table_refreshes"]
        unchanged = decode([0], [97], (1,), reload_inputs=False)
        check_pool_tables(tables.flip(0), "unchanged_page_table")
        assert compare("unchanged_table_decode_vs_private", private_unchanged, unchanged, logits=True)
        for slot in (0, 1):
            assert equal_state(
                logical_state(private_unchanged_states[slot]),
                logical_state(snapshot(gen.caches[slot])),
                f"unchanged_table_state_vs_private/slot{slot}",
            )
        assert gen.counters["page_table_refreshes"] == updates
        assert addresses == [tensor.buffer_address() for pair in pool.pairs.values() for tensor in pair]
        report["checks"].append("unchanged-table-replay-no-refresh-stable-pool-addresses")
        report["counters"] = dict(gen.counters)
        assert not report["failures"], "; ".join(report["failures"])
        report["passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        if gen is not None:
            gen.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
