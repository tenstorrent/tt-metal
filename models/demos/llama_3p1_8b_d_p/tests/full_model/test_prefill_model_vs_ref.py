# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Staged real-checkpoint gates for the prefill wrapper; no decode or migration claims."""

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama_3p1_8b_d_p.tests.full_model.reference import chat_tokens, metrics, reference_prefill
from models.demos.llama_3p1_8b_d_p.tt.input import upload_token_chunk
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import allocate_kv_cache
from models.demos.llama_3p1_8b_d_p.tt.model import PrefillModel

CHECKPOINT = os.environ.get("LLAMA31_8B_CHECKPOINT", "/mnt/models/meta-llama/Llama-3.1-8B-Instruct")
# Full-model final logits retain these composition bounds. Accumulated raw hidden/KV drift is
# recorded in the 2K boundary case; strict same-native-input coverage is required in its companion.
FULL_LIMITS = (0.99, 0.15)


def _positions(start, sp):
    return torch.tensor([p for p in range(start, start + 1024) if (p // 256) % 4 == sp])


def _cache_positions(sp):
    return torch.tensor([p for p in range(2048) if (p // 256) % 4 == sp])


def _addresses(tensor):
    return tuple(int(shard.buffer_address()) for shard in ttnn.get_device_tensors(tensor))


def _snapshot(cache):
    return [[ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(t)] for t in (cache.k, cache.v)]


def _assert_unchanged(cache, before):
    for tensor, snapshots in zip((cache.k, cache.v), before):
        for shard, snapshot in zip(ttnn.get_device_tensors(tensor), snapshots):
            assert torch.equal(ttnn.to_torch(shard), snapshot)


def _seed_cache(mesh_device, model, dtype):
    cache = allocate_kv_cache(mesh_device, model.mesh_config, cache_dtype=dtype)
    for name, sign in (("k", 1), ("v", -1)):
        previous = getattr(cache, name)
        sentinel = (torch.arange(64).reshape(64, 1, 1, 1) % 7 + 1).expand(64, 1, 512, 128) * (sign / 8)
        setattr(
            cache,
            name,
            ttnn.from_torch(
                sentinel.contiguous(),
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=previous.memory_config(),
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
        )
        previous.deallocate(True)
    return cache


def _check_metric(expected, actual, limits, label, records, *, enforce=True):
    pcc, nl2 = metrics(expected, actual)
    within_limits = pcc >= limits[0] and nl2 <= limits[1]
    records.append(
        {"label": label, "pcc": pcc, "nl2": nl2, "limits": limits, "within_limits": within_limits, "enforced": enforce}
    )
    logger.info(f"{label}: PCC={pcc:.9f}, NL2={nl2:.9f}")
    if enforce:
        assert within_limits, (label, pcc, nl2, limits)


def _hidden_limits(num_layers, dtype):
    return ((0.999, 0.025) if dtype == ttnn.bfloat16 else (0.999, 0.05)) if num_layers == 1 else FULL_LIMITS


def _kv_limits(layer_idx, dtype):
    # Layer0 has no preceding device error accumulation, so it retains the strict composed gate.
    if layer_idx == 0:
        return (0.9999, 0.01) if dtype == ttnn.bfloat16 else (0.999, 0.02)
    return FULL_LIMITS


def _check_hidden(hidden, expected, *, start, end, limits, label, records, enforce=True):
    assert tuple(hidden.shape) == (1, 1, 256, 4096)
    assert hidden.dtype == ttnn.bfloat16 and hidden.layout == ttnn.TILE_LAYOUT
    assert hidden.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    for chip, shard in enumerate(ttnn.get_device_tensors(hidden)):
        positions = _positions(start, chip // 8)
        valid = positions < end
        actual = ttnn.to_torch(shard)[0, 0]
        assert torch.isfinite(actual).all()
        if valid.any():
            _check_metric(
                expected[positions[valid]], actual[valid], limits, f"{label} chip={chip}", records, enforce=enforce
            )


def _check_cache(cache, before, reference, *, slot, start, end, records, enforce_accumulated=True):
    layers = reference["layers"]
    for name, tensor, snapshots in zip(("k", "v"), (cache.k, cache.v), before):
        for chip, (shard, snapshot) in enumerate(zip(ttnn.get_device_tensors(tensor), snapshots)):
            actual = ttnn.to_torch(shard)
            assert torch.isfinite(actual).all()
            sp, head = divmod(chip, 8)
            positions = _cache_positions(sp)
            written = (positions >= start) & (positions < end)
            padding = (positions >= end) & (positions < (end + 31) // 32 * 32)
            untouched = ~(written | padding)
            changed_planes = set(range(slot * 32, slot * 32 + len(layers)))
            for plane in range(64):
                if plane not in changed_planes:
                    assert torch.equal(actual[plane], snapshot[plane]), (name, chip, plane, "other plane")
                    continue
                layer_idx = plane - slot * 32
                assert torch.equal(actual[plane, 0, untouched], snapshot[plane, 0, untouched]), (
                    name,
                    chip,
                    plane,
                    "outside chunk",
                )
                assert torch.count_nonzero(actual[plane, 0, padding]) == 0, (name, chip, plane, "padding")
                if written.any():
                    _check_metric(
                        layers[layer_idx][name][head, positions[written]],
                        actual[plane, 0, written],
                        _kv_limits(layer_idx, cache.k.dtype),
                        f"cache {name} slot={slot} layer={layer_idx} chip={chip} head={head}",
                        records,
                        enforce=enforce_accumulated or layer_idx == 0,
                    )


def _check_logits(logits, reference, *, start, end, num_layers, dtype, records):
    assert tuple(logits.shape) == (1, 1, 256, 16032)
    shards = [ttnn.to_torch(shard)[0, 0].float() for shard in ttnn.get_device_tensors(logits)]
    assert all(torch.isfinite(shard).all() for shard in shards)
    positions = reference["logit_positions"]
    selected = positions[(positions >= start) & (positions < end)]
    assert selected.numel(), "every tested chunk needs reference logit positions"
    expected_rows, actual_rows = [], []
    lookup = {int(position): row for row, position in enumerate(positions)}
    for absolute in selected.tolist():
        sp = (absolute // 256) % 4
        row = _positions(start, sp).tolist().index(absolute)
        # TP columns are exact adjacent 16032-wide vocabulary intervals. No logits from padded
        # query rows or a different SP owner may enter token comparisons.
        actual_rows.append(torch.cat([shards[sp * 8 + tp][row] for tp in range(8)]))
        expected_rows.append(reference["logits"][lookup[absolute]])
    expected, actual = torch.stack(expected_rows), torch.stack(actual_rows)
    assert actual.shape[1] == 128256
    limits = _hidden_limits(num_layers, dtype)
    _check_metric(expected, actual, limits, f"logits range=[{start},{end})", records, enforce=True)
    wanted = expected.argmax(dim=-1)
    top1 = (actual.argmax(dim=-1) == wanted).float().mean().item()
    top5 = (actual.topk(5, dim=-1).indices == wanted[:, None]).any(dim=-1).float().mean().item()
    records.append(
        {"label": "teacher-forced token agreement", "positions": selected.tolist(), "top1": top1, "top5": top5}
    )
    logger.info(f"teacher-forced {len(selected)} positions: top1={top1:.6f}, top5={top5:.6f}")
    assert top1 >= 0.90 and top5 >= 0.99
    return actual


def _resource_identity(model):
    return tuple(
        (id(tensor), _addresses(tensor))
        for tensor in (
            model.attention.gathered_k,
            model.attention.gathered_v,
            model.attention.query_position_table,
            model.attention.key_positions,
            *model.rope_tables,
            model.transformation_mat,
        )
    )


def _run(model, cache, ids, reference, *, slot, start, end, records, logits=True, enforce_accumulated=True):
    token_input = upload_token_chunk(model.mesh_device, ids[start:end], actual_start=start, actual_end=end)
    tokens_before = [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(token_input)]
    before = _snapshot(cache)
    observed = []
    identity = _resource_identity(model)

    def observe(layer_idx, hidden):
        observed.append(layer_idx)
        _check_hidden(
            hidden,
            reference["layers"][layer_idx]["hidden"],
            start=start,
            end=end,
            limits=_hidden_limits(layer_idx + 1, cache.k.dtype),
            label=f"layer output slot={slot} layer={layer_idx}",
            records=records,
            enforce=enforce_accumulated or layer_idx == 0,
        )

    output = None
    try:
        ttnn.synchronize_device(model.mesh_device)
        started = time.perf_counter()
        output = model.prefill_chunk(
            token_input,
            cache,
            slot_idx=slot,
            actual_start=start,
            actual_end=end,
            skip_lm_head=not logits,
            layer_observer=observe,
        )
        ttnn.synchronize_device(model.mesh_device)
        # Observer readbacks are included. This is correctness-run wall time, not kernel latency
        # or a production throughput claim. A separate observer-free warm measurement is below.
        records.append(
            {"label": "correctness eager wall seconds (with readbacks)", "seconds": time.perf_counter() - started}
        )
        assert observed == list(range(model.num_layers)), "every real layer must run once in order"
        assert _resource_identity(model) == identity
        for shard, saved in zip(ttnn.get_device_tensors(token_input), tokens_before):
            assert torch.equal(ttnn.to_torch(shard), saved)
        _check_cache(
            cache,
            before,
            reference,
            slot=slot,
            start=start,
            end=end,
            records=records,
            enforce_accumulated=enforce_accumulated,
        )
        if logits:
            return _check_logits(
                output,
                reference,
                start=start,
                end=end,
                num_layers=model.num_layers,
                dtype=cache.k.dtype,
                records=records,
            )
        _check_hidden(
            output,
            reference["layers"][-1]["hidden"],
            start=start,
            end=end,
            limits=_hidden_limits(model.num_layers, cache.k.dtype),
            label="KV-only output",
            records=records,
        )
        return None
    finally:
        if output is not None:
            output.deallocate(True)
        token_input.deallocate(True)


def _make_reference(slot, length, num_layers):
    ids, prompt = chat_tokens(CHECKPOINT, slot=slot, length=length)
    length = len(ids)
    # At 2048, use 256 evenly distributed positions plus exact boundaries and all short-tail rows.
    # The union guarantees meaningful whole-prompt token agreement and tiny continuation coverage.
    positions = set(range(length)) if length <= 65 else set(range(0, length, 8))
    positions.update(p for p in [31, 32, 255, 256, 511, 512, 1023, 1024, 1535, 1536, length - 1] if p < length)
    positions.update(range(1024, min(1033, length)))
    reference = reference_prefill(CHECKPOINT, ids, num_layers=num_layers, selected_logit_positions=sorted(positions))
    return ids, reference, prompt


def _save_report(name, records, prompts):
    directory = os.environ.get("LLAMA_PREFILL_EVIDENCE_DIR")
    if directory:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{name}.json").write_text(
            json.dumps({"records": records, "prompts": prompts}, indent=2, allow_nan=False)
        )


def _free_cache(cache):
    cache.k.deallocate(True)
    cache.v.deallocate(True)


# The reduced wrapper adds the real embedding and terminal path around the accepted layer. A
# two-chunk 1033-token chat prefix checks propagation of true length without changing the layer gate.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_prefill_one_layer_wrapper(mesh_device, cache_dtype):
    records, prompts = [], []
    ids, reference, prompt = _make_reference(0, 1033, 1)
    prompts.append(prompt)
    model = PrefillModel(mesh_device, CHECKPOINT, num_layers=1, cache_dtype=cache_dtype)
    cache = _seed_cache(mesh_device, model, cache_dtype)
    try:
        _run(model, cache, ids, reference, slot=0, start=0, end=1024, records=records)
        _run(model, cache, ids, reference, slot=0, start=1024, end=1033, records=records)
    finally:
        _save_report(f"one-layer-{cache_dtype}", records, prompts)
        _free_cache(cache)
        model.close()


# All 32 layers run by default, with two distinct slots and A/B/A reuse. Every layer's hidden state
# and every K/V head are scored, so a final residual correlation cannot conceal a missing layer.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_full_prefill_short_chat_and_slot_reuse(mesh_device, cache_dtype):
    records = []
    references = [_make_reference(slot, None, 32) for slot in range(2)]
    model = PrefillModel(mesh_device, CHECKPOINT, cache_dtype=cache_dtype)
    assert model.num_layers == len(model.layers) == 32
    assert [layer.layer_idx for layer in model.layers] == list(range(32))
    assert all(layer.attention is model.attention and layer.rope_tables == model.rope_tables for layer in model.layers)
    cache = _seed_cache(mesh_device, model, cache_dtype)
    try:
        first = None
        for slot in (0, 1, 0):
            ids, reference, _ = references[slot]
            actual = _run(model, cache, ids, reference, slot=slot, start=0, end=len(ids), records=records)
            if first is None:
                first = actual
            elif slot == 0:
                assert torch.equal(first, actual), "A/B/A logits must be deterministic"
        # Exercise the KV-only path with identical valid input; it must use all layers as well.
        ids, reference, _ = references[0]
        _run(model, cache, ids, reference, slot=0, start=0, end=len(ids), records=records, logits=False)
    finally:
        _save_report(f"full-short-{cache_dtype}", records, [item[2] for item in references])
        _free_cache(cache)
        model.close()


# Full prefixes precede every continuation. Interleaved slots cover 1024/2048 boundaries, a 1033
# tail, and tile-overlap restarts at 1024 and 1536. Every unaffected decoded cache value must stay unchanged.
# Layer0 and final logits remain hard; later raw hidden/KV drift is recorded with original limits.
# Full 2K acceptance also requires test_prefill_all_layers_from_native_inputs.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_full_prefill_context_boundaries(mesh_device, cache_dtype):
    records = []
    references = [_make_reference(slot, 2048, 32) for slot in range(2)]
    model = PrefillModel(mesh_device, CHECKPOINT, cache_dtype=cache_dtype)
    cache = _seed_cache(mesh_device, model, cache_dtype)
    try:
        for start, end in [(0, 1024), (1024, 1033), (1024, 1537), (1536, 2048)]:
            for slot in (0, 1):
                ids, reference, _ = references[slot]
                _run(
                    model,
                    cache,
                    ids,
                    reference,
                    slot=slot,
                    start=start,
                    end=end,
                    records=records,
                    enforce_accumulated=False,
                )
    finally:
        _save_report(f"full-boundaries-{cache_dtype}", records, [item[2] for item in references])
        _free_cache(cache)
        model.close()


# Argument errors must leave both slots unchanged; a valid recovery follows. Retained inputs
# expose stale-address reuse, then same-signature warm calls prove bounded live DRAM/program state.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_prefill_wrapper_validation_replay_and_resources(mesh_device, expect_error):
    model = PrefillModel(mesh_device, CHECKPOINT, num_layers=1, cache_dtype=ttnn.bfloat8_b)
    cache = _seed_cache(mesh_device, model, ttnn.bfloat8_b)
    ids, reference, prompt = _make_reference(0, 33, 1)
    tokens_a = upload_token_chunk(mesh_device, ids, actual_start=0, actual_end=33)
    tokens_b = upload_token_chunk(mesh_device, ids.flip(0), actual_start=0, actual_end=33)
    reference_b = reference_prefill(CHECKPOINT, ids.flip(0), num_layers=1)
    before_inputs = [
        [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(tokens)] for tokens in (tokens_a, tokens_b)
    ]
    difference = torch.linalg.vector_norm(reference["layers"][0]["hidden"] - reference_b["layers"][0]["hidden"])
    assert difference / torch.linalg.vector_norm(reference["layers"][0]["hidden"]) > 0.01
    outputs, records = [], []
    before = _snapshot(cache)
    try:
        assert all(a != b for a, b in zip(_addresses(tokens_a), _addresses(tokens_b)))
        invalid = [
            ({"slot_idx": 2}, "slot_idx 2 out of range"),
            ({"actual_start": 1}, "aligned to 32 tokens"),
            ({"actual_end": 2049}, "actual range must stay within max_seq_len=2048"),
            ({"actual_end": 0}, "actual range must stay within max_seq_len=2048"),
            ({"actual_end": 1025}, "at most 1024 valid tokens"),
            ({"layer_observer": 1}, "must be callable"),
            ({"skip_lm_head": 1}, "must be a bool"),
        ]
        for change, message in invalid:
            kwargs = dict(slot_idx=0, actual_start=0, actual_end=33)
            kwargs.update(change)
            with expect_error((ValueError, TypeError), message):
                model.prefill_chunk(tokens_a, cache, **kwargs)
            _assert_unchanged(cache, before)
        with expect_error(ValueError, "cache metadata must"):
            model.prefill_chunk(tokens_a, replace(cache, num_users=1), slot_idx=0, actual_start=0, actual_end=33)
        _assert_unchanged(cache, before)
        _run(model, cache, ids, reference, slot=0, start=0, end=33, records=records)
        tokens_b_3d = ttnn.reshape(tokens_b, (1, 1, 256))
        tokens_a_3d = ttnn.reshape(tokens_a, (1, 1, 256))
        for tokens, expected in ((tokens_a, reference), (tokens_b_3d, reference_b), (tokens_a_3d, reference)):
            cache_before = _snapshot(cache)
            outputs.append(model.prefill_chunk(tokens, cache, slot_idx=0, actual_start=0, actual_end=33))
            _check_hidden(
                outputs[-1],
                expected["layers"][0]["hidden"],
                start=0,
                end=33,
                limits=_hidden_limits(1, cache.k.dtype),
                label="retained input replay",
                records=records,
            )
            _check_cache(cache, cache_before, expected, slot=0, start=0, end=33, records=records)
        for tokens, snapshots in zip((tokens_a, tokens_b), before_inputs):
            for shard, snapshot in zip(ttnn.get_device_tensors(tokens), snapshots):
                assert torch.equal(ttnn.to_torch(shard), snapshot)
        for a, b, again in zip(*(ttnn.get_device_tensors(output) for output in outputs)):
            assert a.buffer_address() != b.buffer_address() != again.buffer_address()
            assert torch.equal(ttnn.to_torch(a), ttnn.to_torch(again))
        assert not torch.equal(
            ttnn.to_torch(ttnn.get_device_tensors(outputs[0])[0])[0, 0, :33],
            ttnn.to_torch(ttnn.get_device_tensors(outputs[1])[0])[0, 0, :33],
        )
        for output in outputs:
            output.deallocate(True)
        outputs.clear()
        # Prime the same metadata/signature used for memory and program-cache measurement.
        warm = model.prefill_chunk(tokens_a, cache, slot_idx=0, actual_start=0, actual_end=33)
        warm.deallocate(True)
        ttnn.synchronize_device(mesh_device)
        programs = mesh_device.num_program_cache_entries()
        free = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM).total_bytes_free_per_bank
        for _ in range(3):
            started = time.perf_counter()
            result = model.prefill_chunk(tokens_a, cache, slot_idx=0, actual_start=0, actual_end=33)
            ttnn.synchronize_device(mesh_device)
            records.append(
                {"label": "warm observer-free eager KV-only wall seconds", "seconds": time.perf_counter() - started}
            )
            result.deallocate(True)
            assert mesh_device.num_program_cache_entries() == programs
            assert ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM).total_bytes_free_per_bank == free
        cache_after_replay = _snapshot(cache)
        model.close()
        _assert_unchanged(cache, cache_after_replay)
        for tokens, snapshots in zip((tokens_a, tokens_b), before_inputs):
            for shard, snapshot in zip(ttnn.get_device_tensors(tokens), snapshots):
                assert torch.equal(ttnn.to_torch(shard), snapshot)
        with expect_error(RuntimeError, "PrefillModel is closed"):
            model.prefill_chunk(tokens_a, cache, slot_idx=0, actual_start=0, actual_end=33)
    finally:
        _save_report("wrapper-validation", records, [prompt])
        for output in outputs:
            output.deallocate(True)
        tokens_a.deallocate(True)
        tokens_b.deallocate(True)
        _free_cache(cache)
        model.close()
