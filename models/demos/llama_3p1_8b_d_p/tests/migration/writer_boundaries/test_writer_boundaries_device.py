# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Closed silicon gate for the fixed Llama 2K BFP8 writer-boundary matrix."""

from __future__ import annotations

import ast
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger
from ttnn.device import is_blackhole

import ttnn
from models.demos.llama_3p1_8b_d_p.tt.config import MeshConfig
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import allocate_kv_cache, write_kv_chunk
from models.demos.llama_3p1_8b_d_p.tt.runners.kv_chunk_table import build_kv_chunk_address_table

from . import page_io
from .kv_table_oracle import assert_independent_page, independent_tensor_location
from .writer_boundaries import (
    BF8_PAGE_BYTES,
    CONFIG_NAMES,
    HEAD_DIM,
    MAX_SEQ_LEN,
    NUM_HEADS,
    NUM_LAYERS,
    NUM_SLOTS,
    PAGE_TOKENS,
    WRITER_CASES,
    PageKey,
    case_for_key,
    classify_page_rows,
    decoder_function_source,
    device_major_positions,
    expected_page,
    input_row,
    iter_all_keys,
    run_with_tensor_cleanup,
    snapshot_specs,
    touched_keys,
)

MESH_SHAPE = (4, 8)
SP, TP = MESH_SHAPE
GLOBAL_CHUNK = 1024
LOCAL_CHUNK = GLOBAL_CHUNK // SP
EXPECTED_PAGES = 65_536
EXPECTED_PACKED_BYTES = EXPECTED_PAGES * BF8_PAGE_BYTES
EXPECTED_TOUCHED_PAGES = 208
EXPECTED_TOUCHED_BYTES = EXPECTED_TOUCHED_PAGES * BF8_PAGE_BYTES
EXPECTED_UNTOUCHED_PAGES = EXPECTED_PAGES - EXPECTED_TOUCHED_PAGES
EXPECTED_UNTOUCHED_BYTES = EXPECTED_UNTOUCHED_PAGES * BF8_PAGE_BYTES
EXPECTED_VALID_ROWS = sum(case.end - case.start for case in WRITER_CASES) * len(CONFIG_NAMES)
EXPECTED_PADDING_ROWS = EXPECTED_TOUCHED_PAGES * PAGE_TOKENS - EXPECTED_VALID_ROWS
DECODER_PATH = Path(__file__).resolve().parents[6] / "models/demos/common/prefill/runners/prefill_producer.py"
DECODER_SHA256 = "97853a3c584e18c7acf1f4ee05c228fc1a7f212251ac076ec5585b49f8a5245b"


def _tagged_values(phase, kind, slot, layer, positions):
    positions = list(positions)
    values = torch.full((NUM_HEADS, len(positions), HEAD_DIM), 32.0)
    values[:, :, 0] = 32.0 if kind == "k" else -32.0
    heads = torch.arange(NUM_HEADS, dtype=torch.int64)[:, None]
    position_tensor = torch.tensor(positions, dtype=torch.int64)[None, :]
    cursor = 1
    for coordinate, bits in (
        (heads, 3),
        (torch.full_like(heads, layer), 5),
        (torch.full_like(heads, slot), 1),
        (position_tensor, 11),
    ):
        for bit in range(bits):
            values[:, :, cursor] = torch.where(((coordinate >> bit) & 1).bool(), 64.0, 32.0)
            cursor += 1
    tail = position_tensor >= MAX_SEQ_LEN
    phase_value = torch.full_like(position_tensor, 16.0 if phase == "seed" else -16.0)
    parity = (heads + layer + slot + position_tensor + (1 if phase == "write" else 0)) % 2
    parity_value = torch.where(parity.bool(), -64.0, 64.0)
    high_position_bit = torch.where(((position_tensor >> 11) & 1).bool(), 64.0, 32.0)
    values[:, :, cursor] = torch.where(tail, high_position_bit, phase_value)
    values[:, :, cursor + 1] = torch.where(tail, phase_value, parity_value)
    values[:, :, cursor + 2] = torch.where(tail, parity_value, 32.0)
    values[:, :, cursor + 3] = torch.where(tail, -32.0, 32.0)

    probe_positions = {0, len(positions) - 1}
    for head in (0, NUM_HEADS - 1):
        for index in probe_positions:
            expected = input_row(phase, kind, head, layer, slot, positions[index])
            assert tuple(values[head, index].tolist()) == expected
    return values.to(torch.bfloat16).float()


def _to_chunk(mesh_device, values):
    return ttnn.from_torch(
        values.unsqueeze(0).to(torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(2, 1)),
    )


def _write(mesh_device, cache, *, phase, slot, layer, start, end, cleanup_errors):
    positions = device_major_positions(start)
    tensors = []

    def write():
        tt_k = _to_chunk(mesh_device, _tagged_values(phase, "k", slot, layer, positions))
        tensors.append(("write.k", tt_k))
        tt_v = _to_chunk(mesh_device, _tagged_values(phase, "v", slot, layer, positions))
        tensors.append(("write.v", tt_v))
        write_kv_chunk(
            cache,
            tt_k,
            tt_v,
            slot_idx=slot,
            layer_idx=layer,
            actual_start=start,
            actual_end=end,
        )

    run_with_tensor_cleanup(write, tensors, cleanup_errors)


def _load_decoder():
    source = decoder_function_source(DECODER_PATH, DECODER_SHA256)
    node = ast.parse(source).body[0]
    namespace = {"torch": torch, "np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(DECODER_PATH), "exec"), namespace)
    return namespace["_decode_bfp8_chunk"]


def _torch_page(rows):
    return torch.tensor(rows, dtype=torch.float32).reshape(1, 1, PAGE_TOKENS, HEAD_DIM)


def _validate_table_geometry(mesh_device, table):
    assert page_io.geometry(table) == MAX_SEQ_LEN
    assert table.num_configs() == len(CONFIG_NAMES)
    assert tuple(table.config_name(index) for index in range(table.num_configs())) == CONFIG_NAMES
    assert table.total_entries() == EXPECTED_PAGES
    banks = set()
    for config_id in range(table.num_configs()):
        for layer in range(NUM_LAYERS):
            for slot in range(NUM_SLOTS):
                for position in range(0, MAX_SEQ_LEN, PAGE_TOKENS):
                    location = table.lookup(layer, position, slot, config_id=config_id)
                    assert location.size_bytes == BF8_PAGE_BYTES
                    assert int(location.noc_addr) != 0
                    banks.add(int(location.noc_addr) >> 32)
    assert banks == set(range(mesh_device.dram_grid_size().x))
    return sorted(banks)


def _validate_seed_with_independent_views(cache, table, decode):
    comparisons = 0
    raw_bytes = 0
    for config_id, config_name in enumerate(CONFIG_NAMES):
        kind, head_text = config_name.split("_h")
        head = int(head_text)
        cache_tensor = cache.k if kind == "k" else cache.v
        shards = ttnn.get_device_tensors(cache_tensor)
        assert len(shards) == SP * TP
        for sp_row in range(SP):
            live = ttnn.to_torch(shards[sp_row * TP + head]).to(torch.bfloat16)
            for slot in range(NUM_SLOTS):
                for layer in range(NUM_LAYERS):
                    batch = slot * NUM_LAYERS + layer
                    for position in range(0, MAX_SEQ_LEN, PAGE_TOKENS):
                        owner, local_position = independent_tensor_location(position)
                        if owner != sp_row:
                            continue
                        key = PageKey(kind, head, layer, slot, position)
                        expected = _torch_page(expected_page(key, None)).to(torch.bfloat16)
                        tensor_page = live[
                            batch : batch + 1,
                            :1,
                            local_position : local_position + PAGE_TOKENS,
                            :HEAD_DIM,
                        ].reshape(1, 1, PAGE_TOKENS, HEAD_DIM)
                        assert_independent_page(
                            tensor_page,
                            expected,
                            f"seed live tensor {config_name}/{slot}/{layer}/{position}",
                        )
                        raw = page_io.read_page(table, (config_id, slot, layer, position))
                        table_page = decode(raw, HEAD_DIM).reshape(1, 1, PAGE_TOKENS, HEAD_DIM).to(torch.bfloat16)
                        assert_independent_page(
                            table_page,
                            expected,
                            f"seed table address {config_name}/{slot}/{layer}/{position}",
                        )
                        assert_independent_page(
                            table_page,
                            tensor_page,
                            f"seed independent views {config_name}/{slot}/{layer}/{position}",
                        )
                        comparisons += 1
                        raw_bytes += len(raw)
            del live
    assert comparisons == EXPECTED_PAGES
    assert raw_bytes == EXPECTED_PACKED_BYTES
    return comparisons, raw_bytes


def _snapshot_phase(table, directory, phase):
    receipts = []
    for spec in snapshot_specs(directory):
        if spec.phase == phase:
            receipts.append(page_io.snapshot(table, spec.path, spec.slot, spec.begin, spec.end))
    saved = page_io.SavedPages(receipts, MAX_SEQ_LEN)
    assert sum(receipt["pages"] for receipt in receipts) == EXPECTED_PAGES
    assert sum(receipt["bytes"] for receipt in receipts) == EXPECTED_PACKED_BYTES
    return receipts, saved


def _compare_snapshots(before, after, decode):
    expected_touched = touched_keys()
    assert len(expected_touched) == EXPECTED_TOUCHED_PAGES
    touched_pages = untouched_pages = valid_rows = padding_rows = 0
    for key in iter_all_keys():
        config_id = CONFIG_NAMES.index(f"{key.kind}_h{key.head}")
        packed_key = (config_id, key.slot, key.layer, key.position)
        before_raw = before.get(packed_key)
        after_raw = after.get(packed_key)
        case = case_for_key(key)
        if key not in expected_touched:
            assert case is None
            assert after_raw == before_raw, f"untouched page changed: {key}"
            untouched_pages += 1
            continue

        assert case is not None
        assert after_raw != before_raw, f"touched page retained seed bytes: {key}"
        decoded = decode(after_raw, HEAD_DIM)
        expected = torch.tensor(expected_page(key, case), dtype=torch.float32)
        rows = classify_page_rows(case, key.position)
        for position in rows.valid_positions:
            row = position - key.position
            assert torch.equal(decoded[row].float(), expected[row].float()), f"valid row mismatch: {key}/{position}"
            valid_rows += 1
        for position in rows.padding_positions:
            row = position - key.position
            assert torch.count_nonzero(decoded[row]).item() == 0, f"padding row nonzero: {key}/{position}"
            padding_rows += 1
        touched_pages += 1

    assert touched_pages == EXPECTED_TOUCHED_PAGES
    assert untouched_pages == EXPECTED_UNTOUCHED_PAGES
    assert valid_rows == EXPECTED_VALID_ROWS
    assert padding_rows == EXPECTED_PADDING_ROWS
    return {
        "touched_pages": touched_pages,
        "touched_bytes": touched_pages * BF8_PAGE_BYTES,
        "untouched_pages": untouched_pages,
        "untouched_bytes": untouched_pages * BF8_PAGE_BYTES,
        "valid_rows": valid_rows,
        "padding_rows": padding_rows,
    }


@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-4x8")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    ids=["ring"],
    indirect=True,
)
@pytest.mark.skipif(not is_blackhole(), reason="Llama writer boundary gate targets a Blackhole Galaxy")
@pytest.mark.timeout(0)
# Seed every production page, independently prove tensor/table placement, then require exact bytes
# outside ten boundary writes and exact decoded valid/padding rows inside all 208 touched pages.
def test_llama_bfp8_writer_boundaries_preserve_packed_cache(mesh_device, device_params):
    evidence = Path(os.environ["LLAMA_PREFILL_EVIDENCE_DIR"])
    report_path = evidence / "writer-boundaries-report.json"
    metrics = {
        "status": "running",
        "scope": "2K BFP8 production cache writer boundary gate; no model, H2D runtime, transfer, or decode",
        "seed_writer_calls": 0,
        "test_writer_calls": 0,
    }
    cache = None
    before = after = None
    cleanup_errors = []
    failure = None
    try:
        decode = _load_decoder()
        cache = allocate_kv_cache(
            mesh_device,
            MeshConfig(MESH_SHAPE, TP),
            num_users=NUM_SLOTS,
            num_layers=NUM_LAYERS,
            max_seq_len=MAX_SEQ_LEN,
            cache_dtype=ttnn.bfloat8_b,
        )

        seed_start = time.perf_counter()
        for slot in range(NUM_SLOTS):
            for layer in range(NUM_LAYERS):
                for start in range(0, MAX_SEQ_LEN, GLOBAL_CHUNK):
                    _write(
                        mesh_device,
                        cache,
                        phase="seed",
                        slot=slot,
                        layer=layer,
                        start=start,
                        end=start + GLOBAL_CHUNK,
                        cleanup_errors=cleanup_errors,
                    )
                    metrics["seed_writer_calls"] += 1
        ttnn.synchronize_device(mesh_device)
        metrics["seed_seconds"] = time.perf_counter() - seed_start
        assert metrics["seed_writer_calls"] == NUM_SLOTS * NUM_LAYERS * 2

        table = build_kv_chunk_address_table(mesh_device=mesh_device, kv_cache=cache, chunk_size=GLOBAL_CHUNK)
        metrics["dram_banks"] = _validate_table_geometry(mesh_device, table)

        placement_start = time.perf_counter()
        comparisons, raw_bytes = _validate_seed_with_independent_views(cache, table, decode)
        metrics["independent_seed_comparisons"] = comparisons
        metrics["independent_seed_raw_bytes"] = raw_bytes
        metrics["independent_seed_seconds"] = time.perf_counter() - placement_start

        snapshot_start = time.perf_counter()
        before_receipts, before = _snapshot_phase(table, evidence, "before")
        metrics["before_snapshot"] = before_receipts
        metrics["before_snapshot_seconds"] = time.perf_counter() - snapshot_start

        write_start = time.perf_counter()
        for case in WRITER_CASES:
            _write(
                mesh_device,
                cache,
                phase="write",
                slot=case.slot,
                layer=case.layer,
                start=case.start,
                end=case.end,
                cleanup_errors=cleanup_errors,
            )
            metrics["test_writer_calls"] += 1
        ttnn.synchronize_device(mesh_device)
        metrics["test_write_seconds"] = time.perf_counter() - write_start
        assert metrics["test_writer_calls"] == len(WRITER_CASES) == 10

        snapshot_start = time.perf_counter()
        after_receipts, after = _snapshot_phase(table, evidence, "after")
        metrics["after_snapshot"] = after_receipts
        metrics["after_snapshot_seconds"] = time.perf_counter() - snapshot_start

        compare_start = time.perf_counter()
        metrics.update(_compare_snapshots(before, after, decode))
        metrics["comparison_seconds"] = time.perf_counter() - compare_start
        assert metrics["touched_pages"] == EXPECTED_TOUCHED_PAGES
        assert metrics["touched_bytes"] == EXPECTED_TOUCHED_BYTES
        assert metrics["untouched_pages"] == EXPECTED_UNTOUCHED_PAGES
        assert metrics["untouched_bytes"] == EXPECTED_UNTOUCHED_BYTES
        metrics["status"] = "passed"
        logger.info("writer boundary gate passed: {}", metrics)
    except BaseException as exc:
        failure = exc
        metrics["status"] = "failed"
        metrics["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        for name, saved in (("before", before), ("after", after)):
            if saved is not None:
                try:
                    saved.close()
                except Exception as exc:
                    cleanup_errors.append(f"{name} snapshot close: {type(exc).__name__}: {exc}")
        if cache is not None:
            for name, tensor in (("cache.k", cache.k), ("cache.v", cache.v)):
                try:
                    tensor.deallocate(True)
                except Exception as exc:
                    cleanup_errors.append(f"{name} deallocate: {type(exc).__name__}: {exc}")
        metrics["cleanup_errors"] = cleanup_errors
        report_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        if cleanup_errors and failure is None:
            raise RuntimeError("; ".join(cleanup_errors))
