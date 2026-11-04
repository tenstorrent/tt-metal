# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only: the producer's Gate 1 read-back for V4, over a fake DRAM behind ``read_dram_umd``.

Flash's first five layers (SWA, SWA, CSA, HCA, CSA), two users holding the same prompt. Each config's chunks are
encoded at the addresses the table lists, in the device's format (bf16 tiles, bf16 rows, bfp8 tiles), and the
golden is written as the trace lays it out. The read-back must score every kind 1.0 through the producer's own
dispatch, and lose exactly the kind whose chunk is corrupted.
"""

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import ttnn
from models.demos.common.prefill.runners import prefill_producer
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.hf_config import v4_hf_config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.tests.v4.helpers import build_fake_table, fake_fnids, fake_v4_caches, stage_layout
from models.demos.deepseek_v3_d_p.tt.mla.indexer import normalized_hadamard_matrix
from models.demos.deepseek_v3_d_p.tt.runners.adapters.deepseek_v4 import DeepSeekV4FlashAdapter
from models.demos.deepseek_v3_d_p.tt.v4.kv_caches import CACHE_KINDS
from models.demos.deepseek_v3_d_p.tt.v4.kv_validation import COMPRESSED_STREAM, INDEX_STREAM

CONFIG = v4_hf_config(DeepSeekV4FlashConfig)
LAYERS, USERS, REAL_LEN = 5, 2, 10240
CSA_LAYERS, HCA_LAYERS = (2, 4), (3,)


def _faces(rows):
    """[32, width] -> [tiles, 1024] in the device's tile order: tiles left to right, four 16x16 faces each."""
    tiles = rows.reshape(32, -1, 32).transpose(1, 0, 2)
    return tiles.reshape(-1, 2, 16, 2, 16).transpose(0, 1, 3, 2, 4).reshape(-1, 1024)


def _bf16_tiles(rows):
    return torch.from_numpy(_faces(rows.numpy())).bfloat16().view(torch.int16).numpy().tobytes()


def _bfp8_tiles(rows):
    """Integer rows in [-127, 127]: a shared exponent of 133 is a scale of 1, so they encode exactly."""
    faces = _faces(rows.numpy().astype(np.int32))
    raw = np.empty((len(faces), 1088), dtype=np.uint8)
    raw[:, :64] = 133
    raw[:, 64:] = np.abs(faces).astype(np.uint8) | ((faces < 0).astype(np.uint8) << 7)
    return raw.tobytes()


def _bf16_row(row):
    return row.bfloat16().view(torch.int16).numpy().tobytes()


@pytest.fixture
def gate(monkeypatch, tmp_path):
    """The table, the fake DRAM it addresses, the golden on disk, and the producer pointed at V4."""
    caches = fake_v4_caches(CONFIG, 0, LAYERS)
    layouts = [[stage_layout(stage, fake_fnids(0))] for stage in caches.migration_stages(CONFIG)]
    table = build_fake_table(tmp_path, CONFIG, layouts, (2, 2), USERS)
    device_map = {(0, chip): chip for chip in range(4)}
    memory = {}
    monkeypatch.setattr(ttnn.experimental.disaggregation, "read_dram_umd", lambda uid, addr, size: memory[uid, addr])
    monkeypatch.setattr(prefill_producer, "ADAPTER", DeepSeekV4FlashAdapter())
    monkeypatch.setattr(prefill_producer, "NUM_LAYERS", LAYERS)
    monkeypatch.delenv("PREFILL_PCC_GOLDEN_LEN", raising=False)

    def place(layer, position, config_id, payload):
        for slot in range(USERS):
            loc = table.lookup(layer, position, slot, config_id)
            chip = int(table.get_device_group(loc.device_group_index).fabric_node_ids[0].chip_id)
            memory[chip, loc.noc_addr] = payload

    def golden(stream, layer, rows):
        group, name = stream[0], stream[1].format(layer)
        (tmp_path / group / name).mkdir(parents=True)
        save_file(
            {name: rows.contiguous()}, str(tmp_path / group / name / f"rows_00000000_{len(rows):08d}.safetensors")
        )

    torch.manual_seed(0)
    unrotate = normalized_hadamard_matrix(CONFIG.index_head_dim).float()
    for layer in HCA_LAYERS + CSA_LAYERS:
        rate = CONFIG.compress_rates[CONFIG.layer_types[layer]]
        entries = -(-REAL_LEN // rate // 32) * 32
        compressed = torch.randint(-100, 100, (entries, CONFIG.head_dim)).float()
        golden(COMPRESSED_STREAM, layer, compressed)
        if layer in HCA_LAYERS:
            for p in range(0, entries, 32):
                place(layer, p, 0, _bf16_tiles(compressed[p : p + 32]))
            continue
        for p in range(entries):
            place(layer, p, 1, _bf16_row(compressed[p]))
        # The device holds the keys Hadamard-rotated; the golden holds them as the model computes them.
        rotated = torch.randint(-100, 100, (entries, CONFIG.index_head_dim)).float()
        golden(INDEX_STREAM, layer, rotated @ unrotate)
        for p in range(0, entries, 32):
            place(layer, p, 2, _bfp8_tiles(rotated[p : p + 32]))
    return table, device_map, memory, tmp_path, place


def _read(gate, slot):
    table, device_map, _, trace_dir, _ = gate
    return prefill_producer._read_slot_kv_and_check_pcc(table, device_map, slot, REAL_LEN, trace_dir)


@pytest.mark.parametrize("slot", range(USERS))
def test_every_kind_scores_one(gate, slot):
    assert _read(gate, slot) == {kind: pytest.approx(1.0, abs=1e-6) for kind in CACHE_KINDS}


@pytest.mark.parametrize("config_id, layer", [(0, 3), (1, 4), (2, 2)], ids=list(CACHE_KINDS))
def test_a_corrupted_chunk_lowers_only_its_kind(gate, config_id, layer):
    table, *_, place = gate
    loc = table.lookup(layer, 0, 0, config_id)
    place(layer, 0, config_id, bytes(loc.size_bytes))
    scores = _read(gate, 0)
    for i, kind in enumerate(CACHE_KINDS):
        assert (scores[kind] < 1 - 1e-5) == (i == config_id), scores


def test_a_trace_with_no_golden_verifies_nothing(gate, tmp_path, expect_error):
    table, device_map, *_ = gate
    with expect_error(RuntimeError, "nothing verified"):
        prefill_producer._read_slot_kv_and_check_pcc(table, device_map, 0, REAL_LEN, tmp_path / "empty")
