# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host regression for producer GQA table readback; UMD reads are replaced by known tile bytes."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from models.demos.common.prefill.runners import prefill_producer as producer


def _bfp8_page(values):
    # Integer values below 64 are exact with shared exponent 133. Encode the four tile faces
    # independently of the producer decoder, including the sign bit of every mantissa.
    tiles = values.reshape(32, 4, 32).permute(1, 0, 2)
    faces = tiles.reshape(4, 2, 16, 2, 16).permute(0, 1, 3, 2, 4).reshape(4, 1024)
    mantissas = faces.abs().to(torch.uint8) | ((faces < 0).to(torch.uint8) << 7)
    return torch.cat((torch.full((4, 64), 133, dtype=torch.uint8), mantissas), dim=1).numpy().tobytes()


class _Table:
    def __init__(self, layers, heads):
        self.names = [f"{kind}_h{head}" for kind in ("k", "v") for head in range(heads)]
        self.geometry = SimpleNamespace(
            num_layers=layers, num_slots=2, chunk_n_tokens=32, chunk_size_bytes=4352, max_sequence_length=64
        )
        self.foreign = set()
        self.pages = {}
        self.reads = []

    def num_configs(self):
        return len(self.names)

    def config_name(self, index):
        return self.names[index]

    def config(self, index):
        return self.geometry

    def lookup(self, layer, position, slot, config_id):
        assert slot == 1
        group = 99 if (layer, config_id) in self.foreign else config_id % (len(self.names) // 2)
        return SimpleNamespace(device_group_index=group, noc_addr=(layer, config_id, position), size_bytes=4352)

    def get_device_group(self, index):
        return SimpleNamespace(fabric_node_ids=[SimpleNamespace(mesh_id=0, chip_id=index)])

    def read(self, unique_id, address, size):
        self.reads.append(address)
        assert size == 4352
        return self.pages[address]


class _AckChannel:
    def __init__(self, batches):
        self.batches = list(batches)

    def try_consume_all(self):
        return self.batches.pop(0) if self.batches else 0


@pytest.fixture
def gqa_trace(tmp_path, monkeypatch):
    def make(model_name="llama_3p1_8b"):
        is_llama = model_name == "llama_3p1_8b"
        layers, heads, rotary_dim = (32, 8, 128) if is_llama else (2, 2, 64)
        config = SimpleNamespace(NUM_LAYERS=layers, NUM_KEY_VALUE_HEADS=heads, HEAD_DIM=128, ROTARY_DIM=rotary_dim)
        monkeypatch.setattr(producer, "ADAPTER", SimpleNamespace(name=model_name, model_config=config))
        monkeypatch.setattr(producer, "NUM_LAYERS", layers)
        monkeypatch.delenv("PREFILL_PCC_GOLDEN_LEN", raising=False)
        table = _Table(layers, heads)
        cache_dir = tmp_path / "kv_cache"
        cache_dir.mkdir(exist_ok=True)
        (tmp_path / "metadata.json").write_text(json.dumps({"token_ids": list(range(33))}))
        generator = torch.Generator().manual_seed(823)
        for layer in range(layers):
            key, value = [torch.randint(-63, 64, (1, heads, 64, 128), generator=generator).float() for _ in range(2)]
            save_file(
                {
                    f"key_cache_layer_{layer}": key[:, :, :33].contiguous(),
                    f"value_cache_layer_{layer}": value[:, :, :33].contiguous(),
                },
                cache_dir / f"layer_{layer}.safetensors",
            )
            half = rotary_dim // 2
            rotated = key.clone()
            rotated[..., :rotary_dim:2] = key[..., :half]
            rotated[..., 1:rotary_dim:2] = key[..., half:rotary_dim]
            for kind, tensor in enumerate((rotated, value)):
                for head in range(heads):
                    for position in (0, 32):
                        table.pages[layer, kind * heads + head, position] = _bfp8_page(
                            tensor[0, head, position : position + 32]
                        )
        monkeypatch.setattr(producer.ttnn.experimental.disaggregation, "read_dram_umd", table.read)
        return table, {(0, head): head for head in range(heads)}, tmp_path

    return make


# Exercise the public dispatch and real BFP8 decoder across every Llama layer/head and both pages.
# The GPT-OSS case preserves partial rotary dimensions and rank-local layer filtering.
@pytest.mark.parametrize("model_name", ["llama_3p1_8b", "gpt_oss_d_p"])
def test_producer_gqa_pcc_reads_named_heads_and_rotates_only_keys(gqa_trace, model_name):
    table, devices, trace = gqa_trace(model_name)
    if model_name == "gpt_oss_d_p":
        table.foreign.add((0, 0))
    result = producer._read_slot_kv_and_check_pcc(table, devices, 1, 33, trace)
    assert result == pytest.approx({"k": 1.0, "v": 1.0}, abs=1e-6)
    expected_layers = range(32) if model_name == "llama_3p1_8b" else (1,)
    assert set(table.reads) == {
        (layer, config, position)
        for layer in expected_layers
        for config in range(table.num_configs())
        for position in (0, 32)
    }


# A single-rank Llama verdict must not silently skip missing heads/layers or an empty comparison.
@pytest.mark.parametrize(
    "invalid,error,message",
    [
        ("missing_config", ValueError, "config order"),
        ("wrong_order", ValueError, "config order"),
        ("foreign_layer", KeyError, "no fabric node"),
        ("foreign_head", KeyError, "no fabric node"),
        ("empty_prefix", ValueError, "nonempty prefix"),
    ],
)
def test_producer_llama_gqa_rejects_incomplete_verification(gqa_trace, invalid, error, message, expect_error):
    table, devices, trace = gqa_trace()
    real_len = 33
    if invalid == "missing_config":
        table.names.pop()
    elif invalid == "wrong_order":
        table.names[0], table.names[1] = table.names[1], table.names[0]
    elif invalid == "foreign_layer":
        table.foreign.add((31, 0))
    elif invalid == "foreign_head":
        table.foreign.add((0, 15))
    else:
        real_len = 0
    with expect_error(error, message):
        producer._read_slot_kv_and_check_pcc(table, devices, 1, real_len, trace)


# Reject nonfinite raw data before PCC utilities can sanitize it into a misleading finite score.
def test_producer_llama_gqa_rejects_nonfinite_device_values(gqa_trace, expect_error):
    table, devices, trace = gqa_trace()
    page = np.frombuffer(table.pages[0, 0, 0], dtype=np.uint8).copy().reshape(4, 1088)
    page[:, :64] = 255
    page[:, 64:] = 127
    table.pages[0, 0, 0] = page.tobytes()
    with np.errstate(over="ignore"):
        with expect_error(ValueError, "nonfinite"):
            producer._read_slot_kv_and_check_pcc(table, devices, 1, 33, trace)


# The Llama gate records the measured request range and passes only on one exact ack per layer and chunk.
@pytest.mark.parametrize(
    "channel,received,ok",
    [
        (_AckChannel([128, 0]), 128, True),
        (_AckChannel([127]), 127, False),
        (_AckChannel([129]), 129, False),
        (_AckChannel([128, 1]), 129, False),
        (None, 0, False),
    ],
    ids=["exact", "missing", "same-batch-extra", "late-extra", "missing-channel"],
)
def test_llama_layer_ack_verdict_requires_exact_count(channel, received, ok):
    summary = producer._drain_llama_layer_acks(
        channel,
        layers_per_chunk=32,
        chunks=4,
        request_id_start=0,
        timeout_s=0,
    )

    assert summary == {
        "ok": ok,
        "expected": 128,
        "received": received,
        "layers_per_chunk": 32,
        "chunks": 4,
        "expected_request_id_start": 0,
        "expected_request_id_end": 3,
    }


# A pure token feeder deliberately has no ack channel; Llama warmup must remain valid when verification is off.
def test_llama_warmup_without_verification_does_not_require_layer_acks():
    assert producer._drain_warmup_layer_acks(None, layers_per_chunk=32, chunks=1, strict_llama=False) is None
