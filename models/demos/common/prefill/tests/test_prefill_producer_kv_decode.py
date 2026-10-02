# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from models.demos.common.prefill.runners import prefill_producer as producer
from models.demos.common.prefill.runners.prefill_producer import _decode_kv_chunk


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16], ids=["fp8_e4m3", "bfloat16"])
def test_decode_row_major_kv_chunk(dtype):
    head_dim = 576
    values = torch.linspace(-2, 2, 32 * head_dim, dtype=torch.float32).reshape(32, head_dim).to(dtype)
    raw = values.view(torch.uint8).numpy().tobytes()

    actual = _decode_kv_chunk(raw, head_dim)

    assert torch.equal(actual, values.float())


def test_decode_row_major_fp8_kv_chunk_with_page_padding():
    head_dim = 33
    row_size_bytes = 64
    values = (
        torch.arange(32 * head_dim, dtype=torch.float32).reshape(32, head_dim).remainder(31).to(torch.float8_e4m3fn)
    )
    padded = torch.full((32, row_size_bytes), 0xA5, dtype=torch.uint8)
    padded[:, :head_dim] = values.view(torch.uint8)

    actual = _decode_kv_chunk(padded.numpy().tobytes(), head_dim)

    assert torch.equal(actual, values.float())


def test_decode_packed_scaled_fp8_kv_chunk_with_page_padding():
    latent = torch.arange(32 * 512, dtype=torch.float32).reshape(32, 512).remainder(31).sub(15).to(torch.float8_e4m3fn)
    scales = torch.tensor([0.25, 0.5, 1.0, 2.0], dtype=torch.float32).repeat(32, 1)
    rope = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64).to(torch.bfloat16)
    rows = torch.full((32, 672), 0xA5, dtype=torch.uint8)
    rows[:, :512] = latent.view(torch.uint8)
    rows[:, 512:528] = scales.view(torch.uint8)
    rows[:, 528:656] = rope.view(torch.uint8)

    actual = _decode_kv_chunk(rows.numpy().tobytes(), head_dim=576)
    expected = torch.cat((latent.float() * scales.repeat_interleave(128, dim=-1), rope.float()), dim=-1)

    assert torch.equal(actual, expected)


def test_decode_unknown_kv_chunk_rejected(expect_error):
    with expect_error(ValueError, "unsupported"):
        _decode_kv_chunk(bytes(17), head_dim=576)


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


@pytest.fixture
def gqa_trace(tmp_path, monkeypatch):
    def make(frame="hf"):
        layers, heads, rotary_dim = 32, 8, 128
        config = SimpleNamespace(NUM_LAYERS=layers, NUM_KEY_VALUE_HEADS=heads, HEAD_DIM=128, ROTARY_DIM=rotary_dim)
        monkeypatch.setattr(producer, "ADAPTER", SimpleNamespace(name="llama_3p1_8b", model_config=config))
        monkeypatch.setattr(producer, "NUM_LAYERS", layers)
        monkeypatch.delenv("PREFILL_PCC_GOLDEN_LEN", raising=False)
        table = _Table(layers, heads)
        cache_dir = tmp_path / "kv_cache"
        cache_dir.mkdir(exist_ok=True)
        (tmp_path / "metadata.json").write_text(json.dumps({"rope_frame": frame}))
        generator = torch.Generator().manual_seed(823)
        for layer in range(layers):
            key, value = [torch.randint(-63, 64, (1, heads, 64, 128), generator=generator).float() for _ in range(2)]
            half = rotary_dim // 2
            rotated = key.clone()
            rotated[..., :rotary_dim:2] = key[..., :half]
            rotated[..., 1:rotary_dim:2] = key[..., half:rotary_dim]
            save_file(
                {
                    f"key_cache_layer_{layer}": (rotated if frame == "meta" else key)[:, :, :33].contiguous(),
                    f"value_cache_layer_{layer}": value[:, :, :33].contiguous(),
                },
                cache_dir / f"layer_{layer}.safetensors",
            )
            for kind, tensor in enumerate((rotated, value)):
                for head in range(heads):
                    for position in (0, 32):
                        table.pages[layer, kind * heads + head, position] = _bfp8_page(
                            tensor[0, head, position : position + 32]
                        )
        monkeypatch.setattr(producer.ttnn.experimental.disaggregation, "read_dram_umd", table.read)
        return table, {(0, head): head for head in range(heads)}, tmp_path

    return make


# Both golden frames must match the same device pages; Meta keys must not be permuted a second time.
@pytest.mark.parametrize("frame", ["hf", "meta"])
def test_producer_gqa_pcc_reads_named_heads_and_rotates_only_keys(gqa_trace, frame):
    table, devices, trace = gqa_trace(frame)
    result = producer._read_slot_kv_and_check_pcc(table, devices, 1, 33, trace)
    assert result == pytest.approx({"k": 1.0, "v": 1.0}, abs=1e-6)
    assert set(table.reads) == {
        (layer, config, position)
        for layer in range(32)
        for config in range(table.num_configs())
        for position in (0, 32)
    }


# Missing ownership for the final layer must fail instead of producing PCC from a partial readback.
def test_producer_llama_gqa_rejects_missing_layer(gqa_trace, expect_error):
    table, devices, trace = gqa_trace()
    table.foreign.add((31, 0))
    with expect_error(KeyError, "no fabric node"):
        producer._read_slot_kv_and_check_pcc(table, devices, 1, 33, trace)
