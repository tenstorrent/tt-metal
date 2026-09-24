# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free tests for input packing and raw-HF checkpoint contracts."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from models.demos.llama_3p1_8b_d_p.tt.input import pack_token_ids, validate_chunk_range
from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import PrefillGeometry, validate_mesh
from models.demos.llama_3p1_8b_d_p.tt.weights import CheckpointWeights, validate_checkpoint_config


def _config():
    return {
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "tie_word_embeddings": False,
        "max_position_embeddings": 131072,
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    }


# Absolute-position tags expose wrong SP rotation, wrong encounter order, and padding overwrites.
# The expected mapping is enumerated independently for every row rather than using the packer.
@pytest.mark.parametrize(
    "capacity,start,end",
    [
        (2048, 0, 1),
        (2048, 0, 31),
        (2048, 0, 32),
        (2048, 0, 33),
        (2048, 0, 1024),
        (2048, 32, 65),
        (2048, 224, 257),
        (2048, 480, 511),
        (2048, 480, 512),
        (2048, 480, 513),
        (2048, 992, 1033),
        (2048, 1024, 1025),
        (2048, 1024, 1537),
        (2048, 2016, 2048),
        (4096, 2048, 3072),
        (4096, 2080, 2113),
        (4096, 4064, 4095),
        (3072, 3040, 3072),
        (131072, 131040, 131071),
    ],
)
def test_pack_token_ids_absolute_rows_and_true_length(capacity, start, end):
    ids = torch.arange(start + 100, end + 100, dtype=torch.int64) % 100000
    before = ids.clone()
    actual = pack_token_ids(ids, actual_start=start, actual_end=end, pad_id=17, max_seq_len=capacity).reshape(4, 256)
    for sp in range(4):
        expected = []
        for absolute in range(start, start + 1024):
            if (absolute // 256) % 4 == sp:
                expected.append((absolute + 100) % 100000 if absolute < end else 17)
        assert torch.equal(actual[sp], torch.tensor(expected))
    assert torch.equal(ids, before)


# Invalid true lengths and IDs must fail before transfer or cache mutation. Booleans and floating
# values are rejected explicitly instead of being converted into plausible token IDs.
@pytest.mark.parametrize(
    "ids,start,end,pad,error",
    [
        ([1], 0, 0, 0, ValueError),
        ([1], 1, 2, 0, ValueError),
        ([1], 0, 2, 0, ValueError),
        ([1], 2048, 2049, 0, ValueError),
        ([1], False, 1, 0, TypeError),
        ([128256], 0, 1, 0, ValueError),
        ([-1], 0, 1, 0, ValueError),
        ([True], 0, 1, 0, ValueError),
        ([1.0], 0, 1, 0, ValueError),
        (torch.tensor([1.0]), 0, 1, 0, ValueError),
        ([1], 0, 1, 128256, ValueError),
    ],
)
def test_pack_token_ids_rejects_invalid_input(ids, start, end, pad, error):
    with pytest.raises(error):  # allow-pytest.raises: device-free test also runs outside repository conftest
        pack_token_ids(ids, actual_start=start, actual_end=end, pad_id=pad)


# A checkpoint with GPT-OSS-style or different Llama mathematics cannot enter this wrapper merely
# because some tensor dimensions happen to match. The exact Llama3 frequency policy is checked too.
@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_size", 8192),
        ("rms_norm_eps", 1e-6),
        ("hidden_act", "gelu"),
        ("tie_word_embeddings", True),
        ("attention_bias", True),
        ("mlp_bias", True),
        ("sliding_window", 128),
        ("rope_scaling", {"rope_type": "yarn", "factor": 8.0}),
    ],
)
def test_checkpoint_config_rejects_architecture_substitutions(field, value):
    correct = _config()
    validate_checkpoint_config(correct)
    changed = copy.deepcopy(correct)
    changed[field] = value
    with pytest.raises(ValueError):  # allow-pytest.raises: device-free test also runs outside repository conftest
        validate_checkpoint_config(changed)


# Asymmetric Q/K rows must survive disk loading exactly. This small shard checks the generic read
# boundary directly; the real-checkpoint gate separately verifies all production layer dimensions.
def test_checkpoint_read_preserves_raw_qk_rows(tmp_path):
    q = torch.arange(32).reshape(8, 4).bfloat16()
    k = -q - 10
    names = {"model.layers.0.self_attn.q_proj.weight": q, "model.layers.0.self_attn.k_proj.weight": k}
    save_file(names, tmp_path / "weights.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(_config()))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "weights.safetensors" for name in names}})
    )
    loader = CheckpointWeights(tmp_path)
    loaded = loader._read({name: (8, 4) for name in names})
    for name, expected in names.items():
        assert torch.equal(loaded[name], expected)
    with pytest.raises(ValueError):  # allow-pytest.raises: device-free test also runs outside repository conftest
        loader._read({next(iter(names)): (4, 8)})


class PrefillCapacityHostTests(unittest.TestCase):
    # Wrong cache or RoPE extents would truncate the final SP stripe or its padded physical read.
    def test_capacity_derives_cache_and_padded_rope_extents(self):
        for capacity, local_cache, local_rope in (
            (1024, 256, 512),
            (2048, 512, 768),
            (3072, 768, 1024),
            (4096, 1024, 1280),
            (131072, 32768, 33024),
        ):
            with self.subTest(capacity=capacity):
                geometry = PrefillGeometry(capacity)
                self.assertEqual(geometry.cache_shape, (64, 1, local_cache, 128))
                self.assertEqual(geometry.local_cache_sequence, local_cache)
                self.assertEqual(geometry.rope_local_sequence, local_rope)
                self.assertGreaterEqual(local_rope * 4, capacity - 32 + 1024)

    # Zero, fractional, over-limit or non-chunk-aligned capacities cannot reach an allocator.
    def test_rejects_invalid_allocated_capacities(self):
        for capacity in (0, -1024, 32, 2049, 132096, True, 2048.0, "4096", None):
            with self.subTest(capacity=capacity), self.assertRaises((TypeError, ValueError)):
                PrefillGeometry(capacity)

    # Rank-major gathers must reconstruct every token once; the old eight-block order fails at 4K.
    def test_sp_gather_reconstructs_natural_order(self):
        for capacity in (2048, 3072, 4096, 131072):
            with self.subTest(capacity=capacity):
                natural_blocks = list(range(capacity // 256))
                rank_major = [block for rank in range(4) for block in natural_blocks if block % 4 == rank]
                geometry = PrefillGeometry(capacity)
                restored = [rank_major[index] for index in geometry.gather_block_order]
                self.assertEqual(restored, natural_blocks)
        self.assertEqual(
            PrefillGeometry(4096).gather_block_order, (0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15)
        )

    # A cache from another capacity or packed layout must fail before a model can write its planes.
    def test_cache_metadata_must_match_owning_geometry(self):
        geometry = PrefillGeometry(4096)
        valid = dict(num_users=2, num_layers=32, max_seq_len=4096, sp=4)
        geometry.validate_cache_metadata(SimpleNamespace(**valid))
        for key, value in (("max_seq_len", 2048), ("num_layers", 1), ("num_users", 1), ("sp", 8)):
            changed = dict(valid, **{key: value})
            with self.subTest(field=key), self.assertRaises(ValueError):
                geometry.validate_cache_metadata(SimpleNamespace(**changed))

    # The configured logical end, not physical chunk padding, decides whether a request is valid.
    def test_range_rejects_outside_capacity_and_invalid_start(self):
        validate_chunk_range(4064, 4096, max_seq_len=4096)
        for start, end in ((4096, 4097), (4064, 4097), (2048, 3073), (1, 32), (4096, 4096)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                validate_chunk_range(start, end, max_seq_len=4096)

    # Cache writes retain their existing empty no-op range, while model inputs still require data.
    def test_geometry_distinguishes_empty_cache_write_from_model_input(self):
        geometry = PrefillGeometry(4096)
        geometry.validate_chunk_range(4096, 4096, allow_empty=True)
        with self.assertRaises(ValueError):
            geometry.validate_chunk_range(4096, 4096)
        with self.assertRaises(ValueError):
            geometry.validate_chunk_range(4128, 4128, allow_empty=True)

    # A checkpoint can cover the default but still be too short for a requested 4K cache.
    def test_checkpoint_limit_tracks_requested_capacity(self):
        config = _config()
        config["max_position_embeddings"] = 2048
        validate_checkpoint_config(config)
        with self.assertRaises(ValueError):
            validate_checkpoint_config(config, max_seq_len=4096)
        config["max_position_embeddings"] = 131072
        validate_checkpoint_config(config, max_seq_len=131072)

    # Loader construction must forward the requested limit before opening any tensor shard.
    def test_loader_rejects_short_checkpoint_before_index_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            config = _config()
            config["max_position_embeddings"] = 2048
            (path / "config.json").write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                CheckpointWeights(path, max_seq_len=4096)


CHECKPOINT = Path(os.environ.get("LLAMA31_8B_CHECKPOINT", "/mnt/models/meta-llama/Llama-3.1-8B-Instruct"))


# Every real layer and every weight must equal the independently named HF tensor byte for byte.
# This catches swapped layer mappings even when observer indices still report the correct order.
# Only one layer plus one comparison tensor is resident, bounding host memory during this gate.
@pytest.mark.parametrize("layer_idx", range(32))
def test_prefill_raw_checkpoint_layer_identity(layer_idx):
    loaded = CheckpointWeights(CHECKPOINT).layer(layer_idx)
    index = json.loads((CHECKPOINT / "model.safetensors.index.json").read_text())["weight_map"]
    expected_names = (
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    assert set(loaded) == set(expected_names)
    for name in expected_names:
        full_name = f"model.layers.{layer_idx}.{name}"
        with safe_open(CHECKPOINT / index[full_name], framework="pt", device="cpu") as shard:
            raw = shard.get_tensor(full_name)
        assert loaded[name].dtype == raw.dtype
        assert torch.equal(loaded[name], raw), (layer_idx, name)


# The shared guard must accept the validated Galaxy and reject incorrect mesh metadata or hardware
# before any component can allocate device tensors. These stand-ins keep the check CPU-only.
@pytest.mark.parametrize(
    "config_updates,device_shape,device_count,error",
    [
        ({}, (4, 8), 32, None),
        ({"mesh_shape": (8, 4)}, (4, 8), 32, "requires mesh_shape"),
        ({"sp": 8}, (4, 8), 32, "requires SP=4"),
        ({"tp": 4}, (4, 8), 32, "requires SP=4"),
        ({"sp_axis": 1}, (4, 8), 32, "requires SP=4"),
        ({"tp_axis": 0}, (4, 8), 32, "requires SP=4"),
        ({}, (8, 4), 32, "device requires"),
        ({}, (4, 8), 31, "device requires"),
    ],
)
def test_prefill_mesh_validation(config_updates, device_shape, device_count, error):
    config = SimpleNamespace(**dict(dict(mesh_shape=(4, 8), sp=4, tp=8, sp_axis=0, tp_axis=1), **config_updates))
    device = SimpleNamespace(shape=device_shape, get_num_devices=lambda: device_count)
    if error is None:
        validate_mesh(device, config, "test component")
        del config.tp
        with unittest.TestCase().assertRaisesRegex(ValueError, "mesh_config is missing: tp"):
            validate_mesh(device, config, "test component")
    else:
        with unittest.TestCase().assertRaisesRegex(ValueError, error):
            validate_mesh(device, config, "test component")
