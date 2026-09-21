# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only capacity geometry, token placement and checkpoint-boundary regressions."""

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from models.demos.llama_3p1_8b_d_p.tests.unit.test_prefill_host import _config
from models.demos.llama_3p1_8b_d_p.tt.input import pack_token_ids, validate_chunk_range
from models.demos.llama_3p1_8b_d_p.tt.weights import CheckpointWeights, validate_checkpoint_config


class PrefillCapacityHostTests(unittest.TestCase):
    def geometry(self, max_seq_len=2048):
        try:
            module = importlib.import_module("models.demos.llama_3p1_8b_d_p.tt.prefill_geometry")
        except ModuleNotFoundError:
            self.fail("The requested capacity geometry is not implemented")
        return module.PrefillGeometry(max_seq_len)

    def capacity_call(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except TypeError as error:
            if "unexpected keyword argument 'max_seq_len'" in str(error):
                self.fail("The requested max_seq_len API is missing: " + str(error))
            raise

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
                geometry = self.geometry(capacity)
                self.assertEqual(geometry.cache_shape, (64, 1, local_cache, 128))
                self.assertEqual(geometry.local_cache_sequence, local_cache)
                self.assertEqual(geometry.rope_local_sequence, local_rope)
                self.assertGreaterEqual(local_rope * 4, capacity - 32 + 1024)

    # Zero, fractional, over-limit or non-chunk-aligned capacities cannot reach an allocator.
    def test_rejects_invalid_allocated_capacities(self):
        for capacity in (0, -1024, 32, 2049, 132096, True, 2048.0, "4096", None):
            with self.subTest(capacity=capacity), self.assertRaises((TypeError, ValueError)):
                self.geometry(capacity)

    # Rank-major gathers must reconstruct every token once; the old eight-block order fails at 4K.
    def test_sp_gather_reconstructs_natural_order(self):
        for capacity in (2048, 3072, 4096, 131072):
            with self.subTest(capacity=capacity):
                natural_blocks = list(range(capacity // 256))
                rank_major = [block for rank in range(4) for block in natural_blocks if block % 4 == rank]
                geometry = self.geometry(capacity)
                restored = [rank_major[index] for index in geometry.gather_block_order]
                self.assertEqual(restored, natural_blocks)
        self.assertEqual(self.geometry(4096).gather_block_order, (0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15))

    # A cache from another capacity or packed layout must fail before a model can write its planes.
    def test_cache_metadata_must_match_owning_geometry(self):
        geometry = self.geometry(4096)
        valid = dict(num_users=2, num_layers=32, max_seq_len=4096, sp=4)
        geometry.validate_cache_metadata(SimpleNamespace(**valid))
        for key, value in (("max_seq_len", 2048), ("num_layers", 1), ("num_users", 1), ("sp", 8)):
            changed = dict(valid, **{key: value})
            with self.subTest(field=key), self.assertRaises(ValueError):
                geometry.validate_cache_metadata(SimpleNamespace(**changed))

    # Absolute tags reveal wrong SP rotation for continuations, overlap and the last partial tile.
    def test_configured_capacity_preserves_position_packing(self):
        cases = (
            (4096, 2048, 3072),
            (4096, 2080, 2113),
            (4096, 4064, 4095),
            (3072, 3040, 3072),
            (131072, 131040, 131071),
        )
        for capacity, start, end in cases:
            with self.subTest(capacity=capacity, start=start, end=end):
                ids = (torch.arange(start, end, dtype=torch.int64) + 17) % 100000
                before = ids.clone()
                actual = self.capacity_call(
                    pack_token_ids, ids, actual_start=start, actual_end=end, max_seq_len=capacity, pad_id=7
                ).reshape(4, 256)
                expected_rows = []
                for rank in range(4):
                    expected_rows.append(
                        [
                            ((p + 17) % 100000 if p < end else 7)
                            for p in range(start, start + 1024)
                            if (p // 256) % 4 == rank
                        ]
                    )
                self.assertTrue(torch.equal(actual, torch.tensor(expected_rows)))
                self.assertTrue(torch.equal(ids, before))

    # The configured logical end, not physical chunk padding, decides whether a request is valid.
    def test_range_rejects_outside_capacity_and_invalid_start(self):
        self.capacity_call(validate_chunk_range, 4064, 4096, max_seq_len=4096)
        for start, end in ((4096, 4097), (4064, 4097), (2048, 3073), (1, 32), (4096, 4096)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                self.capacity_call(validate_chunk_range, start, end, max_seq_len=4096)

    # Cache writes retain their existing empty no-op range, while model inputs still require data.
    def test_geometry_distinguishes_empty_cache_write_from_model_input(self):
        geometry = self.geometry(4096)
        geometry.validate_chunk_range(4096, 4096, allow_empty=True)
        with self.assertRaises(ValueError):
            geometry.validate_chunk_range(4096, 4096)
        with self.assertRaises(ValueError):
            geometry.validate_chunk_range(4128, 4128, allow_empty=True)

    # Omitting capacity must keep the accepted 2K interval and reject the first out-of-range token.
    def test_default_input_capacity_remains_2k(self):
        result = pack_token_ids([9], actual_start=2016, actual_end=2017)
        self.assertEqual(tuple(result.shape), (1, 1, 1, 1024))
        with self.assertRaises(ValueError):
            pack_token_ids([9], actual_start=2048, actual_end=2049)

    # A checkpoint can cover the default but still be too short for a requested 4K cache.
    def test_checkpoint_limit_tracks_requested_capacity(self):
        config = _config()
        config["max_position_embeddings"] = 2048
        validate_checkpoint_config(config)
        with self.assertRaises(ValueError):
            self.capacity_call(validate_checkpoint_config, config, max_seq_len=4096)
        config["max_position_embeddings"] = 131072
        self.capacity_call(validate_checkpoint_config, config, max_seq_len=131072)

    # Loader construction must forward the requested limit before opening any tensor shard.
    def test_loader_rejects_short_checkpoint_before_index_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            config = _config()
            config["max_position_embeddings"] = 2048
            (path / "config.json").write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                self.capacity_call(CheckpointWeights, path, max_seq_len=4096)

    # Physical extents for separate model instances cannot change when another capacity is used.
    def test_geometry_is_immutable_and_instance_scoped(self):
        first, second = self.geometry(2048), self.geometry(4096)
        with self.assertRaises((AttributeError, TypeError)):
            first.max_seq_len = 4096
        self.assertEqual(first.cache_shape, (64, 1, 512, 128))
        self.assertEqual(second.cache_shape, (64, 1, 1024, 128))
