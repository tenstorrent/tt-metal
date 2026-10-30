# SPDX-License-Identifier: Apache-2.0
"""Metadata-only regression tests; no HF model or weight tensor is loaded."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

MODULE = Path(__file__).with_name('hf_streaming_reference.py')
spec = importlib.util.spec_from_file_location('hf_streaming_reference', MODULE)
hf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hf)


class MetadataTests(unittest.TestCase):
    def fixture(self, root, dtype='BF16', offsets=(0, 8)):
        name = 'model.language_model.layers.0.weight'
        header = json.dumps({name: {'dtype': dtype, 'shape': [2, 2], 'data_offsets': list(offsets)}}).encode()
        (root/'weights.safetensors').write_bytes(struct.pack('<Q', len(header))+header+b'\0'*8)
        (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {name: 'weights.safetensors'}}))

    def test_index_uses_original_shard_and_remapped_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root)
            index, meta = hf.build_index(root)
            self.assertEqual(index['model.layers.0.weight']['weight_name'], 'model.language_model.layers.0.weight')
            self.assertEqual(index['model.layers.0.weight']['safetensors_file'], str(root/'weights.safetensors'))
            self.assertEqual(meta['weight_bytes'], 8)
            self.assertEqual(len(list(root.iterdir())), 2)

    def test_rejects_precision_or_truncated_weights(self):
        for dtype, offsets in [('F32', (0, 8)), ('BF16', (0, 12))]:
            with self.subTest(dtype=dtype, offsets=offsets), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); self.fixture(root, dtype, offsets)
                with self.assertRaises(ValueError): hf.build_index(root)

    def test_guard_rejects_unlimited_memory_or_tt_devices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'memory.max').write_text('max')
            with self.assertRaises(RuntimeError): hf.check_guard(root, [])
            (root/'memory.max').write_text('35000000000')
            self.assertEqual(hf.check_guard(root, [])['memory_max_bytes'], 35000000000)
            with self.assertRaises(RuntimeError): hf.check_guard(root, ['tenstorrent/0'])
            (root/'memory.max').write_text('35000000001')
            with self.assertRaises(RuntimeError): hf.check_guard(root, [])

    def test_validation_receipt_is_bound_to_source_checkpoint_and_success(self):
        good = {'status':'validated', 'source_sha256':'s', 'checkpoint_identity':{'index':'i'}, 'top100_exact':True}
        hf.require_validation(good, 's', {'index':'i'})
        for patch in [{'status':'failed'}, {'top100_exact':False}, {'source_sha256':'other'}, {'checkpoint_identity':{'index':'other'}}]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                hf.require_validation({**good, **patch}, 's', {'index':'i'})


if __name__ == '__main__': unittest.main()
