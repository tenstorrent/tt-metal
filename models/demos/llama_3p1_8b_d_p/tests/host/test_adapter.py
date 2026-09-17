# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host adapter factory contracts."""

import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter


class AdapterTests(unittest.TestCase):
    def params(self, **overrides):
        values = dict(
            mesh_shape=(4, 8),
            num_layers=32,
            first_layer_idx=0,
            is_first_rank=True,
            is_last_rank=True,
            max_seq_len=2048,
            chunk_size=1024,
            num_users=2,
            capacity_factor=1,
            num_links=1,
            gate_mode_name="DEVICE_FP32",
            kv_only_last_layer=False,
            weight_cache_path=None,
        )
        values.update(overrides)
        return PrefillRunParams(**values)

    def adapter(self):
        with patch.dict(sys.modules, {"loguru": SimpleNamespace(logger=SimpleNamespace(info=lambda *a: None))}):
            try:
                return get_adapter("llama_3p1_8b")
            except KeyError:
                self.fail("Llama adapter is not registered")

    # Registry loading must not import the numerical or device stack.
    def test_registered_import_stays_light(self):
        program = (
            "import sys; from types import SimpleNamespace; "
            "sys.modules['loguru']=SimpleNamespace(logger=SimpleNamespace(info=lambda *a: None)); "
            "from models.demos.common.prefill.adapter import get_adapter; "
            "assert get_adapter('llama_3p1_8b').name == 'llama_3p1_8b'; "
            "assert not set(sys.modules) & {'torch','ttnn','transformers','safetensors'}"
        )
        result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    # Slot and length values reach allocation exactly as resolved by the runner.
    def test_allocation_forwards_geometry(self):
        adapter = self.adapter()
        captured = {}

        def allocate(mesh, mesh_config, **kwargs):
            captured.update(kwargs)
            return "allocated"

        prefix = "models.demos.llama_3p1_8b_d_p.tt."
        dtype = object()
        stubs = {
            "ttnn": SimpleNamespace(bfloat8_b=dtype),
            prefix + "config": SimpleNamespace(MeshConfig=lambda shape, tp: (shape, tp)),
            prefix + "kv_cache": SimpleNamespace(allocate_kv_cache=allocate),
        }
        with patch.dict(sys.modules, stubs):
            self.assertEqual(
                adapter.allocate_kv_cache(mesh_device="mesh", hf_config=None, params=self.params()), "allocated"
            )
        self.assertEqual(captured, dict(num_users=2, num_layers=32, max_seq_len=2048, cache_dtype=dtype))

    # Unsupported shapes fail before device imports or weight loading.
    def test_unsupported_modes_and_lengths_fail_early(self):
        adapter = self.adapter()
        for overrides in (
            dict(max_seq_len=132096),
            dict(num_users=1),
            dict(use_trace=True),
            dict(first_layer_idx=1),
            dict(sp_axis=1, tp_axis=0),
            dict(dflash_enabled=True),
        ):
            with self.assertRaises((ValueError, NotImplementedError)):
                adapter.allocate_kv_cache(mesh_device="mesh", hf_config=None, params=self.params(**overrides))

    # The common runner must resolve two slots from the model manifest before adapter allocation.
    def test_manifest_declares_two_slots(self):
        path = Path(__file__).resolve().parents[2] / "tt/runners/manifests/llama_3p1_8b.json"
        self.assertEqual(json.loads(path.read_text())["env"].get("PREFILL_NUM_USERS"), "2")
