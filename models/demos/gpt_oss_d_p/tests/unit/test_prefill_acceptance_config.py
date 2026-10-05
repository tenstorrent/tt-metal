# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free regression tests for the 1K common-runner acceptance contract."""

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from models.demos.gpt_oss_d_p.tt.runners.acceptance import prefill_runner_scenario, validate_prefill_slot_traces


class PrefillAcceptanceConfigTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_rejects_partial_model_and_single_allocated_slot(self):
        for key, value in (
            ("PREFILL_NUM_LAYERS", "2"),
            ("PREFILL_NUM_USERS", "1"),
            ("PREFILL_MAX_SEQ_LEN", "512"),
            ("GPT_OSS_BOUNDED_SLIDING_KV", "1"),
        ):
            with self.subTest(key=key), patch.dict(os.environ, {key: value}):
                with self.assertRaises(ValueError):
                    prefill_runner_scenario()

    def test_rejects_lowered_or_nonfinite_pcc_gate(self):
        for floor in ("0.82", "nan", "inf", "1.1"):
            with self.subTest(floor=floor), patch.dict(os.environ, {"PREFILL_STANDALONE_CHUNKED_PCC": floor}):
                with self.assertRaises(ValueError):
                    prefill_runner_scenario()

    def test_runner_and_producer_cover_both_slots_and_drain(self):
        scenario = prefill_runner_scenario()
        self.assertEqual((scenario["users"], scenario["layers"], scenario["max_seq_len"]), (2, 36, 1024))
        self.assertEqual(scenario["expected_slots"], 2)
        self.assertEqual(scenario["env"]["PREFILL_CHUNK_SIZE"], "1024")
        self.assertEqual(scenario["producer"]["PREFILL_NUM_USERS"], "2")
        self.assertEqual(scenario["producer"]["PREFILL_PRODUCER_MAX_REQUESTS"], "2")
        self.assertEqual(scenario["producer"]["PREFILL_SEND_SHUTDOWN"], "1")

    def test_kv_pcc_inputs_reject_nonfinite_and_wrong_shape(self):
        from models.demos.gpt_oss_d_p.tt.runners.acceptance import validate_kv_for_pcc

        golden = torch.arange(64).reshape(1, 1, 64).float()
        validate_kv_for_pcc(golden, golden.clone())
        for bad in (golden[:, :, :-1], golden * float("nan"), golden * float("inf")):
            with self.assertRaises(ValueError):
                validate_kv_for_pcc(golden, bad)
            with self.assertRaises(ValueError):
                validate_kv_for_pcc(bad, golden)

    def test_missing_or_extra_layer_completions_cannot_pass(self):
        from models.demos.gpt_oss_d_p.tt.runners.acceptance import require_layer_acks

        require_layer_acks(72, 72)
        for count in (0, 71, 73):
            with self.subTest(count=count), self.assertRaises(RuntimeError):
                require_layer_acks(count, 72)

    def test_forced_or_nonzero_runner_exit_cannot_pass(self):
        from models.demos.gpt_oss_d_p.tt.runners.acceptance import require_clean_runner_exit

        require_clean_runner_exit(0)
        # None means the owner was still running and required harness intervention.
        for returncode in (None, 1, -2, -9):
            with self.subTest(returncode=returncode), self.assertRaises(RuntimeError):
                require_clean_runner_exit(returncode)

    def _trace(self, root, *, tokens=1024, rows=1024, layers=36):
        root = Path(root)
        (root / "kv_cache").mkdir()
        (root / "metadata.json").write_text(json.dumps({"token_ids": list(range(tokens)), "rope_frame": "hf"}))
        # Real sparse safetensors, with independently specified header dimensions.
        size = rows * 8 * 64 * 2
        for layer in range(layers):
            header = {
                f"{kind}_cache_layer_{layer}": {
                    "dtype": "BF16",
                    "shape": [1, 8, rows, 64],
                    "data_offsets": [offset * size, (offset + 1) * size],
                }
                for offset, kind in enumerate(("key", "value"))
            }
            data = json.dumps(header).encode()
            data += b" " * (-len(data) % 8)
            with (root / "kv_cache" / f"layer_{layer}.safetensors").open("wb") as handle:
                handle.write(struct.pack("<Q", len(data)))
                handle.write(data)
                handle.truncate(8 + len(data) + 2 * size)
        return root

    def test_accepts_complete_causal_prefix_for_two_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = self._trace(tmp, tokens=5000, rows=5000)
            validate_prefill_slot_traces(str(trace), prefill_runner_scenario())

    def test_rejects_short_prompt_and_missing_last_layer(self):
        for kwargs in ({"tokens": 1023}, {"layers": 35}, {"rows": 1023}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as tmp:
                trace = self._trace(tmp, **kwargs)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    validate_prefill_slot_traces(str(trace), prefill_runner_scenario())

    def test_rejects_incompatible_rope_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = self._trace(tmp)
            path = trace / "metadata.json"
            metadata = json.loads(path.read_text())
            metadata["rope_frame"] = "meta"
            path.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                validate_prefill_slot_traces(str(trace), prefill_runner_scenario())


if __name__ == "__main__":
    unittest.main()
