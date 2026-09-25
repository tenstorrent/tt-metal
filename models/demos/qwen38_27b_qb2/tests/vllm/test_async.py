# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of the vLLM adapter's async output boundaries."""

import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from vllm_tt_plugin.async_decode import TTAsyncDecodeController, TTDecodeSubmission

from models.demos.qwen38_27b_qb2.tt import generator_vllm as adapter_module
from models.demos.qwen38_27b_qb2.tt.generator_vllm import Qwen38ForCausalLM


class HostAsyncTests(unittest.TestCase):
    def adapter(self, compatibility):
        generator = SimpleNamespace(counters=Counter(), mesh=object())
        with patch.dict("os.environ", {"QWEN_VLLM_HOST_COMPATIBILITY": str(int(compatibility))}):
            return Qwen38ForCausalLM(generator, 2, 128)

    def test_plugin_async_submission_accepts_completed_host_logits(self):
        adapter = self.adapter(True)
        logits = torch.randn(2, 1, 37)
        adapter.decode_forward = Mock(return_value=logits)
        runner = SimpleNamespace(
            model=adapter, kv_caches=object(), request_specific_rope=False, trace_mode="decode_only"
        )
        controller = TTAsyncDecodeController(runner)
        model_input = SimpleNamespace(
            unpadded_batch_size=2,
            tt_sampling_params=SimpleNamespace(enable_log_probs=torch.zeros(2, dtype=torch.bool)),
            perform_device_sampling=False,
            input_tokens=torch.tensor([[1], [2]], dtype=torch.int32),
            input_positions=torch.tensor([31, 63], dtype=torch.int32),
            block_tables=torch.zeros(2, 4, dtype=torch.int32),
            block_tables_per_layer=None,
            slot_remap=None,
        )
        with (
            patch.object(adapter_module.ttnn, "record_event") as record,
            patch.object(adapter_module.ttnn, "event_synchronize") as wait,
            patch.object(adapter_module.ttnn, "to_torch") as convert,
        ):
            submission = controller.submit_decode(model_input, read_from_device=False, async_read=True)
            finalized = controller.finalize_decode(submission)

        self.assertIs(submission.tt_out, logits)
        self.assertEqual(submission.read_events, [])
        self.assertFalse(submission.perform_device_sampling)
        self.assertIs(finalized.tt_out, logits)
        self.assertIsNone(finalized.tt_log_probs)
        self.assertEqual(adapter.generator.counters, {})
        self.assertNotIn("sampling_params", adapter.decode_forward.call_args.kwargs)
        record.assert_not_called()
        wait.assert_not_called()
        convert.assert_not_called()

    def test_host_logits_read_is_identity_in_both_read_modes(self):
        adapter = self.adapter(True)
        logits = torch.randn(2, 1, 37)
        with (
            patch.object(adapter_module.ttnn, "get_device_tensors") as get_shards,
            patch.object(adapter_module.ttnn, "record_event") as record,
        ):
            self.assertIs(adapter.read_decode_output(logits), logits)
            output, events = adapter.read_decode_output(logits, async_read=True)
        self.assertIs(output, logits)
        self.assertEqual(events, [])
        self.assertEqual(adapter.generator.counters, {})
        get_shards.assert_not_called()
        record.assert_not_called()

    def test_direct_host_logits_processing_preserves_shape_values_and_dtype(self):
        adapter = self.adapter(True)
        logits = torch.randn(2, 1, 37, dtype=torch.float32)
        with (
            patch.object(adapter_module.ttnn, "is_tensor_storage_on_device") as storage,
            patch.object(adapter_module.ttnn, "to_torch") as convert,
        ):
            self.assertIs(adapter.process_decode_output_host(logits, is_tokens=False), logits)
        self.assertEqual(adapter.generator.counters, {})
        storage.assert_not_called()
        convert.assert_not_called()

    def test_host_logits_require_explicit_compatibility_at_both_boundaries(self):
        adapter = self.adapter(False)
        logits = torch.randn(2, 1, 37)
        with patch.object(adapter_module.ttnn, "get_device_tensors") as get_shards:
            for async_read in (False, True):
                with self.subTest(async_read=async_read):
                    with self.assertRaisesRegex(ValueError, "QWEN_VLLM_HOST_COMPATIBILITY=1"):
                        adapter.read_decode_output(logits, async_read=async_read)
            with self.assertRaisesRegex(ValueError, "QWEN_VLLM_HOST_COMPATIBILITY=1"):
                adapter.process_decode_output_host(logits, is_tokens=False)
        get_shards.assert_not_called()
        self.assertEqual(adapter.generator.counters, {})

    def test_device_output_cannot_be_misinterpreted_as_host_logits(self):
        for compatibility in (False, True):
            with self.subTest(compatibility=compatibility):
                adapter = self.adapter(compatibility)
                with self.assertRaisesRegex(ValueError, "tokens only"):
                    adapter.process_decode_output_host(object(), is_tokens=False)

    def test_native_async_read_keeps_one_replica_and_event_before_conversion(self):
        for compatibility in (False, True):
            with self.subTest(compatibility=compatibility):
                adapter = self.adapter(compatibility)
                raw, host, event = object(), object(), object()
                replicas = [Mock() for _ in range(4)]
                replicas[0].cpu.return_value = host
                padded_tokens = torch.tensor([71, 93] + [0] * 30, dtype=torch.int32)
                order = []

                def record(mesh, cq):
                    self.assertIs(mesh, adapter.generator.mesh)
                    self.assertEqual(cq, 0)
                    order.append("record")
                    return event

                def wait(read_event):
                    self.assertIs(read_event, event)
                    order.append("wait")

                def convert(output):
                    self.assertIs(output, host)
                    order.append("convert")
                    return padded_tokens

                controller = TTAsyncDecodeController(SimpleNamespace(model=adapter))
                with (
                    patch.object(adapter_module.ttnn, "get_device_tensors", return_value=replicas) as get_shards,
                    patch.object(adapter_module.ttnn, "record_event", side_effect=record),
                    patch.object(adapter_module.ttnn, "event_synchronize", side_effect=wait),
                    patch.object(adapter_module.ttnn, "is_tensor_storage_on_device", return_value=False),
                    patch.object(adapter_module.ttnn, "to_torch", side_effect=convert),
                ):
                    output, events = adapter.read_decode_output(raw, async_read=True)
                    submission = TTDecodeSubmission(
                        tt_out=output,
                        read_events=events,
                        batch_size_per_dp=[2],
                        sampling_params=SimpleNamespace(enable_log_probs=torch.zeros(2, dtype=torch.bool)),
                        perform_device_sampling=True,
                    )
                    finalized = controller.finalize_decode(submission)

                get_shards.assert_called_once_with(raw)
                replicas[0].cpu.assert_called_once_with(blocking=False)
                for replica in replicas[1:]:
                    replica.cpu.assert_not_called()
                self.assertEqual(padded_tokens.numel() * padded_tokens.element_size(), 128)
                self.assertEqual(events, [event])
                self.assertEqual(order, ["record", "wait", "convert"])
                self.assertEqual(adapter.generator.counters, {"token_readbacks": 1})
                self.assertTrue(torch.equal(finalized.tt_out, torch.tensor([[71], [93]], dtype=torch.int64)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
