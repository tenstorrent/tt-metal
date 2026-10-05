# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression for the common runner's GPT-OSS request call contract."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from models.demos.gpt_oss_d_p.tt.tt_prefill_runtime import TtPrefillRuntime


class PrefillRuntimeContractTests(unittest.TestCase):
    def setUp(self):
        sync = patch("models.demos.gpt_oss_d_p.tt.tt_prefill_runtime.ttnn.synchronize_device")
        self.synchronize = sync.start()
        self.addCleanup(sync.stop)

    def runtime(self):
        runtime = TtPrefillRuntime.__new__(TtPrefillRuntime)
        runtime.config = SimpleNamespace(
            default_chunk_size=1024,
            max_seq_len=2048,
            num_users=2,
            is_first_rank=True,
            is_last_rank=True,
            first_layer_idx=0,
        )
        runtime.model_built = True
        runtime.mesh_device = object()
        runtime.rope_indexed = {1024: object()}
        runtime._slot_chunk_size = {1: 1024}
        runtime._resolve_kv = Mock(return_value=object())
        runtime._embed_tokens = Mock(return_value=object())
        runtime._layer_completion_sink = Mock()
        runtime.model = Mock()

        def forward(_input, **kwargs):
            kwargs["on_layer_complete"](0)
            kwargs["on_layer_complete"](35)
            return None

        runtime.model.prefill_forward.side_effect = forward
        return runtime

    @patch("models.demos.gpt_oss_d_p.tt.tt_prefill_runtime.ttnn.deallocate")
    def test_socket_metadata_preserves_slot_position_and_ack_identity(self, deallocate):
        runtime = self.runtime()
        # The common runner already decoded all three metadata words. Its raw device
        # tensor is also passed for runtimes that trace; eager GPT uses these scalars.
        for slot, start, end, request in ((0, 0, 992, 7), (1, 1024, 2016, 11)):
            with self.subTest(slot=slot):
                runtime.prefill_chunk(
                    object(),
                    object(),
                    slot_id=slot,
                    actual_start=start,
                    actual_end=end,
                    request_id=request,
                    d2h_service=None,
                    metadata_msg=object(),
                )
                call = runtime.model.prefill_forward.call_args.kwargs
                self.assertEqual(call["user_id"], slot)
                self.assertEqual(call["cached_len"], start)
                self.assertTrue(call["skip_lm_head"])
        self.assertEqual(
            runtime._layer_completion_sink.call_args_list,
            [
                unittest.mock.call(0, 7),
                unittest.mock.call(35, 7),
                unittest.mock.call(0, 11),
                unittest.mock.call(35, 11),
            ],
        )

    @patch("models.demos.gpt_oss_d_p.tt.tt_prefill_runtime.ttnn.deallocate")
    def test_host_ack_waits_for_queued_device_kv_writes(self, deallocate):
        runtime = self.runtime()
        queued, committed, acks = set(), set(), []

        def finish_device_work(mesh_device):
            self.assertIs(mesh_device, runtime.mesh_device)
            committed.update(queued)
            queued.clear()

        def publish(layer, request):
            self.assertIn(layer, committed, "ACK published before device KV write completed")
            acks.append((layer, request))

        def forward(_input, **kwargs):
            for layer in (0, 1):
                queued.add(layer)
                kwargs["on_layer_complete"](layer)
            return None

        self.synchronize.side_effect = finish_device_work
        runtime._layer_completion_sink = publish
        runtime.model.prefill_forward.side_effect = forward
        runtime.prefill_chunk(
            object(),
            slot_id=1,
            actual_start=0,
            actual_end=1024,
            request_id=13,
            metadata_msg=object(),
        )
        self.assertEqual(acks, [(0, 13), (1, 13)])
        self.assertEqual(queued, set())

    @patch("models.demos.gpt_oss_d_p.tt.tt_prefill_runtime.ttnn.deallocate")
    def test_failed_device_completion_cannot_publish_ack(self, deallocate):
        runtime = self.runtime()
        self.synchronize.side_effect = RuntimeError("device completion failed")
        with self.assertRaisesRegex(RuntimeError, "device completion failed"):
            runtime.prefill_chunk(
                object(),
                slot_id=0,
                actual_start=0,
                actual_end=1024,
                request_id=7,
                metadata_msg=object(),
            )
        runtime._layer_completion_sink.assert_not_called()

    def test_socket_metadata_does_not_bypass_position_validation(self):
        runtime = self.runtime()
        with self.assertRaisesRegex(AssertionError, "not within one chunk"):
            runtime.prefill_chunk(
                object(),
                slot_id=0,
                actual_start=0,
                actual_end=1025,
                metadata_msg=object(),
            )
        runtime.model.prefill_forward.assert_not_called()

    def test_socket_metadata_does_not_silently_enable_device_acks(self):
        runtime = self.runtime()
        with self.assertRaisesRegex(NotImplementedError, "host callback"):
            runtime.prefill_chunk(
                object(),
                slot_id=0,
                actual_start=0,
                actual_end=1024,
                metadata_msg=object(),
                d2h_service=object(),
            )
        runtime.model.prefill_forward.assert_not_called()


if __name__ == "__main__":
    unittest.main()
