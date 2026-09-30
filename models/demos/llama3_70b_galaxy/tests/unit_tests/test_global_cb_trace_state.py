# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

from models.demos.llama3_70b_galaxy.tt.global_cb_trace import GlobalCBTraceState


def make_global_cb():
    sender = SimpleNamespace(x=0, y=0)
    receiver = SimpleNamespace(start=SimpleNamespace(x=1, y=0), end=SimpleNamespace(x=2, y=0))
    return Mock(
        is_suspended=Mock(return_value=False),
        buffer_address=Mock(return_value=1000),
        config_address=Mock(return_value=900),
        size=Mock(return_value=3200),
        buffer_type=Mock(return_value="L1"),
        sender_core_type=Mock(return_value="worker"),
        sender_receiver_core_mapping=Mock(return_value=[(sender, Mock(ranges=Mock(return_value=[receiver])))]),
    )


class TestGlobalCBTraceState(unittest.TestCase):
    def test_preparation_reserves_before_the_first_trace(self):
        state = GlobalCBTraceState()
        global_cb = make_global_cb()
        state.prepare(global_cb)
        self.assertIs(state.global_cb, global_cb)
        state.restore(global_cb)
        global_cb.acknowledge_restored_trace.assert_not_called()
        changed = make_global_cb()
        changed.buffer_address.return_value += 32
        with self.assertRaisesRegex(RuntimeError, "layout changed"):
            state.record_traces("decode", (10,), changed)

    def test_capture_keeps_the_reservation_alive(self):
        state = GlobalCBTraceState()
        global_cb = make_global_cb()
        reference = weakref.ref(global_cb)
        state.record_traces("decode", (10,), global_cb)
        del global_cb
        gc.collect()
        self.assertIsNotNone(reference())
        state.restore(state.global_cb)

    def test_suspended_gcb_cannot_be_used_by_decode(self):
        state = GlobalCBTraceState()
        global_cb = make_global_cb()
        state.record_traces("decode", (10,), global_cb)
        global_cb.is_suspended.return_value = True
        with self.assertRaisesRegex(RuntimeError, "requires a live global circular buffer"):
            state.restore(global_cb)
        global_cb.acknowledge_restored_trace.assert_not_called()

    def test_reconstruction_acknowledges_only_registered_traces(self):
        state = GlobalCBTraceState()
        original = make_global_cb()
        state.record_traces(("decode", False), (10,), original)
        state.record_traces(("decode", True), (11,), original)
        state.record_traces("sampling", (12, 13), original)
        # A sampling recapture replaces released IDs instead of accumulating them.
        state.record_traces("sampling", (14,), original)
        for _ in range(5):
            restored = make_global_cb()
            state.restore(restored)
            self.assertEqual(
                [call.args[0] for call in restored.acknowledge_restored_trace.call_args_list], [10, 11, 14]
            )

    def test_each_layout_change_fails_before_any_acknowledgement(self):
        for field, value in (
            ("buffer_address", 1001),
            ("config_address", 901),
            ("size", 6400),
            ("buffer_type", "L1_SMALL"),
            ("sender_core_type", "dram"),
            ("sender_receiver_core_mapping", []),
        ):
            with self.subTest(field=field):
                state = GlobalCBTraceState()
                state.record_traces("decode", (10,), make_global_cb())
                changed = make_global_cb()
                getattr(changed, field).return_value = value
                with self.assertRaisesRegex(RuntimeError, "layout changed after trace capture"):
                    state.restore(changed)
                changed.acknowledge_restored_trace.assert_not_called()
                # A failed reconstruction must not change the saved layout.
                state.restore(make_global_cb())

    def test_another_trace_cannot_replace_the_capture_layout(self):
        state = GlobalCBTraceState()
        state.record_traces("decode", (10,), make_global_cb())
        changed = make_global_cb()
        changed.config_address.return_value += 32
        with self.assertRaisesRegex(RuntimeError, "layout changed after trace capture"):
            state.record_traces("sampling", (11,), changed)
        restored = make_global_cb()
        state.restore(restored)
        restored.acknowledge_restored_trace.assert_called_once_with(10)

    def test_missing_gcb_is_rejected_before_replay(self):
        state = GlobalCBTraceState()
        with self.assertRaisesRegex(RuntimeError, "requires a live global circular buffer"):
            state.validate(None)

    def test_no_traces_means_no_exemption(self):
        state = GlobalCBTraceState()
        restored = make_global_cb()
        state.restore(restored)
        restored.acknowledge_restored_trace.assert_not_called()
        state.record_traces("sampling", (12,), restored)
        state.record_traces("sampling", (), restored)
        state.restore(restored)
        restored.acknowledge_restored_trace.assert_not_called()


if __name__ == "__main__":
    unittest.main()
