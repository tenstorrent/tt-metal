# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Executed host-boundary sentinels around every captured graph and resident loop."""
from contextlib import contextmanager
from unittest.mock import patch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests import optimized_full_path as probe
from models.autoports.qwen_qwen3_6_27b.tt.generator import QwenGenerator


@contextmanager
def no_host_boundary():
    names = ("from_torch", "to_torch", "as_tensor", "to_device", "copy_host_to_device_tensor", "synchronize_device")
    patches = []

    def forbidden(*args, **kwargs):
        raise AssertionError("Host boundary executed in the captured/resident device path")

    try:
        for name in names:
            item = patch.object(ttnn, name, forbidden)
            item.start()
            patches.append(item)
        yield
    finally:
        for item in reversed(patches):
            item.stop()


original_capture = QwenGenerator._capture
original_decode = QwenGenerator.decode_forward
captured = []
resident_calls = 0


def capture(self, key, function):
    def guarded():
        with no_host_boundary():
            function()
        captured.append(key)

    return original_capture(self, key, guarded)


def decode(self, *args, **kwargs):
    global resident_calls
    if not kwargs.get("reload_inputs", True) and not kwargs.get("read_from_device", True):
        with no_host_boundary():
            result = original_decode(self, *args, **kwargs)
        resident_calls += 1
        return result
    return original_decode(self, *args, **kwargs)


if __name__ == "__main__":
    with no_host_boundary():
        try:
            ttnn.to_torch(None)
        except AssertionError:
            print("SENTINEL_POSITIVE_CONTROL_PASS", flush=True)
        else:
            raise AssertionError("Sentinel was inactive")
    with patch.object(QwenGenerator, "_capture", capture), patch.object(QwenGenerator, "decode_forward", decode):
        probe.main()
    assert set(captured) == {"decode", "sample", "sample_penalties", "reset", "prefill_128"}, captured
    assert resident_calls > 0
    print("RUNTIME_BOUNDARY_AUDIT_PASS", captured, resident_calls, flush=True)
