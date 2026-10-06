# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import unittest

from models.demos.blackhole.qwen38_flash_next.tools.safetensors_metadata import (
    RangeResponseError,
    fetch_safetensors_header,
    read_exact_http_range,
)


class _FakeResponse:
    def __init__(self, payload: bytes, *, status: int, content_range: str | None):
        self._payload = payload
        self.status = status
        self.headers = {}
        if content_range is not None:
            self.headers["Content-Range"] = content_range

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self, size=-1):
        if size < 0:
            size = len(self._payload)
        result = self._payload[:size]
        self._payload = self._payload[size:]
        return result


class SafetensorsMetadataTest(unittest.TestCase):
    def test_exact_range_requires_partial_content_and_exact_bytes(self):
        def opener(request, timeout):
            self.assertEqual(request.headers["Range"], "bytes=8-11")
            self.assertEqual(timeout, 7)
            return _FakeResponse(b"abcd", status=206, content_range="bytes 8-11/99")

        self.assertEqual(read_exact_http_range("https://example.test/shard", 8, 11, timeout=7, opener=opener), b"abcd")

    def test_exact_range_rejects_server_that_ignores_range(self):
        def opener(request, timeout):
            return _FakeResponse(b"whole checkpoint", status=200, content_range=None)

        with self.assertRaisesRegex(RangeResponseError, "HTTP 206"):
            read_exact_http_range("https://example.test/shard", 0, 7, opener=opener)

    def test_fetch_header_decodes_tensor_metadata(self):
        header = json.dumps(
            {
                "__metadata__": {"format": "pt"},
                "model.layers.0.weight": {
                    "dtype": "BF16",
                    "shape": [32, 64],
                    "data_offsets": [0, 4096],
                },
            },
            separators=(",", ":"),
        ).encode("utf-8")
        prefix = len(header).to_bytes(8, "little")
        expected_ranges = {
            "bytes=0-7": (prefix, "bytes 0-7/9999"),
            f"bytes=8-{7 + len(header)}": (header, f"bytes 8-{7 + len(header)}/9999"),
        }

        def opener(request, timeout):
            payload, content_range = expected_ranges[request.headers["Range"]]
            return _FakeResponse(payload, status=206, content_range=content_range)

        result = fetch_safetensors_header("https://example.test/shard", opener=opener)

        self.assertEqual(result["header_length"], len(header))
        self.assertEqual(result["metadata"], {"format": "pt"})
        self.assertEqual(result["tensors"]["model.layers.0.weight"]["shape"], [32, 64])
        self.assertEqual(result["tensors"]["model.layers.0.weight"]["dtype"], "BF16")

    def test_fetch_header_rejects_unbounded_header_length(self):
        prefix = (64 * 1024 * 1024).to_bytes(8, "little")

        def opener(request, timeout):
            return _FakeResponse(prefix, status=206, content_range="bytes 0-7/999999999")

        with self.assertRaisesRegex(RangeResponseError, "header length"):
            fetch_safetensors_header("https://example.test/shard", max_header_bytes=1024, opener=opener)


if __name__ == "__main__":
    unittest.main()
