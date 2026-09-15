# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tt_metal.fabric.debug.visualizer.capture.manifest import load_manifest
from tt_metal.fabric.debug.visualizer.capture.peek import peek_manifest, read_block
from tt_metal.fabric.debug.visualizer.capture.rawfile import RawBlobWriter
from tt_metal.fabric.debug.visualizer.capture.tests.test_peek import (
    FakeContext,
    FakeDevice,
    FakeLoc,
    fixture_manifest,
)


class RawBlobWriterTest(unittest.TestCase):
    def test_round_trip_offsets_and_digests(self):
        blobs = [b"aaa", b"bbbb", b"cc"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "raw.bin"
            writer = RawBlobWriter(path)
            records = [writer.add(blob) for blob in blobs]
            size, digest = writer.close()

            payload = path.read_bytes()
            self.assertEqual(payload, b"".join(blobs))
            self.assertEqual(size, len(payload))
            self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
            offset = 0
            for blob, record in zip(blobs, records):
                blob_offset, blob_size, blob_digest = record
                self.assertEqual(blob_offset, offset)
                self.assertEqual(blob_size, len(blob))
                self.assertEqual(blob_digest, hashlib.sha256(blob).hexdigest())
                self.assertEqual(payload[blob_offset : blob_offset + blob_size], blob)
                offset += blob_size

    def test_abort_drops_the_temporary_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "raw.bin"
            writer = RawBlobWriter(path)
            writer.add(b"partial")
            writer.abort()
            self.assertFalse(path.exists())
            self.assertFalse(writer._temporary_path.exists())


class BlobCaptureTest(unittest.TestCase):
    def load_fixture(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            path.write_text(json.dumps(fixture_manifest()), encoding="utf-8")
            return load_manifest(path)

    def test_blob_offsets_match_sidecar_bytes(self):
        manifest = self.load_fixture()
        devices = {4: FakeDevice(4, [FakeLoc((0, 0)), FakeLoc((0, 1))])}

        def read_u32(device, loc, address):
            if address == 1:
                return 0xA3B3C3D3
            return 0

        def read_bytes(device, loc, address, size):
            return bytes((address + index) % 256 for index in range(size))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "raw.bin"
            writer = RawBlobWriter(path)
            sample = peek_manifest(
                manifest,
                FakeContext(devices),
                read_u32,
                blob_writer=writer,
                read_bytes=read_bytes,
            )
            writer.close()
            payload = path.read_bytes()
            unreserved = sample["routers"][0]["blobs"]["unreserved"]
            self.assertEqual(unreserved["status"], "ok")
            self.assertEqual(unreserved["size"], 1024)
            slice_bytes = payload[unreserved["offset"] : unreserved["offset"] + unreserved["size"]]
            self.assertEqual(slice_bytes, read_bytes(None, None, 0, 1024))
            self.assertEqual(unreserved["sha256"], hashlib.sha256(slice_bytes).hexdigest())
            go_msg = sample["routers"][0]["blobs"]["go_msg"]
            go_bytes = payload[go_msg["offset"] : go_msg["offset"] + go_msg["size"]]
            self.assertEqual(go_bytes, read_bytes(None, None, 4000, 36))
            routing = read_bytes(None, None, 3000, 64)
            self.assertEqual(
                sample["routers"][0]["identity"],
                {
                    "my_mesh_id": int.from_bytes(routing[32:34], "little"),
                    "my_device_id": int.from_bytes(routing[34:36], "little"),
                    "matches_manifest": False,
                },
            )

    def test_chunked_read_reassembles_and_zero_fills_failed_middle(self):
        reads = []

        def read_bytes(device, loc, address, size):
            reads.append((address, size))
            if address == 16:
                return None
            return bytes(range(size))

        payload, status, error = read_block(None, None, 0, 40, 16, read_bytes)
        self.assertEqual(status, "unreadable")
        self.assertIn("chunk at +16", error)
        self.assertEqual(reads, [(0, 16), (16, 16), (32, 8)])
        self.assertEqual(payload[:16], bytes(range(16)))
        self.assertEqual(payload[16:32], bytes(16))
        self.assertEqual(payload[32:], bytes(range(8)))

    def test_no_l1_image_skips_unreserved_only(self):
        manifest = self.load_fixture()
        devices = {4: FakeDevice(4, [FakeLoc((0, 0)), FakeLoc((0, 1))])}

        def read_u32(device, loc, address):
            if address == 1:
                return 0xA3B3C3D3
            return 0

        def read_bytes(device, loc, address, size):
            return bytes((address + index) % 256 for index in range(size))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "raw.bin"
            writer = RawBlobWriter(path)
            sample = peek_manifest(
                manifest,
                FakeContext(devices),
                read_u32,
                blob_writer=writer,
                read_bytes=read_bytes,
                include_unreserved=False,
            )
            writer.close()
            blobs = sample["routers"][0]["blobs"]
            self.assertEqual(blobs["unreserved"]["status"], "unsupported")
            self.assertIn("skipped", blobs["unreserved"]["error"])
            self.assertIsNone(blobs["unreserved"]["offset"])
            self.assertEqual(blobs["go_msg"]["status"], "ok")
            self.assertEqual(
                path.stat().st_size,
                sum(
                    blobs[name]["size"]
                    for name in ("fabric_telemetry", "routing_table", "go_msg", "launch")
                ),
            )


if __name__ == "__main__":
    unittest.main()
