# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression for GPT-OSS table ownership consumed by native KV Manager."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import ttnn
from models.demos.gpt_oss_d_p.tt.runners.kv_chunk_table import build_and_serialize_kv_chunk_table


def _export_table(path, hostname):
    # Replace only device allocation/topology queries; build and serialize the real
    # table so the asserted host is the value a separate native process consumes.
    def tensor(base):
        return SimpleNamespace(shape=(2, 1, 256, 64), dtype=ttnn.bfloat8_b, buffer_address=lambda: base)

    mesh = SimpleNamespace(get_fabric_node_id=lambda coord: ttnn.FabricNodeId(ttnn.MeshId(0), coord[0] * 8 + coord[1]))
    with (
        patch("socket.gethostname", return_value=hostname),
        patch("models.demos.gpt_oss_d_p.tt.runners.kv_chunk_table.get_num_dram_banks", return_value=8),
    ):
        build_and_serialize_kv_chunk_table(
            mesh_device=mesh,
            kv_cache=SimpleNamespace(k=tensor(0x100000), v=tensor(0x200000)),
            seq_len=1024,
            num_layers=1,
            mesh_shape=(4, 8),
            sp_axis=0,
            num_users=2,
            chunk_size=1024,
            num_kv_heads=8,
            head_dim=64,
            path=str(path),
        )


class KvTableHostContractTests(unittest.TestCase):
    def test_exported_owner_matches_native_masked_crc32_identity(self):
        # Fixed native contract vectors cover leading-zero formatting and masking
        # the high CRC bit (host-test-2 has CRC32 0xedccd231).
        for hostname, expected in (("host-test-0", "host-03c2b31d"), ("host-test-2", "host-6dccd231")):
            with self.subTest(hostname=hostname), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "kv_table.pb"
                _export_table(path, hostname)
                table = ttnn.experimental.disaggregation.import_from_protobuf_file(str(path))
                owners = {table.get_host(ttnn.FabricNodeId(ttnn.MeshId(0), chip)) for chip in range(32)}
                self.assertEqual(owners, {expected})


if __name__ == "__main__":
    unittest.main()
