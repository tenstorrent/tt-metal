# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Table population tests use a recording backend, not native protobuf or device I/O."""

import importlib
import unittest
from types import SimpleNamespace

from models.demos.llama_3p1_8b_d_p.tt.runners.kv_layout import PrefillKVLayout


class Table:
    def __init__(self, configs):
        self.configs = dict(sorted(configs.items()))
        self.groups, self.hosts, self.entries = [], {}, {}

    def num_configs(self):
        return len(self.configs)

    def config_name(self, index):
        return list(self.configs)[index]

    def add_device_group(self, nodes):
        self.groups.append(nodes)
        return len(self.groups) - 1

    def set_fabric_node_host(self, node, *, host_name):
        self.hosts[node] = host_name

    def set(self, layer, position, slot, location, config):
        key = (config, layer, position, slot)
        if key in self.entries:
            raise AssertionError("duplicate logical table entry")
        self.entries[key] = location


class TableExportTests(unittest.TestCase):
    # Native manager IDs and names must preserve K0..K7,V0..V7 over a lexically ordered table.
    def test_populate_all_configs_devices_and_slots(self):
        try:
            module = importlib.import_module("models.demos.llama_3p1_8b_d_p.tt.runners.kv_chunk_table")
        except ModuleNotFoundError:
            self.fail("table population has not been implemented")
        layout = PrefillKVLayout(num_banks=8)
        nodes = {(row, col): (11, row * 8 + col) for row in range(4) for col in range(8)}
        api = SimpleNamespace(
            KvChunkAddressTable=Table, KvChunkAddressTableConfig=SimpleNamespace, KvCacheLocation=SimpleNamespace
        )
        table = module.build_address_table(
            layout=layout, base_addresses=(65536, 1048576), fabric_nodes=nodes, host_name="owner", api=api
        )
        self.assertEqual(list(table.configs), list(layout.config_names))
        self.assertEqual(len(table.entries), 65536)
        self.assertEqual(len(table.hosts), 32)
        for index, name in enumerate(layout.config_names):
            cfg = table.configs[name]
            self.assertEqual(
                (cfg.num_layers, cfg.max_sequence_length, cfg.num_slots, cfg.chunk_n_tokens, cfg.chunk_size_bytes),
                (32, 2048, 2, 32, 4352),
            )
            for slot in (0, 1):
                loc = table.entries[index, 31, 2016, slot]
                coord, bank, offset = layout.locate(index, 31, 2016, slot, (65536, 1048576)[index // 8])
                self.assertEqual(table.groups[loc.device_group_index], [nodes[coord]])
                self.assertEqual(loc.noc_addr, (bank << 32) | offset)
                self.assertEqual(loc.size_bytes, 4352)

    # Missing chip ownership must fail before constructing a partial table.
    def test_incomplete_device_map_rejected(self):
        try:
            module = importlib.import_module("models.demos.llama_3p1_8b_d_p.tt.runners.kv_chunk_table")
        except ModuleNotFoundError:
            self.fail("table population has not been implemented")
        with self.assertRaises(ValueError):
            module.build_address_table(
                layout=PrefillKVLayout(num_banks=8),
                base_addresses=(65536, 1048576),
                fabric_nodes={},
                host_name="owner",
                api=None,
            )
