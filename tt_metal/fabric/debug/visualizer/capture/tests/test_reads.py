# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Hand-written cases for what capture reads. Stream register addresses are worked out by hand.

Run from the repository root: python3 -m unittest tt_metal.fabric.debug.visualizer.capture.tests.test_reads
"""

import copy
import unittest

from tt_metal.fabric.debug.visualizer.capture.reads import (
    L1Read,
    ManifestError,
    RegisterRead,
    get_read_plan,
)

L1 = {"category": "control_info", "kind": "l1", "schema": "u32", "cleared_by_host": False}
STREAM = {"category": "flow_control", "kind": "stream", "schema": "u32"}

# The T3K 2D mock manifest's M0/C0/E1, cut down. Sender ch1 is unused, and its credit is an {array, index} reference,
# as on Blackhole. vc2_receiver_free_slots is set, as with VC2 on, so stream 31 has two registers.
ROUTER = {
    "identity": {"eth_chan": 0, "logical_core": [0, 0], "virtual_core": [25, 16]},
    "link": {"edge_capability": "intramesh_cardinal", "peer": "M0/C1/W1", "cross_host": False, "wrap": False},
    "shape": {"num_vcs": 1, "senders_per_vc": [2, 0, 0], "receivers_per_vc": [1, 0, 0], "num_active_eriscs": 1},
    "credit_counters": {
        "to_sender_ack": {
            "address": 0x18020,
            "size": 48,
            "num_elements": 12,
            "size_per_element": 4,
            "schema": "u32",
            "cleared_by_host": False,
            "index_space": "own_sender_compact",
        },
    },
    "channels": {
        "senders": {
            "vc0": {
                "ch0": {
                    "status": "active",
                    "serviced_by": ["erisc0"],
                    "producer": "worker",
                    "ring_buffer": {"address": 0x187A0, "size": 62272, "num_elements": 14, "schema": "bytes"},
                    "credits": {"completed": {"stream_id": 9, "register": "buf_space_available", "schema": "u32"}},
                    "fields": {
                        "is_injection_channel": {"category": "flow_control", "kind": "flag", "value": False},
                        "free_slots": {**STREAM, "stream_id": 0, "register": "buf_space_available"},
                        "producer_credit_return_cmd_buf": {"kind": "enum", "value": 3, "schema": "enum:NocCmdBuf"},
                        "connection": {**L1, "address": 0x181C0, "size": 4},
                    },
                },
                "ch1": {
                    "status": "unused",
                    "serviced_by": [],
                    "producer": "worker",
                    "ring_buffer": None,
                    "credits": {"completed": {"array": "credit_counters/to_sender_ack", "index": 1}},
                    "fields": {"free_slots": None, "connection": None},
                },
            },
        },
        "receivers": {
            "vc0": {
                "ch0": {
                    "status": "active",
                    "serviced_by": ["erisc0"],
                    "forwards_on": "vc0",
                    "ring_buffer": {"address": 0x34B60, "size": 35584, "num_elements": 8, "schema": "bytes"},
                    "fields": {
                        "forward_noc": {"category": "kernel_params", "kind": "number", "value": 1},
                        "pkts_sent": {**STREAM, "stream_id": 4, "register": "buf_space_available"},
                    },
                },
            },
        },
    },
    "intra_chip_downstream_edges": {
        "vc0": {
            "edge3": {
                "downstream_channel": "M0/C0/S1/channels/senders/vc0/ch1",
                "through_tensix_mux": False,
                "fields": {
                    "free_slots": {**STREAM, "stream_id": 15, "register": "buf_space_available"},
                    "teardown_sem": {**L1, "address": 0x186F0, "size": 4},
                },
            },
        },
    },
    "fields": {
        "edm_status": {**L1, "address": 0x18140, "size": 4, "schema": "enum:EDMStatus"},
        "erisc_sync": {**STREAM, "stream_id": 31, "register": "remote_src"},
        "vc2_receiver_free_slots": {**STREAM, "stream_id": 31, "register": "buf_space_available"},
        "context_switch_interval": {"category": "kernel_params", "kind": "number", "value": 16},
        "handshake_sender": {"category": "lifecycle", "kind": "flag", "value": True},
        "my_direction": {"kind": "enum", "value": 0, "schema": "enum:eth_chan_directions"},
        "perf_telemetry": None,
    },
    "eriscs": {
        "erisc0": {
            "processor": "riscv_0",
            "physical_erisc": 0,
            "fields": {"teardown_check_iterations": {"category": "kernel_params", "kind": "number", "value": 32}},
        },
    },
    "leftover_l1": {"address": 0x3D660, "size": 1440, "schema": "bytes", "cleared_by_host": False},
}

# A router with nothing to read but its leftover L1.
BARE_ROUTER = {
    "identity": {"eth_chan": 6, "logical_core": [0, 6], "virtual_core": [22, 16]},
    "leftover_l1": {"address": 0x3D660, "size": 1440, "schema": "bytes", "cleared_by_host": False},
}

MANIFEST = {
    "manifest_version": 1,
    "kind": "fabric_manifest",
    "run": {"arch": "wormhole_b0"},
    "archs": {
        "wormhole_b0": {
            "areas": {
                "heartbeat": {
                    "address": 0x1F80,
                    "size": 4,
                    "schema": "u32",
                    "cleared_by_host": False,
                    "magic": 0xDCBA0000,
                    "magic_mask": 0xFFFF0000,
                    "period_iters": 64,
                },
                "routing_table": {"address": 0x3F440, "size": 2704, "schema": "struct:routing_l1_info_t"},
                "eth_fw_mailbox": None,
            },
            "stream_registers": {
                "base": 0xFFB40000,
                "stride": 0x1000,
                "count": 32,
                "register_offsets": {"buf_space_available": 0x100, "remote_src": 0},
            },
            # Not in the order the manifest writes them, to show that capture keeps the array's order.
            "core_registers": [
                {"name": "wall_clock_low", "address": 0xFFB121F0, "size": 4, "schema": "u32"},
                {"name": "soft_reset", "address": 0xFFB121B0, "size": 4, "schema": "u32", "reset_masks": [0x800]},
                {"name": "wall_clock_high", "address": 0xFFB121F8, "size": 4, "schema": "u32"},
            ],
            "types": {},
        },
    },
    "meshes": {
        "M0": {
            "chips": {
                "C0": {"asic_id": "0x000000036190302b", "is_local": True, "routers": {"E1": ROUTER}},
                "C1": {"mesh_coord": [0, 1], "physical_chip_id": None, "asic_id": None, "is_local": False},
            },
        },
        "M1": {
            "chips": {
                "C0": {"asic_id": "0x00000003619030a1", "is_local": True, "routers": {"S1": BARE_ROUTER}},
            },
        },
    },
}

REGIONS = {
    "credit_counters/to_sender_ack": L1Read(0x18020, 48),
    "channels/senders/vc0/ch0/ring_buffer": L1Read(0x187A0, 62272),
    "channels/senders/vc0/ch0/fields/connection": L1Read(0x181C0, 4),
    "channels/receivers/vc0/ch0/ring_buffer": L1Read(0x34B60, 35584),
    "intra_chip_downstream_edges/vc0/edge3/fields/teardown_sem": L1Read(0x186F0, 4),
    "fields/edm_status": L1Read(0x18140, 4),
    "leftover_l1": L1Read(0x3D660, 1440),
}

# base + stream_id * stride + register_offsets[register]
STREAMS = {
    (0, "buf_space_available"): RegisterRead(0xFFB40100),
    (4, "buf_space_available"): RegisterRead(0xFFB44100),
    (9, "buf_space_available"): RegisterRead(0xFFB49100),
    (15, "buf_space_available"): RegisterRead(0xFFB4F100),
    (31, "buf_space_available"): RegisterRead(0xFFB5F100),
    (31, "remote_src"): RegisterRead(0xFFB5F000),
}

E1 = "meshes.M0.chips.C0.routers.E1"


def router_in(manifest: dict) -> dict:
    return manifest["meshes"]["M0"]["chips"]["C0"]["routers"]["E1"]


class Reads(unittest.TestCase):
    def setUp(self):
        self.plan = get_read_plan(MANIFEST)

    def test_routers_are_those_on_local_chips_in_manifest_order(self):
        self.assertEqual([router.path for router in self.plan.routers], ["M0/C0/E1", "M1/C0/S1"])

    def test_router_identity(self):
        identities = [(r.asic_id, r.eth_chan, r.virtual_core, r.logical_core) for r in self.plan.routers]
        self.assertEqual(identities, [(0x36190302B, 0, (25, 16), (0, 0)), (0x3619030A1, 6, (22, 16), (0, 6))])

    def test_regions_are_every_object_with_an_address_by_path(self):
        # Equality also shows what is skipped: credit references, nulls, number, flag and enum fields, and identity.
        regions = self.plan.routers[0].regions
        self.assertEqual(regions, REGIONS)
        self.assertEqual(list(regions), list(REGIONS), "regions aren't in manifest order")
        self.assertEqual(self.plan.routers[1].regions, {"leftover_l1": L1Read(0x3D660, 1440)})

    def test_streams_are_each_id_and_register_by_id_then_register(self):
        streams = self.plan.routers[0].streams
        self.assertEqual(streams, STREAMS)
        self.assertEqual(list(streams), list(STREAMS), "streams aren't by id and then register")
        self.assertEqual(self.plan.routers[1].streams, {})

    def test_a_region_in_an_array_is_named_by_its_index(self):
        manifest = copy.deepcopy(MANIFEST)
        router_in(manifest)["shape"]["extra"] = [None, {"address": 0x100, "size": 4}]
        self.assertEqual(get_read_plan(manifest).routers[0].regions["shape/extra/1"], L1Read(0x100, 4))

    def test_a_stream_named_twice_is_read_once(self):
        manifest = copy.deepcopy(MANIFEST)
        fields = router_in(manifest)["fields"]
        fields["erisc_sync_again"] = dict(fields["erisc_sync"])
        self.assertEqual(get_read_plan(manifest).routers[0].streams, STREAMS)

    def test_areas_are_the_non_null_ones(self):
        self.assertEqual(self.plan.areas, {"heartbeat": L1Read(0x1F80, 4), "routing_table": L1Read(0x3F440, 2704)})

    def test_core_registers_keep_the_arrays_order(self):
        self.assertEqual(
            self.plan.core_registers,
            {
                "wall_clock_low": RegisterRead(0xFFB121F0),
                "soft_reset": RegisterRead(0xFFB121B0),
                "wall_clock_high": RegisterRead(0xFFB121F8),
            },
        )
        self.assertEqual(list(self.plan.core_registers), ["wall_clock_low", "soft_reset", "wall_clock_high"])

    def test_arch(self):
        self.assertEqual(self.plan.arch, "wormhole_b0")


class BadManifests(unittest.TestCase):
    def test_a_stream_id_outside_the_stream_registers_is_rejected(self):
        for stream_id in (32, -1):
            manifest = copy.deepcopy(MANIFEST)
            router_in(manifest)["fields"]["erisc_sync"]["stream_id"] = stream_id
            named = rf"{E1}\.fields\.erisc_sync\.stream_id is {stream_id}; .*count is 32"
            with self.subTest(stream_id), self.assertRaisesRegex(ManifestError, named):
                get_read_plan(manifest)

    def test_an_unknown_register_is_rejected(self):
        manifest = copy.deepcopy(MANIFEST)
        router_in(manifest)["fields"]["erisc_sync"]["register"] = "remote_dest"
        with self.assertRaisesRegex(ManifestError, rf"{E1}\.fields\.erisc_sync\.register is 'remote_dest'"):
            get_read_plan(manifest)

    def test_an_address_without_a_size_is_rejected(self):
        manifest = copy.deepcopy(MANIFEST)
        del router_in(manifest)["leftover_l1"]["size"]
        with self.assertRaisesRegex(ManifestError, rf"manifest has no {E1}\.leftover_l1\.size"):
            get_read_plan(manifest)

    def test_an_address_that_isnt_an_integer_is_rejected(self):
        manifest = copy.deepcopy(MANIFEST)
        router_in(manifest)["fields"]["edm_status"]["address"] = "0x18140"
        with self.assertRaisesRegex(ManifestError, rf"{E1}\.fields\.edm_status\.address is a string"):
            get_read_plan(manifest)

    def test_a_file_that_isnt_a_fabric_manifest_is_rejected(self):
        manifest = copy.deepcopy(MANIFEST)
        manifest["kind"] = "fabric_snapshot"
        with self.assertRaisesRegex(ManifestError, "kind is 'fabric_snapshot'"):
            get_read_plan(manifest)

    def test_an_arch_without_a_block_is_rejected(self):
        manifest = copy.deepcopy(MANIFEST)
        manifest["run"]["arch"] = "blackhole"
        with self.assertRaisesRegex(ManifestError, r"manifest has no archs\.blackhole"):
            get_read_plan(manifest)

    def test_a_local_chip_without_an_asic_id_is_rejected(self):
        for asic_id, error in ((None, "is null"), ("chip0", "is 'chip0'")):
            manifest = copy.deepcopy(MANIFEST)
            manifest["meshes"]["M0"]["chips"]["C0"]["asic_id"] = asic_id
            named = rf"meshes\.M0\.chips\.C0\.asic_id {error}"
            with self.subTest(asic_id), self.assertRaisesRegex(ManifestError, named):
                get_read_plan(manifest)

    def test_two_core_registers_with_one_name_are_rejected(self):
        manifest = copy.deepcopy(MANIFEST)
        manifest["archs"]["wormhole_b0"]["core_registers"][2]["name"] = "wall_clock_low"
        with self.assertRaisesRegex(ManifestError, r"core_registers\[2\]\.name is 'wall_clock_low'"):
            get_read_plan(manifest)

    def test_a_core_register_that_isnt_one_word_is_rejected(self):
        for size in (2, 8):
            manifest = copy.deepcopy(MANIFEST)
            manifest["archs"]["wormhole_b0"]["core_registers"][1]["size"] = size
            named = rf"archs\.wormhole_b0\.core_registers\[1\]\.size is {size}; .*one 4-byte word"
            with self.subTest(size), self.assertRaisesRegex(ManifestError, named):
                get_read_plan(manifest)


if __name__ == "__main__":
    unittest.main()
