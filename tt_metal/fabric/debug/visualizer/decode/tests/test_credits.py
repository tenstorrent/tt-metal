# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

from tt_metal.fabric.debug.visualizer.decode.credits import annotate_links, decode_channels, stall_score
from tt_metal.fabric.debug.visualizer.decode.inputs import discover_inputs
from tt_metal.fabric.debug.visualizer.decode.output import build_decoded
from tt_metal.fabric.debug.visualizer.decode.tests.fixtures import write_input


def region(region_id: str, **fields):
    payload = {
        "id": region_id,
        "enabled": True,
        "status": "ok",
        "error": None,
        "value": None,
        "count": 0,
    }
    payload.update(fields)
    return payload


def stream(post: int, *, torn: bool = False, status: str = "ok"):
    return {"pre": post, "post": post, "torn": torn} if status != "torn" else {"pre": post, "post": post + 1, "torn": True}


def synthetic_router(regions, *, rings=None, instance=None, status="ok"):
    return {
        "id": {"mesh_id": 0, "chip_id": 0, "eth_chan": 1},
        "capture": {"status": status},
        "instance": instance
        or {
            "worker_sender_channel": 0,
            "sender_channels_per_vc": [1, 1, 0, 0],
            "receiver_channels_per_vc": [1, 0],
            "credit_plan": {
                "vc0_uses_counters": False,
                "vc1_uses_counters": False,
                "vc2_uses_counters": False,
            },
        },
        "regions": regions,
        "rings": rings or [],
        "warnings": [],
    }


IDLE_REGIONS = [
    region("sender.0.ring", count=14),
    region("sender.0.free_slots", value=stream(14)),
    region("sender.0.credits.acked", value=stream(0)),
    region("sender.0.credits.completed", value=stream(0)),
    region("sender.0.control.connection_sem", value={"word": 0, "words": [0]}),
    region("credits.sender.1.free_slots", enabled=False, value=stream(11987)),
    region("receiver.0.ring", count=8),
    region("receiver.0.pkts_sent", value=stream(0)),
    region("credits.downstream.vc0.edge1.free_slots", value=stream(4)),
]


class CreditsTest(unittest.TestCase):
    def test_idle_occupancy_is_zero(self):
        router = synthetic_router(
            IDLE_REGIONS,
            rings=[{"id": "sender.0.ring", "occupied_count": None, "occupancy_source": None, "occupancy_status": "unknown"}],
        )
        channels = decode_channels(router)
        sender = channels["senders"][0]
        self.assertEqual(sender["occupied"], 0)
        self.assertEqual(sender["role"], "worker")
        self.assertEqual(sender["connection"]["name"], "unused")
        self.assertEqual(len(channels["senders"]), 1)
        self.assertEqual(channels["receivers"][0]["pkts_pending"], 0)
        router["channels"] = channels
        self.assertEqual(stall_score(router), 0.0)
        self.assertEqual(router["rings"][0]["occupied_count"], 0)
        self.assertEqual(router["rings"][0]["occupancy_source"], "stream")

    def test_backpressure_scores_one(self):
        regions = [
            region("sender.0.ring", count=14),
            region("sender.0.free_slots", value=stream(0)),
            region("receiver.0.ring", count=8),
            region("receiver.0.pkts_sent", value=stream(8)),
            region("credits.downstream.vc0.edge1.free_slots", value=stream(0)),
        ]
        router = synthetic_router(regions)
        router["channels"] = decode_channels(router)
        self.assertEqual(router["channels"]["senders"][0]["occupied"], 14)
        self.assertEqual(stall_score(router), 1.0)

    def test_disabled_stream_is_ignored(self):
        router = synthetic_router(IDLE_REGIONS)
        channels = decode_channels(router)
        self.assertEqual([sender["index"] for sender in channels["senders"]], [0])

    def test_torn_occupancy_is_null(self):
        regions = [
            region("sender.0.ring", count=14),
            region("sender.0.free_slots", status="torn", value=stream(3, torn=True, status="torn")),
        ]
        router = synthetic_router(regions)
        router["channels"] = decode_channels(router)
        self.assertIsNone(router["channels"]["senders"][0]["occupied"])
        self.assertTrue(router["channels"]["senders"][0]["torn"])
        self.assertIsNone(stall_score(router))

    def test_counter_backed_vc_keeps_raw_counters(self):
        instance = {
            "worker_sender_channel": 0,
            "sender_channels_per_vc": [1, 0, 0, 0],
            "receiver_channels_per_vc": [1, 0],
            "credit_plan": {
                "vc0_uses_counters": True,
                "vc1_uses_counters": False,
                "vc2_uses_counters": False,
            },
        }
        regions = [
            region("sender.0.ring", count=4),
            region("sender.0.free_slots", value=stream(4)),
            region("sender.0.credits.acked", value=stream(9)),
            region("credits.to_sender_ack", value={"counters": [11, 12]}),
            region("credits.to_sender_completion", value={"counters": [21, 22]}),
        ]
        router = synthetic_router(regions, instance=instance)
        sender = decode_channels(router)["senders"][0]
        self.assertEqual(sender["credit_backing"], "counter")
        self.assertIsNone(sender["acked_pending"])
        self.assertIsNone(sender["completed_pending"])
        self.assertEqual(sender["counters"], {"to_sender_ack": 11, "to_sender_completion": 21})

    def test_free_slots_above_depth_is_inconsistent(self):
        regions = [
            region("sender.0.ring", count=2),
            region("sender.0.free_slots", value=stream(4)),
        ]
        router = synthetic_router(regions)
        sender = decode_channels(router)["senders"][0]
        self.assertEqual(sender["status"], "inconsistent")
        self.assertEqual(sender["occupied"], -2)
        self.assertIn("occupancy -2", router["warnings"][0])
        router["channels"] = {"senders": [sender], "receivers": [], "downstream": []}
        self.assertIsNone(stall_score(router))

    def test_links_copy_router_stall_score(self):
        decoded = {
            "routers": [
                {
                    "id": {"mesh_id": 0, "chip_id": 0, "eth_chan": 1},
                    "capture": {"status": "ok"},
                    "stall_score": 0.5,
                }
            ],
            "topology": {
                "links": [
                    {
                        "src": {"mesh_id": 0, "chip_id": 0, "eth_chan": 1},
                        "dst": {"mesh_id": 0, "chip_id": 1, "eth_chan": 2},
                    },
                    {
                        "src": {"mesh_id": 0, "chip_id": 9, "eth_chan": 0},
                        "dst": {"mesh_id": 0, "chip_id": 0, "eth_chan": 1},
                    },
                ]
            },
        }
        annotate_links(decoded)
        self.assertEqual(decoded["topology"]["links"][0]["stall_score"], 0.5)
        self.assertEqual(decoded["topology"]["links"][0]["status"], "ok")
        self.assertIsNone(decoded["topology"]["links"][1]["stall_score"])
        self.assertEqual(decoded["topology"]["links"][1]["status"], "not_captured")

    def test_fixture_decode_is_idle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_input(root)
            decoded = build_decoded(discover_inputs([root]))
            router = decoded["routers"][0]
            self.assertEqual(router["channels"]["senders"][0]["occupied"], 0)
            self.assertEqual(router["channels"]["receivers"][0]["pkts_pending"], 0)
            self.assertEqual(router["channels"]["downstream"][0]["free_slots"], 4)
            self.assertEqual(router["stall_score"], 0.0)
            self.assertEqual(decoded["topology"]["links"][0]["stall_score"], 0.0)
            self.assertEqual(router["rings"][0]["occupancy_status"], "ok")
            self.assertIn("ConnectionState", decoded["enums"])


if __name__ == "__main__":
    unittest.main()
