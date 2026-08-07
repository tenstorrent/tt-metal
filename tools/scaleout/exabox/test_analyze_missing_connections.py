#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for missing-connection parsing in analyze_validation_results.py."""

import io
import unittest
from contextlib import redirect_stdout

from analyze_validation_results import (
    LogAnalysis,
    parse_missing_connections,
    print_missing_links_summary,
    unique_missing_connections,
)

CLEAN_QSFP_LOG = """
[bh-glx-110-a09u02][16:13:51] Physical Discovery found 7 missing channel connections:
[bh-glx-110-a09u02][16:13:51]   - PhysicalChannelEndpoint{hostname='bh-glx-110-a09u08', tray_id=3, asic_channel=AsicChannel{asic_location=1, channel_id=9}} <-> PhysicalChannelEndpoint{hostname='bh-glx-110-a09u14', tray_id=1, asic_channel=AsicChannel{asic_location=1, channel_id=9}}
[bh-glx-110-a09u02][16:13:51]   - PhysicalChannelEndpoint{hostname='bh-glx-110-a09u08', tray_id=3, asic_channel=AsicChannel{asic_location=6, channel_id=8}} <-> PhysicalChannelEndpoint{hostname='bh-glx-110-a09u14', tray_id=1, asic_channel=AsicChannel{asic_location=6, channel_id=8}}
[bh-glx-110-a09u02][16:13:51] Physical Discovery found 6 missing port/cable connections:
[bh-glx-110-a09u02][16:13:51]   - PhysicalPortEndpoint{hostname='bh-glx-110-a09u08', aisle='A', rack=9, shelf_u=8, tray_id=3, port_type=QSFP_DD, port_id=8} <-> PhysicalPortEndpoint{hostname='bh-glx-110-a09u14', aisle='A', rack=9, shelf_u=14, tray_id=1, port_type=QSFP_DD, port_id=8}
[bh-glx-110-a09u02][16:13:51]   - PhysicalPortEndpoint{hostname='bh-glx-110-a09u08', aisle='A', rack=9, shelf_u=8, tray_id=3, port_type=QSFP_DD, port_id=11} <-> PhysicalPortEndpoint{hostname='bh-glx-110-a09u14', aisle='A', rack=9, shelf_u=14, tray_id=1, port_type=QSFP_DD, port_id=11}
"""

INTERLEAVED_LOG = """
[bh-glx-b06u08][16:18:23] Physical Discovery found 8 missing channel connections:
[bh-glx-b06u08][16:18:23]   - PhysicalChannelEndpoint{hostname='bh-glx-b07u02', tray_id=1, asic_channel=AsicChannel{asic_location=4, channel_id=6}} <-> PhysicalChannelEndpoint{hostname='bh-glx-b07u02', tray_id=1, asic_channel=AsicChannel{asic_location=8, channel_id=0}}
[bh-glx-b07u02] 2026-08-06 16:18:23.072 | warning  |     Distributed | Sending ETH_MSG_PORT_ACTION to bring ports down on all links (ethernet_link_api.cpp:148)
[bh-glx-b06u08][16:18:23]   - PhysicalChannelEndpoint{hostname='bh-glx-b07u02', tray_id=1, asic_channel=AsicChannel{asic_location=4, channel_id=7}} <-> PhysicalChannelEndpoint{hostname='bh-glx-b07u02', tray_id=1, asic_channel=AsicChannel{asic_location=8, channel_id=1}}
[bh-glx-b06u02] 2026-08-06 16:18:23.074 | warning  |     Distributed | Waiting for all messages to be processed (ethernet_link_api.cpp:154)
[bh-glx-b06u08][16:18:23]   - PhysicalChannelEndpoint{hostname='bh-glx-b07u02', tray_id=1, asic_channel=AsicChannel{asic_location=8, channel_id=7}} <-> Physical
ChannelEndpoint{hostname='bh-glx-b07u02', tray_id=3, asic_channel=AsicChannel{asic_location=8, channel_id=7}}
[bh-glx-b06u08][16:18:23] Physical Discovery found 4 missing port/cable connections:
[bh-glx-b07u08] 2026-08-06 16:18:23.063 | warning  |     Distributed | Sending ETH_MSG_PORT_ACTION to bring ports down on all links (ethernet_link_api.cpp:148)
[bh-glx-b06u08][16:18:23]   - PhysicalPortEndpoint{hostname='bh-glx-b07u02', aisle='B', rack=7, shelf_u=2, tray_id=1, port_type=TRACE, port_id=15} <-> PhysicalPortEndpoint{hostname='bh-glx-b07u02', aisle='B', rack=7, shelf_u=2, tray_id=1, port_type=TRACE, port_id=16}
[bh-glx-b06u02] 2026-08-06 16:18:23.074 | warning  |     Distributed | Waiting for all messages to be processed (ethernet_link_api.cpp:154)
[bh-glx-b06u08][16:18:23]   - PhysicalPortEndpoint{hostname='bh-glx-b07u02', aisle='B', rack=7, shelf_u=2, tray_id=1, port_type=TRACE, port_id=19} <-> PhysicalPortEndpoint{hostname='bh-glx-b07u02', aisle='B', rack=7, shelf_u=2, tray_id=1, port_type=TRACE, port_id=20}
[bh-glx-b06u08][16:18:23]   - PhysicalPortEndpoint{hostname='bh-glx-b07u02', aisle='B', rack=7, shelf_u=2, tray_id=1, port_type=LINKING_BOARD_2, port_id=2} <-> PhysicalPortEndpoint{hostname='bh-glx-b07u02', aisle='B', rack=7, shelf_u=2, tray_id=3, port_type=LINKING_BOARD_2, port_id=2}
[bh-glx-b06u08][16:18:23]   - PhysicalPortEndpoint{hostname='bh-glx-b07u02', aisle='B', rack=7, shelf_u=2, tray_id=1, port_type=LINKING_BOARD_3, port_id=1} <-> PhysicalPortEndpoint{hostname='bh-glx-b07u02', aisle='B', rack=7, shelf_u=2, tray_id=2, port_type=LINKING_BOARD_3, port_id=1}
"""

REPEATED_DISCOVERY_LOG = """
[host][10:00:00] Physical Discovery found 1 missing port/cable connections:
[host][10:00:00]   - PhysicalPortEndpoint{hostname='h1', aisle='A', rack=1, shelf_u=2, tray_id=1, port_type=QSFP_DD, port_id=3} <-> PhysicalPortEndpoint{hostname='h2', aisle='A', rack=1, shelf_u=8, tray_id=2, port_type=QSFP_DD, port_id=3}
[host][10:01:00] Physical Discovery found 1 missing port/cable connections:
[host][10:01:00]   - PhysicalPortEndpoint{hostname='h1', aisle='A', rack=1, shelf_u=2, tray_id=1, port_type=QSFP_DD, port_id=3} <-> PhysicalPortEndpoint{hostname='h2', aisle='A', rack=1, shelf_u=8, tray_id=2, port_type=QSFP_DD, port_id=3}
[host][10:01:00] Physical Discovery found 1 missing channel connections:
[host][10:01:00]   - PhysicalChannelEndpoint{hostname='h1', tray_id=1, asic_channel=AsicChannel{asic_location=0, channel_id=4}} <-> PhysicalChannelEndpoint{hostname='h2', tray_id=2, asic_channel=AsicChannel{asic_location=0, channel_id=4}}
"""


class TestParseMissingConnections(unittest.TestCase):
    def test_clean_qsfp_port_and_channel_pairs(self):
        conns = parse_missing_connections(CLEAN_QSFP_LOG)
        ports = [c for c in conns if c[0] == "port"]
        channels = [c for c in conns if c[0] == "channel"]
        self.assertEqual(len(ports), 2)
        self.assertEqual(len(channels), 2)
        self.assertEqual(ports[0][1][0], "bh-glx-110-a09u08")
        self.assertEqual(ports[0][1][2], "QSFP_DD")
        self.assertEqual(ports[0][1][3], "8")
        self.assertEqual(channels[0][1][2], "1")
        self.assertEqual(channels[0][1][3], "9")

    def test_interleaved_host_logs_and_wrapped_endpoint(self):
        conns = parse_missing_connections(INTERLEAVED_LOG)
        ports = [c for c in conns if c[0] == "port"]
        channels = [c for c in conns if c[0] == "channel"]
        self.assertEqual(len(ports), 4)
        self.assertEqual(len(channels), 3)
        wrapped = [c for c in channels if c[1][3] == "7" and c[2][3] == "7" and c[1][1] == "1" and c[2][1] == "3"]
        self.assertEqual(len(wrapped), 1)
        port_types = {c[1][2] for c in ports} | {c[2][2] for c in ports}
        self.assertIn("TRACE", port_types)
        self.assertIn("LINKING_BOARD_2", port_types)
        self.assertIn("LINKING_BOARD_3", port_types)

    def test_unique_dedupes_repeated_discovery_blocks(self):
        conns = parse_missing_connections(REPEATED_DISCOVERY_LOG)
        self.assertEqual(len(conns), 3)  # 2 identical ports + 1 channel before dedupe
        analysis = LogAnalysis(filepath="iter.log", missing_connections=conns)
        unique = unique_missing_connections([analysis])
        ports = [c for c in unique if c[0] == "port"]
        channels = [c for c in unique if c[0] == "channel"]
        self.assertEqual(len(ports), 1)
        self.assertEqual(len(channels), 1)

    def test_print_missing_links_summary_includes_port_id(self):
        conns = parse_missing_connections(CLEAN_QSFP_LOG)
        analysis = LogAnalysis(filepath="iter.log", missing_connections=conns)
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_missing_links_summary([analysis])
        out = buf.getvalue()
        self.assertIn("Missing Links", out)
        self.assertIn("Port/cable (2 unique):", out)
        self.assertIn("QSFP_DD | bh-glx-110-a09u08 tray 3 port 8  <->  bh-glx-110-a09u14 tray 1 port 8", out)
        self.assertIn("Channel (2 unique):", out)
        self.assertIn("bh-glx-110-a09u08 <-> bh-glx-110-a09u14", out)


if __name__ == "__main__":
    unittest.main()
