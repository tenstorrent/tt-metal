# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import unittest

from device_triage import hosts
from device_triage.logscan import scan_lines
from device_triage.signatures import classify_check, match_line

# Real lines from /data/ci-provisioner and /data/dcamp logs.
LINES = {
    "Error: POST_RESET failed for device 25. ": ("post_reset_failed", "25"),
    "2026-10-14 06:18:12.362 | error    |             UMD | Sending reset ioctl to device 17 failed: Sending reset "
    "command failed on device 17 with flags 0x6, KMD returned: -19": ("reset_ioctl_failed", "17"),
    "Read 0xffffffff over PCIe ID 6: the board should be reset.": ("pcie_read_ffffffff", "6"),
    "  finding: pci:16 would not open: Query mappings failed on device 16.": ("query_mappings_failed", "16"),
    "terminate called after throwing an instance of 'tt::umd::error::UmdException<tt::umd::error::PcieHangError>'": (
        "pcie_hang",
        None,
    ),
    'C++ exception with description "Firmware startup error on device 29 at core 8-0 over NOC0: scratch_status=0x7, '
    "postcode=0xc0de0041, smc_init_status=0x10 (GDDR training failure) (Timed out after 300000": (
        "fw_startup_gddr_training",
        "29",
    ),
    "ARC startup error at core 8-0 over NOC0: scratch_status=0x7, postcode=0xc0de0041 (Timed out after 300000 ms)": (
        "arc_startup_error",
        None,
    ),
    "TT_FATAL: Graph specified in MGD could not fit in the discovered physical topology.": (
        "mgd_topology_mismatch",
        None,
    ),
    "\x1b[91m Error in resetting galaxy 6u trays!": ("glx_reset_failed", None),
    "WARNING: cross-host port down FAILED (exit code 241); running cleanup reset then retrying port down.": (
        "port_down_failed",
        None,
    ),
    "[Sat Sep 19 04:38:14 2026] tenstorrent 0000:c8:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0018]": (
        "iommu_fault",
        None,
    ),
    "[Tue Oct 13 10:15:21 2026] mce: [Hardware Error]: Machine check events logged": ("machine_check", None),
    "bh-glx-120-b03u08 only shows 24 / 32 devices and needs a reboot": ("devices_short", None),
    "Physical Discovery found 2 missing channel connections:": ("missing_connections", None),
    "user@bh-glx-110-a09u08: Permission denied (publickey).": ("ssh_auth_failed", None),
    # Jira attachments (DIIM-511, DIIM-327, DIIM-550's follow-up HC run).
    'C++ exception with description "Memory barrier timed out after 1000 ms on device 30 core d17-21 (TRANSLATED) at 0x0: '
    "expected 0xaa, read 0xffffffff. 0 of 8 core(s) synced.": ("membar_read_ffffffff", "30"),
    "l: 2026-09-24 23:41:34.005 | critical |          Always | TT_FATAL: Read unexpected run_mailbox value from core 15-3": (
        "run_mailbox_unexpected",
        None,
    ),
    "l: 2026-08-14 18:54:16.713 | critical |            Test | missing links: chip[0 (0000:01:00.0), ubb: 1, chip: 1]: "
    "expected 10 links, got 8 (common.hpp:932)": ("chip_links_missing", None),
    "Ethernet tests\t loop 1 failed": ("eth_test_loop_failed", None),
    'C++ exception with description "Firmware bundle version mismatch for device 5801399391097570079: expected 19.12.0, '
    "got 19.11.0.": ("fw_mismatch", None),
    # Fabric health check console logs attached to DCAMP bot tickets (April-June).
    "fw_bundle_version=255.255.255.255": ("fw_bundle_unreadable", None),
    "Chip 11 Tray: 2 N4 has 1 connections to Chip 3 Tray: 1 N4, expected 2": ("chip_connections_short", None),
    "2026-05-14 07:10:04.914 | error    |             UMD | Expected NOC address: 0x1000000000000000, but got "
    "0x1000000040000000 (silicon_sysmem_manager.cpp:372)": ("sysmem_noc_mismatch", None),
    "2026-05-22 13:58:53.358 | warning  |             UMD | Board 0x47330000000 has 24 chips, but expected 32 chips "
    "for board type ubb_blackhole. (cluster_descriptor.cpp:1232)": ("board_chips_short", None),
    "2026-05-19 07:28:35.015 | warning  |             UMD | ARC core (x=8,y=0) startup timed out after: 300000. "
    "Status: 0xffffffff (utils.hpp:129)": ("arc_startup_error", None),
    'C++ exception with description "ARC core (8, 0) failed to start." thrown in the test body.': (
        "arc_startup_error",
        None,
    ),
    "2026-06-07 07:32:40.913 | warning  |             UMD | TENSTORRENT_IOCTL_SET_POWER_STATE failed on device 0: "
    "Invalid argument (pci_device.cpp:1046)": ("set_power_state_einval", None),
    "TENSTORRENT_IOCTL_SET_POWER_STATE failed on device 12: No such device (pci_device.cpp:1046)": (
        "set_power_state_enodev",
        "12",
    ),
    "2026-05-05 06:55:46.842 | warning  |             UMD | Failed to unmap sysmem buffer (size: 0x40000000, IOVA: "
    "0x40000000). (sysmem_buffer.cpp:148)": ("sysmem_unmap_failed", None),
    "Docker startup error: Error while fetching server API version:": ("docker_failed", None),
}
NON_MATCHING = (
    "fw_bundle_version=19.8.1.0",
    "Mapping validation completed in relaxed mode: 88 warning(s) about channel count mismatches",
    "24/32 chips OK; first failure: 0000:04:00.0 (UBB1/U4) down_ports=[4, 6]",
    "Regenerate on failure: true",
)


class TestLineSignatures(unittest.TestCase):
    def test_known_lines(self):
        for line, (name, dev) in LINES.items():
            with self.subTest(line=line[:60]):
                hit = match_line(line)
                self.assertIsNotNone(hit)
                sig, m = hit
                self.assertEqual(sig.name, name)
                self.assertEqual(m.groupdict().get("dev"), dev)

    def test_non_matching(self):
        for line in NON_MATCHING:
            with self.subTest(line=line[:60]):
                hit = match_line(line)
                self.assertTrue(hit is None or hit[0].severity == "info", hit and hit[0].name)


class TestChecks(unittest.TestCase):
    def test_missing_tray(self):
        check = {
            "name": "asic_location_per_ubb",
            "status": "FAIL",
            "details": "missing UBBs: [4]; expected BDFs: ['0000:81:00.0']",
            "data": {"missing_trays": [4], "missing_bdfs": ["0000:81:00.0"]},
        }
        (f,) = classify_check("snapshot", check, {}, {})
        self.assertEqual((f.category, f.signal, f.severity), ("pcie", "asic_location_missing", "fail"))
        self.assertEqual(f.chips, {"u4"})
        self.assertEqual(hosts.scope_of(f.chips), "tray")

    def test_reset_r_is_weak_unless_fallback_failed(self):
        check = {
            "name": "reset_1_r",
            "status": "FAIL",
            "details": "flag=-r rc=1 post_pcie=31",
            "data": {"post_pcie_count": 31},
        }
        weak = classify_check("reset_loop", check, {}, {"cpld_auto_recover": "PASS"})
        self.assertEqual([(f.category, f.severity) for f in weak], [("reset", "warn"), ("pcie", "fail")])
        strong = classify_check("reset_loop", check, {}, {"cpld_auto_recover": "FAIL"})
        self.assertEqual(strong[0].severity, "fail")

    def test_noise_is_info(self):
        (f,) = classify_check(
            "reset_loop", {"name": "cpld_fw_old", "status": "WARN", "details": "", "data": {}}, {}, {}
        )
        self.assertEqual(f.severity, "info")

    def test_chips_from_details(self):
        check = {
            "name": "gddr_training_per_channel",
            "status": "FAIL",
            "details": "224/256 channels trained; first failures: [0000:01:00.0 (UBB1/U1) ch=0 train_done=0]",
            "data": {"failures": [{"bdf": "0000:02:00.0", "channel": 1}]},
        }
        (f,) = classify_check("snapshot", check, {"0000:02:00.0": "u1c2"}, {})
        self.assertEqual(f.chips, {"u1c1", "u1c2"})

    def test_pcie_gen_only_off_speed_chips(self):
        check = {
            "name": "pcie_gen",
            "status": "WARN",
            "data": {},
            "details": "U6 chips (expected Gen5 on RevC): 0000:06:00.0 (UBB1/U6)=Gen5, 0000:46:00.0 (UBB2/U6)=Gen5, "
            "0000:86:00.0 (UBB4/U6)=Gen5, 0000:c6:00.0 (UBB3/U6)=Gen1(!); below: 0000:c6:00.0 (UBB3/U6)",
        }
        (f,) = classify_check("snapshot", check, {}, {})
        self.assertEqual((f.chips, f.severity), ({"u3c6"}, "warn"))


class TestHosts(unittest.TestCase):
    KNOWN = ["bh-glx-120-a02u08", "bh-glx-110-a02u08", "bh-glx-120-a04u08", "bh-glx-120-d06u14", "bh-glx-120-d07u14"]

    def test_short_names_with_superpod(self):
        found = hosts.ticket_hosts("120A a02u08, a04u08 do not reset properly", self.KNOWN)
        self.assertEqual(found, {"bh-glx-120-a02u08": "short", "bh-glx-120-a04u08": "short"})

    def test_short_name_ambiguous_without_superpod(self):
        found = hosts.ticket_hosts("a02u08 is down", self.KNOWN)
        self.assertEqual(set(found.values()), {"short-ambiguous"})
        self.assertEqual(set(found), {"bh-glx-120-a02u08", "bh-glx-110-a02u08"})

    def test_quad_notation(self):
        found = hosts.ticket_hosts("pcc nondeterminism on 120-D-Quad4-D06_07_08_09u14", self.KNOWN)
        self.assertEqual(set(found), {f"bh-glx-120-d{r:02d}u14" for r in (6, 7, 8, 9)})

    def test_full_names_and_fqdn(self):
        self.assertEqual(hosts.normalize_host("BH-GLX-120-C08U02.exabox.local"), "bh-glx-120-c08u02")
        self.assertEqual(
            hosts.find_hosts("power issues with bh-glx-110-a06u08, bh-glx-110-a06u14"),
            {"bh-glx-110-a06u08", "bh-glx-110-a06u14"},
        )

    def test_compress(self):
        self.assertEqual(hosts.compress_chips({f"u4c{i}" for i in range(1, 9)} | {"u1c3"}), "u1c3,u4")

    def test_summarize_chips(self):
        weights = {f"u3c{i}": 1.0 for i in range(1, 9)} | {"u3": 1.0, "u1c1": 0.9, "u2c2": 0.1}
        self.assertEqual(hosts.summarize_chips(weights), "u3,u1c1")

    def test_ubb_chip_pairs(self):
        line = "missing links: chip[0 (0000:01:00.0), ubb: 1, chip: 1]: expected 10 links, got 8"
        self.assertEqual(hosts.bdf_chip_pairs(line), [("0000:01:00.0", "u1c1")])

    def test_ticket_chip_refs(self):
        refs = hosts.ticket_chip_refs("bh-glx-120-a02u08 Tray 1,2 ; UBB4 device 22; chip 19; u2c5")
        self.assertTrue({"u1", "u4", "dev22", "chip19", "u2c5"} <= refs)


class TestScan(unittest.TestCase):
    def test_prefix_endpoint_and_rollover(self):
        lines = [
            "[bh-glx-110-d03u02][23:59:58] start",
            "[bh-glx-110-d03u02][00:00:05]   - PhysicalChannelEndpoint{hostname='bh-glx-110-d03u20', tray_id=2, "
            "asic_channel=AsicChannel{asic_location=7, channel_id=9}} <-> PhysicalChannelEndpoint{hostname="
            "'bh-glx-110-d03u20', tray_id=4, asic_channel=AsicChannel{asic_location=8, channel_id=9}} missing channel connections",
            "[bh-glx-120-c06u08][00:00:06]  Error: POST_RESET failed for device 25. ",
            "[bh-glx-120-c06u08][00:00:07]  Error: POST_RESET failed for device 25. ",
        ]
        base = 1_760_400_000.0  # 2025-10-14T00:00:00Z
        events = scan_lines(
            lines, default_host=None, base_ts=base, source="t", origin="log:t", path="p", ordinals={25: "u4c2"}
        )
        by_host = {e.host: e for e in events}
        self.assertEqual(by_host["bh-glx-110-d03u20"].chips, "u2c7,u4c8")
        reset = by_host["bh-glx-120-c06u08"]
        self.assertEqual((reset.signal, reset.chips, reset.count), ("post_reset_failed", "u4c2", 2))
        self.assertEqual(reset.ts, base + 86400 + 6)


if __name__ == "__main__":
    unittest.main()
