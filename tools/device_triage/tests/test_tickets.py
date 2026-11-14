# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import unittest

from device_triage import tickets

KNOWN = {
    "bh-glx-120-b02u08",
    "bh-glx-110-c10u08",
    "bh-glx-120-c08u02",
    "bh-glx-110-c08u02",
    "bh-glx-120-a02u08",
    "bh-glx-110-a02u08",
    "bh-glx-110-d09u02",
    "bh-glx-110-d09u08",
    "bh-glx-110-d09u14",
    "bh-glx-110-d09u20",
    "bh-glx-110-d06u14",
    "bh-glx-110-d07u14",
    "bh-glx-110-d08u14",
    "bh-glx-120-c01u02",
    "bh-glx-120-c01u08",
    "bh-glx-120-c01u20",
}


class TicketHostsTest(unittest.TestCase):
    def hosts(self, text: str, known: set[str] = KNOWN) -> dict[str, str]:
        return tickets.ticket_host_map(text, known)

    def test_full_name(self):
        self.assertEqual(self.hosts("bh-glx-120-b02u08 enumerates only 24/32 chips"), {"bh-glx-120-b02u08": "full"})

    def test_superpod_short(self):
        self.assertEqual(self.hosts("110-c10u08 chip missing after reset"), {"bh-glx-110-c10u08": "full"})

    def test_brackets(self):
        found = self.hosts("Recovery fails on bh-glx-110-d09u[02,08,14,20]")
        self.assertEqual(found, {f"bh-glx-110-d09u{s}": "bracket" for s in ("02", "08", "14", "20")})

    def test_bracket_slash_expansion(self):
        self.assertEqual(tickets.expand_brackets("a0[6/7]u0[2,8]"), ["a06u02", "a06u08", "a07u02", "a07u08"])

    def test_slash_slots(self):
        found = self.hosts("bh-glx-120-c01u02/u08/u20: NFS mounts failed at the reboot")
        self.assertEqual(set(found), {"bh-glx-120-c01u02", "bh-glx-120-c01u08", "bh-glx-120-c01u20"})

    def test_prefix_without_superpod_is_ambiguous(self):
        found = self.hosts("bh-glx-c08u02 lost two trays")
        self.assertEqual(found, {"bh-glx-120-c08u02": "short-ambiguous", "bh-glx-110-c08u02": "short-ambiguous"})

    def test_short_with_superpod_in_sight(self):
        self.assertEqual(self.hosts("120A: a02u08 does not come back"), {"bh-glx-120-a02u08": "short"})

    def test_quad(self):
        found = self.hosts("110 D06_07_08_09u14 ring hangs")
        self.assertEqual(found, {f"bh-glx-110-d0{r}u14": "quad" for r in (6, 7, 8, 9)})

    def test_site_and_room_numbers_are_not_superpods(self):
        found = self.hosts("YYZ2 room 803: a02u08 fans loud")
        self.assertEqual(set(found.values()), {"short-ambiguous"})


def ticket(key: str, summary: str, description: str = "", status: str = "Backlog", created: float = 0.0) -> dict:
    return {
        "key": key,
        "created": created,
        "status": status,
        "summary": summary,
        "description": description,
        "comments": "",
    }


class HostViewTest(unittest.TestCase):
    def test_subject_bot_and_mentions(self):
        host = "bh-glx-120-b03u02"
        found = [
            ticket("DIIM-562", "bh-glx-120-b03u02 showing only 24 chips", created=3.0),
            ticket("DCAMP-1756", "[Fabric Health Check] Test failed on bh-glx-120-b03u02", created=2.0),
            ticket(
                "DIIM-536", "Chip 24 becoming unavailable", "seen on bh-glx-120-b03u02", status="Blocked", created=1.0
            ),
            ticket("DIIM-604", "Weka cpuset", " ".join(f"bh-glx-120-b0{i}u02" for i in range(1, 8)), created=4.0),
            ticket("DIIM-1", "bh-glx-110-a01u02 down"),
        ]
        view = tickets.host_view(found, host, {"DIIM-536": {"pcie"}})
        self.assertEqual([v["key"] for v in view], ["DIIM-562", "DIIM-536", "DCAMP-1756", "DIIM-604"])
        by_key = {v["key"]: v for v in view}
        self.assertTrue(by_key["DIIM-562"]["subject"] and by_key["DIIM-562"]["open"])
        self.assertEqual(by_key["DIIM-562"]["categories"], ["pcie"])
        self.assertEqual((by_key["DIIM-536"]["how"], by_key["DIIM-536"]["categories"]), ("description", ["pcie"]))
        self.assertTrue(by_key["DCAMP-1756"]["bot"])
        # A description listing seven hosts is fleet-wide, not about this one.
        self.assertFalse(by_key["DIIM-604"]["subject"])

    def test_siblings(self):
        self.assertEqual(tickets.siblings("bh-glx-120-b02u08"), ["bh-glx-110-b02u08"])
        self.assertEqual(tickets.siblings("qb2-120-p03t08"), [])


if __name__ == "__main__":
    unittest.main()
