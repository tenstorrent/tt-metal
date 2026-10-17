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


if __name__ == "__main__":
    unittest.main()
