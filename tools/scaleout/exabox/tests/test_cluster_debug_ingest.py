#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for cluster_debug_ingest.py: the derivation of health-check
findings from a `tt-bh-glx-cluster-debug` dump.

Every case is built from synthetic records rather than a captured dump, so the
edge cases that matter can be stated one at a time — an untrained port's
zero-filled remote_info, a pair blind at both ends, a port that was never read.
Those are exactly the states a healthy machine never produces and a sick one
produces constantly, so a fixture from a good galaxy would exercise none of
them."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

import cluster_debug_ingest as ingest  # noqa: E402

HOST = "bh-glx-110-a07u02"
TRAINED = "LINK_TRAIN_PASS"

# A zero-filled remote_info is what an untrained port carries: the firmware
# writes a sentinel, not an answer. Reading it as a partner is the mistake this
# module's guard exists to prevent.
DARK_REMOTE = {"asic_id": "0x0000000000000000", "eth_id": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Record builders
# ─────────────────────────────────────────────────────────────────────────────


def envelope(findings=None, descriptor=None, **kw):
    record = {
        "SCHEMA": "snapshot",
        "SCHEMA_VER": "1.0.0",
        "SNAPSHOT_ID": "snap:20260901T174233Z:8f21c4",
        "TIMESTAMP": "2026-09-01T17:42:33Z",
        "DURATION_S": 412.8,
        "TOOL": {"NAME": "bh_glx_cluster_debug.py", "VERSION": "0.1.0", "ARGV": [], "REASON": None},
        "HOST": {"HOSTNAME": HOST, "KERNEL": "6.8.0"},
        "SOURCES": {},
        "FINDINGS": list(findings or []),
    }
    if descriptor is not None:
        record["SOURCES"]["FACTORY_SYSTEM_DESCRIPTOR"] = {
            "FILE": "factory_system_descriptor.textproto",
            "PRESENT": True,
            "MATCHED_HOSTNAME": descriptor,
        }
    record.update(kw)
    return record


def galaxy(absent_ubbs=()):
    ubbs = {
        str(n): (
            {"UBB_UID": None, "UBB_ENTRY": None, "PRESENT": False, "ABSENT_REASON": f"no PCIe devices on bus {n}"}
            if n in absent_ubbs
            else {"UBB_UID": f"ubb:SER{n}", "UBB_ENTRY": f"e{n}", "PRESENT": True}
        )
        for n in range(1, 5)
    }
    return {
        "SCHEMA": "galaxy",
        "ENTRY_ID": "g1",
        "UID": "glx:QTWS7TKC260400005",
        "PATH": f"glx={HOST}",
        "CHASSIS_SERIAL": "QTWS7TKC260400005",
        "GALAXY_ID": HOST,
        "UBBS": ubbs,
        "COLLECTION": {"STATUS": "OK"},
    }


def ubb(num, rev="BH_GALAXY_REV_C", agreement=True, absent_asics=(), status="OK"):
    asics = {
        str(loc): (
            {"ASIC_UID": None, "ASIC_ENTRY": None, "PRESENT": False, "ABSENT_REASON": "no response"}
            if loc in absent_asics
            else {"ASIC_UID": f"asic:0x{num}{loc}", "ASIC_ENTRY": f"a{num}{loc}", "PRESENT": True}
        )
        for loc in range(1, 9)
    }
    collection = {"STATUS": status}
    if status not in ("OK", "HARVESTED"):
        collection["REASON"] = "BMC unreachable"
    return {
        "SCHEMA": "ubb",
        "ENTRY_ID": f"e{num}",
        "UID": f"ubb:SER{num}",
        "PATH": f"glx={HOST}/ubb={num}",
        "UBB_NUM": num,
        "SERIAL": f"SER{num}",
        "ASICS": asics,
        "COLLECTION": collection,
        "BOARD_REV": rev,
        "BOARD_ID_ASIC_AGREEMENT": agreement,
    }


def asic(ubb_num, loc, status="OK"):
    collection = {"STATUS": status}
    if status not in ("OK", "HARVESTED"):
        collection["REASON"] = "L1 read returned all-ones (dark tile)"
    return {
        "SCHEMA": "asic",
        "ENTRY_ID": f"a{ubb_num}{loc}",
        "UID": f"asic:0x{ubb_num}{loc}",
        "PATH": f"glx={HOST}/ubb={ubb_num}/asic={loc}",
        "ASIC_LOCATION": loc,
        "UBB_NUM": ubb_num,
        "COLLECTION": collection,
    }


def port(
    entry_id,
    uid,
    path,
    *,
    partner_uid=None,
    partner_entry=None,
    remote=None,
    train=TRAINED,
    link_up=True,
    harvested=False,
    port_type="CHIP_TO_CHIP",
    status="OK",
    has_status_block=True,
    qsfp=None,
    retrain=0,
    corr_cw=0,
    uncorr_cw=0,
):
    collection = {"STATUS": status}
    if status not in ("OK", "HARVESTED"):
        collection["REASON"] = "a blob did not read"
    record = {
        "SCHEMA": "eth_port",
        "ENTRY_ID": entry_id,
        "UID": uid,
        "PATH": path,
        "QSFP_NAME": qsfp,
        "ETH_PARTNER_UID": partner_uid,
        "ETH_PARTNER_ENTRY": partner_entry,
        "COLLECTION": collection,
        "HARVESTED": harvested,
        "PORT_TYPE": port_type,
        "STATUS": (
            {
                "PORT_STATUS": "PORT_UP" if link_up else "PORT_DOWN",
                "TRAIN_STATUS": train,
                "LINK_UP": link_up,
                "RETRAIN_COUNT": retrain,
                "CORR_CW": corr_cw,
                "UNCORR_CW": uncorr_cw,
            }
            if has_status_block
            else None
        ),
        "DATA": {"eth_status": {"SOURCE": "get_eth_status", "VALUE": {"remote_info": remote}}},
    }
    return record


def cage(ubb_num, num, *, present=True, partner=None, status="OK", module_uid=None):
    collection = {"STATUS": status}
    if status not in ("OK", "HARVESTED"):
        collection["REASON"] = "cage did not answer"
    return {
        "SCHEMA": "qsfp_port",
        "ENTRY_ID": f"q{ubb_num}{num}",
        "UID": f"ubb:SER{ubb_num}/qsfp{num:02d}",
        "PATH": f"glx={HOST}/ubb={ubb_num}/qsfp={num}",
        "QSFP_NUM": num,
        "QSFP_NAME": f"j{num:03d}",
        "UBB_NUM": ubb_num,
        "MODULE_UID": module_uid,
        "QSFP_PARTNER_UID": partner,
        "COLLECTION": collection,
        "PRESENT": present,
    }


def module(ubb_num, num, pn="QDD-400G-AOC3M", sn="APF23140099"):
    return {
        "SCHEMA": "module",
        "ENTRY_ID": f"m{ubb_num}{num}",
        "UID": f"mod:{pn}:{sn}",
        "PATH": f"glx={HOST}/ubb={ubb_num}/qsfp={num}/module",
        "QSFP_NAME": f"j{num:03d}",
        "VENDOR_PN": pn,
        "VENDOR_SN": sn,
        "LENGTH_M": 3.0,
        "COLLECTION": {"STATUS": "OK"},
    }


def linked_pair(train=TRAINED, link_up=(True, True), crossed=False):
    """Two ports facing each other, each naming the other as expected partner.

    ``crossed`` leaves the expectation intact but has the firmware on the first
    port report a third ASIC instead — a cable in the wrong cage.
    """
    a_uid, b_uid = "asic:0x11/eth04", "asic:0x12/eth04"
    a = port(
        "pa",
        a_uid,
        f"glx={HOST}/ubb=1/asic=1/eth=4",
        partner_uid=b_uid,
        partner_entry="pb",
        remote={"asic_id": "0x99" if crossed else "0x12", "eth_id": 4},
        train=train,
        link_up=link_up[0],
    )
    b = port(
        "pb",
        b_uid,
        f"glx={HOST}/ubb=1/asic=2/eth=4",
        partner_uid=a_uid,
        partner_entry="pa",
        remote={"asic_id": "0x11", "eth_id": 4},
        train=train,
        link_up=link_up[1],
    )
    return a, b


def full_galaxy_records(**kw):
    """A dump with the counts a real 6U Galaxy has, so inventory passes."""
    records = [envelope(**kw), galaxy()]
    for num in range(1, 5):
        records.append(ubb(num))
        for loc in range(1, 9):
            records.append(asic(num, loc))
            for eth in range(14):
                records.append(
                    port(
                        f"p{num}{loc}{eth:02d}",
                        f"asic:0x{num}{loc}/eth{eth:02d}",
                        f"glx={HOST}/ubb={num}/asic={loc}/eth={eth}",
                        port_type="PCIE",
                        harvested=False,
                    )
                )
    return records


def find(checks, name):
    for entry in checks:
        if entry["name"] == name:
            return entry
    raise AssertionError(f"no check named {name!r} in {[c['name'] for c in checks]}")


# ─────────────────────────────────────────────────────────────────────────────
# Reading the dump
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadDump(unittest.TestCase):
    def _write(self, text: str) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        handle.write(text)
        handle.close()
        return handle.name

    def test_reads_one_record_per_line(self):
        path = self._write('{"SCHEMA": "galaxy"}\n{"SCHEMA": "ubb"}\n')
        records, malformed = ingest.load_dump(path)
        self.assertEqual([r["SCHEMA"] for r in records], ["galaxy", "ubb"])
        self.assertEqual(malformed, 0)

    def test_truncated_last_line_is_counted_not_fatal(self):
        # JSONL is chosen so a host that fell over mid-run still leaves a
        # readable file; the short line is dropped but must not go unreported.
        path = self._write('{"SCHEMA": "galaxy"}\n{"SCHEMA": "ub\n')
        records, malformed = ingest.load_dump(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(malformed, 1)

    def test_blank_lines_ignored(self):
        path = self._write('\n{"SCHEMA": "galaxy"}\n\n')
        records, malformed = ingest.load_dump(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(malformed, 0)

    def test_non_object_line_is_malformed(self):
        path = self._write('[1, 2, 3]\n{"SCHEMA": "galaxy"}\n')
        records, malformed = ingest.load_dump(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(malformed, 1)


class TestObservedPartner(unittest.TestCase):
    def test_reads_the_far_end_the_firmware_named(self):
        p = port("p", "u", "glx=h/eth=0", remote={"asic_id": "0x00000010deadbeef", "eth_id": 7})
        self.assertEqual(ingest.observed_partner(p), "asic:0x00000010deadbeef/eth07")

    def test_all_zero_asic_id_is_no_answer(self):
        # The guard that matters: without it every untrained link looks trained
        # to the same imaginary far end.
        self.assertIsNone(ingest.observed_partner(port("p", "u", "glx=h/eth=0", remote=DARK_REMOTE)))

    def test_missing_remote_info_is_no_answer(self):
        self.assertIsNone(ingest.observed_partner(port("p", "u", "glx=h/eth=0", remote=None)))

    def test_blob_carrying_an_error_instead_of_a_value(self):
        p = port("p", "u", "glx=h/eth=0")
        p["DATA"] = {"eth_status": {"SOURCE": "get_eth_status", "ERROR": "read timeout"}}
        self.assertIsNone(ingest.observed_partner(p))

    def test_short_path_drops_the_constant_host_segment(self):
        self.assertEqual(ingest.short_path({"PATH": f"glx={HOST}/ubb=3/asic=1/eth=10"}), "ubb=3/asic=1/eth=10")
        self.assertEqual(ingest.short_path({"PATH": "cluster=yyz"}), "cluster=yyz")


# ─────────────────────────────────────────────────────────────────────────────
# Link findings
# ─────────────────────────────────────────────────────────────────────────────


class TestLinkTraining(unittest.TestCase):
    def test_trained_ports_pass(self):
        a, b = linked_pair()
        checks = ingest.summarize([envelope(), a, b])
        self.assertEqual(find(checks, "clusterdbg_link_training")["status"], "PASS")

    def test_untrained_port_fails_and_is_named(self):
        a, b = linked_pair(train="LINK_TRAIN_TIMEOUT", link_up=(False, False))
        entry = find(ingest.summarize([envelope(), a, b]), "clusterdbg_link_training")
        self.assertEqual(entry["status"], "FAIL")
        self.assertIn("ubb=1/asic=1/eth=4", entry["details"])
        self.assertEqual(len(entry["data"]["failures"]), 2)

    def test_harvested_and_pcie_tiles_are_not_in_service(self):
        # 14 tiles per ASIC includes harvested ones and the PCIe pair; counting
        # them would report a healthy chip as having four dead links.
        records = [
            envelope(),
            port("h", "u1", "glx=h/eth=5", harvested=True, train=None, status="HARVESTED"),
            port("p", "u2", "glx=h/eth=12", port_type="PCIE", train=None),
            port("u", "u3", "glx=h/eth=0", port_type="UNCONNECTED", train=None),
        ]
        self.assertEqual(find(ingest.summarize(records), "clusterdbg_link_training")["status"], "SKIP")

    def test_unread_port_is_not_a_down_port(self):
        # A PARTIAL port has no STATUS block at all, so judging it would turn a
        # failed read into a failed link.
        records = [envelope(), port("x", "u", "glx=h/eth=4", status="PARTIAL", has_status_block=False)]
        self.assertEqual(find(ingest.summarize(records), "clusterdbg_link_training")["status"], "SKIP")


class TestLinkAsymmetry(unittest.TestCase):
    def test_agreeing_ends_pass(self):
        a, b = linked_pair()
        self.assertEqual(find(ingest.summarize([envelope(), a, b]), "clusterdbg_link_asymmetry")["status"], "PASS")

    def test_disagreeing_ends_are_reported_once_per_link(self):
        a, b = linked_pair(link_up=(True, False))
        entry = find(ingest.summarize([envelope(), a, b]), "clusterdbg_link_asymmetry")
        self.assertEqual(entry["status"], "FAIL")
        # A link is one thing; both of its records describe the same fault.
        self.assertEqual(len(entry["data"]["asymmetric"]), 1)

    def test_an_unread_end_has_no_state_to_disagree_with(self):
        a, b = linked_pair(link_up=(True, False))
        b["COLLECTION"] = {"STATUS": "READ_FAILED", "REASON": "dark"}
        b["STATUS"] = None
        self.assertEqual(find(ingest.summarize([envelope(), a, b]), "clusterdbg_link_asymmetry")["status"], "PASS")


class TestMissingChannel(unittest.TestCase):
    def test_expected_and_seen_passes(self):
        a, b = linked_pair()
        self.assertEqual(find(ingest.summarize([envelope(), a, b]), "clusterdbg_missing_channel")["status"], "PASS")

    def test_both_ends_blind_collapse_to_one_row(self):
        a, b = linked_pair(train="LINK_TRAIN_TIMEOUT", link_up=(False, False))
        a["DATA"]["eth_status"]["VALUE"]["remote_info"] = DARK_REMOTE
        b["DATA"]["eth_status"]["VALUE"]["remote_info"] = DARK_REMOTE
        entry = find(ingest.summarize([envelope(), a, b]), "clusterdbg_missing_channel")
        self.assertEqual(entry["status"], "FAIL")
        self.assertEqual(len(entry["data"]["missing"]), 1)
        self.assertEqual(entry["data"]["missing"][0]["ends"], "both ends")
        # Collapsed onto the end whose path sorts first, so the row is stable
        # between runs rather than depending on record order.
        self.assertEqual(entry["data"]["missing"][0]["path"], "ubb=1/asic=1/eth=4")

    def test_one_blind_end_stays_its_own_row(self):
        a, b = linked_pair()
        a["DATA"]["eth_status"]["VALUE"]["remote_info"] = DARK_REMOTE
        entry = find(ingest.summarize([envelope(), a, b]), "clusterdbg_missing_channel")
        self.assertEqual(entry["status"], "FAIL")
        self.assertEqual(len(entry["data"]["missing"]), 1)
        self.assertEqual(entry["data"]["missing"][0]["ends"], "this end only")

    def test_no_expectations_skips_rather_than_passing(self):
        # Reporting PASS here would claim coverage the run did not have.
        p = port("p", "u", f"glx={HOST}/ubb=1/asic=1/eth=4", remote=DARK_REMOTE)
        self.assertEqual(find(ingest.summarize([envelope(), p]), "clusterdbg_missing_channel")["status"], "SKIP")


class TestMiscabled(unittest.TestCase):
    def test_wrong_end_fails(self):
        a, b = linked_pair()
        third = port("pc", "asic:0x99/eth04", f"glx={HOST}/ubb=2/asic=1/eth=4", remote={"asic_id": "0x11", "eth_id": 4})
        a["DATA"]["eth_status"]["VALUE"]["remote_info"] = {"asic_id": "0x99", "eth_id": 4}
        entry = find(ingest.summarize([envelope(), a, b, third]), "clusterdbg_miscabled")
        self.assertEqual(entry["status"], "FAIL")
        self.assertEqual(len(entry["data"]["wrong_end"]), 1)

    def test_undescribed_link_without_a_descriptor_is_not_a_finding(self):
        # No descriptor matched, so the cabled links inside the chassis have no
        # expectation; calling each one a surprise would be noise on every run.
        a = port("pa", "asic:0x11/eth10", f"glx={HOST}/ubb=1/asic=1/eth=10", remote={"asic_id": "0x12", "eth_id": 10})
        b = port("pb", "asic:0x12/eth10", f"glx={HOST}/ubb=1/asic=2/eth=10", remote={"asic_id": "0x11", "eth_id": 10})
        entry = find(ingest.summarize([envelope(), a, b]), "clusterdbg_miscabled")
        self.assertEqual(entry["status"], "PASS")
        self.assertEqual(len(entry["data"]["undescribed"]), 2)
        self.assertIn("no factory descriptor matched", entry["details"])

    def test_undescribed_link_with_a_descriptor_warns(self):
        a = port("pa", "asic:0x11/eth10", f"glx={HOST}/ubb=1/asic=1/eth=10", remote={"asic_id": "0x12", "eth_id": 10})
        b = port("pb", "asic:0x12/eth10", f"glx={HOST}/ubb=1/asic=2/eth=10", remote={"asic_id": "0x11", "eth_id": 10})
        entry = find(ingest.summarize([envelope(descriptor=HOST), a, b]), "clusterdbg_miscabled")
        self.assertEqual(entry["status"], "WARN")

    def test_descriptor_present_but_not_naming_this_host_is_no_coverage(self):
        # Supplied and unmatched leaves the cabled links exactly as undescribed
        # as no descriptor at all.
        record = envelope()
        record["SOURCES"]["FACTORY_SYSTEM_DESCRIPTOR"] = {"FILE": "f.textproto", "PRESENT": True}
        self.assertFalse(ingest.descriptor_matched(record))


class TestPartnerDisagreement(unittest.TestCase):
    def test_mutual_naming_passes(self):
        a, b = linked_pair()
        self.assertEqual(
            find(ingest.summarize([envelope(), a, b]), "clusterdbg_partner_disagreement")["status"], "PASS"
        )

    def test_one_sided_naming_fails(self):
        a, b = linked_pair()
        b["DATA"]["eth_status"]["VALUE"]["remote_info"] = {"asic_id": "0x99", "eth_id": 4}
        entry = find(ingest.summarize([envelope(), a, b]), "clusterdbg_partner_disagreement")
        self.assertEqual(entry["status"], "FAIL")


class TestOutsideChannel(unittest.TestCase):
    def test_links_leaving_the_dump_never_fail(self):
        # One galaxy is collected, so every inter-galaxy cable lands here. It is
        # a measure of reach, not a fault.
        p = port("pa", "asic:0x11/eth10", f"glx={HOST}/ubb=1/asic=1/eth=10", remote={"asic_id": "0xbeef", "eth_id": 10})
        entry = find(ingest.summarize([envelope(), p]), "clusterdbg_outside_channel")
        self.assertEqual(entry["status"], "PASS")
        self.assertEqual(entry["data"]["count"], 1)
        self.assertFalse(entry.get("console_visible", True))


# ─────────────────────────────────────────────────────────────────────────────
# Inventory, collection and board findings
# ─────────────────────────────────────────────────────────────────────────────


class TestInventory(unittest.TestCase):
    def test_full_galaxy_passes(self):
        entry = find(ingest.summarize(full_galaxy_records()), "clusterdbg_inventory")
        self.assertEqual(entry["status"], "PASS")
        self.assertEqual(entry["data"]["counts"]["asic"], 32)
        self.assertEqual(entry["data"]["counts"]["eth_port"], 448)

    def test_short_count_fails(self):
        records = [envelope(), galaxy(), ubb(1), asic(1, 1)]
        entry = find(ingest.summarize(records), "clusterdbg_inventory")
        self.assertEqual(entry["status"], "FAIL")
        self.assertIn("ubb 1/4", entry["details"])
        self.assertIn("asic 1/32", entry["details"])

    def test_absent_slot_reasons_are_surfaced(self):
        records = [envelope(), galaxy(absent_ubbs=(2,)), ubb(1, absent_asics=(3,))]
        entry = find(ingest.summarize(records), "clusterdbg_inventory")
        self.assertEqual(entry["status"], "FAIL")
        self.assertTrue(any("UBB2" in a for a in entry["data"]["absent"]))
        self.assertTrue(any("UBB1/U3" in a for a in entry["data"]["absent"]))

    def test_malformed_lines_warn_without_claiming_missing_hardware(self):
        entry = find(ingest.summarize(full_galaxy_records(), malformed=2), "clusterdbg_inventory")
        self.assertEqual(entry["status"], "WARN")
        self.assertIn("2 unparseable line(s)", entry["details"])

    def test_identity_records_the_cross_snapshot_join_keys(self):
        entry = find(ingest.summarize(full_galaxy_records()), "clusterdbg_inventory")
        self.assertEqual(entry["data"]["identity"]["chassis_serial"], "QTWS7TKC260400005")
        self.assertEqual(entry["data"]["identity"]["ubb_uids"]["1"], "ubb:SER1")

    def test_a_dump_with_nothing_in_it_is_a_single_failure(self):
        checks = ingest.summarize([], malformed=3)
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0]["status"], "FAIL")


class TestCollectionStatus(unittest.TestCase):
    def test_hard_failures_fail_and_carry_their_reason(self):
        records = [envelope(), galaxy(), ubb(1), asic(1, 1, status="UNREACHABLE")]
        entry = find(ingest.summarize(records), "clusterdbg_collection_failures")
        self.assertEqual(entry["status"], "FAIL")
        self.assertIn("asic UNREACHABLE x1", entry["details"])
        self.assertEqual(entry["data"]["failures"][0]["reason"], "L1 read returned all-ones (dark tile)")

    def test_partial_and_skipped_are_lost_coverage_not_faults(self):
        records = [envelope(), galaxy(), ubb(1), cage(1, 1, status="SKIPPED")]
        checks = ingest.summarize(records)
        self.assertEqual(find(checks, "clusterdbg_collection_failures")["status"], "PASS")
        partial = find(checks, "clusterdbg_collection_partial")
        self.assertEqual(partial["status"], "WARN")
        self.assertIn("lost coverage", partial["details"])

    def test_harvested_is_not_a_collection_failure(self):
        records = [envelope(), port("h", "u", "glx=h/eth=5", harvested=True, status="HARVESTED")]
        self.assertEqual(find(ingest.summarize(records), "clusterdbg_collection_failures")["status"], "PASS")

    def test_envelope_carries_no_collection_status(self):
        self.assertEqual(find(ingest.summarize([envelope()]), "clusterdbg_collection_failures")["status"], "PASS")


class TestBoardRev(unittest.TestCase):
    def test_uniform_revision_passes(self):
        records = [envelope()] + [ubb(n) for n in range(1, 5)]
        entry = find(ingest.summarize(records), "clusterdbg_board_rev")
        self.assertEqual(entry["status"], "PASS")
        self.assertEqual(entry["data"]["rev"], "BH_GALAXY_REV_C")

    def test_mixed_revision_fails(self):
        records = [envelope(), ubb(1), ubb(2, rev="BH_GALAXY_REV_AB")]
        entry = find(ingest.summarize(records), "clusterdbg_board_rev")
        self.assertEqual(entry["status"], "FAIL")
        self.assertIn("mixed revisions", entry["details"])

    def test_asics_disagreeing_on_board_id_fails(self):
        # board_id can be reset to a uniform default, taking the revision with
        # it, and nothing else looks wrong when it happens.
        records = [envelope(), ubb(1, agreement=False)]
        entry = find(ingest.summarize(records), "clusterdbg_board_rev")
        self.assertEqual(entry["status"], "FAIL")
        self.assertIn("disagree on board_id", entry["details"])


class TestCagesAndModules(unittest.TestCase):
    def test_no_cages_skips(self):
        self.assertEqual(find(ingest.summarize([envelope()]), "clusterdbg_cage_gaps")["status"], "SKIP")

    def test_cages_without_an_expected_partner_skip(self):
        records = [envelope(), cage(1, 1), cage(1, 2, present=False)]
        entry = find(ingest.summarize(records), "clusterdbg_cage_gaps")
        self.assertEqual(entry["status"], "SKIP")
        self.assertIn("cannot be told from an unused one", entry["details"])

    def test_empty_cage_where_a_cable_was_expected_warns(self):
        records = [
            envelope(),
            cage(1, 1, partner="ubb:SER2/qsfp01"),
            cage(1, 2, present=False, partner="ubb:SER2/qsfp02"),
        ]
        entry = find(ingest.summarize(records), "clusterdbg_cage_gaps")
        self.assertEqual(entry["status"], "WARN")
        self.assertEqual(entry["data"]["gaps"][0]["gap"], "empty, a cable was expected")

    def test_unread_cage_is_not_an_empty_cage(self):
        records = [envelope(), cage(1, 1, partner="p"), cage(1, 2, present=False, status="READ_FAILED")]
        entry = find(ingest.summarize(records), "clusterdbg_cage_gaps")
        self.assertEqual(entry["status"], "PASS")

    def test_module_inventory_is_store_only(self):
        records = [envelope(), cage(1, 1, module_uid="mod:x"), module(1, 1)]
        entry = find(ingest.summarize(records), "clusterdbg_modules")
        self.assertEqual(entry["status"], "PASS")
        self.assertEqual(entry["data"]["modules"][0]["vendor_sn"], "APF23140099")
        self.assertFalse(entry.get("console_visible", True))


class TestEthCounters(unittest.TestCase):
    def test_counters_are_totalled_and_never_alert(self):
        records = [
            envelope(),
            port("a", "u1", "glx=h/ubb=1/asic=1/eth=0", retrain=2, corr_cw=10, uncorr_cw=1),
            port("b", "u2", "glx=h/ubb=1/asic=1/eth=1", retrain=3, corr_cw=5, uncorr_cw=0),
        ]
        entry = find(ingest.summarize(records), "clusterdbg_eth_counters")
        self.assertEqual(entry["status"], "PASS")
        self.assertEqual(entry["data"]["totals"], {"retrain_count": 5, "corr_cw": 15, "uncorr_cw": 1})
        self.assertEqual(entry["data"]["worst_uncorr_cw"][0]["path"], "ubb=1/asic=1/eth=0")


# ─────────────────────────────────────────────────────────────────────────────
# The payload as a whole
# ─────────────────────────────────────────────────────────────────────────────


class TestReportShape(unittest.TestCase):
    def test_every_check_matches_the_ingest_contract(self):
        report = ingest.build_report(full_galaxy_records())
        self.assertEqual(report["tool"], ingest.TOOL_NAME)
        self.assertTrue(report["checks"])
        for entry in report["checks"]:
            self.assertEqual(set(entry) - {"console_visible"}, {"name", "status", "details", "ip", "data"})
            self.assertTrue(entry["name"].startswith("clusterdbg_"))
            self.assertIn(entry["status"], ("PASS", "WARN", "FAIL", "SKIP"))
            # An ip outside the health check's groups would reach the JSON and
            # the CSV but never appear in the console summary.
            self.assertIn(entry["ip"], ("board", "pcie", "gddr", "eth", "asic", "fw", "thermal", "other"))
            self.assertIsInstance(entry["data"], dict)

    def test_names_are_unique_so_nothing_is_silently_overwritten(self):
        names = [c["name"] for c in ingest.build_report(full_galaxy_records())["checks"]]
        self.assertEqual(len(names), len(set(names)))

    def test_report_is_json_serializable(self):
        json.dumps(ingest.build_report(full_galaxy_records()), default=str)

    def test_healthy_dump_produces_no_failure(self):
        records = full_galaxy_records()
        # Swap two placeholder tiles for a real trained link rather than adding
        # them, so the dump still holds the 14 ports per ASIC a Galaxy has —
        # otherwise the inventory check rightly objects to the fixture.
        placeholders = [r for r in records if r["SCHEMA"] == "eth_port"][:2]
        for placeholder in placeholders:
            records.remove(placeholder)
        records += list(linked_pair())
        statuses = {c["name"]: c["status"] for c in ingest.summarize(records)}
        self.assertNotIn("FAIL", statuses.values(), f"unexpected failure in {statuses}")

    def test_rollup_precedence(self):
        self.assertEqual(ingest.rollup([{"status": "PASS"}, {"status": "FAIL"}, {"status": "WARN"}]), "FAIL")
        self.assertEqual(ingest.rollup([{"status": "PASS"}, {"status": "WARN"}]), "WARN")
        self.assertEqual(ingest.rollup([{"status": "SKIP"}, {"status": "PASS"}]), "PASS")
        self.assertEqual(ingest.rollup([{"status": "SKIP"}]), "SKIP")
        self.assertEqual(ingest.rollup([]), "SKIP")

    def test_render_lists_every_check(self):
        report = ingest.build_report(full_galaxy_records())
        text = ingest.render(report, source="dump.jsonl")
        for entry in report["checks"]:
            self.assertIn(entry["name"], text)


if __name__ == "__main__":
    unittest.main()
