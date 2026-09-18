"""Saved native serializer snapshots, with no model, native library or device imports."""

import copy
import hashlib
import json
import unittest
from pathlib import Path

from cancel_restart_contract import check_cancelled_pair

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"


class SerializedCancellationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        provenance = json.loads((FIXTURES / "provenance.json").read_bytes())
        cls.saved = {}
        for role, row in provenance["fixtures"].items():
            raw = (HERE / row["fixture"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise AssertionError("Saved native fixture changed: " + role)
            cls.saved[role] = json.loads(raw)
        cls.uuid = provenance["uuid"]

    def pair(self):
        return copy.deepcopy(self.saved["source"]), copy.deepcopy(self.saved["passive"])

    # The native passive serializer has no aggregate successful field; its exact
    # INTERNAL/0 generation and callback are the cancellation evidence.
    def test_actual_serialized_pair_without_passive_aggregate(self):
        source, passive = self.pair()
        self.assertNotIn("successful", passive)
        check_cancelled_pair(source, passive, self.uuid)

    # A successful landing cannot be relabelled as the intentionally cancelled inbound.
    def test_passive_ok_cannot_satisfy_cancelled_pair(self):
        source, passive = self.pair()
        passive["inbound_generations"][0]["status"] = 0
        passive["calls"][0]["status"] = 0
        with self.assertRaises(ValueError):
            check_cancelled_pair(source, passive, self.uuid)

    # Zero transferred tokens is independently required in both passive records
    # and the source-local completion, not inferred from a cancellation flag.
    def test_nonzero_tokens_rejected_in_each_completion_record(self):
        for location in ("inbound", "passive_callback", "source_callback"):
            with self.subTest(location=location):
                source, passive = self.pair()
                record = (
                    passive["inbound_generations"][0]
                    if location == "inbound"
                    else passive["calls"][0]
                    if location == "passive_callback"
                    else source["calls"][-1]
                )
                record["tokens"] = 1
                with self.assertRaises(ValueError):
                    check_cancelled_pair(source, passive, self.uuid)

    # A completion from another transfer cannot validate the selected inbound generation.
    def test_passive_transfer_identity_mismatch_rejected(self):
        for location in ("inbound_generations", "calls"):
            with self.subTest(location=location):
                source, passive = self.pair()
                passive[location][0]["transfer"] += 1
                with self.assertRaises(ValueError):
                    check_cancelled_pair(source, passive, self.uuid)

    # The passive generation and callback must each carry INTERNAL, even if the other agrees.
    def test_wrong_passive_status_rejected_per_record(self):
        for location in ("inbound_generations", "calls"):
            for status in (0, 1, 5, None):
                with self.subTest(location=location, status=status):
                    source, passive = self.pair()
                    passive[location][0]["status"] = status
                    with self.assertRaises(ValueError):
                        check_cancelled_pair(source, passive, self.uuid)

    # Missing or duplicated completion records must not collapse to one apparent success.
    def test_passive_callback_inventory_is_exact(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate):
                source, passive = self.pair()
                passive["calls"] = passive["calls"] * (2 if duplicate else 0)
                with self.assertRaises(ValueError):
                    check_cancelled_pair(source, passive, self.uuid)

    # Cancelling one exact generation does not permit a retired, incomplete,
    # wrong-UUID or differently bounded inbound to satisfy the terminal gate.
    def test_passive_generation_state_and_identity_remain_strict(self):
        for key, value in (
            ("retired", True),
            ("cancelled", False),
            ("complete", False),
            ("uuid", 701),
            ("from", 32),
            ("to", 1024),
        ):
            with self.subTest(key=key):
                source, passive = self.pair()
                passive["inbound_generations"][0][key] = value
                with self.assertRaises(ValueError):
                    check_cancelled_pair(source, passive, self.uuid)

    # Source aggregate status, callback, terminal position and pin release remain
    # mandatory; fixing the passive schema cannot relax the source contract.
    def test_source_terminal_and_claims_remain_strict(self):
        for field in ("successful", "acks", "terminal_position", "cancelled", "pins", "status", "transfer"):
            with self.subTest(field=field):
                source, passive = self.pair()
                if field == "successful":
                    source["successful"] = True
                elif field == "acks":
                    source["acks"] = 31
                elif field in ("terminal_position", "cancelled"):
                    source["generations"][0][field] = 0
                elif field == "pins":
                    slot = next(row for row in source["slots"] if row["slot"] == source["generations"][0]["slot"])
                    slot["pins"] = 1
                elif field == "status":
                    source["calls"][-1]["status"] = 6
                else:
                    source["calls"][-1]["transfer"] += 1
                with self.assertRaises(ValueError):
                    check_cancelled_pair(source, passive, self.uuid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
