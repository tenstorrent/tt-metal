import csv
import json
import re
import unittest
from pathlib import Path

from tools.llk_state_audit import (
    AuditModelError,
    audit,
    inventory,
    load_verification_manifest,
    validate_verification_manifest,
    verify,
)

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "tools" / "llk_state_audit" / "verification_manifest.json"
CSV_PATH = ROOT / "docs" / "llk_state_audit" / "llk_state_map.csv"


class LlkStateCoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = audit(ROOT)
        cls.inventory = inventory(ROOT)
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_every_definition_is_classified_and_represented_in_csv(self) -> None:
        definitions = self.audit["classification"]
        identities = {
            (
                item["architecture"],
                item["name"],
                item["source"]["path"],
                item["source"]["start_line"],
                item["body_fingerprint"],
            )
            for item in definitions
        }
        self.assertEqual(len(identities), len(definitions))

        with CSV_PATH.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        csv_identities = {
            (
                row["Architecture"],
                row["Function"],
                row["Source Path"],
                (
                    int(row["Source Line"])
                    if row["Mapped Status"] == "classification-only"
                    else None
                ),
                row["Body Fingerprint"],
            )
            for row in rows
        }
        for item in definitions:
            prefix = (
                item["architecture"],
                item["name"],
                item["source"]["path"],
            )
            self.assertTrue(
                any(
                    identity[:3] == prefix and identity[4] == item["body_fingerprint"]
                    for identity in csv_identities
                ),
                item["name"],
            )

    def test_every_inventory_sink_has_a_normalized_direct_effect(self) -> None:
        direct_effects = {
            (
                effect["architecture"],
                effect["function"],
                effect["evidence"]["source_path"],
                effect["evidence"]["line"],
                effect["domain"],
                effect["operation"],
            )
            for effect in self.audit["effects"]
            if effect["evidence"]["kind"] == "direct"
        }
        for function in self.inventory["functions"]:
            for sink in function["state_sinks"]:
                identity = (
                    function["architecture"],
                    function["name"],
                    function["source"]["path"],
                    sink["evidence"]["line"],
                    sink["domain"],
                    sink["operation"],
                )
                self.assertIn(identity, direct_effects)

    def test_effect_resources_are_nonblank_and_not_raw_numeric_selectors(
        self,
    ) -> None:
        for effect in self.audit["effects"]:
            resource = effect["resource"]
            self.assertTrue(resource and resource != "-", msg=effect)
            self.assertIsNone(
                re.fullmatch(
                    r"(?:0[xX][0-9A-Fa-f]+|0[bB][01]+|\d+)",
                    resource,
                ),
                msg=effect,
            )

    def test_verification_manifest_paths_and_functions_resolve(self) -> None:
        self.assertEqual(load_verification_manifest(ROOT), self.manifest)
        self.assertEqual(self.manifest["schema_version"], 1)
        functions = {
            (item["architecture"], item["name"])
            for item in self.audit["inventory"]["functions"]
        }
        required_scopes = {
            "runtime_reconfigure",
            "restore",
            "canonical_baseline",
            "synchronization",
        }
        seen_scopes = set()
        for entry in self.manifest["tests"]:
            self.assertTrue((ROOT / entry["path"]).is_file(), entry["path"])
            self.assertTrue(entry["architectures"])
            self.assertTrue(entry["functions"])
            self.assertTrue(entry["domains"])
            self.assertIn(entry["validation_scope"], required_scopes)
            seen_scopes.add(entry["validation_scope"])
            for architecture in entry["architectures"]:
                for function in entry["functions"]:
                    self.assertIn((architecture, function), functions)
        self.assertEqual(seen_scopes, required_scopes)

    def test_verification_manifest_rejects_unknown_scope(self) -> None:
        malformed = json.loads(json.dumps(self.manifest))
        malformed["tests"][0]["validation_scope"] = "not_reviewed"
        temporary = ROOT / "tools" / "llk_state_audit" / ".invalid_manifest.json"
        temporary.write_text(json.dumps(malformed), encoding="utf-8")
        try:
            with self.assertRaises(AuditModelError):
                load_verification_manifest(ROOT, temporary)
        finally:
            temporary.unlink()

    def test_verification_manifest_functions_and_domains_are_audit_backed(
        self,
    ) -> None:
        validate_verification_manifest(self.manifest, self.audit)

    def test_high_risk_families_are_covered_per_architecture(self) -> None:
        covered = {
            (architecture, family)
            for entry in self.manifest["tests"]
            for architecture in entry["architectures"]
            for family in entry["families"]
        }
        required = {
            "math_configure",
            "unpack_configure",
            "unpack_init_or_restore",
            "pack_configure",
            "pack_init_or_restore",
            "mop_or_addrmod",
        }
        for architecture in ("wormhole_b0", "blackhole", "quasar"):
            self.assertTrue(
                required.issubset(
                    {family for arch, family in covered if arch == architecture}
                ),
                architecture,
            )
        self.assertIn(("quasar", "sync_clear_done"), covered)

    def test_generated_artifacts_match_current_sources(self) -> None:
        result = verify(ROOT)
        self.assertTrue(result["valid"], result["mismatches"])


if __name__ == "__main__":
    unittest.main()
