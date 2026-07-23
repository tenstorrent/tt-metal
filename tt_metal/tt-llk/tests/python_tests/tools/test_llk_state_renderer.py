import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.llk_state_audit import (
    CSV_HEADERS,
    audit,
    generate,
    load_effect_model,
    render,
    verify,
)

REVISION = "3218270556c5ff58eba1e68d112d09a11b37dc0e"


def fixture_audit() -> dict:
    source = {
        "path": "tt_llk_wormhole_b0/llk_lib/example.h",
        "start_line": 10,
        "end_line": 12,
    }
    definition = {
        "architecture": "wormhole_b0",
        "name": "_llk_unpack_config_",
        "classification": "included",
        "reason": "reviewed persistent sink",
        "canonical_target": None,
        "source": source,
        "body_fingerprint": "a" * 64,
        "stability": "stable",
    }
    no_effect = {
        **definition,
        "name": "_llk_unpack_get_size_",
        "classification": "excluded",
        "reason": "read/wait-only",
        "source": {**source, "start_line": 20, "end_line": 21},
        "body_fingerprint": "b" * 64,
    }
    effect = {
        "architecture": "wormhole_b0",
        "thread": "T0",
        "stage": "unpack",
        "stability": "stable",
        "function": "_llk_unpack_config_",
        "alias_of": None,
        "lifecycle": "configure",
        "parameter": {"name": "format", "type": "std::uint32_t", "kind": "runtime"},
        "condition": None,
        "condition_kind": None,
        "domain": "cfg_register",
        "resource": "REG_A",
        "operation": "rmw",
        "value_expr": 'format, "quoted"\nnext line',
        "persistence": "persistent",
        "retention_contract": "Retained until the configuration resource is reconfigured.",
        "activation": "immediate",
        "restore": None,
        "confidence": "high",
        "evidence": {
            "kind": "direct",
            "token": "cfg_reg_rmw_tensix",
            "line": 11,
            "source_path": source["path"],
            "body_fingerprint": definition["body_fingerprint"],
            "via": None,
            "via_chain": [],
            "sink": {
                "token": "cfg_reg_rmw_tensix",
                "line": 11,
                "source_path": source["path"],
                "body_fingerprint": definition["body_fingerprint"],
            },
        },
    }
    return {
        "schema_version": 2,
        "inventory": {
            "schema_version": 1,
            "functions": [
                {
                    "name": definition["name"],
                    "architecture": definition["architecture"],
                    "thread": "T0",
                    "stage": "unpack",
                    "stability_tier": "stable",
                    "lifecycle": "configure",
                    "source": source,
                    "body_fingerprint": definition["body_fingerprint"],
                },
                {
                    "name": no_effect["name"],
                    "architecture": no_effect["architecture"],
                    "thread": "T0",
                    "stage": "unpack",
                    "stability_tier": "stable",
                    "lifecycle": "execute",
                    "source": no_effect["source"],
                    "body_fingerprint": no_effect["body_fingerprint"],
                },
            ],
        },
        "classification": [definition, no_effect],
        "effects": [effect],
        "summary": {
            "definition_count": 2,
            "effect_count": 1,
            "by_classification": {"excluded": 1, "included": 1},
            "by_architecture": {"wormhole_b0": {"definitions": 2, "effects": 1}},
        },
    }


class LlkStateRendererTest(unittest.TestCase):
    def test_default_generation_command_matches_readme_invocation(self) -> None:
        rendered = render(fixture_audit(), revision=REVISION)
        self.assertEqual(
            json.loads(rendered["json"])["generation_command"],
            "python3 -m tools.llk_state_audit generate --root .",
        )

    def test_render_has_canonical_json_and_exact_csv_headers(self) -> None:
        fingerprint = "f" * 64
        rendered = render(
            fixture_audit(),
            revision=REVISION,
            source_fingerprint=fingerprint,
            command="generate --fixture",
        )
        self.assertEqual(
            rendered["json"],
            render(
                fixture_audit(),
                revision=REVISION,
                source_fingerprint=fingerprint,
                command="generate --fixture",
            )["json"],
        )
        data = json.loads(rendered["json"])
        self.assertEqual(data["generator_base_commit"], REVISION)
        self.assertEqual(data["analyzed_source_fingerprint"], fingerprint)
        self.assertNotIn("analyzed_commit", data)
        self.assertEqual(data["generator"], {"schema_version": 2, "version": "2.0"})
        self.assertEqual(data["generation_command"], "generate --fixture")
        rows = list(csv.reader(rendered["csv"].splitlines()))
        self.assertEqual(rows[0], CSV_HEADERS)
        self.assertEqual(len(rows), 3)
        self.assertIn("Retention Contract", CSV_HEADERS)
        self.assertIn("Generator Base Commit", CSV_HEADERS)
        self.assertIn("Analyzed Source Fingerprint", CSV_HEADERS)
        self.assertEqual(
            rows[1][CSV_HEADERS.index("Retention Contract")],
            "Retained until the configuration resource is reconfigured.",
        )
        self.assertEqual(
            rows[1][CSV_HEADERS.index("Analyzed Source Fingerprint")],
            fingerprint,
        )

    def test_csv_is_rfc_quoted_and_preserves_classification_only_rows(self) -> None:
        rendered = render(fixture_audit(), revision=REVISION)
        self.assertIn('"format, ""quoted""\nnext line"', rendered["csv"])
        rows = list(csv.DictReader(rendered["csv"].splitlines()))
        excluded = next(
            row for row in rows if row["Function"] == "_llk_unpack_get_size_"
        )
        self.assertEqual(excluded["Mapped Status"], "classification-only")
        self.assertEqual(excluded["Classification"], "excluded")
        self.assertEqual(excluded["Notes"], "excluded: read/wait-only")
        self.assertNotIn("None", rendered["csv"])
        self.assertNotIn("null", rendered["csv"])

    def test_overload_identity_and_every_definition_coverage_are_preserved(
        self,
    ) -> None:
        data = fixture_audit()
        overload = json.loads(json.dumps(data))
        overload["inventory"]["functions"][1]["name"] = "_llk_unpack_config_"
        overload["classification"][1]["name"] = "_llk_unpack_config_"
        rendered = render(overload, revision=REVISION)
        rows = list(csv.DictReader(rendered["csv"].splitlines()))
        self.assertEqual({row["Source Line"] for row in rows}, {"11", "20"})
        self.assertEqual(len(rows), len(overload["classification"]))

    def test_generation_is_byte_for_byte_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "docs"
            first = generate(
                root, output_dir=output, revision=REVISION, audit_data=fixture_audit()
            )
            json_bytes = first["json_path"].read_bytes()
            csv_bytes = first["csv_path"].read_bytes()
            second = generate(
                root, output_dir=output, revision=REVISION, audit_data=fixture_audit()
            )
            self.assertEqual(json_bytes, second["json_path"].read_bytes())
            self.assertEqual(csv_bytes, second["csv_path"].read_bytes())

    def test_readme_does_not_depend_on_git_history_date(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = generate(
                root,
                output_dir=root / "docs",
                revision=REVISION,
                audit_data=fixture_audit(),
            )
            readme = generated["readme_path"].read_text(encoding="utf-8")
        self.assertNotIn("Generator base date", readme)

    def test_verify_succeeds_for_deterministically_generated_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "docs"
            generate(
                root, output_dir=output, revision=REVISION, audit_data=fixture_audit()
            )

            result = verify(
                root, output_dir=output, revision=REVISION, audit_data=fixture_audit()
            )

        self.assertTrue(result["valid"])
        self.assertEqual(result["mismatches"], [])
        self.assertEqual(result["summary"], fixture_audit()["summary"])

    def test_verify_reports_tampered_artifact_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "docs"
            generated = generate(
                root, output_dir=output, revision=REVISION, audit_data=fixture_audit()
            )
            original_csv = generated["csv_path"].read_bytes()
            generated["csv_path"].write_text("tampered\n", encoding="utf-8")

            result = verify(
                root, output_dir=output, revision=REVISION, audit_data=fixture_audit()
            )

            self.assertEqual(result["mismatches"], ["llk_state_map.csv"])
            self.assertEqual(generated["csv_path"].read_bytes(), b"tampered\n")
            self.assertNotEqual(original_csv, generated["csv_path"].read_bytes())

    def test_activation_column_is_rendered_for_effect_rows(self) -> None:
        self.assertIn("Activation", CSV_HEADERS)
        rendered = render(fixture_audit(), revision=REVISION)
        rows = list(csv.DictReader(rendered["csv"].splitlines()))
        effect_row = next(row for row in rows if row["Mapped Status"] == "effect")
        self.assertEqual(effect_row["Activation"], "immediate")
        classification_row = next(
            row for row in rows if row["Mapped Status"] == "classification-only"
        )
        self.assertEqual(classification_row["Activation"], "")

    def test_effect_rows_require_activation_instead_of_rendering_blank(self) -> None:
        data = fixture_audit()
        del data["effects"][0]["activation"]
        with self.assertRaises(KeyError):
            render(data, revision=REVISION)

    def test_restore_owner_pair_is_deduplicated(self) -> None:
        data = fixture_audit()
        data["effects"][0]["restore"] = {
            "kind": "canonical_reset",
            "owner": "_llk_unpack_clear_",
            "pair": "_llk_unpack_clear_",
        }
        row = next(csv.DictReader(render(data, revision=REVISION)["csv"].splitlines()))
        self.assertEqual(row["Restore Owner/Pair"], "_llk_unpack_clear_")


class RealRestoreContractRegressionTest(unittest.TestCase):
    def test_helper_mapped_blackhole_pack_init_has_reviewed_no_op_contract(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[3]
        contracts = load_effect_model(root=root)["restore_contracts"]
        rows = list(
            csv.DictReader(
                render(audit(root), revision=REVISION, restore_contracts=contracts)[
                    "csv"
                ].splitlines()
            )
        )
        row = next(
            row
            for row in rows
            if row["Architecture"] == "blackhole"
            and row["Function"] == "_llk_pack_init_"
            and row["Source Path"] == "tt_llk_blackhole/llk_lib/llk_pack.h"
            and row["Body Fingerprint"]
            == "4ab8e714bc97ec7335e78f5f17eba2165bbe467cb7934da26d03f736cf750e54"
        )
        self.assertEqual(row["Mapped Status"], "effect")
        self.assertEqual(row["Restore Contract Kind"], "no_op_transient")
        self.assertEqual(row["Restore Owner/Pair"], "_llk_pack_uninit_")
        self.assertIn("reviewed restore contract: no_op_transient", row["Notes"])


if __name__ == "__main__":
    unittest.main()
