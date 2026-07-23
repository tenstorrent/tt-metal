import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.llk_state_audit import AuditModelError, inventory, load_effect_model

FIXTURE_HEADER = """\
template <int Faces, typename Format>
inline void _llk_unpack_init_(const std::uint32_t format, bool enable = true)
{
    cfg_reg_rmw_tensix<REG_FORMAT>(format);
    TTI_WRCFG(p_gpr_unpack::L1_BUFFER_ADDR, p_cfg::WRCFG_32b, REG_BASE);
    TTI_WRCFG(p_gpr_unpack::L1_BUFFER_ADDR, p_cfg::WRCFG_32b, REG_BASE_2);
    TTI_SETADCXX(p_setadc::UNP0, 15, 0);
    TTI_SETRWC(p_rwc::SET, 4);
    TTI_SETADDRMOD(p_addrmod::UNP0, 1);
    ckernel_template mop(1, 1, 0);
    mop.program();
    TTI_SETDVALID(0b10);
    TTI_CLEARDVALID(0b10, 0);
    semaphore_post(semaphore::UNPACK_SYNC);
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    set_dst_write_addr(3);
    TTI_SETGPR(p_gpr::R0, format);
    shadow_format = format;
}

inline void _llk_unpack_wrapper_(const std::uint32_t format)
{
    _llk_unpack_init_<4, DataFormat>(format);
}

inline void _llk_unpack_run_(const std::uint32_t address)
{
    _llk_unpack_wrapper_(address);
    semaphore_post(semaphore::UNPACK_SYNC);
}
"""


class LlkStateAuditTest(unittest.TestCase):
    def write_fixture(
        self,
        root: Path,
        architecture: str = "wormhole_b0",
        relative: str = "llk_lib/llk_unpack.h",
    ) -> Path:
        path = root / f"tt_llk_{architecture}" / relative
        path.parent.mkdir(parents=True)
        path.write_text(FIXTURE_HEADER, encoding="utf-8")
        return path

    def test_discovers_definitions_templates_parameters_and_source_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_fixture(root)
            records = inventory(root)

        init = next(
            record
            for record in records["functions"]
            if record["name"] == "_llk_unpack_init_"
        )
        self.assertEqual(init["architecture"], "wormhole_b0")
        self.assertEqual(init["thread"], "T0")
        self.assertEqual(init["stage"], "unpack")
        self.assertEqual(init["template_parameters"], ["int Faces", "typename Format"])
        self.assertEqual(
            init["runtime_parameters"],
            ["const std::uint32_t format", "bool enable = true"],
        )
        self.assertEqual(init["source"]["start_line"], 1)
        self.assertGreater(init["source"]["end_line"], init["source"]["start_line"])
        self.assertEqual(
            init["source"]["path"], "tt_llk_wormhole_b0/llk_lib/llk_unpack.h"
        )

    def test_thread_and_stage_use_hardware_and_pipeline_labels(self) -> None:
        source = """\
inline void _llk_unpack_run_() {}
inline void _llk_math_run_() {}
inline void _llk_pack_run_() {}
inline void _llk_sync_wait_() {}
inline void _llk_misc_run_() {}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "tt_llk_quasar" / "llk_lib" / "labels.h"
            path.parent.mkdir(parents=True)
            path.write_text(source, encoding="utf-8")
            records = {
                record["name"]: record for record in inventory(root)["functions"]
            }

        self.assertEqual(
            (
                records["_llk_unpack_run_"]["thread"],
                records["_llk_unpack_run_"]["stage"],
            ),
            ("T0", "unpack"),
        )
        self.assertEqual(
            (records["_llk_math_run_"]["thread"], records["_llk_math_run_"]["stage"]),
            ("T1", "math"),
        )
        self.assertEqual(
            (records["_llk_pack_run_"]["thread"], records["_llk_pack_run_"]["stage"]),
            ("T2", "pack"),
        )
        self.assertEqual(
            (records["_llk_sync_wait_"]["thread"], records["_llk_sync_wait_"]["stage"]),
            ("shared", "shared"),
        )
        self.assertEqual(
            (records["_llk_misc_run_"]["thread"], records["_llk_misc_run_"]["stage"]),
            ("unknown", "unknown"),
        )

    def test_classifies_lifecycle_stability_stage_and_wrapper_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_fixture(root)
            records = inventory(root)

        by_name = {record["name"]: record for record in records["functions"]}
        self.assertEqual(by_name["_llk_unpack_init_"]["lifecycle"], "init")
        self.assertEqual(by_name["_llk_unpack_init_"]["stability_tier"], "stable")
        self.assertEqual(by_name["_llk_unpack_wrapper_"]["lifecycle"], "wrapper")
        self.assertEqual(
            by_name["_llk_unpack_wrapper_"]["canonical_target"], "_llk_unpack_init_"
        )
        self.assertEqual(by_name["_llk_unpack_run_"]["lifecycle"], "execute")
        self.assertEqual(
            by_name["_llk_unpack_run_"]["direct_calls"], ["_llk_unpack_wrapper_"]
        )

    def test_normalizes_all_seed_state_sink_domains_with_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_fixture(root)
            init = next(
                record
                for record in inventory(root)["functions"]
                if record["name"] == "_llk_unpack_init_"
            )

        sinks = {(sink["domain"], sink["operation"]) for sink in init["state_sinks"]}
        self.assertTrue(
            {
                ("cfg_register", "rmw"),
                ("cfg_register", "write"),
                ("counter", "adc"),
                ("counter", "rwc"),
                ("addr_mod", "program"),
                ("mop_replay", "program"),
                ("dvalid_semaphore", "post"),
                ("dvalid_semaphore", "get"),
                ("dvalid_semaphore", "set"),
                ("dvalid_semaphore", "clear"),
                ("destination", "select"),
                ("gpr", "write"),
                ("software_shadow", "assign"),
            }.issubset(sinks)
        )
        self.assertTrue(
            all(
                sink["evidence"]["line"] > 0 and sink["evidence"]["token"]
                for sink in init["state_sinks"]
            )
        )
        self.assertEqual(
            sum(
                sink["domain"] == "cfg_register" and sink["operation"] == "write"
                for sink in init["state_sinks"]
            ),
            2,
        )

    def test_sink_evidence_reports_exact_source_line(self) -> None:
        source = """\
// leading line
inline void _llk_pack_init_()
{
    TTI_NOP;
    TTI_WRCFG(p_gpr::R0, p_cfg::WRCFG_32b, REG);
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "tt_llk_quasar" / "llk_lib" / "llk_pack.h"
            path.parent.mkdir(parents=True)
            path.write_text(source, encoding="utf-8")
            record = inventory(root)["functions"][0]

        sink = next(
            sink for sink in record["state_sinks"] if sink["operation"] == "write"
        )
        self.assertEqual(sink["evidence"]["line"], 5)

    def test_comments_and_literals_do_not_create_calls_or_sinks(self) -> None:
        source = """\
inline void _llk_math_scan_()
{
    // _llk_comment_call_(); TTI_WRCFG(comment);
    /* _llk_block_call_();
       TTI_SETADCXX(block); */
    const char *text = "_llk_string_call_(); TTI_SETRWC(string);";
    const auto marker = '_llk_char_call_';
    _llk_math_real_();
    TTI_WRCFG(p_gpr::R0, p_cfg::WRCFG_32b, REG);
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "tt_llk_quasar" / "llk_lib" / "llk_math.h"
            path.parent.mkdir(parents=True)
            path.write_text(source, encoding="utf-8")
            record = inventory(root)["functions"][0]

        self.assertEqual(record["direct_calls"], ["_llk_math_real_"])
        self.assertEqual(
            [
                (sink["domain"], sink["operation"], sink["evidence"]["line"])
                for sink in record["state_sinks"]
            ],
            [("cfg_register", "write", 9)],
        )

    def test_nested_template_expression_call_is_discovered(self) -> None:
        source = """\
inline void _llk_unpack_scan_()
{
    // _llk_comment_<Ignored, nested(false)>();
    const char *text = "_llk_string_<Ignored, nested(false)>()";
    _llk_unpack_srcs_config_<
        INSTRN_COUNT,
        srcs_dims::slice_count(true)>();
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "tt_llk_quasar" / "llk_lib" / "llk_unpack.h"
            path.parent.mkdir(parents=True)
            path.write_text(source, encoding="utf-8")
            record = inventory(root)["functions"][0]

        self.assertEqual(record["direct_calls"], ["_llk_unpack_srcs_config_"])

    def test_shadow_comparison_is_not_an_assignment_sink(self) -> None:
        source = """\
inline void _llk_math_init_()
{
    if (shadow_format == value) {
        shadow_format = value;
    }
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "tt_llk_blackhole" / "llk_lib" / "llk_math.h"
            path.parent.mkdir(parents=True)
            path.write_text(source, encoding="utf-8")
            record = inventory(root)["functions"][0]

        assignments = [
            sink
            for sink in record["state_sinks"]
            if sink["domain"] == "software_shadow"
        ]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0]["evidence"]["line"], 4)
        self.assertEqual(assignments[0]["evidence"]["token"], "shadow_format =")

    def test_addrmod_constants_are_not_programming_sinks(self) -> None:
        source = """\
inline void _llk_unpack_addrmod_()
{
    constexpr auto ADDRMOD = 1;
    consume(ADDRMOD);
    TTI_SETADDRMOD(p_addrmod::UNP0, 1);
    addr_mod_t::set(2);
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "tt_llk_wormhole_b0" / "llk_lib" / "llk_unpack.h"
            path.parent.mkdir(parents=True)
            path.write_text(source, encoding="utf-8")
            record = inventory(root)["functions"][0]

        sinks = [sink for sink in record["state_sinks"] if sink["domain"] == "addr_mod"]
        self.assertEqual(
            [(sink["evidence"]["token"], sink["evidence"]["line"]) for sink in sinks],
            [("TTI_SETADDRMOD", 5), ("addr_mod_t::set", 6)],
        )

    def test_chained_addrmod_set_is_a_programming_sink(self) -> None:
        source = """\
inline void _llk_pack_addrmod_()
{
    constexpr auto ADDR_MOD_2 = 2;
    consume(ADDR_MOD_2);
    addr_mod_pack_t {
        .y_src = {.incr = 6},
    }
        .set(ADDR_MOD_2);
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "tt_llk_wormhole_b0" / "llk_lib" / "llk_pack.h"
            path.parent.mkdir(parents=True)
            path.write_text(source, encoding="utf-8")
            record = inventory(root)["functions"][0]

        sinks = [sink for sink in record["state_sinks"] if sink["domain"] == "addr_mod"]
        self.assertEqual(
            [(sink["evidence"]["token"], sink["evidence"]["line"]) for sink in sinks],
            [(".set(ADDR_MOD_2)", 8)],
        )

    def test_fingerprint_detects_body_drift_and_output_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.write_fixture(root)
            first = inventory(root)
            second = inventory(root)
            self.assertEqual(
                json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True)
            )
            original = next(
                record
                for record in first["functions"]
                if record["name"] == "_llk_unpack_run_"
            )["body_fingerprint"]
            path.write_text(
                FIXTURE_HEADER.replace("_llk_unpack_wrapper_(address);", "TTI_NOP;"),
                encoding="utf-8",
            )
            changed = inventory(root)

        replacement = next(
            record
            for record in changed["functions"]
            if record["name"] == "_llk_unpack_run_"
        )["body_fingerprint"]
        self.assertNotEqual(original, replacement)

    def test_fingerprint_ignores_comments_but_preserves_string_literals(self) -> None:
        source = """\
inline void _llk_math_fingerprint_()
{
    const char *url = "https://one";
    TTI_NOP; // first comment
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "tt_llk_blackhole" / "llk_lib" / "llk_math.h"
            path.parent.mkdir(parents=True)
            path.write_text(source, encoding="utf-8")
            original = inventory(root)["functions"][0]["body_fingerprint"]
            path.write_text(
                source.replace("first comment", "changed comment"), encoding="utf-8"
            )
            comment_changed = inventory(root)["functions"][0]["body_fingerprint"]
            path.write_text(
                source.replace("https://one", "https://two"), encoding="utf-8"
            )
            string_changed = inventory(root)["functions"][0]["body_fingerprint"]

        self.assertEqual(original, comment_changed)
        self.assertNotEqual(original, string_changed)

    def test_source_range_ignores_braces_in_comments_and_strings(self) -> None:
        source = """\
inline void _llk_math_run_()
{
    // } must not end the function
    const char *note = "{ not a scope";
    TTI_SETDVALID(1);
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "tt_llk_blackhole" / "llk_lib" / "llk_math.h"
            path.parent.mkdir(parents=True)
            path.write_text(source, encoding="utf-8")
            records = inventory(root)["functions"]

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source"]["end_line"], 6)
        self.assertEqual(records[0]["state_sinks"][0]["operation"], "set")

    def test_numeric_digit_separator_is_not_treated_as_character_literal(self) -> None:
        source = """\
inline void _llk_unpack_first_()
{
    constexpr auto mode = 0b00'01;
}
inline void _llk_unpack_second_()
{
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "tt_llk_blackhole" / "llk_lib" / "llk_unpack.h"
            path.parent.mkdir(parents=True)
            path.write_text(source, encoding="utf-8")
            records = inventory(root)["functions"]

        self.assertEqual(
            [record["name"] for record in records],
            ["_llk_unpack_first_", "_llk_unpack_second_"],
        )

    def test_discovers_nested_templates_and_comma_bearing_return_type(self) -> None:
        source = """\
template <
    typename T,
    typename Guard = std::enable_if_t<
        std::is_same_v<T, std::array<int, 4>>>>
inline std::array<std::pair<int, int>, 2> _llk_math_nested_(
    const std::array<int, 4>& values)
{
    return {};
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "tt_llk_quasar" / "llk_lib" / "llk_math_nested.h"
            path.parent.mkdir(parents=True)
            path.write_text(source, encoding="utf-8")
            records = inventory(root)["functions"]

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["name"], "_llk_math_nested_")
        self.assertEqual(
            records[0]["template_parameters"],
            [
                "typename T",
                "typename Guard = std::enable_if_t< std::is_same_v<T, std::array<int, 4>>>",
            ],
        )
        self.assertEqual(
            records[0]["runtime_parameters"],
            ["const std::array<int, 4>& values"],
        )
        self.assertIn("std::array<std::pair<int, int>, 2>", records[0]["signature"])

    def test_validates_effect_model_schema_and_rejects_unknown_domains(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "effects.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "effects": [
                            {
                                "pattern": "TTI_WRCFG",
                                "domain": "cfg_register",
                                "operation": "write",
                                "retention": "Retained until reconfigured.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(load_effect_model(path)["schema_version"], 1)
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "effects": [
                            {
                                "pattern": "bad",
                                "domain": "unknown",
                                "operation": "write",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(AuditModelError):
                load_effect_model(path)

    def test_rejects_effect_pattern_without_reviewed_retention(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "effects.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "effects": [
                            {
                                "pattern": "TTI_WRCFG",
                                "domain": "cfg_register",
                                "operation": "write",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AuditModelError, "retention"):
                load_effect_model(path)

    def test_rejects_unknown_effect_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "effects.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "effects": [
                            {
                                "pattern": "TTI_WRCFG",
                                "domain": "cfg_register",
                                "operation": "write",
                                "retention": "Retained until reconfigured.",
                                "persistence": "persistent_control",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AuditModelError, "persistence"):
                load_effect_model(path)

    def test_validates_restore_contract_schema_and_source_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "tt_llk_blackhole" / "llk_lib" / "llk_pack.h"
            source.parent.mkdir(parents=True)
            source.write_text(
                "inline void _llk_pack_init_() { TTI_WRCFG(0, 0, REG); }\n",
                encoding="utf-8",
            )
            model = root / "effects.json"
            contract = {
                "architecture": "blackhole",
                "function": "_llk_pack_init_",
                "kind": "no_op_transient",
                "owner": None,
                "pair": None,
                "owner_source": None,
                "pair_source": None,
                "rationale": "Documented no-op teardown.",
                "source": {
                    "path": "tt_llk_blackhole/llk_lib/llk_pack.h",
                    "line": 1,
                    "token": "_llk_pack_init_",
                    "body_fingerprint": inventory(root)["functions"][0][
                        "body_fingerprint"
                    ],
                },
            }
            model.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "effects": [
                            {
                                "pattern": "TTI_WRCFG",
                                "domain": "cfg_register",
                                "operation": "write",
                                "retention": "Retained until reconfigured.",
                            }
                        ],
                        "restore_contracts": [contract],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                load_effect_model(model, root=root)["restore_contracts"], [contract]
            )
            stale = dict(contract)
            stale["source"] = {**contract["source"], "body_fingerprint": "0" * 64}
            model_data = json.loads(model.read_text(encoding="utf-8"))
            model_data["restore_contracts"] = [stale]
            model.write_text(json.dumps(model_data), encoding="utf-8")
            with self.assertRaisesRegex(AuditModelError, "body fingerprint"):
                load_effect_model(model, root=root)
            command = [
                sys.executable,
                "-m",
                "tools.llk_state_audit",
                "check",
                "--root",
                str(root),
                "--effects",
                str(model),
            ]
            checked = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[3],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(checked.returncode, 0)
            self.assertIn("body fingerprint", checked.stderr)
            del contract["rationale"]
            model.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "effects": [
                            {
                                "pattern": "TTI_WRCFG",
                                "domain": "cfg_register",
                                "operation": "write",
                                "retention": "Retained until reconfigured.",
                            }
                        ],
                        "restore_contracts": [contract],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AuditModelError, "restore contract"):
                load_effect_model(model)

    def test_rejects_duplicate_and_nonexistent_restore_contract_references(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "tt_llk_blackhole" / "llk_lib" / "llk_pack.h"
            path.parent.mkdir(parents=True)
            path.write_text(
                "inline void _llk_pack_init_() { TTI_WRCFG(0, 0, REG); }\n"
                "inline void _llk_pack_uninit_() {}\n",
                encoding="utf-8",
            )
            init = next(
                function
                for function in inventory(root)["functions"]
                if function["name"] == "_llk_pack_init_"
            )
            contract = {
                "architecture": "blackhole",
                "function": "_llk_pack_init_",
                "kind": "no_op_transient",
                "owner": None,
                "pair": "_llk_pack_uninit_",
                "owner_source": None,
                "rationale": "Fixture contract.",
                "source": {
                    "path": init["source"]["path"],
                    "line": init["source"]["start_line"],
                    "token": "_llk_pack_init_",
                    "body_fingerprint": init["body_fingerprint"],
                },
            }
            uninit = next(
                function
                for function in inventory(root)["functions"]
                if function["name"] == "_llk_pack_uninit_"
            )
            contract["pair_source"] = {
                "path": uninit["source"]["path"],
                "line": uninit["source"]["start_line"],
                "token": "_llk_pack_uninit_",
                "body_fingerprint": uninit["body_fingerprint"],
            }
            model = root / "effects.json"
            base = {
                "schema_version": 1,
                "effects": [
                    {
                        "pattern": "TTI_WRCFG",
                        "domain": "cfg_register",
                        "operation": "write",
                        "retention": "Retained until reconfigured.",
                    }
                ],
                "restore_contracts": [contract],
            }
            model.write_text(
                json.dumps({**base, "restore_contracts": [contract, contract]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AuditModelError, "unique"):
                load_effect_model(model, root=root)
            missing_owner = {**contract, "owner": "_llk_pack_missing_"}
            model.write_text(
                json.dumps({**base, "restore_contracts": [missing_owner]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AuditModelError, "owner"):
                load_effect_model(model, root=root)
            missing_function = {**contract, "function": "_llk_pack_missing_"}
            model.write_text(
                json.dumps({**base, "restore_contracts": [missing_function]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AuditModelError, "function"):
                load_effect_model(model, root=root)

    def test_rejects_teardown_body_drift_and_stale_reference_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "tt_llk_blackhole" / "llk_lib" / "llk_pack.h"
            path.parent.mkdir(parents=True)
            source = (
                "inline void _llk_pack_init_() { TTI_WRCFG(0, 0, REG); }\n"
                "inline void _llk_pack_uninit_() {}\n"
            )
            path.write_text(source, encoding="utf-8")
            functions = {
                function["name"]: function for function in inventory(root)["functions"]
            }

            def evidence(name):
                function = functions[name]
                return {
                    "path": function["source"]["path"],
                    "line": function["source"]["start_line"],
                    "token": name,
                    "body_fingerprint": function["body_fingerprint"],
                }

            contract = {
                "architecture": "blackhole",
                "function": "_llk_pack_init_",
                "kind": "no_op_transient",
                "owner": None,
                "pair": "_llk_pack_uninit_",
                "owner_source": None,
                "pair_source": evidence("_llk_pack_uninit_"),
                "rationale": "The pair is a documented no-op.",
                "source": evidence("_llk_pack_init_"),
            }
            model = root / "effects.json"
            model_data = {
                "schema_version": 1,
                "effects": [
                    {
                        "pattern": "TTI_WRCFG",
                        "domain": "cfg_register",
                        "operation": "write",
                        "retention": "Retained until reconfigured.",
                    }
                ],
                "restore_contracts": [contract],
            }
            model.write_text(json.dumps(model_data), encoding="utf-8")
            self.assertEqual(
                load_effect_model(model, root=root)["restore_contracts"], [contract]
            )
            path.write_text(
                source.replace(
                    "_llk_pack_uninit_() {}", "_llk_pack_uninit_() { TTI_NOP; }"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AuditModelError, "pair.*body fingerprint"):
                load_effect_model(model, root=root)
            path.write_text(source, encoding="utf-8")
            stale = json.loads(json.dumps(contract))
            stale["pair_source"]["line"] = 1
            model.write_text(
                json.dumps({**model_data, "restore_contracts": [stale]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AuditModelError, "pair.*source token/line"):
                load_effect_model(model, root=root)


if __name__ == "__main__":
    unittest.main()
