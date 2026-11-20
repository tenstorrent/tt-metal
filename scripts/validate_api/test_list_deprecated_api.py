# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import datetime as dt
import json
import unittest
from pathlib import Path

from list_deprecated_api import collect, load_manifest, manifest_names, parse_removal, removal_status, scan_text

REPO = Path(__file__).resolve().parents[2]
API = REPO / "tt_metal" / "api"


def _names(text: str) -> dict[str, str]:
    return {hit.qualified_name: hit.message for hit in scan_text(text)}


class ScannerTest(unittest.TestCase):
    def test_ignores_comments_and_strings(self):
        text = """
        // [[deprecated("not real")]]
        /* [[deprecated("also not real")]] */
        const char* note = "[[deprecated(\\"nope\\")]]";
        [[deprecated("removed after 2026-04-01.")]]
        void real_one();
        """
        hits = scan_text(text)
        self.assertEqual([hit.name for hit in hits], ["real_one"])

    def test_class_scope_survives_operator_and_inline_comparison(self):
        text = """
        namespace tt::tt_metal {
        class Box {
            bool operator<(const Box& other) const;
            bool operator<=(const Box& other) const { return true; }
            [[deprecated("removed after 2026-01-02.")]]
            void old_method();
        };
        inline bool less(int a, int b) { return a < b && a > 0; }
        template <class T>
        struct Hold {
            [[deprecated("removed after 2026-07-08.")]]
            void keep();
        };
        [[deprecated("removed after 2026-03-04.")]]
        void free_old();
        }
        """
        self.assertEqual(
            set(_names(text)),
            {
                "tt::tt_metal::Box::old_method",
                "tt::tt_metal::Hold::keep",
                "tt::tt_metal::free_old",
            },
        )

    def test_nested_namespace_is_one_brace(self):
        text = """
        namespace tt::tt_metal {
        class Allocator;
        }
        namespace tt::tt_metal::distributed {
        class MeshDevice {
            [[deprecated("removed after 2026-01-01.")]]
            void is_local();
        };
        }
        namespace tt::tt_fabric {
        [[deprecated("removed after 2026-12-31.")]]
        void GetFabricConfig();
        }
        """
        self.assertEqual(
            set(_names(text)),
            {
                "tt::tt_metal::distributed::MeshDevice::is_local",
                "tt::tt_fabric::GetFabricConfig",
            },
        )

    def test_using_alias_multiline_message_and_field(self):
        text = """
        namespace tt::tt_metal::experimental {
        using DeviceGroupIndex
            [[deprecated("moved to internal. "
                         "include <internal/disaggregation/kv_chunk_address_table.hpp>")]] =
            internal::DeviceGroupIndex;
        struct Opts {
            [[deprecated("Per-node varargs will be removed.")]]
            int num_runtime_varargs_per_node;
        };
        }
        """
        hits = {hit.name: hit for hit in scan_text(text)}
        self.assertEqual(hits["DeviceGroupIndex"].kind, "alias")
        self.assertEqual(
            hits["DeviceGroupIndex"].qualified_name,
            "tt::tt_metal::experimental::DeviceGroupIndex",
        )
        self.assertIn("moved to internal. include", hits["DeviceGroupIndex"].message)
        self.assertEqual(hits["num_runtime_varargs_per_node"].kind, "field")
        self.assertEqual(
            hits["num_runtime_varargs_per_node"].qualified_name,
            "tt::tt_metal::experimental::Opts::num_runtime_varargs_per_node",
        )

    def test_attribute_then_nodiscard_and_constructor(self):
        text = """
        namespace tt::tt_metal::distributed {
        class MeshDevice : public IDevice, public std::enable_shared_from_this<MeshDevice> {
            [[deprecated("Use the coordinate overload. This will be removed after 2026-11-19.")]]
            [[nodiscard]] std::vector<IDevice*>
            get_devices() const;
            [[deprecated("Use custom alignment. This API will be removed after 2026-12-01.")]]
            MeshDevice(int alignment);
        };
        }
        """
        hits = {hit.name: hit for hit in scan_text(text)}
        self.assertEqual(hits["get_devices"].kind, "function")
        self.assertIn("std::vector<IDevice*> get_devices() const", hits["get_devices"].signature)
        self.assertEqual(hits["MeshDevice"].qualified_name, "tt::tt_metal::distributed::MeshDevice::MeshDevice")

    def test_removal_dates(self):
        self.assertEqual(parse_removal("removed after 2026-11-13.")[0], dt.date(2026, 11, 13))
        self.assertEqual(parse_removal("removed after September 9th, 2026.")[0], dt.date(2026, 9, 9))
        date, raw, style = parse_removal("removed after 28-02-2026.")
        self.assertEqual((date, raw, style), (dt.date(2026, 2, 28), "28-02-2026", "dmy"))
        self.assertEqual(parse_removal("will be removed.")[2], "none")
        self.assertEqual(parse_removal("after 02-03-2026.")[2], "ambiguous")
        today = dt.date(2026, 11, 20)
        self.assertEqual(removal_status(dt.date(2026, 11, 19), today), "overdue")
        self.assertEqual(removal_status(dt.date(2026, 11, 20), today), "due")
        self.assertEqual(removal_status(dt.date(2026, 12, 1), today), "scheduled")
        self.assertEqual(removal_status(None, today), "undated")

    def test_manifest_brace_expansion(self):
        description = (
            "BeginTraceCapture and Synchronize stay. "
            "MeshDevice::{begin,end,replay}_mesh_trace(uint8_t) goes. MeshEvent stays."
        )
        names = manifest_names(description)
        self.assertIn("begin_mesh_trace", names)
        self.assertIn("end_mesh_trace", names)
        self.assertIn("replay_mesh_trace", names)
        self.assertIn("BeginTraceCapture", names)
        self.assertNotIn("begin", names)


class LiveTreeTest(unittest.TestCase):
    def test_live_metal_api_inventory(self):
        if not API.is_dir():
            self.skipTest("tt_metal/api is not in this tree")
        manifest_path = REPO / ".github" / "deprecations.json"
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        symbols = collect(API, REPO, load_manifest(manifest_path), dt.date(2026, 11, 20))
        self.assertGreaterEqual(len(symbols), 1)
        self.assertTrue(all(symbol.name and symbol.name != "<unknown>" for symbol in symbols))

        raw_attrs = 0
        for header in API.rglob("*"):
            if header.suffix not in {".h", ".hpp", ".hh", ".hxx"}:
                continue
            raw_attrs += _count_deprecated_attrs(header.read_text(encoding="utf-8"))
        self.assertEqual(len(symbols), raw_attrs, "scanner dropped or invented a [[deprecated]] attribute")

        by_name = {symbol.qualified_name: symbol for symbol in symbols}
        hal = by_name["tt::tt_metal::hal::get_arch_num_circular_buffers"]
        self.assertEqual(hal.removal_date, dt.date(2026, 10, 23))
        self.assertEqual(by_name["tt::tt_metal::IsGalaxyCluster"].removal_date, dt.date(2026, 11, 13))
        self.assertEqual(
            by_name["tt::tt_metal::distributed::MeshDevice::is_local"].removal_date,
            dt.date(2026, 2, 28),
        )
        self.assertEqual(
            by_name["tt::tt_metal::experimental::disaggregation::DeviceGroupIndex"].kind,
            "alias",
        )
        tracked = [symbol for symbol in symbols if symbol.name == "begin_mesh_trace" and symbol.manifest_ids]
        self.assertTrue(tracked)
        self.assertEqual(tracked[0].manifest_ids, ["mesh-command-queue-id-host-api-overloads"])
        get_devices = by_name["tt::tt_metal::distributed::MeshDevice::get_devices"]
        self.assertEqual(get_devices.manifest_ids, [])
        self.assertIn("mesh-command-queue-id-host-api-overloads", get_devices.manifest_file_ids)
        self.assertGreaterEqual(len(manifest_data["deprecations"]), 1)


def _count_deprecated_attrs(text: str) -> int:
    """Count [[deprecated]] outside comments and string literals. Independent of scan_text."""
    count = 0
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("//", i):
            newline = text.find("\n", i)
            i = n if newline < 0 else newline + 1
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        if text[i] == '"':
            i += 1
            while i < n and text[i] != '"':
                if text[i] == "\\":
                    i += 2
                    continue
                i += 1
            i = min(i + 1, n)
            continue
        if text.startswith("[[deprecated", i):
            end = i + len("[[deprecated")
            if end >= n or not (text[end].isalnum() or text[end] == "_"):
                count += 1
            i = end
            continue
        i += 1
    return count


if __name__ == "__main__":
    unittest.main()
