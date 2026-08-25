#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for resolve_host_ring_order.py."""

import json
import os
import tempfile
import textwrap
import unittest
from pathlib import Path

from resolve_host_ring_order import (
    _build_adjacency_from_cabling,
    _build_adjacency_from_fsd,
    _safe_read_text,
    _walk_ring,
    main,
    parse_textproto,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CABLING_DIR = REPO_ROOT / "tools" / "tests" / "scaleout" / "cabling_descriptors"
DEPLOYMENT_DIR = REPO_ROOT / "tools" / "tests" / "scaleout" / "deployment_descriptors"


class TestTextprotoParser(unittest.TestCase):
    def test_simple_message(self):
        text = 'name: "hello" value: 42'
        result = parse_textproto(text)
        self.assertEqual(result["name"], "hello")
        self.assertEqual(result["value"], "42")

    def test_nested_message(self):
        text = 'outer { inner: "val" count: 3 }'
        result = parse_textproto(text)
        self.assertIsInstance(result["outer"], dict)
        self.assertEqual(result["outer"]["inner"], "val")

    def test_repeated_fields(self):
        text = textwrap.dedent(
            """\
            hosts { host: "a" }
            hosts { host: "b" }
        """
        )
        result = parse_textproto(text)
        self.assertIsInstance(result["hosts"], list)
        self.assertEqual(len(result["hosts"]), 2)

    def test_comments_stripped(self):
        text = '# comment\nname: "test" # trailing'
        result = parse_textproto(text)
        self.assertEqual(result["name"], "test")

    def test_map_syntax(self):
        text = textwrap.dedent(
            """\
            graph_templates {
              key: "tpl"
              value { children { name: "n1" } }
            }
        """
        )
        result = parse_textproto(text)
        self.assertIn("graph_templates", result)


class TestFSDMode(unittest.TestCase):
    def test_four_host_ring(self):
        fsd_text = textwrap.dedent(
            """\
            hosts { hostname: "host-a" }
            hosts { hostname: "host-b" }
            hosts { hostname: "host-c" }
            hosts { hostname: "host-d" }
            eth_connections {
                connection { endpoint_a { host_id: 0 tray_id: 2 } endpoint_b { host_id: 1 tray_id: 1 } }
                connection { endpoint_a { host_id: 1 tray_id: 2 } endpoint_b { host_id: 2 tray_id: 1 } }
                connection { endpoint_a { host_id: 2 tray_id: 2 } endpoint_b { host_id: 3 tray_id: 1 } }
                connection { endpoint_a { host_id: 3 tray_id: 2 } endpoint_b { host_id: 0 tray_id: 1 } }
            }
        """
        )
        fsd = parse_textproto(fsd_text)
        hid_to_name, adj = _build_adjacency_from_fsd(fsd)
        self.assertEqual(len(hid_to_name), 4)
        for hid in range(4):
            self.assertEqual(len(adj[hid]), 2)

        ordered = _walk_ring(["host-a", "host-b", "host-c", "host-d"], hid_to_name, adj)
        self.assertEqual(len(ordered), 4)
        shorts = [h.split(".")[0] for h in ordered]
        idx_a = shorts.index("host-a")
        idx_b = shorts.index("host-b")
        self.assertIn(abs(idx_a - idx_b), [1, 3])

    def test_ignores_same_host_connections(self):
        fsd_text = textwrap.dedent(
            """\
            hosts { hostname: "h1" }
            hosts { hostname: "h2" }
            eth_connections {
                connection { endpoint_a { host_id: 0 tray_id: 1 } endpoint_b { host_id: 0 tray_id: 2 } }
                connection { endpoint_a { host_id: 0 tray_id: 2 } endpoint_b { host_id: 1 tray_id: 1 } }
                connection { endpoint_a { host_id: 1 tray_id: 2 } endpoint_b { host_id: 0 tray_id: 1 } }
            }
        """
        )
        fsd = parse_textproto(fsd_text)
        _, adj = _build_adjacency_from_fsd(fsd)
        self.assertEqual(adj.get(0), {1})
        self.assertEqual(adj.get(1), {0})


class TestCablingMode(unittest.TestCase):
    def test_8x16_superpod_ring(self):
        cabling_path = CABLING_DIR / "8x16_wh_galaxy_xy_torus_superpod.textproto"
        deployment_path = DEPLOYMENT_DIR / "8x16_wh_galaxy_xy_torus_deployment.textproto"
        if not cabling_path.exists() or not deployment_path.exists():
            self.skipTest("Fixture files not found")

        cabling = parse_textproto(cabling_path.read_text())
        deployment = parse_textproto(deployment_path.read_text())

        hid_to_name, adj = _build_adjacency_from_cabling(cabling, deployment)
        self.assertEqual(len(hid_to_name), 4)
        for hid in range(4):
            self.assertIn(hid, adj, f"host_id {hid} missing from adjacency")
            self.assertEqual(
                len(adj[hid]),
                2,
                f"host_id {hid} ({hid_to_name[hid]}) has {len(adj[hid])} neighbors, expected 2",
            )

        ordered = _walk_ring(
            ["wh-glx-1", "wh-glx-2", "wh-glx-3", "wh-glx-4"],
            hid_to_name,
            adj,
        )
        self.assertEqual(len(ordered), 4)
        shorts = [h.split(".")[0] for h in ordered]
        idx_1 = shorts.index("wh-glx-1")
        idx_2 = shorts.index("wh-glx-2")
        self.assertIn(abs(idx_1 - idx_2), [1, 3])

    def test_fqdn_matching(self):
        cabling_path = CABLING_DIR / "8x16_wh_galaxy_xy_torus_superpod.textproto"
        deployment_path = DEPLOYMENT_DIR / "8x16_wh_galaxy_xy_torus_deployment.textproto"
        if not cabling_path.exists() or not deployment_path.exists():
            self.skipTest("Fixture files not found")

        cabling = parse_textproto(cabling_path.read_text())
        deployment = parse_textproto(deployment_path.read_text())
        hid_to_name, adj = _build_adjacency_from_cabling(cabling, deployment)

        ordered = _walk_ring(
            ["wh-glx-1.example.com", "wh-glx-2.example.com", "wh-glx-3.example.com", "wh-glx-4.example.com"],
            hid_to_name,
            adj,
        )
        self.assertEqual(len(ordered), 4)
        for h in ordered:
            self.assertIn(".example.com", h)


class TestRingWalkErrors(unittest.TestCase):
    def test_unknown_host_raises(self):
        hid_to_name = {0: "a", 1: "b"}
        adj = {0: {1}, 1: {0}}
        with self.assertRaises(ValueError):
            _walk_ring(["a", "unknown"], hid_to_name, adj)

    def test_non_ring_raises(self):
        hid_to_name = {0: "a", 1: "b", 2: "c"}
        adj = {0: {1}, 1: {0, 2}, 2: {1}}
        with self.assertRaises(ValueError):
            _walk_ring(["a", "b", "c"], hid_to_name, adj)

    def test_single_host_passthrough(self):
        hid_to_name = {0: "only"}
        adj = {}
        ordered = _walk_ring(["only"], hid_to_name, adj)
        self.assertEqual(ordered, ["only"])


class TestCLI(unittest.TestCase):
    def test_fsd_cli(self):
        import tempfile

        fsd_text = textwrap.dedent(
            """\
            hosts { hostname: "h1" }
            hosts { hostname: "h2" }
            hosts { hostname: "h3" }
            hosts { hostname: "h4" }
            eth_connections {
                connection { endpoint_a { host_id: 0 } endpoint_b { host_id: 1 } }
                connection { endpoint_a { host_id: 1 } endpoint_b { host_id: 2 } }
                connection { endpoint_a { host_id: 2 } endpoint_b { host_id: 3 } }
                connection { endpoint_a { host_id: 3 } endpoint_b { host_id: 0 } }
            }
        """
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".textproto", delete=False) as f:
            f.write(fsd_text)
            fsd_path = f.name

        try:
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(["--hosts", "h1,h2,h3,h4", "--fsd", fsd_path])
            self.assertEqual(rc, 0)
            result = json.loads(buf.getvalue())
            self.assertEqual(result["status"], "success")
            self.assertEqual(len(result["ordered_hosts"].split(",")), 4)
        finally:
            os.unlink(fsd_path)

    def test_cabling_cli(self):
        cabling_path = CABLING_DIR / "8x16_wh_galaxy_xy_torus_superpod.textproto"
        deployment_path = DEPLOYMENT_DIR / "8x16_wh_galaxy_xy_torus_deployment.textproto"
        if not cabling_path.exists() or not deployment_path.exists():
            self.skipTest("Fixture files not found")

        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(
                [
                    "--hosts",
                    "wh-glx-1,wh-glx-2,wh-glx-3,wh-glx-4",
                    "--cabling",
                    str(cabling_path),
                    "--deployment",
                    str(deployment_path),
                ]
            )
        self.assertEqual(rc, 0)
        result = json.loads(buf.getvalue())
        self.assertEqual(result["status"], "success")

    def test_error_on_missing_deployment(self):
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--hosts", "a,b", "--cabling", "/nonexistent"])
        self.assertEqual(rc, 1)

    def test_safe_read_rejects_path_outside_allowed_roots(self):
        with self.assertRaises(ValueError) as ctx:
            _safe_read_text("/etc/passwd")
        self.assertIn("outside allowed descriptor roots", str(ctx.exception))

    def test_safe_read_allows_temp_descriptor(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".textproto", delete=False) as f:
            f.write('name: "ok"')
            path = f.name
        try:
            self.assertEqual(_safe_read_text(path), 'name: "ok"')
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
