# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Cases for check_manifest_routes's --expect-z-peers, whose failure no mock manifest reaches.

Run from the repository root:
    python3 -m unittest tt_metal.fabric.debug.visualizer.decode.tests.test_check_manifest_routes
"""

import contextlib
import io
import json
import pathlib
import tempfile
import unittest

from tt_metal.fabric.debug.visualizer.decode.tests.check_manifest_routes import main
from tt_metal.fabric.debug.visualizer.decode.tests.test_routes import MANIFEST, MESH_MANIFEST, with_value


def check(manifest, *flags):
    """The check's exit code and output on `manifest`, written to a directory of its own."""
    with tempfile.TemporaryDirectory() as directory:
        pathlib.Path(directory, "fabric_manifest_rank_1_of_1.json").write_text(json.dumps(manifest))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main([*flags, directory])
    return code, output.getvalue()


class ExpectZPeersTest(unittest.TestCase):
    # MESH_MANIFEST's C0 has a Z peer.
    BOTH = {**MANIFEST, **MESH_MANIFEST}

    def test_a_manifest_with_a_z_peer_passes(self):
        code, output = check(self.BOTH, "--expect-z-peers")
        self.assertEqual(code, 0, output)
        self.assertIn("Z peers found: 1", output)

    def test_a_changed_peer_format_fails_only_with_it(self):
        # The loaders read any string as a peer, so a new format leaves C0's Z peer unknown without raising.
        routers = "meshes.M0.chips.C0.routers"
        manifest = with_value(self.BOTH, f"{routers}.Z0.link.peer", "M0:C2:Z0")
        manifest = with_value(manifest, f"{routers}.Z1.link.peer", "M0:C2:Z1")

        code, output = check(manifest)
        self.assertEqual(code, 0, output)

        code, output = check(manifest, "--expect-z-peers")
        self.assertEqual(code, 1, output)
        self.assertIn("finds no Z peer in", output)


if __name__ == "__main__":
    unittest.main()
