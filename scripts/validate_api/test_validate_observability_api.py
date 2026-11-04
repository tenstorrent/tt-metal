# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from validate_observability_api import BASE_INCLUDES, validate

LICENSE = "// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.\n// SPDX-License-Identifier: Apache-2.0\n"
BASE = LICENSE + "#pragma once\n" + "".join(f"#include <{name}>\n" for name in sorted(BASE_INCLUDES))


class ValidateObservabilityApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.host = Path(self.tmp.name) / "host"
        self.device = Path(self.tmp.name) / "device"
        self.host.mkdir()
        self.device.mkdir()
        self.header("base.h", BASE)

    def tearDown(self):
        self.tmp.cleanup()

    def header(self, name, body, folder=None):
        path = (folder or self.host) / name
        path.write_text(body)
        return path

    def errors(self):
        return validate(self.host, self.device)[0]

    def test_folder_headers_and_base_are_accepted(self):
        self.header("a.h", LICENSE + '#pragma once\n#include "observability/api/base.h"\n')
        self.header(
            "b.h", LICENSE + "#pragma once\n#include <observability/api/a.h>\n#include <observability/api/base.h>\n"
        )
        self.assertEqual(self.errors(), [])

    def test_standard_header_outside_base_is_rejected(self):
        self.header("a.h", LICENSE + "#pragma once\n#include <cstdint>\n")
        self.assertTrue(any("include only headers of observability/api/" in error for error in self.errors()))

    def test_header_outside_the_folder_is_rejected(self):
        for include in (
            "hostdev/dev_msgs.h",
            "tt_stl/assert.hpp",
            "observability/api/missing.h",
            "api/compute/common.h",
            "tt-metalium/experimental/observability_event_meta.hpp",
        ):
            with self.subTest(include=include):
                self.header("a.h", LICENSE + f'#pragma once\n#include "{include}"\n')
                self.assertTrue(any("include only headers of observability/api/" in error for error in self.errors()))

    def test_base_list_is_closed(self):
        for include in ("iostream", "string", "cstdint", "observability/api/a.h"):
            with self.subTest(include=include):
                self.header("base.h", BASE + f"#include <{include}>\n")
                self.assertTrue(any("base.h may include only" in error for error in self.errors()))

    def test_side_conditionals_are_rejected(self):
        for directive in (
            "#if !defined(__riscv)",
            "#ifdef __riscv",
            "#ifndef COMPILE_FOR_TRISC",
            "#if defined(KERNEL_BUILD) || defined(FW_BUILD)",
            "#if 0\n#elif defined(ARCH_BLACKHOLE)",
        ):
            with self.subTest(directive=directive):
                self.header("a.h", LICENSE + f"#pragma once\n{directive}\n#endif\n")
                self.assertTrue(any("no conditional on the side being compiled" in error for error in self.errors()))

    def test_other_conditionals_are_accepted(self):
        self.header("a.h", LICENSE + "#pragma once\n#if __cplusplus >= 202002L\n#endif\n")
        self.assertEqual(self.errors(), [])

    def test_commented_out_violations_are_ignored(self):
        self.header("a.h", LICENSE + "#pragma once\n// #include <iostream>\n/* #ifdef __riscv */\n")
        self.assertEqual(self.errors(), [])

    def test_pragma_once_is_required(self):
        self.header("a.h", LICENSE + '#include "observability/api/base.h"\n')
        self.assertTrue(any("#pragma once" in error for error in self.errors()))

    def test_non_header_is_rejected(self):
        self.header("a.cpp", LICENSE)
        self.assertTrue(any("only headers belong" in error for error in self.errors()))

    def test_missing_base_fails(self):
        (self.host / "base.h").unlink()
        self.assertTrue(any("no base.h" in error for error in self.errors()))

    def test_device_header_sees_host_and_device_headers(self):
        self.header("shared.h", LICENSE + '#pragma once\n#include "observability/api/base.h"\n')
        self.header("device_only.h", LICENSE + '#pragma once\n#include "observability/api/base.h"\n', self.device)
        self.header(
            "user.h",
            LICENSE
            + '#pragma once\n#include "observability/api/shared.h"\n#include "observability/api/device_only.h"\n',
            self.device,
        )
        self.assertEqual(self.errors(), [])

    def test_host_header_does_not_see_device_headers(self):
        self.header("device_only.h", LICENSE + '#pragma once\n#include "observability/api/base.h"\n', self.device)
        self.header("a.h", LICENSE + '#pragma once\n#include "observability/api/device_only.h"\n')
        self.assertTrue(any("include only headers of observability/api/" in error for error in self.errors()))

    def test_device_headers_follow_the_same_rules(self):
        self.header("a.h", LICENSE + "#pragma once\n#include <iostream>\n#ifdef __riscv\n#endif\n", self.device)
        errors = self.errors()
        self.assertTrue(any("include only headers of observability/api/" in error for error in errors))
        self.assertTrue(any("no conditional on the side being compiled" in error for error in errors))

    def test_device_readme_is_allowed(self):
        self.header("README.md", "# Device versions\n", self.device)
        self.assertEqual(self.errors(), [])

    def test_repository_folders_pass(self):
        repo = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [
                sys.executable,
                Path(__file__).with_name("validate_observability_api.py"),
                "tt_metal/observability/api",
                "tt_metal/hw/observability/api",
            ],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
