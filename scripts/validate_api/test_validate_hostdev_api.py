# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from validate_hostdev_api import BASE_INCLUDES, validate

LICENSE = "// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.\n// SPDX-License-Identifier: Apache-2.0\n"
BASE = LICENSE + "#pragma once\n" + "".join(f"#include <{name}>\n" for name in sorted(BASE_INCLUDES))


class ValidateHostdevApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.header("base.h", BASE)

    def tearDown(self):
        self.tmp.cleanup()

    def header(self, name, body):
        path = self.root / name
        path.write_text(body)
        return path

    def errors(self):
        return validate(self.root)[0]

    def test_folder_headers_and_base_are_accepted(self):
        self.header("a.h", LICENSE + '#pragma once\n#include "hostdev/api/base.h"\n')
        self.header("b.h", LICENSE + "#pragma once\n#include <hostdev/api/a.h>\n#include <hostdev/api/base.h>\n")
        self.assertEqual(self.errors(), [])

    def test_standard_header_outside_base_is_rejected(self):
        self.header("a.h", LICENSE + "#pragma once\n#include <cstdint>\n")
        self.assertTrue(any("include only headers of hostdev/api/" in error for error in self.errors()))

    def test_header_outside_the_folder_is_rejected(self):
        for include in ("hostdev/dev_msgs.h", "tt_stl/assert.hpp", "hostdev/api/missing.h", "api/compute/common.h"):
            with self.subTest(include=include):
                self.header("a.h", LICENSE + f'#pragma once\n#include "{include}"\n')
                self.assertTrue(any("include only headers of hostdev/api/" in error for error in self.errors()))

    def test_base_list_is_closed(self):
        for include in ("iostream", "string", "cstdint", "hostdev/api/a.h"):
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
        self.header("a.h", LICENSE + '#include "hostdev/api/base.h"\n')
        self.assertTrue(any("#pragma once" in error for error in self.errors()))

    def test_non_header_is_rejected(self):
        self.header("a.cpp", LICENSE)
        self.assertTrue(any("only headers belong" in error for error in self.errors()))

    def test_missing_base_fails(self):
        (self.root / "base.h").unlink()
        self.assertTrue(any("no base.h" in error for error in self.errors()))

    def test_repository_folder_passes(self):
        repo = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, Path(__file__).with_name("validate_hostdev_api.py"), "tt_metal/hw/inc/hostdev/api"],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
