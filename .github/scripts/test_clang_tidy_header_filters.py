# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise the root header filters using the same clang-tidy binary as CI."""

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile


def main():
    clang_tidy = sys.argv[1] if len(sys.argv) > 1 else "clang-tidy-20"
    repo = pathlib.Path(__file__).resolve().parents[2]
    config = repo / ".clang-tidy"
    # Include both maintained headers and neighboring generated headers. Every
    # header deliberately violates the same check; only maintained ones report.
    headers = {
        "tt_metal/api/tt-metalium/probe.hpp": True,
        "ttnn/api/ttnn/probe.hpp": True,
        "tt_metal/hw/inc/internal/tt-2xx/quasar/overlay/probe.hpp": True,
        "tt_metal/impl/flatbuffers/probe.hpp": True,
        "tt_metal/impl/flatbuffers/probe_generated.h": False,
        "tt_metal/hw/inc/internal/tt-2xx/quasar/tensix_neo_reg.h": False,
        "tt_metal/hw/inc/internal/tt-2xx/quasar/noc/registers/noc_address_translation_table_a_reg.h": False,
        "generated/probe.pb.h": False,
        "generated/probe.pb.hpp": True,
    }
    with tempfile.TemporaryDirectory(prefix="clang-tidy-header-filters-") as directory:
        root = pathlib.Path(directory)
        shutil.copyfile(config, root / ".clang-tidy")
        for index, header in enumerate(headers):
            path = root / header
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"inline int probe_{index}(int x) {{ if (x) return 1; return 0; }}\n")
        source = root / "probe.cpp"
        source.write_text("".join(f'#include "{header}"\n' for header in headers))

        dumped = subprocess.run(
            [clang_tidy, "--dump-config", str(source), "--", "-std=c++20"],
            capture_output=True,
            text=True,
            check=True,
        )
        if dumped.stderr.strip():
            raise RuntimeError(f"clang-tidy rejected the repository configuration:\n{dumped.stderr}")
        for key in ("HeaderFilterRegex", "ExcludeHeaderFilterRegex"):
            original = re.search(rf"^{key}:\s*(.+)$", config.read_text(), re.MULTILINE)
            effective = re.search(rf"^{key}:\s*(.+)$", dumped.stdout, re.MULTILINE)
            if not original or not effective or original[1].strip("'\"") != effective[1].strip("'\""):
                raise RuntimeError(f"clang-tidy did not retain {key}:\n{dumped.stdout}")

        result = subprocess.run(
            [
                clang_tidy,
                "--checks=-*,readability-braces-around-statements",
                "--warnings-as-errors=*",
                "--use-color=false",
                str(source),
                "--",
                "-std=c++20",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout + result.stderr
        reported = {
            pathlib.Path(path).resolve()
            for path in re.findall(r"^(.+):\d+:\d+: (?:warning|error):", output, re.MULTILINE)
        }
        expected = {(root / path).resolve() for path, included in headers.items() if included}
        if result.returncode != 1 or reported != expected:
            raise RuntimeError(f"Unexpected header diagnostics (exit {result.returncode}):\n{output}")

    print("clang-tidy accepted both filter keys and passed all 9 header diagnostic cases")


if __name__ == "__main__":
    main()
