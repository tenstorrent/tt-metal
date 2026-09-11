# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host regression for the three-stage kernel-prewarm wrapper."""

import os
import stat
import subprocess
from pathlib import Path

WRAPPER = Path(__file__).resolve().parents[4] / "tt_metal" / "tools" / "kernel_prewarm" / "prewarm_and_submit.sh"


def _make_exec(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_capture_stage_exports_capture_only_into_compound_command(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    env_file = tmp_path / "device.env.yaml"
    env_file.write_text(f'TT_METAL_CACHE: "{cache}"\nTT_METAL_HOME: "{tmp_path}"\n')

    tool_dir = tmp_path / "build_Release" / "tools"
    tool_dir.mkdir(parents=True)
    _make_exec(tool_dir / "kernel_prewarm", "#!/usr/bin/env bash\nexit 0\n")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _make_exec(
        fake_bin / "tt-device-mcp",
        "#!/usr/bin/env bash\n"
        'sub="$1"; shift\n'
        'cmd="$1"; shift\n'
        'if [[ "$sub" == "run" ]]; then\n'
        '  printf "%s" "$cmd" > "$TEST_RECORD"\n'
        '  printf "manifest-line\\n" >> "$TEST_CACHE/kernel_prewarm.manifest"\n'
        "fi\n"
        "exit 0\n",
    )

    record = tmp_path / "record"
    subprocess.run(
        [str(WRAPPER), "-e", str(env_file), "-c", "--", "cd /tmp && env"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "TEST_CACHE": str(cache),
            "TEST_RECORD": str(record),
        },
        check=True,
    )

    assert record.read_text() == "export TT_METAL_KERNEL_CAPTURE_ONLY=1; cd /tmp && env"
