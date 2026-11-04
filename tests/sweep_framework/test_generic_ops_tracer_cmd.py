# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys

from model_tracer.generic_ops_tracer import _tee_command


def test_tee_command_passes_args_verbatim_and_keeps_exit_code(tmp_path):
    args = ["-k", "performance and ci-1", "--sampling_params", '{"temperature": 1.0, "top_k": 1}', "$HOME", "it's"]
    script = "import json, sys; print(json.dumps(sys.argv[1:])); sys.exit(3)"
    log = tmp_path / "out dir" / "run.log"
    log.parent.mkdir()

    result = subprocess.run(
        _tee_command([sys.executable, "-c", script, *args], str(log)),
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 3
    assert json.loads(result.stdout) == args
    assert json.loads(log.read_text()) == args
