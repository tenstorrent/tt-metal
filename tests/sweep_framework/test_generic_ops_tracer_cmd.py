# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import sys

from model_tracer.generic_ops_tracer import _run_and_tee


def test_run_and_tee_passes_args_verbatim_and_keeps_exit_code(tmp_path, capsys):
    args = ["-k", "performance and ci-1", "--sampling_params", '{"temperature": 1.0, "top_k": 1}', "$HOME", "it's"]
    script = "import json, sys; print(json.dumps(sys.argv[1:])); print('to stderr', file=sys.stderr); sys.exit(3)"

    returncode, output = _run_and_tee([sys.executable, "-c", script, *args], cwd=tmp_path)

    assert returncode == 3
    # stdout and stderr share the pipe in no fixed order, so check membership, not position.
    lines = output.splitlines()
    assert json.dumps(args) in lines
    assert "to stderr" in lines
    assert capsys.readouterr().out == output
