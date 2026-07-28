# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Full-model prefill accuracy test for gpt_oss_d_p.

The authoritative accuracy test is the standalone script pair that follows the
minimax_m2 testing pattern:

  hf_reference_oracle.py  — run once on CPU to produce ground truth
  first_token_prefill.py  — run TT prefill and compare argmax + PCC to oracle

See those files for run instructions. The standalone scripts open and close the
mesh themselves, which avoids conflicts with pytest fixture mesh lifecycle.

Example (after generating oracle):
  export HF_MODEL=...
  export TT_MESH_GRAPH_DESC_PATH=...
  python3 models/demos/gpt_oss_d_p/tests/accuracy/hf_reference_oracle.py \\
      --out /data/jmalone/gpt_oss_ref \\
      --prompt "What are the prime factors of 1?"

  python3 models/demos/gpt_oss_d_p/tests/accuracy/first_token_prefill.py \\
      --rows 1 --cols 8 \\
      --prompt "What are the prime factors of 1?" \\
      --dump-logits /tmp/tt_logits.npy \\
      --oracle-dir /data/jmalone/gpt_oss_ref

This pytest module runs first_token_prefill.py as a subprocess so CI can
collect and track it without re-implementing the mesh lifecycle here.
The subprocess opens its own mesh; no device fixture is used.
"""

import os
import subprocess
import sys

import pytest


def _available_mesh_shapes():
    """Return the largest mesh shape that fits on this system."""
    import ttnn

    n = ttnn.get_num_devices()
    for shape in [(4, 8), (1, 8), (1, 4), (1, 2), (1, 1)]:
        if shape[0] * shape[1] <= n:
            return shape
    return (1, 1)


@pytest.mark.skipif(not os.getenv("HF_MODEL"), reason="HF_MODEL not set")
def test_prefill_first_token():
    """
    Runs first_token_prefill.py as a subprocess on the largest available mesh.

    To run with oracle comparison:
      export ORACLE_DIR=/data/jmalone/gpt_oss_ref
      pytest models/demos/gpt_oss_d_p/tests/accuracy/test_model.py
    """
    rows, cols = _available_mesh_shapes()
    script = "models/demos/gpt_oss_d_p/tests/accuracy/first_token_prefill.py"
    cmd = [
        sys.executable,
        script,
        "--rows",
        str(rows),
        "--cols",
        str(cols),
        "--prompt",
        "What are the prime factors of 1?",
    ]
    oracle_dir = os.getenv("ORACLE_DIR")
    if oracle_dir:
        cmd += ["--oracle-dir", oracle_dir]

    result = subprocess.run(cmd, env=os.environ)
    assert result.returncode == 0, f"first_token_prefill.py exited {result.returncode}"
