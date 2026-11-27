# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verify graduate_progressive_split.patch without touching the op: the patch is applied to a COPY of
the descriptor (built at import into a temp module), PROGRESSIVE_N_BLOCKS set to PMB_NB, and the op's
create_program_descriptor swapped for the patched one. Skipped unless GENIFY_MATMUL_PERF=1."""
import importlib.util, os, subprocess, sys, tempfile
from pathlib import Path
import pytest
import ttnn
import eval.golden_tests.genify_matmul.helpers as helpers
import ttnn.operations.genify_matmul.genify_matmul  # noqa: F401 (the package re-exports the function name)
import ttnn.operations.genify_matmul.genify_matmul_program_descriptor as pd

op_mod = sys.modules["ttnn.operations.genify_matmul.genify_matmul"]

pytestmark = pytest.mark.skipif(os.environ.get("GENIFY_MATMUL_PERF") != "1", reason="perf bench")
ROOT = Path(pd.__file__).resolve().parents[4]
PATCH = (
    ROOT
    / "ttnn/ttnn/operations/genify_matmul/perf_experiments/progressive_output_multiblock/graduate_progressive_split.patch"
)


def _patched_module():
    tmp = Path(tempfile.mkdtemp())
    rel = "ttnn/ttnn/operations/genify_matmul/genify_matmul_program_descriptor.py"
    (tmp / rel).parent.mkdir(parents=True)
    (tmp / rel).write_text(Path(pd.__file__).read_text())
    subprocess.run(["patch", "-p1", "-s", "-i", str(PATCH)], cwd=tmp, check=True)
    spec = importlib.util.spec_from_file_location("pd_patched", tmp / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pd_patched"] = mod  # dataclasses resolve their module
    spec.loader.exec_module(mod)
    mod.KERNEL_DIR = pd.KERNEL_DIR  # the op's real kernels
    return mod


def test_patch(device, monkeypatch):
    mod = _patched_module()
    mod.PROGRESSIVE_N_BLOCKS = int(os.environ.get("PMB_NB", "2"))
    monkeypatch.setattr(op_mod, "create_program_descriptor", mod.create_program_descriptor)
    mode = os.environ.get("PMB_MODE", "real")
    if mode != "real":
        monkeypatch.setenv("GENIFY_ABLATE", mode)
        monkeypatch.setattr(helpers, "check_output", lambda *a, **k: None)
    orig_plan = mod.plan

    def plan(*a, **k):
        r = orig_plan(*a, **k)
        print(f"\nPMB_PLAN patched nb={mod.PROGRESSIVE_N_BLOCKS} blk={r[2]}")
        return r

    mod.plan = plan
    helpers.run_genify_matmul(
        ((1, 2048, 5120), (5120, 8704)),
        dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        fp32_dest_acc_en=True,
        placement="interleaved",
        epilogue="swiglu",
        device=device,
        extras={"math_fidelity": ttnn.MathFidelity.LoFi},
    )
    print(f"\nPMB_RESULT patched nb={mod.PROGRESSIVE_N_BLOCKS} mode={mode} passed")
