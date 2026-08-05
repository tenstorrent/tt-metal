# SPDX-License-Identifier: Apache-2.0
"""A/B: ttnn.rms_norm vs blaze RMSNorm, at GLM-4.7-Flash decode shape.

Step 1 of the blaze-vs-ttnn campaign: this exists to PROVE THE RIG before any
conclusion is drawn from a bigger cluster. GLM decode shape is hidden=2048 with a
single active row at bs=1.

Run from the tt-blaze root with the profiler env already exported -- ab_harness
requires those vars to be set before ttnn is imported, which under pytest means
setting them in the shell, not in Python.
"""

import importlib.util
import sys
import types

sys.path.insert(0, "/home/ttuser/sdawle/skills/blaze-vs-ttnn-bench/scripts")


def _register_torch_golden() -> None:
    """Make ab_harness's `from tests.blaze.utils.torch_golden import comp_pcc` resolve.

    It cannot resolve on its own here: tt-blaze/tests has no __init__.py (so `tests` is only
    a namespace portion), while tt-blaze/tt-metal/tests DOES have one and is also on
    PYTHONPATH -- a regular package terminates the namespace search and shadows it. So we
    load the real file by path and register it, rather than editing either tree.
    """
    root = "/home/ttuser/sdawle/tt-blaze"
    for name, path in (("tests", f"{root}/tests"), ("tests.blaze", f"{root}/tests/blaze")):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [path]
            sys.modules[name] = mod
    name = "tests.blaze.utils.torch_golden"
    if name not in sys.modules:
        if "tests.blaze.utils" not in sys.modules:
            pkg = types.ModuleType("tests.blaze.utils")
            pkg.__path__ = [f"{root}/tests/blaze/utils"]
            sys.modules["tests.blaze.utils"] = pkg
        spec = importlib.util.spec_from_file_location(name, f"{root}/tests/blaze/utils/torch_golden.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)


_register_torch_golden()

import pytest
import torch

import ttnn
from ab_harness import ABCase, report, run_ab
from blaze.fused_program import MeshFusedProgram
from blaze.ops.rmsnorm import RMSNorm

K_DIM = 2048  # GLM-4.7-Flash hidden_size
EPS = 1e-6


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_ab_rmsnorm(mesh_device):
    torch.manual_seed(0)

    # ---- ONE set of host tensors, ONE golden. Two goldens means two bugs you
    # cannot see, so the golden comes from the blaze op's own contract.
    t_in = torch.randn((1, 1, 32, K_DIM), dtype=torch.bfloat16)
    # gamma is ONE row, broadcast over rows. It must be replicated (not 32 independent
    # random rows) because ttnn.rms_norm treats `weight` as a single broadcast vector --
    # 32 distinct rows silently compares two different functions and shows up as a PCC
    # deficit on BOTH sides, which is a rig bug, not an op difference.
    t_gamma_row = (1.0 + 0.1 * torch.randn((1, 1, 1, K_DIM))).bfloat16()
    t_gamma = t_gamma_row.expand(1, 1, 32, K_DIM).contiguous()
    golden = RMSNorm.golden(t_in, t_gamma_row, EPS)

    # ---- ONE set of device tensors, shared by both sides.
    #
    # Layout has to satisfy BOTH sides, which constrains it more than either alone:
    #   - blaze requires an L1-SHARDED input (tensor_utils.cpp:34 `tensor.is_sharded()`);
    #   - ttnn.rms_norm requires the physical shard to be 32x32-TILE sized
    #     (tensor_layout.cpp:162), so it cannot consume the 1x32 decode tile that blaze
    #     takes natively, AND it rejects HEIGHT_SHARDED outright
    #     (layernorm_device_operation.cpp:166).
    # A 32x2048 WIDTH_SHARDED L1 shard on a single core satisfies both: identical bytes to
    # the height-sharded form, a layout ttnn's layernorm accepts, and still one whole row
    # per core so blaze's per-core reduction stays semantically correct. Note the asymmetry
    # this exposes: at bs=1 only one of those 32 rows is real, and ttnn has no way to say
    # so here -- which is itself a blaze advantage, but one this shared-tensor A/B
    # deliberately does not exploit.
    mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    one_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    l1_rows = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(one_core, (32, K_DIM), ttnn.ShardOrientation.ROW_MAJOR),
    )

    def to_dev(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=mapper,
            memory_config=l1_rows,
        )

    tt_in = to_dev(t_in)
    tt_gamma = to_dev(t_gamma)
    tt_out = to_dev(torch.zeros((1, 1, 32, K_DIM), dtype=torch.bfloat16))

    def ttnn_side():
        return ttnn.rms_norm(tt_in, weight=tt_gamma, epsilon=EPS)

    # ---- Build the blaze program ONCE. Rebuilding per iteration would measure
    # host-side composition instead of kernel time.
    def compose_fn(f, inp, gamma, out):
        h = RMSNorm.emit(f, inp, gamma, prefix="rmsnorm", cores=one_core, epsilon=EPS)
        f._wire_output(h, out)

    # fp32_dest_acc_en is set on the PROGRAM, not on RMSNorm.emit -- emit() deletes its own
    # copy of the flag because "the program's ComputeConfigDescriptor is authoritative"
    # (blaze/ops/rmsnorm/op.py:143). Without it the reduction accumulates in bf16 and lands
    # under the 0.99 PCC floor.
    mfp = MeshFusedProgram(
        mesh_device,
        kernel=None,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        name="ab_rmsnorm",
    )
    mfp.compose(compose_fn, tt_in, tt_gamma, tt_out)
    program = mfp.build([tt_in, tt_gamma, tt_out], mesh_output_tensor=tt_out)

    def blaze_side():
        program.run()
        return tt_out

    rows = run_ab(
        mesh_device,
        [
            ABCase(
                label=f"rmsnorm 1x{K_DIM} (GLM decode)",
                ttnn_fn=ttnn_side,
                blaze_fn=blaze_side,
                golden=lambda: golden,
                pcc_floor=0.99,
            )
        ],
        warmup=2,
        iters=5,
    )
    text = report(rows)
    with open(
        "/tmp/claude-1000/-home-ttuser-sdawle-tt-metal/0247ce80-749c-4947-8139-2d31330dccb7/scratchpad/ab_rmsnorm_result.md",
        "w",
    ) as fh:
        fh.write(text + "\n")
