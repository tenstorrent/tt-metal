# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Small-grid companion to test_softmax.py.
#
# The test_softmax.py::test_softmax_l1 cases are marked blackhole_scale: they reproduce the model's
# exact TILE L1 detect buffers at full size on 10-80 cores, so -m emulator never runs them. These call
# the same test body with sizes that fit 1-2 cores; everything else (layouts, memory
# configs, checks) is unchanged.
#
# Fidelity sweep (issue #63782): on Quasar the math source formats are programmed once, at
# compute_kernel_hw_startup, from the bf16 input, so softmax's fp32 intermediates run under a bf16
# ALU format. ttsim then refuses HiFi2+ phases ("check_fidelity_phase ... one phase covers SrcA").
#   default - the model's call, no compute_kernel_config (op default: HiFi4 on Quasar today)
#   hifi4   - explicit HiFi4, otherwise identical to the bf16-input default config
#   lofi    - explicit LoFi, otherwise identical; runs on ttsim
# On the RTL emulator HiFi4 is measurably more accurate than LoFi (ulp_mean 0.365 vs 0.453, abs_max
# 2.5e-3 vs 3.8e-3), so the extra phases do real work there and the op default stays HiFi4. ttsim's
# check_fidelity_phase refuses them, so default and hifi4 are skipped on ttsim only (Quasar).
# The explicit configs mirror softmax_init_compute_kernel_config's bf16 default (approx on,
# fp32 dest acc on, packer L1 acc off) so only the fidelity differs.
# ---------------------------------------------------------------------------
"""Small-grid cases for the model-faithful ``softmax`` call sites (fit a 1- or 2-compute-node device)."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U
from models.experimental.ops.quasar.tests.yolo_ops import test_softmax as T

_FIDELITIES = [
    pytest.param(None, id="default"),
    pytest.param(ttnn.MathFidelity.HiFi4, id="hifi4"),
    pytest.param(ttnn.MathFidelity.LoFi, id="lofi"),
]


def _compute_config(mesh, fidelity):
    if fidelity is None:
        return None
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


@U.with_default_mesh()
@pytest.mark.parametrize("fidelity", _FIDELITIES)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 4, 16, 400), id="dfl-l1-1x4x16x400"),
    ],
)
def test_softmax_l1_small_grid(ttnn_mesh_device, reset_seeds, shape, fidelity):
    fid_name = "default" if fidelity is None else str(fidelity).split(".")[-1]
    if fidelity != ttnn.MathFidelity.LoFi and U.is_quasar() and U.on_ttsim():
        pytest.skip(
            f"ttsim: {fid_name} (HiFi4) stops on check_fidelity_phase with bf16 math src formats; "
            "runs correctly on the RTL emulator (issue #63782)"
        )
    T.run_softmax_l1(
        ttnn_mesh_device,
        shape,
        _compute_config(ttnn_mesh_device, fidelity),
        label=f"shape={tuple(shape)} fidelity={fid_name}",
    )
