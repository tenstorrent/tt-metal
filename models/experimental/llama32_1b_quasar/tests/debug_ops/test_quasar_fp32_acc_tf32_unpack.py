# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone repro for the Quasar sim's unimplemented bf16 -> Tf32 unpack, seen during llama32_1b decode.

During the first decode-layer sharded RMSNorm the Quasar functional simulator raised:

    [ttsim-qsr] ERROR: UnimplementedFunctionality: qsr_unpack_src_value: in_format=5 out_format=4

in_format=5 = Float16_b (bf16), out_format=4 = Tf32 (tt::DataFormat: Float16_b=5, Tf32=4). i.e. the
unpacker is asked to convert a bf16 source value to Tf32 -- the representation the FPU uses for
fp32-precision math -- and the sim does not implement that conversion. The trigger is a compute op that
does fp32 accumulation of bf16 inputs, i.e. a compute_kernel_config with ``fp32_dest_acc_en=True`` (the
model's rmsnorm_1d.py builds WormholeComputeKernelConfig(math_fidelity=HiFi2, fp32_dest_acc_en=True)).
With ``fp32_dest_acc_en=False`` the math stays bf16 (bf16 -> bf16 unpack, which the sim implements).

Parametrized on ``fp32_dest_acc_en``:
  * fp32acc_off -> PASSES  (validates the workaround: force fp32_dest_acc_en=False on Quasar)
  * fp32acc_on  -> FAILS on Quasar (hits the unimplemented bf16 -> Tf32 unpack); PASSES on WH/BH

The failing case is a plain FAIL (NOT xfail): the craq-sim tooling drives off failing tests. It is likely
a genuine sim gap (bf16 -> Tf32 is a standard HW unpack) -- also worth filing against libttsim.

No weight is used (rms_norm's weight is optional) so the repro isolates the fp32-acc unpack and avoids the
unrelated degenerate-tilize / reshape paths. Input is height-32 (tilizes cleanly).

Run (Quasar sim):
    MESH_DEVICE=<qsr> TT_METAL_SIMULATOR=~/sim/libttsim.so \
        pytest tests/ttnn/unit_tests/operations/test_quasar_fp32_acc_tf32_unpack.py
"""

import pytest
import torch
from loguru import logger

import ttnn


def _readback(tt, mesh_device):
    try:
        num = mesh_device.get_num_devices()
    except Exception:
        num = 1
    if num > 1:
        return ttnn.to_torch(tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return ttnn.to_torch(tt)


def _tile_bf16_dram(t_bf16, mesh_device):
    """bf16 TILE, DRAM-interleaved tensor without the mainline from_torch(TILE) tilize (which hangs on the
    Quasar sim): upload row-major, then tilize with the Gen2-native quasar op where available. Requires a
    tile-aligned height (>=32, %32==0)."""
    rm = ttnn.from_torch(
        t_bf16,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    try:
        return ttnn.experimental.quasar.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    except (AttributeError, RuntimeError) as e:
        logger.info(f"[tf32-repro] quasar.tilize unavailable ({e}); using mainline ttnn.tilize")
        return ttnn.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _compute_cfg(fp32_dest_acc_en):
    # Mirrors the model's rmsnorm compute config (rmsnorm_1d.py:400-403); only fp32_dest_acc_en varies.
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=False,
    )


# NOTE: NOT marked xfail on purpose -- the craq-sim tooling requires a real FAIL on the broken case.
# On Quasar: fp32acc_off passes, fp32acc_on fails (bf16->Tf32 unpack unimplemented). On WH/BH both pass.
_FP32_ACC = [
    pytest.param(False, id="fp32acc_off"),
    pytest.param(True, id="fp32acc_on"),
]


@pytest.mark.parametrize("fp32_dest_acc_en", _FP32_ACC)
def test_rms_norm_fp32_acc_tf32(mesh_device, fp32_dest_acc_en):
    """ttnn.rms_norm on bf16 with fp32_dest_acc_en toggled -- the exact op/config path that faulted.

    fp32_dest_acc_en=True unpacks bf16 -> Tf32 for the fp32 reduce/eltwise math, which the Quasar sim does
    not implement -> this case FAILS on Quasar (and passes on WH/BH). fp32_dest_acc_en=False keeps bf16.
    """
    dim = 2048
    torch.manual_seed(0)
    x = torch.randn(1, 1, 32, dim, dtype=torch.bfloat16)
    xt = _tile_bf16_dram(x, mesh_device)  # height 32 -> tilizes cleanly

    logger.info(f"[tf32-repro] rms_norm (1,1,32,{dim}) bf16 fp32_dest_acc_en={fp32_dest_acc_en}")
    out = ttnn.rms_norm(xt, epsilon=1e-5, compute_kernel_config=_compute_cfg(fp32_dest_acc_en))
    ot = _readback(out, mesh_device).float()
    logger.info("[tf32-repro] rms_norm readback complete")

    assert torch.isfinite(ot).all(), "non-finite after rms_norm"
    # Reference RMSNorm (row-wise, no weight), loose tolerance for bf16.
    xf = x.float()
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)
    assert torch.allclose(ot.reshape(ref.shape), ref, atol=0.1, rtol=0.1), "value mismatch after rms_norm"
