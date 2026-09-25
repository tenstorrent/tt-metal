# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compute-kernel configs and weight-cache naming shared by the modules."""

import ttnn


def compute_config(mesh_device, fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=None):
    """Bring-up default for every projection matmul and norm: HiFi4 with fp32 dest accumulation.
    ``packer_l1_acc=None`` takes the ``MISTRAL_PRECISION`` setting (on by default)."""
    if packer_l1_acc is None:
        from .precision import precision

        packer_l1_acc = precision().packer_l1_acc
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )


def cache_name(tensor_cache_path, name):
    return f"{tensor_cache_path}/{name}" if tensor_cache_path else None


def dtype_tag(dtype) -> str:
    return {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bf8", ttnn.bfloat4_b: "bf4", ttnn.float32: "fp32"}[dtype]
