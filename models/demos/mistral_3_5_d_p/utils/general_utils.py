# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""General utilities for Mistral-Medium-3.5 (d_p). Adapted from ``gpt_oss_d_p/utils``, plus the
shared matmul compute config every projection in this package uses (see
:func:`get_matmul_compute_config`)."""

import ttnn
from models.common.utility_functions import is_blackhole


def get_cache_file_name(tensor_cache_path, name):
    return f"{tensor_cache_path}/{name}" if tensor_cache_path else None


def cache_file_exists(cache_file_name):
    """True iff a tilized tensor cache file for `cache_file_name` exists on disk. ttnn appends a
    `_dtype_<DT>_layout_<L>.tensorbin` suffix, so match by prefix. Used to decide whether to load an
    OPTIONAL weight from cache when the source state_dict is absent (cache-only loading) — its
    presence can't be known from an empty state_dict."""
    if not cache_file_name:
        return False
    import glob

    return bool(glob.glob(f"{cache_file_name}*.tensorbin"))


def get_default_num_links(mesh_device):
    """Default number of fabric links for CCL ops on the given mesh.

    Blackhole exposes 2 fabric links per device; Wormhole exposes 4. Single-row meshes
    (shape[0] == 1) only need 1 link regardless of arch.
    """
    if mesh_device.shape[0] == 1:
        return 1
    return 2 if is_blackhole() else 4


def get_matmul_compute_config(
    mesh_device, *, math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
):
    """Compute-kernel config for this model's PROJECTION matmuls (fused QKV, o_proj, MLP gate/up/down).

    ``fp32_dest_acc_en=True`` is the load-bearing setting, and it is why this helper exists rather
    than letting ``ttnn.linear`` take its defaults the way the gpt-oss donor does. Mistral's
    contractions are far deeper than the donor's (hidden 12288 and intermediate 28672 against
    gpt-oss's 2880), so accumulating the dot products in bf16 destination registers loses real
    precision. Measured on the attention block at seq 512, against the fp32 torch reference:

        fp32_dest_acc_en=False   PCC 0.9922   (LoFi 0.9938, HiFi2 0.9922, HiFi4 0.9922)
        fp32_dest_acc_en=True    PCC 0.9999

    — i.e. the accumulation dtype dominates, and the weight dtype (bf8 vs bf16) and the math fidelity
    barely register next to it. Without this, one decoder layer lands at ~0.989 and misses the spec's
    0.99 bar; with it, the layer clears it with room to spare.

    NOTE this is for matmuls only. The SDPA program configs deliberately keep
    ``fp32_dest_acc_en=False``: the ring cache-read op's streaming online-softmax compute requires it.
    """
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )
