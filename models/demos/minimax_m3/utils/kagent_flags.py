# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill optimization switches (kagent/m3-prefill). M3_KA_ROPE_FUSED and M3_KA_SKIP_IDX_SPLIT are bit-exact and
default ON; the rest default OFF. Set a flag to 0 or 1 to override. Read once at import (set them before the model
modules are imported).

  M3_KA_ROPE_FUSED      1 -> partial RoPE in ONE rotary_embedding_indexed(rotary_dim=64) call instead of
                           slice + slice + rope + concat (bit-exact: the op copies the pass-through channels)
  M3_KA_SKIP_IDX_SPLIT  1 -> skip the MSA index-head split (to_layout/reshape/permute/to_layout) when it maps
                           [1,1,S,D] to itself (one index head per device, i.e. TP=4) (bit-exact)
  M3_KA_MOE_SINGLE_RS   1 -> add the shared-expert and routed partial sums BEFORE one TP reduce-scatter; forces
                           the fused schedule that M3_MOE_FUSE_SHARED_RS already enables by default
  M3_KA_MM_FIDELITY     comma list of matmul groups to run LoFi + fp32 dest acc instead of the HiFi2 default:
                           qkv, index, shared, dense (precision-changing -> L2 gate)
  M3_KA_EXPERT_PLACEMENT  path to a per-layer expert placement (utils/expert_placement.py): relabels the routed
                           experts of each listed MoE layer so the hot ones spread over the 16 chips. The cached
                           expert weights are permuted at load (exact bytes) and the router's expert ids are mapped
                           to labels (M3_KA_EXPERT_RELABEL); routing decisions are unchanged. Column-preserving
                           placements are bit-exact; cross-column ones change the TP reduce-scatter summation
                           grouping (class B). Read at model build.
  M3_KA_EXPERT_RELABEL  gather (default): the router is unchanged and one ttnn.gather per MoE layer maps expert ids
                           to placement labels, so routing decisions stay bit-identical. router: permute the
                           router columns instead (no extra op; changes the gate's TF32-tie resolution -> rejected,
                           docs 32 §3)
"""

import os


def _on(name, default="0"):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


ROPE_FUSED = _on("M3_KA_ROPE_FUSED", "1")
SKIP_IDX_SPLIT = _on("M3_KA_SKIP_IDX_SPLIT", "1")
MOE_SINGLE_RS = _on("M3_KA_MOE_SINGLE_RS")
MM_FIDELITY = {g.strip() for g in os.getenv("M3_KA_MM_FIDELITY", "").split(",") if g.strip()}
EXPERT_PLACEMENT = os.getenv("M3_KA_EXPERT_PLACEMENT", "").strip() or None
EXPERT_RELABEL = os.getenv("M3_KA_EXPERT_RELABEL", "gather").strip().lower()
assert EXPERT_RELABEL in ("gather", "router"), f"M3_KA_EXPERT_RELABEL must be gather or router, got {EXPERT_RELABEL}"


_LOFI = None


def lofi_config(group):
    """Compute-kernel config override for matmul ``group`` (None keeps the op's default). Evaluated per call, so
    a harness may change MM_FIDELITY at run time."""
    global _LOFI
    if group not in MM_FIDELITY:
        return None
    if _LOFI is not None:
        return _LOFI
    import ttnn

    _LOFI = ttnn.init_device_compute_kernel_config(
        ttnn.device.Arch.BLACKHOLE,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    return _LOFI
