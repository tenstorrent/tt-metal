# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Weight dtypes and compute fidelity the TT port chooses, per block.

The architecture is the checkpoint's (`reference/config.py`); this module holds only what the port decides. Re-run
the block PCC tests (`tests/pcc/`) after changing any of these.

Not set here, because tt_dit hard-codes them: the vision blocks' and patch embedding's linears are bf16 (tt_dit
`Linear`'s default; its Qwen3-VL vision modules take no dtype), and the LM attention output projection is bf8
(`LINEAR_DTYPE`). Linears built without a compute config, the LM's and the expert's included, run at tt_dit's
linear default: HiFi2 (for bf16 and bf8 weights), fp32 dest accumulation, packer L1 accumulation.
"""

import ttnn
from models.tt_dit.encoders.transformer import LINEAR_DTYPE

# --- Vision --------------------------------------------------------------------------------------------------------

PROJECTOR_WEIGHT_DTYPE = ttnn.bfloat16

# --- Language model ------------------------------------------------------------------------------------------------

# bf8 (tt_dit's encoder default) holds every LM gate: prefill logits PCC 99.97 %, decode min 99.76 % (README).
LM_QKV_DTYPE = LINEAR_DTYPE
LM_FFN_DTYPE = LINEAR_DTYPE
# bf8. A bf16 head improves the decode logits but not the argmax agreement, for 1.7 % CoT time and 0.26 GB
# (README, "Decode quality and the lm_head dtype").
LM_HEAD_DTYPE = LINEAR_DTYPE

# --- Action expert and action projections --------------------------------------------------------------------------

# Their error feeds the 10 Euler steps. bf16 weights and RoPE tables give a per-step velocity PCC >= 99.998 % on the
# checkpoint; the expert SDPA and the action projections also compute at `accurate_compute_config`.
EXPERT_WEIGHT_DTYPE = ttnn.bfloat16
EXPERT_ROPE_DTYPE = ttnn.bfloat16
ACTION_WEIGHT_DTYPE = ttnn.bfloat16


def accurate_compute_config(mesh_device: ttnn.MeshDevice) -> ttnn.DeviceComputeKernelConfig:
    """HiFi4 with fp32 accumulation: the expert SDPA and the action in/out projections."""
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )
