# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Plain RMSNorm for the DFlash drafter (M1 + M2).

``output = x * rsqrt(mean(x^2) + eps) * weight``

**Not** the target's convention. qwen36 uses *zero-centered* RMSNorm --
``x * rsqrt(...) * (1 + weight)`` -- and pre-adds the 1.0 into every gain at load time
(``tt/rms_norm.py``, ``tt/attention/weights.py:14``). The drafter is ``model_type: qwen3``
with stock ``Qwen3RMSNorm``, so its gains are used as-is. Reusing the target's norm here
would apply a silent +1 offset to all 12 of the drafter's norm weights; reusing the target's
*loader* would bake it into the weights instead. ``weights.py`` never folds, and
``test_rms_norm_tp.py`` grades against the plain HF module.

One function serves both places the drafter normalises, because ``ttnn.rms_norm``
normalises over the last dim and both uses are last-dim:

* **M1, hidden norms** (``input_layernorm``, ``post_attention_layernorm``, ``hidden_norm``,
  ``norm``): ``[..., 5120]`` with a ``[5120]`` gain. The residual stream is replicated under
  TP, so the statistics are already whole -- no distributed-norm CCL.
* **M2, per-head QK norms** (``q_norm``, ``k_norm``): ``[..., heads, 128]`` with a ``[128]``
  gain, normalising each (position, head) vector independently. Heads are sharded across
  devices but ``head_dim`` never is, so this is device-local too.

Note ``k_norm`` is applied to the **concatenated** ctx+block K, i.e. over ``ctx_len + 16``
positions, not just the block -- see ``attention.py``.
"""

import ttnn


def rms_norm(x, weight, eps: float, memory_config=None):
    """Plain RMSNorm over the last dim of ``x``.

    ``weight`` must be the checkpoint's gain, unmodified.
    """
    return ttnn.rms_norm(x, weight=weight, epsilon=eps, memory_config=memory_config)
