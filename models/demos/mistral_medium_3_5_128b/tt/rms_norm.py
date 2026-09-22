# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 RMSNorm. Ported from ``gpt_oss_d_p/tt/rms_norm.py`` (``use_gemma_norm=False``
branch), with the Gemma fold and the distributed branch deleted rather than disabled.

Two model facts make this the whole module:

* **Plain RMSNorm.** ``Ministral3RMSNorm`` is ``weight * x / sqrt(mean(x^2) + eps)``. There is no
  Gemma ``(1 + weight)`` fold, so the ``use_gemma_norm`` switch the sources carry has no valid
  setting here and is dropped. ``tests/unit/test_reference_modeling.py`` pins this on the reference
  side and ``tests/unit/test_norm_vs_ref.py::test_rms_norm_is_not_gemma`` on the device side (zero
  weight must give zero output, which the Gemma form would not).
* **Local, not distributed.** The prefill activation is ``[1, 1, tokens/sp, hidden_size]`` with the
  sequence SP-sharded on the mesh rows and ``hidden`` **replicated** across the TP cols (see
  ``tt/config.py``). Every chip therefore holds the full hidden vector for its own tokens, so the
  RMS reduction is entirely local: one ``ttnn.rms_norm``, no stats all-gather. The sources'
  distributed branch exists for a hidden-sharded residual, which this package does not use.

Two deviations from the sources are forced by ``hidden_size = 12288``, and both are about the
width of the reduction rather than about this model's math.

**The weight is stored ``[1, 1, 1, hidden_size]`` in TILE_LAYOUT**, replicated on every device.
The sources use the row-major ``[1, 1, hidden/TILE_SIZE, TILE_SIZE]`` form instead, and that form
does not survive this width: the row-major path makes ``ttnn.rms_norm`` statically allocate
2565248 B of L1 per core against a 1572864 B budget and the op throws before it runs. A tiled
weight takes a different kernel path with no such blow-up.

**``fp32_dest_acc_en=True`` is mandatory, not a tuning knob.** The default compute config
accumulates the sum of 12288 squares in the bf16 destination register; the small terms stagnate,
the sum comes out ~23% low, and ``rsqrt`` of it scales the whole output up by a *data-dependent*
factor — measured 1.1378 on average with a per-row spread of 1.074..1.202 at hidden 12288. PCC is
invariant to a scale factor, so this is invisible to a norm-level PCC check (0.99985 with the bug,
0.99999 without) and only shows up once something scale-sensitive consumes the output: Q·K logits
grow by 1.1378**2 and sharpen the softmax, and SwiGLU is non-linear, which cost the composed
decoder layer ~0.09 PCC before the flag went in. ``test_norm_vs_ref.py::test_rms_norm_scale``
asserts the output RMS directly so a PCC-only check can never hide it again.

With the flag the single op matches a hand-composed ``pow``/``mean``/``rsqrt`` fp32 chain to
within 1e-6 PCC (0.9999985 vs 0.9999986) at a quarter of the tensors, so the norm stays one op.
"""

import ttnn
from models.demos.mistral_medium_3_5_128b.utils.general_utils import get_cache_file_name


class RMSNorm:
    """``out = x / sqrt(mean(x^2) + eps) * weight``, computed locally on each chip."""

    def __init__(self, mesh_device, config, state_dict, *, mesh_config=None, tensor_cache_path=None):
        """
        Args:
            mesh_device: the open mesh.
            config: a :class:`~...reference.model_config.MistralMediumConfig` (reads ``rms_norm_eps``).
            state_dict: ``{"weight": tensor}`` for this norm. Empty dict => cache-only load.
            mesh_config: unused for the local norm; accepted so every block takes the same arguments.
            tensor_cache_path: directory for the tilized-weight cache, or None.
        """
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.eps = config.rms_norm_eps
        # fp32_dest_acc_en is required for correctness at hidden 12288 — see the module docstring.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        torch_weight = state_dict["weight"].reshape((1, 1, 1, -1)) if state_dict else None
        self.tt_weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def __call__(self, x):
        """``x``: ``[1, 1, tokens_local, hidden_size]`` -> same shape."""
        return ttnn.rms_norm(
            x,
            weight=self.tt_weight,
            epsilon=self.eps,
            compute_kernel_config=self.compute_kernel_config,
        )
