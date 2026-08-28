# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# >>> MACHINE-GENERATED stub (ADAPT — canonical-wrapper path) <<<
"""ADAPT-status stub for `mlp` of `black-forest-labs/FLUX.2-klein-9B (subfolder `vae`)`.

The reuse_registry mapped this component to the canonical TT impl at:

    models/tt_transformers/tt/mlp.py

Status was REUSE; the global PCC gate then failed → force_adapt_all
demoted to ADAPT. This stub IS the per-component starting point:
it imports the canonical class and delegates forward to it.

ADAPT semantics:
  Iter 0 (no LLM): run this stub as-is. If per-component PCC >= 0.99,
    GRADUATE — the canonical impl was already correct for this model.
  Iter 1+ (LLM only enters if iter 0 PCC < 0.99):
    LLM REFINES this stub. Allowed edits: change ModelArgs config,
    adjust constructor args, add small adapter logic in __call__.
    FORBIDDEN: rewriting the canonical class, replacing the import,
    writing your own ttnn ops from scratch, delegating to torch.
"""
from __future__ import annotations

import ttnn
from models.tt_transformers.tt.mlp import MLP


class TtMlp:
    """Adapter that wraps the canonical TT impl from `models/tt_transformers/tt/mlp.py`.

    Delegates __call__ to the canonical instance. The LLM may refine
    __init__ / build / __call__ here on PCC failure, but MUST keep the
    canonical class as the underlying implementation.
    """

    def __init__(self, canonical_instance) -> None:
        self._impl = canonical_instance

    @classmethod
    def build(cls, device, torch_module):
        # Pre-wired construction using tt_transformers helpers.
        # If iter-0 PCC < 0.99, the LLM refines THIS function — usually by
        # adjusting ModelArgs constructor args for this model's specifics.
        try:
            from models.tt_transformers.tt.model_config import ModelArgs

            args = ModelArgs(mesh_device=device, instruct=True)
        except Exception as _ma_exc:
            # ModelArgs construction failed — surface for LLM to handle.
            raise RuntimeError(
                f"ModelArgs(mesh_device=device) failed for 'mlp': "
                f"{type(_ma_exc).__name__}: {_ma_exc}. LLM refinement: "
                f"adjust the ModelArgs constructor args to match this model's "
                f"config."
            )

        # Build the canonical instance. The exact constructor signature
        # varies per class (Attention takes 14 args, MLP takes 11, RMSNorm
        # different, RotaryEmbedding different). The LLM refines this call
        # on PCC failure to pass the right args for this model.
        try:
            canonical = MLP(
                mesh_device=device,
                args=args,
                state_dict=torch_module.state_dict() if torch_module is not None else None,
                layer_num=0,
                dtype=ttnn.bfloat16,
            )
        except Exception as _ctor_exc:
            raise RuntimeError(
                f"Canonical { 'MLP' } constructor failed: "
                f"{type(_ctor_exc).__name__}: {_ctor_exc}. LLM refinement: "
                f"pass the additional required args (tt_ccl, "
                f"transformation_mats, configuration, etc.). See "
                f"`models/tt_transformers/tt/mlp.py` for the full __init__ signature."
            )

        return cls(canonical)

    def __call__(self, *args, **kwargs):
        # Delegate to the canonical impl. LLM may add minor pre/post
        # processing here (input reshape, output unpack) on PCC failure
        # — but NEVER replace this delegation with a custom forward.
        return self._impl(*args, **kwargs)


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtMlp.build(device, torch_module)


# Module-level shim with the component's lowercase slug name. Kept for
# backward compatibility with legacy SMOKE/PCC tests that import the
# slug directly.
def mlp(device, torch_module=None):
    return TtMlp.build(device, torch_module)
