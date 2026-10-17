# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for gpt_oss_120b: the framework reaches the model only through these functions.

Device side (implement stages): device_component, device_model, and optionally device_params. The CPU side
(reference, tokenizer, hf_model, hf_layers) is in reference_hooks.py, the reference stage's.
The block-graph contract is in reference/_bringup_interface.py (vendored, do not edit). This package never imports
the framework (bringup_fw); the framework's lint fails it if it does.
"""

import json
from pathlib import Path

NORM_WEIGHTS = {"attn_norm": "input_layernorm.weight", "ffn_norm": "post_attention_layernorm.weight"}


def _checkpoint(spec):
    from models.demos.gpt_oss_120b.reference.gpt_oss_ref import Checkpoint

    return Checkpoint(Path(spec.hf_dir))


def _config(spec) -> dict:
    return json.loads((Path(spec.hf_dir) / "config.json").read_text())


def _observable_ops():
    """Turn off ttnn's fast runtime mode so ttnn.register_pre_operation_hook sees every op. In fast mode (this
    tt-metal's default) FastOperation never calls PRE_OPERATION_HOOKS, so the framework's device-execution proof
    counts 0 ops for a module that does run on the device. Ops run the same; only their Python wrapper changes."""
    import ttnn

    ttnn.CONFIG.enable_fast_runtime_mode = False


class _HostBoundary:
    """Component harness boundary: [S, H] host in -> device module on the mesh layout -> [S, H] host out."""

    def __init__(self, mesh, module):
        self.mesh, self.module = mesh, module

    def __call__(self, ctx, x):
        from models.demos.gpt_oss_120b.tt.layout import rows_to_device, rows_to_host

        tt_x = rows_to_device(self.mesh, x)
        tt_y = self.module(tt_x)
        out = rows_to_host(self.mesh, tt_y)
        tt_x.deallocate(True)
        tt_y.deallocate(True)
        return out


def device_component(mesh, spec, layer, step):
    _observable_ops()
    if step in NORM_WEIGHTS:
        from models.demos.gpt_oss_120b.tt.norm import TtRmsNorm

        ck = _checkpoint(spec)
        w = ck.get(f"model.layers.{layer}.{NORM_WEIGHTS[step]}")
        ck.close()
        return _HostBoundary(mesh, TtRmsNorm(mesh, w, _config(spec)["rms_norm_eps"]))
    raise NotImplementedError(f"implement stage: no device module for {step} yet")


def device_model(mesh, spec, layers, lm_head=True):
    raise NotImplementedError("assemble stage: no device model yet")
