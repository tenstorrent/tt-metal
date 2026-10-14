# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for deepseek_v41_flash_d_p: the framework reaches the model only through these functions.

CPU side (reference role): reference, hf_model, hf_layers, state_sources.
Device side (implement role): device_params, device_component, device_model.
Contract side (contract role): contract_independent_pcc (optional).
See models/demos/common/bringup/reference/interface.py and testing/harness.py for the contracts.
"""

import torch

ORACLE_MAX_SEQ = 65536  # positions the oracle's caches hold (parity and sanity prompts are far shorter)


def _config(spec):
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import Config

    return Config.from_dir(hf_path(spec))


def reference(spec, layers=None, dtype=None):
    """The standalone CPU reference (reference/ds41_ref.py), the requested layers resident (fp32)."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import DS41Reference

    ref = DS41Reference(hf_path(spec), layers=layers, dtype=dtype or torch.float32)
    for i in ref.layer_ids:  # the state each layer returns is what the spec's block type declares
        assert ref.cfg.state_names(i) == spec.state_names(i), (i, ref.cfg.state_names(i), spec.state_names(i))
    return ref


def hf_model(spec, num_layers):
    """DeepSeek's inference/model.py on the checkpoint, on the CPU (reference/hf_oracle.py). num_layers=None: the
    whole text model in bf16 (the GPU configuration; intake sanity); else the first num_layers blocks in fp32."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.deepseek_v41_flash_d_p.reference.hf_oracle import build_oracle

    dtype = torch.bfloat16 if num_layers is None else torch.float32
    return build_oracle(hf_path(spec), num_layers, dtype, max_seq_len=ORACLE_MAX_SEQ)


def hf_layers(model):
    """Per-block taps: the block output packed like the reference's boundary, [S, 4 * H + 4] (streams, pending pre)."""
    return model.taps


def state_sources(spec, layer):
    """Earlier layers whose state `layer` reads: its compressed-KV source, its top-k source, layer 20's candidates."""
    return _config(spec).sources(layer)


def device_component(mesh, spec, layer, step):
    raise NotImplementedError(f"implement step: no device module for {step} yet")


def device_model(mesh, spec, layers, lm_head=True):
    raise NotImplementedError("implement step: no device model yet")
