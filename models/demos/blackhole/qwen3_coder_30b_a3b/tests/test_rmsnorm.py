# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""First on-device module for Qwen3-Coder-30B-A3B: RMSNorm vs the HF reference.

RMSNorm is the smallest piece of the decoder layer, so it is brought up first --
it proves the whole harness (open a 1x1 mesh, upload real checkpoint weights,
run, compare PCC against the layer-only reference) before GQA, RoPE and the MoE
block are layered on top. When attention PCC later misbehaves, normalisation is
already ruled out.

Two Qwen3-specific details are asserted rather than assumed:
  * eps is 1e-6, not the module default of 1e-5 -- so the config path is used.
  * the norm is the plain variant; ``add_unit_offset`` stays False. Qwen3.5/3.6
    use a zero-centred RMSNorm that folds a "+1" into the weight; this
    checkpoint does not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig
from models.common.utility_functions import comp_allclose, comp_pcc

from .reference import build_reference_layer

LAYER_IDX = 0
PCC_REQUIRED = 0.999  # normalisation alone should be near-exact; the layer bar is 0.995


@pytest.fixture(scope="module")
def reference():
    return build_reference_layer(LAYER_IDX)


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize(
    "norm_name,seq_len,mode",
    [
        ("input_layernorm", 32, "prefill"),
        ("input_layernorm", 128, "prefill"),
        ("post_attention_layernorm", 32, "prefill"),
        ("input_layernorm", 1, "decode"),
    ],
    ids=["input_s32", "input_s128", "postattn_s32", "input_decode"],
)
def test_rmsnorm_vs_reference(
    mesh_device: ttnn.MeshDevice,
    reference,
    norm_name: str,
    seq_len: int,
    mode: str,
):
    layer, config = reference
    torch.manual_seed(0)

    ref_norm = getattr(layer, norm_name)
    dim = config.hidden_size
    assert config.rms_norm_eps == 1e-6, f"unexpected eps {config.rms_norm_eps}"

    # Activations scaled like what actually reaches a decoder layer, per the
    # skill's guidance -- not arbitrary large randoms.
    torch_input = (torch.randn(1, 1, seq_len, dim, dtype=torch.float32) * 0.02).to(torch.bfloat16)

    ttnn.SetDefaultDevice(mesh_device)
    try:
        cache_dir = Path("model_cache/qwen3_coder_30b_a3b/rmsnorm")
        lazy_weight = LazyWeight(
            source=ref_norm.weight.data.clone(),
            dtype=ttnn.bfloat16,
            cache_dir_weight_name=(cache_dir, f"{norm_name}_L{LAYER_IDX}"),
        )
        tt_model = RMSNorm1D.from_config(
            RMSNorm1DConfig(
                weight=lazy_weight,
                eps=config.rms_norm_eps,
                add_unit_offset=False,  # plain RMSNorm, not the zero-centred Qwen3.5/3.6 variant
            )
        )
        tt_out = tt_model.forward(LazyWeight(source=torch_input, dtype=ttnn.bfloat16), mode=mode)
        tt_out_torch = to_torch_auto_compose(tt_out)
    finally:
        ttnn.SetDefaultDevice(None)

    with torch.no_grad():
        ref_out = ref_norm(torch_input.to(torch.float32)).to(torch.bfloat16)

    passing, pcc_message = comp_pcc(ref_out, tt_out_torch, PCC_REQUIRED)
    logger.info(comp_allclose(ref_out, tt_out_torch))
    logger.info(f"RMSNorm[{norm_name}] {mode} seq={seq_len}: {pcc_message}")
    assert passing, f"RMSNorm {norm_name} ({mode}, seq={seq_len}) below {PCC_REQUIRED}: {pcc_message}"
