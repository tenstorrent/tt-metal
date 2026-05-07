# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: Hugging Face ``Ministral3RMSNorm`` vs ``TtMinistralRMSNorm`` on Devstral text weights.

At import time this module patches ``Fp8Dequantize._dequantize_one`` so scalar FP8 scales from
Devstral checkpoints do not trip Hugging Face's fine-grained FP8 dequantizer (same as
``test_ministralattn.py`` / ``demo_devstral2_tt_multimodal.py``).
"""

from __future__ import annotations

import os
import types

import pytest
import torch
from loguru import logger
from transformers.integrations.finegrained_fp8 import Fp8Dequantize
from transformers.models.ministral3.modeling_ministral3 import Ministral3RMSNorm

# Devstral HF checkpoints can ship scalar FP8 scales; ``Fp8Dequantize._dequantize_one`` expects a 2D
# scale grid and raises unless we patch before ``ModelArgs.load_state_dict()`` / ``from_pretrained``.
_ORIGINAL_FP8_DEQUANTIZE_ONE = Fp8Dequantize._dequantize_one


def _dequantize_one_compat(self, quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if scales.ndim == 0:
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if quantized.dtype == torch.int8 or (fp4_dtype is not None and quantized.dtype == fp4_dtype):
            quantized_fp32 = self._unpack_fp4(quantized)
        else:
            quantized_fp32 = quantized.to(torch.float32)
        out_dtype = scales.dtype if scales.dtype.is_floating_point and scales.element_size() >= 2 else torch.bfloat16
        scale = scales.to(torch.float32)
        return (quantized_fp32 * scale).to(out_dtype)
    return _ORIGINAL_FP8_DEQUANTIZE_ONE(self, quantized, scales)


Fp8Dequantize._dequantize_one = _dequantize_one_compat

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstarl2_small.tt.tt_ministralrmsnorm import TtMinistralRMSNorm
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model_config import ModelArgs

DEVSTRAL_REPO_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"


def _text_model_root(multimodal_inner):
    lm = multimodal_inner.language_model
    return lm.model if hasattr(lm, "model") else lm


@pytest.fixture
def trust_remote_ministral(monkeypatch):
    from models.tt_transformers.tt import model_config as mc

    orig_set = mc.ModelArgs._set_hf_params

    def _set_hf_params_trust(self, checkpoint_dir: str):
        self.trust_remote_code_hf = True
        return orig_set(self, checkpoint_dir)

    monkeypatch.setattr(mc.ModelArgs, "_set_hf_params", _set_hf_params_trust)

    def _get_hf_model_cls_devstral_safe(self):
        from transformers import AutoModelForCausalLM
        from transformers.models.auto.modeling_auto import AutoModelForImageTextToText

        if not self.is_multimodal:
            return AutoModelForCausalLM
        if type(self.hf_config) in AutoModelForImageTextToText._model_mapping:
            return AutoModelForImageTextToText
        raise ValueError(
            f"Test supports multimodal configs in AutoModelForImageTextToText only; got {type(self.hf_config)}"
        )

    monkeypatch.setattr(mc.ModelArgs, "get_hf_model_cls", _get_hf_model_cls_devstral_safe)


@torch.no_grad()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (128,))
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("post_attention", (False, True))
def test_ministral3_rmsnorm_pcc_devstral_weights(
    mesh_device,
    seq_len,
    batch_size,
    monkeypatch,
    trust_remote_ministral,
    post_attention,
):
    monkeypatch.setenv("HF_MODEL", DEVSTRAL_REPO_ID)

    dtype = ttnn.bfloat16
    model_args = ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max(512, seq_len),
        dummy_weights=False,
        use_hf_rope=True,
        cache_hf=True,
    )

    # Hidden dim > 4096 + multichip prefill selects distributed RMSNorm; its sharded gamma (e.g. 1280/chip)
    # does not match a fully replicated activations PCC layout vs single-device HF — force replicated path.
    model_args.is_distributed_norm = types.MethodType(lambda self, mode: False, model_args)

    try:
        meta_state_dict = model_args.load_state_dict()
    except Exception as exc:
        pytest.skip(f"Full checkpoint load failed (memory / hub / env): {exc}")

    hf_full = model_args.cached_hf_model
    assert hf_full is not None

    text_root = _text_model_root(hf_full.model)
    hf_norm = text_root.layers[0].post_attention_layernorm if post_attention else text_root.layers[0].input_layernorm
    assert isinstance(hf_norm, Ministral3RMSNorm), type(hf_norm)
    hf_norm.eval()

    torch_in = torch.randn(batch_size, 1, seq_len, model_args.dim, dtype=torch.bfloat16)
    ref_out = hf_norm(torch_in)

    tt_ccl = TT_CCL(mesh_device)
    tt_norm = TtMinistralRMSNorm(
        mesh_device,
        model_args,
        meta_state_dict,
        model_args.weight_cache_path(dtype),
        layer_num=0,
        tt_ccl=tt_ccl,
        post_attention=post_attention,
    )

    tt_in = ttnn.from_torch(
        torch_in,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_out = tt_norm(tt_in, Mode.PREFILL)
    # Replicated norm output per device: concat mesh on dim 0 then take one replica (same pattern as Grok RMS PCC).
    tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
    if tt_torch.shape != ref_out.shape:
        tt_torch = tt_torch.reshape(ref_out.shape)

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(ref_out, tt_torch, pcc_required)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC below {pcc_required}: {pcc_message}"
