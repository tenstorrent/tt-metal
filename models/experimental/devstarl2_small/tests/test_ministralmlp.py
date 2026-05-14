# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: Hugging Face ``Ministral3MLP`` vs ``TtMinistralMLP`` on Devstral text weights.

Applies the shared Devstral FP8 scalar-scale compat patch at import.
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger
from transformers.models.ministral3.modeling_ministral3 import Ministral3MLP

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstarl2_small.devstral_utils import apply_fp8_dequantize_compat
from models.experimental.devstarl2_small.tt.tt_ministralmlp import TtMinistralMLP
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model_config import ModelArgs

apply_fp8_dequantize_compat()

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
def test_ministral3_mlp_pcc_devstral_weights(
    mesh_device,
    seq_len,
    batch_size,
    monkeypatch,
    trust_remote_ministral,
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

    try:
        meta_state_dict = model_args.load_state_dict()
    except Exception as exc:
        pytest.skip(f"Full checkpoint load failed (memory / hub / env): {exc}")

    hf_full = model_args.cached_hf_model
    assert hf_full is not None

    text_root = _text_model_root(hf_full.model)
    hf_mlp = text_root.layers[0].mlp
    assert isinstance(hf_mlp, Ministral3MLP), type(hf_mlp)
    hf_mlp.eval()

    torch_in = torch.randn(1, 1, seq_len, model_args.dim, dtype=torch.bfloat16)
    ref_out = hf_mlp(torch_in)

    tt_ccl = TT_CCL(mesh_device)
    tt_mlp = TtMinistralMLP(
        mesh_device,
        tt_ccl,
        model_args,
        meta_state_dict,
        model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        model_config=model_args.get_model_config(),
    )

    tt_in = ttnn.from_torch(
        torch_in,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 3) if model_args.is_galaxy else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
        dtype=dtype,
        memory_config=model_args.get_mlp_input_mem_config(Mode.PREFILL, None),
        layout=ttnn.TILE_LAYOUT,
    )

    tt_out = tt_mlp(tt_in, Mode.PREFILL)
    tt_torch = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )
    tt_torch = tt_torch[:, :1, :, :]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(ref_out, tt_torch, pcc_required)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC below {pcc_required}: {pcc_message}"
