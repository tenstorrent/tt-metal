# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# PCC: HF Ministral3RotaryEmbedding vs TtMinistral3RotaryEmbedding (cached HF rotary_emb ref).

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.ministral3.modeling_ministral3 import Ministral3RotaryEmbedding

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstarl2_small.devstral_utils import apply_fp8_dequantize_compat
from models.experimental.devstarl2_small.tt.tt_ministral_rotary_emb import TtMinistral3RotaryEmbedding
from models.tt_transformers.tt.model_config import ModelArgs

apply_fp8_dequantize_compat()

DEVSTRAL_REPO_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"


def _text_model_root(multimodal_inner):
    """``Mistral3Model.language_model`` → inner causal stack with ``layers`` and ``rotary_emb``."""
    lm = multimodal_inner.language_model
    return lm.model if hasattr(lm, "model") else lm


@pytest.fixture
def trust_remote_ministral(monkeypatch):
    """Trust remote code for config/load, and avoid ``AutoModelForVision2Seq`` where needed."""

    from models.tt_transformers.tt import model_config as mc

    orig_set = mc.ModelArgs._set_hf_params

    def _set_hf_params_trust(self, checkpoint_dir: str):
        self.trust_remote_code_hf = True
        return orig_set(self, checkpoint_dir)

    monkeypatch.setattr(mc.ModelArgs, "_set_hf_params", _set_hf_params_trust)

    def _get_hf_model_cls_devstral_safe(self):
        """Mistral3 / Devstral map to ``AutoModelForImageTextToText``; skip broken top-level Vision2Seq import."""
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


def _tt_table_to_torch_bf16(tt_tensor: ttnn.Tensor) -> torch.Tensor:
    """Host tensor ``[max_seq_len, head_dim]`` from replicated TT rotary cache ``[1,1,S,D]``."""
    th = ttnn.to_torch(ttnn.get_device_tensors(tt_tensor)[0])
    while th.dim() > 2:
        th = th.squeeze(0)
    return th


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
def test_ministral3_rotary_embedding_pcc_devstral_weights(
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

    text_cfg = model_args.hf_config.text_config
    assert isinstance(text_cfg, Ministral3Config), f"Expected Ministral3Config text_config, got {type(text_cfg)!r}"

    try:
        model_args.load_state_dict()
    except Exception as exc:
        pytest.skip(f"Full checkpoint load failed (memory / hub / env): {exc}")

    hf_full = model_args.cached_hf_model
    assert hf_full is not None, "Expected cached HF model after load_state_dict with cache_hf=True."

    text_root = _text_model_root(hf_full.model)
    rotary = text_root.rotary_emb
    assert isinstance(
        rotary, Ministral3RotaryEmbedding
    ), f"Expected HF Ministral3RotaryEmbedding; got {type(rotary).__module__}.{type(rotary).__name__}"

    rotary.eval()
    hidden_size = model_args.dim
    head_dim = int(text_cfg.head_dim)
    x = torch.zeros(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    cos_ref, sin_ref = rotary(x, position_ids=position_ids)
    assert cos_ref.shape == (batch_size, seq_len, head_dim)
    assert sin_ref.shape == (batch_size, seq_len, head_dim)

    max_rotary_seq = model_args.max_seq_len
    tt_rot = TtMinistral3RotaryEmbedding(
        mesh_device,
        batch_size=batch_size,
        head_dim=head_dim,
        max_seq_len=max_rotary_seq,
        config=text_cfg,
        datatype=dtype,
    )

    cos_tt_2d = _tt_table_to_torch_bf16(tt_rot.cos_matrix)[:seq_len, :]
    sin_tt_2d = _tt_table_to_torch_bf16(tt_rot.sin_matrix)[:seq_len, :]

    cos_tt = cos_tt_2d.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=torch.float32)
    sin_tt = sin_tt_2d.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=torch.float32)

    cos_ref_f = cos_ref.float()
    sin_ref_f = sin_ref.float()

    pcc_required = 0.99
    ok_c, msg_c = comp_pcc(cos_ref_f, cos_tt, pcc_required)
    ok_s, msg_s = comp_pcc(sin_ref_f, sin_tt, pcc_required)
    logger.info(comp_allclose(cos_ref_f, cos_tt))
    logger.info(comp_allclose(sin_ref_f, sin_tt))
    logger.info(f"cos PCC: {msg_c}")
    logger.info(f"sin PCC: {msg_s}")
    assert ok_c, f"cos PCC below {pcc_required}: {msg_c}"
    assert ok_s, f"sin PCC below {pcc_required}: {msg_s}"
