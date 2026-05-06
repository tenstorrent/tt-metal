# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
PCC test: Hugging Face ``Ministral3RotaryEmbedding`` vs
``TtDevstral2LargeRotaryEmbedding`` (TT tables from ``devstral2_large/tt``).

Uses ``mistralai/Devstral-2-123B-Instruct-2512`` config only—no full weight load. HF RoPE cos/sin
are computed from ``Ministral3Config`` (``inv_freq`` / YaRN via ``rope_parameters``), matching what
``TtDevstral2LargeRotaryEmbedding`` builds from the same config on device.

This avoids OOM / SIGKILL from loading ~123B parameters into host RAM for a table PCC test.

Requirements:
- Network or cached Hub files for the repo ``config.json`` (and custom code if required).
- Sequence length multiple of 128 (same convention as ``devstarl2_small`` rotary PCC).
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.ministral3.modeling_ministral3 import Ministral3RotaryEmbedding

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_large.tt.tt_ministral_rotary_emb import TtDevstral2LargeRotaryEmbedding

DEVSTRAL2_LARGE_REPO_ID = "mistralai/Devstral-2-123B-Instruct-2512"


def _ministral3_config_from_hf(hf_cfg) -> Ministral3Config:
    """Multimodal checkpoints expose ``text_config``; causal-only models use the root config."""
    inner = getattr(hf_cfg, "text_config", None)
    out = inner if inner is not None else hf_cfg
    if not isinstance(out, Ministral3Config):
        raise TypeError(f"Expected Ministral3Config, got {type(out)!r}")
    return out


def _tt_table_to_torch_bf16(tt_tensor: ttnn.Tensor) -> torch.Tensor:
    """Host tensor ``[max_seq_len, head_dim]`` from replicated TT rotary cache ``[1,1,S,D]``."""
    # RoPE tables use ``replicate_tensor_to_mesh_mapper``; ``ttnn.to_torch`` on a mesh tensor
    # requires a mesh composer. One replicated shard matches HF (all chips hold the same table).
    device_tensors = ttnn.get_device_tensors(tt_tensor)
    if device_tensors is not None and len(device_tensors) > 0:
        th = ttnn.to_torch(device_tensors[0])
    else:
        th = ttnn.to_torch(tt_tensor)
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
def test_ministral3_rotary_embedding_pcc_devstral2_large_config(
    mesh_device,
    seq_len,
    batch_size,
):
    try:
        hf_cfg = AutoConfig.from_pretrained(
            DEVSTRAL2_LARGE_REPO_ID,
            trust_remote_code=True,
            local_files_only=os.getenv("CI") == "true",
        )
        print(hf_cfg)
    except Exception as exc:
        pytest.skip(f"Could not load Hugging Face config (network / cache): {exc}")

    text_cfg = _ministral3_config_from_hf(hf_cfg)

    rotary = Ministral3RotaryEmbedding(text_cfg)
    assert isinstance(
        rotary, Ministral3RotaryEmbedding
    ), f"Expected HF Ministral3RotaryEmbedding; got {type(rotary).__module__}.{type(rotary).__name__}"

    rotary.eval()
    hidden_size = text_cfg.hidden_size
    head_dim = int(getattr(text_cfg, "head_dim", None) or text_cfg.hidden_size // text_cfg.num_attention_heads)
    x = torch.zeros(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    cos_ref, sin_ref = rotary(x, position_ids=position_ids)
    assert cos_ref.shape == (batch_size, seq_len, head_dim)
    assert sin_ref.shape == (batch_size, seq_len, head_dim)

    max_rotary_seq = max(512, seq_len)
    dtype = ttnn.bfloat16
    tt_rot = TtDevstral2LargeRotaryEmbedding(
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

    pcc_required = 0.999
    ok_c, msg_c = comp_pcc(cos_ref_f, cos_tt, pcc_required)
    ok_s, msg_s = comp_pcc(sin_ref_f, sin_tt, pcc_required)
    logger.info(comp_allclose(cos_ref_f, cos_tt))
    logger.info(comp_allclose(sin_ref_f, sin_tt))
    logger.info(f"cos PCC: {msg_c}")
    logger.info(f"sin PCC: {msg_s}")
    assert ok_c, f"cos PCC below {pcc_required}: {msg_c}"
    assert ok_s, f"sin PCC below {pcc_required}: {msg_s}"
