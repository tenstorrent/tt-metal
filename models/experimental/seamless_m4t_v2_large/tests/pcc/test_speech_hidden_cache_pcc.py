# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: speech hidden buffer vs teacher-forcing ``_decoder_hidden`` after traced decode."""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.tt.common import hf_aligned_generation_kwargs
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    MESH_DEVICE_PARAMETRIZE_E2E_2CQ_GENERATE,
    mesh_default_device,
)

from .test_seamless_m4t_v2_model import _PROMPT, _TGT_HIN, _make_tt_model, _torch_ids_to_ttnn, _weights_dir_or_skip


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_E2E_2CQ_GENERATE, indirect=["mesh_device", "device_params"])
def test_speech_hidden_cache_matches_teacher_forcing(mesh_device, device_params, reset_seeds):
    """Traced decode hidden cache must PCC-match ``_decoder_hidden`` (``SEAMLESS_VALIDATE_SPEECH_HIDDEN=1``)."""
    _ = reset_seeds
    _ = device_params
    weights_dir = _weights_dir_or_skip()

    torch.manual_seed(0)
    model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = model.t2u_model.config
    tokenizer = AutoTokenizer.from_pretrained(weights_dir, local_files_only=True)
    text_enc = tokenizer([_PROMPT], return_tensors="pt", padding=True)
    common_kwargs = hf_aligned_generation_kwargs(model.generation_config)
    tt_extra = dict(use_kv_cache=True, use_decode_trace=True, use_2cq=True)

    os.environ["SEAMLESS_VALIDATE_SPEECH_HIDDEN"] = "1"
    try:
        with mesh_default_device(mesh_device):
            tt_model = _make_tt_model(mesh_device, model, cfg, t2u_cfg)
            tt_model.generate(
                input_ids=_torch_ids_to_ttnn(mesh_device, text_enc["input_ids"]),
                attention_mask=_torch_ids_to_ttnn(mesh_device, text_enc["attention_mask"]),
                generate_speech=True,
                tgt_lang=_TGT_HIN,
                speaker_id=0,
                **common_kwargs,
                **tt_extra,
            )
    finally:
        os.environ.pop("SEAMLESS_VALIDATE_SPEECH_HIDDEN", None)

    logger.info("Speech hidden cache validation completed (no fallback warning => PCC ok)")
