# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Submodule integration PCC (LM hidden state, DPM scheduler).

Full reference generate() parity is in test_e2e_generate_pcc.py.
"""

import sys
from pathlib import Path

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.vibevoice.common.config import (
    MODEL_PATH,
    DEFAULT_TXT_PATH,
    VOICES_DIR,
)
from models.experimental.vibevoice.tt.load_weights import (
    load_vibevoice_state_dict,
    split_submodule_weights,
    remap_lm_keys_to_tt_transformers,
)
from models.experimental.vibevoice.tt.ttnn_vibevoice_lm import (
    preprocess_lm_weights,
    TTVibeVoiceLM,
    create_kv_cache,
)
from models.experimental.vibevoice.tt.ttnn_dpm_scheduler import (
    TTDPMSolverMultistepScheduler,
)
from models.experimental.vibevoice.tt.vibevoice_config import load_vibevoice_model_config

_VIBEVOICE_ROOT = Path(__file__).resolve().parent.parent.parent
_REFERENCE_DIR = _VIBEVOICE_ROOT / "reference"
for _p in (_REFERENCE_DIR, _VIBEVOICE_ROOT.parent.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_VOICE_PATH = str(VOICES_DIR / "en-Alice_woman.wav")
_TEXT_PATH = str(DEFAULT_TXT_PATH)

CFG_SCALE = 1.3
NUM_STEPS = 10
SEQ_LEN = 32


@pytest.fixture(scope="module")
def vv_config():
    return load_vibevoice_model_config(MODEL_PATH)


@pytest.fixture(scope="module")
def lm_state():
    sd = load_vibevoice_state_dict(MODEL_PATH)
    sub = split_submodule_weights(sd)
    return remap_lm_keys_to_tt_transformers(sub["lm"])


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_e2e_lm_hidden_state_pcc(mesh_device, vv_config, lm_state):
    """LM last_hidden_state PCC >= 0.99 after prefill on synthetic tokens.

    Reference is the **full VibeVoice model's own language_model** (the actual
    Qwen2 instance built from VibeVoice's decoder_config with the checkpoint
    weights) — not a hand-built Qwen2Config — so config/weights exactly match.
    """
    torch.manual_seed(0)

    cfg = vv_config.decoder
    input_ids = torch.randint(0, cfg.vocab_size, (1, SEQ_LEN), dtype=torch.long)

    # Reference: the real VibeVoice model, use its language_model's last hidden state.
    from modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference

    ref_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32, device_map="cpu", attn_implementation="sdpa"
    )
    ref_model.eval()
    ref_lm = ref_model.model.language_model
    with torch.no_grad():
        ref_out = ref_lm(input_ids=input_ids).last_hidden_state  # [1, S, hidden]

    # TT
    weights = preprocess_lm_weights(lm_state, mesh_device, cfg)
    lm_tt = TTVibeVoiceLM(weights, mesh_device)
    kv_cache = create_kv_cache(cfg.num_hidden_layers)
    _, tt_hidden = lm_tt.prefill(input_ids, kv_cache=kv_cache, return_last_hidden=True)
    tt_hidden_torch = ttnn.to_torch(tt_hidden).to(torch.float32).squeeze(1)  # [1, S, hidden]

    passed, pcc_val = comp_pcc(ref_out.to(torch.float32), tt_hidden_torch, pcc=0.99)
    assert passed, f"E2E LM hidden PCC {pcc_val:.6f} < 0.99"


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_e2e_dpm_scheduler_pcc(mesh_device):
    """DPM scheduler latent PCC >= 0.99 after 10 steps with synthetic noise."""
    torch.manual_seed(42)

    from schedule.dpm_solver import DPMSolverMultistepScheduler as RefScheduler

    LATENT_SIZE = 64
    ref_sched = RefScheduler(
        num_train_timesteps=1000,
        beta_schedule="cosine",
        solver_order=2,
        prediction_type="v_prediction",
        algorithm_type="dpmsolver++",
        solver_type="midpoint",
    )
    ref_sched.set_timesteps(NUM_STEPS)

    tt_sched = TTDPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_schedule="cosine",
        solver_order=2,
        prediction_type="v_prediction",
    )
    tt_sched.set_timesteps(NUM_STEPS)

    latent = torch.randn(1, LATENT_SIZE, dtype=torch.float32)
    latent_tt = ttnn.as_tensor(
        latent.to(torch.bfloat16).view(1, 1, 1, LATENT_SIZE),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    latent_ref = latent.clone()

    all_eps = [torch.randn(1, LATENT_SIZE, dtype=torch.float32) for _ in range(NUM_STEPS)]

    for step_idx, t_val in enumerate(ref_sched.timesteps):
        eps = all_eps[step_idx]
        result = ref_sched.step(eps, t_val, latent_ref)
        latent_ref = result.prev_sample

    for step_idx in range(NUM_STEPS):
        eps = all_eps[step_idx]
        eps_tt = ttnn.as_tensor(
            eps.to(torch.bfloat16).view(1, 1, 1, LATENT_SIZE),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        latent_tt = tt_sched.step(eps_tt, latent_tt)

    tt_out = ttnn.to_torch(latent_tt).to(torch.float32).view(1, LATENT_SIZE)
    ref_out = latent_ref.view(1, LATENT_SIZE).to(torch.float32)

    passed, pcc_val = comp_pcc(ref_out, tt_out, pcc=0.99)
    assert passed, f"E2E DPM PCC {pcc_val:.6f} < 0.99"
