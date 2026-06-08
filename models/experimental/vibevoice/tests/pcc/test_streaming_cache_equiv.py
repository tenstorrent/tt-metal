# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device streaming-cache equivalence.

Validates that the TTNN tokenizers' streaming path (per-chunk, use_cache=True)
produces the same output as the non-streaming full-sequence path — i.e. the
per-layer causal caches correctly supply each chunk's left context, matching the
reference SConv1d/SConvTranspose1d `_forward_streaming` behaviour.

This is what lets the generator decode/encode on-device per diffusion step
instead of re-running the whole prefix (or the CPU reference).
"""

import sys
from pathlib import Path

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.vibevoice.common.config import MODEL_PATH
from models.experimental.vibevoice.tt.load_weights import (
    load_vibevoice_state_dict,
    split_submodule_weights,
    fold_weight_norm,
)
from models.experimental.vibevoice.tt.ttnn_acoustic_tokenizer import (
    preprocess_acoustic_tokenizer_weights,
    TTAcousticTokenizer,
)
from models.experimental.vibevoice.tt.vibevoice_config import load_vibevoice_model_config

_VIBEVOICE_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_VIBEVOICE_ROOT / "reference", _VIBEVOICE_ROOT.parent.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

pytestmark = pytest.mark.skipif(not Path(MODEL_PATH).is_dir(), reason="VIBEVOICE_MODEL_PATH weights missing")

N_FRAMES = 4


@pytest.fixture(scope="module")
def ac_state():
    sd = load_vibevoice_state_dict(MODEL_PATH)
    return fold_weight_norm(split_submodule_weights(sd)["acoustic_tokenizer"])


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_decode_streaming_equals_full(mesh_device, ac_state):
    cfg = load_vibevoice_model_config(MODEL_PATH).acoustic_tokenizer
    # Separate instances: ttnn.conv2d caches prepared weights per input width, so a
    # full-width pass and a streaming pass must not share a tokenizer object.
    tok_full = TTAcousticTokenizer(preprocess_acoustic_tokenizer_weights(ac_state, mesh_device, cfg), mesh_device)
    tok_stream = TTAcousticTokenizer(preprocess_acoustic_tokenizer_weights(ac_state, mesh_device, cfg), mesh_device)

    vae_dim = cfg.vae_dim
    torch.manual_seed(0)

    # Real (structured) latents from encoding audio — random latents are pathological
    # for PCC because the full vs streaming paths do different bf16 arithmetic.
    ratio = 1
    for r in cfg.encoder_ratios:
        ratio *= r
    audio = torch.randn(1, 1, 1, N_FRAMES * ratio, dtype=torch.bfloat16)
    enc_in = ttnn.as_tensor(
        audio,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    latents = (
        ttnn.to_torch(tok_full.encode(enc_in)).to(torch.bfloat16).reshape(N_FRAMES, vae_dim).t().unsqueeze(0)
    )  # [1, D, N]

    def to_tt(lat_bdt):  # [1, D, t] -> [1, 1, t, D]
        return ttnn.as_tensor(
            lat_bdt.permute(0, 2, 1).unsqueeze(1),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Non-streaming full decode.
    full = ttnn.to_torch(tok_full.decode(to_tt(latents))).to(torch.float32).reshape(-1)

    # Streaming, one latent frame at a time.
    tok_stream.reset_decode_cache()
    chunks = []
    for n in range(N_FRAMES):
        out = tok_stream.decode(to_tt(latents[:, :, n : n + 1]), use_cache=True)
        chunks.append(ttnn.to_torch(out).to(torch.float32).reshape(-1))
    stream = torch.cat(chunks, dim=0)

    m = min(full.numel(), stream.numel())
    passed, pcc_val = comp_pcc(full[:m], stream[:m], pcc=0.99)
    print(f"[decode streaming] full vs streaming PCC = {pcc_val:.6f} (full={full.numel()}, stream={stream.numel()})")
    assert full.numel() == stream.numel(), f"length mismatch: full={full.numel()} stream={stream.numel()}"
    assert passed, f"decode streaming != full (PCC {pcc_val:.6f} < 0.99)"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_encode_streaming_equals_full(mesh_device, ac_state):
    cfg = load_vibevoice_model_config(MODEL_PATH).acoustic_tokenizer
    # Separate instances (see decode test): per-width prepared-weight caching.
    tok_full = TTAcousticTokenizer(preprocess_acoustic_tokenizer_weights(ac_state, mesh_device, cfg), mesh_device)
    tok_stream = TTAcousticTokenizer(preprocess_acoustic_tokenizer_weights(ac_state, mesh_device, cfg), mesh_device)

    ratio = 1
    for r in cfg.encoder_ratios:
        ratio *= r  # 3200
    chunk = ratio
    torch.manual_seed(1)
    audio = torch.randn(1, 1, 1, N_FRAMES * chunk, dtype=torch.bfloat16)

    def to_tt(a):
        return ttnn.as_tensor(
            a,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Non-streaming full encode -> [1, 1, N, vae_dim]
    full = ttnn.to_torch(tok_full.encode(to_tt(audio))).to(torch.float32).reshape(N_FRAMES, -1)

    # Streaming, one audio chunk at a time -> 1 latent frame each.
    tok_stream._encoder_tt.reset_cache()
    frames = []
    for n in range(N_FRAMES):
        a_n = audio[:, :, :, n * chunk : (n + 1) * chunk]
        is_final = n == N_FRAMES - 1
        out = tok_stream.encode(to_tt(a_n), use_cache=True, is_final_chunk=is_final)
        frames.append(ttnn.to_torch(out).to(torch.float32).reshape(-1, full.shape[-1])[-1])  # last frame
    stream = torch.stack(frames, dim=0)

    passed, pcc_val = comp_pcc(full, stream, pcc=0.99)
    print(f"[encode streaming] full vs streaming PCC = {pcc_val:.6f}")
    assert passed, f"encode streaming != full (PCC {pcc_val:.6f} < 0.99)"
