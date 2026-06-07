# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Speech-encoder PCC at several mel sequence lengths (not just the 4096 max the test fixes).

Mirrors tests/pcc/test_speech_encoder.py but sweeps seq in {1934, 3000, 4096} so we can see how
PCC degrades with length on tp=1. 1934 = the demo's real S2TT mel length; 3000 = README's quoted
point (0.9957); 4096 = the test's worst-case max. Random bf16 input, bf16 reference (same as test).

Run (single chip):  SEAMLESS_FORCE_1x1 is implicit here — opens MeshShape(1,1) directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import ttnn
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.seamless_m4t_v2_large.reference.torch_speech_encoder import (
    forward_torch_speech_encoder,
    load_pretrained_speech_encoder,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.common import to_torch_replicated_first_shard
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    DEVICE_PARAMS_P150_E2E_2CQ_GENERATE,
    MESH_SHAPE_P150,
    from_torch_bfloat16_tile,
    mesh_default_device,
)
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_speech_encoder_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_speech_encoder import TTSeamlessM4Tv2SpeechEncoder

# Mid buckets (256..1024) are the range affected by the 1-C threshold change
# (_LONG_AUDIO_RES_DRAM_THRESHOLD 1024->128): they switch from the sharded-LN path (latent crash)
# to the DRAM-bypass path. Validate PCC there; 1934 is a high-seq control (always used DRAM).
SEQS = [256, 512, 768, 1024, 1934]
PCC_THRESHOLD = 0.99


def _run_one(mesh_device, speech_enc, cfg, seq: int) -> float:
    torch.manual_seed(0)
    batch = 1
    n_mels = cfg.feature_projection_input_dim
    input_features = torch.randn(batch, seq, n_mels, dtype=torch.bfloat16)
    attention_mask = torch.ones(batch, seq, dtype=torch.long)

    ref = forward_torch_speech_encoder(speech_enc, input_features, attention_mask).to(torch.bfloat16)

    params = create_speech_encoder_parameters(speech_enc, device=mesh_device)
    tt_model = TTSeamlessM4Tv2SpeechEncoder(
        mesh_device,
        params,
        hidden_size=cfg.hidden_size,
        feature_projection_input_dim=cfg.feature_projection_input_dim,
        speech_encoder_attention_heads=cfg.speech_encoder_attention_heads,
        speech_encoder_intermediate_size=cfg.speech_encoder_intermediate_size,
        speech_encoder_layers=cfg.speech_encoder_layers,
        layer_norm_eps=cfg.layer_norm_eps,
        speech_encoder_chunk_size=cfg.speech_encoder_chunk_size,
        speech_encoder_left_chunk_num=cfg.speech_encoder_left_chunk_num,
        matmul_token_rows=batch * seq,
    )
    tt_x = from_torch_bfloat16_tile(mesh_device, input_features, memory_config=ttnn.L1_MEMORY_CONFIG)
    m1 = from_torch_bfloat16_tile(mesh_device, attention_mask, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_model.pre_warm(batch, seq)
    tt_out = tt_model(tt_x, conv_attention_mask_1d=m1)
    ttnn.synchronize_device(mesh_device)
    tt_cpu = to_torch_replicated_first_shard(tt_out).to(torch.bfloat16)
    ok, msg = check_with_pcc(ref, tt_cpu, pcc=PCC_THRESHOLD)
    logger.info(f"speech encoder PCC @ mel_seq={seq}: {msg} ({'PASS' if ok else 'FAIL'} vs {PCC_THRESHOLD})")
    ttnn.deallocate(tt_out)
    import re

    m = re.search(r"[-+]?\d*\.\d+|\d+", str(msg))
    return float(m.group()) if m else float("nan")


def main() -> None:
    weights_dir = ensure_seamless_m4t_v2_large_weights()
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*MESH_SHAPE_P150), **dict(DEVICE_PARAMS_P150_E2E_2CQ_GENERATE)
    )
    try:
        with mesh_default_device(device):
            speech_enc, cfg = load_pretrained_speech_encoder(weights_dir, dtype=torch.bfloat16)
            results = {}
            for seq in SEQS:
                results[seq] = _run_one(device, speech_enc, cfg, seq)
        print("\n==== speech-encoder PCC vs mel_seq (tp=1, random bf16 input) ====")
        for seq, pcc in results.items():
            print(f"  mel_seq={seq:5d} : PCC={pcc:.6f}  {'PASS' if pcc >= PCC_THRESHOLD else 'FAIL'}")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
