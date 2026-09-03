# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fused-SDPA vs manual-fp32-chain parity on the CP's REAL masked shapes.

``test_qwen3_tts_pcc.py::test_code_predictor_step_pcc`` runs with ``kv_caches=None``
and no mask, so it takes SDPA's ``is_causal=True`` branch and never exercises the
explicit ``decode_attn_mask`` / ``cp_prefill_mask`` the demo actually feeds. This test
does, with real checkpoint weights, by flipping ``cp._fused_sdpa`` between two calls on
ONE CodePredictor instance (the flag is read per call inside ``_layer_forward``), so
weights, KV caches, RoPE tables and masks are bit-identical across the two arms.

Both CP modes are covered:
  prefill  seq=2, cp_prefill_mask   [1, nh, 2, 32]
  decode   seq=1, decode_attn_mask  [1, nh, 1, 32] at start_pos 2..5

Reports PCC and max abs difference on the returned logits. A large delta here means the
fused path is wrong on the masked shapes; a tiny one means the generation-length shift
seen in the demo is downstream chaos (EOS timing), not an attention bug.

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=wormhole_b0 MESH_DEVICE=N300
    pytest -s -q models/demos/qwen3_tts/tests/test_qwen3_tts_cp_sdpa_parity.py
"""

import os

import pytest
import torch

import ttnn

MAX_CP_SEQ_LEN = 32


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if torch.allclose(a, b):
        return 1.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.fixture(scope="module")
def device():
    mesh_shape = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(os.environ.get("MESH_DEVICE"))
    if mesh_shape is None:
        d = ttnn.open_device(device_id=0, l1_small_size=32768)
        d.enable_program_cache()
        yield d
        ttnn.close_device(d)
        return
    if mesh_shape != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    d = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape), l1_small_size=32768)
    d.enable_program_cache()
    yield d
    ttnn.close_mesh_device(d)
    if mesh_shape != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def state_dict():
    from models.demos.qwen3_tts.tt.server import load_weights

    return load_weights()


def test_cp_fused_sdpa_parity(device, state_dict):
    from models.demos.qwen3_tts.tt.code_predictor import CodePredictor
    from models.demos.qwen3_tts.tt.kv_cache import create_kv_cache_list
    from models.demos.qwen3_tts.tt.mesh_utils import get_tp_size, is_mesh_device
    from models.demos.qwen3_tts.tt.mesh_utils import to_torch as _mesh_to_torch
    from models.demos.qwen3_tts.tt.model_config import Qwen3TTSCodePredictorConfig, Qwen3TTSTalkerConfig
    from models.demos.qwen3_tts.tt.rope import get_rope_tensors, get_transformation_mat

    talker_cfg = Qwen3TTSTalkerConfig()
    cp_cfg = Qwen3TTSCodePredictorConfig()
    tp = get_tp_size(device) if is_mesh_device(device) else 1
    nh = cp_cfg.num_attention_heads // tp

    cp = CodePredictor(device=device, config=cp_cfg, talker_hidden_size=talker_cfg.hidden_size, state_dict=state_dict)
    if cp._n150:
        pytest.skip("N150 always uses fused SDPA; there is no manual arm to compare against")

    trans = get_transformation_mat(cp_cfg.head_dim, device)
    torch.manual_seed(11)

    def _run(x_t, cos, sin, start_pos, mode, mask_t, fused):
        """One forward with the flag forced, on a freshly zeroed KV cache."""
        kv = create_kv_cache_list(device, cp_cfg, max_batch_size=1, max_seq_len=MAX_CP_SEQ_LEN)
        x = ttnn.from_torch(
            x_t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        mask = ttnn.from_torch(
            mask_t, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        cp._fused_sdpa = fused
        logits, _ = cp.forward_single_step(
            x,
            cos,
            sin,
            trans,
            generation_step=1,
            kv_caches=kv,
            start_pos=start_pos,
            mode=mode,
            decode_attn_mask=mask if mode == "decode" else None,
            cp_prefill_mask=mask if mode == "prefill" else None,
        )
        out = _mesh_to_torch(logits).float()
        ttnn.deallocate(logits)
        ttnn.deallocate(x)
        ttnn.deallocate(mask)
        for k, v in kv:
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        return out

    print()
    worst = 1.0

    # --- prefill: seq=2, causal mask over the 32-deep cache -------------------
    x_pf = torch.randn(1, 1, 2, talker_cfg.hidden_size, dtype=torch.bfloat16)
    cos_pf, sin_pf = get_rope_tensors(device, cp_cfg.head_dim, 2, torch.arange(2), cp_cfg.rope_theta)
    m_pf = torch.full((1, nh, 2, MAX_CP_SEQ_LEN), float("-inf"))
    m_pf[0, :, 0, 0] = 0.0
    m_pf[0, :, 1, 0:2] = 0.0
    a = _run(x_pf, cos_pf, sin_pf, 0, "prefill", m_pf, True)
    b = _run(x_pf, cos_pf, sin_pf, 0, "prefill", m_pf, False)
    p, d = _pcc(a, b), float((a - b).abs().max())
    worst = min(worst, p)
    print(f"[sdpa_parity] prefill seq=2          PCC={p:.8f}  maxabs={d:.6f}")

    # --- decode: seq=1 at the positions the demo actually walks --------------
    for start_pos in (2, 3, 5):
        x_dc = torch.randn(1, 1, 1, talker_cfg.hidden_size, dtype=torch.bfloat16)
        cos_dc, sin_dc = get_rope_tensors(device, cp_cfg.head_dim, 1, torch.tensor([start_pos]), cp_cfg.rope_theta)
        m_dc = torch.full((1, nh, 1, MAX_CP_SEQ_LEN), float("-inf"))
        m_dc[0, :, 0, : start_pos + 1] = 0.0
        a = _run(x_dc, cos_dc, sin_dc, start_pos, "decode", m_dc, True)
        b = _run(x_dc, cos_dc, sin_dc, start_pos, "decode", m_dc, False)
        p, d = _pcc(a, b), float((a - b).abs().max())
        worst = min(worst, p)
        print(f"[sdpa_parity] decode start_pos={start_pos}    PCC={p:.8f}  maxabs={d:.6f}")

        # argmax agreement is what actually drives the token stream
        ta, tb = int(a.flatten().argmax()), int(b.flatten().argmax())
        print(f"                                     argmax fused={ta} manual={tb} {'OK' if ta == tb else 'DIFFER'}")

    assert worst > 0.99, f"fused SDPA diverges from the manual fp32 chain on masked shapes (worst PCC {worst})"
