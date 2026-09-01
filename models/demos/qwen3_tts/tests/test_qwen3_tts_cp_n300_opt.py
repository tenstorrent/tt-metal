# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Guards for the N300-only CodePredictor fast path (``CodePredictor._n300_cp_opt``).

The fast path swaps three generic ops for sharded equivalents (hidden-size RMSNorm,
nlp_create_qkv_heads, nlp_concat_heads) and replaces the TP=2 all-reduce with a single
all-gather plus a local add. All four are meant to be numerically equivalent to the
generic path, so these tests A/B the two paths on identical synthetic weights and check
that the fast path also survives Metal trace capture (the demo runs the CP under trace).

    MESH_DEVICE is ignored here — these tests always open a (1,2) mesh, since the
    path under test only engages on a 2-chip wormhole mesh.

    pytest models/demos/qwen3_tts/tests/test_qwen3_tts_cp_n300_opt.py -s
"""

import os

import pytest
import torch

import ttnn
from models.demos.qwen3_tts.tests.test_qwen3_tts_profile_single_layer import (
    DEMO_CP_DECODE_SEQ,
    DEMO_CP_KV_MAX,
    DEMO_CP_PREFILL_SEQ,
    _cp_decode_mask,
    _cp_prefill_mask,
    _rope,
    _synthetic_cp_sd,
)

_TRACE_REGION = 23887872


@pytest.fixture(scope="module")
def device():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    d = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2), l1_small_size=32768, trace_region_size=_TRACE_REGION)
    d.enable_program_cache()
    yield d
    ttnn.close_mesh_device(d)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _build(device, opt):
    os.environ["QWEN3_TTS_CP_N300_OPT"] = "1" if opt else "0"
    from models.demos.qwen3_tts.tt.code_predictor import CodePredictor
    from models.demos.qwen3_tts.tt.model_config import Qwen3TTSCodePredictorConfig, Qwen3TTSTalkerConfig

    talker_h = Qwen3TTSTalkerConfig().hidden_size
    cfg = Qwen3TTSCodePredictorConfig(num_hidden_layers=1)
    cp = CodePredictor(
        device=device,
        config=cfg,
        talker_hidden_size=talker_h,
        state_dict=_synthetic_cp_sd(cfg, talker_hidden=talker_h, num_layers=1),
    )
    return cp, cfg


def _run(device, cp, cfg, mode):
    from models.demos.qwen3_tts.tt.kv_cache import create_kv_cache_list
    from models.demos.qwen3_tts.tt.mesh_utils import to_torch

    if mode == "prefill":
        seq, start_pos = DEMO_CP_PREFILL_SEQ, 0
    else:
        seq, start_pos = DEMO_CP_DECODE_SEQ, DEMO_CP_PREFILL_SEQ

    torch.manual_seed(1234)
    xt = torch.randn(1, 1, seq, cfg.hidden_size, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        xt, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    positions = None if mode == "prefill" else torch.tensor([start_pos])
    cos, sin, trans = _rope(device, seq, cfg.head_dim, cfg.rope_theta, positions=positions)
    kv = create_kv_cache_list(device, cfg, max_batch_size=1, max_seq_len=DEMO_CP_KV_MAX)
    if mode == "prefill":
        masks = dict(cp_prefill_mask=_cp_prefill_mask(device, cp.num_heads, seq, DEMO_CP_KV_MAX), decode_attn_mask=None)
    else:
        masks = dict(
            decode_attn_mask=_cp_decode_mask(device, cp.num_heads, DEMO_CP_KV_MAX, valid=start_pos + 1),
            cp_prefill_mask=None,
        )
    y, _ = cp._layer_forward(
        x, cp.layers_w[0], cos, sin, trans, kv_cache=kv[0], start_pos=start_pos, mode=mode, cur_pos_tensor=None, **masks
    )
    return to_torch(y, device=device)[..., :seq, :].float()


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_ab(device, mode):
    ref_cp, cfg = _build(device, opt=False)
    assert not ref_cp._n300_cp_opt
    ref = _run(device, ref_cp, cfg, mode)

    fast_cp, cfg = _build(device, opt=True)
    assert fast_cp._n300_cp_opt, "N300 fast path did not engage — is MESH_DEVICE=N300?"
    got = _run(device, fast_cp, cfg, mode)

    pcc = _pcc(ref, got)
    d = (ref - got).abs()
    # Synthetic weights are un-normalised, so activations here run to ~1e5 where one
    # bf16 ULP is ~512. Judge the error against the tensor's dynamic range in ULPs
    # rather than against each element (near-zero elements make per-element ratios
    # meaningless) and never against absolute difference.
    ulp_rel = (d.max() / ref.abs().max() * 2**8).item()
    print(
        f"\n[{mode}] PCC={pcc:.8f} max|diff|={d.max():.4g} "
        f"(|ref| mean {ref.abs().mean():.3g}, max err {ulp_rel:.2f} bf16 ULP)"
    )
    assert pcc > 0.9999, f"{mode}: PCC {pcc}"
    assert ulp_rel < 4.0, f"{mode}: {ulp_rel:.2f} bf16 ULP is more than rounding"


def test_trace_capture(device):
    """The N300 fast path must survive Metal trace capture/replay: the demo runs the CP
    inside a captured trace, so a non-trace-safe op (notably the CCL) would only show up
    here."""
    from models.demos.qwen3_tts.tt.kv_cache import create_kv_cache_list
    from models.demos.qwen3_tts.tt.mesh_utils import to_torch

    cp, cfg = _build(device, opt=True)
    assert cp._n300_cp_opt

    seq, start_pos = DEMO_CP_DECODE_SEQ, DEMO_CP_PREFILL_SEQ
    torch.manual_seed(99)
    xt = torch.randn(1, 1, seq, cfg.hidden_size, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        xt, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    cos, sin, trans = _rope(device, seq, cfg.head_dim, cfg.rope_theta, positions=torch.tensor([start_pos]))
    kv = create_kv_cache_list(device, cfg, max_batch_size=1, max_seq_len=DEMO_CP_KV_MAX)
    mask = _cp_decode_mask(device, cp.num_heads, DEMO_CP_KV_MAX, valid=start_pos + 1)

    def fwd():
        y, _ = cp._layer_forward(
            x,
            cp.layers_w[0],
            cos,
            sin,
            trans,
            kv_cache=kv[0],
            start_pos=start_pos,
            mode="decode",
            cur_pos_tensor=None,
            decode_attn_mask=mask,
            cp_prefill_mask=None,
        )
        return y

    eager = to_torch(fwd(), device=device)[..., :seq, :].float()

    warm = fwd()
    ttnn.synchronize_device(device)
    ttnn.deallocate(warm)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    try:
        out = fwd()
    finally:
        ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)

    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    traced = to_torch(out, device=device)[..., :seq, :].float()
    ttnn.release_trace(device, tid)

    pcc = _pcc(eager, traced)
    print(f"\n[trace] PCC(eager, traced) = {pcc:.8f} " f"max|diff|={(eager - traced).abs().max().item():.4g}")
    assert pcc > 0.9999, f"trace replay diverged: PCC {pcc}"
