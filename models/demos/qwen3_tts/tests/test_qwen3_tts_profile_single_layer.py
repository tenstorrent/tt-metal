# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Signposted single-block device profiling for every on-device Qwen3-TTS module.

Sequence lengths match ``demo_full_ttnn_tts`` / ``server.py``:

  Talker prefill  32 / 64 / 128   TRACE buckets (different matmul M and kernel path)
  Talker decode   seq=1           QKV/MLP always M=1; attention spans the whole
                                 KV cache (kv_max=352) on the deployed path
  CP prefill      seq=2           always (talker hidden + code0); TILE-padded to 32
  CP decode       seq=1           KV max=32 always
  Speaker         T=384           ~4s jim_reference mel; no TRACE buckets

Talker 32 vs 64/128 is a different QKV path (DRAM-sharded at seq<=32). 64 vs 128
share that family but M and RMSNorm/concat shard height differ — capture each.

CP and Speaker do not have extra seq variants in the demo: CP is always 2 then 1,
Speaker conv length follows the reference wav (one T).

On-device modules (Mimi encode/decode stays on CPU and is not in these tests):

  Talker DecoderLayer     -k talker_layer_prefill_32 | _64 | _128
                          -k talker_layer_decode          eager fallback (cache sliced to 1)
                          -k talker_layer_decode_traced   DEPLOYED path (full-cache SDPA)
  CodePredictor layer     -k cp_layer_prefill | cp_layer_decode
  Speaker TDNN 128→512    -k speaker_tdnn          traced device-conv (k=5 im2col)
  Speaker SERes2Net       -k speaker_block         traced device-conv (one block)
  Speaker full ECAPA      -k speaker_encoder       DEPLOYED ``capture_forward_trace``

Speaker windows replay a Metal trace with ``_se_device_conv`` on (the demo's
``QWEN3_TTS_SE_TRACE=1`` path). Eager device-conv is ~7x slower than host-fuse
and is not what these tests measure.

Run **one** ``-k`` per Tracy capture. Do not use ``-k talker_layer_prefill`` —
that substring matches all three buckets.

Example (N150, Talker prefill 64):

    MESH_DEVICE=N150 TT_METAL_DEVICE_PROFILER=1 \\
      python -m tracy -p -v -r -m pytest -s -v \\
      models/demos/qwen3_tts/tests/test_qwen3_tts_profile_single_layer.py -k talker_layer_prefill_64

    CSV=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
    tt-perf-report --start-signpost start --end-signpost stop $CSV
"""

import os

import pytest
import torch

import ttnn

try:
    from tracy import signpost
except ModuleNotFoundError:

    def signpost(*_a, **_k):
        pass


# Demo TRACE_PREFILL_BUCKETS = (32, 64, 128). Japanese sample pads to 64.
DEMO_TALKER_PREFILL_BUCKETS = (32, 64, 128)
DEMO_TALKER_DECODE_SEQ = 1
# Decode position for the traced-decode window (any pos < kv_max gives the same
# op shapes; 128 matches the qwen3_tts_trace_perf.txt capture).
DEMO_TALKER_DECODE_POS = 128
DEMO_MAX_NEW_TOKENS = 256
_TILE = 32


def _talker_kv_max(padded_seq_len: int) -> int:
    """Matches server.py: tile(padded_prefill + max_new_tokens + 16)."""
    return (((padded_seq_len + DEMO_MAX_NEW_TOKENS + 16) + _TILE - 1) // _TILE) * _TILE


DEMO_CP_PREFILL_SEQ = 2
DEMO_CP_DECODE_SEQ = 1
DEMO_CP_KV_MAX = 32

# ~4.01s @ 24 kHz, hop 256, n_fft 1024 → ~376 mel frames; pad to one tile.
DEMO_SPEAKER_T = 384


# Speaker-encoder Metal traces (im2col + matmul for k>1 convs) need a real trace
# region. The demo uses 200 MB; without it capture_forward_trace fails to allocate.
_TRACE_REGION = 200_000_000


def _open_device():
    mesh_shape = {"N150": (1, 1), "N300": (1, 2)}.get(os.environ.get("MESH_DEVICE"))
    kwargs = dict(l1_small_size=32768, trace_region_size=_TRACE_REGION)
    if mesh_shape is None:
        device = ttnn.open_device(device_id=0, **kwargs)
        device.enable_program_cache()
        return device, None
    if mesh_shape != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape), **kwargs)
    device.enable_program_cache()
    return device, mesh_shape


def _close_device(device, mesh_shape):
    if mesh_shape is None:
        ttnn.close_device(device)
        return
    ttnn.close_mesh_device(device)
    if mesh_shape != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def device():
    d, mesh_shape = _open_device()
    yield d
    _close_device(d, mesh_shape)


def _synthetic_decoder_sd(cfg, prefix="talker.model.layers.0"):
    torch.manual_seed(0)
    h, i = cfg.hidden_size, cfg.intermediate_size
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    p = prefix
    return {
        f"{p}.input_layernorm.weight": torch.ones(h, dtype=torch.bfloat16),
        f"{p}.post_attention_layernorm.weight": torch.ones(h, dtype=torch.bfloat16),
        f"{p}.self_attn.q_proj.weight": torch.randn(nh * hd, h, dtype=torch.bfloat16),
        f"{p}.self_attn.k_proj.weight": torch.randn(nkv * hd, h, dtype=torch.bfloat16),
        f"{p}.self_attn.v_proj.weight": torch.randn(nkv * hd, h, dtype=torch.bfloat16),
        f"{p}.self_attn.o_proj.weight": torch.randn(h, nh * hd, dtype=torch.bfloat16),
        f"{p}.self_attn.q_norm.weight": torch.ones(hd, dtype=torch.bfloat16),
        f"{p}.self_attn.k_norm.weight": torch.ones(hd, dtype=torch.bfloat16),
        f"{p}.mlp.gate_proj.weight": torch.randn(i, h, dtype=torch.bfloat16),
        f"{p}.mlp.up_proj.weight": torch.randn(i, h, dtype=torch.bfloat16),
        f"{p}.mlp.down_proj.weight": torch.randn(h, i, dtype=torch.bfloat16),
    }


def _synthetic_cp_sd(cfg, talker_hidden=2048, num_layers=1):
    """Minimal CodePredictor state dict (one layer + required heads / projection)."""
    torch.manual_seed(0)
    h, i = cfg.hidden_size, cfg.intermediate_size
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    sd = {
        "talker.code_predictor.small_to_mtp_projection.weight": torch.randn(h, talker_hidden, dtype=torch.bfloat16),
        "talker.code_predictor.small_to_mtp_projection.bias": torch.zeros(h, dtype=torch.bfloat16),
        "talker.code_predictor.model.norm.weight": torch.ones(h, dtype=torch.bfloat16),
    }
    for li in range(num_layers):
        p = f"talker.code_predictor.model.layers.{li}"
        sd.update(
            {
                f"{p}.input_layernorm.weight": torch.ones(h, dtype=torch.bfloat16),
                f"{p}.post_attention_layernorm.weight": torch.ones(h, dtype=torch.bfloat16),
                f"{p}.self_attn.q_proj.weight": torch.randn(nh * hd, h, dtype=torch.bfloat16),
                f"{p}.self_attn.k_proj.weight": torch.randn(nkv * hd, h, dtype=torch.bfloat16),
                f"{p}.self_attn.v_proj.weight": torch.randn(nkv * hd, h, dtype=torch.bfloat16),
                f"{p}.self_attn.o_proj.weight": torch.randn(h, nh * hd, dtype=torch.bfloat16),
                f"{p}.self_attn.q_norm.weight": torch.ones(hd, dtype=torch.bfloat16),
                f"{p}.self_attn.k_norm.weight": torch.ones(hd, dtype=torch.bfloat16),
                f"{p}.mlp.gate_proj.weight": torch.randn(i, h, dtype=torch.bfloat16),
                f"{p}.mlp.up_proj.weight": torch.randn(i, h, dtype=torch.bfloat16),
                f"{p}.mlp.down_proj.weight": torch.randn(h, i, dtype=torch.bfloat16),
            }
        )
    for g in range(cfg.num_code_groups - 1):
        sd[f"talker.code_predictor.lm_head.{g}.weight"] = torch.randn(cfg.vocab_size, h, dtype=torch.bfloat16)
    return sd


def _talker_matmul_dtype():
    """The matmul weight dtype the deployed Talker uses.

    Must mirror ``talker.py``'s ``_matmul_dtype`` exactly. This fixture used to
    hardcode bfloat16, so after ``QWEN3_TTS_BF8_WEIGHTS`` became default ON the
    single-layer windows profiled a dtype the demo no longer runs — gate/up read
    116 us here against 92 us in the model.
    """
    return ttnn.bfloat8_b if os.environ.get("QWEN3_TTS_BF8_WEIGHTS", "1") != "0" else ttnn.bfloat16


def _make_talker_layer(device, cfg):
    from models.demos.qwen3_tts.tt.decoder_layer import DecoderLayer

    return DecoderLayer(
        device=device,
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        intermediate_size=cfg.intermediate_size,
        state_dict=_synthetic_decoder_sd(cfg, prefix="talker.model.layers.0"),
        layer_idx=0,
        layer_prefix="talker.model",
        rms_norm_eps=cfg.rms_norm_eps,
        weight_dtype=_talker_matmul_dtype(),
    )


@pytest.fixture(scope="module")
def talker_layer(device):
    from models.demos.qwen3_tts.tt.model_config import Qwen3TTSTalkerConfig

    return _make_talker_layer(device, Qwen3TTSTalkerConfig())


@pytest.fixture(scope="module")
def code_predictor(device):
    """Production CodePredictor with a single layer (random weights)."""
    from models.demos.qwen3_tts.tt.code_predictor import CodePredictor
    from models.demos.qwen3_tts.tt.model_config import Qwen3TTSCodePredictorConfig, Qwen3TTSTalkerConfig

    talker_h = Qwen3TTSTalkerConfig().hidden_size
    cfg = Qwen3TTSCodePredictorConfig(num_hidden_layers=1)
    return CodePredictor(
        device=device,
        config=cfg,
        talker_hidden_size=talker_h,
        state_dict=_synthetic_cp_sd(cfg, talker_hidden=talker_h, num_layers=1),
    )


def _hidden(device, seq_len, hidden):
    x = torch.randn(1, 1, seq_len, hidden, dtype=torch.bfloat16)
    return ttnn.from_torch(
        x,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _rope(device, seq_len, head_dim, rope_theta, positions=None):
    from models.demos.qwen3_tts.tt.rope import get_rope_tensors, get_transformation_mat

    pos = torch.arange(seq_len) if positions is None else positions
    cos, sin = get_rope_tensors(device, head_dim, seq_len, pos, rope_theta)
    trans = get_transformation_mat(head_dim, device)
    return cos, sin, trans


def _profile_forward(device, fn):
    """Compile once, then measure one warm forward between start/stop."""
    fn()
    ttnn.synchronize_device(device)
    signpost("start")
    fn()
    ttnn.synchronize_device(device)
    signpost("stop")


def _enable_traced_device_conv(enc):
    """Same flags ``capture_forward_trace`` forces: every conv on device, no nested SE traces."""
    enc._se_host_fuse = True
    enc._se_device_asp = True
    enc._se_device_conv = True
    enc._se_traces_active = False


def _profile_traced_forward(device, fn):
    """Capture a Metal trace of ``fn``, then replay it once between start/stop.

    Eager device-conv is ~7x slower than the host path; the win only shows up under
    ``execute_trace``. Matches the demo's ``QWEN3_TTS_SE_TRACE=1`` / ``SE_DEVICE_CONV`` path.
    """
    fn()
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    try:
        fn()
    finally:
        ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    signpost("start")
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    signpost("stop")


def _run_talker_prefill(device, talker_layer, seq_len: int):
    from models.demos.qwen3_tts.tt.kv_cache import create_kv_cache_list
    from models.demos.qwen3_tts.tt.model_config import Qwen3TTSTalkerConfig

    assert seq_len in DEMO_TALKER_PREFILL_BUCKETS, seq_len
    cfg = Qwen3TTSTalkerConfig()
    cfg.num_hidden_layers = 1
    kv_max = _talker_kv_max(seq_len)
    x = _hidden(device, seq_len, cfg.hidden_size)
    cos, sin, trans = _rope(device, seq_len, cfg.head_dim, cfg.rope_theta)
    kv_caches = create_kv_cache_list(device, cfg, max_batch_size=1, max_seq_len=kv_max)

    def _fwd():
        y, _ = talker_layer(x, cos, sin, trans, kv_cache=kv_caches[0], start_pos=0, mode="prefill")
        return y

    _profile_forward(device, _fwd)
    print(f"[talker_layer_prefill_{seq_len}] seq_len={seq_len} hidden={cfg.hidden_size} kv_max={kv_max}")


def test_talker_layer_prefill_32(device, talker_layer):
    """Talker DecoderLayer at TRACE bucket 32 (DRAM-sharded QKV, M=1 tile)."""
    _run_talker_prefill(device, talker_layer, 32)


def test_talker_layer_prefill_64(device, talker_layer):
    """Talker DecoderLayer at TRACE bucket 64 (sample_ja; interleaved QKV, M=2 tiles)."""
    _run_talker_prefill(device, talker_layer, 64)


def test_talker_layer_prefill_128(device, talker_layer):
    """Talker DecoderLayer at TRACE bucket 128 (interleaved QKV, M=4 tiles)."""
    _run_talker_prefill(device, talker_layer, 128)


def test_talker_layer_decode(device, talker_layer):
    """One Talker DecoderLayer at demo decode seq=1. QKV/MLP M is always 1."""
    from models.demos.qwen3_tts.tt.kv_cache import create_kv_cache_list
    from models.demos.qwen3_tts.tt.model_config import Qwen3TTSTalkerConfig

    cfg = Qwen3TTSTalkerConfig()
    cfg.num_hidden_layers = 1
    # KV sized for the Japanese-demo bucket (64). Decode QKV/MLP shapes do not
    # depend on bucket; this test slices cache to start_pos+1 so SDPA is 1×1.
    kv_max = _talker_kv_max(64)
    x = _hidden(device, DEMO_TALKER_DECODE_SEQ, cfg.hidden_size)
    cos, sin, trans = _rope(device, DEMO_TALKER_DECODE_SEQ, cfg.head_dim, cfg.rope_theta)
    kv_caches = create_kv_cache_list(device, cfg, max_batch_size=1, max_seq_len=kv_max)

    def _fwd():
        y, _ = talker_layer(x, cos, sin, trans, kv_cache=kv_caches[0], start_pos=0, mode="decode")
        return y

    _profile_forward(device, _fwd)
    print(f"[talker_layer_decode] seq_len={DEMO_TALKER_DECODE_SEQ} hidden={cfg.hidden_size} kv_max={kv_max}")


def test_talker_layer_decode_traced(device, talker_layer):
    """One Talker DecoderLayer on the **deployed** decode path (server.py trace form).

    ``test_talker_layer_decode`` above drives the eager fallback: ``cur_pos_tensor``
    and ``decode_attn_mask`` are None, so the layer writes the cache with
    ``update_cache`` and slices it to ``start_pos+1`` — SDPA then sees a 1-position
    K/V.  The demo never runs that graph.  Under Metal trace, ``server.py`` passes a
    device ``cur_pos_tensor`` + a full ``[1, heads, 1, kv_max]`` mask, so the layer
    uses ``paged_fused_update_cache`` and attends over the WHOLE cache (kv_max=352
    for the Japanese bucket).  That makes every attention op ~kv_max/32 times larger
    and is where the GQA-expansion / typecast overhead actually lands.

    Hoists the cos/sin reshard and the SDPA mask conversion exactly like
    ``Talker._forward_layers`` does, so the window is one deployed layer.
    """
    from models.demos.qwen3_tts.tt.attention import prepare_fused_sdpa_mask, talker_fused_sdpa_enabled
    from models.demos.qwen3_tts.tt.kv_cache import create_kv_cache_list
    from models.demos.qwen3_tts.tt.mesh_utils import get_tp_size
    from models.demos.qwen3_tts.tt.model_config import Qwen3TTSTalkerConfig
    from models.demos.qwen3_tts.tt.rope import shard_decode_rope_tables

    cfg = Qwen3TTSTalkerConfig()
    cfg.num_hidden_layers = 1
    kv_max = _talker_kv_max(64)
    cur_pos = DEMO_TALKER_DECODE_POS
    tp = get_tp_size(device) if device.__class__.__name__ == "MeshDevice" else 1
    local_heads = cfg.num_attention_heads // tp

    x = _hidden(device, DEMO_TALKER_DECODE_SEQ, cfg.hidden_size)
    cos, sin, trans = _rope(
        device,
        DEMO_TALKER_DECODE_SEQ,
        cfg.head_dim,
        cfg.rope_theta,
        positions=torch.tensor([cur_pos]),
    )
    kv_caches = create_kv_cache_list(device, cfg, max_batch_size=1, max_seq_len=kv_max)

    # server.py: int32 [1] position in DRAM (paged_fused_update_cache requires DRAM).
    cur_pos_tt = ttnn.from_torch(
        torch.tensor([cur_pos], dtype=torch.int32),
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    mask_cpu = torch.full((1, local_heads, 1, kv_max), float("-inf"), dtype=torch.float32)
    mask_cpu[0, :, 0, : cur_pos + 1] = 0.0
    mask_tt = ttnn.from_torch(
        mask_cpu,
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Per-step hoists that Talker._forward_layers does once for all 28 layers.
    cos_s, sin_s, _own_rope = shard_decode_rope_tables(cos, sin, cfg.head_dim)
    if talker_fused_sdpa_enabled(device):
        mask_s, _own_mask = prepare_fused_sdpa_mask(mask_tt)
    else:
        mask_s = mask_tt  # manual fp32 chain adds the fp32 mask directly

    def _fwd():
        y, _ = talker_layer(
            x,
            cos_s,
            sin_s,
            trans,
            kv_cache=kv_caches[0],
            mode="decode",
            cur_pos_tensor=cur_pos_tt,
            decode_attn_mask=mask_s,
        )
        return y

    _profile_forward(device, _fwd)
    print(
        f"[talker_layer_decode_traced] seq_len={DEMO_TALKER_DECODE_SEQ} hidden={cfg.hidden_size} "
        f"kv_max={kv_max} cur_pos={cur_pos} local_heads={local_heads} tp={tp}"
    )


def _cp_prefill_mask(device, num_heads, seq_len, max_seq, dtype=ttnn.float32):
    mh = torch.full((1, num_heads, seq_len, max_seq), float("-inf"), dtype=torch.float32)
    for i in range(seq_len):
        mh[0, :, i, : i + 1] = 0.0
    return ttnn.from_torch(
        mh,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _cp_decode_mask(device, num_heads, max_seq, valid, dtype=ttnn.float32):
    mh = torch.full((1, num_heads, 1, max_seq), float("-inf"), dtype=torch.float32)
    mh[0, :, 0, :valid] = 0.0
    return ttnn.from_torch(
        mh,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_cp_layer_prefill(device, code_predictor):
    """One production CodePredictor layer at demo CP prefill seq=2."""
    from models.demos.qwen3_tts.tt.kv_cache import create_kv_cache_list
    from models.demos.qwen3_tts.tt.model_config import Qwen3TTSCodePredictorConfig

    cfg = Qwen3TTSCodePredictorConfig(num_hidden_layers=1)
    seq_len = DEMO_CP_PREFILL_SEQ
    x = _hidden(device, seq_len, cfg.hidden_size)
    cos, sin, trans = _rope(device, seq_len, cfg.head_dim, cfg.rope_theta)
    kv_caches = create_kv_cache_list(device, cfg, max_batch_size=1, max_seq_len=DEMO_CP_KV_MAX)
    # Per-chip head count (TP=1: full heads; TP=2: heads/2). Must match scores.
    # N150 fused SDPA wants bf16 DRAM; create it that way so the layer does not typecast.
    _mask_dt = ttnn.bfloat16 if code_predictor._n150 else ttnn.float32
    mask = _cp_prefill_mask(device, code_predictor.num_heads, seq_len, DEMO_CP_KV_MAX, dtype=_mask_dt)
    lw = code_predictor.layers_w[0]

    def _fwd():
        y, _ = code_predictor._layer_forward(
            x,
            lw,
            cos,
            sin,
            trans,
            kv_cache=kv_caches[0],
            start_pos=0,
            mode="prefill",
            cur_pos_tensor=None,
            decode_attn_mask=None,
            cp_prefill_mask=mask,
        )
        return y

    _profile_forward(device, _fwd)
    print(f"[cp_layer_prefill] seq_len={seq_len} hidden={cfg.hidden_size} kv_max={DEMO_CP_KV_MAX}")


def test_cp_layer_decode(device, code_predictor):
    """One production CodePredictor layer at demo CP decode seq=1."""
    from models.demos.qwen3_tts.tt.kv_cache import create_kv_cache_list
    from models.demos.qwen3_tts.tt.model_config import Qwen3TTSCodePredictorConfig
    from models.demos.qwen3_tts.tt.rope import shard_decode_rope_tables

    cfg = Qwen3TTSCodePredictorConfig(num_hidden_layers=1)
    x = _hidden(device, DEMO_CP_DECODE_SEQ, cfg.hidden_size)
    # First CP decode in the demo is at position 2 (after seq=2 prefill).
    start_pos = DEMO_CP_PREFILL_SEQ
    cos, sin, trans = _rope(
        device, DEMO_CP_DECODE_SEQ, cfg.head_dim, cfg.rope_theta, positions=torch.tensor([start_pos])
    )
    # forward_single_step reshards decode cos/sin once per step; mirror that here
    # so apply_rope_qk skips the per-layer I2S inside the signpost window.
    cos, sin, _own_rope = shard_decode_rope_tables(cos, sin, cfg.head_dim)
    kv_caches = create_kv_cache_list(device, cfg, max_batch_size=1, max_seq_len=DEMO_CP_KV_MAX)
    _mask_dt = ttnn.bfloat16 if code_predictor._n150 else ttnn.float32
    mask = _cp_decode_mask(device, code_predictor.num_heads, DEMO_CP_KV_MAX, valid=start_pos + 1, dtype=_mask_dt)
    lw = code_predictor.layers_w[0]

    def _fwd():
        y, _ = code_predictor._layer_forward(
            x,
            lw,
            cos,
            sin,
            trans,
            kv_cache=kv_caches[0],
            start_pos=start_pos,
            mode="decode",
            cur_pos_tensor=None,
            decode_attn_mask=mask,
            cp_prefill_mask=None,
        )
        return y

    _profile_forward(device, _fwd)
    if _own_rope:
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
    print(
        f"[cp_layer_decode] seq_len={DEMO_CP_DECODE_SEQ} hidden={cfg.hidden_size} "
        f"start_pos={start_pos} kv_max={DEMO_CP_KV_MAX}"
    )


def _synthetic_speaker_tdnn_sd():
    torch.manual_seed(0)
    return {
        "speaker_encoder.blocks.0.conv.weight": torch.randn(512, 128, 5),
        "speaker_encoder.blocks.0.conv.bias": torch.zeros(512),
    }


def _synthetic_speaker_block_sd(block_idx: int = 1):
    """Random weights for one SERes2Net block (512-ch).

    Matches HF: TDNN1/2 are k=1 (device linear); Res2Net parts are k=3 dilated.
    """
    torch.manual_seed(0)
    c, res2_k, scale, se_mid = 512, 3, 8, 128
    part = c // scale
    p = f"speaker_encoder.blocks.{block_idx}."
    sd = {
        f"{p}tdnn1.conv.weight": torch.randn(c, c, 1),
        f"{p}tdnn1.conv.bias": torch.zeros(c),
        f"{p}tdnn2.conv.weight": torch.randn(c, c, 1),
        f"{p}tdnn2.conv.bias": torch.zeros(c),
        f"{p}se_block.conv1.weight": torch.randn(se_mid, c, 1),
        f"{p}se_block.conv1.bias": torch.zeros(se_mid),
        f"{p}se_block.conv2.weight": torch.randn(c, se_mid, 1),
        f"{p}se_block.conv2.bias": torch.zeros(c),
    }
    for i in range(scale - 1):
        sd[f"{p}res2net_block.blocks.{i}.conv.weight"] = torch.randn(part, part, res2_k)
        sd[f"{p}res2net_block.blocks.{i}.conv.bias"] = torch.zeros(part)
    return sd


def _synthetic_speaker_encoder_sd():
    """Entry TDNN + 3 SERes2Net + MFA + ASP + FC — same graph as the demo encoder."""
    sd = dict(_synthetic_speaker_tdnn_sd())
    for idx in (1, 2, 3):
        sd.update(_synthetic_speaker_block_sd(idx))
    c = 512
    mfa = c * 3
    sd["speaker_encoder.mfa.conv.weight"] = torch.randn(mfa, mfa, 1)
    sd["speaker_encoder.mfa.conv.bias"] = torch.zeros(mfa)
    # ASP tdnn is conv([x; mean; std]) → [1536, 4608, 1]; ASP conv is k=1 1536→1536.
    sd["speaker_encoder.asp.tdnn.conv.weight"] = torch.randn(mfa, mfa * 3, 1)
    sd["speaker_encoder.asp.tdnn.conv.bias"] = torch.zeros(mfa)
    sd["speaker_encoder.asp.conv.weight"] = torch.randn(mfa, mfa, 1)
    sd["speaker_encoder.asp.conv.bias"] = torch.zeros(mfa)
    sd["speaker_encoder.fc.weight"] = torch.randn(2048, mfa * 2)
    sd["speaker_encoder.fc.bias"] = torch.zeros(2048)
    return sd


@pytest.mark.timeout(600)
def test_speaker_tdnn(device):
    """SpeakerEncoder entry TDNN (128→512) at demo-like T=384, on-device + traced.

    k=5 reflect-pad conv as im2col + matmul (``_se_device_conv``), captured as a
    Metal trace and replayed between signposts — the same path
    ``capture_forward_trace`` uses. Eager device-conv is not this window.
    """
    from models.demos.qwen3_tts.tt.speaker_encoder import SpeakerEncoder, SpeakerEncoderConfig

    enc = SpeakerEncoder(device, _synthetic_speaker_tdnn_sd(), config=SpeakerEncoderConfig())
    _enable_traced_device_conv(enc)
    mel_ncl = torch.randn(1, 128, DEMO_SPEAKER_T)
    x = enc._torch_ncl_to_ttnn_nlc(mel_ncl)
    w = enc.pytorch_weights["blocks.0.conv.weight"]
    b = enc.pytorch_weights["blocks.0.conv.bias"]

    def _fwd():
        return enc._time_delay_net_block(x, w, b, dilation=1)

    _profile_traced_forward(device, _fwd)
    print(f"[speaker_tdnn] blocks.0 128→512 seq={DEMO_SPEAKER_T} device_conv+trace")


@pytest.mark.timeout(600)
def test_speaker_block(device):
    """One SERes2Net block at T=384, on-device + traced (device-conv for k=3 branches)."""
    from models.demos.qwen3_tts.tt.speaker_encoder import SpeakerEncoder, SpeakerEncoderConfig

    enc = SpeakerEncoder(device, _synthetic_speaker_block_sd(1), config=SpeakerEncoderConfig())
    _enable_traced_device_conv(enc)
    mel_ncl = torch.randn(1, 512, DEMO_SPEAKER_T)
    x = enc._torch_ncl_to_ttnn_nlc(mel_ncl)

    def _fwd():
        return enc._se_res2net_block(x, block_idx=1, scale=8)

    _profile_traced_forward(device, _fwd)
    print(f"[speaker_block] SERes2Net block_idx=1 channels=512 seq={DEMO_SPEAKER_T} device_conv+trace")


@pytest.mark.timeout(900)
def test_speaker_encoder(device):
    """Full ECAPA ``_forward_device`` Metal-trace replay (demo ``SE_TRACE=1`` path).

    ``capture_forward_trace`` forces device conv/ASP so entry TDNN, 3× SERes2Net,
    MFA, ASP and FC stay on device inside the trace. Mel length is the demo's T=384.
    """
    from models.demos.qwen3_tts.tt.speaker_encoder import SpeakerEncoder, SpeakerEncoderConfig

    enc = SpeakerEncoder(device, _synthetic_speaker_encoder_sd(), config=SpeakerEncoderConfig())
    enc.capture_forward_trace(DEMO_SPEAKER_T)
    tid = enc._fwd_traces[DEMO_SPEAKER_T]["trace_id"]
    signpost("start")
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    signpost("stop")
    print(f"[speaker_encoder] capture_forward_trace mel_T={DEMO_SPEAKER_T}")
