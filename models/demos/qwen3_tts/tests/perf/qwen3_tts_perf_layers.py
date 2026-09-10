# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The profiled device graphs for the single-block / single-step perf windows.

Sequence lengths match ``demo_full_ttnn_tts`` / ``server.py``:

  Talker prefill  32 / 64 / 128   TRACE buckets (different matmul M and kernel path)
  Talker decode   seq=1           QKV/MLP always M=1; the deployed path attends over
                                  the whole KV cache (kv_max=352 for the 64 bucket)
  CP prefill      seq=2           always (talker hidden + code0); TILE-padded to 32
  CP decode       seq=1           KV max=32 always

Talker 32 vs 64/128 is a different QKV path (DRAM-sharded at seq<=32). 64 vs 128
share that family but M and RMSNorm/concat shard height differ — capture each.

CP has no extra seq variants in the demo: it is always 2 then 1.

This module owns the layer construction, the deployed-decode buffers and the
``start`` / ``stop`` signposts. ``qwen3_tts_perf_common`` calls into it rather than
redefining any of it — a second, drifting copy of these graphs once profiled
bfloat16 gate/up against the bfloat8_b the model actually ran, so the weight dtype
here is read from the same env flag ``talker.py`` reads.

Deliberately NOT a ``test_*`` module: it defines no tests and pytest must not
collect it. The window functions are plain callables.
"""

import contextlib
import os

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
# Decode position for the traced-decode window. Any pos < kv_max gives the same op
# shapes, so this only has to be a realistic mid-generation position.
DEMO_TALKER_DECODE_POS = 128
DEMO_MAX_NEW_TOKENS = 256
_TILE = 32


def _talker_kv_max(padded_seq_len: int) -> int:
    """Matches server.py: tile(padded_prefill + max_new_tokens + 16)."""
    return (((padded_seq_len + DEMO_MAX_NEW_TOKENS + 16) + _TILE - 1) // _TILE) * _TILE


DEMO_CP_PREFILL_SEQ = 2
DEMO_CP_DECODE_SEQ = 1
DEMO_CP_KV_MAX = 32

# Mirrors the demo's trace region so a traced window allocates the same way it does.
_TRACE_REGION = 200_000_000


def open_device():
    """Device opened the way the demo opens it, honouring MESH_DEVICE."""
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


def close_device(device, mesh_shape):
    if mesh_shape is None:
        ttnn.close_device(device)
        return
    ttnn.close_mesh_device(device)
    if mesh_shape != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


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


def synthetic_cp_sd(cfg, talker_hidden=2048, num_layers=1):
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

    Must mirror ``talker.py``'s ``_matmul_dtype`` exactly. Hardcoding bfloat16 here
    is the bug that made these windows profile a dtype the demo no longer runs after
    ``QWEN3_TTS_BF8_WEIGHTS`` became default ON.
    """
    return ttnn.bfloat8_b if os.environ.get("QWEN3_TTS_BF8_WEIGHTS", "1") != "0" else ttnn.bfloat16


def make_talker_layer(device, cfg):
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


# Signpost names and warmup behaviour for the windows below. The defaults are
# signposts named start/stop and a compile pass before the measured one;
# ``profile_window`` overrides them for callers that compose several windows into ONE
# Tracy capture.
_WINDOW_START, _WINDOW_STOP, _WINDOW_WARMUP = "start", "stop", True


@contextlib.contextmanager
def profile_window(start: str, stop: str, warmup: bool = True):
    """Rename this window's signposts, and optionally skip its compile pass.

    Composing windows into one capture needs both. Distinct names, so the sub-windows
    can be sliced apart afterwards — with every window emitting ``start``/``stop``,
    ``tt-perf-report`` takes the first ``start`` to the first ``stop`` and silently
    reports only the first window. And ``warmup=False`` on the measured pass, because
    ``_profile_forward``'s compile run is NOT signposted, so inside an enclosing
    ``start``/``stop`` it would land in the enclosing window and roughly double it.

    The intended shape is therefore two passes: one to compile (warmup on, throwaway
    names), then the measured one inside the outer signposts.
    """
    global _WINDOW_START, _WINDOW_STOP, _WINDOW_WARMUP
    prev = (_WINDOW_START, _WINDOW_STOP, _WINDOW_WARMUP)
    _WINDOW_START, _WINDOW_STOP, _WINDOW_WARMUP = start, stop, warmup
    try:
        yield
    finally:
        _WINDOW_START, _WINDOW_STOP, _WINDOW_WARMUP = prev


def _profile_forward(device, fn):
    """Compile once, then measure one warm forward between the window's signposts."""
    if _WINDOW_WARMUP:
        fn()
        ttnn.synchronize_device(device)
    signpost(_WINDOW_START)
    fn()
    ttnn.synchronize_device(device)
    signpost(_WINDOW_STOP)


def run_talker_prefill(device, talker_layer, seq_len: int):
    """One Talker DecoderLayer, prefill, at a demo TRACE bucket."""
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


def run_talker_decode_traced(device, talker_layer):
    """One Talker DecoderLayer on the **deployed** decode path (server.py trace form).

    The eager fallback — ``cur_pos_tensor`` and ``decode_attn_mask`` both None — makes
    the layer write the cache with ``update_cache`` and slice it to ``start_pos+1``, so
    SDPA sees a 1-position K/V. The demo never runs that graph. Under Metal trace,
    ``server.py`` passes a device ``cur_pos_tensor`` + a full ``[1, heads, 1, kv_max]``
    mask, so the layer uses ``paged_fused_update_cache`` and attends over the WHOLE
    cache (kv_max=352 for the Japanese bucket). That makes every attention op
    ~kv_max/32 times larger and is where the GQA-expansion / typecast overhead lands.

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


def run_cp_layer_prefill(device, code_predictor):
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


def run_cp_layer_decode(device, code_predictor):
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
