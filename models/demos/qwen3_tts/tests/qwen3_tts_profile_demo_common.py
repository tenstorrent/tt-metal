# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Demo-faithful inputs and buffers for Tracy profile tests.

Builds the same tensors / shapes / server helpers the CLI demo uses
(``demo_full_ttnn_tts.py`` → ``generate_codes_ttnn``), without running the
full generation loop.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
import torch

import ttnn

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REF_AUDIO = REPO_ROOT / "models/demos/qwen3_tts/demo/jim_reference.wav"
DEFAULT_REF_TEXT_PATH = DEFAULT_REF_AUDIO.with_suffix(".txt")
DEFAULT_TARGET_TEXT = (
    "Good morning. Today is a beautiful day for a walk in the park, with bright sun "
    "and a gentle breeze through the trees."
)
DEFAULT_HF_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
_TRACE_REGION = 200_000_000
_L1_SMALL = 32768
_TRACED_PREFILL_BUCKETS = (32, 64, 128)
_TILE = 32
_MAX_CP_SEQ = 32
_PROFILE_CACHE_DIR = REPO_ROOT / "models/demos/qwen3_tts/ops_list/.profile_cache"
_PROFILER_FLUSH_LAYERS = 7


def profile_measure_mode() -> bool:
    return os.environ.get("QWEN3_TTS_PROFILE_MEASURE", "0") != "0"


def profile_warmup_mode() -> bool:
    return os.environ.get("QWEN3_TTS_PROFILE_WARMUP", "0") != "0"


def flush_profiler(device) -> None:
    ttnn.ReadDeviceProfiler(device)


def _profile_cache_key() -> str:
    blob = f"{hf_id()}|{demo_ref_text()}|{demo_target_text()}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _icl_cache_path() -> Path:
    safe_hf = hf_id().replace("/", "_")
    return _PROFILE_CACHE_DIR / f"icl_{safe_hf}_{_profile_cache_key()}.pt"


def _save_icl_cache(state: Dict[str, Any]) -> None:
    from models.demos.qwen3_tts.tt.mesh_utils import to_torch as mesh_to_torch

    _PROFILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "real_seq_len": state["real_seq_len"],
            "config_hidden_size": state["config"].hidden_size,
            "config_top_k": state["config"].top_k,
            "config_temperature": state["config"].temperature,
            "inputs_embeds": mesh_to_torch(state["inputs_embeds_tt"]).bfloat16().cpu(),
            "trailing_text_hidden": state["trailing_text_hidden"].float().cpu(),
            "tts_pad_embed": state["tts_pad_embed"].float().cpu(),
            "code_pred_embeds": [t.float().cpu() for t in state["code_pred_embeds"]],
        },
        _icl_cache_path(),
    )


def _load_icl_cache(device, model) -> Dict[str, Any]:
    from models.demos.qwen3_tts.tt.server import TTSConfig

    payload = torch.load(_icl_cache_path(), map_location="cpu", weights_only=True)
    config = TTSConfig()
    config.hidden_size = int(payload["config_hidden_size"])
    config.top_k = int(payload["config_top_k"])
    config.temperature = float(payload["config_temperature"])
    inputs_embeds_tt = ttnn.from_torch(
        payload["inputs_embeds"],
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return {
        "config": config,
        "inputs_embeds_tt": inputs_embeds_tt,
        "trailing_text_hidden": payload["trailing_text_hidden"],
        "tts_pad_embed": payload["tts_pad_embed"],
        "code_pred_embeds": payload["code_pred_embeds"],
        "real_seq_len": int(payload["real_seq_len"]),
    }


def _ar0_cache_path() -> Path:
    safe_hf = hf_id().replace("/", "_")
    return _PROFILE_CACHE_DIR / f"ar0_{safe_hf}_{_profile_cache_key()}.pt"


def _kv_caches_to_cpu(kv_caches) -> list:
    return [(_mesh_to_torch(k).bfloat16().cpu(), _mesh_to_torch(v).bfloat16().cpu()) for k, v in kv_caches]


def save_ar0_warmup_state(
    *,
    real_seq_len: int,
    padded_seq_len: int,
    max_talker_seq_len: int,
    token_0: int,
    talker_hidden_tt,
    talker_kv_caches,
) -> Path:
    _PROFILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _ar0_cache_path()
    torch.save(
        {
            "real_seq_len": int(real_seq_len),
            "padded_seq_len": int(padded_seq_len),
            "max_talker_seq_len": int(max_talker_seq_len),
            "token_0": int(token_0),
            "talker_hidden": _mesh_to_torch(talker_hidden_tt).bfloat16().cpu(),
            "talker_kv": _kv_caches_to_cpu(talker_kv_caches),
        },
        path,
    )
    return path


def load_ar0_warmup_state() -> dict:
    path = _ar0_cache_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"AR step 0 warmup cache missing at {path}. "
            "Run: QWEN3_TTS_PROFILE_WARMUP=1 pytest .../test_qwen3_tts_profile_ar_step0.py"
        )
    return torch.load(path, map_location="cpu", weights_only=True)


def restore_talker_kv_caches(device, kv_caches, saved_kv: list) -> None:
    for (k_cache, v_cache), (k_cpu, v_cpu) in zip(kv_caches, saved_kv):
        k_host = ttnn.from_torch(k_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        v_host = ttnn.from_torch(v_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(k_host, k_cache)
        ttnn.copy_host_to_device_tensor(v_host, v_cache)


def build_or_load_demo_icl_state(device, model, main_weights, *, use_cache: bool = None) -> Dict[str, Any]:
    """ICL state for the demo prompt, from the on-disk cache when one exists.

    ``use_cache`` defaults to :func:`profile_measure_mode`; pass ``True`` to take the
    cache whenever it is present. Building it runs the SpeakerEncoder and the speech
    tokenizer on device — hundreds of ops that a profiling run does not want to pay
    for or to have sharing the profiler's DRAM buffer with the window under test.
    """
    cache = _icl_cache_path()
    if use_cache is None:
        use_cache = profile_measure_mode()
    if use_cache and cache.is_file():
        return _load_icl_cache(device, model)
    state = build_demo_icl_state(device, model, main_weights)
    if profile_warmup_mode() or not cache.is_file():
        _save_icl_cache(state)
    return state


def hf_id() -> str:
    return os.environ.get("HF_MODEL") or os.environ.get("QWEN3_TTS_HF_ID", DEFAULT_HF_ID)


def demo_ref_text() -> str:
    explicit = os.environ.get("QWEN3_TTS_PROFILE_REF_TEXT")
    if explicit is not None:
        return explicit.strip()
    if DEFAULT_REF_TEXT_PATH.is_file():
        return DEFAULT_REF_TEXT_PATH.read_text().strip()
    raise RuntimeError("Set QWEN3_TTS_PROFILE_REF_TEXT or place a .txt next to jim_reference.wav")


def demo_target_text() -> str:
    return os.environ.get("QWEN3_TTS_PROFILE_TARGET_TEXT", DEFAULT_TARGET_TEXT)


def open_profile_device() -> Tuple[Any, Any]:
    mesh_shape = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(os.environ.get("MESH_DEVICE"))
    if mesh_shape is None:
        device = ttnn.open_device(
            device_id=0,
            l1_small_size=_L1_SMALL,
            trace_region_size=_TRACE_REGION,
        )
        device.enable_program_cache()
        return device, None
    if mesh_shape != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*mesh_shape),
        l1_small_size=_L1_SMALL,
        trace_region_size=_TRACE_REGION,
    )
    device.enable_program_cache()
    return device, mesh_shape


def close_profile_device(device, mesh_shape) -> None:
    if mesh_shape is None:
        ttnn.close_device(device)
        return
    ttnn.close_mesh_device(device)
    if mesh_shape != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def profile_device():
    device, mesh_shape = open_profile_device()
    yield device
    close_profile_device(device, mesh_shape)


@pytest.fixture(scope="module")
def demo_model(profile_device):
    from models.demos.qwen3_tts.tt.model_config import talker_config_for_hf_id
    from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS
    from models.demos.qwen3_tts.tt.server import load_weights

    flush_profiler(profile_device)
    main_weights, _ = load_weights(hf_id())
    talker_config = talker_config_for_hf_id(hf_id())
    model = Qwen3TTS(device=profile_device, state_dict=main_weights, talker_config=talker_config)
    ttnn.synchronize_device(profile_device)
    flush_profiler(profile_device)
    return model, main_weights


def _padded_max_talker_seq(padded_seq_len: int, max_new_tokens: int = 256) -> int:
    raw = padded_seq_len + max_new_tokens + 16
    return ((raw + _TILE - 1) // _TILE) * _TILE


def build_demo_icl_state(device, model, main_weights) -> Dict[str, Any]:
    """Reference encode + speaker embed + ICL, matching ``run_full_ttnn_tts``."""
    from transformers import AutoTokenizer

    from models.demos.qwen3_tts.tt.server import TTSConfig, create_icl_embedding_ttnn, encode_reference_audio

    if os.environ.get("QWEN3_TTS_SE_TRACE", "0") != "0":
        ref_codes, audio_data = encode_reference_audio(str(DEFAULT_REF_AUDIO), main_weights)
        se = model.speaker_encoder
        se.capture_se_block_traces()
        se.capture_fc_trace()
        se.capture_audio_forward_trace(audio_data)
        if not se._audio_traces:
            # Waveform the device mel cannot take; fall back to the mel-in trace.
            se.capture_forward_trace(int(se.compute_mel_spectrogram(audio_data).shape[-1]))
        se.activate_traced_extract()

    ref_codes, audio_data = encode_reference_audio(str(DEFAULT_REF_AUDIO), main_weights)
    speaker_embedding = model.extract_speaker_embedding(audio_data)

    config = TTSConfig()
    config.hidden_size = model.talker_config.hidden_size
    tokenizer = AutoTokenizer.from_pretrained(hf_id(), trust_remote_code=True)
    inputs_embeds_tt, trailing_text_hidden, tts_pad_embed, code_pred_embeds = create_icl_embedding_ttnn(
        target_text=demo_target_text(),
        ref_text=demo_ref_text(),
        ref_codes=ref_codes,
        speaker_embedding=speaker_embedding,
        tokenizer=tokenizer,
        model=model,
        device=device,
        config=config,
        main_weights=main_weights,
    )
    real_seq_len = int(inputs_embeds_tt.shape[2])
    return {
        "config": config,
        "inputs_embeds_tt": inputs_embeds_tt,
        "trailing_text_hidden": trailing_text_hidden,
        "tts_pad_embed": tts_pad_embed,
        "code_pred_embeds": code_pred_embeds,
        "real_seq_len": real_seq_len,
        "ref_codes": ref_codes,
    }


def pad_inputs_to_demo_bucket(device, inputs_embeds_tt, real_seq_len: int, talker_h: int) -> Tuple[ttnn.Tensor, int]:
    """STEP 1 bucket padding from ``generate_codes_ttnn``."""
    if real_seq_len <= _TRACED_PREFILL_BUCKETS[-1]:
        padded_seq_len = next(b for b in _TRACED_PREFILL_BUCKETS if b >= real_seq_len)
    else:
        from models.demos.qwen3_tts.tt.server import get_padded_prefill_len

        padded_seq_len = get_padded_prefill_len(real_seq_len)

    if padded_seq_len > real_seq_len:
        pad_len = padded_seq_len - real_seq_len
        pad_zeros = ttnn.from_torch(
            torch.zeros(1, 1, pad_len, talker_h, dtype=torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        inputs_embeds_tt = ttnn.concat([inputs_embeds_tt, pad_zeros], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(pad_zeros)
    return inputs_embeds_tt, padded_seq_len


def allocate_talker_kv(device, model, padded_seq_len: int, max_new_tokens: int = 256):
    from models.demos.qwen3_tts.tt.server import allocate_kv_cache

    head_dim = model.talker_config.head_dim
    max_talker_seq_len = _padded_max_talker_seq(padded_seq_len, max_new_tokens)
    talker_kv_caches = allocate_kv_cache(
        device=device,
        num_layers=model.talker_config.num_hidden_layers,
        batch_size=1,
        num_kv_heads=model.talker_config.num_key_value_heads,
        max_seq_len=max_talker_seq_len,
        head_dim=head_dim,
    )
    return talker_kv_caches, max_talker_seq_len


def run_talker_prefill_untraced(
    device,
    model,
    inputs_embeds_tt,
    padded_seq_len: int,
    real_seq_len: int,
    talker_kv_caches,
    *,
    profiler_flush_layers: int = 0,
):
    """One untraced Talker prefill + codec_head (same kernels the demo trace captures)."""
    from models.demos.qwen3_tts.tt.rope import get_rope_tensors, get_transformation_mat
    from models.demos.qwen3_tts.tt.server import sample_token

    head_dim = model.talker_config.head_dim
    talker_trans_mat = get_transformation_mat(head_dim, device)
    prefill_cos_tt, prefill_sin_tt = get_rope_tensors(
        device,
        head_dim,
        padded_seq_len,
        torch.arange(padded_seq_len),
        model.talker_config.rope_theta,
    )
    if profiler_flush_layers > 0:
        hidden_out, talker_kv_caches = _talker_prefill_with_profiler_flushes(
            model.talker,
            device,
            inputs_embeds_tt,
            prefill_cos_tt,
            prefill_sin_tt,
            talker_trans_mat,
            talker_kv_caches,
            flush_every=profiler_flush_layers,
        )
    else:
        hidden_out, talker_kv_caches = model.talker.forward_from_hidden(
            inputs_embeds_tt,
            prefill_cos_tt,
            prefill_sin_tt,
            talker_trans_mat,
            kv_caches=talker_kv_caches,
            start_pos=0,
            mode="prefill",
        )
    logits_out = model.talker.get_codec_logits(hidden_out)
    ttnn.synchronize_device(device)
    codec_logits_full = _mesh_to_torch(logits_out).squeeze(1).float()
    codec_logits_torch = codec_logits_full[0, real_seq_len - 1, :]
    token_0 = sample_token(
        codec_logits_torch,
        temperature=1.0,
        top_k=0,
        greedy=False,
        repetition_penalty=1.0,
        generated_tokens=[],
    )

    ttnn.deallocate(prefill_cos_tt)
    ttnn.deallocate(prefill_sin_tt)
    return hidden_out, logits_out, talker_kv_caches, int(token_0)


def _talker_prefill_with_profiler_flushes(
    talker,
    device,
    hidden_states,
    cos,
    sin,
    transformation_mat,
    kv_caches,
    *,
    flush_every: int,
):
    if len(hidden_states.shape) == 3 or hidden_states.shape[1] != 1:
        if len(hidden_states.shape) == 3:
            hidden_states = ttnn.reshape(
                hidden_states,
                (hidden_states.shape[0], 1, hidden_states.shape[1], hidden_states.shape[2]),
            )

    updated_kv_caches = []
    for i, layer in enumerate(talker.layers):
        hidden_states, updated_kv_cache = layer(
            hidden_states,
            cos,
            sin,
            transformation_mat,
            None,
            kv_cache=kv_caches[i],
            start_pos=0,
            mode="prefill",
        )
        updated_kv_caches.append(updated_kv_cache)
        if (i + 1) % flush_every == 0:
            ttnn.synchronize_device(device)
            flush_profiler(device)

    if hidden_states.is_sharded():
        hidden_il = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(hidden_states)
        hidden_states = hidden_il
    hidden_states = talker.norm(hidden_states)
    return hidden_states, updated_kv_caches


def _mesh_to_torch(tensor, dtype=None):
    from models.demos.qwen3_tts.tt.mesh_utils import to_torch as mesh_to_torch

    return mesh_to_torch(tensor, dtype=dtype)


def build_fused_cp_profile_buffers(
    device,
    model,
    config,
    *,
    talker_hidden_tt,
    trailing_text_hidden,
    tts_pad_embed,
    code_pred_embeds,
    talker_embed_dst_tt,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """Persistent buffers for ``capture_fused_cp_trace`` with production weights."""
    from models.demos.qwen3_tts.tt.mesh_utils import get_tp_size, is_mesh_device
    from models.demos.qwen3_tts.tt.rope import compute_rope_frequencies, get_rope_tensors, get_transformation_mat
    from models.demos.qwen3_tts.tt.server import (
        _DeviceSampler,
        _replicate_mapper,
        allocate_kv_cache,
        build_cp_decode_trace_h2d_constants,
        build_trailing_row_h2d,
        upload_embed_tables,
    )

    talker_h = model.talker_config.hidden_size
    cp_cfg = model.code_predictor_config
    cp_head_dim = cp_cfg.head_dim
    tp = get_tp_size(device) if is_mesh_device(device) else 1
    cp_num_heads = cp_cfg.num_attention_heads // tp

    cp_trans_mat = get_transformation_mat(cp_head_dim, device)
    codec_embed_torch = _mesh_to_torch(model.talker.codec_embedding).squeeze(0).squeeze(0).float()
    cp_tables_torch = [t.float() for t in code_pred_embeds]

    codec_embed_tt, cp_embed_tts = upload_embed_tables(device, codec_embed_torch, cp_tables_torch)
    sampler = _DeviceSampler(device, top_k=config.top_k, temperature=config.temperature)
    tok_bufs = [sampler.alloc_token_buf() for _ in range(config.num_code_groups)]
    sampler.warm_ccl()
    mapper = _replicate_mapper(device)

    pf_seq = int(talker_hidden_tt.shape[2])
    if pf_seq > 1:
        src_hidden_tt = ttnn.slice(talker_hidden_tt, [0, 0, pf_seq - 1, 0], [1, 1, pf_seq, talker_h])
    else:
        src_hidden_tt = talker_hidden_tt

    cp_kv_caches = allocate_kv_cache(
        device=device,
        num_layers=cp_cfg.num_hidden_layers,
        batch_size=1,
        num_kv_heads=cp_cfg.num_key_value_heads,
        max_seq_len=_MAX_CP_SEQ,
        head_dim=cp_head_dim,
    )
    cp_kv_zero_hosts = []
    for k_cache, v_cache in cp_kv_caches:
        z = lambda c: ttnn.from_torch(  # noqa: E731
            torch.zeros(c.shape[0], c.shape[1], c.shape[2], c.shape[3], dtype=torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cp_kv_zero_hosts.append((z(k_cache), z(v_cache)))

    cp_prefill_cos_tt, cp_prefill_sin_tt = get_rope_tensors(device, cp_head_dim, 2, torch.arange(2), cp_cfg.rope_theta)
    cp_prefill_cos_src = ttnn.from_torch(
        _mesh_to_torch(cp_prefill_cos_tt).bfloat16(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    cp_prefill_sin_src = ttnn.from_torch(
        _mesh_to_torch(cp_prefill_sin_tt).bfloat16(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    mask_host = torch.full((1, cp_num_heads, 2, _MAX_CP_SEQ), float("-inf"))
    mask_host[0, :, 0, 0] = 0.0
    mask_host[0, :, 1, 0:2] = 0.0
    cp_prefill_mask_tt = ttnn.from_torch(
        mask_host, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    cp_prefill_mask_src = ttnn.from_torch(
        mask_host.float(),
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    cp_prefill_embed_tt = ttnn.from_torch(
        torch.zeros(1, 1, 2, talker_h, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    cp_cos_table, cp_sin_table = compute_rope_frequencies(cp_head_dim, _MAX_CP_SEQ + 5, cp_cfg.rope_theta)
    n_cp_decode = config.num_code_groups - 2
    cos_h2d, sin_h2d, mask_h2d = build_cp_decode_trace_h2d_constants(
        cp_cos_table, cp_sin_table, cp_num_heads, _MAX_CP_SEQ, n_cp_decode
    )
    cp_decode_embed_tt = ttnn.from_torch(
        torch.zeros(1, 1, 1, talker_h, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    cp_decode_cos_tts = [ttnn.to_device(h, device, memory_config=ttnn.L1_MEMORY_CONFIG) for h in cos_h2d]
    cp_decode_sin_tts = [ttnn.to_device(h, device, memory_config=ttnn.L1_MEMORY_CONFIG) for h in sin_h2d]
    cp_decode_mask_tts = [ttnn.to_device(h, device, memory_config=ttnn.L1_MEMORY_CONFIG) for h in mask_h2d]

    trail_row_tt = ttnn.from_torch(
        torch.zeros(1, 1, 1, talker_h, dtype=torch.float32),
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    trail_row_h2d = build_trailing_row_h2d(trailing_text_hidden, tts_pad_embed, max_new_tokens)

    return {
        "model": model,
        "config": config,
        "cp_trans_mat": cp_trans_mat,
        "cp_kv_caches": cp_kv_caches,
        "cp_kv_zero_hosts": cp_kv_zero_hosts,
        "cp_prefill_embed_tt": cp_prefill_embed_tt,
        "cp_prefill_mask_tt": cp_prefill_mask_tt,
        "cp_prefill_cos_tt": cp_prefill_cos_tt,
        "cp_prefill_sin_tt": cp_prefill_sin_tt,
        "cp_prefill_mask_src": cp_prefill_mask_src,
        "cp_prefill_cos_src": cp_prefill_cos_src,
        "cp_prefill_sin_src": cp_prefill_sin_src,
        "cp_decode_embed_tt": cp_decode_embed_tt,
        "cp_decode_cos_tts": cp_decode_cos_tts,
        "cp_decode_sin_tts": cp_decode_sin_tts,
        "cp_decode_mask_tts": cp_decode_mask_tts,
        "talker_hidden_src_tt": src_hidden_tt,
        "talker_embed_dst_tt": talker_embed_dst_tt,
        "codec_embed_tt": codec_embed_tt,
        "cp_embed_tts": cp_embed_tts,
        "talker_h": talker_h,
        "sampler": sampler,
        "tok_bufs": tok_bufs,
        "trail_row_tt": trail_row_tt,
        "trail_row_h2d": trail_row_h2d,
    }


def build_talker_decode_profile_buffers(
    device,
    model,
    *,
    real_seq_len: int,
    max_talker_seq_len: int,
    talker_kv_caches,
) -> Dict[str, Any]:
    """Trace input buffers for one Talker decode step at ``T=real_seq_len``."""
    from models.demos.qwen3_tts.tt.mesh_utils import get_tp_size, is_mesh_device
    from models.demos.qwen3_tts.tt.rope import compute_rope_frequencies, get_transformation_mat
    from models.demos.qwen3_tts.tt.server import build_talker_decode_trace_h2d_constants

    talker_h = model.talker_config.hidden_size
    head_dim = model.talker_config.head_dim
    tp = get_tp_size(device) if is_mesh_device(device) else 1
    num_heads = model.talker_config.num_attention_heads // tp

    talker_trans_mat = get_transformation_mat(head_dim, device)
    talker_cos_table, talker_sin_table = compute_rope_frequencies(
        head_dim, max_talker_seq_len + 50, model.talker_config.rope_theta
    )
    cos_h2d, sin_h2d, mask_h2d, pos_h2d = build_talker_decode_trace_h2d_constants(
        talker_cos_table, talker_sin_table, num_heads, max_talker_seq_len, real_seq_len
    )
    idx = 0  # first decode at T == real_seq_len

    trace_embed_tt = ttnn.from_torch(
        torch.zeros(1, 1, 1, talker_h, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    trace_cos_tt = ttnn.from_torch(
        torch.ones(1, 1, 1, head_dim, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    trace_sin_tt = ttnn.from_torch(
        torch.zeros(1, 1, 1, head_dim, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    trace_cur_pos_tt = ttnn.from_torch(
        torch.tensor([real_seq_len], dtype=torch.int32),
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # bf16 single-head, matching build_talker_decode_trace_h2d_constants (the mask
    # holds only 0.0 and -inf, and every head held an identical row).
    trace_mask_tt = ttnn.from_torch(
        torch.full((1, 1, 1, max_talker_seq_len), float("-inf")).bfloat16(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    return {
        "model": model,
        "talker_trans_mat": talker_trans_mat,
        "talker_kv_caches": talker_kv_caches,
        "trace_embed_tt": trace_embed_tt,
        "trace_cos_tt": trace_cos_tt,
        "trace_sin_tt": trace_sin_tt,
        "trace_cur_pos_tt": trace_cur_pos_tt,
        "trace_mask_tt": trace_mask_tt,
        "cos_h2d": cos_h2d[idx],
        "sin_h2d": sin_h2d[idx],
        "mask_h2d": mask_h2d[idx],
        "cur_pos_h2d": pos_h2d[idx],
    }


def run_talker_decode_untraced(decode_ctx: Dict[str, Any]) -> None:
    """One untraced Talker decode + codec_head (demo trace body)."""
    ttnn.copy_host_to_device_tensor(decode_ctx["cos_h2d"], decode_ctx["trace_cos_tt"])
    ttnn.copy_host_to_device_tensor(decode_ctx["sin_h2d"], decode_ctx["trace_sin_tt"])
    ttnn.copy_host_to_device_tensor(decode_ctx["cur_pos_h2d"], decode_ctx["trace_cur_pos_tt"])
    ttnn.copy_host_to_device_tensor(decode_ctx["mask_h2d"], decode_ctx["trace_mask_tt"])
    hidden, _ = decode_ctx["model"].talker.forward_from_hidden(
        decode_ctx["trace_embed_tt"],
        decode_ctx["trace_cos_tt"],
        decode_ctx["trace_sin_tt"],
        decode_ctx["talker_trans_mat"],
        kv_caches=decode_ctx["talker_kv_caches"],
        cur_pos_tensor=decode_ctx["trace_cur_pos_tt"],
        decode_attn_mask=decode_ctx["trace_mask_tt"],
        mode="decode",
    )
    logits = decode_ctx["model"].talker.get_codec_logits(hidden)
    ttnn.deallocate(logits)
    ttnn.deallocate(hidden)


# ── Demo trace capture ────────────────────────────────────────────────────────
# The demo replays Metal traces; an untraced pass has the same kernel graph but
# host-bound op-to-op gaps that swamp the report. These helpers capture the exact
# traces ``generate_codes_ttnn`` captures, so a profiling window can replay one.


def capture_talker_prefill_trace(device, model, bucket: int, talker_kv_caches) -> Dict[str, Any]:
    """Capture the demo's Talker prefill trace for one bucket (``generate_codes_ttnn`` STEP 3.5).

    Persistent DRAM embed buffer + constant RoPE for positions ``[0, bucket)``, wrapped
    around ``forward_from_hidden(mode="prefill")`` and ``get_codec_logits``. The untraced
    warmup is not optional: trace capture rejects the kernel compiles and the CCL global
    semaphore creation that a first call performs.
    """
    from models.demos.qwen3_tts.tt.rope import get_rope_tensors, get_transformation_mat

    head_dim = model.talker_config.head_dim
    talker_h = model.talker_config.hidden_size
    trans_mat = get_transformation_mat(head_dim, device)

    embed_tt = ttnn.from_torch(
        torch.zeros(1, 1, bucket, talker_h, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cos_tt, sin_tt = get_rope_tensors(device, head_dim, bucket, torch.arange(bucket), model.talker_config.rope_theta)

    wu_hidden, _ = model.talker.forward_from_hidden(
        embed_tt, cos_tt, sin_tt, trans_mat, kv_caches=talker_kv_caches, start_pos=0, mode="prefill"
    )
    model.talker.get_codec_logits(wu_hidden)
    ttnn.synchronize_device(device)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    try:
        hidden_out, _ = model.talker.forward_from_hidden(
            embed_tt, cos_tt, sin_tt, trans_mat, kv_caches=talker_kv_caches, start_pos=0, mode="prefill"
        )
        logits_out = model.talker.get_codec_logits(hidden_out)
    finally:
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
    # Replay once so the profiled replay is not the cold-dispatch one (~10-15 ms of
    # variance on the first execute_trace of a freshly captured trace).
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    ttnn.mark_corruptible(hidden_out)
    ttnn.mark_corruptible(logits_out)
    return {
        "trace_id": trace_id,
        "bucket": bucket,
        "embed_tt": embed_tt,
        "hidden_out": hidden_out,
        "logits_out": logits_out,
    }


def upload_prefill_embeds(device, prefill_trace: Dict[str, Any], inputs_embeds_tt) -> None:
    """H2D the real ICL embeddings into the prefill trace's persistent input buffer."""
    host = ttnn.from_torch(
        _mesh_to_torch(inputs_embeds_tt).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    ttnn.copy_host_to_device_tensor(host, prefill_trace["embed_tt"])


def capture_talker_decode_trace(device, model, decode_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Capture the demo's Talker decode trace (``generate_codes_ttnn`` STEP 5a).

    ``decode_ctx`` comes from :func:`build_talker_decode_profile_buffers`. Run
    :func:`run_talker_decode_untraced` at least once first — same reason as prefill.
    The trace body is forward + ``codec_head``; the in-trace codec-0 argmax exists only
    on the ``greedy`` path, which the demo does not take by default.
    """
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    try:
        hidden_out, _ = model.talker.forward_from_hidden(
            decode_ctx["trace_embed_tt"],
            decode_ctx["trace_cos_tt"],
            decode_ctx["trace_sin_tt"],
            decode_ctx["talker_trans_mat"],
            kv_caches=decode_ctx["talker_kv_caches"],
            cur_pos_tensor=decode_ctx["trace_cur_pos_tt"],
            decode_attn_mask=decode_ctx["trace_mask_tt"],
            mode="decode",
        )
        logits_out = model.talker.get_codec_logits(hidden_out)
    finally:
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    return {"trace_id": trace_id, "hidden_out": hidden_out, "logits_out": logits_out}


def capture_speaker_encoder_forward_trace(device, model, main_weights) -> Dict[str, Any]:
    """Capture the ECAPA forward trace the demo uses with ``QWEN3_TTS_SE_TRACE=1``.

    ``SpeakerEncoder.capture_forward_trace`` forces device conv/ASP for the capture
    region so every conv is on device inside the Metal trace. The partial
    ``speaker_tdnn`` / ``speaker_block`` tests in ``test_qwen3_tts_profile_single_layer``
    profile untraced slices with synthetic weights and default host-fuse — they miss
    most conv work (entry TDNN conv runs on the host; one SERes2Net block is not the
    full encoder).
    """
    from models.demos.qwen3_tts.tt.server import encode_reference_audio

    se = model.speaker_encoder
    _, audio_data = encode_reference_audio(str(DEFAULT_REF_AUDIO), main_weights)
    mel = se.compute_mel_spectrogram(audio_data)
    mel_len = int(mel.shape[-1])

    se.capture_forward_trace(mel_len)
    trace = se._fwd_traces[mel_len]

    mel_host = ttnn.from_torch(mel.permute(0, 2, 1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(mel_host, trace["input_tt"])
    ttnn.execute_trace(device, trace["trace_id"], cq_id=0, blocking=True)
    ttnn.synchronize_device(device)

    return {
        "trace_id": trace["trace_id"],
        "input_tt": trace["input_tt"],
        "mel_len": mel_len,
        "mel_host": mel_host,
    }
