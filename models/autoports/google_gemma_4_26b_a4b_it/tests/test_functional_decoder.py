# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import platform
import struct
import subprocess
import time
from pathlib import Path

import pytest
import requests
import torch
from huggingface_hub import hf_hub_url
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextRotaryEmbedding

import ttnn
import models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder as decoder_module
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
    FULL_BLOCK_SIZE,
    FULL_HEAD_DIM,
    FULL_NUM_KV_HEADS,
    HIDDEN_SIZE,
    MODEL_ID,
    NUM_Q_HEADS,
    SLIDING_BLOCK_SIZE,
    SLIDING_HEAD_DIM,
    SLIDING_NUM_KV_HEADS,
    FULL_KIND,
    PREFILL_SDPA_MAX_SEQ,
    SLIDING_KIND,
    FunctionalDecoder,
    _bounded_cache_fill_plan,
    _make_correctness_compute_config,
    _prefill_attention_path,
)
from models.common.utility_functions import comp_pcc

CONFIG_DIR = Path("models/demos/gemma4/configs/gemma-4-26B-A4B-it")
ARTIFACT_DIR = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder")


def _evidence_provenance(mesh_device, exact_command: str) -> dict:
    """Bind opt-in evidence to its exact source, test, build, and hardware."""

    decoder_path = Path(decoder_module.__file__).resolve()
    test_path = Path(__file__).resolve()
    try:
        checkout_git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        checkout_git_sha = "unavailable"
    try:
        device_ids = [int(device_id) for device_id in mesh_device.get_device_ids()]
    except (AttributeError, TypeError):
        device_ids = []
    extension_path = Path(ttnn._ttnn.__file__).resolve()
    try:
        arch = str(mesh_device.arch())
    except (AttributeError, TypeError):
        arch = os.getenv("ARCH_NAME", "unknown")
    return {
        "functional_decoder_sha256": hashlib.sha256(decoder_path.read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "checkout_git_sha": checkout_git_sha,
        "ttnn_extension_path": str(extension_path),
        "ttnn_extension_sha256": hashlib.sha256(extension_path.read_bytes()).hexdigest(),
        "hardware": {
            "arch": arch,
            "device_ids": device_ids,
            "platform": platform.platform(),
        },
        "exact_command": os.getenv("GEMMA4_EVIDENCE_COMMAND", exact_command),
    }


def _model_snapshot() -> Path | None:
    explicit = os.getenv("GEMMA4_MODEL_SNAPSHOT")
    if explicit:
        candidate = Path(explicit)
        return candidate if candidate.is_dir() else None
    roots = [Path(os.getenv("HF_HOME", Path.home() / ".cache/huggingface")), Path.home() / ".cache/huggingface"]
    for root in roots:
        snapshots = root / "hub/models--google--gemma-4-26B-A4B-it/snapshots"
        if snapshots.is_dir():
            for candidate in sorted(snapshots.iterdir()):
                if (candidate / "model.safetensors.index.json").is_file():
                    return candidate
    return None


def _load_text_config():
    return AutoConfig.from_pretrained(CONFIG_DIR, local_files_only=True).text_config


def _load_layer_state(layer_idx: int) -> dict[str, torch.Tensor]:
    snapshot = _model_snapshot()
    if snapshot is None:
        if os.getenv("GEMMA4_RANGE_DOWNLOAD") == "1":
            return _range_download_layer_state(layer_idx)
        pytest.skip("local HF snapshot not found; set GEMMA4_MODEL_SNAPSHOT or GEMMA4_RANGE_DOWNLOAD=1")
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer_idx}"
    state = {}
    shard_names = sorted({index[k] for k in index if k.startswith(prefix + ".")})
    for shard_name in shard_names:
        with safe_open(snapshot / shard_name, framework="pt", device="cpu") as shard:
            for key, mapped_shard_name in index.items():
                if mapped_shard_name == shard_name and key.startswith(prefix + "."):
                    state[key] = shard.get_tensor(key)
    return state


def _range_download_layer_state(layer_idx: int) -> dict[str, torch.Tensor]:
    """Fetch only one canonical HF layer from the public safetensor shards."""
    cache_dir = Path(os.getenv("GEMMA4_REAL_LAYER_CACHE", "/tmp/gemma4_real_layer_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"layer_{layer_idx}.safetensors"
    if cached.is_file():
        return load_file(cached)

    index_response = requests.get(hf_hub_url(MODEL_ID, "model.safetensors.index.json"), timeout=60)
    index_response.raise_for_status()
    weight_map = index_response.json()["weight_map"]
    prefix = f"model.language_model.layers.{layer_idx}."
    target_names = sorted(name for name in weight_map if name.startswith(prefix))
    if not target_names:
        raise RuntimeError(f"checkpoint index contains no tensors for {prefix}")

    tensors = {}
    for shard_name in sorted({weight_map[name] for name in target_names}):
        shard_url = hf_hub_url(MODEL_ID, shard_name)
        first = requests.get(shard_url, headers={"Range": "bytes=0-7"}, timeout=60)
        first.raise_for_status()
        header_len = struct.unpack("<Q", first.content)[0]
        header_response = requests.get(
            shard_url,
            headers={"Range": f"bytes=8-{8 + header_len - 1}"},
            timeout=60,
        )
        header_response.raise_for_status()
        header = json.loads(header_response.content)
        data_base = 8 + header_len
        for name in target_names:
            if weight_map[name] != shard_name:
                continue
            meta = header[name]
            start, end = meta["data_offsets"]
            response = requests.get(
                shard_url,
                headers={"Range": f"bytes={data_base + start}-{data_base + end - 1}"},
                timeout=600,
            )
            response.raise_for_status()
            dtype = {"BF16": torch.bfloat16, "F32": torch.float32}.get(meta["dtype"])
            if dtype is None:
                raise ValueError(f"unsupported checkpoint dtype {meta['dtype']} for {name}")
            tensors[name] = torch.frombuffer(bytearray(response.content), dtype=dtype).reshape(meta["shape"]).clone()
    save_file(tensors, cached)
    return tensors


def _causal_mask(seq_len: int, *, sliding_window: int | None) -> torch.Tensor:
    mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.bfloat16)
    bad = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    if sliding_window is not None:
        idx = torch.arange(seq_len)
        bad |= idx[None, :] < (idx[:, None] - sliding_window + 1)
    return mask.masked_fill(bad, torch.finfo(torch.bfloat16).min)


def _decode_mask(total_len: int, *, sliding_window: int | None) -> torch.Tensor:
    mask = torch.zeros(1, 1, 1, total_len, dtype=torch.bfloat16)
    if sliding_window is not None:
        idx = torch.arange(total_len)
        bad = idx < (total_len - sliding_window)
        mask = mask.masked_fill(bad.view(1, 1, 1, total_len), torch.finfo(torch.bfloat16).min)
    return mask


def _as_tt(mesh_device, tensor, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.as_tensor(
        tensor,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_torch(mesh_device, tensor):
    return ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=(1, 1)),
    )


def _cache_shape(layer_type: str, *, shared_physical: bool, token_capacity: int | None = None):
    if layer_type == "full_attention" and not shared_physical:
        block_size = FULL_BLOCK_SIZE
        num_heads = FULL_NUM_KV_HEADS
        head_dim = FULL_HEAD_DIM
        default_blocks = 2
    else:
        block_size = SLIDING_BLOCK_SIZE
        num_heads = SLIDING_NUM_KV_HEADS
        head_dim = SLIDING_HEAD_DIM
        default_blocks = 4
    blocks = default_blocks if token_capacity is None else (token_capacity + block_size - 1) // block_size
    return (blocks, num_heads, block_size, head_dim)


def _page_table(layer_type: str, *, shared_physical: bool, token_capacity: int | None = None):
    blocks = _cache_shape(layer_type, shared_physical=shared_physical, token_capacity=token_capacity)[0]
    return torch.arange(blocks, dtype=torch.int32).view(1, blocks)


def test_prefill_attention_dispatch_host():
    assert (
        _prefill_attention_path(
            PREFILL_SDPA_MAX_SEQ,
            is_sliding=False,
            has_paged_cache=True,
        )
        == "non_chunked"
    )
    assert (
        _prefill_attention_path(
            PREFILL_SDPA_MAX_SEQ + 32,
            is_sliding=False,
            has_paged_cache=True,
        )
        == "full_chunked"
    )
    assert (
        _prefill_attention_path(
            PREFILL_SDPA_MAX_SEQ + 32,
            is_sliding=True,
            has_paged_cache=True,
        )
        == "sliding_chunked"
    )
    assert (
        _prefill_attention_path(256, is_sliding=False, has_paged_cache=True, max_non_chunked_seq=128) == "full_chunked"
    )
    with pytest.raises(ValueError, match="paged cache"):
        _prefill_attention_path(
            PREFILL_SDPA_MAX_SEQ + 32,
            is_sliding=False,
            has_paged_cache=False,
        )


@pytest.mark.parametrize(
    "logical_seq_len,expected_prefix,expected_tail",
    [
        (1, 0, tuple(range(1))),
        (31, 0, tuple(range(31))),
        (32, 32, ()),
        (33, 32, (32,)),
        (1023, 992, tuple(range(992, 1023))),
        (1024, 1024, ()),
        (1025, 1024, (1024,)),
        (1055, 1024, tuple(range(1024, 1055))),
        (1056, 1056, ()),
    ],
)
def test_bounded_cache_fill_plan_host(logical_seq_len, expected_prefix, expected_tail):
    assert _bounded_cache_fill_plan(logical_seq_len) == (expected_prefix, expected_tail)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,shared_physical,decode_pcc",
    [
        pytest.param(0, True, 0.995, id="sliding_attention_shared_cache"),
        pytest.param(5, False, 0.995, id="full_attention_natural_cache"),
        pytest.param(5, True, 0.995, id="full_attention_shared_cache_view"),
    ],
)
def test_functional_decoder_real_weights_prefill_decode(
    mesh_device, device_params, layer_idx, shared_physical, decode_pcc
):
    cfg = _load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    state = _load_layer_state(layer_idx)
    seq_len = 32
    torch.manual_seed(layer_idx)

    hidden = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
    decode_hidden = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
    rotary = Gemma4TextRotaryEmbedding(cfg)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    decode_position_ids = torch.tensor([[seq_len]])
    cos, sin = rotary(hidden, position_ids, layer_type=layer_type)
    decode_cos, decode_sin = rotary(decode_hidden, decode_position_ids, layer_type=layer_type)

    prefix = f"model.language_model.layers.{layer_idx}"
    ref = Gemma4TextDecoderLayer(cfg, layer_idx=layer_idx).eval().to(dtype=torch.bfloat16)
    ref.load_state_dict({k[len(prefix) + 1 :]: v for k, v in state.items()}, strict=True)
    cache = DynamicCache(config=cfg)
    sliding_window = cfg.sliding_window if layer_type == "sliding_attention" else None
    with torch.no_grad():
        ref_prefill = ref(
            hidden,
            shared_kv_states={},
            position_embeddings=(cos, sin),
            attention_mask=_causal_mask(seq_len, sliding_window=sliding_window),
            position_ids=position_ids,
            past_key_values=cache,
        )
        ref_decode = ref(
            decode_hidden,
            shared_kv_states={},
            position_embeddings=(decode_cos, decode_sin),
            attention_mask=_decode_mask(seq_len + 1, sliding_window=sliding_window),
            position_ids=decode_position_ids,
            past_key_values=cache,
        )

    decoder = FunctionalDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    page_table = _as_tt(
        mesh_device,
        _page_table(layer_type, shared_physical=shared_physical),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cache_shape = _cache_shape(layer_type, shared_physical=shared_physical)
    kv_cache = (
        _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
        _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
    )

    tt_prefill = decoder.prefill_forward(
        _as_tt(mesh_device, hidden.unsqueeze(1)),
        position_cos=_as_tt(mesh_device, cos.unsqueeze(1)),
        position_sin=_as_tt(mesh_device, sin.unsqueeze(1)),
        page_table=page_table,
        kv_cache=kv_cache,
    )
    tt_decode = decoder.decode_forward(
        _as_tt(mesh_device, decode_hidden.unsqueeze(1)),
        position_cos=_as_tt(mesh_device, decode_cos.unsqueeze(1)),
        position_sin=_as_tt(mesh_device, decode_sin.unsqueeze(1)),
        current_pos=_as_tt(
            mesh_device, torch.tensor([seq_len], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        ),
        page_table=page_table,
        kv_cache=kv_cache,
    )
    ttnn.synchronize_device(mesh_device)

    tt_prefill_torch = _to_torch(mesh_device, tt_prefill).reshape(1, seq_len, HIDDEN_SIZE).to(torch.bfloat16)
    tt_decode_torch = _to_torch(mesh_device, tt_decode).reshape(1, 1, HIDDEN_SIZE).to(torch.bfloat16)
    prefill_ok, prefill_pcc = comp_pcc(ref_prefill, tt_prefill_torch, 0.995)
    decode_ok, actual_decode_pcc = comp_pcc(ref_decode, tt_decode_torch, decode_pcc)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / f"pcc_layer{layer_idx}_{layer_type}_shared{int(shared_physical)}.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "shared_physical_cache": shared_physical,
                "sequence_length": seq_len,
                "decode_current_pos": seq_len,
                "prefill_pcc": float(prefill_pcc),
                "prefill_threshold": 0.995,
                "decode_pcc": float(actual_decode_pcc),
                "decode_threshold": decode_pcc,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    assert prefill_ok, prefill_pcc
    assert decode_ok, actual_decode_pcc


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_functional_decoder_real_shape_batch2_prefill(mesh_device, device_params, layer_idx):
    """Validate the TTNN-only multi-user prefill wrapper for both layer kinds."""
    cfg = _load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    state = _load_layer_state(layer_idx)
    batch = 2
    seq_len = 32
    torch.manual_seed(2400 + layer_idx)
    hidden = torch.randn(batch, seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
    rotary = Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rotary(hidden, position_ids, layer_type=layer_type)

    prefix = f"model.language_model.layers.{layer_idx}"
    reference_layer = Gemma4TextDecoderLayer(cfg, layer_idx=layer_idx).eval().to(dtype=torch.bfloat16)
    reference_layer.load_state_dict({k[len(prefix) + 1 :]: v for k, v in state.items()}, strict=True)
    with torch.no_grad():
        reference = reference_layer(
            hidden,
            shared_kv_states={},
            position_embeddings=(cos, sin),
            attention_mask=_causal_mask(
                seq_len,
                sliding_window=cfg.sliding_window if layer_type == "sliding_attention" else None,
            ).expand(batch, -1, -1, -1),
            position_ids=position_ids,
            past_key_values=DynamicCache(config=cfg),
        )

    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
    )
    one_user_cache_shape = _cache_shape(layer_type, shared_physical=False)
    blocks_per_user = one_user_cache_shape[0]
    cache_shape = (batch * blocks_per_user, *one_user_cache_shape[1:])
    page_table = _as_tt(
        mesh_device,
        torch.arange(batch * blocks_per_user, dtype=torch.int32).view(batch, blocks_per_user),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    kv_cache = (
        _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
        _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
    )
    tt_output = decoder.prefill_forward(
        _as_tt(mesh_device, hidden.unsqueeze(1)),
        position_cos=_as_tt(mesh_device, cos.unsqueeze(1)),
        position_sin=_as_tt(mesh_device, sin.unsqueeze(1)),
        page_table=page_table,
        kv_cache=kv_cache,
    )
    ttnn.synchronize_device(mesh_device)
    actual = _to_torch(mesh_device, tt_output).reshape(batch, seq_len, HIDDEN_SIZE).to(torch.bfloat16)
    passing, pcc = comp_pcc(reference, actual, 0.995)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / f"prefill_batch2_layer{layer_idx}_{layer_type}.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "batch": batch,
                "sequence_length": seq_len,
                "real_shape": [batch, seq_len, HIDDEN_SIZE],
                "hf_vs_ttnn_pcc": float(pcc),
                "threshold": 0.995,
                "page_table_shape": [batch, blocks_per_user],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    assert passing, pcc


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
def test_functional_decoder_decode_trace_replay(mesh_device, device_params):
    cfg = _load_text_config()
    layer_idx = 0
    layer_type = cfg.layer_types[layer_idx]
    state = _load_layer_state(layer_idx)
    seq_len = 32
    torch.manual_seed(7)
    hidden = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
    decode_hidden = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
    rotary = Gemma4TextRotaryEmbedding(cfg)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    decode_position_ids = torch.tensor([[seq_len]])
    cos, sin = rotary(hidden, position_ids, layer_type=layer_type)
    decode_cos, decode_sin = rotary(decode_hidden, decode_position_ids, layer_type=layer_type)

    prefix = f"model.language_model.layers.{layer_idx}"
    ref = Gemma4TextDecoderLayer(cfg, layer_idx=layer_idx).eval().to(dtype=torch.bfloat16)
    ref.load_state_dict({k[len(prefix) + 1 :]: v for k, v in state.items()}, strict=True)
    cache = DynamicCache(config=cfg)
    with torch.no_grad():
        ref(
            hidden,
            shared_kv_states={},
            position_embeddings=(cos, sin),
            attention_mask=_causal_mask(seq_len, sliding_window=cfg.sliding_window),
            position_ids=position_ids,
            past_key_values=cache,
        )
        ref_decode = ref(
            decode_hidden,
            shared_kv_states={},
            position_embeddings=(decode_cos, decode_sin),
            attention_mask=_decode_mask(seq_len + 1, sliding_window=cfg.sliding_window),
            position_ids=decode_position_ids,
            past_key_values=cache,
        )

    decoder = FunctionalDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    page_table = _as_tt(
        mesh_device, torch.tensor([[0, 1, 2, 3]], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    cache_shape = (4, SLIDING_NUM_KV_HEADS, SLIDING_BLOCK_SIZE, SLIDING_HEAD_DIM)
    kv_cache = (
        _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
        _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
    )
    decoder.prefill_forward(
        _as_tt(mesh_device, hidden.unsqueeze(1)),
        position_cos=_as_tt(mesh_device, cos.unsqueeze(1)),
        position_sin=_as_tt(mesh_device, sin.unsqueeze(1)),
        page_table=page_table,
        kv_cache=kv_cache,
    )
    decode_args = dict(
        hidden_states=_as_tt(mesh_device, decode_hidden.unsqueeze(1)),
        position_cos=_as_tt(mesh_device, decode_cos.unsqueeze(1)),
        position_sin=_as_tt(mesh_device, decode_sin.unsqueeze(1)),
        current_pos=_as_tt(
            mesh_device, torch.tensor([seq_len], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        ),
        page_table=page_table,
        kv_cache=kv_cache,
    )
    eager_output = decoder.decode_forward(**decode_args)
    ttnn.synchronize_device(mesh_device)
    eager_torch = _to_torch(mesh_device, eager_output).reshape(1, 1, HIDDEN_SIZE).to(torch.bfloat16)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(**decode_args)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    replay_torch = _to_torch(mesh_device, traced_output).reshape(1, 1, HIDDEN_SIZE).to(torch.bfloat16)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    replay_torch_2 = _to_torch(mesh_device, traced_output).reshape(1, 1, HIDDEN_SIZE).to(torch.bfloat16)
    ttnn.release_trace(mesh_device, trace_id)
    ref_replay_ok, ref_replay_pcc = comp_pcc(ref_decode, replay_torch, 0.995)
    eager_replay_ok, eager_replay_pcc = comp_pcc(eager_torch, replay_torch, 0.9999)
    repeat_ok, repeat_pcc = comp_pcc(replay_torch, replay_torch_2, 0.9999)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "trace_replay_pcc.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "sequence_length": seq_len,
                "decode_current_pos": seq_len,
                "hf_vs_trace_replay_pcc": float(ref_replay_pcc),
                "hf_vs_trace_replay_threshold": 0.995,
                "eager_vs_trace_replay_pcc": float(eager_replay_pcc),
                "eager_vs_trace_replay_threshold": 0.9999,
                "trace_replay_repeat_pcc": float(repeat_pcc),
                "trace_replay_repeat_threshold": 0.9999,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    assert ref_replay_ok, ref_replay_pcc
    assert eager_replay_ok, eager_replay_pcc
    assert repeat_ok, repeat_pcc


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
def test_traced_decode_batch_contract(mesh_device, device_params, layer_idx, batch):
    """Trace/replay paged decode at the functional and serving batches."""
    cfg = _load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    state = _load_layer_state(layer_idx)
    current_position = 32
    if layer_type == "full_attention":
        torch.manual_seed(layer_idx)
        full_prefix = torch.randn(1, current_position, HIDDEN_SIZE, dtype=torch.bfloat16)
        full_decode = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
        prefix_hidden_for_full = full_prefix.expand(batch, -1, -1).clone()
        hidden = full_decode.expand(batch, -1, -1).clone()
    else:
        torch.manual_seed(2000 + layer_idx + batch)
        prefix_hidden_for_full = None
        hidden = torch.randn(batch, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
    positions = torch.full((batch, 1), current_position, dtype=torch.long)
    rotary = Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rotary(hidden, positions, layer_type=layer_type)
    if layer_type == "full_attention":
        num_kv_heads, block_size, head_dim = FULL_NUM_KV_HEADS, FULL_BLOCK_SIZE, FULL_HEAD_DIM
    else:
        num_kv_heads, block_size, head_dim = (
            SLIDING_NUM_KV_HEADS,
            SLIDING_BLOCK_SIZE,
            SLIDING_HEAD_DIM,
        )

    prefix = f"model.language_model.layers.{layer_idx}"
    reference_layer = Gemma4TextDecoderLayer(cfg, layer_idx=layer_idx).eval().to(dtype=torch.bfloat16)
    reference_layer.load_state_dict({k[len(prefix) + 1 :]: v for k, v in state.items()}, strict=True)
    reference_cache = DynamicCache(config=cfg)
    prefix_hidden = None
    prefix_cos = None
    prefix_sin = None
    if layer_type == "full_attention":
        prefix_hidden = prefix_hidden_for_full
        prefix_positions = torch.arange(current_position).view(1, -1).expand(batch, -1)
        prefix_cos, prefix_sin = rotary(
            prefix_hidden,
            prefix_positions,
            layer_type=layer_type,
        )
        with torch.no_grad():
            reference_layer(
                prefix_hidden,
                shared_kv_states={},
                position_embeddings=(prefix_cos, prefix_sin),
                attention_mask=_causal_mask(current_position, sliding_window=None).expand(batch, -1, -1, -1),
                position_ids=prefix_positions,
                past_key_values=reference_cache,
            )
    else:
        reference_cache.update(
            torch.zeros(batch, num_kv_heads, current_position, head_dim, dtype=torch.bfloat16),
            torch.zeros(batch, num_kv_heads, current_position, head_dim, dtype=torch.bfloat16),
            layer_idx=layer_idx,
        )
    with torch.no_grad():
        reference = reference_layer(
            hidden,
            shared_kv_states={},
            position_embeddings=(cos, sin),
            attention_mask=_decode_mask(
                current_position + 1,
                sliding_window=cfg.sliding_window if layer_type == "sliding_attention" else None,
            ).expand(batch, -1, -1, -1),
            position_ids=positions,
            past_key_values=reference_cache,
        )

    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
    )
    blocks_per_user = 2 if layer_type == "full_attention" else 4
    page_table = _as_tt(
        mesh_device,
        torch.arange(batch * blocks_per_user, dtype=torch.int32).view(batch, blocks_per_user),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cache_shape = (batch * blocks_per_user, num_kv_heads, block_size, head_dim)
    kv_cache = (
        _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
        _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
    )
    if layer_type == "full_attention":
        for user_id in range(batch):
            decoder.prefill_forward(
                _as_tt(mesh_device, prefix_hidden[user_id : user_id + 1].unsqueeze(1)),
                position_cos=_as_tt(mesh_device, prefix_cos[user_id : user_id + 1].unsqueeze(1)),
                position_sin=_as_tt(mesh_device, prefix_sin[user_id : user_id + 1].unsqueeze(1)),
                page_table=page_table,
                kv_cache=kv_cache,
                user_id=user_id,
            )
    if layer_type == "sliding_attention":
        tt_cos = cos.unsqueeze(0)
        tt_sin = sin.unsqueeze(0)
    else:
        tt_cos = cos.transpose(0, 1).unsqueeze(0)
        tt_sin = sin.transpose(0, 1).unsqueeze(0)
    decode_args = {
        "hidden_states": _as_tt(mesh_device, hidden.transpose(0, 1).unsqueeze(0)),
        "position_cos": _as_tt(mesh_device, tt_cos),
        "position_sin": _as_tt(mesh_device, tt_sin),
        "current_pos": _as_tt(
            mesh_device,
            torch.full((batch,), current_position, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "page_table": page_table,
        "kv_cache": kv_cache,
    }

    eager_output = decoder.decode_forward(**decode_args)
    ttnn.synchronize_device(mesh_device)
    eager = _to_torch(mesh_device, eager_output).reshape(1, batch, HIDDEN_SIZE).transpose(0, 1).to(torch.bfloat16)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(**decode_args)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    replay = _to_torch(mesh_device, traced_output).reshape(1, batch, HIDDEN_SIZE).transpose(0, 1).to(torch.bfloat16)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    replay_repeat = (
        _to_torch(mesh_device, traced_output).reshape(1, batch, HIDDEN_SIZE).transpose(0, 1).to(torch.bfloat16)
    )
    ttnn.release_trace(mesh_device, trace_id)

    hf_ok, hf_pcc = comp_pcc(reference, replay, 0.995)
    eager_ok, eager_pcc = comp_pcc(eager, replay, 0.9999)
    repeat_ok, repeat_pcc = comp_pcc(replay, replay_repeat, 0.9999)
    if not hf_ok:
        reference_flat = torch.nn.functional.normalize(reference.float().reshape(batch, -1), dim=-1)
        replay_flat = torch.nn.functional.normalize(replay.float().reshape(batch, -1), dim=-1)
        user_correlation = reference_flat @ replay_flat.T
        print("TRACE_BATCH_CORRELATION_DIAGONAL", user_correlation.diag().tolist())
        print("TRACE_BATCH_CORRELATION_BEST_REPLAY_USER", user_correlation.argmax(dim=1).tolist())
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / f"trace_{layer_type}_batch{batch}.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "batch": batch,
                "current_positions": [current_position] * batch,
                "page_table_shape": [batch, blocks_per_user],
                "hf_vs_trace_replay_pcc": float(hf_pcc),
                "hf_vs_trace_replay_threshold": 0.995,
                "eager_vs_trace_replay_pcc": float(eager_pcc),
                "eager_vs_trace_replay_threshold": 0.9999,
                "repeat_replay_pcc": float(repeat_pcc),
                "repeat_replay_threshold": 0.9999,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    assert hf_ok, hf_pcc
    assert eager_ok, eager_pcc
    assert repeat_ok, repeat_pcc


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_paged_prefill_logical_boundary_lengths(mesh_device, device_params, layer_idx):
    cfg = _load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    state = _load_layer_state(layer_idx)
    if layer_type == "sliding_attention":
        default_lengths = [1, 31, 32, 33, 63, 64, 65, 1023, 1024, 1025]
        num_kv_heads, block_size, head_dim = (
            SLIDING_NUM_KV_HEADS,
            SLIDING_BLOCK_SIZE,
            SLIDING_HEAD_DIM,
        )
    else:
        default_lengths = [1, 31, 32, 33, 127, 128, 129, 1023, 1024, 1025]
        num_kv_heads, block_size, head_dim = FULL_NUM_KV_HEADS, FULL_BLOCK_SIZE, FULL_HEAD_DIM
    override = os.getenv("GEMMA4_BOUNDARY_LENGTHS")
    lengths = [int(value) for value in override.split(",")] if override else default_lengths

    prefix = f"model.language_model.layers.{layer_idx}"
    reference_layer = Gemma4TextDecoderLayer(cfg, layer_idx=layer_idx).eval().to(dtype=torch.bfloat16)
    reference_layer.load_state_dict({k[len(prefix) + 1 :]: v for k, v in state.items()}, strict=True)
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
    )
    results = []
    for seq_len in lengths:
        torch.manual_seed(3000 + layer_idx + seq_len)
        hidden = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
        positions = torch.arange(seq_len).unsqueeze(0)
        rotary = Gemma4TextRotaryEmbedding(cfg)
        cos, sin = rotary(hidden, positions, layer_type=layer_type)
        with torch.no_grad():
            reference = reference_layer(
                hidden,
                shared_kv_states={},
                position_embeddings=(cos, sin),
                attention_mask=_causal_mask(
                    seq_len,
                    sliding_window=cfg.sliding_window if layer_type == "sliding_attention" else None,
                ),
                position_ids=positions,
            )

        padded_len = ((seq_len + 31) // 32) * 32
        num_blocks = (padded_len + block_size - 1) // block_size
        physical_order = torch.roll(torch.arange(num_blocks, dtype=torch.int32), shifts=1)
        page_table = _as_tt(
            mesh_device,
            physical_order.view(1, num_blocks),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        cache_shape = (num_blocks, num_kv_heads, block_size, head_dim)
        kv_cache = (
            _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
            _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
        )
        actual_tt = decoder.prefill_forward(
            _as_tt(mesh_device, hidden.unsqueeze(1)),
            position_cos=_as_tt(mesh_device, cos.unsqueeze(1)),
            position_sin=_as_tt(mesh_device, sin.unsqueeze(1)),
            page_table=page_table,
            kv_cache=kv_cache,
        )
        actual = _to_torch(mesh_device, actual_tt).reshape(1, seq_len, HIDDEN_SIZE).to(torch.bfloat16)
        passed, pcc = comp_pcc(reference, actual, 0.995)
        results.append(
            {
                "logical_sequence_length": seq_len,
                "physical_sequence_length": padded_len,
                "page_table": physical_order.tolist(),
                "pcc": float(pcc),
                "threshold": 0.995,
            }
        )
        assert actual_tt.shape[-2] == seq_len
        assert passed, pcc
        ttnn.deallocate(actual_tt)
        ttnn.deallocate(kv_cache[0])
        ttnn.deallocate(kv_cache[1])

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / f"prefill_boundaries_{layer_type}.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_advertised_context_traced_decode(mesh_device, device_params, layer_idx):
    """Exercise traced batch-1 decode at the final HF-advertised position.

    The cache is intentionally allocated directly on device: constructing
    multi-gigabyte host zero tensors would test host RAM and PCIe transfer,
    rather than the decoder's physical device-capacity contract.
    """
    if os.getenv("GEMMA4_FUNCTIONAL_DECODER_CONTEXT") != "1":
        pytest.skip("set GEMMA4_FUNCTIONAL_DECODER_CONTEXT=1 to run the advertised-context capacity test")

    cfg = _load_text_config()
    advertised_context = int(cfg.max_position_embeddings)
    current_position = advertised_context - 1
    layer_type = cfg.layer_types[layer_idx]
    state = _load_layer_state(layer_idx)
    if layer_type == "sliding_attention":
        num_kv_heads, block_size, head_dim = (
            SLIDING_NUM_KV_HEADS,
            SLIDING_BLOCK_SIZE,
            SLIDING_HEAD_DIM,
        )
    else:
        num_kv_heads, block_size, head_dim = FULL_NUM_KV_HEADS, FULL_BLOCK_SIZE, FULL_HEAD_DIM

    num_blocks = (advertised_context + block_size - 1) // block_size
    cache_shape = (num_blocks, num_kv_heads, block_size, head_dim)
    bytes_per_cache = num_blocks * num_kv_heads * block_size * head_dim * 2
    physical_order = torch.roll(torch.arange(num_blocks, dtype=torch.int32), shifts=7)
    page_table = _as_tt(
        mesh_device,
        physical_order.view(1, num_blocks),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    kv_cache = (
        ttnn.full(
            cache_shape,
            0.0078125,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        ttnn.full(
            cache_shape,
            -0.015625,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    )

    torch.manual_seed(9000 + layer_idx)
    hidden = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
    rotary = Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rotary(
        hidden,
        torch.tensor([[current_position]], dtype=torch.long),
        layer_type=layer_type,
    )
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
    )
    sentinel_positions = [0, current_position // 2, current_position - 1]
    sentinel_values = [(0.125, 4.0), (-0.25, -8.0), (0.5, 16.0)]
    update_mem_config = decoder_module._make_single_user_cache_update_memory_config(mesh_device, head_dim)
    update_kwargs = decoder._cache_view_kwargs(prefill=False)
    for sentinel_position, (key_value, value_value) in zip(sentinel_positions, sentinel_values):
        position_tensor = _as_tt(
            mesh_device,
            torch.tensor([sentinel_position], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        key_token = ttnn.to_memory_config(
            _as_tt(mesh_device, torch.full((1, 1, num_kv_heads, head_dim), key_value, dtype=torch.bfloat16)),
            update_mem_config,
        )
        value_token = ttnn.to_memory_config(
            _as_tt(mesh_device, torch.full((1, 1, num_kv_heads, head_dim), value_value, dtype=torch.bfloat16)),
            update_mem_config,
        )
        ttnn.experimental.paged_update_cache(
            kv_cache[0],
            key_token,
            update_idxs_tensor=position_tensor,
            page_table=page_table,
            **update_kwargs,
        )
        ttnn.experimental.paged_update_cache(
            kv_cache[1],
            value_token,
            update_idxs_tensor=position_tensor,
            page_table=page_table,
            **update_kwargs,
        )

    def read_logical_cache_row(cache_tensor, logical_position):
        physical_block = int(physical_order[logical_position // block_size].item())
        physical_cache_block = _to_torch(
            mesh_device,
            ttnn.slice(
                cache_tensor,
                starts=[physical_block, 0, 0, 0],
                ends=[physical_block + 1, num_kv_heads, block_size, head_dim],
                steps=[1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        )
        return physical_cache_block[0, :, logical_position % block_size, :]

    expected_rows = {}
    for sentinel_position, (key_value, value_value) in zip(sentinel_positions, sentinel_values):
        key_row = read_logical_cache_row(kv_cache[0], sentinel_position)
        value_row = read_logical_cache_row(kv_cache[1], sentinel_position)
        torch.testing.assert_close(
            key_row,
            torch.full_like(key_row, key_value),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            value_row,
            torch.full_like(value_row, value_value),
            rtol=0,
            atol=0,
        )
        expected_rows[str(sentinel_position)] = {
            "key_mean": float(key_row.float().mean()),
            "value_mean": float(value_row.float().mean()),
        }
    baseline_position = current_position - 2
    baseline_key = read_logical_cache_row(kv_cache[0], baseline_position)
    baseline_value = read_logical_cache_row(kv_cache[1], baseline_position)
    torch.testing.assert_close(baseline_key, torch.full_like(baseline_key, 0.0078125), rtol=0, atol=0)
    torch.testing.assert_close(baseline_value, torch.full_like(baseline_value, -0.015625), rtol=0, atol=0)
    decode_args = dict(
        hidden_states=_as_tt(mesh_device, hidden.unsqueeze(1)),
        position_cos=_as_tt(mesh_device, cos.unsqueeze(1)),
        position_sin=_as_tt(mesh_device, sin.unsqueeze(1)),
        current_pos=_as_tt(
            mesh_device,
            torch.tensor([current_position], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        page_table=page_table,
        kv_cache=kv_cache,
    )

    decoder.decode_forward(**decode_args)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(**decode_args)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    first = _to_torch(mesh_device, traced_output)
    start = time.perf_counter()
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    warmed_trace_ms = (time.perf_counter() - start) * 1000
    second = _to_torch(mesh_device, traced_output)
    ttnn.release_trace(mesh_device, trace_id)
    preserved_key = read_logical_cache_row(kv_cache[0], current_position - 1)
    preserved_value = read_logical_cache_row(kv_cache[1], current_position - 1)
    torch.testing.assert_close(preserved_key, torch.full_like(preserved_key, 0.5), rtol=0, atol=0)
    torch.testing.assert_close(preserved_value, torch.full_like(preserved_value, 16.0), rtol=0, atol=0)

    repeat_ok, repeat_pcc = comp_pcc(first, second, 0.9999)
    assert torch.isfinite(second).all()
    assert repeat_ok, repeat_pcc
    try:
        dram_bytes = int(mesh_device.num_dram_channels()) * int(mesh_device.dram_size_per_channel())
    except (AttributeError, TypeError):
        dram_bytes = None
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / f"advertised_context_decode_{layer_type}.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "advertised_context": advertised_context,
                "decode_current_position": current_position,
                "batch": 1,
                "page_table_permutation": "roll_by_7",
                "page_table_shape": [1, num_blocks],
                "cache_shape_each": cache_shape,
                "cache_bytes_each": bytes_per_cache,
                "cache_bytes_total": 2 * bytes_per_cache,
                "device_dram_bytes": dram_bytes,
                "history_initialization": {
                    "key_baseline": 0.0078125,
                    "value_baseline": -0.015625,
                    "sentinel_positions": sentinel_positions,
                    "sentinel_key_values": [pair[0] for pair in sentinel_values],
                    "sentinel_values": [pair[1] for pair in sentinel_values],
                    "baseline_position": baseline_position,
                    "device_readback_verified": True,
                    "observed_sentinel_rows": expected_rows,
                    "preserved_after_trace": current_position - 1,
                },
                "traced_warmed_host_ms": warmed_trace_ms,
                "finite_output": True,
                "repeat_replay_pcc": float(repeat_pcc),
                "repeat_replay_threshold": 0.9999,
                "provenance": _evidence_provenance(
                    mesh_device,
                    "GEMMA4_FUNCTIONAL_DECODER_CONTEXT=1 GEMMA4_RANGE_DOWNLOAD=1 "
                    "pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/"
                    "test_functional_decoder.py::test_advertised_context_traced_decode",
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_prefill_capacity_probe(mesh_device, device_params, layer_idx):
    """Run one real-weight prefill length for serialized capacity probing."""
    requested = os.getenv("GEMMA4_PREFILL_CAPACITY_LENGTH")
    if requested is None:
        pytest.skip("set GEMMA4_PREFILL_CAPACITY_LENGTH to run a physical prefill-capacity probe")
    seq_len = int(requested)
    if seq_len < 1:
        raise ValueError("capacity probe length must be positive")

    cfg = _load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    state = _load_layer_state(layer_idx)
    if layer_type == "sliding_attention":
        num_kv_heads, block_size, head_dim = (
            SLIDING_NUM_KV_HEADS,
            SLIDING_BLOCK_SIZE,
            SLIDING_HEAD_DIM,
        )
    else:
        num_kv_heads, block_size, head_dim = FULL_NUM_KV_HEADS, FULL_BLOCK_SIZE, FULL_HEAD_DIM

    torch.manual_seed(10000 + layer_idx + seq_len)
    hidden = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
    positions = torch.arange(seq_len).unsqueeze(0)
    rotary = Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rotary(hidden, positions, layer_type=layer_type)
    num_blocks = (seq_len + block_size - 1) // block_size
    page_table = _as_tt(
        mesh_device,
        torch.roll(torch.arange(num_blocks, dtype=torch.int32), shifts=3).view(1, num_blocks),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cache_shape = (num_blocks, num_kv_heads, block_size, head_dim)
    kv_cache = (
        ttnn.zeros(
            cache_shape,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        ttnn.zeros(
            cache_shape,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    )
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
    )
    start = time.perf_counter()
    output = decoder.prefill_forward(
        _as_tt(mesh_device, hidden.unsqueeze(1)),
        position_cos=_as_tt(mesh_device, cos.unsqueeze(1)),
        position_sin=_as_tt(mesh_device, sin.unsqueeze(1)),
        page_table=page_table,
        kv_cache=kv_cache,
    )
    ttnn.synchronize_device(mesh_device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    sample = _to_torch(
        mesh_device,
        ttnn.slice(
            output,
            starts=[0, 0, seq_len - 1, 0],
            ends=[1, 1, seq_len, HIDDEN_SIZE],
            steps=[1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    )
    assert output.shape[-2] == seq_len
    assert torch.isfinite(sample).all()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / f"prefill_capacity_{layer_type}_{seq_len}.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "sequence_length": seq_len,
                "real_weights": True,
                "page_table_permutation": "roll_by_3",
                "cache_shape_each": cache_shape,
                "finite_last_token": True,
                "host_elapsed_ms": elapsed_ms,
                "provenance": _evidence_provenance(
                    mesh_device,
                    f"GEMMA4_PREFILL_CAPACITY_LENGTH={seq_len} GEMMA4_RANGE_DOWNLOAD=1 "
                    "pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/"
                    "test_functional_decoder.py::test_prefill_capacity_probe",
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_bounded_modulo_prefill_tail_cache_integrity(mesh_device, device_params):
    """A 1025-token prefill may replace slot 0, but must preserve slots 1..1023."""
    cfg = _load_text_config()
    layer_idx = 0
    state = _load_layer_state(layer_idx)
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
    )
    torch.manual_seed(4217)
    hidden_long = torch.randn(1, 1025, HIDDEN_SIZE, dtype=torch.bfloat16)
    rotary = Gemma4TextRotaryEmbedding(cfg)
    cos_long, sin_long = rotary(
        hidden_long,
        torch.arange(1025).unsqueeze(0),
        layer_type="sliding_attention",
    )
    page_table_host = torch.roll(torch.arange(16, dtype=torch.int32), shifts=5).view(1, 16)
    page_table = _as_tt(
        mesh_device,
        page_table_host,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    def run_attention_only(logical_len):
        padded_len = ((logical_len + 31) // 32) * 32
        hidden = hidden_long[:, :logical_len]
        cos = cos_long[:, :logical_len]
        sin = sin_long[:, :logical_len]
        if padded_len != logical_len:
            hidden = torch.nn.functional.pad(hidden, (0, 0, 0, padded_len - logical_len))
            cos = torch.nn.functional.pad(cos, (0, 0, 0, padded_len - logical_len))
            sin = torch.nn.functional.pad(sin, (0, 0, 0, padded_len - logical_len))
        cache_shape = (16, SLIDING_NUM_KV_HEADS, SLIDING_BLOCK_SIZE, SLIDING_HEAD_DIM)
        cache = (
            _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
            _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
        )
        x = _as_tt(mesh_device, hidden.unsqueeze(1))
        x = decoder._rms_norm(x, decoder.weights.input_ln)
        attention_output = decoder._attention_prefill(
            x,
            position_cos=_as_tt(mesh_device, cos.unsqueeze(1)),
            position_sin=_as_tt(mesh_device, sin.unsqueeze(1)),
            page_table=page_table,
            chunk_page_table=None,
            kv_cache=cache,
            user_id=0,
            cache_position_modulo=1024,
            logical_seq_len=logical_len,
        )
        attention_output.deallocate(True)
        return tuple(_to_torch(mesh_device, tensor) for tensor in cache)

    baseline_cache = run_attention_only(1024)
    wrapped_cache = run_attention_only(1025)

    def logical_order(cache):
        physical_blocks = page_table_host.flatten().to(torch.long)
        return (
            cache[physical_blocks]
            .permute(1, 0, 2, 3)
            .reshape(
                SLIDING_NUM_KV_HEADS,
                1024,
                SLIDING_HEAD_DIM,
            )
        )

    for baseline, wrapped in zip(baseline_cache, wrapped_cache):
        baseline = logical_order(baseline)
        wrapped = logical_order(wrapped)
        preserved, preserved_pcc = comp_pcc(baseline[:, 1:], wrapped[:, 1:], 0.9999)
        assert preserved, preserved_pcc
        assert not torch.equal(baseline[:, 0], wrapped[:, 0])


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize(
    "layer_kind",
    [pytest.param(SLIDING_KIND, id="sliding"), pytest.param(FULL_KIND, id="full")],
)
def test_long_prefill_attention_correctness(mesh_device, device_params, layer_kind):
    """Validate selected rows beyond the real 32768-token SDPA cliff."""
    if os.getenv("GEMMA4_LONG_ATTN_TEST") != "1":
        pytest.skip("set GEMMA4_LONG_ATTN_TEST=1 for the >32768 attention-only regression")
    seq_len = PREFILL_SDPA_MAX_SEQ + 32
    torch.manual_seed(8900 if layer_kind.name == "sliding_attention" else 8905)
    q = torch.randn(1, NUM_Q_HEADS, seq_len, layer_kind.head_dim, dtype=torch.bfloat16)
    k = torch.randn(1, layer_kind.num_kv_heads, seq_len, layer_kind.head_dim, dtype=torch.bfloat16)
    v = torch.randn(1, layer_kind.num_kv_heads, seq_len, layer_kind.head_dim, dtype=torch.bfloat16)
    q_tt, k_tt, v_tt = (_as_tt(mesh_device, tensor) for tensor in (q, k, v))

    decoder = object.__new__(FunctionalDecoder)
    decoder.layer_kind = layer_kind
    decoder.mesh_device = mesh_device
    decoder.correctness_compute_config = _make_correctness_compute_config(mesh_device)
    if layer_kind.name == "sliding_attention":
        output = decoder._sliding_chunked_prefill_attention(q_tt, k_tt, v_tt)
    else:
        num_blocks = (seq_len + layer_kind.block_size - 1) // layer_kind.block_size
        page_table_host = torch.roll(torch.arange(num_blocks, dtype=torch.int32), shifts=11).view(1, -1)
        page_table = _as_tt(
            mesh_device,
            page_table_host,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        cache_shape = (num_blocks, layer_kind.num_kv_heads, layer_kind.block_size, layer_kind.head_dim)
        cache = (
            _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
            _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
        )
        ttnn.experimental.paged_fill_cache(cache[0], k_tt, page_table, batch_idx=0, block_size=layer_kind.block_size)
        ttnn.experimental.paged_fill_cache(cache[1], v_tt, page_table, batch_idx=0, block_size=layer_kind.block_size)
        output = decoder._full_chunked_prefill_attention(
            q_tt,
            cache[0],
            cache[1],
            page_table,
            user_id=0,
        )

    check_positions = [PREFILL_SDPA_MAX_SEQ - 1, PREFILL_SDPA_MAX_SEQ, seq_len - 1]
    results = []
    for position in check_positions:
        got = _to_torch(
            mesh_device,
            ttnn.slice(
                output,
                starts=[0, 0, position, 0],
                ends=[1, NUM_Q_HEADS, position + 1, layer_kind.head_dim],
                steps=[1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        ).float()
        first_key = max(0, position - layer_kind.sliding_window + 1) if layer_kind.sliding_window is not None else 0
        groups = NUM_Q_HEADS // layer_kind.num_kv_heads
        q_ref = (
            q[:, :, position : position + 1]
            .float()
            .reshape(
                1,
                layer_kind.num_kv_heads,
                groups,
                1,
                layer_kind.head_dim,
            )
        )
        k_ref = k[:, :, first_key : position + 1].float()
        v_ref = v[:, :, first_key : position + 1].float()
        scores = torch.einsum("bhgqd,bhkd->bhgqk", q_ref, k_ref)
        reference = torch.einsum(
            "bhgqk,bhkd->bhgqd",
            torch.softmax(scores, dim=-1),
            v_ref,
        ).reshape(1, NUM_Q_HEADS, 1, layer_kind.head_dim)
        passing, pcc = comp_pcc(reference, got, 0.995)
        results.append({"position": position, "pcc": float(pcc), "threshold": 0.995})
        assert passing, f"position={position}: {pcc}"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / f"long_prefill_attention_{layer_kind.name}.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "layer_type": layer_kind.name,
                "sequence_length": seq_len,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,shared_physical",
    [
        pytest.param(0, True, id="sliding_attention_1024"),
        pytest.param(5, False, id="full_attention_1024"),
    ],
)
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
def test_functional_decoder_perf_profile(mesh_device, device_params, layer_idx, shared_physical, batch):
    if os.getenv("GEMMA4_FUNCTIONAL_DECODER_PERF") != "1":
        pytest.skip("set GEMMA4_FUNCTIONAL_DECODER_PERF=1 to run the profiler harness")

    try:
        from tracy import signpost
    except ImportError:
        signpost = lambda *_, **__: None

    cfg = _load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    state = _load_layer_state(layer_idx)
    seq_len = int(os.getenv("GEMMA4_FUNCTIONAL_DECODER_SEQ_LEN", "1024"))
    if seq_len % 32 != 0:
        raise ValueError(f"sequence length must be tile-aligned, got {seq_len}")
    token_capacity = seq_len + 1
    torch.manual_seed(1000 + layer_idx)
    hidden = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
    decode_hidden = torch.randn(batch, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
    rotary = Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rotary(hidden, torch.arange(seq_len).unsqueeze(0), layer_type=layer_type)
    decode_positions = torch.full((batch, 1), seq_len, dtype=torch.long)
    decode_cos, decode_sin = rotary(decode_hidden, decode_positions, layer_type=layer_type)

    decoder = FunctionalDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    blocks_per_user = _cache_shape(
        layer_type,
        shared_physical=shared_physical,
        token_capacity=token_capacity,
    )[0]
    page_table = _as_tt(
        mesh_device,
        torch.arange(batch * blocks_per_user, dtype=torch.int32).view(batch, blocks_per_user),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    one_user_cache_shape = _cache_shape(
        layer_type,
        shared_physical=shared_physical,
        token_capacity=token_capacity,
    )
    cache_shape = (batch * one_user_cache_shape[0], *one_user_cache_shape[1:])
    kv_cache = (
        _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
        _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
    )
    prefill_args = dict(
        hidden_states=_as_tt(mesh_device, hidden.unsqueeze(1)),
        position_cos=_as_tt(mesh_device, cos.unsqueeze(1)),
        position_sin=_as_tt(mesh_device, sin.unsqueeze(1)),
        page_table=page_table,
        kv_cache=kv_cache,
    )
    if layer_type == "sliding_attention":
        tt_decode_cos = decode_cos.unsqueeze(0)
        tt_decode_sin = decode_sin.unsqueeze(0)
    else:
        tt_decode_cos = decode_cos.transpose(0, 1).unsqueeze(0)
        tt_decode_sin = decode_sin.transpose(0, 1).unsqueeze(0)
    decode_args = dict(
        hidden_states=_as_tt(mesh_device, decode_hidden.transpose(0, 1).unsqueeze(0)),
        position_cos=_as_tt(mesh_device, tt_decode_cos),
        position_sin=_as_tt(mesh_device, tt_decode_sin),
        current_pos=_as_tt(
            mesh_device,
            torch.full((batch,), seq_len, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        page_table=page_table,
        kv_cache=kv_cache,
    )

    decoder.prefill_forward(**prefill_args)
    decoder.decode_forward(**decode_args)
    ttnn.synchronize_device(mesh_device)

    measured = {}
    case_id = f"layer{layer_idx}_{layer_type}_seq{seq_len}_batch{batch}"
    if batch == 1:
        signpost(f"PERF_PREFILL_{case_id}", f"cache_shape={cache_shape}")
        start = time.perf_counter()
        prefill_output = decoder.prefill_forward(**prefill_args)
        ttnn.synchronize_device(mesh_device)
        measured["prefill_host_ms"] = (time.perf_counter() - start) * 1000
        signpost(f"PERF_PREFILL_{case_id}_END", f"cache_shape={cache_shape}")

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_decode_output = decoder.decode_forward(**decode_args)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)

    signpost(f"PERF_DECODE_{case_id}", f"cache_shape={cache_shape}")
    start = time.perf_counter()
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    measured["decode_trace_host_ms"] = (time.perf_counter() - start) * 1000
    signpost(f"PERF_DECODE_{case_id}_END", f"cache_shape={cache_shape}")
    ttnn.release_trace(mesh_device, trace_id)

    if batch == 1:
        assert prefill_output.shape[-2] == seq_len
    assert traced_decode_output.shape[-2] >= 1
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / f"{case_id}_host_timings.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "shared_physical_cache": shared_physical,
                "sequence_length": seq_len,
                "decode_current_pos": seq_len,
                "decode_batch": batch,
                "cache_shape": cache_shape,
                "provenance": _evidence_provenance(
                    mesh_device,
                    "GEMMA4_FUNCTIONAL_DECODER_PERF=1 GEMMA4_RANGE_DOWNLOAD=1 "
                    "python -m tracy -r -p pytest -- "
                    "models/autoports/google_gemma_4_26b_a4b_it/tests/"
                    f"test_functional_decoder.py::test_functional_decoder_perf_profile -k '{layer_type} and batch{batch}'",
                ),
                **measured,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def test_sparse_moe_prefill_delegates_to_canonical_path(monkeypatch):
    """Keep canonical sparse prefill delegation covered without TT hardware."""
    decoder = object.__new__(FunctionalDecoder)
    decoder.expert_weights = object()
    decoder.expert_config = object()
    decoder.expert_prefill_sparsity = object()
    hidden_states = object()
    routing_weights = object()
    prefill_result = object()
    calls = {}

    def fake_prefill(**kwargs):
        calls["prefill"] = kwargs
        return prefill_result

    monkeypatch.setattr(decoder_module, "sparse_expert_prefill", fake_prefill)

    assert decoder._moe_prefill_chunk(hidden_states, routing_weights) is prefill_result
    assert calls["prefill"] == {
        "hidden_states": hidden_states,
        "routing_weights": routing_weights,
        "weights": decoder.expert_weights,
        "config": decoder.expert_config,
        "prefill_sparsity": decoder.expert_prefill_sparsity,
    }


def test_sparse_moe_canonical_hot_path_audit():
    import inspect

    decode_source = inspect.getsource(FunctionalDecoder._moe_decode_single_user)
    prefill_source = inspect.getsource(decoder_module.sparse_expert_prefill)
    wrapper_source = inspect.getsource(FunctionalDecoder._moe_decode) + inspect.getsource(
        FunctionalDecoder._moe_prefill_chunk
    )

    assert decode_source.count("ttnn.sparse_matmul") == 3
    assert decode_source.count("compute_kernel_config=self.correctness_compute_config") == 1
    decode_batch_source = inspect.getsource(FunctionalDecoder._moe_decode)
    assert "ttnn.slice" in decode_batch_source
    assert "ttnn.concat(outputs, dim=2" in decode_batch_source
    assert "ttnn.sparse_matmul" in inspect.getsource(
        decoder_module.sparse_expert_prefill.__globals__["_process_prefill_chunk"]
    )
    assert "ttnn.repeat(hidden_states" not in wrapper_source
    for token in ("torch.", "import torch", "ttnn.from_torch", "ttnn.to_torch"):
        assert token not in decode_source
        assert token not in prefill_source


def test_functional_decoder_hot_path_fallback_audit():
    import inspect

    forbidden = ("torch.", "import torch", "ttnn.from_torch", "ttnn.to_torch")
    methods = [
        FunctionalDecoder.prefill_forward,
        FunctionalDecoder._prefill_forward_single_user,
        FunctionalDecoder.decode_forward,
        FunctionalDecoder._attention_prefill,
        FunctionalDecoder._fill_prefill_cache,
        FunctionalDecoder._full_chunked_prefill_attention,
        FunctionalDecoder._sliding_chunked_prefill_attention,
        FunctionalDecoder._attention_decode,
        FunctionalDecoder._dense_mlp,
        FunctionalDecoder._router_weights,
        FunctionalDecoder._moe_prefill,
        FunctionalDecoder._moe_prefill_chunk,
        FunctionalDecoder._moe_decode,
        FunctionalDecoder._moe_decode_single_user,
    ]
    source = "\n".join(inspect.getsource(method) for method in methods)
    for token in forbidden:
        assert token not in source
