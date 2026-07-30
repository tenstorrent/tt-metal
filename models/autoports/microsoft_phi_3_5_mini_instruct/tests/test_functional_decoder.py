# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import importlib
import inspect
import json
import os
from pathlib import Path

import pytest
import torch
import ttnn
from safetensors import safe_open
from transformers import AutoConfig

from models.autoports.microsoft_phi_3_5_mini_instruct.tt.functional_decoder import FunctionalDecoder
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

try:
    from tracy import signpost
except ImportError:

    def signpost(*_args, **_kwargs):
        return None


MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
SNAPSHOT = "2fe192450127e6a83f7441aef6e3ca586c338b77"
HF_CACHE = Path.home() / ".cache/huggingface/hub/models--microsoft--Phi-3.5-mini-instruct/snapshots" / SNAPSHOT
PCC_THRESHOLD = 0.995
TEST_TABLE_LEN = 512
AUTO_DIR = Path(__file__).resolve().parents[1]
WEIGHT_STATS_PATH = AUTO_DIR / "doc/functional_decoder/weight_stats_layer0.json"
LAYER_WEIGHT_SHAPES = {
    "self_attn.qkv_proj.weight": (9216, 3072),
    "self_attn.o_proj.weight": (3072, 3072),
    "mlp.gate_up_proj.weight": (16384, 3072),
    "mlp.down_proj.weight": (3072, 8192),
    "input_layernorm.weight": (3072,),
    "post_attention_layernorm.weight": (3072,),
}


@pytest.fixture
def mesh_device():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        yield device
    finally:
        if os.getenv("PHI35_SKIP_MESH_CLOSE") == "1":
            return
        ttnn.close_mesh_device(device)


class Phi3LayerCache:
    """Compatibility cache for the remote Phi-3 modeling file in this checkout."""

    def __init__(self):
        self.data = {}

    def get_usable_length(self, _kv_seq_len, layer_idx):
        if layer_idx not in self.data:
            return 0
        return self.data[layer_idx][0].shape[-2]

    def get_seq_length(self, layer_idx=0):
        return self.get_usable_length(0, layer_idx)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx in self.data:
            key_states = torch.cat([self.data[layer_idx][0], key_states], dim=-2)
            value_states = torch.cat([self.data[layer_idx][1], value_states], dim=-2)
        self.data[layer_idx] = (key_states, value_states)
        return key_states, value_states

    def __getitem__(self, layer_idx):
        return self.data[layer_idx]


def _hf_config():
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    cfg._attn_implementation = "eager"
    return cfg


def _hf_layer_class(cfg):
    module = importlib.import_module(cfg.__class__.__module__.replace("configuration_phi3", "modeling_phi3"))
    return module.Phi3DecoderLayer


def _make_hf_layer(cfg, state_dict):
    layer = _hf_layer_class(cfg)(cfg, layer_idx=0).eval().to(torch.bfloat16)
    layer.load_state_dict({k: v.to(torch.bfloat16) for k, v in state_dict.items()})
    return layer


def _synthetic_state_dict(seed=0):
    generator = torch.Generator().manual_seed(seed)
    if WEIGHT_STATS_PATH.exists():
        stats = json.loads(WEIGHT_STATS_PATH.read_text())["tensors"]
        state = {}
        for name, shape in LAYER_WEIGHT_SHAPES.items():
            record = stats[name]
            if tuple(record["shape"]) != shape:
                raise ValueError(f"{name} stats shape {record['shape']} != expected {shape}")
            mean = float(record["mean"])
            std = float(record["std"])
            if std == 0.0:
                state[name] = torch.full(shape, mean, dtype=torch.float32)
            else:
                state[name] = torch.randn(shape, generator=generator, dtype=torch.float32) * std + mean
        return state

    return {name: torch.randn(shape, generator=generator, dtype=torch.float32) * 0.01 for name, shape in LAYER_WEIGHT_SHAPES.items()}


def _real_layer0_state_dict():
    index_path = HF_CACHE / "model.safetensors.index.json"
    if not index_path.exists():
        pytest.skip(f"real Phi-3.5 weights are not present at {HF_CACHE}")
    index = json.loads(index_path.read_text())
    wanted_prefix = "model.layers.0."
    by_shard = {}
    for key, shard in index["weight_map"].items():
        if key.startswith(wanted_prefix):
            by_shard.setdefault(shard, []).append(key)

    state = {}
    for shard, keys in by_shard.items():
        with safe_open(HF_CACHE / shard, framework="pt", device="cpu") as f:
            for key in keys:
                state[key[len(wanted_prefix) :]] = f.get_tensor(key)
    return state


def _causal_mask(seq_len):
    mask = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min)
    return torch.triu(mask, diagonal=1).reshape(1, 1, seq_len, seq_len)


def _page_table(num_blocks):
    # Non-identity permutation catches page-table routing bugs.
    values = list(range(num_blocks))
    if len(values) > 1:
        values = values[1:] + values[:1]
    return torch.tensor([values], dtype=torch.int32)


def _position_tensors(mesh_device, pos):
    current_pos = ttnn.Tensor(torch.tensor([pos], dtype=torch.int32), ttnn.int32).to(mesh_device)
    position_ids = ttnn.Tensor(torch.tensor([pos], dtype=torch.uint32), ttnn.uint32).to(mesh_device)
    return current_pos, position_ids


def _run_prefill_decode_pcc(mesh_device, state_dict, *, seq_len=32, seed=1, trace_decode=True):
    cfg = _hf_config()
    hf_layer = _make_hf_layer(cfg, state_dict)
    torch.manual_seed(seed)
    x_prefill = (torch.randn(1, seq_len, cfg.hidden_size, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    with torch.no_grad():
        ref_prefill = hf_layer(
            x_prefill,
            attention_mask=_causal_mask(seq_len),
            position_ids=torch.arange(seq_len).reshape(1, seq_len),
            use_cache=False,
        )[0]

    decoder = FunctionalDecoder.from_state_dict(
        state_dict,
        hf_config=cfg,
        layer_idx=0,
        mesh_device=mesh_device,
        max_position_embeddings=max(TEST_TABLE_LEN, seq_len + 32),
    )
    kv_cache = FunctionalDecoder.allocate_paged_kv_cache(
        hf_config=cfg,
        mesh_device=mesh_device,
        max_batch_size=1,
        max_seq_len=seq_len + 32,
        block_size=32,
    )
    page_table_host = _page_table((seq_len + 31) // 32 + 1)
    page_table = ttnn.Tensor(page_table_host, ttnn.int32).to(mesh_device)
    x_tt = ttnn.Tensor(x_prefill.reshape(1, 1, seq_len, cfg.hidden_size), ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(
        mesh_device
    )

    warm_prefill = decoder.prefill_forward(
        x_tt,
        page_table=page_table,
        kv_cache=kv_cache,
        start_pos=0,
        rope_sequence_length=seq_len,
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm_prefill)

    signpost("PERF_PREFILL")
    tt_prefill = decoder.prefill_forward(
        x_tt,
        page_table=page_table,
        kv_cache=kv_cache,
        start_pos=0,
        rope_sequence_length=seq_len,
    )
    ttnn.synchronize_device(mesh_device)
    signpost("PERF_PREFILL_END")

    got_prefill = ttnn.to_torch(tt_prefill).reshape(1, seq_len, cfg.hidden_size)
    prefill_ok, prefill_msg = comp_pcc(ref_prefill.float(), got_prefill.float(), pcc=PCC_THRESHOLD)
    print(f"HF-vs-TTNN prefill PCC: {prefill_msg}")

    cache = Phi3LayerCache()
    with torch.no_grad():
        hf_layer(
            x_prefill,
            attention_mask=_causal_mask(seq_len),
            position_ids=torch.arange(seq_len).reshape(1, seq_len),
            past_key_value=cache,
            use_cache=True,
        )
        x_decode = (torch.randn(1, 1, cfg.hidden_size, dtype=torch.float32) * 0.1).to(torch.bfloat16)
        ref_decode = hf_layer(
            x_decode,
            attention_mask=None,
            position_ids=torch.tensor([[seq_len]]),
            past_key_value=cache,
            use_cache=True,
        )[0]

    x_decode_tt = ttnn.Tensor(x_decode.reshape(1, 1, 1, cfg.hidden_size), ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(
        mesh_device
    )
    current_pos, position_ids = _position_tensors(mesh_device, seq_len)

    # Compile/warm eager decode once. The same K/V slot is overwritten with the
    # same tensors during traced capture/replay, so the measured output remains stable.
    decoder.decode_forward(
        x_decode_tt,
        current_pos=current_pos,
        position_ids=position_ids,
        page_table=page_table,
        kv_cache=kv_cache,
        rope_sequence_length=seq_len + 1,
    )
    ttnn.synchronize_device(mesh_device)

    if trace_decode:
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_decode = decoder.decode_forward(
            x_decode_tt,
            current_pos=current_pos,
            position_ids=position_ids,
            page_table=page_table,
            kv_cache=kv_cache,
            rope_sequence_length=seq_len + 1,
        )
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh_device)
        signpost("PERF_DECODE")
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh_device)
        signpost("PERF_DECODE_END")
        ttnn.release_trace(mesh_device, trace_id)
    else:
        tt_decode = decoder.decode_forward(
            x_decode_tt,
            current_pos=current_pos,
            position_ids=position_ids,
            page_table=page_table,
            kv_cache=kv_cache,
            rope_sequence_length=seq_len + 1,
        )

    got_decode = ttnn.to_torch(tt_decode).reshape(1, 1, cfg.hidden_size)
    decode_ok, decode_msg = comp_pcc(ref_decode.float(), got_decode.float(), pcc=PCC_THRESHOLD)
    print(f"HF-vs-TTNN decode PCC: {decode_msg}")
    if os.getenv("PHI35_READ_DEVICE_PROFILER") == "1":
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
    return {
        "prefill_ok": prefill_ok,
        "prefill_msg": prefill_msg,
        "decode_ok": decode_ok,
        "decode_msg": decode_msg,
        "prefill_output": got_prefill,
        "decode_output": got_decode,
    }


@pytest.mark.timeout(240)
def test_dense_layer_synthetic_prefill_decode_pcc_and_traced_decode(mesh_device):
    result = _run_prefill_decode_pcc(mesh_device, _synthetic_state_dict(), seq_len=32, trace_decode=True)
    assert result["prefill_ok"], result["prefill_msg"]
    assert result["decode_ok"], result["decode_msg"]


@pytest.mark.timeout(300)
def test_dense_layer_real_weights_prefill_decode_pcc(mesh_device):
    result = _run_prefill_decode_pcc(mesh_device, _real_layer0_state_dict(), seq_len=32, seed=7, trace_decode=True)
    assert result["prefill_ok"], result["prefill_msg"]
    assert result["decode_ok"], result["decode_msg"]


@pytest.mark.timeout(300)
def test_repeated_input_determinism(mesh_device):
    state = _synthetic_state_dict(seed=11)
    first = _run_prefill_decode_pcc(mesh_device, state, seq_len=32, seed=13, trace_decode=False)
    second = _run_prefill_decode_pcc(mesh_device, state, seq_len=32, seed=13, trace_decode=False)
    prefill_ok, prefill_msg = comp_pcc(first["prefill_output"].float(), second["prefill_output"].float(), pcc=0.9999)
    decode_ok, decode_msg = comp_pcc(first["decode_output"].float(), second["decode_output"].float(), pcc=0.9999)
    assert prefill_ok, prefill_msg
    assert decode_ok, decode_msg


def test_runtime_forward_fallback_audit_static():
    runtime_callables = [
        FunctionalDecoder.prefill_forward,
        FunctionalDecoder.decode_forward,
        FunctionalDecoder._mlp_forward,
        FunctionalDecoder._prefill_rope_tables,
        FunctionalDecoder._decode_rope_tables,
        importlib.import_module(
            "models.autoports.microsoft_phi_3_5_mini_instruct.tt.functional_decoder"
        )._apply_rope,
        importlib.import_module(
            "models.autoports.microsoft_phi_3_5_mini_instruct.tt.functional_decoder"
        )._rotate_half,
        importlib.import_module(
            "models.autoports.microsoft_phi_3_5_mini_instruct.tt.functional_decoder"
        )._typecast_if_needed,
    ]
    for callable_obj in runtime_callables:
        source = inspect.getsource(callable_obj)
        forbidden = ("torch.", "ttnn.from_torch", "ttnn.to_torch", "from_torch(", "to_torch(")
        hits = [token for token in forbidden if token in source]
        assert not hits, f"{callable_obj.__name__} contains forbidden runtime fallback tokens: {hits}"


@pytest.mark.skipif(os.getenv("PHI35_RUN_LONG_CONTEXT") != "1", reason="set PHI35_RUN_LONG_CONTEXT=1 for full-context stress")
@pytest.mark.timeout(600)
def test_full_context_decode_current_position_and_page_table(mesh_device):
    cfg = _hf_config()
    seq_len = cfg.max_position_embeddings
    state = _synthetic_state_dict(seed=17)
    decoder = FunctionalDecoder.from_state_dict(state, hf_config=cfg, layer_idx=0, mesh_device=mesh_device)
    kv_cache = FunctionalDecoder.allocate_paged_kv_cache(
        hf_config=cfg,
        mesh_device=mesh_device,
        max_batch_size=1,
        max_seq_len=seq_len,
        block_size=32,
    )
    page_blocks = seq_len // 32
    page_table = ttnn.Tensor(torch.arange(page_blocks, dtype=torch.int32).reshape(1, page_blocks), ttnn.int32).to(
        mesh_device
    )
    x_decode = ttnn.Tensor(torch.zeros(1, 1, 1, cfg.hidden_size, dtype=torch.bfloat16), ttnn.bfloat16).to(
        ttnn.TILE_LAYOUT
    ).to(mesh_device)
    current_pos, position_ids = _position_tensors(mesh_device, seq_len - 1)
    out = decoder.decode_forward(
        x_decode,
        current_pos=current_pos,
        position_ids=position_ids,
        page_table=page_table,
        kv_cache=kv_cache,
        rope_sequence_length=seq_len,
    )
    got = ttnn.to_torch(out)
    assert got.shape == (1, 1, 1, cfg.hidden_size)
    assert torch.isfinite(got).all()


@pytest.mark.skipif(os.getenv("PHI35_RUN_LONG_PREFILL") != "1", reason="set PHI35_RUN_LONG_PREFILL=1 for long prefill")
@pytest.mark.timeout(900)
def test_long_prefill_page_table(mesh_device):
    cfg = _hf_config()
    seq_len = int(os.getenv("PHI35_LONG_PREFILL_LEN", "4096"))
    assert seq_len % 32 == 0
    state = _synthetic_state_dict(seed=19)
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=0,
        mesh_device=mesh_device,
        max_position_embeddings=seq_len,
    )
    kv_cache = FunctionalDecoder.allocate_paged_kv_cache(
        hf_config=cfg,
        mesh_device=mesh_device,
        max_batch_size=1,
        max_seq_len=seq_len,
        block_size=32,
    )
    page_blocks = seq_len // 32
    page_table_host = _page_table(page_blocks)
    page_table = ttnn.Tensor(page_table_host, ttnn.int32).to(mesh_device)
    x_prefill = ttnn.Tensor(torch.zeros(1, 1, seq_len, cfg.hidden_size, dtype=torch.bfloat16), ttnn.bfloat16).to(
        ttnn.TILE_LAYOUT
    ).to(mesh_device)
    out = decoder.prefill_forward(
        x_prefill,
        page_table=page_table,
        kv_cache=kv_cache,
        start_pos=0,
        rope_sequence_length=seq_len,
    )
    got = ttnn.to_torch(out)
    assert got.shape == (1, 1, seq_len, cfg.hidden_size)
    assert torch.isfinite(got).all()
