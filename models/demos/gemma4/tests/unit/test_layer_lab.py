# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Layer lab: one decoder layer of a dense Gemma 4 variant on ONE chip with the per-chip shard shapes of its serving
mesh, driven through the served decode step (presliced RoPE rows, paged KV cache, device position tensors), traced
and replayed under the device profiler the way the traced prefill tracy session is (one read-back before the
``start`` signpost, one after ``stop``). The per-op ledger (kernel / firmware us, DRAM bytes and bandwidth floor,
FLOPs and compute floor, program count) is printed by ``_logs/g6_layer_ledger.py`` from the tracy report.

One (model, layer kind, tp, batch) is one job on one chip (``-k 12B-sliding-tp4-b32``): the HF text config is
re-cut to the shard (q / kv heads and the MLP width divided by ``tp``, kv heads at least 1 as on the replicated
global layers), the layer takes random weights of the shard's shapes and dtypes (attention / MLP bfp8 as served,
norms bf16, KV cache bfp8 under GEMMA4_KV_BFP8=1), and ``GEMMA4_DRAM_SHARD_TP1`` keeps the served DRAM-sharded
decode matmuls on the single chip. CCL ops are out of scope here (tp=1 on the chip: the two all-reduces of the
served layer are not in the ledger). The decode fusion flags are read from the environment (no model build, no
profile): the lab runs the base path unless a job sets them.

    HF_MODEL unused; ids: {12B,31B}-{sliding,global}-tp{4,8}-b{1,32}
    python -m tracy -r -p -v --no-web-server -m pytest test_layer_lab.py -k 12B-sliding-tp4-b32 -s
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4.tt.attention import Gemma4AttentionConfig
from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
from models.demos.gemma4.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4.tt.model import create_rope_caches
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.tt_transformers.tt.common import PagedAttentionConfig

MODELS = {"12B": "google/gemma-4-12B-it", "31B": "google/gemma-4-31B-it"}
# The lab is a measurement harness run by the device-queue lab jobs (one chip per case, tracy session); the CI
# unit tiers (CI=true, E2B / E4B checkpoints) skip it: it names the dense checkpoints itself and needs their
# configs from the HF cache.
pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="layer lab: devq measurement harness, not a CI unit test"
)
KINDS = {"sliding": "sliding_attention", "global": "full_attention"}
CASES = [(m, k, tp, b) for m in ("12B", "31B") for k in ("sliding", "global") for tp in (4, 8) for b in (1, 32)]
PREFILL_LEN = int(os.environ.get("GEMMA4_LAB_PREFILL_LEN", "128"))  # positions already in the cache
N_STEPS = int(os.environ.get("GEMMA4_LAB_STEPS", "3"))
WARM = 2
BLOCK = 64


def _signpost(name):
    try:
        from tracy import signpost
    except ImportError:
        return
    signpost(name)


def shard_text_config(model_name: str, tp: int):
    """The HF text config of ``model_name`` cut to one shard of a TP=``tp`` mesh."""
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(MODELS[model_name], trust_remote_code=True).text_config
    except Exception as e:  # no local config for the dense checkpoint: not this machine's job
        pytest.skip(f"{MODELS[model_name]} config unavailable: {e}")
    assert cfg.num_attention_heads % tp == 0 and cfg.num_key_value_heads % tp == 0, (model_name, tp)
    cfg.num_attention_heads //= tp
    cfg.num_key_value_heads //= tp
    g = getattr(cfg, "num_global_key_value_heads", None) or cfg.num_key_value_heads * tp
    cfg.num_global_key_value_heads = max(1, g // tp)  # replicated when fewer global kv heads than chips
    assert cfg.intermediate_size % tp == 0
    cfg.intermediate_size //= tp
    return cfg


def _layer_index(cfg, kind: str) -> int:
    return next(i for i, lt in enumerate(cfg.layer_types) if lt == KINDS[kind])


def _ids(case):
    m, k, tp, b = case
    return f"{m}-{k}-tp{tp}-b{b}"


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)], device_params_extra={"trace_region_size": 128_000_000})
@pytest.mark.parametrize("model_name, kind, tp, batch", CASES, ids=[_ids(c) for c in CASES])
def test_layer_lab(mesh_device, model_name, kind, tp, batch, monkeypatch):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

    monkeypatch.setenv("GEMMA4_DRAM_SHARD_TP1", "1")
    torch.manual_seed(0)
    cfg = shard_text_config(model_name, tp)
    layer_idx = _layer_index(cfg, kind)
    model_args = Gemma4ModelArgs.from_hf_config(cfg)
    attn_cfg = Gemma4AttentionConfig(model_args, layer_idx)
    hidden = int(cfg.hidden_size)
    max_seq_len = 4096
    logger.info(
        f"LAB layer: {model_name} {kind} tp={tp} batch={batch}: hidden {hidden}, q heads {cfg.num_attention_heads}, "
        f"kv heads {attn_cfg.num_key_value_heads}, head_dim {attn_cfg.head_dim}, mlp {cfg.intermediate_size}, "
        f"layer_idx {layer_idx}"
    )

    # the layer: random HF weights of the shard's shapes, the served dtypes
    hf_layer = Gemma4TextDecoderLayer(cfg, layer_idx=layer_idx)
    tt_state = {f"model.layers.{layer_idx}.{k}": v for k, v in hf_layer.state_dict().items()}
    del hf_layer
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=1))
    layer = Gemma4DecoderLayer(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=tt_state,
        layer_idx=layer_idx,
        ccl_manager=None,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=max_seq_len,
        max_local_batch_size=batch,
        shared_mlp_dtype=ttnn.bfloat8_b,
        attention_dtype=ttnn.bfloat8_b,
    )
    del tt_state

    # paged KV cache + identity page table, positions at PREFILL_LEN (the step attends PREFILL_LEN + 1 rows)
    blocks_per_user = (PREFILL_LEN + N_STEPS + WARM + 2 + BLOCK - 1) // BLOCK
    paged = PagedAttentionConfig(block_size=BLOCK, max_num_blocks=batch * blocks_per_user)
    kv_dtype = ttnn.bfloat8_b if os.environ.get("GEMMA4_KV_BFP8", "1") == "1" else ttnn.bfloat16
    kv_cache = init_kv_cache(
        mesh_device, attn_cfg, max_batch_size=batch, max_seq_len=max_seq_len, paged_attention_config=paged,
        cache_dtype=kv_dtype,
    )  # fmt: skip
    pt_host = torch.arange(paged.max_num_blocks, dtype=torch.int32).reshape(batch, blocks_per_user)
    page_table = ttnn.from_torch(pt_host, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)
    pos_host = torch.full((1, 32), PREFILL_LEN, dtype=torch.int64)
    pos_u32 = ttnn.from_torch(pos_host, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)
    pos_i32 = ttnn.from_torch(
        torch.full((batch,), PREFILL_LEN, dtype=torch.int32), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
    )  # fmt: skip
    # the model's per-step RoPE prologue: one position gather per layer type
    _caches_4d, caches_2d = create_rope_caches(mesh_device, cfg, max_seq_len)
    cos_2d, sin_2d = caches_2d[KINDS[kind]]
    cos_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(pos_u32, cos_2d, layout=ttnn.TILE_LAYOUT))
    sin_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(pos_u32, sin_2d, layout=ttnn.TILE_LAYOUT))
    x = ttnn.from_torch(
        (torch.randn(1, 1, batch, hidden) * 2).to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )  # fmt: skip
    rope_prepared = {}  # the model keeps the prepared cos / sin of a layer type for the whole step: built once here

    def step():
        # the layer frees its input (the residual) -- a clone keeps ``x`` for the next step (one Copy program,
        # listed in the ledger as the lab's input copy)
        x_in = ttnn.clone(x)
        return layer(
            x_in,
            rope_mats=(cos_pos, sin_pos),
            position_idx=pos_u32,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=True,
            token_index=None,
            position_idx_cache=pos_i32,
            rope_presliced=True,
            rope_prepared=rope_prepared,
        )

    for _ in range(WARM):  # compile + the prepared RoPE tensors
        out = step()
        ttnn.synchronize_device(mesh_device)
        out.deallocate(True)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = step()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for _ in range(WARM):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    try:
        ttnn.ReadDeviceProfiler(mesh_device)
    except Exception:
        pass
    _signpost("start")
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    wall_ms = (time.perf_counter() - t0) / N_STEPS * 1e3
    _signpost("stop")
    try:
        ttnn.ReadDeviceProfiler(mesh_device)
    except Exception:
        pass
    o = ttnn.to_torch(out).float()
    assert torch.isfinite(o).all(), "layer output not finite"
    logger.info(
        f"LAB_WALL model={model_name} kind={kind} tp={tp} batch={batch} traced: {wall_ms:.3f} ms/step "
        f"({N_STEPS} replays; |out| mean {o.abs().mean():.3f})"
    )
    ttnn.release_trace(mesh_device, tid)
