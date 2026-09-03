# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Profile the CodePredictor frame the **demo actually runs**.

``test_qwen3_tts_profile_single_layer.py -k cp_layer_*`` measures one CP layer
*eagerly*, with a 1-layer CodePredictor and a fresh DRAM-interleaved input. The demo
runs something structurally different: ``capture_fused_cp_trace`` (``tt/server.py``)
puts the **whole** frame into ONE Metal trace —

    KV/constant restore -> [talker_hidden ; embed(code0)] concat -> CP prefill (seq=2,
    5 layers) -> sample -> 14 x (device embedding -> CP decode (seq=1, 5 layers) ->
    sample) -> concat 15 ids -> 16 device embeddings + adds -> next Talker embedding

— and replays it once per audio frame. This test builds that same trace by calling the
**production** ``capture_fused_cp_trace`` with the production ``CodePredictor`` (real
5-layer config, synthetic weights) and replays it **once** between signposts, so the op
stream in the report is the deployed one.

Weights are random: shapes, memory configs, program configs and the op graph are what
device time depends on, and all of those come from the config, not the values.

Run (N300):

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=wormhole_b0 MESH_DEVICE=N300
    python -m tracy -p -v -r -m pytest -s -q \\
      models/demos/qwen3_tts/tests/test_qwen3_tts_profile_cp_frame.py -k cp_fused_frame
    CSV=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
    tt-perf-report --start-signpost start --end-signpost stop "$CSV"

One replay per capture — the profiler raises ``Device data mismatch`` across multiple
traced replays, and a whole-demo capture overflows the zone count.
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


# Demo constants (server.py generate_codes_ttnn).
MAX_CP_SEQ_LEN = 32
CP_PREFILL_SEQ = 2
TALKER_CODEC_VOCAB = 3072


def _open_device():
    mesh_shape = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(os.environ.get("MESH_DEVICE"))
    if mesh_shape is None:
        d = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=200000000)
        d.enable_program_cache()
        return d, None
    if mesh_shape != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    d = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*mesh_shape),
        l1_small_size=32768,
        trace_region_size=200000000,
    )
    d.enable_program_cache()
    return d, mesh_shape


@pytest.fixture(scope="module")
def device():
    d, mesh_shape = _open_device()
    yield d
    if mesh_shape is None:
        ttnn.close_device(d)
        return
    ttnn.close_mesh_device(d)
    if mesh_shape != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _synthetic_cp_sd(cfg, talker_hidden):
    """Full CodePredictor state dict — all ``cfg.num_hidden_layers`` layers."""
    torch.manual_seed(0)
    h, i = cfg.hidden_size, cfg.intermediate_size
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    sd = {
        "talker.code_predictor.small_to_mtp_projection.weight": torch.randn(h, talker_hidden, dtype=torch.bfloat16),
        "talker.code_predictor.small_to_mtp_projection.bias": torch.zeros(h, dtype=torch.bfloat16),
        "talker.code_predictor.model.norm.weight": torch.ones(h, dtype=torch.bfloat16),
    }
    for li in range(cfg.num_hidden_layers):
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


class _ModelStub:
    """``capture_fused_cp_trace`` only reaches ``model.code_predictor``."""

    def __init__(self, code_predictor):
        self.code_predictor = code_predictor


def _build_fused_cp_frame(device, *, max_new_tokens=8, signpost_warmup=False, capture=True):
    """Mirror server.py's CP setup exactly, then capture the production fused trace."""
    from models.demos.qwen3_tts.tt.code_predictor import CodePredictor
    from models.demos.qwen3_tts.tt.mesh_utils import get_tp_size, is_mesh_device
    from models.demos.qwen3_tts.tt.model_config import Qwen3TTSCodePredictorConfig, Qwen3TTSTalkerConfig
    from models.demos.qwen3_tts.tt.rope import compute_rope_frequencies, get_rope_tensors, get_transformation_mat
    from models.demos.qwen3_tts.tt.server import (
        TTSConfig,
        _DeviceSampler,
        _replicate_mapper,
        allocate_kv_cache,
        build_cp_decode_trace_h2d_constants,
        build_trailing_row_h2d,
        capture_fused_cp_trace,
        upload_embed_tables,
    )

    talker_cfg = Qwen3TTSTalkerConfig()
    cp_cfg = Qwen3TTSCodePredictorConfig()
    talker_h = talker_cfg.hidden_size
    cp_head_dim = cp_cfg.head_dim

    tp = get_tp_size(device) if is_mesh_device(device) else 1
    cp_num_heads = cp_cfg.num_attention_heads // tp

    config = TTSConfig()
    config.hidden_size = talker_h
    config.max_new_tokens = max_new_tokens
    config.num_code_groups = cp_cfg.num_code_groups

    code_predictor = CodePredictor(
        device=device,
        config=cp_cfg,
        talker_hidden_size=talker_h,
        state_dict=_synthetic_cp_sd(cp_cfg, talker_h),
    )
    model = _ModelStub(code_predictor)

    cp_trans_mat = get_transformation_mat(cp_head_dim, device)

    # --- persistent state, all allocated BEFORE the capture (server.py ordering) ---
    torch.manual_seed(0)
    codec_embed_torch = torch.randn(TALKER_CODEC_VOCAB, talker_h).bfloat16().float()
    cp_tables_torch = [
        torch.randn(cp_cfg.vocab_size, talker_h).bfloat16().float() for _ in range(cp_cfg.num_code_groups - 1)
    ]
    codec_embed_tt, cp_embed_tts = upload_embed_tables(device, codec_embed_torch, cp_tables_torch)

    sampler = _DeviceSampler(device, top_k=config.top_k, temperature=config.temperature)
    tok_bufs = [sampler.alloc_token_buf() for _ in range(config.num_code_groups)]
    sampler.warm_ccl()
    mapper = _replicate_mapper(device)
    trail_row_tt = ttnn.from_torch(
        torch.zeros(1, 1, 1, talker_h, dtype=torch.float32),
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    src_hidden_tt = ttnn.from_torch(
        torch.zeros(1, 1, 1, talker_h, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    talker_embed_dst_tt = ttnn.from_torch(
        torch.zeros(1, 1, 1, talker_h, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )

    cp_kv_caches = allocate_kv_cache(
        device=device,
        num_layers=cp_cfg.num_hidden_layers,
        batch_size=1,
        num_kv_heads=cp_cfg.num_key_value_heads,
        max_seq_len=MAX_CP_SEQ_LEN,
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

    cp_cos_table, cp_sin_table = compute_rope_frequencies(cp_head_dim, MAX_CP_SEQ_LEN + 5, cp_cfg.rope_theta)
    cp_prefill_cos_tt, cp_prefill_sin_tt = get_rope_tensors(
        device, cp_head_dim, CP_PREFILL_SEQ, torch.arange(CP_PREFILL_SEQ), cp_cfg.rope_theta
    )
    from models.demos.qwen3_tts.tt.mesh_utils import to_torch as _mesh_to_torch

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
    mask_host = torch.full((1, cp_num_heads, CP_PREFILL_SEQ, MAX_CP_SEQ_LEN), float("-inf"))
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
        torch.zeros(1, 1, CP_PREFILL_SEQ, talker_h, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    n_cp_decode = config.num_code_groups - 2
    cos_h2d, sin_h2d, mask_h2d = build_cp_decode_trace_h2d_constants(
        cp_cos_table, cp_sin_table, cp_num_heads, MAX_CP_SEQ_LEN, n_cp_decode
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

    trail_rows = build_trailing_row_h2d(
        torch.zeros(1, max_new_tokens, talker_h), torch.zeros(1, 1, talker_h), max_new_tokens
    )

    state = dict(
        model=model,
        code_predictor=code_predictor,
        config=config,
        cp_cfg=cp_cfg,
        talker_h=talker_h,
        cp_trans_mat=cp_trans_mat,
        cp_kv_caches=cp_kv_caches,
        cp_prefill_embed_tt=cp_prefill_embed_tt,
        cp_prefill_cos_tt=cp_prefill_cos_tt,
        cp_prefill_sin_tt=cp_prefill_sin_tt,
        cp_prefill_mask_tt=cp_prefill_mask_tt,
        cp_decode_embed_tt=cp_decode_embed_tt,
        cp_decode_cos_tts=cp_decode_cos_tts,
        cp_decode_sin_tts=cp_decode_sin_tts,
        cp_decode_mask_tts=cp_decode_mask_tts,
        sampler=sampler,
        tok_bufs=tok_bufs,
    )
    if capture is False:
        return None, state

    fused = capture_fused_cp_trace(
        device,
        model,
        config,
        cp_trans_mat=cp_trans_mat,
        cp_kv_caches_persistent=cp_kv_caches,
        cp_kv_zero_hosts=cp_kv_zero_hosts,
        cp_prefill_embed_tt=cp_prefill_embed_tt,
        cp_prefill_mask_tt=cp_prefill_mask_tt,
        cp_prefill_cos_tt=cp_prefill_cos_tt,
        cp_prefill_sin_tt=cp_prefill_sin_tt,
        cp_prefill_mask_src=cp_prefill_mask_src,
        cp_prefill_cos_src=cp_prefill_cos_src,
        cp_prefill_sin_src=cp_prefill_sin_src,
        cp_decode_embed_tt=cp_decode_embed_tt,
        cp_decode_cos_tts=cp_decode_cos_tts,
        cp_decode_sin_tts=cp_decode_sin_tts,
        cp_decode_mask_tts=cp_decode_mask_tts,
        talker_hidden_src_tt=src_hidden_tt,
        talker_embed_dst_tt=talker_embed_dst_tt,
        codec_embed_tt=codec_embed_tt,
        cp_embed_tts=cp_embed_tts,
        talker_h=talker_h,
        sampler=sampler,
        tok_bufs=tok_bufs,
        trail_row_tt=trail_row_tt,
        trail_row_h2d=trail_rows,
        signpost_warmup=signpost_warmup,
    )
    return fused, state


def test_cp_fused_frame(device):
    """One warm pass of the demo's fused CodePredictor frame body, signpost-bounded.

    ``capture_fused_cp_trace(signpost_warmup=True)`` wraps a second **untraced** pass of
    the exact body it then captures. Trace capture records the programs that body
    enqueues, so the op graph and the per-op kernel durations are the trace's; only host
    dispatch differs, and that is not what ``DEVICE KERNEL DURATION`` measures.

    Profiling the replay itself does not work here: the replayed programs only reach the
    ops CSV through ``cpp_device_perf_report.csv``, and that post-processor asserts on
    this graph (``Device data missing: Op N not present``). The legacy parser drops them
    silently, leaving an empty signpost window.
    """
    fused, _state = _build_fused_cp_frame(device, signpost_warmup=True)
    assert fused.trace_id is None, "signpost_warmup returns before capture — nothing to release"
    print("[cp_fused_frame] one fused CodePredictor frame profiled (prefill seq=2 + 14 decode, 5 layers)")


def test_cp_decode_step(device):
    """ONE CP decode step — the loop-iteration harness for the optimization loop.

    The frame is 1 prefill + 14 *identical* decode steps, so one step is 1/15 of the
    ops and post-processing (the full-frame capture writes a >1 GB device log and takes
    minutes to parse; this lands in well under one). Everything the demo's traced decode
    step does is here and nothing else: ``small_to_mtp_projection`` -> 5 layers -> final
    RMSNorm -> ``lm_head[k]`` -> the in-trace Gumbel sampler.

    Use ``test_cp_fused_frame`` for the frame-level baseline and the final check; use
    this one to A/B a lever.
    """
    _, st = _build_fused_cp_frame(device, capture=False)
    cp = st["code_predictor"]
    slot, code_idx = 1, 2

    def _step():
        logits, _ = cp.forward_single_step(
            st["cp_decode_embed_tt"],
            st["cp_decode_cos_tts"][slot - 1],
            st["cp_decode_sin_tts"][slot - 1],
            st["cp_trans_mat"],
            generation_step=code_idx,
            kv_caches=st["cp_kv_caches"],
            start_pos=code_idx,
            mode="decode",
            decode_attn_mask=st["cp_decode_mask_tts"][slot - 1],
        )
        st["sampler"].append_sampling(logits, slot, st["tok_bufs"][code_idx])
        ttnn.deallocate(logits)

    _step()
    ttnn.synchronize_device(device)
    signpost("start")
    _step()
    ttnn.synchronize_device(device)
    signpost("stop")
    print("[cp_decode_step] one CP decode step (proj + 5 layers + norm + lm_head + sampler)")


def test_cp_prefill_step(device):
    """ONE CP prefill step (seq=2) — the other half of the frame, same harness."""
    _, st = _build_fused_cp_frame(device, capture=False)
    cp = st["code_predictor"]

    def _step():
        logits, _ = cp.forward_single_step(
            st["cp_prefill_embed_tt"],
            st["cp_prefill_cos_tt"],
            st["cp_prefill_sin_tt"],
            st["cp_trans_mat"],
            generation_step=1,
            kv_caches=st["cp_kv_caches"],
            start_pos=0,
            mode="prefill",
            cp_prefill_mask=st["cp_prefill_mask_tt"],
        )
        vocab = int(logits.shape[3])
        lg1 = ttnn.slice(logits, [0, 0, 1, 0], [1, 1, 2, vocab])
        ttnn.deallocate(logits)
        st["sampler"].append_sampling(lg1, 0, st["tok_bufs"][1])
        ttnn.deallocate(lg1)

    _step()
    ttnn.synchronize_device(device)
    signpost("start")
    _step()
    ttnn.synchronize_device(device)
    signpost("stop")
    print("[cp_prefill_step] one CP prefill step (seq=2, proj + 5 layers + norm + lm_head + sampler)")
