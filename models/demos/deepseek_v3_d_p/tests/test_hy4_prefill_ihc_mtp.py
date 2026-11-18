# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-preview prefill MTP with iHC on device vs the fp32 reference, after a truncated (5-layer) trunk.

What Hy4's MTP layer consumes (vLLM hy_v4, ``vllm/models/hy_v4/nvidia/{model,mtp}.py``; transformers on this host has
no hy_v4 and #65936's reference has no MTP):

  * the main model's output on its last rank, ``h0 = norm(hc_head(streams))``: the iHC head collapses the 4 streams
    and the model's final norm runs on that; that is the ``previous_hidden_states`` the MTP layer reads;
  * the MTP layer is a plain-residual decoder layer (``enable_ihc=False``; the checkpoint's ``model.mtp_layers.0``
    has no ``hc_*`` tensors): ``eh_proj(cat[enorm(embed(next ids)), hnorm(h0)])`` -> attention (its own full DSA
    indexer, sink, output gate) -> MoE -> ``final_layernorm`` (``HYV4SharedHead`` is the identity).

Device: ``TtPrefillTransformer`` with 5 trunk layers (iHC on) and a ``TtMTPPredictor`` (K = 1) built at layer index 5
from ``model.mtp_layers.0`` (``hy4_checkpoint.load_hy4_mtp_state_dict``), one 4096-token chunk of the 4K golden prompt
with the golden's next token as the provided lookahead (so no level generates). Layout: the dev
``glm_prefill_sp8_tp4`` (8x4 mesh, SP 8 on axis 0, TP 4 on axis 1).

Reference (``ref_mtp.py`` next to the test env, #65936's ``Hy4PrefillReference``): the same truncated model, written
as a golden trace -- layers 0..4 are the 4K golden's own, layer 5 is the MTP layer. Scored: KVPE and index-K for
every slot (the MTP's at slot 5), the iHC streams at every trunk site, h0, and the MTP layer's output and its
final_layernorm output.

Env: as ``test_hy4_prefill_ihc_layers.py``, plus ``HY4_MTP_TRACE`` (the truncated-model trace dir) and
``HY4_MTP_TTNN_CACHE`` (MTP weight cache root).
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import models.demos.deepseek_v3_d_p.tests.test_hy4_prefill_device_smoke as smoke
import ttnn
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.test_hy4_prefill_ihc_layers import REF_DIR, _SiteRecorder
from models.demos.deepseek_v3_d_p.tt.hy4 import ihc as hy4_ihc

chunked = smoke.chunked
TRUNK = 5  # the MTP layer sits at index 5
TRACE = Path(os.environ.get("HY4_MTP_TRACE", "/mnt/tt-data/dwadhwani/hy4-golden/hy4-4k-v2-mtp-L5"))
MTP_CACHE_ROOT = Path(
    os.environ.get("HY4_MTP_TTNN_CACHE", "/mnt/tt-data/dwadhwani/Hy4-Preview-Cache/hy4_mtp_ttnn_cache")
)
SEQ_CACHE = 2 * smoke.CHUNK_SIZE
PCC_FLOOR = 0.99
# The MTP block's output is a plain residual dominated by its MoE, whose top-8 routing amplifies the upstream error
# (the trunk's MoE sites at the same depth: 0.985 / 0.982 update PCC). GLM-5.3's MTP test budgets the same output at
# 0.96 (tests/mtp_prefill/test_mtp_transformer_chunks.py MTP_MODULE_OUTPUT_PCC).
MTP_OUTPUT_FLOOR = 0.96

pytestmark = smoke.pytestmark + [
    pytest.mark.skipif(not (TRACE / "mtp_tensors.pt").is_file(), reason=f"no truncated-model MTP reference at {TRACE}"),
    pytest.mark.skipif(
        not (REF_DIR / f"streams_L{TRUNK - 1}_ffn.pt").is_file(), reason=f"no reference streams at {REF_DIR}"
    ),
]


def _pcc(a, b) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, sp_axis",
    [
        pytest.param(
            smoke.MESH,
            torus_xy_device_params(
                fabric_payload_size=chunked.moe_fabric_payload(GLM53Config),
                l1_small_size=chunked.GLM_L1_SMALL_SIZE,
                trace_region_size=chunked.GLM_TRACE_REGION_SIZE,
            ),
            2,
            0,
            id="dev-sp8tp4-torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["hy4_preview"], indirect=True, ids=["hy4"])
@pytest.mark.timeout(0)
def test_hy4_prefill_ihc_mtp(variant, config_only, mesh_device, device_params, num_links, sp_axis, monkeypatch):
    from models.common.utility_functions import is_blackhole
    from models.demos.common.prefill.runners.runner_utils import num_mtp_tokens
    from models.demos.deepseek_v3_d_p.reference.hy4_config import Hy4Config
    from models.demos.deepseek_v3_d_p.tests.mtp_prefill.test_mtp_transformer_chunks import (
        _from_device,
        _mtp_union,
        mtp_chunk_stream,
    )
    from models.demos.deepseek_v3_d_p.tt.hy4 import layout as hy4_layout
    from models.demos.deepseek_v3_d_p.tt.mla.indexer import full_indexer_rank
    from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
    from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE
    from models.demos.deepseek_v3_d_p.tt.mtp_prefill.mtp_config import MTPConfig
    from models.demos.deepseek_v3_d_p.tt.mtp_prefill.tt_mtp import TtMTPPredictor
    from models.demos.deepseek_v3_d_p.tt.mtp_prefill.utils import MTP_CACHE_PREFIX, enable_mtp_indexer_slot
    from models.demos.deepseek_v3_d_p.tt.runners import hy4_checkpoint as hc
    from models.demos.deepseek_v3_d_p.tt.runners import weight_categories as wc
    from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
    from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache, init_mla_kv_cache

    monkeypatch.setenv(hy4_layout.LAYOUT_ENV, hy4_layout.DEV_LAYOUT)
    sp, tp = smoke.MESH[sp_axis], smoke.MESH[1 - sp_axis]
    tp_axis = 1 - sp_axis
    model_cfg = variant.model_config
    assert model_cfg is Hy4Config and hy4_ihc.ihc_enabled(model_cfg)
    topology = per_axis_topology(device_params["fabric_config"])
    S = smoke.CHUNK_SIZE
    K = 1
    n_mtp = num_mtp_tokens(K)
    experts_per_chip = model_cfg.NUM_ROUTED_EXPERTS // (sp * tp)
    fmt = chunked.GLM_KV_CACHE_FORMAT

    config = copy.deepcopy(config_only)
    config.max_seq_len = SEQ_CACHE
    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and rope_scaling.get("factor", 1.0) == 1.0:
        rope_scaling["original_max_position_embeddings"] = SEQ_CACHE  # as the chunked driver does
    config.indexer_types = list(config.indexer_types)[:TRUNK]
    assert enable_mtp_indexer_slot(config, TRUNK) == TRUNK  # the MTP layer owns a (full) indexer slot

    # --- weights: the trunk's missing categories from the checkpoint (the runner's rebuild), the MTP's own cache ---
    cache = smoke._mesh_cache_dir(mesh_device.get_num_devices())
    missing = wc.missing_trunk_categories(
        cache,
        num_layers=TRUNK,
        first_layer_idx=0,
        experts_per_chip=experts_per_chip,
        model_cfg=model_cfg,
        routed_expert_weights_dtype=DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE,
        is_first_rank=True,
        is_last_rank=True,
        mtp_levels=K,
        kv_only_last_layer=False,
        hf_config=config,
        tp_factor=tp,
    )
    logger.info(f"[hy4 mtp] trunk cache {cache}: building {wc.format_missing(missing) if missing else 'nothing'}")
    trunk_sd = (
        hc.load_hy4_trunk_state_dict(
            smoke._checkpoint_dir(), config, model_cfg, missing=missing, first_layer_idx=0, num_layers=TRUNK
        )
        if missing
        else {}
    )
    mtp_cache = MTP_CACHE_ROOT / f"hy4_preview_{'bh' if is_blackhole() else 'wh'}_{ttnn.get_num_devices()}dev"
    mtp_cache = mtp_cache / f"{sp}x{tp}_L{TRUNK}"
    mtp_cache.mkdir(parents=True, exist_ok=True)
    init_checker(mtp_cache)
    mtp_cached = TtMTPPredictor.check_cache_complete(
        mtp_cache, TRUNK, cache_name_prefix=MTP_CACHE_PREFIX, experts_per_chip=experts_per_chip, model_cfg=model_cfg
    )
    init_checker(cache)
    logger.info(f"[hy4 mtp] MTP cache {mtp_cache}: {'present' if mtp_cached else 'building from model.mtp_layers.0'}")
    mtp_sd = (
        {"mtp": {}, "layer": {}}
        if mtp_cached
        else hc.load_hy4_mtp_state_dict(smoke._checkpoint_dir(), config, model_cfg, TRUNK)
    )

    common = dict(
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        dispatch_buffer_capacity_factor=8,
        is_chunked=True,
        max_seq_len=SEQ_CACHE,
        slot_num=1,
        routing_use_l1_small_for_semaphores=True,
        tp_shard_kv=True,
        sparse_kv_cache_format=fmt,
    )
    predictor = TtMTPPredictor(
        mesh_device,
        config,
        model_cfg,
        mtp_sd,
        MTPConfig.from_hf_config(config, num_levels=K),
        seq_len=S,
        num_levels=K,
        layer_idx=TRUNK,
        first_cache_slot=TRUNK,
        weight_cache_path=mtp_cache,
        cache_name_prefix=MTP_CACHE_PREFIX,
        layer_num=TRUNK + K,
        **common,
    )
    assert not predictor.module.layer.ihc, "the MTP layer must be built without iHC"
    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        model_cfg=model_cfg,
        state_dict=trunk_sd,
        num_layers=TRUNK,
        seq_len=S,
        weight_cache_path=cache,
        is_balanced=False,
        mtp_predictor=predictor,
        **common,
    )
    assert transformer.ihc_head is not None and transformer.norm is not None
    del trunk_sd, mtp_sd

    kvpe_cache = init_mla_kv_cache(
        cache_format=fmt,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=SEQ_CACHE,
        mesh_shape=list(mesh_device.shape),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_kvpe_cache_layers=transformer.num_kvpe_cache_layers,
        num_users=1,
    )
    index_kv_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.index_head_dim,
        mesh_device=mesh_device,
        seq_len=SEQ_CACHE,
        mesh_shape=list(mesh_device.shape),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_kvpe_cache_layers=full_indexer_rank(config, TRUNK + K),
        num_users=1,
        dtype=ttnn.bfloat8_b,
    )

    meta = json.loads((TRACE / "metadata.json").read_text())
    prompt = list(meta["token_ids"]) + [int(meta["next_token_id"])]
    stream, real_len, provided = mtp_chunk_stream(prompt, 0, S, n_mtp, K, actual_isl=len(prompt))
    assert (real_len, provided) == (S, K), (real_len, provided)

    ckpt = hc.Hy4Checkpoint(smoke._checkpoint_dir())
    sites = _SiteRecorder(monkeypatch, mesh_device, sp_axis, {i: hy4_ihc.read_layer_ihc(ckpt, i) for i in range(TRUNK)})
    kv = smoke._PccRecorder(monkeypatch)
    monkeypatch.setattr(smoke, "NUM_LAYERS", TRUNK + K)

    h0 = {}
    real_run_mtp = transformer.run_mtp

    def capture_h0(h_normed, *a, **kw):
        h0["h"] = _from_device(h_normed, mesh_device).float()[0, 0]
        return real_run_mtp(h_normed, *a, **kw)

    monkeypatch.setattr(transformer, "run_mtp", capture_h0)
    captured = {}
    # The fused projection's output, read to host the moment it is produced (res.x is read after the whole run).
    fused_cls = type(predictor.module.fused)
    real_fused = fused_cls.forward

    def fused_and_capture(self_, embed, hidden):
        x = real_fused(self_, embed, hidden)
        ttnn.synchronize_device(mesh_device)
        captured["x_early"] = _from_device(x, mesh_device).float()[0, 0]
        return x

    monkeypatch.setattr(fused_cls, "forward", fused_and_capture)

    union = _mtp_union(transformer, stream, n_mtp, K, 0, real_len)
    transformer.forward(
        union.trunk,
        kvpe_cache,
        actual_isl=real_len,
        actual_start=0,
        actual_end=real_len,
        cache_user_id=0,
        index_kv_cache=index_kv_cache,
        mtp_union=union,
        provided_levels=provided,
        input_is_embedded=True,
        on_mtp_complete=lambda out, gen: captured.update(out=out, gen=list(gen)),
    )
    ttnn.synchronize_device(mesh_device)
    union.deallocate()
    assert captured.get("gen") == [], f"no level should generate with the lookahead provided: {captured.get('gen')}"

    ref = torch.load(TRACE / "mtp_tensors.pt", map_location="cpu", weights_only=True)
    res = captured["out"]
    mtp_rows = {
        "h0 = norm(hc_head(streams))": _pcc(ref["h0"], h0["h"]),
        "MTP block input eh_proj(cat[enorm(e), hnorm(h0)])": _pcc(ref["mtp_in"], captured["x_early"]),
        "MTP block out": _pcc(ref["out"], _from_device(res.out[0], mesh_device).float()[0, 0]),
        "MTP final_layernorm out": _pcc(ref["normed"], _from_device(res.out_head_normed[0], mesh_device).float()[0, 0]),
    }

    if os.environ.get("HY4_MTP_DUMP"):
        torch.save(
            {
                "x_early": captured["x_early"],
                "x_late": _from_device(res.x[0], mesh_device).float()[0, 0],
                "h0": h0["h"],
            },
            os.environ["HY4_MTP_DUMP"],
        )
    layout = variant.prefill_trace_layout
    chunked._record_kv_cache_pcc(
        TRACE, layout, kvpe_cache, mesh_device, sp, TRUNK + K, SEQ_CACHE, S, config.kv_lora_rank,
        tp_shard_kv=True, assert_threshold=None, sp_axis=sp_axis, chunk_size=S,
    )  # fmt: skip
    chunked._record_indexer_k_cache_pcc(
        TRACE, layout, index_kv_cache, mesh_device, sp, TRUNK + K, SEQ_CACHE, S, config,
        tp_shard_kv=True, sp_axis=sp_axis, chunk_size=S,
    )  # fmt: skip

    kv_rows = {r["layer"]: r for r in kv.table()}
    logger.info(
        "Hy4 prefill + iHC + MTP (layout glm_prefill_sp8_tp4), 5-layer trunk, MTP at slot 5, vs fp32 reference:"
    )
    logger.info("slot  kvpe_nope  kvpe_pe   index_k_rope  index_k_nope")
    for layer, r in sorted(kv_rows.items()):
        ik = f"{r['index_k_rope']:12.6f}  {r['index_k_nope']:12.6f}" if "index_k_rope" in r else f"{'-':>12}  {'-':>12}"
        logger.info(f"{layer:4d}  {r['kvpe_nope']:9.6f}  {r['kvpe_pe']:8.6f}  {ik}")
    for name, v in mtp_rows.items():
        logger.info(f"{name}: pcc={v:.6f}")
    logger.info("site      streams_in  y          streams_out  update")
    for name, r in sites.rows.items():
        logger.info(
            f"{name:8s}  {r.get('in', float('nan')):.6f}  {r.get('y', float('nan')):.6f}  "
            f"{r.get('out', float('nan')):.6f}     {r.get('update', float('nan')):.6f}"
        )
    if os.environ.get("HY4_IHC_REPORT"):
        Path(os.environ["HY4_IHC_REPORT"]).write_text(
            json.dumps(
                {"layout": hy4_layout.DEV_LAYOUT, "kv": list(kv_rows.values()), "mtp": mtp_rows, "sites": sites.rows},
                indent=1,
            )
        )
    predictor_ok = TRUNK in kv_rows and "index_k_rope" in kv_rows[TRUNK]
    assert predictor_ok, f"no KV / index-K recorded for the MTP slot {TRUNK}: {sorted(kv_rows)}"
    bad = [f"slot {l} {k}={v:.4f}" for l, r in kv_rows.items() for k, v in r.items() if k != "layer" and v < PCC_FLOOR]
    bad += [
        f"{k}={v:.4f}"
        for k, v in mtp_rows.items()
        if v < (MTP_OUTPUT_FLOOR if k.startswith("MTP block out") or "final_layernorm" in k else PCC_FLOOR)
    ]
    assert not bad, f"below {PCC_FLOOR}: {bad}"
