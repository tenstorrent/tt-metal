# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-preview prefill with iHC (#65891): the first layers on device vs the HF hy_v4 golden.

Same driver and golden as ``test_hy4_prefill_device_smoke.py`` (``TtPrefillTransformer`` through the chunked driver,
one 4096-token chunk = the whole 4K golden, the dev / numerics layout ``glm_prefill_sp8_tp4`` on the 8x4 mesh), with the
model now running iHC: embedding -> 4 fp32 streams -> per layer, an iHC site around attention and one around the FFN /
MoE. Scored per layer:

  * the KVPE cache (nope / pe) and the index-key cache (full indexer layers), against the golden;
  * the iHC streams at every site against the fp32 CPU reference's streams (``HY4_IHC_REF_STREAMS``):
      - ``in``:     the streams entering the site,
      - ``y``:      the sublayer input sum_i pre_i x_i (the reference's own hc_pre on the reference streams),
      - ``out``:    the streams leaving the site,
      - ``update``: out - in, i.e. post_j * F(y): the sublayer output as the streams receive it (the residual would
                    otherwise hide a wrong sublayer).

The golden stores KV / index-K / top-k per layer but no residual streams; the reference streams are the same
``Hy4PrefillReference`` (reference/hy4/model.py, #65936) that wrote the golden, rerun with its ``_attention`` / ``_ffn``
hooked to save ``x4`` after each site: ``streams_embed.pt``, ``streams_L{i}_attn.pt``, ``streams_L{i}_ffn.pt``
(``[S, 4, 6144]`` fp32).

The last layer is built kv-only (as the driver and the runner do), so with ``HY4_IHC_LAYERS=5`` layers 0-3 are fully
scored and layer 4 contributes its KV. Weight-cache pieces the cache lacks (the ``ihc`` category, the learnable sink in
``mla``) are read from the checkpoint exactly as the runner's rebuild does (``missing_trunk_categories`` +
``load_hy4_trunk_state_dict``) and written to the cache while the model builds.

Env: ``HY4_HF_MODEL`` (checkpoint), ``TT_HY4_PREFILL_TTNN_CACHE`` / ``PREFILL_TTNN_CACHE`` (cache root),
``PREFILL_TRACE_DIR`` (golden), ``HY4_IHC_REF_STREAMS`` (reference streams dir), ``HY4_IHC_LAYERS`` (default 5),
``HY4_IHC_REPORT`` (optional JSON path), ``PREFILL_CHUNK_SIZE=4096``.
"""

from __future__ import annotations

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
from models.demos.deepseek_v3_d_p.tt.hy4 import ihc as hy4_ihc

chunked = smoke.chunked
NUM_LAYERS = int(os.environ.get("HY4_IHC_LAYERS", "5"))
REF_DIR = Path(os.environ.get("HY4_IHC_REF_STREAMS", "/mnt/tt-data/dwadhwani/hy4-golden/hy4-4k-v3-streams"))
PCC_FLOOR = 0.99  # KV / index-K / streams, every scored layer
# The update of an MoE site is the MoE output as the streams receive it: GLM-5.3's / #65923's MoE bar. Its input already
# carries the upstream layers' error, which top-8 routing amplifies (the module alone, on the reference input, is 0.9999).
MOE_UPDATE_FLOOR = 0.971

pytestmark = smoke.pytestmark + [
    pytest.mark.skipif(not (REF_DIR / "streams_embed.pt").is_file(), reason=f"no reference streams at {REF_DIR}")
]


def _pcc(a, b) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _ref_streams(name: str) -> torch.Tensor:
    return torch.load(REF_DIR / f"streams_{name}.pt", map_location="cpu", weights_only=True)


class _SiteRecorder:
    """Scores every iHC site as the block runs it (patched ``TtPrefillBlock._ihc_pre`` / ``_ihc_post``). Sites run in
    order L0.attn, L0.ffn, L1.attn, ...; a kv-only last layer runs only its attention hc_pre."""

    def __init__(self, monkeypatch, mesh_device, sp_axis, ihc_weights):
        from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
        from models.experimental.hy4_preview.reference.ihc import IHCConfig, IHCWrap

        self.rows: dict[str, dict] = {}
        self.k_pre = 0
        self.k_post = 0
        self.ref_in = None
        tp_axis = 1 - sp_axis
        n = 4

        def host(t):
            ttnn.synchronize_device(mesh_device)
            return hy4_ihc.streams_to_host(t, mesh_device, sp_axis=sp_axis, tp_axis=tp_axis, n=n)

        def host_hidden(t):
            dims = [None, None]
            dims[sp_axis], dims[tp_axis] = 2, 3
            return ttnn.to_torch(
                t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=mesh_device.shape)
            ).float()[0, 0]

        def site(k):
            return k // 2, ("attn", "ffn")[k % 2]

        def ref_before(layer, s):
            if s == "attn":
                return _ref_streams("embed" if layer == 0 else f"L{layer - 1}_ffn")
            return _ref_streams(f"L{layer}_attn")

        orig_pre, orig_post = TtPrefillBlock._ihc_pre, TtPrefillBlock._ihc_post

        def pre(wrap, streams):
            layer, s = site(self.k_pre)
            self.k_pre += 1
            y, post = orig_pre(wrap, streams)
            name = f"L{layer}.{s}"
            dev_in = host(streams)
            ref_in = ref_before(layer, s)
            w = ihc_weights[layer][s]
            ref = IHCWrap(IHCConfig()).eval()
            ref.fn.data, ref.base.data, ref.scale.data = w["fn"], w["base"], w["scale"]
            with torch.no_grad():
                ref_y, _ = ref.hc_pre(ref_in.unsqueeze(0))
            row = self.rows.setdefault(name, {})
            row["in"] = _pcc(ref_in, dev_in)
            row["y"] = _pcc(ref_y[0], host_hidden(y))
            self.ref_in, self.dev_in = ref_in, dev_in
            logger.info(f"[hy4 ihc] {name} in pcc={row['in']:.6f} y pcc={row['y']:.6f}")
            return y, post

        def post_(wrap, sublayer_out, streams, post, free_streams):
            layer, s = site(self.k_post)
            self.k_post += 1
            new = orig_post(wrap, sublayer_out, streams, post, free_streams)
            name = f"L{layer}.{s}"
            dev_out = host(new)
            ref_out = _ref_streams(f"L{layer}_{s}")
            row = self.rows.setdefault(name, {})
            row["out"] = _pcc(ref_out, dev_out)
            row["update"] = _pcc(ref_out - self.ref_in, dev_out - self.dev_in)
            row["max_abs_out"] = (ref_out - dev_out).abs().max().item()
            logger.info(f"[hy4 ihc] {name} out pcc={row['out']:.6f} update pcc={row['update']:.6f}")
            return new

        monkeypatch.setattr(TtPrefillBlock, "_ihc_pre", staticmethod(pre))
        monkeypatch.setattr(TtPrefillBlock, "_ihc_post", staticmethod(post_))


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
def test_hy4_prefill_ihc_layers_vs_golden(
    variant, config_only, mesh_device, device_params, num_links, sp_axis, monkeypatch
):
    from models.demos.deepseek_v3_d_p.reference.hy4_config import Hy4Config
    from models.demos.deepseek_v3_d_p.tt.hy4 import layout as hy4_layout
    from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
    from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE
    from models.demos.deepseek_v3_d_p.tt.runners import hy4_checkpoint as hc
    from models.demos.deepseek_v3_d_p.tt.runners import weight_categories as wc
    from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

    monkeypatch.setenv(hy4_layout.LAYOUT_ENV, hy4_layout.DEV_LAYOUT)
    hy4_layout.resolve_layout(smoke.MESH[sp_axis], smoke.MESH[1 - sp_axis], sp_axis, Hy4Config.NUM_ATTENTION_HEADS)
    monkeypatch.setattr(smoke, "NUM_LAYERS", NUM_LAYERS)  # the KV / index-K recorders count layers from it

    cache = smoke._mesh_cache_dir(mesh_device.get_num_devices())
    if not cache.is_dir():
        pytest.skip(f"no Hy4 weight cache at {cache}: build it with the prefill runner (PREFILL_MODEL=hy4_preview)")
    stale = hc.stale_mla_cache_layers(cache, range(NUM_LAYERS))
    assert not stale, f"Hy4 MLA/indexer cache at {cache} for layers {stale} predates format {hc.HY4_MLA_CACHE_FORMAT}"

    # The runner's rebuild: the categories this cache lacks, read from the checkpoint, written while the model builds.
    sp, tp = smoke.MESH[sp_axis], smoke.MESH[1 - sp_axis]
    missing = wc.missing_trunk_categories(
        cache,
        num_layers=NUM_LAYERS,
        first_layer_idx=0,
        experts_per_chip=Hy4Config.NUM_ROUTED_EXPERTS // (sp * tp),
        model_cfg=variant.model_config,
        routed_expert_weights_dtype=DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE,
        is_first_rank=True,
        is_last_rank=True,
        mtp_levels=0,
        kv_only_last_layer=True,
        hf_config=config_only,
        tp_factor=tp,
    )
    # Never the routed experts here (tens of GB from the checkpoint): build those with the prefill runner.
    assert wc.ROUTED_EXPERTS not in missing, f"cache lacks the routed experts: {missing}"
    partial = (
        hc.load_hy4_trunk_state_dict(
            smoke._checkpoint_dir(),
            config_only,
            variant.model_config,
            missing=missing,
            first_layer_idx=0,
            num_layers=NUM_LAYERS,
        )
        if missing
        else {}
    )
    logger.info(f"[hy4 ihc] cache {cache}: building {wc.format_missing(missing) if missing else 'nothing'}")
    orig_acquire = chunked.acquire_transformer

    def acquire(*a, **kw):
        kw["state_dict"] = partial
        return orig_acquire(*a, **kw)

    monkeypatch.setattr(chunked, "acquire_transformer", acquire)
    orig_check = chunked.TtPrefillTransformer.check_cache_complete
    monkeypatch.setattr(
        chunked.TtPrefillTransformer,
        "check_cache_complete",
        staticmethod(lambda *a, **kw: True if missing else orig_check(*a, **kw)),
    )

    ckpt = hc.Hy4Checkpoint(smoke._checkpoint_dir())
    ihc_weights = {i: hy4_ihc.read_layer_ihc(ckpt, i) for i in range(NUM_LAYERS)}
    sites = _SiteRecorder(monkeypatch, mesh_device, sp_axis, ihc_weights)
    kv = smoke._PccRecorder(monkeypatch)

    chunked.run_chunked_transformer_updated(
        variant,
        config_only,
        mesh_device,
        cache.parent,
        NUM_LAYERS,
        1,
        GateComputeMode.DEVICE_FP32,
        num_links,
        per_axis_topology(device_params["fabric_config"]),
        1,
        routing_use_l1_small_for_semaphores=True,
        preload_isl=0,
        check_pcc=True,
        check_layer_pcc=False,
        use_trace=False,
        seq_cache=2 * smoke.CHUNK_SIZE,
        kv_cache_format=chunked.GLM_KV_CACHE_FORMAT,
        sp_axis=sp_axis,
        chunk_size=smoke.CHUNK_SIZE,
    )

    kv_rows = {r["layer"]: r for r in kv.table()}
    logger.info("Hy4 prefill with iHC vs golden (layout glm_prefill_sp8_tp4: SP 8 on axis 0, TP 4 on axis 1):")
    logger.info("layer  kvpe_nope  kvpe_pe   index_k_rope  index_k_nope")
    for layer, r in sorted(kv_rows.items()):
        ik = f"{r['index_k_rope']:12.6f}  {r['index_k_nope']:12.6f}" if "index_k_rope" in r else f"{'-':>12}  {'-':>12}"
        logger.info(f"{layer:5d}  {r['kvpe_nope']:9.6f}  {r['kvpe_pe']:8.6f}  {ik}")
    logger.info("site      streams_in  y          streams_out  update")
    for name, r in sites.rows.items():
        logger.info(
            f"{name:8s}  {r.get('in', float('nan')):.6f}  {r.get('y', float('nan')):.6f}  "
            f"{r.get('out', float('nan')):.6f}     {r.get('update', float('nan')):.6f}"
        )
    if os.environ.get("HY4_IHC_REPORT"):
        Path(os.environ["HY4_IHC_REPORT"]).write_text(
            json.dumps({"layout": hy4_layout.DEV_LAYOUT, "kv": list(kv_rows.values()), "sites": sites.rows}, indent=1)
        )

    bad = [
        f"L{layer} {k}={v:.4f}" for layer, r in kv_rows.items() for k, v in r.items() if k != "layer" and v < PCC_FLOOR
    ]

    def floor(name, k):
        layer = int(name[1 : name.index(".")])
        moe = name.endswith(".ffn") and layer >= Hy4Config.NUM_DENSE_LAYERS
        return MOE_UPDATE_FLOOR if (moe and k == "update") else PCC_FLOOR

    bad += [
        f"{name} {k}={v:.4f}"
        for name, r in sites.rows.items()
        for k, v in r.items()
        if k != "max_abs_out" and v < floor(name, k)
    ]
    assert len(kv_rows) == NUM_LAYERS, f"KV recorded for layers {sorted(kv_rows)}"
    assert not bad, f"below {PCC_FLOOR} (MoE update: {MOE_UPDATE_FLOOR}): {bad}"
