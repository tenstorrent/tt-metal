# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic (tt_metal_tracker-q171.2.11): where do eager host-scalar and metadata forwards of the tier-1 reduced
stack differ on a partial chunk?

q171.2.6 saw host-scalar (``actual_isl=end-start``) and metadata (``actual_isl=None``) eager forwards first differ at
the first MoE layer's output on slot 1's 96-token chunk. TtMoe enables padding awareness only when ``actual_isl`` is
not None, on both paths (production's traced forward passes ``actual_isl=chunk_size`` as that guard), so the two
paths compared there also differed in padding awareness. This test runs the 96-token chunk four ways and logs, per
pair, whether the first MoE call's input, gate scores, gate indices and output, and every step's output, are equal on
the chunk's real rows:

* ``host96``:   eager host scalars, actual_isl=96 (production eager)
* ``meta_on``:  metadata, actual_isl=CHUNK (production traced guard; padding config built on device)
* ``meta_off``: metadata, actual_isl=None (the q171.2.6 harness; padding awareness off)
* ``host_off``: eager host scalars, actual_isl=None (padding awareness off)

Real rows: the chunk starts at 0, so SP chip 0's first 96 local rows are positions 0..95 (rotated layout).

Findings (b06u02, 8x4, glm54-orch/logs/cachediff/diag{1..4}-*.log):
* ``host96`` == ``meta_on`` bit for bit at every tap: with the production guard, host-scalar and metadata agree.
* ``meta_off`` == ``host_off``; vs ``host96`` the gate (scores, indices) is equal but the routed experts' outputs for
  real tokens differ (MoE output max 0.10), because the padding rows join the experts' token counts.
* ``composite_only``: all four paths agree bit for bit. ``fused_band`` (every expert on moe_fused_swiglu): they still
  differ, for exactly the experts whose count changed. moe_fused_swiglu's per-token output depends on its expert's
  count (test_moe_fused_swiglu_count_dependence.py), and the 320 split switches ops by count as well.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.chunk_layout import pack_forward_tokens
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import moe_fabric_payload, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.glm54.test_glm54_loop_device import CHUNK, TOKEN_SEED, _build, _scalar
from models.demos.deepseek_v3_d_p.tests.glm54_weights import (
    GLM54ReducedConfig,
    glm54_reduced_config,
    glm54_tier1_weights,
)
from models.demos.deepseek_v3_d_p.tt.mla.rope import ChunkMetadata
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import mesh_shard_dims
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import TtMoe
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import TtMoEGatePrefill
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup
from models.demos.deepseek_v3_d_p.tt.moe.tt_reduce import TtReduceModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import TtSharedExpert
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_3 import GLM53Adapter
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP

REAL = 96
PATHS = ("host96", "meta_on", "meta_off", "host_off")
PAIRS = (("host96", "meta_on"), ("host96", "meta_off"), ("meta_off", "host_off"), ("host96", "host_off"))


def _dev0(t: ttnn.Tensor) -> torch.Tensor:
    """Device 0's (SP chip 0, TP chip 0) shard on host."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=moe_fabric_payload(GLM54ReducedConfig), l1_small_size=GLM53Adapter.l1_small_size
            ),
            2,
            id="torus-xy-8x4",
        )
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "routed_expert_split",
    ["model", "composite_only", "fused_band"],
    ids=["split320", "composite_only", "fused_band"],
)
@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA ops are Blackhole-only")
@pytest.mark.timeout(0)
def test_glm54_moe_padding_diag(mesh_device, device_params, num_links, monkeypatch, routed_expert_split):
    # The routed experts run on two ops split by an expert's token count (ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD, 320
    # from GLM-5.3): "model" keeps the split, "composite_only" sends every expert to unified_routed_expert_moe, and
    # "fused_band" puts every expert of this chunk in the fused band (threshold 5000: above every count here, max 3138,
    # and below max_tokens 5120, which the one-program overlap requires).
    if routed_expert_split == "composite_only":
        monkeypatch.setattr(GLM54ReducedConfig, "ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD", None)
    elif routed_expert_split == "fused_band":
        monkeypatch.setattr(GLM54ReducedConfig, "ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD", 5000)
    logger.info(
        f"routed-expert split: {routed_expert_split}, threshold "
        f"{getattr(GLM54ReducedConfig, 'ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD', None)}"
    )
    sp, chunk_local, max_seq = mesh_device.shape[0], PREFILL_CHUNK_TOKENS_PER_CHIP, 3 * CHUNK
    config = glm54_reduced_config(max_seq)
    config.loop_window_format = "bf16"
    weights = glm54_tier1_weights(config)
    transformer, kvpe_cache, index_cache = _build(
        mesh_device, device_params, num_links, config, weights, num_slots=2, max_seq=max_seq
    )
    prompt = torch.randint(0, config.vocab_size, (CHUNK,), generator=torch.Generator().manual_seed(TOKEN_SEED))

    def tokens(n_real: int):
        chunk = torch.zeros(CHUNK, dtype=torch.int64)
        chunk[:n_real] = prompt[:n_real]
        return ttnn.from_torch(
            torch.tensor(pack_forward_tokens([chunk.tolist()], [0], sp), dtype=torch.int64).reshape(sp, 1, chunk_local),
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=mesh_shard_dims(0, 0)
            ),
        )

    # Taps of the first MoE call of a forward: its input, the gate's (scores, indices), its output.
    rec = {}
    moe_forward, gate_forward = TtMoe.forward, TtMoEGatePrefill.forward

    def moe_tap(self, x, *args, **kwargs):
        first = "moe_in" not in rec
        if first:
            rec["moe_in"] = _dev0(x)
            rec["actual_isl"] = kwargs.get("actual_isl")
        out = moe_forward(self, x, *args, **kwargs)
        if first:
            rec["moe_out"] = _dev0(out[0] if isinstance(out, tuple) else out)
            rec["capacity"] = self.dispatch_module.max_dispatch_buffer_token_size
        return out

    def gate_tap(self, x, *args, **kwargs):
        out = gate_forward(self, x, *args, **kwargs)
        if "scores" not in rec:
            rec["scores"], rec["indices"] = _dev0(out[0]), _dev0(out[1])
            rec["padding_config"] = kwargs.get("padding_config") is not None
        return out

    shared_forward, reduce_forward, routing_forward = (
        TtSharedExpert.forward,
        TtReduceModule.forward,
        TtMoERoutingSetup.forward,
    )

    def shared_tap(self, x):
        out = shared_forward(self, x)
        rec.setdefault("shared_out", _dev0(out))
        return out

    def reduce_tap(self, combine_output, *args, **kwargs):
        out = reduce_forward(self, combine_output, *args, **kwargs)
        if "combine" not in rec:
            rec["combine"], rec["reduce_out"] = _dev0(combine_output), _dev0(out)
        return out

    def routing_tap(self, *args, **kwargs):
        out = routing_forward(self, *args, **kwargs)
        if "token_counts" not in rec:
            rec["token_counts"] = _dev0(out[1]).flatten().to(torch.int64)
            rec["region_offsets"] = _dev0(out[2]).flatten().to(torch.int64)
        return out

    monkeypatch.setattr(TtMoe, "forward", moe_tap)
    monkeypatch.setattr(TtMoEGatePrefill, "forward", gate_tap)
    monkeypatch.setattr(TtSharedExpert, "forward", shared_tap)
    monkeypatch.setattr(TtReduceModule, "forward", reduce_tap)
    monkeypatch.setattr(TtMoERoutingSetup, "forward", routing_tap)

    def run(path: str) -> dict:
        rec.clear()
        steps = []
        hidden = dict(
            on_layer_hidden=lambda layer, x: steps.append(
                (
                    layer,
                    ttnn.to_torch(
                        x,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)
                        ),
                    ),
                )
            )
        )
        slot = 1
        if path.startswith("meta"):
            meta = ChunkMetadata(*(_scalar(mesh_device, v) for v in (slot, 0, REAL)))
            transformer.forward(
                tokens(REAL),
                kvpe_cache,
                actual_isl=CHUNK if path == "meta_on" else None,
                actual_start=None,
                actual_end=None,
                cache_user_id=0,
                index_kv_cache=index_cache,
                metadata=meta,
                **hidden,
            )
        else:
            transformer.forward(
                tokens(REAL),
                kvpe_cache,
                actual_isl=REAL if path == "host96" else None,
                actual_start=0,
                actual_end=REAL,
                cache_user_id=slot,
                index_kv_cache=index_cache,
                **hidden,
            )
        ttnn.synchronize_device(mesh_device)
        out = dict(rec)
        out["steps"] = steps
        return out

    runs = {p: run(p) for p in PATHS}
    again = run("host96")  # determinism of one path, same state
    for p in PATHS:
        r = runs[p]
        logger.info(
            f"[{p}] first MoE call: actual_isl={r['actual_isl']} padding_config={r['padding_config']} "
            f"moe_in {list(r['moe_in'].shape)} indices {list(r['indices'].shape)}"
        )

    def cmp(a: torch.Tensor, b: torch.Tensor, rows: slice) -> str:
        x, y = a[..., rows, :], b[..., rows, :]
        if torch.equal(x, y):
            return "equal"
        d = (x.double() - y.double()).abs()
        return f"DIFF max {float(d.max()):.4g} rows {int((d.reshape(-1, d.shape[-1]).amax(-1) > 0).sum())}"

    for p in PATHS:
        r = runs[p]
        logger.info(
            f"[{p}] token counts (device 0 view) sum {int(r['token_counts'].sum())} max {int(r['token_counts'].max())} "
            f"capacity {r['capacity']}; {len(r['token_counts'])} count entries, first 64 {r['token_counts'].tolist()[:64]}; "
            f"combine {list(r['combine'].shape)} reduce_out {list(r['reduce_out'].shape)}"
        )

    c0, c1 = runs["host96"]["combine"][:, :1, :REAL], again["combine"][:, :1, :REAL]
    unwritten = ((c0.double() - c1.double()).abs().amax(-1).nan_to_num(nan=1.0) > 0).reshape(1, REAL, -1)
    unwritten |= ~torch.isfinite(c0.double()).all(-1).reshape(1, REAL, -1)
    logger.info(f"unwritten (remote-expert) combine slots on device 0: {int(unwritten.sum())} of {unwritten.numel()}")

    for a, b in PAIRS + (("host96", "host96 again"),):
        ra, rb = runs[a], (again if b == "host96 again" else runs[b])
        logger.info(f"[{a} vs {b}] moe_in {cmp(ra['moe_in'], rb['moe_in'], slice(0, REAL))}")
        logger.info(f"[{a} vs {b}] gate scores {cmp(ra['scores'], rb['scores'], slice(0, REAL))}")
        ia, ib = ra["indices"][..., :REAL, :], rb["indices"][..., :REAL, :]
        same_sets = torch.equal(ia.sort(-1).values, ib.sort(-1).values)
        logger.info(f"[{a} vs {b}] gate indices {'equal' if torch.equal(ia, ib) else 'DIFF'} (same sets {same_sets})")
        logger.info(f"[{a} vs {b}] shared_out {cmp(ra['shared_out'], rb['shared_out'], slice(0, REAL))}")
        ca, cb = ra["combine"][..., :REAL, :, :], rb["combine"][..., :REAL, :, :]  # [1, dg, seq, topk, emb]
        per_slot = (ca.double() - cb.double()).abs().amax(-1).reshape(-1, REAL, ca.shape[-2])  # [dg, token, k]
        logger.info(
            f"[{a} vs {b}] combine rows differing per dispatch-group index: {(per_slot > 0).flatten(1).sum(-1).tolist()}"
        )
        per_slot = per_slot[:1]  # dispatch-group index 0: this chip's own rows (tokens 0..95 real)
        # Slots of experts not on this chip are never written (the reduce skips them via the dispatch table), so they hold
        # whatever the buffer held: mask the slots that differ between two identical host96 runs.
        per_slot = torch.where(unwritten, torch.zeros_like(per_slot), per_slot.nan_to_num(nan=float("inf")))
        bad = (per_slot > 0).nonzero().tolist()
        logger.info(
            f"[{a} vs {b}] combine (per token, per top-k slot, real tokens): "
            f"{'equal' if not bad else f'DIFF in {len(bad)} of {per_slot.numel()} slots, max {float(per_slot.max()):.4g}'}"
        )
        if bad:
            ids = ra["indices"][:REAL]
            hit = sorted({int(ids[t, k]) for _, t, k in bad})
            logger.info(f"[{a} vs {b}] differing slots' experts {hit[:40]}; first slots (dg, token, k) {bad[:10]}")
        logger.info(f"[{a} vs {b}] reduce_out {cmp(ra['reduce_out'], rb['reduce_out'], slice(0, REAL))}")
        logger.info(f"[{a} vs {b}] moe_out {cmp(ra['moe_out'], rb['moe_out'], slice(0, REAL))}")
        for (la, xa), (lb, xb) in zip(ra["steps"], rb["steps"]):
            logger.info(f"[{a} vs {b}] step layer {la}: real rows {cmp(xa, xb, slice(0, REAL))}")
    # The padding rows' routing per path (indices past the real rows on SP chip 0).
    for p in PATHS:
        pad = runs[p]["indices"][..., REAL:, :]
        logger.info(f"[{p}] pad-row indices: unique {torch.unique(pad).tolist()[:12]}")
