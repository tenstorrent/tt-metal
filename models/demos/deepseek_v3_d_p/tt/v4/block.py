# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``TtV4PrefillBlock``: one DeepSeek-V4-Flash decoder layer for the pure-ttnn prefill.

    streams -> attn_hc (pre) -> input_layernorm(collapsed) -> {TtSWA | TtCSA | TtHCA} -> attn_hc.mix
            -> ffn_hc (pre) -> post_attention_layernorm(collapsed) -> TtMoe -> ffn_hc.mix -> streams

(reference ``DeepseekV4DecoderLayer.forward``). The residual is a LIST of 4 streams ``[1, 1, S_l, D_l]`` (bf16,
TP-sharded on D, SP-sharded on S). Per (slot) the block keeps the attention module's chunk state (window carry,
entry count, compressor priors); ``forward`` takes the engine's ``(slot, actual_start, actual_end)`` and mirrors the
attention KV into the engine-owned unified caches (``tt/v4/kv_contract.py``). Hooks kept from the V3 block:
``on_layer_complete(global_layer_idx)`` after the KV write (host callback, flushed), ``on_layer_hidden(idx,
streams)`` at the end, ``kv_only`` (last layer of a KV-only worker: attention + write only, returns None).

Weights arrive under the reference module's names (``tt/v4/weights/hf_names.py``); the routed experts as a list
of per-expert HF-orientation dicts under ``"__experts__"``.

``TtV4Block`` (bottom of this file) is main's public DeepSeek-V4 decoder block (tt-metal #56723, exported by
``tt/v4/__init__.py``): ``TtMHCWrap`` hyper-connection residual, ``tt/mla`` sliding-window / heavily-compressed attention
that owns its own KV state, and ``TtPrefillBlock._build_moe``. It is an independent implementation with its own callers
(``tests/v4/test_block.py``, ``tests/v4/test_block_chunked.py``); ``TtV4PrefillBlock`` above is the one the V4-Flash /
V4.1 production prefill runs.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch

_TRACED_RAGGED = os.environ.get("PREFILL_TRACED_RAGGED", "0") == "1"  # DS4F-0268: ragged final chunk on the islands
# DS4F-0300: a ragged FIRST chunk (every prompt shorter than one chunk) replays islands A + B like a full chunk 0; only its
# attention runs eager (trace_ready is False at kv_actual 0). Without it the whole layer runs eager (~5.6 s at 43 layers).
_TRACED_RAGGED0 = os.environ.get("PREFILL_TRACED_CHUNK0_RAGGED", "1") == "1"  # default on: bit-exact (DS4F-0300 v1)
# DS4F-0300: chunk 0's attention on a second captured A2 island per slot (the modules' forward_attn(chunk0=True): CSA repacks
# its index rows, HCA / SWA mask the carry columns) instead of the eager path (trace_ready is False at kv_actual 0).
_TRACED_CHUNK0 = (
    os.environ.get("PREFILL_TRACED_CHUNK0", "1") == "1"
)  # default on (DS4F-0300: closer to the torch reference)
# DS4F-0300: the shortest ragged chunk (chunk 0 or a final tail) that runs its attention on the islands. Was 256 (= 32 x sp, the
# EAGER path's alignment step -- no traced-path constraint behind it): every prompt < 256 tokens paid ~1.5 s of eager attention issue
# and every tail < 256 tokens ran the whole layer eager. 256 restores the old floor (A/B).
_TRACED_MIN_LEN = max(1, int(os.environ.get("PREFILL_TRACED_MIN_LEN", "1")))
# DS4F-0300 measurement only (PREFILL_ISLAND_TIMING=1): synchronize after every island / glue step of a traced layer and add the
# wall time to ISLAND_TIMES[(kind, step)]; the runtime logs and clears it per chunk. Serialises host and device -- timing runs only.
_ISLAND_TIMING = os.environ.get("PREFILL_ISLAND_TIMING", "0") == "1"
ISLAND_TIMES: dict = {}

# ---- main's TtV4Block dependencies (tt-metal #56723). tt/mla's sliding-window attention (whose name, TtSWA, collides
# with tt/v4/attention/swa.TtSWA) and ``TtPrefillBlock`` are imported where TtV4Block is built: they pull in
# utils.kv_cache_utils (deepseek_v3_b1 / dflash / the migration runner) and ttMLA / tt_ffn, which the production
# TtV4PrefillBlock import path keeps lazy.
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import MHCConfig
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import TtMHCWrap
from models.demos.deepseek_v3_d_p.tt.mla.heavily_compressed_attention import TtHCA, TtHCACompressor
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.v4.attention.csa import TtCSA, TtCSACompressor, TtCSAIndexer
from models.demos.deepseek_v3_d_p.tt.v4.attention.swa import TtSWA
from models.demos.deepseek_v3_d_p.tt.v4.hyper_connection import TtHyperConnection
from models.demos.deepseek_v3_d_p.tt.v4.layer_kinds import CSA, HCA, SLIDING, layer_kinds
from models.demos.deepseek_v3_d_p.tt.v4.moe import build_v4_moe, is_hash_layer
from models.demos.deepseek_v3_d_p.tt.v4.trace_island import TraceIsland, copy_into

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig

    from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TopologyArg

HC = 4


def build_attention(
    mesh_device,
    cfg,
    layer_idx: int,
    w: dict,
    rotary_emb,
    *,
    sp_axis=0,
    tp_axis=1,
    topology=ttnn.Topology.Linear,
    weight_cache_path=None,
):
    """The layer's attention module from reference-named weights. ``weight_cache_path`` (the runner's per-mesh
    .tensorbin dir) caches the projection weights under ``layer_{idx}.attn.*`` / ``.compressor.*`` / ``.indexer.*``."""
    kind = layer_kinds(cfg)[layer_idx]
    mesh = dict(sp_axis=sp_axis, tp_axis=tp_axis, topology=topology)

    def cached(name):
        return dict(weight_cache_path=weight_cache_path, cache_name_prefix=f"layer_{layer_idx}.{name}")

    common = dict(
        q_a_proj_weight=w["self_attn.q_a_proj.weight"],
        q_a_norm_weight=w["self_attn.q_a_norm.weight"],
        q_b_proj_weight=w["self_attn.q_b_proj.weight"],
        kv_proj_weight=w["self_attn.kv_proj.weight"],
        kv_norm_weight=w["self_attn.kv_norm.weight"],
        sinks=w["self_attn.sinks"],
        o_a_proj_weight=w["self_attn.o_a_proj.weight"],
        o_b_proj_weight=w["self_attn.o_b_proj.weight"],
        rotary_emb=rotary_emb,
        num_heads=cfg.num_attention_heads,
        head_dim=cfg.head_dim,
        rope_head_dim=cfg.qk_rope_head_dim,
        sliding_window=cfg.sliding_window,
        o_groups=cfg.o_groups,
        rms_norm_eps=cfg.rms_norm_eps,
        **mesh,
    )
    common.update(cached("attn"))
    if kind == SLIDING:
        return TtSWA(mesh_device, compressor=None, rope_layer_type="main", **common)
    if kind == HCA:
        comp = TtHCACompressor(
            mesh_device,
            kv_proj_weight=w["self_attn.compressor.kv_proj.weight"],
            gate_proj_weight=w["self_attn.compressor.gate_proj.weight"],
            position_bias=w["self_attn.compressor.position_bias"],
            kv_norm_weight=w["self_attn.compressor.kv_norm.weight"],
            head_dim=cfg.head_dim,
            compress_rate=cfg.compress_rates[HCA],
            rope_head_dim=cfg.qk_rope_head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=cfg.rms_norm_eps,
            **mesh,
            **cached("compressor"),
        )
        return TtHCA(mesh_device, compressor=comp, rope_layer_type="compress", **common)
    assert kind == CSA, kind
    rate = cfg.compress_rates[CSA]
    comp = TtCSACompressor(
        mesh_device,
        kv_proj_weight=w["self_attn.compressor.kv_proj.weight"],
        gate_proj_weight=w["self_attn.compressor.gate_proj.weight"],
        position_bias=w["self_attn.compressor.position_bias"],
        kv_norm_weight=w["self_attn.compressor.kv_norm.weight"],
        head_dim=cfg.head_dim,
        compress_rate=rate,
        rope_head_dim=cfg.qk_rope_head_dim,
        rotary_emb=rotary_emb,
        rms_norm_eps=cfg.rms_norm_eps,
        **mesh,
        **cached("compressor"),
    )
    icomp = TtCSACompressor(
        mesh_device,
        kv_proj_weight=w["self_attn.compressor.indexer.kv_proj.weight"],
        gate_proj_weight=w["self_attn.compressor.indexer.gate_proj.weight"],
        position_bias=w["self_attn.compressor.indexer.position_bias"],
        kv_norm_weight=w["self_attn.compressor.indexer.kv_norm.weight"],
        head_dim=cfg.index_head_dim,
        compress_rate=rate,
        rope_head_dim=cfg.qk_rope_head_dim,
        rotary_emb=rotary_emb,
        rms_norm_eps=cfg.rms_norm_eps,
        **mesh,
        **cached("indexer.compressor"),
    )
    indexer = TtCSAIndexer(
        mesh_device,
        compressor=icomp,
        q_b_proj_weight=w["self_attn.compressor.indexer.q_b_proj.weight"],
        weights_proj_weight=w["self_attn.compressor.indexer.scorer.weights_proj.weight"],
        n_heads=cfg.index_n_heads,
        head_dim=cfg.index_head_dim,
        rope_head_dim=cfg.qk_rope_head_dim,
        topk=cfg.index_topk,
        **mesh,
        **cached("indexer"),
    )
    return TtCSA(mesh_device, compressor=comp, indexer=indexer, **common)


class TtV4PrefillBlock(LightweightModule):
    def __init__(
        self,
        mesh_device,
        cfg,
        layer_idx: int,
        w: dict,
        *,
        rotary_emb,
        seq_len_per_chip: int,
        sp_axis: int = 0,
        tp_axis: int = 1,
        topology=ttnn.Topology.Linear,
        num_links: int = 2,
        kv_only: bool = False,
        num_routed_experts: Optional[int] = None,
        dispatch_buffer_capacity_factor: int = 2,
        weight_cache_path=None,
    ):
        super().__init__()
        self.mesh_device, self.cfg, self.layer_idx = mesh_device, cfg, int(layer_idx)
        self.kind = layer_kinds(cfg)[layer_idx]
        self.kv_only = bool(kv_only)
        self.hash_layer = is_hash_layer(cfg, layer_idx)
        self.sp_axis, self.tp_axis = sp_axis, tp_axis
        D = cfg.hidden_size
        hc = dict(
            hidden=D,
            rms_eps=cfg.rms_norm_eps,
            hc_eps=cfg.hc_eps,
            sinkhorn_iters=cfg.hc_sinkhorn_iters,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
        )
        self.attn_hc = TtHyperConnection(
            mesh_device, fn=w["attn_hc.fn"], base=w["attn_hc.base"], scale=w["attn_hc.scale"], **hc
        )
        self.input_norm = TtDistributedRmsNorm(
            mesh_device=mesh_device,
            emb_dim=D,
            epsilon=cfg.rms_norm_eps,
            torch_weight=w["input_layernorm.weight"],
            cluster_axis=tp_axis,
            num_links=num_links,
            topology=topology,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.input_norm",
        )
        self.attn = build_attention(
            mesh_device,
            cfg,
            layer_idx,
            w,
            rotary_emb,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            topology=topology,
            weight_cache_path=weight_cache_path,
        )
        self.states: dict = {}
        self._islands = None  # (A, B, S_in, y_buf, ids_buf) once enable_trace_islands ran
        if self.kv_only:
            self.ffn_hc = self.post_norm = self.moe = None
            return
        self.ffn_hc = TtHyperConnection(
            mesh_device, fn=w["ffn_hc.fn"], base=w["ffn_hc.base"], scale=w["ffn_hc.scale"], **hc
        )
        self.post_norm = TtDistributedRmsNorm(
            mesh_device=mesh_device,
            emb_dim=D,
            epsilon=cfg.rms_norm_eps,
            torch_weight=w["post_attention_layernorm.weight"],
            cluster_axis=tp_axis,
            num_links=num_links,
            topology=topology,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.post_attention_norm",
        )
        self.moe = build_v4_moe(
            mesh_device,
            cfg,
            layer_idx,
            seq_len_per_chip=seq_len_per_chip,
            gate_weight=w["mlp.gate.weight"].float(),
            gate_bias=None
            if w.get("mlp.gate.e_score_correction_bias") is None
            else w["mlp.gate.e_score_correction_bias"].float(),
            tid2eid=w.get("mlp.gate.tid2eid"),
            routed_expert_weights=w.get("__experts__"),
            shared_expert_weights={
                "gate_proj": w["mlp.shared_experts.gate_proj.weight"].float(),
                "up_proj": w["mlp.shared_experts.up_proj.weight"].float(),
                "down_proj": w["mlp.shared_experts.down_proj.weight"].float(),
            },
            num_links=num_links,
            topology=topology,
            dispatch_buffer_capacity_factor=dispatch_buffer_capacity_factor,
            weight_cache_path=weight_cache_path,
            num_routed_experts=num_routed_experts,
        )

    # ---- chunk state per slot ------------------------------------------------------------------------------------
    def alloc_states(self, num_users: int, max_seq_len: int, chunk_tokens: int) -> None:
        for slot in range(int(num_users)):
            self.states[slot] = self.attn.alloc_state(max_seq_len, chunk_tokens=chunk_tokens, slot=slot)

    def reset_slot(self, slot: int) -> None:
        """A new prompt starts in ``slot``: counters back to zero (stale rows are masked by the counters), CSA
        compressor priors back to empty. A slot nothing has been written to since its last reset is left alone,
        so the reset at chunk 0 inside ``forward`` allocates nothing when the caller already reset it -- which is
        what lets a trace capture of chunk 0 follow its warm-up without a host write (DS4F-0247)."""
        st = self.states[slot]
        if getattr(st, "fresh", False):
            return
        st.kv_actual, st.entry_count = 0, 0
        if hasattr(st, "prior_c"):
            # in place: the prior tensors keep their addresses (a captured chunk-0 trace reads them there)
            self.attn.compressor.reset_prior(st.prior_c)
            self.attn.indexer.compressor.reset_prior(st.prior_i)
            if getattr(st, "score_mask", None) is not None:
                self.attn.reset_score_mask(st)  # traced path A3: every entry column back to -inf
        st.fresh = True

    # ---- trace islands (DS4F-0246/0247) --------------------------------------------------------------------------
    def enable_trace_islands(self, streams: list, input_ids=None, moe_meta=None) -> None:
        """Capture the position-independent slices of this block as traces, ONCE, after an eager warm-up has compiled
        their programs. ``streams`` are live activations of the shape every chunk uses (their contents do not matter);
        ``input_ids`` a device ids tensor of the gate's layout for a hash-routed layer (``None`` otherwise).
        Island A = attn_hc + input_norm -> (post, comb, h). Island B = attn_hc.mix + ffn_hc + post_norm + MoE +
        ffn_hc.mix -> streams (absent on a kv_only layer). Inputs: the block-owned persistent copies S_in of the four
        streams, y_buf for the attention output, ids_buf for the hash gate; B also reads A's persistent outputs."""
        assert self._islands is None, "islands already enabled"
        S_in = [ttnn.clone(t) for t in streams]  # persistent input copies, same shape/dtype/layout as every chunk

        def island_a(*s):
            post, comb, x = self.attn_hc(list(s))
            h = self.input_norm(x)
            ttnn.deallocate(x)
            return post, comb, h

        A = TraceIsland(self.mesh_device, island_a, S_in, name=f"layer{self.layer_idx}.A")
        post, comb, h = A.capture()
        self._attn_islands = {}
        self._a2_buckets = {}  # slot -> [(scorer width, A2 bucket island)], ascending (DS4F-0300 capacity buckets)
        if getattr(self.attn, "traceable", lambda: False)():
            # PATH A3 (DS4F-0246, galaxy: CSA issue 104 ms vs 20 ms device; HCA 26 vs 13.5): the attention itself as two
            # islands per slot -- A1 = stems + compressor(s) (+ SP gather), A2 = the attention core + o-proj + export prep
            # -- with eager glue / epilogue for the position-dependent writes. Protocol shared by the CSA and HCA modules:
            # prepare_chunk / forward_pre / glue_chunk / forward_attn / epilogue_chunk + trace_ready(state). Warm each
            # phase eagerly first (compiles; mutates the slot, reset at request start), then capture with the same
            # scalars pushed so no host write lands inside the capture.
            S_l = streams[0].shape[2]
            chunk_tokens = S_l * self.mesh_device.shape[self.sp_axis]
            for slot, state in self.states.items():
                pre = getattr(self.attn, "precreate_constants", None)
                if pre is not None:
                    pre(
                        state, S_l, chunk_tokens
                    )  # lazily-built constants must exist before any capture (DS4F-0262 hazards)
                self.attn.prepare_chunk(state, chunk_tokens)
                warm = self.attn.forward_pre(h, state, chunk_tokens)
                self.attn.glue_chunk(state, warm, chunk_tokens, None)
                a2_warm = self.attn.forward_attn(h, warm, state, chunk_tokens)
                # DS4F-0300 capacity buckets (CSA): scorer widths of the extra A2 islands, ascending; compile each variant
                # and the copy into the full island's outputs (same shapes) now, before any attention island is captured
                buckets = getattr(self.attn, "capacity_buckets", lambda *_: [])(state, S_l)
                for kv_b in buckets:
                    for dst, t in zip(a2_warm, self.attn.forward_attn(h, warm, state, chunk_tokens, kv_bucket=kv_b)):
                        if t is not None:
                            copy_into(dst, t)
                            ttnn.deallocate(t)
                c0_bucket = buckets[0] if buckets else None  # chunk 0's live columns fit the smallest bucket
                chunk0_island = _TRACED_CHUNK0 and getattr(self.attn, "supports_chunk0_island", False)
                if chunk0_island:
                    # compile the chunk-0 variant too (DS4F-0300) before any attention island is captured
                    c0_kw = {} if c0_bucket is None else {"kv_bucket": c0_bucket}
                    a2_warm = list(a2_warm) + list(
                        self.attn.forward_attn(h, warm, state, chunk_tokens, chunk0=True, **c0_kw)
                    )
                for t in list(a2_warm) + list(warm):
                    if t is not None:  # the SWA phases carry None for entries / mask
                        ttnn.deallocate(t)
                if os.environ.get("PREFILL_A1_OPTIME", "0") == "1" and self.kind == CSA:
                    # DS4F-0300 measurement: one more EAGER pass (compiled above) with a synchronize after every step
                    import time as _time

                    from loguru import logger

                    marks = []

                    def _t(name, _m=marks):
                        ttnn.synchronize_device(self.mesh_device)
                        _m.append((name, _time.perf_counter()))

                    _t("start")
                    w2 = self.attn.forward_pre(h, state, chunk_tokens, _t=_t)
                    self.attn.glue_chunk(state, w2, chunk_tokens, None)
                    _t("glue")
                    a2t = self.attn.forward_attn(h, w2, state, chunk_tokens, _t=_t)
                    logger.info(
                        f"[v4 optime] layer {self.layer_idx} CSA eager A1/glue/A2 ms: "
                        + ", ".join(f"{n} {(t1 - t0) * 1e3:.2f}" for (_, t0), (n, t1) in zip(marks, marks[1:]))
                    )
                    for t in list(a2t) + list(w2):
                        if t is not None:
                            ttnn.deallocate(t)
                self.attn.prepare_chunk(state, chunk_tokens)
                A1 = TraceIsland(
                    self.mesh_device,
                    lambda hh, _st=state: self.attn.forward_pre(hh, _st, chunk_tokens),
                    [h],
                    name=f"layer{self.layer_idx}.A1.slot{slot}",
                )
                outs = A1.capture()
                self.attn.glue_chunk(state, outs, chunk_tokens, None)
                A2 = TraceIsland(
                    self.mesh_device,
                    lambda _o=outs, _st=state: self.attn.forward_attn(h, _o, _st, chunk_tokens),
                    [],
                    name=f"layer{self.layer_idx}.A2.slot{slot}",
                )
                a2_out = A2.capture()
                bucket_islands = []
                for kv_b in buckets:
                    # the bucket island writes the FULL island's persistent outputs (one set of output buffers per layer and
                    # slot, whichever island ran): its own results are capture intermediates, copied then freed

                    def _bucket_fn(_o=outs, _st=state, _b=kv_b, _dst=a2_out):
                        res = self.attn.forward_attn(h, _o, _st, chunk_tokens, kv_bucket=_b)
                        for dst, t in zip(_dst, res):
                            if t is not None:
                                copy_into(dst, t)
                                ttnn.deallocate(t)
                        return _dst

                    isl = TraceIsland(
                        self.mesh_device, _bucket_fn, [], name=f"layer{self.layer_idx}.A2b{kv_b}.slot{slot}"
                    )
                    isl.capture()
                    bucket_islands.append((kv_b, isl))
                self._a2_buckets[slot] = bucket_islands
                if buckets:
                    from loguru import logger

                    logger.info(
                        f"[v4 buckets] layer {self.layer_idx} slot {slot}: A2 scorer widths {buckets} + full "
                        f"{(int(state.index_k.shape[2]) // 64) * 64}; chunk 0 on {c0_bucket}"
                    )
                A2c0 = None
                if chunk0_island:
                    A2c0 = TraceIsland(
                        self.mesh_device,
                        lambda _o=outs, _st=state, _kw=c0_kw: self.attn.forward_attn(
                            h, _o, _st, chunk_tokens, chunk0=True, **_kw
                        ),
                        [],
                        name=f"layer{self.layer_idx}.A2c0.slot{slot}",
                    )
                    A2c0.capture()
                self._attn_islands[slot] = (A1, A2, A2c0)
                self._chunk_tokens = chunk_tokens
        if self.kv_only:
            self._islands = (A, None, S_in, None, None)
            return
        y_buf = ttnn.clone(h)  # the attention output has h's shape
        ids_buf = None
        if self.hash_layer:
            assert isinstance(input_ids, ttnn.Tensor), "a hash-routed layer's islands need the device ids tensor"
            ids_buf = ttnn.clone(input_ids)
        S_l, D_l = streams[0].shape[2], streams[0].shape[3]
        chunk_tokens = S_l * self.mesh_device.shape[self.sp_axis]

        def island_b(*inputs):
            s = list(inputs[:HC])
            y = inputs[HC]
            streams2 = self.attn_hc.mix(s, y, post, comb)
            post2, comb2, x = self.ffn_hc(streams2)
            h2 = self.post_norm(x)
            ttnn.deallocate(x)
            h3 = ttnn.reshape(h2, [1, S_l, D_l])
            # moe_meta (DS4F-0300): the padding config is built ON DEVICE from this chunk's (slot, start, end) tensors, so a ragged
            # chunk's pad rows are not dispatched; None = the capture's full-chunk config (every row routed)
            moe_out, _ = self.moe(h3, input_ids=ids_buf, actual_isl=chunk_tokens, actual_start=0, metadata=moe_meta)
            moe_out = ttnn.reshape(moe_out, [1, 1, S_l, D_l])
            out = self.ffn_hc.mix(streams2, moe_out, post2, comb2)
            ttnn.deallocate(moe_out)
            for t in streams2:
                ttnn.deallocate(t)
            return out

        if moe_meta is not None:
            # allocate the persistent device padding-config row and compile its op BEFORE the capture (DS4F-0271 rule)
            self.moe.gate.build_padding_config_device(moe_meta)
        B = TraceIsland(self.mesh_device, island_b, S_in + [y_buf], moe=self.moe, name=f"layer{self.layer_idx}.B")
        B.capture()
        self._islands = (A, B, S_in, y_buf, ids_buf)

    def islands_enabled(self) -> bool:
        return self._islands is not None

    def release_islands(self) -> None:
        """Free this layer's captured traces (A, B and the per-slot attention islands A1/A2) and the sub-device managers
        that own them. Must run BEFORE the mesh device closes: a MeshTraceBuffer destroyed by the SubDeviceManager
        tracker during close segfaults in BankManager::deallocate_buffer (DS4F-0258; same mechanism as the MLA runtime's
        release_trace). Idempotent."""
        for islands in (getattr(self, "_attn_islands", None) or {}).values():
            for i in islands:
                if i is not None:
                    i.release()
        self._attn_islands = {}
        for bucket_islands in (getattr(self, "_a2_buckets", None) or {}).values():
            for _, i in bucket_islands:
                i.release()
        self._a2_buckets = {}
        if self._islands is not None:
            for i in self._islands[:2]:
                if i is not None:
                    i.release()
            self._islands = None
        release = getattr(self.moe, "release_sub_device_manager", None) or getattr(
            self.moe, "release_sub_device_managers", None
        )
        if release is not None:
            release()

    def _tick(self, step: str, t0: float) -> float:
        """PREFILL_ISLAND_TIMING: drain the device, charge the wall since t0 to (kind, step), return the new t0."""
        import time as _time

        ttnn.synchronize_device(self.mesh_device)
        t1 = _time.perf_counter()
        key = (str(self.kind), step)
        ISLAND_TIMES[key] = ISLAND_TIMES.get(key, 0.0) + (t1 - t0)
        return t1

    def _forward_traced(self, streams, *, slot, caches, real_len, input_ids, on_layer_complete, on_layer_hidden):
        A, B, S_in, y_buf, ids_buf = self._islands
        state = self.states[slot]
        if _ISLAND_TIMING:
            import time as _time

            ttnn.synchronize_device(self.mesh_device)
            t0 = _time.perf_counter()
        for dst, src in zip(S_in, streams):
            copy_into(dst, src)
        post, comb, h = A.replay()
        if _ISLAND_TIMING:
            t0 = self._tick("A", t0)
        state.fresh = False
        attn_islands = self._attn_islands.get(slot) if getattr(self, "_attn_islands", None) else None
        traced_attn = attn_islands is not None and self.attn.trace_ready(state)
        a2_label = "A2"
        if traced_attn:
            A1, A2 = attn_islands[0], attn_islands[1]
            need = None
            for kv_b, isl in (getattr(self, "_a2_buckets", None) or {}).get(slot, ()):
                # DS4F-0300: the smallest capacity bucket whose scorer width holds this chunk's live columns
                need = self.attn.bucket_need(state, self._chunk_tokens) if need is None else need
                if need <= kv_b:
                    A2, a2_label = isl, "A2b"
                    break
        elif (
            attn_islands is not None
            and len(attn_islands) > 2
            and attn_islands[2] is not None
            and int(state.kv_actual) == 0
            and (real_len >= _TRACED_MIN_LEN or real_len == getattr(self, "_chunk_tokens", -1))
        ):
            # chunk 0 on its own A2 island (DS4F-0300), any length >= PREFILL_TRACED_MIN_LEN (default 1)
            A1, A2 = attn_islands[0], attn_islands[2]
            a2_label = "A2c0"
            traced_attn = True
        if traced_attn:
            export = self._export_target(caches, slot)
            self.attn.prepare_chunk(state, real_len)
            outs = A1.replay()
            if _ISLAND_TIMING:
                t0 = self._tick("A1", t0)
            self.attn.glue_chunk(state, outs, real_len, export)
            if _ISLAND_TIMING:
                t0 = self._tick("glue", t0)
            a2 = A2.replay()
            if _ISLAND_TIMING:
                t0 = self._tick(a2_label, t0)
            y = a2[0]
            if getattr(on_layer_hidden, "detail", False):  # DS4F-0272 probe: island outputs before the epilogue writes
                on_layer_hidden(f"{self.layer_idx}:A1.outs", list(outs))
                on_layer_hidden(f"{self.layer_idx}:A2.rest", list(a2[1:]))
            self.attn.epilogue_chunk(state, outs, a2, export, real_len)
            if _ISLAND_TIMING:
                t0 = self._tick("epilogue", t0)
        else:
            y = self.attn(h, seq_len_actual=real_len, state=state, export=self._export_target(caches, slot))
            if _ISLAND_TIMING:
                t0 = self._tick("attn_eager", t0)
        if getattr(on_layer_hidden, "detail", False):
            on_layer_hidden(f"{self.layer_idx}:attn.y", [y])
        if on_layer_complete is not None:
            on_layer_complete(self.layer_idx)  # the runtime decides when (and whether) to drain the device first
        if self.kv_only:
            if not traced_attn:
                ttnn.deallocate(y)
            return None
        copy_into(y_buf, y)
        if not traced_attn:
            ttnn.deallocate(y)  # nothing eager may outlive the replay below
        if ids_buf is not None:
            copy_into(ids_buf, input_ids)
        out = list(B.replay())
        if _ISLAND_TIMING:
            t0 = self._tick("B", t0)
        if on_layer_hidden is not None:
            on_layer_hidden(self.layer_idx, out)
        return out

    def _export_target(self, caches, slot: int):
        """(unified cache tensor, batch index) of this layer for ``slot``; None when the kind has no export yet."""
        if caches is None:
            return None
        geom = caches.geometry
        if self.kind == HCA:
            layers, tensors = geom.hca_layers, (caches.hca_unified,)
        elif self.kind == SLIDING:
            layers, tensors = geom.swa_layers, (caches.swa_window,)
        else:
            layers, tensors = geom.csa_layers, (caches.csa_unified, caches.csa_index_k)
        if any(t is None for t in tensors):
            return None
        batch_idx = int(slot) * len(layers) + list(layers).index(self.layer_idx)
        if self.kind == CSA:
            # + the compressor pending-state group (contract config 4, DS4F-0242): None when not allocated
            return (*tensors, batch_idx, caches.csa_pending)
        return (*tensors, batch_idx)

    # ---- forward ------------------------------------------------------------------------------------------------
    def forward(
        self,
        streams: list,
        *,
        slot: int,
        caches,
        actual_start: int,
        actual_end: int,
        input_ids: Optional[torch.Tensor] = None,
        on_layer_complete: Optional[Callable[[int], None]] = None,
        on_layer_hidden: Optional[Callable[[int, list], None]] = None,
    ):
        assert len(streams) == HC
        state = self.states[slot]
        if actual_start == 0:
            self.reset_slot(slot)
        assert (
            state.kv_actual == actual_start
        ), f"slot {slot}: state at {state.kv_actual}, chunk starts at {actual_start}"
        real_len = int(actual_end) - int(actual_start)
        S_l, D_l = streams[0].shape[2], streams[0].shape[3]
        chunk = S_l * self.mesh_device.shape[self.sp_axis]
        # DS4F-0268: a ragged FINAL chunk may ride the full-chunk islands (PREFILL_TRACED_RAGGED=1): the input buffer is
        # always chunk-wide, the attention is causal (pad rows never feed a real row), the eager glue / epilogue take the
        # real length (the HCA/CSA compressor masks read it from a scalar; the ring rows are re-derived; entries past the
        # real ones stay -inf in the score mask and beyond the migrated range). A tail shorter than PREFILL_TRACED_MIN_LEN
        # (default 1; was 256 = the eager path's 32 * sp alignment step) stays eager.
        ragged_traced = (
            _TRACED_RAGGED
            and real_len < chunk
            and real_len >= _TRACED_MIN_LEN
            and int(actual_start) > 0
            and bool(getattr(self, "_attn_islands", None))
            and slot in self._attn_islands
            and self.attn.trace_ready(state)
        )
        # DS4F-0300: a ragged chunk 0 (the whole prompt fits in one chunk) takes the same route as a full chunk 0: islands A
        # + B replay on the chunk-wide buffers (the MoE island routes the pad rows as tokens, as for the traced ragged tail;
        # attention is causal, so no real row reads a pad row) and the attention runs eager with the real length.
        ragged0_traced = _TRACED_RAGGED0 and 0 < real_len < chunk and int(actual_start) == 0
        if self._islands is not None and (real_len == chunk or ragged_traced or ragged0_traced):
            # full chunk: the captured islands' MoE padding config (actual_isl = chunk) and the hash gate's device ids
            # buffer hold; a ragged FINAL chunk (real_len < chunk) runs the eager path below unless ragged_traced
            if not self.hash_layer or isinstance(input_ids, ttnn.Tensor):
                return self._forward_traced(
                    streams,
                    slot=slot,
                    caches=caches,
                    real_len=real_len,
                    input_ids=input_ids,
                    on_layer_complete=on_layer_complete,
                    on_layer_hidden=on_layer_hidden,
                )

        post, comb, x = self.attn_hc(streams)
        h = self.input_norm(x)
        ttnn.deallocate(x)
        state.fresh = False  # the attention below writes this slot's state
        y = self.attn(h, seq_len_actual=real_len, state=state, export=self._export_target(caches, slot))
        ttnn.deallocate(h)
        if getattr(on_layer_hidden, "detail", False):  # DS4F-0272 probe
            on_layer_hidden(f"{self.layer_idx}:attn.y", [y])
        if on_layer_complete is not None:
            on_layer_complete(self.layer_idx)  # the runtime drains the device before the ack fires (per layer or chunk)
        if self.kv_only:
            return None
        streams = self.attn_hc.mix(streams, y, post, comb)
        ttnn.deallocate(y)

        post, comb, x = self.ffn_hc(streams)
        h = self.post_norm(x)
        ttnn.deallocate(x)
        h3 = ttnn.reshape(h, [1, S_l, D_l])
        # actual_start=0: this block's chunk is laid out contiguously from SP row 0 (no block-cyclic rotation)
        moe_out, _ = self.moe(h3, input_ids=input_ids if self.hash_layer else None, actual_isl=real_len, actual_start=0)
        moe_out = ttnn.reshape(moe_out, [1, 1, S_l, D_l])
        if getattr(on_layer_hidden, "detail", False):
            on_layer_hidden(f"{self.layer_idx}:moe.out", [moe_out])
        streams = self.ffn_hc.mix(streams, moe_out, post, comb)
        ttnn.deallocate(moe_out)
        if on_layer_hidden is not None:
            on_layer_hidden(self.layer_idx, streams)
        return streams


# =====================================================================================================================
# TtV4Block -- main's public DeepSeek-V4 decoder block (tt-metal #56723), independent of TtV4PrefillBlock above.
# =====================================================================================================================

# Main's ``_ATTENTION`` map, resolved lazily (see the import note at the top of this file).
_ATTENTION_KINDS = ("sliding_attention", "heavily_compressed_attention")


def _attention_cls(kind: str):
    if kind == "sliding_attention":
        from models.demos.deepseek_v3_d_p.tt.mla.sliding_window_attention import TtSWA as TtMlaSWA

        return TtMlaSWA
    return TtHCA


# The only gate modes that read the hash table; the rest route by top-k and ignore it, which on a
# hash layer means picking other experts entirely.
_HASH_GATE_MODES = (GateComputeMode.HASH_HOST, GateComputeMode.HASH_DEVICE)

_SUBLAYER_DTYPE = ttnn.bfloat16


class TtV4Block(LightweightModule):
    """A norm, the layer's attention, a norm, the layer's MoE, and two hyper-connection residuals.

    Two properties shape the class:

      * attention is per layer. ``config.layer_types[layer_idx]`` names it, and V4's attention modules
        own their KV state and build their own rope, so the block supplies no cache, no rope tensors
        and no indexer, and holds no cache-migration handoff.
      * the residual is hyper-connections, so the block's own input and output carry ``hc_mult``
        parallel streams in fp32. Every sublayer between them still sees one hidden state.

    Shapes, taking a single device first: input and output are ``[1, 1, seq, hc_mult * hidden]`` fp32,
    and each sublayer sees ``[1, 1, seq, hidden]``. On a mesh the sequence is sharded on the SP axis
    and hidden on the TP axis, giving ``[1, 1, seq/sp, hc_mult * hidden/tp]`` and
    ``[1, 1, seq/sp, hidden/tp]``. Hidden is never gathered across the block. Within a chip the columns
    hold its own hidden slice of every stream, which is what ``mhc_expand`` and this block's own
    output already produce -- chaining layers needs no repacking. Only a host upload has to permute
    into that order, since a mesh mapper splits the last dim globally (``_pack_streams`` in the test).

    The MoE comes from ``TtPrefillBlock._build_moe``: the same TtMoe off the same model config,
    reading ``state_dict["hash_table"]`` for V4's hash-routed layers.

    ``weight_cache_path`` lets the norms and the MoE write their caches, but there is no
    check-complete / build-cache pre-pass yet -- that belongs with the transformer that drives a
    whole model's worth of these.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        config: PretrainedConfig,
        model_cfg: type,
        state_dict: dict,
        layer_idx: int,
        seq_len: int,
        attn_reference,
        mhc_weights: dict,
        num_links: int = 1,
        topology: TopologyArg = ttnn.Topology.Linear,
        sp_axis: int = 0,
        tp_axis: int = 1,
        max_seq_len: Optional[int] = None,
        dispatch_buffer_capacity_factor: int = 2,
        gate_fallback_mode: Optional[GateComputeMode] = None,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        weight_cache_path: Optional[Path] = None,
        is_balanced: bool = False,
        routing_use_l1_small_for_semaphores: bool = False,
        overlap_shared_expert_with_dispatch: bool = True,
    ):
        """``seq_len`` is the padded per-chunk length; ``max_seq_len`` the full per-user length the
        attention state must span, defaulting to one chunk. ``attn_reference`` is the torch attention
        module the weights are read off, and ``mhc_weights`` the ``(fn, base, scale)`` triple per site.
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.num_links = num_links
        # A (SP-axis, TP-axis) tuple configures each mesh axis separately; a scalar applies to both.
        assert (
            not isinstance(topology, tuple) or len(topology) == 2
        ), f"per-axis topology must be a 2-tuple (sp_axis, tp_axis), got {topology!r}"
        tp_topology = topology[1] if isinstance(topology, tuple) else topology
        self.layer_idx = layer_idx

        attn_kind = config.layer_types[layer_idx]
        assert (
            attn_kind in _ATTENTION_KINDS
        ), f"layer {layer_idx} wants attention {attn_kind!r}, have {_ATTENTION_KINDS}"

        mlp_kind = config.mlp_layer_types[layer_idx]
        hash_layer = mlp_kind == "hash_moe"
        if gate_fallback_mode is None:
            gate_fallback_mode = GateComputeMode.HASH_DEVICE if hash_layer else GateComputeMode.DEVICE_FP32
        hash_mode = gate_fallback_mode in _HASH_GATE_MODES
        assert hash_layer == hash_mode, f"layer {layer_idx} is {mlp_kind}, which {gate_fallback_mode} cannot route"

        logger.info(f"Building TtV4Block layer_idx={layer_idx} ({attn_kind}, {mlp_kind})")

        self.attn_norm = TtDistributedRmsNorm(
            mesh_device=mesh_device,
            emb_dim=config.hidden_size,
            torch_weight=state_dict.get("attn_norm_weight"),
            epsilon=config.rms_norm_eps,
            cluster_axis=tp_axis,
            num_links=num_links,
            topology=tp_topology,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.attn_norm",
        )
        self.ffn_norm = TtDistributedRmsNorm(
            mesh_device=mesh_device,
            emb_dim=config.hidden_size,
            torch_weight=state_dict.get("ffn_norm_weight"),
            epsilon=config.rms_norm_eps,
            cluster_axis=tp_axis,
            num_links=num_links,
            topology=tp_topology,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.ffn_norm",
        )

        # The full per-axis topology: the attention's SP-axis gather needs the SP element and its
        # other collectives the TP one. Its weights come off the torch module, not a state dict, so
        # there is nothing for a weight cache to hold for this half of the block.
        self.attn = _attention_cls(attn_kind).from_reference(
            mesh_device, attn_reference, config, sp_axis=sp_axis, tp_axis=tp_axis, topology=topology
        )
        # Allocated once: the state's shape is the same for every chunk, only its contents advance.
        # TODO once a V4 transformer owns this state instead of the block, forward() needs the chunk
        # offset back the way TtPrefillBlock takes actual_start: a caller-owned cache does not
        # advance itself, so the position has to come in with the chunk.
        self.attn_state = self.attn.alloc_state(max_seq_len or seq_len, chunk_tokens=seq_len)

        from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock

        self.ffn = TtPrefillBlock._build_moe(
            mesh_device=mesh_device,
            model_cfg=model_cfg,
            config=config,
            state_dict=state_dict,
            seq_len=seq_len,
            sp_axis=sp_axis,
            emb_dim=config.hidden_size,
            num_links=num_links,
            topology=topology,
            gate_fallback_mode=gate_fallback_mode,
            routed_expert_activations_dtype=routed_expert_activations_dtype,
            routed_expert_weights_dtype=routed_expert_weights_dtype,
            shared_expert_activations_dtype=shared_expert_activations_dtype,
            shared_expert_weights_dtype=shared_expert_weights_dtype,
            weight_cache_path=weight_cache_path,
            layer_idx=layer_idx,
            dispatch_buffer_capacity_factor=dispatch_buffer_capacity_factor,
            routing_use_l1_small_for_semaphores=routing_use_l1_small_for_semaphores,
            is_balanced=is_balanced,
            overlap_shared_expert_with_dispatch=overlap_shared_expert_with_dispatch,
        )

        assert mhc_weights is not None and set(mhc_weights) >= {
            "attn",
            "ffn",
        }, f"both sites need an (fn, base, scale) triple; got keys {sorted(mhc_weights or ())}"
        mhc_cfg = MHCConfig(
            dim=config.hidden_size,
            n=config.hc_mult,
            sinkhorn_iters=config.hc_sinkhorn_iters,
            eps=config.hc_eps,
            norm_eps=config.rms_norm_eps,
        )
        self.attn_res = TtMHCWrap(
            mesh_device, mhc_cfg, *mhc_weights["attn"], tp_axis=tp_axis, num_links=num_links, topology=tp_topology
        )
        self.ffn_res = TtMHCWrap(
            mesh_device, mhc_cfg, *mhc_weights["ffn"], tp_axis=tp_axis, num_links=num_links, topology=tp_topology
        )

    def forward(
        self,
        x: ttnn.Tensor,
        actual_isl: Optional[int] = None,
        input_ids: Optional[torch.Tensor] = None,
        padding_side: str = "right",
    ) -> ttnn.Tensor:
        """One chunk of packed residual streams in, the same shape out.

        ``actual_isl`` is the chunk's real pre-pad length and ``input_ids`` the chunk's token ids,
        which a hash-routed layer indexes its expert table with and a top-k layer ignores. Where the
        chunk begins is not an argument: the attention state tracks it and advances itself.

        Each norm sits inside its site's sublayer, not before it: the hyper-connection collapses its
        streams first, and the norm belongs on what comes out of that collapse.
        """
        assert padding_side == "right", f"V4 attention is right-padded by construction, got {padding_side!r}"

        def _attn(h):
            normed = self.attn_norm(h)
            out = self.attn(normed, seq_len_actual=actual_isl, state=self.attn_state)
            ttnn.deallocate(normed)
            return out

        def _ffn(h):
            normed = self.ffn_norm(h)
            # TtMoe works in 3D; the block's own tensors are 4D.
            out, _ = self.ffn(
                ttnn.squeeze(normed, dim=0),
                actual_isl=actual_isl,
                padding_side=padding_side,
                # The MoE reads a non-zero actual_start as "your rows are rotated across the SP
                # axis", which is ttMLA's layout, not ours: this block uploads the sequence as a
                # plain contiguous shard, so the sequential row counts are the correct ones.
                actual_start=0,
                input_ids=input_ids,
            )
            ttnn.deallocate(normed)
            return ttnn.unsqueeze(out, dim=0)

        x = self.attn_res(x, _attn, sublayer_dtype=_SUBLAYER_DTYPE)
        return self.ffn_res(x, _ffn, sublayer_dtype=_SUBLAYER_DTYPE)
