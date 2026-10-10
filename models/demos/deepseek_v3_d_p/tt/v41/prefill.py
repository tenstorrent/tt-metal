# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash prefill for the disaggregated split (tt-blaze DS41F-0037): the ENCODER (layers 0..19, Engram
before 1 and 14) in full and layer 20 KV-only, eager, chunked.

    pf = V41Prefill(mesh, cfg, checkpoint(), max_seq_len=..., chunk_tokens=...)
    pf.prefill(ids, engram_host)            # tt-blaze EngramHost: the n-gram hashes + table rows per position
    snap = pf.cache_snapshot()              # == tt-blaze golden ``cache_snapshot_of`` (prefill_trace.py) for layers 0..20

What the decode ring needs after the prompt (tt-blaze ``docs/deepseek_v4_1_flash/disagg_prefill_plan.md``): every layer's
window (the last 128 positions), the KV sources' compressed entries (2 / 8 / 14 at ratio 2; layer 20 at ratio 1) and the
index keys (2 / 8 / 14 / 20). The decoder layers 21..39 start empty: the ring replays the prompt's last R >= 2541 tokens,
after which its state equals a full prefill's (layer-20 entries and keys are exact because they come from the encoder
output; every decoder-side dependency is a window, a top-k or a candidate pick per query).

Streams are 4 bf16 ``[1, 1, S_l, D_l]`` (SP over mesh dim ``sp_axis``, TP over ``tp_axis``) as in V4-Flash; the embedding
lookup and the Engram hash / gather run on the host.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm

from .attention import V41CSA, v41_hf_config
from .block import HC, V41HyperConnection, V41PrefillBlock, identity_pre_row
from .engram import V41PrefillEngram


def _first_device(t) -> torch.Tensor:
    """A replicated device tensor -> the first chip's copy on the host (fp32)."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


PROFILE = os.environ.get("V41_PREFILL_PROFILE", "0") == "1"


class PhaseProfile:
    """V41_PREFILL_PROFILE=1: per phase name (summed over the layers), the HOST time to issue its ops and the DEVICE
    drain after it (a synchronize closes every phase). A phase whose drain is ~0 is host-issue bound: the device waited
    on Python. The syncs serialise host and device, so the total is an upper bound of the unprofiled chunk."""

    def __init__(self, mesh):
        self.mesh, self.acc = mesh, {}
        ttnn.synchronize_device(mesh)
        self.t = time.perf_counter()

    def __call__(self, name: str) -> None:
        t1 = time.perf_counter()
        ttnn.synchronize_device(self.mesh)
        t2 = time.perf_counter()
        a = self.acc.setdefault(name, [0.0, 0.0, 0])
        a[0], a[1], a[2] = a[0] + t1 - self.t, a[1] + t2 - t1, a[2] + 1
        self.t = t2

    def report(self, what: str) -> dict:
        tot_i = sum(v[0] for v in self.acc.values())
        tot_w = sum(v[1] for v in self.acc.values())
        rows = ", ".join(
            f"{k} {v[0] * 1e3:.0f}+{v[1] * 1e3:.0f} ms/{v[2]}"
            for k, v in sorted(self.acc.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))
        )
        logger.info(
            f"[v41 prefill][profile] {what}: issue {tot_i:.2f} s + device drain {tot_w:.2f} s | phase issue+drain/calls: {rows}"
        )
        return dict(self.acc)


class V41Prefill:
    CHUNK_ALIGN = (
        64  # tokens per chip: a multiple of the MoE routing's 64-core masked_bincount grid (and of TILE / ratio)
    )

    def __init__(
        self,
        mesh_device,
        cfg,
        ck,
        *,
        max_seq_len: int,
        chunk_tokens: int,
        n_layers: int | None = None,
        kv_only: bool = True,
        sp_axis: int = 0,
        tp_axis: int = 1,
        topology=ttnn.Topology.Linear,
        num_links: int = 2,
        weight_cache_path=None,
        load_routed_from_cache: bool = False,
        num_users: int = 1,
    ):
        self.mesh, self.cfg, self.ck = mesh_device, cfg, ck
        self.num_users, self.slot = int(num_users), 0
        self.last_profile = None  # V41_PREFILL_PROFILE: the last chunk's PhaseProfile
        weight_cache_path = Path(weight_cache_path) if weight_cache_path else None  # the modules join it with `/`
        self.sp_axis, self.tp_axis = sp_axis, tp_axis
        self.sp, self.tp = mesh_device.shape[sp_axis], mesh_device.shape[tp_axis]
        self.chunk = int(chunk_tokens)
        # 64 tokens per chip per step: the MoE's masked_bincount splits each chip's tokens over a 64-core grid
        assert self.chunk % (self.CHUNK_ALIGN * self.sp) == 0, (self.chunk, self.CHUNK_ALIGN * self.sp)
        self.max_seq_len = int(max_seq_len)
        self.n_layers = cfg.first_decoder_layer if n_layers is None else int(n_layers)
        self.kv_only_layer = (
            cfg.first_decoder_layer if kv_only else None
        )  # layer 20: the decoder's shared KV + index keys
        S_l = self.chunk // self.sp
        self.rotary_emb = DeepseekV4RotaryEmbedding(v41_hf_config(cfg, max_seq=self.max_seq_len))
        mk = dict(sp_axis=sp_axis, tp_axis=tp_axis, topology=topology)
        self.sources: dict = {}
        self.blocks, self.engrams = [], {}
        t0 = time.time()
        for L in range(self.n_layers):
            w = ck.layer(L)
            if cfg.role(L).engram:
                self.engrams[L] = V41PrefillEngram(mesh_device, cfg, L, w, sp_axis=sp_axis, tp_axis=tp_axis)
            blk = V41PrefillBlock(
                mesh_device,
                cfg,
                L,
                w,
                rotary_emb=self.rotary_emb,
                seq_len_per_chip=S_l,
                sources=self.sources,
                ck=ck,
                num_links=num_links,
                weight_cache_path=weight_cache_path,
                load_routed_from_cache=load_routed_from_cache,
                **mk,
            )
            for slot in range(self.num_users):  # one attention state per user (the weights are shared)
                blk.alloc_state(self.max_seq_len, self.chunk, slot=slot)
            self.blocks.append(blk)
            logger.info(f"[v41 prefill] layer {L} ({cfg.role(L).mode}) built, {time.time() - t0:.0f} s")
        self.kv_attn = self.kv_hc = self.kv_norm = None
        if self.kv_only_layer is not None:
            L = self.kv_only_layer
            w = ck.layer(L)
            self.kv_hc = V41HyperConnection(
                mesh_device,
                hidden=cfg.dim,
                fn=w["hc_attn_fn"],
                base=w["hc_attn_base"],
                scale=w["hc_attn_scale"],
                rms_eps=cfg.norm_eps,
                hc_eps=cfg.hc_eps,
                sinkhorn_iters=cfg.hc_sinkhorn_iters,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
            )
            self.kv_norm = TtDistributedRmsNorm(
                mesh_device=mesh_device,
                emb_dim=cfg.dim,
                epsilon=cfg.norm_eps,
                torch_weight=w["attn_norm.weight"].float(),
                cluster_axis=tp_axis,
                num_links=num_links,
                topology=topology,
                weight_cache_path=weight_cache_path,
                cache_name_prefix=f"v41_layer_{L}.attn_norm",
            )
            self.kv_attn = V41CSA.from_weights(
                mesh_device, cfg, L, w, self.rotary_emb, weight_cache_path=weight_cache_path, **mk
            )
            self.kv_states = {
                s: self.kv_attn.alloc_state(self.max_seq_len, chunk_tokens=self.chunk) for s in range(self.num_users)
            }
        self._embed = None
        self.kv_actuals = {s: 0 for s in range(self.num_users)}

    # the CURRENT slot's layer-20 state and token count (use_slot): the chunk / export / snapshot code reads these
    @property
    def kv_state(self):
        return self.kv_states[self.slot]

    @property
    def kv_actual(self) -> int:
        return self.kv_actuals[self.slot]

    @kv_actual.setter
    def kv_actual(self, v: int) -> None:
        self.kv_actuals[self.slot] = int(v)

    def use_slot(self, slot: int) -> None:
        assert 0 <= slot < self.num_users, (slot, self.num_users)
        self.slot = int(slot)
        for b in self.blocks:
            b.use_slot(self.slot)

    def reset(self) -> None:
        """A new prompt in the CURRENT slot: its attention states' counters back to 0. Path B's masks are built from the counters
        (entries / window rows past them read -inf, chunk 0's carry columns are masked), so no device contents need to
        be cleared; V4.1's non-overlap compressor carries no state between chunks."""
        states = [b.state for b in self.blocks] + ([self.kv_state] if self.kv_attn is not None else [])
        for st in states:
            st.kv_actual, st.entry_count = 0, 0
        self.kv_actual = 0

    # ---- host-side inputs ---------------------------------------------------------------------------------------
    def _embedding(self) -> torch.Tensor:
        if self._embed is None:
            self._embed = self.ck.raw("embed.weight")  # [vocab, D] bf16
        return self._embed

    def _streams(self, ids: list[int]) -> list:
        h = torch.zeros(1, 1, self.chunk, self.cfg.dim, dtype=torch.bfloat16)
        h[0, 0, : len(ids)] = self._embedding()[torch.tensor(ids, dtype=torch.long)].to(torch.bfloat16)
        mapper = ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(2, 3))
        t = ttnn.from_torch(h, device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
        return [t] + [ttnn.clone(t) for _ in range(HC - 1)]  # model.py: h.unsqueeze(2).repeat(1, 1, hc, 1)

    # ---- the prefill --------------------------------------------------------------------------------------------
    def prefill(self, ids: list[int], engram_host=None, *, on_hidden=None) -> None:
        """All of ``ids`` from position 0 (chunks of ``chunk_tokens``; every chunk but the last full)."""
        assert self.kv_actual == 0, "one prompt per V41Prefill (states are not reset)"
        S = len(ids)
        assert S <= self.max_seq_len
        for c0 in range(0, S, self.chunk):
            part = ids[c0 : c0 + self.chunk]
            t0 = time.time()
            self._chunk(part, c0, engram_host, on_hidden)
            ttnn.synchronize_device(self.mesh)
            logger.info(f"[v41 prefill] chunk [{c0}, {c0 + len(part)}) in {time.time() - t0:.1f} s")

    def _chunk(self, part: list[int], start: int, engram_host, on_hidden):
        real_len = len(part)
        prof = PhaseProfile(self.mesh) if PROFILE else None
        mark = prof if prof is not None else (lambda name: None)
        streams = self._streams(part)
        pre = identity_pre_row(self.mesh, self.chunk // self.sp, self.sp_axis)
        mark("embed")
        rows = engram_host.rows_for(self.slot, part, start) if (self.engrams and engram_host is not None) else {}
        mark("engram.host")
        for blk in self.blocks:
            L = blk.layer
            if L in self.engrams:
                assert L in rows, f"layer {L} needs Engram rows (pass an EngramHost)"
                rd = self.engrams[L].upload_rows(rows[L], self.chunk)
                streams = self.engrams[L](streams, rd)
                ttnn.deallocate(rd)
                mark("engram.dev")
            streams, pre = blk(streams, pre, real_len=real_len, on_hidden=on_hidden, prof=prof)
            if on_hidden is not None:
                on_hidden(f"{L}:out", streams, pre)
        if self.kv_attn is not None:
            h = self.kv_norm(self.kv_hc.collapse(streams, pre))
            self.kv_attn.kv_only(h, seq_len_actual=real_len, state=self.kv_state)
            ttnn.deallocate(h)
            mark("kv_only")
        self.kv_actual += real_len
        if prof is not None:
            self.last_profile = prof.report(f"chunk [{start}, {start + real_len}) slot {self.slot}")

    # ---- export ---------------------------------------------------------------------------------------------------
    def _window(self, state) -> torch.Tensor:
        """The carry (positions [S - 128, S) in order) -> model.py's ``window_kv_cache`` layout (slot = position % 128)."""
        carry = _first_device(state.sliding_carry)[0, 0]  # [128, 512]
        win = int(carry.shape[0])
        S = self.kv_actual
        out = torch.zeros_like(carry)
        for i in range(win):
            q = S - win + i
            if q >= 0:
                out[q % win] = carry[i]
        return out

    def cache_snapshot(self) -> dict:
        """tt-blaze ``prefill_trace.cache_snapshot_of`` for layers 0..n_layers-1 (+ the KV-only layer): window ring
        (``[1, 128, 512]``, slot p % 128), and for KV sources the compressed rows ``[1, S // ratio, 512]`` and the index keys
        ``[1, S // ratio, 128]``; ``kv_state`` / ``score_state`` are omitted (V4.1's pooling never straddles an even S).
        """
        S = self.kv_actual
        out = {}
        mods = [(b.layer, b.attn, b.state) for b in self.blocks]
        if self.kv_attn is not None:
            mods.append((self.kv_only_layer, self.kv_attn, self.kv_state))
        for L, attn, st in mods:
            d = dict(window_kv_cache=self._window(st)[None])
            r = self.cfg.role(L)
            if r.mode == "full":
                n = S // r.compress_ratio
                d["compress_kv_cache"] = _first_device(st.compressed_kv)[0, 0, :n][None]
                d["index_k_cache"] = _first_device(st.index_k)[0, 0, :n][None]
            out[L] = d
        return dict(end_pos=S, layers=out)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.cache_snapshot(), path)
