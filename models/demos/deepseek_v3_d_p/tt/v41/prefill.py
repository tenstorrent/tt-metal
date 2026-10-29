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
# DS41F-0068: the chunk embedding uploaded S-sharded / TP-replicated and split on device (0 = the D-over-TP mapper)
EMBED_PARTITION = os.environ.get("V41_PREFILL_EMBED_PARTITION", "1") == "1"
ENGRAM_EARLY = (
    os.environ.get("V41_PREFILL_ENGRAM_EARLY", "1") == "1"
)  # DS41F-0068: the Engram fetch before the embedding
# DS41F-0119 (prefill bench 10-29: chunks are host-issue bound): V41_PREFILL_DEVICE_EMBED=1 looks a full text chunk's rows up on
# device from the runner's H2D ids ([1, 1, S_l] uint32 per chip, SP-sharded, TP-replicated) in the table TP-sharded on D
# (~331 MB per chip), instead of the host gather (84 MB) + tilize + TP-replicated upload; ragged / image chunks keep the host
# path. Same bf16 rows by construction; V41_PREFILL_DEVICE_EMBED_CHECK=N compares both paths on the first N chunks.
DEVICE_EMBED = os.environ.get("V41_PREFILL_DEVICE_EMBED", "0") == "1"
DEVICE_EMBED_CHECK = int(os.environ.get("V41_PREFILL_DEVICE_EMBED_CHECK", "0"))


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


def _decoder_cache():
    """DS41F-0073: V41_PREFILL_DECODER_CACHE = a weight-cache dir for the decoder layers (unset: converted at every start)."""
    p = os.environ.get("V41_PREFILL_DECODER_CACHE", "")
    return Path(p) if p else None


def _decoder_expert_dtype():
    """DS41F-0073: V41_PREFILL_DECODER_EXPERT_DTYPE=bfp4|bfp8 for the decoder layers' routed experts (unset: the encoder's).
    BFP4 is what the decode ring runs those layers with (tt-blaze DS41F-0019) and halves their ~9 GB per chip."""
    v = os.environ.get("V41_PREFILL_DECODER_EXPERT_DTYPE", "")
    return {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}[v] if v else None


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
        self.islands = False  # enable_islands: blocks replay their mHC / MoE trace islands
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
            # DS41F-0073: the decoder layers (20..39, V41_PREFILL_DECODER) take their own weight cache (default none: the shared
            # disk cannot hold another ~13.7 GB per BFP8 layer) and expert dtype (default the encoder's)
            dec = L >= cfg.first_decoder_layer
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
                weight_cache_path=_decoder_cache() if dec else weight_cache_path,
                load_routed_from_cache=load_routed_from_cache and not dec,
                expert_dtype=_decoder_expert_dtype() if dec else None,
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
        # DEVICE_EMBED: the table on device, allocated here -- before any trace-island capture (a buffer allocated after a
        # capture can sit in the islands' intermediates, DS41F-0037 10-10 21:25); _ids_dev = the current chunk's H2D ids
        self._ids_dev = None
        self._embed_dev = self._device_embedding() if DEVICE_EMBED else None
        self._embed_checks = DEVICE_EMBED_CHECK
        self.kv_actuals = {s: 0 for s in range(self.num_users)}
        # ViT V2 (tt-blaze vit_plan.md E1 / G1): the current prompt's image-span rows (virtual id vocab + k -> row k) and
        # each gate's (text, bias_vl) bias rows; ``_vl_live`` = some gate holds bias_vl rows from the previous chunk
        self._image_rows = None
        self._gate_bias_rows: dict = {}
        self._vl_live = False

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
        # DS41F-0068 path A3: the static scorer reads the slot's score mask (shared by the slot's sources), which each traced
        # chunk's glue opens column by column -> every entry column back to -inf (a device copy), once per mask
        seen = set()
        for b in self.blocks:
            m = getattr(b.state, "score_mask", None)
            if m is not None and b._attn_islands_enabled() and id(m) not in seen:
                seen.add(id(m))
                b.attn.reset_score_mask(b.state)

    # ---- host-side inputs ---------------------------------------------------------------------------------------
    def _embedding(self) -> torch.Tensor:
        if self._embed is None:
            self._embed = self.ck.raw("embed.weight")  # [vocab, D] bf16
        return self._embed

    def _device_embedding(self):
        """DEVICE_EMBED: the [vocab, D] bf16 table, replicated over SP and sharded on D over TP (TtParallelEmbedding's layout)."""
        assert (self.sp_axis, self.tp_axis) == (0, 1), (self.sp_axis, self.tp_axis)
        t0 = time.time()
        mapper = ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(None, 1))
        t = ttnn.from_torch(
            self._embedding().contiguous(),
            device=self.mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        logger.info(f"[v41 prefill] device embedding table {tuple(t.shape)} per chip in {time.time() - t0:.1f} s")
        return t

    def _streams_device(self, ids_dev):
        """DEVICE_EMBED: [1, 1, S_l, D_l] bf16 TILE per chip straight from the H2D ids (no host round trip)."""
        S_l, D_l = self.chunk // self.sp, self.cfg.dim // self.tp
        t = ttnn.embedding(ids_dev, self._embed_dev, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if tuple(t.shape) != (1, 1, S_l, D_l):
            t = ttnn.reshape(t, (1, 1, S_l, D_l))
        return t

    def _streams(self, ids: list[int]) -> list:
        ids_dev = self._ids_dev
        if (
            self._embed_dev is not None
            and ids_dev is not None
            and len(ids) == self.chunk
            and max(ids) < self._embedding().shape[0]
        ):
            t = self._streams_device(ids_dev)
            if (
                self._embed_checks > 0
            ):  # DEVICE_EMBED_CHECK: the host path's tensor for the same ids, compared on chip 0..n
                self._embed_checks -= 1
                self._ids_dev = None
                ref = self._streams(ids)[0]
                self._ids_dev = ids_dev
                a = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)]
                b = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(ref)]
                same = all(
                    x.shape == y.shape and torch.equal(x.view(torch.int16), y.view(torch.int16)) for x, y in zip(a, b)
                )
                logger.warning(
                    f"[v41 prefill] DEVICE_EMBED_CHECK: device rows {'BIT-IDENTICAL' if same else 'DIFFER'} to the host path on "
                    f"{len(a)} chips (shape {tuple(a[0].shape)} vs {tuple(b[0].shape)})"
                )
            return [t] + [ttnn.clone(t) for _ in range(HC - 1)]
        emb = self._embedding()
        t = torch.tensor(ids, dtype=torch.long)
        vl = t >= emb.shape[0]  # ViT V2 E1: image-span VIRTUAL ids (vocab + k) embed as the request's span row k
        rows = emb[t.clamp(max=emb.shape[0] - 1)].to(torch.bfloat16)
        if bool(vl.any()):
            assert self._image_rows is not None, "image-span (virtual) ids need prefill(..., image_rows=...)"
            k = t[vl] - emb.shape[0]
            assert int(k.max()) < self._image_rows.shape[0], (int(k.max()), tuple(self._image_rows.shape))
            rows[vl] = self._image_rows[k].to(torch.bfloat16)
        if len(ids) == self.chunk and rows.is_contiguous():
            h = rows.view(
                1, 1, self.chunk, self.cfg.dim
            )  # DS41F-0068: a full chunk needs no zero-filled copy (84 MB twice)
        else:
            h = torch.zeros(1, 1, self.chunk, self.cfg.dim, dtype=torch.bfloat16)
            h[0, 0, : len(ids)] = rows
        if EMBED_PARTITION:
            # DS41F-0068: S-sharded + TP-replicated upload (contiguous host slices: ~25 ms for the 8192-row chunk, vs ~104 ms for
            # the D-over-TP mapper's strided copies, MEASURED ut/a3/upload_bench.py), then each chip keeps its D / tp on device
            mapper = ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(2, None))
            full = ttnn.from_torch(
                h, device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
            )
            t = ttnn.mesh_partition(full, dim=3, cluster_axis=self.tp_axis)
            ttnn.deallocate(full)
        else:
            mapper = ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(2, 3))
            t = ttnn.from_torch(h, device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
        return [t] + [ttnn.clone(t) for _ in range(HC - 1)]  # model.py: h.unsqueeze(2).repeat(1, 1, hc, 1)

    # ---- the prefill --------------------------------------------------------------------------------------------
    def set_image_rows(self, rows) -> None:
        """ViT V2: the CURRENT prompt's image-span rows ``[n_span, dim]`` (pod-NFS ``prof/vit_prompt.py``'s span rows:
        START, the aligner rows with a NEWLINE after each grid row, END); virtual id ``vocab + k`` embeds as row ``k`` and
        routes with ``bias_vl``. None = a text-only prompt."""
        self._image_rows = None if rows is None else torch.as_tensor(rows).to(torch.bfloat16).reshape(-1, self.cfg.dim)

    def _gate_rows(self, blk):
        """(text, bias_vl) bias rows of ``blk``'s gate, bf16 [n_experts]: the text row read back from the gate's own device
        bias (bit-identical to what it routes text with), bias_vl centred on its own midrange before the bf16 cast (DS41F-0040's
        rule: top-k is shift-invariant per token, so each vector may carry its own constant)."""
        L = blk.layer
        if L not in self._gate_bias_rows:
            g = blk.moe.gate
            text = ttnn.to_torch(ttnn.get_device_tensors(g.bias)[0])[0].to(torch.bfloat16)
            vl = self.ck.raw(f"layers.{L}.ffn.gate.bias_vl").float()
            vl = (vl - (vl.min() + vl.max()) / 2).to(torch.bfloat16)
            self._gate_bias_rows[L] = (text, vl)
        return self._gate_bias_rows[L]

    def _route_bias(self, part: list[int]) -> None:
        """ViT V2 G1: model.py routes image-span tokens with ``bias_vl`` (``Gate.forward``'s image_mask). Rewrite each gate's
        per-row bias [sp_dim, n_experts] (the islands captured its buffer; the CONTENT changes, as the ragged padding
        config does) with bias_vl rows at this chunk's span positions; the next text-only chunk restores the text rows.
        """
        if (
            os.environ.get("V41_PREFILL_VL_BIAS", "1") == "0"
        ):  # the causal control: span tokens route with the text bias
            return
        vocab = self._embedding().shape[0]
        vl = [i for i, t in enumerate(part) if int(t) >= vocab]
        if not vl and not self._vl_live:
            return
        mapper = ttnn.ShardTensor2dMesh(self.mesh, dims=(0, None), mesh_shape=tuple(self.mesh.shape))
        for blk in self.blocks:
            g = blk.moe.gate
            assert int(g.bias.shape[0]) * self.sp == self.chunk, (tuple(g.bias.shape), self.chunk, self.sp)
            text, vlb = self._gate_rows(blk)
            rows = text.unsqueeze(0).repeat(self.chunk, 1)
            if vl:
                rows[torch.tensor(vl, dtype=torch.long)] = vlb
            host = ttnn.from_torch(rows, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
            ttnn.copy_host_to_device_tensor(host, g.bias)
        self._vl_live = bool(vl)

    def prefill(self, ids: list[int], engram_host=None, *, on_hidden=None, image_rows=None) -> None:
        """All of ``ids`` from position 0 (chunks of ``chunk_tokens``; every chunk but the last full). ``image_rows``: the
        prompt's image-span rows (``set_image_rows``) when ``ids`` carry virtual image ids."""
        assert self.kv_actual == 0, "one prompt per V41Prefill (states are not reset)"
        S = len(ids)
        assert S <= self.max_seq_len
        if image_rows is not None:
            self.set_image_rows(image_rows)
        for c0 in range(0, S, self.chunk):
            part = ids[c0 : c0 + self.chunk]
            t0 = time.time()
            self._chunk(part, c0, engram_host, on_hidden)
            ttnn.synchronize_device(self.mesh)
            logger.info(f"[v41 prefill] chunk [{c0}, {c0 + len(part)}) in {time.time() - t0:.1f} s")

    def enable_islands(self) -> None:
        """Capture every block's trace islands (block.enable_trace_islands) once, after eager warm-up chunks have built every
        program and every lazily cached constant at every live width (the runtime's warm-all). The CSA sources get their
        persistent selection-mask buffers first (they must exist before any capture). Templates: zero streams of the chunk
        shape, each block's capture feeding the next its persistent outputs (contents do not matter)."""
        assert not self.islands
        for blk in self.blocks:
            if hasattr(blk.attn, "precreate_shared_masks"):
                blk.attn.precreate_shared_masks(blk.state)
        streams = self._streams([0] * self.chunk)
        pre = identity_pre_row(self.mesh, self.chunk // self.sp, self.sp_axis)
        t0 = time.time()
        for blk in self.blocks:
            streams, pre = blk.enable_trace_islands(streams, pre, self.chunk)
        self.islands = True
        logger.info(f"[v41 prefill] trace islands captured for {len(self.blocks)} blocks in {time.time() - t0:.1f} s")

    def _chunk(self, part: list[int], start: int, engram_host, on_hidden):
        if self.islands:
            return self._chunk_islands(part, start, engram_host)
        real_len = len(part)
        prof = PhaseProfile(self.mesh) if PROFILE else None
        mark = prof if prof is not None else (lambda name: None)
        streams = self._streams(part)
        self._route_bias(part)
        pre = identity_pre_row(self.mesh, self.chunk // self.sp, self.sp_axis)
        mark("embed")
        rows = engram_host.rows_for(self.slot, part, start) if (self.engrams and engram_host is not None) else {}
        mark("engram.host")
        for blk in self.blocks:
            L = blk.layer
            if L in self.engrams:
                assert L in rows, f"layer {L} needs Engram rows (pass an EngramHost)"
                rd = self.engrams[L].upload_rows(rows[L], self.chunk)
                s_in = streams
                streams = self.engrams[L](streams, rd)
                ttnn.deallocate(rd)
                mark("engram.dev")
                if (
                    on_hidden is not None
                ):  # the Engram output (block L's input) and its input (ViT V2: identity on the span)
                    on_hidden(f"{L}:engram", streams, s_in)
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

    def _chunk_islands(self, part: list[int], start: int, engram_host):
        """``_chunk`` on the captured islands: the embedding streams, the pre row and the Engram outputs are eager (owned,
        freed by the block after its copy-in); each block's outputs are its island's persistent buffers."""
        real_len = len(part)
        prof = PhaseProfile(self.mesh) if PROFILE else None
        mark = prof if prof is not None else (lambda name: None)
        rows = {}

        def _fetch():
            # layer 14's rows are only needed at block 14: gather them in the background while blocks 0..13 run
            fetch = getattr(engram_host, "rows_async", None)
            return fetch(self.slot, part, start) if fetch is not None else engram_host.rows_for(self.slot, part, start)

        if self.engrams and engram_host is not None and ENGRAM_EARLY:
            # DS41F-0068: the fetch starts BEFORE the embedding (its host gather + upload, ~140 ms at 8192) so layer 1's rows are
            # not waited for (V41_PREFILL_ENGRAM_EARLY=0: after it, as before)
            rows = _fetch()
            mark("engram.host")
        streams = self._streams(part)
        self._route_bias(part)
        pre = identity_pre_row(self.mesh, self.chunk // self.sp, self.sp_axis)
        mark("embed")
        if self.engrams and engram_host is not None and not ENGRAM_EARLY:
            rows = _fetch()
            mark("engram.host")
        owned = True
        for blk in self.blocks:
            L = blk.layer
            if L in self.engrams:
                r = rows[L].result() if hasattr(rows[L], "result") else rows[L]
                mark("engram.wait")
                rd = self.engrams[L].upload_rows(r, self.chunk)
                new = self.engrams[L](streams, rd)
                ttnn.deallocate(rd)
                if owned:
                    for t in streams:
                        ttnn.deallocate(t)
                streams, owned = new, True
                mark("engram.dev")
            streams, pre = blk.forward_islands(
                streams, pre, real_len=real_len, owned=owned, owned_pre=blk is self.blocks[0]
            )
            owned = False
            mark(f"block.{type(blk.attn).__name__}")
        if self.kv_attn is not None:
            h = self.kv_norm(self.kv_hc.collapse(streams, pre))
            self.kv_attn.kv_only(h, seq_len_actual=real_len, state=self.kv_state)
            ttnn.deallocate(h)
            mark("kv_only")
        self.kv_actual += real_len
        if prof is not None:
            self.last_profile = prof.report(f"chunk [{start}, {start + real_len}) slot {self.slot} (islands)")

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
