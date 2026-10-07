# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1-Flash on the 4x8 Blackhole galaxy as ONE model object with GPT-OSS-style entry points (``prefill_forward``,
``prepare_inputs_prefill``, ``decode_forward``, ``prepare_inputs_decode``), built once and used for prefill AND decode over the SAME paged KV pool.

Layering: demo/text_demo.py -> tt/generator.py (``Generator``) -> this ``Model`` -> tt/layer.py (``DSV41Layer``), tt/paged_attention.py, tt/prefill_*.py.
The hand-off prefill -> decode is in tt/prefill_handoff.py (ring rows, compressed latents, ratio-2 compressor state, first token) and here (Engram token
history, per-user positions, device token feedback).
"""

import functools
import gc
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt import uni_policy
from models.demos.blackhole.deepseek_v41_flash.tt.decoder import DSV41Decoder
from models.demos.blackhole.deepseek_v41_flash.tt.device_head import DSV41DeviceEmbedding, DSV41DeviceHead
from models.demos.blackhole.deepseek_v41_flash.tt.engram import DSV41DeviceEngram, HostEngramRows
from models.demos.blackhole.deepseek_v41_flash.tt.engram_ragged import RaggedNgramHash
from models.demos.blackhole.deepseek_v41_flash.tt.layer import DSV41Layer
from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer
from models.demos.blackhole.deepseek_v41_flash.tt.model import DSV41DecodeChain
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.blackhole.deepseek_v41_flash.tt.paged_attention import (
    DSV41PagedAttention,
    DSV41PagedCompressedAttention,
    DSV41PagedStepState,
)
from models.demos.blackhole.deepseek_v41_flash.tt.paged_ops import PAGE_TOKENS, PagedKVPool
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_attention import DSV41PrefillAttention, clear_chunk_caches
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_handoff import GenPrefillModel, PagedStateSink
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_layer import DSV41PrefillLayer, DSV41PrefillMoE

# prefill routed experts via ttnn.experimental.deepseek_prefill reading the decode ring weights in place: default ON with automatic fallback
# (tt/uni_policy.py); DSV41_PREFILL_MOE=off forces the moe_compute prefill path


def UNI_LAYERS(layer_ids):
    """Layers that use the unified prefill MoE: DSV41_UNI_LAYERS="all" (default) or e.g. "2-11,20"."""
    v = os.environ.get("DSV41_UNI_LAYERS", "all")
    if v.startswith(
        "auto"
    ):  # DSV41_UNI_LAYERS=auto: as many layers as fit next to the decode weights, added after the rest of the model is built (Model._uni_auto)
        return set()
    if v == "all":
        return set(layer_ids)
    out = set()
    for part in v.split(","):
        lo, _, hi = part.partition("-")
        out |= set(range(int(lo), int(hi or lo) + 1))
    return out & set(layer_ids)


from models.demos.blackhole.deepseek_v41_flash.tt.prefill_model import T

GATE_CUTOFFS = json.loads((Path(__file__).resolve().parents[1] / "configs" / "gate_cutoffs.json").read_text())


class Model:
    def __init__(self, mesh_device, args, max_ctx, num_pages=None, kv_dtype=ttnn.bfloat16, log=print):
        """args: ``DSV41ModelArgs`` (layer ids, users per row). ``max_ctx``: longest context (prompt + generated) of any user. ``num_pages``: pages of
        128 tokens PER MESH ROW shared by the users of that row (default users_per_row * ceil(max_ctx / 128))."""
        self.md, self.log = mesh_device, log
        self._keep = (
            None  # batch independent state that ``reconfigure`` carries over (filled at the end of every build)
        )
        self._build(args, max_ctx, num_pages, kv_dtype)

    def _build(self, args, max_ctx, num_pages, kv_dtype):
        """Everything of the model: the batch independent weights (attention / shared / mHC / norms, routed experts, embedding, head, Engram device
        weights + host tables) and the state that depends on the users per row ``U`` / ``max_ctx`` (KV pool, page tables, rings, per-user buffers, MoE
        dispatch buffers, prefill model, ...). With ``self._keep`` set (``reconfigure``) the weights that do not depend on them are reused.
        """
        mesh_device, log = self.md, self.log
        keep = self._keep
        self.args, self.kv_dtype = args, kv_dtype
        self.rows, self.cols = tuple(mesh_device.shape)
        self.U = args.users_per_row
        self.uni = uni_policy.decide(
            mesh_device, self.U, log
        )  # unified prefill MoE (+ ring weights) on, or the logged fallback
        self.B = self.rows * self.U
        self.max_ctx = max_ctx
        self.layer_ids = list(args.layer_ids)
        self.timing = {}
        ratios = {R.model_args().compress_ratios[L] for L in self.layer_ids}
        dl = 512 if 1 in ratios else 1024 if 2 in ratios else 1 << 30
        ui = os.environ.get("DSV41_INDEXER", "auto")
        self.use_indexer = (
            (max_ctx > dl) if ui == "auto" else ui == "1"
        )  # indexer top-512 (decode) + sparse prefill: needed beyond 512 compressed entries
        self.dec_idx, self.index_owner = {}, {}
        pages_per_user = -(-(max_ctx + 128) // PAGE_TOKENS)
        self.num_pages = num_pages or self.U * pages_per_user
        t0 = time.time()
        if keep is None:
            self.l1_base = self._l1_alloc()
        chain = DSV41DecodeChain(mesh_device, users_per_row=self.U, log=log)  # mesh config, CCL, shared MoE buffers
        self.chain = chain
        self.mc, self.ccl = chain.mesh_config, chain.ccl
        self.pool = PagedKVPool(
            mesh_device,
            self.U,
            self.num_pages,
            len(self.layer_ids),
            max_ctx + 128,
            ring_rows=int(
                os.environ.get("DSV41_RING_ROWS", "128")
            ),  # 160 = window + speculative-decoding slack (tt/spec_paged.py RING_SPEC)
            dtype=kv_dtype,
        )
        self.sink = PagedStateSink(mesh_device, self.pool, self.U)
        self.sources, self.attns, self.built, self.step_groups, pls = {}, {}, [], {}, []
        sh = _Shards()
        pool = ThreadPoolExecutor(max_workers=2)
        futs = {}
        submit = lambda L: (
            futs.setdefault(
                L, pool.submit(load_layer, L, True, max_ctx + 128, self.use_indexer, load_experts=keep is None)
            )
            if L in self.layer_ids
            else None
        )
        for L in self.layer_ids[:2]:
            submit(L)
        first_pmoe = None
        for i, L in enumerate(self.layer_ids):
            submit(L + 1), submit(L + 2)
            w = futs.pop(L).result()
            attn = self._build_attention(L, w, slot=i)
            layer = DSV41Layer(
                mesh_device,
                self.mc,
                self.ccl,
                attn,
                w["norms"],
                w["mhc"],
                w["moe"],
                gate_bias_shift=GATE_CUTOFFS[str(L)],
                users_per_row=self.U,
                moe_buffers=chain.moe_buffers,
                expert_state=None if keep is None else keep["experts"][L],
            )
            if chain.moe_buffers is None:
                chain.moe_buffers = layer.moe.decode.buffers
            pa = DSV41PrefillAttention(attn, w["attn"]["attn_sink"].float())
            pa.state_sink = functools.partial(self.sink.write, attn)
            attn.prefill = pa
            pmoe = DSV41PrefillMoE(layer.moe, T=T, buffers=None if first_pmoe is None else first_pmoe.decode.buffers)
            first_pmoe = first_pmoe or pmoe
            pls.append((L, DSV41PrefillLayer(layer, pa, pmoe, T=T)))
            if self.uni and L in UNI_LAYERS(
                self.layer_ids
            ):  # deepseek_prefill routed-expert pipeline (tt/prefill_unified_moe.py)
                from models.demos.blackhole.deepseek_v41_flash.tt.prefill_unified_moe import DSV41UnifiedMoE

                if uni_policy.ring_requested():
                    # ONE weight copy: the unified op reads the decode moe_compute (ring layout) expert weights in place (RING_WEIGHTS mode, needs the
                    # private/updated _ttnncpp); decode and interleaved prefill share the same read-only buffers
                    es = layer.moe.decode.expert_state
                    pls[-1][1].umoe = DSV41UnifiedMoE(mesh_device, L, log=log, ring=(es.tt_w0_w1, es.tt_w2))
                else:
                    pls[-1][1].umoe = DSV41UnifiedMoE(
                        mesh_device, L, log=log, weights=None if keep is None else keep["umoe"].get(L)
                    )
            key = getattr(attn, "ratio", 0)
            if key not in self.step_groups:
                self.step_groups[key] = DSV41StepState_paged(attn, max_ctx + 64, self.use_indexer)
            self.built.append((L, layer, key))
            self.attns[L] = attn
            del w
            gc.collect()
            if i % 5 == 0 or i == len(self.layer_ids) - 1:
                log(f"built layer {L} ({time.time() - t0:.0f}s)")
        self.log_dram("after layers built")
        if os.environ.get("DSV41_PF_SPARSE") == "1" or self.use_indexer:
            self.enable_prefill_sparse(c_max=int(os.environ.get("DSV41_PF_CMAX", "2048")))
        self.log_dram("after prefill sparse")
        engram_ids = [l for l in (1, 14) if l in self.layer_ids]
        self.engram_ids = engram_ids
        self.host_rows = (
            HostEngramRows(
                tuple(engram_ids),
                max_batch_size=self.B,
                max_seq_len=max_ctx + 1024,
                tables=None if keep is None else keep["tables"],
            )
            if engram_ids
            else None
        )
        if self.host_rows is not None:
            self.hasher = RaggedNgramHash(self.host_rows.engram.hash)
            if os.environ.get("DSV41_ENGRAM_RAM", "1") == "1" and not all(
                t.in_ram for t in self.host_rows.tables.values()
            ):
                t1 = time.time()
                self.host_rows.load_ram()
                log(f"Engram tables in process memory ({time.time() - t1:.0f}s)")
        if keep is None:
            dev_engram = {
                l: DSV41DeviceEngram(mesh_device, l, sh, mesh_config=self.mc, ccl=self.ccl) for l in engram_ids
            }
            embedding = DSV41DeviceEmbedding(mesh_device, sh.get("embed.weight"), users_per_row=self.U)
            self.head = DSV41DeviceHead(
                mesh_device, sh.get("norm.weight").float(), sh.get("head.weight"), norm_eps=R.model_args().norm_eps
            )
        else:  # batch independent device weights of the first build
            dev_engram = keep["dev_engram"]
            for (
                e
            ) in (
                dev_engram.values()
            ):  # the CCL semaphores were released with the rest of the L1 state (see _release_batch_state)
                e.mesh_config, e.ccl = self.mc, self.ccl
            embedding = keep["embedding"].rebatch(self.U)
            self.head = keep["head"]
        self.dec = DSV41Decoder(mesh_device, self.built, embedding, self.head, dev_engram, step_states=self.step_groups)
        self.dec.mesh_config, self.dec.ccl = self.mc, self.ccl
        self.prefill_model = GenPrefillModel(mesh_device, pls, embedding, self.head, dev_engram, self.host_rows, self.U)
        self.prefill_model.set_head_sampling(self.mc, self.ccl)
        self.prefill_model.sink = self.sink
        self.engram_kin = {l: e.kin for l, e in dev_engram.items()}
        self.rows_cat = None
        self.trace_id = None
        self.admitted = False
        self.pool_pages_free = None
        if self.uni and os.environ.get("DSV41_UNI_LAYERS", "all").startswith("auto"):
            self._uni_auto(pls)
        self._keep = {
            "experts": {L: layer.moe.decode.expert_state for L, layer, _ in self.built},
            "umoe": {  # unified-layout expert weights that are a COPY (not the ring-layout decode weights read in place): batch independent, kept too
                L: (pl.umoe.gate_projs, pl.umoe.up_projs, pl.umoe.down_projs)
                for L, pl in pls
                if getattr(pl, "umoe", None) is not None
                and pl.umoe.gate_projs[0] is not self._keep_es(L).get("tt_w0_w1", object())
            },
            "tables": None if self.host_rows is None else self.host_rows.tables,
            "dev_engram": dev_engram,
            "embedding": embedding,
            "head": self.head,
        }
        self.log_dram("model built")
        if os.environ.get("DSV41_MEMLOG") in (
            "1",
            "2",
            "3",
        ):  # trace the DRAM cost of every phase of the traced-chunk prefill
            pm_ = self.prefill_model

            def wrap(name):
                orig = getattr(pm_, name)

                def f(*a, **k):
                    self.log_dram(f"before {name}")
                    if name == "forward_device":
                        self.l1_report("before forward_device")
                    try:
                        r = orig(*a, **k)
                    except Exception:
                        if name == "forward_device":
                            self.l1_report("AT FAILURE")
                        raise
                    self.log_dram(f"after  {name}")
                    return r

                setattr(pm_, name, f)

            for nm in (
                "setup_dyn",
                "teardown_dyn",
                "alloc_inputs",
                "forward_device",
                "begin_chunk",
                "last_logits_traced",
            ):
                if hasattr(pm_, nm):
                    wrap(nm)
            for lid_, pl_ in pm_.layers:  # per-layer DRAM growth inside a (compile or captured) chunk forward
                if lid_ in (0, 2, 3, 10, 20, 21, 30, 39):

                    def fl(*a, _o=pl_.forward, _l=lid_, **k):
                        r = _o(*a, **k)
                        self.log_dram(f"  layer {_l} forward done")
                        return r

                    pl_.forward = fl

                    def wrapm(obj, meth, label, _l=lid_):
                        if not hasattr(obj, meth):
                            return
                        o_ = getattr(obj, meth)

                        def g(*a, **k):
                            self.log_dram(f"  L{_l} before {label}")
                            r = o_(*a, **k)
                            self.log_dram(f"  L{_l} after  {label}")
                            return r

                        setattr(obj, meth, g)

                    if lid_ in (0, 2):  # finer: which stage of the eager compile forward holds the transient peak
                        wrapm(pl_.pa, "forward_dyn", "attention.forward_dyn")
                        wrapm(pl_.pmoe, "forward", "moe.forward")
                        if getattr(pl_.pa, "sparse", None) is not None:
                            wrapm(pl_.pa.sparse, "attend_dyn", "sparse.attend_dyn")
                            if pl_.pa.sparse.indexer is not None:
                                wrapm(pl_.pa.sparse.indexer, "select_dyn", "indexer.select_dyn")
            if getattr(
                ttnn, "_dsv41_memlog_wrapped", False
            ):  # a rebuild (reconfigure): the capture wrappers of the first build stay
                ob, oe = None, None
            else:
                ob, oe = ttnn.begin_trace_capture, ttnn.end_trace_capture
            if ob is not None:
                ttnn._dsv41_memlog_wrapped = True
                ttnn.begin_trace_capture = lambda *a, **k: (
                    self.log_dram("before begin_trace_capture"),
                    setattr(self, "_capturing", True),
                    ob(*a, **k),
                )[2]
                ttnn.end_trace_capture = lambda *a, **k: (
                    oe(*a, **k),
                    setattr(self, "_capturing", False),
                    self.log_dram("after end_trace_capture"),
                )[0]
        log(
            f"model built: {len(self.layer_ids)} layers, U={self.U} users/row (batch {self.B}), pool {self.num_pages} pages/row ({time.time() - t0:.0f}s)"
        )

    def _keep_es(self, L):
        """{'tt_w0_w1': ...} of the routed-expert state of layer L (to tell the ring-layout weights read in place from a separate unified-layout copy)."""
        es = next(layer.moe.decode.expert_state for l, layer, _ in self.built if l == L)
        return {"tt_w0_w1": getattr(es, "tt_w0_w1", None)}

    def _uni_auto(self, pls):
        """DSV41_UNI_LAYERS=auto: the unified-layout expert weights (~54 MiB/bank/layer) are a SECOND copy next to the moe_compute decode weights, so only
        some layers fit. Add them layer by layer, after everything else (pool, decode weights, Engram, head) is resident, while the free DRAM per bank
        stays above DSV41_UNI_RESERVE_MIB (default 450: the growth from 'model built' to the peak of the traced prefill + decode, 130-390 MiB in the grid
        logs, plus the unified shared buffers). DSV41_UNI_MAX=<n> caps the count."""
        from models.demos.blackhole.deepseek_v41_flash.tt.prefill_unified_moe import DSV41UnifiedMoE

        mib = 2**20
        per = float(os.environ.get("DSV41_UNI_LAYER_MIB", "54"))
        reserve = float(os.environ.get("DSV41_UNI_RESERVE_MIB", "450"))
        cap = int(os.environ.get("DSV41_UNI_MAX", "99"))
        n = 0
        for L, pl in pls:
            mv = ttnn.get_memory_view(self.md, ttnn.BufferType.DRAM)
            free = mv.total_bytes_free_per_bank / mib
            if n >= cap or free - per < reserve:
                break
            kept = (self._keep or {}).get("umoe", {}).get(L)
            pl.umoe = DSV41UnifiedMoE(self.md, L, log=self.log, weights=kept)
            n += 1
        self.uni_layers = [L for L, pl in pls if pl.umoe is not None]
        self.log(
            f"UNI auto: unified prefill MoE on {n}/{len(pls)} layers {self.uni_layers} (reserve {reserve} MiB/bank)"
        )

    def log_dram(self, tag):
        """DSV41_MEMLOG=1: allocated / free DRAM per bank (MiB) at this point (diagnosing capacity / leaks across ISLs)."""
        if os.environ.get("DSV41_MEMLOG") not in ("1", "2", "3"):
            return
        if getattr(self, "_capturing", False):
            return
        ttnn.synchronize_device(self.md)
        mv = ttnn.get_memory_view(self.md, ttnn.BufferType.DRAM)
        l1 = ttnn.get_memory_view(self.md, ttnn.BufferType.L1)
        self.log(
            f"MEMLOG {tag:40s} allocated {mv.total_bytes_allocated_per_bank / 2**20:8.1f} MiB/bank  free {mv.total_bytes_free_per_bank / 2**20:8.1f}  largest free block {mv.largest_contiguous_bytes_free_per_bank / 2**20:8.1f}"
            f"  | L1 alloc {l1.total_bytes_allocated_per_bank} B largest free {l1.largest_contiguous_bytes_free_per_bank} B"
        )
        if os.environ.get(
            "DSV41_L1_DUMP"
        ):  # per-buffer allocator dump (generated/reports/<prefix>detailed_memory_usage.csv), one device
            try:
                ttnn.dump_device_memory_state(
                    self.md, prefix=f"{os.environ['DSV41_L1_DUMP']}_{tag.strip().replace(' ', '_')}_"
                )
            except Exception as e:  # noqa: BLE001
                self.log(f"L1 dump failed: {type(e).__name__}: {str(e)[:200]}")

    def live_tensor_report(self, tag, top=25):
        """DSV41_MEMLOG=2: the largest live ttnn tensors (python objects) with their referrer types: which allocation survives a teardown."""
        if os.environ.get("DSV41_MEMLOG") != "2":
            return

        sizes = {
            "BFLOAT16": 2,
            "FLOAT32": 4,
            "BFLOAT8_B": 1.06,
            "UINT32": 4,
            "INT32": 4,
            "UINT16": 2,
            "BFLOAT4_B": 0.56,
            "UINT8": 1,
        }
        rows, seen = [], set()
        for o in gc.get_objects():
            if type(o).__name__ == "Tensor" and type(o).__module__.startswith("ttnn"):
                try:
                    if id(o) in seen or not o.is_allocated():
                        continue
                    seen.add(id(o))
                    n = 1
                    for d in o.shape:
                        n *= int(d)
                    b = n * sizes.get(str(o.dtype).split(".")[-1], 2)
                    ref = [type(r).__name__ for r in gc.get_referrers(o)][:3]
                    rows.append((b, tuple(o.shape), str(o.dtype).split(".")[-1], ref))
                except Exception:
                    pass
        rows.sort(key=lambda r: -r[0])
        tot = sum(r[0] for r in rows)
        self.log(
            f"LIVE {tag}: {len(rows)} tensors, {tot / 2**20:.0f} MiB (logical, per mesh shard x devices counted once)"
        )
        for b, s, d, ref in rows[:top]:
            self.log(f"LIVE   {b / 2**20:8.1f} MiB {s} {d} refs={ref}")

    def l1_report(self, tag):
        """DSV41_MEMLOG=3: L1 allocator state per bank + every live python ttnn tensor that sits in L1 (the persistent L1 buffers that squeeze the static CBs of
        the next program: 'Statically allocated circular buffers ... clash with L1 buffers')."""
        if os.environ.get("DSV41_MEMLOG") != "3":
            return
        ttnn.synchronize_device(self.md)
        mv = ttnn.get_memory_view(self.md, ttnn.BufferType.L1)
        self.log(
            f"L1VIEW {tag}: allocated {mv.total_bytes_allocated_per_bank} B/bank, largest free {mv.largest_contiguous_bytes_free_per_bank} B"
        )
        rows = []
        for o in gc.get_objects():
            try:
                if type(o).__name__ == "Tensor" and type(o).__module__.startswith("ttnn") and o.is_allocated():
                    mc = o.memory_config()
                    if mc.buffer_type == ttnn.BufferType.L1:
                        rows.append(
                            (
                                tuple(o.shape),
                                str(o.dtype),
                                str(o.layout),
                                str(mc.memory_layout),
                                str(getattr(o, "shard_spec", None))[:80],
                            )
                        )
            except Exception:
                pass
        self.log(f"L1VIEW {tag}: {len(rows)} live L1 tensors")
        for r in rows[:40]:
            self.log(f"L1VIEW   {r}")

    # ---- construction ---------------------------------------------------------------------------------------------------------------
    def _build_attention(self, L, w, slot):
        meta = w["meta"]
        kw = dict(users_per_row=self.U)
        a = (self.md, self.mc, self.ccl, w["attn"], w["freqs_cis"])
        if meta["ratio"] == 0:
            return DSV41PagedAttention(*a, self.pool, slot, **kw)
        idx = self._build_indexer(L, meta, w)
        if meta["is_kv_source"]:
            attn = DSV41PagedCompressedAttention(
                *a, meta["ratio"], w["compressor"], self.pool, slot, L, indexer=idx, **kw
            )
            self.sources[L] = attn
            if (
                meta["ratio"] > 1
            ):  # the decode compressor reads the previous token's [kv|score]; the prefill hand-off fills it per user
                attn.prev_cs = ttnn.zeros(
                    [1, 1, self.U, 1024], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.md
                )
            return attn
        return DSV41PagedCompressedAttention(
            *a,
            meta["ratio"],
            None,
            self.pool,
            slot,
            meta["kv_source"],
            source=self.sources[meta["kv_source"]],
            indexer=idx,
            **kw,
        )

    def _build_indexer(self, L, meta, w):
        """Decode indexer of an index-source layer (same construction as DSV41DecodeChain's paged builder); key owners get the key slab, the other index
        sources alias the slab of their kv source."""
        if not (self.use_indexer and "indexer" in w):
            return None
        from models.demos.blackhole.deepseek_v41_flash.tt.indexer import DSV41DecodeIndexer, default_backend

        iw = w["indexer"]
        n_alloc = -(-(self.max_ctx // meta["ratio"] + 32) // 32) * 32
        idx = DSV41DecodeIndexer(
            self.md,
            iw,
            w["freqs_cis"],
            users_per_row=self.U,
            n_alloc=n_alloc,
            ratio=meta["ratio"],
            key_dtype=ttnn.bfloat8_b,
            fp4_q=True,
            backend=default_backend(self.max_ctx // meta["ratio"]),
        )
        if meta["is_kv_source"]:
            idx.set_key_weights(iw["wk"], iw["k_norm"])
            idx.load_keys(torch.zeros(self.B, 0, 128))
            self.index_owner[L] = idx
        else:
            idx.k_cache = self.index_owner[meta["kv_source"]].k_cache
        self.dec_idx[L] = idx
        return idx

    def enable_prefill_sparse(self, c_max=2048, max_tokens=None, enable=True):
        """Prefill indexer top-512 + sparse_sdpa for the compressed layers (tt/prefill_sparse.py): exact CSA selection for prompts with > 512 compressed
        entries. Also by env DSV41_PF_SPARSE=1 at model build. ``c_max``: largest prefill chunk. enable=False detaches (dense prefill attention).
        """
        from models.demos.blackhole.deepseek_v41_flash.tt.prefill_sparse import attach_prefill_sparse

        pas = {L: self.attns[L].prefill for L in self.layer_ids}
        if not enable:
            for pa in pas.values():
                pa.sparse = None
            return {}
        sh = _Shards()
        idx_w = {}
        for L in self.layer_ids:
            if L in R.model_args().index_source_layers and L < 40:
                idx_w[L] = load_layer(L, with_moe=False, max_seq_len=8, with_indexer=True)["indexer"]
        sinks = {L: sh.get(f"layers.{L}.attn.attn_sink").float() for L in self.layer_ids}
        self.prefill_sparse = attach_prefill_sparse(
            pas, idx_w, self.U, max_tokens or self.max_ctx, c_max, sinks, decode_indexers=self.dec_idx, enable=True
        )
        return self.prefill_sparse

    def dense_limit(self):
        """Longest context (tokens) whose compressed attention is exact WITHOUT the indexer: every compressed entry is selected while there are <= 512 of them
        (ratio 1: ctx <= 512; ratio 2: ctx <= 1024)."""
        ratios = {getattr(self.attns[L], "ratio", 0) for L in self.layer_ids}
        return 512 if 1 in ratios else 1024 if 2 in ratios else 1 << 30

    def check_context_supported(self, ctx):
        if ctx > self.dense_limit() and not self.use_indexer and os.environ.get("DSV41_ALLOW_DENSE") != "1":
            raise NotImplementedError(
                f"context {ctx} > {self.dense_limit()}: more than 512 compressed entries need the indexer top-512 selection in prefill and decode "
                "(paged decode indexer exists in tt/indexer.py but its key-slab write from the prefill and the prefill-side selection are not wired; "
                "DSV41_ALLOW_DENSE=1 runs the dense approximation for experiments)"
            )

    # ---- paged users ----------------------------------------------------------------------------------------------------------------
    def admit_users(self, prompt_lens, max_new_tokens, active=None):
        """(Re)admit every user: pages for the prompt + the generated tokens (+1), page table uploaded into the persistent device tensor.
        ``active`` (bool [B], serving / vLLM interface only): re-admit only the users where it is True; the other users keep their pages.
        """
        users = range(self.B) if active is None else [b for b in range(self.B) if bool(active[b])]
        for b in users:
            r, k = self.pool.user_key(b)
            if k in self.pool.allocs[r].pages:
                self.pool.release(b)
        for b in users:
            self.pool.admit(b, int(prompt_lens[b]) + 1, reserve_tokens=max_new_tokens)
        self.pool.sync_page_table()
        self.admitted = True

    # ---- prefill ----------------------------------------------------------------------------------------------------------------------
    def prepare_inputs_prefill(self, tokens, prompt_lens, chunk=None):
        """tokens [B, L] (right padded, any pad value) + prompt_lens [B] -> (padded tokens [B, W] with every user's tail = its last real token,
        chunk plan [(s0, C)])."""
        B, L = tokens.shape
        lens = torch.as_tensor(prompt_lens).long()
        S = int(lens.max())
        plan = self.prefill_model.chunk_plan(S, chunk)
        W = plan[-1][0] + plan[-1][1]
        tp = torch.zeros(B, W, dtype=torch.long)
        for b in range(B):
            n = int(lens[b])
            tp[b, :n] = tokens[b, :n].long()
            tp[b, n:] = tokens[b, n - 1]
        return tp, plan

    def prefill_forward(
        self,
        tokens,
        prompt_lens,
        chunk=None,
        max_new_tokens=0,
        want_logits=False,
        hook=None,
        enable_trace=True,
        s_pad_max=None,
        active=None,
    ):
        """``active`` (bool [B], serving / vLLM interface only, traced-chunk path only): prefill only the users where it is True; the others (prompt_lens
        must be 0 for them) keep their KV / ring / compressor state, pages and Engram history untouched (their rows are computed on filler tokens but every
        hand-off write is masked) and get first token -1. The index-key export (decode indexer on, ``use_indexer``) rewrites the key slab of EVERY user of a
        mesh row, so with the indexer on the caller must not leave a live user inactive (it re-prefills them: ``generator_vllm``).
        Traced-chunk prefill (default): ONE chunk of ``chunk`` tokens per user is captured once and REPLAYED for every chunk of the prompt, the per-chunk values
        (positions, rope rows, masks, page-table / write-index tensors of the hand-off) being persistent device tensors refreshed before each replay
        (tt/prefill_dyn.py of the prefill model + ``PagedStateSink.update``). ``enable_trace=False``: the same dynamic chunk driven eagerly (compile / reference).
        Traced-chunk prefill is the default (verified at 40 layers, ISL 128); DSV41_PREFILL_DYN=0 selects the eager per-chunk reference path.
        """
        if (
            hasattr(self.prefill_model, "run_traced_chunks") and os.environ.get("DSV41_PREFILL_DYN", "1") != "0"
        ):  # UNVALIDATED until h44p validates the traced-chunk path
            return self.prefill_forward_dyn(
                tokens, prompt_lens, chunk, max_new_tokens, want_logits, enable_trace, s_pad_max, active
            )
        assert active is None, "partial prefill (active mask) needs the traced-chunk path (DSV41_PREFILL_DYN != 0)"
        return self.prefill_forward_legacy(tokens, prompt_lens, chunk, max_new_tokens, want_logits, hook)

    def _filler_tokens(self, n_rows, width):
        """Token ids for the rows of users that are NOT prefilled in a partial (serving) prefill: the model computes those rows anyway (every row of the batch goes through every layer).
        A constant filler (token 0) routes all of them to the SAME experts (hash-routed layers 0-2 by token id, learned layers by the identical hidden states): ~16k identical tokens per
        chunk on a few experts, far from the balanced routing of a real batch. The vLLM serving sweep hung deterministically (all devices in ReduceScatter, 4 devices in the unified-MoE
        CombineDeviceOperation, log dsv4-logs/triage/vllm_hang_run2_full.log) on the 5th request in a row; diverse deterministic pseudo-random ids keep the routing spread like a batch of
        real prompts. Cached per shape."""
        cache = self.__dict__.setdefault("_filler_cache", {})
        key = (n_rows, width)
        if key not in cache:
            g = torch.Generator().manual_seed(20260507)
            cache.clear()  # one shape at a time (S_pad changes rarely)
            cache[key] = torch.randint(1000, int(self.args.vocab_size), (n_rows, width), generator=g)
        return cache[key]

    def admit_idle_users(self):
        """Serving / vLLM interface only: ``decode_forward`` steps ALL B users (``pool.ensure`` grows every user), so a user that was never prefilled or whose request finished
        (``pool.release``) needs an allocator entry: it gets ONE page (it decodes a pad token at position 0). -> True when the page table changed (uploaded).
        """
        changed = False
        for b in range(self.B):
            r, k = self.pool.user_key(b)
            if k not in self.pool.allocs[r].pages:
                self.pool.admit(b, 1)
                changed = True
        if changed:
            self.pool.sync_page_table()
        return changed

    def prepare_for_traces(self, lens):
        """Allocate EVERY persistent device tensor and run every compile pass BEFORE the first trace capture: a persistent tensor created after a capture can
        land on the (freed) intermediate buffers of a captured trace and is then clobbered by the next replay (or corrupts it). Creates the head's column-id
        constant, the decode device-loop buffers (tokens / positions), the packed Engram-rows buffer, and compiles one decode step (state restored).
        """
        if getattr(self, "_warm", False):
            return
        head = self.head
        if getattr(head, "_col_ids", None) is None:
            head._col_ids = ttnn.from_torch(
                torch.arange(head.cols, dtype=torch.float32).reshape(1, 1, 1, head.cols),
                device=self.md,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.md),
            )
        zeros = torch.zeros(self.B, dtype=torch.long)
        self._set_loop_state(zeros, torch.as_tensor(lens).long())
        if self.engram_ids:
            k = sum(self.engram_kin[l] for l in self.engram_ids)
            host = ttnn.from_torch(
                torch.zeros(self.B, 1, 1, k, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self._mp(),
            )
            self._upload_rows(host)
        if (
            os.environ.get("DSV41_UNI_NODECODE") == "1"
        ):  # prefill-only unified mode: no decode weights, no decode compile pass
            self._warm = True
            return
        snaps = self.dec.snapshot_states()
        self.last_logits = (
            self.dec.forward()
        )  # compile pass (writes pool rows at position = prompt length: rewritten by the prefill / first real step)
        ttnn.synchronize_device(self.md)
        self.dec.restore_states(snaps)
        del snaps
        self._set_loop_state(zeros, torch.as_tensor(lens).long())
        self._warm = True

    def _export_index_keys(self, lens):
        """Hand-off of the index keys: the prefill indexers' key slabs (key owners 2/8/14/20) -> the decode indexers' key slabs ``k_cache``."""
        if not self.use_indexer or not getattr(self, "prefill_sparse", None):
            return
        for L, dec in self.dec_idx.items():
            if L not in self.index_owner:
                continue  # layers 24..36 alias layer 20's slab
            sp = self.attns[L].prefill.sparse
            ix = None if sp is None else sp.indexer
            if (
                ix is None
                or ix.key_owner is not None
                or not getattr(sp, "dyn_on", True)
                and os.environ.get("DSV41_PREFILL_DYN", "1") != "0"
            ):
                continue
            _t = time.perf_counter()
            ix.export_keys(dec.k_cache, int(torch.as_tensor(lens).max()) // self.attns[L].ratio)
            if os.environ.get("DSV41_PF_TIMING") == "1":
                ttnn.synchronize_device(self.md)
                self.log(f"  export_keys layer {L}: {(time.perf_counter() - _t) * 1e3:.0f} ms")
        ttnn.synchronize_device(self.md)

    def _post_chunk(self, s0, C):
        """Host-only (no device allocation): read the ragged-head trace outputs of this chunk for the users whose last prompt token is inside it."""
        pm, U, rows, cols = self.prefill_model, self.U, self.rows, self.cols
        K = C // 32
        for u in range(U):
            lg, tk = pm.head_out[u]
            todo = [
                (r, (int(self._last_pos[r * U + u]) - s0) % 32, r * U + u)
                for r in range(rows)
                if s0 <= int(self._last_pos[r * U + u]) < s0 + C
            ]
            if not todo:
                continue
            devs = ttnn.get_device_tensors(ttnn.from_device(tk))
            full = pm.head.gather_logits(lg).reshape(rows, 32, -1) if self._want_logits else None
            for r, off, b in todo:
                tok = int(ttnn.to_torch(devs[r * cols]).reshape(-1)[off])
                self._res[b] = (tok, None if full is None else full[r, off].clone())

    def prefill_forward_dyn(
        self, tokens, prompt_lens, chunk, max_new_tokens, want_logits, enable_trace, s_pad_max, active=None
    ):
        B = self.B
        assert tokens.shape[0] == B
        lens = torch.as_tensor(prompt_lens).long()
        if active is not None:
            active = torch.as_tensor(active).bool()
            assert not bool((lens[~active] != 0).any()), "inactive users must have prompt length 0"
            assert bool((lens[active] > 0).all()), "active users need a non-empty prompt"
        assert int(lens.max()) + max_new_tokens <= self.max_ctx, "prompt + generated tokens exceed the model's max_ctx"
        self.check_context_supported(int(lens.max()) + max_new_tokens)
        t_start = time.perf_counter()
        S = int(lens.max())
        C = chunk or -(-S // 128) * 128
        S_pad = -(-S // C) * C
        # DSV41_PREFILL_SPAD_MAX (tokens): size the chunk trace's per-context tables (sparse / indexer / latent) ONCE for this padded length, so a later
        # LONGER prompt of the same chunk size replays the same capture instead of re-capturing (and OOMing). Only the real chunks run.
        spad_env = int(os.environ.get("DSV41_PREFILL_SPAD_MAX", "0") or 0)
        if (
            not spad_env and os.environ.get("DSV41_PREFILL_ROW_TOKENS") == "auto2"
        ):  # auto2: sized once for the build's max context
            spad_env = int(self.max_ctx)
        s_pad_max = max(s_pad_max or 0, spad_env)
        s_pad_max = -(-s_pad_max // C) * C if s_pad_max else None
        S_dyn = max(S_pad, s_pad_max or 0)
        self.log(f"  prefill_dyn: admit users")
        self.admit_users(lens, max_new_tokens, active)
        self.sink.set_lengths(lens)
        self.sink.bind(C)
        bis = os.environ.get("DSV41_BISECT", "")
        if "nosink" in bis and not getattr(self, "_nosink_done", False):
            for _, pl in self.prefill_model.layers:
                pl.pa.state_sink = None
            self._nosink_done = True
        self.log_dram("prefill start")
        self.prepare_for_traces(lens)
        self.log_dram("after prepare_for_traces")
        pm = self.prefill_model
        pm.timing = {}
        tp = torch.zeros(B, S_pad, dtype=torch.long)
        for b in range(B):
            n = int(lens[b])
            tp[b, :n] = tokens[b, :n].long()
            tp[b, n:] = tokens[b, n - 1]
        if active is not None and not bool(active.all()):
            tp[~active] = self._filler_tokens(int((~active).sum()), S_pad)
        if active is not None and self.host_rows is not None:
            # the ragged hash writes every user's token history (``cache``): keep the history of the inactive (decoding) users
            st_cache, keep = self.hasher.st.cache, (~active).nonzero().reshape(-1)
            saved = st_cache[keep].clone()
        hashes = self.hasher(tp, torch.zeros(B, dtype=torch.long)) if self.host_rows is not None else None
        if active is not None and self.host_rows is not None:
            st_cache[keep] = saved
        self._res, self._last_pos, self._want_logits = {}, lens - 1, want_logits
        if active is not None:  # inactive users: no last token inside any chunk -> first token -1
            self._res = {b: (-1, None) for b in range(B) if not bool(active[b])}
        if getattr(self, "_hooks_set", None) is not pm:
            pm.pre_replay_hooks.append(lambda s0, C_: self.sink.update(s0, C_))
            hook = lambda s0, C_: self._post_chunk(s0, C_)
            # the async replay loop only synchronizes (and calls the hook) for chunks that hold some user's last prompt token
            hook.needs_sync = lambda s0, C_: bool(((self._last_pos >= s0) & (self._last_pos < s0 + C_)).any())
            hook.is_post_chunk = True
            pm.post_replay_hooks.append(hook)
            self._hooks_set = pm
        if "nohead" in bis:
            pm.post_replay_hooks[:] = [h for h in pm.post_replay_hooks if not getattr(h, "is_post_chunk", False)]
        self.log(
            f"  prefill_dyn: run chunks (trace={enable_trace}, C={C}, S_pad={S_pad}, S_dyn={S_dyn}, bisect={bis!r})"
        )
        if enable_trace and getattr(pm, "dyn", None) is not None and (pm.dyn.C != C or pm.dyn.S_pad != S_dyn):
            self.log_dram("before teardown (old S_pad %d)" % pm.dyn.S_pad)
            pm.teardown_dyn()
            pm.dyn_out = None  # the last chunk's output streams (fp32 [32,1,4,5120] tiles: ~320 MiB/bank per 4096 tokens/row) of the captured trace
            pm.head_out = []
            gc.collect()
            self.log_dram("after teardown")
            self.live_tensor_report("after teardown")
        if enable_trace:
            pm.run_traced_chunks(tp, C, hashes=hashes, S_pad_max=s_pad_max)
            self.log_dram("after run_traced_chunks")
            self.log("  prefill_dyn: chunks done")
        else:
            pm.setup_dyn(C, S_dyn)
            for _, pl in pm.layers:
                pl.pa.reset_dyn()
            for ci in range(S_pad // C):
                s0 = ci * C
                hs = None if hashes is None else hashes[:, s0 : s0 + C]
                bufs = pm.alloc_inputs(C)
                pm.upload_inputs(pm.prep_inputs(tp[:, s0 : s0 + C], hs), bufs)
                pm.begin_chunk(s0, C)
                pm.forward_device(bufs, S, s0, C, dyn=True)
                ttnn.synchronize_device(self.md)
                self._post_chunk(s0, C)
        _t0 = time.perf_counter()
        ttnn.synchronize_device(self.md)
        _t1 = time.perf_counter()
        self._export_index_keys(lens)
        _t2 = time.perf_counter()
        if os.environ.get("DSV41_PF_TIMING") == "1":
            self.log(
                f"  prefill_dyn tail: sync {(_t1 - _t0) * 1e3:.0f} ms, export_index_keys {(_t2 - _t1) * 1e3:.0f} ms"
            )
        self.log_dram("prefill end")
        self.timing = dict(pm.timing, total=time.perf_counter() - t_start)
        first = torch.tensor([self._res[b][0] for b in range(B)], dtype=torch.long)
        logits = (
            torch.stack(
                [
                    self._res[b][1] if self._res[b][1] is not None else torch.zeros(self.args.vocab_size)
                    for b in range(B)
                ]
            )
            if want_logits
            else None
        )
        return first, logits

    def prefill_forward_legacy(self, tokens, prompt_lens, chunk=None, max_new_tokens=0, want_logits=False, hook=None):
        """tokens [B, L] right-padded prompts, prompt_lens [B] -> (first generated token [B] (greedy), logits [B, vocab] fp32 or None). Leaves, for every
        user: KV pages + rings + compressor state of all layers in the decode pool, the Engram token history on the host.
        """
        B = self.B
        assert tokens.shape[0] == B
        lens = torch.as_tensor(prompt_lens).long()
        assert int(lens.max()) + max_new_tokens <= self.max_ctx, "prompt + generated tokens exceed the model's max_ctx"
        self.check_context_supported(int(lens.max()) + max_new_tokens)
        self.timing = {}
        t_start = time.perf_counter()
        self.admit_users(lens, max_new_tokens)
        self.sink.set_lengths(lens)
        tp, plan = self.prepare_inputs_prefill(tokens, lens, chunk)
        pm = self.prefill_model
        pm.timing = {}
        pm.S = int(lens.max())
        for _, pl in pm.layers:
            pl.pa.begin()
        last_pos = lens - 1
        S = int(lens.max())

        def prep(ci):
            s0, C = plan[ci]
            tk = tp[:, s0 : s0 + C]
            hs = self.hasher(tk, torch.full((B,), s0)) if self.host_rows is not None else None
            return pm.prep_inputs(tk, hs)

        res = {}
        ex = ThreadPoolExecutor(1)
        fut = ex.submit(prep, 0)
        for ci, (s0, C) in enumerate(plan):
            t0 = time.perf_counter()
            pre = fut.result()
            pm.timing["host_wait"] = pm.timing.get("host_wait", 0.0) + time.perf_counter() - t0
            if ci + 1 < len(plan):
                fut = ex.submit(prep, ci + 1)
            bufs = pm.alloc_inputs(C)
            pm.upload_inputs(pre, bufs)
            self.sink.update(s0, C)
            out = pm.forward_device_legacy(
                bufs,
                S,
                s0,
                C,
                hook=hook if len(plan) == 1 else None,
                profile=True,
                last_pos=last_pos,
                want_logits=want_logits,
            )
            res.update(out)
            if len(plan) > 1:
                ttnn.synchronize_device(self.md)
                self.log(
                    f"  prefill chunk {ci + 1}/{len(plan)} (s0={s0}, C={C}) done at {time.perf_counter() - t_start:.1f} s"
                )
                clear_chunk_caches()
        ex.shutdown()
        ttnn.synchronize_device(self.md)
        self._export_index_keys(lens)
        self.timing = dict(pm.timing, total=time.perf_counter() - t_start)
        first = torch.tensor([res[b][0] for b in range(B)], dtype=torch.long)
        logits = torch.stack([res[b][1] for b in range(B)]) if want_logits else None
        return first, logits

    # ---- decode -------------------------------------------------------------------------------------------------------------------------
    def prepare_inputs_decode(self, tokens, current_pos):
        """Engram rows of the step's input tokens at their per-user positions -> host tensor for the persistent rows buffer (None without Engram)."""
        if not self.engram_ids:
            return None
        hashes = self.hasher(tokens.reshape(self.B, 1).long(), current_pos.long())
        rows = self._rows_threads(hashes)
        cat = torch.cat([rows[l].reshape(self.B, 1, 1, -1) for l in self.engram_ids], dim=-1).to(torch.bfloat16)
        return ttnn.from_torch(
            cat,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.md, dims=(0, None), mesh_shape=(self.rows, self.cols)),
        )

    _rpool = None

    def _rows_threads(self, hashes):
        if Model._rpool is None:
            Model._rpool = ThreadPoolExecutor(max(1, len(self.engram_ids)))
        fs = {l: Model._rpool.submit(self.host_rows.rows, l, hashes) for l in self.engram_ids}
        return {l: f.result() for l, f in fs.items()}

    def _engram_rows_fn(self, tokens, pos):
        """In-trace: slice the persistent packed rows buffer into the per-layer rows the Engram layers read."""
        Tn = self.U
        rc = ttnn.reshape(self.rows_cat, [1, 1, Tn, self.rows_cat.shape[-1]])
        out, off = {}, 0
        for l in self.engram_ids:
            k = self.engram_kin[l]
            out[l] = ttnn.to_layout(ttnn.slice(rc, [0, 0, 0, off], [1, 1, Tn, off + k]), ttnn.TILE_LAYOUT)
            off += k
        return out

    def _mp(self):
        return ttnn.ShardTensor2dMesh(self.md, dims=(0, None), mesh_shape=(self.rows, self.cols))

    def _set_loop_state(self, tokens, current_pos):
        tok = ttnn.from_torch(
            tokens.reshape(-1, 1).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._mp(),
        )
        pos = ttnn.from_torch(
            current_pos.reshape(-1).to(torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._mp(),
        )
        if getattr(self.dec, "tok_dev", None) is None:
            self.dec.enable_device_loop(self.mc, self.ccl, tokens.reshape(-1), current_pos.reshape(-1))
            self.dec.engram_rows_fn = self._engram_rows_fn if self.engram_ids else None
        else:
            ttnn.copy_host_to_device_tensor(tok, self.dec.tok_dev)
            ttnn.copy_host_to_device_tensor(pos, self.dec.pos_dev)

    def _upload_rows(self, host_rows):
        if host_rows is None:
            return
        if self.rows_cat is None:
            self.rows_cat = ttnn.to_device(host_rows, self.md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            ttnn.copy_host_to_device_tensor(host_rows, self.rows_cat)

    def _read_tokens(self):
        devs = ttnn.get_device_tensors(ttnn.from_device(self.dec.tok_dev))
        return torch.cat([ttnn.to_torch(devs[r * self.cols]).reshape(-1) for r in range(self.rows)]).long()

    def decode_forward(self, tokens, current_pos, enable_trace=True, reload_inputs=True):
        """One decode step of every user. tokens [B] = the token fed at position current_pos [B] (per user). Returns the next greedy tokens [B].
        ``reload_inputs=False`` (steady state): the device already holds the fed-back token / position (device loop); only the Engram rows of
        ``tokens`` are uploaded."""
        t0 = time.perf_counter()
        self.pool.ensure(
            current_pos, lookahead=16
        )  # pages for the next replays (no-op, no upload, unless a page boundary is near)
        if reload_inputs or getattr(self.dec, "tok_dev", None) is None:
            self._set_loop_state(tokens, current_pos)
        host_rows = self.prepare_inputs_decode(tokens, current_pos)
        t1 = time.perf_counter()
        self._upload_rows(host_rows)
        if enable_trace and self.trace_id is None:
            self._capture_decode(tokens, current_pos)
            self._set_loop_state(tokens, current_pos)
            self._upload_rows(host_rows)
        t2 = time.perf_counter()
        if enable_trace:
            ttnn.execute_trace(self.md, self.trace_id, cq_id=0, blocking=False)
        else:
            self.last_logits = self.dec.forward()
        out = self._read_tokens()  # blocking read: waits for the step
        t3 = time.perf_counter()
        self.timing["decode_host_prep"] = t1 - t0
        self.timing["decode_upload"] = t2 - t1
        self.timing["decode_device_read"] = t3 - t2
        return out

    def _capture_decode(self, tokens, current_pos):
        """Compile pass (restores the step-carried compressor state) + trace capture of one device-loop step."""
        snaps = self.dec.snapshot_states()
        self.dec.forward()
        ttnn.synchronize_device(self.md)
        self.dec.restore_states(snaps)
        self._set_loop_state(tokens, current_pos)
        self.trace_id = ttnn.begin_trace_capture(self.md, cq_id=0)
        self.last_logits = self.dec.forward()
        ttnn.end_trace_capture(self.md, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.md)
        self.dec.restore_states(snaps)

    def read_logits(self):
        """Host copy [B, vocab] fp32 of the logits of the last decode step (diagnostics; the loop itself only reads tokens)."""
        ttnn.synchronize_device(self.md)
        return self.head.gather_logits(self.last_logits)[: self.B]

    def release_trace(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.md, self.trace_id)
            self.trace_id = None

    # ---- one model build, many batch sizes / context lengths ---------------------------------------------------------------------------
    def _l1_alloc(self):
        ttnn.synchronize_device(self.md)
        return ttnn.get_memory_view(self.md, ttnn.BufferType.L1).total_bytes_allocated_per_bank

    def dram_snapshot(self):
        """(allocated, free, largest free block) in MiB per DRAM bank, after a device synchronise."""
        ttnn.synchronize_device(self.md)
        mv = ttnn.get_memory_view(self.md, ttnn.BufferType.DRAM)
        l1 = ttnn.get_memory_view(self.md, ttnn.BufferType.L1)
        self.l1_last = (l1.total_bytes_allocated_per_bank, l1.largest_contiguous_bytes_free_per_bank)
        mib = 2**20
        return (
            mv.total_bytes_allocated_per_bank / mib,
            mv.total_bytes_free_per_bank / mib,
            mv.largest_contiguous_bytes_free_per_bank / mib,
        )

    def _release_prefill_traces(self):
        pm = getattr(self, "prefill_model", None)
        if pm is None:
            return
        pm.teardown_dyn()  # chunk trace, fused / per-column head traces, per-chunk buffers
        if getattr(pm, "trace_id", None) is not None:  # legacy whole-prefill trace
            ttnn.release_trace(self.md, pm.trace_id)
            pm.trace_id = None
        from models.demos.blackhole.deepseek_v41_flash.tt import reconfigure as RC

        for t in list(getattr(pm, "_bufs", {}).values()):
            RC.free_tensors(t)
        pm._bufs = {}
        pm.dyn_out = None
        pm.head_out = []

    def _release_batch_state(self, generator_hooks=()):
        """Release EVERY piece of device state that depends on the users per row / max context, keeping the weights and the Engram host tables
        (``self._keep``): traces first (a trace owns the intermediate buffers of its capture and replays with fixed addresses), then the objects
        that own persistent tensors (KV pool, page tables, per-user buffers, prefill input buffers, MoE dispatch buffers + semaphores), the module level
        caches of device constants, and finally the program cache (programs own L1 semaphores / were specialised to the old shapes).
        """
        from models.demos.blackhole.deepseek_v41_flash.tt import reconfigure as RC

        for h in generator_hooks:
            h()
        self.release_trace()  # decode trace
        self._release_prefill_traces()  # (a separate method: no local keeps the old prefill model alive through the gc below)
        ttnn.synchronize_device(self.md)
        for name in (
            "chain",
            "pool",
            "sink",
            "sources",
            "attns",
            "built",
            "step_groups",
            "dec",
            "prefill_model",
            "prefill_sparse",
            "dec_idx",
            "index_owner",
            "host_rows",
            "hasher",
            "rows_cat",
            "last_logits",
            "uni_layers",
            "_hooks_set",
            "_nosink_done",
            "_warm",
            "_moe_warm",
            "_res",
            "_last_pos",
            "_want_logits",
            "engram_kin",
            "head",
            "mc",
            "ccl",
        ):
            self.__dict__.pop(name, None)
        self.trace_id, self.admitted, self.pool_pages_free = None, False, None
        # EVERY persistent L1 allocation goes (the CCL semaphores too, they are re-created first by the rebuild like in a fresh process): the L1 allocator is
        # first-fit, so gaps left by buffers of the old batch size would shift the new persistent buffers below the static circular-buffer region of the
        # big programs ('Statically allocated circular buffers ... clash with L1 buffers')
        for e in self._keep["dev_engram"].values():
            e.mesh_config = e.ccl = None
            e._w_rows = (
                {}
            )  # compact weight rows per token count T (built lazily, one per batch size seen): not weights, rebuilt on first use
        RC.reset_module_caches()
        for _ in range(3):
            gc.collect()
        ttnn.synchronize_device(self.md)
        if os.environ.get("DSV41_RECONFIG_DEBUG") == "1":
            RC.debug_l1_referrers(self.log, self.md)
        dbg = os.environ.get("DSV41_RECONFIG_DEBUG") == "1"
        if dbg:
            self.log(
                f"RC_DEBUG before clear_program_cache: {self.md.num_program_cache_entries()} entries, L1 alloc {self._l1_alloc()} B"
            )
        if os.environ.get("DSV41_RECONFIG_CLEAR_PCACHE", "1") == "1":
            self.md.clear_program_cache()  # programs of the old shapes, and the L1 semaphores they own
        ttnn.synchronize_device(self.md)
        if dbg:
            self.log(
                f"RC_DEBUG after clear_program_cache: {self.md.num_program_cache_entries()} entries, L1 alloc {self._l1_alloc()} B"
            )

    def reconfigure(self, args, max_ctx, num_pages=None, kv_dtype=None, generator_hooks=()):
        """Serve another batch size (``args.users_per_row``), max context or KV pool dtype with the SAME weights: releases all batch dependent device
        state (``_release_batch_state``) and rebuilds it (``_build`` with ``self._keep``: attention / shared-expert / mHC / norm weights are re-read
        from the checkpoint, the routed-expert weights (95 % of the device bytes), the ring-layout weights read in place by the unified prefill MoE, the
        embedding / head / Engram device weights and the Engram host tables are NOT touched). Returns a dict with the timings and the free DRAM per bank
        before / after the release / after the rebuild (MiB). The caller must drop its own references to the old pool / generator state
        (``generator_hooks``: callables run first, e.g. releasing the speculative runner)."""
        assert self._keep is not None, "reconfigure needs a completed first build"
        t0 = time.time()
        old = (self.B, self.U, self.max_ctx)
        before = self.dram_snapshot()
        l1_before = self.l1_last
        self.log_dram("reconfigure: before release")
        self._release_batch_state(generator_hooks)
        t1 = time.time()
        mid = self.dram_snapshot()
        l1_mid = self.l1_last
        self.log_dram("reconfigure: after release")
        self.timing = {}
        self._build(args, max_ctx, num_pages, kv_dtype if kv_dtype is not None else self.kv_dtype)
        t2 = time.time()
        after = self.dram_snapshot()
        info = dict(
            old=old,
            new=(self.B, self.U, self.max_ctx),
            release_s=t1 - t0,
            rebuild_s=t2 - t1,
            total_s=t2 - t0,
            dram_before=before,
            dram_released=mid,
            dram_after=after,
        )
        f = lambda t: f"alloc {t[0]:.1f} / free {t[1]:.1f} / largest block {t[2]:.1f}"
        l1_after = self.l1_last
        self.log(
            f"RECONFIGURE B {old[0]} (U={old[1]}, ctx {old[2]}) -> B {self.B} (U={self.U}, ctx {self.max_ctx}): release {t1 - t0:.1f} s, rebuild {t2 - t1:.1f} s, total {t2 - t0:.1f} s; "
            f"DRAM MiB/bank before [{f(before)}] | weights only [{f(mid)}] | after [{f(after)}]"
            f" | L1 B/bank allocated / largest free: before {l1_before[0]} / {l1_before[1]}, released {l1_mid[0]} / {l1_mid[1]} (fresh process at start: {self.l1_base}), after {l1_after[0]} / {l1_after[1]}"
        )
        return info


def DSV41StepState_paged(attn, max_pos, with_indexer=False):
    return DSV41PagedStepState(attn, max_pos=max_pos, with_indexer=with_indexer, per_user_valid=with_indexer)
