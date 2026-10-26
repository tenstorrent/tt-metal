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
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt import prefill_paged as PGM
from models.demos.blackhole.deepseek_v41_flash.tt import prefill_single as PSG
from models.demos.blackhole.deepseek_v41_flash.tt import uni_policy
from models.demos.blackhole.deepseek_v41_flash.tt.decoder import DSV41Decoder
from models.demos.blackhole.deepseek_v41_flash.tt.device_head import DSV41DeviceEmbedding, DSV41DeviceHead
from models.demos.blackhole.deepseek_v41_flash.tt.engram import DSV41DeviceEngram, HostEngramRows
from models.demos.blackhole.deepseek_v41_flash.tt.engram_ragged import RaggedNgramHash
from models.demos.blackhole.deepseek_v41_flash.tt.h2d import replay
from models.demos.blackhole.deepseek_v41_flash.tt.indexer import pos_buckets
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
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_taps import PrefillTaps, taps_possible

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


def _return_lag():
    """windows allowed in flight when a call with no final row returns (``DSV41_PF_RETURN_LAG``; ``DSV41_PF_RETURN_LAG_FILE`` = a file holding the value, read at every call: a live switch for A/B runs):
    -1 = do not return early (wait for the device like before), 0 = wait for the last window, 1 = the last window has started (default), 2 = no wait
    """
    f = os.environ.get("DSV41_PF_RETURN_LAG_FILE")
    if f:
        try:
            return int(open(f).read().strip())
        except Exception:
            pass
    return int(os.environ.get("DSV41_PF_RETURN_LAG", "1"))


_PREP_POOL = [None]


def _prep_pool():
    """thread pool of the window builds (``DSV41_PF_BUILD_AHEAD`` windows gathered concurrently)"""
    if _PREP_POOL[0] is None:
        _PREP_POOL[0] = ThreadPoolExecutor(max(2, int(os.environ.get("DSV41_PF_BUILD_AHEAD", "2")) + 1))
    return _PREP_POOL[0]


class _UpOnly:
    """marker: the recorded upload ops of the token half of a window (``Model._one`` lookahead); the position half is recorded when the window starts"""

    __slots__ = ("ops",)

    def __init__(self, ops):
        self.ops = ops


def window_needs_sync(users, C, async_win, apc_wants=None, vis=None, final=None):
    """Does the host have to wait for the device after the replay of a window of the interleaved prefill? ``users``: [(user, window start, end of its chunk)] of the users in the window.
    Yes when the window holds a user's last token of the call (the ragged head output is read), when a prefix snapshot the cache wants ends at the window (its device reads must precede the
    next replay), or when asynchronous windows are off; otherwise the window is only enqueued (``DSV41_PF_ASYNC_SERVER``).
    ``final`` (set of users, or None = unknown): the users whose sequence ENDS in this call; with it only their last-token windows are read (the token of any other row is discarded by the scheduler).
    """
    if not async_win:
        return True
    for b, s0, e in users:
        if e <= s0 + C and (final is None or b in final):
            return True
        if apc_wants is not None and s0 + C <= e and not (vis and b in vis) and apc_wants(s0 + C):
            return True
    return False


def key_export_fill_max(slab_rows):
    """Largest prefix [0, j0 + P) (rows) of the decode index-key slab that ``_export_user_keys`` writes in place (fill_cache); ``DSV41_KEY_EXPORT_FILL_MAX`` overrides. Default: the larger of 524,288 and
    the slab's rows, so every piece of every build is written in place (the slice_write fallback re-allocates the slab the captured decode traces read by address).
    """
    env = os.environ.get("DSV41_KEY_EXPORT_FILL_MAX")
    return int(env) if env not in (None, "") else max(524288, int(slab_rows))


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
        # users per mesh row of the PREFILL trace (= the decode's by default; fewer: interleaved prefill of a few users per row at a time, see INTERLEAVE_NOTES.md)
        self.Up = int(os.environ.get("DSV41_PREFILL_UP", str(self.U)))
        assert 1 <= self.Up <= self.U, f"DSV41_PREFILL_UP={self.Up} must be in 1..{self.U}"
        self.max_ctx = max_ctx
        from models.demos.blackhole.deepseek_v41_flash.tt import moe_overlap

        moe_overlap.configure(self.B, max_ctx)
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
            scratch_rows=PGM.scratch_rows(self.Up, int(os.environ.get("DSV41_PF_CMAX", "2048")))
            if PGM.MODE != "0"
            else 0,
        )
        self.sink = PagedStateSink(mesh_device, self.pool, self.Up, decode_users=self.U)
        PGM.bind(self.sink)
        # speculative builds (pool ring rows for the spec runners): the prefill chunk trace also writes the drafter's taps (layers 37-39 stream means of the last 128 positions of every
        # user, tt/prefill_taps.py), which seed the drafter after the prefill instead of replaying the prompt tail through the verify trace. ``tap_n[user]`` = prompt length the stash holds.
        self.tap_n = {}
        if self.pool.ring_rows >= 128 + 127 and taps_possible(self.layer_ids):
            self.sink.taps = PrefillTaps(mesh_device, self.U)
        log(
            f"prefill drafter taps {'ON' if self.sink.taps is not None else 'OFF'} (ring rows {self.pool.ring_rows}, layers {len(self.layer_ids)})"
        )
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
            for L_next in self.layer_ids[
                i + 1 : i + 3
            ]:  # the next two layers of the list (= L + 1, L + 2 for a contiguous list; layer subsets of tests need not be)
                submit(L_next)
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
            pa = DSV41PrefillAttention(attn, w["attn"]["attn_sink"].float(), users=self.Up)
            pa.state_sink = functools.partial(self.sink.write, attn)
            pa.lat_sink = functools.partial(self.sink.write_lat, attn)
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
                from models.demos.blackhole.deepseek_v41_flash.tt import moe_overlap

                if (
                    moe_overlap.prep_enabled()
                ):  # DSV41_MO_OVERLAP=1|prep: separate gate / up shared-expert weights for the sub-device matmuls
                    moe_overlap.SDOverlap.get(mesh_device).split_weights(layer.shared)
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
        self.dec.samp = self.samp = self._make_sampler(self.U, plain=True)
        self.prefill_model = GenPrefillModel(
            mesh_device, pls, embedding, self.head, dev_engram, self.host_rows, self.Up
        )
        self.prefill_model.set_head_sampling(self.mc, self.ccl)
        self.prefill_model.sink = self.sink
        self.engram_kin = {l: e.kin for l, e in dev_engram.items()}
        self.rows_cat = None
        self.trace_id = None
        self.dtr, self._dpos = {}, None  # decode traces per position bucket, host-tracked positions of the device loop
        self.samp_trace = None  # trace of the sampler (replayed after the step trace by a step with sampled rows)
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

    def mem_report(self, tag):
        """One log line with the free DRAM per bank and the trace-region occupancy (always on, a device sync + two view queries: used at the end of the serving warm-up)."""
        try:
            ttnn.synchronize_device(self.md)
            mv = ttnn.get_memory_view(self.md, ttnn.BufferType.DRAM)
            mib = 2**20
            s = (
                f"MEMREPORT {tag}: DRAM allocated {mv.total_bytes_allocated_per_bank / mib:.1f} MiB/bank, free {mv.total_bytes_free_per_bank / mib:.1f}, "
                f"largest free block {mv.largest_contiguous_bytes_free_per_bank / mib:.1f}"
            )
            try:
                tv = ttnn.get_memory_view(self.md, ttnn.BufferType.TRACE)
                s += (
                    f"; trace region allocated {tv.total_bytes_allocated_per_bank / mib:.1f} MiB/bank, free {tv.total_bytes_free_per_bank / mib:.1f}, "
                    f"largest free block {tv.largest_contiguous_bytes_free_per_bank / mib:.1f}"
                )
            except Exception as e:  # noqa: BLE001
                s += f"; trace view unavailable ({type(e).__name__})"
            self.log(s)
        except Exception as e:  # noqa: BLE001
            self.log(f"MEMREPORT {tag}: failed ({type(e).__name__}: {str(e)[:100]})")

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
            idx._slab_owner = self.index_owner[
                meta["kv_source"]
            ]  # (decode buckets share one prefix view of the slab per step: tt/decode_buckets.py)
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
            pas, idx_w, self.Up, max_tokens or self.max_ctx, c_max, sinks, decode_indexers=self.dec_idx, enable=True
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
            self.Up != self.U
        ):  # fewer prefill slots than decode users: the batch is prefilled slot group by slot group (prefill_interleaved)
            B = self.B
            lens = torch.as_tensor(prompt_lens).long()
            users = [b for b in range(B) if int(lens[b]) > 0 and (active is None or bool(active[b]))]
            res = self.prefill_interleaved(
                [(b, tokens[b], 0, int(lens[b])) for b in users],
                chunk or 1024,
                s_pad_max=s_pad_max,
                want_logits=want_logits,
            )
            first = torch.full((B,), -1, dtype=torch.long)
            logits = torch.zeros(B, self.args.vocab_size) if want_logits else None
            for b, (tk, lg) in res.items():
                first[b] = int(tk)
                if want_logits:
                    logits[b] = lg
            return first, logits
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
        for (
            P
        ) in (
            self._dec_bucket_list()
        ):  # (position buckets: every scored length is compiled, its constants created, before the first trace)
            self._dec_set_bucket(P)
            self.last_logits = (
                self.dec.forward()
            )  # compile pass (writes pool rows at position = prompt length: rewritten by the prefill / first real step)
            ttnn.synchronize_device(self.md)
            self.dec.restore_states(snaps)
        del snaps
        if self.samp is not None:  # compile pass of the sampler (it writes the token buffer: set again below)
            self.dec.forward_sampler(self.last_logits, False)
            self.dec.forward_sampler(self.last_logits, True)
            ttnn.synchronize_device(self.md)
        self._set_loop_state(zeros, torch.as_tensor(lens).long())
        self._warm = True

    def single_spad(self, C):
        """DSV41_PF_SINGLE=1: the padded context of THE chunk trace = the build's maximum context rounded up to the chunk (every prompt of the build fits)."""
        return -(-(self.max_ctx - 1) // C) * C

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

    def _post_todo(self, u, s0, C):
        """(mesh row, offset in the 32-token tile, decode user) of the users whose last prompt token is inside the window starting at ``s0`` (per slot ``u`` of the traced chunk)"""
        rows, U = self.rows, self.Up
        sb = getattr(self, "_slot_b", None)  # prefill slot (r * Up + u) -> global decode user (None: identity)
        s0s = PSG.slot_s0(s0, rows * U)  # (per-row chunk starts: one start per prefill slot)
        return [
            (r, (int(self._last_pos[r * U + u]) - s0s[r * U + u]) % 32, r * U + u if sb is None else sb[r * U + u])
            for r in range(rows)
            if s0s[r * U + u] <= int(self._last_pos[r * U + u]) < s0s[r * U + u] + C
        ]

    def post_chunk_prefetch(self, s0, C):
        """Enqueue the NON-blocking reads of the ragged-head outputs (tokens, and the vocab shards of the logits of the users that need them) of the window just enqueued: the command queue is in order, so
        they run right behind its trace and the host copies are ready when ``_post_chunk`` runs after the synchronisation (the blocking reads cost about 40 ms per call with the device idle).
        """
        pm, cols = self.prefill_model, self.cols
        pf = self.__dict__.setdefault("_post_pf", {})
        pf.clear()
        for u in range(self.Up):
            todo = self._post_todo(u, s0, C)
            if not todo:
                continue
            lg, tk = pm.head_out[u]
            ent = {"tk": ttnn.from_device(tk, blocking=False)}
            if self._want_logits:
                lgd = ttnn.get_device_tensors(lg)
                ent["lg"] = {
                    (r, c): ttnn.from_device(lgd[r * cols + c], blocking=False) for r, _, _ in todo for c in range(cols)
                }
            pf[u] = ent

    def _post_chunk(self, s0, C):
        """Host-only (no device allocation): read the ragged-head trace outputs of this chunk for the users whose last prompt token is inside it."""
        pm, rows, cols = self.prefill_model, self.rows, self.cols
        pf = self.__dict__.get("_post_pf") or {}
        for u in range(self.Up):
            lg, tk = pm.head_out[u]
            todo = self._post_todo(u, s0, C)
            if not todo:
                continue
            ent = pf.pop(u, None)
            t_p0 = time.perf_counter()
            devs = ttnn.get_device_tensors(ent["tk"] if ent is not None else ttnn.from_device(tk))
            pm.timing["post_fd"] = pm.timing.get("post_fd", 0.0) + time.perf_counter() - t_p0
            lgd = ttnn.get_device_tensors(lg) if self._want_logits else None
            for r, off, b in todo:
                tok = int(ttnn.to_torch(devs[r * cols]).reshape(-1)[off])
                row = None
                if (
                    lgd is not None
                ):  # only the vocab shards of the mesh row of this user (8 devices x [32, vocab / 8]), not the logits of every row / token
                    row = torch.cat(
                        [
                            ttnn.to_torch(ent["lg"][(r, c)] if ent is not None and "lg" in ent else lgd[r * cols + c])
                            .reshape(32, -1)[off]
                            .float()
                            for c in range(cols)
                        ]
                    )
                self._res[b] = (tok, row)
            pm.timing["post_tt"] = pm.timing.get("post_tt", 0.0) + time.perf_counter() - t_p0

    def prefill_forward_dyn(
        self, tokens, prompt_lens, chunk, max_new_tokens, want_logits, enable_trace, s_pad_max, active=None
    ):
        B = self.B
        assert tokens.shape[0] == B
        assert (
            self.Up == self.U
        ), "the whole-batch prefill needs DSV41_PREFILL_UP == users per row (use prefill_interleaved)"
        self._slot_b = None
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
        if PSG.enabled():
            s_pad_max = self.single_spad(C)
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
        self.flush_key_exports(why="non-interleaved prefill")  # (its replays overwrite the carried key FIFOs)
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
        for b in range(B):  # the drafter taps of the last 128 positions of every active user are in the stash
            if active is None or bool(active[b]):
                self.tap_n[b] = int(lens[b])
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

    # ---- interleaved (chunked, per-user) prefill for continuous batching (tt/generator_vllm.py; INTERLEAVE_NOTES.md) ----------------------------------
    def _admit_idle(self):
        """Every user owns at least one page: the decode loop grows the pages of ALL users (``pool.ensure``), including the idle ones."""
        for b in range(self.B):
            r, k = self.pool.user_key(b)
            if k not in self.pool.allocs[r].pages:
                self.pool.admit(b, 1)

    def release_user(self, b):
        """A finished / preempted request: free its pages, keep the user admitted with one page (see ``_admit_idle``)."""
        self.pool.release(b)
        self.pool.admit(b, 1)
        self.pool.sync_page_table()
        getattr(self, "pf_resume", {}).pop(b, None)
        self.drop_key_exports([b])

    def _export_state(self, NA, UD):  # NA: entries per piece
        """Persistent device tensors of the key hand-off for slab length ``NA`` / ``UD`` users per row (allocated before any trace capture by ``warm_key_export``; the values are rewritten
        per call): gather indices [1, NA], entry mask [1,1,NA,1], (mesh row, user) mask [rows * UD,1,1,1]."""
        st = self.__dict__.setdefault("_exp_state", {})
        if (NA, UD) not in st:
            rep = ttnn.ReplicateTensorToMesh(self.md)
            st[(NA, UD)] = dict(
                idx=ttnn.from_torch(
                    torch.zeros(1, NA, dtype=torch.int32),
                    device=self.md,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=rep,
                ),
                emask=ttnn.from_torch(
                    torch.zeros(1, 1, NA, 1),
                    device=self.md,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=rep,
                ),
                umask=ttnn.from_torch(
                    torch.zeros(self.rows * UD, 1, 1, 1),
                    device=self.md,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=self._mp(),
                ),
            )
        return st[(NA, UD)]

    def _export_user_keys(self, ix, k_cache, i_src, u_dst, rows_sel, off, n, warm_limit=None):
        """Hand-off of the index keys of the users held by prefill slot ``i_src`` of the mesh rows ``rows_sel`` (decode user ``u_dst`` of each of those rows) only: slab
        slots [off, off + n) of the prefill key FIFO -> decode key slab entries [0, n). The other mesh rows / users / entries keep their decode keys (bfp8 -> bf16 -> bfp8 is exact).
        The decode key slab ``k_cache`` is referenced by the CAPTURED decode traces, so it is updated IN PLACE (``ttnn.copy`` of the rebuilt slab; ``slice_write`` re-allocates).
        Every op has a SHAPE that depends only on (slot, slab) and every per-call value (gather indices, entry / user masks) lives in a persistent tensor: no new program, no new device
        buffer after the traces are captured (the first version sliced at value dependent offsets: each new prompt length created programs whose buffers were allocated under live traces,
        found by TT_METAL_TRACE_ALLOC_TRACKING; ``warm_key_export`` compiles the few programs before the captures).
        """
        sh = ix.sh
        IDIM, NA, UD = ix.keys.shape[3], k_cache.shape[2], k_cache.shape[0]
        P = min(NA, int(os.environ.get("DSV41_KEY_EXPORT_PIECE", "65536")))
        st = self._export_state(P, UD)
        rep = ttnn.ReplicateTensorToMesh(self.md)
        um = torch.zeros(self.rows, UD, 1, 1, 1)
        for r in rows_sel:
            um[r, u_dst] = 1.0
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                um.reshape(self.rows * UD, 1, 1, 1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=self._mp(),
            ),
            st["umask"],
        )
        # Source blocks of the key FIFO (a whole-FIFO gather / whole-slab rebuild needs several 256 MiB temporaries per chip at 1M entries): block b = FIFO slots [base, base + SB).
        # Column-sharded FIFO: slot cols * q + c - dL sits on column c at row q; a block is the gather of Lq rows (fixed shape, static row offset q0). Replicated FIFO: one block.
        if sh is None:
            KL = ix.keys.shape[2]
            blocks = [(0, KL, None)]
        else:
            cols, Lc = sh["cols"], sh["Lc"]
            dL = cols * Lc - ix.KL_full
            KL = ix.KL_full
            Lq = min(Lc, P // cols)
            blocks = []
            for q0 in range(0, Lc, Lq):
                q0 = min(q0, Lc - Lq)
                blocks.append((cols * q0 - dL, cols * Lq, q0))
            blocks = list(dict.fromkeys(blocks))
        warm = n == 0 and off == 0 and not rows_sel
        # DEFAULT ON (DSV41_KEY_EXPORT_FILL=0 selects the slice_write export). The slice_write export re-allocates the decode key slab, which the captured decode traces reference by address: with
        # prefill slots < users per row (DSV41_PREFILL_UP=1, the serving default) the decode steps then score stale / misplaced keys (B=16 / 262144: wrong greedy answers, logits 20-31 instead
        # of 40-60; bisected in evals_gpqa_tau3/b16_isolation: FILL=1 alone fixes the 4,934-token probe in every slot). 1M builds already ran the fill export (generator_vllm).
        KEY_FILL = os.environ.get("DSV41_KEY_EXPORT_FILL", "1") == "1"
        FILL_MAX = key_export_fill_max(
            NA
        )  # (rows of the prefix written in one fill_cache: beyond it the slice_write path, which RE-ALLOCATES the slab the decode traces read. Default max(524288, slab rows): the whole slab is written in place whatever the build (a ratio-1 slab of a 262,144-token build has 262,176 rows, of a 1M build 1,048,640: the old fixed 524,288 sent the pieces of prompts beyond ~524k tokens to slice_write); DSV41_KEY_EXPORT_FILL_MAX overrides)
        NP = -(-NA // P)
        for base, SB, q0 in blocks:
            hi = min(base + SB, KL)
            pieces = []
            for j0 in dict.fromkeys(
                min(pj * P, NA - P) for pj in range(NP)
            ):  # the last piece is shifted back (its overlap is served twice with equal values)
                j1 = j0 + P
                a_, b_ = max(off + j0, base, 0), min(off + min(j1, n), hi)  # source slots served by this (block, piece)
                if (warm and (warm_limit is None or j0 < warm_limit)) or a_ < b_:
                    pieces.append((j0, j1, a_, b_))
            if not pieces:
                continue
            if sh is None:
                fifo = ttnn.slice(ix.keys, [i_src, 0, 0, 0], [i_src + 1, 1, KL, IDIM])
                table = ttnn.to_layout(
                    fifo, ttnn.ROW_MAJOR_LAYOUT
                )  # a full-extent slice may alias ix.keys: never freed
                tmp_blk = [table, fifo]
            else:
                k = ttnn.to_layout(
                    ttnn.slice(ix.keys, [i_src, 0, q0, 0], [i_src + 1, 1, q0 + Lq, IDIM]), ttnn.ROW_MAJOR_LAYOUT
                )
                g = sh["mc"].allgather(
                    k, sh["ccl"], axis=1, dim=1
                )  # [1, cols, Lq, 128]: g[:, c, q] = padded slot cols * (q0 + q) + c
                g = ttnn.permute(g, (0, 2, 1, 3))
                table = ttnn.reshape(g, [1, 1, SB, IDIM])
                tmp_blk = [k, g, table]
            table = ttnn.reshape(table, [1, 1, table.shape[2], IDIM])
            for j0, j1, a_, b_ in pieces:
                idx = torch.zeros(1, P, dtype=torch.int32)
                em = torch.zeros(1, 1, P, 1)
                if a_ < b_:
                    js = torch.arange(a_ - off, b_ - off) - j0
                    idx[0, js] = torch.arange(a_ - base, b_ - base, dtype=torch.int32)
                    em[0, 0, js, 0] = 1.0
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep), st["idx"]
                )
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(em, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep), st["emask"]
                )
                emb = ttnn.embedding(st["idx"], table, layout=ttnn.TILE_LAYOUT)  # [1, P, IDIM]
                emb = ttnn.reshape(emb, [1, 1, P, IDIM])
                emb_full = ttnn.repeat(emb, ttnn.Shape([UD, 1, 1, 1]))
                cond = ttnn.multiply(st["umask"], st["emask"])  # [UD, 1, P, 1]
                old = ttnn.slice(k_cache, [0, 0, j0, 0], [UD, 1, j0 + P, IDIM])
                old_bf = old if k_cache.dtype == ttnn.bfloat16 else ttnn.typecast(old, ttnn.bfloat16)
                new = ttnn.where(cond, emb_full, old_bf)
                out = new if k_cache.dtype == ttnn.bfloat16 else ttnn.typecast(new, k_cache.dtype)
                if KEY_FILL and j0 + P <= FILL_MAX:
                    # in-place write of the prefix [0, j0 + P) of every user with fill_cache (tile native): slice_write converts the WHOLE slab to row-major first (a 256 MiB temporary at a
                    # 1M context: 32 MiB/bank contiguous, and the cost of the export). The rows [0, j0) are the slab's own (read back, written unchanged).
                    full = (
                        out
                        if j0 == 0
                        else ttnn.concat([ttnn.slice(k_cache, [0, 0, 0, 0], [UD, 1, j0, IDIM]), out], dim=2)
                    )
                    for u_ in range(UD):
                        inp = full if UD == 1 else ttnn.slice(full, [u_, 0, 0, 0], [u_ + 1, 1, j0 + P, IDIM])
                        ttnn.fill_cache(k_cache, inp, u_)
                        if inp is not full:
                            ttnn.deallocate(inp)
                    self.log_dram(f"key export in place j0={j0} P={P} (temporaries live)")  # (DSV41_MEMLOG)
                    if full is not out:
                        ttnn.deallocate(full)
                else:
                    ttnn.experimental.slice_write(out, k_cache, [0, 0, j0, 0], [UD, 1, j0 + P, IDIM], [1, 1, 1, 1])
                seen = {ix.keys.buffer_unique_id(), k_cache.buffer_unique_id()}
                for t in (emb, emb_full, cond, old, old_bf, new, out):
                    if t.is_allocated() and t.buffer_unique_id() not in seen:
                        seen.add(t.buffer_unique_id())
                        ttnn.deallocate(t)
            seen = {ix.keys.buffer_unique_id(), k_cache.buffer_unique_id()}
            for t in tmp_blk:
                if t.is_allocated() and t.buffer_unique_id() not in seen:
                    seen.add(t.buffer_unique_id())
                    ttnn.deallocate(t)

    def selftest_key_export(self):
        """Device self-test of ``_export_user_keys`` on the idle model: random content in the prefill key FIFO of the first key-owner layer, an export of FIFO[off, off + n) into user ``u`` of
        mesh rows 1 and 3 only, and a check of the decode key slab on the host (exported entries equal up to bfp8 rounding, every other row / user / entry unchanged). Returns the report string.
        """
        if not self.use_indexer or not getattr(self, "prefill_sparse", None):
            return "no decode indexer: nothing to check"
        for L, dec in self.dec_idx.items():
            if L not in self.index_owner:
                continue
            sp = self.attns[L].prefill.sparse
            ix = None if sp is None else sp.indexer
            if ix is not None and ix.key_owner is None and ix.keys is not None:
                break
        else:
            return "no key owner"
        if ix.sh is not None:
            return "column-sharded prefill indexer: self-test skipped"
        rows, cols = self.rows, self.cols
        Up, KL, IDIM = ix.keys.shape[0], ix.keys.shape[2], ix.keys.shape[3]
        k = dec.k_cache
        UD, NA = k.shape[0], k.shape[2]

        def slab():
            ttnn.synchronize_device(self.md)
            devs = ttnn.get_device_tensors(k)
            return torch.stack([ttnn.to_torch(devs[r * cols]).float().reshape(UD, NA, -1) for r in range(rows)])

        g = torch.Generator().manual_seed(7)
        fifo = torch.randn(rows * Up, 1, KL, IDIM, generator=g)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                fifo.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=self._mp()
            ),
            ix.keys,
        )
        before = slab()
        n, off, u, i_src, sel = min(NA, 96), 32, min(2, UD - 1), Up - 1, [1, 3]
        self._export_user_keys(ix, k, i_src, u, sel, off, n)
        after = slab()
        want = fifo.reshape(rows, Up, KL, IDIM)[:, i_src, off : off + n].float()

        def pcc(a, b):
            a, b = a.flatten(), b.flatten()
            a, b = a - a.mean(), b - b.mean()
            return float((a @ b) / (a.norm() * b.norm() + 1e-12))

        ok_rows = [pcc(after[r, u, :n], want[r]) for r in sel]
        unchanged = after.clone()
        changed = torch.zeros_like(unchanged, dtype=torch.bool)
        for r in sel:
            changed[r, u, :n] = True
        untouched = bool((unchanged[~changed] == before[~changed]).all())
        res = f"key export self-test (layer {L}, slot {i_src}, user {u}, rows {sel}, off {off}, n {n}): PCC of the exported entries {[round(x, 5) for x in ok_rows]}, everything else untouched: {untouched}"
        # restore: the slab and the FIFO of an idle model carry no state, but leave zeros
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.zeros_like(fifo).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=self._mp(),
            ),
            ix.keys,
        )
        return res

    def _prealloc_export_state(self):
        """Create the persistent state of the key export (``_export_state``) of every decode key slab without running an export."""
        if not self.use_indexer or not getattr(self, "prefill_sparse", None):
            return
        for L, dec in self.dec_idx.items():
            if L in self.index_owner:
                P = min(dec.k_cache.shape[2], int(os.environ.get("DSV41_KEY_EXPORT_PIECE", "65536")))
                self._export_state(P, dec.k_cache.shape[0])

    def warm_key_export(self, limit=None):
        """Allocate the persistent key-export tensors and compile the export programs of every key-owner layer for every prefill slot (a no-change export: empty user mask), before any
        trace is captured."""
        if not self.use_indexer or not getattr(self, "prefill_sparse", None):
            return
        for L, dec in self.dec_idx.items():
            if L not in self.index_owner:
                continue
            sp = self.attns[L].prefill.sparse
            ix = None if sp is None else sp.indexer
            if ix is None or ix.key_owner is not None or ix.keys is None:
                continue
            for i_src in range(ix.keys.shape[0]):
                self._export_user_keys(ix, dec.k_cache, i_src, 0, [], 0, 0, warm_limit=limit)
        ttnn.synchronize_device(self.md)

    def _export_keys_users(self, users, ends, s_end, slot_of):
        """Index-key hand-off (prefill key FIFOs -> decode key slabs) of the users whose prompt ended inside the window that ended at ``s_end``;
        ``slot_of(b)`` = prefill slot (index within its mesh row) of user b."""
        if not self.use_indexer or not getattr(self, "prefill_sparse", None):
            return
        from models.demos.blackhole.deepseek_v41_flash.tt.prefill_sparse import ceil32

        by_u = {}
        for b in users:
            by_u.setdefault((slot_of(b), b % self.U), []).append(b // self.U)
        mx = max(ends[b] for b in users)
        for L, dec in self.dec_idx.items():
            if L not in self.index_owner:
                continue  # layers 24..36 alias layer 20's slab
            sp = self.attns[L].prefill.sparse
            ix = None if sp is None else sp.indexer
            if ix is None or ix.key_owner is not None or not getattr(sp, "dyn_on", False):
                continue
            r = self.attns[L].ratio
            off = (
                ix.KL_full if ix.sh is not None else ix.KL
            ) - s_end // r  # column-sharded FIFO: ix.KL is the per-column length
            if ix.sh is not None and ix.sh.get("single"):
                off = ix.sh["Cc"]  # single trace: entry j sits at slot Cc + j
            n = min(ceil32(mx // r), s_end // r, dec.k_cache.shape[2])
            for (i_src, u_dst), rws in by_u.items():
                self._export_user_keys(ix, dec.k_cache, i_src, u_dst, sorted(rws), off, n)
        ttnn.synchronize_device(self.md)

    # ---- lazy key hand-off (DSV41_PF_LAZY_EXPORT=1, default): the server does not know which chunk step is the LAST of a prompt (the plugin sends the END of the scheduled chunk, never the prompt length),
    # so a step that ends a user's chunk only RECORDS the export (``_kx_pending[b] = (window end, prompt end of the call, prefill slot in the mesh row)``) and the export of the whole key prefix runs
    # once, when the keys are needed: before the first decode / verify step (``flush_key_exports()`` at the top of ``decode_forward`` / ``decode_forward_bucket`` and of the adapter's decode and
    # drafter entry points), before the carried state of the user's slot is overwritten (``_slot_alloc`` eviction, a trace re-capture, an unmasked replay, a non-interleaved prefill). A new prompt of the
    # user, ``release_user`` and a state release drop the record (the keys of an unfinished prompt are never read). Without it every chunk step re-exported the whole prefix (0.09 s early, 0.2 s at 64k,
    # 0.58 s late in a 1M prompt) although only the last export is ever read.
    def _lazy_key_export(self):
        return os.environ.get("DSV41_PF_LAZY_EXPORT", "1") == "1"

    def flush_key_exports(self, users=None, why=""):
        """Run the recorded key hand-offs (of ``users``; default every user) now. -> seconds (0.0 when nothing was pending)."""
        pend = self.__dict__.get("_kx_pending")
        if not pend:
            return 0.0
        sel = list(pend) if users is None else [b for b in users if b in pend]
        if not sel:
            return 0.0
        t0 = time.perf_counter()
        by_end = {}
        for b in sel:
            by_end.setdefault(pend[b][0], []).append(b)
        ends = {b: pend[b][1] for b in sel}
        i_src = {b: pend[b][2] for b in sel}
        for b in sel:
            del pend[b]
        for s_end in sorted(by_end):
            self._export_keys_users(by_end[s_end], ends, s_end, lambda b: i_src[b])
        dt = time.perf_counter() - t0
        st = self.__dict__.setdefault("kx_stats", {"flushes": 0, "users": 0, "seconds": 0.0})
        st["flushes"] += 1
        st["users"] += len(sel)
        st["seconds"] += dt
        self.log(f"  key export flush ({why or 'on demand'}): users {sorted(sel)}, {dt * 1e3:.0f} ms")
        return dt

    def drop_key_exports(self, users=None):
        """Forget the recorded key hand-offs of ``users`` (default all): the state they refer to is gone or superseded."""
        pend = self.__dict__.get("_kx_pending")
        if pend:
            for b in list(pend) if users is None else users:
                pend.pop(b, None)

    # prefill slots (DSV41_PREFILL_UP < users per row): the traced chunk holds Up users per mesh row; a user that is being prefilled owns one slot of its row (its carried state
    # lives there) until it is finished or evicted by another user
    def _slot_alloc(self, wave, resume):
        """assign a prefill slot (global index r * Up + i) to every user of ``wave`` [(b, tokens, start, end)]; users that continue keep theirs. Returns {b: slot}."""
        Up, U = self.Up, self.U
        owner = self.__dict__.setdefault("pf_slot_owner", [None] * (self.rows * Up))
        of = self.__dict__.setdefault("pf_slot_of", {})
        out, taken = {}, set()
        for b, _, st, _ in wave:
            if st > 0:
                assert of.get(b) is not None and owner[of[b]] == b, f"user {b} has no prefill slot to continue"
                out[b] = of[b]
                taken.add(of[b])
        for b, _, st, _ in wave:
            if st > 0:
                continue
            r = b // U
            cand = [r * Up + i for i in range(Up) if r * Up + i not in taken]
            assert cand, "more users of one mesh row than prefill slots in one wave"
            if of.get(b) in cand:
                sl = of[b]
            else:  # a slot whose owner is finished / not resumable first, else evict (its owner has to recompute from position 0)
                free = [c for c in cand if owner[c] is None or owner[c] not in resume]
                sl = free[0] if free else cand[0]
            o = owner[sl]
            if o is not None and o != b:
                if self.__dict__.get("_kx_pending"):
                    self.flush_key_exports([o], "slot eviction")  # (its carried key FIFO is about to be overwritten)
                resume.pop(o, None)
                of.pop(o, None)
            if b in of and of[b] != sl:
                owner[of[b]] = None
            owner[sl], of[b], out[b] = b, sl, sl
            taken.add(sl)
        return out

    def enable_apc(self, budget_bytes, policy="pow2"):
        """Automatic prefix caching of the interleaved prefill (tt/prefix_cache.py): host snapshots of the carried prefill state at chunk boundaries, restored into any user."""
        from models.demos.blackhole.deepseek_v41_flash.tt.prefix_cache import PrefixCache

        if (
            PGM.read_on() and self.pool.dtype != ttnn.bfloat16
        ):  # read_pool_latents (prefix_cache.py) needs a bf16 pool: refuse here instead of asserting at the first snapshot
            self.log(
                "prefix caching OFF: paged prefill read mode with a non-bf16 KV pool (DSV41_PF_PAGED_FP8=1); APC needs the bf16 pool"
            )
            self.apc = None
            return None
        self.apc = PrefixCache(self, int(os.environ.get("DSV41_VLLM_CHUNK", "512")), budget_bytes, policy, log=self.log)
        self.apc.set_signature((tuple(self.layer_ids), self.pool.ring_rows, str(self.kv_dtype), self.Up, PGM.MODE))
        return self.apc

    def reset_prefix_cache(self):
        """vLLM's reset_prefix_cache (the plugin forwards it) / a weight reload: drop every snapshot."""
        if getattr(self, "apc", None) is not None:
            self.apc.clear()
            self.log("prefix cache reset")

    def _apc_restore_wave(self, wave, apc, vis, zone=None):
        """The users of ``wave`` that start at 0 (and are no image prompts) take the deepest snapshot of their token prefix (users that take the SAME snapshot are restored together);
        -> the wave with the new start positions."""
        found = {}
        for b, t, st, e in wave:
            if st == 0 and not (vis and b in vis):
                snap = apc.lookup(b, t, e, zone=None if zone is None else zone.get(b))
                if snap is not None:
                    found.setdefault(id(snap), (snap, []))[1].append(b)
        start = {}
        for snap, bs in found.values():
            apc.restore_group(snap, bs)
            for b in bs:
                start[b] = snap.s
                e_b = next(e for b2, _, _, e in wave if b2 == b)
                self.pool.grow(b, e_b + 1)
        if start:
            self.pool.sync_page_table()
        return [(b, t, start.get(b, st), e) for b, t, st, e in wave]

    def _pf_pick_bucket(self, e, C, buckets, first):
        """S_pad context for a prompt whose first call ends at ``e``: the smallest context >= e when this call certainly holds the WHOLE prompt (``e`` is not a multiple of the chunk: the plugin's
        scheduler ends every non-final piece of a split prompt on a chunk multiple), else the largest (a continuation must stay in the context of its first call).
        """
        need = -(-e // C) * C
        if first and e % C == 0:
            return buckets[-1]
        for b in buckets:
            if b >= need:
                return b
        return buckets[-1]

    def prefill_interleaved(self, items, chunk, s_pad_max=None, want_logits=False, vis=None, final=None, totals=None):
        """See ``_prefill_interleaved_one``. With several captured S_pad contexts (``warm_serving(extra_spads=...)``) the users of the call are split by the context their carried state lives in
        (``pf_bucket_of``): a new prompt takes the smallest context that holds it when its first call is certainly the whole prompt, otherwise the largest; a continuation stays in its context, a
        prompt that outgrows it is recomputed from position 0 in a larger one (always correct). Each group is one ``_prefill_interleaved_one`` call on its context.
        """
        pm = self.prefill_model
        buckets = pm.dyn_buckets() if os.environ.get("DSV41_PF_SMALL_BUCKET", "1") != "0" else []
        if len(buckets) <= 1 or self.Up != self.U:
            return self._prefill_interleaved_one(items, chunk, s_pad_max, want_logits, vis, final, totals)
        C = int(chunk)
        of = self.__dict__.setdefault("pf_bucket_of", {})
        resume = self.__dict__.setdefault("pf_resume", {})
        groups = {}
        for b, t, st, e in items:
            b, st, e = int(b), int(st), int(e)
            cb = of.get(b)
            if st > 0 and cb is not None and resume.get(b) == st and e <= cb:
                k = cb
            else:
                if (
                    st > 0
                ):  # the carried state is gone or lives in a context that is too small: recompute from position 0 in a context chosen like a new prompt's
                    if resume.get(b) == st:
                        self.log(
                            f"prefill_interleaved: user {b} outgrew its S_pad {cb} context at {e}: recomputing from 0"
                        )
                    resume.pop(b, None)
                    st = 0
                k = self._pf_pick_bucket(e, C, buckets, True)
            groups.setdefault(k, []).append((b, t, st, e))
        res, tms = {}, []
        self.flush_key_exports(
            why="S_pad contexts"
        )  # (the key FIFOs of the contexts are separate: every group exports eagerly)
        self._kx_eager = True
        try:
            for k in sorted(groups, reverse=True):  # (continuations of the large context first: they are the long ones)
                grp = groups[k]
                pm.use_dyn(k)
                if getattr(self, "apc", None) is not None:
                    self.apc.store.tag = k
                vg = None if not vis else {b: v for b, v in vis.items() if b in {g[0] for g in grp}}
                res.update(self._prefill_interleaved_one(grp, C, k, want_logits, vg or None, final, totals))
                for g in grp:
                    of[g[0]] = k
                tms.append((k, [g[0] for g in grp], dict(self.timing)))
        finally:
            self._kx_eager = False
        tm = {}
        for _, _, d in tms:
            for key, v in d.items():
                if isinstance(v, (int, float)):
                    tm[key] = tm.get(key, 0.0) + v
        tm["spad_groups"] = ";".join(f"{k}:{us}" for k, us, _ in tms)
        self.timing = tm
        return res

    def _prefill_interleaved_one(
        self, items, chunk, s_pad_max=None, want_logits=False, vis=None, final=None, totals=None
    ):
        """Prefill of SOME users (new requests / prompt chunks) through the captured chunk trace while the other users keep their decode state.

        ``items``: [(user b, tokens 1-D [>= end], start, end)]: positions [start, end) of user b are computed. ``start`` must be 0 or the position the previous call for
        that user ended at (``self.pf_resume[b]``: only valid while the user's carried state is intact, i.e. the previous end was a multiple of ``chunk``).
        The users are processed in windows of ``chunk`` positions (one trace replay per window; the traced chunk holds ``Up`` = ``DSV41_PREFILL_UP`` prefill slots per mesh row
        (default: all U users, slot = user); the slots that are not part of the window are masked: no write of the pool / ring / compressor state / carried prefill state
        (``DSV41_PF_UMASK=1``) / Engram history). With Up < U a user takes a free slot of its mesh row (users of the same row beyond Up are processed in further waves).
        A user whose ``end`` is not a multiple of the chunk has its last window padded (its carried state is then no longer resumable: it is the end of its prompt).
        ``vis`` (DSV41_VISION=1): {user: ``VisionSeq``} for the users whose prompt has images: their image positions take the vision encoder's rows as embeddings, route with
        bias_vl and have no Engram contribution / n-gram (tt/vision_state.py).
        Returns {user: (greedy token of position end - 1, logits [vocab] fp32 or None)}.
        """
        C = int(chunk)
        if vis and not self.prefill_model.vis:
            raise RuntimeError("a prompt has images but the model was built without DSV41_VISION=1")
        assert C % 128 == 0, "chunk must be a multiple of 128"
        B, pm, Up, U, rows = self.B, self.prefill_model, self.Up, self.U, self.rows
        general = Up != U
        resume = self.__dict__.setdefault("pf_resume", {})
        items = [(int(b), torch.as_tensor(t).long(), int(st), int(e)) for b, t, st, e in items]
        ends = {b: e for b, _, _, e in items}
        assert len(ends) == len(items), "duplicate users in one prefill call"
        for k, (b, t, st, e) in enumerate(items):
            assert 0 < e <= self.max_ctx - 1 and 0 <= st < e and t.numel() >= e, (b, st, e, t.numel())
            if st > 0 and (resume.get(b) != st or self.__dict__.get("pf_slot_of", {}).get(b) is None and general):
                # the carried prefill state of this user is gone (its prefill slot was taken by another prompt of its mesh row, or the trace was re-captured): recompute from position 0
                # (always correct, never fast; ``tokens`` hold the prompt from position 0)
                self.log(
                    f"prefill_interleaved: user {b} cannot resume at {st} (carried position {resume.get(b)}): recomputing from 0"
                )
                items[k] = (b, t, 0, e)
        max_end = max(ends.values())
        self.check_context_supported(max_end)
        S_pad = max(s_pad_max or 0, -(-max_end // C) * C)
        if PSG.enabled():
            S_pad = self.single_spad(C)  # ONE chunk trace for every prompt length (tt/prefill_single.py)
        assert S_pad <= self.max_ctx + 128, (
            f"padded prompt length {S_pad} (chunk {C}) exceeds the RoPE tables of the model (max_ctx {self.max_ctx} + 128): build the model with "
            f"max_ctx >= {S_pad - 128} (a multiple of the chunk is the simplest)"
        )
        t_start = time.perf_counter()
        self._admit_idle()
        self.sink.set_lengths(torch.zeros(B, dtype=torch.long))  # (capture / compile pass: every user masked)
        self.sink.set_map([None] * (rows * Up) if general else None)
        self.sink.bind(C)
        self.prepare_for_traces(torch.zeros(B, dtype=torch.long))
        if getattr(self, "_hooks_set", None) is not pm:
            pm.pre_replay_hooks.append(lambda s0, C_: self.sink.update(s0, C_))
            pm.post_replay_hooks.append(self._post_chunk)
            self._hooks_set = pm
        pm.timing = {}
        pm._call_t0 = t_start
        if self.__dict__.get("_kx_pending") and (
            pm.dyn is None or pm.dyn.C != C or pm.dyn.S_pad != S_pad or pm.dyn_trace is None
        ):
            self.flush_key_exports(why="trace re-capture")  # (the key FIFOs of the finished prompts are re-allocated)
        if pm.capture_dyn(
            C, S_pad
        ):  # (re)captured: the carried state of every user is gone, the decode trace shares the freed DRAM
            resume.clear()
            self.__dict__.pop("_look", None)
            if getattr(self, "apc", None) is not None:
                self.apc.clear()  # (the FIFO geometry follows S_pad)
            self.__dict__.pop("pf_slot_of", None)
            self.__dict__.pop("pf_slot_owner", None)
            self.release_trace()
            self.log_dram("after capture_dyn")
        masked = getattr(pm.dyn, "umask", None) is not None
        lazy_kx = (
            masked and self._lazy_key_export() and not self.__dict__.get("_kx_eager", False)
        )  # (an unmasked trace overwrites the carried state of every user: no deferral)
        if not lazy_kx:
            self.flush_key_exports(why="eager mode")
        async_win = lazy_kx and os.environ.get("DSV41_PF_ASYNC_SERVER", "1") == "1"
        for b, t, st, e in items:
            if st == 0:
                self.drop_key_exports([b])  # (a new prompt of the user: the keys of the previous one are never read)
                self.pool.release(b)
                self.pool.admit(b, e + 1)
            else:
                self.pool.grow(b, e + 1)
        self.pool.sync_page_table()
        if self.host_rows is not None:
            cap = self.hasher.st.cache.shape[1]
            assert -(-max_end // C) * C <= cap, f"Engram history capacity {cap} < padded prompt {-(-max_end // C) * C}"
        res, self._res, self._want_logits = {}, {}, want_logits
        n_replays, t_rep, t_host, t_exp, t_hash, t_apc = 0, 0.0, 0.0, 0.0, 0.0, 0.0
        apc = getattr(self, "apc", None)
        if apc is not None and not apc.active():
            apc = None
        apc_defer = (
            apc is not None and async_win and apc.defer()
        )  # (deferred snapshots need the asynchronous windows: their reads are ordered by the queue, not by a host wait)
        t_apc_ovl = [0.0]  # seconds of snapshot completions run while the device replays a window
        early_bk = os.environ.get("DSV41_PF_EARLY_BK", "1") == "1"
        post_pf = async_win and os.environ.get("DSV41_PF_POST_PREFETCH", "1") == "1"
        fin_users = (
            None if final is None else {int(b) for b, f in final.items() if f}
        )  # (None: the plugin does not say which rows end their sequence)
        defer_return = (
            async_win
            and fin_users is not None
            and not fin_users
            and os.environ.get("DSV41_PF_DEFER_RETURN", "1") == "1"
            and _return_lag() >= 0
        )  # (a call with no final row returns once its windows are enqueued)
        # bounded replay (DSV41_PF_REPLAY, tt/prefill_model.py): ``zone[b]`` = first chunk start of user b that runs layers 21-39 (the chunks of the last R tokens of its prompt); the total length of the
        # prompt is ``totals[b]`` (plugin kwarg prefill_total) or, for a row that ends its sequence in this call, its end; unknown -> 0 (every chunk runs every layer)
        rep = getattr(pm, "dyn_trace_b", None) is not None and masked
        zone = {}
        if rep:
            for b, _, _, e in items:
                tot = None if totals is None else totals.get(b)
                if tot is None and fin_users is not None and b in fin_users:
                    tot = e
                zone[b] = pm.zone_start(tot, C)
        lookahead = os.environ.get("DSV41_PF_LOOKAHEAD", "1") == "1"
        spec_enq = os.environ.get("DSV41_PF_SPEC_ENQ", "1") == "1"
        t_first = 0.0  # host seconds of window builds that were not overlapped with a replay
        t_pre = time.perf_counter() - t_start
        per_row = (
            bool(getattr(pm.dyn, "per_row", False)) and masked
        )  # per-row chunk starts (tt/prefill_single.py): users at different positions replay in ONE pass
        prebuild_next = (
            async_win and per_row and os.environ.get("DSV41_PF_PREBUILD", "1") == "1"
        )  # (position-only tables of the next call's first window, see DynCtx.prebuild)
        if per_row:
            groups = [
                sorted(items, key=lambda it: it[2] == 0)
            ]  # (continuations first: a new prompt of the same call must not evict a slot a continuation still needs)
        else:
            groups = [
                [it for it in items if it[2] == st0] for st0 in sorted({st for _, _, st, _ in items}, reverse=True)
            ]  # continuations FIRST: a new prompt of the same call must not evict a slot a continuation still needs
        for grp in groups:
            if general:  # waves: at most Up users of a mesh row per wave
                by_row = {}
                for it in grp:
                    by_row.setdefault(it[0] // U, []).append(it)
                waves = [
                    [it for r in sorted(by_row) for it in by_row[r][k * Up : (k + 1) * Up]]
                    for k in range(-(-max(len(v) for v in by_row.values()) // Up))
                ]
            else:
                waves = [grp]
            for wave in waves:
                if apc is not None and any(it[2] == 0 for it in wave):
                    t_apc += apc.finish_pending()  # (a lookup must see every snapshot taken so far)
                    new_ = self._apc_restore_wave(
                        [it for it in wave if it[2] == 0], apc, vis, zone if rep else None
                    )  # new prompts take the deepest stored snapshot (earlier waves of THIS call may have just stored one)
                    wave = [it for it in wave if it[2] != 0] + new_ if per_row else new_
                slot_of = self._slot_alloc(wave, resume) if general else {b: b for b, _, _, _ in wave}
                if (
                    rep
                ):  # first replay chunk of the rows of this wave (per row: every row its own; else one value for all rows)
                    if per_row:
                        pm.set_zone({slot_of[b] // Up: zone[b] for b, _, _, _ in wave})
                    else:
                        pm.set_zone({r_: min(zone[b] for b, _, _, _ in wave) for r_ in range(rows)})
                if per_row:
                    # window k of user it = positions [start + k * C, ...): every user advances at ITS OWN position in a replay
                    win_keys = list(range(max(-(-(e - st) // C) for _, _, st, e in wave)))
                    s0_of = lambda it, k: it[2] + k * C  # noqa: E731
                else:
                    win_keys = list(range(min(it[2] for it in wave), max(e for _, _, _, e in wave), C))
                    s0_of = lambda it, k: k  # noqa: E731

                def _build(k, wave=wave, aprep=False):
                    """host inputs of the window ``k`` (an absolute chunk start; per-row: the round) (tokens, Engram hashes, Engram rows gathered / dequantised): None when no user of the wave is active in it. Pure host work: DSV41_PF_OVERLAP=1 (default) runs it for the NEXT window while the device replays the current one."""
                    th_h = 0.0
                    part = [it for it in wave if (s0_of(it, k) < it[3] if per_row else it[2] <= k < it[3])]
                    if not part:
                        return None
                    active, lens = torch.zeros(rows * Up, dtype=torch.bool), torch.zeros(B, dtype=torch.long)
                    tok = self._filler_tokens(
                        rows * Up, C
                    ).clone()  # diverse filler (a constant token routes every filler row to the same experts: see _filler_tokens)
                    slot_b = [None] * (rows * Up)
                    s0v = [0] * (rows * Up)
                    for it in part:
                        b, t, st, e = it
                        sl = slot_of[b]
                        s0 = s0_of(it, k)
                        n = min(e - s0, C)
                        tok[sl, :n] = t[s0 : s0 + n]
                        tok[sl, n:] = t[e - 1]
                        active[sl], lens[b], slot_b[sl], s0v[sl] = True, e, b, s0
                    idx = active.nonzero().reshape(-1)
                    hs = None
                    vw = None  # (image mask [rows*Up, C], embedding override [rows*Up, C, dim]) of this window, None without images
                    if vis:
                        img, ovr = torch.zeros(rows * Up, C, dtype=torch.bool), None
                        for it in part:
                            b, t, st, e = it
                            vs = vis.get(b)
                            if vs is None:
                                continue
                            s0 = s0_of(it, k)
                            n = min(e - s0, C)
                            m_, emb_ = vs.window(s0, n)
                            if m_.any():
                                ovr = torch.zeros(rows * Up, C, vs.dim, dtype=torch.bfloat16) if ovr is None else ovr
                                img[slot_of[b], :n], ovr[slot_of[b], :n] = m_, emb_
                        vw = (img, ovr) if ovr is not None else None
                    look = self.__dict__.pop(
                        "_look", None
                    )  # (hashes / Engram rows of this window built during the previous call's last replay: ``_look_build``)
                    if look is not None and vw is None and look[0] == _look_key(part, k):
                        pm.timing["lookahead_hits"] = pm.timing.get("lookahead_hits", 0) + 1
                        in_place = look[4] is not None and look[4] == getattr(pm, "_win_count", None)
                        return (
                            part,
                            active,
                            lens,
                            tok,
                            slot_b,
                            vw,
                            look[1],
                            look[2],
                            0.0,
                            s0v,
                            _UpOnly(None if in_place else look[3]),
                        )
                    th0 = time.perf_counter()
                    if self.host_rows is not None:
                        hs = self.hasher(
                            tok[idx],
                            torch.tensor([s0v[i] for i in idx.tolist()], dtype=torch.long),
                            rows=torch.tensor([slot_b[i] for i in idx.tolist()]),
                            token_mask=None if vw is None else ~vw[0][idx],
                        )
                    th_h = time.perf_counter() - th0
                    if (
                        aprep and vw is None
                    ):  # the Engram row gather + dequant (the bulk of the host work of a window) runs in a pool thread: several windows ahead are gathered concurrently
                        prepped = _prep_pool().submit(pm.prep_inputs, tok, hs, active.nonzero().reshape(-1), None)
                    else:
                        prepped = pm.prep_inputs(tok, hs, active.nonzero().reshape(-1), vis=vw)
                    return part, active, lens, tok, slot_b, vw, hs, prepped, th_h, s0v

                def _bookkeep(part, k):
                    """host bookkeeping of a replayed window: the key hand-off record (or the eager export), ``resume``, prefix snapshot starts. Needs no result of the window: with deferred snapshots it
                    runs from ``_overlap`` (while the device still replays the window: the snapshot reads queue right behind its trace, and at the last window of a call the host work does not
                    wait for the device to go idle first)."""
                    nonlocal t_exp, t_apc
                    done = [(it[0], s0_of(it, k) + C) for it in part if it[3] <= s0_of(it, k) + C]
                    te0 = time.perf_counter()
                    if lazy_kx:
                        pend = self.__dict__.setdefault("_kx_pending", {})
                        for b, x in done:
                            pend[b] = (x, ends[b], slot_of[b] % Up if general else b % U)
                    else:
                        for s_end in sorted({x for _, x in done}):
                            self._export_keys_users(
                                [b for b, x in done if x == s_end],
                                ends,
                                s_end,
                                (lambda b: slot_of[b] % Up) if general else (lambda b: b % U),
                            )
                    t_exp += time.perf_counter() - te0
                    for it in part:
                        b, _, _, e = it
                        s_end = s0_of(it, k) + C
                        if e <= s_end:
                            resume.pop(b, None)
                            if e % C == 0:
                                resume[b] = e
                        else:
                            resume[b] = s_end
                    if (
                        apc is not None
                    ):  # prefix snapshots of the users that just crossed a boundary the policy wants (the state of a user is complete and resumable here)
                        for it in part:
                            b, t, st, e = it
                            if s0_of(it, k) + C <= e and not (vis and b in vis):
                                t_apc0 = time.perf_counter()
                                full_ = not rep or zone.get(b, 0) == 0
                                if apc_defer:
                                    apc.begin_snapshot(b, s0_of(it, k) + C, t, full=full_)
                                else:
                                    apc.maybe_snapshot(b, s0_of(it, k) + C, t, full=full_)
                                t_apc += time.perf_counter() - t_apc0

                bk_done = set()

                def _fp(t_, s0_, n_):
                    # fingerprint of the tokens a window hashes: the window and the 16 tokens before it (the n-gram context of the hash: max_ngram_size - 1 tokens, checked below): a different request that reuses the
                    # user / slot at the same position can never take another prompt's inputs
                    return hashlib.blake2b(t_[max(0, s0_ - 16) : s0_ + n_].numpy().tobytes(), digest_size=12).digest()

                def _look_key(part_, k_):
                    return (C,) + tuple(
                        sorted(
                            (
                                slot_of[it_[0]],
                                it_[0],
                                s0_of(it_, k_),
                                min(it_[3] - s0_of(it_, k_), C),
                                _fp(it_[1], s0_of(it_, k_), min(it_[3] - s0_of(it_, k_), C)),
                            )
                            for it_ in part_
                        )
                    )

                def _look_build():
                    """The NEXT call's first window, token part: when a user's chunk ended on a chunk boundary and the tokens passed for it go further (the plugin sends the whole prompt, or at least the tokens
                    of the longest user of the call), its next window is [e, e + n) of those tokens whatever the next call's end is. The Engram hashes + rows and their uploads are built NOW, during the last
                    replay of this call (the first window of a call has nothing to hide behind), and used by the next call's first window iff it asks for exactly the same (slot, user, start, length) set.
                    """
                    ent = [
                        it_
                        for it_ in wave
                        if it_[3] % C == 0 and it_[1].numel() > it_[3] and not (vis and it_[0] in vis)
                    ]
                    if not ent or self.host_rows is None or self.hasher.st.layout.max_ngram_size > 17:
                        return
                    t_l0 = time.perf_counter()
                    tok_l = self._filler_tokens(rows * Up, C).clone()
                    act_l, slot_l, s0_l, key_l = (
                        torch.zeros(rows * Up, dtype=torch.bool),
                        [None] * (rows * Up),
                        [0] * (rows * Up),
                        [],
                    )
                    for b_, t_, st_, e_ in ent:
                        sl_, n_ = slot_of[b_], min(t_.numel() - e_, C)
                        tok_l[sl_, :n_] = t_[e_ : e_ + n_]
                        tok_l[sl_, n_:] = t_[e_ + n_ - 1]
                        act_l[sl_], slot_l[sl_], s0_l[sl_] = True, b_, e_
                        key_l.append((sl_, b_, e_, n_, _fp(t_, e_, n_)))
                    idx_l = act_l.nonzero().reshape(-1)
                    hs_l = self.hasher(
                        tok_l[idx_l],
                        torch.tensor([s0_l[i] for i in idx_l.tolist()], dtype=torch.long),
                        rows=torch.tensor([slot_l[i] for i in idx_l.tolist()]),
                    )
                    prepped_l = pm.prep_inputs(tok_l, hs_l, idx_l, vis=None)
                    ops_l = pm.record_upload(prepped_l, C)
                    enq_l = None
                    if (
                        spec_enq
                    ):  # the copies go into the queue right behind the trace being replayed: the next call's first window finds the inputs in place
                        replay(ops_l)
                        enq_l = pm._win_count
                    self._look = ((C,) + tuple(sorted(key_l)), hs_l, prepped_l, ops_l, enq_l)
                    pm.timing["lookahead_build"] = pm.timing.get("lookahead_build", 0.0) + time.perf_counter() - t_l0

                ahead = {}
                launched = (
                    {}
                )  # windows whose hash is done and whose row gather runs in the pool (DSV41_PF_BUILD_AHEAD > 1), by window key
                build_ahead = int(os.environ.get("DSV41_PF_BUILD_AHEAD", "2"))

                def _resolve(nbw):
                    if nbw is not None and hasattr(nbw[7], "result"):
                        nbw = nbw[:7] + (nbw[7].result(),) + nbw[8:]
                    return nbw

                if (
                    pm.__dict__.get("build_prof") is not None
                ):  # (DSV41_PF_PROFILE_BUILD diagnostics: the host work of every window build is profiled)
                    _build_plain = _build

                    def _build(k, _b=_build_plain, _p=pm.build_prof):  # noqa: F811
                        _p.enable()
                        try:
                            return _b(k)
                        finally:
                            _p.disable()

                for k_s0, k in enumerate(win_keys):
                    if k in ahead:
                        bw = ahead.pop(k)
                    elif k in launched:
                        bw = _resolve(launched.pop(k))
                    else:  # a window nothing built during the previous replay (the first of a call): its host work is not hidden
                        t_b0 = time.perf_counter()
                        bw = _build(k)
                        t_first += time.perf_counter() - t_b0
                    if bw is None:
                        continue
                    part, active, lens, tok, slot_b, vw, hs, prepped, th_h, s0v = bw[:10]
                    rec_ops = bw[10] if len(bw) > 10 else None
                    if isinstance(
                        rec_ops, _UpOnly
                    ):  # (the token half of the uploads was recorded during the previous call: add the position half now)
                        self.sink.set_lengths(lens)
                        self.sink.set_map(slot_b if general else None)
                        rec_ops = (rec_ops.ops or []) + pm.record_begin(s0v if per_row else k, C, active)
                    s0 = s0v if per_row else k
                    t_hash += th_h
                    self.sink.set_lengths(lens)
                    self.sink.set_map(slot_b if general else None)
                    self._slot_b = slot_b if general else None
                    self._last_pos = (
                        torch.tensor([-1 if b is None else int(lens[b]) - 1 for b in slot_b]) if general else lens - 1
                    )

                    def _overlap(k_s0=k_s0, lens=lens, slot_b=slot_b, part=part, k=k):
                        if post_pf and sync_k:
                            self.post_chunk_prefetch(s0, C)  # (head read-backs queued right behind this window's trace)
                        if (
                            apc_defer
                        ):  # (the reads of a snapshot started after the previous window precede this window's replay: finish them while it runs)
                            t_apc_ovl[0] += apc.finish_pending()
                            if early_bk:
                                bk_done.add(k_s0)
                                _bookkeep(
                                    part, k
                                )  # (this window's own bookkeeping and snapshot starts: queued behind its trace)
                        if os.environ.get("DSV41_PF_OVERLAP", "1") != "1":
                            return
                        if k_s0 + 1 >= len(win_keys) and prebuild_next:
                            # last window of the call: nothing to build ahead. The NEXT call (the same users continuing, if their chunk ended on a chunk boundary) starts its first window
                            # with nothing to hide behind: build its position-only host tensors now (RoPE / mask tables) while this window replays
                            nxt = [0] * (rows * Up)
                            for it_ in wave:
                                if it_[3] % C == 0 and it_[3] < self.max_ctx - 1:
                                    nxt[slot_of[it_[0]]] = it_[3]
                            pm.dyn.prebuild(nxt)
                            if spec_enq:
                                pm.dyn.enqueue_prebuilt(pm._win_count)
                            if lookahead:
                                _look_build()
                        if build_ahead > 1:  # hash (serial, in window order) + start the gathers of the next windows
                            for kn in win_keys[k_s0 + 1 : k_s0 + 1 + build_ahead]:
                                if kn not in launched and kn not in ahead:
                                    launched[kn] = _build(kn, aprep=True)
                        for kn in win_keys[k_s0 + 1 :]:
                            nbw = _resolve(launched.pop(kn)) if kn in launched else _build(kn)
                            if nbw is None:
                                ahead[kn] = None
                                continue
                            # the whole host side of the next window's uploads (RoPE tables, masks, hand-off index tables, Engram rows) is built NOW, the device copies are recorded and replayed right before its trace
                            self.sink.set_lengths(nbw[2])
                            self.sink.set_map(nbw[4] if general else None)
                            ops = pm.record_window(nbw[7], nbw[9] if per_row else kn, C, nbw[1])
                            self.sink.set_lengths(lens)
                            self.sink.set_map(slot_b if general else None)
                            ahead[kn] = nbw + (ops,)
                            return

                    # a window whose result nothing reads is only enqueued (DSV41_PF_ASYNC_SERVER=1, default): the next window's copies + trace queue behind it, the host work of the window after
                    # that runs meanwhile. Synchronised: a window that holds the last token of a user of the call (head read-back), a snapshot boundary the prefix cache wants (its device reads must
                    # precede the next replay), an eager key export, an unmasked trace
                    sync_k = window_needs_sync(
                        [(it[0], s0_of(it, k), it[3]) for it in part],
                        C,
                        async_win,
                        None if apc is None or apc_defer else apc.wants,
                        vis,
                        fin_users,
                    )
                    th, td = pm.replay_window(
                        tok,
                        hs,
                        s0,
                        C,
                        active,
                        vis=vw,
                        prepped=prepped,
                        overlap=_overlap,
                        ops=rec_ops,
                        sync=sync_k,
                        back=not rep or any(s0_of(it, k) >= zone[it[0]] for it in part),
                    )
                    n_replays, t_host, t_rep = n_replays + 1, t_host + th, t_rep + td
                    t_ret = time.perf_counter()
                    if k_s0 not in bk_done:
                        _bookkeep(part, k)
                    if pm.__dict__.get("winlog") is not None and pm.winlog["windows"]:
                        pm.winlog["windows"][-1]["t_ret"] = t_ret  # (replay_window returned: post hooks included)
                        pm.winlog["windows"][-1][
                            "t_bk"
                        ] = time.perf_counter()  # (export record, resume, snapshot starts done)
                    if not masked:  # unmasked trace: every user outside this replay had its carried state overwritten
                        for b in [b for b in resume if b not in {p[0] for p in part}]:
                            resume.pop(b)
        t_loop_end = time.perf_counter()
        if apc is not None and not (
            apc_defer
            and os.environ.get("DSV41_PF_OVERLAP", "1") == "1"
            and os.environ.get("DSV41_APC_FINISH_NEXT", "1") == "1"
        ):
            # (snapshots started in the last windows of the call. With deferred snapshots they stay pending: the first replay of the NEXT call finishes them while the device works, or the
            # lookup of a new prompt / clear() does, whichever comes first; nothing reads the store in between)
            t_apc += apc.finish_pending()
        t_apc_end = time.perf_counter()
        wl_call = pm.__dict__.get("winlog")
        if wl_call is not None:
            pm.winlog_drain()
            wl_call["calls"].append(
                {
                    "t_start": t_start,
                    "t_pre_end": t_start + t_pre,
                    "t_end": time.perf_counter(),
                    "first_build_s": t_first,
                    "t_loop_end": t_loop_end,
                    "t_apc_end": t_apc_end,
                }
            )
        if defer_return:
            # (no user of the call ends its sequence: the scheduler discards every token of this step; the windows stay in flight and the NEXT call, a decode step or any read orders itself behind them.)
            # The step boundary the scheduler sees moves earlier by the windows still in flight: a request that arrives in that span misses the next step. DSV41_PF_RETURN_LAG = windows allowed to be
            # in flight at the return (default 1: the last window has started; 0 = wait for the device, 2 = no wait): it bounds that added admission latency to one window.
            lag = _return_lag()
            h_ = pm.__dict__.get("_ev_hist") or []
            if lag in (0, 1) and len(h_) >= lag + 1:
                ttnn.event_synchronize(h_[-(lag + 1)])
        elif pm.__dict__.get(
            "_ev_hist"
        ):  # (a trailing enqueued window: nothing reads its result, but the call must not return with the device still busy)
            ttnn.synchronize_device(self.md)
            pm._ev_hist.clear()
        for b, _, _, _ in items:
            res[b] = self._res[b] if (fin_users is None or b in fin_users) else self._res.get(b, (0, None))
            self.tap_n[b] = ends[b]
        self.timing = dict(
            pm.timing,
            total=time.perf_counter() - t_start,
            replays=n_replays,
            replay_s=t_rep,
            host_s=t_host,
            pre_s=t_pre,
            hash_s=t_hash,
            export_s=t_exp,
            apc_snapshot_s=t_apc,
            apc_overlapped_s=t_apc_ovl[0],
            first_build_s=t_first,
            dyn_pre_hits=getattr(pm.dyn, "pre_hits", 0),
            dyn_pre_misses=getattr(pm.dyn, "pre_misses", 0),
            dyn_prebuild_s=getattr(pm.dyn, "t_prebuild", 0.0),
        )
        return res

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

    def _make_sampler(self, users_per_row, plain=False, spec=False):
        """The in-trace sampler (tt/device_sampler.py, default; DSV41_INTRACE_SAMPLE=0 builds none): temperature / top-k / top-p draw of the next token over the step's logits, in its OWN trace that a decode
        step replays after the step trace only when a row is sampled (``decode_forward(samp=...)``: the per-row parameters, uploaded before the replay); greedy steps never touch it.
        """
        if os.environ.get("DSV41_INTRACE_SAMPLE", "1") != "1":
            return None
        if (plain and os.environ.get("DSV41_SHARED_SAMPLER", "0") == "1") or (
            spec and os.environ.get("DSV41_SHARED_SAMPLER_SPEC", "0") == "1"
        ):  # (the vLLM interface defaults DSV41_SHARED_SAMPLER to 1 for plain launches, generator_vllm; DSV41_SHARED_SAMPLER_SPEC is the speculative verify)
            # plain decode traces: the shared tt_transformers sampler (top-32 candidates, ~4x faster; tt/shared_sampler.py). The speculative verify keeps our own sampler.
            from models.demos.blackhole.deepseek_v41_flash.tt.shared_sampler import SharedSamplerAdapter

            smp = SharedSamplerAdapter(self.md, self.mc, self.ccl, users_per_row, exact_greedy=spec)
            smp.alloc_params()
            return smp
        from models.demos.blackhole.deepseek_v41_flash.tt.device_sampler import DSV41DeviceSampler

        smp = DSV41DeviceSampler(
            self.md,
            self.mc,
            self.ccl,
            users_per_row,
            J=int(os.environ.get("DSV41_SAMP_J", "16")),
            levels=int(os.environ.get("DSV41_SAMP_LEVELS", "3")),
            levels_k=int(os.environ.get("DSV41_SAMP_LEVELS_K", "4")),
        )
        smp.alloc_params()
        return smp

    def decode_forward(self, tokens, current_pos, enable_trace=True, reload_inputs=True, samp=None):
        """One decode step of every user. tokens [B] = the token fed at position current_pos [B] (per user). Returns the next greedy tokens [B].
        ``reload_inputs=False`` (steady state): the device already holds the fed-back token / position (device loop); only the Engram rows of
        ``tokens`` are uploaded."""
        self.flush_key_exports(why="decode")
        t0 = time.perf_counter()
        self.pool.ensure(
            current_pos, lookahead=16
        )  # pages for the next replays (no-op, no upload, unless a page boundary is near)
        if reload_inputs or getattr(self.dec, "tok_dev", None) is None:
            self._set_loop_state(tokens, current_pos)
            self._dpos = torch.as_tensor(current_pos).reshape(-1).long().clone()
        elif getattr(self, "_dpos", None) is None:
            self._dpos = torch.as_tensor(current_pos).reshape(-1).long().clone()
        host_rows = self.prepare_inputs_decode(tokens, current_pos)
        t1 = time.perf_counter()
        self._upload_rows(host_rows)
        if enable_trace and self.trace_id is None:
            self._capture_decode(tokens, current_pos)
            self._set_loop_state(tokens, current_pos)
            self._upload_rows(host_rows)
        if enable_trace:
            self._dec_bind(self._dec_bucket_for(self._dpos))  # the traces of the position bucket this step falls in
            self._dpos = self._dpos + 1  # (the device loop advances every position by one per replay)
        t2 = time.perf_counter()
        if samp is not None and self.samp is not None:
            self.samp.set_params(*samp)
        sampled = samp is not None and self.samp is not None
        if enable_trace:
            from models.demos.blackhole.deepseek_v41_flash.tt.decode_buckets import check_trace_allocations

            check_trace_allocations(self.md, self.trace_id, f"decode B={self.B}")
            ttnn.execute_trace(self.md, self.trace_id, cq_id=0, blocking=False)
            if sampled:  # the sampled tokens replace the greedy ones the step wrote (same buffer)
                self.samp_trace.replay()
        else:
            self.last_logits = self.dec.forward()
            if sampled:
                self.dec.forward_sampler(self.last_logits, self.samp.fast)
        out = self._read_tokens()  # blocking read: waits for the step
        t3 = time.perf_counter()
        self.timing["decode_host_prep"] = t1 - t0
        self.timing["decode_upload"] = t2 - t1
        self.timing["decode_device_read"] = t3 - t2
        return out

    # ---- position buckets of the plain decode step (tt/indexer.py ``set_pos_bucket``): one trace set per bucket ------------------------------------------------
    def _dec_indexers(self):
        seen, out = set(), []
        for idx in self.dec_idx.values():
            if id(idx) not in seen:
                seen.add(id(idx))
                out.append(idx)
        return out

    def _dec_bucket_list(self):
        """position buckets (ascending, DSV41_POS_BUCKETS) of the decode traces, then None = the whole key slab."""
        key = (self.max_ctx, os.environ.get("DSV41_POS_BUCKETS"))
        c = self.__dict__.get("_dbl")
        if c is None or c[0] != key:
            if not any(getattr(i, "backend", None) == "matmul" for i in self._dec_indexers()):
                c = (key, [None])
            else:
                c = (key, pos_buckets(self.max_ctx + 64) + [None])
            self._dbl = c
        return c[1]

    def _dec_set_bucket(self, P):
        for idx in self._dec_indexers():
            idx.set_pos_bucket(P)

    def _dec_bucket_for(self, pos):
        """smallest bucket that covers every position fed by this step (the entries written / read stay below it)."""
        mx = int(pos.max()) + 1
        return next((b for b in self._dec_bucket_list() if b is not None and mx <= b), None)

    def _dec_bind(self, P):
        t = self.dtr[P]
        self.trace_id, self.samp_trace, self.last_logits = t["trace_id"], t["samp_trace"], t["last_logits"]

    def _capture_decode(self, tokens, current_pos, compile_pass=True):
        """Compile pass (restores the step-carried compressor state) + trace capture of one device-loop step, for every position bucket. ``compile_pass=False``: everything was compiled before (warm_serving)."""
        from models.demos.blackhole.deepseek_v41_flash.tt.decode_buckets import corruptible
        from models.demos.blackhole.deepseek_v41_flash.tt.device_sampler import SamplerTraces

        snaps = self.dec.snapshot_states()
        Ps = self._dec_bucket_list()
        if compile_pass:
            for P in Ps:  # every bucket's shapes / constants before ANY trace exists
                self._dec_set_bucket(P)
                self.dec.forward()
                ttnn.synchronize_device(self.md)
                self.dec.restore_states(snaps)
            self._set_loop_state(tokens, current_pos)
        self.dtr = {}
        for P in Ps:
            self._dec_set_bucket(P)
            t = {"samp_trace": None}
            t["trace_id"] = ttnn.begin_trace_capture(self.md, cq_id=0)
            with corruptible(self.md):
                t["last_logits"] = self.dec.forward()
            ttnn.end_trace_capture(self.md, t["trace_id"], cq_id=0)
            ttnn.synchronize_device(self.md)
            self.dec.restore_states(snaps)
            if self.samp is not None:  # the sampler's own trace over the logits of the step trace
                t["samp_trace"] = SamplerTraces(self.samp)
                t["samp_trace"].capture(lambda fast, t=t: self.dec.forward_sampler(t["last_logits"], fast))
                ttnn.synchronize_device(self.md)
            self.dtr[P] = t
        self._dec_set_bucket(None)
        self._dec_bind(None)

    def read_logits(self):
        """Host copy [B, vocab] fp32 of the logits of the last decode step (diagnostics; the loop itself only reads tokens)."""
        ttnn.synchronize_device(self.md)
        return self.head.gather_logits(self.last_logits)[: self.B]

    def release_trace(self):
        for t in getattr(self, "dtr", {}).values():
            if t.get("trace_id") is not None:
                ttnn.release_trace(self.md, t["trace_id"])
            if t.get("samp_trace") is not None:
                t["samp_trace"].release()
        self.dtr = {}
        self.trace_id = None
        self.samp_trace = None
        for b in getattr(self, "buckets", {}).values():
            b.release_trace()

    # ---- serving warm-up and decode buckets (tt/decode_buckets.py, VLLM_BUCKETS_NOTES.md) -----------------------------------------------------
    def decode_filler(self):
        """Deterministic diverse token ids [B] fed to the users that hold no request in a decode step (a constant token routes every filler row to the same experts)."""
        f = self.__dict__.get("_dfill")
        if f is None:
            if os.environ.get("DSV41_VLLM_FILLER", "zero") == "diverse":
                g = torch.Generator().manual_seed(20260508)
                f = torch.randint(1000, int(self.args.vocab_size), (self.B,), generator=g)
            else:
                # token 0 (the old adapter's idle rows): identical idle rows route to the SAME few experts, so they add almost no expert-weight streaming to a decode step; diverse ids made the
                # idle rows of a bucket activate up to 6 new experts each (+9 ms of a 16-row step at 40 layers). The blow-up seen with a constant filler was the 16k-token PREFILL chunk.
                f = torch.zeros(self.B, dtype=torch.long)
            self._dfill = f
        return f

    def _hold_block(self, mib):
        """a DRAM tensor of ``mib`` MiB per bank (kept until freed: the hole it leaves is contiguous). The largest size <= mib that fits, down to 16 MiB."""
        while True:
            try:
                return ttnn.zeros(
                    [1, 1, mib * 8 * 1024, 512], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.md
                )
            except Exception:
                if mib <= 16:
                    raise
                self.log(f"guard hole of {mib} MiB/bank does not fit, trying {mib - 4}")
                mib -= 4

    def warm_serving(self, chunk, s_pad, bucket_users=(), extra_spads=()):
        """Everything a server needs before its first request, in the order that keeps device allocation out of any captured trace: (1) allocate every persistent buffer (prefill
        per-chunk buffers, the full decode loop buffers, every decode bucket), (2) compile every program (prefill chunk, full decode step, every bucket step), (3) capture the traces
        (prefill chunk, full decode, every bucket). A request afterwards only replays traces and copies into persistent tensors. ``bucket_users``: users per mesh row of the
        decode buckets below U. Returns the seconds each phase took."""
        t0 = time.perf_counter()
        B, rows, Up, U, pm = self.B, self.rows, self.Up, self.U, self.prefill_model
        general = Up != U
        zeros = torch.zeros(B, dtype=torch.long)
        filler = self.decode_filler()
        self._admit_idle()
        self.sink.set_lengths(zeros)
        self.sink.set_map([None] * (rows * Up) if general else None)
        self.sink.bind(chunk)
        self.prepare_for_traces(zeros)  # decode loop buffers of the full model + its compile pass
        if getattr(self, "_hooks_set", None) is not pm:
            pm.pre_replay_hooks.append(lambda s0, C_: self.sink.update(s0, C_))
            pm.post_replay_hooks.append(self._post_chunk)
            self._hooks_set = pm
        pm.timing = {}
        extra = sorted({int(x) for x in extra_spads if 0 < int(x) < int(s_pad)}) if Up == U else []
        if PSG.enabled():
            extra = []  # ONE chunk trace serves every prompt length: no small-S_pad context
        first = []
        if extra and os.environ.get("DSV41_PF_EXTRA_FIRST", "0") == "1":
            # ALLOCATION ORDER: the persistent tables of the small context are allocated BEFORE the max-context ones. Allocated afterwards they land in the holes the max-context compile left
            # (its eager temporaries are 16.8 MiB/bank blocks), the eager passes of the captures then find fewer than the 3 such blocks they need (1M, 4 users: OOM). First, they sit at the
            # bottom and every later block stays as contiguous as in a build without the small context.
            cur = pm.dyn
            if not (cur is not None and cur.C == chunk and cur.S_pad == int(s_pad) and pm.dyn_trace is not None):
                first = extra[:1]
        if first:
            self.log_dram(f"before the S_pad {first[0]} prefill context (allocated first)")
            pm.teardown_dyn()
            pm.compile_dyn(chunk, first[0])
            self.warm_key_export(limit=first[0])
            self.log_dram(f"after the S_pad {first[0]} prefill context")
            pm.compile_dyn_extra(chunk, int(s_pad))  # (the small context stays the active one)
            pm.use_dyn(int(s_pad))
            compiled = True
            self.warm_key_export()
        else:
            compiled = pm.compile_dyn(chunk, s_pad)
            self.warm_key_export()  # (the max-context export first: with the slice_write export its temporaries need a 32 MiB/bank hole that exists only now; the fill_cache export warms every prefix size)
        # smaller S_pad contexts (DSV41_PF_SMALL_BUCKET): a short prompt replays a chunk trace whose FIFOs / masks / index scores have S_pad / ratio entries instead of max_ctx / ratio.
        # Each owns its carried tables and trace; ALL of them are allocated + compiled here, before the first capture.
        extra_all = extra if compiled else []
        extra = [x for x in extra_all if x not in first]
        for sp_ in extra:
            self.log_dram(f"before the S_pad {sp_} prefill context")
            pm.compile_dyn_extra(chunk, sp_)
            pm.use_dyn(sp_)
            self.warm_key_export(
                limit=sp_
            )  # (a prompt of this context has <= sp_ entries per ratio: only the first prefix sizes ever run)
            pm.use_dyn(int(s_pad))
            self.log_dram(f"after the S_pad {sp_} prefill context")
        guard = None
        if extra:  # (allocated AFTER the max-context tables: DSV41_PF_EXTRA_FIRST=0)
            # the eager pass right before the max-context capture needs ~17 MiB/bank contiguous (the sharded indexer's scores): keep a hole for it until then (the small contexts' tables and
            # program buffers would otherwise split every such block)
            gm = int(os.environ.get("DSV41_PF_GUARD_MIB", "36"))
            if gm > 0:
                self.mem_report("before the guard hole of the max-context capture")
                guard = self._hold_block(gm)
        t1 = time.perf_counter()
        from models.demos.blackhole.deepseek_v41_flash.tt.decode_buckets import DecodeBucket

        self.buckets = {}
        feeds = {}
        for Ub in sorted(set(int(u) for u in bucket_users if 0 < int(u) < U)):
            b = DecodeBucket(self, Ub, self.log)
            phys = torch.tensor([(i // Ub) * U + i % Ub for i in range(rows * Ub)], dtype=torch.long)
            tok, pos = filler[phys].clone(), torch.zeros(rows * Ub, dtype=torch.long)
            b.prepare(tok, pos, phys)
            b.compile(tok, pos, phys)
            self.buckets[Ub] = b
            feeds[Ub] = (tok, pos)
            self.log_dram(f"decode bucket U'={Ub} compiled")
        self.buckets_x = {}
        for name in [
            x for x in os.environ.get("DSV41_BUCKET_ABLATE", "").split(",") if x
        ]:  # timing experiments (see DecodeBucket)
            Ub = min(self.buckets) if self.buckets else None
            if Ub is None:
                break
            b = DecodeBucket(self, Ub, self.log, ablate=tuple(name.split("+")))
            phys = torch.tensor([(i // Ub) * U + i % Ub for i in range(rows * Ub)], dtype=torch.long)
            tok, pos = filler[phys].clone(), torch.zeros(rows * Ub, dtype=torch.long)
            b.prepare(tok, pos, phys)
            b.compile(tok, pos, phys)
            self.buckets_x[name] = (b, tok, pos, phys)
        ttnn.synchronize_device(self.md)
        if compiled and os.environ.get("DSV41_WARM_APC", "1") == "1":
            self.warm_apc()  # (compile-before-capture: the snapshot / restore programs must not first compile while a trace is live)
        gc.collect()
        t2 = time.perf_counter()
        if compiled:  # (a (re)capture of the prefill trace: the carried prefill state of every user is gone)
            if guard is not None:
                ttnn.deallocate(guard)
                guard = None
            pm.capture_dyn_trace(chunk)
            self.mem_report(f"after the S_pad {s_pad} chunk trace")
            for sp_ in extra_all:
                pm.use_dyn(sp_)
                pm.capture_dyn_trace(chunk)
                self.mem_report(f"after the S_pad {sp_} chunk trace")
            if extra_all:
                pm.use_dyn(int(s_pad))
            self.__dict__.pop("pf_bucket_of", None)
            if extra and int(os.environ.get("DSV41_PF_GUARD_MIB", "36")) > 0:
                # the eager decode step right before the decode capture needs a ~17 MiB/bank contiguous block (the indexer's score mask): hold one until then
                self.mem_report("before the guard hole of the decode precapture")
                guard = self._hold_block(int(os.environ.get("DSV41_PF_GUARD2_MIB", "36")))
            self.__dict__.setdefault("pf_resume", {}).clear()
            self.__dict__.pop("pf_slot_of", None)
            self.__dict__.pop("pf_slot_owner", None)
        self.release_trace()
        for Ub, b in self.buckets.items():
            b.capture(*feeds[Ub])
        for name, (b, tok, pos, phys) in self.buckets_x.items():
            b.capture(tok, pos)
        if guard is not None:
            ttnn.deallocate(guard)
            guard = None
        if self.trace_id is None:
            # one more eager decode step right before the capture (the program of paged_update_cache of the decode indexer was missing from the program cache at a 1M context with the
            # spec runner built: "Cannot load new binaries during trace capture"); costs one step, state restored
            self._capture_decode(filler, zeros, compile_pass=os.environ.get("DSV41_DEC_PRECAPTURE_WARM", "1") == "1")
        self.log_dram("serving warm-up done (all traces captured)")
        self.mem_report("after the model warm-up (prefill chunk + decode traces captured)")
        t3 = time.perf_counter()
        self.timing["warm_serving"] = {"alloc_compile": t2 - t0, "prefill_compile": t1 - t0, "capture": t3 - t2}
        self.log(
            f"warm_serving: prefill chunk {chunk} / S_pad {s_pad}, decode buckets U' {sorted(self.buckets)} + {U}: compile {t2 - t0:.1f} s, capture {t3 - t2:.1f} s"
        )
        return self.timing["warm_serving"]

    def warm_apc(self, max_size=None):
        """Pre-compile the prefix cache's device ops BEFORE any trace is captured (``warm_serving``): a snapshot (ttnn.slice of the carried tables, ttnn.embedding pool gather, reads) and a restore
        (write_batch / paged_scatter_rows) of every size 512 .. ``max_size`` (powers of two of the chunk, the sizes the ``pow2`` policy takes) on an idle user. The shapes depend on the size only,
        so each first-seen size cost a kernel-cache compile (0.4 s each, 3-4 s over the first prompts) in the first requests. The state of the idle user is zeros and is released afterwards.
        -> seconds (0.0 without prefix caching)."""
        apc = getattr(self, "apc", None)
        if apc is None or not apc.active():
            return 0.0
        from models.demos.blackhole.deepseek_v41_flash.tt import prefix_cache as PC

        t0 = time.perf_counter()
        b, chunk = 0, apc.chunk
        top = min(int(max_size or os.environ.get("DSV41_WARM_APC_MAX", "32768")), self.max_ctx - 1)
        sizes = [chunk * 2**k for k in range(32) if chunk * 2**k <= top]
        resume = self.__dict__.setdefault("pf_resume", {})
        try:
            for s_ in sizes:
                self.release_user(b)
                self.pool.release(b)
                self.pool.admit(b, s_ + 1)
                self.pool.sync_page_table()
                if self.Up != self.U:
                    self._slot_alloc([(b, None, 0, s_)], resume)
                resume[b] = s_
                toks = torch.zeros(s_, dtype=torch.long)
                if apc.defer():
                    snap = PC.snapshot_finish(self, PC.snapshot_begin(self, b, s_, toks))
                else:
                    snap = PC.snapshot_user(self, b, s_, toks)
                PC.restore_users(self, snap, [b])
                del snap
        except PC.ApcUnsupported as e:
            self.log(f"APC warm-up skipped: {e}")
        finally:
            self.release_user(b)
            resume.pop(b, None)
            self.__dict__.pop("pf_slot_of", None)
            self.__dict__.pop("pf_slot_owner", None)
        ttnn.synchronize_device(self.md)
        dt = time.perf_counter() - t0
        self.log(f"warm_apc: snapshot + restore of {sizes} on an idle user: {dt:.1f} s")
        return dt

    def warm_requests(self, chunk):
        """One synthetic multi-window request through ``prefill_interleaved`` with every real path on (after every trace was captured: the replays need the chunk trace): two users of different mesh
        rows with DIFFERENT chunk starts in one call (per-row starts), the lazy key hand-off, prefix snapshots on the way (APC on), then a third user that takes a prefix hit (restore + continuation).
        Nothing of it may compile after the traces exist: the number of programs the device cache gained is logged (a non-zero count is a missing warm-up step). The model is left idle
        (users released, snapshot store emptied). -> (seconds, new program-cache entries)."""
        t0 = time.perf_counter()
        pm, B, U = self.prefill_model, self.B, self.U
        if B < 3 or pm.dyn is None or pm.dyn_trace is None:
            return 0.0, 0
        n0 = self.md.num_program_cache_entries()
        apc = getattr(self, "apc", None)
        if apc is not None and not apc.active():
            apc = None
        stats0 = dict(apc.stats) if apc is not None else None
        g = torch.Generator().manual_seed(7)
        L0, L1 = 9 * chunk, 8 * chunk
        t0_ = torch.randint(1000, 100000, (L0,), generator=g, dtype=torch.int64)
        t1_ = torch.randint(1000, 100000, (L1,), generator=g, dtype=torch.int64)
        users = [0, U, 2 * U] if self.rows >= 3 else [0, 1, 2]
        spad = pm.dyn.S_pad
        calls = [
            [(users[0], t0_, 0, 4 * chunk), (users[1], t1_, 0, 2 * chunk)],
            [(users[0], t0_, 4 * chunk, L0), (users[1], t1_, 2 * chunk, L1)],
        ]
        try:
            for items in calls:
                self.prefill_interleaved(items, chunk, s_pad_max=spad, want_logits=False)
            self.flush_key_exports(why="warm-up")
            if apc is not None:  # a new prompt with the same tokens as user 0: takes the deepest snapshot and continues
                self.prefill_interleaved([(users[2], t0_, 0, L0)], chunk, s_pad_max=spad, want_logits=False)
                self.flush_key_exports(why="warm-up")
        finally:
            if apc is not None:
                apc.finish_pending()
                apc.store.clear()
                apc.stats.update(stats0)
            for u in users:
                self.release_user(u)
                self.__dict__.get("pf_resume", {}).pop(u, None)
                self.tap_n.pop(u, None)
            self.__dict__.pop("pf_slot_of", None)
            self.__dict__.pop("pf_slot_owner", None)
            self.drop_key_exports()
        ttnn.synchronize_device(self.md)
        dn = self.md.num_program_cache_entries() - n0
        dt = time.perf_counter() - t0
        self.log(
            f"warm_requests: 2 users with different starts + a prefix hit: {dt:.1f} s, program cache {n0} -> {n0 + dn} entries"
            + (
                ""
                if dn == 0
                else f" ({dn} NEW program(s) compiled after the traces were captured: a warm-up step is missing)"
            )
        )
        return dt, dn

    def bench_buckets(self, n=40):
        """Device time per replay (ms, blocking replay + token read) of every decode bucket, the ablation variants (DSV41_BUCKET_ABLATE) and the full step, on idle filler inputs."""
        out = {}
        items = [(f"bucket B'={4 * u}", b, None) for u, b in sorted(self.buckets.items())]
        items += [(f"ablation {k} (B'={4 * v[0].U})", v[0], v) for k, v in self.buckets_x.items()]
        for name, b, v in items:
            tok = (
                v[1]
                if v
                else self.decode_filler()[torch.tensor([(i // b.U) * self.U + i % b.U for i in range(self.rows * b.U)])]
            )
            pos = torch.zeros(self.rows * b.U, dtype=torch.long)
            phys = torch.tensor([(i // b.U) * self.U + i % b.U for i in range(self.rows * b.U)])
            b.step(tok, pos, phys)
            t0 = time.perf_counter()
            for _ in range(n):
                b.step(tok, pos, phys, reload_inputs=False)
            out[name] = (time.perf_counter() - t0) / n * 1e3
        zeros = torch.zeros(self.B, dtype=torch.long)
        self.decode_forward(self.decode_filler(), zeros, enable_trace=True, reload_inputs=True)
        t0 = time.perf_counter()
        for _ in range(n):
            self.decode_forward(self.decode_filler(), zeros, enable_trace=True, reload_inputs=False)
        out[f"full B={self.B}"] = (time.perf_counter() - t0) / n * 1e3
        return out

    def decode_forward_bucket(self, Ub, tokens, current_pos, phys, enable_trace=True, samp=None):
        """One decode step at ``Ub`` users per mesh row (``Ub`` = U: the full model). tokens / current_pos [4 Ub] in bucket row order, ``phys`` [4 Ub] the model user of every row
        (``r * U + u``, u < Ub). -> next greedy tokens [4 Ub]."""
        self.flush_key_exports(why="decode bucket")
        if Ub == self.U:
            return self.decode_forward(tokens, current_pos, enable_trace=enable_trace, reload_inputs=True, samp=samp)
        return self.buckets[Ub].step(tokens, current_pos, phys, enable_trace=enable_trace, samp=samp)

    def read_logits_bucket(self, Ub):
        return self.read_logits() if Ub == self.U else self.buckets[Ub].read_logits()

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

        self.drop_key_exports()
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
            "buckets",
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
        self.samp_trace = None
        self.dtr, self._dpos = {}, None
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
