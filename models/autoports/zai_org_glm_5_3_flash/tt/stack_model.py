# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14b (PRODUCTION_PATH.md M5 / M5c): the persistent-stack model behind the Stage 6 generator contract
(doc/ttnn/06a-full-model-prep/contract.md, doc/ttnn/09-vllm/m6_serving_contract.md), selected in serving with
GLM53_BACKEND_MODULE=models.autoports.zai_org_glm_5_3_flash.tt.stack_model.

One decode step = ONE trace replay, no host work inside: MkEmbed (the device-resident tokens -> the stacked residual of
every 4-row chunk) -> per chunk, ONE stack program per run of same-kind layers (tt/mk_stack.py / tt/tp_stack.py: the
layer loop inside the kernels, per-layer weight / state bases, layer i+1's weights streamed while layer i's tail runs;
on a TP mesh the in-program fabric all-reduces) -> MkHead (final norm, lm_head vocab shard, device top-K, the chips'
candidates merged in the program, the positions advanced) -> the same token tensor the next step's MkEmbed reads.
`samples_in_decode_trace` = True (one execute_trace per step incl. sampling).

Layer kinds: dense_kda (TP mesh: Stage 14's TPDenseKDALayer shards; one chip: 7a's MkLayer), sparse_kda (one chip: 7a's
MkSparseLayer; TP8 / EP8: Stage 14, pending), sparse_dsa (Stage 7b, pending). `layers=` restricts the model to a prefix /
subset for development (the tokens are then not the model's). Sampling: greedy (the device top-1 with the lowest-id tie
break); temperature / top-k / top-p rows are refused until the device sampler lands. Prefill: the prompt runs through the
decode step one token at a time (exact KDA / MLA semantics; TTFT is not this module's target).
"""

from __future__ import annotations

import dataclasses
import os
import time as _time
from typing import Any, Optional, Sequence

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.config import DENSE_KDA, SPARSE_DSA, SPARSE_KDA
from models.autoports.zai_org_glm_5_3_flash.tt.generator import (
    BLOCK_SIZE,
    MODEL_PREFILL_ALIGN,
    PREFILL_CHUNK,
    DecodeIO,
    ModelBackend,
)
from models.autoports.zai_org_glm_5_3_flash.tt.rt_guard import checked

VOCAB = 154880
MOCK = os.environ.get("GLM53_MOCK") == "1"  # mock cluster: no trace capture (a "trace" replays the step eagerly)


def _mock_capture() -> bool:
    """GLM53_MOCK_CAPTURE=1 on the mock: each capture runs its step once for the allocations (tt/mock_dram.py)"""
    return MOCK and os.environ.get("GLM53_MOCK_CAPTURE", "0") == "1"


# the fused_experts gate_up permutations of all 42 sparse layers, prebuilt on the share (Stage 9's cache: byte-identical
# to the derived files, 107 GB): the load reads them instead of permuting 12,096 experts on the host
_PERM_CACHE = "/home/alex/mpi-shfs/llm-wksp/tt-cache/glm53-flash-fused-perm"
if os.path.isdir(_PERM_CACHE):
    os.environ.setdefault("GLM53_FUSED_EXPERTS_CACHE", _PERM_CACHE)
CHUNK = 4  # rows per stack launch (7a's megakernel serves t <= 4)
# the megakernel knobs per layer kind (read by the layer classes at construction and by the programs at build): the
# dense layers' deep qkv ring must not reach the sparse layers (their reader region holds the expert buffers too)
KNOB_KEYS = ("GLM53_MK_KB_QKV", "GLM53_MK_KB_O", "GLM53_MK_RING_KB_QKV", "GLM53_MK_RING_KB_O", "GLM53_MK_RING_KB",
             "GLM53_MK_OPT", "GLM53_MK_READERS", "GLM53_MKX_RING", "GLM53_MKX_MBUF", "GLM53_MKX_INFLIGHT",
             "GLM53_MK_DEFER_INV", "GLM53_MHC_POST_ROWWRITE", "GLM53_MK_W_DELAY_US", "GLM53_MHC_PRE_EARLY",
             "GLM53_MHC_HN_FIRST", "GLM53_TP_SHARED", "GLM53_MK_FOLD_Y2", "GLM53_REC_EARLY_SIGNAL", "GLM53_REC_MODE",
             "GLM53_MHC_MODE", "GLM53_MHC_ROWREAD", "GLM53_REC_V2_UG", "GLM53_REC_UGROUPS", "GLM53_MK_ROUTER_KSPLIT",
             "GLM53_KDA_RSPLIT",
             # merged MRs' knobs (Stage 14 boundary prefetch; 7a MR3): popped per kind like the rest (A/B via
             # GLM53_STACK_KNOB_SET, e.g. "s:GLM53_MK_SPARSE_DEFER_INV=1")
             "GLM53_MK_PRE_PREFETCH", "GLM53_STACK_P1_PREFETCH", "GLM53_MHC_FPU_COLLAPSE", "GLM53_MHC_PRE_SPLIT", "GLM53_MK_SPARSE_DEFER_INV", "GLM53_MHC_DIAG",
             # 7a MR4 + MR5 (per kind via GLM53_STACK_KNOB_SET, e.g. "s:GLM53_KDA_TILES_MIN=9"; popped per kind so a
             # sparse_kda value never reaches the dense layers' kda_pre / fidelity)
             "GLM53_MK_QKV_ALIAS", "GLM53_MK_QKV_R0", "GLM53_KDA_PACK", "GLM53_KDA_TILES_MIN", "GLM53_MK_DS_FIDELITY",
             "GLM53_MK_LR_FIDELITY",
             # 7a 2026-10-21: Stage 14's first-item state prefetch, -8.5 us per sparse_kda layer at B=1 on 7a's B=1 profile
             # (vtp8 L4, outputs identical, 30-s soak bitwise): per kind, e.g. "s:GLM53_KDA_PREFETCH=1" on the B=1 backend
             "GLM53_KDA_PREFETCH",
             # 14b int25: a narrow width's stacked s1 in DRAM (set by _narrow_knobs only)
             "GLM53_MK_S1_DRAM", "GLM53_MOE_LIVE", "GLM53_MKX_XSL_SEED")  # fmt: skip
# one chip (Stage 7a's measured-best serving knobs); keys: False = dense_kda, True = sparse_kda, "dsa" = sparse_dsa's
# MkMoE (Stage 7b's TPMla takes its own measured-best defaults)
_S7A_SPARSE = {"GLM53_MK_OPT": "O3", "GLM53_MK_READERS": "3", "GLM53_MKX_RING": "3", "GLM53_MKX_INFLIGHT": "2"}
KNOBS = {
    False: {"GLM53_MK_OPT": "O3", "GLM53_MK_READERS": "3", "GLM53_MK_RING_KB_QKV": "700"},
    True: _S7A_SPARSE,
    "dsa": _S7A_SPARSE,
}
# a TP mesh: Stage 14's M3 configurations (doc/ttnn/14-production-tp8-megakernel perf.jsonl): dense = the glue trims +
# weight-start delay + PRE_EARLY + HN_FIRST (TP8 chain-45 237.5 us / layer, tag tp8-hnfirst-cn-chain45); sparse_kda =
# TP_SHARED + 3 readers / bank + the trims (375.8 us, tp8-sparse-tps-chain45-cn-r3); sparse_dsa's TPMoE keeps 7a's
# sparse knobs (its 6-slot default expert ring does not fit under the step's L1 floor). GLM53_STACK_KNOBS=s7a keeps the
# one-chip profile on a mesh (A/B)
_TRIMS = {"GLM53_MHC_POST_ROWWRITE": "1", "GLM53_MK_W_DELAY_US": "10", "GLM53_MHC_PRE_EARLY": "1"}
# dense_kda + 7a's generic knobs (Stage 14 evidence ring.jsonl ring-vtp8-dense-knobs: vtp8 3-layer stack 220 vs 259
# us/layer in-program; TP2 vs Stage 3 L0 B=1 0.999951, L1 B=1 0.999995, L0 B=4 0.999953);
# kda_rec v2 asserts rec_spread 1 (the stack's TP layers are built with 1)
_GEN7A = {
    "GLM53_REC_MODE": "v2",
    "GLM53_MHC_MODE": "dz,pv2,cp",
    "GLM53_MHC_ROWREAD": "1",
    "GLM53_REC_EARLY_SIGNAL": "1",
}
KNOBS_TP = {
    False: {
        "GLM53_MK_READERS": "4",
        "GLM53_MK_KB_QKV": "4",
        "GLM53_MK_RING_KB_QKV": "450",
        "GLM53_MK_DEFER_INV": "1",
        "GLM53_MHC_HN_FIRST": "1",
        **_TRIMS,
        **_GEN7A,
    },
    True: {"GLM53_MK_READERS": "3", "GLM53_TP_SHARED": "1", **_TRIMS},
    "dsa": _S7A_SPARSE,
}


# Stage 7a's 283-us TP8 sparse_kda knob set (GLM53_STACK_KNOBS=s7a283, the served profile): stack-checked
# by review (tests/s14b notes: fold_y2's CB_Y2 now rewound per layer; DEFER_INV / ROWWRITE inert in sparse_kda stacks;
# pv2 / cp only at t > 4; MKX_XROWS not in this tree); B > 4 adds 7a's user groups
KNOBS_7A283 = {"GLM53_MK_READERS": "3", "GLM53_MK_KB_QKV": "4", "GLM53_MK_FOLD_Y2": "1", "GLM53_MK_DEFER_INV": "1",
               "GLM53_REC_EARLY_SIGNAL": "1", "GLM53_MK_W_DELAY_US": "10", "GLM53_MHC_POST_ROWWRITE": "1",
               "GLM53_MHC_PRE_EARLY": "1", "GLM53_REC_MODE": "v2", "GLM53_MHC_MODE": "dz,pv2,cp", "GLM53_TP_SHARED": "1",
               "GLM53_MHC_ROWREAD": "1"}  # fmt: skip


def _live_device_tensors() -> dict:
    import gc

    out = {}
    for o in gc.get_objects():
        try:
            if isinstance(o, ttnn.Tensor) and o.storage_type() == ttnn.StorageType.DEVICE and o.is_allocated():
                out[id(o)] = o
        except Exception:  # noqa: BLE001 (gc walk)
            continue
    return out


def l1_report(mesh, tag: str, root=None) -> list:
    """GLM53_L1_REPORT=1 (Stage 12, coordinator 2026-10-23T04:26Z; the n29h6 w4 CB clash with 'an L1 buffer live at
    capture time'): every allocated L1 block of the mesh (chip 0's allocator = every chip's, lock-step) and the live
    ttnn tensors whose buffer starts in it (shape, dtype, memory config), printed before a decode capture (and the
    mock's stand-in step). Host only: an allocator view + a gc walk. Returns the rows."""
    try:
        base = int(ttnn._ttnn.device.GetAllocatorBaseAddress(mesh, ttnn.BufferType.L1))
        mv = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
        blocks = sorted((base + int(b["address"]), int(b["size"])) for b in mv.block_table
                        if str(b.get("allocated", "")).lower() in ("yes", "true", "1"))  # fmt: skip
    except Exception as e:  # noqa: BLE001  (a diagnostic never stops a capture)
        print(f"L1_REPORT {tag}: no allocator view ({type(e).__name__}: {e})", flush=True)
        return []
    owners = {}
    found = []
    if root is not None:  # the model's object graph (tp_image's walk): owner paths
        from models.autoports.zai_org_glm_5_3_flash.tt.tp_image import device_tensors  # noqa: PLC0415

        found += device_tensors(root, "be")
    found += [("gc", t) for t in _live_device_tensors().values()]
    for path, t in found:
        try:
            mc = t.memory_config()
            if mc.buffer_type != ttnn.BufferType.L1:
                continue
            ws = owners.setdefault(int(t.buffer_address()), [])
            w = f"{path} {tuple(int(v) for v in t.shape)} {t.dtype} {mc.memory_layout}"
            if path != "gc" or not ws:
                ws.append(w)
        except Exception:  # noqa: BLE001
            continue
    rows = []
    for addr, size in blocks:
        who = [w for a, ws in owners.items() if addr <= a < addr + size for w in ws]
        rows.append(dict(address=addr, size=size, end=addr + size, tensors=who))
    lowest = min((r["address"] for r in rows), default=None)
    print(
        f"L1_REPORT {tag}: {len(rows)} allocated L1 blocks, {sum(r['size'] for r in rows)} B per bank, lowest at "
        f"{lowest}; unowned blocks = buffers held outside a ttnn tensor (e.g. a C++ object / global semaphore)",
        flush=True,
    )
    for r in rows:
        print(
            f"L1_REPORT {tag}:   [{r['address']}, {r['end']}) {r['size']} B  {r['tensors'] or 'NO LIVE TENSOR'}",
            flush=True,
        )
    return rows


def set_knobs(sparse, tp: int = 1, t: int = 1):
    """the knob env of a layer kind (False dense_kda, True sparse_kda, "dsa" sparse_dsa): set before a kind's layers are
    constructed AND before its programs are first built (both read the env)"""
    key = sparse if sparse == "dsa" else bool(sparse)
    for k in KNOB_KEYS:
        os.environ.pop(k, None)
    # default s7a283 (Stage 14 ring item: the sparse_kda 3-layer vtp8 stack 304 vs 361 us/layer in-program under "s14")
    mode = os.environ.get("GLM53_STACK_KNOBS", "s7a283")
    prof = KNOBS_TP if tp > 1 and mode in ("s14", "s7a283") else KNOBS
    os.environ.update(prof[key])
    if tp > 1 and mode == "s7a283" and key is True:
        os.environ.update(KNOBS_7A283)
        if t > 4:
            os.environ.update({"GLM53_REC_V2_UG": "1", "GLM53_REC_UGROUPS": "8"})
    if prof is KNOBS_TP and key is True and t > 1:
        # 4-row chunks: the sparse stack's 4-slot expert ring (14's B=1 default) puts the reader CBs 5 KB above the
        # step's L1 floor (0x1466c0 > 0x145400 in mock); 3 slots fit (7a's serving value)
        os.environ["GLM53_MKX_RING"] = "3"
    # GLM53_STACK_KNOB_SET="d:K=V;s:K=V;a:K=V" (dense_kda / sparse_kda / sparse_dsa; K=- drops K; ";" or ","
    # between items when no value holds a comma): overrides (A/B)
    spec = os.environ.get("GLM53_STACK_KNOB_SET", "")
    for item in filter(None, spec.split(";") if ";" in spec else spec.split(",")):
        kind, kv = item.split(":", 1)
        k, v = kv.split("=", 1)
        if {"d": False, "s": True, "a": "dsa"}[kind] == key:
            if v == "-":
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _narrow_knobs(sparse, tp: int, t: int):
    """set_knobs for a narrow width's layers: a t = 1 sparse_kda layer takes the 3-slot expert ring (as t = 2..4) --
    the 4-slot ring of the B = 1 backend puts its static CBs 16 KB above the native B = 32 backend's L1 floor (CB end
    1,336,768 > L1 buffer 1,320,576 in mock; narrow width 2 builds clean)"""
    set_knobs(sparse, tp, t)
    if sparse is True and t == 1 and tp > 1:
        os.environ["GLM53_MKX_RING"] = "3"
    # the stacked s1 [t, 1, 4, 4096] fp32 (P6 -> P7) of a narrow width in DRAM: its 20,480 B per bank (t = 4) were the
    # one L1 tensor a narrow step allocated below the native B = 32 programs' CB ends (TP8 mock, 7.5 KB under the
    # floor); the full width's s1 is flat [1, 1, 32, 16384], not a view target (tile reshapes copy). Its writer
    # (mhc_post/writer_p.cpp, whole pages + write barrier before the hand-off) and readers take TensorAccessor args.
    if os.environ.get("GLM53_MK_S1_DRAM_OFF") != "1":  # A/B
        os.environ["GLM53_MK_S1_DRAM"] = "1"


DRAM = ttnn.DRAM_MEMORY_CONFIG
KDIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels")


def serving_fabric_config() -> dict:
    """the ttnn.set_fabric_config keywords this model is validated with (Stage 14's in-program all-reduce: FABRIC_1D,
    the default router config -- the binding takes a FabricRouterConfig, not None; the adapter maps both keys)"""
    return {"config": ttnn.FabricConfig.FABRIC_1D, "router_config": ttnn._ttnn.fabric.FabricRouterConfig()}


def build_model_backend(model_dir, mesh_device, **kw) -> "StackBackend":
    """GLM53_BACKEND_MODULE entry point (m6_serving_contract.md §1); unknown keys of the Stage 9 adapter are ignored."""
    known = {k: kw[k] for k in ("max_batch", "layers", "seq", "hf_config", "topk", "ep_weights", "native_t",
                                "weight_cache", "sample") if k in kw}  # fmt: skip
    return StackBackend(mesh_device, **known)


@dataclasses.dataclass
class Buf:
    """a DecodeIO field: the device tensor the trace binds + its name (write / read dispatch on the name)"""

    name: str
    t: Any
    shape: tuple

    @property
    def dtype(self):
        return torch.int64


@dataclasses.dataclass
class StackCache:
    max_batch: int
    max_seq_len: int
    num_blocks: int
    blocks_per_seq: int
    states: list  # [chunk][layer] (S, C) per-chip device tensors
    res: list  # [chunk] stacked residual [4, 1, 4, 4096] fp32 DRAM
    masks: list  # [chunk] the decoder masks (fp32, bf16, lower_bound * mask) [1, 1, 4, 1]
    stacks: list  # [chunk] [(layer indices, MkStack)]
    extra: dict = dataclasses.field(default_factory=dict)


def layer_kind(cfg, L: int) -> str:
    """the evidence / stage names of tt/config.py's layer kinds (kda_dense -> dense_kda, kda_moe -> sparse_kda,
    mla_moe -> sparse_dsa)"""
    return {"kda_dense": DENSE_KDA, "kda_moe": SPARSE_KDA, "mla_moe": SPARSE_DSA}[cfg.layer_kind(L)]


def dsa_paged_pools() -> bool:
    """GLM53_DSA_PAGED_POOLS=1 (Stage 12, doc/ttnn/12-fast-start/paged_pooled_keys.md rev 3; default off): the sharded
    pooled-key store is PAGED -- one [32, 128] page per local latent block (its 16 pooled rows in rows 0-15), sized by
    the device pool instead of rows x max context. Needs GLM53_DSA_SHARD and kernels that address it (mk_mla
    DSA_PAGED_POOLS, Stage 14 / 16)."""
    return os.environ.get("GLM53_DSA_PAGED_POOLS", "0") == "1"


def dsa_units_to_unit_order(per_chip: torch.Tensor, C: int, units: Optional[int], tpu: int = 4) -> torch.Tensor:
    """GLM53_DSA_SHARD host export: per-chip unit-paged rows [C, tiles, 32, 32] (chip c's local unit k = global unit
    k C + c) -> [1, U tpu, 32, 32] in GLOBAL unit order (chip-independent: served_1m_memory_plan.md E's unit). U = units
    (None: every local unit of every chip)."""
    C_, T = int(per_chip.shape[0]), int(per_chip.shape[1])
    if C_ != C:
        raise ValueError(f"unit-order export: {C_} chip rows for {C} chips")
    local_units = T // tpu
    U = local_units * C if units is None else int(units)
    out = per_chip.new_zeros(1, U * tpu, *per_chip.shape[2:])
    for u in range(U):
        c, k = u % C, u // C
        if k >= local_units:
            raise ValueError(f"unit {u} needs local unit {k} of chip {c}; the export holds {local_units} per chip")
        out[0, u * tpu : (u + 1) * tpu] = per_chip[c, k * tpu : (k + 1) * tpu]
    return out


def dsa_unit_order_to_chips(unit_order: torch.Tensor, C: int, local_tiles: int, tpu: int = 4) -> torch.Tensor:
    """inverse of dsa_units_to_unit_order: [1, U tpu, 32, 32] -> [C, local_tiles, 32, 32] (local units past the blob's
    stay zero: never read before the continuation writes them)"""
    if int(unit_order.shape[0]) != 1:
        raise ValueError(f"unit-order blob has {int(unit_order.shape[0])} rows")
    U = int(unit_order.shape[1]) // tpu
    out = unit_order.new_zeros(C, local_tiles, *unit_order.shape[2:])
    for u in range(U):
        c, k = u % C, u // C
        if (k + 1) * tpu > local_tiles:
            raise ValueError(f"unit {u} -> chip {c} local unit {k} exceeds the {local_tiles // tpu} local units")
        out[c, k * tpu : (k + 1) * tpu] = unit_order[0, u * tpu : (u + 1) * tpu]
    return out


class StackBackend(ModelBackend):
    vocab_size = VOCAB
    block_size = BLOCK_SIZE
    prefill_align = MODEL_PREFILL_ALIGN
    continuation_align = BLOCK_SIZE
    width_bucketed_decode = False  # one decode trace for every live count (the full width does the work anyway)
    samples_in_decode_trace = True
    copy_slot_keeps_sampler_rng = True  # the RNG row moves with the slot (copy_slot); False without device sampling

    @property
    def prefill_chunk_tokens(self) -> int:
        """the generator's chunk size: the TP prefill's largest bucket when attached (Stage 16 P3: 8,192-token buckets
        chunk a long prompt at 8,192; the default buckets end at PREFILL_CHUNK = 2,048, the earlier constant)"""
        tpp = getattr(self, "tpp", None)
        return max(tpp.buckets) if tpp is not None else PREFILL_CHUNK

    def __init__(self, mesh, *, max_batch: int = 32, layers: Optional[Sequence[int]] = None, seq: Optional[list] = None,
                 hf_config=None, topk: int = 32, ep_weights: Optional[bool] = None, native_t: bool = False,
                 weight_cache: Optional[dict] = None, sample: bool = True):  # fmt: skip
        """topk / sample: the head's candidate count and device sampling over the candidates (the serving contract's
        samples_in_decode_trace: k / p / temperature / seeds per row; top_k 0 or > topk samples from the topk best;
        sample=False or topk=1: greedy requests only).
        weight_cache (TP): a dict that keeps the layers' device weights for the next backend of this process (its
        own max_batch / chunk: only the programs are built; the 45-layer image loads once).
        seq (one chip, development): prebuilt layers [dict(L, mk, sparse, kda)] (tests/s14b/m5common.build_layers);
        otherwise the TP layers are built here (dense_kda / sparse_kda: Stage 14's TP layers; sparse_dsa: tt/tp_dsa.py,
        Stage 7b's TPMla + TPMoE)."""
        from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST
        from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config as _hf
        from models.autoports.zai_org_glm_5_3_flash.tt import mk_head as MKH
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource

        assert 1 <= max_batch <= 32, max_batch
        self.mesh, self.max_batch = mesh, max_batch
        self.weight_cache = weight_cache
        self.tp = int(mesh.get_num_devices())
        self.ep_weights = self.tp > 1 if ep_weights is None else bool(ep_weights)
        self.cfg = hf_config or _hf()
        # Stage 14 DSA decode (GLM53_STACK_DSA=1, default off): the sparse_dsa layers run MkMla(dsa=DsaIndexer) on a
        # per-layer pooled-key store + ring (allocate_kv_cache; Stage 16's TP prefill writes the same tensors), so the
        # context is no longer bounded by the dense-exact 2,051 rows. The DSA program runs the indexer every step (its
        # replicated projections stream 16.8 MB per chip per layer) and, at t > 4, token lanes + o_T since MkMla revs
        # 39 / 40: off keeps the dense MkMla until the A/B (card0 units 37 / 40) decides
        self.dsa_decode = os.environ.get("GLM53_STACK_DSA", "0") == "1"
        # GLM53_STACK_DSA_DEFER=1 (with GLM53_STACK_DSA=1): the sparse_dsa layers are CONSTRUCTED dense (the upload
        # sequence of a DRAM image recorded without the DSA stack: tt/tp_image.py replays only an identical one) and
        # get their DSA attention halves at the first allocate_kv_cache (attach_dsa: after the restore, real reads)
        self.dsa_defer = self.dsa_decode and os.environ.get("GLM53_STACK_DSA_DEFER", "0") == "1"
        # rows per stack launch: 4-row chunks (the stacked residual, every layer kind incl. sparse_dsa), or native_t:
        # ONE launch of max_batch rows per kind run (the native-t layers, the flat residual; sparse_dsa: MkMla t <= 32);
        # B < 4: one chunk of B rows (B=1 runs the t=1 programs, not a 4-row launch with 3 idle rows)
        self.chunk = max_batch if (native_t and max_batch > CHUNK) or max_batch < CHUNK else CHUNK
        # native-t on a mesh: the TP fabric's consumer side at t > 4 is mhc_post/reader.cpp's MK_POST_Y_FABRIC branch,
        # and the stack's argument CB is placed first (tt/cb_guard.py: otherwise tt-metal may place it inside a weight
        # ring at t = 8 under the serving knobs)
        self.n_chunks = -(-max_batch // self.chunk)
        # 4-row-chunk mode only: one decode trace per row-chunk count (rows 0..w-1 run chunks 0..w/chunk-1), so a few
        # live users decode at the cost of their chunks (the generator keeps live requests in the low slots); each width
        # trace holds its chunks' launches in the trace region. GLM53_STACK_WIDTHS (default "none": one full-width
        # trace; e.g. "4,8,16"), GLM53_STACK_PREFILL_TRACES (default "0": prefill replays the full decode trace; "1":
        # one prefill trace per chunk). The native-t mode uses GLM53_STACK_NATIVE_WIDTHS / _NARROW_WIDTHS below.
        spec = os.environ.get("GLM53_STACK_WIDTHS", "none")
        ws = () if spec in ("", "none") else tuple(int(v) for v in spec.split(","))
        self.width_bucketed_decode = tuple(w for w in ws if w < max_batch and w % self.chunk == 0) or False
        # native-t width buckets: extra programs for t = w (e.g. 8 / 16) on the SHARED 32-row state (per-user rows:
        # kda_rec S pages, kda_pre C tile rows, masks, latent cache / page table / positions), the full-width residual
        # through a w-row view and ONE aliased L1 pool (mk_stack.AllocPool.alias) -> a few live users (slots 0..w-1)
        # decode at the cost of w rows. GLM53_STACK_NATIVE_WIDTHS (default "none"; the served construction: "8,16")
        nspec = os.environ.get("GLM53_STACK_NATIVE_WIDTHS", "none")
        nws = () if nspec in ("", "none") else tuple(int(v) for v in nspec.split(","))
        self.native_widths = (tuple(sorted(w for w in nws if 4 < w < max_batch))
                              if self.chunk == max_batch and max_batch > CHUNK else ())  # fmt: skip
        if self.native_widths:
            self.width_bucketed_decode = self.native_widths
        # narrow widths (w <= 4): the STACKED t = w layer programs (the B <= 4 backends' programs) on the SHARED 32-row
        # state with their own stacked residual and t = w views of the embed / head; built after the image-recorded
        # construction (allocate_kv_cache: the DRAM image map stays the 32-row one's). GLM53_STACK_NARROW_WIDTHS ("none")
        aspec = os.environ.get("GLM53_STACK_NARROW_WIDTHS", "none")
        aws = () if aspec in ("", "none") else tuple(int(v) for v in aspec.split(","))
        self.narrow_widths = (tuple(sorted(w for w in aws if 1 <= w <= CHUNK))
                              if self.chunk == max_batch and max_batch > CHUNK else ())  # fmt: skip
        if self.narrow_widths:
            assert weight_cache is not None, "narrow widths reuse the cached layer weights (weight_cache)"
            self.width_bucketed_decode = tuple(sorted(set(self.narrow_widths) | set(self.native_widths)))
        self.nseq, self.nheads, self.nembeds = {}, {}, {}
        self.wseq = {}
        self.prefill_traces = os.environ.get("GLM53_STACK_PREFILL_TRACES", "0") == "1"
        # prefix-cache snapshot pool (tt/prefix_cache.py): GLM53_PREFIX_DEVICE_ENTRIES entries (default 32 = one state
        # chunk; 0 = none), each the non-paged state of one slot, allocated as extra state chunks after everything
        # else in allocate_kv_cache (the image-recorded construction's addresses do not move) and addressed as the
        # virtual slots max_batch + k by SlotOps / the prefill trace's state_io (one more chunk row each)
        # exactly the requested entries: the pool's last chunk holds the remainder rows (snapshot_chunk_rows); virtual
        # slot max_batch + k -> chunk (max_batch + k) // chunk, row (max_batch + k) % chunk; per-user pages come from
        # each state tensor's own rows. _snapshot_want is the request (never reset); snapshot_entries = what the
        # current full-size cache holds
        self._snapshot_want = max(0, int(os.environ.get("GLM53_PREFIX_DEVICE_ENTRIES", "32")))
        self.snapshot_entries = self._snapshot_want
        # GLM53_DSA_SHARD=1 (with GLM53_STACK_DSA=1; doc/ttnn/14-production-tp8-megakernel/dsa/shard_tp8_design.md
        # section 6.2): the sequence-sharded DSA pool -- 128-token unit u of a slot lives on chip u % tp only (latent
        # rows + pooled keys, local blocks / local pools), the sparse_dsa layers decode on tp_dsa_shard.TPMlaShard
        # (local top-512, candidate all-gather + exact merge, split-K attention, partial reduce-scatter); the
        # generator maps vLLM's blocks to owner-aware physical blocks (dsa_shard_owners, tt/owner_blocks.py) and every
        # chip gets its LOCAL page table. Scope: decode rows t <= SHARD_T_MAX = 32 (rev 60, W: native B=32 and the
        # native / narrow widths on token lanes; TPMlaShard builds streams for GLM53_STACK_NATIVE_WIDTHS), the
        # decode-program prefill or the TP prefill (GLM53_STACK_PREFILL=tp with GLM53_PF_DSA=1: tt/prefill_tp.py writes
        # each unit's latent pages / pooled keys to its owner chip only and reads the cached context through the
        # per-chip local tables), no decode tee; refused otherwise. The Stage 17 spec step only with
        # GLM53_SPEC_ON_SHARD=1 (v0, dsa/spec_on_shard_design.md: the verify on ONE token lane -- MkMla rev 61 refuses
        # otherwise -- its page table the slot's per-chip LOCAL row: shard_local_rows)
        self.dsa_shard = os.environ.get("GLM53_DSA_SHARD", "0") == "1"
        if self.dsa_shard:
            from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as _MK

            why = [w for w, bad in (
                ("GLM53_STACK_DSA=1", not self.dsa_decode), ("a TP mesh (tp > 1)", self.tp < 2),
                (f"decode rows t <= {_MK.SHARD_T_MAX} (chunk {self.chunk})", self.chunk > _MK.SHARD_T_MAX),
                ("GLM53_STACK_PREFILL=decode or tp", os.environ.get("GLM53_STACK_PREFILL", "decode") not in ("decode", "tp")),
                ("GLM53_PF_DSA=1 with GLM53_STACK_PREFILL=tp", os.environ.get("GLM53_STACK_PREFILL", "decode") == "tp"
                 and os.environ.get("GLM53_PF_DSA", "0") != "1"),
                ("GLM53_SPEC_K=0 or GLM53_SPEC_ON_SHARD=1", int(os.environ.get("GLM53_SPEC_K", "0")) > 0
                 and os.environ.get("GLM53_SPEC_ON_SHARD", "0") != "1"),
                ("GLM53_DECODE_TEE=0", os.environ.get("GLM53_DECODE_TEE", "0") == "1")) if bad]  # fmt: skip
            if why:
                raise ValueError(f"GLM53_DSA_SHARD=1 needs: {', '.join(why)}")
        self.prefill_state_io = False  # True once the TP prefill's in-trace StateIO exists (prepare_prefill)
        source = FdTileCacheSource(CACHE_DIR, MANIFEST)
        ids = list(layers) if layers is not None else list(range(self.cfg.num_hidden_layers))
        self.fl = None
        if seq is not None:
            assert self.tp == 1, "prebuilt layers: one chip"
            self.seq = seq
        else:
            self.seq = self._build_tp(source, ids)
        kinds = [layer_kind(self.cfg, e["L"]) for e in self.seq]
        self.layer_kinds = tuple(dict.fromkeys(kinds))
        # MLA / DSA bounds the context (dense-exact regime) once sparse_dsa layers are in the model
        if self.dsa_decode and not self.dsa_defer and not all(_has_dsa(e) for e in self.seq if e["sparse"] == "dsa"):
            raise ValueError("GLM53_STACK_DSA=1: a sparse_dsa layer without dsa= (prebuilt / weight-cached layers)")
        self.supported_context = 2051 if "sparse_dsa" in kinds and not self.dsa_decode else 1048576
        if self.dsa_shard:
            mlas = sorted({type(e["mla"]).__name__ for e in self.seq if e["sparse"] == "dsa"})
            print(f"StackBackend GLM53_DSA_SHARD=1: sparse_dsa layers on {mlas} over C = {self.tp} owner chips",
                  flush=True)  # fmt: skip
        self.logits_width = self.vocab_size
        lm, norm, emb = MKH.lm_head_host(source), MKH.norm_host(source), MKH.embed_host(source)
        self.sample = bool(sample) and topk > 1
        self.copy_slot_keeps_sampler_rng = self.sample
        # the Stage 9 adapter narrows vLLM's device-sampling route to this (sampled requests beyond it: host sampler)
        self.max_device_top_k = int(topk) if self.sample else 1
        hfab = self.fl.fabric if self.fl is not None else None
        if hfab is not None and -(-max_batch // hfab.brows) * self.tp * topk * 6 > 64 * 32:
            # the candidate gather of max_batch rows does not fit the layers' packed lanes (4-row chunks: 8 row groups
            # of tp x topk x 6 columns): the head gets its own fabric phase of max_batch rows (whole tiles at B > 16;
            # same lane cores, its own sync words and lane CBs)
            from models.autoports.zai_org_glm_5_3_flash.tt import fabric_ar as FAR

            hfab = FAR.FabricPhase(mesh, max_batch, lanes=hfab.lanes, mode="cb")
            hfab.probe()
        self.head = MKH.MkHead(mesh, lm, norm, t=max_batch, eps=self.cfg.rms_norm_eps, topk=topk, fabric=hfab,
                               n_prod=self.fl.n_readers if self.fl is not None else None, sample=self.sample)  # fmt: skip
        self._rng_counter = 0
        del lm
        self.embed = MKH.MkEmbed(mesh, emb, t=max_batch, place=self.head.place)
        del emb
        repl = ttnn.ReplicateTensorToMesh(mesh) if self.tp > 1 else None
        self.pos = ttnn.from_torch(torch.full((1, 32), -1, dtype=torch.int32), dtype=ttnn.int32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM, mesh_mapper=repl)  # fmt: skip
        self.slot_ops = SlotOps(mesh)
        self.moe_live_all = os.environ.get("GLM53_MOE_LIVE_ALL", "0") == "1"  # MKX_LIVE A/B: every row live (_moe_live)
        self._live_page = None
        self.pos_c = []  # per-chunk position copies (sparse_dsa layers present: allocate_kv_cache)
        self.params = dict(k=torch.ones(32, dtype=torch.int64), p=torch.zeros(32), temp=torch.ones(32))
        self.counters = dict(
            decode_eager=0, decode_replays=0, captures=0, prefill_tokens=0, slot_ops=0, seed_writes=0, idle_resets=0
        )
        self.traces = []
        self.wheads = {}  # native width -> its MkHead.width_view (GLM53_STACK_WIDTH_HEAD, default on)

    # ------------------------------------------------------------------ construction
    def _build_tp(self, source, ids):
        from models.autoports.zai_org_glm_5_3_flash.tt import ep_experts as EPX
        from models.autoports.zai_org_glm_5_3_flash.tt import tp8_layer as TPL
        from models.autoports.zai_org_glm_5_3_flash.tt import tp8_sparse as TPSP
        from models.autoports.zai_org_glm_5_3_flash.tt import tp_dsa as TPD
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import OptimizedDecoderConfig

        seq, dsa_share = [], None
        wc = self.weight_cache if self.weight_cache is not None else {}
        self._wc = wc  # the narrow widths' layer objects reuse these weights (allocate_kv_cache)
        build = EPX.build_moe_ep if self.ep_weights else TPSP.build_moe
        # the layers whose weights are not resident yet: their files prefetched by a thread pool AHEAD of the
        # constructor (tt/load_prefetch.py: parallel NFS reads + the expert permutations; GLM53_LOAD_PREFETCH=0: off)
        cached = lambda L: any((k, L) in wc for k in ("dense", "sparse", "dsa"))  # noqa: E731
        todo = [L for L in ids if not cached(L)]
        pf = None
        if todo and os.environ.get("GLM53_LOAD_PREFETCH", "1") == "1" and hasattr(source, "tensors"):
            from models.autoports.zai_org_glm_5_3_flash.tt.load_prefetch import LayerPrefetcher

            pf = LayerPrefetcher(source, todo, num_experts=self.cfg.n_routed_experts,
                                 expert_layers=[L for L in todo if layer_kind(self.cfg, L) != DENSE_KDA] if self.ep_weights
                                 else [], fused_cache_dir=OptimizedDecoderConfig().fused_experts_cache_dir)  # fmt: skip
        try:
            seq, dsa_share = self._build_tp_layers(source, ids, wc, build, pf, TPL, TPSP, TPD, OptimizedDecoderConfig)
        finally:
            if pf is not None:
                pf.close()
                self.load_info = dict(prefetched_gb=round(pf.bytes / 1e9, 1), prefetch_errors=pf.errors[:3])
        # the head's fabric phase and the state sharding (every KDA kind shards alike): the first KDA layer's
        self.fl = next(e["tp"] for e in seq if e["sparse"] != "dsa")
        for w in self.native_widths:
            self.wseq[w] = self._build_width(source, ids, wc, seq, w, TPL, TPSP, OptimizedDecoderConfig)
        return seq

    def _build_width(self, source, ids, wc, seq, w, TPL, TPSP, OptimizedDecoderConfig):
        """the TP layers of native width w: programs only (every weight from the cache wc, sliced experts included),
        each KDA kind on the full width's 32-row fabric phase (whole-tile packets: the t > 4 consumer); the sparse_dsa
        entries are the full width's (their TPDsaLayer objects carry a fabric phase per width)"""
        fabs = {}
        for e in seq:
            if e["sparse"] != "dsa":
                fabs.setdefault(SPARSE_KDA if e["sparse"] else DENSE_KDA, e["tp"].fabric)
        sliced = os.environ.get("GLM53_TP_XSL") == "1" and os.environ.get("GLM53_TP_XSL_SLICED") == "1"
        out = []
        for L, e in zip(ids, seq):
            assert e["L"] == L, (e["L"], L)
            if e["sparse"] == "dsa":
                out.append(e)
                continue
            kind = layer_kind(self.cfg, L)
            set_knobs(kind == SPARSE_KDA, self.tp, w)
            if kind == DENSE_KDA:
                tp = TPL.TPDenseKDALayer(self.mesh, self.cfg, L, source, OptimizedDecoderConfig(), b=w,
                                         weights=wc[("dense", L)], fabric=fabs.get(DENSE_KDA))  # fmt: skip
            else:
                moe = wc[("moe_rs", L) if sliced else ("moe", L)]
                tp = TPSP.TPSparseKDALayer(self.mesh, self.cfg, L, source, OptimizedDecoderConfig(), b=w, moe=moe,
                                           weights=wc[("sparse", L)], fabric=fabs.get(SPARSE_KDA))  # fmt: skip
            assert tp.fabric is fabs.get(SPARSE_KDA if kind == SPARSE_KDA else DENSE_KDA), ("width fabric", L, w)
            out.append(dict(L=L, mk=tp.mk, sparse=kind == SPARSE_KDA, kda=tp.attn, tp=tp))
        return out

    def _build_tp_layers(self, source, ids, wc, build, pf, TPL, TPSP, TPD, OptimizedDecoderConfig):
        seq, dsa_share = [], None
        # GLM53_DSA_PREFETCH=1 (default, with GLM53_STACK_DSA_DEFER): the weight-cache files the deferred DSA halves
        # read at attach_dsa (11 layers x ~15 files) go into the page cache NOW, by the restore's prefetch child, while
        # the image replay (no file reads of its own) runs: attach_dsa's reads then hit memory (int27c: the KV phase
        # held 11 attach rebuilds). Stopped (stats) by attach_dsa.
        self._dsa_pf = None
        if self.dsa_defer and os.environ.get("GLM53_DSA_PREFETCH", "1") == "1":
            paths = [p for L in ids if layer_kind(self.cfg, L) == SPARSE_DSA and ("dsa", L) not in wc
                     for p in TPD.TPDsaLayer.deferred_paths(source, L)]  # fmt: skip
            if paths:
                from models.autoports.zai_org_glm_5_3_flash.tt import tp_image as TI

                self._dsa_pf = TI.prefetch_files(paths, label="dsa attach")
        # one fabric phase per KDA kind (a stack program runs on its first layer's; each phase's sync buffer reserves
        # its address range on every core: 34 of them cost native B=32 its static-CB headroom). GLM53_STACK_SHARE_FABRIC
        # =0: one phase per layer (the earlier construction)
        fabs = {} if os.environ.get("GLM53_STACK_SHARE_FABRIC", "1") == "1" else None
        for L in ids:
            if pf is not None and L in pf.layers:
                pf.wait(L)
            kind = layer_kind(self.cfg, L)
            set_knobs("dsa" if kind == SPARSE_DSA else kind == SPARSE_KDA, self.tp, self.chunk)
            if kind == DENSE_KDA:
                tp = TPL.TPDenseKDALayer(self.mesh, self.cfg, L, source, OptimizedDecoderConfig(), b=self.chunk,
                                         weights=wc.get(("dense", L)),
                                         fabric=fabs.get(DENSE_KDA) if fabs is not None else None)  # fmt: skip
                wc[("dense", L)] = tp.w
                if fabs is not None:
                    fabs[DENSE_KDA] = tp.fabric
            elif kind == SPARSE_KDA:  # Stage 14: KDA heads TP, routed experts EP, router / shared expert replicated
                # the routed expert WEIGHTS EP-sharded too (tt/ep_experts.py: 36 per chip, the 45-layer image)
                # Stage 7a's sliced XSL layout (GLM53_TP_XSL=1 + GLM53_TP_XSL_SLICED=1): the layer loads its own
                # sliced experts (fused_decoder.load_xsl_experts) -> the MoE here brings router / shared only
                sliced = os.environ.get("GLM53_TP_XSL") == "1" and os.environ.get("GLM53_TP_XSL_SLICED") == "1"
                mkey = ("moe_rs", L) if sliced else ("moe", L)
                moe = wc.get(mkey) or (build(self.mesh, self.cfg, source, L, OptimizedDecoderConfig(), routed=False)
                                       if sliced and self.ep_weights else
                                       build(self.mesh, self.cfg, source, L, OptimizedDecoderConfig()))  # fmt: skip
                wc[mkey] = moe
                tp = TPSP.TPSparseKDALayer(self.mesh, self.cfg, L, source, OptimizedDecoderConfig(), b=self.chunk,
                                           moe=moe, weights=wc.get(("sparse", L)),
                                           fabric=fabs.get(SPARSE_KDA) if fabs is not None else None)  # fmt: skip
                wc[("sparse", L)] = tp.w
                if fabs is not None:
                    fabs[SPARSE_KDA] = tp.fabric
            else:  # sparse_dsa: Stage 7b's TPMla (MLA heads TP) + TPMoE (routed experts EP), two programs per chip
                d = wc.get(("dsa", L))
                if d is not None and (d.mla.mk.dsa is not None or getattr(d, "dsa_pending", False)) != self.dsa_decode:
                    raise ValueError(f"weight cache: sparse_dsa layer {L} built with dsa={d.mla.mk.dsa is not None}, "
                                     f"GLM53_STACK_DSA={int(self.dsa_decode)}")  # fmt: skip
                if d is not None and bool(getattr(d, "shard", False)) != self.dsa_shard:
                    raise ValueError(f"weight cache: sparse_dsa layer {L} built with shard={getattr(d, 'shard', False)}, "
                                     f"GLM53_DSA_SHARD={int(self.dsa_shard)}")  # fmt: skip
                if d is None:  # a cached layer serves every t <= 4 + this chunk (its fabric phases per row count)
                    moe = build(self.mesh, self.cfg, source, L, OptimizedDecoderConfig())
                    # (native widths > 4 run on the 32-row phases: tp_mla.fabric / tp_dsa._hooked fall back to them)
                    ts = tuple(sorted({1, 2, 3, 4, self.chunk})) if self.weight_cache is not None else (self.chunk,)
                    d = TPD.TPDsaLayer(self.mesh, self.cfg, L, source, OptimizedDecoderConfig(), moe=moe, ts=ts,
                                       share=dsa_share, dsa=self.dsa_decode, defer=self.dsa_defer,
                                       shard=self.dsa_shard)  # fmt: skip
                    wc[("dsa", L)] = d
                assert d.mla.nofab or (
                    self.chunk in d.mla.fabrics
                    and self.chunk in d.moe.fabrics
                    and (not self.native_widths or (32 in d.mla.fabrics and 32 in d.moe.fabrics))
                ), (L, self.chunk)
                dsa_share = dsa_share or d
                seq.append(dict(L=L, mla=d.mla, moe=d.moe, attn=d.attn, sparse="dsa", kda=None, tp=d))
                continue
            seq.append(dict(L=L, mk=tp.mk, sparse=kind == SPARSE_KDA, kda=tp.attn, tp=tp))
        return seq, dsa_share

    def _runs(self):
        runs, cur = [], []
        for i, e in enumerate(self.seq):
            if cur and (self.seq[cur[-1]]["sparse"] != e["sparse"] or e["sparse"] == "dsa"):  # dsa: runs of 1
                runs.append(cur)
                cur = []
            cur.append(i)
        return runs + [cur]

    def _kda0(self):
        """the first KDA layer's attention (state layout, the decoder masks' lower bound)"""
        return next(e["kda"] for e in self.seq if e["sparse"] != "dsa")

    def _new_state(self, rows: Optional[int] = None):
        """one state chunk of `rows` user rows (default the chunk; the prefix-cache pool's last chunk may be shorter)"""
        rows = int(rows or self.chunk)
        if self.fl is not None:
            sh = self.fl.shards
            if os.environ.get("GLM53_STATE_ZERO_ONCE", "1") == "1":  # (0: the per-state host tilize, the A/B)
                return sh.shard_state_zeros(self.chunk)
            return sh.shard_state(torch.zeros(rows, 64, 128, 128), torch.zeros(1, 3, rows, 24576, dtype=torch.bfloat16))
        st = self._kda0().allocate_state(rows)
        return st.S, st.C

    def _masks(self, active: torch.Tensor):
        lb = float(self._kda0().lower_bound)
        m = active.float().reshape(1, 1, self.chunk, 1)
        repl = ttnn.ReplicateTensorToMesh(self.mesh) if self.tp > 1 else None
        mk = lambda v, dt: ttnn.from_torch(
            v, dtype=dt, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=DRAM, mesh_mapper=repl
        )  # fmt: skip  # noqa: E731
        return mk(m, ttnn.float32), mk(m, ttnn.bfloat16), mk(m * lb, ttnn.float32)

    # ------------------------------------------------------------------ the contract
    def _moe_live(self):
        """the live-row page the routed experts read (ReaderExperts MKX_LIVE: an idle row, < 0, routes to no expert ->
        a live row's bits and the step's expert stream do not depend on idle rows; coordinator 2026-10-22 10:51Z) on a
        one-chunk step (the served native B = 32 / its widths / narrow), else None (chunked steps keep their programs).
        GLM53_MOE_LIVE=0: off. The page is the host's copy of cur_pos (write(cur_pos) mirrors it: liveness only changes
        by host writes, the trace's +1 keeps the sign), allocated here = after any DRAM-image restore and before the
        first capture; moe_live_all (runtime, GLM53_MOE_LIVE_ALL=1) writes every row live = the pre-fix routing on
        the same programs (the A/B of one build)."""
        if os.environ.get("GLM53_MOE_LIVE", "1") != "1" or self.n_chunks != 1:
            return None
        if getattr(self, "_live_page", None) is None:
            repl = ttnn.ReplicateTensorToMesh(self.mesh) if self.tp > 1 else None
            self._live_page = ttnn.from_torch(torch.full((1, 32), -1, dtype=torch.int32), dtype=ttnn.int32,
                                              layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, memory_config=DRAM,
                                              mesh_mapper=repl)  # fmt: skip
        return self._live_page

    def allocate_kv_cache(self, *, max_batch: int, max_seq_len: int, num_blocks: Optional[int] = None) -> StackCache:
        from models.autoports.zai_org_glm_5_3_flash.tt import mk_stack as MKS
        from models.autoports.zai_org_glm_5_3_flash.tt import tp_stack as TPS
        from models.autoports.zai_org_glm_5_3_flash.tt.generator import blocks_per_seq

        assert max_batch <= self.max_batch, (max_batch, self.max_batch)
        t_all = _time.perf_counter()
        sub = self.startup_sub = dict(getattr(self, "startup_sub", {}) or {})
        lap = [_time.perf_counter()]

        def mark(name):  # wall seconds since the previous mark (the KV-phase breakdown: adapter event `startup`)
            t = _time.perf_counter()
            sub[name] = round(t - lap[0], 3)
            lap[0] = t

        self.attach_dsa()
        mark("attach_s")
        bps = blocks_per_seq(max_seq_len)
        nb = num_blocks or max_batch * bps
        repl = ttnn.ReplicateTensorToMesh(self.mesh) if self.tp > 1 else None
        dsa = [i for i, e in enumerate(self.seq) if e["sparse"] == "dsa"]
        # MkMla reads cur_pos / page-table rows 0..3 of its tensors: per-chunk copies (the head writes the positions'
        # copies after its in-trace advance; the host writes both)
        repl_ = ttnn.ReplicateTensorToMesh(self.mesh) if self.tp > 1 else None
        self.pos_c = [ttnn.from_torch(torch.full((1, 32), -1, dtype=torch.int32), dtype=ttnn.int32,
                                      layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, memory_config=DRAM,
                                      mesh_mapper=repl_) for _ in range(self.n_chunks)] if dsa else []  # fmt: skip
        states = [[None if e["sparse"] == "dsa" else self._new_state() for e in self.seq] for _ in range(self.n_chunks)]
        rshape = (self.chunk, 1, 4, 4096) if self.chunk <= CHUNK else (1, 1, self.chunk, 16384)
        res = [ttnn.from_torch(torch.zeros(rshape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                               device=self.mesh, memory_config=DRAM, mesh_mapper=repl) for _ in range(self.n_chunks)]  # fmt: skip
        masks = [self._masks(torch.zeros(self.chunk, dtype=torch.bool)) for _ in range(self.n_chunks)]
        stacks, pool = [], None
        for c in range(self.n_chunks):
            per = []
            for r in self._runs():
                if self.seq[r[0]]["sparse"] == "dsa":  # MkMla + MkMoE, launched per step (no layer loop: runs of 1)
                    per.append((r, None))
                    continue
                mks = [self.seq[i]["mk"] for i in r]
                if self.fl is not None:  # the run's own fabric layer (lanes / FabricPhase of its kind)
                    cls = TPS.TPSparseStack if self.seq[r[0]]["sparse"] else TPS.TPStack
                    stk = cls(self.mesh, mks, self.seq[r[0]]["tp"], t=self.chunk, res=res[c], pool=pool,
                              probe=os.environ.get("GLM53_STACK_PROBE") == "1" and hasattr(mks[0], "PROBE_ROWS"),
                              live=self._moe_live() if self.seq[r[0]]["sparse"] else None)  # fmt: skip
                else:
                    stk = MKS.MkStack(self.mesh, mks, t=self.chunk, res=res[c], pool=pool,
                                      live=self._moe_live() if self.seq[r[0]]["sparse"] else None)  # fmt: skip
                pool = stk.pool
                per.append((r, stk))
            stacks.append(per)
        kv = StackCache(max_batch=max_batch, max_seq_len=max_seq_len, num_blocks=nb, blocks_per_seq=bps,
                        states=states, res=res, masks=masks, stacks=stacks)  # fmt: skip
        mark("states_stacks_s")
        if self.native_widths:  # the width stacks: w-row views of the residual / masks, the full width's pool aliased
            kv.extra["pool"] = pool
            kv.extra["wres"], kv.extra["wmasks"], kv.extra["wstacks"] = {}, {}, {}
            pool.alias = True
            try:
                for w in self.native_widths:
                    view = lambda t_, rows: ttnn.reshape(  # noqa: E731
                        t_, ttnn.Shape([int(v) for v in t_.shape][:-2] + [rows, int(t_.shape[-1])]),
                        ttnn.Shape([int(v) for v in t_.padded_shape]))  # fmt: skip
                    rw = view(res[0], w)
                    kv.extra["wres"][w] = rw
                    kv.extra["wmasks"][w] = tuple(view(m, w) for m in masks[0])
                    per = []
                    for r in self._runs():
                        ws_ = self.wseq[w]
                        if ws_[r[0]]["sparse"] == "dsa":
                            per.append((r, None))
                            continue
                        for i in r:  # the run's width layers on the full-width run's hand-off buffers (one L1 set)
                            _alias_persist(ws_[i]["mk"], self.seq[r[0]]["mk"], w, self.chunk)
                            if hasattr(ws_[i]["mk"], "experts"):
                                _alias_persist(ws_[i]["mk"].experts, self.seq[i]["mk"].experts, w, self.chunk)
                        cls = TPS.TPSparseStack if ws_[r[0]]["sparse"] else TPS.TPStack
                        per.append((r, cls(self.mesh, [ws_[i]["mk"] for i in r], ws_[r[0]]["tp"], t=w, res=rw,
                                           pool=pool, live=self._moe_live() if ws_[r[0]]["sparse"] else None)))  # fmt: skip
                    kv.extra["wstacks"][w] = per
            finally:
                pool.alias = False
        mark("width_stacks_s")
        # the sparse_dsa layers' paged latent caches (bf16 [blocks, 1, 64, 512], 7b's layout); GLM53_DSA_SHARD: every
        # chip's LOCAL pool (one address, its own content), local block 0 = the chip's null block (owner_blocks.py)
        page_tokens = bps * (self.seq[dsa[0]]["attn"].block_size if dsa else 64)
        nb_dev = nb
        if self.dsa_shard and dsa:
            from models.autoports.zai_org_glm_5_3_flash.tt import dsa_shard as DSH
            from models.autoports.zai_org_glm_5_3_flash.tt.owner_blocks import OwnerBlockMap

            bs = self.seq[dsa[0]]["attn"].block_size
            bpu = DSH.UNIT // bs
            nb_dev = OwnerBlockMap.local_blocks_for(self.tp, nb, max_batch, bpu)
            page_tokens = DSH.ShardMap(self.tp, bs).local_cap(bps * bs)  # a slot's local tokens on EVERY chip
            kv.extra["dsa_shard"] = (self.tp, nb_dev, bpu)
            kv.extra["bps_local"] = page_tokens // bs
        kv.extra["latent"] = {i: self.seq[i]["attn"].allocate_cache(nb_dev) for i in dsa}
        mark("latent_s")
        if self.dsa_decode:  # per sparse_dsa layer: the pooled-key store + ring of every row (slot b = row b)
            rows = self.n_chunks * self.chunk
            if self._snapshot_want and max_batch == self.max_batch:
                # + the prefix-cache snapshot pool's virtual slots max_batch + k (the pool's state chunks are
                # appended below): an entry holds the slot's pooled keys + ring too (_dsa_ops); the decode never
                # reads these rows. GLM53_PREFIX_DEVICE_ENTRIES=0: no pool, int26's allocation unchanged
                rows += self._snapshot_want
            # GLM53_DSA_STORE_SEATS=1 (n29h7 DRAM, coordinator 10:13:08Z / 10:15:53Z): the pooled-key store holds the
            # cache's SEATS (max_batch slots) only, not every decode row: at 1M x 32 rows it is 2.96 GB/chip. The
            # prefix cache's virtual slots get no pooled capacity (no device prefix pool with this knob). Invariant:
            # slot ids < max_batch (Glm53Generator binding), rows >= max_batch keep cur_pos < 0 (host writes -1;
            # topk_select advances only cur_pos >= 0), and an idle row addresses no pool (ix_dm1 index_row returns
            # for p < 0, att_dsa(_shard)_dm0 reads pools only for sparse(p))
            seats = None
            if os.environ.get("GLM53_DSA_STORE_SEATS", "0") == "1":
                if rows != self.n_chunks * self.chunk:
                    raise ValueError(
                        f"GLM53_DSA_STORE_SEATS=1: the device prefix pool ({self._snapshot_want} entries) would need "
                        "pooled-key capacity for its virtual slots; set GLM53_PREFIX_DEVICE_ENTRIES=0"
                    )
                seats = max_batch
            if dsa_paged_pools():  # GLM53_DSA_PAGED_POOLS: one pooled-key page per local latent block, not per slot
                kv.extra["dsa"] = {i: self._paged_pool_state(i, nb_dev, rows, page_tokens) for i in dsa}
            else:
                kv.extra["dsa"] = {i: self.seq[i]["mla"].mk.dsa.stack_state(self.mesh, rows, page_tokens,
                                                                            pool_slots=seats) for i in dsa}  # fmt: skip
            from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import dsa_dense_window

            if dsa_dense_window():  # GLM53_DSA_DENSE_WINDOW (dsa/dense_window_design.md): the replicated window store
                from models.autoports.zai_org_glm_5_3_flash.tt.mk_mla import DSA_WIN_BLOCKS

                # restores (LMCache / page DMA hits, device prefix attaches) rebuild the window from the restored latent
                # pages: rebuild_window, called by the generator before the row's first chunk (dense_window_restore_design.md)
                why = [w for w, bad in (("GLM53_DSA_SHARD=1", not self.dsa_shard),) if bad]
                if why:
                    raise ValueError(f"GLM53_DSA_DENSE_WINDOW=1 needs: {', '.join(why)}")
                # one slot per SEAT (slot ids < max_batch; idle rows read no window: att_dsa_shard_dm0.cpp)
                wz = torch.zeros(int(max_batch) * DSA_WIN_BLOCKS, 1, 64, 512, dtype=torch.bfloat16)
                for i in dsa:
                    kv.extra["dsa"][i] = dict(kv.extra["dsa"][i], window=ttnn.from_torch(
                        wz, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)),
                        window_slots=int(max_batch))  # fmt: skip
                del wz
                # the window's restore staging (dsa/dense_window_restore_design.md W1): one slot's 33 blocks of 32 tiles,
                # allocated with the cache, BEFORE any capture (rebuild_window moves pages through it between steps)
                kv.extra["win_stage"] = ttnn.zeros((DSA_WIN_BLOCKS * 32, 1, 32, 32), dtype=ttnn.bfloat16,
                                                   layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=DRAM)  # fmt: skip
            self._dsa_stage_with_cache(kv)  # per-slot pooled keys / GLM53_DSA_EXPORT_STAGE_TILES: with the cache
        mark("dsa_state_s")
        kv.extra["pool"] = pool
        if self.narrow_widths:
            self._build_narrow(kv)
        mark("narrow_s")
        # the prefix-cache snapshot pool: state chunks max_batch / chunk .. holding entries 0 .. snapshot_entries - 1
        # (virtual slot max_batch + k = chunk (max_batch + k) // chunk, row (max_batch + k) % chunk); the decode step
        # never runs them (it loops over self.n_chunks), slot ops and the prefill's state_io reach them by index
        kv.extra["snapshot_chunks"] = 0
        if self._snapshot_want and max_batch == self.max_batch:
            # the virtual slots continue the live chunks' row numbering: max_batch must fill whole chunks (else entry 0
            # would alias a live chunk's padding rows)
            assert max_batch % self.chunk == 0, (max_batch, self.chunk)
            for rows in snapshot_chunk_rows(self._snapshot_want, self.chunk):
                states.append([None if e["sparse"] == "dsa" else self._new_state(rows) for e in self.seq])
                kv.extra["snapshot_chunks"] += 1
        kv.extra["snapshot_entries"] = (
            self._snapshot_want if kv.extra["snapshot_chunks"] else 0
        )  # what this cache holds
        if max_batch == self.max_batch:
            self.snapshot_entries = kv.extra["snapshot_entries"]
        if os.environ.get("GLM53_DECODE_TEE", "0") == "1" and kv.extra["snapshot_chunks"]:
            kv.extra["tee"] = DecodeTee(self, kv)  # G snapshots inside the decode trace (its last program)
        mark("snapshot_pool_s")
        sub["allocate_kv_cache_s"] = round(_time.perf_counter() - t_all, 3)
        print(f"StackBackend.allocate_kv_cache: {sub['allocate_kv_cache_s']:.2f} s "
              f"({ {k: v for k, v in sub.items() if k.endswith('_s') and k != 'allocate_kv_cache_s'} })", flush=True)  # fmt: skip
        return kv

    def _build_narrow(self, kv: StackCache) -> None:
        """the narrow widths' layers (programs only: every weight from the weight cache, sliced experts included), one
        packed b = w fabric phase per KDA kind on that kind's lane cores, the stacked residual [w, 1, 4, 4096] of each
        width, w-row views of the masks, the width's stacks (the full width's L1 pool aliased where shapes match); the
        sparse_dsa layers are the full width's objects (their fabric phases already cover t = 1..4)"""
        from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST
        from models.autoports.zai_org_glm_5_3_flash.tt import fabric_ar as FAR
        from models.autoports.zai_org_glm_5_3_flash.tt import tp8_layer as TPL
        from models.autoports.zai_org_glm_5_3_flash.tt import tp8_sparse as TPSP
        from models.autoports.zai_org_glm_5_3_flash.tt import tp_stack as TPS
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import (
            FdTileCacheSource,
            OptimizedDecoderConfig,
        )

        assert self.fl is not None, "narrow widths: the TP stack (a mesh)"
        source = FdTileCacheSource(CACHE_DIR, MANIFEST)
        wc = self._wc
        sliced = os.environ.get("GLM53_TP_XSL") == "1" and os.environ.get("GLM53_TP_XSL_SLICED") == "1"
        repl = ttnn.ReplicateTensorToMesh(self.mesh) if self.tp > 1 else None
        pool = kv.extra["pool"]
        for w in self.narrow_widths:
            fabs, seq = {}, []
            for e in self.seq:
                L = e["L"]
                if e["sparse"] == "dsa":
                    assert w in e["tp"].mla.fabrics, ("sparse_dsa fabric phase for t", w, sorted(e["tp"].mla.fabrics))
                    seq.append(e)
                    continue
                kind = layer_kind(self.cfg, L)
                _narrow_knobs(kind == SPARSE_KDA, self.tp, w)
                if kind not in fabs:
                    f = FAR.FabricPhase(self.mesh, w, lanes=e["tp"].fabric.lanes, mode="cb")
                    f.probe()
                    fabs[kind] = f
                if kind == DENSE_KDA:
                    tp = TPL.TPDenseKDALayer(self.mesh, self.cfg, L, source, OptimizedDecoderConfig(), b=w,
                                             weights=wc[("dense", L)], fabric=fabs[kind])  # fmt: skip
                else:
                    tp = TPSP.TPSparseKDALayer(self.mesh, self.cfg, L, source, OptimizedDecoderConfig(), b=w,
                                               moe=wc[("moe_rs", L) if sliced else ("moe", L)],
                                               weights=wc[("sparse", L)], fabric=fabs[kind])  # fmt: skip
                seq.append(dict(L=L, mk=tp.mk, sparse=kind == SPARSE_KDA, kda=tp.attn, tp=tp))
            self.nseq[w] = seq
            rw = ttnn.from_torch(torch.zeros(w, 1, 4, 4096), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                 device=self.mesh, memory_config=DRAM, mesh_mapper=repl)  # fmt: skip
            per = []
            pool.alias = True
            try:
                for r in self._runs():
                    if seq[r[0]]["sparse"] == "dsa":
                        per.append((r, None))
                        continue
                    cls = TPS.TPSparseStack if seq[r[0]]["sparse"] else TPS.TPStack
                    per.append((r, cls(self.mesh, [seq[i]["mk"] for i in r], seq[r[0]]["tp"], t=w, res=rw, pool=pool,
                                       live=self._moe_live() if seq[r[0]]["sparse"] else None)))  # fmt: skip
            finally:
                pool.alias = False
            kv.extra.setdefault("nres", {})[w] = rw
            kv.extra.setdefault("nmasks", {})[w] = tuple(_rows_view(m, w) for m in kv.masks[0])
            kv.extra.setdefault("nstacks", {})[w] = per

    def _narrow_io(self, w: int):
        """the t = w embed / head of a narrow width: views of the 32-row ones (created at the width's first, eager step,
        after the 32-row head's first call: its h / outputs exist)"""
        from models.autoports.zai_org_glm_5_3_flash.tt import mk_head as MKH

        if w not in self.nheads:
            em = MKH.MkEmbed.__new__(MKH.MkEmbed)
            em.__dict__.update(self.embed.__dict__)
            em.t = w
            self.nembeds[w] = em
            self.nheads[w] = MKH.MkHead.width_view(self.head, w, _rows_view)
        return self.nembeds[w], self.nheads[w]

    def _step_narrow(self, kv: StackCache, w: int):
        """a narrow width's step: the t = w embed into its stacked residual, the t = w stacks / sparse_dsa programs on
        the shared 32-row state (rows 0..w-1), the t = w head (top-K / sampling / position advance of rows < w)"""
        rw, masks, pool = kv.extra["nres"][w], kv.extra["nmasks"][w], kv.extra["pool"]
        embed, head = self._narrow_io(w)
        embed(self.head.topk_out, [rw])
        pool.alias = True
        try:
            for r, stk in kv.extra["nstacks"][w]:
                if stk is None:
                    e = self.seq[r[0]]
                    set_knobs("dsa", self.tp, w)
                    s1 = e["mla"](rw, kv.extra["latent"][r[0]], kv.extra["pt_c"][0], self.pos_c[0], t=w, keep=("s1",),
                                  stack=True, dsa_state=self._dsa_st(kv, r[0], 0))["s1"]  # fmt: skip
                    e["moe"](s1, t=w, out=rw, live=self._moe_live())
                    ttnn.deallocate(s1)
                    del s1
                    continue
                _narrow_knobs(self.nseq[w][r[0]]["sparse"], self.tp, w)
                stk([kv.states[0][i][1] for i in r], masks, [kv.states[0][i][0] for i in r])
        finally:
            pool.alias = False
        head([rw], self.pos, pos_copies=self.pos_c)

    def dsa_shard_owners(self, kv: StackCache):
        """GLM53_DSA_SHARD: (C chips, physical blocks per chip incl. its null block 0, blocks per 128-token unit) for
        the generator's owner-aware block map (tt/owner_blocks.py); None = the replicated pool"""
        return kv.extra.get("dsa_shard")

    def latent_pages(self, kv: StackCache) -> dict:
        """{decoder layer: its paged latent [num_blocks, 1, 64, 512] bf16, replicated on the mesh} (KV connectors);
        GLM53_DSA_SHARD: every chip's LOCAL pool [nb_local, 1, 64, 512] -- one mesh tensor at one address, per-chip
        content, page index = LOCAL block (dsa_shard_owners gives C / nb_local / bpu; the connector reads block g from
        chip g % C at local g // C and writes it there only: design 6.3)"""
        return {self.seq[i]["L"]: t for i, t in kv.extra["latent"].items()}

    def allocate_decode_io(self, kv: StackCache) -> DecodeIO:
        B = kv.max_batch
        repl = ttnn.ReplicateTensorToMesh(self.mesh) if self.tp > 1 else None
        # rows for every chunk row (MkMla reads rows 0..3 of a chunk; rows past B map like the others, never active)
        rows = max(B, self.n_chunks * self.chunk)
        shard_pt = kv.extra.get("dsa_shard") is not None
        if (
            shard_pt
        ):  # every chip's LOCAL table [rows, bps_local], distributed like the shard unit's (write(): per chip)
            ncs = int(self.mesh.get_num_devices())
            pt = ttnn.from_torch(torch.zeros(ncs * rows, kv.extra["bps_local"], dtype=torch.int32), dtype=ttnn.int32,
                                 layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, memory_config=DRAM,
                                 mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0))  # fmt: skip
        else:
            pt = ttnn.from_torch((torch.arange(rows * kv.blocks_per_seq, dtype=torch.int32) % kv.num_blocks)
                                 .reshape(rows, kv.blocks_per_seq), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                 device=self.mesh, memory_config=DRAM, mesh_mapper=repl)  # fmt: skip
        io = DecodeIO(tokens=Buf("tokens", self.head.topk_out, (B,)), cur_pos=Buf("cur_pos", self.pos, (B,)),
                      page_table=Buf("page_table", pt, (B, kv.blocks_per_seq)),
                      logits=Buf("logits", self.head, (B, self.logits_width)),
                      sampled=Buf("sampled", self.head.sampled, (B,)), k=Buf("k", None, (B,)),
                      p=Buf("p", None, (B,)), temp=Buf("temp", None, (B,)), seeds=Buf("seeds", None, (B,)))  # fmt: skip
        kv.extra["io"] = io
        # Stage 17 (GLM53_SPEC_K = K, default off): the speculative step of a lone user in slot 0 (tt/spec_mtp.py),
        # its programs built here, compiled by prepare_decode before any capture
        self.spec = None
        if int(os.environ.get("GLM53_SPEC_K", "0")) > 0 and self.fl is not None:
            from models.autoports.zai_org_glm_5_3_flash.tt.spec_mtp import SpecMTP

            # GLM53_SPEC_DRAFTER (alias GLM53_SPEC_KIND, the served name): "mtp" (default, K <= 3), "none" (verify +
            # ACCEPT only, K <= 7: external drafts) or "dflash2" (the Apache-2.0 DFlash2 block drafter INSIDE the
            # step, K <= 7, tt/dflash2_kind.py; the served line: GLM53_SPEC_KIND=dflash2 GLM53_SPEC_K=4 on the rung-2
            # ops path (the kind's defaults); the rung-3 mk path only by GLM53_DF2_LAYER=mk once its silicon numerics pass)
            self.spec = SpecMTP(self, kv, K=int(os.environ["GLM53_SPEC_K"]),
                                drafter=os.environ.get("GLM53_SPEC_KIND") or os.environ.get("GLM53_SPEC_DRAFTER", "mtp"))  # fmt: skip
        if self.pos_c:  # per-chunk page tables (rows 4c..4c+3 at rows 0..3)
            host = (torch.arange(rows * kv.blocks_per_seq, dtype=torch.int32) % kv.num_blocks).reshape(rows, -1)
            if shard_pt:
                z = torch.zeros(ncs * self.chunk, kv.extra["bps_local"], dtype=torch.int32)
                kv.extra["pt_c"] = [ttnn.from_torch(z, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                                                    memory_config=DRAM, mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0))
                                    for c in range(self.n_chunks)]  # fmt: skip
            else:
                kv.extra["pt_c"] = [ttnn.from_torch(host[c * self.chunk : (c + 1) * self.chunk].contiguous(),
                                                    dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                                                    memory_config=DRAM, mesh_mapper=repl)
                                    for c in range(self.n_chunks)]  # fmt: skip
        self._kv = kv
        return io

    def _write_local_tables(self, buf: Buf, value: torch.Tensor) -> None:
        """GLM53_DSA_SHARD: the generator's physical global table [B, bps] (block g of column j on chip g % C, local block
        g // C) -> chip r's LOCAL table [rows, bps_local] (column j = local block j of the slot's local positions;
        tt/dsa_shard.py ShardMap.page_table_local), every chip its own content, in place (the live decode trace keeps
        its buffer); rows past B and columns past the table point at local block 0 (every chip's null block)"""
        from models.autoports.zai_org_glm_5_3_flash.tt import dsa_shard as DSH

        # GLM53_PT_INCREMENTAL (dsa_shard.LocalTableCache): only the changed rows are re-localized on the host, and a
        # per-chunk table no changed row falls in keeps its device copy; the main table is always copied whole
        pt_c = list(self._kv.extra.get("pt_c", []))
        cache = self.__dict__.setdefault("_pt_local", DSH.LocalTableCache())
        local, changed = cache.update((buf.t, *pt_c), value, int(buf.t.shape[0]), self.shard_local_rows)
        shard = ttnn.ShardTensorToMesh(self.mesh, dim=0)  # rank r = mesh device r (row-major: TPMlaShard's rank)
        to = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,  # noqa: E731
                                       mesh_mapper=shard)  # fmt: skip
        ttnn.copy_host_to_device_tensor(to(torch.cat(local)), buf.t)
        for c, ptc in enumerate(pt_c):
            if changed and not any(c * self.chunk <= s < (c + 1) * self.chunk for s in changed):
                continue  # no changed row in this chunk: its device table already holds these rows
            ttnn.copy_host_to_device_tensor(to(torch.cat([lt[c * self.chunk : (c + 1) * self.chunk] for lt in local])),
                                            ptc)  # fmt: skip

    def shard_local_rows(self, value: torch.Tensor, rows: int) -> list:
        """GLM53_DSA_SHARD: physical global page-table rows [n, bps] (block g of column j on chip g % C, local block
        g // C) -> [chip r's LOCAL rows [rows, bps_local] for r < C] (tt/dsa_shard.py ShardMap.page_table_local; rows past
        n: every column's owner null block = local block 0); raises on an off-owner entry"""
        from models.autoports.zai_org_glm_5_3_flash.tt import dsa_shard as DSH

        C, _nb_local, bpu = self._kv.extra["dsa_shard"]
        return DSH.local_rows(value, rows, C, bpu, int(self._kv.extra["bps_local"]))

    def _rows(self, value: torch.Tensor) -> ttnn.Tensor:
        rows = torch.zeros(32, 64, dtype=torch.int32)
        rows[: value.numel(), 0] = value.reshape(-1).to(torch.int32)
        repl = ttnn.ReplicateTensorToMesh(self.mesh) if self.tp > 1 else None
        return ttnn.from_torch(rows, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=repl)

    def write(self, buf: Buf, value: torch.Tensor) -> None:
        value = torch.as_tensor(value)
        repl = ttnn.ReplicateTensorToMesh(self.mesh) if self.tp > 1 else None
        if buf.name in ("tokens", "sampled"):  # sampled: the readback copy (prefill_chunk puts its rows back)
            ttnn.copy_host_to_device_tensor(self._rows(value), buf.t)
        elif buf.name == "cur_pos":
            pos = torch.full((1, 32), -1, dtype=torch.int32)
            pos[0, : value.numel()] = value.reshape(-1).to(torch.int32)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                                            mesh_mapper=repl), buf.t)  # fmt: skip
            for c, pc in enumerate(getattr(self, "pos_c", [])):
                cp = torch.full((1, 32), -1, dtype=torch.int32)
                cp[0, : 32 - c * self.chunk] = pos[0, c * self.chunk :]
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(cp, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                                                mesh_mapper=repl), pc)  # fmt: skip
            self._write_masks(value)
            if getattr(self, "_live_page", None) is not None:  # MKX_LIVE: the routed experts' live-row page
                lp = torch.zeros_like(pos) if self.moe_live_all else pos
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(lp, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                                                mesh_mapper=repl), self._live_page)  # fmt: skip
        elif buf.name == "page_table" and self._kv.extra.get("dsa_shard") is not None:
            self._pt_last = (buf.t, value.to(torch.int32).clone())  # set_page_table_host's same-value check
            self._pt_host = value.to(torch.int64).clone()  # the PHYSICAL global table (owner-aware, owner_blocks.py)
            self._write_local_tables(buf, value)
            if (
                getattr(self, "spec", None) is not None
            ):  # Stage 17 on the shard: slot 0's PHYSICAL row (spec_mtp localizes)
                self.spec.set_page_table(value[0])
        elif buf.name == "page_table":
            self._pt_last = (buf.t, value.to(torch.int32).clone())  # set_page_table_host's same-value check
            self._pt_host = value.to(torch.int64).clone()  # host shadow of the device page table
            rows = int(buf.t.shape[0])  # padded to every chunk row (allocate_decode_io)
            if value.shape[0] < rows:
                pad = (torch.arange(rows * value.shape[1], dtype=torch.int32) % self._kv.num_blocks).reshape(rows, -1)
                pad[: value.shape[0]] = value.to(torch.int32)
                value = pad
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(value.to(torch.int32), dtype=ttnn.int32,
                                                            layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=repl), buf.t)  # fmt: skip
            for c, ptc in enumerate(self._kv.extra.get("pt_c", [])):
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(value[c * self.chunk : (c + 1) * self.chunk].to(torch.int32)
                                                                .contiguous(), dtype=ttnn.int32,
                                                                layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=repl), ptc)  # fmt: skip
            if getattr(self, "spec", None) is not None:  # Stage 17: the speculating user's row (slot 0) -> pt_v rows
                self.spec.set_page_table(value[0])
        elif buf.name in ("k", "p", "temp"):
            self.params[buf.name][: value.numel()] = value.reshape(-1).to(self.params[buf.name].dtype)
            if buf.name == "temp":  # the generator writes k, p, temp in this order: the row set is complete
                n = value.numel()
                sampled = (self.params["k"][:n] != 1) & (self.params["temp"][:n] > 0)
                if not self.sample:
                    if bool(sampled.any()):
                        raise NotImplementedError("StackBackend(sample=False): greedy requests only")
                    return
                self._write_sampler_params()
        elif buf.name == "seeds":
            if self.sample:
                self._write_seeds(value.reshape(-1).to(torch.int64))
        else:
            raise ValueError(f"StackBackend.write: {buf.name} is not host-writable")

    # ------------------------------------------------------------------ device sampler (kernels/head/topk_select.cpp)
    def _params_match(self, row: int, kpt) -> bool:
        """the generator's params row equals (k, p, temperature) (a prefill's override needs no restore then)"""
        k, p, t = kpt
        return (int(self.params["k"][row]) == int(k) and float(self.params["p"][row]) == float(p)
                and float(self.params["temp"][row]) == float(t))  # fmt: skip

    def _write_sampler_params(self, override: Optional[dict] = None) -> None:
        """the params page: row r = (k, p, 1 / temperature, flags bit 0 = draw); greedy rows (k = 1 or temperature
        0) keep the argmax. override: {row: (k, p, temperature)} on top of the generator's params (a prefill)"""
        import struct

        page = torch.zeros(32, 4, dtype=torch.int64)
        for r in range(32):
            k, p, temp = int(self.params["k"][r]), float(self.params["p"][r]), float(self.params["temp"][r])
            if override and r in override:
                k, p, temp = int(override[r][0]), float(override[r][1]), float(override[r][2])
            draw = k != 1 and temp > 0
            page[r, 0] = max(0, k)
            page[r, 1] = struct.unpack("<I", struct.pack("<f", p if draw else 1.0))[0]
            page[r, 2] = struct.unpack("<I", struct.pack("<f", 1.0 / temp if draw else 1.0))[0]
            page[r, 3] = int(draw)
        self._write_page(self.head.samp_par, page)

    def _write_page(self, t, page: torch.Tensor) -> None:
        repl = ttnn.ReplicateTensorToMesh(self.mesh) if self.tp > 1 else None
        bits = ((page.reshape(1, 128).to(torch.int64) & 0xFFFFFFFF) ^ 0x80000000) - 0x80000000  # uint32 bits as int32
        host = ttnn.from_torch(bits.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=repl)
        ttnn.copy_host_to_device_tensor(host, t)

    def _read_rng(self) -> torch.Tensor:
        """the device RNG page (chip 0: every chip draws identically) -> [32, 4] int64 (uint32 values)"""
        d = ttnn.get_device_tensors(self.head.samp_rng)[0] if self.tp > 1 else self.head.samp_rng
        return ttnn.to_torch(d).to(torch.int64).reshape(32, 4) & 0xFFFFFFFF

    @staticmethod
    def _rng_state(seed: int) -> list:
        """xoshiro128** state of a 64-bit seed (splitmix64 -> 4 x 32 bits; never all zero)"""
        m = (1 << 64) - 1
        x, out = seed & m, []
        for _ in range(2):
            x = (x + 0x9E3779B97F4A7C15) & m
            z = x
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & m
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & m
            z ^= z >> 31
            out += [z & 0xFFFFFFFF, z >> 32]
        return out if any(out) else [1, 0, 0, 0]

    def _write_seeds(self, seeds: torch.Tensor) -> None:
        """generator contract: per row SEED_SKIP (-2) keeps the RNG, SEED_UNSEEDED (-1) a fresh unseeded stream,
        seed >= 0 that request's stream (the write itself re-seeds); read-modify-write of the device page"""
        if bool((seeds == -2).all()):
            return
        page = self._read_rng()
        for r, s_ in enumerate(seeds.tolist()[:32]):
            if s_ == -2:
                continue
            if s_ == -1:
                self._rng_counter += 1
                s_ = (0x5DEECE66D * self._rng_counter + 7919 * r) | (1 << 62)  # distinct from explicit seeds
            page[r] = torch.tensor(self._rng_state(int(s_)), dtype=torch.int64)
        self._write_page(self.head.samp_rng, page)
        self.counters["seed_writes"] += 1

    def _copy_rng_row(self, src: int, dst: int) -> None:
        page = self._read_rng()
        page[dst] = page[src]
        self._write_page(self.head.samp_rng, page)

    def _write_masks(self, cur_pos: torch.Tensor):
        """the decoder masks follow cur_pos (an inactive row's KDA state stays bit-identical): written with the host
        cur_pos write; the device only advances active rows (+1), which never changes a row's activeness"""
        kv = self._kv
        act = torch.zeros(self.n_chunks * self.chunk, dtype=torch.bool)
        act[: cur_pos.numel()] = cur_pos.reshape(-1) >= 0
        for c in range(self.n_chunks):
            new = self._masks(act[c * self.chunk : (c + 1) * self.chunk])
            for dst, src in zip(kv.masks[c], new):
                ttnn.copy(src, dst)

    def read(self, buf: Buf) -> torch.Tensor:
        if buf.name in ("tokens", "sampled"):
            rows = ttnn.to_torch(ttnn.get_device_tensors(buf.t)[0]).to(torch.int64).reshape(32, 64)
            return rows[: buf.shape[0], 0].clone()
        if buf.name == "cur_pos":
            return ttnn.to_torch(ttnn.get_device_tensors(buf.t)[0]).to(torch.int64).reshape(-1)[: buf.shape[0]]
        if buf.name == "logits":
            return self.head.gather(self.head.outs)[: buf.shape[0], : self.logits_width]
        if buf.name in self.params:
            return self.params[buf.name][: buf.shape[0]].clone()
        raise ValueError(f"StackBackend.read: {buf.name}")

    def read_async(self, buf):
        """tokens / sampled: the device -> host copy enqueued behind the submitted work (non-blocking) + an event;
        finish_read converts the host copy (what read() would have returned at the enqueue point). Others: blocking."""
        if buf.name not in ("tokens", "sampled") or MOCK:
            return self.read(buf), []
        host = ttnn.from_device(buf.t, blocking=False)
        return ("async", host, buf.shape[0]), [ttnn.record_event(self.mesh)]

    def finish_read(self, buf, pending):
        if isinstance(pending, tuple) and len(pending) == 3 and pending[0] == "async":
            _, host, n = pending
            parts = ttnn.get_device_tensors(host) if self.tp > 1 else [host]
            rows = ttnn.to_torch(parts[0]).to(torch.int64).reshape(32, 64)
            return rows[:n, 0].clone()
        return pending

    def attach_dsa(self) -> int:
        """GLM53_STACK_DSA_DEFER: give every deferred sparse_dsa layer its DSA attention half (TPDsaLayer.attach_dsa:
        real reads, after a DRAM-image restore) before any KV cache / stack program uses it; the stack entries (also the
        native widths', the same dicts) follow. Returns the number of layers attached (0: nothing deferred)."""
        n, share = 0, None
        t0 = _time.perf_counter()
        per = []
        for e in self.seq:
            d = e.get("tp")
            if e["sparse"] != "dsa" or not getattr(d, "dsa_pending", False):
                continue
            t = _time.perf_counter()
            d.attach_dsa(share)
            per.append(round(_time.perf_counter() - t, 2))
            share = share or d
            e["mla"], e["attn"] = d.mla, d.attn
            n += 1
        if n:
            assert all(
                _has_dsa(e) for e in self.seq if e["sparse"] == "dsa"
            ), "attach_dsa: a sparse_dsa layer without dsa"
            pf = getattr(self, "_dsa_pf", None)
            pf_stats = None
            if pf is not None:
                pf_stats = pf.stop()
                self._dsa_pf = None
            src = sorted({getattr(e.get("tp"), "slices_from", "?") for e in self.seq if e["sparse"] == "dsa"})
            self.startup_sub = dict(getattr(self, "startup_sub", {}) or {}, attach_dsa_s=round(_time.perf_counter() - t0, 3),
                                    attach_dsa_layers_s=per, dsa_prefetch=pf_stats, dsa_slices=src)  # fmt: skip
            print(f"StackBackend.attach_dsa: {n} sparse_dsa layers now run MkMla(dsa) (deferred past the image restore): "
                  f"{self.startup_sub['attach_dsa_s']:.2f} s (per layer {per}; slices {src}; prefetch {pf_stats})",
                  flush=True)  # fmt: skip
            # the attach's device temporaries sit in reference cycles until a collection: ~40 MB/bank on the n29h7
            # 2x4 mock, still allocated through the KV pool and the warm-up unless collected here
            import gc

            gc.collect()
        return n

    def set_page_table_host(self, kv, page_table: torch.Tensor) -> None:
        # the generator publishes with write(io.page_table) and then this hook, the same value into the same buffer on
        # this backend: the second write (on the sharded pool 8 chips' local tables + the per-chunk tables; 14 % of the
        # engine thread in the cf3g host profile) is skipped when nothing changed in between
        buf = kv.extra["io"].page_table
        last = getattr(self, "_pt_last", None)
        if last is not None and last[0] is buf.t and torch.equal(last[1], torch.as_tensor(page_table).to(torch.int32)):
            return
        self.write(buf, page_table)

    def reset_slot(self, kv: StackCache, slot: int) -> None:
        self.slot_ops.run(self._state_ops(kv, slot, None))
        self.counters["slot_ops"] += 1

    def release_slot(self, kv: StackCache, slot: int) -> None:
        """Canonical idle row (coordinator 2026-10-22 10:51:21Z): a released slot's per-request state goes back to the
        fresh-server value (zero KDA state / conv rows, as reset_all leaves every slot), so the idle rows of a width
        trace compute what they compute on a fresh server instead of a finished request's history: stale idle rows
        cost the lone user after a 32-user burst 4.4 ms per step (TPOT 28.5 vs 24.1 ms fresh) and flipped a near-tie of
        the live row (pagegrowth req 1). Enqueued behind any in-flight step (same command queue); the token row goes
        to 0 at the generator's next whole-buffer token write. GLM53_IDLE_CANON=0: no device work at release."""
        if os.environ.get("GLM53_IDLE_CANON", "1") != "1":
            return
        self.reset_slot(kv, slot)
        self.counters["idle_resets"] += 1

    def copy_slot(self, kv: StackCache, src: int, dst: int) -> None:
        self.slot_ops.run(self._state_ops(kv, dst, src))
        if self.sample:  # the sampler RNG row moves with the slot (copy_slot_keeps_sampler_rng)
            self._copy_rng_row(src, dst)
        self.counters["slot_ops"] += 1

    def reset_all(self, kv: StackCache) -> None:
        for s in range(kv.max_batch):
            self.reset_slot(kv, s)

    # ---- prefix-cache snapshots (plain DMA between steps: SlotOps kinds 1 / 2 over every KDA layer, plus the DSA
    # pooled-key store pages and ring tiles when the TP prefill keeps them; one launch)
    def _entry_slot(self, kv: StackCache, entry: int) -> int:
        n = int(kv.extra.get("snapshot_entries", 0))
        if not 0 <= int(entry) < n:
            raise ValueError(f"snapshot entry {entry} outside the pool of {n} entries")
        return int(kv.max_batch) + int(entry)

    def _state_copy(self, kv: StackCache, src: int, dst: int, upto: Optional[int] = None) -> None:
        """slot dst <- slot src (virtual slots allowed): the TP prefill's state_io copy program (one core per KDA
        layer, ~ms) when it exists, else SlotOps (one core, serialized pages: ~50 ms per chip at TP8); then the DSA
        pooled keys / ring (SlotOps page copies) in the sparse regime. upto: the tokens the state covers (a multiple
        of 128 at every snapshot point): only the pooled-key store pages of positions < upto move (None: all)"""
        tpp = getattr(self, "tpp", None)
        sio = getattr(tpp, "sio", None) if tpp is not None and tpp.kv is kv else None
        if sio is not None:
            sio.copy(src, dst)
            ops = self._dsa_ops(kv, dst, src, upto)
            self.counters["state_copy_sio"] = self.counters.get("state_copy_sio", 0) + 1
        else:
            ops = self._state_ops(kv, dst, src, dsa=False) + self._dsa_ops(kv, dst, src, upto)
        if ops:
            self.slot_ops.run(ops)
            self.counters["slot_ops"] += 1
        self.counters["snapshot_ops"] = self.counters.get("snapshot_ops", 0) + 1

    def decode_tee(self, kv: StackCache) -> bool:
        """the decode trace tees crossing slots into their snapshot entries (GLM53_DECODE_TEE=1, DecodeTee)"""
        return kv.extra.get("tee") is not None

    def set_decode_tee(self, kv: StackCache, pairs) -> None:
        """the next decode step's tee pairs [(slot, entry, t0, t1)] in tokens (DecodeTee.set); [] clears"""
        kv.extra["tee"].set(pairs)

    def snapshot_slot(self, kv: StackCache, slot: int, entry: int, upto: Optional[int] = None) -> None:
        self._state_copy(kv, int(slot), self._entry_slot(kv, entry), upto)

    def restore_slot(self, kv: StackCache, entry: int, slot: int, upto: Optional[int] = None) -> None:
        self._state_copy(kv, self._entry_slot(kv, entry), int(slot), upto)

    def _dsa_ops(self, kv: StackCache, dst: int, src: int, upto: Optional[int] = None) -> list:
        """the per-slot DSA indexer state, dst <- src (contiguous pages per slot: SlotOps kind 1): every tensor of
        _dsa_tensors. Tiles per slot = the tensor's tiles / its rows (the store: pages_per_slot x 4 tiles; the ring: 8).
        upto: of a unit-paged tensor (a pooled-key store) only the 128-token units of positions < upto move (a snapshot
        at boundary B needs the pools of [0, B), 2026-10-22); None: the whole slot. A slot outside a tensor's rows (a
        snapshot entry the store was not sized for) raises: never a copy out of bounds."""
        ops = []
        for _key, t, rows, paged in self._dsa_tensors(kv):
            if not (0 <= src < rows and 0 <= dst < rows):
                raise ValueError(f"DSA state copy {src} -> {dst}: a per-slot store of {rows} rows")
            n = _pages(t) // rows
            units = None if upto is None else -(-int(upto) // self.DSA_UNIT)
            if units is not None and kv.extra.get("dsa_shard") is not None:
                # the sharded store holds a slot's LOCAL units: chip r owns ceil((U - r) / C) of [0, U); every chip
                # copies the same ceil(U / C) (one program; the extra unit's rows are never selected)
                units = -(-units // kv.extra["dsa_shard"][0])
            m = n if units is None or not paged else min(n, units * 4)
            if m:
                addr = int(t.buffer_address())
                ops.append((1, addr, src * n, addr, dst * n, m, _pages_bytes(t), 0))
        return ops

    def _dsa_tensors(self, kv: StackCache) -> list:
        """[(key, tensor, rows, unit_paged)] of every distinct per-slot DSA tensor in a fixed order: the decode's
        per-layer pooled-key store + ring (GLM53_STACK_DSA: kv.extra["dsa"], rows = its "slots"), then the TP prefill's
        (sparse regime, rows = tpp.slot_rows), deduplicated by buffer (the prefill aliases the decode's store when both
        exist). unit_paged: the slot's rows are 128-token units of 4 tiles (a [32, 128] bf16 page of 32 pooled keys);
        else (the ring) the slot's tiles are one record."""
        out, seen = [], set()

        def add(key, t, rows, paged):
            a = int(t.buffer_address())
            if a not in seen:
                seen.add(a)
                out.append((key, t, int(rows), paged))

        for i in sorted(kv.extra.get("dsa", {})):
            st = kv.extra["dsa"][i]
            if st.get("paged", False):  # GLM53_DSA_PAGED_POOLS: the store is per block, not per slot. Its buffer counts
                # as seen, so the TP prefill's alias of it (MlaPrefill.idx.store = st["pools"]) is skipped too: as a
                # per-slot store it would export / import / zero OTHER blocks' pages as this slot's units
                seen.add(int(st["pools"].buffer_address()))
            else:
                add(f"L{self.seq[i]['L']}.pools", st["pools"], st.get("pool_slots", st["slots"]), True)
            add(f"L{self.seq[i]['L']}.ring", st["ring"], st["slots"], False)
        tpp = getattr(self, "tpp", None)
        if tpp is not None and tpp.kv is kv and tpp.has_dsa():
            for j, pl in enumerate(tpp.layers):
                idx = getattr(getattr(pl, "mla", None), "idx", None)  # MlaPrefill.idx (PrefillLayer has none)
                if idx is None:
                    continue
                add(f"pf{j}.store", idx.store, tpp.slot_rows, True)
                add(f"pf{j}.ring", idx.ring, tpp.slot_rows, False)
        return out

    def pooled_store_tensors(self, kv: Optional[StackCache] = None) -> list:
        """Stage 12 (GLM53_KV_POOLED_PAGES): the 11 sparse_dsa layers' pooled-key stores in layer order (the latent
        caches' order) -- the page-DMA's pool_tensors; [] when the cache has no single set of them (no DSA state, or a
        decode and a prefill store that are not aliased)"""
        kv = kv or self._kv
        if self.pooled_paged(kv):  # GLM53_DSA_PAGED_POOLS: the per-block stores (not in _dsa_tensors)
            dsa = kv.extra["dsa"]
            tens = [(dsa[i]["pools"], 1) for i in sorted(dsa)]
        else:
            tens = [(t, rows) for _k, t, rows, paged in self._dsa_tensors(kv) if paged]
        if not tens or len(tens) != len(kv.extra.get("latent", {})):
            return []
        if len({(_pages(t), rows) for t, rows in tens}) != 1:
            return []
        return [t for t, _rows in tens]

    def pooled_unit_pages(self, kv: StackCache, slot: int, units) -> list:
        """(chip, page) of slot's 128-token units in the pooled-key stores (the same page in every layer): GLM53_DSA_SHARD
        = unit u on chip u % C, local page slot x pps + u // C; replicated = (None, slot x pps + u)"""
        tens = [(t, rows) for _k, t, rows, paged in self._dsa_tensors(kv) if paged]
        if not tens:
            raise ValueError("no pooled-key store")
        t, rows = tens[0]
        if not 0 <= int(slot) < int(rows):
            raise ValueError(f"slot {slot} outside the pooled-key store's {rows} rows")
        pps = _pages(t) // int(rows) // self.DSA_TILES_PER_UNIT
        shard = kv.extra.get("dsa_shard")
        C = int(shard[0]) if shard is not None else 1
        out = []
        for u in units:
            c, k = (int(u) % C, int(u) // C) if shard is not None else (None, int(u))
            if not 0 <= k < pps:
                raise ValueError(f"unit {u} needs local unit {k}; a slot holds {pps}")
            out.append((c, int(slot) * pps + k))
        return out

    def _paged_pool_state(self, i: int, nb_local: int, rows: int, page_tokens: int) -> dict:
        """GLM53_DSA_PAGED_POOLS: layer i's DSA state with the PAGED pooled-key store -- pools [nb_local, 1, 32, 128]
        (local latent block lb's 16 pooled rows = rows 0-15 of page lb; the null block 0 owns page 0), the ring per row
        as before; "paged" marks it (no per-slot pooled rows: _dsa_tensors leaves the store out of slot copies and
        checkpoints). Refused until the decode / prefill kernels address it (mk_mla.DSA_PAGED_POOLS)."""
        from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as MKM  # noqa: PLC0415

        if not self.dsa_shard:
            raise ValueError("GLM53_DSA_PAGED_POOLS=1 needs GLM53_DSA_SHARD=1 (the store follows the owner blocks)")
        if not getattr(MKM, "DSA_PAGED_POOLS", False):
            raise NotImplementedError(
                "GLM53_DSA_PAGED_POOLS=1: the decode / prefill kernels do not address the paged pooled-key store yet "
                "(Stage 14 / 16, doc/ttnn/12-fast-start/paged_pooled_keys.md)")  # fmt: skip
        ix = self.seq[i]["mla"].mk.dsa
        z = lambda shape: ttnn.zeros(
            shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=DRAM  # noqa: E731
        )
        return dict(pools=z((int(nb_local), 1, 32, ix.DIM)), ring=z((int(rows), 1, 32, 2 * ix.DIM)),
                    tp=ix.pool_rows(page_tokens), slots=int(rows), s0=0, paged=True)  # fmt: skip

    def pooled_paged(self, kv: Optional[StackCache] = None) -> bool:
        """the cache's pooled-key stores are paged by latent block (GLM53_DSA_PAGED_POOLS)"""
        dsa = (kv or self._kv).extra.get("dsa", {})
        return bool(dsa) and all(st.get("paged", False) for st in dsa.values())

    def pooled_block_pages(self, kv: StackCache, phys_blocks) -> list:
        """GLM53_DSA_PAGED_POOLS: (owner chip, page) of physical global blocks g in the paged stores = (g % C, g // C)"""
        C = int(kv.extra["dsa_shard"][0])
        return [(int(g) % C, int(g) // C) for g in phys_blocks]

    def has_dsa_state(self, kv: Optional[StackCache] = None) -> bool:
        """the cache keeps per-slot DSA indexer state (the sparse regime): a prefix checkpoint must carry it"""
        return bool(self._dsa_tensors(kv or self._kv))

    # ---- the DSA host export (Stage 12 host KV tier, doc/ttnn/12-fast-start/page_dma/host_kv_tier.md 5b): a slot's
    # pooled-key rows of its first `units` 128-token units (served_1m_memory_plan.md E's unit: one [32, 128] bf16 page =
    # 4 tiles per unit per layer, 64 B per token) + its ring tiles, moved between the slot and a host blob through a
    # staging buffer allocated with the cache (before any capture) by SlotOps (compiled at reset_slot, before any
    # capture): no allocation and no program compile after a capture; plain DMA between steps, like snapshot_slot.
    DSA_UNIT = 128
    DSA_BLOB = "glm53_dsa_v1"
    DSA_TILES_PER_UNIT = 4  # one [32, 128] bf16 pooled-key page per unit per layer

    def ensure_dsa_export(self, kv: Optional[StackCache] = None) -> bool:
        """the DSA host export's staging buffer, allocated on demand by its consumers (the adapter when a host tier or a
        KV connector is configured) and only BEFORE the first capture (a buffer allocated after a capture could sit on
        a captured program's transient memory). False: no DSA state, or GLM53_DSA_EXPORT_STAGE_TILES=0."""
        kv = kv or self._kv
        if not self._dsa_tensors(kv):
            return False
        if kv.extra.get("dsa_stage") is None and self.counters.get("captures", 0):
            raise RuntimeError("the DSA export staging buffer must be allocated before the first trace capture")
        self._ensure_dsa_stage(kv)
        return kv.extra.get("dsa_stage") is not None

    def ensure_state_export(self, kv: Optional[StackCache] = None) -> bool:
        """every prefix-checkpoint staging buffer (before the first capture, on demand): the KDA pack (S pages of
        GLM53_KDA_EXPORT_STAGE_LAYERS layers per round, default 8; one conv-row tile set holding 32 layers' rows; 0 = the
        per-layer host gather of whole chunks) and the DSA rows. True when the KDA pack is available."""
        kv = kv or self._kv
        if kv.extra.get("kda_stage_s") is None and self.counters.get("captures", 0):
            raise RuntimeError("the state export staging buffers must be allocated before the first trace capture")
        self.ensure_dsa_export(kv)
        if kv.extra.get("kda_stage_s") is None:
            lay = self._kda_layers(kv, 0)
            ls = min(int(os.environ.get("GLM53_KDA_EXPORT_STAGE_LAYERS", "8")), len(lay))
            if ls > 0:
                S0, C0 = lay[0][2]
                n_s, n_c = _pages(S0) // self.chunk, _pages(C0)
                assert all(_pages(S) // self.chunk == n_s and _pages(C) == n_c and S.dtype == S0.dtype and C.dtype == C0.dtype
                           for _i, _L, (S, C) in lay), "KDA layers with different state geometries"  # fmt: skip
                kv.extra["kda_stage_s"] = ttnn.zeros((ls * n_s, 1, 32, 32), dtype=S0.dtype, layout=ttnn.TILE_LAYOUT,
                                                     device=self.mesh, memory_config=DRAM)  # fmt: skip
                kv.extra["kda_stage_c"] = ttnn.zeros((n_c, 1, 32, 32), dtype=C0.dtype, layout=ttnn.TILE_LAYOUT,
                                                     device=self.mesh, memory_config=DRAM)  # fmt: skip
        return kv.extra.get("kda_stage_s") is not None

    def _kda_layers(self, kv: StackCache, c: int) -> list:
        """[(layer index, L, (S, C))] of the KDA layers of state chunk c"""
        return [(i, e["L"], kv.states[c][i]) for i, e in enumerate(self.seq) if kv.states[c][i] is not None]

    # ---- the in-replay checkpoint source (one-replay store, doc/ttnn/09-vllm/perf/one_replay_store/design.md: K1, K2
    # and M1 write it; M2 = the readers below). Contract: kv.extra["kda_snap_stage"] = [(S_t, s_page0, C_t, c_row)] per
    # KDA layer in _kda_layers order. S_t's pages [s_page0, s_page0 + n_s) hold the layer's recurrent state at the cut
    # in the live state tensor's slot-row page layout (the n_s pages one slot occupies); C_t (the live conv tensor's
    # page geometry and dtype) holds the layer's conv carry at the cut in tile row c_row. kv.extra["kda_snap_at"] =
    # (slot, cut) is set by the host BEFORE the replay that writes them (with its snap_end). An export from the
    # snapshot consumes the tag: one export per snapshot, so a later request in the same slot never reads a stale one.
    KDA_SOURCES = ("live", "snapshot")

    @property
    def kda_checkpoint_in_replay(self) -> bool:
        """the TP prefill writes the KDA state at a requested cut inside the replay (GLM53_PF_KDA_SNAPSHOT=1): a storing
        request runs whole and exports its checkpoint at the cut from that snapshot (one-replay store A). Before
        prepare_prefill builds the TP prefill (warmup phase 1, after initialize_kv_cache: the vLLM connector's
        from_runner guard asks first) the answer is the configuration prepare_prefill builds it from. Not with a PMK
        chain attached (its KDA scan writes no cut state; PmkStack.attach refuses the combination)"""
        tpp = getattr(self, "tpp", None)
        if tpp is None:
            return (os.environ.get("GLM53_STACK_PREFILL", "decode") == "tp"
                    and os.environ.get("GLM53_PF_KDA_SNAPSHOT", "0") == "1")  # fmt: skip
        return bool(getattr(tpp, "kda_snapshot", False) and not getattr(tpp, "pmk_layers", None))

    def kda_snapshot_holds(self, kv: StackCache, slot: int, n_tokens: int) -> bool:
        """True when the snapshot staging holds slot's state at n_tokens (the replay that wrote it carried this cut)"""
        at = kv.extra.get("kda_snap_at")
        return kv.extra.get("kda_snap_stage") is not None and at is not None and tuple(at) == (int(slot), int(n_tokens))

    def _kda_state_src(self, kv: StackCache, slot: int, n_tokens: Optional[int], source: str) -> list:
        """[(L, S, first S page, C, conv row)] per KDA layer: where slot's checkpoint state lives. "live": the slot's
        rows of the state tensors (its state after everything it processed); "snapshot": the in-replay staging at the
        cut n_tokens (the contract above; raises ValueError when it does not hold slot's state at n_tokens or does not
        match the state geometry -- the caller then has no checkpoint at the cut)"""
        if source not in self.KDA_SOURCES:
            raise ValueError(f"KDA state source {source!r} not in {self.KDA_SOURCES}")
        c, rd = divmod(int(slot), self.chunk)
        lay = self._kda_layers(kv, c)
        if not lay:
            return []
        n_s = _pages(lay[0][2][0]) // self.chunk
        if source == "live":
            return [(L, S, rd * n_s, C, rd) for _i, L, (S, C) in lay]
        stage = kv.extra.get("kda_snap_stage")
        if n_tokens is None or not self.kda_snapshot_holds(kv, slot, n_tokens):
            have = kv.extra.get("kda_snap_at") if stage is not None else "no staging"
            raise ValueError(f"the KDA snapshot staging does not hold slot {slot} at {n_tokens} tokens ({have})")
        if len(stage) != len(lay):
            raise ValueError(f"the KDA snapshot staging has {len(stage)} layers, the cache {len(lay)}")
        out = []
        for (_i, L, (S, C)), (S_t, s0, C_t, crow) in zip(lay, stage):
            ok = (S_t.dtype == S.dtype and C_t.dtype == C.dtype and _pages(C_t) == _pages(C) and 0 <= int(crow) < 32
                  and 0 <= int(s0) <= _pages(S_t) - n_s)  # fmt: skip
            if not ok:
                raise ValueError(f"the KDA snapshot staging of layer {L} does not match the state geometry")
            out.append((L, S_t, int(s0), C_t, int(crow)))
        return out

    KDA_PACK = "glm53_kda_pack_v1"

    def export_kda_pack(self, kv: StackCache, slot: int, n_tokens: Optional[int] = None, source: str = "live") -> dict:
        """slot's KDA state of every KDA layer through the staging buffers (no whole-chunk gather): S = the slot's
        contiguous pages (SlotOps kind 1, per round GLM53_KDA_EXPORT_STAGE_LAYERS layers), conv = the slot's row of every
        tile (kind 2, row j of the conv staging = layer j, 32 layers per round). {"kind": KDA_PACK, "layers": [L],
        "S": [chips, layers, pages, 32, 32], "C": [chips, layers, tiles, 32]} -- chip-local tiles (each chip its own
        heads / channels), restored only by import_kda_pack on the same geometry. source (_kda_state_src): "live" = the
        slot's rows, "snapshot" = the in-replay staging at the cut n_tokens (consumes its tag)."""
        src = self._kda_state_src(kv, slot, n_tokens, source)
        ss, sc = kv.extra["kda_stage_s"], kv.extra["kda_stage_c"]
        n_s = _pages(self._kda_layers(kv, int(slot) // self.chunk)[0][2][0]) // self.chunk
        ls = int(ss.padded_shape[0]) // n_s
        sa, ca = int(ss.buffer_address()), int(sc.buffer_address())
        s_parts, c_parts = [], []
        for g in range(0, len(src), ls):
            grp = src[g : g + ls]
            self.slot_ops.run([(1, int(S.buffer_address()), s0, sa, j * n_s, n_s, _pages_bytes(S), 0)
                               for j, (_L, S, s0, _C, _r) in enumerate(grp)])  # fmt: skip
            self.counters["slot_ops"] += 1
            host = self._stage_read(ss)
            s_parts.append(host[:, : len(grp) * n_s].reshape(host.shape[0], len(grp), n_s, 32, 32).clone())
        for g in range(0, len(src), 32):
            grp = src[g : g + 32]
            self.slot_ops.run([(2, int(C.buffer_address()), 0, ca, 0, _pages(C), _pages_bytes(C), (r << 16) | j)
                               for j, (_L, _S, _s0, C, r) in enumerate(grp)])  # fmt: skip
            self.counters["slot_ops"] += 1
            host = self._stage_read(sc)  # [chips, tiles, 32, 32]: layer j = row j of every tile
            c_parts.append(host[:, :, : len(grp), :].permute(0, 2, 1, 3).clone())
        self.counters["kda_pack_exports"] = self.counters.get("kda_pack_exports", 0) + 1
        if source == "snapshot":
            kv.extra["kda_snap_at"] = None
            self.counters["kda_snapshot_exports"] = self.counters.get("kda_snapshot_exports", 0) + 1
        return dict(kind=self.KDA_PACK, layers=[L for L, _S, _s0, _C, _r in src], n_s=n_s, S=torch.cat(s_parts, dim=1),
                    C=torch.cat(c_parts, dim=1))  # fmt: skip

    def import_kda_pack(self, kv: StackCache, slot: int, blob: dict) -> None:
        """inverse of export_kda_pack: the blob's S pages and conv rows -> slot's rows (no other slot changes)"""
        c, rd = divmod(int(slot), self.chunk)
        lay = self._kda_layers(kv, c)
        if blob.get("kind") != self.KDA_PACK or list(blob["layers"]) != [L for _i, L, _st in lay]:
            raise ValueError(f"KDA pack for layers {blob.get('layers')} does not match this cache's KDA layers")
        ss, sc = kv.extra["kda_stage_s"], kv.extra["kda_stage_c"]
        n_s = _pages(lay[0][2][0]) // self.chunk
        if int(blob["n_s"]) != n_s:
            raise ValueError(f"KDA pack of {blob['n_s']} pages per slot, this cache has {n_s}")
        ls = int(ss.padded_shape[0]) // n_s
        sa, ca = int(ss.buffer_address()), int(sc.buffer_address())
        S_all, C_all = blob["S"], blob["C"]  # [chips, layers, n_s, 32, 32], [chips, layers, tiles, 32]
        for g in range(0, len(lay), ls):
            grp = lay[g : g + ls]
            host = torch.zeros(S_all.shape[0], ls * n_s, 32, 32, dtype=S_all.dtype)
            host[:, : len(grp) * n_s] = S_all[:, g : g + len(grp)].reshape(S_all.shape[0], len(grp) * n_s, 32, 32)
            self._stage_write(ss, host)
            self.slot_ops.run([(1, sa, j * n_s, int(S.buffer_address()), rd * n_s, n_s, _pages_bytes(S), 0)
                               for j, (_i, _L, (S, _C)) in enumerate(grp)])  # fmt: skip
            self.counters["slot_ops"] += 1
            ttnn.synchronize_device(self.mesh)  # the next round rewrites the staging pages
        for g in range(0, len(lay), 32):
            grp = lay[g : g + 32]
            host = torch.zeros(C_all.shape[0], C_all.shape[2], 32, 32, dtype=C_all.dtype)
            host[:, :, : len(grp), :] = C_all[:, g : g + len(grp)].permute(0, 2, 1, 3)
            self._stage_write(sc, host)
            self.slot_ops.run([(2, ca, 0, int(C.buffer_address()), 0, _pages(C), _pages_bytes(C), (j << 16) | rd)
                               for j, (_i, _L, (_S, C)) in enumerate(grp)])  # fmt: skip
            self.counters["slot_ops"] += 1
            ttnn.synchronize_device(self.mesh)
        self.counters["kda_pack_imports"] = self.counters.get("kda_pack_imports", 0) + 1

    def _dsa_stage_with_cache(self, kv: StackCache) -> None:
        """the DSA export staging allocated WITH the cache (before any capture) when every checkpoint must carry DSA rows:
        per-slot pooled keys (GLM53_DSA_PAGED_POOLS=0: a KDA-only checkpoint would restore zero pooled keys, see
        export_kda_state) or GLM53_DSA_EXPORT_STAGE_TILES set explicitly. Paged pools (the served construction) keep the
        on-demand allocation (ensure_dsa_export: the adapter's KV connector / host tier): their pooled keys ride the pages.
        """
        if not dsa_paged_pools() or os.environ.get("GLM53_DSA_EXPORT_STAGE_TILES"):
            self._ensure_dsa_stage(kv)

    def _ensure_dsa_stage(self, kv: StackCache) -> None:
        """GLM53_DSA_EXPORT_STAGE_TILES (default 256 = 512 KiB per chip; 0: no host export) single-tile pages"""
        if kv.extra.get("dsa_stage") is not None:
            return
        n = int(os.environ.get("GLM53_DSA_EXPORT_STAGE_TILES", "256"))
        if n <= 0:
            return
        kv.extra["dsa_stage"] = ttnn.zeros((n, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                           device=self.mesh, memory_config=DRAM)  # fmt: skip

    def _dsa_rounds(self, kv: StackCache, slot: int, units: Optional[int]) -> list:
        """the copy plan of slot's DSA rows: rounds of [(key, tensor tile0, tiles, staging tile0)], each round filling at
        most the staging buffer; a unit-paged tensor moves its first `units` units (None: every unit of the slot)"""
        cap = int(kv.extra["dsa_stage"].padded_shape[0])
        rounds, cur, used = [], [], 0
        for key, t, rows, paged in self._dsa_tensors(kv):
            if not 0 <= slot < rows:
                raise ValueError(f"DSA export of slot {slot}: {key} has {rows} rows")
            per = _pages(t) // rows
            n = per if not paged or units is None else min(per, 4 * int(units))
            off = 0
            while off < n:
                take = min(n - off, cap - used)
                cur.append((key, slot * per + off, take, used))
                used, off = used + take, off + take
                if used == cap:
                    rounds.append(cur)
                    cur, used = [], 0
        if cur:
            rounds.append(cur)
        return rounds

    def _stage_read(self, stage) -> torch.Tensor:
        """the staging tiles of every chip, [chips, tiles, 32, 32] (each tile row-major: an exact permutation of its
        bytes, restored by _stage_write). Mock (GLM53_MOCK, GLM53_MOCK_STATE_STUB default 1): zeros, no host transfer --
        the plan and the SlotOps calls still run, the bytes are not modelled (a mock has no data to move)."""
        n = int(stage.padded_shape[0])
        if MOCK and os.environ.get("GLM53_MOCK_STATE_STUB", "1") == "1":
            dt = torch.float32 if stage.dtype == ttnn.float32 else torch.bfloat16
            return torch.zeros(self.tp if self.tp > 1 else 1, n, 32, 32, dtype=dt)
        if self.tp > 1:
            t = ttnn.to_torch(stage, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
        else:
            t = ttnn.to_torch(stage)
        return t.reshape(-1, n, 32, 32)

    def _stage_write(self, stage, host: torch.Tensor) -> None:
        """host [chips or 1, tiles, 32, 32] -> the staging tiles (1: the same tiles on every chip); a mock (stub) skips it"""
        n = int(stage.padded_shape[0])
        if MOCK and os.environ.get("GLM53_MOCK_STATE_STUB", "1") == "1":
            return
        if self.tp > 1 and host.shape[0] > 1:
            mapper = ttnn.ShardTensorToMesh(self.mesh, dim=0)
        else:
            mapper = ttnn.ReplicateTensorToMesh(self.mesh) if self.tp > 1 else None
        h = ttnn.from_torch(host.reshape(-1, 1, 32, 32)[: n * host.shape[0]].contiguous(), dtype=stage.dtype,
                            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)  # fmt: skip
        ttnn.copy_host_to_device_tensor(h, stage)

    def export_dsa_state(self, kv: StackCache, slot: int, n_tokens: Optional[int] = None,
                         pools: bool = True) -> Optional[dict]:  # fmt: skip
        """slot's DSA indexer state -> {"kind": DSA_BLOB, "units", "n_tokens", "chips", "replicated", "layout", "pools",
        "tensors": {key: [chips or 1, tiles, 32, 32] bf16}} (None: no DSA state). Units = ceil(n_tokens / 128) of every
        pooled-key store (None: all of the slot's rows), the whole ring. pools=False (Stage 12, GLM53_KV_POOLED_PAGES:
        the pooled keys ride in the LMCache pages): units 0, the blob carries the ring only. Every chip's copy is read; equal copies are kept
        once. GLM53_DSA_SHARD (layout "unit"): a pooled-key store's rows are read from every chip's LOCAL units and
        converted to the CHIP-INDEPENDENT global unit order (unit u = chip u % C, local u // C; served_1m_memory_plan.md
        E, shard_tp8_design.md 6.2), one row [1, 4 U, 32, 32], so the blob restores into a cache of any C; the ring
        (not unit-paged) stays per chip when the chips differ."""
        tens = self._dsa_tensors(kv)
        if not tens:
            return None
        stage = kv.extra.get("dsa_stage")
        if stage is None:
            raise RuntimeError("DSA host export: no staging buffer (GLM53_DSA_EXPORT_STAGE_TILES=0)")
        # GLM53_DSA_PAGED_POOLS: the record carries no pooled keys (they live in the blocks' pages), and says so -- a
        # connector hit's fresh blocks then need the pooled pages (GLM53_KV_POOLED_PAGES) or restart at 0
        # (generator_vllm _states_pools_less); 'pools' True here resumed hits on the blocks' stale keys (n29h7s2)
        pools = bool(pools) and not self.pooled_paged(kv)
        units = 0 if not pools else None if n_tokens is None else -(-int(n_tokens) // self.DSA_UNIT)
        addr = {key: int(t.buffer_address()) for key, t, _r, _p in tens}
        paged = {key: p for key, _t, _r, p in tens}
        parts = {key: [] for key, _t, _r, _p in tens}
        sa = int(stage.buffer_address())
        for rnd in self._dsa_rounds(kv, int(slot), units):
            self.slot_ops.run([(1, addr[k], t0, sa, s0, n, 2048, 0) for k, t0, n, s0 in rnd])
            self.counters["slot_ops"] += 1
            host = self._stage_read(stage)
            for k, _t0, n, s0 in rnd:
                parts[k].append(host[:, s0 : s0 + n].clone())
        n_chips = next((v[0].shape[0] for v in parts.values() if v), 1)
        # a store with no rows to move (pools=False, or units 0) leaves an empty [chips, 0, 32, 32] record
        out = {k: torch.cat(v, dim=1) if v else torch.zeros(n_chips, 0, 32, 32, dtype=torch.bfloat16)
               for k, v in parts.items()}  # fmt: skip
        chips = next(iter(out.values())).shape[0]
        shard = kv.extra.get("dsa_shard")
        if shard is not None:  # per key: unit-paged stores -> global unit order; the rest kept once when equal
            C = int(shard[0])
            for k, v in out.items():
                if paged[k]:
                    out[k] = dsa_units_to_unit_order(v, C, units, self.DSA_TILES_PER_UNIT)
                elif all(torch.equal(v.view(torch.int16)[c], v.view(torch.int16)[0]) for c in range(chips)):
                    out[k] = v[:1].clone()
            same = all(v.shape[0] == 1 for v in out.values())
        else:
            same = all(
                torch.equal(v.view(torch.int16)[c], v.view(torch.int16)[0]) for v in out.values() for c in range(chips)
            )
            if same:
                out = {k: v[:1].clone() for k, v in out.items()}
        self.counters["dsa_exports"] = self.counters.get("dsa_exports", 0) + 1
        self.counters["dsa_export_bytes"] = self.counters.get("dsa_export_bytes", 0) + sum(
            v.numel() * v.element_size() for v in out.values())  # fmt: skip
        self.counters["dsa_export_replicated"] = self.counters.get("dsa_export_replicated", 0) + int(same)
        return dict(kind=self.DSA_BLOB, unit_tokens=self.DSA_UNIT, units=units, n_tokens=n_tokens, chips=chips,
                    replicated=same, layout="unit" if shard is not None else "chip", pools=bool(pools),
                    tensors=out)  # fmt: skip

    def rebuild_window(self, kv: StackCache, slot: int, start: int, blocks) -> int:
        """GLM53_DSA_DENSE_WINDOW under restores (dsa/dense_window_restore_design.md W1): window blocks
        [0, ceil(min(start, DENSE_MAX) / 64)) of `slot` <- the request's restored latent pages, every sparse_dsa layer.
        The window rows ARE latent rows (block j of a slot = the request's latent page j, the same tile layout), so a
        connector hit / prefix attach restores them by copy. `blocks`: the slot's PHYSICAL global blocks per column (the
        sharded pool: block g on chip g % C at local g // C, written there only, so the replicated window gathers each
        block from its owner through the host). SlotOps (compiled before any capture) + the window staging (allocated
        with the cache): no compile, no allocation after a capture; plain DMA between steps, before the row's first
        chunk (which writes rows >= start itself). Returns the blocks rebuilt per layer (0: no window or nothing to do).
        """
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_mla import DSA_DENSE_MAX, DSA_WIN_BLOCKS

        wins = [(i, d["window"]) for i, d in sorted(kv.extra.get("dsa", {}).items()) if d.get("window") is not None]
        stage = kv.extra.get("win_stage")
        if not wins or stage is None or int(start) <= 0:
            return 0
        nb = min(-(-min(int(start), DSA_DENSE_MAX) // 64), DSA_WIN_BLOCKS)
        phys = [int(g) for g in torch.as_tensor(blocks).reshape(-1)[:nb].tolist()]
        if len(phys) < nb or min(phys) < 0:
            raise ValueError(f"rebuild_window slot {slot}: {len(phys)} blocks for {nb} window blocks: {phys}")
        if not 0 <= int(slot) < int(next(iter(kv.extra["dsa"].values()))["window_slots"]):
            raise ValueError(f"rebuild_window: slot {slot} outside the window's seats")
        shard = kv.extra.get("dsa_shard")
        C = int(shard[0]) if shard is not None else 1
        tpb, cap, sa = 32, int(stage.padded_shape[0]), int(stage.buffer_address())
        for i, win in wins:
            la, wa = int(kv.extra["latent"][i].buffer_address()), int(win.buffer_address())
            # every chip copies local block g // C into staging block j; only the owner's copy (chip g % C) is the page
            self.slot_ops.run([(1, la, (g // C) * tpb, sa, j * tpb, tpb, 2048, 0) for j, g in enumerate(phys)])
            host = self._stage_read(stage)  # [chips, cap, 32, 32]
            pick = torch.zeros(1, cap, 32, 32, dtype=host.dtype)
            for j, g in enumerate(phys):
                pick[0, j * tpb : (j + 1) * tpb] = host[(g % C) if host.shape[0] > 1 else 0, j * tpb : (j + 1) * tpb]
            self._stage_write(stage, pick)  # the same pages on every chip
            self.slot_ops.run([(1, sa, 0, wa, int(slot) * DSA_WIN_BLOCKS * tpb, nb * tpb, 2048, 0)])
            ttnn.synchronize_device(self.mesh)  # the next layer rewrites the staging
        self.counters["window_rebuilds"] = self.counters.get("window_rebuilds", 0) + 1
        self.counters["window_rebuild_blocks"] = self.counters.get("window_rebuild_blocks", 0) + nb * len(wins)
        return nb

    def import_dsa_state(self, kv: StackCache, slot: int, blob: dict) -> None:
        """inverse of export_dsa_state: the blob's rows -> slot's rows (rows past the blob's units are left as they are:
        never read before the continuation writes them)"""
        if not isinstance(blob, dict) or blob.get("kind") != self.DSA_BLOB:
            raise TypeError(f"not a DSA export blob: {type(blob).__name__}")
        tens = self._dsa_tensors(kv)
        keys = [key for key, _t, _r, _p in tens]
        if sorted(keys) != sorted(blob["tensors"]):
            raise ValueError(f"DSA blob tensors {sorted(blob['tensors'])} != this cache's {sorted(keys)}")
        stage = kv.extra.get("dsa_stage")
        if stage is None:
            raise RuntimeError("DSA host import: no staging buffer (GLM53_DSA_EXPORT_STAGE_TILES=0)")
        addr = {key: int(t.buffer_address()) for key, t, _r, _p in tens}
        cap, sa = int(stage.padded_shape[0]), int(stage.buffer_address())
        src_all = self._dsa_import_rows(kv, blob, tens)  # {key: [1 or C, tiles, 32, 32]} in THIS cache's chip layout
        chips = max(int(v.shape[0]) for v in src_all.values())
        cursor = dict.fromkeys(keys, 0)
        for rnd in self._dsa_rounds(kv, int(slot), blob["units"]):
            host = torch.zeros(chips, cap, 32, 32, dtype=torch.bfloat16)
            for k, _t0, n, s0 in rnd:
                src = src_all[k][:, cursor[k] : cursor[k] + n]
                if src.shape[1] != n:
                    raise ValueError(f"DSA blob {k}: {cursor[k] + src.shape[1]} tiles, the copy plan needs more")
                host[:, s0 : s0 + n] = src
                cursor[k] += n
            self._stage_write(stage, host)
            self.slot_ops.run([(1, sa, s0, addr[k], t0, n, 2048, 0) for k, t0, n, s0 in rnd])
            self.counters["slot_ops"] += 1
            ttnn.synchronize_device(self.mesh)  # the next round rewrites the staging tiles
        left = {k: int(src_all[k].shape[1]) - cursor[k] for k in keys if int(src_all[k].shape[1]) != cursor[k]}
        if left:
            raise ValueError(f"DSA blob tiles not imported: {left}")
        self.counters["dsa_imports"] = self.counters.get("dsa_imports", 0) + 1

    def _dsa_import_rows(self, kv: StackCache, blob: dict, tens: list) -> dict:
        """the blob's tensors in THIS cache's chip layout: a unit-order ("unit" layout, or a replicated blob whose C = 1
        makes unit order = local order) pooled-key store -> per chip LOCAL units under GLM53_DSA_SHARD (unit u -> chip
        u % C, local u // C), one row on a replicated cache; a per-chip raw dump needs the same C; the ring as it is"""
        shard = kv.extra.get("dsa_shard")
        C = int(shard[0]) if shard is not None else 1
        tpu = self.DSA_TILES_PER_UNIT
        units = blob.get("units")
        out = {}
        for key, t, rows, paged in tens:
            v = blob["tensors"][key]
            if (
                paged
                and int(v.shape[0]) == 1
                and (blob.get("layout", "chip") == "unit" or int(blob["chips"]) == 1 or blob["replicated"])
            ):
                if C > 1:  # unit order -> this cache's local units; tiles per chip = the copy plan's count
                    per = _pages(t) // int(rows)
                    local = per if units is None else min(per, tpu * int(units))
                    v = dsa_unit_order_to_chips(v, C, local, tpu)
            elif int(v.shape[0]) not in (1, C):
                raise ValueError(f"DSA blob {key}: {int(v.shape[0])} chip rows for a cache of {C} chips")
            out[key] = v
        return out

    def export_snapshot(self, kv: StackCache, entry: int, n_tokens: Optional[int] = None) -> Any:
        """the host tier's blob of a snapshot entry (virtual slot max_batch + entry): KDA state + (sparse regime) the
        entry's DSA rows of its first ceil(n_tokens / 128) units"""
        return self.export_kda_state(kv, self._entry_slot(kv, entry), n_tokens=n_tokens)

    def import_snapshot(self, kv: StackCache, entry: int, states: Any) -> None:
        self.import_kda_state(kv, self._entry_slot(kv, entry), states)

    def _state_ops(self, kv: StackCache, dst: int, src: Optional[int], dsa: bool = True) -> list:
        """KDA S rows (contiguous pages per user) and conv-history rows (a row of every tile) of slot dst <- slot src
        (None: zeros), every layer; dsa=False: not the sparse_dsa slot state (Stage 17's speculative hand-off moves
        only the KDA candidate row `sel` -> row 0 of ONE user; that user's pooled keys / ring stay in slot 0 -- copying
        slot sel's would overwrite them with another slot's)"""
        cd, rd = divmod(dst, self.chunk)
        ops = []
        for li in range(len(self.seq)):
            if self.seq[li]["sparse"] == "dsa":
                st = kv.extra.get("dsa", {}).get(li)
                if (
                    st is not None and src is not None and dsa
                ):  # GLM53_STACK_DSA: the slot's pooled keys + ring are per slot
                    # (not paged): they move with the slot; a reset needs nothing (rows past cur_pos are never read)
                    pps, ap, ar = st["tp"] // 32 * 4, st["pools"].buffer_address(), st["ring"].buffer_address()
                    if not st.get("paged"):  # rev 63: paged pooled rows follow the slot's blocks (the page-table remap)
                        ops.append((1, ap, src * pps, ap, dst * pps, pps, 2048, 0))
                    ops.append((1, ar, src * 8, ar, dst * 8, 8, 2048, 0))
                # the latent cache: nothing to reset (rows past cur_pos are never read) and nothing to copy — the
                # pages follow the page-table row, which Glm53Generator.remap_slots moves after copy_slot
                # (ModelBackend.copy_slot contract); the destination's row still lists its previous owner's blocks
                # then, which vLLM may have given to another live request
                continue
            Sd, Cd = kv.states[cd][li]
            n_s = _pages(Sd) // _rows(Sd)  # user rows are the outer dim of S [rows, H, 128, 128]: contiguous tiles (the
            # pool's last chunk may hold fewer rows than a chunk; every chunk has the same pages per user)
            if src is None:
                ops.append((0, 0, 0, Sd.buffer_address(), rd * n_s, n_s, 4096, 0))
                ops.append((2, 0, 0, Cd.buffer_address(), 0, _pages(Cd), 2048, rd))
            else:
                cs, rs = divmod(src, self.chunk)
                Ss, Cs = kv.states[cs][li]
                assert _pages(Ss) // _rows(Ss) == n_s and _pages(Cs) == _pages(Cd), (li, src, dst)
                ops.append((1, Ss.buffer_address(), rs * n_s, Sd.buffer_address(), rd * n_s, n_s, 4096, 0))
                ops.append((2, Cs.buffer_address(), 0, Cd.buffer_address(), 0, _pages(Cd), 2048, (rs << 16) | rd))
        return ops

    def prepare_prefill(self, kv) -> None:
        """GLM53_STACK_PREFILL=tp (Stage 16): the batched TP prefill (tt/prefill_tp.py) -- every bucket compiled here (one
        eager forward each, before any capture); its traces are captured with the full-width decode trace"""
        self._kv = kv
        if os.environ.get("GLM53_STACK_PREFILL", "decode") == "tp" and getattr(self, "tpp", None) is None:
            from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import TPPrefill

            # default 256 / 512 / 2,048 (coordinator 2026-10-22 15:50Z): every bucket sequence-parallel at TP8 (256 = 32 x 8),
            # so a prompt's bucket MODE never flips between its solo and its grouped (packed / batched) admission
            spec = os.environ.get("GLM53_PREFILL_BUCKETS", "256,512,2048")
            self.tpp = TPPrefill(self, kv, buckets=tuple(int(v) for v in spec.split(",")))
            if self.tpp.has_dsa() and os.environ.get("GLM53_DSA_EXPORT_STAGE_TILES"):
                self._ensure_dsa_stage(kv)  # explicit: the DSA host export's staging, before any capture
            # the generator refuses continuation starts the prefill's SDPA cannot take (q 128: 128-aligned chunks; the
            # vLLM plugin's tt.prefill_chunk_alignment is 128 in the served config)
            self.continuation_align = max(BLOCK_SIZE, self.tpp.start_align)
            # PREFILL MEGAKERNEL stack (tt/pmk_stack.py, Stage 16 B5): GLM53_PMK_LAYERS = n -> layers [0, n) of every
            # planned bucket run as persistent programs inside the bucket trace (built here, inside the backend build
            # window, so a resident image replays its uploads; the compile launches below are the programs' first
            # launches, before any capture). n = 0 (default): nothing is installed, the op-graph path is unchanged.
            n_pmk = int(os.environ.get("GLM53_PMK_LAYERS", "0") or 0)
            # GLM53_PF_MODE=pmk (Stage 9's serving switch, pmk/stack README §11): every planned bucket's trace runs
            # the layers as the pieces' program sequence (tt/pmk_stack.ServedLayer: A1..A5 + program B + op fillers
            # for the slots no piece has delivered) on the persistent residual; GLM53_PMK_LAYERS = n restricts it to
            # the first n layers (0 = all); GLM53_PMK_SLOTS picks the slots. "op" (default): the op graph, unchanged.
            mode = os.environ.get("GLM53_PF_MODE", "op")
            assert mode in ("op", "pmk"), f"GLM53_PF_MODE={mode}: op | pmk"
            if mode == "pmk":
                from models.autoports.zai_org_glm_5_3_flash.tt.pmk_stack import PmkStack, slot_modes_from_env

                n = min(n_pmk, len(self.tpp.layers)) if n_pmk > 0 else len(self.tpp.layers)
                opts = dict(
                    pfs_park=int(os.environ.get("GLM53_PMK_A_PARK", "4")),
                    router_in=os.environ.get("GLM53_PMK_ROUTER_IN", "fp32"),
                )  # program B: b_config (the record + GLM53_PMK_B_*)
                self.pmk = PmkStack(self.tpp, n, impl="served", slots=slot_modes_from_env(), opts=opts,
                                    probe=os.environ.get("GLM53_PMK_PROBE", "0") == "1").attach()  # fmt: skip
                self.counters["pmk_layers"] = n
                self.counters["pmk_slots"] = dict(self.pmk.slots)
                self.counters["pmk_signature"] = self.pmk.signature()["signature"]  # Stage 12's image window key
            elif n_pmk > 0:  # the family-skeleton driver (the first stack mock; unchanged)
                from models.autoports.zai_org_glm_5_3_flash.tt.pmk_stack import PmkStack

                self.pmk = PmkStack(self.tpp, min(n_pmk, len(self.tpp.layers))).attach()
            self.prefill_state_io = self.tpp.sio is not None  # attach / snapshot ride inside the prefill trace
            self._prep_dram("before TPPrefill.compile")
            self.tpp.compile()
            self._prep_dram("after TPPrefill.compile")
            # the eager compile forwards leave cyclic Python garbage that still holds device buffers (cf3h-j image mock
            # 2010, 9 2026-10-25 20:20Z: ~40 MB/bank, and the W1 group compile's s4 'mb' reduce_scatter then found
            # 7.5 MB free / 5.6 MB largest per bank for 8 MiB): collect it before the group compile
            import gc  # noqa: PLC0415

            gc.collect()
            self._prep_dram("after gc")
            self._compile_groups()  # W1: the prewarmed group keys' compile launches, before any capture
            if getattr(self, "_group_keys", None):
                self._prep_dram("after the W1 group compile")

    def _prep_dram(self, stage: str) -> None:
        """one PREP_DRAM line: DRAM allocated / free / largest free block per bank (lock-step: chip 0 = every chip) after
        a prepare_prefill stage (9's cf3h-j DRAM fit at the W1 group compile, 2026-10-25)"""
        try:
            v = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)
            mb = lambda b: f"{int(b) / 2**20:.1f}"  # noqa: E731
            print(f"PREP_DRAM {stage}: alloc {mb(v.total_bytes_allocated_per_bank)} / free {mb(v.total_bytes_free_per_bank)} / "
                  f"largest free {mb(v.largest_contiguous_bytes_free_per_bank)} MiB per bank", flush=True)  # fmt: skip
        except Exception as e:  # noqa: BLE001  (a diagnostic never stops the build)
            print(f"PREP_DRAM {stage}: no DRAM view ({type(e).__name__}: {e})", flush=True)

    def _step(self, kv: StackCache, chunks=None, width=None):
        """the decode step (eager here; the same calls run inside the captured trace): embed -> the layers of the row
        chunks `chunks` (default all) -> head. A subset serves the low widths (rows 0..w-1: chunks 0..w/chunk-1) and
        one slot's prefill (its own chunk); rows of chunks that do not run keep their states / caches untouched (their
        head rows are never read: inactive)."""
        if width is not None and int(width) in self.narrow_widths:
            self._step_narrow(kv, int(width))
            return
        self.embed(self.head.topk_out, kv.res)
        if width is not None and width in kv.extra.get("wstacks", {}):
            self._step_width(kv, int(width))
            if os.environ.get("GLM53_STACK_WIDTH_HEAD", "1") == "1":  # the head over rows 0..w-1 only
                self._width_head(int(width))([kv.extra["wres"][int(width)]], self.pos, pos_copies=self.pos_c)
            else:  # the 32-row head (its top-K scan costs every row: 3.7 ms at TP8 B=32 vs ~1 ms at t = 8)
                self.head(kv.res, self.pos, pos_copies=self.pos_c)
            return
        for c in range(self.n_chunks) if chunks is None else chunks:
            for r, stk in kv.stacks[c]:
                if stk is None:  # sparse_dsa: 7b's MkMla (stack mode: reads / keeps the persistent residual) + MkMoE
                    e = self.seq[r[0]]
                    if self.fl is not None:
                        set_knobs("dsa", self.tp, self.chunk)
                    s1 = e["mla"](kv.res[c], kv.extra["latent"][r[0]], kv.extra["pt_c"][c], self.pos_c[c], t=self.chunk,
                                  keep=("s1",), stack=True,
                                  dsa_state=self._dsa_st(kv, r[0], c * self.chunk))["s1"]  # fmt: skip
                    e["moe"](s1, t=self.chunk, out=kv.res[c], live=self._moe_live())
                    # freed explicitly (not just dropped): an s1 still referenced elsewhere stays allocated after a
                    # capture -> buffers alive at the replay of an earlier-captured width trace
                    ttnn.deallocate(s1)
                    del s1  # L1: must not stay allocated under the next stack program's CB region
                    continue
                if self.fl is not None:  # the TP layers were built with the per-kind knobs (one chip: by the caller)
                    set_knobs(self.seq[r[0]]["sparse"], self.tp, self.chunk)
                stk([kv.states[c][i][1] for i in r], kv.masks[c], [kv.states[c][i][0] for i in r])
        self.head(kv.res, self.pos, pos_copies=self.pos_c)

    def _dsa_st(self, kv: StackCache, i: int, s0: int):
        """sparse_dsa layer i's DSA state for a launch on slots s0.. (GLM53_STACK_DSA; None: the dense MkMla)"""
        st = kv.extra.get("dsa", {}).get(i)
        return None if st is None else dict(st, s0=s0)

    def _width_head(self, w: int):
        """MkHead.width_view of the 32-row head for native width w (created at the width's first, eager step, after
        the 32-row head's first call: its h / outputs exist and are viewed)"""
        from models.autoports.zai_org_glm_5_3_flash.tt import mk_head as MKH

        hd = self.wheads.get(w)
        if hd is None:
            hd = self.wheads[w] = MKH.MkHead.width_view(self.head, w, _rows_view)
        return hd

    def _step_width(self, kv: StackCache, w: int):
        """the layers at native width w (rows 0..w-1) on the shared 32-row state; the embed / head stay 32-row (rows
        >= w are idle: never read)"""
        rw, masks, pool = kv.extra["wres"][w], kv.extra["wmasks"][w], kv.extra["pool"]
        pool.alias = True  # a width stack's first call builds its program: its intermediates alias the full width's
        try:
            for r, stk in kv.extra["wstacks"][w]:
                if stk is None:
                    e = self.seq[r[0]]
                    if self.fl is not None:
                        set_knobs("dsa", self.tp, w)
                    for o in (getattr(e["mla"], "mk", None), getattr(e["moe"], "mk", None)):
                        if o is not None:  # the full width's buffers exist (prepare_decode ran it first)
                            _seed_width(o, w, self.chunk)
                    s1 = e["mla"](rw, kv.extra["latent"][r[0]], kv.extra["pt_c"][0], self.pos_c[0], t=w, keep=("s1",),
                                  stack=True, dsa_state=self._dsa_st(kv, r[0], 0))["s1"]  # fmt: skip
                    e["moe"](s1, t=w, out=rw, live=self._moe_live())
                    ttnn.deallocate(s1)
                    del s1
                    continue
                if self.fl is not None:
                    set_knobs(self.wseq[w][r[0]]["sparse"], self.tp, w)
                stk([kv.states[0][i][1] for i in r], masks, [kv.states[0][i][0] for i in r])
        finally:
            pool.alias = False

    def prepare_decode(self, io: DecodeIO, kv: StackCache, widths: Sequence[int]) -> None:
        """compile every program (one eager step) before any capture, then restore the request state it touched"""
        self._kv = kv
        tok = self.read(io.tokens)
        pos = self.read(io.cur_pos)
        self._step(kv)
        if kv.extra.get("tee") is not None:  # compiled before any capture (ctl holds no pairs: reads ctl, returns)
            kv.extra["tee"].run()
        for w in self.native_widths + self.narrow_widths:
            self._step(kv, width=w)
        if getattr(self, "spec", None) is not None:  # Stage 17: every spec program compiled before the first capture
            self.spec.prepare()
        ttnn.synchronize_device(self.mesh)
        self.reset_all(kv)
        self.write(io.tokens, tok)
        self.write(io.cur_pos, pos)

    def _width_chunks(self, width: int):
        return range(min(self.n_chunks, -(-int(width) // self.chunk)))

    def decode(self, io: DecodeIO, kv: StackCache, width: int) -> None:
        self._kv = kv
        if int(width) in self.native_widths or int(width) in self.narrow_widths:
            self._step(kv, width=int(width))
        else:
            self._step(kv, self._width_chunks(width))
        if kv.extra.get("tee") is not None:
            kv.extra["tee"].run()
        self.counters["decode_eager"] += 1

    def sample(self, io: DecodeIO, mode: str) -> None:
        pass  # the decode step samples (samples_in_decode_trace)

    def capture_decode(self, io: DecodeIO, kv: StackCache, width: int) -> Any:
        """the width-w decode trace (the chunks covering rows 0..w-1); the full width's is also kv.extra["decode_trace"]
        and, with several chunks, its capture adds one PREFILL trace per chunk (kv.extra["prefill_traces"][c]: that
        chunk alone, used by prefill_chunk for a slot of chunk c)"""
        self._kv = kv
        tpp = getattr(self, "tpp", None)
        if tpp is not None and tpp.kv is kv and not tpp._warmed and not tpp.traces:
            # Stage 16: every prefill key's eager forward BEFORE the first trace capture (the decode traces included):
            # a program first compiled after a capture allocates a buffer of its own that stays alive in that trace's
            # memory (02:04Z leak_micro on the pair; the TP8 128-bucket 'alive before replay' at 00:32Z)
            tpp.warm()
            self._warm_groups()  # W1: the prewarmed group keys too, under the same capture-time device state
        self._capture_spec()
        if int(width) in self.native_widths or int(width) in self.narrow_widths:  # its own programs, shared state
            if (
                not MOCK
                and os.environ.get("GLM53_PF_CAPTURE_REPORT", "0") == "1"
                and getattr(self, "_cap_view", None) is None
            ):
                from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import trace_view

                self._cap_view = trace_view(self.mesh)  # the baseline before the first decode trace
            if MOCK and _mock_capture():  # the capture's allocations, in the device's order (tt/mock_dram.py)
                self._mock_capture_step(kv, None, int(width))
            tr = ("mock", kv, None, int(width)) if MOCK else self._capture(kv, None, width=int(width))
            kv.extra.setdefault("decode_traces", {})[int(width)] = tr
            if not MOCK and os.environ.get("GLM53_PF_CAPTURE_REPORT", "0") == "1":  # the per-trace size table
                from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import report_capture

                self._cap_view = report_capture(self.mesh, f"decode w{int(width)}", getattr(self, "_cap_view", None))
            # Stage 9 (n29h7s2, 2026-10-23): a launch whose vLLM seats stop at the largest native width (16 seats on
            # the 32-row chunk) never captures the full width, so the TP-prefill traces are captured here, after that
            # width's decode trace (the full-width path below does the same; capture() skips captured keys)
            if (tpp is not None and tpp.kv is kv and self.native_widths and int(width) == max(self.native_widths)
                    and not tpp.traces):  # fmt: skip
                tpp.capture()
                self._capture_groups()
                if os.environ.get("GLM53_L1_REPORT") == "1":  # Stage 9 cf3f: what the TP-prefill captures left in L1
                    l1_report(self.mesh, "after TP-prefill capture", root=self)
            return tr
        chunks = self._width_chunks(width)
        full = len(chunks) == self.n_chunks
        if MOCK:
            if _mock_capture():
                self._mock_capture_step(kv, chunks)
            tr = ("mock", kv, chunks)
        else:
            tr = self._capture(kv, chunks)
        kv.extra.setdefault("decode_traces", {})[int(width)] = tr
        if full:
            kv.extra["decode_trace"] = tr
            old = kv.extra.get("prefill_traces") or []
            if old and not MOCK and any(t not in self.traces for t in old):  # released with the generator's traces
                for t in old:
                    if t in self.traces:
                        self.release(t)
                kv.extra.pop("prefill_traces")
            if self.n_chunks > 1 and self.prefill_traces and "prefill_traces" not in kv.extra:
                if MOCK and _mock_capture():
                    for c in range(self.n_chunks):
                        self._mock_capture_step(kv, [c])
                kv.extra["prefill_traces"] = [("mock", kv, [c]) if MOCK else self._capture(kv, [c])
                                              for c in range(self.n_chunks)]  # fmt: skip
            if getattr(self, "tpp", None) is not None and self.tpp.kv is kv:
                self.tpp.capture()  # Stage 16: one batched-prefill trace per bucket (after every compile)
                self._capture_groups()
        return tr

    def _capture_spec(self):
        """Stage 17 cf3f (coordinator 2026-10-23T18:11:19Z): the speculative traces (SpecMTP.capture: the lone step /
        boot and every multi-user layout's) are captured at the FIRST decode capture -- after the prefill keys' eager
        warm (no program compiles after a capture), ahead of every decode trace and every prefill key. Captured lazily
        at the first speculative request (the old order), their programs compiled after every other capture and met an
        L1 buffer allocated after the decode captures (n29h7s2 cf3e run 1756: a static-CB clash at 1,242,240 on cores
        0-0..2-0, TPMoE at the verify width). Idempotent (capture() returns the captured set)."""
        spec = getattr(self, "spec", None)
        if spec is None or spec.traces:
            return
        if os.environ.get("GLM53_L1_REPORT") == "1":
            l1_report(self.mesh, "spec capture (startup)", root=self)
        view = None
        if not MOCK and os.environ.get("GLM53_PF_CAPTURE_REPORT", "0") == "1":
            from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import trace_view

            view = trace_view(self.mesh)
        spec.capture()
        if view is not None:  # the per-trace size table: the spec set as one entry (the decode baseline follows it)
            from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import report_capture

            report_capture(self.mesh, "spec (step / boot / layouts)", view)

    def _mock_capture_step(self, kv: StackCache, chunks, width=None):
        """GLM53_MOCK_CAPTURE: what _capture records, run once on the mock for its allocations (no trace)"""
        if os.environ.get("GLM53_L1_REPORT") == "1":
            l1_report(self.mesh, f"mock capture width={width} chunks={chunks}", root=self)
        self._step(kv, chunks, width)
        if kv.extra.get("tee") is not None:
            kv.extra["tee"].run()

    def _capture(self, kv: StackCache, chunks, width=None):
        if os.environ.get("GLM53_L1_REPORT") == "1":
            l1_report(self.mesh, f"capture width={width} chunks={chunks}", root=self)
        before = _live_device_tensors() if os.environ.get("GLM53_TRACE_LEAK_DEBUG") == "1" else None
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        try:
            self._step(kv, chunks, width)
            if kv.extra.get("tee") is not None:
                kv.extra["tee"].run()
        finally:
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        if before is not None:  # tensors created inside the capture and still referenced after it (they would be
            # overwritten by the replay of an earlier trace): shape, dtype and who holds them
            import gc

            gc.collect()
            for t in _live_device_tensors().values():
                if id(t) in before:
                    continue
                holders = [type(r).__name__ + (f"[{list(r.keys())[:6]}]" if isinstance(r, dict) else "")
                           for r in gc.get_referrers(t) if not isinstance(r, list) or len(r) < 64][:4]  # fmt: skip
                print(f"TRACE_LEAK chunks={list(chunks)} {tuple(t.shape)} {t.dtype} held by {holders}", flush=True)
        self.traces.append(tid)
        self.counters["captures"] += 1
        return tid

    def capture_sampling(self, io: DecodeIO, mode: str) -> Any:
        raise NotImplementedError("samples_in_decode_trace: the decode trace samples")

    def replay(self, trace) -> None:
        if isinstance(trace, tuple):  # mock
            self._step(trace[1], trace[2] if len(trace) > 2 else None, trace[3] if len(trace) > 3 else None)
        else:
            ttnn.execute_trace(self.mesh, trace, cq_id=0, blocking=False)
        self.counters["decode_replays"] += 1

    def release(self, trace) -> None:
        if isinstance(trace, tuple) or trace not in self.traces:  # mock, or already released (with its decode trace)
            return
        ttnn.release_trace(self.mesh, trace)
        self.traces.remove(trace)
        kv = getattr(self, "_kv", None)
        if kv is not None and kv.extra.get("decode_trace") == trace:  # the full width's: its prefill traces go too
            for t in kv.extra.pop("prefill_traces", None) or []:
                if t in self.traces:
                    ttnn.release_trace(self.mesh, t)
                    self.traces.remove(t)
            kv.extra.pop("decode_trace", None)

    def prefill_chunk(self, tokens: torch.Tensor, *, kv_cache: StackCache, slot: int, chunk_start: int, n_valid: int,
                      logits_rows: str, sample=None, io: Optional[DecodeIO] = None, state_src: Optional[int] = None,
                      state_snap: Optional[int] = None, snap_at: Optional[int] = None):  # fmt: skip
        """prefill as decode steps of slot `slot` alone (every other row inactive: their state, positions and token rows
        are restored afterwards) -> the logits of the chunk's last position ([1, V], logits_rows "last"), of every
        position ([n, V], "all"), nothing ("none"), or with `sample` the device-sampled token of the last position
        (left in the slot's decode token row)"""
        io = io or kv_cache.extra["io"]
        self._kv = kv_cache
        tpp = getattr(self, "tpp", None)
        if tpp is not None and tpp.kv is kv_cache and logits_rows != "all":
            return self._prefill_chunk_tp(tokens, kv_cache=kv_cache, slot=slot, chunk_start=chunk_start,
                                          n_valid=n_valid, logits_rows=logits_rows, sample=sample, io=io,
                                          state_src=state_src, state_snap=state_snap, snap_at=snap_at)  # fmt: skip
        if snap_at is not None:
            raise ValueError("a KDA snapshot at a cut needs the TP prefill (GLM53_PF_KDA_SNAPSHOT)")
        # the decode-step prefill has no in-trace state I/O: a prefix attach is a plain DMA before the chunk, a
        # snapshot one after it (the generator does both itself while prefill_state_io is False; the "all" logits
        # path of a TP backend lands here too, so they are honoured here as well)
        if state_src is not None:
            assert chunk_start > 0, "an attach continues a cached prefix"
            self.restore_slot(kv_cache, int(state_src), slot, upto=chunk_start)
        elif chunk_start == 0:
            self.reset_slot(kv_cache, slot)
        if state_snap is not None:
            out = self._prefill_chunk_steps(tokens, kv_cache=kv_cache, slot=slot, chunk_start=chunk_start,
                                            n_valid=n_valid, logits_rows=logits_rows, sample=sample, io=io)  # fmt: skip
            self.snapshot_slot(kv_cache, slot, int(state_snap), upto=chunk_start + n_valid)
            return out
        return self._prefill_chunk_steps(tokens, kv_cache=kv_cache, slot=slot, chunk_start=chunk_start,
                                         n_valid=n_valid, logits_rows=logits_rows, sample=sample, io=io)  # fmt: skip

    def _prefill_chunk_steps(self, tokens, *, kv_cache, slot, chunk_start, n_valid, logits_rows, sample, io):
        """the decode-step prefill proper (one decode replay per token of the chunk)"""
        keep_tok, keep_pos = self.read(io.tokens), self.read(io.cur_pos)
        # every prefill replay rewrites ALL rows of the readback copy too: a slot whose next input lives only there (an
        # async decode step not read yet) would lose it (Stage 9 M6 review) -> put the other rows back afterwards
        keep_samp = self.read(io.sampled)
        rows = []
        # the device sampler during a prefill: the slot's row draws ONCE, at the last position and with `sample`'s
        # params (the prompt steps before it stay greedy: no RNG draw for a discarded token); every other row keeps
        # its params; the generator's page is restored after the chunk (16:30Z: a sampled request's first token used
        # whatever the slot's params row held from the previous request)
        samp_rows = None
        if self.sample:
            samp_rows = {slot: (1, 0.0, 1.0)}
            self._write_sampler_params(override=samp_rows)
        for i in range(n_valid):
            if samp_rows is not None and sample is not None and i == n_valid - 1:
                samp_rows[slot] = tuple(sample.device_values(self.head.topk))
                self._write_sampler_params(override=samp_rows)
            tok = keep_tok.clone()
            tok[slot] = int(tokens.reshape(-1)[i])
            pos = torch.full_like(keep_pos, -1)
            pos[slot] = chunk_start + i
            self.write(io.tokens, tok)
            self.write(io.cur_pos, pos)
            pts = kv_cache.extra.get("prefill_traces")
            trace = pts[slot // self.chunk] if pts else kv_cache.extra.get("decode_trace")
            wts = kv_cache.extra.get("decode_traces", {})
            nw = [w for w in self.native_widths + self.narrow_widths if w > slot and w in wts]
            if nw:  # a native width covering the slot: the prompt runs at w rows, not max_batch
                trace = wts[min(nw)]
            if trace is not None:
                self.replay(trace)
            else:
                self._step(kv_cache)
            if logits_rows == "all" or (logits_rows == "last" and i == n_valid - 1 and sample is None):
                rows.append(self.read(io.logits)[slot : slot + 1])
            self.counters["prefill_tokens"] += 1
        first = self.read(io.tokens)[slot]
        tok = keep_tok.clone()
        if sample is not None:
            tok[slot] = first  # the device's greedy sample of the last position = the slot's next input
        self.write(io.tokens, tok)
        samp = keep_samp.clone()
        if sample is not None:
            samp[slot] = first
        self.write(io.sampled, samp)
        self.write(io.cur_pos, keep_pos)  # the generator owns the positions (its shadow == the device copy)
        if samp_rows is not None:
            self._write_sampler_params()  # back to the generator's params view
        if sample is not None:
            return int(first)  # the device sampler's token of the last position, in the slot's decode token row
        if logits_rows == "all":
            return torch.cat(rows)
        return rows[-1] if rows else None

    def _prefill_chunk_tp(self, tokens, *, kv_cache, slot, chunk_start, n_valid, logits_rows, sample, io,
                          state_src=None, state_snap=None, snap_at=None, group=None):  # fmt: skip
        """Stage 16 batched prefill of one chunk (tt/prefill_tp.py: ONE trace replay for all layers + the head on the
        chunk's last valid row). The head runs with only `slot` active at the last position; the other rows' tokens /
        positions / sampler params are restored afterwards (as the decode-step prefill does). A fresh request needs no
        reset_slot: the trace zeroes the KDA scratch (state_io, fresh flag) and writes it back into the slot."""
        tm = [("start", _time.perf_counter())] if os.environ.get("GLM53_PF_HOST_TIMING") == "1" else None
        if chunk_start == 0 and self.tpp.sio is None:
            self.reset_slot(kv_cache, slot)
        # W1 (prefill_group): `group` = a pmk_stack.GroupStack and n_valid its chunks' valid rows -- the same flow, the
        # group's ONE replay instead of the chunk's (the head on the group's last valid row)
        n_tot = int(sum(n_valid)) if group is not None else int(n_valid)

        def run(head: bool):
            if group is not None:  # a group never carries a store-A cut (the generator runs that chunk alone)
                assert snap_at is None, "a KDA snapshot cut inside a prefill group"
                return group.prefill(tokens, slot=slot, chunk_start=chunk_start, n_valid=list(n_valid), head=head,
                                     state_src=state_src, state_snap=state_snap)  # fmt: skip
            return self.tpp.prefill(tokens, slot=slot, chunk_start=chunk_start, n_valid=n_valid, head=head,
                                    state_src=state_src, state_snap=state_snap, snap_at=snap_at)  # fmt: skip

        nohead_ok = (group.ready(group.windows(chunk_start, list(n_valid)), head=False) if group is not None
                     else self.tpp.nohead_ready(chunk_start, n_valid))  # fmt: skip
        if logits_rows == "none" and sample is None and nohead_ok:
            # a non-final chunk (the generator asks nothing of it): the head-less trace leaves the decode rows alone ->
            # no reads, waits or restores; the replay is only enqueued (the host runs ahead to the next chunk's inputs)
            run(False)
            self.counters["prefill_tokens"] += n_tot
            self.counters["prefill_chunks_tp"] = self.counters.get("prefill_chunks_tp", 0) + 1
            self.counters["prefill_chunks_nohead"] = self.counters.get("prefill_chunks_nohead", 0) + 1
            return None
        keep_tok, keep_pos = self.read(io.tokens), self.read(io.cur_pos)
        # the prefill trace's head rewrites EVERY row of the sampled readback copy too: another slot's next input that
        # lives only there (an async decode step not read yet) would be lost -> the other rows go back afterwards (as
        # the decode-step prefill does; Stage 16 interim review 2026-10-21)
        keep_samp = self.read(io.sampled)
        tm is not None and tm.append(("reads", _time.perf_counter()))
        samp_rows = None
        if self.sample:  # the slot draws once, at the last position, with the request's params
            samp_rows = {slot: tuple(sample.device_values(self.head.topk)) if sample is not None else (1, 0.0, 1.0)}
            self._write_sampler_params(override=samp_rows)
        tm is not None and tm.append(("sampler", _time.perf_counter()))
        pos = torch.full_like(keep_pos, -1)
        pos[slot] = chunk_start + n_tot - 1
        self.write(io.cur_pos, pos)
        tm is not None and tm.append(("cur_pos", _time.perf_counter()))
        run(True)
        tm is not None and tm.append(("prefill_host", _time.perf_counter()))
        self.counters["prefill_tokens"] += n_tot
        self.counters["prefill_chunks_tp"] = self.counters.get("prefill_chunks_tp", 0) + 1
        out = None
        if logits_rows == "last" and sample is None:
            out = self.read(io.logits)[slot : slot + 1]
        first = self.read(io.tokens)[slot]
        tm is not None and tm.append(("first_wait", _time.perf_counter()))  # includes the device replay
        # restore the other rows (a B = 1 backend has none: its row 0 already holds the head's own token / sample, and
        # the sampler row equals the request's params -- 3 host -> device copies saved per request, ~4.5 ms at TP2)
        lone = int(self.max_batch) == 1 and sample is not None and slot == 0
        if not lone:
            tok = keep_tok.clone()
            if sample is not None:
                tok[slot] = first
            self.write(io.tokens, tok)
            samp = keep_samp.clone()
            if sample is not None:
                samp[slot] = first
            self.write(io.sampled, samp)
        self.write(io.cur_pos, keep_pos)
        if samp_rows is not None and not (lone and self._params_match(slot, samp_rows[slot])):
            self._write_sampler_params()
        if tm is not None:  # GLM53_PF_HOST_TIMING: per-step host ms accumulated over requests (Stage 16 TTFT budget)
            tm.append(("restore", _time.perf_counter()))
            acc = self.counters.setdefault("pf_host_ms", {})
            for (_, a), (lab, b) in zip(tm, tm[1:]):
                acc[lab] = acc.get(lab, 0.0) + (b - a) * 1e3
            acc["n"] = acc.get("n", 0) + 1
        if sample is not None:
            return int(first)
        return out

    def _group_prewarm_keys(self) -> list:
        """GLM53_PF_GROUP_PREWARM = "n@start,..." (W1 serving; with GLM53_PF_GROUP_MAX): the group keys to compile at
        prepare_prefill and capture with the TP-prefill traces -- group size n from chunk start `start`, both the head
        and the head-less trace; e.g. "16@0,16@32768" = the first two 16-chunk groups of a long prompt"""
        spec = os.environ.get("GLM53_PF_GROUP_PREWARM", "")
        if int(os.environ.get("GLM53_PF_GROUP_MAX", "0") or 0) < 2 or not spec:
            return []
        out = []
        for item in spec.split(","):
            n_, start_ = (int(v) for v in item.strip().split("@"))
            if not self.prefill_group_ok(n_):
                continue
            g = self.prefill_group_obj(n_)
            ts = g.windows(start_, [g.c] * n_)
            out += [(n_, start_, ts, True), (n_, start_, ts, False)]
        return out

    def _compile_groups(self):
        """the prewarmed group keys' eager forwards = their programs' compile launches (inside prepare_prefill, after
        tpp.compile, before any capture; slot 0 with zero tokens, as TPPrefill.compile)"""
        self._group_keys = self._group_prewarm_keys()
        for n_, start_, ts, head in self._group_keys:
            g = self.prefill_group_obj(n_)
            g.write(torch.zeros(n_ * g.c, dtype=torch.int64), slot=0, chunk_start=start_, n_valid=[g.c] * n_)
            g.forward(ts, head=head)
            self.counters["prefill_group_compiles"] = self.counters.get("prefill_group_compiles", 0) + 1

    def _warm_groups(self):
        """the prewarmed group keys' eager forwards AGAIN, under the device state the captures will see (called with
        tpp.warm in capture_decode, before the first trace capture). The decode preparation between prepare_prefill and
        the captures re-keys some programs (tpp.warm re-made +4 at cf3h-j); _compile_groups ran before it, so a group
        capture hit a program not in the cache: the EP layers' moe_fused_swiglu prologue (pad + to_layout at the
        group's n x C rows) -- cf3h-j d0 op-arm capture, 2026-10-25 18:05Z, TT_FATAL 'Cannot load new binaries during
        trace capture'. Same rule and env as TPPrefill.warm (GLM53_PF_WARM_CAPTURE; the mock only with
        GLM53_MOCK_PF_WARM=1); overwrites slot 0's state (warm-up only), as _compile_groups does."""
        mock_warm = os.environ.get("GLM53_MOCK") == "1" and os.environ.get("GLM53_MOCK_PF_WARM") == "1"
        if (os.environ.get("GLM53_MOCK") == "1" and not mock_warm) or os.environ.get(
            "GLM53_PF_WARM_CAPTURE", "1"
        ) != "1":
            return
        for n_, start_, ts, head in getattr(self, "_group_keys", []):
            g = self.prefill_group_obj(n_)
            if g.ready(ts, head=head):  # captured already: nothing may compile any more
                continue
            g.write(torch.zeros(n_ * g.c, dtype=torch.int64), slot=0, chunk_start=start_, n_valid=[g.c] * n_)
            g.forward(ts, head=head)
            self.counters["prefill_group_warms"] = self.counters.get("prefill_group_warms", 0) + 1

    def _capture_groups(self):
        """capture the prewarmed group keys (with the TP-prefill traces, after every compile)"""
        for n_, _start, ts, head in getattr(self, "_group_keys", []):
            self.prefill_group_obj(n_).capture(ts, head=head)
            self.counters["prefill_group_captures"] = self.counters.get("prefill_group_captures", 0) + 1

    def prefill_group_ok(self, n: int) -> bool:
        """W1: a layer-major group of n chunks needs the TP prefill with the served PMK chain attached (the op arm =
        GLM53_PMK_SLOTS=none + opts group_op_ffn; the hybrid = the PMK attention slots + the op FFN grouped)"""
        pmk = getattr(self, "pmk", None)
        return (
            n >= 2
            and getattr(self, "tpp", None) is not None
            and pmk is not None
            and getattr(pmk, "impl", "") == "served"
        )

    def prefill_group_obj(self, n: int):
        """the cached pmk_stack.GroupStack of n chunks over the attached served chain (built on first use: its
        transients / per-chunk residuals are allocated then; prewarm with .prepare([(windows, head), ..]))"""
        from models.autoports.zai_org_glm_5_3_flash.tt import pmk_stack as PS  # noqa: PLC0415

        groups = self.__dict__.setdefault("_pf_groups", {})
        if (
            self.__dict__.get("_pf_groups_owner") is not self.pmk
        ):  # a new PMK chain (re-attach / another arm): its own groups
            for g_ in groups.values():
                g_.release()
            groups.clear()
            self._pf_groups_owner = self.pmk
        if n not in groups:
            groups[n] = PS.GroupStack(self.pmk, n, PS.V1_BUCKET)  # the group's chunk bucket: the family engine's 2,048
        return groups[n]

    def prefill_group(self, tokens: torch.Tensor, *, kv_cache, slot: int, chunk_start: int, n_valid: list,
                      logits_rows: str, sample=None, io=None, state_src=None, state_snap=None):  # fmt: skip
        """W1 (design/w1_grpmax_design.md §8): n consecutive chunks of ONE request in ONE layer-major replay (the MoE /
        FFN of every grouped layer once over the n chunks). n_valid: each chunk's valid rows (all full but the last).
        Returns NotImplemented (the caller runs the chunks one by one) when the group cannot serve: no served chain,
        or its (windows, head) trace was not prewarmed -- serving never compiles or captures at request time."""
        n = len(n_valid)
        if not self.prefill_group_ok(n) or logits_rows == "all":
            return NotImplemented
        groups = self.__dict__.get("_pf_groups", {})
        if self.__dict__.get("_pf_groups_owner") is not self.pmk or n not in groups:
            # no group of n chunks was built at prewarm (GLM53_PF_GROUP_PREWARM / GroupStack.prepare): never build one at
            # request time -- realgate9 2026-10-25 19:32Z: GROUP_SIZES 4,2 with PREWARM 4@8192, a run's tail group of 2
            # built GroupStack(2) after the captures (305 buffers) and the next replay was refused by the trace tracker
            self.counters["prefill_group_cold"] = self.counters.get("prefill_group_cold", 0) + 1
            return NotImplemented
        grp = groups[n]
        head = not (logits_rows == "none" and sample is None)
        try:
            ts = grp.windows(chunk_start, [int(v) for v in n_valid])
        except AssertionError:  # a chunk planned to another bucket than the group's
            return NotImplemented
        if not grp.ready(ts, head=head):
            self.counters["prefill_group_cold"] = self.counters.get("prefill_group_cold", 0) + 1
            return NotImplemented
        io = io or kv_cache.extra["io"]
        self._kv = kv_cache
        out = self._prefill_chunk_tp(tokens, kv_cache=kv_cache, slot=slot, chunk_start=chunk_start, n_valid=list(n_valid),
                                     logits_rows=logits_rows, sample=sample, io=io, state_src=state_src,
                                     state_snap=state_snap, group=grp)  # fmt: skip
        self.counters["prefill_groups"] = self.counters.get("prefill_groups", 0) + 1
        return out

    def prefill_slice_plan(self, chunk_start: int, n_valid: int, front: int) -> tuple:
        """Stage 16's slice-boundary rule for one generator chunk (TPPrefill.slice_plan): (chunk lengths, branch); the
        chunk unchanged without the TP prefill"""
        tpp = getattr(self, "tpp", None)
        if tpp is None:
            return [int(n_valid)], ""
        return tpp.slice_plan(int(chunk_start), int(n_valid), int(front))

    def prefill_slice_rows(self, rows: int) -> int:
        """rows per chip of a `rows`-row TP prefill replay (bucket, packed T, batched nb x s) when sequence-parallel; 0
        when replicated, with the rule off (GLM53_PF_SLICE_RULE=0) or without the TP prefill (the generator's packed /
        batched slice-boundary check)"""
        tpp = getattr(self, "tpp", None)
        return tpp.slice_rows(int(rows)) if tpp is not None and tpp.slice_rule else 0

    def prefill_batch_size(self) -> tuple:
        """(bucket, nb) of the captured batched-admission trace (GLM53_PF_BATCH), or (0, 0)"""
        tpp = getattr(self, "tpp", None)
        if tpp is None or not tpp.batch_keys:
            return (0, 0)
        return tpp.batch_keys[0]

    def prefill_packed_size(self) -> tuple:
        """(T, KMAX) of the LARGEST captured packed bucket (GLM53_PF_PACKED), or (0, 0); every bucket:
        prefill_packed_sizes"""
        sizes = self.prefill_packed_sizes()
        return sizes[-1] if sizes else (0, 0)

    def prefill_packed_sizes(self) -> list:
        """[(T, KMAX)] of every captured packed bucket (GLM53_PF_PACKED="T:K[,T:K]"), ascending T; [] without"""
        tpp = getattr(self, "tpp", None)
        keys = (getattr(tpp, "packed_keys", None) or []) if tpp is not None else []
        return [(t_, max(k_ for tt, k_ in keys if tt == t_)) for t_ in sorted({t_ for t_, _ in keys})]

    def prefill_packed_align(self) -> int:
        """rows a packed segment rounds up to (32; 128 with the DSA indexer: whole pooled-key pages per segment)"""
        tpp = getattr(self, "tpp", None)
        return int(getattr(tpp, "packed_align", 32)) if tpp is not None else 32

    def prefill_batch_tp(self, reqs: list, *, kv_cache: StackCache, io: DecodeIO, sample=None) -> list:
        """Stage 16 BATCHED prefill: reqs = [(tokens, slot, n_valid)] fresh requests (chunk start 0, n_valid <= the
        largest bucket, distinct slots) in ONE replay of the batched trace (TPPrefill.prefill_batch: the routed
        experts' weights read once for every request) -> each request's first token (sampled on device by the head at
        its last position). The other decode rows' tokens / positions / sampler rows are restored afterwards."""
        tpp = self.tpp
        assert tpp is not None and tpp.kv is kv_cache, "prefill_batch_tp needs the attached TP prefill"
        n_max = max(int(nv) for _, _, nv in reqs)
        # the captured batch key's bucket when it holds them (shorter prompts are masked rows of it: no eager forward
        # of a smaller uncaptured key)
        s = min((s_ for s_, nb_ in tpp.batch_keys if nb_ == len(reqs) and s_ >= n_max), default=tpp.bucket_for(n_max))
        keep_tok, keep_pos, keep_samp = self.read(io.tokens), self.read(io.cur_pos), self.read(io.sampled)
        samp_rows = None
        if self.sample:  # sample: one SamplingParams for every request, or one per request
            sps = list(sample) if isinstance(sample, (list, tuple)) else [sample] * len(reqs)
            dv = lambda sp: tuple(sp.device_values(self.head.topk)) if sp is not None else (1, 0.0, 1.0)  # noqa: E731
            samp_rows = {int(slot): dv(sp) for (_, slot, _), sp in zip(reqs, sps)}
            self._write_sampler_params(override=samp_rows)
        pos = torch.full_like(keep_pos, -1)
        for _, slot, nv in reqs:
            pos[int(slot)] = int(nv) - 1
        self.write(io.cur_pos, pos)
        tpp.prefill_batch(s, [(tk, int(slot), int(nv)) for tk, slot, nv in reqs])
        got = self.read(io.tokens)
        first = [int(got[int(slot)]) for _, slot, _ in reqs]
        tok, samp = keep_tok.clone(), keep_samp.clone()
        for (_, slot, _), f in zip(reqs, first):
            tok[int(slot)] = f
            samp[int(slot)] = f
        self.write(io.tokens, tok)
        self.write(io.sampled, samp)
        self.write(io.cur_pos, keep_pos)  # the generator owns the positions
        if samp_rows is not None:
            self._write_sampler_params()
        self.counters["prefill_tokens"] += sum(int(nv) for _, _, nv in reqs)
        self.counters["prefill_batches_tp"] = self.counters.get("prefill_batches_tp", 0) + 1
        return first

    def prefill_packed_tp(self, reqs: list, *, kv_cache: StackCache, io: DecodeIO, t: int = 2048, sample=None) -> list:
        """Stage 16 PACKED prefill (mixed ISL, coordinator 2026-10-22 03:15Z): reqs = [(tokens, slot, n_valid)] fresh
        requests of DIFFERENT lengths packed back to back in ONE replay of the (t, K) packed trace
        (TPPrefill.prefill_packed) -> each request's first token (sampled on device at its last row). The other decode
        rows' tokens / positions / sampler rows are restored afterwards (as prefill_batch_tp)."""
        tpp = self.tpp
        assert tpp is not None and tpp.kv is kv_cache, "prefill_packed_tp needs the attached TP prefill"
        full = tpp.packed_reqs(reqs)  # (tokens, slot, n_valid, chunk_start, head)
        keep_tok, keep_pos, keep_samp = self.read(io.tokens), self.read(io.cur_pos), self.read(io.sampled)
        samp_rows = None
        if self.sample:
            sps = list(sample) if isinstance(sample, (list, tuple)) else [sample] * len(full)
            dv = lambda sp: tuple(sp.device_values(self.head.topk)) if sp is not None else (1, 0.0, 1.0)  # noqa: E731
            samp_rows = {slot: dv(sp) for (_, slot, _, _, hd), sp in zip(full, sps) if hd}
            self._write_sampler_params(override=samp_rows)
        pos = torch.full_like(keep_pos, -1)
        for _, slot, nv, st, hd in full:
            if hd:
                pos[slot] = st + nv - 1
        self.write(io.cur_pos, pos)
        tpp.prefill_packed(t, full)
        got = self.read(io.tokens)
        # head-less segments (intermediate chunks: state only) keep their decode rows like the non-participating ones
        first = [int(got[slot]) if hd else None for _, slot, _, _, hd in full]
        tok, samp = keep_tok.clone(), keep_samp.clone()
        for (_, slot, _, _, hd), f in zip(full, first):
            if hd:
                tok[slot] = f
                samp[slot] = f
        self.write(io.tokens, tok)
        self.write(io.sampled, samp)
        self.write(io.cur_pos, keep_pos)
        if samp_rows is not None:
            self._write_sampler_params()
        self.counters["prefill_tokens"] += sum(nv for _, _, nv, _, _ in full)
        self.counters["prefill_packed_tp"] = self.counters.get("prefill_packed_tp", 0) + 1
        return first

    def prefill_packed_caps(self) -> dict:
        """what the packed call accepts (Stage 9's prefill_packed_caps): fresh segments always; continuation (chunk start
        > 0, multiples of `align_start`, rows <= `dense_limit` with the DSA indexer, the sum of the replay's prefixes <=
        `max_prefix` rows) and head-less (intermediate) segments with GLM53_PF_PACKED_CONT=1; `align` = the segment row
        rounding; `sizes` = [(T, KMAX)]"""
        tpp = getattr(self, "tpp", None)
        if tpp is None or not getattr(tpp, "packed_keys", None):
            return dict(fresh=False, continuation=False, headless=False)
        cont = bool(getattr(tpp, "packed_cont", False))
        from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import DENSE_EXACT

        return dict(fresh=True, continuation=cont, headless=cont, dsa_indexer=tpp.has_dsa(), align=tpp.packed_align,
                    align_start=max(BLOCK_SIZE, tpp.start_align), max_prefix=tpp.packed_prefix if cont else 0,
                    dense_limit=DENSE_EXACT if tpp.has_dsa() else None, sizes=self.prefill_packed_sizes())  # fmt: skip

    # ------------------------------------------------------------------ LMCache hooks (doc/ttnn/15-lmcache)
    def kv_cache_connector_views(self, kv: Optional[StackCache] = None) -> dict:
        """{"model.layers.{L}.self_attn": the DSA layer's latent cache as the [num_blocks, 64, 512] bf16 view} (the
        connector's per-layer page view, vLLM's MLA [num_pages, page_size, head_dim]; same buffer as the device's
        [num_blocks, 1, 64, 512] — replicated on every chip), in layer order"""
        kv = kv or self._kv
        out = {}
        for i in sorted(kv.extra.get("latent", {}), key=lambda j: self.seq[j]["L"]):
            t = kv.extra["latent"][i]
            v = ttnn.reshape(t, (int(t.shape[0]), int(t.shape[-2]), int(t.shape[-1])))
            assert v.buffer_address() == t.buffer_address(), "the connector view must alias the cache"
            out[f"model.layers.{self.seq[i]['L']}.self_attn"] = v
        return out

    def export_kda_state(self, kv: StackCache, slot: int, n_tokens: Optional[int] = None, pools: bool = True,
                         source: str = "live") -> list:  # fmt: skip
        """slot's KDA recurrent state of every KDA layer (host, the full single-chip layout): [{L, S [64, 128, 128]
        fp32, C [3, 24576] bf16}] — the per-request checkpoint a prefix hit needs (page_contract.md §5.3) — and, when
        the cache keeps DSA indexer state (sparse regime) and has the export's staging buffer, one {"dsa":
        export_dsa_state(slot, n_tokens)} record (the slot's pooled keys of its first ceil(n_tokens / 128) units +
        ring). source="snapshot": the KDA part from the in-replay staging at the cut n_tokens (_kda_state_src; needs
        the KDA pack). The DSA part stays the slot's live rows: its units below the cut are complete and final after
        the replay, and its ring is rewritten by any hit's first chunk (one_replay_store/design.md K3)."""
        c, rd = divmod(int(slot), self.chunk)
        out = []
        t0 = _time.perf_counter()
        if kv.extra.get("kda_stage_s") is not None and self._kda_layers(kv, c):
            out.append(dict(kda=self.export_kda_pack(kv, slot, n_tokens, source)))  # the slot's rows, via the staging
        elif source != "live":
            raise RuntimeError(f"a KDA export from the {source} needs the KDA pack staging (ensure_state_export)")
        else:
            for i, e in enumerate(self.seq):
                st = kv.states[c][i]
                if st is None:  # sparse_dsa: no recurrent state
                    continue
                S, C = self._gather_state(e, *st)
                out.append(dict(L=e["L"], S=S[rd].clone(), C=C[0, :, rd].clone()))
        t1 = _time.perf_counter()
        dsa_t = self._dsa_tensors(kv)
        if dsa_t and kv.extra.get("dsa_stage") is not None:
            out.append(dict(dsa=self.export_dsa_state(kv, slot, n_tokens, pools=pools)))
        elif pools and any(paged for _k, _t, _r, paged in dsa_t):
            # a per-slot pooled-key store (GLM53_DSA_PAGED_POOLS=0) and no DSA export staging: a KDA-only checkpoint
            # would restore a slot with ZERO pooled keys for its prefix (2026-10-24 selfhit: L3 0.0625 off in another
            # slot). Callers allocate the staging before the first capture (ensure_state_export; the vLLM adapter does
            # it for a KV connector / host tier). pools=False (the pooled keys ride the LMCache pages) needs no rows.
            raise RuntimeError("export_kda_state: the cache keeps per-slot DSA pooled keys but has no DSA export "
                               "staging; call ensure_state_export(kv) before the first capture")  # fmt: skip
        self._time("kda_export", t1 - t0)
        self._time("dsa_export", _time.perf_counter() - t1)
        return out

    # ------------------------------------------------------------------ device snapshot of a slot's checkpoint (Stage 12)
    def state_snapshot_tensors(self, kv: Optional[StackCache] = None) -> list:
        """persistent device buffers that hold ONE slot's whole checkpoint at once (coordinator 2026-10-23T03:54Z: the
        prompt-end KDA export off the first-token path): every KDA layer's S pages ("snap_s"), the conv rows of every
        layer (32 layers per set of conv tiles, "snap_c"), the DSA rows of up to GLM53_STATE_SNAPSHOT_DSA_TILES tiles
        ("snap_d", default 4,096 = 8 MiB per chip; a checkpoint with more DSA rows is exported synchronously). Allocated
        before the first capture (DRAM); the page-DMA saves them (page_pool.LatentHitDma(state_tensors=)). [] without
        the KDA pack."""
        kv = kv or self._kv
        if kv.extra.get("snap_s") is None:
            if self.counters.get("captures", 0):
                raise RuntimeError("the state snapshot buffers must be allocated before the first trace capture")
            if not self.ensure_state_export(kv):
                return []
            lay = self._kda_layers(kv, 0)
            S0, C0 = lay[0][2]
            n_s, n_c = _pages(S0) // self.chunk, _pages(C0)
            mk = lambda n, dt: ttnn.zeros(
                (n, 1, 32, 32), dtype=dt, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=DRAM  # noqa: E731
            )
            kv.extra["snap_s"] = mk(len(lay) * n_s, S0.dtype)
            kv.extra["snap_c"] = mk(-(-len(lay) // 32) * n_c, C0.dtype)
            nd = int(os.environ.get("GLM53_STATE_SNAPSHOT_DSA_TILES", "4096"))
            kv.extra["snap_d"] = mk(nd, ttnn.bfloat16) if nd > 0 and self._dsa_tensors(kv) else None
        return [kv.extra[k] for k in ("snap_s", "snap_c", "snap_d") if kv.extra.get(k) is not None]

    def snapshot_state(self, kv: StackCache, slot: int, n_tokens: Optional[int] = None, pools: bool = True,
                       source: str = "live") -> Optional[dict]:  # fmt: skip
        """slot's checkpoint copied ON DEVICE into the snapshot buffers (SlotOps only: enqueued after the step's work,
        no host transfer, no sync); returns the plan collect_snapshot needs: {"pages": [S, conv, DSA pages used], ...}.
        None when it does not fit (no snapshot buffers, DSA rows past the budget): the caller exports synchronously.
        source="snapshot": the KDA pages / conv rows from the in-replay staging at the cut n_tokens (_kda_state_src,
        raises ValueError when it does not hold it); the copies are enqueued in queue order after the replay that wrote
        the staging and before any later replay rewrites it, so the tag is consumed here."""
        ss, sc, sd = kv.extra.get("snap_s"), kv.extra.get("snap_c"), kv.extra.get("snap_d")
        c, rd = divmod(int(slot), self.chunk)
        lay = self._kda_layers(kv, c) if ss is not None else []
        if not lay:
            return None
        S0, C0 = lay[0][2]
        n_s, n_c = _pages(S0) // self.chunk, _pages(C0)
        dsa, used = None, 0
        pools = bool(pools) and not self.pooled_paged(kv)  # GLM53_DSA_PAGED_POOLS: as export_dsa_state
        if self._dsa_tensors(kv) and kv.extra.get("dsa_stage") is not None:
            if sd is None:
                return None
            units = 0 if not pools else None if n_tokens is None else -(-int(n_tokens) // self.DSA_UNIT)
            cap, dsa = int(sd.padded_shape[0]), []
            for key, t, rows, paged in self._dsa_tensors(kv):
                if not 0 <= slot < rows:
                    raise ValueError(f"DSA snapshot of slot {slot}: {key} has {rows} rows")
                per = _pages(t) // rows
                n = per if not paged or units is None else min(per, 4 * int(units))
                if used + n > cap:
                    self.counters["state_snapshot_too_big"] = self.counters.get("state_snapshot_too_big", 0) + 1
                    return None
                dsa.append((key, int(t.buffer_address()), slot * per, n, used, bool(paged)))
                used += n
        src = self._kda_state_src(kv, slot, n_tokens, source)
        sa, ca = int(ss.buffer_address()), int(sc.buffer_address())
        ops = [(1, int(S.buffer_address()), s0, sa, j * n_s, n_s, _pages_bytes(S), 0)
               for j, (_L, S, s0, _C, _r) in enumerate(src)]  # fmt: skip
        ops += [(2, int(C.buffer_address()), 0, ca, (j // 32) * n_c, _pages(C), _pages_bytes(C), (r << 16) | (j % 32))
                for j, (_L, _S, _s0, C, r) in enumerate(src)]  # fmt: skip
        if dsa:
            da = int(sd.buffer_address())
            ops += [(1, a, t0, da, s0, n, 2048, 0) for _k, a, t0, n, s0, _p in dsa if n]
        t0 = _time.perf_counter()
        self.slot_ops.run(ops)
        self.counters["slot_ops"] += 1
        self.counters["state_snapshots"] = self.counters.get("state_snapshots", 0) + 1
        if source == "snapshot":
            kv.extra["kda_snap_at"] = None
            self.counters["kda_snapshot_exports"] = self.counters.get("kda_snapshot_exports", 0) + 1
        self._time("state_snapshot", _time.perf_counter() - t0)
        units = 0 if not pools else None if n_tokens is None else -(-int(n_tokens) // self.DSA_UNIT)
        return dict(layers=[L for _i, L, _st in lay], n_s=n_s, n_c=n_c, n_tokens=n_tokens, units=units, pools=bool(pools),
                    s_float=S0.dtype == ttnn.float32, c_float=C0.dtype == ttnn.float32,
                    dsa=None if dsa is None else [(k, n, s0, p) for k, _a, _t0, n, s0, p in dsa],
                    shard_c=(int(kv.extra["dsa_shard"][0]) if kv.extra.get("dsa_shard") is not None else None),
                    pages=[len(lay) * n_s, -(-len(lay) // 32) * n_c] + ([used] if sd is not None else []))  # fmt: skip

    @classmethod
    def collect_snapshot(cls, plan: dict, tiles) -> list:
        """export_kda_state's output (bitwise: the same tiles, the same layout) from a snapshot: tiles(i) -> torch [chips,
        pages, 32, 32] of snapshot buffer i (0 S, 1 conv, 2 DSA): the page-DMA's bytes (page_pool.StateTicket, any
        thread: no ttnn call here) or _stage_read of the buffers (main thread)"""
        nl, n_s, n_c = len(plan["layers"]), plan["n_s"], plan["n_c"]
        hs = tiles(0)
        S = hs[:, : nl * n_s].reshape(hs.shape[0], nl, n_s, 32, 32).clone()
        hc = tiles(1)
        C = torch.cat([hc[:, r * n_c : (r + 1) * n_c, : len(plan["layers"][r * 32 : (r + 1) * 32]), :].permute(0, 2, 1, 3)
                       for r in range(-(-nl // 32))], dim=1).clone()  # fmt: skip
        out = [dict(kda=dict(kind=cls.KDA_PACK, layers=list(plan["layers"]), n_s=n_s, S=S, C=C))]
        if plan["dsa"] is not None:
            hd = tiles(2)
            parts = {e[0]: hd[:, e[2] : e[2] + e[1]].clone() for e in plan["dsa"]}
            paged = {e[0]: (e[3] if len(e) > 3 else True) for e in plan["dsa"]}
            chips = next(iter(parts.values())).shape[0]
            shard_c = plan.get("shard_c")
            if shard_c is not None:  # GLM53_DSA_SHARD: as export_dsa_state -- unit-paged stores in global unit order
                to_units = globals()["dsa_units_to_unit_order"]
                for k, v in parts.items():
                    if paged[k]:
                        parts[k] = to_units(v, int(shard_c), plan["units"], getattr(cls, "DSA_TILES_PER_UNIT", 4))
                    elif all(torch.equal(v.view(torch.int16)[c], v.view(torch.int16)[0]) for c in range(chips)):
                        parts[k] = v[:1].clone()
                same = all(v.shape[0] == 1 for v in parts.values())
            else:
                same = all(torch.equal(v.view(torch.int16)[c], v.view(torch.int16)[0]) for v in parts.values()
                           for c in range(chips))  # fmt: skip
                if same:
                    parts = {k: v[:1].clone() for k, v in parts.items()}
            out.append(dict(dsa=dict(kind=cls.DSA_BLOB, unit_tokens=cls.DSA_UNIT, units=plan["units"],
                                     n_tokens=plan["n_tokens"], chips=chips, replicated=same,
                                     layout="unit" if shard_c is not None else "chip", pools=plan.get("pools", True),
                                     tensors=parts)))  # fmt: skip
        return out

    def export_kda_state_async(self, kv: StackCache, slot: int, n_tokens: Optional[int] = None, dma=None,
                               pools: bool = True, source: str = "live"):  # fmt: skip
        """the prompt-end checkpoint OFF the first-token path: snapshot_state (device copies), one sync for those copies
        (the step's own work is already done: the sampler read its tokens), then the page-DMA SAVEs the snapshot while
        the next steps run (dma: page_pool.LatentHitDma built with state_tensors=state_snapshot_tensors()). Returns a
        zero-argument callable -> export_kda_state's output (waits for the DMA; no ttnn call: any thread), or None
        (no DMA / does not fit: the caller exports synchronously). source: as snapshot_state."""
        if dma is None or not getattr(dma, "state", None):
            return None
        plan = self.snapshot_state(kv, slot, n_tokens, pools=pools, source=source)
        if plan is None:
            return None
        t0 = _time.perf_counter()
        ttnn.synchronize_device(self.mesh)  # the snapshot copies landed: the page-DMA reads the buffers, not the queue
        ticket = dma.save_state(plan["pages"])
        self._time("state_snapshot_save", _time.perf_counter() - t0)
        import numpy as np  # noqa: PLC0415

        def tiles(i):
            elem = np.float32 if (i == 0 and plan["s_float"]) or (i == 1 and plan["c_float"]) else np.uint16
            t = torch.from_numpy(ticket.tiles(i, elem))
            return t if elem is np.float32 else t.view(torch.bfloat16)

        return lambda: self.collect_snapshot(plan, tiles)

    def _time(self, name: str, dt: float) -> None:
        """host wall seconds of a prefix-state transfer (the serving events' report: T1 spill / promote timing)"""
        self.counters[name + "_s"] = round(self.counters.get(name + "_s", 0.0) + dt, 6)
        self.counters[name + "_last_s"] = round(dt, 6)

    def import_kda_state(self, kv: StackCache, slot: int, states: list) -> None:
        """inverse of export_kda_state: write a checkpoint into slot's rows of every KDA layer (the chunk's other rows
        are read back and written unchanged)"""
        c, rd = divmod(int(slot), self.chunk)
        by_l = {int(d["L"]): d for d in states if "L" in d}
        dsa = [d["dsa"] for d in states if "dsa" in d]
        packs = [d["kda"] for d in states if "kda" in d]
        t0 = _time.perf_counter()
        for blob in packs:  # the staged path: no gather, no device allocation, no program after a capture
            self.import_kda_pack(kv, slot, blob)
        for i, e in enumerate(self.seq):
            st = kv.states[c][i]
            if st is None or packs:
                continue
            d = by_l[e["L"]]
            S, C = self._gather_state(e, *st)
            S[rd] = d["S"].to(S.dtype)
            C[0, :, rd] = d["C"].to(C.dtype)
            Sn, Cn = self._shard_state(e, S, C)
            ttnn.copy(Sn, st[0])
            ttnn.copy(Cn, st[1])
            ttnn.deallocate(Sn)
            ttnn.deallocate(Cn)
        ttnn.synchronize_device(self.mesh)
        t1 = _time.perf_counter()
        for blob in dsa:  # the slot's DSA rows of the checkpoint's units (sparse regime)
            self.import_dsa_state(kv, slot, blob)
        ttnn.synchronize_device(self.mesh)
        self._time("kda_import", t1 - t0)
        self._time("dsa_import", _time.perf_counter() - t1)

    def _gather_state(self, e, Sm, Cm):
        if self.tp > 1:
            return e["tp"].shards.gather_state(Sm, Cm)
        return ttnn.to_torch(Sm).float(), ttnn.to_torch(Cm)

    def _shard_state(self, e, S, C):
        if self.tp > 1:
            return e["tp"].shards.shard_state(S, C)
        mk = lambda x, dt: ttnn.from_torch(
            x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=DRAM  # noqa: E731
        )
        return mk(S, ttnn.float32), mk(C, ttnn.bfloat16)

    def close(self) -> None:
        if getattr(self, "spec", None) is not None:
            self.spec.release()
        for t in list(self.traces):
            self.release(t)
        if getattr(self, "tpp", None) is not None:
            self.tpp.release()

    def synchronize(self) -> None:
        """Wait for every enqueued command on the mesh (Stage 9's decode host-overhead probes)."""
        ttnn.synchronize_device(self.mesh)

    def trace_report(self) -> dict:
        mv = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)  # lock-step allocation: chip 0 = every chip
        dram = int(mv.total_bytes_allocated_per_bank) * int(mv.num_banks)
        tpp = getattr(self, "tpp", None)
        # the TP prefill's own replays / captures / runtime eager fallbacks (operator 2026-10-22 09:18Z: eager execution
        # is prohibited -- a served run's 'eager' must stay 0; warm-up forwards before capture are not counted there)
        prefill = dict(tpp.counters) if tpp is not None and isinstance(getattr(tpp, "counters", None), dict) else None
        return dict(self.counters, traces=len(self.traces), tp=self.tp, layers=[e["L"] for e in self.seq],
                    load=getattr(self, "load_info", None), prefill=prefill,
                    program_cache_entries=self.mesh.num_program_cache_entries(),
                    dram_gib_per_chip=round(dram / 2**30, 3), trace_mib_per_chip=self._trace_mib())  # fmt: skip

    def _trace_mib(self):
        """MiB of the trace region the captured traces occupy per chip (Stage 9: size the serving trace region as
        used + margin); None where the memory view has no trace buffer type"""
        try:
            v = ttnn.get_memory_view(self.mesh, ttnn.BufferType.TRACE)
            return round(int(v.total_bytes_allocated_per_bank) * int(v.num_banks) / 2**20, 2)
        except Exception:
            return None


class DecodeTee:
    """Prefix-cache G snapshots INSIDE the decode trace (GLM53_DECODE_TEE=1 with the snapshot pool): ONE program, the
    last of every decode trace (kernels/prefill/decode_tee.cpp), copies the post-step state of each slot listed in
    `ctl` into its trailing snapshot-pool entry: KDA S pages + conv rows (one core per KDA layer) and the DSA
    pooled-key pages [p0, p1) + ring (one core per sparse_dsa layer). The host writes `ctl` only around a step in which
    a slot crosses a 128-token boundary (no pairs: the program reads ctl and returns), so a decode step stays one
    replay and no copy launch runs between steps (doc/ttnn/14-production-tp8-megakernel/dsa/g_snapshot_design.md)."""

    MAX_PAIRS = 32
    CTL_WORDS = 192  # one 768-B ROW_MAJOR page: word 0 = pairs, then (src slot, dst slot, p0, p1) per pair

    def __init__(self, backend: "StackBackend", kv: StackCache):
        self.mesh, self.max_batch = backend.mesh, int(kv.max_batch)
        self.chunk, self.n_chunks = int(backend.chunk), len(kv.states)
        self.entries = int(kv.extra.get("snapshot_entries", 0))
        assert self.entries > 0, "the decode tee writes snapshot-pool entries (GLM53_PREFIX_DEVICE_ENTRIES)"
        repl = ttnn.ReplicateTensorToMesh(self.mesh) if backend.tp > 1 else None
        self.repl = repl
        self.ctl = ttnn.from_torch(torch.zeros(1, self.CTL_WORDS, dtype=torch.int32), dtype=ttnn.uint32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, memory_config=DRAM,
                                   mesh_mapper=repl)  # fmt: skip
        kda = [li for li, e in enumerate(backend.seq) if e["sparse"] != "dsa"]
        dsa = sorted(kv.extra.get("dsa", {}))
        grid = self.mesh.compute_with_storage_grid_size()
        cores = [(x, y) for y in range(int(grid.y)) for x in range(int(grid.x))]
        assert len(kda) + len(dsa) <= len(cores), (len(kda), len(dsa))
        self.kda_rt, self.dsa_rt = {}, {}
        n_s = n_c = None
        for (x, y), li in zip(cores, kda):
            row = [self.ctl.buffer_address()]
            for c in range(self.n_chunks):
                S, C = kv.states[c][li]
                assert _pages(S) % _rows(S) == 0, (li, c)
                n_s, n_c = (n_s or _pages(S) // _rows(S)), (n_c or _pages(C))
                assert (_pages(S) // _rows(S), _pages(C)) == (n_s, n_c), (li, c)
                row += [S.buffer_address(), C.buffer_address()]
            self.kda_rt[(x, y)] = row
        self.n_s, self.n_c = n_s, n_c
        self.dsa_ct = None
        for (x, y), li in zip(cores[len(kda) :], dsa):
            st = kv.extra["dsa"][li]
            if st.get("paged"):
                raise NotImplementedError("DecodeTee with GLM53_DSA_PAGED_POOLS=1 (block-addressed pooled keys)")
            slots, ps = int(st["slots"]), int(st.get("pool_slots", st["slots"]))
            assert ps >= self.max_batch + self.entries, (ps, slots, self.max_batch, self.entries)
            # a 128-token store page = 32 pools = one tile row of [32, 128] bf16 = 128 / 32 tiles
            ct = [_pages(st["pools"]) // ps, int(st["pools"].padded_shape[-1]) // 32, _pages(st["ring"]) // slots]
            assert self.dsa_ct in (None, ct), (self.dsa_ct, ct)
            self.dsa_ct = ct
            self.dsa_rt[(x, y)] = [self.ctl.buffer_address(), st["pools"].buffer_address(), st["ring"].buffer_address()]
        crs = lambda xy: ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))  # noqa: E731
                                            for x, y in xy])  # fmt: skip
        self.kda_cores = crs(list(self.kda_rt)) if self.kda_rt else None
        self.dsa_cores = crs(list(self.dsa_rt)) if self.dsa_rt else None
        self.pairs: list = []
        self.counters = dict(runs=0, writes=0, tees=0)

    def _kernel(self, cores, rt_rows: dict, ct: list, mode: str):
        rt = ttnn.RuntimeArgs()
        for (x, y), row in rt_rows.items():
            rt[x][y] = list(row)
        return ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, "prefill", "decode_tee.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=cores, compile_time_args=ct,
            runtime_args=rt, defines=[("TEE_CB", "0"), (mode, "1"), ("TEE_REV", "2")],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                     noc=ttnn.NOC.RISCV_0_default))  # fmt: skip

    def run(self) -> None:
        """the tee program (inside the decode trace capture; eagerly once in prepare_decode = its compile)"""
        kernels, cbs = [], []
        size = 2 * 4096 + 2 * 2048 + self.CTL_WORDS * 4
        kda_ct = [self.n_s, self.n_c, self.chunk, self.n_chunks, self.max_batch + self.entries]  # 4: the slot bound
        for cores, rt_rows, ct, mode in ((self.kda_cores, self.kda_rt, kda_ct, "TEE_KDA"),
                                         (self.dsa_cores, self.dsa_rt, self.dsa_ct, "TEE_DSA")):  # fmt: skip
            if cores is None:
                continue
            kernels.append(self._kernel(cores, rt_rows, ct, mode))
            cbs.append(ttnn.CBDescriptor(total_size=size, core_ranges=cores, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=size)]))  # fmt: skip
        ttnn.generic_op([self.ctl, self.ctl], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        self.counters["runs"] += 1

    PAGE = 128  # tokens per DSA pooled-key store page (32 pools of 4)

    def set(self, pairs) -> None:
        """the pairs the NEXT decode step tees: [(slot, entry, t0, t1)] (entry = snapshot-pool entry index; the entry
        already holds the slot's DSA pooled keys of positions < t0, the step ends at t1 tokens: the store pages
        [t0 // 128, ceil(t1 / 128)) move). Written only when it changes ([] after a crossing step)."""
        pairs = [(int(s_), int(e_), int(a_), int(b_)) for s_, e_, a_, b_ in pairs]
        if pairs == self.pairs:
            return
        assert len(pairs) <= self.MAX_PAIRS, len(pairs)
        v = torch.zeros(1, self.CTL_WORDS, dtype=torch.int32)
        v[0, 0] = len(pairs)
        for i, (s_, e_, a_, b_) in enumerate(pairs):
            assert 0 <= s_ < self.max_batch and 0 <= e_ < self.entries and 0 <= a_ <= b_, (s_, e_, a_, b_)
            p0, p1 = a_ // self.PAGE, -(-b_ // self.PAGE)
            v[0, 1 + 4 * i : 5 + 4 * i] = torch.tensor([s_, self.max_batch + e_, p0, p1], dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(v, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                                        mesh_mapper=self.repl), self.ctl)  # fmt: skip
        self.pairs = pairs
        self.counters["writes"] += 1
        self.counters["tees"] += len(pairs)


def _pages_bytes(t) -> int:
    """bytes of one tile page of a TILE-layout tensor (fp32 4096, bf16 2048)"""
    return 32 * 32 * {ttnn.float32: 4, ttnn.bfloat16: 2, ttnn.int32: 4, ttnn.uint32: 4}[t.dtype]


def snapshot_chunk_rows(entries: int, chunk: int) -> list:
    """rows of each prefix-cache pool state chunk for `entries` entries: whole chunks, then the remainder"""
    full, rem = divmod(max(0, int(entries)), int(chunk))
    return [int(chunk)] * full + ([rem] if rem else [])


def _rows(t) -> int:
    """user rows of a KDA state tensor (S [rows, H, 128, 128]: its outer dim)"""
    return int(t.shape[0])


def _pages(t) -> int:
    """tiles of a TILE-layout tensor (per chip)"""
    sh = [int(v) for v in t.padded_shape]
    n = 1
    for v in sh[:-2]:
        n *= v
    return n * (sh[-2] // 32) * (sh[-1] // 32)


def _rows_view(t_, rows: int):
    """a tile tensor seen with `rows` logical rows (same buffer, same padded shape)"""
    shp = [int(v) for v in t_.shape]
    if shp[-2] == rows:
        return t_
    return ttnn.reshape(t_, ttnn.Shape(shp[:-2] + [rows, shp[-1]]), ttnn.Shape([int(v) for v in t_.padded_shape]))


def _seed_width(o, w: int, full: int) -> None:
    """an object shared by every width (the sparse_dsa layers' MkMla / MkMoE): its (name, w) persistent buffers as
    w-row views of its own (name, full) ones, created before its first width call"""
    ps = getattr(o, "persist", None)
    if ps is None:
        return
    for k in [k for k in ps if isinstance(k, tuple) and len(k) == 2 and k[1] == full]:
        if (k[0], w) not in ps:
            ps[(k[0], w)] = _rows_view(ps[k], w)


def _alias_persist(dst, src, w: int, full: int, seed: bool = True) -> int:
    """native-t width buckets: the width-w object dst takes the full-width object src's persistent hand-off buffers —
    (name, w) keys as w-row views of src's (name, full), t-independent keys as src's own tensors; dst's are freed (one
    L1 set for every width; programs of different widths never run concurrently). seed: also create the (name, w)
    views dst has not allocated yet (its first width call would allocate them outside any pool)"""
    pd, ps = getattr(dst, "persist", None), getattr(src, "persist", None)
    if pd is None or ps is None or pd is ps:
        return 0
    n = 0
    keys = set(pd) | (
        {(k[0], w) for k in ps if isinstance(k, tuple) and len(k) == 2 and k[1] == full} if seed else set()
    )
    # keys already aliased by an earlier call (a second allocate_kv_cache on the same backend: the generator's own
    # cache + vLLM's) hold views of src's buffers -- deallocating them would free src's (name, full) tensors
    done = dst.__dict__.setdefault("_aliased_keys", set())
    for k in keys:
        v = pd.get(k)
        if isinstance(k, tuple) and len(k) == 2 and k[1] == w and (k[0], full) in ps:
            new = _rows_view(ps[(k[0], full)], w)
        elif v is not None and k in ps and tuple(ps[k].shape) == tuple(v.shape):
            new = ps[k]
        else:
            continue
        if v is not None and v is not new and k not in done:
            ttnn.deallocate(v)
        pd[k] = new
        done.add(k)
        n += 1
    return n


def _has_dsa(e: dict) -> bool:
    """a stack entry's MLA half runs MkMla(dsa=...) (TPMla -> .mk, or a one-chip MkMla)"""
    m = e.get("mla")
    return getattr(getattr(m, "mk", m), "dsa", None) is not None


class SlotOps:
    """per-user state operations between steps (kernels/head/slot_op.cpp), one core, <= 42 ops per launch"""

    MAX_OPS = 42
    OP_WORDS = 8  # kind, src, src0, dst, dst0, pages, page bytes, rows

    def __init__(self, mesh):
        self.mesh = mesh
        repl = ttnn.ReplicateTensorToMesh(mesh) if int(mesh.get_num_devices()) > 1 else None
        self.dummy = ttnn.from_torch(torch.zeros(1, 32, dtype=torch.int32), dtype=ttnn.int32,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM, mesh_mapper=repl)  # fmt: skip

    def run(self, ops: list):
        core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        for i in range(0, len(ops), self.MAX_OPS):
            part = ops[i : i + self.MAX_OPS]
            rt = ttnn.RuntimeArgs()
            # fixed length (1 + MAX_OPS x 8 words): generic_op's program cache keys on the descriptor WITHOUT the
            # runtime args, so a cached launch with fewer args would be reused for a longer op list (args past the end
            # never written -> the kernel reads stale words)
            args = [len(part)] + [int(v) for op in part for v in op]
            assert all(len(op) == self.OP_WORDS for op in part), part
            rt[0][0] = args + [0] * (1 + self.MAX_OPS * self.OP_WORDS - len(args))
            kd = ttnn.KernelDescriptor(
                kernel_source=os.path.join(KDIR, "head", "slot_op.cpp"),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=core, compile_time_args=[],
                runtime_args=rt, defines=[("SLOT_CB", "0")],
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                         noc=ttnn.NOC.RISCV_0_default))  # fmt: skip
            cb = ttnn.CBDescriptor(total_size=8192, core_ranges=core, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=8192)])  # fmt: skip
            ttnn.generic_op([self.dummy, self.dummy],
                            checked(ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[cb])))  # fmt: skip
