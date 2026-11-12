# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash on the persistent-stack port (models/autoports/zai_org_glm_5_3_flash/tt: ``StackBackend`` under
``Glm53Generator``) for the TT vLLM plugin -- the FIRST adapter: text only, greedy on device (the head's top-1 inside
the one full-width decode trace), host-sampled rows through full logits, no prefix cache, no multimodal, synchronous
scheduling (every decode reloads its inputs). GLM53_STACK_PREFIX=1 (opt-in) adds vLLM's automatic
prefix caching from the stack's device snapshot pool (``StackPrefixCache``). Image input:
``GLM53StackForConditionalGeneration`` at the bottom of this module (the same adapter + vLLM's ``SupportsMultiModal``
and the upstream GLM-5.3-Flash processor). Speculative decode (opt-in GLM53_STACK_VLLM_SPEC=1 with GLM53_SPEC_K;
--no-async-scheduling, or async scheduling with GLM53_STACK_VLLM_SPEC_ASYNC=1: spec_async_enabled): the backend's MTP
step for the lone live greedy request (section "Stage 17: speculation").

Selected with ``TT_MODEL_CLASS_OVERRIDES="Glm5NextForCausalLM=<ADAPTER_TARGET>,Glm5NextForConditionalGeneration=
<ADAPTER_TARGET>"``. The plugin opens the mesh BEFORE ``initialize_vllm_model`` (worker.py:274-281) with
``trace_region_size`` / ``worker_l1_size`` / ``l1_small_size`` from the launch's ``tt`` config (worker.py:1026-1041)
and the fabric from ``model_capabilities["fabric_config"]`` (worker.py:994-1010), so the validated 8x4 launch needs
``trace_region_size = galaxy_defaults.trace_mb(max_model_len) << 20``, ``worker_l1_size 1280896``, ``l1_small_size
32768`` and ``--block-size 64`` on the command line; ``galaxy_defaults.apply`` sets the remaining GLM53_* knobs at
model construction (its GLM53_MODEL_TRACE_MB is dead here: the device is already open).

Slot aliasing: the plugin's ``empty_slots`` / ``slot_remap`` name PLUGIN slots, which the adapter maps onto the
generator's slots (``_CacheBinding.slots``) through a bijection it keeps (``_alias``). The plugin's decode gather
(model_runner.py:1366-1421: a full permutation of every slot after it condenses the rows) is absorbed into the alias
instead of moving state: row ``i`` decodes in the generator slot its request was admitted to
(``Glm53Generator.decode_forward(slots=...)``), so no ``remap_slots`` copies run and a gather that is a cycle over a
full batch (every slot live: the generator's "cyclic slot_remap needs a free scratch slot" refusal) is fine. A new
request takes the lowest free generator slot. The plugin references in this module (``worker.py:...``,
``model_runner.py:...``, ``async_decode.py:...``) are to vllm-tt-plugin main at 042a948 (2026-11-09). vLLM's logical blocks travel
``pt_shadow -> OwnerBlockMap -> per-chip local tables`` inside the generator / backend once the cache is allocated
with ``null_block=0``.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import gc
import hashlib
import importlib
import os
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch
from loguru import logger

from models.autoports.zai_org_glm_5_3_flash.tt.generator import (
    BLOCK_SIZE,
    MAX_DECODE_SLOTS,
    DecodeOutput,
    PendingTokens,
)
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams

HF_ARCHITECTURES = ("Glm5NextForConditionalGeneration", "Glm5NextForCausalLM")
ADAPTER_TARGET = "models.demos.glm53_flash.tt.generator_vllm_stack:GLM53StackForCausalLM"
MM_ADAPTER_TARGET = "models.demos.glm53_flash.tt.generator_vllm_stack:GLM53StackForConditionalGeneration"
MM_ENV = "GLM53_STACK_MM"  # "1": the stack's embedding table carries the image rows (mk_head.mm_extra_rows)
MM_ENCODER_ENV = "GLM53_STACK_MM_ENCODER"  # the multimodal class's image encoder: device (default) | hf_cpu | none
MM_CACHE_ENV = "GLM53_STACK_MM_CACHE_MB"  # the image feature cache's bound: MB of feature rows (default 256)
MM_ALIAS_CAP = 4096  # vLLM item identifiers remembered as aliases of a content key
LATENT_DIM = 512  # kv_lora_rank: the latent page row, the only "KV" vLLM accounts for (generator_vllm.LATENT_DIM)


def _mesh_shape_from_env() -> tuple[int, int]:
    """The mesh the plugin will open (``generator_vllm._mesh_shape_from_env``): ``GLM53_MESH_SHAPE`` (rows,cols) if
    set, else the plugin's ``MESH_DEVICE`` presets this model runs on or a ``(r, c)`` tuple."""
    spec = os.environ.get("GLM53_MESH_SHAPE")
    if spec:
        r, c = (int(v) for v in spec.split(","))
        return r, c
    md = os.environ.get("MESH_DEVICE", "")
    if md.startswith("("):
        r, c = (int(v) for v in md.strip("() ").split(","))
        return r, c
    return {"BH-Galaxy": (8, 4), "TG": (8, 4), "P150x8": (1, 8), "P150x4": (1, 4)}.get(md, (8, 4))


def _device_top_k() -> int:
    """The head's device sampler bound (kernels/head/topk_select.cpp, StackBackend(topk=K, sample=True)): temperature /
    top-k / top-p rows are drawn from the K best candidates inside the traced decode step; a request without a top-k
    bound (GLM's generation config sets none) or above K is drawn from those K (device_top_k_unbounded_as). K is the
    largest the head's in-program candidate gather carries: every chip's K candidates x 6 bf16 fields in lane 0's
    64 tiles (mk_head.MkHead), 10 on the 32-chip 8x4, 32 (the sampler's cap) on 8 chips.
    GLM53_STACK_DEVICE_SAMPLING=0: greedy on device only, sampled rows on the host sampler over the full logits."""
    if os.environ.get("GLM53_STACK_DEVICE_SAMPLING", "1") != "1":
        return 1
    rows, cols = _mesh_shape_from_env()
    return max(1, min(32, (64 * 32) // (6 * rows * cols)))


DEVICE_TOP_K = _device_top_k()
# the resident-input chain (DECODE_RELOAD_CONTRACT.md): a steady device-sampled decode keeps its tokens and positions on
# device (reload_inputs=False), so vLLM's async scheduling builds step N + 1 while step N runs -- the server's host
# input work (~3-5 ms per step at ISL 128, py-spy 2026-11-10) overlaps the device step instead of adding to it.
# GLM53_STACK_ASYNC_DECODE=0: every decode reloads its inputs (no async scheduling)
ASYNC_DECODE = DEVICE_TOP_K > 1 and os.environ.get("GLM53_STACK_ASYNC_DECODE", "1") == "1"
# GLM53_STACK_PREFIX=1 (default off): vLLM's automatic prefix caching served from the stack's device snapshot pool
# (StackPrefixCache) -- the adapter declares supports_prefix_caching and confirms_prefix_cache_hits (the plugin's
# scheduler trims vLLM's block-hash hit to a position the device holds a snapshot of). Read at import: the plugin
# reads the capabilities from the class.
PREFIX_CACHE = os.environ.get("GLM53_STACK_PREFIX", "0") == "1"
# GLM53_STACK_PREFIX_HEAD (tokens, default 16,384 = two 8,192 chunks): snapshots at or below it are kept (a shared system /
# tool prompt serves every conversation); above it each request keeps only its LATEST snapshot -- its previous one is
# evicted when the next commits -- so one long prompt holds head + 1 pool entries instead of one per chunk (AgentX
# 2026-11-12: a 400 K prompt rotated all 32 entries and every next turn re-prefilled from 0, hit rate 0.7 %)
PREFIX_HEAD = int(os.environ.get("GLM53_STACK_PREFIX_HEAD", "16384"))
SPEC_ENV = "GLM53_STACK_VLLM_SPEC"
SPEC_MAX_ROWS = 4  # tt/spec_mtp.py: the speculative step runs at the B <= 4 chunk width (B 4 validated, chain_mk201)
# GLM53_SPEC_MULTI=1 (default off; design doc/perf/mtp_multi_user_design.md): every live user speculates on a B 16
# build (GLM53_SPEC_MULTI_W, default 16) with generator.spec_multi_plan's row groups. The backend's multi-user step
# (``spec_multi_capable``: batch 2's fixed G = GLM53_SPEC_MULTI_USERS) verifies exactly G greedy requests on the group
# bases; any other step keeps DraftOutput.num_valid = 0 (plain decode), exactly as the single-user plan's multi-row steps
SPEC_MULTI_ENV = "GLM53_SPEC_MULTI"


def spec_enabled(env=os.environ) -> bool:
    """Stage 17 single-user speculation through the plugin's contract (SPEC_DECODE_CONTRACT.md, method custom_class
    with the model-owned drafter): GLM53_STACK_VLLM_SPEC=1 (default off) with GLM53_SPEC_K = K > 0 (the backend builds
    the MTP speculative step: tt/spec_mtp.py; GLM53_SPEC_ON_SHARD=1 on the 8x4 shard)."""
    return env.get(SPEC_ENV, "0") == "1" and int(env.get("GLM53_SPEC_K", "0") or 0) > 0


SPEC = spec_enabled()
SPEC_ASYNC_ENV = "GLM53_STACK_VLLM_SPEC_ASYNC"


def spec_async_enabled(env=os.environ, async_decode: bool | None = None) -> bool:
    """``supports_async_spec_decode`` (SPEC_DECODE_CONTRACT.md 4c): GLM53_STACK_VLLM_SPEC_ASYNC=1 (default off) on a
    speculating launch that also declares the ordinary async decode (``supports_async_decode``: the resident-input
    chain) -- the profile may then keep vLLM's async scheduling. A verify submitted with ``read_from_device=False``
    returns ``SpecReadback`` (the step replayed, its record's copy enqueued behind it): ``read_decode_output`` hands
    the record's event to the plugin, ``process_decode_output_host`` completes it into the ``[B, 1+K]`` argmax ids at
    the plugin's finalize, before the accept walk; every speculation entry point completes an outstanding one first."""
    a = ASYNC_DECODE if async_decode is None else bool(async_decode)
    return spec_enabled(env) and a and env.get(SPEC_ASYNC_ENV, "0") == "1"


SPEC_ASYNC = spec_async_enabled()


@dataclasses.dataclass
class SpecReadback:
    """A deferred verify's answer (``VerifyOutput.argmax_ids`` of a ``read_from_device=False`` fused step): the
    generator's ``SpecPending`` plus the plugin row it answers; ``ids`` once completed (``_spec_collect``)."""

    handle: object
    row: int
    B: int
    W: int
    ids: torch.Tensor | None = None


def spec_plan_for(max_num_seqs: int, requested_k: int, types: dict, env=os.environ):
    """The plugin's ``SpecPlan`` / ``SpecReject`` for one launch point (``GLM53StackForCausalLM.spec_plan``; pure): the
    backend's step drafts exactly K = GLM53_SPEC_K (num_speculative_tokens must be >= K; effective_k = K) for ONE live
    user in generator slot 0 of a B in [K + 1, 4] build; ``k_by_rows`` ((1, K), (max_num_seqs, 1)) and
    ``DraftOutput.num_valid`` = 0 whenever more than one row is live keep every other step an ordinary decode;
    ``argmax_ids`` only (greedy: sampled rows decode plainly through ``supports_narrow_decode``), the drafter's state
    on device (no hidden feed: the step reads the head's h itself)."""
    if not spec_enabled(env):
        return types["SpecReject"](reason=f"{SPEC_ENV}=1 with GLM53_SPEC_K > 0 enables the stack's speculative step")
    K = int(env.get("GLM53_SPEC_K"))
    B, req = int(max_num_seqs), int(requested_k)
    if req < K:
        return types["SpecReject"](
            reason=f"the stack's speculative step drafts exactly K = {K} (GLM53_SPEC_K); num_speculative_tokens {req}",
            supported_k=(K,),
        )
    if env.get(SPEC_MULTI_ENV, "0") == "1":
        from models.autoports.zai_org_glm_5_3_flash.tt.generator import spec_multi_k_by_rows

        W = int(env.get("GLM53_SPEC_MULTI_W", "16") or 16)
        if not K + 1 <= B <= W:
            return types["SpecReject"](
                reason=f"{SPEC_MULTI_ENV}=1: max_num_seqs {B} outside [K + 1, {W}] (the B {W} build's verify width)",
                supported_k=tuple(k for k in (1, 2, 3) if k + 1 <= B) if B <= W else (),
            )
        return types["SpecPlan"](
            effective_k=K,
            lanes_per_request=1,
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            accept_modes=("argmax_ids",),
            drafter_state="internal",
            supports_narrow_decode=True,
            k_by_rows=spec_multi_k_by_rows(K, B, W),
        )
    if not K + 1 <= B <= SPEC_MAX_ROWS:
        ok = tuple(k for k in (1, 2, 3) if k + 1 <= B) if B <= SPEC_MAX_ROWS else ()
        return types["SpecReject"](
            reason=f"max_num_seqs {B}: the speculative step verifies K + 1 = {K + 1} rows on a B <= {SPEC_MAX_ROWS} build",
            supported_k=ok,
        )
    return types["SpecPlan"](
        effective_k=K,
        lanes_per_request=1,
        extra_bytes_per_seq=0,
        extra_bytes_per_token=0,
        accept_modes=("argmax_ids",),
        drafter_state="internal",
        supports_narrow_decode=True,
        k_by_rows=((1, K), (B, 1)) if B > 1 else (),
    )


def _fabric_capability() -> dict | None:
    """``model_capabilities["fabric_config"]`` (MODEL_CAPABILITIES.md:126; merged into ``ttnn.set_fabric_config(**kw)``
    by worker.py:994-1010): ``config`` from ``stack_model.serving_fabric_config`` (FABRIC_1D, the fabric every stack
    row was validated with -- the EP all-to-all and the sharded pool's streams, GLM53_FABRIC=1d) plus the relaxed-init
    ``reliability_mode`` and the maximum-payload ``router_config`` of ``tt/mesh.py::mesh_device_params`` (what the e2e
    test's device fixture set), the keys ``generator_vllm._fabric_capability`` declares. Built only in a process that
    already imported ttnn (the plugin does before it resolves the model class): importing this module stays ttnn-free.
    """
    if "ttnn" not in sys.modules:
        if "vllm" not in sys.modules:  # a host process (the host test keeps this module ttnn-free)
            return None
        import ttnn  # noqa: F401  (a vLLM process: the validated fabric must reach the plugin before it opens the mesh)
    # no fallback: a failure here must stop the launch (the plugin's hardware default on a Blackhole Galaxy is the 2D
    # torus, a fabric the stack was never validated on), so the exception propagates
    from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import serving_fabric_config
    from models.demos.glm53_flash.tt.mesh import mesh_device_params

    out = dict(serving_fabric_config())
    params = mesh_device_params(*_mesh_shape_from_env())
    if "reliability_mode" in params:
        out["reliability_mode"] = params["reliability_mode"]
    if "fabric_router_config" in params:
        out["router_config"] = params["fabric_router_config"]
    return out


def cache_defaults(env=os.environ) -> dict:
    """The stack's caches under the serving launcher's per-model cache dir (tt-inference-server exports
    ``TT_CACHE_PATH``): the reference-format tile cache ``glm53_tile_cache_v3`` (built on the first boot, read
    afterwards: mk_sources.MkTileCacheSource) and the permuted expert copies ``glm53_fused_experts_cache``
    (ep_experts under GLM53_EP_KEEP_PERMUTED) -- unless the environment names other locations; nothing without
    TT_CACHE_PATH (the e2e test names its caches itself)."""
    root = env.get("TT_CACHE_PATH")
    if not root:
        return {}
    want = {
        "GLM53_MK_TILE_CACHE_V3": str(Path(root) / "glm53_tile_cache_v3"),
        "GLM53_FUSED_EXPERTS_CACHE": str(Path(root) / "glm53_fused_experts_cache"),
    }
    return {k: v for k, v in want.items() if not env.get(k)}


def _snapshot_dir(hf_config, default_snapshot_dir) -> Path:
    """The checkpoint directory: the HF config's ``_name_or_path`` when it is a snapshot, else the default."""
    name = getattr(hf_config, "_name_or_path", "") or ""
    p = Path(name)
    if name and p.is_dir() and (p / "config.json").exists():
        return p
    return default_snapshot_dir()


def trim_page_rows(table) -> torch.Tensor:
    """The plugin's block table ``[B, blocks]`` (input_batch.py:789-808: zero past each row's live blocks; 0 is vLLM's
    null block, never allocated) -> the generator's negative-terminated rows: every column from the first zero at
    column >= 1 on becomes -1. ``Glm53Generator._write_page_rows`` cuts a row at its first negative entry; a zero
    there would be the null block in a covered column (``allocate_kv_cache(null_block=0)`` refuses it) and would also
    defeat the unchanged-row fast path (``_rows_unchanged`` wants a negative right after the coverage). Column 0 is
    kept as given: a live row never holds the null block there, and a dead row (``start_pos`` -1) is never read."""
    t = torch.as_tensor(table).to(torch.int64).clone()
    if t.dim() == 1:
        t = t[None]
    if t.shape[1] > 1:
        pad = (t[:, 1:] == 0).cumsum(dim=1) > 0
        t[:, 1:] = torch.where(pad, torch.full_like(t[:, 1:], -1), t[:, 1:])
    return t


def row_params(sampling_params, r: int) -> SamplingParams:
    """Row ``r`` of the plugin's per-row lists (``TTSamplingParams``, model_input.py:69-84; on device-sampling calls
    the fields are lists, a None seed = vLLM gave none) -> the generator's ``SamplingParams``: temperature 0 or top_k 1
    is greedy (``is_greedy``: the head's argmax, the only device route of this adapter); vLLM's "no top-k" (0 / -1)
    becomes None. Every field is a per-row list on the plugin's object and only ``seed`` carries None (vLLM gave no
    seed), so a missing field or a short list is a contract break and raises (AttributeError / IndexError)."""
    t = float(sampling_params.temperature[r])
    k = int(sampling_params.top_k[r])
    seed = sampling_params.seed[r]
    return SamplingParams(
        temperature=t,
        top_k=None if k <= 0 else k,
        top_p=float(sampling_params.top_p[r]),
        seed=None if seed is None or int(seed) < 0 else int(seed),
    )


def _is_host_output(out) -> bool:
    """async_decode.py:126-133 ``_is_host_decode_output``: a tensor or a tuple of tensors / None needs no read."""
    return isinstance(out, torch.Tensor) or (
        isinstance(out, tuple) and all(t is None or isinstance(t, torch.Tensor) for t in out)
    )


def resident_blocks(obm, blocks) -> bool:
    """Whether the logical blocks ``blocks`` (a page-table row's first columns, in column order) hold device content.
    The sequence-sharded pool (``Glm53Generator.dsa_shard_map``: tt/owner_blocks.py's OwnerBlockMap) keeps a cached
    block mapped to a physical block of its column's owner chip and evicts the least recently released ones when a chip
    runs dry: an evicted block has no mapping. On the replicated pool (None) vLLM's cached blocks are always resident.
    A null / missing entry (<= 0) never is."""
    for j, b in enumerate(int(v) for v in blocks):
        if b <= 0:
            return False
        if obm is not None:
            g = obm.physical(b)
            if g is None or g % obm.C != obm.owner(j):
                return False
    return True


class StackPrefixCache:
    """GLM53_STACK_PREFIX=1: the adapter's prefix cache -- tt/prefix_cache.py's ``PrefixStore`` bookkeeping over the
    backend's device snapshot pool (``GLM53_PREFIX_DEVICE_ENTRIES`` entries, the virtual slots max_batch + k). vLLM
    shares a cached prefix's latent pages (its block pool; ``Glm53Generator.shared_pages``); the KDA state and the DSA
    pooled keys / ring are per slot, so a hit needs a snapshot taken exactly at its position:
      snapshot  a prefill chunk that ends on the grid (the backend's prefill chunk, 8,192 tokens with the TP prefill:
                vLLM cuts long prompts there, ``prefill_chunk_alignment``) also writes the slot's state into a reserved
                pool entry inside its prefill trace (``snapshot_after``), registered under the chained hash of tokens
                [0, end) once the call returned; a prefix registered already is not snapshotted again; the store
                reuses its least recently used unpinned entry when the pool is full; above ``PREFIX_HEAD`` tokens a
                request keeps only its latest snapshot (the next turn of a conversation extends its whole prompt):
                its previous one is evicted when the next commits, unless a slot reads or a hit pins it;
      confirm   (the plugin's scheduler, before it commits vLLM's hit) the longest registered position <= vLLM's
                proposal whose pages are still resident, else 0; that entry is pinned (never reused) until the
                request's prefill attaches it;
      attach    a NEW request admitted at start > 0 (vLLM's num_computed_tokens = the confirmed position) starts from
                the entry (``prefix_attach``) on vLLM's cached blocks; without an entry at exactly that position or with
                a block of [0, start) evicted meanwhile it is prefilled from position 0 instead (the plugin passes the
                whole history; vLLM's accounting is unaffected);
      release   the request's slot detaches from its entry.
    Image requests (GLM53StackForConditionalGeneration) take no part: the key hashes token ids, which carry no pixels
    (every image row is the same ``<|image|>`` id), so ``confirm`` refuses a prompt with mm items and ``plan`` neither
    attaches nor snapshots the rows its caller names in ``skip`` (rows holding vision tokens).
    ``confirm`` runs on vLLM's scheduler thread, the rest on the worker's: one lock around the store."""

    def __init__(self, gen, kv, max_batch: int):
        from models.autoports.zai_org_glm_5_3_flash.tt.prefix_cache import BOUNDARY, PrefixStore, signature

        entries = int(getattr(gen.backend, "snapshot_entries", 0) or 0)
        if entries <= 0:
            raise ValueError("GLM53_STACK_PREFIX=1 needs the device snapshot pool (GLM53_PREFIX_DEVICE_ENTRIES > 0)")
        self.gen, self.kv = gen, kv
        self.store = PrefixStore(
            max_batch=max_batch,
            device_entries=entries,
            boundary=BOUNDARY,
            sig=signature(model="GLM-5.3-Flash", adapter="stack", max_batch=max_batch),
        )
        self.lock = threading.Lock()
        self.pins: OrderedDict = OrderedDict()  # prompt fingerprint -> the key of the entry its confirmed hit reads
        self.pin_cap = 4 * int(max_batch)  # hits confirmed but never scheduled (aborted while waiting) expire
        self._pending: list = []  # the current prefill call's reserved snapshots: (key, n_tokens, pages, entry, slot)
        self.head = PREFIX_HEAD  # snapshots at or below it are kept; above it one (the latest) per request
        self.tail: dict = {}  # generator slot -> the key of its request's latest snapshot above the head
        self.stats = {
            k: 0
            for k in ("confirmed", "confirmed_tokens", "not_resident", "attaches", "fallbacks", "snapshots",
                      "snapshots_skipped", "mm_excluded", "tail_superseded")  # fmt: skip
        }

    @property
    def grid(self) -> int:
        """the snapshot grid: the backend's prefill chunk (the TP prefill's largest bucket: known after the warm-up)"""
        return int(self.gen.backend.prefill_chunk_tokens)

    # ---- pins: a confirmed hit's entry is never reused before its request's prefill attaches it
    def _pin(self, fp: int, e) -> None:
        e.readers += 1
        self.pins[fp] = e.key
        while len(self.pins) > self.pin_cap:
            _, k = self.pins.popitem(last=False)
            self._unpin(k)

    def _unpin(self, key: str) -> None:
        e = self.store.entries.get(key)
        if e is not None and e.readers > 0:
            e.readers -= 1

    def _take_pin(self, key: str) -> None:
        fp = next((f for f, k in self.pins.items() if k == key), None)
        if fp is not None:
            del self.pins[fp]
            self._unpin(key)

    # ---- the plugin's hit confirmation (scheduler thread)
    def confirm(self, prompt_token_ids, mm_items, num_computed_tokens: int) -> int:
        """the cached tokens the device serves for vLLM's proposed hit of ``num_computed_tokens`` (0: none)"""
        if int(num_computed_tokens) <= 0 or mm_items:
            return 0
        ids = [int(t) for t in prompt_token_ids]
        fp = hash(tuple(ids))
        obm = self.gen.dsa_shard_map(self.kv)
        with self.lock:
            old = self.pins.pop(fp, None)
            if old is not None:  # the same prompt confirmed again (the scheduler retries a waiting request)
                self._unpin(old)
            cap = int(num_computed_tokens)
            while cap >= self.store.boundary:
                pos, e = self.store.longest_cached(ids, max_tokens=cap)
                if e is None:
                    break
                if e.tier == "device" and resident_blocks(obm, e.pages[: -(-pos // BLOCK_SIZE)]):
                    self._pin(fp, e)
                    self.stats["confirmed"] += 1
                    self.stats["confirmed_tokens"] += pos
                    return pos
                self.stats["not_resident"] += 1
                cap = pos - 1  # a shorter registered prefix may still be resident
        return 0

    # ---- one prefill call (worker thread)
    def plan(self, tokens: torch.Tensor, rows: torch.Tensor, gslots: list, starts: list, ends: list, skip=()):
        """-> (the rows' starts: 0 for a hit that cannot attach; the generator keywords ``prefix_attach`` /
        ``snapshot_after``). A row at start > 0 in a generator slot that holds no live request is a NEW request vLLM
        admitted with its first ``start`` tokens cached (a continuation's slot is live). ``tokens`` = every row's
        history from position 0, ``rows`` = the trimmed page-table rows. Rows in ``skip`` (image requests) neither
        attach (one admitted at start > 0 is prefilled from 0; the multimodal plan already re-runs those) nor
        snapshot."""
        starts = [int(v) for v in starts]
        attach, snap = {}, {}
        obm = self.gen.dsa_shard_map(self.kv)
        live = set(self.gen.live_slots(self.kv))
        with self.lock:
            self._pending = []
            for i, (g, s0) in enumerate(zip(gslots, starts)):
                if s0 <= 0 or g in live:
                    continue
                if i in skip:  # an image request: never confirmed (confirm refuses mm items)
                    starts[i] = 0
                    self.stats["fallbacks"] += 1
                    logger.warning(
                        f"GLM53 stack prefix: generator slot {g}'s hit at {s0}: an image request, prefilled from 0"
                    )
                    continue
                key = self.store.key(tokens[i, :s0].tolist(), s0) if s0 % self.store.boundary == 0 else None
                e = self.store.entries.get(key) if key is not None else None
                if key is not None:
                    self._take_pin(key)
                blocks = rows[i, : -(-s0 // BLOCK_SIZE)].tolist()
                if e is not None and e.tier == "device" and e.n_tokens == s0 and resident_blocks(obm, blocks):
                    self.store.attach(key, g)
                    attach[i] = int(e.device_entry)
                    self.stats["attaches"] += 1
                    logger.info(
                        f"GLM53 stack prefix: generator slot {g} attaches snapshot entry {e.device_entry} at {s0}"
                    )
                else:
                    starts[i] = 0
                    self.stats["fallbacks"] += 1
                    why = "no snapshot at that position" if e is None else "a cached block was evicted"
                    logger.warning(f"GLM53 stack prefix: generator slot {g}'s hit at {s0}: {why}, prefilled from 0")
            planned = set()
            for i, (s0, end) in enumerate(zip(starts, ends)):
                end = int(end)
                if end % self.grid or end % self.store.boundary or end <= s0:
                    continue
                if i in skip:  # an image request's state depends on pixels its token-id key does not carry
                    self.stats["mm_excluded"] += 1
                    continue
                key = self.store.key(tokens[i, :end].tolist(), end)
                if key in self.store.entries or key in planned:
                    self.stats["snapshots_skipped"] += 1
                    continue
                ent = self.store.reserve_device()  # the least recently used unpinned entry when the pool is full
                if ent is None:
                    self.stats["snapshots_skipped"] += 1
                    continue
                planned.add(key)
                snap[i] = int(ent)
                self._pending.append((key, end, rows[i, : -(-end // BLOCK_SIZE)].tolist(), int(ent), int(gslots[i])))
        extra = {}
        if attach:
            extra["prefix_attach"] = attach
        if snap:
            extra["snapshot_after"] = snap
        return starts, extra

    def commit(self) -> None:
        """the prefill call returned: its snapshots exist on the device"""
        with self.lock:
            for key, n, pages, ent, g in self._pending:
                self.store.register(key, n_tokens=n, pages=pages, kind="P", device_entry=ent)
                self.stats["snapshots"] += 1
                if n > self.head:  # the request's previous tail snapshot is superseded (free unless read / pinned)
                    prev = self.tail.get(g)
                    old = self.store.entries.get(prev) if prev is not None and prev != key else None
                    if old is not None and old.readers == 0 and old.n_tokens > self.head:
                        self.store.evict(prev)
                        self.stats["tail_superseded"] += 1
                    self.tail[g] = key
                logger.info(
                    f"GLM53 stack prefix: snapshot of tokens [0, {n}) in entry {ent} "
                    f"({len(self.store.entries)} of {self.store.device_entries} entries registered)"
                )
            self._pending = []

    def abort(self) -> None:
        """the prefill call raised: its reserved entries were never written"""
        with self.lock:
            for _key, _n, _pages, ent, _g in self._pending:
                self.store.unreserve(ent)
            self._pending = []

    def release(self, gslot: int) -> None:
        with self.lock:
            self.store.release_slot(int(gslot))
            self.tail.pop(int(gslot), None)  # a later request in the slot starts its own chain

    def report(self) -> dict:
        with self.lock:
            return {**self.store.report(), "pins": len(self.pins), **self.stats}


class GLM53StackForCausalLM:
    """GLM-5.3-Flash text generation on the persistent stack (one Blackhole Galaxy mesh) for the TT vLLM plugin."""

    # DECODE_RELOAD_CONTRACT.md / async_decode.py:1398-1412: the four explicit reload commands are accepted
    decode_input_update_contract = 1

    model_capabilities = {
        # MODEL_CAPABILITIES.md:56-59 scheduling flags: a prompt split across engine steps resumes through the
        # generator's continuation chunks (64-aligned, 128 with the TP prefill's SDPA: ``continuation_align``)
        "supports_chunked_prefill": True,
        # GLM53_STACK_PREFIX=1: vLLM's prefix cache, its hits confirmed against the device snapshots (StackPrefixCache)
        "supports_prefix_caching": PREFIX_CACHE,
        "confirms_prefix_cache_hits": PREFIX_CACHE,
        # MODEL_CAPABILITIES.md:85-91 sampling / async flags: the head's argmax is the only device sampler
        # (``StackBackend(topk=1, sample=False)``), so greedy rows (and top_k == 1) decode on device and every other
        # sampled row goes to the host sampler over full logits; no resident-input chain (every decode reloads)
        "supports_sample_on_device": True,
        "max_device_top_k": DEVICE_TOP_K,
        "device_top_k_unbounded_as": DEVICE_TOP_K if DEVICE_TOP_K > 1 else None,
        "supports_device_penalties": False,
        "supports_device_grammar": False,
        "supports_async_decode": ASYNC_DECODE,
        "supports_async_spec_decode": SPEC_ASYNC,
        "supports_compact_host_logits": False,
        "supports_selective_host_readback": False,
        # MODEL_CAPABILITIES.md:126 device setup: worker.py:274-281 reads the class attribute before it opens the mesh
        "fabric_config": _fabric_capability(),
        # SPEC_DECODE_CONTRACT.md: the stack's speculative step serves one live user in generator slot 0 (GLM53_STACK_VLLM_SPEC,
        # default off): the model-owned drafter (custom_class) proposes on device; its state stays on device (no hidden
        # feed); synchronous scheduling (--no-async-scheduling) unless GLM53_STACK_VLLM_SPEC_ASYNC=1 declares the
        # deferred verify readback (supports_async_spec_decode above: spec_async_enabled)
        "supports_spec_decode": SPEC,
        "spec_requirements": ["device_propose"] if SPEC else [],
    }

    # model_runner.py:2301-2320: the one captured decode width (fixed user slots, no smaller bucket); set by
    # ``warmup_model_decode(enable_trace=True)``, cleared by ``release_persistent_capture``
    tt_supported_decode_batch_sizes: tuple[int, ...] = ()
    _HYBRID_KV_CACHE_GROUPS_ENABLED = False  # worker.py:799-807: one logical KV group
    supports_multimodal = False  # vLLM's ``SupportsMultiModal`` flag: text only

    def __init__(self, generator, max_batch_size: int, max_seq_len: int, *, source=None, spec: bool | None = None,
                 spec_async: bool | None = None, **vllm_kwargs):  # fmt: skip
        # ``**vllm_kwargs`` (``vllm_config`` / ``prefix``) is never used: the plugin constructs the adapter through
        # ``initialize_vllm_model``; vLLM >= 0.30's ``is_vllm_model`` wants the keyword accepted (see the shims below)
        self.gen = generator
        self.max_batch = int(max_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.source = source
        self.vocab_size = int(generator.vocab_size)
        self.kv_cache = None
        self._alias = list(range(self.max_batch))  # plugin slot -> generator slot (identity until the first gather)
        self._chain = []  # generator slots of the last reload step's decode rows: the resident chain's batch
        self.prefix = None  # GLM53_STACK_PREFIX: the StackPrefixCache of the allocated cache
        # Stage 17 (SPEC): the speculative step of the lone live greedy request in generator slot 0 behind the plugin's
        # verify / propose calls; the greedy flag of every generator slot (the last params the plugin sent for it)
        self.spec = SPEC if spec is None else bool(spec)
        self._greedy: dict = {}
        self._spec_n = None  # the last fused verify's accepted drafts (its proposal must continue exactly that)
        self._no_spec: set = set()  # generator slots whose request never speculates (_spec_admit)
        self.spec_async = self.spec and (SPEC_ASYNC if spec_async is None else bool(spec_async))
        self._spec_pending = None  # the outstanding deferred verify (SpecReadback) until _spec_collect
        self._spec_lock = threading.Lock()  # (the plugin may finalize a verify off the engine thread)
        self.spec_stats = dict(fused=0, fallback=0, committed=0, enters=0, proposals=0, declines=0, seeds=0,
                               deferred=0)  # fmt: skip

    # ================================================================================= construction
    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size: int,
        max_seq_len: int,
        n_layers: int | None = None,
        tt_data_parallel: int = 1,
        optimizations: str | None = None,
    ):
        """loader.py:38-45: ``initialize_vllm_model(hf_config, device, max_batch_size, max_seq_len=max_model_len,
        tt_data_parallel=..., optimizations=...)`` on the already-open mesh. ``max_num_seqs`` is the backend's
        construction ``max_batch`` (<= 32; the trace sizes, sparse ladder and buckets are construction-time: another
        ``max_model_len`` is a rebuild); ``tt_data_parallel`` must be 1 (one engine per mesh). The validated 8x4
        knobs come from ``galaxy_defaults.apply`` (an explicit environment value always wins), the weights from
        ``mk_sources.install_weight_source`` (GLM53_MK_TILE_CACHE_V3, built on a cold cache), the backend as the e2e
        test builds it (topk 1, sample False, EP experts, native_t above 4 users). ``optimizations`` is ignored."""
        if int(tt_data_parallel) != 1:
            raise ValueError("GLM53StackForCausalLM runs one engine per mesh (tt_data_parallel must be 1)")
        B, seq = int(max_batch_size), int(max_seq_len)
        if not 1 <= B <= MAX_DECODE_SLOTS:
            raise ValueError(f"max_num_seqs {B} outside [1, {MAX_DECODE_SLOTS}] (the stack's decode slots)")
        caches = cache_defaults()
        os.environ.update(caches)  # before the stack modules are imported (their own defaults read the environment)
        from models.autoports.zai_org_glm_5_3_flash.tt import galaxy_defaults as GD
        from models.autoports.zai_org_glm_5_3_flash.tt.config import load_hf_text_config
        from models.autoports.zai_org_glm_5_3_flash.tt.generator import Glm53Generator
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_sources import MkTileCacheSource, install_weight_source
        from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import StackBackend
        from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer
        from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir

        snapshot = _snapshot_dir(hf_config, default_snapshot_dir)
        shape = tuple(int(v) for v in list(mesh_device.shape))
        applied = GD.apply(shape, seq=seq)  # (its prefill buckets are the ones that fit `seq`)
        logger.info(
            f"GLM53 stack vLLM: mesh {shape}, max_num_seqs {B}, max_model_len {seq}, galaxy defaults {applied}, "
            f"caches {caches or 'from the environment'}"
        )
        cfg = load_hf_text_config(str(snapshot))
        source = install_weight_source(snapshot, cfg)
        layers = list(range(int(n_layers) if n_layers else int(cfg.num_hidden_layers)))
        # the narrow widths (GLM53_STACK_NARROW_WIDTHS) reuse the cached layer weights: an empty dict this build fills
        narrow = os.environ.get("GLM53_STACK_NARROW_WIDTHS", "none") not in ("", "none")
        t0 = time.perf_counter()
        backend = StackBackend(
            mesh_device,
            max_batch=B,
            layers=layers,
            hf_config=cfg,
            topk=DEVICE_TOP_K,
            sample=DEVICE_TOP_K > 1,
            ep_weights=True,
            native_t=B > 4,
            weight_cache={} if narrow else None,
        )
        if isinstance(source, MkTileCacheSource) and not source.ready:
            source.complete()  # built by this construction: a later boot reads every tensor from the cache
        logger.info(f"GLM53 stack vLLM: StackBackend built in {time.perf_counter() - t0:.0f} s ({len(layers)} layers)")
        gen = Glm53Generator(backend, tokenizer=load_tokenizer(snapshot), max_batch=B, max_seq_len=seq)
        return cls(gen, B, seq, source=source)

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs: int, requested_k: int):
        """SPEC_DECODE_CONTRACT.md section 2 (config time, before any instance; never raises): ``spec_plan_for``"""
        from models.demos.glm53_flash.tt.vllm_spec import contract_types

        return spec_plan_for(int(max_num_seqs), int(requested_k), contract_types())

    # ================================================================================= KV cache
    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        **kwargs,
    ) -> int:
        """worker.py:737-785: the all-user token budget vLLM sizes its block pool from (plus one block per user on
        top): every slot may hold ``max_model_len`` latent pages, so ``max_num_seqs x max_model_len``."""
        if max_model_len is None or max_num_seqs is None:
            raise ValueError("get_max_tokens_all_users needs max_model_len and max_num_seqs")
        return int(max_num_seqs) * int(max_model_len)

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """worker.py:349-416 / 630-652: one logical full-attention group of latent pages, ``[64, 1, 512]`` bf16 per
        block, so vLLM's admission accounting counts what the stack pages; the device cache is model-owned. The block
        size must be the stack's 64-token page (``--block-size 64``)."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        block_size = int(vllm_config.cache_config.block_size)
        if block_size != BLOCK_SIZE:
            raise ValueError(f"--block-size {block_size} must be {BLOCK_SIZE}: the stack's latent page is 64 tokens")
        return {
            "glm53_latent": FullAttentionSpec(
                block_size=block_size, num_kv_heads=1, head_size=LATENT_DIM, dtype=torch.bfloat16
            )
        }

    def allocate_kv_cache(self, shape, dtype, num_layers: int):
        """model_runner.py:786-816: ``allocate_kv_cache(shape=(num_blocks, heads, block_size, head), dtype,
        num_layers)``; the return value comes back as ``kv_cache=`` on every hook (model_runner.py:750). The
        generator binds the cache by identity (``_binding``: ``id(kv)``), so the ``StackCache`` itself is returned;
        ``num_blocks`` sizes the page pool, block 0 is vLLM's null block (the tail of every page-table row)."""
        num_blocks, _, block_size, _ = (int(v) for v in shape)
        if block_size != BLOCK_SIZE:
            raise ValueError(f"vLLM block size {block_size} != the stack's {BLOCK_SIZE}")
        self.kv_cache = self.gen.allocate_kv_cache(
            self.max_batch, self.max_seq_len, num_blocks=num_blocks, null_block=0
        )
        self._alias = list(range(self.max_batch))  # a fresh cache: every generator slot is free
        self._chain = []
        self.prefix = None
        if PREFIX_CACHE:  # live requests may share the fully computed pages of a cached prefix (read-only)
            self.prefix = StackPrefixCache(self.gen, self.kv_cache, self.max_batch)
            self.gen.shared_pages = True
        return self.kv_cache

    def _kv(self, kv_cache):
        kv = kv_cache if kv_cache is not None else self.kv_cache
        if kv is None:
            raise RuntimeError("allocate_kv_cache must run before this hook")
        return kv

    # ================================================================================= warm-up
    def warmup_model_prefill(self, kv_cache=None, enable_trace: bool = True, can_sample_on_device: bool = False, **kw):
        """model_runner.py:4770-4799 phase 1 (``enable_trace=False``) -> ``warmup_serving(phase="prepare")``: every
        prefill bucket and the decode compiled eagerly, the idle state restored; phase 2 (``enable_trace=True``, before
        the decode hook's) -> ``phase="capture"``: the full-width decode trace and, behind it, the TP prefill traces
        (``StackBackend.capture_decode``). Both phases are idempotent in the generator, so the decode hook's calls are
        no-ops after these."""
        self.gen.warmup_serving(self._kv(kv_cache), phase="capture" if enable_trace else "prepare")

    def warmup_model_decode(
        self,
        kv_cache=None,
        enable_trace: bool = True,
        max_batch_size: int | None = None,
        num_blocks: int | None = None,
        can_sample_on_device: bool = False,
        **kw,
    ):
        """model_runner.py:4770-4799: the same two phases (prepared / captured already by ``warmup_model_prefill`` when
        the plugin traces prefill too; ``trace_mode=decode_only`` captures here). After the capture the one decode
        width is declared (model_runner.py:2301-2320: every decode is padded to ``max_num_seqs`` rows)."""
        if max_batch_size is not None and int(max_batch_size) != self.max_batch:
            raise ValueError(
                f"warmup_model_decode max_batch_size {max_batch_size} != the construction's {self.max_batch}"
            )
        self.gen.warmup_serving(self._kv(kv_cache), phase="capture" if enable_trace else "prepare")
        if enable_trace:
            self.tt_supported_decode_batch_sizes = (self.max_batch,)

    # ================================================================================= prefill
    def prefill_forward(
        self,
        tokens: torch.Tensor,
        page_table=None,
        kv_cache=None,
        enable_trace: bool = True,
        prompt_lens=None,
        start_pos=None,
        empty_slots=None,
        sampling_params=None,
        **kwargs,
    ):
        """model_runner.py:4009-4063 / MODEL_EXECUTION.md:37-61: row ``i`` of ``tokens [B, T]`` (history from position
        0) is the chunk ``tokens[i, start_pos[i]:prompt_lens[i]]`` of the request in plugin slot ``empty_slots[i]``
        (allocated by the plugin: model_runner.py:1299-1364; a first chunk takes the lowest free generator slot, a
        continuation keeps its slot: ``_admit``). The generator takes chunk-local tokens with their length and the
        absolute start (a continuation at the slot's cached length, 64 / 128-aligned), page-table rows trimmed to the
        live blocks. Returns ``(ids [B, 1], None)`` with ``sampling_params`` (device greedy: the first token
        sampled inside the prefill trace; every row must be greedy, which the plugin guarantees for
        ``max_device_top_k=1``) or ``logits [B, 1, V]`` fp32 for host sampling, which the plugin also selects when a
        row is an intermediate chunk (its head output is discarded)."""
        mm = [k for k in kwargs if k.startswith(("pixel_values", "image_", "video_", "audio_", "mm_"))]
        if mm and not self.supports_multimodal:
            raise NotImplementedError(f"multimodal prompt on the text-only stack adapter ({mm[0]})")
        kv = self._kv(kv_cache)
        self._spec_settle()
        tokens = torch.as_tensor(tokens)
        if tokens.dim() == 1:
            tokens = tokens[None]
        B = int(tokens.shape[0])
        if prompt_lens is None:
            raise ValueError("prefill_forward needs prompt_lens (the exclusive end of every row's chunk)")
        ends = [int(v) for v in prompt_lens]
        starts = [0] * B if start_pos is None else [int(v) for v in torch.as_tensor(start_pos).reshape(-1).tolist()]
        slots = list(range(B)) if empty_slots is None else [int(s) for s in empty_slots]
        if len(ends) != B or len(starts) != B or len(slots) != B:  # one entry per row (MODEL_EXECUTION.md "Prefill")
            raise ValueError(f"prefill rows {B} vs prompt_lens {len(ends)} start_pos {len(starts)} slots {len(slots)}")
        lens = [e - s for s, e in zip(starts, ends)]
        if any(n < 1 for n in lens) or any(e > int(tokens.shape[1]) for e in ends):
            raise ValueError(
                f"prefill chunks [start, end) {list(zip(starts, ends))} outside tokens [{B}, {tokens.shape[1]}]"
            )
        # GLM53StackForConditionalGeneration (``_mm_plan``): the rows' image feature rows, an off-grid continuation
        # re-run from position 0, the rows of failed image requests; the text class never gets here with one
        plan = self._mm_plan(tokens, starts, ends, slots, kwargs) if mm else None
        gen_starts = starts if plan is None else plan["starts"]
        lens = [e - s for s, e in zip(gen_starts, ends)]
        chunk = torch.zeros(B, max(lens), dtype=torch.int64)
        for i in range(B):
            chunk[i, : lens[i]] = tokens[i, gen_starts[i] : ends[i]].to(torch.int64)
        if page_table is None:
            raise ValueError("prefill_forward needs the rows' page table")
        rows = trim_page_rows(page_table)
        if len(rows) < B:
            raise ValueError(f"prefill page_table has {len(rows)} rows for {B} prompts")
        rows = rows[:B]
        params = None
        if sampling_params is not None:
            params = [row_params(sampling_params, i) for i in range(B)]
            bad = [] if DEVICE_TOP_K > 1 else [i for i, p in enumerate(params) if not p.is_greedy]
            if bad:
                raise ValueError(
                    f"prefill rows {bad} are sampled (temperature > 0 without top_k 1): the stack samples greedily on "
                    "device only; the plugin routes such rows to host sampling under max_device_top_k=1"
                )
        if self.spec:  # Stage 17: another slot's prefill would overwrite the step's candidate KDA rows -> leave first
            self._spec_stop(kv)
        gslots = self._admit(kv, slots, starts)  # (a re-run continuation keeps its slot: admitted at its own start)
        extra = {"mm_rows": plan["rows"]} if plan is not None and plan["rows"] else {}
        if self.prefix is not None:  # GLM53_STACK_PREFIX: attach confirmed hits, snapshot the on-grid chunk ends
            skip = plan["vision"] if plan is not None else ()  # image requests neither attach nor snapshot
            gen_starts, pfx = self.prefix.plan(tokens, rows, gslots, gen_starts, ends, skip=skip)
            extra.update(pfx)
            lens = [e - s for s, e in zip(gen_starts, ends)]
            chunk = torch.zeros(B, max(lens), dtype=torch.int64)
            for i in range(B):
                chunk[i, : lens[i]] = tokens[i, gen_starts[i] : ends[i]].to(torch.int64)
        # Stage 17: a chunk of generator slot 0 (the slot that can speculate) also fills the MTP layer's cache (the
        # TP prefill's in-trace fill, StackBackend.arm_mtp_prefill) from its history; a chunk's last row (the next
        # chunk's first token unknown here) and the prompt's last one (the seed below) are left to the step. A prefix
        # attach / snapshot of this call happens inside it, after the _spec_stop above; a request that attached or
        # holds image rows never speculates (_spec_admit) and is not armed
        if self.spec:
            self._spec_admit(gslots, gen_starts, extra.get("prefix_attach", {}), plan["vision"] if plan else ())
        armed = False
        if self.spec and self.spec_multi_capable and callable(getattr(self.gen.backend, "arm_mtp_prefill", None)):
            # GLM53_SPEC_MULTI: every group base of the call fills its user's MTP cache
            bases = set(self.gen.spec_slots())
            for i, g in enumerate(gslots):
                if g in bases and g not in self._no_spec:
                    armed = bool(self.gen.backend.arm_mtp_prefill(g, tokens[i, : ends[i]].to(torch.int64))) or armed
        elif (self.spec and 0 in gslots and 0 not in self._no_spec
                and callable(getattr(self.gen.backend, "arm_mtp_prefill", None))):  # fmt: skip
            i0 = gslots.index(0)
            armed = bool(self.gen.backend.arm_mtp_prefill(0, tokens[i0, : ends[i0]].to(torch.int64)))
        try:
            out = self.gen.prefill_forward(
                chunk,
                page_table=rows,
                kv_cache=kv,
                prompt_lens=lens,
                empty_slots=gslots,
                start_pos=gen_starts,
                sampling_params=params,
                **extra,
            )
        except Exception:
            if self.prefix is not None:
                self.prefix.abort()
            raise
        finally:
            if armed:
                self.gen.backend.disarm_mtp_prefill()
        if self.prefix is not None:
            self.prefix.commit()
        if self.spec:
            for i, g in enumerate(gslots):
                self._greedy[g] = params is not None and params[i].is_greedy
            if params is not None:
                self._spec_seed(kv)
        if plan is not None and plan["failed"]:
            return self._mm_failed_rows(out, plan["failed"], [gslots[i] for i in plan["failed"]], params is not None)
        if params is not None:
            return out.reshape(B, 1).to(torch.int32), None
        return out.reshape(B, 1, -1).float()

    # ================================================================================= decode
    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table=None,
        kv_cache=None,
        enable_trace: bool = True,
        read_from_device: bool = True,
        sampling_params=None,
        slot_remap=None,
        *,
        reload_inputs: bool = True,
        reload_page_table: bool = False,
        reload_sampling_params: bool = False,
        reset_sampling_state: bool = False,
        **kwargs,
    ):
        """async_decode.py:1335-1440: one decode step of ``tokens [B_pad, 1]`` / ``start_pos [B_pad]`` (-1 on padding
        rows, ``B_pad`` = the declared width) where row ``i`` is plugin slot ``i`` once ``slot_remap`` (row ``i`` reads
        plugin slot ``remap[i]``, a full permutation: model_runner.py:1366-1421) is applied -- absorbed into the slot
        alias (``_permuted``): every row decodes in the generator slot its request lives in (``slots=``), nothing is
        copied, and the alias advances only once the generator accepted the step (the plugin commits its own map then:
        ``note_decode_state_slots_settled``). ``reload_inputs`` is always True for this adapter (async_decode.py:423-481: no ``supports_async_decode``), so tokens / positions /
        page rows are restaged from the host (the generator writes only what changed). Device route (``sampling_params``
        given): ``read_from_device=False`` returns the generator's ``DecodeOutput`` handle (DP=1 always sends it:
        model_runner.py:4106-4108), ``True`` the ids ``[B_pad, 1]``. Host route (no ``sampling_params``): the logits
        are read back regardless (the generator needs ``read_from_device=True`` without a sampler) and returned as
        ``(logits [B_pad, 1, V] fp32, None)``, which the plugin reads as ``tt_out[rows, -1, :]``
        (model_runner.py:4239-4245)."""
        if "reset_batch" in kwargs:
            raise TypeError("legacy reset_batch keyword: this adapter declares decode_input_update_contract = 1")
        self._spec_settle()
        if kwargs.get("spec_mode") is not None or kwargs.get("num_valid_drafts") is not None:
            if not self.spec:
                raise NotImplementedError(
                    f"speculative decode is not enabled on the stack adapter ({SPEC_ENV}=1 with GLM53_SPEC_K)"
                )
            if not reload_inputs:
                raise ValueError("a verify reloads its inputs (DECODE_RELOAD_CONTRACT.md: its block is new every step)")
            return self._verify(
                tokens, start_pos, num_valid_drafts=kwargs.get("num_valid_drafts"), spec_mode=kwargs.get("spec_mode"),
                page_table=page_table, kv_cache=kv_cache, enable_trace=enable_trace, sampling_params=sampling_params,
                slot_remap=slot_remap, reload_page_table=reload_page_table,
                reload_sampling_params=reload_sampling_params, reset_sampling_state=reset_sampling_state,
                read_from_device=read_from_device,
            )  # fmt: skip
        if self.spec:  # an ordinary decode continues every row from its committed state: no speculation under it
            self._spec_stop(self._kv(kv_cache))
        if not reload_inputs:
            if not ASYNC_DECODE:
                raise ValueError(
                    "reload_inputs=False: the stack adapter declares no resident-input chain (supports_async_decode is "
                    "False), every decode reloads its inputs"
                )
            if slot_remap is not None or reload_sampling_params or reset_sampling_state or sampling_params is None:
                raise ValueError(
                    "a resident chain step (reload_inputs=False) carries no slot remap and no sampling reload / reset, "
                    "and is device-sampled (DECODE_RELOAD_CONTRACT.md: those come with reload_inputs)"
                )
            return self._chain_decode(
                self._kv(kv_cache),
                int(torch.as_tensor(tokens).numel()),
                page_table if reload_page_table else None,
                enable_trace,
                read_from_device,
            )
        return self._reload_decode(
            self._kv(kv_cache), tokens, start_pos, page_table, enable_trace, read_from_device, sampling_params,
            slot_remap, reload_page_table, reload_sampling_params, reset_sampling_state,
        )  # fmt: skip

    def _reload_decode(self, kv, tokens, start_pos, page_table, enable_trace, read_from_device, sampling_params,
                       slot_remap, reload_page_table, reload_sampling_params, reset_sampling_state):  # fmt: skip
        """decode_forward's reload step (every input restaged from the host): see decode_forward"""
        tokens = torch.as_tensor(tokens).reshape(-1).to(torch.int64)
        positions = torch.as_tensor(start_pos).reshape(-1).to(torch.int64)
        B = int(tokens.numel())
        if int(positions.numel()) != B:
            raise ValueError(f"tokens {B} rows vs start_pos {int(positions.numel())}")
        if B > self.max_batch:
            raise ValueError(f"{B} decode rows exceed the {self.max_batch} slots")
        rows = None
        if page_table is not None:
            rows = trim_page_rows(page_table)
            if int(rows.shape[0]) < B:
                raise ValueError(f"page_table {int(rows.shape[0])} rows < {B} decode rows")
            rows = rows[:B]
        alias = self._permuted(slot_remap)
        common = dict(
            page_table=rows,
            kv_cache=kv,
            slots=alias[:B],
            enable_trace=enable_trace,
            reload_inputs=True,
            reload_page_table=reload_page_table,
            reload_sampling_params=reload_sampling_params,
            reset_sampling_state=reset_sampling_state,
        )
        if sampling_params is None:
            logits = self.gen.decode_forward(tokens, positions, sampling_params=None, read_from_device=True, **common)
            self._alias = alias
            self._chain = [alias[i] for i in range(B) if int(positions[i]) >= 0]
            return logits.reshape(B, 1, -1).float(), None
        params = [row_params(sampling_params, i) for i in range(B)]
        bad = (
            [] if DEVICE_TOP_K > 1 else [i for i, p in enumerate(params) if not p.is_greedy and int(positions[i]) >= 0]
        )
        if bad:
            raise ValueError(
                f"decode rows {bad} are sampled (temperature > 0 without top_k 1): the stack samples greedily on device "
                "only; the plugin routes such rows to host sampling under max_device_top_k=1"
            )
        out = self.gen.decode_forward(
            tokens, positions, sampling_params=params, read_from_device=read_from_device, **common
        )
        self._alias = alias
        self._chain = [alias[i] for i in range(B) if int(positions[i]) >= 0]
        if self.spec:
            for i in range(B):
                if int(positions[i]) >= 0:
                    self._greedy[alias[i]] = params[i].is_greedy
        if isinstance(out, DecodeOutput):
            return out
        return self.gen.process_decode_output_host(out, is_tokens=True).to(torch.int32)

    def _chain_decode(self, kv, B: int, page_table, enable_trace: bool, read_from_device: bool):
        """A resident chain step: the last reload step's decode slots (``_chain``; a request mid-prefill holds an active
        generator slot but is not in the plugin's decode batch until a reload adds it) that are still live decode from
        their device-resident token and position (``Glm53Generator.decode_forward(reload_inputs=False, slots=...)``), in
        the generator's slot order. The plugin reads its
        rows in its own order, so the output rows are the alias (plugin row ``i`` = generator slot ``alias[i]``; the
        layout is the last reload step's: a chain step carries no remap); a page-table refresh
        (``reload_page_table``) is reordered from plugin rows to the live slots the same way."""
        b = self.gen._binding(kv)
        live = [s for s in self._chain if b.slots[s].active and not b.slots[s].finished]
        if not live:
            raise ValueError("a resident chain step (reload_inputs=False) without a live decode row from a reload step")
        live.sort()
        pt = None
        if page_table is not None:
            rows = trim_page_rows(page_table)
            row_of = {self._alias[i]: i for i in range(min(B, int(rows.shape[0])))}
            missing = [s for s in live if s not in row_of]
            if missing:
                raise ValueError(f"page-table refresh without rows for the live generator slots {missing}")
            pt = torch.stack([rows[row_of[s]] for s in live]) if live else None
        empty = torch.zeros(0, dtype=torch.int64)
        out = self.gen.decode_forward(
            empty,
            empty,
            page_table=pt,
            kv_cache=kv,
            slots=live,
            enable_trace=enable_trace,
            sampling_params=[],
            read_from_device=read_from_device,
            reload_inputs=False,
            reload_page_table=pt is not None,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        order = self._alias[:B]
        if isinstance(out, DecodeOutput):
            return dataclasses.replace(out, rows=order)
        by_slot = dict(zip(live, torch.as_tensor(out).reshape(-1).tolist()))
        ids = torch.tensor([by_slot.get(s, 0) for s in order], dtype=torch.int32)
        return ids.reshape(B, 1)

    # ================================================================================= Stage 17: speculation
    # SPEC_DECODE_CONTRACT.md with the model-owned drafter: the backend's speculative step (tt/spec_mtp.py: the verify
    # of [b, d1..dK], the accept on device, the MTP drafts of the next step -- ONE trace replay) serves the lone live
    # greedy request in generator slot 0. propose_draft_tokens enters it (the bootstrap drafts) or hands out the drafts
    # the last fused step made; a verify of exactly those drafts runs the step (its K + 1 argmax ids: the plugin's walk
    # accepts what the device accepted); any other verify (more live rows, a truncated / foreign block) leaves the
    # speculation and answers column 0 with an ordinary decode (a one-column argmax: one token per row committed).
    # With the prefix cache / image input (both opt-in): every prefill leaves the speculation before its prefix attach /
    # snapshot (the generator refuses a prefill while slot 0 speculates), and a request that attached a prefix-cache
    # entry or holds image rows never speculates (_spec_admit: its MTP cache would lack rows) -- its steps decline.
    def _spec_stop(self, kv) -> None:
        """leave the speculation (plain decode inputs restored from the device's committed state), if any"""
        if getattr(self.gen._binding(kv), "spec", None) is not None:
            self.gen.spec_leave(kv)
        self._spec_n = None

    def _spec_lone(self, kv) -> bool:
        """generator slot 0 holds the only live request, a greedy one that may speculate, and the next step fits the
        context"""
        b = self.gen._binding(kv)
        live = [s for s in range(b.max_batch) if b.slots[s].active and not b.slots[s].finished]
        return (live == [0] and bool(self._greedy.get(0, False)) and 0 not in self._no_spec
                and self.gen.spec_room(kv))  # fmt: skip

    @property
    def spec_multi_capable(self) -> bool:
        """GLM53_SPEC_MULTI with the backend's multi-user step (tt/spec_mtp.py groups: G = GLM53_SPEC_MULTI_USERS users
        on the group bases, one step) -- the opt-in plan's multi-row verifies run on it"""
        return bool(self.spec) and self.gen.spec_groups() > 0

    def _spec_multi_ok(self, kv) -> bool:
        """GLM53_SPEC_MULTI v1 gate: the live requests are exactly the G group bases, every one greedy and allowed to
        speculate, and every user's next step fits (the context, GLM53_SPEC_MAX_CTX: Glm53Generator.spec_room)"""
        b = self.gen._binding(kv)
        live = [s for s in range(b.max_batch) if b.slots[s].active and not b.slots[s].finished]
        bases = list(self.gen.spec_slots())
        return (bool(bases) and live == bases and all(bool(self._greedy.get(s, False)) for s in bases)
                and not set(bases) & self._no_spec and self.gen.spec_room(kv))  # fmt: skip

    def _spec_gate(self, kv) -> bool:
        return self._spec_multi_ok(kv) if self.spec_multi_capable else self._spec_lone(kv)

    def _spec_admit(self, gslots, starts, attach, vision) -> None:
        """the speculation verdict of the requests one prefill call admits or continues: a request whose MTP cache
        would lack rows never speculates (its steps decline: plain decode) --
          a prefix-cache attach (GLM53_STACK_PREFIX: ``prefix_attach``) -- its [0, start) came from a snapshot entry
            and vLLM's shared blocks; the MTP cache (private to generator slot 0) never saw those positions for it;
          an image request (``vision``: image tokens in its history, a failed one included) -- the MTP-cache fill
            embeds token ids, so its image positions would carry the placeholder's embedding, not the feature rows;
        a fresh text request (generator start 0) clears its slot's verdict, a continuation keeps it."""
        for i, g in enumerate(gslots):
            if i in attach or i in vision:
                self._no_spec.add(int(g))
            elif int(starts[i]) == 0:
                self._no_spec.discard(int(g))

    def _spec_seed(self, kv) -> None:
        """right after a lone greedy request's prefill (its first token sampled, the head's h = the last prompt row's):
        one bootstrap fills the MTP cache's row L - 1 (the first token after the prompt: the prefill's in-trace fill
        stops at L - 2) and is left again (GLM53_STACK_VLLM_SPEC_SEED=0: off) -- the plugin's first decode is an
        ordinary one, after which the speculation enters at L"""
        if os.environ.get("GLM53_STACK_VLLM_SPEC_SEED", "1") != "1" or not self._spec_gate(kv):
            return
        try:
            self.gen.spec_enter(kv)
        except ValueError as e:  # (page coverage / context: no seed; the speculation simply starts without it)
            logger.warning(f"GLM53 stack spec: no MTP seed ({e})")
            return
        self.gen.spec_leave(kv)
        self.spec_stats["seeds"] += 1

    def _verify(self, tokens, start_pos, *, num_valid_drafts, spec_mode, page_table, kv_cache, enable_trace,
                sampling_params, slot_remap, reload_page_table, reload_sampling_params, reset_sampling_state,
                read_from_device=True):  # fmt: skip
        """SPEC_DECODE_CONTRACT.md 4a: ``tokens`` / ``start_pos`` ``[B, 1+K]`` (column 0 the committed token, 1..K the
        drafts), ``num_valid_drafts [B]`` -> ``VerifyOutput(argmax_ids)``: the fused step's ``[B, 1+K]`` (return
        column j = the model's choice after column j; the lone row's K + 1 verify argmax ids, padding rows 0) when the
        block is exactly the speculating request's (its bonus at its cache length, the K drafts the device made, no
        other live row); else ``[B, 1]`` from an ordinary decode of column 0 (the plugin commits one token per row:
        model_runner trims the drafts to the answered width). ``read_from_device=False`` on a ``spec_async`` adapter
        (the plugin's async verify): the fused step's ids come back as a ``SpecReadback`` (read_decode_output /
        process_decode_output_host complete it); the fallback's ids are host ids (no read event)."""
        from models.demos.glm53_flash.tt.vllm_spec import contract_types

        if spec_mode != "argmax_ids":
            raise NotImplementedError(f"spec_mode {spec_mode!r}: the stack adapter verifies in argmax_ids only")
        kv = self._kv(kv_cache)
        tok = torch.as_tensor(tokens).to(torch.int64)
        pos = torch.as_tensor(start_pos).to(torch.int64)
        if tok.dim() != 2 or tuple(pos.shape) != tuple(tok.shape):
            raise ValueError(
                f"a verify block is [B, 1+K] tokens and positions, got {tuple(tok.shape)} / {tuple(pos.shape)}"
            )
        B, W = (int(v) for v in tok.shape)
        nv = torch.as_tensor(num_valid_drafts).reshape(-1).to(torch.int64)
        if int(nv.numel()) != B:
            raise ValueError(f"num_valid_drafts {int(nv.numel())} rows vs the block's {B}")
        VerifyOutput = contract_types()["VerifyOutput"]
        alias = self._permuted(slot_remap)
        fused = self._fused_rows if self.spec_multi_capable else self._fused_row  # (GLM53_SPEC_MULTI: every user's row)
        r = fused(kv, alias, tok, pos, nv)
        if r is not None:
            rows = trim_page_rows(page_table)[:B] if page_table is not None else None
            if isinstance(r, dict):  # GLM53_SPEC_MULTI: {plugin row: group base} -> the users' rows
                row = None if rows is None else {g: rows[i] for i, g in r.items()}
            else:
                row = None if rows is None else rows[r]
            self._alias, self._chain = alias, []
            if (
                self.spec_async and not read_from_device
            ):  # the deferred readback: the record's copy rides behind the step
                self._spec_pending = rb = SpecReadback(self.gen.spec_step_submit(kv, page_row=row), r, B, W)
                return VerifyOutput(spec_mode="argmax_ids", argmax_ids=rb)
            return VerifyOutput(spec_mode="argmax_ids", argmax_ids=self._spec_ids(self.gen.spec_step(kv, page_row=row),
                                                                                  r, B, W))  # fmt: skip
        # not the speculating request's block: the ordinary decode of column 0, one committed token per row
        self._spec_stop(kv)
        self.spec_stats["fallback"] += 1
        out = self._reload_decode(kv, tok[:, 0], pos[:, 0], page_table, enable_trace, True, sampling_params, slot_remap,
                                  reload_page_table, reload_sampling_params, reset_sampling_state)  # fmt: skip
        if isinstance(out, tuple):  # host route: (logits [B, 1, V], None) -> the greedy rows' argmax
            ids = out[0][:, 0, : self.vocab_size].argmax(dim=-1).to(torch.int32)
        else:
            ids = torch.as_tensor(out).to(torch.int32)
        return VerifyOutput(spec_mode="argmax_ids", argmax_ids=ids.reshape(B, 1))

    def _fused_row(self, kv, alias, tok, pos, nv):
        """the plugin row of the speculating request when the verify block is exactly its next step (else None): one
        live row, in generator slot 0, column 0 = the step's bonus at the slot's cache length, the K consecutive
        positions after it, and all K of the drafts the device made"""
        b = self.gen._binding(kv)
        sp = getattr(b, "spec", None)
        K = self.gen.spec_k()
        B, W = (int(v) for v in tok.shape)
        live = [i for i in range(B) if int(pos[i, 0]) >= 0]
        if sp is None or len(live) != 1 or W != K + 1:
            return None
        r = live[0]
        p0 = int(pos[r, 0])
        if alias[r] != 0 or int(nv[r]) != K or p0 != int(b.slots[0].seq_len) or int(tok[r, 0]) != int(sp["bonus"]):
            return None
        if [int(v) for v in tok[r, 1:].tolist()] != [int(v) for v in sp["drafts"]]:
            return None
        if [int(v) for v in pos[r].tolist()] != list(range(p0, p0 + W)):
            return None
        return r

    def _fused_rows(self, kv, alias, tok, pos, nv):
        """GLM53_SPEC_MULTI: {plugin row: group base} when the verify block is exactly the G users' next step (every
        live row a group base's request, all G of them, each _fused_row's conditions against its own bonus / drafts /
        cache length), else None"""
        b = self.gen._binding(kv)
        sp = getattr(b, "spec", None)
        K = self.gen.spec_k()
        B, W = (int(v) for v in tok.shape)
        live = [i for i in range(B) if int(pos[i, 0]) >= 0]
        bases = list(self.gen.spec_slots())
        if sp is None or "groups" not in sp or W != K + 1 or sorted(alias[i] for i in live) != bases:
            return None
        out = {}
        for i in live:
            g, p0 = alias[i], int(pos[i, 0])
            mine = sp["groups"][g]
            if int(nv[i]) != K or p0 != int(b.slots[g].seq_len) or int(tok[i, 0]) != int(mine["bonus"]):
                return None
            if [int(v) for v in tok[i, 1:].tolist()] != [int(v) for v in mine["drafts"]]:
                return None
            if [int(v) for v in pos[i].tolist()] != list(range(p0, p0 + W)):
                return None
            out[i] = g
        return out

    def _spec_ids(self, rec: dict, r, B: int, W: int) -> torch.Tensor:
        """a fused step's record -> the verify's ``[B, W]`` argmax ids (row r: the K + 1 verify argmax, padding rows 0)
        + the adapter's bookkeeping (the accepted count its proposal must continue, the stats). GLM53_SPEC_MULTI: r =
        {plugin row: group base}, rec = {group base: record}"""
        if isinstance(r, dict):
            ids = torch.zeros(B, W, dtype=torch.int32)
            self._spec_n = {}
            for i, g in r.items():
                ids[i] = torch.tensor([int(v) for v in rec[g]["a"][:W]], dtype=torch.int32)
                self._spec_n[g] = int(rec[g]["n"])
                self.spec_stats["committed"] += int(rec[g]["n"]) + 1
            self.spec_stats["fused"] += 1
            if self.spec_stats["fused"] % 256 == 0:
                st = self.spec_stats
                logger.info(f"GLM53 stack spec (multi): {st} (tau per user "
                            f"{st['committed'] / max(1, st['fused'] * len(r)):.2f})")  # fmt: skip
            return ids
        ids = torch.zeros(B, W, dtype=torch.int32)
        ids[r] = torch.tensor([int(v) for v in rec["a"][:W]], dtype=torch.int32)
        self._spec_n = int(rec["n"])
        self.spec_stats["fused"] += 1
        self.spec_stats["committed"] += int(rec["n"]) + 1
        if self.spec_stats["fused"] % 256 == 0:
            st = self.spec_stats
            logger.info(f"GLM53 stack spec: {st} (tau {st['committed'] / max(1, st['fused']):.2f})")
        return ids

    def _spec_collect(self, rb: SpecReadback) -> torch.Tensor:
        """complete a deferred verify (once; later calls return its ids)"""
        with self._spec_lock:
            if rb.ids is None:
                rb.ids = self._spec_ids(self.gen.spec_step_collect(rb.handle), rb.row, rb.B, rb.W)
                self.spec_stats["deferred"] += 1
            if self._spec_pending is rb:
                self._spec_pending = None
            return rb.ids

    def _spec_settle(self) -> None:
        """every entry point that reads or moves slot state first completes an outstanding deferred verify (the plugin
        completes it at its finalize; this covers a caller that never did)"""
        if self._spec_pending is not None:
            self._spec_collect(self._spec_pending)

    def propose_draft_tokens(self, num_drafts, committed_tokens, committed_positions, accepted_counts, hidden=None):
        """SPEC_DECODE_CONTRACT.md 4b (after every commit): ``DraftOutput([B, K] ids, num_valid [B])`` -- K drafts for
        the lone live greedy request in generator slot 0 (the bootstrap's when it is not speculating yet, else the ones
        the last fused step made, which continue exactly its committed block), 0 for every row otherwise (declining
        leaves the speculation: the next step is an ordinary decode, or the plugin's one verify that resolves a
        multi-token commit, answered by column 0)"""
        from models.demos.glm53_flash.tt.vllm_spec import PLACEHOLDER_TOKEN_ID, contract_types

        DraftOutput = contract_types()["DraftOutput"]
        K = int(num_drafts)
        ct = torch.as_tensor(committed_tokens).to(torch.int64)
        cp = torch.as_tensor(committed_positions).to(torch.int64)
        ac = torch.as_tensor(accepted_counts).reshape(-1).to(torch.int64)
        B = int(ct.shape[0])
        drafts = torch.full((B, K), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        nvalid = torch.zeros(B, dtype=torch.int32)
        if not self.spec or self.kv_cache is None:
            return DraftOutput(draft_token_ids=drafts, num_valid=nvalid)
        self._spec_settle()
        kv = self.kv_cache
        b = self.gen._binding(kv)
        live = [i for i in range(B) if int(cp[i, 0]) >= 0]
        if self.spec_multi_capable:
            return self._propose_multi(K, ct, cp, ac, b, kv, live, drafts, nvalid, DraftOutput)
        r = live[0] if len(live) == 1 else None
        ok = r is not None and self._alias[r] == 0 and K == self.gen.spec_k() and self._spec_lone(kv)
        if ok:
            last = int(ac[r]) - 1
            # the block's last committed token sits at the slot's cache length (the next input: model_runner
            # _committed_positions -- column j at the verify input's position + 1 + j)
            ok = int(cp[r, last]) == int(b.slots[0].seq_len)
        sp = getattr(b, "spec", None)
        if ok and sp is not None:  # (_spec_n None: entered, no step yet -- a repeated proposal of the same state)
            if (self._spec_n is not None and last != self._spec_n) or int(ct[r, last]) != int(sp["bonus"]):
                raise RuntimeError(
                    f"proposal after a fused verify: the plugin committed {last + 1} tokens ending {int(ct[r, last])}, "
                    f"the device {self._spec_n + 1 if self._spec_n is not None else None} ending {sp['bonus']}"
                )
            d = list(sp["drafts"])
        elif ok:
            try:
                d = self.gen.spec_enter(kv)
            except ValueError as e:  # (a refusal of the generator's: page coverage / context -> decline this step)
                logger.warning(f"GLM53 stack spec: no bootstrap ({e})")
                ok = False
            else:
                self._spec_n = None
                self.spec_stats["enters"] += 1
        if not ok:
            self._spec_stop(kv)
            self.spec_stats["declines"] += 1
            return DraftOutput(draft_token_ids=drafts, num_valid=nvalid)
        drafts[r] = torch.tensor([int(v) for v in d[:K]], dtype=torch.int32)
        nvalid[r] = K
        self.spec_stats["proposals"] += 1
        return DraftOutput(draft_token_ids=drafts, num_valid=nvalid)

    def _propose_multi(self, K, ct, cp, ac, b, kv, live, drafts, nvalid, DraftOutput):
        """propose_draft_tokens of GLM53_SPEC_MULTI: K drafts for every live row when the live requests are exactly the
        G greedy users on the group bases (_spec_multi_ok) and each row's committed block ends at its slot's cache
        length -- the bootstrap's drafts (entering) or the last fused step's, which continue exactly its committed
        block per user; else every row declines (the speculation is left: the next step decodes plainly)"""
        rows = {i: self._alias[i] for i in live}
        ok = K == self.gen.spec_k() and sorted(rows.values()) == list(self.gen.spec_slots()) and self._spec_multi_ok(kv)
        last = {i: int(ac[i]) - 1 for i in live}
        if ok:
            ok = all(int(cp[i, last[i]]) == int(b.slots[g].seq_len) for i, g in rows.items())
        sp = getattr(b, "spec", None)
        d = None
        if ok and sp is not None:  # (_spec_n None: entered, no step yet -- a repeated proposal of the same state)
            for i, g in rows.items():
                n = self._spec_n.get(g) if isinstance(self._spec_n, dict) else None
                if (n is not None and last[i] != n) or int(ct[i, last[i]]) != int(sp["groups"][g]["bonus"]):
                    raise RuntimeError(
                        f"proposal after a fused verify (slot {g}): the plugin committed {last[i] + 1} tokens ending "
                        f"{int(ct[i, last[i]])}, the device {None if n is None else n + 1} ending {sp['groups'][g]['bonus']}"
                    )
            d = {g: list(sp["groups"][g]["drafts"]) for g in rows.values()}
        elif ok:
            try:
                d = self.gen.spec_enter(kv)
            except ValueError as e:  # (a refusal of the generator's: page coverage / context -> decline this step)
                logger.warning(f"GLM53 stack spec: no multi-user bootstrap ({e})")
                ok = False
            else:
                self._spec_n = None
                self.spec_stats["enters"] += 1
        if not ok:
            self._spec_stop(kv)
            self.spec_stats["declines"] += 1
            return DraftOutput(draft_token_ids=drafts, num_valid=nvalid)
        for i, g in rows.items():
            drafts[i] = torch.tensor([int(v) for v in d[g][:K]], dtype=torch.int32)
            nvalid[i] = K
        self.spec_stats["proposals"] += 1
        return DraftOutput(draft_token_ids=drafts, num_valid=nvalid)

    # ================================================================================= slot aliasing
    def _permuted(self, slot_remap) -> list:
        """The alias after the plugin's gather ``slot_remap`` (row ``i`` reads plugin slot ``remap[i]``: the one
        encoding the plugin sends, a full-width int tensor that permutes every slot, model_runner.py:1366-1421), the
        current alias without one. Row ``i`` keeps the generator slot its request already decodes in, so no state
        moves."""
        n = self.max_batch
        if slot_remap is None:
            return self._alias
        remap = [int(x) for x in torch.as_tensor(slot_remap).reshape(-1).tolist()]
        if len(remap) != n or sorted(remap) != list(range(n)):
            raise ValueError(f"slot_remap {remap} is not a full-width permutation of the {n} slots")
        return [self._alias[s] for s in remap]

    def _admit(self, kv, slots: list, starts: list) -> list:
        """The generator slots of a prefill batch's plugin slots. A new request (chunk start 0 in a plugin slot whose
        generator slot holds none) takes the lowest free generator slot: its alias entry is swapped with the free
        plugin slot that held that generator slot, invisible to the plugin, so the live generator slots stay as low as
        the plugin's rows (the generator's decode width covers slots 0..max live). A continuation keeps its slot."""
        n = self.max_batch
        if any(not 0 <= s < n for s in slots):
            raise ValueError(f"empty_slots {slots} outside [0, {n})")
        live = set(self.gen.live_slots(kv))
        alias, taken = list(self._alias), set(live)
        for s, start in zip(slots, starts):
            g = alias[s]
            # a continuation: its request already holds this generator slot (its first chunk made it live); with the
            # prefix cache a start > 0 in a slot without a live request is a NEW request vLLM admitted with its first
            # `start` tokens cached (StackPrefixCache.plan attaches it), admitted like a first chunk below
            if start > 0 and (self.prefix is None or g in live):
                if g not in live:
                    raise RuntimeError(
                        f"prefill continuation at {start} in plugin slot {s}: generator slot {g} holds no live request "
                        f"(alias {alias}, live {sorted(live)})"
                    )
                taken.add(g)
                continue
            if (
                g in live
            ):  # the plugin never hands a fresh request a held slot (model_runner._alloc_prefill_state_slots)
                raise RuntimeError(
                    f"new request in plugin slot {s}: generator slot {g} still holds a live request (alias {alias}, "
                    f"live {sorted(live)}): the plugin's ownership map and the alias diverged"
                )
            free = [x for x in range(n) if x not in taken]
            if self.spec and self.spec_multi_capable:  # GLM53_SPEC_MULTI: the free group bases first (the users that
                # can speculate sit there: no slot moves in v1)
                bases = set(self.gen.spec_slots())
                free.sort(key=lambda x: (x not in bases, x))
            if not free:
                raise RuntimeError(f"no free generator slot for the prefill batch {slots} (live {sorted(live)})")
            lowest = free[0]
            if lowest != g:
                j = alias.index(lowest)
                alias[s], alias[j] = lowest, g
            taken.add(lowest)
        self._alias = alias
        return [alias[s] for s in slots]

    def read_decode_output(self, tt_out, async_read: bool = False, sample_rows=None):
        """async_decode.py:1474-1486 (the async read of a ``read_from_device=False`` submission): the generator
        enqueues the copy of the sampled rows behind the step and returns ``(PendingTokens, events)``; the plugin waits
        on the events, then ``process_decode_output_host`` yields the ids. A host output passes through. Selective
        readback (``sample_rows``) is not declared."""
        if sample_rows is not None:
            raise NotImplementedError("selective host readback is not declared by the stack adapter")
        if isinstance(tt_out, SpecReadback):  # a deferred verify: the record's event, completed at the finalize
            if async_read:
                return tt_out, ([] if tt_out.ids is not None else list(tt_out.handle.events))
            return self._spec_collect(tt_out)
        if _is_host_output(tt_out):
            return (tt_out, []) if async_read else tt_out
        if not isinstance(tt_out, DecodeOutput):
            raise TypeError(f"read_decode_output expects a DecodeOutput, got {type(tt_out).__name__}")
        return self.gen.read_decode_output(tt_out, async_read=async_read)

    def process_decode_output_host(self, tt_out, is_tokens: bool = False, sample_rows=None):
        """async_decode.py:1651-1666: the ids ``[B_pad, 1]`` int32 of a device-sampled step (padding rows 0). The
        synchronous path hands ``decode_forward``'s raw ``DecodeOutput`` here, so it is read first
        (``Glm53Generator.process_decode_output_host`` takes ``PendingTokens`` or a tensor); a host output passes
        through. Compact host logits (``sample_rows``) are not declared."""
        if sample_rows is not None:
            raise NotImplementedError("compact host logits are not declared by the stack adapter")
        if isinstance(tt_out, SpecReadback):  # a deferred verify -> its [B, 1+K] argmax ids (before the accept walk)
            return self._spec_collect(tt_out)
        if _is_host_output(tt_out):
            return tt_out
        if isinstance(tt_out, DecodeOutput):
            tt_out = self.gen.read_decode_output(tt_out)
        if not isinstance(tt_out, (PendingTokens, torch.Tensor)):
            raise TypeError(
                f"process_decode_output_host expects a DecodeOutput / PendingTokens, got {type(tt_out).__name__}"
            )
        return self.gen.process_decode_output_host(tt_out, is_tokens=True).to(torch.int32)

    # ================================================================================= request lifecycle
    def release_request(self, slot: int) -> None:
        """model_runner.py:965-983 / 2880: the request in plugin ``slot`` (its CURRENT slot, after every gather)
        finished or was preempted -> ``Glm53Generator.release_slot`` on its generator slot: the slot stops decoding,
        its page coverage and (sharded pool) physical blocks are released, the backend's release hook runs once."""
        if self.kv_cache is None:
            raise RuntimeError("release_request before allocate_kv_cache")
        if not 0 <= int(slot) < self.max_batch:
            raise ValueError(f"release_request({slot}) outside the {self.max_batch} slots")
        g = self._alias[int(slot)]
        self._spec_settle()
        if (self.spec and self.spec_multi_capable and g in self.gen.spec_slots()
                and getattr(self.gen._binding(self.kv_cache), "spec", None) is not None):  # fmt: skip
            # GLM53_SPEC_MULTI: the other users' committed KDA rows sit in their candidate rows -> leave (every user's
            # row to its base), not abort; this request's own copy is dead state
            self._spec_stop(self.kv_cache)
        elif self.spec and g == 0 and getattr(self.gen._binding(self.kv_cache), "spec", None) is not None:
            self.gen.spec_abort(self.kv_cache)  # the speculating request ended: its candidate rows are dead state
            self._spec_n = None
        if self.spec and g == 0:  # (the served smoke's evidence: tau = committed tokens per fused step so far)
            st = self.spec_stats
            logger.info(f"GLM53 stack spec: {st} (tau {st['committed'] / max(1, st['fused']):.2f})")
        self._greedy.pop(g, None)
        self._no_spec.discard(g)
        self.gen.release_slot(self.kv_cache, g)
        if self.prefix is not None:  # its prefix-cache entry loses a reader
            self.prefix.release(g)

    def confirm_prefix_cache_hit(self, prompt_token_ids, mm_items=(), num_computed_tokens: int = 0) -> int:
        """vllm-tt-plugin's hit confirmation (``confirms_prefix_cache_hits``; scheduler.py ``_confirmed_computed_blocks``,
        called on vLLM's scheduler thread before the hit is committed): the cached tokens the device serves for vLLM's
        proposed hit of ``num_computed_tokens`` -- the longest snapshot position at or below it whose pages are still
        resident (``StackPrefixCache.confirm``) -- or 0 (no prefix cache, a multimodal prompt, nothing resident)."""
        if self.prefix is None:
            return 0
        return self.prefix.confirm(prompt_token_ids, mm_items, num_computed_tokens)

    def note_state_slots_moved(self, moves) -> None:
        """model_runner.py:282 / MODEL_CAPABILITIES.md:73-75: the plugin's ``old -> new`` notification of the gather
        ``decode_forward`` already absorbed into the slot alias -- nothing to do (no state moved)."""
        return None

    def release_persistent_capture(self) -> None:
        """model_runner.py:525-540 (teardown, before the mesh closes): every captured trace is released
        (``Glm53Generator.release_traces``) and the decode width declaration withdrawn."""
        self.gen.release_traces()
        self.tt_supported_decode_batch_sizes = ()

    # ================================================================================= vLLM protocol shims
    # vLLM >= 0.30 admits the generate runner only for classes passing ``is_vllm_model`` (an ``__init__`` accepting
    # ``vllm_config``, ``embed_input_ids``, ``forward(input_ids, positions)``, ``compute_logits``); never called.
    def embed_input_ids(self, input_ids):  # pragma: no cover - protocol shim
        raise NotImplementedError("TT models do not expose embed_input_ids")

    def forward(self, input_ids, positions, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError("TT models run through prefill_forward / decode_forward")

    def compute_logits(self, hidden_states, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError("TT models return logits from decode_forward")


# ===================================================================================== image input
class HostVisionEncoder:
    """The image encoder of ``GLM53StackForConditionalGeneration`` until the stack carries its own tower: HF's
    ``Glm5NextVisionModel`` in fp32 on the host CPU with the checkpoint's ``model.visual.*`` weights (the construction
    of ``reference/hf_multimodal_reference.build_hf_vision``: the features the HF greedy reference consumed) -> per
    image item ``[n_tokens, 4096]`` rows rounded to bf16 (``Glm5NextModel.forward`` casts them). Built on first use
    (``build()`` at model construction); ~2 s of CPU per 448 x 448 image, the weights ~2.3 GB of host memory."""

    def __init__(self, snapshot: Path, dtype: torch.dtype = torch.float32):
        self.snapshot, self.dtype, self.model = Path(snapshot), dtype, None

    def build(self) -> "HostVisionEncoder":
        if self.model is None:
            from models.demos.glm53_flash.reference.hf_multimodal_reference import build_hf_vision
            from models.demos.glm53_flash.tt.vision import load_vision_weights

            t0 = time.perf_counter()
            self.model = build_hf_vision(self.snapshot, load_vision_weights(self.snapshot), self.dtype)
            logger.info(f"GLM53 stack vLLM: host vision tower ({self.dtype}) built in {time.perf_counter() - t0:.1f} s")
        return self

    @torch.no_grad()
    def __call__(self, items: list) -> list[torch.Tensor]:
        self.build()
        out = []
        for it in items:
            if it.kind != "image":
                raise NotImplementedError(f"the stack adapter encodes images only (item kind {it.kind!r})")
            f = self.model(it.pixel_values.to(self.dtype), grid_thw=it.grid_thw).pooler_output
            out.append(f.float().to(torch.bfloat16))
        return out


class DeviceVisionEncoder:
    """The image encoder on the stack's own mesh (GLM53_STACK_MM_ENCODER=device): ``multimodal.VisionRuntime`` -- the
    resident ``tt/vision.py`` tower, weights replicated on every chip (~1.1 GB each) -- with the L1-bounded programs the
    stack's worker L1 needs (``TtVisionTower(mm_l1_budget=, sdpa_chunks=, split_swiglu=True)``: GLM53_STACK_MM_TOWER_MM_KB
    per-core matmul CBs, default 512; GLM53_STACK_MM_TOWER_SDPA "q,k", default "32,1024"). Construction registers
    ``attach`` as a pre-capture hook of the backend (``StackBackend.pre_capture_hooks``): the tower is built and every
    patch bucket's programs (1024 .. GLM53_STACK_MM_TOWER_PATCHES, default 4096, in steps of 1024) are compiled and run
    once right before the first trace capture, after every other allocation, so no tower buffer or program sits in a
    trace's transients and the text programs and buffers are the text build's. A call (after the captures) settles the
    device, then runs the tower with a host sync before every op launch (GLM53_STACK_MM_TOWER_FENCE=1: a trace replay can
    leave the dispatcher's expected worker count short of the device's, so eager programs right after replays could
    otherwise overlap; ``TPPrefill.quiesce``) and with program-cache misses forbidden (GLM53_STACK_MM_TOWER_SEAL=1: a
    tower program first built after the captures would put its binaries under live traces -- the call fails naming the
    op instead) -> per item ``[n_tokens, 4096]`` bf16 rows (chip 0's; HF's cast)."""

    def __init__(self, backend, snapshot: Path, env=os.environ):
        self.backend, self.mesh, self.snapshot = backend, backend.mesh, Path(snapshot)
        self.max_patches = int(env.get("GLM53_STACK_MM_TOWER_PATCHES", "4096"))
        q, k = (int(v) for v in env.get("GLM53_STACK_MM_TOWER_SDPA", "32,1024").split(","))
        self.tower_kw = dict(mm_l1_budget=int(env.get("GLM53_STACK_MM_TOWER_MM_KB", "512")) * 1024, sdpa_chunks=(q, k),
                             split_swiglu=True)  # fmt: skip
        self.fence = env.get("GLM53_STACK_MM_TOWER_FENCE", "1") == "1"
        self.seal = env.get("GLM53_STACK_MM_TOWER_SEAL", "1") == "1"
        self.runtime, self.warm_s, self.calls, self.encode_s, self.fenced = None, None, 0, 0.0, 0
        backend.pre_capture_hooks.append(self.attach)

    def _l1(self) -> dict:
        import ttnn

        try:
            v = ttnn.get_memory_view(self.mesh, ttnn.BufferType.L1)
            return dict(alloc_per_bank=int(v.total_bytes_allocated_per_bank),
                        largest_free_per_bank=int(v.largest_contiguous_bytes_free_per_bank))  # fmt: skip
        except Exception as e:  # noqa: BLE001  (a diagnostic never stops the build)
            return dict(error=str(e)[:120])

    def attach(self) -> None:
        """the pre-capture hook: build the resident tower and warm every patch bucket (eager; before any capture)"""
        if self.runtime is not None:
            return
        import ttnn
        from models.demos.glm53_flash.tt.multimodal import VisionRuntime

        t0 = time.perf_counter()
        self.runtime = VisionRuntime(self.mesh, self.snapshot, resident=True, max_patches=self.max_patches,
                                     tower_kw=self.tower_kw)  # fmt: skip
        built = time.perf_counter() - t0
        l1 = self._l1()
        self.warm_s = self.runtime.warmup()
        if int(getattr(self.backend, "mm_extra_rows", 0) or 0) > 0:  # the image warm-up's staging half: M1's host write
            self.backend.stage_mm_rows(
                torch.zeros(1, 4096, dtype=torch.bfloat16)
            )  # + SlotOps copy, before the captures
        ttnn.synchronize_device(self.mesh)
        gc.collect()
        self.programs = int(self.mesh.num_program_cache_entries())
        plans = getattr(self.runtime.tower.mm, "plans", {}) if self.runtime.tower is not None else {}
        logger.info(
            f"GLM53 stack vision: tower built in {built:.1f} s, every patch bucket up to {self.max_patches} warmed in "
            f"{self.warm_s:.1f} s before the first capture ({self.tower_kw}); L1 {l1}; matmul plans "
            f"{len(plans)}, largest {max([p_['l1_bytes'] for p_ in plans.values()] or [0])} B per core; "
            f"{self.programs} programs cached"
        )

    @contextlib.contextmanager
    def _fenced(self):
        """a host sync before every ttnn op launch (prefill_tp.install_launch_fence's mechanism, scoped to the call)"""
        if not self.fence:
            yield
            return
        import ttnn.decorators as D

        import ttnn

        call, gop, busy = D.FastOperation.__call__, ttnn.generic_op, [False]

        def sync():
            if not busy[0]:
                busy[0] = True
                try:
                    ttnn.synchronize_device(self.mesh)
                    self.fenced += 1
                finally:
                    busy[0] = False

        def fast_call(op, *a, **k):
            sync()
            return call(op, *a, **k)

        def generic_op(*a, **k):
            sync()
            return gop(*a, **k)

        D.FastOperation.__call__ = fast_call
        if not isinstance(gop, D.FastOperation):
            ttnn.generic_op = generic_op
        try:
            yield
        finally:
            D.FastOperation.__call__ = call
            ttnn.generic_op = gop

    @contextlib.contextmanager
    def _sealed(self):
        if not self.seal:
            yield
            return
        self.mesh.set_program_cache_misses_allowed(False)
        try:
            yield
        finally:
            self.mesh.set_program_cache_misses_allowed(True)

    def __call__(self, items: list) -> list[torch.Tensor]:
        if self.runtime is None:
            raise RuntimeError("the device vision tower is built by the backend's pre-capture hook (warm-up capture)")
        for it in items:
            if it.kind != "image":
                raise NotImplementedError(f"the stack adapter encodes images only (item kind {it.kind!r})")
        import ttnn

        t0 = time.perf_counter()
        ttnn.synchronize_device(self.mesh)  # an in-flight step (a head-less prefill, an async decode) settles first
        before = int(self.mesh.num_program_cache_entries())
        with self._fenced(), self._sealed():
            feats = self.runtime.encode(items)
        ttnn.synchronize_device(self.mesh)
        after = int(self.mesh.num_program_cache_entries())
        if after != before:  # (only reachable with GLM53_STACK_MM_TOWER_SEAL=0)
            logger.error(f"GLM53 stack vision: {after - before} tower program(s) built after the captures")
        gc.collect()
        self.calls += 1
        self.encode_s += time.perf_counter() - t0
        return [torch.as_tensor(f).float().to(torch.bfloat16) for f in feats]


def make_mm_encoder(snapshot: Path, env=os.environ, backend=None):
    """``GLM53_STACK_MM_ENCODER``: ``device`` (the default on a mesh backend) = ``DeviceVisionEncoder`` (built by the
    backend's pre-capture hook), ``hf_cpu`` (the default without a backend) = ``HostVisionEncoder`` (weights loaded
    now), ``none`` = no encoder (an image request is refused)"""
    kind = env.get(MM_ENCODER_ENV) or ("device" if backend is not None else "hf_cpu")
    if kind == "none":
        return None
    if kind == "device":
        if backend is None:
            raise ValueError(f"{MM_ENCODER_ENV}=device needs the stack backend (its mesh and pre-capture hooks)")
        return DeviceVisionEncoder(backend, snapshot, env)
    if kind != "hf_cpu":
        raise ValueError(f"{MM_ENCODER_ENV}={kind!r}: device | hf_cpu | none")
    return HostVisionEncoder(snapshot).build()


# where vLLM keeps the upstream GLM-5.3-Flash processor: glm5next.common.multimodal (0.31), .nvidia.multimodal (0.30)
GLM5NEXT_PROCESSOR_MODULES = ("vllm.models.glm5next.common.multimodal", "vllm.models.glm5next.nvidia.multimodal")


def mm_content_key(item) -> str:
    """An image item's feature-cache key: blake2b-128 over its kind, its processed pixels (dtype, shape, raw bytes) and
    its grid -- equal for identical images under the same processing, whatever identifier vLLM gave the item (a
    per-request uuid without vLLM's processor cache)."""
    pv = item.pixel_values.detach().cpu().contiguous()
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{item.kind}|{pv.dtype}|{tuple(pv.shape)}|{item.grid_thw.reshape(-1).tolist()}".encode())
    h.update(pv.reshape(-1).view(torch.uint8).numpy())
    return h.hexdigest()


def _nbytes(t: torch.Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def _vllm_multimodal_api():
    """vLLM's multimodal declaration pieces in a vLLM process -- ``SupportsMultiModal`` and the registration of
    upstream's ``Glm5NextMultiModalProcessor`` / ``Glm5NextProcessingInfo`` with the Glm4v dummy-inputs builder
    (exactly what upstream's ``Glm5NextForConditionalGeneration`` registers; the module per vLLM version:
    ``GLM5NEXT_PROCESSOR_MODULES``) -- or None: a host process (vLLM not imported: importing it would run its platform
    plugins; the plugin imports vLLM before it resolves the model class) or a vLLM without them."""
    if "vllm" not in sys.modules:
        return None
    try:
        from vllm.model_executor.models.glm4_1v import Glm4vDummyInputsBuilder
        from vllm.model_executor.models.interfaces import SupportsMultiModal
        from vllm.multimodal import MULTIMODAL_REGISTRY
    except ImportError as exc:
        logger.warning(f"GLM53 stack vLLM: multimodal class declared without vLLM's multimodal API ({exc})")
        return None
    errors = []
    for name in GLM5NEXT_PROCESSOR_MODULES:
        try:
            mod = importlib.import_module(name)
            proc, info = mod.Glm5NextMultiModalProcessor, mod.Glm5NextProcessingInfo
        except (ImportError, AttributeError) as exc:
            errors.append(f"{name}: {exc}")
            continue
        return SimpleNamespace(
            SupportsMultiModal=SupportsMultiModal,
            register_processor=MULTIMODAL_REGISTRY.register_processor(
                proc, info=info, dummy_inputs=Glm4vDummyInputsBuilder
            ),
            processor_module=name,
        )
    logger.warning(f"GLM53 stack vLLM: no GLM-5.3-Flash multimodal processor in this vLLM ({'; '.join(errors)})")
    return None


def declare_multimodal_class(api=None):
    """``GLM53StackForConditionalGeneration``: the stack adapter served with image input. A separate class (as
    ``generator_vllm.GLM53FlashForConditionalGeneration``): the text class stays text-only, bit for bit (without
    ``SupportsMultiModal`` vLLM puts no processor in front of it and sends it no pixel lists). Selected with
    ``TT_MODEL_CLASS_OVERRIDES="Glm5NextForConditionalGeneration=<MM_ADAPTER_TARGET>"``.

    Per prefill row with image items (the plugin's per-row ``pixel_values`` / ``image_grid_thw`` lists + placeholder
    ranges + identifiers, ``tt/mm_inputs.py``; ``_mm_plan``): the items' prompt rows (``feature_rows`` on the row's
    prefix), the items whose rows meet this chunk encoded once each (``mm_encoder``: ``DeviceVisionEncoder``; the
    features cached by vLLM's identifier across the request's chunks and identical images), the chunk's rows handed to
    the generator as ``mm_rows`` -> the stack's extra embedding rows (``Glm53Generator._mm_stage``, GLM53_STACK_MM=1, set
    by this class's construction). An off-grid continuation (the plugin leaves an image prompt's chunk ends where vLLM
    put them) re-runs the request from position 0 in its slot; a request whose image input fails (an encoder error, a
    video: images only) ends on a stop token while the engine and the other rows go on.

    ``api``: ``_vllm_multimodal_api()`` (the class gains ``SupportsMultiModal`` and the processor registration, which
    is what makes ``vllm serve`` accept ``image_url`` content) or None (the same flags on a plain subclass)."""
    bases = (GLM53StackForCausalLM,) + ((api.SupportsMultiModal,) if api is not None else ())

    class GLM53StackForConditionalGeneration(*bases):
        """GLM-5.3-Flash text + image generation on the persistent stack (one Blackhole Galaxy mesh)."""

        supports_multimodal = True  # the SupportsMultiModal ClassVar, restated for the vLLM-less declaration
        supports_multimodal_raw_input_only = False
        supports_encoder_tp_data = False
        supports_mm_device_do_normalize = False
        requires_raw_input_tokens = False
        mm_encoder = None  # the image encoder (initialize_vllm_model: make_mm_encoder)

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            # image features by content (``mm_content_key``: the processed pixels + grid): one encoder call per
            # distinct image for every chunk of its request and every request showing it; vLLM's item identifier is
            # an alias of the key (a request's later chunks skip the hash); least recently used out past
            # GLM53_STACK_MM_CACHE_MB of feature rows (default 256 MB: ~32 images of 972 rows), the newest kept
            self._feat_cache: collections.OrderedDict = collections.OrderedDict()  # content key -> [n, 4096] bf16
            self._feat_alias: collections.OrderedDict = collections.OrderedDict()  # identifier -> content key
            self._feat_bytes = 0
            mb = float(os.environ.get(MM_CACHE_ENV, "256"))
            if mb < 0:
                raise ValueError(f"{MM_CACHE_ENV}={mb}: the image feature cache bound must be >= 0 MB")
            self._feat_bytes_cap = int(mb * 2**20)
            self._mm_failed: set = set()  # generator slots of image requests that failed (their chunks run as text)
            self.mm_stats = dict(items_encoded=0, cache_hits=0, content_hits=0, evicted=0, rows_staged=0, restarts=0,
                                 failed=0, encode_s=0.0)  # fmt: skip

        def allocate_kv_cache(self, shape, dtype, num_layers: int):
            self._mm_failed = set()
            return super().allocate_kv_cache(shape, dtype, num_layers)

        def release_request(self, slot: int) -> None:
            if self.kv_cache is not None and 0 <= int(slot) < self.max_batch:
                self._mm_failed.discard(self._alias[int(slot)])
            super().release_request(slot)

        @classmethod
        def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size: int, max_seq_len: int, *a, **kw):
            """The text construction with GLM53_STACK_MM=1 (the table's extra rows; an explicit 0 is refused: this
            class serves images) + the image encoder (``GLM53_STACK_MM_ENCODER``: the device tower by default, built
            by the backend's pre-capture hook during the warm-up's capture phase)."""
            os.environ.setdefault(MM_ENV, "1")
            if os.environ[MM_ENV] != "1":
                raise ValueError(f"{cls.__name__} serves images: {MM_ENV}={os.environ[MM_ENV]!r} (needs 1)")
            from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir

            model = super().initialize_vllm_model(hf_config, mesh_device, max_batch_size, max_seq_len, *a, **kw)
            if int(getattr(model.gen.backend, "mm_extra_rows", 0) or 0) <= 0:
                raise RuntimeError(f"{cls.__name__}: the backend was built without the extra embedding rows")
            model.mm_encoder = make_mm_encoder(
                _snapshot_dir(hf_config, default_snapshot_dir), backend=model.gen.backend
            )
            return model

        def _mm_plan(self, tokens: torch.Tensor, starts: list, ends: list, slots: list, kwargs: dict) -> dict:
            """The multimodal side of one prefill call (``tokens [B, T]`` = every row's history from 0, row i's chunk
            [starts[i], ends[i])): {"starts": the generator's chunk starts, "rows": {row: (positions [m] absolute,
            rows [m, 4096] bf16)}, "failed": rows whose request failed, "vision": rows holding vision tokens (+ the
            failed ones): the prefix cache neither attaches nor snapshots them}.
            - An off-grid continuation (start not a multiple of the backend's continuation_align, 128 with the TP
              prefill: the plugin leaves an image prompt's chunk ends where vLLM put them -- before an image item, at
              the budget) cannot resume the slot's state: the row re-runs [0, end) from position 0 in its own slot (the
              history is in ``tokens``; the earlier images come from the feature cache), wherever the cut lies, an
              image item included.
            - GLM53_STACK_PREFIX: a NEW image request admitted at start > 0 (a prefix hit; never confirmed, as
              ``confirm`` refuses mm items) re-runs [0, end) from 0 too, so every image row before the start is staged.
            - Image rows: every image item whose prompt rows (``feature_rows`` on the row's prefix) meet the chunk is
              encoded once per item (``_mm_features``: cached) and its rows inside the chunk staged (M1).
            - A row whose image input fails (an encoder error, a video, no encoder, no extra rows, inconsistent items)
              fails ITS request, not the engine: the chunk runs as text (its slot stays consistent) and the head
              returns a stop token (``_mm_failed_rows``); its later chunks do the same until the request is released."""
            from models.demos.glm53_flash.tt.mm_inputs import (
                IMAGE_TOKEN_ID,
                feature_rows,
                items_in_range,
                mm_items,
                mm_present,
            )

            align = int(getattr(self.gen.backend, "continuation_align", 1) or 1)
            present = mm_present(kwargs)  # (a text batch on this class arrives with pixel_values=[None] * B)
            live = set(self.gen.live_slots(self.kv_cache)) if self.prefix is not None else None
            gen_starts, out, failed, vision = list(starts), {}, set(), set()
            for i in range(int(tokens.shape[0])):
                s, e = starts[i], ends[i]
                if bool((tokens[i, :e] == IMAGE_TOKEN_ID).any()):  # (video frames use the same id)
                    vision.add(i)
                if s > 0 and s % align:
                    gen_starts[i] = 0
                    self.mm_stats["restarts"] += 1
                    logger.info(
                        f"GLM53 stack MM: plugin slot {slots[i]}: chunk [{s}, {e}) starts off the {align}-token grid: "
                        "re-run from 0"
                    )
                elif s > 0 and i in vision and live is not None and self._alias[slots[i]] not in live:
                    gen_starts[i] = 0
                    self.mm_stats["restarts"] += 1
                    logger.info(
                        f"GLM53 stack MM: plugin slot {slots[i]}: a new image request admitted with [0, {s}) cached: "
                        "prefilled from 0 (image requests take no prefix hit)"
                    )
                if s > 0 and self._alias[slots[i]] in self._mm_failed:
                    failed.add(i)
                    continue
                if not present:
                    continue
                try:
                    rows_i = self._mm_row(tokens, i, gen_starts[i], e, kwargs, (feature_rows, items_in_range, mm_items))
                except Exception as exc:  # noqa: BLE001  (this request fails; the engine and the other rows go on)
                    failed.add(i)
                    self.mm_stats["failed"] += 1
                    logger.error(
                        f"GLM53 stack MM: plugin slot {slots[i]}, chunk [{s}, {e}): image input failed "
                        f"({type(exc).__name__}: {exc}); the request ends with a stop token"
                    )
                    continue
                if rows_i is not None:
                    out[i] = rows_i
                    self.mm_stats["rows_staged"] += int(rows_i[0].numel())
            return dict(starts=gen_starts, rows=out, failed=failed, vision=vision | failed)

        def _mm_row(self, tokens, i: int, start: int, end: int, kwargs: dict, fns) -> tuple | None:
            """row i's (positions, rows) inside [start, end), or None (no image row there)"""
            feature_rows, items_in_range, mm_items = fns
            items = mm_items(kwargs, i)
            if not items:
                return None
            kinds = sorted({it.kind for it in items} - {"image"})
            if kinds:
                raise NotImplementedError(f"{kinds} items: the stack adapter serves images only")
            if int(getattr(self.gen.backend, "mm_extra_rows", 0) or 0) <= 0:
                raise NotImplementedError(f"image input needs the stack's extra embedding rows ({MM_ENV}=1)")
            rows = feature_rows(tokens[i, :end], items)
            need = items_in_range(rows, start, end)
            if not need:
                return None
            if self.mm_encoder is None:
                raise NotImplementedError(f"an image prompt without an image encoder ({MM_ENCODER_ENV}=none)")
            ids = list((kwargs.get("mm_identifiers") or [None] * (i + 1))[i] or [])
            pos, vals = [], []
            for k in need:
                feats = self._mm_features(items[k], ids[k] if k < len(ids) else None)
                r = rows[k]
                f = feats[: int(r.numel())]  # an item the prefix cuts: its rows so far
                sel = (r >= start) & (r < end)
                pos.append(r[sel])
                vals.append(f[sel])
            p_, v_ = torch.cat(pos).to(torch.int64), torch.cat(vals).to(torch.bfloat16)
            order = torch.argsort(p_)
            return p_[order], v_[order]

        def _mm_features(self, item, identifier) -> torch.Tensor:
            """one image item's [n_tokens, 4096] bf16 rows: from the feature cache, else ONE encoder call. The cache
            is keyed by content (``mm_content_key``), so identical images of different requests encode once; vLLM's
            identifier only aliases that key (a request's later chunks find the item without re-hashing it)."""
            alias = str(identifier) if identifier else None
            key = self._feat_alias.get(alias) if alias is not None else None
            hit = self._feat_cache.get(key) if key is not None else None
            if hit is not None and int(hit.shape[0]) == item.n_tokens:
                self._feat_alias.move_to_end(alias)
                self._feat_cache.move_to_end(key)
                self.mm_stats["cache_hits"] += 1
                return hit
            key = mm_content_key(item)
            if alias is not None:
                self._feat_alias[alias] = key
                self._feat_alias.move_to_end(alias)
                while len(self._feat_alias) > MM_ALIAS_CAP:
                    self._feat_alias.popitem(last=False)
            hit = self._feat_cache.get(key)
            grid = item.grid_thw.reshape(-1).tolist()
            if hit is not None:
                self._feat_cache.move_to_end(key)
                self.mm_stats["cache_hits"] += 1
                self.mm_stats["content_hits"] += 1
                logger.info(
                    f"GLM53 stack MM: image item grid {grid} -> {item.n_tokens} rows from the feature cache (the pixels "
                    f"of an earlier item, {key[:12]})"
                )
                return hit
            t0 = time.perf_counter()
            (feats,) = self.mm_encoder([item])
            dt = time.perf_counter() - t0
            feats = torch.as_tensor(feats).to(torch.bfloat16)
            if feats.dim() != 2 or int(feats.shape[0]) != item.n_tokens:
                raise ValueError(f"{tuple(feats.shape)} features for an item of {item.n_tokens} tokens")
            self.mm_stats["items_encoded"] += 1
            self.mm_stats["encode_s"] += dt
            logger.info(
                f"GLM53 stack MM: image item grid {grid} -> {item.n_tokens} rows encoded in {dt * 1e3:.0f} ms "
                f"({type(self.mm_encoder).__name__}, {key[:12]})"
            )
            self._feat_put(key, feats)
            return feats

        def _feat_put(self, key: str, feats: torch.Tensor) -> None:
            """cache a new item's rows, then drop the least recently used past GLM53_STACK_MM_CACHE_MB (never the
            newest: one item larger than the bound is kept until the next one arrives)"""
            self._feat_cache[key] = feats
            self._feat_bytes += _nbytes(feats)
            while len(self._feat_cache) > 1 and self._feat_bytes > self._feat_bytes_cap:
                _, old = self._feat_cache.popitem(last=False)
                self._feat_bytes -= _nbytes(old)
                self.mm_stats["evicted"] += 1

        def _stop_id(self) -> int:
            stops = sorted(int(t) for t in (getattr(self.gen, "stop_token_ids", ()) or ()))
            if not stops:
                raise RuntimeError("a failed image request needs a stop token to end on (the tokenizer has none)")
            return stops[0]

        def _mm_failed_rows(self, out, rows: set, gen_slots: list, sampled: bool):
            """the call's output with every failed row's head on the stop token (vLLM finishes the request there); the
            generator slots are remembered so the request's later chunks fail too (until release_request)"""
            self._mm_failed.update(int(g) for g in gen_slots)
            stop, idx = self._stop_id(), sorted(rows)
            if sampled:
                ids = out.reshape(-1, 1).to(torch.int32).clone()
                ids[idx, 0] = stop
                return ids, None
            logits = out.reshape(int(out.shape[0]), 1, -1).float().clone()
            logits[idx] = -1e9
            logits[idx, 0, stop] = 0.0
            return logits

        @classmethod
        def get_placeholder_str(cls, modality: str, i: int) -> str | None:
            """The chat template's image marker (upstream ``Glm4vForConditionalGeneration``): the processor expands
            ``<|image|>`` to the image tokens"""
            if modality.startswith("image"):
                return "<|begin_of_image|><|image|><|end_of_image|>"
            raise ValueError("Only image modality is supported")

        # vLLM nn.Module multimodal protocol shims (the TT runner hands the raw items to prefill_forward instead)
        def embed_multimodal(self, **kwargs):  # pragma: no cover - protocol shim
            raise NotImplementedError("TT models encode multimodal items inside prefill_forward")

        def get_language_model(self):  # pragma: no cover - protocol shim
            raise NotImplementedError("TT models expose no nn.Module language model")

    GLM53StackForConditionalGeneration.__qualname__ = GLM53StackForConditionalGeneration.__name__
    GLM53StackForConditionalGeneration.__module__ = __name__
    if api is not None:
        GLM53StackForConditionalGeneration = api.register_processor(GLM53StackForConditionalGeneration)
    return GLM53StackForConditionalGeneration


GLM53StackForConditionalGeneration = declare_multimodal_class(_vllm_multimodal_api())
