# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""vLLM serving adapter for MiMo-V2.6-Flash on a Blackhole Galaxy (vllm-tt-plugin, tenstorrent/vllm-tt-plugin#216).

Data parallelism: ``--data-parallel-size 4`` runs as four in-process TT lanes
(``model_capabilities["folds_dp_into_lanes"]``): one engine drives MiMoGalaxyGenerator's four TP=8 replicas, one
per row of the 4x8 mesh. Lane ``l`` owns the plugin's slots [l * per_lane, (l + 1) * per_lane), which are decode
slots 0 .. per_lane - 1 of replica ``l``. Each lane has its own vLLM scheduler and block pool, so block ids are
lane-local and each replica has its own KV storage.

The four lanes of one engine prefill or decode together. ``serving/serve_rows.sh`` runs one engine per row instead
(``--data-parallel-size 1`` on the row's chips, ``TT_VISIBLE_DEVICES``), behind a proxy, so a row prefills while the
others decode. A plugin that runs a model folding DP into lanes as one lane at ``--data-parallel-size 1`` serves each
of them through the same lane path (``dp`` 1).

KV cache: vLLM owns the blocks. ``get_kv_cache_spec`` reports a FullAttentionSpec for the global layers and a
SlidingWindowSpec for the SWA layers (the model attends a window of 128; the spec declares 128 + the chunk start
alignment - 1, ``swa_spec_window``), both K 192 / V 128 at block size 64 (128 for the global layers with
MIMO_KV_PAIR_SPLIT, below). vLLM's hybrid KV manager splits them into KV cache groups of equal size (for 9 global
and 39 SWA layers: one global group and five SWA groups of up to 9 layers), every group with its own block table and
all of them drawing block ids from one pool.
The model gets one page table per group (TtMiMoModel.setup_vllm_kv_cache). A SWA group's table is indexed by
absolute position with the blocks no window can read set to vLLM's null block 0, the layout the model's own
kv.SwaBlockTable keeps.

Storage follows vLLM's sharing rule: the i-th layer of every group shares one buffer (``tensor_idx``), which is
safe because a block id belongs to one group at a time. On a chip every layer's K cache is
[num_blocks, 1, 64, 192] and every layer's V cache [num_blocks, 1, 64, 128] (attention.py keeps V's real dims on
the SWA layers too), each shared that way: 9 K and 9 V buffers for 9 global and 39 SWA layers.

A global layer's KV head is served by a chip pair, and a block can be shared by requests in slots of either parity
(prefix caching). With MIMO_KV_PAIR_SPLIT (the default) the pair splits every block (reference/kv_pair_split.py): the
global group's FullAttentionSpec has block size 128, whose page has the SWA page's bytes (4 KV heads x 128 tokens
against 8 x 64), and each chip keeps its 64-token halves of a block in rows 0..63 of the same physical page, so a
global token costs half a pool token and a lane holds twice the global context (4 x 1M at MIMO_KV_GB_PER_CHIP=6.7).
A later prefill chunk must then start on a 128-token block. vLLM's prefix-cache hits land on its scheduler block,
lcm(64, 128) = 128, and the plugin's TTScheduler ends a partial prefill chunk on its 128-token grid
(``prefill_chunk_alignment``), except for a running request whose leftover step budget cannot reach the grid: that one
gets the shorter chunk, so its next chunk starts off the grid. ``prefill_forward`` moves such a start back to the
block and recomputes the positions in between, which read sliding windows up to 127 positions further back than vLLM
would keep for the scheduled start, so the SWA specs declare a window that much longer (``swa_spec_window``). With
MIMO_KV_PAIR_SPLIT=0 the global group keeps 64-token blocks (its page padded to the SWA page), chunk starts move back to
64-token blocks, and the pair keeps both copies of its KV head (TtMiMoModel.setup_vllm_kv_cache).

Sampling: prefill returns last-token logits [B, 1, vocab] for the host to sample. Decode replays one of two traced
steps per replica (captured in ``warmup_model_decode``) that also computes each slot's token and writes it, with the
slot's position plus one, into the trace's own inputs (TtMiMoModel._feed_back): the greedy step (GreedyHead), or,
while a lane holds a row with temperature > 0, the sampled step (DeviceSampler: temperature, top_k up to
``max_device_top_k``, top_p; its random stream is on device, seeded with the parameters, a seeded row's by (seed,
position)). With ``sample_on_device_mode`` the plugin samples on device unless a row asks for what the sampler lacks
(a top_k past its candidates, pure temperature sampling without top_p or top_k, penalties or logprobs
send the whole batch to host sampling, which reads the logit rows) and overlaps the steps (``decode_input_update_contract`` 1,
``supports_async_decode``): a steady step copies nothing (the trace reads its RoPE
cos / sin from device tables at its own positions, at every context length; with MIMO_DEVICE_ROPE=0 it copies those
of the adapter's own position counter), and its tokens are read back without blocking. Prefill runs
eagerly in chunks of up to 4096 tokens, between the replays. A program first compiled after the capture would keep buffers the replays overwrite, so
``warmup_model_prefill`` compiles every prefill program a served chunk can run first: they key only on the chunk's
padded length and the alignment of its start (attention.py), a bounded set.

Structured output (``supports_device_grammar``, the plugin's docs/DEVICE_GRAMMAR_CONTRACT.md; it needs
``--no-async-scheduling``): vLLM builds a step's grammar bitmask after the step is submitted, so a decode step with a
structured request comes as ``decode_forward(defer_device_sampling=True)``, which keeps the step's arguments
(DeferredDecode), then ``sample_deferred_decode`` with the rows' sampling parameters and the packed mask, which runs the
step: each lane with a constrained row uploads its mask rows and replays the sampled decode trace, whose sampler sets
the logits a row's mask disallows below every other before it draws (DeviceSampler.set_grammar), greedy rows included.
The forward therefore waits for the mask instead of overlapping vLLM's work on it. Structured prefill (the first
token) samples on host. A replica with more than MAX_USERS decode users has no device sampler: such a launch keeps host
grammar (``enable_device_grammar``).

Speculative decoding (MIMO_SPEC_DECODE=mtp; the plugin's model-owned drafter contract, docs/SPEC_DECODE_CONTRACT.md,
with ``--speculative-config '{"method": "custom_class", "model": "vllm_tt_plugin.model_owned_drafter",
"num_speculative_tokens": 3}'``): the MTP layers draft up to MAX_DRAFTS tokens for a lane that ran one request in its
last step (MIMO_MTP_LAYERS=3, the default: draft i + 1 from MTP layer i; 1: layer MIMO_MTP_LAYER drafts every
token), and a verify step runs that request in every decode slot of its replica (reference/spec_decode.py). Lanes
with several requests decline to draft and decode as usual; a verify block that still carries drafts beside other
rows (a request joined after the proposal) is packed into passes of SPEC_SLOTS slots. A verify answers the
plugin's argmax ids when every request of the step is greedy. A step with a sampled request is accepted on device
(``fused_sample``, tt/verify_sampler.py) when every request's sampling is one the device sampler serves (the plugin
decides from ``max_device_top_k`` and ``supports_device_top_p_without_top_k``, and sends no penalties, logprobs,
grammar or token filters there): the plugin sends the requests' temperature / top_k / top_p and the uniforms it drew
from their generators, the device runs the plugin's accept walk on each slot's candidates, and the step returns each
request's committed tokens and count. Otherwise the verify returns its logits and the plugin walks acceptance on the
host with each request's own sampling (temperature, top-p, penalties, ...). Either way sampled requests speculate
losslessly. A verify and a draft each copy one packed int32 tensor of their tokens and positions (plus the RoPE cos /
sin with MIMO_DEVICE_ROPE=0); their traces derive every other input from it on device
(tt/spec_inputs.py). A step that verifies nothing decodes as above, a sampled request's on the sampled decode trace.
The verify and the MTP draft run only as traces, which ``warmup_model_decode`` captures with the decode trace, so
``spec_plan`` refuses a launch whose tt options capture no decode traces (``_untraced_decode``).

The draft is chained behind the verify (tt/spec_chain.py, reference/spec_chain.py): an argmax_ids verify accepts on
device too, every request greedy (its committed tokens are the plugin's greedy walk over the ids), and on a lane that
speculates the host enqueues, without waiting, the verify trace, the accept trace (SampleVerify, then the chain step,
which writes the MTP draft's inputs from the accepted count and committed tokens on device) and the MTP draft trace,
copies one 320-byte result page per pass to the host behind it without blocking (every pass of every lane is
enqueued before the host waits): the argmax ids or the committed tokens and count, and the drafts.
propose_draft_tokens returns those drafts when the plugin committed what the device did; it drafts from the host plan
after a plain decode step, a logits verify, without the decode RoPE tables (MIMO_DEVICE_ROPE=0), or when the commits differ.

The MTP rings only hold what the lane's drafts wrote. With MIMO_MTP_BACKFILL (default on, with the decode RoPE tables)
each decode slot that may speculate keeps a history of the target's final normed hidden (tt/mtp_backfill.py: written
by every decode step, verify pass and prefill chunk), and a draft whose request's rows the rings do not continue
(``mtp_ring``: after the request's prompt, after the lane served several requests) is preceded by the backfill trace,
which fills every MTP layer's ring from that history: its drafts are those of an MTP that saw every position.

Asynchronous scheduling (MIMO_ASYNC_SPEC=1, the default: ``supports_async_spec_decode``). An ordinary step of a
speculating launch overlaps the step in flight as it does without speculation, and the plugin proposes for it when it
commits, a step after its submission. By then the lane's next step may already be enqueued, and the resident hidden
the MTP would read is that step's: such a propose enqueues nothing and offers placeholder drafts (``_stale_source``),
which make the plugin drain the newer step and propose for it, and that propose drafts.

A fused_sample verify asked to defer its read (``defer_read``; MIMO_SPEC_DEFER=0 answers host tensors) returns a
VerifyHandle instead of its tensors: the step's result pages are on their way to the host, and the plugin reads the
handle a step later, possibly on another thread (``read_decode_output``), and proposes from it (the drafts its pages
carry). Any other verify answers its mode's host tensors. The plugin may chain the next verify onto the deferred one
in flight (``chain_rows`` / ``chain_tip``, its section 4j): a chain row's block (the last committed token at the
position after the pending step's commit, then the drafts chained behind it) exists only on device, so the host
uploads the row's pass laid out at the plugin's provisional positions (page tables only) and marks it relayed, and
the verify trace's relay step (tt/spec_chain.py) writes its tokens and positions, and its drafts into the accept
step's control page, from the result page the lane's last pass left. The handle's readout reports the positions the
device used (p0). Every busy lane then runs verify, accept, chained draft, the relay of the next verify, ... with the
host one step behind, reading one page per step.
"""

import math
import os
import threading
import time

import numpy as np
import torch
from loguru import logger

import ttnn
from models.common.sampling.tt_sampling import format_grammar_bitmask

from ..reference import kv_pair_split
from ..reference.checkpoint import MiMoCheckpoint, default_model_path
from ..reference.config import MiMoConfig
from ..reference.sample_verify import commit_pass, pass_inputs
from ..reference.spec_chain import chain_row, chained_draft
from ..reference.spec_decode import (
    DRAFT_GROUPS,
    GROUP_SLOTS,
    MAX_DRAFTS,
    SPEC_SLOTS,
    DraftSource,
    draft_sources,
    plan_mtp,
    plan_verify_passes,
    planned_drafts,
)
from .attention import PAIR_SPLIT_OPS, page_table_width, verify_grouped
from .device_sampler import CANDIDATES_PER_CHIP, MAX_USERS, stream_states
from .generator import MiMoGalaxyGenerator
from .model import PREFILL_CHUNK_TRACE, prefill_padded_len, prefill_warmup_batches, prefill_warmup_chunks

TP = 8
BLOCK_SIZE = 64
# MIMO_KV_PAIR_SPLIT=0: the global layers' chip pair keeps both copies of its KV head (64-token global blocks) instead
# of splitting every 128-token global block between its two chips (module docstring).
KV_PAIR_SPLIT = os.environ.get("MIMO_KV_PAIR_SPLIT", "1") == "1"
# Per-chip bytes per pool token (a 64-token page / 64) of the vLLM KV layout above for MiMo-V2.6-Flash (9 global, 39
# SWA layers, so 9 shared K buffers and 9 shared V buffers), BFP8: (9 * 192 + 9 * 128) values * 1088 / 1024 bytes =
# 3060. A SWA token takes one pool token per SWA group; a global token one, or half of one with MIMO_KV_PAIR_SPLIT
# (a page holds 128 global tokens: kv_pair_split.global_pool_tokens).
KV_BYTES_PER_TOKEN = (9 * 192 + 9 * 128) * 1088 / 1024
# Per-chip DRAM given to the KV pool when MIMO_KV_GB_PER_CHIP is unset (weights take 23.7 of 34 GB per chip): with the
# pair split 6.7, the least that holds 4 requests of 1M per lane (4 x 546,368 pool tokens x 3060 B = 6.69 GB), else 6.
DEFAULT_KV_GB_PER_CHIP = 6.0
PAIR_SPLIT_KV_GB_PER_CHIP = 6.7
# MiMo-V2.6-Flash's sliding window (config.json): a SWA layer's query attends its last 128 positions.
SWA_WINDOW = 128
# MIMO_VLLM_TIMING=1 logs each prefill's host enqueue and total time, decode's host time every 128 steps, and every 128
# speculative steps where a step's time goes (verify, MTP draft, and the host gap between them; _spec_timing).
TIMING = os.environ.get("MIMO_VLLM_TIMING") == "1"
# MIMO_SAMPLER_CHECK=1 (debug): each device-sampled decode step blocks, recomputes every row's draw on the host from the
# step's logit rows and the row's uniform (DeviceSampler.check) and logs the counts every 64 steps.
SAMPLER_CHECK = os.environ.get("MIMO_SAMPLER_CHECK") == "1"
# MIMO_PREFILL_BATCH=0: prefill every row on its own (TtMiMoModel.prefill_enqueue_batch is not used).
PREFILL_BATCH = os.environ.get("MIMO_PREFILL_BATCH", "1") == "1" and os.environ.get("MIMO_SPEC_DECODE") != "dflash"
# MIMO_PREFILL_TRACE=0: run prefill batches eagerly instead of replaying their traces.
PREFILL_TRACE = os.environ.get("MIMO_PREFILL_TRACE", "1") == "1"
# MIMO_PREFILL_CHUNK_TRACE (model.PREFILL_CHUNK_TRACE; default 1, 0 opts out): a lane's later prefill chunks of the
# step's full token budget (padded) replay a trace of the whole chunk (TtMiMoModel.setup_prefill_chunk_traces) instead
# of dispatching its ops, so one host thread keeps every lane busy. A setup or capture that fails (a trace region too
# small for them, say) drops them with a warning: later chunks then run eagerly and the server still starts.
# MIMO_PREFILL_CHUNK_TRACE_SDPA (default "256"): the global SDPA chunk sizes traced (a chunk whose start is an odd
# multiple of 128 runs 128; "256,128" traces those too).
CHUNK_TRACE_SDPA = tuple(int(c) for c in os.environ.get("MIMO_PREFILL_CHUNK_TRACE_SDPA", "256").split(","))
# Decode, verify and MTP draft steps read their RoPE cos / sin from device tables of every position
# (TtMiMoModel.setup_decode_rope_tables); MIMO_DEVICE_ROPE=0 uploads them every step instead.
DEVICE_ROPE = os.environ.get("MIMO_DEVICE_ROPE", "1") == "1"
# Up to this context the tables hold whole 192-wide rows (201 MB per chip at 128K); past it only the 64 rotated dims
# (``compact``: 537 MB per chip at 1M instead of 1.61 GB), each read appending the constant 1 / 0 of the other dims.
ROPE_TABLE_FULL_WIDTH_MAX_LEN = 131072
# MIMO_SPEC_DECODE=mtp: load the MTP layers and serve the plugin's model-owned drafter contract (module docstring).
# MIMO_MTP_LAYERS=3 (default): draft i + 1 from MTP layer i (vLLM 0.31's MultiModuleMTPSpeculator, which the GPU
# references run with the num_nextn_predict_layers patch; needs the decode RoPE tables); 1: MTP layer MIMO_MTP_LAYER
# (default 0) drafts every token, feeding itself its own output (vLLM's MTP proposer, upstream vLLM for MiMo).
# MIMO_SPEC_DECODE=dflash: the checkpoint's DFlash draft model (``dflash/``, tt/dflash.py) drafts instead, from the
# target layers' outputs the decode and verify steps keep, chained behind the verify as the MTP's; prompts prefill one
# row at a time (each chunk writes its last rows into the lane's DFlash context ring).
SPEC_DECODE = os.environ.get("MIMO_SPEC_DECODE", "")
DFLASH = SPEC_DECODE == "dflash"
# One MTP layer per draft by default: 3, or 1 with one-draft groups (MIMO_SPEC_GROUPS=4, K = 1: c=16 drafting).
MTP_LAYERS = int(os.environ.get("MIMO_MTP_LAYERS", str(min(3, MAX_DRAFTS))))
MTP_LAYER = int(os.environ.get("MIMO_MTP_LAYER", "0"))
# MIMO_MTP_BACKFILL=0: no target-hidden history, no ring backfill (the rings start empty for each prompt and keep the
# gaps of the steps the lane did not draft); it needs the decode RoPE tables.
MTP_BACKFILL = os.environ.get("MIMO_MTP_BACKFILL", "1") == "1"
# MIMO_SPEC_CHAIN=0: no draft chained behind a verify (every draft host-planned after the plugin's walk, and an
# argmax_ids verify reads its greedy tokens without an accept step), the flow before tt/spec_chain.py, for comparison.
SPEC_CHAIN = os.environ.get("MIMO_SPEC_CHAIN", "1") == "1"
# MIMO_SPEC_CHAIN_CHECK=1 (debug): a draft chained behind a verify is drafted again from the host plan of the commit it
# drafted from (the same MTP inputs, rewriting the same ring entries) and compared; mismatches are logged
# (``chain_stats``). A synchronous verify's chained draft is checked at its propose when the plugin committed what the
# device did; a deferred verify reads its pages and checks its drafts before it returns (``_check_deferred``), so the
# host waits for the device on every verify.
SPEC_CHAIN_CHECK = os.environ.get("MIMO_SPEC_CHAIN_CHECK") == "1"
# MIMO_ASYNC_SPEC=0: declare no asynchronous speculative decoding (``supports_async_spec_decode``), so a speculating
# launch runs synchronously (vLLM's --no-async-scheduling), for comparison.
ASYNC_SPEC = os.environ.get("MIMO_ASYNC_SPEC", "1") == "1"
# MIMO_SPEC_DEFER=0: a fused_sample verify asked to defer its read (async scheduling) answers host tensors, so nothing
# chains onto it; by default it returns a handle and is read a step later (VerifyHandle), and the plugin may chain the
# next verify onto it.
SPEC_DEFER = os.environ.get("MIMO_SPEC_DEFER", "1") == "1"


def _untraced_decode(vllm_config):
    """Why the plugin captures no decode traces for this launch, or None. Its warmup captures them when the operator's
    tt options (``additional_config["tt"]``, as its worker reads them) keep trace_mode "all" or "decode_only" and
    enable_model_warmup true, the defaults."""
    additional = getattr(vllm_config, "additional_config", None)
    tt = additional.get("tt") if isinstance(additional, dict) else None
    tt = tt if isinstance(tt, dict) else {}
    trace_mode, warmup = tt.get("trace_mode", "all"), tt.get("enable_model_warmup", True)
    if trace_mode in ("all", "decode_only") and warmup:
        return None
    return (
        "MiMo verifies and drafts only by replaying the traces warmup_model_decode captures: launch with the tt "
        f"options trace_mode 'all' or 'decode_only' and enable_model_warmup true (got trace_mode {trace_mode!r}, "
        f"enable_model_warmup {warmup!r})"
    )


def chunk_align():
    """A later prefill chunk's start alignment (``prefill_forward`` moves an unaligned start back to it): the 64-token
    block, or with MIMO_KV_PAIR_SPLIT the global group's 128-token block."""
    return kv_pair_split.PAIR if KV_PAIR_SPLIT else BLOCK_SIZE


def swa_spec_window(window=SWA_WINDOW):
    """The sliding window the SWA layers' SlidingWindowSpec declares to vLLM: the model's ``window`` plus
    ``chunk_align() - 1``. vLLM keeps a request's positions from ``processed - spec window + 1`` on and frees the
    sliding-window blocks below them (SlidingWindowManager.get_num_skipped_tokens), which is what the window of the
    next scheduled position reads. But ``prefill_forward`` moves an unaligned chunk start ``s`` back to
    ``s0 >= s - chunk_align() + 1`` and recomputes from there, and position ``s0`` attends from ``s0 - window + 1``
    (TtMiMoModel._window_blocks reads from ``s0 - window``, in the same block: ``s0`` and ``window`` are multiples of
    the block size). With this spec window vLLM keeps every position from ``s - window - chunk_align() + 2``, at or
    before ``s0 - window + 1``. The margin cannot go into the spec's extra_retained_tokens: vLLM overwrites that field
    (get_kv_cache_configs sets it from the speculative config). The model still attends over ``window``; the spec only
    decides what vLLM keeps."""
    return window + chunk_align() - 1


def swa_tokens_per_user():
    """Pool tokens vLLM's sliding-window groups can hold per request while a prefill chunk of up to 4096 tokens is in
    flight (SlidingWindowSpec.max_admission_blocks_per_request): 5 SWA groups x (the spec window - 1 + 4096 tokens,
    plus one block)."""
    return 5 * (math.ceil((swa_spec_window() - 1 + 4096) / BLOCK_SIZE) + 1) * BLOCK_SIZE


def _check_pair_split_ops():
    """MIMO_KV_PAIR_SPLIT runs the MLA attention ops with ``return_lse`` (attention.PAIR_SPLIT_OPS)."""
    if KV_PAIR_SPLIT and not PAIR_SPLIT_OPS:
        raise ValueError(
            "MIMO_KV_PAIR_SPLIT needs a ttnn build whose MLA attention ops return their LSE (return_lse); "
            "launch with MIMO_KV_PAIR_SPLIT=0 on this build"
        )


def _model_path(hf_config):
    path = getattr(hf_config, "_name_or_path", None)
    if path and os.path.exists(os.path.join(path, "config.json")):
        return path
    return default_model_path()


def _cache_dir(model_path):
    """The converted TTNN weights, as the tests keep them (tests/common.py cache_dir("model_bfp4experts"))."""
    root = os.environ.get("MIMO_TT_CACHE", "/home/zni/models/mimo-tt-cache")
    return os.path.join(root, os.path.basename(os.path.normpath(model_path)), "model_bfp4experts")


class VerifyHandle:
    """An accept-mode verify's result pages, one per pass, each copied to the host right behind its pass without
    blocking (SpecChain.read_async), and the events recorded after each lane's last copy. A deferred verify returns
    it (VerifyOutput.handle) with its ``tip_id``; a synchronous one reads it before returning.

    ``read`` and ``readout`` block on the events and touch nothing but the handle, so the plugin may call them from a
    reader thread; the pages are host copies, so they stay valid however many steps the device runs meanwhile, until
    the handle is dropped (after its propose, ``_propose_deferred``)."""

    def __init__(self, rows, width, first_positions, fused):
        self.tip_id = None
        self.rows, self.width, self.fused = rows, width, fused
        # The positions the plugin gave each row's column 0 (-1: a gap).
        self.p0 = torch.as_tensor(first_positions).to(torch.int32).clone()
        self.pages = []  # (lane, VerifyPass, SpecChain, host page), in enqueue order
        self.events = {}  # lane: the event after its last page copy
        self.timing_events = {}  # lane: the event at its accept trace's end (MIMO_VLLM_TIMING)
        self.sources = {}  # lane: the DraftSources whose drafts chained behind its pass
        self.relayed = set()  # the chain rows, whose column-0 position the device chose
        self.hints = None  # per row: the drafts verified
        self.prev = None  # a chained verify's: the deferred verify it chained onto, until this one is read
        self.proposed = False
        self.waited = self.draft_wait = 0.0
        self._lock = threading.Lock()
        self._results = self._readout = None

    def read(self):
        """[(lane, VerifyPass, ChainResult)] in enqueue order (blocks until the pages are on the host)."""
        with self._lock:
            if self._results is None:
                for lane, event in self.events.items():
                    t = time.perf_counter()
                    timing = self.timing_events.get(lane)
                    if timing is not None:
                        ttnn.event_synchronize(timing)
                        self.waited += time.perf_counter() - t
                        t = time.perf_counter()
                    ttnn.event_synchronize(event)
                    if timing is not None:
                        self.draft_wait += time.perf_counter() - t
                    else:
                        self.waited += time.perf_counter() - t
                self._results = [(lane, vp, chain.unpack(host)) for lane, vp, chain, host in self.pages]
            return self._results

    def p0_range(self, row):
        """(lowest, highest) real column-0 position of ``row``: exact for a row the host built or once this step is
        read; for a chain row the step it chained onto's range plus its count (1 .. 1 + its drafts), exact once that
        step is read (its p0 plus its count). Engine thread (``_check_chain``), while a reader may be reading: the
        readout is set before ``prev`` is dropped."""
        readout = self._readout
        if readout is None and row in self.relayed:
            prev = self.prev
            if prev is not None:
                before = prev._readout
                if before is not None:
                    p = int(before.p0[row]) + int(before.accepted_counts[row])
                    return p, p
                low, high = prev.p0_range(row)
                return low + 1, high + 1 + int(prev.hints[row])
            readout = self._readout
        p = int((self.p0 if readout is None else readout.p0)[row])
        return p, p

    def readout(self):
        """The plugin's FusedReadout: accepted_token_ids [B, 1 + K] int32 (-1 past each row's count; a gap commits
        token 0, count 1), accepted_counts [B] int32 and p0 [B] int32, the position the device gave each row's
        column 0 (the page's positions section: a chain row's is the device's own, at or below the provisional one;
        any other row's must be the one submitted)."""
        from vllm_tt_plugin.spec_decode import FusedReadout

        results = self.read()
        with self._lock:
            if self._readout is None:
                committed = torch.full((self.rows, self.width), -1, dtype=torch.int32)
                committed[:, 0] = 0
                counts = torch.ones(self.rows, dtype=torch.int32)
                p0 = self.p0.clone()
                for _, vp, result in results:
                    for s, row, col in vp.answers:
                        if col != 0:
                            continue
                        got = int(result.positions[s])
                        if row in self.relayed and got < 0:
                            raise RuntimeError(
                                f"chain row {row}: the step it chained onto drafted nothing on device (its chain "
                                "declined), so the device verified no drafts of its own"
                            )
                        if row in self.relayed and got > int(p0[row]):
                            raise RuntimeError(
                                f"chain row {row}: the device ran column 0 at {got}, above {int(p0[row])}"
                            )
                        if row not in self.relayed and got != int(p0[row]):
                            raise RuntimeError(
                                f"row {row}: the device ran column 0 at {got}, submitted at {int(p0[row])}"
                            )
                        p0[row] = got
                    commit_pass(vp, result.accept, committed, counts)
                self._readout = FusedReadout(accepted_token_ids=committed, accepted_counts=counts, p0=p0)
                # Its positions are known now: the steps chained onto this one stop at it (no chain of handles).
                self.prev = None
            return self._readout


class DeferredDecode:
    """A decode step submitted with ``defer_device_sampling`` (module docstring): its ``decode_forward`` arguments,
    which ``sample_deferred_decode`` runs once with the rows' sampling parameters and grammar mask. Nothing of it is on
    the device yet."""

    def __init__(self, args):
        self.args = args


class MiMoV2ForCausalLM:
    model_capabilities = {
        "supports_chunked_prefill": True,
        "supports_prefix_caching": True,
        "folds_dp_into_lanes": True,
        "fabric_config": {"config": ttnn.FabricConfig.FABRIC_1D_RING},
        # Device sampling: the decode trace's greedy token or DeviceSampler's draw (module docstring). Its candidates
        # are each chip's CANDIDATES_PER_CHIP largest logits, which bound top_k and stand in for the vocabulary when
        # a top_p < 1 row sets no top_k; a row with neither (pure temperature sampling) samples on host.
        "supports_sample_on_device": True,
        "supports_async_decode": True,
        "max_device_top_k": TP * CANDIDATES_PER_CHIP,
        "supports_device_top_p_without_top_k": True,
        "supports_device_penalties": False,
        "supports_device_logprobs": False,
        # Structured output samples on device: DeviceSampler applies vLLM's grammar mask (sample_deferred_decode).
        "supports_device_grammar": True,
        # The MTP drafter (spec_plan, propose_draft_tokens): it keeps the target hidden on device, so it needs no
        # hidden feed. Under async scheduling (MIMO_ASYNC_SPEC, module docstring) ordinary steps overlap as in a
        # launch that does not speculate, a fused verify is read a step later, and the next verify may chain onto it
        # on device.
        "supports_spec_decode": True,
        "spec_requirements": ("device_propose",),
        "output_tokens_per_step": 1,
        "supports_async_spec_decode": ASYNC_SPEC,
        # Lane mode under async scheduling (the plugin's section 4j): a prefill may run while a deferred verify is in
        # flight (prefill_forward writes only its slots; the verify's pages are host copies).
        "supports_prefill_over_pending_spec": ASYNC_SPEC,
    }
    # The four reload commands of the plugin's decode reload contract (docs/DECODE_RELOAD_CONTRACT.md).
    decode_input_update_contract = 1
    # Emits SlidingWindowSpec groups: the plugin's KV budget adds its sliding-window headroom.
    _HYBRID_KV_CACHE_GROUPS_ENABLED = True
    # (vllm_config, specs) recorded by get_kv_cache_spec, which the plugin calls in the worker process that later
    # allocates the KV cache. allocate_kv_cache_per_layer groups the specs with vLLM's own function, as the engine
    # did: vLLM resolves the KV cache layout the grouping reads only after collecting the specs.
    _kv_cache_spec_source = None
    # vLLM's KV cache group of each layer, when set directly (tests driving the adapter without vLLM).
    _layer_groups = None
    # Tokens vLLM schedules per step (--max-num-batched-tokens), which bounds a prefill chunk: None reads it from the
    # vLLM config get_kv_cache_spec recorded (set directly when driving the adapter without vLLM).
    prefill_token_budget = None

    def __init__(self, generator, max_seq_len):
        self.gen = generator
        self.models = generator.models
        self.dp = generator.dp
        self.per_lane = generator.users_per_replica
        self.max_seq_len = max_seq_len
        self.already_warmed_up_prefill = False
        self.prefill_programs_compiled = False
        self.layer_groups = None
        self.kv_caches = None
        self.decode_recaptures = 0  # decode traces captured again after a prefill compiled new programs
        # The KV pair split (allocate_kv_cache_per_layer, from the global group's block size) and each KV cache group's
        # block size in positions (64; 128 for the global group under the pair split).
        self.pair_split = False
        self.group_block_sizes = {}
        # Per lane: int32 [max_batch] positions the next decode step reads (-1: idle slot), set by a reloading step
        # and advanced with every step, as the trace advances its own (a steady step reads no host positions).
        self.decode_positions = [None] * self.dp
        # Per lane: whether its rows' sampling parameters (set by a reloading step) include a sampled row, which
        # selects the sampled decode trace.
        self.lane_sampled = [False] * self.dp
        # Device grammar (module docstring): whether it is on (``enable_device_grammar``), the step waiting for its
        # mask, per lane whether the sampler's parameters still flag rows of the last mask (``_set_grammar``), and why
        # a deferred step failed (every later decode raises).
        self.device_grammar_enabled = False
        self._deferred = None
        self.lane_grammar = [False] * self.dp
        self._grammar_failure = None
        # Per lane: where the last step left its requests' committed tokens (reference.spec_decode.DraftSource), one
        # per draft group, none when that step ran no request or more than DRAFT_GROUPS; propose_draft_tokens drafts
        # from them.
        self.spec_sources = [[] for _ in range(self.dp)]
        # Per lane: {row: the position of the token it committed} when the last step was a plain decode step (empty
        # after a verify, whose propose always follows it directly): a propose for an older step is told apart by its
        # committed position (``_stale_source``).
        self.source_positions = [{} for _ in range(self.dp)]
        # Per lane: {row: the draft its last verify chained on device for that request} (reference.spec_chain
        # ChainedDraft); propose_draft_tokens returns it when the plugin committed what the device did.
        self.chained = [{} for _ in range(self.dp)]
        self.chain_check = SPEC_CHAIN_CHECK
        # Drafts returned from the chain and drafted from the host plan; with chain_check, chained drafts checked
        # against the host plan's and the mismatches; proposes for a step older than the lane's last one
        # (``_stale_source``: one step behind, or anywhere else); chain rows whose page tables have holes
        # (``_check_chain_tables``); ring backfills.
        self.chain_stats = {
            "chained": 0,
            "host": 0,
            "checked": 0,
            "mismatched": 0,
            "stale": 0,
            "misplaced": 0,
            "uncovered": 0,
            "backfills": 0,
        }
        # The deferred verifies' tip ids, and the newest one while no other step followed it (the one a chained
        # verify may build on).
        self._next_tip = 0
        self._tip = None
        # Per lane and draft group: (lane slot, last row) of the request whose rows the group's MTP rings hold up to
        # that row (the last draft's B - 1), None after a prompt reset them; (lane slot, VerifyHandle) after a draft
        # chained behind a verify, whose row the handle's page gives (``_ring``). A draft for another request, or past
        # a gap, backfills first.
        self.mtp_ring = [[None] * DRAFT_GROUPS for _ in range(self.dp)]
        # Per lane and draft group: (lane slot, last position) of the request whose context the group's DFlash ring
        # holds up to that position, None when it holds none (tt/dflash.py), or (lane slot, VerifyHandle) after a
        # draft chained behind a verify (``_ring``); a draft or prefill chunk that does not continue it resets it.
        self.dflash_ring = [[None] * DRAFT_GROUPS for _ in range(self.dp)]
        # Per lane: each draft group's request (its lane slot; ``_draft_groups``), and the step each slot last ran in.
        self.group_owners = [[None] * DRAFT_GROUPS for _ in range(self.dp)]
        self.slot_seen = [{} for _ in range(self.dp)]
        self._steps = 0
        self.chain_stats["dflash_resets"] = 0
        # Per lane and slot: where the slot's last prefill chunk ended (a chunk starting there continues its request).
        self.prefill_ends = [{} for _ in range(self.dp)]

    # ------------------------------------------------------------------ construction and sizing
    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None, **kwargs
    ):
        if optimizations not in (None, "performance"):
            raise ValueError(f"MiMo-V2.6-Flash serves one datatype policy; optimizations={optimizations!r}")
        dp = tt_data_parallel
        num_devices = mesh_device.get_num_devices()
        if num_devices != TP * dp:
            raise ValueError(
                f"MiMo runs one TP={TP} replica per lane: {dp} lanes need {TP * dp} chips, got {num_devices}"
            )
        if max_batch_size % dp:
            raise ValueError(f"max_batch_size {max_batch_size} does not split over {dp} lanes")
        if max_seq_len % BLOCK_SIZE:
            raise ValueError(f"max_model_len {max_seq_len} must be a multiple of the block size {BLOCK_SIZE}")
        path = _model_path(hf_config)
        cfg = MiMoConfig.from_model_path(path)
        num_layers = hf_config.num_hidden_layers
        if num_layers > cfg.num_hidden_layers:
            raise ValueError(f"num_hidden_layers {num_layers} > the checkpoint's {cfg.num_hidden_layers}")
        mtp_layers = dflash_path = None
        if SPEC_DECODE:
            if SPEC_DECODE not in ("mtp", "dflash"):
                raise ValueError(f"MIMO_SPEC_DECODE={SPEC_DECODE!r}: MiMo drafts with its MTP layers (mtp) or DFlash")
            if max_batch_size // dp < SPEC_SLOTS:
                raise ValueError(
                    f"a verify runs one request in {SPEC_SLOTS} decode slots: launch with --max-num-seqs >= "
                    f"{SPEC_SLOTS} per lane (got {max_batch_size // dp})"
                )
        if DFLASH:
            if MAX_DRAFTS < 3:
                raise ValueError(
                    f"MIMO_SPEC_DECODE=dflash drafts a block of 3 or 7: groups of {GROUP_SLOTS} slots hold 1"
                )
            if not os.path.exists(os.path.join(path, "dflash", "config.json")):
                raise ValueError(f"MIMO_SPEC_DECODE=dflash: no DFlash draft model in {path}/dflash")
            dflash_path = path
        elif SPEC_DECODE:
            if MTP_LAYERS not in (1, MAX_DRAFTS):
                raise ValueError(f"MIMO_MTP_LAYERS={MTP_LAYERS}: 1, or {MAX_DRAFTS} (one MTP layer per draft)")
            if MTP_LAYERS > cfg.num_nextn_predict_layers:
                raise ValueError(f"MIMO_MTP_LAYERS={MTP_LAYERS}: the checkpoint has {cfg.num_nextn_predict_layers}")
            if planned_drafts(MTP_LAYERS) and not DEVICE_ROPE:
                raise ValueError("the planned MTP program plans its passes on device: it needs MIMO_DEVICE_ROPE=1")
            mtp_layers = (MTP_LAYER,) if MTP_LAYERS == 1 else tuple(range(MAX_DRAFTS))
            if MTP_BACKFILL and not DEVICE_ROPE:
                logger.warning("MiMo vLLM: MIMO_MTP_BACKFILL needs the decode RoPE tables (MIMO_DEVICE_ROPE=1): off")
        logger.info(f"MiMo vLLM: {dp} replicas x {max_batch_size // dp} users, {num_layers} layers, {path}")
        generator = MiMoGalaxyGenerator(
            mesh_device,
            cfg,
            MiMoCheckpoint(path, cfg),
            _cache_dir(path),
            dp=dp,
            users_per_replica=max_batch_size // dp,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            block_size=BLOCK_SIZE,
            setup_kv=False,
            mtp_layers=mtp_layers,
            mtp_backfill=MTP_BACKFILL and DEVICE_ROPE,
            dflash_path=dflash_path,
        )
        return cls(generator, max_seq_len)

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        """MTP drafts up to MAX_DRAFTS tokens: the last committed token and its drafts fill one verify pass. A launch
        that captures no decode traces serves no draft length (``_untraced_decode``)."""
        from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

        if SPEC_DECODE not in ("mtp", "dflash"):
            return SpecReject(
                reason="MiMo drafts with its MTP layers or DFlash: launch with MIMO_SPEC_DECODE=mtp or dflash",
                supported_k=(MAX_DRAFTS,),
            )
        untraced = _untraced_decode(vllm_config)
        if untraced is not None:
            return SpecReject(reason=untraced)
        if requested_k < 1:
            return SpecReject(reason=f"num_speculative_tokens {requested_k} < 1", supported_k=(MAX_DRAFTS,))
        k = min(requested_k, MAX_DRAFTS)
        return SpecPlan(
            effective_k=k,
            lanes_per_request=k + 1,
            # The MTP ring cache is per lane, not per request; its weights are the replica's.
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            # argmax_ids for all-greedy steps; fused_sample (acceptance on device) when a sampled request shares
            # the step and the device sampler serves every request's sampling; logits otherwise (the plugin's host
            # accept walk then samples losslessly, ``_verify``).
            # The device verify sampler serves as many slots as the replica's decode users (up to 8: launch
            # MIMO_SPEC_SLOTS=8 with 8 sequences per lane); a verify with more accepts greedy requests only
            # (GreedyVerify) and raises on a sampled one.
            accept_modes=("argmax_ids", "logits", "fused_sample"),
            drafter_state="internal",
            supports_narrow_decode=True,
        )

    @classmethod
    def get_max_tokens_all_users(cls, model_name="", num_devices=TP, tt_data_parallel=1, **kwargs):
        """Pool tokens of one lane (64-token pages x 64: the plugin sizes the pool as this over --block-size 64):
        enough for ``max_num_seqs`` requests of ``max_model_len`` (their global group's pages, half as many with
        MIMO_KV_PAIR_SPLIT, plus what vLLM's SWA groups hold per request), capped by MIMO_KV_GB_PER_CHIP of DRAM per
        chip (default 6.7 with the pair split: 4 x 1M per lane; else 6). Warns when the cap holds fewer requests."""
        default = PAIR_SPLIT_KV_GB_PER_CHIP if KV_PAIR_SPLIT else DEFAULT_KV_GB_PER_CHIP
        gb = float(os.environ.get("MIMO_KV_GB_PER_CHIP", default))
        capacity = int(gb * 1e9 // KV_BYTES_PER_TOKEN)
        max_model_len, max_num_seqs = kwargs.get("max_model_len"), kwargs.get("max_num_seqs")
        if not max_model_len or not max_num_seqs:
            return capacity
        global_tokens = kv_pair_split.global_pool_tokens(max_model_len) if KV_PAIR_SPLIT else max_model_len
        per_request = global_tokens + swa_tokens_per_user()
        if capacity < max_num_seqs * per_request:
            # vLLM still starts (one request fits) but queues what does not: say so instead of serving fewer users.
            need = math.ceil(max_num_seqs * per_request * KV_BYTES_PER_TOKEN / 1e7) / 100
            logger.warning(
                f"MiMo vLLM KV: MIMO_KV_GB_PER_CHIP={gb:g} holds {capacity / per_request:.2f} requests of "
                f"--max-model-len {max_model_len} per lane, fewer than --max-num-seqs {max_num_seqs}; the rest wait "
                f"for blocks. {max_num_seqs} need MIMO_KV_GB_PER_CHIP >= {need:.2f}"
            )
        return min(capacity, max_num_seqs * per_request)

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        hf = vllm_config.model_config.hf_config
        block_size = vllm_config.cache_config.block_size
        if block_size != BLOCK_SIZE:
            raise ValueError(
                f"MiMo's paged caches use {BLOCK_SIZE}-token blocks; launch with --block-size {BLOCK_SIZE}"
            )
        _check_pair_split_ops()
        full = dict(num_kv_heads=hf.num_key_value_heads, head_size=hf.head_dim, head_size_v=hf.v_head_dim)
        swa = dict(num_kv_heads=hf.swa_num_key_value_heads, head_size=hf.swa_head_dim, head_size_v=hf.swa_v_head_dim)
        dtype = torch.bfloat16
        # The pair split's global blocks hold 128 tokens: 64 on each chip of the pair (module docstring).
        full_block = kv_pair_split.PAIR if KV_PAIR_SPLIT else block_size
        # vLLM's hybrid manager needs one page size for every group; pad the global layers' page to the SWA page (with
        # the pair split they are equal).
        page = max(
            bs * d["num_kv_heads"] * (d["head_size"] + d["head_size_v"]) * 2
            for bs, d in ((full_block, full), (block_size, swa))
        )
        # vLLM's window keeps the SWA blocks a chunk start moved back by prefill_forward reads (swa_spec_window).
        window = swa_spec_window(getattr(hf, "sliding_window_size", None) or hf.sliding_window)
        specs = {}
        for i in range(hf.num_hidden_layers):
            name = f"model.layers.{i}.self_attn"
            if hf.hybrid_layer_pattern[i] == 1:
                specs[name] = SlidingWindowSpec(
                    block_size=block_size, dtype=dtype, sliding_window=window, page_size_padded=page, **swa
                )
            else:
                specs[name] = FullAttentionSpec(block_size=full_block, dtype=dtype, page_size_padded=page, **full)
        cls._kv_cache_spec_source = (vllm_config, specs)
        return specs

    @classmethod
    def _vllm_layer_groups(cls):
        """vLLM's KV cache group of each layer, from the specs get_kv_cache_spec returned."""
        if cls._layer_groups is not None:
            return cls._layer_groups
        if cls._kv_cache_spec_source is None:
            raise RuntimeError("allocate_kv_cache_per_layer before get_kv_cache_spec: no KV cache groups")
        from vllm.v1.core.kv_cache_utils import get_kv_cache_groups

        vllm_config, specs = cls._kv_cache_spec_source
        groups = [None] * len(specs)
        for g, group in enumerate(get_kv_cache_groups(vllm_config, dict(specs))):
            for name in group.layer_names:
                groups[int(name.split(".")[2])] = g
        if None in groups:
            raise RuntimeError(f"layers without a KV cache group: {groups}")
        logger.info(f"MiMo vLLM KV cache groups per layer: {groups}")
        return groups

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """per_layer_specs: (shape, dtype, tensor_idx) per layer, shape[0] the lane's block count and shape[2] its
        group's block size (64; 128 for the global layers with MIMO_KV_PAIR_SPLIT). Returns, per lane, [K, V] per
        layer, with buffers shared as in the module docstring (every physical page [1, 64, dim] per chip)."""
        groups = self._vllm_layer_groups()
        if len(groups) != len(per_layer_specs):
            raise RuntimeError(f"{len(per_layer_specs)} layer KV specs but {len(groups)} layer groups")
        _check_pair_split_ops()
        num_blocks = per_layer_specs[0][0][0]
        assert all(s[0][0] == num_blocks for s in per_layer_specs)
        # Each layer's block size is its group's: SWA 64, global 64 or (pair split) 128.
        block_sizes = {}
        for layer, g, (shape, _, _) in zip(self.models[0].layers, groups, per_layer_specs):
            want = kv_pair_split.PAIR if KV_PAIR_SPLIT and not layer.is_swa else BLOCK_SIZE
            if shape[2] != want or block_sizes.setdefault(g, shape[2]) != shape[2]:
                raise ValueError(
                    f"layer {layer.layer_idx} ({'SWA' if layer.is_swa else 'global'}, group {g}): block size "
                    f"{shape[2]}, expected {want} (MIMO_KV_PAIR_SPLIT={int(KV_PAIR_SPLIT)})"
                )
        # Layers sharing a buffer must come from different groups: their block ids never overlap.
        seen = set()
        for i, (_, _, t) in enumerate(per_layer_specs):
            assert (groups[i], t) not in seen, f"layer {i}: two layers of group {groups[i]} share buffer {t}"
            seen.add((groups[i], t))
        # vLLM trims its tables to the pool; the padding entries stay 0 (its null block, set_vllm_page_tables). A
        # 128-token global group's table is half as wide.
        table_widths = {
            g: page_table_width(min(math.ceil(self.max_seq_len / bs), num_blocks)) for g, bs in block_sizes.items()
        }
        self.pair_split = KV_PAIR_SPLIT
        self.group_block_sizes = block_sizes
        self.layer_groups = groups
        compact_rope = self.max_seq_len > ROPE_TABLE_FULL_WIDTH_MAX_LEN
        # The replicas' RoPE tables are equal: computed once on the host (~2.5 s at 1M), uploaded to each.
        rope_host = self.models[0].decode_rope_tables_host(compact_rope) if DEVICE_ROPE else None
        self.kv_caches = []
        for model in self.models:
            buffers, caches = {}, []
            for layer, (_, _, t) in zip(model.layers, per_layer_specs):
                attn = layer.attn
                caches.append(
                    [
                        self._buffer(model, buffers, ("k", t, attn.head_dim), num_blocks, attn.head_dim),
                        self._buffer(model, buffers, ("v", t, attn.v_head_dim), num_blocks, attn.v_head_dim),
                    ]
                )
            model.setup_vllm_kv_cache(self.per_lane, caches, groups, table_widths, pair_split=self.pair_split)
            if DEVICE_ROPE:
                model.setup_decode_rope_tables(compact_rope, rope_host)
            if model.mtp is not None:
                model.mtp.setup_cache()  # after the RoPE tables: the backfill reads them
            if model.dflash is not None:
                model.dflash.setup_cache()
            self.kv_caches.append(caches)
        logger.info(
            f"MiMo vLLM KV: {num_blocks} blocks per lane (group block sizes {block_sizes}, pair split "
            f"{self.pair_split}), {len(buffers)} buffers per replica, page tables {table_widths} wide"
        )
        if rope_host is not None:
            rows, width = rope_host[0].shape
            logger.info(
                f"MiMo vLLM: decode RoPE tables 4 x {rows} x {width} ({4 * rows * width * 2 / 1e6:.0f} MB per chip)"
            )
        return self.kv_caches

    @staticmethod
    def _buffer(model, buffers, key, num_blocks, dim):
        if key not in buffers:
            cache = ttnn.allocate_tensor_on_device(
                ttnn.Shape((num_blocks, 1, BLOCK_SIZE, dim)),
                model.kv_dtype,
                ttnn.TILE_LAYOUT,
                model.mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.fill(cache, 0.0, output_tensor=cache)
            buffers[key] = cache
        return buffers[key]

    # ------------------------------------------------------------------ page tables
    def _group_rows(self, tables, row):
        """{group: vLLM block table row ``row``}, one per KV cache group (the layers of a group share a table)."""
        out = {}
        for i, g in enumerate(self.layer_groups):
            if g not in out:
                out[g] = torch.as_tensor(tables[i][row])
        return out

    def _layer_tables(self, page_table, page_tables_per_layer):
        """One block table per layer. The plugin sends ``page_tables_per_layer`` when vLLM made several KV cache
        groups; with one group, ``page_table`` serves every layer."""
        if page_tables_per_layer is None and len(set(self.layer_groups)) == 1:
            return [page_table] * len(self.layer_groups)
        if page_tables_per_layer is None or len(page_tables_per_layer) != len(self.layer_groups):
            raise ValueError(
                f"MiMo needs a page table per attention layer ({len(self.layer_groups)}); got "
                f"{None if page_tables_per_layer is None else len(page_tables_per_layer)}"
            )
        return page_tables_per_layer

    # ------------------------------------------------------------------ prefill
    def prefill_forward(
        self,
        tokens,
        start_pos,
        prompt_lens,
        page_table,
        kv_cache,
        enable_trace=False,
        empty_slots=None,
        page_tables_per_layer=None,
        sampling_params=None,
        **kwargs,
    ):
        """Row ``i`` computes tokens[i, start_pos[i]:prompt_lens[i]] for the request in slot ``empty_slots[i]``.
        Returns its last-token logits, float32 [B, 1, vocab] (host sampling).

        It may run while a deferred verify is in flight (``supports_prefill_over_pending_spec``): it writes only its
        slots' page-table rows and K / V, after the verify on the device queue; the verify's result pages are copied
        to the host behind it, so its read and propose need nothing the prefill touches. Nothing chains onto that
        verify afterwards, and a lane taking a prompt restarts its MTP ring (its drafter serves one request)."""
        if sampling_params is not None:
            raise ValueError("MiMo samples on the host; got device sampling_params")
        tables = self._layer_tables(page_table, page_tables_per_layer)
        rows = tokens.shape[0]
        starts = np.asarray(start_pos).reshape(-1).astype(np.int64)
        ends = np.asarray(prompt_lens).reshape(-1).astype(np.int64)
        slots = list(range(rows)) if empty_slots is None else [int(s) for s in empty_slots]
        placed = []
        table_rows = [{} for _ in range(self.dp)]
        continues = {}
        align = self._chunk_align()
        for i in range(rows):
            lane, slot = divmod(slots[i], self.per_lane)
            # A chunk that starts off the alignment (vLLM's prefix hits land on its scheduler block, but the plugin's
            # scheduler gives a running request a short chunk when its leftover budget cannot reach its grid) is moved
            # back to it and recomputes the positions in front of it (their K / V are rewritten with the same values).
            # The SWA specs' window keeps the sliding-window blocks those positions read (swa_spec_window). With the
            # pair split the alignment is the global group's 128-token block.
            start = int(starts[i]) // align * align
            cached = 0
            if (
                start > 0
                and PREFILL_BATCH
                and prefill_padded_len(int(ends[i])) <= self.models[0].PREFILL_BATCH_MAX_PADDED
            ):
                # A short prompt whose first blocks are cached (a prefix hit) is recomputed from position 0 in a batch
                # pass (with the lane's other first chunks, traced) instead of an eager later chunk. Its cached blocks
                # are not written: another request may read them, and a pass of another length could compute their
                # K / V slightly differently. Their fill entries go to the null block.
                cached, start = start // BLOCK_SIZE, 0
            placed.append((i, lane, slot, start, int(ends[i]), cached))
            table_rows[lane][slot] = self._group_rows(tables, i)
            # The MTP history: a chunk starting where the slot's last one ended continues its request.
            continues[i] = start > 0 and self.prefill_ends[lane].get(slot) == start
            self.prefill_ends[lane][slot] = int(ends[i])
        for lane, rows_of_lane in enumerate(table_rows):
            if rows_of_lane:
                self.models[lane].set_vllm_page_tables(rows_of_lane)
                # A lane taking a prompt drafts again only after a decode step of its requests; with one draft group
                # its MTP ring starts empty for the new request.
                self.spec_sources[lane], self.source_positions[lane], self.chained[lane] = [], {}, {}
                if DRAFT_GROUPS == 1:
                    self.mtp_ring[lane][0] = None
                    if self.models[lane].mtp is not None:
                        self.models[lane].mtp.reset()
        # A request's first chunk: its draft group (``_draft_groups``), whose rings hold none of its rows (with draft
        # groups its first draft backfills the group's MTP rings).
        draft_group = {}
        for i, lane, slot, *_ in placed:
            draft_group[i] = self._draft_groups(lane, [slot], prompt=True)[slot]
            if not continues[i]:
                self.mtp_ring[lane][draft_group[i]] = self.dflash_ring[lane][draft_group[i]] = None
        # A lane's first chunks of one padded length (at most PREFILL_BATCH_MAX_PADDED) go in one pass
        # (TtMiMoModel.prefill_enqueue_batch, which pads every chunk to the longest); the other rows one by one. Every
        # pass is enqueued before any is read, the lanes interleaved so all replicas have work.
        # A lane's replayed later chunks, then its batch passes, go before its other rows: a batch reads no cached KV
        # (it starts at 0), and a replay must not run while an eager pass's output of that lane is alive.
        by_lane, traced_rows = [], set()
        for lane in range(self.dp):
            groups = {}
            for p in placed:
                if p[1] == lane:
                    key = prefill_padded_len(p[4]) if self._batchable(p) else ("row", p[0])
                    groups.setdefault(key, []).append(p)
            batches = [g for key, g in groups.items() if not isinstance(key, tuple)]
            rows_ = [g for key, g in groups.items() if isinstance(key, tuple)]
            # A replayed later chunk goes first: a replay must not run while an eager pass's output of that lane is
            # alive (its own output is copied into the slot's persistent logits row, which a batch replay leaves be).
            # A row is traced only when every chunk prefill_enqueue splits it into replays (chunk_traced); one that
            # would mix replays and eager chunks (an unaligned start moved back past the chunk budget) runs all eager
            # (allow_trace=False), after the replays, so no replay runs while an eager output of the lane is alive.
            model = self.models[lane]
            traced = [
                g for g in rows_ if PREFILL_CHUNK_TRACE and model.chunk_traced(g[0][2], g[0][3], g[0][4] - g[0][3])
            ]
            traced_rows.update(g[0][0] for g in traced)
            by_lane.append(traced + batches + [g for g in rows_ if not any(g is t for t in traced)])
        order = [passes[k] for k in range(max(map(len, by_lane))) for passes in by_lane if k < len(passes)]
        # Nothing chains onto a verify a prefill followed (the plugin builds the next decode from host state).
        self._tip = None
        programs = [model.mesh_device.num_program_cache_entries() for model in self.models]
        t0 = time.perf_counter()
        pending = []
        for rows_of_pass in order:
            lane = rows_of_pass[0][1]
            chunks = [
                (torch.as_tensor(tokens[i, start:end]).to(torch.int32), slot, cached)
                for i, _, slot, start, end, cached in rows_of_pass
            ]
            if self._batchable(rows_of_pass[0]):
                handle = self.models[lane].prefill_enqueue_batch(chunks)
            else:
                (chunk, slot, _), start, i = chunks[0], rows_of_pass[0][3], rows_of_pass[0][0]
                group = draft_group[i]
                self._dflash_continue(lane, slot, start, start + chunk.numel() - 1, group)
                handle = self.models[lane].prefill_enqueue(
                    chunk,
                    slot,
                    start_pos=start,
                    history_continues=continues[i],
                    draft_group=group,
                    allow_trace=i in traced_rows,
                )
            pending.append((rows_of_pass, lane, handle))
        t1 = time.perf_counter()
        logits = [None] * rows
        for rows_of_pass, lane, handle in pending:
            out = self.models[lane].prefill_read(handle)[0].reshape(len(rows_of_pass), -1)
            for k, p in enumerate(rows_of_pass):
                logits[p[0]] = out[k]
        if TIMING:
            logger.info(
                f"MiMo vLLM timing: prefill {rows} rows in {len(pending)} passes "
                f"{[[p[4] - p[3] for p in rows_of] for rows_of, _, _ in pending]}: enqueue {1e3 * (t1 - t0):.0f} ms, "
                f"total {1e3 * (time.perf_counter() - t0):.0f} ms"
            )
        self._recapture_after_new_programs(programs)
        return torch.stack(logits).unsqueeze(1)

    def _chunk_align(self):
        """A prefill chunk's start alignment (``chunk_align``) for this adapter's KV layout."""
        return kv_pair_split.PAIR if self.pair_split else BLOCK_SIZE

    def _batchable(self, placed_row):
        """A first chunk (start 0) whose padded length a prefill batch takes."""
        start, end = placed_row[3], placed_row[4]
        return PREFILL_BATCH and start == 0 and prefill_padded_len(end) <= self.models[0].PREFILL_BATCH_MAX_PADDED

    def _recapture_after_new_programs(self, programs):
        """``programs``: each replica's program-cache size before the prefill. A prefill that compiled new programs
        (a chunk length or offset first seen while serving) put their program-cache buffers in memory the replica's
        decode trace may write on replay (tech_reports/.../TraceCorrectness.md, unsafe allocations), so that replica's
        decode trace is captured again, around them. Every slot is idle during the capture, so no KV is written."""
        for lane, model in enumerate(self.models):
            if model.decode_trace is None or model.mesh_device.num_program_cache_entries() == programs[lane]:
                continue
            new = model.mesh_device.num_program_cache_entries() - programs[lane]
            # Every trace of the replica, in warmup_model_decode's order: the greedy and sampled decode steps, the
            # verify step and the MTP draft program, then the later-chunk prefill traces.
            model.release_spec_traces()
            model.release_decode_trace()
            model.release_prefill_chunk_traces()
            model.capture_decode_trace()
            if model.mtp is not None or model.dflash is not None:
                model.capture_spec_traces()
            model.capture_prefill_chunk_traces()  # optional, so last: if they no longer fit they are dropped
            self.decode_recaptures += 1
            logger.info(
                f"MiMo vLLM: replica {lane} decode trace recaptured after {new} new prefill programs "
                f"({self.decode_recaptures} recaptures so far)"
            )

    # ------------------------------------------------------------------ decode
    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        enable_trace=True,
        read_from_device=True,
        page_tables_per_layer=None,
        sampling_params=None,
        slot_remap=None,
        prompt_tokens=None,
        output_tokens=None,
        *,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        spec_mode=None,
        num_valid_drafts=None,
        accepted_counts=None,
        defer_read=False,
        chain_rows=None,
        chain_tip=None,
        **kwargs,
    ):
        """One decode step for every lane's slots: tokens [B, 1], start_pos [B] (-1: idle slot), B = lanes x per_lane.
        With ``spec_mode`` it is a verify of the plugin's [B, 1 + K] candidate block (``_verify``); a plain decode
        ignores ``accepted_counts`` (it keeps no per-candidate state). ``defer_read`` (a fused_sample verify under
        async scheduling): the verify returns a handle and is read later, or answers host tensors with
        MIMO_SPEC_DEFER=0; ``chain_rows`` / ``chain_tip``: a verify chained onto the deferred one in flight
        (``_verify``).

        ``sampling_params`` None: host sampling, every step reloads the inputs and the step returns logits
        [B, 1, vocab]. Otherwise device sampling: the step returns the tokens [B], greedy or drawn by DeviceSampler with
        the rows' parameters, which a step with ``reload_inputs`` or ``reload_sampling_params`` uploads (the rows'
        order is the one ``tokens`` has then). ``reload_inputs`` False (steady device decode) leaves ``tokens`` and
        ``start_pos`` unread: the
        trace consumes the tokens and positions its last replay wrote, and the adapter copies the RoPE inputs of its
        own position counter (``decode_positions``); ``reload_page_table`` then copies the page tables.

        The model keeps no other slot state (no penalty or RNG state), so a reloading step needs no ``slot_remap``:
        it rewrites every slot's tokens and positions. A steady step with a non-identity remap raises.

        ``defer_device_sampling`` (device grammar, module docstring): a device-sampled step without its
        ``sampling_params``; returns a DeferredDecode for ``sample_deferred_decode``, and no decode may come before
        that consumes it."""
        if self._grammar_failure is not None:
            raise RuntimeError(
                f"MiMo device grammar sampling failed after a decode submission: {self._grammar_failure}"
            )
        if self._deferred is not None:
            raise RuntimeError("a deferred decode step is pending: sample_deferred_decode must run it first")
        if "reset_batch" in kwargs:
            raise ValueError("MiMo implements decode_input_update_contract 1; got the legacy reset_batch keyword")
        defer = bool(kwargs.pop("defer_device_sampling", False))
        if defer:
            if not self.device_grammar_enabled:
                raise ValueError("defer_device_sampling needs device grammar (enable_device_grammar)")
            if spec_mode is not None or sampling_params is not None or read_from_device:
                raise ValueError(
                    "a deferred decode is a plain step without sampling_params or a read: got spec_mode "
                    f"{spec_mode!r}, sampling_params {sampling_params is not None}, read_from_device {read_from_device}"
                )
        if defer_read and spec_mode != "fused_sample":
            raise ValueError(f"defer_read comes with a fused_sample verify, got spec_mode {spec_mode!r}")
        if (chain_rows is not None or chain_tip is not None) and spec_mode != "fused_sample":
            raise ValueError(f"chain_rows / chain_tip come with a fused_sample verify, got spec_mode {spec_mode!r}")
        if spec_mode is not None:
            return self._verify(
                tokens,
                start_pos,
                page_table,
                page_tables_per_layer,
                num_valid_drafts,
                spec_mode,
                kwargs.pop("spec_sampling", None),
                defer_read=defer_read,
                chain_rows=chain_rows,
                chain_tip=chain_tip,
                accepted_counts=accepted_counts,
            )
        if kwargs.get("spec_sampling") is not None:
            raise ValueError("spec_sampling comes with a fused_sample verify; got it on a plain decode")
        if reload_inputs and reload_page_table:
            raise ValueError("reload_inputs already copies the page tables; got reload_page_table too")
        if reset_sampling_state and not reload_inputs:
            raise ValueError("reset_sampling_state without reload_inputs")
        if not reload_inputs:
            if sampling_params is None and not defer:
                raise ValueError("host sampling decodes from host inputs: reload_inputs must be True")
            if not enable_trace:
                raise ValueError("a steady decode (reload_inputs False) replays the decode trace: enable_trace False")
            if slot_remap is not None and any(int(j) != i for i, j in enumerate(slot_remap)):
                raise ValueError(f"a steady decode cannot move resident slot state: slot_remap {list(slot_remap)}")
        args = {
            "tokens": tokens,
            "start_pos": start_pos,
            "page_table": page_table,
            "page_tables_per_layer": page_tables_per_layer,
            "enable_trace": enable_trace,
            "reload_inputs": reload_inputs,
            "reload_page_table": reload_page_table,
            "reload_sampling_params": reload_sampling_params,
        }
        if defer:
            if reload_inputs:
                args.update(tokens=torch.as_tensor(tokens).clone(), start_pos=torch.as_tensor(start_pos).clone())
            self._deferred = DeferredDecode(args)
            return self._deferred
        return self._decode(**args, sampling_params=sampling_params, read_from_device=read_from_device)

    def _decode(
        self,
        tokens,
        start_pos,
        page_table,
        page_tables_per_layer,
        enable_trace,
        reload_inputs,
        reload_page_table,
        reload_sampling_params,
        sampling_params,
        read_from_device,
        grammar=None,
    ):
        """A plain decode step (``decode_forward``'s arguments, checked). ``grammar``: int32 [lanes x per_lane, words],
        each row's packed grammar mask (-1 words: allowed), for a device-sampled step (``sample_deferred_decode``)."""
        on_device = sampling_params is not None
        t0 = time.perf_counter()
        tables = self._layer_tables(page_table, page_tables_per_layer)
        rows = self.dp * self.per_lane
        if reload_inputs:
            tokens = torch.as_tensor(tokens).reshape(-1)
            positions = torch.as_tensor(start_pos).reshape(-1)
            rows = positions.numel()
            if rows > self.dp * self.per_lane:
                raise ValueError(f"{rows} decode rows > {self.dp} lanes x {self.per_lane} slots")
        handles = []
        self.spec_sources = [[] for _ in range(self.dp)]
        self.source_positions = [{} for _ in range(self.dp)]
        self.chained = [{} for _ in range(self.dp)]
        self._tip = None
        for lane, model in enumerate(self.models):
            first = lane * self.per_lane
            last = min(first + self.per_lane, rows)
            if first >= rows:
                break
            if reload_inputs:
                tok = torch.zeros(model.max_batch, dtype=torch.int32)
                pos = torch.full((model.max_batch,), -1, dtype=torch.int32)
                tok[: last - first] = tokens[first:last].to(torch.int32)
                pos[: last - first] = positions[first:last].to(torch.int32)
            else:
                pos = self.decode_positions[lane]
                if pos is None:
                    raise RuntimeError(f"lane {lane}: a steady decode before any reloading decode")
            active = pos >= 0
            if on_device and (reload_inputs or reload_sampling_params):
                self._set_sampling(lane, model, sampling_params, first, last, pos)
            if not bool(active.any()):
                self.decode_positions[lane] = pos
                handles.append(None)
                continue
            # A row under a grammar draws on the sampled trace (the greedy one applies no mask).
            masked = on_device and self._set_grammar(lane, model, grammar, first, last, active)
            sample = on_device and (self.lane_sampled[lane] or masked)
            if reload_inputs or reload_page_table:
                model.set_vllm_page_tables(
                    {s: self._group_rows(tables, first + s) for s in range(last - first) if active[s]}
                )
            if enable_trace and model.decode_trace is None:
                raise RuntimeError("decode trace requested before warmup_model_decode captured it")
            # MIMO_SAMPLER_CHECK's host draw knows no grammar: it checks unconstrained steps only.
            rng = model.sampler.read_rng() if sample and SAMPLER_CHECK and not masked else None
            if reload_inputs:
                handle = model.decode_enqueue(tok, pos, sample=sample)
            else:
                handle = model.decode_enqueue_resident(pos, sample=sample)
            if rng is not None:
                self._check_sampling(lane, model, handle, rng, active)
            self.decode_positions[lane] = torch.where(active, pos + 1, pos)
            handles.append((lane, last - first, handle))
            live = [s for s in range(last - first) if active[s]]
            self._seen(lane, live)
            groups = self._draft_groups(lane, live) if len(live) <= DRAFT_GROUPS else {}
            if live and all(groups.get(s) is not None for s in live):
                self.spec_sources[lane] = [DraftSource(row=first + s, slots=[s], group=groups[s]) for s in live]
                self.source_positions[lane] = {first + s: int(pos[s]) + 1 for s in live}
        out = {"rows": rows, "handles": handles, "tokens": on_device}
        if TIMING:
            self._decode_timing(t0, reload_inputs, on_device)
        if read_from_device:
            return self.process_decode_output_host(self.read_decode_output(out), is_tokens=on_device)
        return out

    def _decode_timing(self, t0, reload_inputs, on_device):
        """Every 128 decode steps: the mean host time of decode_forward's submission, and the share of steps that
        reloaded their inputs or sampled on device."""
        now = time.perf_counter()
        stats = getattr(self, "_timing", None) or {"steps": 0, "host": 0.0, "reload": 0, "device": 0, "since": now}
        stats["steps"] += 1
        stats["host"] += now - t0
        stats["reload"] += int(reload_inputs)
        stats["device"] += int(on_device)
        if stats["steps"] == 128:
            logger.info(
                f"MiMo vLLM timing: decode {stats['steps']} steps in {1e3 * (now - stats['since']):.0f} ms, submit "
                f"{1e3 * stats['host'] / stats['steps']:.2f} ms/step, {stats['reload']} reloads, "
                f"{stats['device']} device-sampled"
            )
            stats = None
        self._timing = stats

    # ------------------------------------------------------------------ speculative decoding
    def _verify(
        self,
        tokens,
        start_pos,
        page_table,
        page_tables_per_layer,
        num_valid_drafts,
        spec_mode,
        spec_sampling=None,
        defer_read=False,
        chain_rows=None,
        chain_tip=None,
        accepted_counts=None,
    ):
        """A verify of the plugin's candidate block tokens / start_pos [B, 1 + K] (-1: no position): each lane's
        active rows packed into verify passes (reference.spec_decode.plan_verify_passes), a lane's passes one after
        another, the lanes side by side. Returns argmax_ids [B, 1 + K] (column j the target's greedy token after
        column j, unused columns 0), or with ``spec_mode`` "logits" the fp32 logits [B, 1 + K, vocab] of the same
        cells for the plugin's host accept walk: every answered cell holds its slot's full logit row; the others
        (gap rows, columns past a row's drafts) are never read and keep a finite earlier value (``_spec_logits``).

        With "fused_sample" each pass's acceptance runs on device right after it (VerifySampler; its inputs from
        reference.sample_verify.pass_inputs) and the verify returns accepted_token_ids [B, 1 + K] int32 (each row's
        committed tokens, -1 past its count) and accepted_counts [B] int32 in [1, 1 + K]; a row without a position (a
        gap) commits token 0 with count 1, what its argmax_ids answer commits. ``spec_sampling``
        (``_check_spec_sampling``) carries the sampled rows' parameters and the uniforms the plugin drew; None: every
        row greedy.

        An argmax_ids verify runs the same accept trace, every request greedy (its committed tokens are the plugin's
        greedy walk over the ids), so both modes read one result page per pass (tt/spec_chain.py): the argmax ids of
        the pass's slots, its accept page. On a lane whose step is one request in one pass (its DraftSource), with the
        decode RoPE tables, the accept trace's chain step also writes the MTP draft's inputs from what the device
        committed, and the draft trace is enqueued right behind it (``mtp.enqueue_chained``): the page then carries
        the drafts, kept in ``chained`` for propose_draft_tokens. The draft is declined on device when the bonus
        position reaches max_seq_len - K. Each pass's page is copied to the host right behind it without blocking
        (SpecChain.read_async), so every pass of every lane is enqueued before the host waits (VerifyHandle).
        MIMO_SPEC_CHAIN=0: no chain, and an argmax_ids verify reads the verify's greedy tokens. A logits verify chains
        nothing.

        ``defer_read`` (a fused_sample verify under async scheduling, the plugin's section 4j): nothing is read. The
        verify returns VerifyOutput(handle=VerifyHandle), whose ``tip_id`` names the step; ``read_decode_output``
        waits for its pages and returns the FusedReadout, and propose_draft_tokens returns the drafts its pages carry
        (``_propose_deferred``). Without the chain (MIMO_SPEC_CHAIN=0, no decode RoPE tables) the verify answers host
        tensors instead: its drafts would be planned from a resident hidden that later steps overwrite.

        ``chain_rows`` (bool [B]) and ``chain_tip`` (a verify chained onto the deferred one in flight, which must be
        ``chain_tip``): a chain row's block is the device's, built from the pending step's page (``_check_chain``):
        its last committed token at p0 = the pending p0 + its count, then the drafts chained behind it. The host lays
        it out as the plugin sent it (a pass of 1 + num_valid_drafts slots at the provisional positions, which only
        set the page tables) and marks the source relayed in the chain row; the verify trace's relay step writes the
        tokens and positions from the page (reference.spec_chain.relay_step). The readout reports the device's
        positions as p0, and raises for a chain row the device could not build (its draft was declined)."""
        from vllm_tt_plugin.spec_decode import VerifyOutput

        t0 = time.perf_counter()
        self._verify_tables_s = 0.0
        if spec_mode not in ("argmax_ids", "logits", "fused_sample"):
            raise ValueError(f"MiMo verifies argmax ids, logits or fused_sample; got spec_mode {spec_mode!r}")
        if any(model.verify_trace is None for model in self.models):
            raise RuntimeError("a verify before warmup_model_decode captured the verify traces (MIMO_SPEC_DECODE=mtp)")
        fused = spec_mode == "fused_sample"
        accept = fused or (spec_mode == "argmax_ids" and SPEC_CHAIN)  # the pass's accept trace and result page
        if spec_sampling is not None and not fused:
            raise ValueError(f"spec_sampling is only valid with spec_mode 'fused_sample', got {spec_mode!r}")
        defer = defer_read and fused and self._chains_on_device()
        tables = self._layer_tables(page_table, page_tables_per_layer)
        tokens = torch.as_tensor(tokens).to(torch.int64)
        positions = torch.as_tensor(start_pos).to(torch.int64)
        counts = torch.as_tensor(num_valid_drafts).reshape(-1)
        rows = tokens.shape[0]
        if rows > self.dp * self.per_lane:
            raise ValueError(f"{rows} verify rows > {self.dp} lanes x {self.per_lane} slots")
        pending = self._tip
        relayed = self._check_chain(chain_rows, chain_tip, defer, tokens, positions, counts, accepted_counts)
        if relayed:
            # The block the host lays out for a chain row: placeholder tokens at the provisional positions (the page
            # tables are the row's whole table, indexed by absolute position: any real p0 below the bound reads them).
            tokens = tokens.clone()
            for r in relayed:
                tokens[r] = 0
                self._check_chain_tables(tables, r, pending.p0_range(r)[0] + 1, int(positions[r, int(counts[r])]))
        if fused:
            spec_sampling = self._check_spec_sampling(spec_sampling, rows, tokens.shape[1] - 1)
        # The next verify writes the bonus and K drafts at positions bonus .. bonus + K: the chain drafts below that.
        limit = self.max_seq_len - max(tokens.shape[1] - 1, 1)
        lanes, chains, sources = [], [], [[] for _ in range(self.dp)]
        for lane, model in enumerate(self.models):
            first, last = lane * self.per_lane, min((lane + 1) * self.per_lane, rows)
            active = [
                (r, tokens[r].tolist(), positions[r].tolist(), int(counts[r]))
                for r in range(first, last)
                if int(positions[r, 0]) >= 0
            ]
            live = [r - first for r, *_ in active]
            self._seen(lane, live)
            groups = None
            if DRAFT_GROUPS > 1:  # each row in its request's draft group
                groups = {first + s: g for s, g in self._draft_groups(lane, live).items() if g is not None}
            # The grouped verify attention scans each group's KV once: one request per group (verify_grouped).
            passes = plan_verify_passes(active, groups=groups, aligned=KV_PAIR_SPLIT and verify_grouped())
            lanes.append(passes)
            lane_sources = sources[lane] = draft_sources(passes)
            # The draft sources run one pass: its chain row goes with it, a block per source.
            chained = accept and SPEC_CHAIN and bool(lane_sources) and self._drafter(lane) is not None
            chained = chained and model.rope_tables is not None
            relay = any(src.row in relayed for src in lane_sources)
            if relay and not all(src.row in relayed for src in lane_sources):
                raise RuntimeError(f"lane {lane}: chain rows {sorted(relayed)} beside unchained sources {lane_sources}")
            chains.append((lane_sources, limit, relay) if chained else None)
            # The decode trace's resident tokens and positions are stale now: the plugin reloads after a verify.
            self.decode_positions[lane] = None
        # A deferred verify's sources travel with its handle: a propose without one (an ordinary step's) drafts
        # nothing from them.
        self.spec_sources = [[] for _ in range(self.dp)] if defer else sources
        self.source_positions = [{} for _ in range(self.dp)]
        self.chained = [{} for _ in range(self.dp)]
        self._tip = None
        if not accept:
            return self._verify_unchained(spec_mode, lanes, tables, tokens.shape, t0)
        handle = VerifyHandle(rows, tokens.shape[1], positions[:, 0], fused)
        handle.relayed = set(relayed)
        handle.hints = counts.to(torch.int64).clone()
        handle.prev = pending if relayed else None
        for k in range(max(map(len, lanes))):
            for lane, p in enumerate(lanes):
                if k >= len(p):
                    continue
                model = self.models[lane]
                # The accept step's inputs are staged before the verify: its relay step writes a chain row's drafts.
                self._enqueue_verify_pass(lane, p[k], tables, chains[lane], pass_inputs(p[k], spec_sampling))
                model.verify_accept_enqueue()
                if chains[lane] is not None:
                    # MIMO_VLLM_TIMING splits the wait at the accept trace's end: verify vs chained draft.
                    if TIMING:
                        handle.timing_events[lane] = ttnn.record_event(model.mesh_device, 0)
                    rings = self.mtp_ring if model.mtp is not None else self.dflash_ring
                    for source in sources[lane]:
                        slot = source.row - lane * self.per_lane
                        if not chains[lane][2]:
                            # A pass the host built: a draft's first row is its column 0's position, and the rings
                            # may not hold the rows before it (a prompt over the deferred verify before it reset
                            # them). A relayed pass continues the draft chained behind the step it chained onto.
                            first = int(p[k].positions[source.slots[0]])
                            if model.mtp is not None:
                                self._backfill(lane, slot, first, source.group)
                            else:
                                self._dflash_continue(lane, slot, first, group=source.group)
                        # The group's rings now hold the request's rows up to the bonus - 1, which the page gives.
                        rings[lane][source.group] = (slot, handle)
                    self._drafter(lane).enqueue_chained()
                    handle.sources[lane] = sources[lane]
                handle.pages.append((lane, p[k], model.spec_chain, model.spec_chain.read_async()))
        for lane, p in enumerate(lanes):
            if p:
                handle.events[lane] = ttnn.record_event(self.models[lane].mesh_device, 0)
        chained = sum(map(len, handle.sources.values()))
        if defer:
            self._next_tip += 1
            handle.tip_id = self._next_tip
            self._tip = handle
            if self.chain_check:
                self._check_deferred(handle)
            if TIMING:
                self._spec_timing("verify", t0, 0.0, sum(map(len, lanes)), self._verify_tables_s, chained=chained)
            self._log_fused()
            return VerifyOutput(spec_mode="fused_sample", handle=handle)
        results = handle.read()
        for lane, lane_sources in handle.sources.items():
            result = next(res for ln, _, res in results if ln == lane)
            for source in lane_sources:
                self.chained[lane][source.row] = chained_draft(result, source, layers=self._mtp_layers(lane))
        if TIMING:
            self._spec_timing(
                "verify",
                t0,
                handle.waited,
                sum(map(len, lanes)),
                self._verify_tables_s,
                draft_wait=handle.draft_wait,
                chained=chained,
            )
        if fused:
            for lane, _, result in results:
                self.models[lane].verify_sampler.tally(result.accept)
            readout = handle.readout()
            self._log_fused()
            return VerifyOutput(
                spec_mode="fused_sample",
                accepted_token_ids=readout.accepted_token_ids,
                accepted_counts=readout.accepted_counts,
            )
        argmax = torch.zeros(tokens.shape, dtype=torch.int32)
        for _, vp, result in results:
            for s, row, col in vp.answers:
                argmax[row, col] = int(result.argmax[s])
        return VerifyOutput(spec_mode="argmax_ids", argmax_ids=argmax)

    def _check_chain(self, chain_rows, chain_tip, defer, tokens, positions, counts, accepted_counts):
        """The chain rows of a verify chained onto the deferred one in flight, {row: drafts to verify}, checked
        against the plugin's contract (docs/SPEC_DECODE_CONTRACT.md 4j): ``chain_tip`` is the newest deferred verify
        with no step after it; a chain row is the source of a lane whose draft chained behind that verify, carries
        placeholder tokens (-1) and accepted count -1, verifies 1 .. MAX_DRAFTS of the device's drafts, and is laid
        out from a provisional column-0 position at or past the highest the device can have put it; its lane runs
        nothing else (MiMo drafts for the requests of a lane step that ran one pass)."""
        if chain_rows is None:
            if chain_tip is not None:
                raise ValueError("chain_tip without chain_rows")
            return {}
        rows, width = tokens.shape
        chain_rows = torch.as_tensor(chain_rows)
        if chain_rows.dtype != torch.bool or tuple(chain_rows.shape) != (rows,):
            raise ValueError(f"chain_rows must be bool [{rows}], got {chain_rows.dtype} {tuple(chain_rows.shape)}")
        if not defer:
            raise ValueError("chain_rows come with a deferred fused_sample verify (defer_read)")
        pending = self._tip
        if pending is None or chain_tip != pending.tip_id:
            newest = None if pending is None else pending.tip_id
            raise ValueError(
                f"chain_tip {chain_tip} is not the newest deferred verify with no step after it ({newest})"
            )
        out = {}
        for r in torch.nonzero(chain_rows).reshape(-1).tolist():
            if r not in {src.row for src in pending.sources.get(r // self.per_lane, [])}:
                raise ValueError(
                    f"chain row {r}: deferred verify {chain_tip} drafted nothing for it on device (MiMo drafts behind "
                    "a lane step of one pass)"
                )
            if not bool((tokens[r] == -1).all()):
                raise ValueError(f"chain row {r} carries host tokens {tokens[r].tolist()}")
            if accepted_counts is not None and int(torch.as_tensor(accepted_counts).reshape(-1)[r]) != -1:
                raise ValueError(f"chain row {r} must carry accepted_counts -1")
            hint = int(counts[r])
            if not 1 <= hint <= min(MAX_DRAFTS, width - 1):
                raise ValueError(f"chain row {r} asks to verify {hint} drafts; the device drafted {MAX_DRAFTS}")
            bound = int(positions[r, 0])
            if positions[r, : 1 + hint].tolist() != list(range(bound, bound + 1 + hint)):
                raise ValueError(f"chain row {r}: positions {positions[r].tolist()} are not laid out from {bound}")
            high = pending.p0_range(r)[1] + 1 + int(pending.hints[r])
            if bound < high:
                raise ValueError(f"chain row {r}: provisional position {bound} below the device's possible {high}")
            out[r] = hint
        lanes = {r // self.per_lane for r in out}
        for r in range(rows):
            if r not in out and r // self.per_lane in lanes and int(positions[r, 0]) >= 0:
                raise ValueError(f"row {r} beside chain row(s) {sorted(out)} in lane {r // self.per_lane}")
        return out

    def _check_chain_tables(self, tables, row, low, high):
        """A chain row's page tables must hold a block (not vLLM's null block 0) at every position its verify may
        read or write: from the sliding window's start of the lowest real column-0 position ``low`` to the last
        provisional position ``high``. vLLM frees sliding-window blocks by the scheduler's optimistic length, which
        runs ahead of the device's real one; a hole is counted (``chain_stats["uncovered"]``) and logged: those
        positions read another block's K / V."""
        window = getattr(self.gen.cfg, "sliding_window", None) or 128
        for g, t in self._group_rows(tables, row).items():
            bs = self.group_block_sizes.get(g, BLOCK_SIZE)
            first, last = max(low - window + 1, 0) // bs, high // bs
            if last >= t.numel() or bool((t[first : last + 1] == 0).any()):
                self.chain_stats["uncovered"] += 1
                if self.chain_stats["uncovered"] % 64 == 1:
                    logger.warning(
                        f"MiMo chained verify: row {row}'s group {g} page table has no block for some of positions "
                        f"{first * bs} .. {high} ({self.chain_stats['uncovered']} such rows so far)"
                    )
                return

    def _chains_on_device(self):
        """A fused verify may be deferred (MIMO_SPEC_DEFER): every lane chains its draft behind a verify
        (MIMO_SPEC_CHAIN, the decode RoPE tables, a drafter)."""
        drafters = all(self._drafter(lane) is not None for lane in range(self.dp))
        return SPEC_DEFER and SPEC_CHAIN and drafters and all(m.rope_tables is not None for m in self.models)

    def _drafter(self, lane):
        """The lane's drafter: its TtMiMoMTP or TtDFlash (None: none)."""
        model = self.models[lane]
        return model.mtp if model.mtp is not None else model.dflash

    def _verify_unchained(self, spec_mode, lanes, tables, shape, t0):
        """A logits verify, or an argmax_ids verify with MIMO_SPEC_CHAIN=0: no accept trace; each pass's greedy
        tokens (and logit rows) read before the next pass of its lane overwrites them."""
        from vllm_tt_plugin.spec_decode import VerifyOutput

        argmax = torch.zeros(shape, dtype=torch.int32)
        logits = self._spec_logits(tuple(shape)) if spec_mode == "logits" else None
        waited = 0.0
        for k in range(max(map(len, lanes))):
            pending = []
            for lane, p in enumerate(lanes):
                if k < len(p):
                    pending.append((lane, p[k], self._enqueue_verify_pass(lane, p[k], tables)))
            for lane, vp, handle in pending:
                model = self.models[lane]
                t = time.perf_counter()
                out = model.read_next_tokens(handle)
                rows = model.decode_read_rows(handle) if logits is not None else None
                waited += time.perf_counter() - t
                for s, row, col in vp.answers:
                    argmax[row, col] = int(out[s])
                    if rows is not None:
                        logits[row, col] = rows[s]
        if TIMING:
            self._spec_timing("verify", t0, waited, sum(map(len, lanes)), self._verify_tables_s)
        if logits is not None:
            return VerifyOutput(spec_mode="logits", logits=logits)
        return VerifyOutput(spec_mode="argmax_ids", argmax_ids=argmax)

    def _check_spec_sampling(self, spec_sampling, rows, num_drafts):
        """The plugin's ``spec_sampling`` of a fused_sample verify (vllm-tt-plugin docs/SPEC_DECODE_CONTRACT.md):
        host tensors over the block's B rows: temperature fp32 [B] (0: greedy), top_k int32 [B] (below 1 or at least
        the vocabulary: none), top_p fp32 [B], greedy bool [B], uniforms fp32 [B, K + 2] in [0, 1) (column 0 the
        bonus, 1..K the accept draws, K + 1 the correction). None: every row greedy. A sampled row must be one the
        device sampler serves, as the plugin routes it (``model_capabilities``)."""
        if spec_sampling is None:
            return None
        want = {
            "temperature": ((rows,), torch.float32),
            "top_k": ((rows,), torch.int32),
            "top_p": ((rows,), torch.float32),
            "greedy": ((rows,), torch.bool),
            "uniforms": ((rows, num_drafts + 2), torch.float32),
        }
        if set(spec_sampling) != set(want):
            raise ValueError(f"spec_sampling keys {sorted(spec_sampling)} != {sorted(want)}")
        for name, (shape, dtype) in want.items():
            t = spec_sampling[name]
            if not isinstance(t, torch.Tensor) or tuple(t.shape) != shape or t.dtype != dtype:
                got = (tuple(t.shape), t.dtype) if isinstance(t, torch.Tensor) else type(t).__name__
                raise ValueError(f"spec_sampling[{name!r}] must be a {dtype} tensor of shape {shape}, got {got}")
        u = spec_sampling["uniforms"]
        if bool(((u < 0) | (u >= 1.0)).any()):
            raise ValueError("spec_sampling['uniforms'] must lie in [0, 1)")
        vocab = self.gen.cfg.vocab_size
        max_k = self.model_capabilities["max_device_top_k"]
        for r in torch.nonzero(~spec_sampling["greedy"]).reshape(-1).tolist():
            t = float(spec_sampling["temperature"][r])
            k, p = int(spec_sampling["top_k"][r]), float(spec_sampling["top_p"][r])
            unbounded = k < 1 or k >= vocab
            if t <= 0.0 or (not unbounded and k > max_k) or (unbounded and p >= 1.0):
                raise ValueError(
                    f"row {r}: device acceptance takes a temperature > 0 with top_k <= {max_k}, or none with "
                    f"top_p < 1; got temperature {t}, top_k {k}, top_p {p}"
                )
        return spec_sampling

    def _log_fused(self):
        """Every 1024 fused verifies: the requests accepted on device, and how many of them on truncated candidates
        (a nucleus reaching past the candidates: accepted on them, as the decode sampler draws)."""
        self._fused_steps = getattr(self, "_fused_steps", 0) + 1
        if self._fused_steps % 1024 == 0:
            groups = sum(m.verify_sampler.groups for m in self.models)
            truncated = sum(m.verify_sampler.truncated for m in self.models)
            logger.info(
                f"MiMo vLLM fused accept: {self._fused_steps} verifies, {groups} requests accepted on device, "
                f"{truncated} on truncated candidates"
            )

    def _spec_logits(self, shape):
        """The host buffer a logits verify of candidate block ``shape`` [B, 1 + K] fills, kept across steps: a
        fresh [B, 1 + K, vocab] fp32 tensor would be zeroed every step (16 x 4 x vocab: ~40 MB). Lane DP is
        synchronous, so the plugin has finished reading a step's logits before the next verify writes them."""
        buffers = getattr(self, "_spec_logit_buffers", None)
        if buffers is None:
            buffers = self._spec_logit_buffers = {}
        if shape not in buffers:
            buffers[shape] = torch.zeros(*shape, self.gen.cfg.vocab_size, dtype=torch.float32)
        return buffers[shape]

    def _enqueue_verify_pass(self, lane, vp, tables, chain=None, accept=None):
        """``chain``: ([DraftSource], limit, relay) of the drafts chained behind the pass, None when none does. The
        pass's chain row (reference.spec_chain.chain_row) carries them and, with the MTP history, each slot's request
        slot (the verify trace writes the slots' final norm rows into their requests' rings). ``accept``: its accept
        step's (slots, groups), staged before the verify (TtMiMoModel.verify_enqueue)."""
        model = self.models[lane]
        hslots = None
        if model.mtp is not None and model.mtp.history is not None:
            first = lane * self.per_lane
            hslots = [None if row is None else row - first for row in vp.table_rows]
        if chain is not None or hslots is not None:
            chain = chain_row(model.max_batch, *(chain or ([], 0, False)), hslots=hslots)
        tok = torch.zeros(model.max_batch, dtype=torch.int32)
        pos = torch.full((model.max_batch,), -1, dtype=torch.int32)
        tok[:SPEC_SLOTS] = torch.tensor(vp.tokens, dtype=torch.int32)
        pos[:SPEC_SLOTS] = torch.tensor(vp.positions, dtype=torch.int32)
        t = time.perf_counter()
        # A request's candidates fill several slots: build its group rows once, not once per slot.
        by_row = {}
        for row in vp.table_rows:
            if row is not None and row not in by_row:
                by_row[row] = self._group_rows(tables, row)
        model.set_vllm_page_tables({s: by_row[row] for s, row in enumerate(vp.table_rows) if row is not None})
        self._verify_tables_s += time.perf_counter() - t
        chain = None if chain is None else torch.tensor(chain, dtype=torch.int32)
        return model.verify_enqueue(tok, pos, chain, accept=accept)

    def propose_draft_tokens(self, num_drafts, committed_tokens, committed_positions, accepted_counts, hidden=None):
        """Drafts for the next step: the drafter drafts for the requests of each lane whose last step ran one pass (one
        per draft group), from that step's resident final hidden / taps; every other row offers none. After a verify that chained the lane's draft on
        device (``_verify``), the drafts are already on the host: they are returned when the plugin committed what the
        device committed (the row's count, tokens and positions). Otherwise (a plain decode step, a logits verify, no
        decode RoPE tables, MIMO_SPEC_CHAIN=0, a draft the device declined at max_seq_len - K that ``num_drafts``
        allows, or a different commit) the MTP drafts now from the host plan (reference.spec_decode.plan_mtp), or with
        every MTP layer from the commit's description (the device plans the passes: their rows depend on its stash). A
        propose for an ordinary step older than the lane's last one (async scheduling) drafts nothing on device
        (``_stale_source``). Only the lanes it consumed forget their source. ``hidden``: a deferred verify's handle,
        whose propose returns the drafts its pages carry (``_propose_deferred``)."""
        from vllm_tt_plugin.spec_decode import DraftOutput

        if isinstance(hidden, VerifyHandle):
            return self._propose_deferred(hidden, num_drafts, committed_tokens, committed_positions, accepted_counts)
        t0 = time.perf_counter()
        committed_tokens = torch.as_tensor(committed_tokens)
        committed_positions = torch.as_tensor(committed_positions)
        counts = torch.as_tensor(accepted_counts).reshape(-1)
        rows = committed_tokens.shape[0]
        drafts = torch.zeros(rows, num_drafts, dtype=torch.int32)
        valid = torch.zeros(rows, dtype=torch.int32)
        jobs, inputs_s, hits, kept = [], 0.0, 0, set()
        for lane, lane_sources in enumerate(self.spec_sources):
            drafter = self._drafter(lane)
            if not lane_sources or drafter is None or not 1 <= num_drafts <= MAX_DRAFTS:
                continue
            commits = []  # the sources the host plans a draft for: (source, tokens, positions, m, hit)
            for source in lane_sources:
                if source.row >= rows:
                    continue
                r, m = source.row, int(counts[source.row])
                # A row whose request is gone has position -1; the next verify writes the bonus token and the drafts
                # at positions bonus .. bonus + num_drafts.
                if not 0 <= int(committed_positions[r, m - 1]) < self.max_seq_len - num_drafts:
                    continue
                stale = self._stale_source(lane, r, committed_positions, m)
                if stale is not None:
                    # The source is a newer step's: kept for that step's propose.
                    kept.add(lane)
                    if stale:
                        drafts[r] = committed_tokens[r, m - 1].to(torch.int32)
                        valid[r] = num_drafts
                    continue
                chained = self.chained[lane].get(r)
                hit = chained is not None
                hit = hit and chained.matches(committed_tokens[r].tolist(), committed_positions[r].tolist(), m)
                if hit:
                    drafts[r] = torch.tensor(chained.drafts[:num_drafts], dtype=torch.int32)
                    valid[r] = num_drafts
                    hits += 1
                    if not self.chain_check:
                        continue
                commits.append((source, committed_tokens[r].tolist(), committed_positions[r].tolist(), m, hit))
            if commits:
                anchors = self._enqueue_host_drafts(lane, [c[:4] for c in commits])
                inputs_s += drafter.last_inputs_s
                jobs += [(lane, c[0], anchor, c[4]) for c, anchor in zip(commits, anchors)]
        waited = 0.0
        for lane, source, anchor, hit in jobs:
            r = source.row
            t = time.perf_counter()
            host = torch.tensor(self._read_drafts(lane, anchor, source.group)[:num_drafts], dtype=torch.int32)
            waited += time.perf_counter() - t
            if hit:
                self._check_chained(lane, drafts[r], host)
                continue
            drafts[r] = host
            valid[r] = num_drafts
        planned = sum(not hit for *_, hit in jobs)
        self.chain_stats["chained"] += hits
        self.chain_stats["host"] += planned
        for lane in range(self.dp):
            if lane not in kept:
                self.spec_sources[lane], self.source_positions[lane], self.chained[lane] = [], {}, {}
        if TIMING:
            self._spec_timing("propose", t0, waited, planned, inputs_s, chained=hits)
        return DraftOutput(draft_token_ids=drafts, num_valid=valid)

    def _read_drafts(self, lane, anchor, group=0):
        """Draft group ``group``'s drafts of the lane's last host-planned draft (blocks): the MTP's of slot ``anchor``,
        or DFlash's."""
        model = self.models[lane]
        if model.mtp is not None:
            return model.mtp.read_drafts(anchor, group)
        return model.dflash.read_drafts(group)

    def _enqueue_host_drafts(self, lane, commits):
        """The drafter's host-planned draft for ``commits`` [(DraftSource, tokens, positions, m)] of the lane's last
        step, one per draft group, in one program (non-blocking). Returns each commit's anchor slot (MTP's; None with
        DFlash)."""
        if self.models[lane].mtp is None:
            return self._enqueue_dflash(lane, commits)
        return self._enqueue_host_draft(lane, commits)

    def _enqueue_host_draft(self, lane, commits):
        """The MTP draft of the host plan for ``commits`` [(DraftSource, tokens, positions, m)]: per commit of ``m``
        tokens ``tokens`` at ``positions`` (non-blocking), behind the backfill when its group's rings do not hold the
        request's rows before it: with one layer (one commit) reference.spec_decode.plan_mtp, with every layer the
        commits' descriptions (the device plans the passes: their rows depend on its stash). It reads the resident
        final hidden, which must still be that step's. Returns each commit's anchor slot, whose drafts ``read_drafts``
        returns."""
        mtp = self.models[lane].mtp
        for source, tokens, positions, m in commits:
            slot = source.row - lane * self.per_lane
            self._backfill(lane, slot, int(positions[0]) - 1, source.group)
            self.mtp_ring[lane][source.group] = (slot, int(positions[m - 1]) - 1)
        if mtp.multi:
            mtp.enqueue_layers(commits)
            return [(source.group + 1) * GROUP_SLOTS - 1 for source, *_ in commits]
        assert len(commits) == 1, "one MTP layer drafts for one request"
        ((source, tokens, positions, m),) = commits
        plan = plan_mtp(source, positions, m)
        slot_tokens = [0] * SPEC_SLOTS
        for c in range(m):
            slot_tokens[source.slots[c]] = int(tokens[c])
        mtp.enqueue(slot_tokens, plan)
        return [plan.anchor]

    def _draft_groups(self, lane, slots, prompt=False):
        """{slot: draft group} of the requests in lane ``slots`` (a step's live slots, or a prompt's slot): a request
        keeps its group while it runs, so its drafter state (rings, stash) stays its own. A slot without one takes a
        group no live slot owns, the one whose request ran least recently; None when every group belongs to another
        live slot (more requests than groups: the step drafts for none). A prompt's slot always takes one (the request
        that owned it is gone or will be found without one). One draft group: always 0."""
        if DRAFT_GROUPS == 1:
            return {s: 0 for s in slots}
        owners, seen = self.group_owners[lane], self.slot_seen[lane]
        out = {}
        for s in slots:
            if s in owners:
                out[s] = owners.index(s)
                continue
            taken = set(slots) if not prompt else {s}
            free = [g for g in range(DRAFT_GROUPS) if owners[g] is None or owners[g] not in taken]
            if not free:
                out[s] = None
                continue
            g = min(free, key=lambda g: (owners[g] is not None, seen.get(owners[g], -1)))
            owners[g], out[s] = s, g
            self.mtp_ring[lane][g] = self.dflash_ring[lane][g] = None  # another request's rows
        return out

    def _seen(self, lane, slots):
        """A step ran the lane's ``slots`` (the groups' recency, ``_draft_groups``)."""
        self._steps += 1
        for s in slots:
            self.slot_seen[lane][s] = self._steps

    def _dflash_continue(self, lane, slot, first, last=None, group=0):
        """Before DFlash context rows ``first`` .. ``last`` of the request in lane ``slot`` (a prefill chunk, or a
        draft's committed rows) in draft group ``group``: reset the group's ring unless it holds that request's rows
        up to ``first`` - 1 (a new request, another request's rows, or a gap the lane did not draft), then record it
        up to ``last`` (None: a chained draft, which the caller records)."""
        dflash = self.models[lane].dflash
        if dflash is None:
            return
        ring = self._ring(lane, self.dflash_ring, group)
        if ring is None or ring[0] != slot or ring[1] < first - 1:
            dflash.reset(group)
            self.chain_stats["dflash_resets"] += 1
        if last is not None:
            held = max(last, ring[1]) if ring is not None and ring[0] == slot else last
            self.dflash_ring[lane][group] = (slot, held)

    def _enqueue_dflash(self, lane, commits):
        """The DFlash draft for ``commits`` [(DraftSource, tokens, positions, m)] (non-blocking): per commit of ``m``
        tokens ``tokens`` at ``positions``, the committed columns' slots of the last step hold the target layers'
        outputs at positions ``positions[c] - 1``, which become its group's ring context; its anchor is the last
        committed token. It reads the taps that step left, which must still be its own. Returns None per commit
        (read_drafts takes no anchor)."""
        drafts = []
        for source, tokens, positions, m in commits:
            slot = source.row - lane * self.per_lane
            first, last = int(positions[0]) - 1, int(positions[m - 1]) - 1
            self._dflash_continue(lane, slot, first, last, source.group)
            context = [(source.slots[c], int(positions[c]) - 1) for c in range(m)]
            drafts.append((source.group, context, int(tokens[m - 1]), int(positions[m - 1]), source.slots[m - 1]))
        self.models[lane].dflash.enqueue(drafts)
        return [None] * len(commits)

    def _check_deferred(self, handle):
        """MIMO_SPEC_CHAIN_CHECK on a deferred verify, right after its enqueue: waits for its pages, then drafts each
        draft chained behind a pass again from the host plan of the commit the device drafted from, while nothing is
        enqueued behind the verify (the resident hidden is still its own), and compares. The plugin reads the handle's
        page copies as before, and a verify chained onto this one relays the page the redraft rewrote with the same
        drafts; a mismatch is logged, and the relay then verifies the host plan's drafts."""
        results = handle.read()
        jobs = []
        for lane, lane_sources in handle.sources.items():
            result = next(res for ln, _, res in results if ln == lane)
            redrafts = []
            for source in lane_sources:
                chained = chained_draft(result, source, layers=self._mtp_layers(lane))
                if chained is not None:
                    redrafts.append((source, chained))
            if redrafts:
                commits = [(src, c.tokens, c.positions, len(c.tokens)) for src, c in redrafts]
                anchors = self._enqueue_host_drafts(lane, commits)
                jobs += [(lane, src, c, anchor) for (src, c), anchor in zip(redrafts, anchors)]
        for lane, source, chained, anchor in jobs:
            host = torch.tensor(self._read_drafts(lane, anchor, source.group), dtype=torch.int32)
            self._check_chained(lane, torch.tensor(chained.drafts, dtype=torch.int32), host)

    def _propose_deferred(self, handle, num_drafts, committed_tokens, committed_positions, accepted_counts):
        """The propose of a deferred verify (once, on the engine thread, after its read, possibly steps later): the
        drafts the device chained behind each source pass, from that step's pages, when the plugin committed what the
        device did; nothing is enqueued and nothing waits on later steps. A row whose request is gone (position -1),
        and a draft the chain declined at the context end, offer none: the resident hidden a host plan would read
        may be a later step's."""
        from vllm_tt_plugin.spec_decode import DraftOutput

        t0 = time.perf_counter()
        if handle.proposed:
            raise RuntimeError(f"deferred verify {handle.tip_id} is proposed a second time")
        handle.proposed = True
        results = handle.read()
        committed_tokens = torch.as_tensor(committed_tokens)
        committed_positions = torch.as_tensor(committed_positions)
        counts = torch.as_tensor(accepted_counts).reshape(-1)
        rows = committed_tokens.shape[0]
        drafts = torch.zeros(rows, num_drafts, dtype=torch.int32)
        valid = torch.zeros(rows, dtype=torch.int32)
        if handle.fused:
            for lane, _, result in results:
                self.models[lane].verify_sampler.tally(result.accept)
        hits = 0
        for lane, lane_sources in handle.sources.items():
            result = next(res for ln, _, res in results if ln == lane)
            for source in lane_sources:
                chained = chained_draft(result, source, layers=self._mtp_layers(lane))
                r = source.row
                if chained is None or r >= rows or int(committed_positions[r, 0]) < 0:
                    continue
                if not 1 <= num_drafts <= MAX_DRAFTS:
                    continue
                if chained.matches(committed_tokens[r].tolist(), committed_positions[r].tolist(), int(counts[r])):
                    drafts[r] = torch.tensor(chained.drafts[:num_drafts], dtype=torch.int32)
                    valid[r] = num_drafts
                    hits += 1
        self.chain_stats["chained"] += hits
        if TIMING:
            self._spec_timing("propose", t0, 0.0, 0, chained=hits, verify_wait=handle.waited + handle.draft_wait)
        return DraftOutput(draft_token_ids=drafts, num_valid=valid)

    def _mtp_layers(self, lane):
        """The MTP layers drafting on ``lane`` (1, or MAX_DRAFTS: the chain's anchor is then the last slot); DFlash's
        chain anchors at the last committed column, as one layer's."""
        mtp = self.models[lane].mtp
        return 1 if mtp is None else len(mtp.layers)

    def _ring(self, lane, rings=None, group=0):
        """``rings[lane][group]`` (default ``mtp_ring``; ``dflash_ring``): (lane slot, last row) or None. A draft
        chained behind a verify left (slot, its VerifyHandle): the rows run up to its page's bonus - 1, or, when the
        chain declined (nothing written), unknown (None). Reads the handle on the engine thread (by then the plugin
        has read it: a later step was built from it), so the answer is ordered with the prefills that reset the
        ring."""
        rings = self.mtp_ring if rings is None else rings
        ring = rings[lane][group]
        if ring is not None and isinstance(ring[1], VerifyHandle):
            chain = next(res for ln, _, res in ring[1].read() if ln == lane).groups[group]
            drafted = chain.drafting and chain.draft_anchor == chain.anchor
            ring = rings[lane][group] = (ring[0], chain.bonus - 1) if drafted else None
        return ring

    def _backfill(self, lane, slot, a, group=0):
        """Before a draft of the request in lane ``slot`` whose first row is at ``a`` (non-blocking, enqueued right
        before it), in draft group ``group``: the backfill trace when the group's MTP rings do not hold that request's
        rows up to a - 1 (after its prompt, after a gap without drafts: ``mtp_ring``), with the MTP history
        (tt/mtp_backfill.py)."""
        mtp = self.models[lane].mtp
        if mtp.history is None or slot >= SPEC_SLOTS:
            return
        ring = self._ring(lane, group=group)
        if ring is None or ring[0] != slot or ring[1] < a - 1:
            mtp.enqueue_backfill(slot, a, group)
            self.chain_stats["backfills"] += 1

    def _stale_source(self, lane, row, committed_positions, count):
        """None when the lane's DraftSource is the step this propose is for. Under async scheduling the plugin
        proposes for an ordinary step when it commits it, and the lane's next step may be enqueued by then: the
        source (and the resident hidden the MTP reads) is then that newer step's, whose token sits one position past
        the one proposed for. True for that case: the propose enqueues nothing and offers the committed token as
        placeholder drafts, so that the plugin drains the newer step before it builds a verify and proposes for it
        (which drafts on device); the placeholders never reach a verify, and would only be rejected if they did.
        False for any other position (no drafts, counted in ``chain_stats``)."""
        expected = self.source_positions[lane].get(row)
        if expected is None:
            return None
        got = int(committed_positions[row, count - 1])
        if got == expected:
            return None
        stale = count == 1 and got == expected - 1
        self.chain_stats["stale" if stale else "misplaced"] += 1
        return stale

    def _check_chained(self, lane, chained, host):
        """MIMO_SPEC_CHAIN_CHECK: a draft returned from the chain against the host plan's draft of the same commit."""
        stats = self.chain_stats
        stats["checked"] += 1
        if not torch.equal(chained, host):
            stats["mismatched"] += 1
            logger.warning(
                f"MiMo spec chain check: lane {lane}: chained drafts {chained.tolist()}, host {host.tolist()}"
            )
        if stats["checked"] % 64 == 0:
            logger.info(
                f"MiMo spec chain check: {stats['checked']} chained drafts checked, {stats['mismatched']} differ"
            )

    def _spec_timing(self, kind, t0, waited, count, part=0.0, draft_wait=0.0, chained=0, verify_wait=0.0):
        """Every 128 verifies: per verify step, the mean host time of the verify (its enqueues: page tables, uploads,
        the verify, accept and chained draft traces) and its time blocked reading back, split at the accept trace's
        end on a lane that chained its draft (the verify's device share, then the draft's); the plugin's accept walk
        (from the verify's return to propose_draft_tokens); propose_draft_tokens' host time (with the host plan's
        inputs) and its time blocked on host-planned drafts; and the gap from its return to the next verify (the
        plugin's commit and vLLM's scheduling and output). ``count``: the verify's passes or the lanes the propose
        drafted from the host plan; ``chained``: the lanes whose draft the verify chained, or the propose returned.
        A deferred verify's wait is its read's (on the plugin's reader thread), counted at its propose
        (``verify_wait``)."""
        now = time.perf_counter()
        stats = getattr(self, "_spec_timing_stats", None) or {
            "verifies": 0,
            "passes": 0,
            "chained": 0,
            "hits": 0,
            "drafts": 0,
            "verify_host": 0.0,
            "verify_wait": 0.0,
            "draft_wait": 0.0,
            "propose_host": 0.0,
            "propose_wait": 0.0,
            "verify_tables": 0.0,
            "draft_inputs": 0.0,
            "accept": 0.0,
            "accepts": 0,
            "gap": 0.0,
            "gaps": 0,
            "verified": None,
            "last": None,
            "since": t0,
        }
        if kind == "verify":
            if stats["last"] is not None:
                stats["gap"] += t0 - stats["last"]
                stats["gaps"] += 1
            stats["verifies"] += 1
            stats["passes"] += count
            stats["chained"] += chained
            stats["verify_host"] += now - t0 - waited - draft_wait
            stats["verify_wait"] += waited
            stats["draft_wait"] += draft_wait
            stats["verify_tables"] += part
            stats["verified"] = now
        else:
            if stats["verified"] is not None:
                stats["accept"] += t0 - stats["verified"]
                stats["accepts"] += 1
                stats["verified"] = None
            stats["drafts"] += count
            stats["hits"] += chained
            stats["verify_wait"] += verify_wait
            stats["propose_host"] += now - t0 - waited
            stats["propose_wait"] += waited
            stats["draft_inputs"] += part
            stats["last"] = now
        if kind == "propose" and stats["verifies"] >= 128:
            n = stats["verifies"]

            def ms(key, per=n):
                return f"{1e3 * stats[key] / max(per, 1):.2f}"

            logger.info(
                f"MiMo vLLM timing: spec {n} verifies in {1e3 * (now - stats['since']):.0f} ms "
                f"({1e3 * (now - stats['since']) / n:.2f} ms/step), {stats['passes']} passes, {stats['chained']} lane "
                f"drafts chained on device ({stats['hits']} returned), {stats['drafts']} host-planned; per step: "
                f"verify host {ms('verify_host')} (tables {ms('verify_tables')}) + wait {ms('verify_wait')} ms, "
                f"chained draft wait {ms('draft_wait')} ms, accept walk {ms('accept', stats['accepts'])} ms, "
                f"propose host {ms('propose_host')} (inputs {ms('draft_inputs')}) + wait {ms('propose_wait')} ms, "
                f"gap {ms('gap', stats['gaps'])} ms"
            )
            stats = None
        self._spec_timing_stats = stats

    def _check_sampling(self, lane, model, handle, rng, active):
        """MIMO_SAMPLER_CHECK: the step's tokens against the host's draw on its logit rows (blocking)."""
        logits = model.plan.to_torch_tp_concat(handle["logits_rows"], dim=-1).float().reshape(model.decode_users, -1)
        tokens = ttnn.to_torch(ttnn.get_device_tensors(handle["next_tokens"])[0]).reshape(-1)
        checked, exact, bad = model.sampler.check(logits, tokens, rng, active)
        stats = getattr(self, "_check_stats", None) or {"steps": 0, "rows": 0, "exact": 0, "bad": 0}
        stats["steps"] += 1
        stats["rows"] += checked
        stats["exact"] += exact
        stats["bad"] += len(bad)
        for r, got, want in bad:
            logger.warning(f"MiMo sampler check: lane {lane} row {r}: device {got}, host {want}")
        if stats["steps"] % 64 == 0:
            logger.info(
                f"MiMo sampler check: {stats['steps']} lane-steps, {stats['rows']} rows, {stats['exact']} with the "
                f"nucleus in the candidates, {stats['bad']} mismatches"
            )
        self._check_stats = stats

    def _set_sampling(self, lane, model, sampling_params, first, last, pos):
        """Uploads the sampling parameters of the lane's rows [first, last) (idle rows, position -1, greedy) and their
        random streams, and records whether any row samples. A seeded row's stream is keyed by (seed, position)
        (device_sampler.stream_states), so its draws do not depend on which steps reload it. A sampled row must be
        within what the plugin routes to the device (``model_capabilities``): top_k up to ``max_device_top_k``, or
        none with top_p < 1."""
        n = last - first
        temperature, top_k, top_p = self._check_device_sampling(sampling_params, first, last, pos[:n] >= 0)
        seeds = list(sampling_params.seed[first:last]) if sampling_params.seed is not None else [None] * n
        rows = model.max_batch
        states = stream_states(seeds + [None] * (rows - n), pos.tolist())
        hosts, sampled = model.sampler.host_params(temperature, top_k, top_p, states=states)
        model.sampler.set_params(hosts)
        self.lane_sampled[lane] = sampled
        self.lane_grammar[lane] = False  # the parameters flag no grammar row now

    def _check_device_sampling(self, sampling_params, first, last, active):
        """The temperature (0 on idle rows), top_k and top_p of rows [first, last) (torch, each [last - first]); raises
        when an active sampled row asks for more than the device sampler serves (``_set_sampling``)."""
        temperature = torch.tensor([float(t) for t in sampling_params.temperature[first:last]])
        top_k = torch.tensor([int(k) for k in sampling_params.top_k[first:last]])
        top_p = torch.tensor([float(p) for p in sampling_params.top_p[first:last]])
        temperature = torch.where(active, temperature, torch.zeros_like(temperature))
        vocab = self.gen.cfg.vocab_size
        max_k = self.model_capabilities["max_device_top_k"]
        for s in range(last - first):
            if temperature[s] == 0 or top_k[s] == 1:
                continue
            k, p = int(top_k[s]), float(top_p[s])
            unbounded = k < 1 or k >= vocab
            if (not unbounded and k > max_k) or (unbounded and p >= 1.0):
                raise ValueError(
                    f"row {first + s}: device sampling takes top_k <= {max_k}, or none with top_p < 1; got "
                    f"temperature {float(temperature[s])}, top_k {k}, top_p {p}"
                )
        return temperature, top_k, top_p

    def _set_grammar(self, lane, model, grammar, first, last, active):
        """Uploads the grammar mask of the lane's rows [first, last) (``grammar``'s rows, or None) and flags its active
        constrained rows in the sampler's parameters, or lifts the flags the lane's last mask left; returns whether a
        row of the lane is constrained."""
        rows = None
        if grammar is not None:
            mask = grammar[first:last]
            rows = active[: last - first] & (mask != -1).any(dim=1)
        if rows is not None and bool(rows.any()):
            model.sampler.set_grammar(mask, rows)
            self.lane_grammar[lane] = True
            return True
        if self.lane_grammar[lane]:
            model.sampler.set_grammar()
            self.lane_grammar[lane] = False
        return False

    def read_decode_output(self, out, async_read=False, **kwargs):
        """Starts reading a decode step's output: its greedy tokens (device sampling), without blocking when
        ``async_read`` (returns the per-lane host tensors and their read events), or its logit rows (host sampling;
        read now, returned with no events). A verify's output is on the host already (the plugin unwraps its mode's
        tensor, or passes a fused_sample VerifyOutput whole): returned as it is, with no events. A deferred verify's
        handle (VerifyHandle; any thread, blocking on its step's events) returns its FusedReadout."""
        if kwargs.get("sample_rows") is not None:
            raise ValueError("MiMo declares no selective host readback; got sample_rows")
        if isinstance(out, VerifyHandle):
            if async_read or kwargs:
                raise ValueError("a deferred verify's handle is read with read_decode_output(handle) alone")
            return out.readout()
        if isinstance(out, torch.Tensor) or hasattr(out, "spec_mode"):
            return (out, []) if async_read else out
        if not out["tokens"]:
            logits = self._read_logits(out)
            return (logits, []) if async_read else logits
        reads, events = [], []
        for entry in out["handles"]:
            if entry is None:
                continue
            lane, n, handle = entry
            reads.append((lane, n, handle["next_tokens"].cpu(blocking=not async_read)))
            if async_read:
                events.append(ttnn.record_event(self.models[lane].mesh_device, 0))
        host = {"rows": out["rows"], "reads": reads}
        return (host, events) if async_read else host

    def _read_logits(self, out):
        vocab = self.gen.cfg.vocab_size
        logits = torch.zeros(out["rows"], 1, vocab, dtype=torch.float32)
        for entry in out["handles"]:
            if entry is None:
                continue
            lane, n, handle = entry
            logits[lane * self.per_lane : lane * self.per_lane + n, 0] = self.models[lane].decode_read_rows(handle)[:n]
        return logits

    def process_decode_output_host(self, out, is_tokens=False, **kwargs):
        """Host output of ``read_decode_output``: greedy tokens int32 [B] in slot order (``is_tokens``; idle slots
        0), else the logits it read."""
        if kwargs.get("sample_rows") is not None:
            raise ValueError("MiMo declares no compact host logits; got sample_rows")
        if not is_tokens:
            if not isinstance(out, torch.Tensor):
                raise ValueError("host sampling: read_decode_output returns the logits")
            return out
        if isinstance(out, torch.Tensor) or "reads" not in out:
            raise ValueError("device sampling: expected the token reads of read_decode_output")
        tokens = torch.zeros(out["rows"], dtype=torch.int32)
        for lane, n, host in out["reads"]:
            row = ttnn.to_torch(ttnn.get_device_tensors(host)[0]).reshape(-1)[:n]
            tokens[lane * self.per_lane : lane * self.per_lane + n] = row.to(torch.int32)
        return tokens

    # ------------------------------------------------------------------ device grammar (module docstring)
    def enable_device_grammar(self):
        """The plugin calls it once after load, when it may sample structured decode steps on device. The sampler's
        mask buffers exist from the KV cache setup on, before any capture, and the sampled decode trace applies them,
        so this only checks that every replica has a device sampler (at most MAX_USERS decode users); otherwise
        ``device_grammar_enabled`` stays False and the plugin keeps host grammar."""
        if self.per_lane > MAX_USERS:
            logger.warning(
                f"MiMo vLLM: device grammar needs at most {MAX_USERS} decode users per lane (got {self.per_lane}): "
                "structured output samples on host"
            )
            return
        self.device_grammar_enabled = True

    def sample_deferred_decode(self, payload, sampling_params=None, *, grammar_bitmask=None):
        """Runs the deferred step ``payload`` (``decode_forward(defer_device_sampling=True)``) with the rows'
        ``sampling_params`` and ``grammar_bitmask``: vLLM's packed mask, int32 [rows, ceil(vocab / 32)] in decode row
        order (lane after lane, rows past it and rows without a grammar allowed everything). Returns what a
        device-sampled ``decode_forward(..., read_from_device=False)`` returns. Checks everything before any state
        changes; once the step started, a failure is final (every later decode raises)."""
        if self._grammar_failure is not None:
            raise RuntimeError(
                f"MiMo device grammar sampling failed after a decode submission: {self._grammar_failure}"
            )
        if not isinstance(payload, DeferredDecode) or payload is not self._deferred:
            raise ValueError("not the pending deferred decode (stale, already sampled, or never submitted)")
        if sampling_params is None:
            raise ValueError("sample_deferred_decode needs the rows' sampling_params")
        if any(bool(x) for x in (getattr(sampling_params, "enable_log_probs", None) or [])):
            raise ValueError("MiMo's device sampler returns no logprobs: a logprobs request samples on host")
        if grammar_bitmask is None:
            raise ValueError("sample_deferred_decode needs the step's grammar_bitmask")
        rows = self.dp * self.per_lane
        vocab = self.gen.cfg.vocab_size
        grammar = format_grammar_bitmask(
            grammar_bitmask, vocab_size=vocab, padded_vocab_size=-(-vocab // 32) * 32, max_batch_size=rows
        )
        args = payload.args
        if args["reload_inputs"]:
            positions = torch.as_tensor(args["start_pos"]).reshape(-1)
        elif any(p is None for p in self.decode_positions):
            raise RuntimeError("a steady decode before any reloading decode")
        else:
            positions = torch.cat([p[: self.per_lane] for p in self.decode_positions])
        self._check_device_sampling(sampling_params, 0, positions.numel(), positions >= 0)
        self._deferred = None
        try:
            return self._decode(**args, sampling_params=sampling_params, read_from_device=False, grammar=grammar)
        except Exception as e:
            self._grammar_failure = f"{type(e).__name__}: {e}"
            raise

    # ------------------------------------------------------------------ warmup and lifetime
    def _max_prefill_tokens(self):
        """The most tokens one prefill chunk of the model can hold: vLLM's per-step token budget plus the
        ``_chunk_align() - 1`` positions prefill_forward recomputes in front of an unaligned start, at most the model's
        max_prefill_chunk (prefill_enqueue splits longer chunks) and the context."""
        budget = self.prefill_token_budget
        if budget is None and self._kv_cache_spec_source is not None:
            budget = self._kv_cache_spec_source[0].scheduler_config.max_num_batched_tokens
        model = self.models[0]
        if budget is None:
            budget = model.max_prefill_chunk
        return min(budget + self._chunk_align() - 1, model.max_prefill_chunk, self.max_seq_len)

    def warmup_model_prefill(self, kv_cache=None, enable_trace=False, can_sample_on_device=False, **kwargs):
        """Compile every prefill program a served chunk can run, on every replica, before any decode trace is
        captured (prefill runs eagerly, so ``enable_trace`` changes nothing). One chunk per padded length at start 0
        and per later start that changes what the programs key on (model.prefill_warmup_chunks), each through one
        layer of each kind (TtMiMoModel.prefill_warmup_layers). Writes only KV blocks no request owns yet."""
        if self.already_warmed_up_prefill or self.prefill_programs_compiled:
            self.already_warmed_up_prefill = True
            return
        t0 = time.time()
        # Decode makes persistent L1 buffers on its first step (the MoE fabric row gather's among them). Layernorm and
        # some reductions size their circular buffers from the free L1 and key the program on that plan
        # (ttnn/cpp/ttnn/operations/core/program_cache_l1.hpp), so prefill is compiled with those buffers in place, as
        # served prefills run. The plugin warms prefill before decode and, with trace_mode decode_only, not again after.
        idle = torch.full((self.models[0].max_batch,), -1, dtype=torch.int32)  # every slot idle: no KV is written
        handles = [(m, m.decode_enqueue(torch.zeros_like(idle), idle)) for m in self.models if m.decode_trace is None]
        for m, handle in handles:
            m.decode_read(handle)
        model0 = self.models[0]
        groups = sorted(set(self.layer_groups))
        widths = {g: model0.vllm_tables[g].shape[1] for g in groups}  # a pair-split global table is half as wide
        width = max(widths.values())
        num_blocks = self.kv_caches[0][0][0].shape[0]
        window = next((l.attn.sliding_window for l in model0.layers if l.is_swa), BLOCK_SIZE)
        max_tokens = self._max_prefill_tokens()
        chunks = prefill_warmup_chunks(max_tokens, self.max_seq_len, BLOCK_SIZE, window, self.pair_split)
        layers = model0.prefill_warmup_layers()
        # Slot 0 of every group maps positions from 0 to blocks 1, 2, ..: free while vLLM has not started.
        blocks = torch.arange(1, min(width, num_blocks - 1) + 1, dtype=torch.int32)
        if blocks.numel() * BLOCK_SIZE < max(start for _, start in chunks):  # the later chunks' window blocks
            raise RuntimeError(f"{num_blocks} KV blocks cannot back the prefill warmup's chunks {chunks}")
        for model in self.models:
            model.set_vllm_page_tables({0: {g: blocks[: widths[g]] for g in groups}})
        programs = [m.mesh_device.num_program_cache_entries() for m in self.models]
        tokens = torch.arange(max_tokens, dtype=torch.int32) % 1000 + 1000
        for n, start in chunks:
            handles = [(m, m.prefill_enqueue(tokens[:n], 0, start_pos=start, layers=layers)) for m in self.models]
            for m, h in handles:
                m.prefill_read(h)
        # Batches of first chunks (prefill_forward): every user count and padded length a lane's step can hold. Slots
        # 1 .. users - 1 write the blocks after slot 0's.
        batches = prefill_warmup_batches(max_tokens, self.per_lane, model0.PREFILL_BATCH_MAX_PADDED, self.max_seq_len)
        batches = batches if PREFILL_BATCH else []  # prefill_forward runs every batchable first chunk as a batch
        span = -(-model0.PREFILL_BATCH_MAX_PADDED // BLOCK_SIZE)
        if self.per_lane * span > blocks.numel():
            raise RuntimeError(f"{num_blocks} KV blocks cannot back the prefill warmup's batches {batches}")
        for model in self.models:
            model.set_vllm_page_tables(
                {s: {g: blocks[s * span :][: widths[g]] for g in groups} for s in range(self.per_lane)}
            )
        for users, n in batches:
            chunks_of = [(tokens[:n], s) for s in range(users)]
            handles = [(m, m.prefill_enqueue_batch(chunks_of, layers=layers)) for m in self.models]
            for m, h in handles:
                m.prefill_read(h)
        for model in self.models:
            empty = {g: torch.zeros(widths[g], dtype=torch.int32) for g in groups}
            model.set_vllm_page_tables({s: dict(empty) for s in range(self.per_lane)})
        # Traced batches: their persistent buffers now, before any capture, and one eager pass each through every
        # layer (compiling whatever the capture will record); warmup_model_decode captures them after the decode trace.
        if PREFILL_TRACE and PREFILL_BATCH:
            for model in self.models:
                for users, n in batches:
                    padded = prefill_padded_len(n)
                    if (users, padded) not in model.prefill_traces:
                        model.setup_prefill_batch_trace(users, padded)
                        model.warm_prefill_batch_trace(users, padded)
            for model in self.models:
                ttnn.synchronize_device(model.mesh_device)
        # Traced later chunks: the same, for a chunk of the step's whole budget and for the longest chunk (an unaligned
        # start moved back), each SDPA program a start can bring.
        chunk_traces = {}  # replica -> its traced later chunks (none: dropped, it runs them eagerly)
        if PREFILL_CHUNK_TRACE and not DFLASH and not all(m.chunk_traces_supported() for m in self.models):
            logger.warning(
                "MiMo vLLM: traced later prefill chunks (MIMO_PREFILL_CHUNK_TRACE) off: this ttnn build's "
                "chunked_flash_mla_prefill takes no chunk_start_idx_tensor, so a later-chunk trace could only replay "
                "at its capture's start; later chunks run eagerly"
            )
        elif PREFILL_CHUNK_TRACE and not DFLASH:
            pair = self.pair_split
            lengths = sorted({prefill_padded_len(n, self._chunk_align(), pair) for n in self._chunk_trace_tokens()})
            for lane, model in enumerate(self.models):
                try:
                    chunk_traces[lane] = model.setup_prefill_chunk_traces(lengths, CHUNK_TRACE_SDPA)
                    model.warm_prefill_chunk_traces()
                except Exception as e:  # e.g. out of DRAM for their buffers: serve with eager later chunks
                    model.drop_prefill_chunk_traces(f"replica {lane}: setting up later-chunk traces failed: {e}")
                    chunk_traces[lane] = []
        # One entry when every replica traces the same chunks; per replica when one dropped them.
        traced = list(chunk_traces.values())
        traced = traced[0] if traced and all(t == traced[0] for t in traced) else (chunk_traces or [])
        self.already_warmed_up_prefill = self.prefill_programs_compiled = True
        new = [m.mesh_device.num_program_cache_entries() - p for m, p in zip(self.models, programs)]
        logger.info(
            f"MiMo vLLM: prefill warmed up in {time.time() - t0:.1f}s: {len(chunks)} chunks of up to {max_tokens} "
            f"tokens and {len(batches)} batches through layers {layers}, {new} new programs per replica; "
            f"traced later chunks {traced}"
        )

    def _chunk_trace_tokens(self):
        """Chunk lengths whose later chunks are traced: the step's token budget and the longest chunk."""
        budget = self.prefill_token_budget
        if budget is None and self._kv_cache_spec_source is not None:
            budget = self._kv_cache_spec_source[0].scheduler_config.max_num_batched_tokens
        longest = self._max_prefill_tokens()
        return {min(budget, longest), longest} if budget else {longest}

    def warmup_model_decode(self, kv_cache=None, enable_trace=False, max_batch_size=None, **kwargs):
        """``enable_trace``: capture each replica's decode trace (every slot idle, so no KV is written); otherwise
        compile the eager step. The plugin always runs the eager pass first; with MTP the verify and draft programs
        compile in their own captures, so that pass is the same as without it."""
        for lane, model in enumerate(self.models):
            if enable_trace:
                if model.decode_trace is None:
                    model.capture_decode_trace()
                    logger.info(f"MiMo vLLM: replica {lane} decode trace captured")
                pending = [key for key, t in model.prefill_traces.items() if t["id"] is None]
                for k, (users, padded) in enumerate(pending):
                    logger.debug(f"MiMo vLLM: replica {lane} capturing prefill trace {k + 1}/{len(pending)}")
                    model.capture_prefill_batch_trace(users, padded)
                if pending:
                    logger.info(f"MiMo vLLM: replica {lane} prefill traces captured for (users, length) {pending}")
                if (model.mtp is not None or model.dflash is not None) and model.verify_trace is None:
                    model.capture_spec_traces()
                    logger.info(f"MiMo vLLM: replica {lane} verify and draft traces captured")
                # Last: they are optional, so every trace the replica needs is in the region before them; one that
                # does not fit drops them all (capture_prefill_chunk_traces warns) and later chunks run eagerly.
                if any(t["id"] is None for t in model.chunk_traces.values()):
                    if model.capture_prefill_chunk_traces():
                        logger.info(f"MiMo vLLM: replica {lane} later-chunk prefill traces {list(model.chunk_traces)}")
            else:
                idle = torch.full((model.max_batch,), -1, dtype=torch.int32)
                model.decode_read(model.decode_enqueue(torch.zeros(model.max_batch, dtype=torch.int32), idle))

    def release_persistent_capture(self):
        for model in self.models:
            model.release_decode_trace()
            model.release_prefill_traces()
            model.release_prefill_chunk_traces()
            model.release_spec_traces()
