# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Chat session over the single-trace HEAD/TAIL chain of the B=1 timing runner.

The served decode path is the runner's chain unchanged: 4 HEAD + 4 TAIL traces
of the position-generic body, replayed as ``HEAD[t mod 4]``, host PLE row
refresh, ``TAIL[t mod 4]``.  Prefill has two modes.  ``teacher_forced`` writes
each prompt token into the persistent token row before its HEAD, so the
device's own greedy resolve (the last op of every TAIL) is overwritten before
the embedding reads the row.  ``chunked`` (the default when the chain captured
the chunk trace) runs all prompt tokens but the last through the 32-row chunk
trace with ``Qwen38ChunkPrefill``: teacher-forced steps up to the next multiple
of 32, the chunk carry seeded from the decode buffers, full chunks, a padded
tail chunk, the eager hand-off (``finish_prefill``), then the last prompt token
is teacher-forced as the first decode replay.  Suffixes shorter than
``CHUNK_PREFILL_MIN_ROWS`` after the alignment steps are teacher-forced.
Generation is the runner's evented loop (non-blocking row read, event, HEAD,
event wait, PLE refresh, TAIL).  Multi-turn requests reuse the device state when
the new prompt extends the committed input sequence (the suffix is prefilled
from the committed position); otherwise the generic state is reset in place.

The prompt-end snapshot (``REPLY-TAIL-SPLICE-DESIGN-20260906.md``): before the last prompt token's forced step
the chain copies the generic state's recurrent buffers (GDN recurrent states and ring slots, PLE slots, QSA staging
and raw-key rings; the caches are positional) into a resident snapshot and the session records the ids consumed so
far.  A later request whose render extends those ids but not the committed ones (a thinking conversation: the
template renders the past turn's think block as ``<think>\n\n</think>``, one token off the device's ``<think>\n``)
restores the snapshot and prefills only the rendered tail instead of resetting.

Sampling is an optional tail of the same chain (``tools/qwen38_sampling_step.py``):
TAIL's epilogue also writes a candidate row, and a request with ``temperature > 0``
runs the sampled loop (read the row after TAIL, sample on the host, write the token
into the row before HEAD).  Greedy requests take the loop above untouched.

MTP drafting is opt-in (``mtp=K``, K in 3, 4 or 5; ``ttnn/mtp_v2.py``): the chain also
holds the verify / draft / commit traces and, in every TAIL and in the chunk body, the
MTP layer's rows, so the MTP layer follows the target through prefill.  A greedy
request on such a chain runs its prefill as above, reads the first token, switches
the device state into verify mode (``mtp_enter``) and generates with the pass loop:
draft replay, readback, PLE rows, commit, verify replay, readback, ``a + 1`` tokens
per pass streamed as they commit.  At the end of the request (or before a forced
token) the last pass's rows are committed as far as the request consumed them and
the 1-row buffers are rebuilt (``mtp_leave``), so 1-row, sampled and MTP requests
alternate on one server.  Sampled requests and ``prefill_mode`` ``teacher_forced``
requests take the loops above, unless the chain captured the split verify beside
the fused one (``mtp_sampled``: an ``--mtp --sampling`` server's default, off with
the server's ``QWEN38_MTP_SAMPLED=0``): then a sampled request the pass loop can
bound (``sampling_step.drafting_admission``) drafts too, through the split form,
the host deciding every pass by exact speculative sampling
(``ttnn/speculative_sampling.py``) on the rows' candidate distributions, while a
greedy request keeps the fused verify (the pinned greedy stream: the traces a
``QWEN38_MTP_SAMPLED=0`` server runs; ``mtp_enter`` routes by ``decide``).  With
``QWEN38_MTP_DEVICE_ACCEPT=1`` beside it (default off) a third form is captured,
the head, the device's point-mass acceptance (``fused.mtp_accept``) and the tail
in one trace, for the sampled requests the device acceptance admits
(``mtp_enter``'s ``before_verify_sampled``, the pass's uniforms; the acceptance
is ``tools/qwen38_mtp_device_accept.py``'s).

``Qwen38ChatSession`` speaks to the device only through a chain object with the
per-step primitives; ``Qwen38TracedChain`` is the hardware one, the no-device
test drives the session with a scripted chain.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
from ttnn.tools.trace_allocation_tracker import TRACE_ALLOC_TRACKING, TraceAllocationTracker, acknowledge_corruptible

import ttnn
from models.demos.blackhole.qwen38_flash_next import vision_splice
from models.demos.blackhole.qwen38_flash_next.chat import (
    EOS_TOKEN_IDS,
    IM_START_ID,
    REASONING_EFFORTS,
    TOKENIZER_SIZE,
    VOCAB_SIZE,
    Qwen38ChatFormatError,
)
from models.demos.blackhole.qwen38_flash_next.config import LAYER_PATTERN
from models.demos.blackhole.qwen38_flash_next.tools import hardware_profiles, physical_route
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_protocol as protocol
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_sampling_step as sampling_step
from models.demos.blackhole.qwen38_flash_next.tools import resident_decode
from models.demos.blackhole.qwen38_flash_next.tools.evidence_records import Marker, utc_now
from models.demos.blackhole.qwen38_flash_next.tools.hardware_profiles import ResidentHardwareProfile
from models.demos.blackhole.qwen38_flash_next.tools.live_decode_diagnostic import (
    Qwen38LiveDecodeConstruction,
    construct_live_decode_diagnostic,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_prefill_driver import (
    Qwen38ChunkPrefill,
    Qwen38PrefillResult,
    alignment_steps,
)
from models.demos.blackhole.qwen38_flash_next.ttnn import fused as fused_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn.bf4 import (
    BF4_TILE_BYTES,
    BLACKHOLE_RING_SIZES,
    packed_bf4_bytes_per_device,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import (
    RESIDENT_CONTEXT_HEADROOM,
    RESIDENT_MAX_QSA_CACHE_CAPACITY,
    RESIDENT_QSA_CACHE_CAPACITIES,
    Qwen38ResidentContext,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    CHUNK_ROWS,
    FABRIC_CONFIG,
    LONG_CHUNK_ROWS,
    MESH_SHAPE,
    TP_SIZE,
    Qwen38MeshContract,
    is_slab_rows,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import ZERO_EMBEDDING_TOKEN
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_rows_scan as gdn_rows_scan_module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import mtp_accept as mtp_accept_module
from models.demos.blackhole.qwen38_flash_next.ttnn.model import GENERIC_HEAD_LAYERS, Qwen38TTNNGenericTraceKey

# The device position P must stay below allocated_context (RoPE table lookup, KV write); the consumed EOS step and a
# little slack are the headroom.  CONTEXT_LIMIT is the default (32k) build's; a session derives its own from the
# chain's allocated context.
CONTEXT_HEADROOM = RESIDENT_CONTEXT_HEADROOM
CONTEXT_LIMIT = Qwen38ResidentContext().context_limit
PREFILL_EVENT_INTERVAL = 16
# A request's completion budget is bounded by the remaining context, context_limit - prompt tokens, and defaults to
# it (require_budget).  MAX_TOKENS_BOUND only types an explicit max_tokens before the prompt is known: the largest
# admitted context.
MAX_TOKENS_BOUND = max(RESIDENT_QSA_CACHE_CAPACITIES)
PREFILL_MODES = ("chunked", "teacher_forced")
DEFAULT_PREFILL_MODE = "chunked"
# Rows left for the chunk trace after the alignment steps below which the chunk path costs more than teacher forcing
# them at about 50 ms per token: the serving hand-off measured 240-340 ms (4x p150b, 2026-09-03) on top of the seed and
# the padded chunk replay, so the break-even is about 10-12 rows.
CHUNK_PREFILL_MIN_ROWS = 16
# The chunk warm pass embeds one token per vocabulary owner in every lane group (the decode warm pass's ids).
WARM_CHUNK_TOKEN_IDS = tuple(
    resident_decode.SEQUENTIAL_TRACE_WARM_EMBEDDING_TOKEN_IDS[index % TP_SIZE] for index in range(CHUNK_ROWS)
)
WARM_LONG_CHUNK_TOKEN_IDS = tuple(
    resident_decode.SEQUENTIAL_TRACE_WARM_EMBEDDING_TOKEN_IDS[index % TP_SIZE] for index in range(LONG_CHUNK_ROWS)
)
# The CPU acceptance study's template flags (mtp_acceptance_cpu_v2 at a97cb9e6b0): the 12 prompt records render
# identically only with these.  The records carry their own system turn inside their prompt ids; the session adds no
# system prompt of its own.
ENABLE_THINKING = False
PRESERVE_THINKING = True
REASONING_EFFORT = "low"
RESIDUE_CLASSES = resident_decode.SINGLE_TRACE_RESIDUE_CLASS_TRACES
SEED_TOKEN_ID = IM_START_ID
THINK_END_ID = protocol.THINK_END_ID
# MTP drafting: the draft counts a server may be opened with (the chain timing tool's measured arms; k = 3 and 4
# run the 5-row verify MoE form, k = 5 the 32-row form, and the QSA verify path admits no more than k + 1 = 6 rows),
# the GDN state re-anchor settings (off = every GDN layer commits through the chunk kernel; layer0 = layer 0's
# commits run the 1-row fp32 step recurrence over the committed rows), the bootstrap pass's placeholder drafts, the
# resident expert pairs with the MTP layer's.
MTP_DRAFTS = (3, 4, 5)
MTP_GDN_ANCHORS = ("off", "layer0")
MTP_GDN_ANCHOR_LAYERS = {"off": (), "layer0": (0,)}
MTP_BOOTSTRAP_DRAFT_TOKEN = 0
MTP_EXPECTED_CACHE_LOADS = resident_decode.EXPECTED_CACHE_LOADS + 1
# The verify forms a chain captures per switch pair (mtp_sampled, mtp_device_accept) (Qwen38TracedChain.open): the
# fused verify alone; the fused verify and the split head + tail beside it (QWEN38_MTP_SAMPLED); with the device
# acceptance (QWEN38_MTP_DEVICE_ACCEPT, default off) also the device-decided sampled form, the head, the accept program
# and the tail in one trace.  The admission's traces term counts them; the server's pre-mesh refusal and the open's
# gate both read this one table, so the two records cannot drift.
MTP_VERIFY_FORMS_BY_SWITCH = {
    (False, False): ("fused",),
    (True, False): ("fused", "split"),
    (True, True): ("fused", "split", "sampled"),
}
# The device-decided form's warm: k + 1 uniforms (multiples of 2**-24, exact in fp32), distinct per row, under the
# sampling extension's warm policy; the eager pass's statistics row must equal the host reference bitwise.
MTP_WARM_ACCEPT_UNIFORMS = tuple(
    n * 2**-24 for n in (1_000_003, 5_000_011, 9_000_017, 13_000_019, 16_000_023, 7_000_027)
)


def mtp_verify_forms(mtp_sampled: bool, mtp_device_accept: bool = False) -> tuple[str, ...]:
    """The verify forms an MTP chain captures with the switches (``MTP_VERIFY_FORMS_BY_SWITCH``); the device
    acceptance needs the split verify."""

    if type(mtp_sampled) is not bool or type(mtp_device_accept) is not bool:
        raise ValueError(
            f"mtp_sampled and mtp_device_accept must be a bool each, got {mtp_sampled!r}, {mtp_device_accept!r}"
        )
    if mtp_device_accept and not mtp_sampled:
        raise ValueError("the device acceptance needs the split verify (mtp_sampled)")
    return MTP_VERIFY_FORMS_BY_SWITCH[(mtp_sampled, mtp_device_accept)]


# The MTP chain's DRAM growth per bank beyond the resident build, by the part the open measures it in.  `components`
# is the MTP layer's weights: its 49th BF4 pair (interleaved over the banks) and its non-expert weights beyond the pair
# (k-independent).  `states` is the layer's QSA state at the allocated context, then the verify / draft / split states,
# the step inputs and the chunk extension, sized by the verify MoE form ``moe_rows_for(k + 1)`` (5 rows for k = 3 and
# 4, 32 rows for k >= 5).  `traces` is the captured MTP traces, sized by the number of verify forms captured: one (the
# fused verify or the split head + tail, with the draft and the commit: three or four traces), each further form adding
# its verify trace and its draft trace (two forms: six traces; up to three: the fused greedy verify, a fused sampled
# verify and the split head + tail).  Measured on the 4x p150
# line (ring 8, allocated context 32,768) 2026-09-25: k = 4 with both verify forms {'components': 50468032, 'states':
# 12938112, 'traces': 11107904}; k = 5 with the split verify alone {'components': 50468032, 'states': 20920192,
# 'traces': 6390144}; the ring-8 pair per bank is 44,236,800 and the QSA state per bank 4,462,592 at 32,768.  The
# admission's estimate (its required side) is the exact pair and QSA state plus these measured remainders, each with
# the margin (the 24 MiB flat bound before it admitted k = 4 with one form only: 73,865,216 against the 74,514,048
# and 77,778,368 measured); the open checks the measured growth against the estimate.  Its free side decides on the
# live allocator (mtp_capacity_admission with ``live``: the mesh's DRAM view read after the resident weights are
# built, the chain's open), from which the resident build's own remaining allocations are taken: the context state
# (Qwen38ResidentContext.context_state_bytes_per_device) and RESIDENT_POST_BUILD_BYTES_PER_BANK_UPPER_BOUND, the
# context-independent rest (the GDN states, the decode and chunk traces, the sampling chain: 47.5 MB per bank at
# 32,768 and 49.9 MB at 262,144, measured 2026-09-25 on the compact expert layout with bf8 dense weights).  Without a
# device (the no-device tests) the table RESIDENT_FREE_BYTES_PER_BANK_AFTER_CAPTURES stands in: the free bytes per
# bank a resident build left after its captures without MTP, measured 2026-09-04 (head 4149197252, 8 banks of
# 4,272,341,376 bytes; free and largest contiguous) on the layout of that date -- today's build leaves about 1.24 GB
# more per bank, so the table refuses 262,144 where the live read admits it.  A resident BF4 payload is interleaved
# over the banks one 576-byte tile page at a time, and the resident loader needs 128 MB contiguous per bank.
# components 50,468,032 - the pair 44,236,800; states 12,938,112 (5 rows) / 20,920,192 (32 rows) - the QSA state
# 4,462,592; traces 6,390,144 with one verify form and 11,107,904 with two, so 4,717,760 per further form (its verify
# trace and its draft trace; the third form's figure is that line continued, not a measurement).
MTP_COMPONENTS_BEYOND_PAIR_BYTES_PER_BANK = 6_231_232
MTP_STATES_BEYOND_QSA_STATE_BYTES_PER_BANK_BY_MOE_ROWS = {
    5: 8_475_520,
    # the 6-row verify MoE form: the 5-row term plus the measured growth of a 6-row open over a 5-row one at k = 5
    # (+744,064 B per bank after the captures; +252,928 at k = 4, measured 2026-09-26 on the 4x p150 line at 32,768)
    6: 8_475_520 + 744_064,
    32: 16_457_600,
}
MTP_STATES_ESTIMATE_PROVISIONAL_MOE_ROWS = frozenset()  # every row count's term is measured (6 re-seeded 2026-09-26)
MTP_TRACES_BYTES_PER_BANK_ONE_VERIFY_FORM = 6_390_144
MTP_TRACES_BYTES_PER_BANK_PER_ADDITIONAL_VERIFY_FORM = 4_717_760
MTP_VERIFY_FORMS_MAX = 3
MTP_GROWTH_ESTIMATE_MARGIN_PERCENT = 10
RESIDENT_POST_BUILD_BYTES_PER_BANK_UPPER_BOUND = 64 << 20
# A --long-chunks build's 128-row chunk state and trace beyond the build the table and the post-build bound describe
# (measured 2026-09-25 on the QuietBox at 32,768 with the compact expert layout: 1,603,483,392 bytes free per bank
# without, 1,569,890,816 with --long-chunks): taken off the free side when the admission is for a --long-chunks chain
# (the table's build had none; the live after-build read precedes their allocation).  And the MTP layer's own 128-row
# chunk extension (its QSA chunk state, its rows-128 MoE instance without a combine buffer of its own, a [1,1,4,32]
# token row) beyond the states remainder, a remainder of the states term with the same margin, measured 2026-09-26 on
# the 4-chip p150 line at 32,768 with k = 4 and two verify forms as the states term of the --mtp --long-chunks chain's
# DRAM growth record less the plain --mtp chain's (15,339,392 - 15,313,792 bytes per bank; components and traces
# identical, 50,468,032 and 7,118,400 in both records).
LONG_CHUNKS_BYTES_PER_BANK_AFTER_CAPTURES = 1_603_483_392 - 1_569_890_816
MTP_LONG_CHUNK_EXTENSION_BYTES_PER_BANK = 15_339_392 - 15_313_792
# The 128-row twin's slab form (2026-09-26): per 128-row slice one FP32 TILE token tile ``[1,1,4,32]`` (one 32 x 32 tile
# page of 4,096 bytes) and, after the first slice, one UINT32 ``[1,1,1,1]`` position offset (one 32-byte DRAM page).
MTP_SLAB_TOKEN_TILE_BYTES = 32 * 32 * 4
MTP_SLAB_OFFSET_PAGE_BYTES = 32


def mtp_slab_form_bytes_per_bank(slab_rows: int) -> int:
    """The slab form's DRAM per bank as an upper bound: every page on one bank (the pages are interleaved)."""

    if not is_slab_rows(slab_rows):
        raise ValueError(f"slab_rows must be a slab row count, got {slab_rows!r}")
    slices = slab_rows // LONG_CHUNK_ROWS
    return slices * MTP_SLAB_TOKEN_TILE_BYTES + (slices - 1) * MTP_SLAB_OFFSET_PAGE_BYTES


RESIDENT_DRAM_BANKS = 8
RESIDENT_MIN_CONTIGUOUS_BYTES_PER_BANK = 128 << 20
RESIDENT_FREE_BYTES_PER_BANK_AFTER_CAPTURES = {
    32768: (478_238_784, 477_584_640),
    65536: (423_336_000, 422_681_856),
    131072: (313_694_272, 313_040_128),
    262144: (94_378_048, 93_723_904),
}
# The verify-rows fold (``gdn_rows_scan``, the default; ``QWEN38_FUSED_OFF`` restores the wrap) keeps every GDN layer's k + 1
# prefix states in DRAM for the pass: ``[k + 1, 12, 128, 128]`` fp32 per layer = (k + 1) x 786,432 bytes per layer per
# device (3,932,160 at k = 4; 141,557,760 over the 36 GDN layers), interleaved one 4 KiB tile page at a time over the
# banks, so the states term grows by GDN layers x ceil((k + 1) x 192 pages / banks) x 4,096 bytes per bank (17,694,720 at
# k = 4).  Measured on the 4x p150 line 2026-09-26 (k = 4, two verify forms, 32,768, head e31a751e483, run
# q38-chat-server-p150-line-20260926T130417Z-3556384): {'components': 50468032, 'states': 31093824, 'traces': 5832256}
# = 87,394,112 bytes per bank, refused by the estimate without this term (77,095,515, of which states 13,785,664:
# +17,308,160 measured over the estimate; the 2026-09-25 record without the fold measured states 12,938,112, so the
# growth beyond the prefix states themselves is 461,000 bytes per bank, inside the margin).  The traces fell 11,107,904
# -> 5,832,256 (one program per GDN layer where the wrap runs six, one pick where the commit re-ran the two prims); the
# estimate keeps the wrap's traces figure as an over-estimate.  The fold attaches only where its admission holds (rows
# k + 1 <= gdn_rows_scan.MAX_ROWS), so beyond that the term is zero.
GDN_LAYERS = LAYER_PATTERN.count("linear_attention")
GDN_STATE_TILE_PAGES = gdn_rows_scan_module.HEADS * (gdn_rows_scan_module.HEAD_DIM // gdn_rows_scan_module.TILE) ** 2
GDN_STATE_PAGE_BYTES = gdn_rows_scan_module.TILE * gdn_rows_scan_module.TILE * 4  # one fp32 tile


def fold_prefix_states_bytes_per_bank(drafts: int, banks: int = RESIDENT_DRAM_BANKS) -> int:
    """The verify-rows fold's prefix states per bank at ``drafts`` drafts: every GDN layer's ``[drafts + 1, 12, 128, 128]``
    fp32 tensor, its 4 KiB tile pages spread over the ``banks``; 0 where the fold does not attach (rows past its
    admission, :data:`ttnn.fused.gdn_rows_scan.MAX_ROWS`)."""

    rows = drafts + 1
    if rows > gdn_rows_scan_module.MAX_ROWS:
        return 0
    return GDN_LAYERS * -(-(rows * GDN_STATE_TILE_PAGES) // banks) * GDN_STATE_PAGE_BYTES


def resident_post_build_bytes_per_bank(allocated_context: int) -> int:
    """What the resident build allocates per bank after its weights, before the MTP chain can be measured: the
    context-scaled state (the QSA caches and the RoPE tables) and the context-independent rest under
    :data:`RESIDENT_POST_BUILD_BYTES_PER_BANK_UPPER_BOUND`."""

    context_state = Qwen38ResidentContext(allocated_context).context_state_bytes_per_device
    return -(-context_state // RESIDENT_DRAM_BANKS) + RESIDENT_POST_BUILD_BYTES_PER_BANK_UPPER_BOUND


def mtp_capacity_admission(
    allocated_context: int,
    *,
    drafts: int = mtp_v2.DEFAULT_DRAFTS,
    verify_forms: int = 1,
    ring_size: int = max(BLACKHOLE_RING_SIZES),
    live: Mapping[str, Any] | None = None,
    live_point: str = "after_build",
    long_chunks: bool = False,
    moe_rows: int | None = None,
    components_shared: bool = False,
    gdn_rows_scan: bool = False,
    slab_rows: int | None = None,
    vision_resident_bytes_per_bank: int = 0,
    vision_peak_activation_bytes_per_bank: int = 0,
) -> dict[str, Any]:
    """Whether the MTP chain fits beside the resident build at ``allocated_context`` with ``drafts`` drafts per pass
    and ``verify_forms`` captured verify forms (1..:data:`MTP_VERIFY_FORMS_MAX`).  The required side is the estimate
    of the growth the open measures per bank: the components with the 49th BF4 pair (its two payloads interleaved
    over the banks and needing their contiguous room), the states of the verify MoE form ``moe_rows_for(drafts + 1)``
    with the MTP layer's QSA state at that context, the traces of one form plus one verify and one draft trace per
    further form, :data:`MTP_GROWTH_ESTIMATE_MARGIN_PERCENT` over every remainder.  The free side is the free bytes
    per bank the build leaves after its captures, live or from the table.
    ``drafts`` is any count whose verify MoE form exists (rows up to 32); which counts the verify path runs is
    ``mtp_v2.SUPPORTED_DRAFTS``, which a server may open with :data:`MTP_DRAFTS`.

    ``live`` is the mesh allocator's DRAM view (``free_bytes_per_bank``, ``largest_contiguous_bytes_free_per_bank``:
    the record ``hardware_profiles.symmetric_mesh_dram_memory`` returns) read at ``live_point``: ``after_build``
    (the chain's open, once the resident weights are built and before the MTP layer's) takes the resident build's
    remaining allocations (:func:`resident_post_build_bytes_per_bank`) off both readings; ``after_captures`` reads
    them as they are.  Without ``live`` the 2026-09-04 table :data:`RESIDENT_FREE_BYTES_PER_BANK_AFTER_CAPTURES`
    stands in (no device: the no-device tests).  Every byte count is in the record, the estimate by part beside the
    measured remainders it adds and ``decided_by`` naming the free side read and the required side's parts, with
    the shortfalls of a refusal; ``fits`` decides.

    ``long_chunks`` (an ``--mtp --long-chunks`` chain) takes the 128-row chunk state and trace
    (:data:`LONG_CHUNKS_BYTES_PER_BANK_AFTER_CAPTURES`) off the free side where they are not yet in the reading (the
    table, the live ``after_build`` read) and adds the MTP layer's 128-row extension
    (:data:`MTP_LONG_CHUNK_EXTENSION_BYTES_PER_BANK`) to the states remainder.

    ``gdn_rows_scan`` (the verify-rows fold serving: the registry's default less ``QWEN38_FUSED_OFF``) adds the fold's
    persistent prefix states to the states remainder, derived from ``drafts`` and the GDN layer count
    (:func:`fold_prefix_states_bytes_per_bank`),
    with the same margin; the traces remainder stays the wrap's (an over-estimate under the fold).  The term is per
    chain: a second drafting chain (``components_shared``) allocates its own GDN rows states, so its admission
    charges its own ``drafts + 1`` rows.

    ``vision_resident_bytes_per_bank`` (a server with the vision tower resident: the tower's BF16 weights, replicated
    per die, loaded in the chain's warm hook after the resident build and before any capture; the tower's
    ``resident_bytes_per_bank()``, 125.2 MB measured on the line at 2026-09-29) comes off the free side where the
    reading predates the load (the table, the live ``after_build`` read), not off an ``after_captures`` read that
    holds it.  ``vision_peak_activation_bytes_per_bank`` (the tower's transient DRAM per bank for the largest image
    the server admits, ``peak_activation_bytes_per_bank(patches)``) is a required-side part with the growth margin:
    an image request allocates it after every resident allocation, so it must stay free beside the MTP chain."""

    if isinstance(drafts, bool) or type(drafts) is not int or not 1 <= drafts < CHUNK_ROWS:
        raise ValueError(f"MTP drafts must be an int in [1, {CHUNK_ROWS - 1}], got {drafts!r}")
    if isinstance(verify_forms, bool) or type(verify_forms) is not int or not 1 <= verify_forms <= MTP_VERIFY_FORMS_MAX:
        raise ValueError(f"verify forms must be an int in [1, {MTP_VERIFY_FORMS_MAX}], got {verify_forms!r}")
    moe_rows = mtp_v2.resolve_moe_rows(drafts + 1, moe_rows)  # the switch's row count keys the states term too
    if moe_rows not in MTP_STATES_BEYOND_QSA_STATE_BYTES_PER_BANK_BY_MOE_ROWS:
        raise ValueError(f"no MTP states estimate for a {moe_rows}-row verify MoE form")
    context = Qwen38ResidentContext(allocated_context).allocated_context
    if live_point not in ("after_build", "after_captures"):
        raise ValueError(f"live_point must be after_build or after_captures, got {live_point!r}")
    if type(long_chunks) is not bool:
        raise ValueError(f"long_chunks must be a bool, got {long_chunks!r}")
    if type(gdn_rows_scan) is not bool:
        raise ValueError(f"gdn_rows_scan must be a bool, got {gdn_rows_scan!r}")
    if slab_rows is not None and (not is_slab_rows(slab_rows) or not long_chunks):
        raise ValueError(f"slab_rows takes a slab row count with long_chunks, got {slab_rows!r}")
    for name, value in (
        ("vision_resident_bytes_per_bank", vision_resident_bytes_per_bank),
        ("vision_peak_activation_bytes_per_bank", vision_peak_activation_bytes_per_bank),
    ):
        if isinstance(value, bool) or type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative int, got {value!r}")
    long_chunks_bytes = LONG_CHUNKS_BYTES_PER_BANK_AFTER_CAPTURES if long_chunks else 0
    # The tower loads in the warm hook: after the build's read, before the captures (so in an after-captures read).
    pre_capture_bytes = long_chunks_bytes + vision_resident_bytes_per_bank
    if live is None:
        # A context below the smallest measured one (8,192: the batched lanes' small context) is admitted against the
        # smallest measured context's readings: its resident build leaves more room, so the admission is conservative.
        measured_at = [c for c in sorted(RESIDENT_FREE_BYTES_PER_BANK_AFTER_CAPTURES) if c >= context]
        if not measured_at:
            raise ValueError(f"no free-bytes-after-captures measurement at or above {context} tokens")
        free, largest = RESIDENT_FREE_BYTES_PER_BANK_AFTER_CAPTURES[measured_at[0]]
        # The table's build had no 128-row chunk state or trace (nor a tower): they come off its free bytes and, as
        # an upper bound on what they take from the largest contiguous block, off that block too.
        free, largest = free - pre_capture_bytes, max(largest - pre_capture_bytes, 0)
        source: dict[str, Any] = {
            "free_bytes_source": "table_2026-09-04",
            "free_bytes_measured_at_context": measured_at[0],
        }
    else:
        banks = int(live.get("num_banks", RESIDENT_DRAM_BANKS))
        if banks != RESIDENT_DRAM_BANKS:
            raise ValueError(
                f"the live DRAM view has {banks} banks, the admission is written for {RESIDENT_DRAM_BANKS}"
            )
        live_free = int(live["free_bytes_per_bank"])
        live_largest = int(live["largest_contiguous_bytes_free_per_bank"])
        if live_free < 0 or live_largest < 0 or live_largest > live_free:
            raise ValueError(f"inconsistent live DRAM view: free {live_free}, largest contiguous {live_largest}")
        # After the build the 128-row chunk state and trace are still to come (they are allocated with the chunk
        # states and captured after the chunk trace): with the resident's own remaining allocations, off both readings.
        remaining = (
            resident_post_build_bytes_per_bank(context) + pre_capture_bytes if live_point == "after_build" else 0
        )
        free, largest = live_free - remaining, max(live_largest - remaining, 0)
        source = {
            "free_bytes_source": f"measured_{live_point}",
            "free_bytes_measured_at_context": context,
            "live_free_bytes_per_bank": live_free,
            "live_largest_contiguous_bytes_free_per_bank": live_largest,
            "resident_post_build_bytes_per_bank": remaining,
        }
    w01_bytes, w2_bytes = packed_bf4_bytes_per_device(ring_size=ring_size)
    w01_per_bank = -(-(w01_bytes // BF4_TILE_BYTES) // RESIDENT_DRAM_BANKS) * BF4_TILE_BYTES
    w2_per_bank = -(-(w2_bytes // BF4_TILE_BYTES) // RESIDENT_DRAM_BANKS) * BF4_TILE_BYTES
    qsa_state = -(-Qwen38ResidentContext(allocated_context).qsa_generic_state_bytes // RESIDENT_DRAM_BANKS)
    # ``slab_rows`` (a ``--prefill-slab`` chain under ``--mtp``): the 128-row twin's slab form adds one 128-lane FP32
    # token tile (one 32x32 tile page) per 128-row slice and one UINT32 position-offset page per slice after the first,
    # charged whole to one bank (an upper bound: the pages are interleaved over the banks).
    slab_form = mtp_slab_form_bytes_per_bank(slab_rows) if slab_rows is not None else 0
    extension = (MTP_LONG_CHUNK_EXTENSION_BYTES_PER_BANK if long_chunks else 0) + slab_form
    prefix_states = fold_prefix_states_bytes_per_bank(drafts) if gdn_rows_scan else 0
    remainders = {
        "components_beyond_pair": MTP_COMPONENTS_BEYOND_PAIR_BYTES_PER_BANK,
        "states_beyond_qsa_state": MTP_STATES_BEYOND_QSA_STATE_BYTES_PER_BANK_BY_MOE_ROWS[moe_rows],
        "long_chunk_extension": extension,
        "gdn_prefix_states": prefix_states,
        "traces": MTP_TRACES_BYTES_PER_BANK_ONE_VERIFY_FORM
        + (verify_forms - 1) * MTP_TRACES_BYTES_PER_BANK_PER_ADDITIONAL_VERIFY_FORM,
        "traces_per_additional_verify_form": MTP_TRACES_BYTES_PER_BANK_PER_ADDITIONAL_VERIFY_FORM,
    }

    def with_margin(remainder: int) -> int:
        return -(-remainder * (100 + MTP_GROWTH_ESTIMATE_MARGIN_PERCENT) // 100)

    if type(components_shared) is not bool:
        raise ValueError(f"components_shared must be a bool, got {components_shared!r}")
    # a second chain over one model (QWEN38_MTP_DRAFTS_PER_REQUEST) reuses the first's MTP components and its
    # alignment history: its growth is its own states (the verify window, the draft) and its traces
    components = (
        0 if components_shared else w01_per_bank + w2_per_bank + with_margin(remainders["components_beyond_pair"])
    )
    estimate = {
        "components": components,
        "states": qsa_state
        + with_margin(
            remainders["states_beyond_qsa_state"] + remainders["long_chunk_extension"] + remainders["gdn_prefix_states"]
        ),
        "traces": with_margin(remainders["traces"]),
    }
    if vision_peak_activation_bytes_per_bank:
        # A part only on a server with the tower: the records of every other server keep their three parts.
        estimate["vision_activation"] = with_margin(vision_peak_activation_bytes_per_bank)
    required = sum(estimate.values())
    contiguous = (
        RESIDENT_MIN_CONTIGUOUS_BYTES_PER_BANK
        if components_shared
        else max(w01_per_bank, w2_per_bank, RESIDENT_MIN_CONTIGUOUS_BYTES_PER_BANK)
    )
    shortfalls = [
        name
        for name, short in (
            ("free_bytes_below_estimate", free < required),
            ("largest_contiguous_below_pair_room", largest < contiguous),
        )
        if short
    ]
    return {
        "allocated_context": allocated_context,
        "drafts": drafts,
        "mtp_moe_rows": moe_rows,
        # the states term for this row count is an interpolation until a served open measures it (the growth gate
        # compares the measured sum against the estimate, so a wrong term shows as a refusal, never silently)
        "mtp_states_estimate_provisional": moe_rows in MTP_STATES_ESTIMATE_PROVISIONAL_MOE_ROWS,
        "verify_forms": verify_forms,
        "components_shared": components_shared,
        "ring_size": ring_size,
        "long_chunks": long_chunks,
        "gdn_rows_scan": gdn_rows_scan,
        "num_banks": RESIDENT_DRAM_BANKS,
        **source,
        "long_chunks_bytes_per_bank_after_captures": long_chunks_bytes,
        "mtp_long_chunk_extension_bytes_per_bank": extension,
        "mtp_slab_form_bytes_per_bank": slab_form,
        "slab_rows": slab_rows,
        "mtp_gdn_prefix_states_bytes_per_bank": prefix_states,
        "vision_resident_bytes_per_bank": vision_resident_bytes_per_bank,
        "vision_peak_activation_bytes_per_bank": vision_peak_activation_bytes_per_bank,
        "free_bytes_per_bank_after_captures": free,
        "largest_contiguous_bytes_free_per_bank_after_captures": largest,
        "resident_pair_bytes_per_bank": w01_per_bank + w2_per_bank,
        "mtp_qsa_state_bytes_per_bank": qsa_state,
        "mtp_growth_remainders_bytes_per_bank": remainders,
        "mtp_growth_estimate_margin_percent": MTP_GROWTH_ESTIMATE_MARGIN_PERCENT,
        "mtp_growth_estimate_bytes_per_bank": estimate,
        "required_free_bytes_per_bank": required,
        "required_largest_contiguous_bytes_per_bank": contiguous,
        "headroom_bytes_per_bank": free - required,
        # which side decided: the free side as read (live after build / after captures, or the table) against the
        # required side's estimate by part for this k and these forms; the shortfalls name a refusal's reason(s)
        "decided_by": {
            "free_side": source["free_bytes_source"],
            "required_side": f"estimate k={drafts} moe_rows={moe_rows} verify_forms={verify_forms}"
            + (" gdn_rows_scan" if gdn_rows_scan else "")
            + (" vision" if vision_resident_bytes_per_bank or vision_peak_activation_bytes_per_bank else ""),
            "shortfalls": shortfalls,
        },
        "fits": not shortfalls,
    }


# The lanes' DRAM per bank (``--lanes B``), MODELED from the lane families' shapes (ttnn/lanes.py ``_lane_shapes``, the
# MTP-lane extras of ttnn/mtp_lanes.py) with the slope correction of the one measurement past B = 1 (the stage-1
# B = 1 / 2 / 4 lane verify states at 32,768: 71.0 MB per bank per lane against the model's 66.0, +7 %) and the growth
# margin on every modeled term; the required side of ``lanes_capacity_admission``.  MEASURED reference (a 4x p150 hold,
# head 1ce94766f7b1, 2026-09-28): the 4 x 4 lane verify state at 32,768 is 297.3 MB per bank
# (LANES_MEASURED_VERIFY_STATE_BYTES_PER_BANK_4X4_32K), the model's estimate with its margins 27 % above it.
LANES_IMAGE_BYTES_PER_DEVICE_SLOPE = 13_056  # per context row: the 12 KV slabs and the 12 compressed caches
LANES_IMAGE_BYTES_PER_DEVICE_FIXED = 29_684_736  # the recurrent states, ring slots, stagings, rings, PLE slots
LANES_FIXED_BYTES_PER_DEVICE = 60_424_192  # the KV import scratch (13 layers x 4,096 rows) and the GDN lane rows qkv
LANES_MOE_ROWS_SLOPE_BYTES_PER_BANK = 295_632  # the states remainder per verify MoE row (the 5- and 32-row terms' line)
LANES_PER_LANE_SLOPE_CORRECTION_PERCENT = 7  # the measured per-lane increment over the modeled one at 32,768
LANES_TRACES_BYTES_PER_BANK = (51_000_000 + 47_000_000) // 8 + 7_400_000  # the three lane traces, measured 2026-09-20
LANE_KV_STAGINGS = 2  # ttnn/lanes.py KV_STAGINGS: the pagers' whole-lane KV slab pair
# MEASURED per bank on a 4x p150 hold at head 1ce94766f7b1, 2026-09-28 (the allocator's view, 4 x 4 lanes, k = 4, the
# B = 1 chain kept beside them): the lane verify state 297,300,000 at 32,768 and 993,686,336 at 131,072 (a per-lane
# slope of 1,771 B per bank per context row, the model's 1,768), the two pagers 85.0 MB per bank at 131,072 (the
# model's 84.5), the three lane traces 7.3 MB per bank (the 2026-09-20 figure above is the upper bound the estimate
# keeps).  The lanes' net cost at 131,072 was 917,362,880 B per bank with 456,988,928 free after the captures
# (largest contiguous 373,234,752); a served process carries about 100,444,352 B per bank more than the tool's chain
# (the long-chunks state and trace, the slab buffers, the sampled state), leaving about 356 MB per bank at 4 x 128k.
LANES_MEASURED_VERIFY_STATE_BYTES_PER_BANK_4X4_32K = 297_300_000
LANES_MEASURED_VERIFY_STATE_BYTES_PER_BANK_4X4_128K = 993_686_336
LANES_MEASURED_PAGERS_BYTES_PER_BANK_128K = 85_000_000
LANES_MEASURED_TRACES_BYTES_PER_BANK = 7_300_000
LANES_MEASURED_FREE_AFTER_CAPTURES_BYTES_PER_BANK_4X4_128K = 456_988_928
LANES_SERVED_EXTRAS_BYTES_PER_BANK = 100_444_352


def lanes_image_bytes_per_device(allocated_context: int) -> int:
    """One lane's image per device (the seven families at ``allocated_context`` rows)."""

    return LANES_IMAGE_BYTES_PER_DEVICE_SLOPE * int(allocated_context) + LANES_IMAGE_BYTES_PER_DEVICE_FIXED


def lanes_mtp_extra_bytes_per_device(allocated_context: int, rows: int) -> int:
    """Beyond the lane image, per lane and device: the MTP layer's own lane QSA state, the backbone QSA lane verify
    states, the GDN lane rows buffers and constants, the PLE lane rows state, the residual and draft rows."""

    context = int(allocated_context)
    qsa_layers, gdn_layers, heads, head_dim = 12, GDN_LAYERS, gdn_module.VALUE_HEADS_PER_DEVICE, gdn_module.HEAD_DIM
    qkv_width, chunk_rows = gdn_module.QKV_WIDTH_PER_DEVICE, CHUNK_ROWS
    mtp_layer_qsa = Qwen38ResidentContext(context).qsa_generic_state_bytes + 2 * 32 * 128 * 2
    backbone_qsa_verify = qsa_layers * 2 * 32 * 128 * 2
    gdn_rows_per_layer = (
        2 * chunk_rows * qkv_width * 2 + 3 * chunk_rows * heads * head_dim * 2 + 2 * chunk_rows * heads * 4
    )
    gdn_rows = gdn_layers * gdn_rows_per_layer
    gdn_lane_constants = 2 * chunk_rows * chunk_rows * 2 + 4 * chunk_rows * 4
    ple_rows = (9 + rows) * 4 * 640 * 2
    misc = 4 * 640 * 2 + rows * 640 * 2 + 2 * 32 * 128 * 2
    return mtp_layer_qsa + backbone_qsa_verify + gdn_rows + gdn_lane_constants + ple_rows + misc


def lanes_pager_bytes_per_device(allocated_context: int) -> int:
    """The two 1-lane pagers (the generic state's and the alignment layer's): their pack buffers and the whole-lane KV
    staging pair each, from the family shapes."""

    context = int(allocated_context)
    compressed_rows = context // 4 + 32
    backbone = (
        12 * compressed_rows * 128 * 2
        + 36 * gdn_module.VALUE_HEADS_PER_DEVICE * gdn_module.HEAD_DIM * gdn_module.HEAD_DIM * 4
        + 12 * 32 * 512 * 2
        + 12 * 32 * 128 * 2
        + 144 * gdn_module.QKV_WIDTH_PER_DEVICE * 2
        + 9 * 4 * 640 * 2
    )
    alignment = compressed_rows * 128 * 2 + 32 * 512 * 2 + 32 * 128 * 2
    stagings = 2 * LANE_KV_STAGINGS * context * 512 * 2
    return backbone + alignment + stagings


def lanes_fold_prefix_states_bytes_per_bank(total_rows: int, banks: int = RESIDENT_DRAM_BANKS) -> int:
    """The verify-rows fold's prefix states for a lane tile of ``total_rows`` real rows (B x R): ``[total_rows, 12, 128,
    128]`` fp32 per GDN layer, interleaved one 4 KiB tile page at a time over the banks (the B = 1 form's rule,
    :func:`fold_prefix_states_bytes_per_bank`, at the lanes' row count); 70,778,880 B per bank at 20 rows."""

    pages = int(total_rows) * GDN_STATE_TILE_PAGES
    return GDN_LAYERS * -(-pages // banks) * GDN_STATE_PAGE_BYTES


def lanes_capacity_admission(
    allocated_context: int,
    *,
    lanes: int,
    drafts: int,
    live: Mapping[str, Any] | None = None,
    reserved_bytes_per_bank: int = 0,
    gdn_rows_scan: bool = False,
) -> dict[str, Any]:
    """Whether ``lanes`` MTP lanes at ``drafts`` drafts fit beside the resident build and its single-lane MTP chain
    at ``allocated_context``: the one-tile rule (``B x (k + 1) <= 32``), then the DRAM per bank.

    The required side is MODELED: the lane states (``lanes`` x (the lane image + the MTP-lane extra) x the slope
    correction + the fixed part) over the banks, the verify MoE instances at ``B x R`` rows (the states remainder's
    line), the two pagers, and the three lane traces (measured 2026-09-20), each with the growth margin; the free side
    is the live allocator's reading (``free_bytes_per_bank``, ``largest_contiguous_bytes_free_per_bank``) less
    ``reserved_bytes_per_bank`` (what the chain still allocates after the reading: its traces), or the 2026-09-04 table
    without a device.  ``gdn_rows_scan`` (the verify-rows fold serving the lanes) adds the fold's prefix states at the
    lanes' row count (:func:`lanes_fold_prefix_states_bytes_per_bank`) to the states remainder.  The largest single
    lane tensor, the flat KV ``[1, 1, B x C + scratch, 512]`` of one QSA layer,
    must fit the largest contiguous block.  The open measures the growth (states, traces) and refuses READY when it
    exceeds the required side: the estimate's margins are what the measurement is judged against."""

    if isinstance(lanes, bool) or type(lanes) is not int or not 2 <= lanes <= 8:
        raise ValueError(f"lanes must be an int in [2, 8], got {lanes!r}")
    if isinstance(drafts, bool) or type(drafts) is not int or drafts not in mtp_v2.SUPPORTED_DRAFTS:
        raise ValueError(f"drafts must be one of {mtp_v2.SUPPORTED_DRAFTS}, got {drafts!r}")
    rows = drafts + 1
    total_rows = lanes * rows
    if total_rows > CHUNK_ROWS:
        raise ValueError(f"{lanes} lanes x {rows} rows = {total_rows} exceed the {CHUNK_ROWS}-row tile")
    context = Qwen38ResidentContext(allocated_context).allocated_context
    if (
        isinstance(reserved_bytes_per_bank, bool)
        or type(reserved_bytes_per_bank) is not int
        or reserved_bytes_per_bank < 0
    ):
        raise ValueError(f"reserved_bytes_per_bank must be a non-negative int, got {reserved_bytes_per_bank!r}")
    if type(gdn_rows_scan) is not bool:
        raise ValueError(f"gdn_rows_scan must be a bool, got {gdn_rows_scan!r}")
    if live is None:
        measured_at = [c for c in sorted(RESIDENT_FREE_BYTES_PER_BANK_AFTER_CAPTURES) if c >= context]
        if not measured_at:
            raise ValueError(f"no free-bytes-after-captures measurement at or above {context} tokens")
        free, largest = RESIDENT_FREE_BYTES_PER_BANK_AFTER_CAPTURES[measured_at[0]]
        source: dict[str, Any] = {
            "free_bytes_source": "table_2026-09-04",
            "free_bytes_measured_at_context": measured_at[0],
        }
    else:
        banks = int(live.get("num_banks", RESIDENT_DRAM_BANKS))
        if banks != RESIDENT_DRAM_BANKS:
            raise ValueError(
                f"the live DRAM view has {banks} banks, the admission is written for {RESIDENT_DRAM_BANKS}"
            )
        live_free, live_largest = int(live["free_bytes_per_bank"]), int(live["largest_contiguous_bytes_free_per_bank"])
        if live_free < 0 or live_largest < 0 or live_largest > live_free:
            raise ValueError(f"inconsistent live DRAM view: free {live_free}, largest contiguous {live_largest}")
        free, largest = live_free - reserved_bytes_per_bank, max(live_largest - reserved_bytes_per_bank, 0)
        source = {
            "free_bytes_source": "measured_live",
            "live_free_bytes_per_bank": live_free,
            "live_largest_contiguous_bytes_free_per_bank": live_largest,
            "reserved_bytes_per_bank": reserved_bytes_per_bank,
        }

    def with_margin(remainder: int) -> int:
        return -(-remainder * (100 + MTP_GROWTH_ESTIMATE_MARGIN_PERCENT) // 100)

    per_lane = lanes_image_bytes_per_device(context) + lanes_mtp_extra_bytes_per_device(context, rows)
    per_lane_corrected = -(-per_lane * (100 + LANES_PER_LANE_SLOPE_CORRECTION_PERCENT) // 100)
    lane_states_per_device = lanes * per_lane_corrected + LANES_FIXED_BYTES_PER_DEVICE
    pagers_per_device = lanes_pager_bytes_per_device(context)
    moe_states = (
        MTP_STATES_BEYOND_QSA_STATE_BYTES_PER_BANK_BY_MOE_ROWS[5]
        + (total_rows - 5) * LANES_MOE_ROWS_SLOPE_BYTES_PER_BANK
    )
    remainders = {
        "lane_states": -(-lane_states_per_device // RESIDENT_DRAM_BANKS),
        "pagers": -(-pagers_per_device // RESIDENT_DRAM_BANKS),
        "moe_rows_instances": moe_states,
        "fold_prefix_states": lanes_fold_prefix_states_bytes_per_bank(total_rows) if gdn_rows_scan else 0,
        "traces": LANES_TRACES_BYTES_PER_BANK,
    }
    estimate = {
        "states": with_margin(
            remainders["lane_states"]
            + remainders["pagers"]
            + remainders["moe_rows_instances"]
            + remainders["fold_prefix_states"]
        ),
        "traces": with_margin(remainders["traces"]),
    }
    required = sum(estimate.values())
    # the largest single lane tensor: one QSA layer's flat KV over the banks (page-interleaved)
    flat_kv_per_bank = -(-(lanes * context + mtp_lanes_scratch_rows(context)) * 512 * 2 // RESIDENT_DRAM_BANKS)
    contiguous = max(flat_kv_per_bank, RESIDENT_MIN_CONTIGUOUS_BYTES_PER_BANK)
    shortfalls = [
        name
        for name, short in (
            ("free_bytes_below_estimate", free < required),
            ("largest_contiguous_below_lane_kv", largest < contiguous),
        )
        if short
    ]
    return {
        "allocated_context": context,
        "lanes": lanes,
        "drafts": drafts,
        "rows": rows,
        "total_rows": total_rows,
        "one_tile": total_rows <= CHUNK_ROWS,
        "gdn_rows_scan": gdn_rows_scan,
        "num_banks": RESIDENT_DRAM_BANKS,
        **source,
        "free_bytes_per_bank": free,
        "largest_contiguous_bytes_free_per_bank": largest,
        "per_lane_bytes_per_device": per_lane,
        "per_lane_slope_correction_percent": LANES_PER_LANE_SLOPE_CORRECTION_PERCENT,
        "lanes_growth_remainders_bytes_per_bank": remainders,
        "lanes_growth_estimate_margin_percent": MTP_GROWTH_ESTIMATE_MARGIN_PERCENT,
        "lanes_growth_estimate_bytes_per_bank": estimate,
        "required_free_bytes_per_bank": required,
        "required_largest_contiguous_bytes_per_bank": contiguous,
        "headroom_bytes_per_bank": free - required,
        "decided_by": {
            "free_side": source["free_bytes_source"],
            "required_side": f"estimate lanes={lanes} k={drafts} rows={total_rows} (modeled, +{LANES_PER_LANE_SLOPE_CORRECTION_PERCENT} % slope, {MTP_GROWTH_ESTIMATE_MARGIN_PERCENT} % margin)"
            + (" gdn_rows_scan" if gdn_rows_scan else ""),
            "shortfalls": shortfalls,
        },
        "fits": not shortfalls,
    }


def mtp_lanes_scratch_rows(allocated_context: int) -> int:
    """The lane KV cache's scratch rows past the last lane: one import chunk (ttnn/mtp_lanes.py ``kv_scratch_rows``)."""

    return min(4096, int(allocated_context))


class Qwen38ChatRequestError(ValueError):
    """A request the session refuses (HTTP 400): bad messages, context length, bad token budget."""


def mtp_tail_epilogue(model, lm_head, chain_mtp, output, token_row_io, sampling=None):
    """The MTP chain's TAIL epilogue, one function for the warm pass (eager) and the TAIL capture (recorded), so the
    capture asks only for programs the warm compiled: the greedy candidates (with a sampling chain, the form its
    extension uses: the folded candidate row when ``candidate_row`` is on), the device resolve into a fresh row (the
    MTP row reads it before the copy), the MTP layer's row on the retained residual / RoPE rows / QSA position inputs,
    the copy into the persistent token row, and with a sampling chain its candidate row from those candidates (the
    2026-09-25 miss: the row built without them took the chain's form, whose typecasts no warm had compiled).
    Returns ``(candidates, token_row, row)`` (``row`` None without a sampling chain); the caller checks or releases."""

    if output.logits is None or output.residual is None:
        raise Qwen38ChatChainError("the MTP tail epilogue needs the step's logits and its retained MTP inputs")
    if sampling is not None and sampling.candidate_row:
        candidates = lm_head.greedy_candidates(output.logits, candidate_row=sampling.constants)
    else:
        candidates = lm_head.greedy_candidates(output.logits)
    token_row = lm_head.resolve_greedy_on_device(candidates)
    mtp_v2.forward_mtp_step_row(
        model,
        chain_mtp.alignment,
        chain_mtp.step_inputs,
        output.residual,
        token_row,
        rope=output.rope,
        qsa_position=output.qsa_position,
    )
    ttnn.copy(token_row, token_row_io)
    row = None
    if sampling is not None:
        row = lm_head.sampling_candidates(output.logits, sampling.constants, candidates=candidates)
    return candidates, token_row, row


class Qwen38ChatChainError(RuntimeError):
    """The device chain disagrees with its host mirror; the model owner cannot be trusted afterwards."""


@dataclass(frozen=True)
class Qwen38ChatCompletion:
    token_ids: list[int]
    finish_reason: str
    prompt_tokens: int
    prefix_reused: int
    reset: bool
    prefill_tokens: int
    prefill_seconds: float
    ttft_seconds: float
    decode_seconds: float
    tokens_per_second: float | None
    position: int
    # The prefill path this request took ("chunked" or "teacher_forced") and its shape: teacher-forced steps
    # (alignment steps plus the last prompt token, or every token), chunk replays, real rows of the padded tail,
    # the eager hand-off time and the wall per prompt token.
    prefill_mode: str = "teacher_forced"
    prefill_forced_tokens: int = 0
    prefill_chunks: int = 0  # the 32-row chunk replays (full and the padded tail)
    prefill_long_chunks: int = 0  # the 128-row chunk replays ahead of them (--long-chunks / a slab's remainder)
    prefill_slabs: int = 0  # the slab replays ahead of those (--prefill-slab)
    prefill_tail_rows: int = 0
    prefill_handoff_ms: float = 0.0
    # The slabs' host work, summed over the request's slabs: the input preparation (the n-gram lookup and the row
    # packing), the input copies' enqueue, the wait at the slab event syncs.
    prefill_slab_prepare_ms: float = 0.0
    prefill_slab_upload_ms: float = 0.0
    prefill_slab_wait_ms: float = 0.0
    # MTP drafting when the request generated through the pass loop: {k, anchor, sampled, passes, accepted_drafts,
    # tokens_per_pass} and, on the split verify, accept_checks and the sampled_* counters, every field the counts this
    # request added (the chain's cumulative counters are /health.mtp); None for the 1-row loops.
    mtp: dict[str, Any] | None = None
    # The prompt-end snapshot was restored: prefix_reused is the snapshot's length, the rest of the prompt the tail.
    restored: bool = False
    # The schedule of the snapshot this request restored (SNAPSHOT_SCHEDULES; None when it did not restore one): a
    # restored row is classified against a fresh prefill of the same prompt without its history -- "chunked" = the
    # fresh state bitwise, "forced-tail" = the fresh state to rounding.  ``snapshot_captured`` is the schedule of the
    # snapshot this request left for the next one (None when its prefill did not reach the last prompt token).
    snapshot_schedule: str | None = None
    snapshot_captured: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "prefix_reused": self.prefix_reused,
            "reset": self.reset,
            "prefix_restored": self.restored,
            "snapshot_schedule": self.snapshot_schedule,
            "snapshot_captured": self.snapshot_captured,
            "prefill_tokens": self.prefill_tokens,
            "prefill_seconds": round(self.prefill_seconds, 4),
            "prefill_mode": self.prefill_mode,
            "prefill_forced_tokens": self.prefill_forced_tokens,
            "prefill_chunks": self.prefill_chunks,
            "prefill_long_chunks": self.prefill_long_chunks,
            "prefill_slabs": self.prefill_slabs,
            "prefill_tail_rows": self.prefill_tail_rows,
            "prefill_handoff_ms": round(self.prefill_handoff_ms, 3),
            "prefill_slab_prepare_ms": round(self.prefill_slab_prepare_ms, 3),
            "prefill_slab_upload_ms": round(self.prefill_slab_upload_ms, 3),
            "prefill_slab_wait_ms": round(self.prefill_slab_wait_ms, 3),
            "prefill_ms_per_prompt_token": (
                None if not self.prefill_tokens else round(1000.0 * self.prefill_seconds / self.prefill_tokens, 3)
            ),
            "ttft_seconds": round(self.ttft_seconds, 4),
            "decode_seconds": round(self.decode_seconds, 4),
            "tokens_per_second": None if self.tokens_per_second is None else round(self.tokens_per_second, 3),
            "position": self.position,
            "mtp": self.mtp,
        }


# The schedule that produced a snapshot's state (``Qwen38PromptSnapshot.schedule``, the ledger's
# ``qwen38.snapshot_schedule``): ``chunked`` = every position came from a prefill of exactly these ids from position 0
# (the chunk driver, or its own teacher-forced form for a prompt below CHUNK_PREFILL_MIN_ROWS), the schedule a fresh
# prefill of the ids takes, so an exact repeat restores the fresh state bitwise; ``forced-tail`` = a restored or
# extended request captured it: positions a fresh prefill would compute inside the chunk driver came from 1-row steps
# (a teacher-forced tail, a decoded reply) or from the tail's own alignment steps and chunk boundaries, so the state
# agrees with a fresh prefill's to rounding only (docs/NUMERICS.md).  A restore whose tail is the last prompt token
# alone recaptures the restored state and inherits its value.
SNAPSHOT_SCHEDULES = ("chunked", "forced-tail")


@dataclass(frozen=True)
class Qwen38PromptSnapshot:
    """The host record of the chain's prompt-end snapshot: the ids consumed when it was taken (the prompt without
    its last token), the n-gram context after them, the schedule that produced the state (SNAPSHOT_SCHEDULES) and
    the images' pad spans with their digests (``vision_splice.Qwen38VisionPrompt.spans``; empty for a text prompt):
    the ids alone do not identify an image prompt's pixels."""

    ids: tuple[int, ...]
    ple_context: tuple[int, int] | None
    schedule: str = "chunked"
    vision: tuple[tuple[int, int, str], ...] = ()


def vision_spans_compatible(
    committed: Sequence[tuple[int, int, str]], wanted: Sequence[tuple[int, int, str]], common: int
) -> bool:
    """Whether a committed prefix of ``common`` ids whose images are ``committed`` (spans with digests) may serve a
    prompt whose images are ``wanted``: every image span that starts below ``common`` must be the same span with the
    same digest on both sides (two images of one grid render the same pad ids; a span the prefix cuts through is not
    reused)."""

    below = lambda spans: {span for span in spans if span[0] < common}  # noqa: E731
    return below(committed) == below(wanted) and all(stop <= common for _, stop, _ in below(wanted))


@dataclass(frozen=True)
class Qwen38TeacherForcedRows:
    """:meth:`Qwen38ChatSession.teacher_force`'s result: per wanted position p the resolved argmax and the candidate
    row of the TAIL that consumed tokens 0..p, and the path the replay took.  ``full_logits`` holds, when asked, that
    TAIL's full-vocabulary logits per wanted position (the LM head's bf16 row gathered eagerly, fp32 on the host) and
    ``full_logits_seconds`` the time those gathers and readbacks took."""

    rows: dict[int, tuple[int, Any]]
    prefill_mode: str
    chunks: int
    forced_tokens: int
    seconds: float
    full_logits: dict[int, Any] = field(default_factory=dict)
    full_logits_seconds: float = 0.0


@dataclass
class Qwen38MTPPassRun:
    """The pass loop's last pass until :meth:`Qwen38ChatSession._mtp_settle` commits it: the position it started at,
    the rows it fed (``[t_P, d_1 .. d_a]``), the tokens it emitted (``[d_1 .. d_a, t']``) and how many of them the
    request took."""

    position: int
    rows: list[int]
    emitted: list[int]
    yielded: int
    host_committed: bool = False  # the rows are in the session's committed list (the pass was consumed whole)


class Qwen38ChatSession:
    """One conversation owner: the committed input sequence, the PLE n-gram context, the traced chain.

    ``template`` is the pinned ``Qwen38OfficialChatTemplate`` (``render`` and
    ``tokenizer.decode`` are all it uses).  ``chain`` offers the per-step
    primitives of ``Qwen38TracedChain``.
    """

    def __init__(
        self,
        chain: Any,
        template: Any,
        *,
        context_limit: int | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        prefill_mode: str = DEFAULT_PREFILL_MODE,
        allocated_context: int | None = None,
    ) -> None:
        # The chain's allocated context (a scripted test chain without one is the default build's; a renderer
        # without a chain names it); the limit defaults to that context less the headroom.
        if allocated_context is None:
            allocated_context = getattr(chain, "allocated_context", RESIDENT_MAX_QSA_CACHE_CAPACITY)
        if context_limit is None:
            context_limit = Qwen38ResidentContext(allocated_context).context_limit
        if type(context_limit) is not int or not 1 < context_limit <= allocated_context:
            raise ValueError(f"context limit must be in (1, {allocated_context}], got {context_limit!r}")
        self.allocated_context = allocated_context
        if prefill_mode not in PREFILL_MODES:
            raise ValueError(f"prefill mode must be one of {PREFILL_MODES}, got {prefill_mode!r}")
        self.chain = chain
        self.template = template
        self.context_limit = context_limit
        self.clock_ns = clock_ns
        # A chain without the chunk trace (teacher-forced open, the scripted test chain) serves teacher forcing.
        self.chunk_trace_available = getattr(chain, "chunk_trace_id", None) is not None
        self.prefill_mode = prefill_mode if self.chunk_trace_available else "teacher_forced"
        self.sampling = getattr(chain, "sampling", None)  # the candidate-row extension, or None: greedy only
        self.mtp = getattr(chain, "mtp", None)  # the MTP drafting extension (verify / draft / commit traces), or None
        # The drafting chains by draft count (QWEN38_MTP_DRAFTS_PER_REQUEST): a request may bind one of them as
        # ``self.mtp`` (and the chain's) for its duration; ``mtp_default`` is the server's --mtp chain, restored after.
        self.mtp_default = self.mtp
        chains = getattr(chain, "mtp_chains", None) or {}
        self.mtp_chains: dict[int, Qwen38ChainMTP] = (
            dict(chains) if chains else ({} if self.mtp is None else {self.mtp.drafts: self.mtp})
        )
        # A chain without the snapshot primitives (an older scripted chain) resets where a restore would apply.
        self.snapshot_available = callable(getattr(chain, "capture_prompt_snapshot", None))
        self.snapshot: Qwen38PromptSnapshot | None = None
        self.committed: list[int] = []
        self.ple_context: tuple[int, int] | None = None
        self.last_finish: str | None = None
        self.row_unconsumed = False  # the next token sits in the row (after length, disconnected, a hook stop)
        self.row_token: int | None = None  # a length finish's last token: read, delivered to the client, not committed
        # The committed prompt's images: their pad spans with digests (empty for a text prompt); the reuse key
        # beside the ids (``vision_spans_compatible``).
        self.committed_vision: tuple[tuple[int, int, str], ...] = ()
        self.requests_served = 0
        # The device acceptance's diagnostics (QWEN38_MTP_DEVICE_ACCEPT_DUMP, dev): with a dump directory every
        # device-decided pass records the rows it decided on and complete() writes one JSON per request for
        # the acceptance's gate tool (dev), which re-derives every pass on the host.
        self.device_accept_dump: Path | None = None
        self.device_accept_records: list[dict[str, Any]] = []
        self.last_tokens_per_second: float | None = None
        self.poisoned = False
        self._mtp_run: Qwen38MTPPassRun | None = None  # the pass loop's last pass until complete() settles it

    # -- request rendering -------------------------------------------------------------------

    def render(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        enable_thinking: bool = ENABLE_THINKING,
        reasoning_effort: str = REASONING_EFFORT,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> list[int]:
        """Prompt token ids of exactly ``messages`` (and the client's ``tools``) under the acceptance study's flags;
        no system turn is added when the client sends none.  The protocol module renders: bitwise the reference
        template."""

        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence) or not messages:
            raise Qwen38ChatRequestError("messages must be a nonempty list")
        if not all(isinstance(message, Mapping) for message in messages):
            raise Qwen38ChatRequestError("every message must be an object")
        if reasoning_effort not in REASONING_EFFORTS:
            raise Qwen38ChatRequestError(
                f"reasoning_effort must be one of {REASONING_EFFORTS}, got {reasoning_effort!r}"
            )
        try:
            return protocol.render_prompt(
                self.template.tokenizer,
                list(messages),
                list(tools),
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
            )
        except Qwen38ChatFormatError as error:
            raise Qwen38ChatRequestError(str(error)) from error

    def require_budget(self, prompt_tokens: int, max_tokens: int | None) -> int:
        """The request's completion budget, refused before touching the device when the prompt and the budget do
        not fit the context limit (which already holds the consumed EOS step).  ``None`` (the client sent no
        max_tokens) is the remaining context, ``context_limit - prompt_tokens``."""

        remaining = self.context_limit - prompt_tokens
        if max_tokens is None:
            if remaining < 1:
                raise Qwen38ChatRequestError(
                    f"context_length_exceeded: prompt {prompt_tokens} tokens leave {remaining} of the limit "
                    f"{self.context_limit} for the completion; at least 1 is needed"
                )
            return remaining
        if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_TOKENS_BOUND:
            raise Qwen38ChatRequestError(
                f"max_tokens must be an integer in [1, {MAX_TOKENS_BOUND}], got {max_tokens!r}"
            )
        if max_tokens > remaining:
            raise Qwen38ChatRequestError(
                f"context_length_exceeded: prompt {prompt_tokens} + max_tokens {max_tokens} = "
                f"{prompt_tokens + max_tokens} exceeds the limit {self.context_limit}; "
                f"at most {max(remaining, 0)} tokens remain"
            )
        return max_tokens

    # -- the steps ---------------------------------------------------------------------------

    def _forced_step(self, token_id: int, next_token_id: int | None = None) -> None:
        """A' write x_t into the token row (behind TAIL(t-1)), C HEAD(t), F PLE row for x_t, G TAIL(t).

        On an MTP chain the TAIL also runs the MTP layer's row at t, which consumes the token at t + 1:
        ``next_token_id`` when the host knows it (the next prompt token), else the step's own resolved argmax.
        """

        residue = len(self.committed) % RESIDUE_CLASSES
        self.chain.write_token_row(token_id)
        if self.mtp is not None:
            self.chain.write_mtp_next_token(next_token_id)
        self.chain.execute_head(residue)
        self.ple_context = self.chain.refresh_ple_row(token_id, self.ple_context)
        self.chain.execute_tail(residue)
        self.committed.append(token_id)

    def _prefill(
        self,
        token_ids: Sequence[int],
        should_stop: Callable[[], str | None] | None,
        before_last: Callable[[], None] | None = None,
    ) -> str | None:
        """Forced steps; at every event sync the hook may end the request (the row then holds an unconsumed token).
        ``before_last`` runs before the last token's step (the prompt-end snapshot)."""

        for index, token_id in enumerate(token_ids, start=1):
            if index == len(token_ids) and before_last is not None:
                before_last()
            self._forced_step(token_id, token_ids[index] if index < len(token_ids) else None)
            if index % PREFILL_EVENT_INTERVAL == 0:
                # Bounds the host run-ahead and surfaces a device error at a known token.
                self.chain.event_synchronize(self.chain.record_event())
                reason = None if should_stop is None else should_stop()
                if reason is not None:
                    return reason
        return None

    def resolve_prefill_mode(self, prefill_mode: str | None) -> str:
        """The session's mode, or a request's override; ``chunked`` needs the chain's chunk trace."""

        if prefill_mode is None:
            return self.prefill_mode
        if prefill_mode not in PREFILL_MODES:
            raise Qwen38ChatRequestError(f"prefill_mode must be one of {PREFILL_MODES}, got {prefill_mode!r}")
        if prefill_mode == "chunked" and not self.chunk_trace_available:
            raise Qwen38ChatRequestError("prefill_mode chunked: this server did not capture the chunk trace")
        return prefill_mode

    def chunk_prefill_rows(self, count: int) -> int:
        """Rows the chunk trace would run for ``count`` tokens from the committed position (after the alignment steps)."""

        return count - alignment_steps(len(self.committed), count) if count > 0 else 0

    def _prefill_chunked(
        self,
        token_ids: Sequence[int],
        following_token: int,
        should_stop: Callable[[], str | None] | None,
        vision: vision_splice.Qwen38VisionPrompt | None = None,
        *,
        between_chunks: Callable[[int, int], None] | None = None,
        event_rows: int | None = None,
    ) -> Qwen38PrefillResult:
        """The chunk driver over ``token_ids`` from the committed position; its alignment steps are this session's
        forced steps (with the prefill event cadence).  Runs outside the loop guard: the seed and the hand-off
        synchronize and allocate.  The committed sequence and the n-gram context follow the device: a stop at one
        of the driver's event syncs (``result.stopped``) leaves the prompt committed up to the hand-off.
        ``following_token`` (the last prompt token, teacher-forced afterwards) is the MTP layer's token at the last
        prefilled position.  ``between_chunks`` and ``event_rows`` are the lanes admission's hooks (the driver's:
        other device work between chunk groups, and the event cadence in rows); the single stream passes neither."""

        forced = 0
        start = len(self.committed)
        ahead = [*token_ids[1:], following_token]

        def forced_step(token_id: int, context: tuple[int, int] | None) -> tuple[int, int] | None:
            nonlocal forced
            if context != self.ple_context:
                raise Qwen38ChatChainError(f"chunk prefill context {context} vs the session's {self.ple_context}")
            self._forced_step(token_id, ahead[forced])
            forced += 1
            if forced % PREFILL_EVENT_INTERVAL == 0:
                self.chain.event_synchronize(self.chain.record_event())
            return self.ple_context

        # An image prompt's rotary positions and feature rows ride along, and so do the lanes admission's hooks; a
        # text prompt's plain call is the one it always was.
        hooks: dict[str, Any] = {} if vision is None else {"positions": vision.positions, "features": vision.features}
        if between_chunks is not None:
            hooks["between_chunks"] = between_chunks
        if event_rows is not None:
            hooks["event_rows"] = event_rows
        result = self.chain.chunk_prefill(
            token_ids,
            start_position=start,
            ple_context=self.ple_context,
            forced_step=forced_step,
            following_token=following_token,
            should_stop=should_stop,
            **hooks,
        )
        if result.timing.alignment_steps != forced:
            raise Qwen38ChatChainError(f"chunk prefill forced {forced} alignment steps, reported {result.timing}")
        self.committed.extend(token_ids[forced : result.position - start])
        self.ple_context = result.ple_context
        if result.position != len(self.committed) or (
            result.stopped is None and result.position != start + len(token_ids)
        ):
            raise Qwen38ChatChainError(
                f"chunk prefill ended at position {result.position} (stopped {result.stopped!r}) vs committed input "
                f"sequence length {len(self.committed)} of {start + len(token_ids)}"
            )
        return result

    def _generate(
        self,
        max_new_tokens: int,
        stop_ids: Sequence[int],
        think_budget: int | None,
        should_stop: Callable[[], str | None] | None,
    ) -> Iterator[tuple[int | None, str | None]]:
        """The runner's evented A-G loop; yields (x_t, finish) with finish set on the last item (x_t None: a hook stop).

        EOS is consumed (its HEAD is already queued when the host learns it), so
        every layer and the counter end at the same position and the next turn
        continues from ``<|im_end|>``.  On ``max_tokens`` the last token is read
        by the blocking read and not consumed.  A token at or above the tokenizer
        size is an LM-head padding row: the step is finished and the request
        ends with ``error``.  Before a step the hook may end the request with its
        own reason (the row keeps the unconsumed token), and a thinking budget
        forces ``</think>`` as a teacher-forced step once that many reasoning
        tokens were produced without the model closing the block.
        """

        produced = 0
        reasoning_tokens = 0
        thinking_open = think_budget is not None
        while True:
            reason = None if should_stop is None else should_stop()
            if reason is not None:
                yield None, reason
                return
            if thinking_open and reasoning_tokens >= think_budget:
                self._forced_step(THINK_END_ID)
                thinking_open = False
                produced += 1
                yield THINK_END_ID, "length" if produced == max_new_tokens else None
                if produced == max_new_tokens:
                    return
                continue
            if produced + 1 == max_new_tokens:
                token_id = self.chain.read_token_row()
                self._require_vocabulary(token_id)
                self.row_token = token_id
                yield token_id, "length"
                return
            residue = len(self.committed) % RESIDUE_CLASSES
            pending = self.chain.read_token_row_nonblocking()
            event = self.chain.record_event()
            self.chain.execute_head(residue)
            self.chain.event_synchronize(event)
            token_id = self.chain.pending_value(pending)
            self._require_vocabulary(token_id)
            produced += 1
            if thinking_open:
                thinking_open = token_id != THINK_END_ID
                reasoning_tokens += 1
            self.ple_context = self.chain.refresh_ple_row(token_id, self.ple_context)
            self.chain.execute_tail(residue)
            self.committed.append(token_id)
            if token_id >= TOKENIZER_SIZE or token_id in stop_ids:
                self.chain.read_token_row()  # completes TAIL(t); x_{t+1} is discarded
                yield token_id, "error" if token_id >= TOKENIZER_SIZE else "stop"
                return
            yield token_id, None

    @staticmethod
    def _require_vocabulary(token_id: int) -> None:
        if not 0 <= token_id < VOCAB_SIZE:
            raise Qwen38ChatChainError(f"token row holds {token_id}, outside the vocabulary [0, {VOCAB_SIZE})")

    # -- the MTP pass loop -------------------------------------------------------------------

    def _generate_mtp(
        self,
        max_new_tokens: int,
        stop_ids: Sequence[int],
        think_budget: int | None,
        should_stop: Callable[[], str | None] | None,
        sampling: sampling_step.Qwen38SamplingRequest | None = None,
    ) -> Iterator[tuple[int | None, str | None]]:
        """The pass loop; yields (x_t, finish) like :meth:`_generate`, ``a + 1`` tokens per pass.

        Entry: the last prompt token's TAIL is enqueued (or the row holds an unconsumed token); the blocking
        read gives the model's next token, which the first pass consumes as its row 0.  Each pass feeds the rows
        ``[t_P, d_1 .. d_k]`` and commits ``a + 1`` of them on the device; the host learns ``[d_1 .. d_a, t']``,
        streams them and keeps the last one (``t'``, not yet fed) as the next pass's row 0.  The pass's device
        commit is deferred to the next pass (or to :meth:`_mtp_settle`, which commits as many rows as the request
        consumed and rebuilds the 1-row buffers).  A forced ``</think>`` needs the 1-row chain: the pass loop
        settles, forces the token, reads the model's next token and re-enters.  When the cache has no room for a
        pass the 1-row loop finishes the request.

        A sampled request (``sampling``, admitted by ``sampling_step.drafting_admission``) runs the same loop with the
        host deciding every pass (``sampling_step.accept_pass`` on the split verify): where the greedy loop reads the
        model's next token from the row (the entry, after a forced ``</think>``) the sampled loop samples it from the
        candidate row the last TAIL wrote, and every hand-off to the 1-row chain feeds the pending token by a forced
        step (``generate_sampled`` samples from a fresh row; a sampled token left in the row would be sampled twice),
        streaming it first when the client has not seen it yet.
        """

        prompt_tokens = len(self.committed)
        decide = None if sampling is None else partial(sampling_step.accept_pass, self, sampling, prompt_tokens)
        # The device acceptance (the sampling extension's, admitted per request): its hook writes each pass's uniforms
        # and the pass loop runs the device-decided form; a refused request keeps the host-decided split form.
        before_verify_sampled = None
        if sampling is not None and getattr(self.mtp, "device_accept", False):
            _refusal, before_verify_sampled = self.chain.sampling.device_acceptance_for(sampling)
            if before_verify_sampled is not None:
                decide = None
        record_rows = self.device_accept_dump is not None

        def next_pending() -> int:
            # The model's next token after the last TAIL: read from the row (greedy), or sampled from the candidate row.
            if sampling is None:
                pending = self.chain.read_token_row()
            else:
                pending = sampling_step.sample_next_token(self, sampling, prompt_tokens=prompt_tokens).token_id
            self._require_vocabulary(pending)
            return pending

        pending = next_pending()  # completes the last TAIL: the model's next token, not consumed
        pending_yielded = False  # a pass's t' was streamed with its pass; a token read from the row was not
        produced = 0
        reasoning_tokens = 0
        thinking_open = think_budget is not None
        entered = False

        def force_think_end(settle_finish: str) -> None:
            # The 1-row loop's forced </think>: the pass loop settles first ("budget": the tokens streamed so far
            # consumed; "length": a pending token the client saw is fed, one it never saw is dropped for the forced
            # one, as _generate drops the row's token).
            nonlocal entered
            if entered:
                self._mtp_settle(settle_finish)
                entered = False
            if settle_finish == "length" and pending_yielded:
                self._forced_step(pending)
            self._forced_step(THINK_END_ID)
            if sampling is not None:
                if not pending_yielded:
                    sampling.samples.pop()  # the dropped token's sample: samples stay index-aligned with the stream
                sampling.samples.append(None)

        def finish_on_one_row_loop():
            # The 1-row loop takes the rest of the request; the pass loop settles and leaves the pending token
            # where that loop expects it: in the row when it was not streamed yet, consumed by a forced step when it
            # was (a sampled request: always fed, streamed first when the client has not seen it).
            nonlocal entered, produced, thinking_open, reasoning_tokens
            if entered:
                self._mtp_settle("length")
                entered = False
            if sampling is None and not pending_yielded:
                self.chain.write_token_row(pending)
            else:
                if not pending_yielded:
                    produced += 1
                    if thinking_open:
                        thinking_open = pending != THINK_END_ID
                        reasoning_tokens += 1
                    if pending >= TOKENIZER_SIZE or pending in stop_ids:
                        self._forced_step(pending)  # consumed, as the 1-row loops consume EOS
                        self.chain.read_token_row()  # completes its TAIL; the model's next token is discarded
                        yield pending, "error" if pending >= TOKENIZER_SIZE else "stop"
                        return
                    yield pending, None
                self._forced_step(pending)
            budget = None if not thinking_open else max(think_budget - reasoning_tokens, 0)
            with self.chain.loop_guard():
                if sampling is None:
                    yield from self._generate(max_new_tokens - produced, stop_ids, budget, should_stop)
                else:
                    yield from sampling_step.generate_sampled(
                        self,
                        sampling,
                        max_new_tokens - produced,
                        stop_ids=stop_ids,
                        tokenizer_size=TOKENIZER_SIZE,
                        think_budget=budget,
                        should_stop=should_stop,
                        forced_step=self._forced_step,
                        clock_ns=self.clock_ns,
                        prompt_tokens=prompt_tokens,  # the tokens emitted so far are output, not prompt
                    )

        while True:
            reason = None if should_stop is None else should_stop()
            if reason is not None:
                yield None, reason
                return
            if thinking_open and reasoning_tokens >= think_budget:
                force_think_end("length")
                thinking_open = False
                produced += 1
                yield THINK_END_ID, "length" if produced == max_new_tokens else None
                if produced == max_new_tokens:
                    return
                pending = next_pending()
                pending_yielded = False
                continue
            position = len(self.committed)
            if not pending_yielded:
                if produced + 1 == max_new_tokens:
                    self.row_token = pending
                    if sampling is not None:
                        # The row holds the token the client saw, as after generate_sampled.
                        self.chain.write_token_row(pending)
                    yield pending, "length"
                    return
                if pending >= TOKENIZER_SIZE or pending in stop_ids or not self.chain.mtp_pass_fits(position):
                    yield from finish_on_one_row_loop()
                    return
                produced += 1
                if thinking_open:
                    thinking_open = pending != THINK_END_ID
                    reasoning_tokens += 1
                yield pending, None
                pending_yielded = True
                reason = None if should_stop is None else should_stop()
                if reason is not None:
                    # The hit token is consumed as the 1-row loop consumes it: the previous pass settles whole, the
                    # token is fed by a forced step (the row then holds the model's next token).
                    if entered:
                        self._mtp_settle("length")
                        entered = False
                    self._forced_step(pending)
                    yield None, reason
                    return
            elif not self.chain.mtp_pass_fits(position):
                yield from finish_on_one_row_loop()
                return
            uniforms_before = 0 if sampling is None else len(sampling.uniforms)
            if entered:
                with self.chain.loop_guard():
                    record = self.chain.mtp_step()
            else:
                # The eager seed, then the bootstrap pass; a sampled request's decision rides on the chain (a greedy
                # request keeps the call form every chain knows; the device acceptance's hook the device form).
                if before_verify_sampled is not None:
                    record = self.chain.mtp_enter(
                        pending,
                        self.ple_context,
                        before_verify_sampled=before_verify_sampled,
                        record_candidate_rows=record_rows,
                    )
                elif decide is None:
                    record = self.chain.mtp_enter(pending, self.ple_context)
                else:
                    record = self.chain.mtp_enter(pending, self.ple_context, decide=decide)
                entered = True
            if getattr(record, "arithmetic", None) == mtp_v2.DEVICE_ACCEPT_ARITHMETIC:
                # The device-decided pass's ledger (the extension's statistics reader) and, with the dump on, its
                # record (the pass's own uniforms beside the request's ledger: the gate's slicing has both).
                sampling.mtp.record_device(record.statistics, self.mtp.drafts)
                if record.candidate_rows is not None:
                    self.device_accept_records.append(
                        {
                            "index": len(self.device_accept_records),
                            "candidate_rows": [float(value) for value in record.candidate_rows.reshape(-1).tolist()],
                            "tokens": [int(token) for token in record.tokens],
                            "statistics": [float(value) for value in record.statistics],
                            "uniforms": [float(value) for value in sampling.uniforms[uniforms_before:]],
                            "committed_tokens": len(self.committed),  # the pass's history: the stream cut here
                        }
                    )
            accepted = record.accepted
            emitted = [int(token) for token in record.argmaxes[: accepted + 1]]
            for token_id in emitted:
                self._require_vocabulary(token_id)
            run = Qwen38MTPPassRun(position, [pending, *emitted[:accepted]], emitted, 0)
            self._mtp_run = run
            budget_hit = False
            for index, token_id in enumerate(emitted):
                run.yielded = index + 1
                produced += 1
                if thinking_open:
                    thinking_open = token_id != THINK_END_ID
                    reasoning_tokens += 1
                if sampling is not None:
                    sampling.samples.append(None)  # a pass token carries no per-token sample (no logprobs drafted)
                if token_id >= TOKENIZER_SIZE or token_id in stop_ids:
                    yield token_id, "error" if token_id >= TOKENIZER_SIZE else "stop"
                    return
                if produced == max_new_tokens:
                    self.row_token = token_id  # _mtp_settle leaves it in the row, as _generate does
                    yield token_id, "length"
                    return
                yield token_id, None
                reason = None if should_stop is None else should_stop()
                if reason is not None:
                    yield None, reason
                    return
                if thinking_open and reasoning_tokens >= think_budget:
                    budget_hit = True  # the 1-row loop forces </think> here: the rest of the pass is rolled back
                    break
            if budget_hit:
                force_think_end("budget")
                thinking_open = False
                produced += 1
                yield THINK_END_ID, "length" if produced == max_new_tokens else None
                if produced == max_new_tokens:
                    return
                pending = next_pending()
                pending_yielded = False
                continue
            # The whole pass was consumed: its rows are the device's; its device commit is the next pass's first
            # job or the settle's, so the run stays until then.
            self.committed.extend(run.rows)
            run.host_committed = True
            pending = emitted[accepted]
            pending_yielded = True

    def _mtp_settle(self, finish: str) -> None:
        """Leave the pass loop after its last pass (``self._mtp_run``): commit the rows the request consumed and
        rebuild the 1-row buffers, then leave the row as :meth:`_generate` would.

        Of the pass's ``[t_P, d_1 .. d_a]`` rows and ``[d_1 .. d_a, t']`` emitted tokens, the request took the
        first ``yielded`` tokens; the last one is consumed (fed to the device) unless the request ended on
        ``length``, where the 1-row loop leaves the last token in the row unconsumed.  A consumed ``t'`` (not a row
        of the pass) is fed by a forced step after the hand-off.  The rows past the consumed prefix (drafts the
        device already ran) are rolled back by committing fewer rows.
        """

        run = self._mtp_run
        if run is None:
            return
        accepted = len(run.rows) - 1
        consume_last = finish != "length"
        consumed = min(run.yielded - 1 + consume_last, accepted)
        rows = 1 + consumed
        force_last = consume_last and run.yielded - 1 + consume_last > accepted
        if run.host_committed and rows != len(run.rows):
            raise Qwen38ChatChainError(f"a consumed pass settles whole: {rows} of {len(run.rows)} rows")
        self.ple_context = self.chain.mtp_leave(position=run.position + rows, committed_rows=rows)
        if not run.host_committed:
            self.committed.extend(run.rows[:rows])
        self._mtp_run = None
        if force_last:
            self._forced_step(run.emitted[accepted])
        else:
            self.chain.write_token_row(run.emitted[rows - 1])  # the model's next token, unconsumed, as after _generate

    # -- one request -------------------------------------------------------------------------

    def _write_device_accept_dump(self, sampling: sampling_step.Qwen38SamplingRequest) -> Path:
        """One JSON per request for the device acceptance's gate tool (dev): k, the request's sampling parameters,
        its uniform ledger and the device-decided passes' records (``index`` in the order the passes ran,
        ``candidate_rows`` the rows the device decided on as (k + 1) * 256 floats, ``tokens`` the pass's verify tokens
        ``[t, d_1 .. d_k]``, ``statistics`` the 16 lanes, ``uniforms`` the k + 1 draws that pass consumed, and
        ``committed_tokens`` the length of the committed stream at the pass, its rows' history)."""

        parameters = sampling.parameters
        run = {
            "drafts": self.mtp.drafts,
            "parameters": {
                "temperature": parameters.temperature,
                "top_k": parameters.top_k,
                "top_p": parameters.top_p,
                "min_p": parameters.min_p,
            },
            "uniforms": [float(value) for value in sampling.uniforms],
            "records": list(self.device_accept_records),
        }
        path = self.device_accept_dump / f"device-accept-{self.requests_served:05d}.json"
        path.write_text(json.dumps(run) + "\n", encoding="utf-8")
        return path

    def reset(self) -> None:
        """Position 0 and position-zero state at the captured addresses; the committed sequence is dropped, and so
        is the prompt-end snapshot's record: the prefill that follows writes the KV and compressed caches from row 0,
        so the snapshot's positional half (the caches below its position, which the snapshot does not copy) is gone
        the moment that prefill starts, whether or not it reaches its last token."""

        self.chain.reset_and_seed(SEED_TOKEN_ID)
        self.committed = []
        self.snapshot = None
        self.ple_context = None
        self.last_finish = None
        self.row_unconsumed = False
        self.row_token = None
        self.committed_vision = ()

    def reusable_prefix(self, token_ids: Sequence[int], vision: Sequence[tuple[int, int, str]] = ()) -> tuple[int, str]:
        """How the device meets ``token_ids`` whose images are ``vision`` (their pad spans with digests, empty for
        text): ``(n, "extends")`` when they extend the committed ``n`` ids (or repeat them exactly while the
        unconsumed next token is still in the row; a partial match cannot be rewound) and the images inside those
        ids are the committed ones, ``(n, "snapshot")`` when they extend the ``n`` ids of the prompt-end snapshot
        instead (its images alike), else ``(0, "reset")``.  Two images of one grid render identical ids, so the
        spans' digests are part of the key."""

        common = 0
        while common < len(self.committed) and common < len(token_ids) and self.committed[common] == token_ids[common]:
            common += 1
        if (
            common == len(self.committed)
            and (common < len(token_ids) or self.row_unconsumed)
            and vision_spans_compatible(self.committed_vision, vision, common)
        ):
            return common, "extends"
        snapshot = self.snapshot
        if (
            snapshot is not None
            and len(token_ids) > len(snapshot.ids)
            and tuple(token_ids[: len(snapshot.ids)]) == snapshot.ids
            and vision_spans_compatible(snapshot.vision, vision, len(snapshot.ids))
        ):
            return len(snapshot.ids), "snapshot"
        return 0, "reset"

    def _restore_prompt_snapshot(self) -> None:
        """The chain's snapshot back on device (position, phases and rotary shift included); the host follows its
        record."""

        self.chain.restore_prompt_snapshot()
        self.committed = list(self.snapshot.ids)
        self.ple_context = self.snapshot.ple_context
        self.committed_vision = self.snapshot.vision
        self.last_finish = None
        self.row_unconsumed = False

    def _bind_mtp(self, chain_mtp: Qwen38ChainMTP | None) -> None:
        """The drafting chain the next passes run: the session's ``mtp`` and the traced chain's (its step writes, the
        pass loop's entry and settle read that attribute).  The default chain when no request binds another."""

        if chain_mtp is not self.mtp:
            if self.mtp is not None and self.mtp.chain is not None:
                raise Qwen38ChatChainError("cannot rebind the drafting chain while a pass loop is open")
            self.mtp = chain_mtp
            if hasattr(self.chain, "mtp"):
                self.chain.mtp = chain_mtp

    def _capture_prompt_snapshot(self, schedule: str = "chunked") -> None:
        """The device state after the committed ids (every prompt token but the last), copied on the chain; the
        record makes a later render that extends these ids a restore instead of a reset.  ``schedule`` names what
        produced the state (SNAPSHOT_SCHEDULES)."""

        if schedule not in SNAPSHOT_SCHEDULES:
            raise ValueError(f"snapshot schedule must be one of {SNAPSHOT_SCHEDULES}, got {schedule!r}")
        if not self.snapshot_available or not self.committed:
            return
        self.chain.capture_prompt_snapshot(len(self.committed))
        self.snapshot = Qwen38PromptSnapshot(tuple(self.committed), self.ple_context, schedule, self.committed_vision)

    def complete(
        self,
        token_ids: Sequence[int],
        max_tokens: int | None,
        *,
        stop_ids: Sequence[int] = EOS_TOKEN_IDS,
        on_token: Callable[[int], None] | None = None,
        prefill_mode: str | None = None,
        think_budget: int | None = None,
        should_stop: Callable[[], str | None] | None = None,
        sampling: sampling_step.Qwen38SamplingRequest | None = None,
        speculative: bool = True,
        verify_each_step: bool = False,
        mtp_drafts: int | None = None,
        vision: vision_splice.Qwen38VisionPrompt | None = None,
    ) -> Qwen38ChatCompletion:
        """Prefill what the device does not already hold, then generate up to ``max_tokens`` tokens (the remaining
        context when ``None``; ``require_budget``).

        ``vision`` carries an image prompt's rotary positions and feature rows (``token_ids`` are the expanded ids:
        one ``<|image_pad|>`` per merged image token) and its images' digests; the committed prefix and the
        prompt-end snapshot are keyed on the ids AND the images' pad spans with their digests, so a follow-up turn on
        the same image prefills only the new turn while a prompt with another image of the same grid (the same ids)
        resets.  The pads of the extension never take 1-row steps: an extension whose alignment steps or forced tail
        would hold a pad resets instead.

        Any failure inside the device section leaves the chain's queue and the
        model owner in an unknown state: the session is poisoned and must not
        serve again.  ``on_token`` raising ``OSError`` (the client went away) is
        not such a failure: it is called between steps, so generation stops with
        ``disconnected`` and, as after ``length``, the next token stays in the row.
        ``prefill_mode`` overrides the session's mode for this request.
        ``should_stop`` is polled between steps and at the prefill's event syncs
        (the teacher-forced cadence, the chunk driver's per-chunk events) and
        ends the request with the reason it returns, the row unconsumed (a stop
        inside the chunk driver hands off at the chunks replayed so far; the row
        then holds no prediction and an exact repeat of the committed prefix
        resets);
        ``think_budget`` forces ``</think>`` after that many reasoning tokens
        (the caller passes it only when the prompt left the think block open).
        ``sampling`` (a request with ``temperature > 0``) runs the sampled loop over
        the chain's candidate row instead of the greedy loop; ``None`` is greedy.  On
        a device-sampler chain a request the device policy admits runs the device loop
        (the greedy loop plus a draw write per step; ``verify_each_step`` adds the
        per-step host check of the discriminator); the others keep the host loop.
        On an MTP chain a greedy chunked-mode request generates through the pass
        loop unless ``speculative`` is False (the 1-row loop, for the hand-off gate).
        """

        token_ids = list(token_ids)
        if not token_ids or any(type(value) is not int or not 0 <= value < VOCAB_SIZE for value in token_ids):
            raise Qwen38ChatRequestError("prompt token ids must be a nonempty list of vocabulary ids")
        max_tokens = self.require_budget(len(token_ids), max_tokens)
        mode = self.resolve_prefill_mode(prefill_mode)
        has_image_pads = bool(vision_splice.image_lanes(token_ids))
        if vision is not None:
            try:
                vision.validate_prompt(token_ids)
            except ValueError as error:
                raise Qwen38ChatRequestError(f"image prompt: {error}") from error
            if mode != "chunked":
                raise Qwen38ChatRequestError(
                    "image prompts need the chunked prefill mode (their pads never take 1-row steps)"
                )
            # An image prompt prefills from position 0 (below): all but its last token must fill the chunk trace's
            # minimum (a real image prompt is at least 66 tokens: 64 pads and the two markers).
            if len(token_ids) - 1 < CHUNK_PREFILL_MIN_ROWS:
                raise Qwen38ChatRequestError("image prompt too short for the chunked prefill")
        elif has_image_pads:
            raise Qwen38ChatRequestError("the prompt holds image pads but no vision inputs")
        if sampling is not None and self.sampling is None:
            raise Qwen38ChatRequestError("sampling is unavailable: this chain captured no candidate row (greedy only)")
        if self.poisoned:
            raise Qwen38ChatChainError("session is poisoned by an earlier device failure")
        # The request's drafting chain (extra_body.mtp_drafts on a QWEN38_MTP_DRAFTS_PER_REQUEST server): bound as
        # the session's and the traced chain's ``mtp`` for the request (the pass loop, the 1-row decode's step write,
        # the settle, the summary go through it), the default restored on the way out.
        if mtp_drafts is not None:
            if mtp_drafts not in self.mtp_chains:
                raise Qwen38ChatRequestError(
                    f"mtp_drafts {mtp_drafts!r} is not a chain of this session (captured: {sorted(self.mtp_chains)})"
                )
            self._bind_mtp(self.mtp_chains[mtp_drafts])
        # MTP drafting serves the greedy requests of the chunked mode and, with the chain's switch on, the sampled
        # requests the pass loop can bound (drafting_admission); the rest take the 1-row loops (speculative=False is
        # the diagnostic form: a greedy request on the 1-row loop of an MTP chain).
        drafting = self.mtp is not None and speculative and mode == "chunked"
        started_ns = self.clock_ns()
        spans = () if vision is None else vision.spans(token_ids)
        common, reuse = self.reusable_prefix(token_ids, spans)
        if common and vision is not None:
            # The extension's tokens that the 1-row body would take (the alignment steps of a chunked extension, or
            # the whole forced tail of a short one) must be text: image pads never take 1-row steps.
            head = list(token_ids[common:-1])
            aligned = alignment_steps(common, len(head))
            chunkable = mode == "chunked" and len(head) - aligned >= CHUNK_PREFILL_MIN_ROWS
            if vision_splice.image_lanes(head[:aligned] if chunkable else head):
                common, reuse = 0, "reset"
        # The device sampler's per-request writes (the policy, the greedy flag, the first draw) precede every
        # prompt step: the last prompt TAIL chooses the first token under this request's policy.
        device_loop = False
        if (
            getattr(self.sampling, "sampler", None) is not None
            or getattr(self.sampling, "device_accept", None) is not None
        ):
            # The device sampler's writes, or the device acceptance's request start (its policy row, its uniform
            # stream restarted: an --mtp server samples on the host, so the sampler alone would never reach it).
            self.sampling.begin_request(sampling)
            device_loop = sampling is not None and bool(sampling.uniforms)
        if sampling is not None:
            refusal = sampling_step.drafting_admission(self.mtp, sampling, device_loop=device_loop)
            if not drafting:
                refusal = refusal or f"refused: {'teacher-forced prefill' if mode != 'chunked' else 'not speculative'}"
            drafting = drafting and refusal is None
            sampling.mtp_drafting = "drafted" if drafting else refusal
        snapshot_before = self.snapshot
        restored_schedule = snapshot_before.schedule if reuse == "snapshot" and snapshot_before is not None else None
        try:
            self.row_token = None
            self.device_accept_records = []
            if reuse == "snapshot":
                self._restore_prompt_snapshot()
            elif reuse == "reset":
                self.reset()  # drops the snapshot record too: the caches below its position are rewritten from row 0
            # The snapshot stays valid until a prefill reaching its last token replaces it or a reset drops it (its
            # recurrent buffers change on no other path; a stopped prefill that extends the committed ids leaves the
            # caches below the snapshot's position untouched), so a stopped extension leaves the earlier one restorable.
            suffix = token_ids[common:]
            chunked: Qwen38PrefillResult | None = None
            # The schedule the capture records (SNAPSHOT_SCHEDULES): a prefill from position 0 is the schedule a fresh
            # prefill of these ids takes; a restore whose tail is the last token alone recaptures the restored state
            # itself; every other restore or extension leaves positions a fresh prefill would chunk to the 1-row steps
            # or to the tail's own schedule.
            if reuse == "reset" or common == 0:
                schedule = "chunked"  # a prefill from position 0 (a reset, or a new session's first request)
            elif reuse == "snapshot" and len(suffix) == 1:
                schedule = snapshot_before.schedule
            else:
                schedule = "forced-tail"
            capture = partial(self._capture_prompt_snapshot, schedule)
            # All but the last prompt token through the chunk trace when enough rows remain after the alignment
            # steps; the last one is the first decode replay and is always teacher-forced inside the guard.  The
            # prompt-end snapshot is taken before that last token: after the hand-off, or inside the forced prefill.
            before_last: Callable[[], None] | None = capture
            # The prompt's images are the committed ones from here (the snapshot records them with the ids): the
            # spans below ``common`` are the committed prefix's own, the rest this prefill's.
            self.committed_vision = spans
            prefill_vision = vision
            if vision is not None and common:
                # The extension carries the feature rows of the pads it covers (the prefix's pads are on device).
                pads_before = len(vision_splice.image_lanes(token_ids[:common]))
                pads_in = len(vision_splice.image_lanes(suffix[:-1]))
                prefill_vision = vision_splice.Qwen38VisionPrompt(
                    vision.positions,
                    vision.features[pads_before : pads_before + pads_in],
                    vision.digest,
                    vision.image_digests,
                )
            if mode == "chunked" and self.chunk_prefill_rows(len(suffix) - 1) >= CHUNK_PREFILL_MIN_ROWS:
                chunked = self._prefill_chunked(suffix[:-1], suffix[-1], should_stop, prefill_vision)
                suffix = suffix[-1:]
                before_last = None
                if chunked.stopped is None:
                    capture()
            # A stop inside the chunk driver ended the request at its hand-off: nothing more runs on the device.
            chunk_stopped = chunked is not None and chunked.stopped is not None
            generated: list[int] = []
            finish: str | None = chunked.stopped if chunk_stopped else None
            hook_stopped = chunk_stopped
            first_ns = last_ns = started_ns
            mtp_before = None if not drafting else self.mtp.counters()

            def consume(steps) -> None:
                nonlocal finish, hook_stopped, first_ns, last_ns
                for token_id, finish_reason in steps:
                    if token_id is None:
                        finish, hook_stopped = finish_reason, True
                        break
                    last_ns = self.clock_ns()
                    if not generated:
                        first_ns = last_ns
                    generated.append(token_id)
                    if finish_reason is not None:
                        finish = finish_reason
                    if on_token is not None and finish_reason != "error" and token_id not in stop_ids:
                        try:
                            on_token(token_id)
                        except OSError:
                            if finish_reason is None:
                                finish = "disconnected"
                            break

            if chunk_stopped:
                prefill_done_ns = self.clock_ns()
                first_ns = last_ns = prefill_done_ns
            elif drafting:
                # The pass loop takes the guard per pass (its mode switches synchronize) and settles outside it.
                with self.chain.loop_guard():
                    finish = self._prefill(suffix, should_stop, before_last)
                hook_stopped = finish is not None
                prefill_done_ns = self.clock_ns()
                first_ns = last_ns = prefill_done_ns
                if not hook_stopped:
                    finish = "length"
                    consume(self._generate_mtp(max_tokens, stop_ids, think_budget, should_stop, sampling))
                    self._mtp_settle(finish)
            else:
                with self.chain.loop_guard():
                    finish = self._prefill(suffix, should_stop, before_last)
                    hook_stopped = finish is not None
                    prefill_done_ns = self.clock_ns()
                    first_ns = last_ns = prefill_done_ns
                    if not hook_stopped:
                        finish = "length"
                        if sampling is None:
                            steps = self._generate(max_tokens, stop_ids, think_budget, should_stop)
                        elif device_loop:
                            steps = sampling_step.generate_sampled_on_device(
                                self,
                                sampling,
                                max_tokens,
                                stop_ids=stop_ids,
                                tokenizer_size=TOKENIZER_SIZE,
                                prefilled=bool(suffix),
                                think_budget=think_budget,
                                should_stop=should_stop,
                                forced_step=self._forced_step,
                                clock_ns=self.clock_ns,
                                verify_each_step=verify_each_step,
                            )
                        else:
                            steps = sampling_step.generate_sampled(
                                self,
                                sampling,
                                max_tokens,
                                stop_ids=stop_ids,
                                tokenizer_size=TOKENIZER_SIZE,
                                think_budget=think_budget,
                                should_stop=should_stop,
                                forced_step=self._forced_step,
                                clock_ns=self.clock_ns,
                            )
                        consume(steps)
            self.last_finish = finish
            self.row_unconsumed = not chunk_stopped and (hook_stopped or finish in ("length", "disconnected"))
            position = self.chain.position()
            if position != len(self.committed):
                raise Qwen38ChatChainError(
                    f"device position {position} vs committed input sequence length {len(self.committed)}"
                )
        except BaseException:
            self.poisoned = True
            self._bind_mtp(self.mtp_default)
            raise
        decode_seconds = (last_ns - first_ns) / 1e9
        tokens_per_second = (
            (len(generated) - 1) / decode_seconds if len(generated) >= 2 and decode_seconds > 0 else None
        )
        self.requests_served += 1
        self.last_tokens_per_second = tokens_per_second
        prefill_tokens = len(token_ids) - common
        if self.device_accept_dump is not None and self.device_accept_records and sampling is not None:
            self._write_device_accept_dump(sampling)
        completion = Qwen38ChatCompletion(
            token_ids=generated,
            finish_reason=finish,
            prompt_tokens=len(token_ids),
            prefix_reused=common,
            reset=reuse == "reset",
            restored=reuse == "snapshot",
            snapshot_schedule=restored_schedule,
            snapshot_captured=(
                None if self.snapshot is None or self.snapshot is snapshot_before else self.snapshot.schedule
            ),
            prefill_tokens=prefill_tokens,
            prefill_seconds=(prefill_done_ns - started_ns) / 1e9,
            ttft_seconds=(first_ns - started_ns) / 1e9,
            decode_seconds=decode_seconds,
            tokens_per_second=tokens_per_second,
            position=position,
            prefill_mode="teacher_forced" if chunked is None else "chunked",
            prefill_forced_tokens=(
                prefill_tokens if chunked is None else chunked.timing.alignment_steps + (0 if chunk_stopped else 1)
            ),
            prefill_chunks=0 if chunked is None else chunked.timing.chunks,
            prefill_long_chunks=0 if chunked is None else chunked.timing.long_chunks,
            prefill_slabs=0 if chunked is None else chunked.timing.slabs,
            prefill_tail_rows=0 if chunked is None else chunked.timing.tail_rows,
            prefill_handoff_ms=0.0 if chunked is None else chunked.timing.handoff_ms,
            prefill_slab_prepare_ms=0.0 if chunked is None else sum(chunked.timing.slab_prepare_ms),
            prefill_slab_upload_ms=0.0 if chunked is None else sum(chunked.timing.slab_upload_ms),
            prefill_slab_wait_ms=0.0 if chunked is None else sum(chunked.timing.slab_wait_ms),
            mtp=None if not drafting else self.mtp.summary(since=mtp_before),
        )
        self._bind_mtp(self.mtp_default)
        return completion

    # -- teacher forcing with rows (the agreement records) -----------------------------------------

    def teacher_force(
        self,
        token_ids: Sequence[int],
        *,
        positions: Sequence[int],
        prefill_mode: str | None = None,
        full_logits: bool = False,
    ) -> Qwen38TeacherForcedRows:
        """From position 0 through ``token_ids``: at every wanted position p the resolved argmax and the candidate
        row of the TAIL that consumed tokens 0..p, so p's row must be a forced step.  In the chunked mode a stretch of
        unwanted positions goes through the chunk driver when enough rows remain after its alignment steps (the
        chunk rows yield no logits row); the forced mode forces every token.  Feeds up to the last wanted position
        and leaves the row unconsumed, like a ``length`` finish.  Needs the candidate row (``--sampling``).
        ``full_logits`` also gathers that TAIL's full-vocabulary logits (the sampler's fallback read) at every wanted
        position."""

        if self.sampling is None:
            raise Qwen38ChatRequestError("teacher forcing with rows needs the candidate row: this chain captured none")
        wanted_set = set(positions)
        wanted = sorted(wanted_set)
        if not wanted or wanted[0] < 0 or wanted[-1] >= len(token_ids):
            raise Qwen38ChatRequestError(
                f"wanted positions must lie in [0, {len(token_ids)}), got {wanted[:1]}..{wanted[-1:]}"
            )
        token_ids = list(token_ids[: wanted[-1] + 1])
        if any(type(value) is not int or not 0 <= value < VOCAB_SIZE for value in token_ids):
            raise Qwen38ChatRequestError("token ids must be vocabulary ids")
        mode = self.resolve_prefill_mode(prefill_mode)
        if self.poisoned:
            raise Qwen38ChatChainError("session is poisoned by an earlier device failure")
        started_ns = self.clock_ns()
        rows: dict[int, tuple[int, Any]] = {}
        full: dict[int, Any] = {}
        gather_ns = 0
        chunks = forced = 0
        if getattr(self.sampling, "sampler", None) is not None:
            self.sampling.begin_request(
                None
            )  # the device sampler under the greedy flag: TAIL's token row is the argmax
        try:
            self.reset()
            next_wanted = 0  # wanted[next_wanted] is the first wanted position at or after the committed length
            while len(self.committed) < len(token_ids):
                index = len(self.committed)
                gap = wanted[next_wanted] - index
                if mode == "chunked" and self.chunk_prefill_rows(gap) >= CHUNK_PREFILL_MIN_ROWS:
                    result = self._prefill_chunked(
                        token_ids[index : wanted[next_wanted]], token_ids[wanted[next_wanted]], None
                    )
                    chunks += result.timing.chunks
                    forced += result.timing.alignment_steps
                    continue
                # A forced run: through this wanted position and every later one until a stretch the chunk path takes.
                end = wanted[next_wanted]
                next_wanted += 1
                while next_wanted < len(wanted):
                    gap = wanted[next_wanted] - (end + 1)
                    if mode == "chunked" and gap - alignment_steps(end + 1, gap) >= CHUNK_PREFILL_MIN_ROWS:
                        break
                    end = wanted[next_wanted]
                    next_wanted += 1
                with self.chain.loop_guard():
                    for position in range(index, end + 1):
                        following = token_ids[position + 1] if position + 1 < len(token_ids) else None
                        self._forced_step(token_ids[position], following)
                        forced += 1
                        if position in wanted_set:
                            rows[position] = (self.chain.read_token_row(), self.sampling.read_candidate_row())
                            if full_logits:
                                # TAIL(position) ran with residue position mod 4 (_forced_step, before the append)
                                gather_started_ns = self.clock_ns()
                                full[position] = self.sampling.read_full_logits(position % RESIDUE_CLASSES)
                                gather_ns += self.clock_ns() - gather_started_ns
                        elif forced % PREFILL_EVENT_INTERVAL == 0:
                            self.chain.event_synchronize(self.chain.record_event())
            position = self.chain.position()
            if position != len(self.committed):
                raise Qwen38ChatChainError(
                    f"device position {position} vs committed input sequence length {len(self.committed)}"
                )
        except BaseException:
            self.poisoned = True
            raise
        self.last_finish = "length"
        self.row_unconsumed = True
        self.row_token = None
        return Qwen38TeacherForcedRows(
            rows, mode, chunks, forced, (self.clock_ns() - started_ns) / 1e9, full, gather_ns / 1e9
        )


class Qwen38TextStream:
    """Incremental detokenizer: decode the unemitted tail, hold it while it ends in U+FFFD (a split UTF-8 sequence)."""

    def __init__(self, decode: Callable[[list[int]], str]) -> None:
        self.decode = decode
        self.pending: list[int] = []

    def push(self, token_id: int) -> str:
        self.pending.append(token_id)
        text = self.decode(self.pending)
        if text.endswith("�"):
            return ""
        self.pending = []
        return text


def template_decoder(template: Any) -> Callable[[list[int]], str]:
    return lambda ids: template.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


# -- hardware ----------------------------------------------------------------------------------


def resolve_route(hardware_profile: ResidentHardwareProfile) -> tuple[ResidentHardwareProfile, dict]:
    """The profile with its route: derived from the cluster descriptor (no device is opened; a ring also asks the
    fabric's control plane for its line order) and adopted when the profile leaves it ``None`` (the p150 line and the
    QuietBox 2: recorded, not pinned), checked against the pinned one otherwise."""

    import yaml

    lane = f"{hardware_profile.host} partition-{hardware_profile.partition.upper()}"
    descriptor_path = Path(ttnn.cluster.serialize_cluster_descriptor()).resolve(strict=True)
    document = yaml.safe_load(descriptor_path.read_bytes())
    chips_with_mmio = physical_route.parse_chips_with_mmio(
        document, expected_device_nodes=set(hardware_profile.device_nodes)
    )
    derivation: dict[str, Any] = {"cluster_descriptor": str(descriptor_path), "chips_with_mmio": chips_with_mmio}
    if hardware_profile.ethernet_graph == "line":
        route = physical_route.derive_canonical_line_route(document, chips_with_mmio)
        derivation["route_derivation"] = physical_route.derive_canonical_line_route.__name__
    else:
        # A ring under the 1x4 LINE descriptor: the fabric's topology solver chose which three of the four links
        # carry the line, and the mesh must open in that order (QuietBox (0, 2, 1, 3) = its ring walk; QuietBox 2
        # (1, 0, 3, 2), 2026-09-18, where the ring walk (0, 1, 2, 3) put a mesh neighbour pair on the unused link
        # and the first all_gather failed).  The walk is recorded next to the route for comparison.
        route = physical_route.derive_fabric_line_route(
            document,
            chips_with_mmio,
            lambda mesh_id, chip_id: int(ttnn.cluster.get_chip_unique_id_from_fabric_node_id(mesh_id, chip_id)),
        )
        ring_walk = physical_route.derive_ring_walk_route(document, chips_with_mmio)
        derivation["route_derivation"] = physical_route.derive_fabric_line_route.__name__
        derivation["ring_walk_route"] = list(ring_walk)
        derivation["ring_walk_agrees"] = ring_walk == route
    route_nodes = physical_route.route_device_nodes(route, chips_with_mmio)
    if hardware_profile.route is None:
        hardware_profile = replace(hardware_profile, route=route, route_nodes=route_nodes)
    elif route != hardware_profile.route or route_nodes != hardware_profile.route_nodes:
        raise Qwen38ChatChainError(
            f"{lane} route actual=(logical={route}, nodes={route_nodes}) "
            f"expected=(logical={hardware_profile.route}, nodes={hardware_profile.route_nodes})"
        )
    return hardware_profile, derivation


def open_partition_b_mesh(
    marker: Marker, hardware_profile: ResidentHardwareProfile, *, command_queues: int = 1
) -> tuple[Any, dict]:
    """The runner's mesh open for one lane: the profile's route derivation, FABRIC_1D, one 1D four-device mesh as
    the logical 1x4.  ``command_queues`` (1 by default: the open call as it always was) opens a second queue for the
    MTP pass's early rows read (``QWEN38_MTP_PLE_EARLY``); nothing but buffer reads may ever run on it.

    Same calls and checks as ``run()`` of the timing runner on a larger
    host (its partition-B values are the profile's defaults, hence the
    name; a partition-A profile brings its own nodes, route and locks; both
    are a 4x1 line reshaped to 1x4, ``derive_canonical_line_route``).  A ring
    profile (the QuietBox, the QuietBox 2) opens the 1x4 in the order the fabric
    embedded its LINE descriptor, ``derive_fabric_line_route``.  Every check before the fabric enable
    raises with the fabric untouched; a failure after it disables the fabric
    before re-raising, so the caller owns the fabric only once this returns.
    """

    if isinstance(command_queues, bool) or type(command_queues) is not int or command_queues not in (1, 2):
        raise Qwen38ChatChainError(f"the mesh opens with 1 or 2 command queues, got {command_queues!r}")
    lane = f"{hardware_profile.host} partition-{hardware_profile.partition.upper()}"
    discovered = (int(ttnn.GetNumAvailableDevices()), int(ttnn.get_num_pcie_devices()), int(ttnn.get_num_devices()))
    if discovered != (4, 4, 4):
        raise Qwen38ChatChainError(f"{lane} visibility actual={discovered} expected=(4, 4, 4)")
    descriptor = ttnn._ttnn.multi_device.SystemMeshDescriptor()
    physical_shape = tuple(int(value) for value in descriptor.local_shape())
    if physical_shape != hardware_profile.system_mesh_local_shape or not bool(descriptor.all_local()):
        raise Qwen38ChatChainError(
            f"{lane} physical mesh is not one local {hardware_profile.system_mesh_local_shape} line: "
            f"actual={physical_shape}, all_local={bool(descriptor.all_local())}"
        )
    hardware_profile, derivation = resolve_route(hardware_profile)
    chips_with_mmio = derivation["chips_with_mmio"]
    route, route_nodes = hardware_profile.route, hardware_profile.route_nodes
    lock_proof = hardware_profiles.verify_inherited_locks(hardware_profile)
    marker("before-fabric-enable")
    ttnn.set_fabric_config(**FABRIC_CONFIG)
    mesh = None
    try:
        marker("before-mesh-open")
        mesh = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(*physical_shape),
            physical_device_ids=list(route),
            l1_small_size=24576,
            trace_region_size=0,
            **({"num_command_queues": command_queues} if command_queues != 1 else {}),
        )
        if tuple(int(value) for value in mesh.shape) != MESH_SHAPE:
            mesh.reshape(ttnn.MeshShape(*MESH_SHAPE))
        Qwen38MeshContract(route).validate_mesh(mesh)
        live_mapping = hardware_profiles.live_mapping(mesh, chips_with_mmio, hardware_profile)
        mesh.enable_program_cache()
    except BaseException:
        if mesh is not None:  # a failed identity check after the open: release the cards before the fabric
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        raise
    return mesh, {
        "partition": hardware_profile.partition,
        "command_queues": command_queues,
        "device_nodes": list(hardware_profile.device_nodes),
        "physical_open_shape": list(physical_shape),
        "logical_mesh_shape": list(MESH_SHAPE),
        "canonical_logical_route": list(route),
        "canonical_device_node_route": list(route_nodes),
        "route_derivation": derivation["route_derivation"],
        "ring_walk_route": derivation.get("ring_walk_route"),
        "ring_walk_agrees": derivation.get("ring_walk_agrees"),
        "cluster_descriptor": derivation["cluster_descriptor"],
        "fabric_config": "FABRIC_1D",
        "collective_topology": "Linear",
        "live_mapping": live_mapping,
        "preopen_runner_lock_proof": lock_proof,
    }


@dataclass
class Qwen38ChainMTP:
    """What an MTP-drafting chain adds (``mtp_v2``): the MTP components, the verify / draft states and their traces
    (``traces``: the fused verify, the commit and the draft; with ``sampled`` also ``split_traces``: the verify head
    and tail with a draft of their own, the commit shared), the TAIL rows' step inputs, the chunk extension (and, on
    a ``--long-chunks`` chain, its 128-row twin ``long_chunk_extension``), the live pass loop and the cumulative
    counters.

    ``step_written`` tracks the TAIL contract: every TAIL reads the step inputs written for its step (a forced step
    writes the next prompt token; a device-token step selects the resolved argmax).
    """

    drafts: int
    anchor: str
    components: Any
    verify: mtp_v2.Qwen38TTNNVerifyState
    draft: mtp_v2.Qwen38TTNNDraftState
    step_inputs: mtp_v2.Qwen38TTNNMTPStepInputs
    chunk_extension: mtp_v2.Qwen38TTNNMTPChunkExtension | None
    long_chunk_extension: mtp_v2.Qwen38TTNNMTPChunkExtension | None = None
    traces: mtp_v2.Qwen38TTNNMTPTraces | None = None
    verify_output: mtp_v2.Qwen38TTNNVerifyOutput | None = None
    chain: mtp_v2.Qwen38TTNNMTPChain | None = None
    step_written: bool = False
    passes: int = 0
    accepted_drafts: int = 0
    capture_ms: dict[str, float] = field(default_factory=dict)
    trace_dram_bytes_per_bank: dict[str, int] = field(default_factory=dict)
    admission: dict[str, Any] = field(default_factory=dict)  # mtp_capacity_admission at the build's context and k
    dram_bytes_per_bank: dict[str, int] = field(default_factory=dict)  # measured growth: components, states, traces
    # The split verify (an --mtp --sampling server's default; QWEN38_MTP_SAMPLED=0 turns it off), captured beside
    # the fused one: sampled requests draft through it, the host deciding every pass; greedy requests keep the fused
    # traces (``Qwen38TracedChain.mtp_enter`` routes by ``decide``).
    sampled: bool = False
    split_traces: mtp_v2.Qwen38TTNNMTPTraces | None = None  # verify_head, verify_tail, their draft; commit = traces'
    split_verify_output: mtp_v2.Qwen38TTNNVerifyOutput | None = None  # the tail's readback row (the split draft's)
    head_output: mtp_v2.Qwen38TTNNVerifyHeadOutput | None = None
    # The device-decided sampled form (QWEN38_MTP_DEVICE_ACCEPT=1 beside the split verify, default off): the head, the
    # accept program and the tail in one trace with a draft of its own; the pass loop of a sampled request the device
    # acceptance admits (the tool side's decision) runs it, the others keep the host-decided split form.
    device_accept: bool = False
    sampled_traces: mtp_v2.Qwen38TTNNMTPTraces | None = None  # verify_sampled, its draft; commit = traces'
    sampled_verify_output: mtp_v2.Qwen38TTNNVerifyOutput | None = None
    # The early rows read (QWEN38_MTP_PLE_EARLY, the server's default with --mtp since 2026-09-26; the mesh opened with
    # two command queues): the fused and the device-decided pass loops read the verify lanes on the second queue and
    # look the next pass's PLE rows 0-1 up under the draft (mtp_v2.Qwen38TTNNEarlyRowsReader); the host-decided split
    # keeps its order.  False here = the one-queue form (a caller that did not ask; QWEN38_MTP_PLE_EARLY=0).
    ple_early: bool = False
    # Greedy split passes whose host decision (decide_greedy) was checked against the device lanes.  The served path
    # routes greedy requests to the fused traces, so it stays 0 there; a caller running the split form with
    # decide_greedy (a diagnostic) counts here.
    accept_checks: int = 0
    sampled_passes: int = 0
    sampled_accepted_drafts: int = 0
    sampled_draws: int = 0
    sampled_fallbacks: int = 0
    device_accept_passes: int = 0  # sampled passes the device decided (counted in the sampled_* counters too)
    device_accept_guard_deviations: int = 0  # rows of those passes whose kept minimum did not clear the shard floor

    @property
    def alignment(self) -> mtp_v2.Qwen38TTNNVerifyAlignment:
        return self.verify.alignment

    def captured_trace_ids(self) -> list[int]:
        """Every captured trace once (the commit is shared by the forms), the fused form's first."""

        ids: list[int] = []
        for traces in (self.traces, self.split_traces, self.sampled_traces):
            if traces is not None:
                ids.extend(trace_id for trace_id in traces.ids() if trace_id not in ids)
        return ids

    def record(self, pass_record: mtp_v2.Qwen38TTNNMTPPassRecord) -> mtp_v2.Qwen38TTNNMTPPassRecord:
        self.passes += 1
        self.accepted_drafts += pass_record.accepted
        statistics = {} if pass_record.decision is None else pass_record.decision.statistics
        self.accept_checks += int(statistics.get("accept_checks", 0))
        if statistics.get("sampled"):
            self.sampled_passes += 1
            self.sampled_accepted_drafts += pass_record.accepted
            self.sampled_draws += int(statistics["draws"])
            self.sampled_fallbacks += int(statistics["fallbacks"])
        if getattr(pass_record, "arithmetic", None) == mtp_v2.DEVICE_ACCEPT_ARITHMETIC:
            # The device-decided sampled pass: a sampled pass whose draws are the uniforms the program consumed
            # (u_0 .. u_a* and v: a* + 2, or k + 1 with every draft accepted); the device never falls back, the rows
            # whose guard failed are counted instead.
            self.sampled_passes += 1
            self.sampled_accepted_drafts += pass_record.accepted
            self.sampled_draws += min(pass_record.accepted + 2, self.drafts + 1)
            self.device_accept_passes += 1
            self.device_accept_guard_deviations += int(pass_record.guard_deviations or 0)
        return pass_record

    # The counters: cumulative on the chain (``/health.mtp``); a request snapshots them (``counters()``) before it
    # runs and reports what it added (``summary(since=)``), so every field of its ``qwen38.mtp`` is its own.
    COUNTERS = (
        "passes",
        "accepted_drafts",
        "accept_checks",
        "sampled_passes",
        "sampled_accepted_drafts",
        "sampled_draws",
        "sampled_fallbacks",
        "device_accept_passes",
        "device_accept_guard_deviations",
    )

    def counters(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.COUNTERS}

    def summary(self, *, since: Mapping[str, int] | None = None) -> dict[str, Any]:
        """The ``qwen38.mtp`` object: k, anchor, the switch, passes, accepted drafts and tokens per pass (``a + 1``
        per pass); with the switch on also the split form's counters (``accept_checks``, the greedy split passes
        decided on the host: 0 on the served path, which runs greedy requests through the fused verify; the sampled
        passes with their accepted drafts, tokens per pass, draws and fallbacks).  Cumulative since the chain opened,
        or, with ``since`` a ``counters()`` snapshot, the counts added after it: a request's response reports every
        field over that request alone."""

        counts = self.counters()
        if since is not None:
            counts = {name: value - since[name] for name, value in counts.items()}
        passes, accepted_drafts = counts["passes"], counts["accepted_drafts"]
        summary = {
            "k": self.drafts,
            "anchor": self.anchor,
            "sampled": self.sampled,
            "passes": passes,
            "accepted_drafts": accepted_drafts,
            "tokens_per_pass": None if not passes else round((passes + accepted_drafts) / passes, 4),
        }
        if self.sampled:
            sampled_passes, sampled_accepted = counts["sampled_passes"], counts["sampled_accepted_drafts"]
            summary.update(
                accept_checks=counts["accept_checks"],
                sampled_passes=sampled_passes,
                sampled_accepted_drafts=sampled_accepted,
                sampled_tokens_per_pass=(
                    None if not sampled_passes else round((sampled_passes + sampled_accepted) / sampled_passes, 4)
                ),
                sampled_draws=counts["sampled_draws"],
                sampled_fallbacks=counts["sampled_fallbacks"],
                device_accept=self.device_accept,
                device_accept_passes=counts["device_accept_passes"],
                device_accept_guard_deviations=counts["device_accept_guard_deviations"],
            )
        return summary


@dataclass
class Qwen38TracedChain:
    """The runner's single-trace chain kept open for serving: 8 decode traces, the persistent token row, the PLE
    row, (``chunked_prefill``) the chunk state with its one chunk trace captured after the decode traces,
    (``sampling``) the candidate-row epilogue with its readback row, and (``mtp``) the MTP drafting extension."""

    construction: Qwen38LiveDecodeConstruction
    built_target: Any
    state: Any
    prepared: Any
    token_row_io: Any
    ple_row_mapper: Any
    head_trace_ids: list[int]
    tail_trace_ids: list[int]
    captures: list[Any]
    trace_candidates: list[Any]
    trace_token_rows: list[Any]
    capture_ms: float
    program_cache_entries: int
    open_seconds: float
    misses_forbidden: bool = True
    chunk_state: Any = None
    chunk_trace_id: int | None = None
    chunk_capture_ms: float = 0.0
    # (``long_chunks``) the 128-row chunk state beside the 32-row one and its trace, captured after the chunk trace;
    # the driver runs the 128-row chunks first, then the 32-row chunks and the padded tail.
    long_chunk_state: Any = None
    long_chunk_trace_id: int | None = None
    long_chunk_capture_ms: float = 0.0
    # (``slab_rows``) the slab chunk state beside the 32-row one and its trace, captured after the long chunk trace;
    # the driver runs the slabs first, then the 128-row chunks, then the 32-row chunks and the padded tail.
    slab_state: Any = None
    slab_trace_id: int | None = None
    slab_capture_ms: float = 0.0
    # The GDN state re-anchor the chunk trace was captured with (every chunk replay commits through it).
    chunk_gdn_step_anchor: bool = False
    # The allocation tracker verified every trace once after the captures; per-prefill re-verification (140-290 ms
    # in the full-model gate) is a diagnostic switch.
    verify_each_prefill: bool = False
    sampling: sampling_step.Qwen38SamplingChainExtension | None = None
    mtp: Qwen38ChainMTP | None = None
    # (QWEN38_MTP_DRAFTS_PER_REQUEST) every drafting chain captured at open by draft count, ``mtp`` (the default)
    # included; the others share the MTP layer's generic state, the step inputs and the chunk extensions with it and
    # own their verify window, draft state and traces.  A request binds one as ``mtp`` for its duration.
    mtp_chains: dict[int, Qwen38ChainMTP] = field(default_factory=dict)
    # The prompt-end snapshot buffers (model.allocate_generic_snapshot over the generic state and the MTP alignment
    # layer's), allocated before the warm pass; the session captures and restores through the two methods below.
    snapshot: Any = None
    closed: bool = field(default=False, repr=False)

    @property
    def mesh(self) -> Any:
        return self.construction.builder.mesh_device

    @property
    def allocated_context(self) -> int:
        """The build's allocated context: the KV caches, RoPE tables and position constants are sized by it."""

        return self.built_target.model.allocated_context

    # -- the per-step primitives the session calls (the runner's loop, one call each) --------

    def write_token_row(self, token_id: int) -> None:
        ttnn.copy_host_to_device_tensor(resident_decode.device_token_row(self.mesh, token_id), self.token_row_io)

    def drafting_chains(self) -> list[Qwen38ChainMTP]:
        """Every drafting chain this chain captured (``mtp_chains`` when the per-request form opened them, else the
        default alone), the default first."""

        if self.mtp is None:
            return []
        chains = [chain_mtp for _drafts, chain_mtp in sorted(self.mtp_chains.items()) if chain_mtp is not self.mtp]
        return [self.mtp] + chains

    def write_mtp_next_token(self, next_token_id: int | None) -> None:
        """MTP chains: the TAIL's MTP row consumes the token at P + 1: this one, or None for the step's own argmax."""

        self.mtp.step_inputs.write_next_token(next_token_id)
        self.mtp.step_written = True

    def execute_head(self, residue: int) -> None:
        if self.mtp is not None and not self.mtp.step_written:
            self.write_mtp_next_token(None)  # a device-token step: the MTP row takes the resolved argmax
        ttnn._ttnn_execute_trace(self.mesh, self.head_trace_ids[residue], cq_id=0, blocking=False)

    def execute_tail(self, residue: int) -> None:
        ttnn._ttnn_execute_trace(self.mesh, self.tail_trace_ids[residue], cq_id=0, blocking=False)
        if self.mtp is not None:
            self.mtp.step_written = False

    def refresh_ple_row(self, token_id: int, context: tuple[int, int] | None) -> tuple[int, int]:
        """Scalar n-gram hash of (x_t, context), 16 PLE rows by pread, 5,120 B ROW_MAJOR write behind HEAD(t)."""

        ple_owner = self.built_target.model.layers[1].ple
        payload, next_context = ple_owner.resident_lookup.lookup_token(token_id, context)
        host_row, _row = resident_decode.ple_host_row(self.ple_row_mapper, payload)
        ttnn.copy_host_to_device_tensor(host_row, self.prepared.ple.embedding_sharded)
        return next_context

    def read_token_row_nonblocking(self) -> Any:
        return ttnn.from_device(ttnn.get_device_tensors(self.token_row_io)[0], blocking=False)

    def record_event(self) -> Any:
        return ttnn.record_event(self.mesh, cq_id=0)

    @staticmethod
    def event_synchronize(event: Any) -> None:
        ttnn.event_synchronize(event)

    @staticmethod
    def pending_value(pending: Any) -> int:
        return resident_decode.token_row_host_value(ttnn.to_torch(pending))

    def read_token_row(self) -> int:
        return resident_decode.token_row_value(self.token_row_io)

    def position(self) -> int:
        return self.state.position.read()

    def loop_guard(self):
        """The runner's replay-loop guard: only the loop's own host I/O stays callable, a synchronize fails."""

        return resident_decode.forbid_trace_body_host_io_and_sync(
            phase="chat request device loop", allowed=resident_decode.REPLAY_LOOP_HOST_IO_CALLS
        )

    def reset_and_seed(self, token_id: int) -> None:
        """In-place state reset to position 0 and the seed token in the persistent row (the runner's prologue)."""

        self.built_target.model.reset_generic_state_inplace(self.state)
        if self.mtp is not None:
            self.mtp.alignment.layer.reset_generic_state_inplace(self.mtp.alignment.generic_state)
            self.mtp.step_written = False
        self.write_token_row(token_id)
        ttnn.synchronize_device(self.mesh)
        position = self.state.position.read()
        if position != 0:
            raise Qwen38ChatChainError(f"reset left the device position at {position}, expected 0")
        resident_decode.require_token_row_holds(
            self.token_row_io, resident_decode.host_token_row(token_id), label="reset seed token row"
        )

    def capture_prompt_snapshot(self, position: int) -> None:
        """The generic state's recurrent buffers into the resident snapshot: device copies queued behind the TAIL
        whose state they record (nothing allocated, no synchronize: callable under the loop guard)."""

        self.built_target.model.capture_generic_snapshot(self.state, self.snapshot, position=position)

    def restore_prompt_snapshot(self) -> None:
        """The snapshot back into the generic state (its position and GDN phases included), then the device
        position read back against it."""

        self.built_target.model.restore_generic_snapshot(self.snapshot, self.state)
        if self.mtp is not None:
            self.mtp.step_written = False
        ttnn.synchronize_device(self.mesh)
        position = self.state.position.read()
        if position != self.snapshot.position:
            raise Qwen38ChatChainError(
                f"restore left the device position at {position}, expected {self.snapshot.position}"
            )

    def chunk_prefill(
        self,
        token_ids: Sequence[int],
        *,
        start_position: int,
        ple_context: tuple[int, int] | None,
        forced_step: Callable[[int, tuple[int, int] | None], tuple[int, int] | None],
        following_token: int | None = None,
        should_stop: Callable[[], str | None] | None = None,
        between_chunks: Callable[[int, int], None] | None = None,
        event_rows: int | None = None,
        positions=None,
        features=None,
    ) -> Qwen38PrefillResult:
        """The chunk driver on this chain (the full-model gate's prefill path): alignment steps through
        ``forced_step``, the seed, chunk replays (``should_stop`` polled at their event syncs), the padded tail,
        the hand-off.  Not under the loop guard.  ``following_token`` is the MTP layer's token at the last
        prefilled position (MTP chains).  ``between_chunks`` / ``event_rows``: the driver's hooks (the lanes
        admission runs the decoding lanes' passes between chunk groups, one event per 128 rows).  ``positions`` /
        ``features`` are an image prompt's rotary positions and feature rows (``Qwen38ChunkPrefill.run``)."""

        if self.chunk_trace_id is None:
            raise Qwen38ChatChainError("the chain was opened without the chunk trace")
        return Qwen38ChunkPrefill(
            self.built_target.model,
            self.mesh,
            self.state,
            self.chunk_state,
            self.chunk_trace_id,
            forced_step=forced_step,
            verify_allocations=self.verify_each_prefill,
            gdn_step_anchor=self.chunk_gdn_step_anchor,
            long_chunk_state=self.long_chunk_state,
            long_chunk_trace_id=self.long_chunk_trace_id,
            mtp=None if self.mtp is None else self.mtp.chunk_extension,
            slab_state=self.slab_state,
            slab_trace_id=self.slab_trace_id,
            long_mtp=None if self.mtp is None else self.mtp.long_chunk_extension,
            **({} if event_rows is None else {"event_rows": event_rows}),
        ).run(
            token_ids,
            start_position=start_position,
            ple_context=ple_context,
            following_token=following_token,
            should_stop=should_stop,
            positions=positions,
            features=features,
            **({} if between_chunks is None else {"between_chunks": between_chunks}),
        )

    # -- the MTP pass loop primitives (mtp chains) ---------------------------------------------

    def mtp_pass_fits(self, position: int) -> bool:
        return mtp_v2.verify_pass_fits(position, self.built_target.model.allocated_context)

    def _replay(self, trace_id: int) -> None:
        ttnn._ttnn_execute_trace(self.mesh, trace_id, cq_id=0, blocking=True)

    def _enqueue(self, trace_id: int) -> None:
        ttnn._ttnn_execute_trace(self.mesh, trace_id, cq_id=0, blocking=False)

    def mtp_enter(
        self,
        first_token: int,
        ple_context: tuple[int, int] | None,
        *,
        decide: Callable[..., Any] | None = None,
        before_verify_sampled: Callable[[list[int]], Any] | None = None,
        record_candidate_rows: bool = False,
    ) -> mtp_v2.Qwen38TTNNMTPPassRecord:
        """Eager switch into verify mode at the device position (the host's committed count), then the bootstrap
        pass whose row 0 is ``first_token`` (placeholder drafts).  Returns the pass record.  The pass loop's form:
        neither hook (a greedy request) runs the fused verify traces, the device's verdict inside the body (no head
        readback, no host decision: the pinned greedy stream); ``decide`` (a sampled request's acceptance) runs the
        split traces, the host deciding between the head and the tail (``mtp_sampled`` chains only);
        ``before_verify_sampled`` (a sampled request the device acceptance admits: the callable writes the pass's
        uniforms to the accept program's constants) runs the device-decided traces (``mtp_device_accept`` chains
        only), ``record_candidate_rows`` putting the rows the device decided on into every pass record."""

        mtp = self.mtp
        model = self.built_target.model
        if mtp.chain is not None:
            raise Qwen38ChatChainError("the MTP pass loop is already active")
        if decide is not None and before_verify_sampled is not None:
            raise Qwen38ChatChainError("a pass loop decides on the host or on the device, not both")
        if decide is not None and (not mtp.sampled or mtp.split_traces is None):
            raise Qwen38ChatChainError(
                "a host decision needs the split verify (the chain was opened without mtp_sampled)"
            )
        if before_verify_sampled is not None and mtp.sampled_traces is None:
            raise Qwen38ChatChainError(
                "the device acceptance needs its captured form (the chain was opened without mtp_device_accept)"
            )
        position = self.state.position.read()
        mtp_v2.enter_verify_mode(model, self.state, mtp.verify, position=position, ple_context=ple_context)
        if before_verify_sampled is not None:
            # The device-decided form: no head output, no host decision; the hook writes each pass's uniforms.
            traces, verify_output = mtp.sampled_traces, mtp.sampled_verify_output
            head_output, decision = None, mtp_v2.decide_greedy
        elif decide is None:
            # The fused form: no head output; decide_greedy is the chain's default, never called on this form.
            traces, verify_output, head_output, decision = mtp.traces, mtp.verify_output, None, mtp_v2.decide_greedy
        else:
            traces, verify_output = mtp.split_traces, mtp.split_verify_output
            head_output, decision = mtp.head_output, decide
        early_reader = None
        if mtp.ple_early and decide is None:
            # The fused and the device-decided forms: the verify lanes read on the second queue under the draft.
            # The host-decided split keeps today's order (its head readback blocks the host before the tail).
            early_reader = mtp_v2.Qwen38TTNNEarlyRowsReader(self.mesh, verify_output.readback, cq_id=1, main_cq_id=0)
        mtp.chain = mtp_v2.Qwen38TTNNMTPChain(
            model,
            mtp.verify,
            mtp.draft,
            traces,
            verify_output,
            replay=self._replay,
            position=position,
            enqueue=self._enqueue,
            head_output=head_output,
            decide=decision,
            before_verify_sampled=before_verify_sampled,
            record_candidate_rows=record_candidate_rows,
            early_reader=early_reader,
        )
        return mtp.record(mtp.chain.bootstrap([first_token] + [MTP_BOOTSTRAP_DRAFT_TOKEN] * mtp.drafts))

    def mtp_read_full_logits_rows(self) -> torch.Tensor:
        """The split verify's fallback: the eager gather of the head's retained verify logits, fp32 ``[rows, VOCAB]``
        (a row whose candidate guard failed is sampled over it)."""

        if self.mtp is None or self.mtp.head_output is None or self.sampling is None:
            raise Qwen38ChatChainError("the verify rows' full logits need the split verify and the candidate row")
        return self.sampling.read_full_logits_rows(self.mtp.head_output.logits)

    def mtp_step(self) -> mtp_v2.Qwen38TTNNMTPPassRecord:
        if self.mtp.chain is None:
            raise Qwen38ChatChainError("the MTP pass loop is not active")
        return self.mtp.record(self.mtp.chain.step())

    def mtp_leave(self, *, position: int, committed_rows: int) -> tuple[int, int] | None:
        """Commit ``committed_rows`` of the last pass (its commit trace) and rebuild the 1-row buffers at ``position``."""

        mtp = self.mtp
        if mtp.chain is None:
            raise Qwen38ChatChainError("the MTP pass loop is not active")
        mtp.chain = None
        mtp.step_written = False
        return mtp_v2.leave_verify_mode(
            self.built_target.model,
            self.state,
            mtp.verify,
            position=position,
            committed_rows=committed_rows,
            commit=lambda: self._replay(mtp.traces.commit),
        )

    # -- open / close ------------------------------------------------------------------------

    @classmethod
    def open(
        cls,
        construction: Qwen38LiveDecodeConstruction,
        *,
        marker: Marker,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        chunked_prefill: bool = True,
        sampling: bool = False,
        warm_hook: Callable[[Qwen38TracedChain], None] | None = None,
        chunk_gdn_step_anchor: bool = False,
        long_chunks: bool = False,
        mtp: int | None = None,
        mtp_gdn_anchor: str = "off",
        device_sampler: bool = False,
        slab_rows: int | None = None,
        mtp_sampled: bool = False,
        mtp_device_accept: bool = False,
        mtp_moe_rows: int | None = None,
        mtp_alternates: Sequence[int] = (),
        mtp_ple_early: bool = False,
    ) -> Qwen38TracedChain:
        """Target build, generic state (+ chunk state), warm pass (+ one eager chunk and both hand-off forms), miss
        guard, 8 decode captures (+ the chunk capture): the runner's chain prologue and the full-model gate's order.

        The chunk state is allocated before any capture (the decode traces bake every address in) and the chunk
        trace is captured after the decode traces on the same generic state, with ``chunk_gdn_step_anchor`` (the
        GDN state re-anchor) baked into the warm chunk and the capture.  ``sampling`` (off by default: the
        greedy loop keeps its measured period) adds the candidate-row epilogue to every TAIL (its constants are
        allocated before the warm pass, its programs compile in the warm pass, its row is checked there against
        torch.topk of the eager full gather); ``device_sampler`` (needs ``sampling``) adds the on-device sampler
        after the row, so TAIL writes the sampled token itself (``ttnn/device_sampler.py``; its constants are
        allocated with the row's, its programs compile in the warm pass, checked there against the host reference
        and the greedy row).  ``warm_hook`` runs after the warm pass, misses still allowed, on
        the open chain: a caller with an eager path of its own (the long-context chain's hidden windows) compiles
        its programs there.  ``mtp`` (off by default) builds the MTP components and allocates the verify / draft
        states, the TAIL step inputs and the chunk extension (with ``long_chunks`` its 128-row twin too) before the
        warm pass, adds the MTP layer's row to every TAIL and its rows to the chunk bodies, warms the pass loop and
        both mode switches at every position residue, and captures the verify, commit and draft traces after the
        chunk traces.  ``mtp_sampled`` (needs
        ``mtp`` and ``sampling``) allocates the split verify's buffers beside the verify state, warms the split
        form in every round (the head, the greedy decision checked against the device lanes, the rows candidates
        against the eager rows gather, the tail; every op of the fused body runs in the head or the tail on tensors
        of the same specs, so the fused capture needs no round of its own) and captures the verify head and tail,
        with a draft of their own, after the fused verify, commit and draft (the commit is shared): greedy requests
        run the fused traces, sampled requests the split ones (``mtp_enter``).  ``mtp_device_accept`` (needs
        ``mtp_sampled``; the sampling extension's ``device_accept.constants``, the sampler tail's built with rows k + 1
        from ``QWEN38_MTP_DEVICE_ACCEPT``) warms and captures the device-decided sampled form beside them (the head,
        the accept program and the tail in one trace, with a draft of its own): the warm round runs that very
        sequence under the extension's warm policy and checks the program's statistics row against the host reference.
        ``mtp_ple_early`` (``QWEN38_MTP_PLE_EARLY``; the mesh opened with two command queues) makes the fused and the
        device-decided pass loops read the verify lanes on the second queue and look the next pass's PLE rows 0-1 up
        under the draft (``mtp_enter`` builds the reader; the host-decided split keeps its order).
        """

        if type(chunk_gdn_step_anchor) is not bool:
            raise ValueError(f"chunk_gdn_step_anchor must be a bool, got {chunk_gdn_step_anchor!r}")
        if type(long_chunks) is not bool:
            raise ValueError(f"long_chunks must be a bool, got {long_chunks!r}")
        if type(device_sampler) is not bool:
            raise ValueError(f"device_sampler must be a bool, got {device_sampler!r}")
        if device_sampler and not sampling:
            raise ValueError("device_sampler needs sampling: the composite reads the candidate row")
        if long_chunks and (not chunked_prefill or chunk_gdn_step_anchor):
            raise ValueError("long chunks need the chunked prefill and run without the GDN step anchor")
        if mtp is not None and mtp not in MTP_DRAFTS:
            raise ValueError(f"mtp drafts must be one of {MTP_DRAFTS} or None, got {mtp!r}")
        if mtp_gdn_anchor not in MTP_GDN_ANCHORS:
            raise ValueError(f"mtp_gdn_anchor must be one of {MTP_GDN_ANCHORS}, got {mtp_gdn_anchor!r}")
        if slab_rows is not None and (not is_slab_rows(slab_rows) or not long_chunks):
            raise ValueError(f"a prefill slab needs a slab row count and the long chunks, got {slab_rows!r}")
        # A prefill slab with MTP drafting runs the MTP layer's rows through the 128-row twin's slab form (2026-09-26:
        # the slab as 128-row slices inside the slab body); the slab already requires the long chunks, so the twin exists.
        if type(mtp_sampled) is not bool:
            raise ValueError(f"mtp_sampled must be a bool, got {mtp_sampled!r}")
        if mtp_sampled and (mtp is None or not sampling):
            raise ValueError("mtp_sampled needs mtp and sampling: the split verify's candidates epilogue")
        if type(mtp_device_accept) is not bool:
            raise ValueError(f"mtp_device_accept must be a bool, got {mtp_device_accept!r}")
        if mtp_device_accept and not mtp_sampled:
            raise ValueError("mtp_device_accept needs mtp_sampled: the device decides the split verify's pass")
        if type(mtp_ple_early) is not bool:
            raise ValueError(f"mtp_ple_early must be a bool, got {mtp_ple_early!r}")
        if mtp_ple_early and mtp is None:
            raise ValueError("mtp_ple_early needs mtp: the early rows are the MTP pass loop's")
        started_ns = clock_ns()
        runtime_surface = resident_decode.b5b_runtime_surface()
        if runtime_surface["nonblocking_read"] != "ttnn.from_device(local, blocking=False)":
            raise Qwen38ChatChainError(
                f"the binary offers {runtime_surface['nonblocking_read']!r}, expected ttnn.from_device(blocking=False)"
            )
        if not TRACE_ALLOC_TRACKING:
            raise Qwen38ChatChainError("TT_METAL_TRACE_ALLOC_TRACKING=1 is required for the trace chain")
        builder = construction.builder
        mesh = builder.mesh_device
        set_misses_allowed = getattr(mesh, "set_program_cache_misses_allowed", None)
        if not callable(set_misses_allowed):
            raise Qwen38ChatChainError("MeshDevice.set_program_cache_misses_allowed is required")
        if RESIDUE_CLASSES != gdn_module.CONV_KERNEL_SIZE:
            raise Qwen38ChatChainError(
                f"residue classes {RESIDUE_CLASSES} vs GDN conv ring length {gdn_module.CONV_KERNEL_SIZE}"
            )

        def synchronize() -> None:
            ttnn.synchronize_device(mesh)

        def dram_allocated_per_bank() -> int:
            return int(ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank)

        def dram_free_view() -> dict[str, int]:
            # the mesh allocator's DRAM view (one virtual allocator, every card alike): what the admission reads
            view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
            return {
                "num_banks": int(view.num_banks),
                "free_bytes_per_bank": int(view.total_bytes_free_per_bank),
                "largest_contiguous_bytes_free_per_bank": int(view.largest_contiguous_bytes_free_per_bank),
            }

        if slab_rows is not None:
            builder.enable_prefill_slab(slab_rows)  # the slab's dense-linear residents, behind the DRAM admission
        marker("before-chat-target-build")
        built_target = builder.build_target()
        marker("after-chat-target-build")
        synchronize()
        resident_owner = builder.expert_streamer
        if len(built_target.components.layers) != 48 or built_target.components.expert_streamer is not resident_owner:
            raise Qwen38ChatChainError(f"target graph has {len(built_target.components.layers)} layers, expected 48")
        if resident_owner.cache_load_success_count != resident_decode.EXPECTED_CACHE_LOADS:
            raise Qwen38ChatChainError(
                f"resident expert pairs loaded {resident_owner.cache_load_success_count}, "
                f"expected {resident_decode.EXPECTED_CACHE_LOADS}"
            )
        model = built_target.model
        lm_head = model.model_io.lm_head
        mtp_components = None
        mtp_dram_bytes_per_bank: dict[str, int] = {}
        if mtp is not None:
            # The admission on the live allocator: the free bytes per bank after the resident weights, less what the
            # resident build still allocates (its context state, its traces), against the MTP pair, state and the
            # growth estimate for this k and the verify forms; the measured growth of the MTP build, its states and
            # its traces is checked against that estimate.  The verify forms the open captures (one table, the switch)
            # are counted in the estimate and named in the record.
            forms = mtp_verify_forms(mtp_sampled, mtp_device_accept)
            mtp_admission = mtp_capacity_admission(
                model.allocated_context,
                drafts=mtp,
                ring_size=builder.identity.ring_size,
                live=dram_free_view(),
                verify_forms=len(forms),
                long_chunks=long_chunks,
                slab_rows=slab_rows,  # the twin's slab form under --prefill-slab: the default chain's
                gdn_rows_scan=fused_module.enabled(gdn_rows_scan_module.NAME),
                moe_rows=mtp_moe_rows,
            )
            mtp_admission["verify_forms_captured"] = list(forms)
            # the record at the decision, admitted or not (READY carries it again; a refused or crashed open has only
            # this line): the live view, what the resident build still takes, what the MTP chain needs, the verdict
            print(
                json.dumps({"utc": utc_now(), "event": "mtp_admission_live", **mtp_admission}, sort_keys=True),
                flush=True,
            )
            if not mtp_admission["fits"]:
                raise Qwen38ChatChainError(
                    f"MTP drafting does not fit at allocated context {model.allocated_context}: {mtp_admission}"
                )
            marker("before-chat-mtp-build")
            allocated_before_mtp_build = dram_allocated_per_bank()
            mtp_components = builder.build_mtp_components()
            synchronize()
            mtp_dram_bytes_per_bank["components"] = dram_allocated_per_bank() - allocated_before_mtp_build
            if resident_owner.cache_load_success_count != MTP_EXPECTED_CACHE_LOADS:
                raise Qwen38ChatChainError(
                    f"resident expert pairs loaded with the MTP layer {resident_owner.cache_load_success_count}, "
                    f"expected {MTP_EXPECTED_CACHE_LOADS}"
                )
            marker("after-chat-mtp-build")
        state = model.allocate_generic_state()
        chunk_state = None
        long_chunk_state = None
        if chunked_prefill:
            model.reset_generic_state_inplace(state)
            chunk_state = model.allocate_chunk_state(state)
            model.reset_chunk_state_inplace(state, chunk_state)
            if long_chunks:
                long_chunk_state = model.allocate_chunk_state(state, rows=LONG_CHUNK_ROWS, base=chunk_state)
                model.reset_chunk_state_inplace(state, long_chunk_state)
        slab_state = None
        if slab_rows is not None:
            slab_state = model.allocate_chunk_state(state, rows=slab_rows, base=chunk_state)
            model.reset_chunk_state_inplace(state, slab_state)
        # The MTP states sit beside the generic and chunk states, before any capture: every trace bakes their
        # addresses in (the verify / draft states, the TAIL step inputs, the chunk extension).  The split verify's
        # candidates epilogue rebases by the sampling chain's constants, so with the switch on the extension is
        # built first (the default order is unchanged otherwise).
        chain_mtp = None
        sampling_extension = None
        if mtp_sampled:
            sampling_extension = sampling_step.Qwen38SamplingChainExtension(
                lm_head,
                mesh,
                device_sampler=device_sampler,
                device_accept=sampling_step.device_acceptance_for_chain(mesh, lm_head.mesh_contract, drafts=mtp),
            )
        if mtp is not None:
            allocated_before_mtp_states = dram_allocated_per_bank()
            verify = mtp_v2.allocate_verify_state(
                model,
                state,
                drafts=mtp,
                moe_rows=mtp_moe_rows,  # QWEN38_MTP_MOE_ROWS (the server) or None = moe_rows_for
                mtp_components=mtp_components,
                gdn_step_anchor_layers=MTP_GDN_ANCHOR_LAYERS[mtp_gdn_anchor],
                candidates_constants=None if sampling_extension is None else sampling_extension.constants,
            )
            chain_mtp = Qwen38ChainMTP(
                drafts=mtp,
                anchor=mtp_gdn_anchor,
                components=mtp_components,
                verify=verify,
                draft=mtp_v2.allocate_draft_state(model, verify),
                step_inputs=mtp_v2.Qwen38TTNNMTPStepInputs.allocate(mesh, model.mesh_contract),
                chunk_extension=(
                    None
                    if chunk_state is None
                    else mtp_v2.Qwen38TTNNMTPChunkExtension.allocate(model, verify, chunk_state)
                ),
                admission=mtp_admission,
                dram_bytes_per_bank=mtp_dram_bytes_per_bank,
                sampled=mtp_sampled,
                device_accept=mtp_device_accept,
                ple_early=mtp_ple_early,
            )
            if long_chunk_state is not None:
                # The 128-row twin runs the MTP layer's rows inside the 128-row chunk body (base= the 32-row extension,
                # the 128-row chunk state's shared combine buffer); it is part of the states term measured below.
                chain_mtp.long_chunk_extension = mtp_v2.Qwen38TTNNMTPChunkExtension.allocate(
                    model, verify, long_chunk_state, base=chain_mtp.chunk_extension, slab_rows=slab_rows
                )
            synchronize()
            mtp_dram_bytes_per_bank["states"] = dram_allocated_per_bank() - allocated_before_mtp_states
        # The device acceptance is the sampling extension's (built from QWEN38_MTP_DEVICE_ACCEPT at its construction,
        # tools/qwen38_mtp_device_accept.py): its constants are the accept program's (the sampler tail's, rows = k + 1);
        # the switch the caller resolved and the extension's build must agree, one switch name.
        device_acceptance = None if sampling_extension is None else getattr(sampling_extension, "device_accept", None)
        if (device_acceptance is not None) != mtp_device_accept:
            raise Qwen38ChatChainError(
                f"mtp_device_accept={mtp_device_accept} but the sampling extension "
                f"{'built' if device_acceptance is not None else 'did not build'} its device acceptance "
                "(QWEN38_MTP_DEVICE_ACCEPT decides both)"
            )
        accept_constants = None if device_acceptance is None else device_acceptance.constants
        # The other drafting chains (mtp_alternates: QWEN38_MTP_DRAFTS_PER_REQUEST): admitted per k without the
        # shared components, their states beside the default chain's BEFORE any capture (every trace bakes its
        # addresses), the MTP layer's generic state, the step inputs and the chunk extensions the default chain's
        # (one committed history, one decode tail; the sharing checked by identity), their own verify window, draft
        # and traces.  The device acceptance's constants are one k's (rows = k + 1), so its form admits no alternate.
        mtp_chains: dict[int, Qwen38ChainMTP] = {}
        if mtp_alternates:
            if chain_mtp is None:
                raise Qwen38ChatChainError("mtp_alternates need the default drafting chain (mtp)")
            if mtp_device_accept:
                raise Qwen38ChatChainError(
                    "mtp_alternates and the device acceptance do not combine: its accept constants are one k's"
                )
            alternates = tuple(mtp_alternates)
            if (
                len(set(alternates)) != len(alternates)
                or mtp in alternates
                or any(count not in mtp_v2.SUPPORTED_DRAFTS for count in alternates)
            ):
                raise Qwen38ChatChainError(
                    f"mtp_alternates must be distinct supported draft counts other than {mtp}, got {alternates}"
                )
            mtp_chains[mtp] = chain_mtp
            for count in alternates:
                alternate_admission = mtp_capacity_admission(
                    model.allocated_context,
                    drafts=count,
                    live=dram_free_view(),
                    verify_forms=len(forms),
                    long_chunks=False,  # the 128-row twin is the default chain's (shared): counted once, by its owner
                    components_shared=True,
                    # the fold's prefix states are per chain (every chain allocates its own GDN rows states): its k + 1
                    gdn_rows_scan=fused_module.enabled(gdn_rows_scan_module.NAME),
                )
                alternate_admission["verify_forms_captured"] = list(forms)
                print(
                    json.dumps(
                        {"utc": utc_now(), "event": "mtp_admission_live", **alternate_admission}, sort_keys=True
                    ),
                    flush=True,
                )
                if not alternate_admission["fits"]:
                    raise Qwen38ChatChainError(
                        f"the k={count} drafting chain does not fit beside the k={mtp} chain at allocated context "
                        f"{model.allocated_context}: {alternate_admission}"
                    )
                allocated_before_alternate = dram_allocated_per_bank()
                alternate_verify = mtp_v2.allocate_verify_state(
                    model,
                    state,
                    drafts=count,
                    mtp_components=mtp_components,
                    gdn_step_anchor_layers=MTP_GDN_ANCHOR_LAYERS[mtp_gdn_anchor],
                    candidates_constants=None if sampling_extension is None else sampling_extension.constants,
                    alignment_generic_state=chain_mtp.alignment.generic_state,
                )
                mtp_v2.validate_verify_state_sharing(model, alternate_verify, chain_mtp.verify)
                alternate_dram: dict[str, int] = {"components": 0}
                mtp_chains[count] = Qwen38ChainMTP(
                    drafts=count,
                    anchor=mtp_gdn_anchor,
                    components=mtp_components,
                    verify=alternate_verify,
                    draft=mtp_v2.allocate_draft_state(model, alternate_verify),
                    # shared: the decode TAIL trace bakes the default chain's step inputs (the hand-off token at
                    # P + 1) and its chunk extensions seed the one alignment history; a request bound to this chain
                    # writes through the same objects
                    step_inputs=chain_mtp.step_inputs,
                    chunk_extension=chain_mtp.chunk_extension,
                    long_chunk_extension=chain_mtp.long_chunk_extension,
                    admission=alternate_admission,
                    dram_bytes_per_bank=alternate_dram,
                    sampled=mtp_sampled,
                    ple_early=mtp_ple_early,
                )
                synchronize()
                alternate_dram["states"] = dram_allocated_per_bank() - allocated_before_alternate
        # The prompt-end snapshot buffers beside the states, before any capture (the tracker's post-capture check
        # then sees no later allocation); with MTP the alignment layer's generic state is a 49th layer of it.
        snapshot = model.allocate_generic_snapshot(
            state,
            extra_layers=() if chain_mtp is None else ((chain_mtp.alignment.layer, chain_mtp.alignment.generic_state),),
        )
        token_row_io = model.model_io.embedding.upload_token_row(SEED_TOKEN_ID)
        prepared = model.prepare_generic_decode_inputs(SEED_TOKEN_ID, state, device_token=token_row_io)
        ple_row_mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=MESH_SHAPE, dims=(None, 3))
        synchronize()
        chain = cls(
            construction=construction,
            built_target=built_target,
            state=state,
            prepared=prepared,
            token_row_io=token_row_io,
            ple_row_mapper=ple_row_mapper,
            head_trace_ids=[],
            tail_trace_ids=[],
            captures=[],
            trace_candidates=[],
            trace_token_rows=[],
            capture_ms=0.0,
            program_cache_entries=0,
            open_seconds=0.0,
            misses_forbidden=False,
            chunk_state=chunk_state,
            sampling=(
                sampling_extension
                if sampling_extension is not None
                else (
                    sampling_step.Qwen38SamplingChainExtension(lm_head, mesh, device_sampler=device_sampler)
                    if sampling
                    else None
                )
            ),
            chunk_gdn_step_anchor=chunk_gdn_step_anchor,
            mtp=chain_mtp,
            mtp_chains=mtp_chains,
            snapshot=snapshot,
        )

        def warm_step(position: int, warm_token_id: int, ple_context, *, mtp_next: int | None):
            """One eager 1-row step: PLE row (checked against the host oracle), the body, the epilogue's ops."""

            chain.write_token_row(warm_token_id)
            if chain_mtp is not None:
                chain.write_mtp_next_token(mtp_next)
            synchronize()
            actual = state.position.read()
            if actual != position:
                raise Qwen38ChatChainError(f"warm position counter {actual} vs expected {position}")
            oracle_embedding, oracle_context = model.layers[1].ple.host_embedding.lookup(
                torch.tensor([[warm_token_id]], dtype=torch.long),
                None if ple_context is None else torch.tensor([ple_context], dtype=torch.long),
            )
            payload, ple_context = model.layers[1].ple.resident_lookup.lookup_token(warm_token_id, ple_context)
            host_row, row = resident_decode.ple_host_row(ple_row_mapper, payload)
            ttnn.copy_host_to_device_tensor(host_row, prepared.ple.embedding_sharded)
            if (
                not torch.equal(
                    row.reshape(-1).view(torch.int16), oracle_embedding.reshape(-1).contiguous().view(torch.int16)
                )
                or list(ple_context) != oracle_context.reshape(-1).tolist()
            ):
                raise Qwen38ChatChainError(f"warm position {position} resident PLE lookup differs from the host oracle")
            if chain_mtp is None:
                output = model.forward_decode_generic(prepared, state)
            else:
                head = model.forward_decode_generic_head(prepared, state)
                output = model.forward_decode_generic_tail(head, prepared, state, retain_mtp_inputs=True)
            if output.logits is None:
                raise Qwen38ChatChainError(f"warm position {position} returned no logits")
            if chain_mtp is None:
                candidates = lm_head.greedy_candidates(output.logits)
                # The plain greedy body's last op is the copy into the persistent token row, done by the resolve (its
                # ``into``).  The warm run compiles the traced programs.
                resolved_row = lm_head.resolve_greedy_on_device(candidates, into=token_row_io)
            else:
                # The MTP body: the capture's own epilogue (candidates, resolve, the MTP row, the copy), eager here.
                candidates, resolved_row, warm_row = mtp_tail_epilogue(
                    model, lm_head, chain_mtp, output, token_row_io, chain.sampling
                )
                chain_mtp.step_written = False
                if warm_row is not None:
                    ttnn.deallocate(warm_row)  # the warm's copy of the row (the extension's own warm reads its own)
            synchronize()
            actual = state.position.read()
            if actual != position + 1:
                raise Qwen38ChatChainError(f"warm position counter after step {actual} vs expected {position + 1}")
            resolved = resident_decode.require_token_row_resolves(
                resolved_row, candidates, lm_head, label=f"warm position {position} device greedy resolve"
            )
            resident_decode.require_token_row_holds(
                token_row_io, resident_decode.host_token_row(resolved), label=f"warm position {position} row copy"
            )
            if chain.sampling is not None:
                chain.sampling.warm(output.logits, token_row_io, label=f"warm position {position} candidate row")
            ttnn.deallocate(candidates.local_indices)
            ttnn.deallocate(candidates.local_values)
            ttnn.deallocate(resolved_row)
            output.release_tensors()
            return resolved, ple_context

        # Warm pass, eager, misses allowed: every kernel of the generic body
        # compiles here.  Kept gates (actual vs expected): position counter,
        # resident PLE lookup vs the host oracle, device greedy resolve vs CPU,
        # the candidate row vs torch.topk of the full gather.
        marker("before-chat-warm-pass")
        ple_context: tuple[int, int] | None = None
        warm_tokens = [resident_decode.SEQUENTIAL_TRACE_WARM_EMBEDDING_TOKEN_IDS[p % TP_SIZE] for p in range(TP_SIZE)]
        for position in range(resident_decode.SINGLE_TRACE_WARM_POSITIONS):
            # The MTP row takes the next warm token as a prompt token would, the argmax at the last position.
            mtp_next = warm_tokens[position + 1] if position + 1 < len(warm_tokens) else None
            resolved, ple_context = warm_step(position, warm_tokens[position % TP_SIZE], ple_context, mtp_next=mtp_next)
        synchronize()
        marker("after-chat-warm-pass")
        if warm_hook is not None:
            marker("before-chat-warm-hook")
            warm_hook(chain)
            synchronize()
            marker("after-chat-warm-hook")

        if chain_mtp is not None:
            # The pass loop's programs and both eager mode switches at every position residue: the seed's ring
            # slot slices are position-dependent programs and the server switches at any P.  Each round: the
            # switch in, one verify pass (its MoE rows / alignment / accept programs), the draft rows, the switch
            # out through the commit (the request's settle form), then 1-row steps to the next residue.  With the
            # split verify captured beside the fused one the rounds run the head / tail form: every op of the fused
            # body (the prologue, the embed, the 48 layers, the final mixer, the rows resolve and the accept in the
            # head; the alignment, the readback concat, the accept-tile copy and the position adds in the tail) runs
            # there on tensors of the same dtype, layout and memory config (the split's host-written scalars and
            # lanes are allocated as the fused body's), so the fused capture's programs are all compiled here.  With
            # the device acceptance every residue runs a second round, the device-decided form's: the capture's eager
            # sequence (the head, the accept program, the tail), so the accept program compiles before its capture.
            marker("before-chat-mtp-warm-pass")
            warm_fed = resident_decode.SINGLE_TRACE_WARM_POSITIONS  # positions the warm pass consumed so far

            def warm_mtp_chain(target: Qwen38ChainMTP) -> None:
                """The rounds for one drafting chain (its verify MoE rows / alignment / accept / draft / commit programs
                in every captured form at every position residue); with QWEN38_MTP_DRAFTS_PER_REQUEST every chain has its
                rounds, the default first, so no capture meets a program its warm never compiled."""

                nonlocal warm_fed, resolved, ple_context
                warm_forms = (
                    ("fused",) if not target.sampled else ("split", "sampled") if target.device_accept else ("split",)
                )
                rounds = [(residue, warm_form) for residue in range(RESIDUE_CLASSES) for warm_form in warm_forms]
                for index, (residue, warm_form) in enumerate(rounds):
                    # 1-row steps to this round's residue class (the seed's programs at every P mod 4; the first chain
                    # starts aligned, a later chain wherever the previous one's rounds left P)
                    while warm_fed % RESIDUE_CLASSES != residue:
                        resolved, ple_context = warm_step(warm_fed, resolved, ple_context, mtp_next=None)
                        warm_fed += 1
                    position = state.position.read()
                    if position != warm_fed or position % RESIDUE_CLASSES != residue:
                        raise Qwen38ChatChainError(
                            f"MTP warm round {residue} ({warm_form}) at position {position}, fed {warm_fed}"
                        )
                    mtp_v2.enter_verify_mode(model, state, target.verify, position=position, ple_context=ple_context)
                    warm_tokens_pass = [resolved] + [MTP_BOOTSTRAP_DRAFT_TOKEN] * target.drafts
                    mtp_v2.write_verify_inputs(model, target.verify, warm_tokens_pass)
                    head = None
                    if warm_form == "fused":
                        output = mtp_v2.forward_verify(model, target.verify, state, catch_up=False)
                    elif warm_form == "sampled":
                        # The device-decided form, the capture's eager sequence under the extension's warm policy and
                        # k + 1 distinct uniforms; the program's statistics row must equal the host reference on the rows
                        # the head landed, bitwise (the fusion gate's check, here in the warm).
                        accept_constants.write_policy(sampling_step.WARM_POLICY)
                        warm_uniforms = list(MTP_WARM_ACCEPT_UNIFORMS[: target.drafts + 1])
                        accept_constants.write_uniforms(warm_uniforms)
                        output = mtp_v2.forward_verify_sampled(
                            model, target.verify, state, accept_constants, catch_up=False
                        )
                        synchronize()
                        reference = mtp_accept_module.accept_reference(
                            mtp_v2.read_candidate_rows(target.verify),
                            warm_tokens_pass[1:],
                            sampling_step.WARM_POLICY,
                            warm_uniforms,
                            sentinel=ZERO_EMBEDDING_TOKEN,
                        )
                        statistics = mtp_v2.read_accept_statistics(target.verify)
                        expected = reference.statistics_row()
                        actual = torch.tensor(statistics.row, dtype=torch.float32)
                        if not torch.equal(actual.view(torch.int32), expected.view(torch.int32)):
                            raise Qwen38ChatChainError(
                                f"MTP warm round {residue} device acceptance statistics {list(statistics.row)} vs the "
                                f"host reference {expected.tolist()}"
                            )
                    else:
                        # The split form: the head, its row, the greedy decision checked against the device lanes, the rows
                        # candidates against torch.topk of the eager rows gather (the fallback's program), the tail.
                        head = mtp_v2.forward_verify_head(model, target.verify, state, catch_up=False)
                        synchronize()
                        head_readback = mtp_v2.read_verify_head(head, rows=target.verify.rows)
                        chain.sampling.warm_rows(
                            head.logits, head_readback.candidate_rows, label=f"MTP warm round {residue} verify rows"
                        )
                        mtp_v2.write_verify_decision(
                            model, target.verify, mtp_v2.decide_greedy(warm_tokens_pass, head_readback)
                        )
                        output = mtp_v2.forward_verify_tail(model, target.verify, state, head, catch_up=False)
                    mtp_v2.forward_draft(model, target.verify, target.draft, state, output)
                    synchronize()
                    if head is not None:
                        head.release_tensors()
                    # The pass row: the verify's accept row and the draft's token chain (checked to start at its t', d_1').
                    readback, _ = mtp_v2.read_pass_row(target.verify, target.draft)
                    mtp_v2.commit_verify_host(target.verify, readback.accepted)
                    output.release_tensors()
                    if state.position.read() != position + readback.accepted + 1:
                        raise Qwen38ChatChainError(
                            f"MTP warm verify pass left P = {state.position.read()}, expected "
                            f"{position + readback.accepted + 1}"
                        )
                    # The settle form: every committed row this round, the next token unconsumed in the row.
                    committed_rows = readback.accepted + 1
                    ple_context = mtp_v2.leave_verify_mode(
                        model,
                        state,
                        target.verify,
                        position=position + committed_rows,
                        committed_rows=committed_rows,
                        commit=lambda: mtp_v2.forward_commit(model, target.verify, state),
                    )
                    warm_fed += committed_rows
                    resolved = readback.argmaxes[readback.accepted]

            for target in [chain_mtp] + [
                mtp_chains[count] for count in sorted(mtp_chains) if count != chain_mtp.drafts
            ]:
                warm_mtp_chain(target)
            synchronize()
            marker("after-chat-mtp-warm-pass")

        if slab_rows is not None:
            # One eager slab from the reset state: its programs compile here, before the miss guard.
            marker("before-chat-slab-warm-pass")
            slab_extension = None if chain_mtp is None else chain_mtp.long_chunk_extension
            model.reset_generic_state_inplace(state)
            model.reset_chunk_state_inplace(state, chunk_state)
            model.reset_chunk_state_inplace(state, slab_state)
            warm_slab_tokens = list(WARM_LONG_CHUNK_TOKEN_IDS) * (slab_rows // LONG_CHUNK_ROWS)
            model.write_chunk_inputs(slab_state, warm_slab_tokens, ple_context=None)
            if slab_extension is not None:
                # With MTP the 128-row twin's slab form compiles here too (its slices' programs are the 128-row body's).
                chain_mtp.alignment.layer.reset_generic_state_inplace(chain_mtp.alignment.generic_state)
                chain_mtp.chunk_extension.reset_chunk()
                slab_extension.reset_chunk()
                slab_extension.write_slab_tokens(model, [*warm_slab_tokens[1:], warm_slab_tokens[0]])
            synchronize()
            model.forward_prefill_chunk_generic(slab_state, state, mtp=slab_extension)
            synchronize()
            actual = state.position.read()
            if actual != slab_rows:
                raise Qwen38ChatChainError(f"warm slab position counter {actual} vs expected {slab_rows}")
            marker("after-chat-slab-warm-pass")
        if long_chunks:
            # One eager 128-row chunk from the reset state: its programs compile here, before the miss guard (with
            # MTP the 128-row twin's too: the MTP layer's 128-row body, the mixer's 128-row form, the 128-row token
            # rows' embedding; a program the capture asks for that no warm compiled is a miss under the guard).
            marker("before-chat-long-chunk-warm-pass")
            long_chunk_extension = None if chain_mtp is None else chain_mtp.long_chunk_extension
            model.reset_generic_state_inplace(state)
            model.reset_chunk_state_inplace(state, chunk_state)
            model.reset_chunk_state_inplace(state, long_chunk_state)
            model.write_chunk_inputs(long_chunk_state, list(WARM_LONG_CHUNK_TOKEN_IDS), ple_context=None)
            if long_chunk_extension is not None:
                chain_mtp.alignment.layer.reset_generic_state_inplace(chain_mtp.alignment.generic_state)
                chain_mtp.chunk_extension.reset_chunk()
                long_chunk_extension.reset_chunk()
                long_chunk_extension.write_tokens(model, [*WARM_LONG_CHUNK_TOKEN_IDS[1:], WARM_LONG_CHUNK_TOKEN_IDS[0]])
            synchronize()
            model.forward_prefill_chunk_generic(long_chunk_state, state, mtp=long_chunk_extension)
            synchronize()
            actual = state.position.read()
            if actual != LONG_CHUNK_ROWS:
                raise Qwen38ChatChainError(f"warm long chunk position counter {actual} vs expected {LONG_CHUNK_ROWS}")
            marker("after-chat-long-chunk-warm-pass")
        if chunked_prefill:
            # One eager chunk from the reset state (the chunk body's programs compile here) and both hand-off
            # forms (a closed block fills the staging tile; an open block copies it and selects a non-empty ring).
            marker("before-chat-chunk-warm-pass")
            chunk_extension = None if chain_mtp is None else chain_mtp.chunk_extension
            model.reset_generic_state_inplace(state)
            model.reset_chunk_state_inplace(state, chunk_state)
            model.write_chunk_inputs(chunk_state, list(WARM_CHUNK_TOKEN_IDS), ple_context=None)
            if chunk_extension is not None:
                chain_mtp.alignment.layer.reset_generic_state_inplace(chain_mtp.alignment.generic_state)
                chunk_extension.reset_chunk()
                chunk_extension.write_tokens(model, [*WARM_CHUNK_TOKEN_IDS[1:], WARM_CHUNK_TOKEN_IDS[0]])
            synchronize()
            model.forward_prefill_chunk_generic(
                chunk_state, state, gdn_step_anchor=chunk_gdn_step_anchor, mtp=chunk_extension
            )
            synchronize()
            actual = state.position.read()
            if actual != CHUNK_ROWS:
                raise Qwen38ChatChainError(f"warm chunk position counter {actual} vs expected {CHUNK_ROWS}")
            model.finish_prefill(state, chunk_state, CHUNK_ROWS)
            model.finish_prefill(state, chunk_state, CHUNK_ROWS - 1)
            if chunk_extension is not None:
                chunk_extension.finish_chunk(model, prefilled=CHUNK_ROWS)
                chunk_extension.finish_chunk(model, prefilled=CHUNK_ROWS - 1)
            synchronize()
            actual = state.position.read()
            if actual != CHUNK_ROWS - 1:
                raise Qwen38ChatChainError(f"warm hand-off position counter {actual} vs expected {CHUNK_ROWS - 1}")
            marker("after-chat-chunk-warm-pass")

        # The snapshot round trip (its copy programs compile here; the raw-key ring copy has no other warm form):
        # capture at the warm position, read every recurrent buffer, reset the state, restore, read again: the
        # restored buffers must be the captured ones bitwise and the position the captured one.
        marker("before-chat-snapshot-warm-pass")

        def snapshot_rows() -> dict[str, list[torch.Tensor]]:
            return {
                label: [ttnn.to_torch(local) for local in ttnn.get_device_tensors(source)]
                for label, source, _ in snapshot.pairs
            }

        snapshot_position = state.position.read()
        chain.capture_prompt_snapshot(snapshot_position)
        synchronize()
        expected_rows = snapshot_rows()
        model.reset_generic_state_inplace(state)
        synchronize()
        chain.restore_prompt_snapshot()
        for label, expected_locals in snapshot_rows().items():
            for device, (expected, actual) in enumerate(zip(expected_rows[label], expected_locals, strict=True)):
                bits = torch.int16 if expected.dtype == torch.bfloat16 else torch.int32
                if not torch.equal(expected.view(bits), actual.view(bits)):
                    raise Qwen38ChatChainError(
                        f"snapshot round trip: {label} on device {device} differs after the restore at position "
                        f"{snapshot_position}: max abs {float((actual.float() - expected.float()).abs().max())}"
                    )
        marker("after-chat-snapshot-warm-pass")

        # Pre-capture reset (the in-place reset programs compile here), the
        # allocation tracker's acknowledgements of every host- or trace-written
        # buffer, then the miss guard closes.
        chain.reset_and_seed(SEED_TOKEN_ID)
        for tensor in (prepared.ple.embedding_sharded, token_row_io, state.position.scalar):
            acknowledge_corruptible(tensor)
        if chunked_prefill:
            model.reset_chunk_state_inplace(state, chunk_state)
            for tensor in (chunk_state.token_row, chunk_state.ple_rows.embedding_rows, chunk_state.accepted):
                acknowledge_corruptible(tensor)
        if chain.sampling is not None:
            chain.sampling.mark_corruptible()
        if chain_mtp is not None:
            verify = chain_mtp.verify
            for tensor in (
                chain_mtp.step_inputs.host_lane,
                chain_mtp.step_inputs.select_index,
                verify.token_row,
                verify.draft_lanes,
                verify.ple_rows.embedding_rows,
                verify.accepted,
                chain_mtp.draft.pass_row,
            ):
                acknowledge_corruptible(tensor)
            if verify.split is not None:
                for tensor in (*verify.split.host_written(), *verify.split.device_written()):
                    acknowledge_corruptible(tensor)
            if chain_mtp.chunk_extension is not None:
                chain_mtp.chunk_extension.reset_chunk()
                acknowledge_corruptible(chain_mtp.chunk_extension.token_row)
            if chain_mtp.long_chunk_extension is not None:
                chain_mtp.long_chunk_extension.reset_chunk()
                acknowledge_corruptible(chain_mtp.long_chunk_extension.token_row)
        chain.program_cache_entries = resident_decode.program_cache_count(mesh)
        if chain.program_cache_entries <= 0:
            raise Qwen38ChatChainError("program cache is empty after the warm pass")
        set_misses_allowed(False)
        chain.misses_forbidden = True

        def conv_phases() -> dict[int, int]:
            return {
                layer.layer_index: layer_state.attention.conv_phase
                for layer, layer_state in zip(model.layers, state.layers, strict=True)
                if isinstance(layer.attention, gdn_module.Qwen38TTNNGDN)
            }

        def require_ring_phases(label: str, residue: int) -> None:
            """HEAD_r advances the HEAD layers' ring phase, TAIL_r the rest: a gate around each capture."""

            marker(f"chat-capture-{label}-residue-{residue}")
            next_phase = (residue + 1) % RESIDUE_CLASSES
            head_phase = residue if label == "before-head-capture" else next_phase
            tail_phase = next_phase if label == "after-tail-capture" else residue
            actual = conv_phases()
            expected = {index: head_phase if index < GENERIC_HEAD_LAYERS else tail_phase for index in actual}
            if actual != expected:
                raise Qwen38ChatChainError(
                    f"GDN ring phases {label} residue {residue}: actual {actual} vs expected {expected}"
                )

        def capture_epilogue(trace_output: Any) -> tuple[Any, Any]:
            """TAIL's last ops: local candidates, device greedy resolve, copy into the persistent token row; a
            sampling chain's epilogue runs the same three and then the candidate row.  An MTP chain's epilogue
            runs the MTP layer's row (the retained residual, RoPE rows and position inputs, the resolved argmax
            as the fallback token) between the resolve and the row copy, then releases what the TAIL retained."""

            if chain_mtp is not None:
                candidates, trace_token_row, trace_row = mtp_tail_epilogue(
                    model, lm_head, chain_mtp, trace_output, token_row_io, chain.sampling
                )
                ttnn.deallocate(trace_output.residual)
                trace_output.rope.deallocate()
                trace_output.qsa_position.deallocate()
                trace_output.residual = trace_output.rope = trace_output.qsa_position = None
                if chain.sampling is not None:
                    chain.sampling.trace_rows.append(trace_row)
                    chain.sampling.trace_logits.append(trace_output.logits)
                return candidates, trace_token_row
            if chain.sampling is not None:
                return chain.sampling.capture_epilogue(trace_output, token_row_io)
            if trace_output.logits is None:
                raise Qwen38ChatChainError("TAIL capture returned no logits")
            candidates = lm_head.greedy_candidates(trace_output.logits)
            trace_token_row = lm_head.resolve_greedy_on_device(candidates, into=token_row_io)
            return candidates, trace_token_row

        marker("before-chat-captures")
        capture_ns = 0
        dram_before_captures = dram_allocated_per_bank()
        for phase in range(RESIDUE_CLASSES):
            capture = model.capture_decode_generic(
                prepared,
                state,
                residue=phase,
                split=True,
                guard=lambda label: resident_decode.forbid_trace_body_host_io_and_sync(phase=f"chat {label}"),
                epilogue=capture_epilogue,
                phase_observer=lambda label, residue=phase: require_ring_phases(label, residue),
                regime=resident_decode.SINGLE_TRACE_INDEXER_REGIME,
                cq_id=0,
                clock_ns=clock_ns,
                retain_mtp_inputs=chain_mtp is not None,
            )
            if (
                capture.parts != resident_decode.SINGLE_TRACE_TRACE_PARTS
                or capture.head is None
                or not capture.head.active
            ):
                raise Qwen38ChatChainError(
                    f"capture residue {phase} parts {capture.parts} did not retain a HEAD handoff"
                )
            if capture.guard_attempts:
                raise Qwen38ChatChainError(
                    f"capture residue {phase} attempted host I/O in-body: {capture.guard_attempts}"
                )
            chain.captures.append(capture)
            chain.head_trace_ids.append(
                capture.trace_ids[Qwen38TTNNGenericTraceKey("head", phase, resident_decode.SINGLE_TRACE_INDEXER_REGIME)]
            )
            chain.tail_trace_ids.append(
                capture.trace_ids[Qwen38TTNNGenericTraceKey("tail", phase, resident_decode.SINGLE_TRACE_INDEXER_REGIME)]
            )
            capture_ns += sum(capture.capture_ns.values())
            candidates, trace_token_row = capture.epilogue
            chain.trace_candidates.append(candidates)
            chain.trace_token_rows.append(trace_token_row)
            acknowledge_corruptible(candidates.local_indices)
            acknowledge_corruptible(candidates.local_values)
            if chain.sampling is not None:
                chain.sampling.mark_trace_rows_corruptible()
            synchronize()
        marker("after-chat-captures")
        chain.capture_ms = capture_ns / 1e6
        dram_after_decode = dram_allocated_per_bank()
        if chunked_prefill:
            # The chunk trace after the 8 decode traces, on the same generic state; capture records without
            # executing, so the ring phases and the position stay at 0.
            marker("before-chat-chunk-capture")
            chunk_capture_started_ns = clock_ns()
            chain.chunk_trace_id = model.capture_prefill_chunk(
                chunk_state,
                state,
                guard=lambda label: resident_decode.forbid_trace_body_host_io_and_sync(phase=f"chat {label}"),
                cq_id=0,
                mtp=None if chain_mtp is None else chain_mtp.chunk_extension,
                gdn_step_anchor=chunk_gdn_step_anchor,
            )
            chain.chunk_capture_ms = (clock_ns() - chunk_capture_started_ns) / 1e6
            synchronize()
            marker("after-chat-chunk-capture")
        dram_after_chunk = dram_allocated_per_bank()
        if long_chunks:
            marker("before-chat-long-chunk-capture")
            long_chunk_capture_started_ns = clock_ns()
            chain.long_chunk_state = long_chunk_state
            chain.long_chunk_trace_id = model.capture_prefill_chunk(
                long_chunk_state,
                state,
                guard=lambda label: resident_decode.forbid_trace_body_host_io_and_sync(phase=f"chat long {label}"),
                cq_id=0,
                mtp=None if chain_mtp is None else chain_mtp.long_chunk_extension,
            )
            chain.long_chunk_capture_ms = (clock_ns() - long_chunk_capture_started_ns) / 1e6
            synchronize()
            marker("after-chat-long-chunk-capture")
        dram_after_long_chunk = dram_allocated_per_bank()
        if slab_rows is not None:
            marker("before-chat-slab-capture")
            slab_capture_started_ns = clock_ns()
            chain.slab_state = slab_state
            chain.slab_trace_id = model.capture_prefill_chunk(
                slab_state,
                state,
                guard=lambda label: resident_decode.forbid_trace_body_host_io_and_sync(phase=f"chat slab {label}"),
                cq_id=0,
                mtp=None if chain_mtp is None else chain_mtp.long_chunk_extension,
            )
            chain.slab_capture_ms = (clock_ns() - slab_capture_started_ns) / 1e6
            synchronize()
            marker("after-chat-slab-capture")
        # Every prefill chunk trace is booked before the MTP captures: the MTP traces term of the growth record is what
        # the verify / commit / draft captures add after the last prefill capture (the 128-row trace, MTP rows
        # included, is its own term, as the 32-row chunk trace with the extension's rows always was).
        dram_after_prefill_captures = dram_allocated_per_bank()

        def capture_mtp_chain(target: Qwen38ChainMTP, dram_baseline: int) -> int:
            """One drafting chain's traces: the verify (first pass), commit and draft traces after the prefill traces,
            the fused form under both switch values (a greedy request's traces: the QWEN38_MTP_SAMPLED=0 server's
            exactly); the draft body reads the verify output's readback address, so the verify capture comes first.
            With the split verify the head, the tail and a second draft follow (each verify body allocates its readback
            row inside its own capture, so the tail lands a row of its own and the split form needs a draft captured on
            that row; the commit reads neither row and is shared), and with the device acceptance its one trace and
            draft.  Capture records without executing: the device state is unchanged.  ``dram_baseline`` is the
            allocation the chain's traces grow from (the last prefill capture's for the default chain, the previous
            chain's for an alternate); the allocation after them is returned.  The same sequence of calls for every
            chain (QWEN38_MTP_DRAFTS_PER_REQUEST: the alternates after the default, in draft-count order)."""

            def guard(label: str):
                return resident_decode.forbid_trace_body_host_io_and_sync(phase=f"chat k={target.drafts} {label}")

            capture_started_ns = clock_ns()
            verify_first, verify_output = mtp_v2.capture_verify(
                model, target.verify, state, catch_up=False, guard=guard, cq_id=0
            )
            target.capture_ms["verify_first"] = (clock_ns() - capture_started_ns) / 1e6
            capture_started_ns = clock_ns()
            commit = mtp_v2.capture_commit(model, target.verify, state, guard=guard, cq_id=0)
            target.capture_ms["commit"] = (clock_ns() - capture_started_ns) / 1e6
            capture_started_ns = clock_ns()
            draft = mtp_v2.capture_draft(model, target.verify, target.draft, state, verify_output, guard=guard, cq_id=0)
            target.capture_ms["draft"] = (clock_ns() - capture_started_ns) / 1e6
            target.traces = mtp_v2.Qwen38TTNNMTPTraces(verify_first=verify_first, draft=draft, commit=commit)
            target.verify_output = verify_output
            acknowledge_corruptible(verify_output.readback)
            synchronize()
            dram_after_fused = dram_allocated_per_bank()
            if target.sampled:
                # The split verify: the head, the tail that reads its roots, the draft that reads the tail's row.
                capture_started_ns = clock_ns()
                verify_head, head_output = mtp_v2.capture_verify_head(
                    model, target.verify, state, catch_up=False, guard=guard, cq_id=0
                )
                target.capture_ms["verify_head"] = (clock_ns() - capture_started_ns) / 1e6
                capture_started_ns = clock_ns()
                verify_tail, split_verify_output = mtp_v2.capture_verify_tail(
                    model, target.verify, state, head_output, catch_up=False, guard=guard, cq_id=0
                )
                target.capture_ms["verify_tail"] = (clock_ns() - capture_started_ns) / 1e6
                capture_started_ns = clock_ns()
                split_draft = mtp_v2.capture_draft(
                    model, target.verify, target.draft, state, split_verify_output, guard=guard, cq_id=0
                )
                target.capture_ms["split_draft"] = (clock_ns() - capture_started_ns) / 1e6
                target.split_traces = mtp_v2.Qwen38TTNNMTPTraces(
                    verify_first=None,
                    draft=split_draft,
                    commit=commit,
                    verify_head=verify_head,
                    verify_tail=verify_tail,
                )
                target.split_verify_output = split_verify_output
                target.head_output = head_output
                for tensor in (
                    split_verify_output.readback,
                    head_output.readback,
                    head_output.roots,
                    head_output.logits.tensor,
                ):
                    acknowledge_corruptible(tensor)
                synchronize()
            dram_after_split = dram_allocated_per_bank()
            if target.device_accept:
                # The device-decided sampled form: one trace (the head, the accept program, the tail), the draft on its
                # row; the statistics row it writes and the accept constants were marked with the split's buffers.
                capture_started_ns = clock_ns()
                verify_sampled, sampled_verify_output = mtp_v2.capture_verify_sampled(
                    model, target.verify, state, accept_constants, catch_up=False, guard=guard, cq_id=0
                )
                target.capture_ms["verify_sampled"] = (clock_ns() - capture_started_ns) / 1e6
                capture_started_ns = clock_ns()
                sampled_draft = mtp_v2.capture_draft(
                    model, target.verify, target.draft, state, sampled_verify_output, guard=guard, cq_id=0
                )
                target.capture_ms["sampled_draft"] = (clock_ns() - capture_started_ns) / 1e6
                target.sampled_traces = mtp_v2.Qwen38TTNNMTPTraces(
                    verify_first=None, draft=sampled_draft, commit=commit, verify_sampled=verify_sampled
                )
                target.sampled_verify_output = sampled_verify_output
                acknowledge_corruptible(sampled_verify_output.readback)
                synchronize()
            dram_after_target = dram_allocated_per_bank()
            target.trace_dram_bytes_per_bank = {
                "decode_traces": dram_after_decode - dram_before_captures,
                "chunk_trace": dram_after_chunk - dram_after_decode,
                "long_chunk_trace": dram_after_long_chunk - dram_after_chunk,
                "mtp_fused_traces": dram_after_fused - dram_baseline,
                "mtp_traces": dram_after_target - dram_baseline,
            }
            if target.sampled:
                target.trace_dram_bytes_per_bank["mtp_split_traces"] = dram_after_split - dram_after_fused
            if target.device_accept:
                target.trace_dram_bytes_per_bank["mtp_sampled_traces"] = dram_after_target - dram_after_split
            target.dram_bytes_per_bank["traces"] = target.trace_dram_bytes_per_bank["mtp_traces"]
            target.admission["measured_after_captures"] = dram_free_view()  # the allocator with the MTP chain in
            captured_forms = (
                ["fused"]
                + (["split"] if target.split_traces is not None else [])
                + (["sampled"] if target.sampled_traces is not None else [])
            )
            if captured_forms != target.admission["verify_forms_captured"]:
                raise Qwen38ChatChainError(
                    f"captured verify forms {captured_forms} vs the admission's {target.admission['verify_forms_captured']}"
                )
            mtp_growth = sum(target.dram_bytes_per_bank.values())
            if mtp_growth > target.admission["required_free_bytes_per_bank"]:
                raise Qwen38ChatChainError(
                    f"MTP DRAM growth {mtp_growth} bytes per bank {target.dram_bytes_per_bank} exceeds the admission's "
                    f"estimate {target.admission['required_free_bytes_per_bank']} "
                    f"{target.admission['mtp_growth_estimate_bytes_per_bank']} for k={target.drafts} "
                    f"(verify MoE rows {target.admission['mtp_moe_rows']}"
                    f"{' [states estimate PROVISIONAL]' if target.admission.get('mtp_states_estimate_provisional') else ''}, "
                    f"{target.admission['verify_forms']} verify form(s); the required side decided, the free side "
                    f"read {target.admission['free_bytes_source']}) at allocated context {model.allocated_context}"
                )
            return dram_after_target

        if chain_mtp is not None:
            marker("before-chat-mtp-captures")
            dram_after_previous = capture_mtp_chain(chain_mtp, dram_after_prefill_captures)
            for count in sorted(mtp_chains):
                if count != chain_mtp.drafts:
                    dram_after_previous = capture_mtp_chain(mtp_chains[count], dram_after_previous)
            marker("after-chat-mtp-captures")
        phases, position = conv_phases(), state.position.read()
        if set(phases.values()) != {0} or position != 0:
            raise Qwen38ChatChainError(f"after the captures ring phases {phases} and position {position}, expected 0")
        after_capture = resident_decode.program_cache_count(mesh)
        if after_capture != chain.program_cache_entries:
            raise Qwen38ChatChainError(
                f"program cache {after_capture} entries after capture vs {chain.program_cache_entries} before"
            )
        for trace_id in chain.trace_ids():
            TraceAllocationTracker.verify_before_replay(mesh, trace_id)
        chain.reset_and_seed(SEED_TOKEN_ID)
        chain.open_seconds = (clock_ns() - started_ns) / 1e9
        return chain

    def trace_ids(self) -> list[int]:
        """Every captured trace: 4 HEAD, 4 TAIL, then the chunk trace, the long chunk trace and the MTP traces
        when captured."""

        return (
            self.head_trace_ids
            + self.tail_trace_ids
            + [
                trace_id
                for trace_id in (self.chunk_trace_id, self.long_chunk_trace_id, self.slab_trace_id)
                if trace_id is not None
            ]
            + [trace_id for chain_mtp in self.drafting_chains() for trace_id in chain_mtp.captured_trace_ids()]
        )

    def close(self) -> None:
        """The runner's release order; skipped by the caller when a request failed mid-loop."""

        if self.closed:
            return
        mesh = self.mesh
        ttnn.synchronize_device(mesh)
        if self.misses_forbidden:
            mesh.set_program_cache_misses_allowed(True)
            self.misses_forbidden = False
        for trace_id in self.trace_ids():
            ttnn.release_trace(mesh, trace_id)
        self.head_trace_ids.clear()
        self.tail_trace_ids.clear()
        self.chunk_trace_id = None
        self.long_chunk_trace_id = None
        self.slab_trace_id = None
        for chain_mtp in self.drafting_chains():
            chain_mtp.traces = None
            chain_mtp.split_traces = None
            chain_mtp.sampled_traces = None
            for name in ("verify_output", "split_verify_output", "sampled_verify_output", "head_output"):
                output = getattr(chain_mtp, name)
                if output is not None:
                    output.release_tensors()
                    setattr(chain_mtp, name, None)
        for candidates in self.trace_candidates:
            ttnn.deallocate(candidates.local_indices)
            ttnn.deallocate(candidates.local_values)
        self.trace_candidates.clear()
        for trace_token_row in self.trace_token_rows:
            ttnn.deallocate(trace_token_row)
        self.trace_token_rows.clear()
        if self.sampling is not None:
            self.sampling.release()
        for capture in self.captures:
            if capture.active:
                capture.release_tensors()
        self.captures.clear()
        if self.prepared.active:
            self.prepared.release()
        ttnn.deallocate(self.token_row_io)
        if self.slab_state is not None:  # before the 32-row state whose histories it shares
            self.built_target.model.release_chunk_state(self.slab_state)
            self.slab_state = None
        if self.long_chunk_state is not None:  # before the 32-row state whose histories it shares
            self.built_target.model.release_chunk_state(self.long_chunk_state)
            self.long_chunk_state = None
        if self.mtp is not None:
            # The MTP states before the chunk and generic states they sit beside.
            model = self.built_target.model
            # The sharing chains first (their release leaves the shared generic state, step inputs and extensions
            # alone), the owning default last.
            for chain_mtp in self.drafting_chains():
                if chain_mtp is not self.mtp:
                    mtp_v2.release_draft_state(model, chain_mtp.verify, chain_mtp.draft)
                    mtp_v2.release_verify_state(model, chain_mtp.verify)
            if self.mtp.long_chunk_extension is not None:  # before the 32-row twin it was allocated beside
                self.mtp.long_chunk_extension.release()
            if self.mtp.chunk_extension is not None:
                self.mtp.chunk_extension.release()
            self.mtp.step_inputs.deallocate()
            mtp_v2.release_draft_state(model, self.mtp.verify, self.mtp.draft)
            mtp_v2.release_verify_state(model, self.mtp.verify)
        if self.chunk_state is not None:
            self.built_target.model.release_chunk_state(self.chunk_state)
            self.chunk_state = None
        if self.snapshot is not None:
            self.built_target.model.release_generic_snapshot(self.snapshot)
            self.snapshot = None
        self.built_target.model.release_generic_state(self.state)
        ttnn.synchronize_device(mesh)
        self.built_target.components.close_resident_experts()
        ttnn.synchronize_device(mesh)
        self.closed = True


def construct_chain(
    prepared: Any,
    mesh: Any,
    *,
    marker: Marker,
    chunked_prefill: bool = True,
    sampling: bool = False,
    bf4_stage_limit: int | None = None,
    long_chunks: bool = False,
    mtp: int | None = None,
    mtp_gdn_anchor: str = "off",
    device_sampler: bool = False,
    slab_rows: int | None = None,
    mtp_sampled: bool = False,
    mtp_device_accept: bool = False,
    mtp_moe_rows: int | None = None,
    mtp_alternates: Sequence[int] = (),
    mtp_ple_early: bool = False,
    warm_hook: Callable[[Qwen38TracedChain], None] | None = None,
) -> Qwen38TracedChain:
    """Live construction on the open mesh (missing BF4 layers converted first), then the chain prologue.
    ``warm_hook`` is the open's (after the warm pass, before any capture: the lanes server allocates its lane states
    and compiles their programs there)."""

    construction = construct_live_decode_diagnostic(
        prepared,
        mesh_device=mesh,
        collective_topology=ttnn.Topology.Linear,
        marker=marker,
        bf4_stage_limit=bf4_stage_limit,
    )
    return Qwen38TracedChain.open(
        construction,
        marker=marker,
        chunked_prefill=chunked_prefill,
        sampling=sampling,
        long_chunks=long_chunks,
        mtp=mtp,
        mtp_gdn_anchor=mtp_gdn_anchor,
        device_sampler=device_sampler,
        slab_rows=slab_rows,
        mtp_sampled=mtp_sampled,
        mtp_device_accept=mtp_device_accept,
        mtp_moe_rows=mtp_moe_rows,
        mtp_alternates=mtp_alternates,
        mtp_ple_early=mtp_ple_early,
        warm_hook=warm_hook,
    )
