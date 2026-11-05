# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Chat session over the traced HEAD/TAIL decode chain.

Decode replays ``HEAD[t mod 4]``, the host PLE row refresh and ``TAIL[t mod 4]``. Prefill is ``chunked`` (the chunk
traces, then the last prompt token teacher-forced as the first decode step) or ``teacher_forced``. Sampling and MTP
drafting are optional extensions of the same chain. ``Qwen38ChatSession`` reaches the device only through a chain
object; ``Qwen38TracedChain`` is the hardware one.
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
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_sampling as sampling_step
from models.demos.blackhole.qwen38_flash_next.tools import resident_decode
from models.demos.blackhole.qwen38_flash_next.tools.evidence_records import Marker, utc_now
from models.demos.blackhole.qwen38_flash_next.tools.hardware_profiles import ResidentHardwareProfile
from models.demos.blackhole.qwen38_flash_next.tools.live_decode_construction import (
    Qwen38LiveDecodeConstruction,
    construct_live_decode,
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

# Device positions must stay below allocated_context (RoPE lookup, KV write). CONTEXT_LIMIT is the default build's;
# a session derives its own from the chain's allocated context.
CONTEXT_HEADROOM = RESIDENT_CONTEXT_HEADROOM
CONTEXT_LIMIT = Qwen38ResidentContext().context_limit
PREFILL_EVENT_INTERVAL = 16
# Bounds an explicit max_tokens before the prompt is known; the request budget is the remaining context.
MAX_TOKENS_BOUND = max(RESIDENT_QSA_CACHE_CAPACITIES)
PREFILL_MODES = ("chunked", "teacher_forced")
DEFAULT_PREFILL_MODE = "chunked"
# Fewer rows than this after the alignment steps are teacher-forced: cheaper than the chunk hand-off. From a device
# measurement.
CHUNK_PREFILL_MIN_ROWS = 16
# The chunk warm pass embeds one token per vocabulary owner in every lane group (the decode warm pass's ids).
WARM_CHUNK_TOKEN_IDS = tuple(
    resident_decode.SEQUENTIAL_TRACE_WARM_EMBEDDING_TOKEN_IDS[index % TP_SIZE] for index in range(CHUNK_ROWS)
)
WARM_LONG_CHUNK_TOKEN_IDS = tuple(
    resident_decode.SEQUENTIAL_TRACE_WARM_EMBEDDING_TOKEN_IDS[index % TP_SIZE] for index in range(LONG_CHUNK_ROWS)
)
# Template flags of the acceptance records: they render identically only with these. The records carry their own
# system turn; the session adds none.
ENABLE_THINKING = False
PRESERVE_THINKING = True
REASONING_EFFORT = "low"
RESIDUE_CLASSES = resident_decode.SINGLE_TRACE_RESIDUE_CLASS_TRACES
SEED_TOKEN_ID = IM_START_ID
THINK_END_ID = protocol.THINK_END_ID
# MTP: the draft counts a server may open with (the QSA verify path admits at most k + 1 = 6 rows), the GDN state
# re-anchor settings (layer0: layer 0 commits through the 1-row fp32 step recurrence) and the bootstrap placeholder.
MTP_DRAFTS = (3, 4, 5)
MTP_GDN_ANCHORS = ("off", "layer0")
MTP_GDN_ANCHOR_LAYERS = {"off": (), "layer0": (0,)}
MTP_BOOTSTRAP_DRAFT_TOKEN = 0
MTP_EXPECTED_CACHE_LOADS = resident_decode.EXPECTED_CACHE_LOADS + 1
# Verify forms a chain captures per (mtp_sampled, mtp_device_accept). The server's pre-mesh refusal and the open's
# gate both read this table.
MTP_VERIFY_FORMS_BY_SWITCH = {
    (False, False): ("fused",),
    (True, False): ("fused", "split"),
    (True, True): ("fused", "split", "sampled"),
}
# Warm uniforms for the device-decided form: distinct multiples of 2**-24 (exact in fp32), so the eager pass's
# statistics row must equal the host reference bitwise.
MTP_WARM_ACCEPT_UNIFORMS = tuple(
    n * 2**-24 for n in (1_000_003, 5_000_011, 9_000_017, 13_000_019, 16_000_023, 7_000_027)
)


def mtp_verify_forms(mtp_sampled: bool, mtp_device_accept: bool = False) -> tuple[str, ...]:
    """Verify forms an MTP chain captures for the switches; the device acceptance needs the split verify."""

    if type(mtp_sampled) is not bool or type(mtp_device_accept) is not bool:
        raise ValueError(
            f"mtp_sampled and mtp_device_accept must be a bool each, got {mtp_sampled!r}, {mtp_device_accept!r}"
        )
    if mtp_device_accept and not mtp_sampled:
        raise ValueError("the device acceptance needs the split verify (mtp_sampled)")
    return MTP_VERIFY_FORMS_BY_SWITCH[(mtp_sampled, mtp_device_accept)]


# MTP chain DRAM growth per bank beyond the resident build, by the part the open measures: components (the MTP
# layer's weights beyond its BF4 pair), states (beyond its QSA state, by verify MoE rows) and traces (one verify form,
# plus a verify and a draft trace per further form). From device measurements; the admission adds the margin and the
# open checks the measured growth against the estimate.
MTP_COMPONENTS_BEYOND_PAIR_BYTES_PER_BANK = 6_231_232
MTP_STATES_BEYOND_QSA_STATE_BYTES_PER_BANK_BY_MOE_ROWS = {
    5: 8_475_520,
    # the 5-row term plus a 6-row open's measured growth over a 5-row one
    6: 8_475_520 + 744_064,
}
MTP_TRACES_BYTES_PER_BANK_ONE_VERIFY_FORM = 6_390_144
MTP_TRACES_BYTES_PER_BANK_PER_ADDITIONAL_VERIFY_FORM = 4_717_760
MTP_VERIFY_FORMS_MAX = 3
MTP_GROWTH_ESTIMATE_MARGIN_PERCENT = 10
RESIDENT_POST_BUILD_BYTES_PER_BANK_UPPER_BOUND = 64 << 20
# A --long-chunks build's 128-row chunk state and trace (off the free side) and the MTP layer's own 128-row chunk
# extension (added to the states remainder): device-measured builds without and with the long chunks.
FREE_BYTES_PER_BANK_AFTER_CAPTURES_WITHOUT_LONG_CHUNKS = 1_603_483_392
FREE_BYTES_PER_BANK_AFTER_CAPTURES_WITH_LONG_CHUNKS = 1_569_890_816
LONG_CHUNKS_BYTES_PER_BANK_AFTER_CAPTURES = (
    FREE_BYTES_PER_BANK_AFTER_CAPTURES_WITHOUT_LONG_CHUNKS - FREE_BYTES_PER_BANK_AFTER_CAPTURES_WITH_LONG_CHUNKS
)
MTP_STATES_BYTES_PER_BANK_WITHOUT_LONG_CHUNKS = 15_313_792
MTP_STATES_BYTES_PER_BANK_WITH_LONG_CHUNKS = 15_339_392
MTP_LONG_CHUNK_EXTENSION_BYTES_PER_BANK = (
    MTP_STATES_BYTES_PER_BANK_WITH_LONG_CHUNKS - MTP_STATES_BYTES_PER_BANK_WITHOUT_LONG_CHUNKS
)
# Slab form of the 128-row twin: per 128-row slice one FP32 token tile page ``[1,1,4,32]`` and, after the first
# slice, one 32-byte UINT32 position-offset page.
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
# (free, largest contiguous) bytes per bank after a non-MTP build's captures; the no-device stand-in for the live read.
RESIDENT_FREE_BYTES_PER_BANK_AFTER_CAPTURES = {
    32768: (478_238_784, 477_584_640),
    65536: (423_336_000, 422_681_856),
    131072: (313_694_272, 313_040_128),
    262144: (94_378_048, 93_723_904),
}
# The verify-rows fold (gdn_rows_scan) keeps every GDN layer's k + 1 fp32 prefix states ``[k + 1, 12, 128, 128]`` in
# DRAM for the pass, interleaved one 4 KiB tile page at a time over the banks.
GDN_LAYERS = LAYER_PATTERN.count("linear_attention")
GDN_STATE_TILE_PAGES = gdn_rows_scan_module.HEADS * (gdn_rows_scan_module.HEAD_DIM // gdn_rows_scan_module.TILE) ** 2
GDN_STATE_PAGE_BYTES = gdn_rows_scan_module.TILE * gdn_rows_scan_module.TILE * 4  # one fp32 tile


def fold_prefix_states_bytes_per_bank(drafts: int, banks: int = RESIDENT_DRAM_BANKS) -> int:
    """The verify-rows fold's prefix states per bank at ``drafts`` drafts; 0 where the fold does not attach."""

    rows = drafts + 1
    if rows > gdn_rows_scan_module.MAX_ROWS:
        return 0
    return GDN_LAYERS * -(-(rows * GDN_STATE_TILE_PAGES) // banks) * GDN_STATE_PAGE_BYTES


def resident_post_build_bytes_per_bank(allocated_context: int) -> int:
    """Resident build allocations per bank after its weights: the context-scaled state plus the context-independent
    upper bound."""

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
    gdn_rows_scan: bool = False,
    slab_rows: int | None = None,
    vision_resident_bytes_per_bank: int = 0,
    vision_peak_activation_bytes_per_bank: int = 0,
) -> dict[str, Any]:
    """Whether the MTP chain fits beside the resident build at ``allocated_context``; ``fits`` decides.

    The required side estimates the per-bank growth the open measures, with the margin over every measured remainder;
    the free side is the build's free bytes per bank after its captures. ``live`` is the mesh allocator's DRAM view
    read at ``live_point``: ``after_build`` takes the build's remaining allocations, the 128-row chunk state and the
    vision tower off it, ``after_captures`` reads it as is. Without ``live`` the measured table stands in.
    ``vision_peak_activation_bytes_per_bank`` is a required-side part: an image request allocates it after every
    resident allocation."""

    if isinstance(drafts, bool) or type(drafts) is not int or drafts not in mtp_v2.SUPPORTED_DRAFTS:
        raise ValueError(f"MTP drafts must be one of {mtp_v2.SUPPORTED_DRAFTS}, got {drafts!r}")
    if isinstance(verify_forms, bool) or type(verify_forms) is not int or not 1 <= verify_forms <= MTP_VERIFY_FORMS_MAX:
        raise ValueError(f"verify forms must be an int in [1, {MTP_VERIFY_FORMS_MAX}], got {verify_forms!r}")
    moe_rows = mtp_v2.moe_rows_for(drafts + 1)
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
        # A context below the smallest measured one uses that one's readings: conservative, a smaller build leaves more.
        measured_at = [c for c in sorted(RESIDENT_FREE_BYTES_PER_BANK_AFTER_CAPTURES) if c >= context]
        if not measured_at:
            raise ValueError(f"no free-bytes-after-captures measurement at or above {context} tokens")
        free, largest = RESIDENT_FREE_BYTES_PER_BANK_AFTER_CAPTURES[measured_at[0]]
        # The table's build had no 128-row chunk state or tower: off its free bytes and, as an upper bound, off the
        # largest contiguous block.
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
        # After the build, the resident's remaining allocations and the 128-row chunk state are still to come.
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

    estimate = {
        "components": w01_per_bank + w2_per_bank + with_margin(remainders["components_beyond_pair"]),
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
    contiguous = max(w01_per_bank, w2_per_bank, RESIDENT_MIN_CONTIGUOUS_BYTES_PER_BANK)
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
        "verify_forms": verify_forms,
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
        # the free side as read against the required side's estimate; the shortfalls name a refusal's reasons
        "decided_by": {
            "free_side": source["free_bytes_source"],
            "required_side": f"estimate k={drafts} moe_rows={moe_rows} verify_forms={verify_forms}"
            + (" gdn_rows_scan" if gdn_rows_scan else "")
            + (" vision" if vision_resident_bytes_per_bank or vision_peak_activation_bytes_per_bank else ""),
            "shortfalls": shortfalls,
        },
        "fits": not shortfalls,
    }


# The lanes' DRAM per bank (``--lanes B``), modeled from the lane families' shapes (ttnn/lanes.py, ttnn/mtp_lanes.py)
# with a measured per-lane slope correction and the growth margin on every term; the required side of
# ``lanes_capacity_admission``.
LANES_IMAGE_BYTES_PER_DEVICE_SLOPE = 13_056  # per context row: the 12 KV slabs and the 12 compressed caches
LANES_IMAGE_BYTES_PER_DEVICE_FIXED = 29_684_736  # the recurrent states, ring slots, stagings, rings, PLE slots
LANES_FIXED_BYTES_PER_DEVICE = 60_424_192  # the KV import scratch (13 layers x 4,096 rows) and the GDN lane rows qkv
LANES_MOE_ROWS_SLOPE_BYTES_PER_BANK = 295_632  # the states remainder per verify MoE row (the 5- and 32-row terms' line)
LANES_PER_LANE_SLOPE_CORRECTION_PERCENT = 7  # the measured per-lane increment over the modeled one
# The three lane traces, an upper bound: two measured per device (spread over the banks) and one measured per bank.
LANES_TRACES_MEASURED_BYTES_PER_DEVICE = (51_000_000, 47_000_000)
LANES_TRACE_MEASURED_BYTES_PER_BANK = 7_400_000
LANES_TRACES_BYTES_PER_BANK = (
    sum(LANES_TRACES_MEASURED_BYTES_PER_DEVICE) // RESIDENT_DRAM_BANKS + LANES_TRACE_MEASURED_BYTES_PER_BANK
)
LANE_KV_STAGINGS = 2  # ttnn/lanes.py KV_STAGINGS: the pagers' whole-lane KV slab pair
# Per-bank device measurements of a 4 x 4 lane chain (k = 4) the model is checked against; the served extras are what a
# served process carries beyond the timing tool's chain.
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
    """The verify-rows fold's prefix states per bank for a lane tile of ``total_rows`` (B x R) rows."""

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
    """Whether ``lanes`` MTP lanes at ``drafts`` drafts fit beside the resident build and its single-lane MTP chain.

    The one-tile rule (``B x (k + 1) <= 32``) first, then DRAM per bank: the modeled required side against the live
    reading less ``reserved_bytes_per_bank`` (or the table without a device); one QSA layer's flat lane KV must fit
    the largest contiguous block. The open refuses READY when its measured growth exceeds the required side."""

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
    """The MTP chain's TAIL epilogue, shared by the warm pass and the TAIL capture so the capture only uses programs
    the warm compiled. Returns ``(candidates, token_row, row)``; ``row`` is None without a sampling chain."""

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
    # The prefill path this request took ("chunked" or "teacher_forced") and its shape.
    prefill_mode: str = "teacher_forced"
    prefill_forced_tokens: int = 0
    prefill_chunks: int = 0  # the 32-row chunk replays (full and the padded tail)
    prefill_long_chunks: int = 0  # the 128-row chunk replays ahead of them (--long-chunks / a slab's remainder)
    prefill_slabs: int = 0  # the slab replays ahead of those (--prefill-slab)
    prefill_tail_rows: int = 0
    prefill_handoff_ms: float = 0.0
    # The slabs' host work summed over the request's slabs: input preparation, input copy enqueue, event waits.
    prefill_slab_prepare_ms: float = 0.0
    prefill_slab_upload_ms: float = 0.0
    prefill_slab_wait_ms: float = 0.0
    # This request's MTP pass-loop counters (the chain's cumulative ones are /health.mtp); None for the 1-row loops.
    mtp: dict[str, Any] | None = None
    # The prompt-end snapshot was restored: prefix_reused is the snapshot's length, the rest of the prompt the tail.
    restored: bool = False
    # Schedules (SNAPSHOT_SCHEDULES) of the snapshot this request restored and of the one it left; None when absent.
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


# The schedule that produced a snapshot's state: ``chunked`` = a fresh prefill of exactly these ids from position 0,
# so an exact repeat restores the fresh state bitwise; ``forced-tail`` = a restored or extended request captured it,
# so the state matches a fresh prefill's to rounding only. A restore whose tail is the last prompt token alone
# recaptures the restored state and inherits its schedule.
SNAPSHOT_SCHEDULES = ("chunked", "forced-tail")


@dataclass(frozen=True)
class Qwen38PromptSnapshot:
    """Host record of the chain's prompt-end snapshot. ``vision`` holds the image pad spans with their digests: the
    ids alone do not identify an image prompt's pixels."""

    ids: tuple[int, ...]
    ple_context: tuple[int, int] | None
    schedule: str = "chunked"
    vision: tuple[tuple[int, int, str], ...] = ()


def vision_spans_compatible(
    committed: Sequence[tuple[int, int, str]], wanted: Sequence[tuple[int, int, str]], common: int
) -> bool:
    """Whether a committed prefix of ``common`` ids may serve a prompt with images ``wanted``: every image span below
    ``common`` must match in span and digest (images of one grid render the same pad ids) and lie wholly inside it."""

    below = lambda spans: {span for span in spans if span[0] < common}  # noqa: E731
    return below(committed) == below(wanted) and all(stop <= common for _, stop, _ in below(wanted))


@dataclass(frozen=True)
class Qwen38TeacherForcedRows:
    """:meth:`Qwen38ChatSession.teacher_force`'s result: per wanted position p the argmax and candidate row of the
    TAIL that consumed tokens 0..p; ``full_logits`` holds that TAIL's full-vocabulary logits when asked."""

    rows: dict[int, tuple[int, Any]]
    prefill_mode: str
    chunks: int
    forced_tokens: int
    seconds: float
    full_logits: dict[int, Any] = field(default_factory=dict)
    full_logits_seconds: float = 0.0


@dataclass
class Qwen38MTPPassRun:
    """The pass loop's last pass until :meth:`Qwen38ChatSession._mtp_settle` commits it: rows fed
    ``[t_P, d_1 .. d_a]``, tokens emitted ``[d_1 .. d_a, t']`` and how many the request took."""

    position: int
    rows: list[int]
    emitted: list[int]
    yielded: int
    host_committed: bool = False  # the rows are in the session's committed list (the pass was consumed whole)


class Qwen38ChatSession:
    """One conversation owner: the committed input sequence, the PLE n-gram context, the traced chain.

    ``template`` is the pinned ``Qwen38OfficialChatTemplate``; ``chain`` offers ``Qwen38TracedChain``'s per-step
    primitives.
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
        # The limit defaults to the chain's allocated context less the headroom.
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
        # A chain without the snapshot primitives resets where a restore would apply.
        self.snapshot_available = callable(getattr(chain, "capture_prompt_snapshot", None))
        self.snapshot: Qwen38PromptSnapshot | None = None
        self.committed: list[int] = []
        self.ple_context: tuple[int, int] | None = None
        self.last_finish: str | None = None
        self.row_unconsumed = False  # the next token sits in the row (after length, disconnected, a hook stop)
        self.row_token: int | None = None  # a length finish's last token: read, delivered to the client, not committed
        # The committed prompt's image pad spans with digests: the reuse key beside the ids.
        self.committed_vision: tuple[tuple[int, int, str], ...] = ()
        self.requests_served = 0
        # QWEN38_MTP_DEVICE_ACCEPT_DUMP: complete() writes the device-decided passes' rows, one JSON per request.
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
        """Prompt token ids of exactly ``messages`` and ``tools``; the session adds no system turn."""

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
        """The request's completion budget, refused before touching the device when it does not fit the context
        limit; ``None`` means the remaining context."""

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
        """Write x_t into the token row (behind TAIL(t-1)), HEAD(t), the PLE row for x_t, TAIL(t).

        On an MTP chain the TAIL's MTP row consumes the token at t + 1: ``next_token_id`` when the host knows it,
        else the step's own resolved argmax.
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
        """The chunk driver over ``token_ids`` from the committed position; its alignment steps are forced steps.

        Runs outside the loop guard: the seed and the hand-off synchronize and allocate. A stop at an event sync
        leaves the prompt committed up to the hand-off. ``following_token`` is the MTP layer's token at the last
        prefilled position; ``between_chunks`` and ``event_rows`` are the lanes admission's hooks."""

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

        # An image prompt's rotary positions and feature rows, and the lanes admission's hooks, only when present.
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
        """The evented decode loop; yields (x_t, finish) with finish set on the last item (x_t None: a hook stop).

        EOS is consumed (its HEAD is already queued when the host learns it), so the next turn continues from
        ``<|im_end|>``. On ``max_tokens`` the last token is read but not consumed. A token at or above the tokenizer
        size is an LM-head padding row and ends the request with ``error``. A thinking budget forces ``</think>``.
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

        Each pass feeds ``[t_P, d_1 .. d_k]`` and commits ``a + 1`` rows; the host streams ``[d_1 .. d_a, t']`` and
        keeps ``t'`` as the next pass's row 0. A pass's device commit is deferred to the next pass or to
        :meth:`_mtp_settle`. A forced ``</think>`` or a full cache hands off to the 1-row chain. A sampled request
        samples the pending token from the candidate row and feeds it by a forced step on every hand-off, so it is
        never sampled twice.
        """

        prompt_tokens = len(self.committed)
        decide = None if sampling is None else partial(sampling_step.accept_pass, self, sampling, prompt_tokens)
        # Device acceptance, admitted per request: its hook writes each pass's uniforms; a refused request keeps the
        # host-decided split form.
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
            # Settle the pass loop first; on "length" a pending token the client saw is fed, an unseen one dropped.
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
            # Leave the pending token where the 1-row loop expects it: in the row if unstreamed, else forced (a
            # sampled request always forces it, streaming it first if unseen).
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
                    # Consume the token as the 1-row loop does: settle the previous pass, then a forced step.
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
                # The eager seed, then the bootstrap pass.
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
                # The device-decided pass's ledger and, with the dump on, its record.
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
            # The whole pass was consumed; its device commit is the next pass's or the settle's, so the run stays.
            self.committed.extend(run.rows)
            run.host_committed = True
            pending = emitted[accepted]
            pending_yielded = True

    def _mtp_settle(self, finish: str) -> None:
        """Leave the pass loop: commit the rows the request consumed, rebuild the 1-row buffers and leave the row as
        :meth:`_generate` would.

        The last yielded token is consumed unless the request ended on ``length``; a consumed ``t'`` is fed by a
        forced step. Drafts past the consumed prefix are rolled back by committing fewer rows.
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
        """One JSON per request of the device-decided passes: k, sampling parameters, uniforms and per-pass records."""

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
        """Position 0 and position-zero state at the captured addresses. The snapshot record is dropped too: the next
        prefill rewrites the caches below its position, which the snapshot does not copy."""

        self.chain.reset_and_seed(SEED_TOKEN_ID)
        self.committed = []
        self.snapshot = None
        self.ple_context = None
        self.last_finish = None
        self.row_unconsumed = False
        self.row_token = None
        self.committed_vision = ()

    def reusable_prefix(self, token_ids: Sequence[int], vision: Sequence[tuple[int, int, str]] = ()) -> tuple[int, str]:
        """How the device meets ``token_ids``: ``(n, "extends")`` when they extend the committed ``n`` ids,
        ``(n, "snapshot")`` when they extend the prompt-end snapshot's, else ``(0, "reset")``. The image spans'
        digests are part of the key; a partial match cannot be rewound."""

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
        """Restore the chain's snapshot on device; the host follows its record."""

        self.chain.restore_prompt_snapshot()
        self.committed = list(self.snapshot.ids)
        self.ple_context = self.snapshot.ple_context
        self.committed_vision = self.snapshot.vision
        self.last_finish = None
        self.row_unconsumed = False

    def _capture_prompt_snapshot(self, schedule: str = "chunked") -> None:
        """Copy the device state after the committed ids (every prompt token but the last) on the chain."""

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
        vision: vision_splice.Qwen38VisionPrompt | None = None,
    ) -> Qwen38ChatCompletion:
        """Prefill what the device does not already hold, then generate up to ``max_tokens`` tokens.

        Any failure inside the device section poisons the session: it must not serve again. ``on_token`` raising
        ``OSError`` (client gone) is not such a failure: generation stops with ``disconnected`` and the next token
        stays in the row. ``should_stop`` is polled between steps and at the prefill's event syncs. Image pads
        (``vision``) never take 1-row steps: an extension whose 1-row steps would hold a pad resets instead.
        ``speculative=False`` keeps an MTP chain's greedy request on the 1-row loop.
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
            # All but the last token of an image prompt must fill the chunk trace's minimum.
            if len(token_ids) - 1 < CHUNK_PREFILL_MIN_ROWS:
                raise Qwen38ChatRequestError("image prompt too short for the chunked prefill")
        elif has_image_pads:
            raise Qwen38ChatRequestError("the prompt holds image pads but no vision inputs")
        if sampling is not None and self.sampling is None:
            raise Qwen38ChatRequestError("sampling is unavailable: this chain captured no candidate row (greedy only)")
        if self.poisoned:
            raise Qwen38ChatChainError("session is poisoned by an earlier device failure")
        # MTP drafting serves chunked-mode greedy requests and the sampled ones drafting_admission admits.
        drafting = self.mtp is not None and speculative and mode == "chunked"
        started_ns = self.clock_ns()
        spans = () if vision is None else vision.spans(token_ids)
        common, reuse = self.reusable_prefix(token_ids, spans)
        if common and vision is not None:
            # The extension's 1-row steps (alignment steps, or a short forced tail) must be text.
            head = list(token_ids[common:-1])
            aligned = alignment_steps(common, len(head))
            chunkable = mode == "chunked" and len(head) - aligned >= CHUNK_PREFILL_MIN_ROWS
            if vision_splice.image_lanes(head[:aligned] if chunkable else head):
                common, reuse = 0, "reset"
        # The device sampler's per-request writes precede every prompt step: the last prompt TAIL chooses the first
        # token under this request's policy.
        device_loop = False
        if (
            getattr(self.sampling, "sampler", None) is not None
            or getattr(self.sampling, "device_accept", None) is not None
        ):
            # The device acceptance needs its request start too: an --mtp server samples on the host.
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
            # The snapshot stays valid until a prefill reaching its last token replaces it or a reset drops it, so a
            # stopped extension leaves the earlier one restorable.
            suffix = token_ids[common:]
            chunked: Qwen38PrefillResult | None = None
            if reuse == "reset" or common == 0:
                schedule = "chunked"  # a prefill from position 0 (a reset, or a new session's first request)
            elif reuse == "snapshot" and len(suffix) == 1:
                schedule = snapshot_before.schedule
            else:
                schedule = "forced-tail"
            capture = partial(self._capture_prompt_snapshot, schedule)
            # The last prompt token is always teacher-forced inside the guard as the first decode replay; the
            # prompt-end snapshot is taken just before it.
            before_last: Callable[[], None] | None = capture
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
        """From position 0 through ``token_ids``: at every wanted position p the argmax and candidate row of the TAIL
        that consumed tokens 0..p, so wanted positions are forced steps; in chunked mode unwanted stretches may go
        through the chunk driver. Leaves the row unconsumed like a ``length`` finish. Needs the candidate row."""

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
    """The profile with its route derived from the cluster descriptor (no device is opened): adopted when the
    profile leaves it ``None``, checked against the pinned one otherwise."""

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
        # carry the line, and the mesh must open in that order (the ring walk can put a neighbour pair on the unused
        # link). The walk is recorded next to the route for comparison.
        route = physical_route.derive_fabric_line_route(
            document, chips_with_mmio, hardware_profiles.system_mesh_chip_id
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
    """Open the profile's four devices as the logical 1x4 mesh on FABRIC_1D, in the profile's route order.

    ``command_queues=2`` opens a second queue for the MTP pass's early rows read; nothing but buffer reads may run on
    it. A check before the fabric enable raises with the fabric untouched; a failure after it disables the fabric
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
    """What an MTP-drafting chain adds (``mtp_v2``): components, verify / draft states and traces, step inputs, chunk
    extensions, the live pass loop and the cumulative counters.

    ``step_written`` tracks the TAIL contract: every TAIL reads the step inputs written for its step.
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
    # The split verify (--mtp --sampling): sampled requests draft through it, the host deciding every pass; greedy
    # requests keep the fused traces.
    sampled: bool = False
    split_traces: mtp_v2.Qwen38TTNNMTPTraces | None = None  # verify_head, verify_tail, their draft; commit = traces'
    split_verify_output: mtp_v2.Qwen38TTNNVerifyOutput | None = None  # the tail's readback row (the split draft's)
    head_output: mtp_v2.Qwen38TTNNVerifyHeadOutput | None = None
    # The device-decided sampled form (QWEN38_MTP_DEVICE_ACCEPT=1): head, accept program and tail in one trace, for
    # the sampled requests the device acceptance admits.
    device_accept: bool = False
    sampled_traces: mtp_v2.Qwen38TTNNMTPTraces | None = None  # verify_sampled, its draft; commit = traces'
    sampled_verify_output: mtp_v2.Qwen38TTNNVerifyOutput | None = None
    # Early rows read (needs the second command queue): the fused and device-decided pass loops read the verify lanes
    # on queue 1 and look up the next pass's PLE rows under the draft.
    ple_early: bool = False
    # Greedy split passes whose host decision was checked against the device lanes (0 on the served path).
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
            # Draws are the uniforms the program consumed (a* + 2, or k + 1 with every draft accepted); the device
            # never falls back, it counts the rows whose guard failed instead.
            self.sampled_passes += 1
            self.sampled_accepted_drafts += pass_record.accepted
            self.sampled_draws += min(pass_record.accepted + 2, self.drafts + 1)
            self.device_accept_passes += 1
            self.device_accept_guard_deviations += int(pass_record.guard_deviations or 0)
        return pass_record

    # Cumulative on the chain (/health.mtp); a request reports what it added via ``summary(since=counters())``.
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
        """The ``qwen38.mtp`` object: cumulative since the chain opened, or the counts added since a ``counters()``
        snapshot ``since``."""

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
    """The traced decode chain kept open for serving: 8 decode traces, the persistent token and PLE rows, and the
    optional chunk prefill, sampling and MTP extensions."""

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
    # The driver runs slabs first, then 128-row chunks, then 32-row chunks and the padded tail.
    long_chunk_state: Any = None
    long_chunk_trace_id: int | None = None
    long_chunk_capture_ms: float = 0.0
    slab_state: Any = None
    slab_trace_id: int | None = None
    slab_capture_ms: float = 0.0
    # The GDN state re-anchor the chunk trace was captured with (every chunk replay commits through it).
    chunk_gdn_step_anchor: bool = False
    sampling: sampling_step.Qwen38SamplingChainExtension | None = None
    mtp: Qwen38ChainMTP | None = None
    # The prompt-end snapshot buffers, allocated before the warm pass.
    snapshot: Any = None
    closed: bool = field(default=False, repr=False)

    @property
    def mesh(self) -> Any:
        return self.construction.builder.mesh_device

    @property
    def allocated_context(self) -> int:
        """The build's allocated context: the KV caches, RoPE tables and position constants are sized by it."""

        return self.built_target.model.allocated_context

    # -- the per-step primitives the session calls --------------------------------------------

    def write_token_row(self, token_id: int) -> None:
        ttnn.copy_host_to_device_tensor(resident_decode.device_token_row(self.mesh, token_id), self.token_row_io)

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
        """Replay-loop guard: only the loop's own host I/O stays callable, a synchronize fails."""

        return resident_decode.forbid_trace_body_host_io_and_sync(
            phase="chat request device loop", allowed=resident_decode.REPLAY_LOOP_HOST_IO_CALLS
        )

    def reset_and_seed(self, token_id: int) -> None:
        """In-place state reset to position 0 and the seed token in the persistent row."""

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
        """The snapshot back into the generic state, then the device position checked against it."""

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
        """Run the chunk driver on this chain (not under the loop guard): alignment steps through ``forced_step``,
        the seed, chunk replays, the padded tail and the hand-off."""

        if self.chunk_trace_id is None:
            raise Qwen38ChatChainError("the chain was opened without the chunk trace")
        return Qwen38ChunkPrefill(
            self.built_target.model,
            self.mesh,
            self.state,
            self.chunk_state,
            self.chunk_trace_id,
            forced_step=forced_step,
            verify_allocations=False,  # verified once after capture
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
        """Eager switch into verify mode, then the bootstrap pass with row 0 ``first_token``; returns its record.

        Neither hook runs the fused verify traces (greedy); ``decide`` runs the split traces, the host deciding
        between head and tail; ``before_verify_sampled`` writes each pass's uniforms and runs the device-decided
        traces."""

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
            # The verify lanes read on the second queue under the draft. Not for the host-decided split: its head
            # readback blocks the host before the tail.
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
        """Eager fp32 ``[rows, VOCAB]`` gather of the head's verify logits: the split verify's fallback for a row
        whose candidate guard failed."""

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
        mtp_ple_early: bool = False,
    ) -> Qwen38TracedChain:
        """Build the target, allocate the states, run the warm pass under the miss guard and capture the traces.

        Every state is allocated before any capture: the traces bake their addresses in. Chunk, sampling and MTP
        programs compile in the warm pass and are checked there against host references; ``warm_hook`` runs after
        the warm pass, misses still allowed, for a caller's own eager programs. ``mtp_ple_early`` needs the mesh
        opened with two command queues.
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
        # A slab under MTP runs the MTP rows through the 128-row twin's slab form; the slab requires the long chunks.
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
            # the mesh allocator's DRAM view (one virtual allocator, every card alike)
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
            # Admission on the live allocator; the measured growth of the MTP build, states and traces is checked
            # against its estimate below.
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
            )
            mtp_admission["verify_forms_captured"] = list(forms)
            # printed admitted or not: a refused or crashed open has only this record
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
        # The MTP states are allocated before any capture: every trace bakes their addresses in. With ``mtp_sampled``
        # the sampling extension is built first: the split verify's candidates epilogue uses its constants.
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
                # The 128-row twin runs the MTP rows inside the 128-row chunk body; part of the states term below.
                chain_mtp.long_chunk_extension = mtp_v2.Qwen38TTNNMTPChunkExtension.allocate(
                    model, verify, long_chunk_state, base=chain_mtp.chunk_extension, slab_rows=slab_rows
                )
            synchronize()
            mtp_dram_bytes_per_bank["states"] = dram_allocated_per_bank() - allocated_before_mtp_states
        # The sampling extension builds the device acceptance from QWEN38_MTP_DEVICE_ACCEPT; the caller's switch must
        # agree with it.
        device_acceptance = None if sampling_extension is None else getattr(sampling_extension, "device_accept", None)
        if (device_acceptance is not None) != mtp_device_accept:
            raise Qwen38ChatChainError(
                f"mtp_device_accept={mtp_device_accept} but the sampling extension "
                f"{'built' if device_acceptance is not None else 'did not build'} its device acceptance "
                "(QWEN38_MTP_DEVICE_ACCEPT decides both)"
            )
        accept_constants = None if device_acceptance is None else device_acceptance.constants
        # The snapshot buffers before any capture (the tracker's post-capture check sees no later allocation); with
        # MTP the alignment layer's generic state is a 49th layer of it.
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
                # The greedy body's last op: the resolve copies into the persistent token row (``into``).
                resolved_row = lm_head.resolve_greedy_on_device(candidates, into=token_row_io)
            else:
                # The MTP body: the capture's own epilogue, eager here.
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

        # Warm pass, eager, misses allowed: every kernel of the generic body compiles here.
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
            # The pass loop's programs and both mode switches at every position residue: the seed's ring slot slices
            # are position-dependent and the server switches at any P. The split head / tail rounds run every op of
            # the fused body on tensors of the same specs, so they also compile the fused capture's programs.
            marker("before-chat-mtp-warm-pass")
            warm_fed = resident_decode.SINGLE_TRACE_WARM_POSITIONS  # positions the warm pass consumed so far

            def warm_mtp_chain(target: Qwen38ChainMTP) -> None:
                """Warm every captured form at every position residue, so no capture meets an uncompiled program."""

                nonlocal warm_fed, resolved, ple_context
                warm_forms = (
                    ("fused",) if not target.sampled else ("split", "sampled") if target.device_accept else ("split",)
                )
                rounds = [(residue, warm_form) for residue in range(RESIDUE_CLASSES) for warm_form in warm_forms]
                for index, (residue, warm_form) in enumerate(rounds):
                    # 1-row steps to this round's residue class (the seed's programs at every P mod 4)
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
                        # The device-decided form under the warm policy; its statistics row must equal the host
                        # reference bitwise.
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
                        # The split form: head, rows candidates checked against the eager rows gather, decision, tail.
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
                    # The pass row: the verify's accept row and the draft's token chain.
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

            warm_mtp_chain(chain_mtp)
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
                # With MTP the 128-row twin's slab form compiles here too.
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
            # One eager 128-row chunk from the reset state (with MTP the 128-row twin's too): its programs compile
            # here, before the miss guard.
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

        # The snapshot round trip compiles its copy programs; the restored buffers must equal the captured ones
        # bitwise.
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

        # Pre-capture reset, the tracker's acknowledgements of every host- or trace-written buffer, then the miss
        # guard closes.
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
            """TAIL's last ops: local candidates, device greedy resolve, copy into the persistent token row, plus the
            sampling chain's candidate row and the MTP layer's row when present."""

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
        # The MTP traces term is what the MTP captures add after the last prefill capture.
        dram_after_prefill_captures = dram_allocated_per_bank()

        def capture_mtp_chain(target: Qwen38ChainMTP, dram_baseline: int) -> None:
            """Capture the drafting chain's traces; ``dram_baseline`` is the allocation they grow from.

            Each verify body allocates its readback row inside its own capture and the draft reads that row, so every
            verify form is captured before its own draft; the commit reads neither row and is shared."""

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
                # The device-decided sampled form: one trace (head, accept program, tail) and the draft on its row.
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
                    f"(verify MoE rows {target.admission['mtp_moe_rows']}, "
                    f"{target.admission['verify_forms']} verify form(s); the required side decided, the free side "
                    f"read {target.admission['free_bytes_source']}) at allocated context {model.allocated_context}"
                )

        if chain_mtp is not None:
            marker("before-chat-mtp-captures")
            capture_mtp_chain(chain_mtp, dram_after_prefill_captures)
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
            + ([] if self.mtp is None else self.mtp.captured_trace_ids())
        )

    def close(self) -> None:
        """Release everything in dependency order; skipped by the caller when a request failed mid-loop."""

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
        if self.mtp is not None:
            self.mtp.traces = None
            self.mtp.split_traces = None
            self.mtp.sampled_traces = None
            for name in ("verify_output", "split_verify_output", "sampled_verify_output", "head_output"):
                output = getattr(self.mtp, name)
                if output is not None:
                    output.release_tensors()
                    setattr(self.mtp, name, None)
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
    mtp_ple_early: bool = False,
    warm_hook: Callable[[Qwen38TracedChain], None] | None = None,
) -> Qwen38TracedChain:
    """Live construction on the open mesh (missing BF4 layers converted first), then the chain open."""

    construction = construct_live_decode(
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
        mtp_ple_early=mtp_ple_early,
        warm_hook=warm_hook,
    )
