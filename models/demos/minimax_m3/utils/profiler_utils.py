# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Zone markers for MiniMax-M3 prefill profiling (Tracy).

A "zone" is a named region of the forward pass. Every zone emits a pair of Tracy
signposts around the ops it contains::

    M3_ZONE_START <name>   ... the zone's ttnn ops ...   M3_ZONE_END <name>

Signposts land in the tracy ops CSV as rows with ``OP TYPE == "signpost"`` and
``OP CODE == "<prefix> <name>"``, interleaved with the op rows in host-enqueue
order. Because a zone's ops are exactly the ops enqueued between its two
signposts, summing ``DEVICE KERNEL DURATION [ns]`` between the markers gives that
zone's device time — regardless of when the device actually ran them. This is the
same mechanism deepseek_v3_d_p uses (``forward_layer_{i}_start`` in
``tt/tt_prefill_transformer.py``, ``MLA_START``/``MLA_END`` in ``tt/mla/mla.py``),
just with a nested zone hierarchy instead of two flat regions.

A matching Tracy *host* zone is emitted too, so the regions also show up as
nested zones on the host timeline in the Tracy GUI / WASM viewer.

Everything here is OFF by default: unless ``M3_PROFILE_ZONES=1`` is set, ``zone()``
returns a shared no-op context manager and nothing is emitted. With the flag set,
zones at or below ``M3_PROFILE_LEVEL`` become real. Signposts are host-only
messages (no device op, no sync), so an enabled zone does not perturb device
timings.

Usage::

    from models.demos.minimax_m3.utils.profiler_utils import zone

    with zone("msa/indexer"):
        block_scores = ttnn.experimental.indexer_score_msa(...)
        block_ids = ttnn.experimental.topk_large_indices(...)

Profiler reads: the device profiler buffer holds ~1000 ops per device, and one M3
prefill chunk enqueues far more than that (~50-60 ops per layer x 60 layers), so
``read_profiler()`` MUST be called periodically or device data is silently
dropped. ``tests/perf/profile_prefill.py`` wires it to the model's
``on_layer_complete`` seam.

See models/demos/minimax_m3/tests/perf/README_profiling.md.
"""

from __future__ import annotations

import contextlib
import os

import ttnn


def _signpost(header: str) -> None:
    """Emit a Tracy signpost, byte-identical to `tracy.signpost(header)` minus its loguru line.

    `tracy.signpost` logs every call at INFO. A 60-layer chunk opens ~20 zones per layer, so going
    through it would print ~2.4k lines per chunk (~30k across warmup + a 12-chunk prefix) and burn real
    time in log formatting. The wire format is the contract with tools/tracy/process_ops_logs.py: the
    backticks are the message CSV's quotechar and "TT_SIGNPOST: " is what marks the row as a signpost,
    whose remainder becomes the CSV's OP CODE.
    """
    try:
        ttnn.tracy_message(f"`TT_SIGNPOST: {header}`")
    except Exception:  # tracy disabled in this build — zones become inert
        pass


ZONE_START_PREFIX = "M3_ZONE_START"
ZONE_END_PREFIX = "M3_ZONE_END"

# Zone detail levels. A zone is emitted only when its level <= M3_PROFILE_LEVEL, so one set of call
# sites serves every depth of investigation:
#
#   1 COARSE  per layer: attn vs mlp. ~3 zones/layer — start here, it answers "which block".
#   2 MEDIUM  + every block that costs real time: sdpa, the CCLs, cache_read, indexer, and the MoE
#             stages (dispatch / experts_mm / combine / reduce). ~20 zones/layer. The default.
#   3 FINE    + norms, residuals, rope, head splits, and the sub-splits of the medium zones
#             (deshard vs slice, weighted-sum vs reduce-scatter). ~35 zones/layer.
#
# Levels are not just presentation: each zone is two Tracy signposts, and Tracy caps a trace at 32K
# source locations, so a coarse level also buys headroom on long captures.
COARSE, MEDIUM, FINE = 1, 2, 3

# Read once at import: the harness sets these before the model is built.
ZONES_ENABLED = os.getenv("M3_PROFILE_ZONES", "0") == "1"
LEVEL = int(os.getenv("M3_PROFILE_LEVEL", str(MEDIUM)))

# Reused singleton for the disabled path — nullcontext carries no per-use state, so one
# instance is safe to enter/exit repeatedly (and re-entrantly).
_NULL_ZONE = contextlib.nullcontext()

# Host-side Tracy zones are cosmetic (the signposts are what the parser reads). Kept behind
# their own flag so a build without the bindings can still produce a zone CSV.
_HOST_ZONES = os.getenv("M3_PROFILE_HOST_ZONES", "1") == "1"


@contextlib.contextmanager
def _zone(name: str):
    _signpost(f"{ZONE_START_PREFIX} {name}")
    if _HOST_ZONES:
        try:
            ttnn.start_tracy_zone("minimax_m3", name, 0)
        except Exception:  # bindings absent / tracy disabled in this build
            pass
    try:
        yield
    finally:
        if _HOST_ZONES:
            try:
                ttnn.stop_tracy_zone(name)
            except Exception:
                pass
        _signpost(f"{ZONE_END_PREFIX} {name}")


def zone(name: str, level: int = MEDIUM):
    """Context manager marking ``name`` as a profiling zone.

    No-op unless M3_PROFILE_ZONES=1 and ``level <= M3_PROFILE_LEVEL`` (see COARSE/MEDIUM/FINE above).
    Suppressing a zone does not lose its ops: they are charged to the nearest enclosing zone that is
    still open, so a coarse run still accounts for 100% of the time, just in fewer buckets.

    Zones nest by call site — the parser builds the full path from the nesting, so names here are
    local (``"indexer"``, not ``"attn/indexer"``). The same name is entered once per layer and the
    parser accumulates across layers.
    """
    if not ZONES_ENABLED or level > LEVEL:
        return _NULL_ZONE
    return _zone(name)


def read_profiler(mesh_device) -> None:
    """Flush the device profiler buffers to host. No-op unless M3_PROFILE_ZONES=1.

    Required every <1000 ops per device (see the module docstring). Blocking: it reads the
    device-side profiler buffers, so it inflates host wall-clock. Device kernel durations are
    unaffected — measure wall-clock in a separate, unprofiled run.
    """
    if not ZONES_ENABLED:
        return
    ttnn.ReadDeviceProfiler(mesh_device)
