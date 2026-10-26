# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Zone markers for chunked-prefill profiling (Tracy).

A "zone" is a named region of the forward pass. Every zone emits a pair of Tracy signposts around the
ops it contains::

    <PREFIX>_START <name>   ... the zone's ttnn ops ...   <PREFIX>_END <name>

Signposts land in the tracy ops CSV as rows with ``OP TYPE == "signpost"`` and ``OP CODE == "<prefix>
<name>"``, interleaved with the op rows in host-enqueue order. Because a zone's ops are exactly the
ops enqueued between its two signposts, summing ``DEVICE KERNEL DURATION [ns]`` between the markers
gives that zone's device time — regardless of when the device actually ran them. This is the same
mechanism deepseek_v3_d_p uses (``forward_layer_{i}_start`` in ``tt/tt_prefill_transformer.py``,
``MLA_START``/``MLA_END`` in ``tt/mla/mla.py``), just with a nested zone hierarchy instead of two flat
regions. A matching Tracy *host* zone is emitted too, so the regions also show up as nested zones on
the host timeline in the Tracy GUI / WASM viewer.

Everything is OFF by default: unless the model's ``<ENV_PREFIX>_ZONES=1`` is set, ``zone()`` returns a
shared no-op context manager and nothing is emitted. With the flag set, zones at or below
``<ENV_PREFIX>_LEVEL`` become real. Signposts are host-only messages (no device op, no sync), so an
enabled zone does not perturb device timings.

Each model instantiates one :class:`ZoneProfiler` from its :class:`~.spec.ZoneSpec` in its
``utils/profiler_utils.py`` and re-exports ``zone`` / ``read_profiler``::

    from models.demos.gpt_oss_d_p.utils.profiler_utils import zone

    with zone("ring_joint_sdpa"):
        out = dense_sp_attention(...)

Profiler reads: the device profiler buffer holds ``TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT`` programs
per device (default 1000) and a prefill chunk enqueues far more than that over a full model, so
``read_profiler()`` MUST be called periodically or device data is silently dropped — the parser
reports how many device ops lost their duration that way.

Failure policy: ``ttnn.tracy_message`` / ``start_tracy_zone`` / ``stop_tracy_zone`` are compiled as
no-ops in a build without Tracy, so there is nothing to catch here. An exception from them while
zones are enabled is a real bug in an opt-in profiling run and is left to propagate — a silently
missing marker would mis-attribute every op after it.
"""

from __future__ import annotations

import contextlib
import os

from loguru import logger

from models.demos.common.prefill.profiling.spec import ZoneSpec

# Zone detail levels. A zone is emitted only when its level <= the configured level, so one set of
# call sites serves every depth of investigation:
#
#   1 COARSE  per layer: attn vs mlp. A few zones per layer — start here, it answers "which block".
#   2 MEDIUM  + every block that costs real time: the SDPAs, the CCLs, and the MoE stages. The default.
#   3 FINE    + norms, residuals, rope, head splits, and the sub-splits of the medium zones.
#
# Levels are not just presentation: each zone is two Tracy signposts, and Tracy caps a trace at 32K
# source locations, so a coarse level also buys headroom on long captures.
COARSE, MEDIUM, FINE = 1, 2, 3

# Reused singleton for the disabled path — nullcontext carries no per-use state, so one instance is
# safe to enter/exit repeatedly (and re-entrantly).
_NULL_ZONE = contextlib.nullcontext()

# The signpost wire format is the contract with tools/tracy/process_ops_logs.py: the backticks are the
# message CSV's quotechar and "TT_SIGNPOST: " marks the row as a signpost, whose remainder becomes the
# CSV's OP CODE. Same bytes as `tracy.signpost(header)`, minus its per-call loguru INFO line: a
# full-model chunk opens hundreds of zones, and importing tools/tracy into model code would drag its
# CSV post-processing (pandas) into every model import.
SIGNPOST_FORMAT = "`TT_SIGNPOST: {header}`"


def parse_level(raw: str | None, *, var: str, default: int = MEDIUM) -> int:
    """Zone level from an env value; a bad value warns and falls back rather than breaking model imports."""
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"{var}={raw!r} is not an integer zone level (1=coarse, 2=medium, 3=fine); using {default}")
        return default


def signpost(header: str) -> None:
    """Emit a Tracy signpost row that the ops CSV will carry as ``OP CODE == header``."""
    import ttnn

    ttnn.tracy_message(SIGNPOST_FORMAT.format(header=header))


class ZoneProfiler:
    """The ``zone()`` / ``read_profiler()`` pair for one model, configured from its env vars.

    ``enabled`` / ``level`` / ``host_zones`` default to the spec's env vars (read once, at construction,
    which the model does at import — so the harness must set them before importing the model) and can
    be passed explicitly, which is what the unit tests do.
    """

    def __init__(self, spec: ZoneSpec, *, enabled: bool | None = None, level: int | None = None, host_zones=None):
        self.spec = spec
        self.enabled = os.getenv(spec.zones_env, "0") == "1" if enabled is None else bool(enabled)
        self.level = parse_level(os.getenv(spec.level_env), var=spec.level_env) if level is None else int(level)
        # Host-side Tracy zones are cosmetic (the signposts are what the parser reads); their own flag
        # keeps a long capture under Tracy's source-location cap without losing the CSV zones.
        self.host_zones = os.getenv(spec.host_zones_env, "1") == "1" if host_zones is None else bool(host_zones)

    def zone(self, name: str, level: int = MEDIUM):
        """Context manager marking ``name`` as a profiling zone.

        No-op unless zones are enabled and ``level <= self.level`` (see COARSE/MEDIUM/FINE). Suppressing
        a zone does not lose its ops: they are charged to the nearest enclosing zone that is still open,
        and the report shows them as that zone's ``(self)`` bucket, so a coarse run still accounts for
        100% of the time, just in fewer buckets.

        Zones nest by call site — the parser builds the full path from the nesting, so names here are
        local (``"dispatch"``, not ``"mlp/dispatch"``). The same name is entered once per layer and the
        parser accumulates across layers.
        """
        if not self.enabled or level > self.level:
            return _NULL_ZONE
        return self._zone(name)

    @contextlib.contextmanager
    def _zone(self, name: str):
        import ttnn

        signpost(f"{self.spec.zone_start} {name}")
        if self.host_zones:
            ttnn.start_tracy_zone(self.spec.host_zone_scope, name, 0)
        try:
            yield
        finally:
            if self.host_zones:
                ttnn.stop_tracy_zone(name)
            signpost(f"{self.spec.zone_end} {name}")

    def read_profiler(self, mesh_device) -> None:
        """Flush the device profiler buffers to host. No-op unless zones are enabled.

        Blocking: it reads the device-side profiler buffers, so it inflates host wall-clock and lands in
        the trace as a gap before the next op. Device kernel durations are unaffected.
        """
        if not self.enabled:
            return
        import ttnn

        ttnn.ReadDeviceProfiler(mesh_device)
