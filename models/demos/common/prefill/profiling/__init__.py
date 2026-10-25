# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Zone profiling for chunked prefill, shared by the galaxy prefill models.

One mechanism, parameterized by a per-model :class:`~.spec.ZoneSpec`:

* :mod:`.zones`            — ``ZoneProfiler``: the ``zone()`` context manager the model code wraps its
                             ops in, emitting paired Tracy signposts (off unless the model's env flag is set)
* :mod:`.parse_zone_perf`  — stream a tracy ops CSV, rebuild the zone hierarchy from the signposts,
                             roll device-kernel time / ops / bytes up per zone, per device, per layer class
* :mod:`.visualize_zones`  — the text + self-contained HTML report on top of the parser

Each model keeps a thin ``utils/profiler_utils.py`` that declares its spec (signpost prefix, env-var
prefix, layer classes, which zones are communication vs memory) and re-exports ``zone`` /
``read_profiler``, plus two one-line CLI shims under ``tests/perf/`` so the README commands stay
model-local. Everything that can silently produce a wrong report lives here, once.
"""
