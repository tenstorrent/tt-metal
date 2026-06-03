# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated, verbatim copy of origin/main's sampling code, for perf/feature comparison only.

These are NOT the modules the models use. They are unmodified copies of
``models/common/sampling/{tt_sampling,tt_log_probs,_utils}.py`` taken from origin/main
(commit 7b7a26f14687dbfb85263982694fc3c4debf19e3, 2026-06-03), with ONLY their intra-package
imports rewritten to resolve within this package. They exist so the comparison harness
(``models/common/tests/modules/sampling/test_sampling_perf_compare.py``) can benchmark TTTv2
``Sampling1D`` against the up-to-date TTTv1 ``TTSampling`` from main without touching the
current branch's (older) ``models/common/sampling`` copy.

Source-of-truth for the gap analysis: ``dev-tools/agents-context/ttv2_sampling_main_comparison.md``.
"""
