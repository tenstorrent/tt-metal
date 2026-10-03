# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced per-stage cost of the attention (in-trace per-call time). Superseded by tests/test_attn_profile.py (the old
version of this file crashed: MeshConfig.allreduce deallocates its input, so a replayed call used a freed tensor).
Run:  pytest tests/test_attention_microbench.py -s   (prints 'PR ...' lines; ATTN_IMPL=ref for the baseline)."""

from models.demos.blackhole.deepseek_v41_flash.tests.test_attn_profile import test_attn_profile  # noqa: F401
