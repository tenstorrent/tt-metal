# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device: the prefill attention's Q split and SDPAProgramConfig.prefix_reuse (tp_common.prefill_sdpa_prefix_reuse)
leave every byte of a prefill unchanged.

Qwen36ForCausalLM.prefill_forward (eager, 2048-token chunks) prefills four rows whose masked tails sit after a
2048- or 4096-token prefix: 128- and 256-token buckets (32-row Q chunks with K/V read batches), a 1024-token bucket
and 2048-token chunks (one Q chunk per core). The logits, the attention K and V of the rows' pages and
the GDN state they leave must match the same step with every call at its own Q chunk and prefix_reuse 0.

    MESH_DEVICE=P150x8 HF_MODEL=<Qwen3.8-27B dir> pytest models/demos/blackhole/qwen36/tests/test_prefill_sdpa_prefix_reuse.py
"""

import pytest
import torch
from loguru import logger

from models.demos.blackhole.qwen36.tests import test_prefill_checkpoint_split as split_test
from models.demos.blackhole.qwen36.tests.test_gdn_block_state import _prompts
from models.demos.blackhole.qwen36.tt import tp_common

LENS = (2100, 2300, 2600, 4200)


@torch.no_grad()
@pytest.mark.timeout(7200)
@split_test._parametrize
def test_prefix_reuse_keeps_the_prefill_bytes(mesh_device, monkeypatch, reset_seeds, ensure_gc):
    monkeypatch.setattr(split_test, "CTX", 8192)
    rows = list(range(len(LENS)))
    t = split_test.Rows(mesh_device, monkeypatch, 8, len(rows))
    prompts = _prompts(len(rows), max(LENS), seed=73)
    policy = tp_common.prefill_sdpa_prefix_reuse
    picked = set()

    def recorded(*args, **kwargs):
        got = policy(*args, **kwargs)
        picked.add((args[0], *got))
        return got

    with monkeypatch.context() as m:
        m.setattr(tp_common, "prefill_sdpa_prefix_reuse", lambda rows, q_chunk, *args, **kwargs: (q_chunk, 0))
        want = t.snapshot(LENS, rows, t.prefill(LENS, rows, prompts))
    monkeypatch.setattr(tp_common, "prefill_sdpa_prefix_reuse", recorded)
    got = t.snapshot(LENS, rows, t.prefill(LENS, rows, prompts))
    logger.info(f"PREFIX REUSE: (rows, q_chunk, prefix_reuse) of the prefill attention calls: {sorted(picked)}")
    assert any(q < 128 and bits == 5 for _, q, bits in picked), "no split call took the K/V read batches"
    assert any(bits == 4 for *_, bits in picked), "no call took one Q chunk per core"
    bad = split_test._report(split_test.compare_dumps(got, want), "prefix reuse vs off")
    assert bad == 0
