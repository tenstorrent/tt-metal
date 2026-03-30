#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for NextN MTP + record loading on CPU (no full DeepSeek load).

Run from repo root with venv (or env where torch/transformers are installed):
  ./venv/bin/python models/demos/speculative_deepseek_r1_broad/scripts/test_mtp_record_cpu_smoke.py
Optional: TRANSFORMERS_CACHE=/tmp/hf_smoke HF_HOME=/tmp/hf_smoke to avoid writing to home cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_imports():
    """Test that required modules import."""
    print("Test 1: Imports ...")
    from models.demos.speculative_deepseek_r1_broad.config import EagleConfig, PathProposal
    from models.demos.speculative_deepseek_r1_broad.nextn_sglang_cpu_draft import NextNSglangCPUDraftAdapter
    from models.demos.speculative_deepseek_r1_broad.trace_replay_base import (
        TraceReplayBaseAdapter,
        load_mtp_reference_bundle,
        load_trace_or_mtp_reference,
    )
    assert EagleConfig and PathProposal and NextNSglangCPUDraftAdapter and TraceReplayBaseAdapter
    assert load_mtp_reference_bundle and load_trace_or_mtp_reference
    print("  OK: all imports succeeded")
    return True


def test_nextn_head_adapter_creation():
    """NextNSglangCPUDraftAdapter on CPU (unbound until bind_from_nextn_paths)."""
    print("Test 2: NextNSglangCPUDraftAdapter(device='cpu') ...")
    from models.demos.speculative_deepseek_r1_broad.nextn_sglang_cpu_draft import NextNSglangCPUDraftAdapter

    adapter = NextNSglangCPUDraftAdapter(device="cpu")
    assert adapter.device.type == "cpu"
    assert not adapter.bound
    print("  OK: NextNSglangCPUDraftAdapter created on CPU, bound=False")
    return True


def test_record_load_minimal_payload():
    """Test record loading with a minimal in-memory payload (no file)."""
    print("Test 3: Load MTP reference from minimal payload ...")
    import torch
    from models.demos.speculative_deepseek_r1_broad.default_paths import NEXTN_HF_REPO_ID
    from models.demos.speculative_deepseek_r1_broad.trace_replay_base import load_mtp_reference_bundle

    hidden_size = 128
    num_steps = 4
    batch_size = 1
    payload = {
        "hidden_states": torch.randn(num_steps, batch_size, hidden_size, dtype=torch.float32),
        "next_tokens": torch.randint(0, 1000, (num_steps, batch_size), dtype=torch.long),
        "start_tokens": torch.tensor([1], dtype=torch.long),
        "metadata": {"model_id": "deepseek-ai/DeepSeek-R1-0528"},
    }
    trace = load_mtp_reference_bundle(payload=payload, batch_index=0)
    assert trace.model_id == "deepseek-ai/DeepSeek-R1-0528"
    assert trace.tokenizer_hub_id == NEXTN_HF_REPO_ID
    assert len(trace.step_next_tokens) == num_steps
    assert trace.step_last_hidden.shape == (num_steps, hidden_size)
    assert len(trace.prompt_token_ids) == 1
    assert trace.mtp_prefix_source == "start_tokens_tensor"
    assert trace.mtp_batch_start_tokens == (1,)
    assert trace.prompt_token_ids[0] == 1
    print("  OK: minimal record -> TraceBundle (tokenizer_hub_id -> NextN, not full R1 Hub)")
    return True


def main():
    print("=" * 60)
    print("MTP + record CPU smoke tests (no full DeepSeek)")
    print("=" * 60)
    ok = 0
    if test_imports():
        ok += 1
    if test_nextn_head_adapter_creation():
        ok += 1
    if test_record_load_minimal_payload():
        ok += 1
    print("=" * 60)
    print(f"Passed: {ok}/3")
    return 0 if ok == 3 else 1


if __name__ == "__main__":
    sys.exit(main())
