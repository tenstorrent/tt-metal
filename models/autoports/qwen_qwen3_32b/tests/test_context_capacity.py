# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, one-length-per-process capacity probe for functional or optimized prefill."""

from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path

import pytest
import torch

from models.autoports.qwen_qwen3_32b.tests.test_functional_decoder import (
    EMITTED_BATCH,
    REPRESENTATIVE_LAYER,
    FunctionalDecoder,
    _config,
    _empty_caches,
    _synthetic_state,
    _to_host,
    _tt_tensor,
)

PROBE_LENGTH_ENV = "QWEN3_32B_CONTEXT_PROBE_LEN"
EXPECT_OOM_ENV = "QWEN3_32B_CONTEXT_EXPECT_OOM"
DECODER_ENV = "QWEN3_32B_CONTEXT_DECODER"
RESULTS_DIR_ENV = "QWEN3_32B_CONTEXT_RESULTS_DIR"


def _write_capacity_result(seq_len: int, payload: dict) -> None:
    directory = os.getenv(RESULTS_DIR_ENV)
    if not directory:
        return
    repo_root = Path(__file__).resolve().parents[4]
    source_paths = (
        "models/autoports/qwen_qwen3_32b/tt/optimized_decoder.py",
        "models/autoports/qwen_qwen3_32b/tests/test_context_capacity.py",
        "models/autoports/qwen_qwen3_32b/doc/context_contract.json",
    )
    payload = {
        **payload,
        "source_sha256": {path: hashlib.sha256((repo_root / path).read_bytes()).hexdigest() for path in source_paths},
    }
    output = Path(directory) / f"capacity_{seq_len}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_batch32_prefill_capacity_probe(mesh_device):
    length_text = os.environ.get(PROBE_LENGTH_ENV)
    if not length_text:
        pytest.skip(f"Set {PROBE_LENGTH_ENV} to run an isolated context-capacity measurement")
    seq_len = int(length_text)

    config = _config()
    state = _synthetic_state(config)
    decoder_kind = os.environ.get(DECODER_ENV, "functional")
    if decoder_kind == "optimized":
        from models.autoports.qwen_qwen3_32b.tt.optimized_decoder import OptimizedDecoder

        decoder_class = OptimizedDecoder
    elif decoder_kind == "functional":
        decoder_class = FunctionalDecoder
    else:
        raise ValueError(f"{DECODER_ENV} must be 'functional' or 'optimized', got {decoder_kind!r}")
    decoder = decoder_class.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH,
        max_cache_len=seq_len,
    )
    if decoder_kind == "optimized":
        key_cache, value_cache = decoder.allocate_kv_cache(seq_len)
    else:
        key_cache, value_cache = _empty_caches(config, mesh_device, max_cache_len=seq_len)
    hidden = torch.zeros((1, EMITTED_BATCH, seq_len, config.hidden_size), dtype=torch.bfloat16)

    expect_oom = os.environ.get(EXPECT_OOM_ENV) == "1"
    try:
        actual = decoder.prefill_forward(_tt_tensor(hidden, mesh_device), key_cache, value_cache)
    except RuntimeError as error:
        if not expect_oom:
            raise
        message = str(error)
        assert "Out of Memory" in message, message
        detail = next(line.strip() for line in message.splitlines() if "Out of Memory" in line)
        print(
            f"capacity probe EXPECTED OOM: decoder={decoder_kind}, batch={EMITTED_BATCH}, "
            f"seq_len={seq_len}: {detail}"
        )
        _write_capacity_result(
            seq_len,
            {
                "decoder": decoder_kind,
                "batch": EMITTED_BATCH,
                "sequence_length": seq_len,
                "result": "expected_out_of_memory",
                "error": detail,
            },
        )
        return

    if expect_oom:
        pytest.fail(f"Expected a DRAM allocation failure at batch={EMITTED_BATCH}, seq_len={seq_len}")
    host = _to_host(actual)
    assert tuple(host.shape) == tuple(hidden.shape)
    print(
        f"capacity probe PASS: decoder={decoder_kind}, batch={EMITTED_BATCH}, "
        f"seq_len={seq_len}, output_shape={tuple(host.shape)}"
    )
    _write_capacity_result(
        seq_len,
        {
            "decoder": decoder_kind,
            "batch": EMITTED_BATCH,
            "sequence_length": seq_len,
            "result": "pass",
            "output_shape": list(host.shape),
        },
    )
