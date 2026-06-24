# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-layer prefill/decode for Tracy. Same pipeline as text prefill/decode with n_layers=1."""

from __future__ import annotations

import bz2

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole

from models.experimental.glm4_moe_lite.tests.pipeline_tests.test_utils import (
    PROMPT_FILE,
    apply_single_layer_env,
    alloc_kv_cache_and_page_table,
    compute_max_seq_len,
    create_runner,
    fabric_1d_trace_device_params,
    load_tokenizer,
    mesh_shape_param,
    require_snapshot,
    scale_page_params,
)

MESH = [mesh_shape_param()]


@torch.no_grad()
@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("seq_len", (1024,), ids=["1k"])
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize("page_params", [{"page_block_size": 64, "page_max_num_blocks": 1024}])
@pytest.mark.parametrize("device_params", fabric_1d_trace_device_params(num_command_queues=2), indirect=True)
@pytest.mark.parametrize("mesh_device", MESH, indirect=True)
def test_single_layer_prefill(
    seq_len,
    batch_size,
    page_params,
    mesh_device,
    reset_seeds,
    monkeypatch: pytest.MonkeyPatch,
):
    apply_single_layer_env(monkeypatch)

    snap = require_snapshot()
    page_params = scale_page_params(page_params, seq_len, batch_size)
    tok = load_tokenizer(snap)
    with bz2.open(PROMPT_FILE, "rt", encoding="utf-8") as f:
        encoded_prompt = tok(f.read(), add_special_tokens=True)["input_ids"][:seq_len]

    block_size = int(page_params["page_block_size"])
    total_tokens = seq_len + 32
    max_seq_len = compute_max_seq_len(total_tokens, block_size)
    runner = create_runner(
        mesh_device=mesh_device,
        snapshot_dir=snap,
        max_seq_len=max_seq_len,
        cache_subdir="pipeline_single_layer_prefill",
    )
    kv_cache, page_table, _ = alloc_kv_cache_and_page_table(
        mesh_device=mesh_device,
        runner=runner,
        batch_size=batch_size,
        total_tokens=total_tokens,
        block_size=block_size,
    )

    runner.prefill(
        tokens=torch.tensor([encoded_prompt], dtype=torch.int32),
        prompt_lens=[seq_len],
        page_table=page_table,
        kv_cache=kv_cache,
        seq_pad_multiple=int(page_params["page_block_size"]),
    )
    ttnn.synchronize_device(mesh_device)


@torch.no_grad()
@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize("page_params", [{"page_block_size": 64, "page_max_num_blocks": 1024}])
@pytest.mark.parametrize("device_params", fabric_1d_trace_device_params(num_command_queues=2), indirect=True)
@pytest.mark.parametrize("mesh_device", MESH, indirect=True)
def test_single_layer_decode(
    batch_size,
    page_params,
    mesh_device,
    reset_seeds,
    monkeypatch: pytest.MonkeyPatch,
):
    apply_single_layer_env(monkeypatch)

    snap = require_snapshot()
    prefill_len = 32
    tok = load_tokenizer(snap)
    enc = tok("Hello.", add_special_tokens=True)["input_ids"][:prefill_len]

    block_size = int(page_params["page_block_size"])
    total_tokens = prefill_len + 64
    max_seq_len = compute_max_seq_len(total_tokens, block_size)
    runner = create_runner(
        mesh_device=mesh_device,
        snapshot_dir=snap,
        max_seq_len=max_seq_len,
        cache_subdir="pipeline_single_layer_decode",
    )
    kv_cache, page_table, _ = alloc_kv_cache_and_page_table(
        mesh_device=mesh_device,
        runner=runner,
        batch_size=batch_size,
        total_tokens=total_tokens,
        block_size=block_size,
    )

    runner.prefill(
        tokens=torch.tensor([enc], dtype=torch.int32),
        prompt_lens=[prefill_len],
        page_table=page_table,
        kv_cache=kv_cache,
        seq_pad_multiple=int(page_params["page_block_size"]),
    )

    token_id = int(enc[-1])
    tt_out = runner.decode(
        tokens=torch.tensor([[token_id]], dtype=torch.int32),
        start_pos=torch.tensor([prefill_len], dtype=torch.int32),
        page_table=page_table,
        kv_cache=kv_cache,
    )
    del tt_out
    ttnn.synchronize_device(mesh_device)
