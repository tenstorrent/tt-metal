# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed one-layer Qwen3-32B execution smoke on a physical BlackHole TP4 product."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.models.qwen3_32b.executor import EagerQwen3_32BExecutor
from models.common.models.qwen3_32b.hf_adaptor import from_pretrained
from models.common.models.qwen3_32b.model import QWEN3_32B_ACCURACY, QWEN3_32B_BH_TP4_CLUSTER_TYPES
from models.common.tests.demos.cleanup_utils import cleanup_model_case
from models.common.tests.demos.run_helpers import make_contiguous_page_table

_HF_MODEL = "Qwen/Qwen3-32B"
_BLOCK_SIZE = 32
_PROMPT_LEN = 128
_MAX_SEQ_LEN = 512


pytestmark = [
    pytest.mark.timeout(1800),
    pytest.mark.parametrize(
        "ttnn_mesh_device",
        [
            {
                "mesh_shape": (1, 4),
                "trace_region_size": 0,
                "num_command_queues": 1,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            }
        ],
        indirect=True,
        scope="module",
        ids=["physical-P150x4-ring"],
    ),
]


def _assert_physical_bh_tp4(mesh_device: ttnn.MeshDevice) -> None:
    assert ttnn.device.is_blackhole(), "BlackHole TP4 smoke requires BlackHole"
    assert (
        ttnn.cluster.get_cluster_type() in QWEN3_32B_BH_TP4_CLUSTER_TYPES
    ), "BlackHole TP4 smoke requires a physical P150_X4 or P300_X2 product"
    assert mesh_device.get_num_devices() == 4
    assert tuple(mesh_device.shape) == (1, 4)


def _cache_dir(hf_model: str) -> Path:
    if root := os.getenv("TT_CACHE_PATH"):
        return Path(root) / "P150x4"
    return Path("model_cache") / hf_model.strip("/") / "P150x4"


def _cache_slice(mesh_tensor, block_start: int, block_end: int) -> torch.Tensor:
    shards = []
    for shard in ttnn.get_device_tensors(mesh_tensor):
        shape = tuple(int(value) for value in shard.shape)
        sliced = ttnn.slice(shard, (block_start, 0, 0, 0), (block_end, shape[1], shape[2], shape[3]))
        shards.append(ttnn.to_torch(sliced).clone())
    return torch.cat(shards, dim=1)


def _kv_block_snapshot(kv_cache, block: int):
    return tuple(tuple(_cache_slice(tensor, block, block + 1) for tensor in layer) for layer in kv_cache)


def _assert_kv_changed(before, after) -> None:
    comparisons = [
        torch.equal(before_tensor, after_tensor)
        for before_layer, after_layer in zip(before, after)
        for before_tensor, after_tensor in zip(before_layer, after_layer)
    ]
    assert comparisons and not all(comparisons), "decode did not advance the position-128 KV block"


def _assert_logits(logits: torch.Tensor, *, vocab_size: int) -> None:
    assert isinstance(logits, torch.Tensor)
    assert tuple(logits.shape) == (1, 1, vocab_size)
    assert torch.isfinite(logits).all()


@pytest.fixture(scope="module")
def production_model(ttnn_mesh_device, require_blackhole_mesh_device):
    _assert_physical_bh_tp4(ttnn_mesh_device)
    ttnn_mesh_device.enable_program_cache()
    ttnn_mesh_device.clear_program_cache()
    llm = None
    try:
        llm = from_pretrained(
            ttnn_mesh_device,
            hf_model=os.getenv("HF_MODEL", _HF_MODEL),
            max_batch_size=1,
            max_seq_len=_MAX_SEQ_LEN,
            n_layers=1,
            optimizations=QWEN3_32B_ACCURACY,
            cache_dir=_cache_dir(os.getenv("HF_MODEL", _HF_MODEL)),
        )
        assert llm.runtime_config.disable_batched_prefill, "P150x4 must retain Qwen3-32B sequential prefill"
        assert llm.model.config.block_configs[0].attention_config.topology == ttnn.Topology.Ring
        yield llm
    finally:
        cleanup_model_case(None if llm is None else llm.model, ttnn_mesh_device)
        ttnn_mesh_device.disable_and_clear_program_cache()
        ttnn.SetDefaultDevice(None)


def test_qwen3_32b_one_layer_prefill_decode_smoke(ttnn_mesh_device, production_model):
    """Exercise production prefill/decode, KV advancement, and warm-cache reuse."""

    model = production_model.model
    runtime = production_model.runtime_config
    executor = EagerQwen3_32BExecutor(model, ttnn_mesh_device)
    try:
        kv_shape = (
            (_MAX_SEQ_LEN // _BLOCK_SIZE),
            runtime.n_kv_heads // ttnn_mesh_device.get_num_devices(),
            _BLOCK_SIZE,
            runtime.head_dim,
        )
        kv_cache = executor.allocate_kv_cache(kv_shape, torch.bfloat16, runtime.n_layers)
        page_table = make_contiguous_page_table(1, _MAX_SEQ_LEN, _BLOCK_SIZE)
        tokens = (torch.arange(_PROMPT_LEN, dtype=torch.long).reshape(1, -1) + 17) % 32000
        prefill_kwargs = {
            "page_table": page_table,
            "kv_cache": kv_cache,
            "prompt_lens": torch.tensor([_PROMPT_LEN], dtype=torch.long),
            "empty_slots": [0],
            "execution": executor.eager_execution,
        }

        logits = executor.prefill_forward(tokens, **prefill_kwargs)
        _assert_logits(logits, vocab_size=model.vocab_size)
        cached_programs = ttnn_mesh_device.num_program_cache_entries()
        assert cached_programs > 0

        repeated_logits = executor.prefill_forward(tokens, **prefill_kwargs)
        _assert_logits(repeated_logits, vocab_size=model.vocab_size)
        assert ttnn_mesh_device.num_program_cache_entries() == cached_programs

        decode_block = _PROMPT_LEN // _BLOCK_SIZE
        kv_before_decode = _kv_block_snapshot(kv_cache, decode_block)
        decode_output = executor.decode_forward(
            torch.tensor([64], dtype=torch.long),
            torch.tensor([_PROMPT_LEN], dtype=torch.long),
            page_table,
            kv_cache=kv_cache,
            execution=executor.eager_execution,
        )
        assert isinstance(decode_output, tuple) and len(decode_output) == 2
        decode_logits, log_probs = decode_output
        assert log_probs is None
        _assert_logits(decode_logits, vocab_size=model.vocab_size)
        _assert_kv_changed(kv_before_decode, _kv_block_snapshot(kv_cache, decode_block))
    finally:
        executor.cleanup()
