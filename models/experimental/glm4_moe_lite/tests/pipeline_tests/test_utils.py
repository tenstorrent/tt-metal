# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for GLM-4.7-Flash pipeline integration tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import ttnn
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.experimental.glm4_moe_lite.tt.layer0_tt import (
    _alloc_contiguous_page_table,
    _alloc_paged_kvpe_cache,
    _round_up,
)
from models.experimental.glm4_moe_lite.tt.model_tt import Glm4MoeLiteDenseOnlyTT
from models.experimental.glm4_moe_lite.tt.weights import find_missing_shards, resolve_best_effort_snapshot_dir

MODEL_ID = "zai-org/GLM-4.7-Flash"

TT_METAL_HOME = os.environ.get(
    "TT_METAL_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)
PROMPT_FILE = os.path.join(TT_METAL_HOME, "models/tt_transformers/tests/tale-of-two-cities.txt.bz2")

TRACE_REGION_SIZE_WORMHOLE = 30_000_000
TRACE_REGION_SIZE_BLACKHOLE = 35_000_000


def fabric_1d_trace_device_params(*, num_command_queues: int = 1):
    from models.common.utility_functions import is_blackhole, is_wormhole_b0

    trace_region_size = TRACE_REGION_SIZE_WORMHOLE if is_wormhole_b0() else TRACE_REGION_SIZE_BLACKHOLE
    if is_blackhole():
        num_command_queues = 2
    params: dict = {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "trace_region_size": trace_region_size,
        "num_command_queues": num_command_queues,
    }
    # QB-2 / p150 Blackhole: match physical 4-card count when using the 1x4 GLM mesh.
    if is_blackhole() and len(ttnn.get_device_ids()) == 4:
        params["require_exact_physical_num_devices"] = True
    return [params]


def mesh_shape_param():
    """Default mesh for pipeline tests.

    Blackhole QB-2 (4 cards): 1x4 — matches GLM production bring-up (not bh_1d_mesh_device's 4x1).
    Wormhole / other: 1 x num_devices, or MESH_DEVICE=P150x4 -> 1x4.
    """
    from models.common.utility_functions import is_blackhole

    env = os.environ.get("MESH_DEVICE")
    if env == "P150x4":
        return (1, 4)
    n = max(1, len(ttnn.get_device_ids()))
    if is_blackhole() and n == 4:
        return (1, 4)
    return n


def scale_page_params(page_params: dict, seq_len: int, batch_size: int) -> dict:
    block_size = int(page_params["page_block_size"])
    num_blocks = max(int(page_params["page_max_num_blocks"]), -(-seq_len // block_size))
    num_blocks = -(-num_blocks // batch_size) * batch_size
    return {"page_block_size": block_size, "page_max_num_blocks": num_blocks}


def require_snapshot() -> Path:
    snap = Path(resolve_best_effort_snapshot_dir(MODEL_ID))
    missing = find_missing_shards(snap)
    if missing:
        pytest.skip(f"GLM snapshot missing {len(missing)} shards; run ensure_glm47_weights.sh first.")
    return snap


def load_tokenizer(snapshot_dir: Path) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True, use_fast=True)


def load_hf_causal_lm(snapshot_dir: Path) -> AutoModelForCausalLM:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            snapshot_dir,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
    except Exception as exc:
        pytest.skip(f"Unable to load HF reference model from {snapshot_dir}: {exc}")
    model.eval()
    return model


@torch.no_grad()
def hf_prefill_last_token_logits(model: AutoModelForCausalLM, input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
    return outputs.logits[:, -1:, :vocab_size].to(dtype=torch.float32)


@torch.no_grad()
def hf_decode_logits(
    model: AutoModelForCausalLM,
    *,
    token_id: int,
    past_key_values,
    vocab_size: int,
) -> tuple[torch.Tensor, object]:
    input_ids = torch.tensor([[int(token_id)]], dtype=torch.long)
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )
    logits = outputs.logits[:, :, :vocab_size].to(dtype=torch.float32)
    return logits, outputs.past_key_values


def create_runner(
    *,
    mesh_device,
    snapshot_dir: Path,
    max_seq_len: int,
    cache_subdir: str,
) -> Glm4MoeLiteDenseOnlyTT:
    cache_dir = Path(os.path.expanduser(f"~/.cache/ttnn/models/glm4_moe_lite/{cache_subdir}"))
    runner = Glm4MoeLiteDenseOnlyTT.create(
        device=mesh_device,
        snapshot_dir=snapshot_dir,
        cache_dir=cache_dir,
        max_seq_len=int(max_seq_len),
    )
    for layer_idx in range(runner.num_layers_to_run):
        runner._ensure_layer_weights(layer_idx)
    return runner


def compute_max_seq_len(total_tokens: int, block_size: int = 64) -> int:
    blocks_per_seq = max(1, _round_up(total_tokens, block_size) // block_size)
    min_blocks_per_seq = max(1, _round_up(128, block_size) // block_size)
    blocks_per_seq = max(blocks_per_seq, min_blocks_per_seq)
    return int(blocks_per_seq * block_size)


def alloc_kv_cache_and_page_table(
    *,
    mesh_device,
    runner: Glm4MoeLiteDenseOnlyTT,
    batch_size: int,
    total_tokens: int,
    block_size: int = 64,
    kv_cache_dtype=ttnn.bfloat16,
):
    max_seq_len = compute_max_seq_len(total_tokens, block_size)
    blocks_per_seq = max_seq_len // block_size

    kvpe_dim = int(runner.hparams.kv_lora_rank + runner.hparams.qk_rope_head_dim)
    kv_cache = [
        _alloc_paged_kvpe_cache(
            device=mesh_device,
            max_num_blocks=int(batch_size * blocks_per_seq),
            block_size=block_size,
            kvpe_dim=kvpe_dim,
            dtype=kv_cache_dtype,
        )
        for _ in range(int(runner.num_layers_to_run))
    ]
    page_table = _alloc_contiguous_page_table(batch=batch_size, blocks_per_seq=blocks_per_seq)
    return kv_cache, page_table, max_seq_len


def apply_correctness_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Conservative runtime flags for full-model logits PCC vs HF."""
    monkeypatch.setenv("GLM4_MOE_LITE_ENABLE_MOE", "1")
    monkeypatch.setenv("GLM4_MOE_LITE_EXPERTS_TT_DTYPE", "bf16")
    monkeypatch.setenv("GLM4_MOE_LITE_MOE_FP32_ACC", "1")
    # Replicated weights on multi-card mesh (no TP) for straightforward HF logits compare.
    monkeypatch.setenv("GLM4_MOE_LITE_TP", "0")
    monkeypatch.setenv("GLM4_MOE_LITE_ATTN_DP", "0")


def apply_wh_correctness_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reference-precision runtime flags that FIT the Wormhole LoudBox (T3K).

    The Blackhole ``apply_correctness_env`` uses bf16 experts, which OOM WH DRAM
    (~7 GB/chip for experts alone, even EP-sharded). On WH the highest expert
    precision that fits is **bf8**, which reaches ~0.954 prefill-logits PCC vs HF
    (bf4=0.935, bf8=0.954; bf16≈0.97 but OOMs -> Blackhole-only).

    Like the Blackhole env this uses TP=0 (replicated, no tensor-parallel): the
    production TP=1 path currently drops PCC to ~0.65 (a separate accuracy bug),
    so the correctness reference deliberately isolates it with TP=0.

    Do NOT add MOE_SPARSE_FP32_ACC / MOE_SPARSE_FIDELITY / MOE_SPARSE_APPROX=0
    here: measured, they REGRESS PCC (0.954 -> 0.54), i.e. those precision paths
    are currently buggy.
    """
    monkeypatch.setenv("GLM4_MOE_LITE_ENABLE_MOE", "1")
    monkeypatch.setenv("GLM4_MOE_LITE_EXPERTS_TT_DTYPE", "bf8")  # bf16 OOMs WH; bf8 is the fitting max
    monkeypatch.setenv("GLM4_MOE_LITE_MOE_FP32_ACC", "1")
    monkeypatch.setenv("GLM4_MOE_LITE_TP", "0")
    monkeypatch.setenv("GLM4_MOE_LITE_ATTN_DP", "0")


def apply_wh_tp1_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """WH **1x8** tensor-parallel (TP=1) reference-precision env.

    The 2x4 mesh scores only ~0.65 PCC at TP=1 (2D-mesh MoE-reduce bug: the fused reduce
    covers only the col axis, missing the DP-row expert sum; a2a dispatch deadlocks). A
    **1x8** mesh is 1D, so the MoE all-reduce is single-axis (cols) — the same path that
    works on Blackhole 1x4 — which should make TP=1 accurate on all 8 WH chips.

    TP=8 does not divide num_attention_heads=20, so head-parallel attention is disabled
    (attention falls back to full-head; the hidden-dim-sharded projections still shard over
    the 8 columns). bf8 experts (bf16 OOMs WH). Must be run on the conftest mesh fixture
    (MeshShape(1, param) => 1x8) which sets FABRIC_1D via device_params; a bare
    set_fabric_config + manual open segfaults on 1x8.
    """
    monkeypatch.setenv("GLM4_MOE_LITE_ENABLE_MOE", "1")
    monkeypatch.setenv("GLM4_MOE_LITE_EXPERTS_TT_DTYPE", "bf8")
    monkeypatch.setenv("GLM4_MOE_LITE_MOE_FP32_ACC", "1")
    monkeypatch.setenv("GLM4_MOE_LITE_TP", "1")
    monkeypatch.setenv("GLM4_MOE_LITE_ATTN_DP", "0")
    monkeypatch.setenv("GLM4_MOE_LITE_HEAD_PARALLEL_ATTN", "0")  # 20 heads not divisible by TP=8
    monkeypatch.setenv("GLM4_MOE_LITE_HEAD_PARALLEL_KVB2", "0")


def apply_single_layer_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLM4_MOE_LITE_DEBUG_ALLOW_PARTIAL_LAYERS", "1")
    monkeypatch.setenv("GLM4_MOE_LITE_NUM_LAYERS", "1")
    monkeypatch.setenv("GLM4_MOE_LITE_ENABLE_MOE", "0")
