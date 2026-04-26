# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ttnn

from models.experimental.glm4_moe_lite.tt.config import Glm4MoeLiteHParams
from models.experimental.glm4_moe_lite.tt.model_tt import Glm4MoeLiteDenseOnlyTT
from models.experimental.glm4_moe_lite.tt.layer0_tt import (
    _alloc_contiguous_page_table,
    _alloc_paged_kvpe_cache,
    _round_up,
)
from models.experimental.glm4_moe_lite.tt.weights import find_missing_shards, resolve_best_effort_snapshot_dir


def _load_hparams(snapshot_dir: Path) -> Glm4MoeLiteHParams:
    cfg = json.loads((Path(snapshot_dir) / "config.json").read_text())
    hparams = Glm4MoeLiteHParams.from_hf_config(SimpleNamespace(**cfg))
    hparams.validate()
    return hparams


def _default_golden_path(snapshot_dir: Path, *, num_layers: int, max_new_tokens: int) -> Path:
    snap_name = Path(snapshot_dir).name
    root = Path(os.path.expanduser("~/.cache/ttnn/models/glm4_moe_lite/golden"))
    return root / f"golden_tokens_{snap_name}_layers{num_layers}_new{max_new_tokens}.json"


def _assert_next_token_matches_or_is_near_tie(
    *,
    step_label: str,
    logits: torch.Tensor,  # [1,1,V]
    expected_token_id: int,
    topk: int = 5,
    max_logit_gap: float = 0.75,
) -> int:
    """Assert greedy token matches expected, or the expected is a very close runner-up.

    Why: exact greedy tokens can be unstable across backends for MoE models due to near-ties.
    We use teacher forcing for sequence alignment and allow only *small* logit gaps.
    """
    # Some paths return [1,1,1,V] (extra singleton sequence axis). Normalize.
    if logits.ndim == 4 and int(logits.shape[2]) == 1:
        logits = logits.squeeze(2)
    if logits.ndim != 3 or int(logits.shape[0]) != 1 or int(logits.shape[1]) != 1:
        raise ValueError(f"{step_label}: expected logits shape [1,1,V], got {tuple(logits.shape)}")

    log = logits[0, 0].to(dtype=torch.float32)
    pred_token_id = int(log.argmax().item())
    if pred_token_id == int(expected_token_id):
        return pred_token_id

    k = min(int(topk), int(log.numel()))
    topv, topi = torch.topk(log, k=k)
    topi_list = [int(x) for x in topi.tolist()]
    topv_list = [float(x) for x in topv.tolist()]

    expected_token_id = int(expected_token_id)
    expected_logit = float(log[expected_token_id].item())
    pred_logit = float(log[pred_token_id].item())
    gap = pred_logit - expected_logit

    if expected_token_id not in set(topi_list) or gap > float(max_logit_gap):
        pairs = list(zip(topi_list, [round(x, 4) for x in topv_list]))
        raise AssertionError(
            f"{step_label}: expected={expected_token_id} pred={pred_token_id} "
            f"expected_logit={expected_logit:.4f} pred_logit={pred_logit:.4f} gap={gap:.4f} "
            f"top{int(k)}={pairs}"
        )

    return pred_token_id


@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_GOLDEN_TESTS") != "1",
    reason="Enable with TT_ENABLE_GOLDEN_TESTS=1 (runs slow correctness test vs offline golden tokens).",
)
@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_HW_TESTS") != "1",
    reason="Enable with TT_ENABLE_HW_TESTS=1 (requires Tenstorrent device access).",
)
@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_LARGE_MODEL_TESTS") != "1",
    reason="Enable with TT_ENABLE_LARGE_MODEL_TESTS=1 (loads expert weights).",
)
def test_tt_truncated_2layer_greedy_matches_offline_golden(monkeypatch: pytest.MonkeyPatch) -> None:
    snap = resolve_best_effort_snapshot_dir("zai-org/GLM-4.7-Flash")
    missing = find_missing_shards(snap)
    if missing:
        pytest.skip(f"GLM snapshot missing {len(missing)} shards; run ensure_glm47_weights.sh first.")

    num_layers = 2
    max_new_tokens = 32
    golden_path = _default_golden_path(Path(snap), num_layers=num_layers, max_new_tokens=max_new_tokens)
    if not golden_path.is_file():
        pytest.skip(f"Golden file not found: {golden_path}. Run generate_truncated_golden_tokens.py first.")

    golden = json.loads(golden_path.read_text())
    record = golden["records"][0]
    prompt_ids = torch.tensor([record["prompt_input_ids"]], dtype=torch.int32)
    expected = list(record["generated_ids"])

    monkeypatch.setenv("GLM4_MOE_LITE_ENABLE_MOE", "1")
    monkeypatch.setenv("GLM4_MOE_LITE_NUM_LAYERS", str(num_layers))
    monkeypatch.setenv("GLM4_MOE_LITE_DEBUG_ALLOW_PARTIAL_LAYERS", "1")
    monkeypatch.setenv("GLM4_MOE_LITE_EXPERTS_TT_DTYPE", "bf16")
    # Correctness-first precision for bring-up; required to avoid greedy flips on near-ties.
    monkeypatch.setenv("GLM4_MOE_LITE_MOE_FP32_ACC", "1")

    hparams = _load_hparams(Path(snap))

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        physical_device_ids=[0],
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )
    try:
        # Allocate minimal KV cache for (prompt + decode) length.
        block_size = 64
        prompt_len = int(prompt_ids.shape[1])
        total_len = prompt_len + int(max_new_tokens)
        blocks_per_seq = max(1, _round_up(total_len, block_size) // block_size)
        # Match vLLM-style behavior: allocate at least 128 tokens worth of blocks to avoid
        # short-sequence edge cases in paged-decode kernels.
        min_blocks_per_seq = max(1, _round_up(128, block_size) // block_size)
        blocks_per_seq = max(blocks_per_seq, min_blocks_per_seq)

        kvpe_dim = int(hparams.kv_lora_rank + hparams.qk_rope_head_dim)
        kv_cache = [
            _alloc_paged_kvpe_cache(
                device=mesh_device,
                max_num_blocks=int(1 * blocks_per_seq),
                block_size=block_size,
                kvpe_dim=kvpe_dim,
                dtype=ttnn.bfloat16,
            )
            for _ in range(num_layers)
        ]
        page_table = _alloc_contiguous_page_table(batch=1, blocks_per_seq=blocks_per_seq)

        runner = Glm4MoeLiteDenseOnlyTT.create(
            device=mesh_device,
            snapshot_dir=Path(snap),
            cache_dir=Path(os.path.expanduser("~/.cache/ttnn/models/glm4_moe_lite/truncated2_tt_cache")),
            max_seq_len=int(blocks_per_seq * block_size),
            hparams=hparams,
        )

        # Prefill -> compare logits for the first generated token.
        logits = runner.prefill(
            tokens=prompt_ids,
            prompt_lens=[prompt_len],
            page_table=page_table,
            kv_cache=kv_cache,
            seq_pad_multiple=block_size,
        )
        _assert_next_token_matches_or_is_near_tie(
            step_label="prefill",
            logits=logits,
            expected_token_id=int(expected[0]),
        )

        # Decode loop (teacher forced): we always feed the golden token to keep KV cache in-sync.
        token_in = int(expected[0])
        for step in range(max_new_tokens - 1):
            start_pos = torch.tensor([prompt_len + step], dtype=torch.int32)
            tokens = torch.tensor([[token_in]], dtype=torch.int32)
            logits = runner.decode(tokens=tokens, start_pos=start_pos, page_table=page_table, kv_cache=kv_cache)
            _assert_next_token_matches_or_is_near_tie(
                step_label=f"decode_step{step}",
                logits=logits,
                expected_token_id=int(expected[step + 1]),
            )
            token_in = int(expected[step + 1])

    finally:
        # Best-effort cleanup.
        try:
            for t in kv_cache:
                ttnn.deallocate(t)
        except Exception:
            pass
        ttnn.close_mesh_device(mesh_device)

    # Assertions are performed step-by-step above.
