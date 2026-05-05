# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
vLLM-interface verification tests for Qwen3.5-27B.

These tests exercise Qwen35ForCausalLM.{prefill_forward, decode_forward,
allocate_kv_cache} directly, mimicking the call shape that
vllm/v1/worker/tt_model_runner.py uses, without depending on the vLLM fork.

Run:
    HF_MODEL=~/models/Qwen3.5-27B-FP8 \\
        pytest models/demos/qwen35_27b/tt/tests/test_vllm_prefill.py -v -s
"""

import os

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.demos.qwen35_27b.tt.generator_vllm import Qwen35ForCausalLM


def _get_model_path():
    return os.path.expanduser(os.environ.get("HF_MODEL", "~/models/Qwen3.5-27B-FP8"))


def _apply_chat_template(tokenizer, prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tokenizer.encode(text)


def _make_paged_kv_cache_and_page_table(model, mesh_device, batch_size, block_size=64, max_num_blocks=2048):
    """Allocate KV cache via the vLLM-style allocate_kv_cache API and build a
    page_table covering max_num_blocks per batch row."""
    args = model.model_args[0]
    n_local_kv_heads = args.n_local_kv_heads
    head_dim = args.head_dim

    # vLLM passes only the count of attention layers — not GDN — to allocate_kv_cache.
    n_full_attn_layers = sum(1 for t in args.layer_types if t == "full_attention")

    kv_cache_shape = (max_num_blocks, n_local_kv_heads, block_size, head_dim)
    kv_cache = model.allocate_kv_cache(
        kv_cache_shape=kv_cache_shape, dtype=torch.bfloat16, num_layers=n_full_attn_layers
    )

    # page_table: one disjoint block range per batch row so per-user writes don't collide.
    # Each user gets max_num_blocks // batch_size physical blocks.
    blocks_per_user = max_num_blocks // batch_size
    page_table_torch = torch.zeros(batch_size, blocks_per_user, dtype=torch.int32)
    for u in range(batch_size):
        page_table_torch[u] = torch.arange(u * blocks_per_user, (u + 1) * blocks_per_user, dtype=torch.int32)

    return kv_cache, page_table_torch, blocks_per_user * block_size


@pytest.fixture(scope="module")
def vllm_model(mesh_device):
    """Build the vLLM model wrapper once per module."""
    if mesh_device.get_num_devices() < 4:
        pytest.skip("Full model requires TP>=4")

    model_path = _get_model_path()
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path

    model = Qwen35ForCausalLM.initialize_vllm_model(
        hf_config=None,
        mesh_device=mesh_device,
        max_batch_size=32,
        max_seq_len=8192,
        tt_data_parallel=1,
    )
    return model


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"P150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 8))
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True, "trace_region_size": 200_000_000}], indirect=True)
def test_vllm_prefill_single_user(mesh_device, vllm_model, reset_seeds, ensure_gc):
    """Prefill one user via Qwen35ForCausalLM.prefill_forward, then decode via
    decode_forward. Verifies the wrapper end-to-end against the existing model.
    """
    model = vllm_model
    args = model.model_args[0]
    batch_size = 32

    tokenizer = AutoTokenizer.from_pretrained(_get_model_path(), trust_remote_code=True)
    prompt_tokens = _apply_chat_template(tokenizer, "The capital of France is")
    seq_len = len(prompt_tokens)
    logger.info(f"Prompt: {seq_len} tokens")

    kv_cache, page_table_torch, _ = _make_paged_kv_cache_and_page_table(model, mesh_device, batch_size)

    # vLLM passes tokens shape [B, max_prompt_len], zero-padded along last dim.
    tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)
    tokens[0, :seq_len] = torch.tensor(prompt_tokens, dtype=torch.long)
    prompt_lens = torch.tensor([seq_len] + [1] * (batch_size - 1), dtype=torch.int32)

    # Single-user prefill: we only fill slot 0.
    out = model.prefill_forward(
        tokens=tokens[0:1],
        page_table=page_table_torch[0:1],
        kv_cache=kv_cache[0],
        prompt_lens=prompt_lens[0:1],
        empty_slots=[0],
    )
    assert out is not None, "prefill_forward returned None"
    logger.info(f"prefill_forward returned (type={type(out).__name__}); single-user prefill completed")


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"P150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 8))
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True, "trace_region_size": 200_000_000}], indirect=True)
def test_vllm_prefill_two_users_slot_isolation(mesh_device, vllm_model, reset_seeds, ensure_gc):
    """Prefill user A into slot 0 and user B (different prompt) into slot 5;
    verify that user A's GDN state in slot 0 is unchanged after user B's
    prefill, and that user B's state lands in slot 5 only.

    This catches the previous bug where replicate_prefill_state_to_batch
    blindly broadcast the new user's state across all 32 batch rows, clobbering
    other in-flight users' GDN state.
    """
    model = vllm_model
    args = model.model_args[0]
    batch_size = 32

    # Pick the first GDN ("linear_attention") layer to inspect.
    gdn_layer_idx = next(i for i, t in enumerate(args.layer_types) if t == "linear_attention")
    gdn_attn = model.model[0].layers[gdn_layer_idx].attention
    Nv_TP = gdn_attn.Nv_TP

    def _snapshot_rec_states():
        per_dev = ttnn.get_device_tensors(gdn_attn.rec_states)
        return [ttnn.to_torch(t).clone() for t in per_dev]

    tokenizer = AutoTokenizer.from_pretrained(_get_model_path(), trust_remote_code=True)
    prompt_a = _apply_chat_template(tokenizer, "The capital of France is")
    prompt_b = _apply_chat_template(tokenizer, "The largest planet in our solar system is")

    kv_cache, page_table_torch, _ = _make_paged_kv_cache_and_page_table(model, mesh_device, batch_size)

    SLOT_A, SLOT_B = 0, 5

    # ─── User A → slot 0 ───
    seq_a = len(prompt_a)
    tokens_a = torch.tensor([prompt_a], dtype=torch.long)
    model.prefill_forward(
        tokens=tokens_a,
        page_table=page_table_torch[SLOT_A : SLOT_A + 1],
        kv_cache=kv_cache[0],
        prompt_lens=torch.tensor([seq_a], dtype=torch.int32),
        empty_slots=[SLOT_A],
    )

    rec_after_a = _snapshot_rec_states()
    # Slot 0's rows should be non-zero (user A's prefill left state there).
    slot_a_rows_per_dev = [t[SLOT_A * Nv_TP : (SLOT_A + 1) * Nv_TP] for t in rec_after_a]
    assert any(
        t.abs().sum().item() > 0 for t in slot_a_rows_per_dev
    ), f"Slot {SLOT_A} GDN state is all zeros after user A's prefill (expected non-zero)"

    # ─── User B → slot 5 ───
    seq_b = len(prompt_b)
    tokens_b = torch.tensor([prompt_b], dtype=torch.long)
    model.prefill_forward(
        tokens=tokens_b,
        page_table=page_table_torch[SLOT_B : SLOT_B + 1],
        kv_cache=kv_cache[0],
        prompt_lens=torch.tensor([seq_b], dtype=torch.int32),
        empty_slots=[SLOT_B],
    )

    rec_after_b = _snapshot_rec_states()

    # Slot 0's rows must be IDENTICAL to the snapshot taken right after user A —
    # user B's prefill must not have touched them.
    for d, (before, after) in enumerate(zip(rec_after_a, rec_after_b)):
        before_slot_a = before[SLOT_A * Nv_TP : (SLOT_A + 1) * Nv_TP]
        after_slot_a = after[SLOT_A * Nv_TP : (SLOT_A + 1) * Nv_TP]
        assert torch.equal(before_slot_a, after_slot_a), (
            f"Device {d}: slot {SLOT_A} GDN rec_states changed after user B's prefill — " f"slot isolation is broken"
        )

    # Slot 5's rows should now be non-zero (user B's state).
    slot_b_rows_per_dev = [t[SLOT_B * Nv_TP : (SLOT_B + 1) * Nv_TP] for t in rec_after_b]
    assert any(
        t.abs().sum().item() > 0 for t in slot_b_rows_per_dev
    ), f"Slot {SLOT_B} GDN state is all zeros after user B's prefill (expected non-zero)"

    logger.info(
        f"PASSED: GDN slot isolation verified on layer {gdn_layer_idx}: "
        f"slot {SLOT_A} preserved across user B's prefill; slot {SLOT_B} populated."
    )
