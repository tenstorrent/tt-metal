# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in Tracy harnesses for the Qwen3.6 kernels targeted by issue #50475.

These tests load one real checkpoint layer, compile once, then place ``start`` / ``stop``
signposts around exactly one production-path invocation.  They are skipped unless
``QWEN36_KERNEL_PROFILE`` selects ``gdn_prefill`` or ``attention_prefill``.
"""

import os
import time

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.modules.tt_ccl import get_tt_ccl
from models.demos.blackhole.qwen36.tests.test_factory import (
    load_attn_layer,
    load_gdn_layer,
    model_path,
    parametrize_mesh_tp,
    shard_to_device,
)
from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_prefill
from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention, load_attention_weights_tp
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs


def _selected(component):
    if os.environ.get("QWEN36_KERNEL_PROFILE") != component:
        pytest.skip(f"set QWEN36_KERNEL_PROFILE={component} to run this Tracy harness")


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_prefill_profile(mesh_device, reset_seeds, ensure_gc):
    _selected("gdn_prefill")
    os.environ.setdefault("HF_MODEL", model_path())
    seq_len = int(os.environ.get("QWEN36_KERNEL_PROFILE_SEQ_LEN", "2048"))
    assert seq_len > 0 and seq_len % 128 == 0

    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=seq_len)
    layer_idx = next(i for i, kind in enumerate(args.attention_type_list) if kind == "linear_attention")
    weights = load_gdn_weights_tp(mesh_device, load_gdn_layer(args.CKPT_DIR, layer_idx), args)
    gdn = TPGatedDeltaNet(mesh_device, args, weights, get_tt_ccl(mesh_device))
    x = torch.randn(1, 1, seq_len, args.dim, dtype=torch.bfloat16)

    # Compile and populate the program cache outside the measured interval.
    out = gdn.forward_prefill(shard_to_device(mesh_device, x, dim=-1), chunk_size=128, borrow_output=True)
    ttnn.synchronize_device(mesh_device)
    if gdn.rec_state is not None:
        ttnn.deallocate(gdn.rec_state)
        gdn.rec_state = None

    signpost("start")
    begin = time.perf_counter()
    out = gdn.forward_prefill(shard_to_device(mesh_device, x, dim=-1), chunk_size=128, borrow_output=True)
    ttnn.synchronize_device(mesh_device)
    elapsed_ms = (time.perf_counter() - begin) * 1000.0
    signpost("stop")
    logger.info(f"GDN_PREFILL_PROFILE_RESULT layer={layer_idx} seq_len={seq_len} elapsed_ms={elapsed_ms:.3f}")
    assert out.shape[-2] == seq_len


@torch.no_grad()
@parametrize_mesh_tp()
def test_attention_prefill_profile(mesh_device, reset_seeds, ensure_gc):
    _selected("attention_prefill")
    os.environ.setdefault("HF_MODEL", model_path())
    seq_len = int(os.environ.get("QWEN36_KERNEL_PROFILE_SEQ_LEN", "2048"))
    chunk_start = int(os.environ.get("QWEN36_KERNEL_PROFILE_CHUNK_START", "0"))
    block_size = 64
    assert seq_len > 0 and seq_len % block_size == 0
    assert chunk_start >= 0 and chunk_start % seq_len == 0
    max_seq_len = chunk_start + seq_len

    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=max_seq_len)
    layer_idx = next(i for i, kind in enumerate(args.attention_type_list) if kind == "full_attention")
    weights = load_attention_weights_tp(mesh_device, load_attn_layer(args.CKPT_DIR, layer_idx), args)
    attention = TPAttention(mesh_device, args, weights, get_tt_ccl(mesh_device))
    num_blocks = (max_seq_len + block_size - 1) // block_size
    k_cache_dtype = ttnn.bfloat8_b if attention._sdpa_k_bf8 else ttnn.bfloat16
    v_cache_dtype = ttnn.bfloat8_b if attention._sdpa_v_bf8 else ttnn.bfloat16

    def make_cache(cache_dtype):
        return ttnn.from_torch(
            torch.zeros(num_blocks, args.n_local_kv_heads, block_size, args.head_dim, dtype=torch.bfloat16),
            dtype=cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    attention.set_paged_kv_cache(make_cache(k_cache_dtype), make_cache(v_cache_dtype))
    full_page_table = ttnn.from_torch(
        torch.arange(num_blocks, dtype=torch.int32).reshape(1, -1),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
    )
    first_chunk_block = chunk_start // block_size
    chunk_blocks = seq_len // block_size
    chunk_page_table = ttnn.from_torch(
        torch.arange(first_chunk_block, first_chunk_block + chunk_blocks, dtype=torch.int32).reshape(1, -1),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
    )
    x = torch.randn(1, 1, seq_len, args.dim, dtype=torch.bfloat16)
    cos, sin = rot_mats_prefill(mesh_device, args.rope_head_dim, seq_len, args.rope_theta)

    def run_once():
        return attention.forward_prefill_paged(
            shard_to_device(mesh_device, x, dim=-1),
            cos,
            sin,
            full_page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start,
            borrow_output=True,
        )

    out = run_once()
    ttnn.synchronize_device(mesh_device)

    signpost("start")
    begin = time.perf_counter()
    out = run_once()
    ttnn.synchronize_device(mesh_device)
    elapsed_ms = (time.perf_counter() - begin) * 1000.0
    signpost("stop")
    logger.info(
        f"ATTENTION_PREFILL_PROFILE_RESULT layer={layer_idx} seq_len={seq_len} "
        f"chunk_start={chunk_start} k_cache_dtype={k_cache_dtype} v_cache_dtype={v_cache_dtype} "
        f"elapsed_ms={elapsed_ms:.3f}"
    )
    assert out.shape[-2] == seq_len
