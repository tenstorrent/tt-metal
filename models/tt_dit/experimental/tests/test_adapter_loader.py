# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for the runtime-LoRA adapter loader.

Builds a tiny WanTransformer3DModel (1 block) with ``lora_enabled=True``,
writes a synthetic LoRA safetensors file, runs ``load_adapter_into``, and
checks that:

  - every LoRA-aware Linear in the block got an adapter registered
  - ``bind_active`` marks the right modules active across all paths
  - ``unbind_active`` clears the active state

The forward math is covered by ``test_lora_variants.py``; this file
exercises the loader's key-mapping logic and the pipeline-mixin bind
flow against the default (``fuse``) mode.

Run with:
    pytest -xvs models/tt_dit/experimental/tests/test_adapter_loader.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

import ttnn
from models.tt_dit.experimental.lora.adapter_loader import load_adapter_into
from models.tt_dit.layers.lora import LoRAMixin
from models.tt_dit.models.transformers.wan2_2.transformer_wan import WanTransformer3DModel
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager


def _save_synthetic_adapter(
    path: Path,
    *,
    num_blocks: int = 1,
    dim: int = 128,
    ffn_dim: int = 256,
    rank: int = 8,
) -> None:
    """Write a fake LoRA safetensors that covers all 6 LoRA-targeted Linears
    per block, using the lightx2v naming convention."""
    from safetensors.torch import save_file

    state: dict[str, torch.Tensor] = {}
    dtype = torch.bfloat16

    def add_pair(base: str, in_dim: int, out_dim: int):
        state[f"{base}.lora_A.weight"] = torch.randn(rank, in_dim, dtype=dtype) * 0.02
        state[f"{base}.lora_B.weight"] = torch.randn(out_dim, rank, dtype=dtype) * 0.02

    for i in range(num_blocks):
        # self-attn Q, K, V, O
        add_pair(f"blocks.{i}.self_attn.q", dim, dim)
        add_pair(f"blocks.{i}.self_attn.k", dim, dim)
        add_pair(f"blocks.{i}.self_attn.v", dim, dim)
        add_pair(f"blocks.{i}.self_attn.o", dim, dim)
        # cross-attn Q, K, V, O
        add_pair(f"blocks.{i}.cross_attn.q", dim, dim)
        add_pair(f"blocks.{i}.cross_attn.k", dim, dim)
        add_pair(f"blocks.{i}.cross_attn.v", dim, dim)
        add_pair(f"blocks.{i}.cross_attn.o", dim, dim)
        # ffn ff1, ff2 (lightx2v naming: ffn.0 = ff1, ffn.2 = ff2)
        add_pair(f"blocks.{i}.ffn.0", dim, ffn_dim)
        add_pair(f"blocks.{i}.ffn.2", ffn_dim, dim)

    save_file(state, str(path))


def _count_registered(transformer) -> int:
    """Count LoRA modules with at least one entry in their bank."""
    n = 0
    for mod in _iter_lora_modules(transformer):
        if any(a is not None for a in mod.lora_bank):
            n += 1
    return n


def _iter_lora_modules(module):
    if isinstance(module, LoRAMixin):
        yield module
    for _, child in module.named_children():
        yield from _iter_lora_modules(child)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_adapter_loader_registers_all_lora_linears(mesh_device: ttnn.MeshDevice) -> None:
    """Loader picks up Q/K/V/O for both attn1/attn2 plus ff1/ff2 per block."""
    num_heads = 4
    head_dim = 32
    dim = num_heads * head_dim
    ffn_dim = 2 * dim
    rank = 8
    num_blocks = 1

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=1, mesh_axis=1),
        sequence_parallel=ParallelFactor(factor=1, mesh_axis=0),
    )
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)

    transformer = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_heads=num_heads,
        dim=dim,
        in_channels=16,
        out_channels=16,
        text_dim=128,
        freq_dim=64,
        ffn_dim=ffn_dim,
        num_layers=num_blocks,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=False,
        lora_enabled=True,
    )

    # Every block should have 5 LoRA-aware Linears wired in: to_qkv (self),
    # to_q (cross), to_kv (cross), to_out for each attention (×2), ff1, ff2.
    # That's 1 + 1 + 1 + 2 + 1 + 1 = 7 modules per block.
    expected_per_block = 7
    n_modules = sum(1 for _ in _iter_lora_modules(transformer))
    assert (
        n_modules == num_blocks * expected_per_block
    ), f"expected {num_blocks * expected_per_block} LoRA modules; got {n_modules}"

    with tempfile.TemporaryDirectory() as tmp:
        adapter_path = Path(tmp) / "synthetic.safetensors"
        _save_synthetic_adapter(adapter_path, num_blocks=num_blocks, dim=dim, ffn_dim=ffn_dim, rank=rank)

        handle = load_adapter_into(transformer, str(adapter_path), scale=1.0, name="synthetic")

    # All 7 LoRA modules per block should have a registered adapter.
    registered = _count_registered(transformer)
    assert (
        registered == num_blocks * expected_per_block
    ), f"expected {num_blocks * expected_per_block} registered adapters; got {registered}"

    # Fused-QKV (to_qkv self-attn): rank should be 3*r. Fused KV (cross-attn
    # to_kv): rank should be 2*r. Singletons: rank = r.
    block0 = transformer.blocks[0]
    assert block0.attn1.to_qkv.lora_bank[0].rank == 3 * rank
    assert block0.attn2.to_kv.lora_bank[0].rank == 2 * rank
    assert block0.attn2.to_q.lora_bank[0].rank == rank
    assert block0.attn1.to_out.lora_bank[0].rank == rank
    assert block0.attn2.to_out.lora_bank[0].rank == rank
    assert block0.ffn.ff1.lora_bank[0].rank == rank
    assert block0.ffn.ff2.lora_bank[0].rank == rank

    # handle.rank tracks the max rank across targets (3*rank for the fused-QKV group).
    assert handle.rank == 3 * rank


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_pipeline_bind_unbind_walks_all_modules(mesh_device: ttnn.MeshDevice) -> None:
    """The pipeline mixin's bind/unbind walks every LoRA module on each set_active_lora."""
    from models.tt_dit.experimental.pipelines.pipeline_wan_runtime_lora import _iter_lora_modules as pipeline_iter

    num_heads = 4
    head_dim = 32
    dim = num_heads * head_dim
    ffn_dim = 2 * dim
    rank = 8

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=1, mesh_axis=1),
        sequence_parallel=ParallelFactor(factor=1, mesh_axis=0),
    )
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)

    transformer = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_heads=num_heads,
        dim=dim,
        in_channels=16,
        out_channels=16,
        text_dim=128,
        freq_dim=64,
        ffn_dim=ffn_dim,
        num_layers=1,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=False,
        lora_enabled=True,
    )

    with tempfile.TemporaryDirectory() as tmp:
        adapter_path = Path(tmp) / "synthetic.safetensors"
        _save_synthetic_adapter(adapter_path, num_blocks=1, dim=dim, ffn_dim=ffn_dim, rank=rank)
        handle = load_adapter_into(transformer, str(adapter_path), name="synthetic")

    # Simulate the bind path the pipeline mixin runs.
    for dotted, bank_idx in handle.target_indices.items():
        cur = transformer
        for part in dotted.split("."):
            cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
        cur.bind_active(bank_idx)

    # All modules covered by the handle should be active (delta merged).
    for dotted in handle.target_indices:
        cur = transformer
        for part in dotted.split("."):
            cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
        assert cur.is_lora_active, f"{dotted} not active after bind_active"

    # Unbind everything → delta subtracted, is_lora_active False.
    for mod in pipeline_iter(transformer):
        if mod.is_lora_active:
            mod.unbind_active()
    for dotted in handle.target_indices:
        cur = transformer
        for part in dotted.split("."):
            cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
        assert not cur.is_lora_active
        assert cur.active_idx is None
