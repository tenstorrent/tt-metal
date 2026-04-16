# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Decode-only baseline for ``ttnn.experimental.rotary_embedding_hf`` (no sweep framework).

Deterministic tensors are generated once and saved under ``rotary_embedding_hf_fixtures/``.
Each run reloads the same files so PCC tracks kernel changes, not input noise.

Attention-aligned path: explicit height-sharded Q/K (``nearest_32`` head padding),
sharded cos/sin like ``HfRotarySetupNew.get_rot_mats``, HiFi4 compute config,
no ``memory_config`` on the op (matches ``Attention._hf_rope_new_decode``).

Regenerate fixtures: ``HF_ROPE_DECODE_FIXTURE_REGEN=1 pytest ...`` (or delete the fixture directory).

**Machine-readable summary (always printed):** each parametrized case emits a single tab-separated line
prefixed with ``HF_ROPE_DECODE_BASELINE_TSV`` (columns: batch, num_heads, head_dim, min_pcc, mean_pcc,
max_pcc). Use ``pytest -s`` so stdout is not captured. Parse logs with ``grep '^HF_ROPE_DECODE_BASELINE_TSV'``
(no regex on numeric fields).
"""

from __future__ import annotations

import os
import pathlib

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole, nearest_32
from ttnn.types import BlackholeComputeKernelConfig

FIXTURE_DIR = pathlib.Path(__file__).parent / "rotary_embedding_hf_fixtures"
# Log line prefix for deterministic extraction (tab-separated batch, heads, dim, min/mean/max PCC).
BASELINE_TSV_PREFIX = "HF_ROPE_DECODE_BASELINE_TSV"
N_SEEDS = 5
# Low bar so CI/dev can land the harness before the kernel is fixed; raise as PCC improves.
PCC_THRESHOLD = 0.90

CONFIGS = [
    (1, 32, 128),
    (8, 32, 128),
    (32, 32, 128),
    (1, 8, 128),
    (8, 8, 128),
    (32, 8, 128),
    (1, 32, 64),
    (8, 16, 64),
    (32, 8, 64),
]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_hf(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


def _fixture_rng_seed(batch: int, num_heads: int, head_dim: int, seed_idx: int) -> int:
    """Unique deterministic seed per (batch, heads, dim, seed_idx)."""
    return (batch * 7919 + num_heads * 6247 + head_dim * 9973 + seed_idx * 100_003) % (2**31 - 1)


def _fixture_path(batch: int, num_heads: int, head_dim: int, seed_idx: int) -> pathlib.Path:
    return FIXTURE_DIR / f"b{batch}_h{num_heads}_d{head_dim}_s{seed_idx}.pt"


def generate_all_fixtures() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for batch, num_heads, head_dim in CONFIGS:
        for seed_idx in range(N_SEEDS):
            path = _fixture_path(batch, num_heads, head_dim, seed_idx)
            torch.manual_seed(_fixture_rng_seed(batch, num_heads, head_dim, seed_idx))
            inp = torch.randn(1, batch, num_heads, head_dim, dtype=torch.float32)
            cos = torch.randn(1, batch, 1, head_dim, dtype=torch.float32)
            sin = torch.randn(1, batch, 1, head_dim, dtype=torch.float32)
            cos_e = cos.expand(-1, -1, num_heads, -1)
            sin_e = sin.expand(-1, -1, num_heads, -1)
            golden = apply_rotary_pos_emb_hf(inp, cos_e, sin_e)
            torch.save(
                {
                    "batch": batch,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "seed_idx": seed_idx,
                    "input": inp,
                    "cos": cos,
                    "sin": sin,
                    "golden": golden,
                },
                path,
            )


def load_fixture(batch: int, num_heads: int, head_dim: int, seed_idx: int) -> dict:
    path = _fixture_path(batch, num_heads, head_dim, seed_idx)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing fixture {path}. Run with HF_ROPE_DECODE_FIXTURE_REGEN=1 once or run tests to auto-generate."
        )
    return torch.load(path, map_location="cpu")


def _attention_rotary_embedding_hf_compute_kernel_config():
    cls = BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    return cls(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _decode_qk_heads_mem_config(device, batch: int, num_heads: int, head_dim: int) -> ttnn.MemoryConfig:
    """HEIGHT-sharded L1 for decode Q/K (B1 sweep / explicit WH shard)."""
    padded_heads = nearest_32(num_heads)
    shard_h = padded_heads
    if is_blackhole():
        return ttnn.create_sharded_memory_config(
            shape=(shard_h, head_dim),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    grid_size = device.compute_with_storage_grid_size()
    batch_grid = ttnn.num_cores_to_corerangeset(batch, grid_size, row_wise=True)
    return ttnn.create_sharded_memory_config(
        shape=(padded_heads, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _decode_hf_cos_sin_sharded(device, batch: int, head_dim: int, cos_torch, sin_torch, *, dtype):
    """Match ``HfRotarySetupNew.get_rot_mats`` sharding: HEIGHT (TILE_SIZE, head_dim) on batch core grid."""
    core_grid = device.compute_with_storage_grid_size()
    num_cores = min(batch, core_grid.x * core_grid.y)
    batch_grid = ttnn.num_cores_to_corerangeset(num_cores, core_grid, row_wise=True)
    mem_config = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    pad_h = ttnn.TILE_SIZE - cos_torch.shape[2]
    if pad_h > 0:
        z = torch.zeros(1, batch, pad_h, head_dim, dtype=cos_torch.dtype, device=cos_torch.device)
        cos_torch = torch.cat([cos_torch, z], dim=2)
        sin_torch = torch.cat([sin_torch, z], dim=2)
    cos_interleaved = ttnn.from_torch(
        cos_torch,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sin_interleaved = ttnn.from_torch(
        sin_torch,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if batch % ttnn.TILE_SIZE != 0:
        cos_interleaved = cos_interleaved[:, :batch, :, :]
        sin_interleaved = sin_interleaved[:, :batch, :, :]
    cos_tensor = ttnn.interleaved_to_sharded(cos_interleaved, mem_config)
    sin_tensor = ttnn.interleaved_to_sharded(sin_interleaved, mem_config)
    return cos_tensor, sin_tensor


@pytest.fixture(scope="session", autouse=True)
def ensure_decode_rope_fixtures():
    regen = os.environ.get("HF_ROPE_DECODE_FIXTURE_REGEN", "").lower() in ("1", "true", "yes")
    if regen and FIXTURE_DIR.exists():
        for p in FIXTURE_DIR.glob("*.pt"):
            p.unlink()
    if regen or not FIXTURE_DIR.exists() or not any(FIXTURE_DIR.glob("*.pt")):
        generate_all_fixtures()


@pytest.mark.parametrize("batch,num_heads,head_dim", CONFIGS)
def test_rotary_embedding_hf_decode_baseline(batch, num_heads, head_dim, device):
    dtype = ttnn.bfloat16
    rope_cfg = _attention_rotary_embedding_hf_compute_kernel_config()
    pccs: list[float] = []

    for seed_idx in range(N_SEEDS):
        data = load_fixture(batch, num_heads, head_dim, seed_idx)
        torch_input = data["input"].to(torch.float32)
        torch_cos = data["cos"].to(torch.float32)
        torch_sin = data["sin"].to(torch.float32)
        torch_golden = data["golden"].to(torch.float32)

        padded_heads = nearest_32(num_heads)
        inp_for_dev = torch_input
        if padded_heads != num_heads:
            pad_h = padded_heads - num_heads
            z = torch.zeros(1, batch, pad_h, head_dim, dtype=torch_input.dtype)
            inp_for_dev = torch.cat([torch_input, z], dim=2)

        qk_mem = _decode_qk_heads_mem_config(device, batch, num_heads, head_dim)
        input_tensor = ttnn.from_torch(
            inp_for_dev.to(torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=qk_mem,
        )
        cos_tt, sin_tt = _decode_hf_cos_sin_sharded(
            device,
            batch,
            head_dim,
            torch_cos.to(torch.bfloat16),
            torch_sin.to(torch.bfloat16),
            dtype=dtype,
        )

        out_tt = ttnn.experimental.rotary_embedding_hf(
            input_tensor,
            cos_tt,
            sin_tt,
            is_decode=True,
            compute_kernel_config=rope_cfg,
        )
        out_torch = ttnn.to_torch(out_tt).to(torch.float32)
        if padded_heads != num_heads:
            out_torch = out_torch[:, :, :num_heads, :]

        _, pcc_val = comp_pcc(torch_golden, out_torch, pcc=0.0)
        pccs.append(float(pcc_val))

        ttnn.deallocate(out_tt)
        ttnn.deallocate(input_tensor)
        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)

    min_pcc = min(pccs)
    mean_pcc = sum(pccs) / len(pccs)
    max_pcc = max(pccs)
    # TSV first: stable prefix + tabs only — grep/cut/sort friendly (no regex on floats).
    print(
        f"{BASELINE_TSV_PREFIX}\t{batch}\t{num_heads}\t{head_dim}\t{min_pcc}\t{mean_pcc}\t{max_pcc}",
        flush=True,
    )
    print(
        f"rotary_embedding_hf decode baseline: batch={batch} num_heads={num_heads} head_dim={head_dim} "
        f"min_pcc={min_pcc:.6f} mean_pcc={mean_pcc:.6f} max_pcc={max_pcc:.6f}",
        flush=True,
    )
    assert min_pcc >= PCC_THRESHOLD, (
        f"min PCC {min_pcc} below threshold {PCC_THRESHOLD} " f"(mean={mean_pcc:.6f} max={max_pcc:.6f} per-seed={pccs})"
    )
