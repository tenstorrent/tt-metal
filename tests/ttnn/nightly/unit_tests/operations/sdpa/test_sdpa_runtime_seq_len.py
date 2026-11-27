# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Non-causal SDPA with runtime_seq_len: one program serves every Q and K length.

Each case runs a sequence of (Sq, Sk) pairs against the same configuration in one process and
compares every output with torch. A new length adds a program only where it changes whether a core
may get more than one Q chunk, which sizes the Q buffer. The lengths cover whole and partial tiles and chunks, Sq != Sk,
fewer Q chunks than cores and more, and a GQA case whose K/V chains span several Q heads.
"""

import os
from pathlib import Path

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

PCC_MIN = 0.999


def _sdpa_kernel_builds() -> int:
    """Kernel binaries of the SDPA reader, writer and compute in the JIT cache, over all build keys.

    The JIT build writes each new binary to <cache>/<build key>/kernels/<kernel name>/<hash>/, so the
    count grows by one for every compile of these kernels.
    """
    if "TT_METAL_CACHE" in os.environ:
        root = Path(os.environ["TT_METAL_CACHE"]) / "tt-metal-cache"
    elif Path(os.environ.get("HOME", "")).is_dir():
        root = Path(os.environ["HOME"]) / ".cache" / "tt-metal-cache"
    else:
        root = Path("/tmp/tt-metal-cache")
    return sum(
        1 for name in ("sdpa", "reader_interleaved", "writer_interleaved") for _ in root.glob(f"*/kernels/{name}/*")
    )


def _run(
    device, *, b: int, nh: int, nkv: int, d: int, lengths: list[tuple[int, int]], q_chunk: int, k_chunk: int
) -> None:
    torch.manual_seed(1234)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )

    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    q_buffer_factors = set()
    for i, (sq, sk) in enumerate(lengths):
        q = torch.randn(b, nh, sq, d)
        k = torch.randn(b, nkv, sk, d)
        v = torch.randn(b, nkv, sk, d)
        tt_q, tt_k, tt_v = (
            ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for t in (q, k, v)
        )

        entries = device.num_program_cache_entries()
        builds = _sdpa_kernel_builds()
        out = ttnn.transformer.scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            is_causal=False,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            runtime_seq_len=True,
        )
        # A new length reuses the program of its Q buffer factor, and the kernels the first length compiled.
        q_buffer_factor = 2 if -(-b * nh * -(-sq // q_chunk) // num_cores) > 1 else 1
        added = device.num_program_cache_entries() - entries
        assert added == (0 if q_buffer_factor in q_buffer_factors else 1), f"Sq {sq}, Sk {sk}: {added} programs"
        q_buffer_factors.add(q_buffer_factor)
        if i > 0:
            assert _sdpa_kernel_builds() == builds, f"Sq {sq}, Sk {sk} compiled new SDPA kernels"

        got = ttnn.to_torch(out)[:, :, :sq, :]
        rep = nh // nkv
        expected = torch.nn.functional.scaled_dot_product_attention(
            q, k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1), is_causal=False
        )
        passing, pcc = comp_pcc(expected, got, PCC_MIN)
        assert passing, f"Sq {sq}, Sk {sk}: {pcc}"


# The shape of the VAE mid-block attention: one head of 512 channels, local queries against all keys.
@pytest.mark.parametrize(
    "lengths",
    [
        [(4608, 9216), (1024, 2048), (2516, 20128), (100, 1000), (8192, 8192), (33, 4128)],
    ],
    ids=["vae"],
)
def test_sdpa_runtime_seq_len_single_head(device, lengths: list[tuple[int, int]]) -> None:
    _run(device, b=1, nh=1, nkv=1, d=512, lengths=lengths, q_chunk=128, k_chunk=128)


# More Q chunks than cores (Q double-buffered), and K/V chains over the Q heads of a KV head.
@pytest.mark.parametrize(
    "lengths",
    [
        [(2048, 2048), (16384, 1024), (300, 777), (4096, 4096)],
    ],
    ids=["gqa"],
)
def test_sdpa_runtime_seq_len_gqa(device, lengths: list[tuple[int, int]]) -> None:
    _run(device, b=2, nh=8, nkv=2, d=128, lengths=lengths, q_chunk=128, k_chunk=256)


def test_sdpa_runtime_seq_len_rejects_causal(device) -> None:
    t = ttnn.from_torch(torch.randn(1, 1, 64, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(RuntimeError, match="runtime_seq_len"):
        ttnn.transformer.scaled_dot_product_attention(t, t, t, is_causal=True, runtime_seq_len=True)


# fp32_dest_acc_en=False selects the streaming compute path, which bakes the K length into its kernel.
def test_sdpa_runtime_seq_len_rejects_streaming(device) -> None:
    t = ttnn.from_torch(torch.randn(1, 1, 64, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    compute_kernel_config = ttnn.init_device_compute_kernel_config(device.arch(), fp32_dest_acc_en=False)
    with pytest.raises(RuntimeError, match="fp32_dest_acc_en"):
        ttnn.transformer.scaled_dot_product_attention(
            t, t, t, is_causal=False, compute_kernel_config=compute_kernel_config, runtime_seq_len=True
        )
