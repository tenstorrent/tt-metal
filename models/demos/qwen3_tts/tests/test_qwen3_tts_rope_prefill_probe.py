# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Why prefill RoPE costs 40 us for Q and 25 us for K, and whether a reshape fixes it.

``rotary_embedding_llama_multi_core_program_factory.cpp:86`` splits the work like this::

    batch_parallel_factor = min(batch, num_cores)
    seq_parallel_factor   = min(num_cores / batch_parallel_factor, seq_len_t)
    num_rows_per_core     = ceil(seq_len_t / seq_parallel_factor) * n_heads

The head axis appears only as a MULTIPLIER on per-core work — it is never a parallel
axis. Talker prefill Q is ``[1, 16, 64, 128]``: batch=1, seq_len_t=2, so the factor is
1 x 2 = **2 cores**, each rotating 16 head-rows, and ``num_rows_per_core = 16 > 8`` also
selects ``use_reload_impl``, the slower path that re-reads cos/sin per row. That is why
the op measures the same ~40 us at seq=64 and seq=128: per-core work is identical.

The prefill validation only requires ``cos.shape[0] == 1`` and
``cos.shape[1] in (input.shape[1], 1)``, so a head-broadcast cos/sin stays legal if the
heads are moved into dim 0. Q as ``[16, 1, 64, 128]`` gives batch=16 -> 16 x 2 = 32 cores
and ``num_rows_per_core = 1``, taking the fast path too. ``[1,16,S,D]`` and ``[16,1,S,D]``
have identical linear tile order, so the reshape should be metadata only.

This probe checks all three things at once: does it run, is it BIT-EXACT against the
shipped call, and is it faster.

    python -m tracy -p -v -r --op-support-count 100000 \
        -m pytest -s -q models/demos/qwen3_tts/tests/test_qwen3_tts_rope_prefill_probe.py
"""

from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn

TILE = 32
HEAD_DIM = 128
HEADS, KV_HEADS = 16, 8
REPS = 8


@pytest.fixture(scope="module")
def device():
    d = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768)
    d.enable_program_cache()
    yield d
    ttnn.close_mesh_device(d)


def _kcfg():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )


@pytest.mark.parametrize("seq", [64, 128], ids=["seq64", "seq128"])
@pytest.mark.parametrize("nh", [HEADS, KV_HEADS], ids=["q16", "k8"])
def test_rope_prefill_head_to_batch(device, seq, nh):
    from models.demos.qwen3_tts.tt.rope import get_rope_tensors, get_transformation_mat

    torch.manual_seed(0)
    x = torch.randn(1, nh, seq, HEAD_DIM, dtype=torch.bfloat16)
    cos, sin = get_rope_tensors(device, HEAD_DIM, seq, torch.arange(seq), 1000000.0)
    trans = get_transformation_mat(HEAD_DIM, device)
    kc = _kcfg()
    seq_t = seq // TILE
    print(f"\n### nh={nh} seq={seq} (seq_len_t={seq_t})")
    print(
        f"    shipped [1,{nh},{seq},{HEAD_DIM}]: batch=1 -> parallel 1 x {min(64, seq_t)} = {min(64, seq_t)} cores,"
        f" rows/core={seq_t // min(64, seq_t) * nh}"
    )
    bpf = min(nh, 64)
    spf = min(64 // bpf, seq_t)
    print(
        f"    probe   [{nh},1,{seq},{HEAD_DIM}]: batch={nh} -> parallel {bpf} x {spf} = {bpf*spf} cores,"
        f" rows/core={(seq_t + spf - 1)//spf}"
    )

    results = {}
    for label, shape in (("shipped [1,nh,S,D]", (1, nh, seq, HEAD_DIM)), ("probe [nh,1,S,D]", (nh, 1, seq, HEAD_DIM))):
        xt = ttnn.from_torch(
            x.reshape(*shape).contiguous(),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        try:
            out = ttnn.experimental.rotary_embedding_llama(
                xt, cos, sin, trans, is_decode_mode=False, compute_kernel_config=kc
            )
            ttnn.synchronize_device(device)
        except Exception as e:
            msg = next((l.strip() for l in str(e).splitlines() if "TT_FATAL" in l), str(e)[:140])
            print(f"  {label:20s} -> REFUSED: {msg[:140]}")
            ttnn.deallocate(xt)
            continue
        t0 = time.perf_counter()
        for _ in range(REPS):
            o = ttnn.experimental.rotary_embedding_llama(
                xt, cos, sin, trans, is_decode_mode=False, compute_kernel_config=kc
            )
            ttnn.synchronize_device(device)
            ttnn.deallocate(o)
        wall = (time.perf_counter() - t0) / REPS * 1e6
        results[label] = ttnn.to_torch(out).reshape(1, nh, seq, HEAD_DIM).clone()
        print(f"  {label:20s} -> ran, {wall:.1f} us wall/launch (host-timed; read device time from the CSV)")
        ttnn.deallocate(out)
        ttnn.deallocate(xt)

    if len(results) == 2:
        a, b = results["shipped [1,nh,S,D]"], results["probe [nh,1,S,D]"]
        md = float((a.float() - b.float()).abs().max())
        print(f"  max|shipped - probe| = {md:.3e}  {'BIT-EXACT' if md == 0.0 else 'DIFFERS'}")
        assert md == 0.0, f"reshape changed the result (max diff {md})"
