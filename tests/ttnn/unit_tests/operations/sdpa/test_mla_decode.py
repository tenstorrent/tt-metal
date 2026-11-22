# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import run_flash_mla_decode_impl


@pytest.mark.parametrize(
    "batch, seq_len, nh, nkv, kv_lora_rank, d_rope, q_num_cores, q_mem_config",
    # batch, seq_len, num heads q, num heads kv, kv lora rank, dim rope, number of cores to shard q on
    [
        (4, 1024, 128, 1, 512, 64, 64, None),  # DeepSeek V3 TG full DP
        (2, 1024, 8, 1, 128, 64, 0, None),  # small config, DRAM Q
    ],
)
@pytest.mark.parametrize(
    "q_dtype, dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b),
    ],
)
@pytest.mark.parametrize(
    "use_paged_attention",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "block_size",
    [
        64,
    ],
)
@pytest.mark.parametrize(
    "reuse_k",
    [
        True,
    ],
)
def test_flash_mla_decode(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_num_cores,
    q_dtype,
    q_mem_config,
    dtype,
    use_paged_attention,
    block_size,
    reuse_k,
    function_level_defaults,
    reset_seeds,
):
    run_flash_mla_decode_impl(
        device,
        batch,
        seq_len,
        nh,
        nkv,
        kv_lora_rank,
        d_rope,
        q_num_cores,
        q_dtype,
        q_mem_config,
        dtype,
        use_paged_attention,
        block_size,
        reuse_k,
        max_cores_per_head_batch=4,
    )


@pytest.mark.parametrize("batch, nh", [(4, 128), (2, 128)])
@pytest.mark.parametrize("use_paged_attention", [True, False])
def test_flash_mla_decode_column_groups(device, batch, nh, use_paged_attention, function_level_defaults, reset_seeds):
    grid = device.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < 8:
        pytest.skip("needs an 8x8 compute grid")
    run_flash_mla_decode_impl(
        device,
        batch,
        1024,
        nh,
        1,
        512,
        64,
        64,
        ttnn.bfloat16,
        None,
        ttnn.bfloat8_b,
        use_paged_attention=use_paged_attention,
        block_size=64,
        reuse_k=True,
        q_column_groups=True,
    )


@pytest.mark.parametrize("pos", [76, 3000])
def test_flash_mla_decode_kv_groups_as_batch(device, pos, function_level_defaults, reset_seeds):
    """GQA (8 kv heads x 3 q heads, one call) vs the kv heads as the batch (16-row Q tiles, a per-batch sink, exact
    exp): bit-equal outputs."""
    import torch

    nkv, g, w, vw, s = 8, 3, 288, 160, 4096
    kv = ttnn.from_torch(torch.randn(1, nkv, s, w), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    kv_b = ttnn.reshape(kv, [nkv, 1, s, w])
    qt = torch.randn(1, 1, nkv * g, w) * 0.1
    sink = torch.randn(nkv * g, 32)
    sink_b = torch.zeros(nkv, 32, 32)
    sink_b[:, :g] = sink.view(nkv, g, 32)
    to_dev = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    grid = device.compute_with_storage_grid_size()
    ck = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )

    def run(q, k, sink_t, n_pos, **opts):
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            q_chunk_size=0,
            k_chunk_size=128,
            exp_approx_mode=False,
            **opts,
        )
        cp = ttnn.from_torch(torch.tensor([pos] * n_pos, dtype=torch.int32), dtype=ttnn.int32, device=device)
        out = ttnn.transformer.flash_multi_latent_attention_decode(
            q,
            k,
            head_dim_v=vw,
            is_causal=True,
            cur_pos_tensor=cp,
            attention_sink=sink_t,
            scale=0.088,
            program_config=cfg,
            compute_kernel_config=ck,
        )
        return ttnn.to_torch(out).float()

    ref = run(to_dev(qt), kv, to_dev(sink), 1).view(nkv, g, vw)
    out = run(to_dev(qt.view(1, nkv, g, w)), kv_b, to_dev(sink_b.view(nkv * 32, 32)), nkv, exact_exp_half_tile=True)
    assert torch.equal(out.reshape(nkv, -1, vw)[:, :g], ref)
