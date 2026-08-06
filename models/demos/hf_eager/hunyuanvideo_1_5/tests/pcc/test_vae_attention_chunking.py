# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Equivalence and memory-scaling tests for chunked VAE mid-block attention.

The host cases are pure math and run without a device.  The device cases at the
bottom follow ``test_vae_spatial_sharding.py`` and need a real mesh.
"""

import os
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.hf_eager.hunyuanvideo_1_5.tt.vae_decoder import AttnBlock, _block_causal_mask
from models.demos.hf_eager.hunyuanvideo_1_5.tt.vae_spatial import (
    SpatialShardPlan,
    attention_chunk_tokens_from_env,
    block_causal_chunk_plan,
    chunk_plan_peak_score_elements,
    replicate_pad_to_plan,
    tile_padded,
    unchunked_score_elements,
)
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.tensor import fast_device_to_host, typed_tensor_2dshard

BF16_BYTES = 2

# (latent_frames, latent_h, latent_w) actually decoded on Blackhole Galaxy.
PRODUCTION_LATENTS = {
    "480p": (31, 30, 53),
    "720p": (31, 45, 80),
}


# ---------------------------------------------------------------------------
# Host-only torch oracles
# ---------------------------------------------------------------------------


def full_attention_reference(q, k, v, n_frame, n_hw):
    """Monolithic block-causal attention, mirroring the current TTNN path.

    ``q``/``k``/``v`` are ``(B, seq, C)``.  The additive mask is the same tensor
    ``AttnBlock`` uploads today.
    """
    scale = q.shape[-1] ** -0.5
    mask = _block_causal_mask(n_frame, n_hw, dtype=q.dtype).to(q.device)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale + mask
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def chunked_attention_reference(q, k, v, n_frame, n_hw, q_chunk_tokens):
    """Torch mirror of ``AttnBlock._chunked_attention``, op for op."""
    scale = q.shape[-1] ** -0.5
    plan = block_causal_chunk_plan(n_frame, n_hw, q_chunk_tokens)
    keys_t = k.transpose(-2, -1)
    blocks = []
    for chunk in plan:
        q_block = q[:, chunk.q_start : chunk.q_stop, :]
        scores = torch.matmul(q_block, keys_t[:, :, : chunk.kv_stop]) * scale
        blocks.append(torch.matmul(torch.softmax(scores, dim=-1), v[:, : chunk.kv_stop, :]))
    return torch.cat(blocks, dim=1)


def _qkv(batch, n_frame, n_hw, channels, seed, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    shape = (batch, n_frame * n_hw, channels)
    return tuple(torch.randn(shape, generator=generator, dtype=dtype) for _ in range(3))


# ---------------------------------------------------------------------------
# Chunk plan structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_frame,n_hw", [(1, 1), (3, 8), (5, 24), (31, 64), (31, 1590), (31, 3600)])
@pytest.mark.parametrize("q_chunk_tokens", [0, 1, 7, 32, 512, 1024, 4096])
def test_chunk_plan_partitions_the_query_sequence_exactly(n_frame, n_hw, q_chunk_tokens):
    plan = block_causal_chunk_plan(n_frame, n_hw, q_chunk_tokens)
    seq = n_frame * n_hw

    assert plan[0].q_start == 0
    assert plan[-1].q_stop == seq
    for previous, current in zip(plan, plan[1:]):
        assert current.q_start == previous.q_stop
    assert sum(chunk.q_len for chunk in plan) == seq
    assert all(chunk.q_len > 0 for chunk in plan)


@pytest.mark.parametrize("n_frame,n_hw", [(1, 1), (3, 8), (5, 24), (31, 64)])
@pytest.mark.parametrize("q_chunk_tokens", [0, 1, 7, 32, 512])
def test_every_chunk_stays_inside_one_frame_so_no_mask_is_needed(n_frame, n_hw, q_chunk_tokens):
    """A block may drop the mask only if all its rows share a key prefix."""
    plan = block_causal_chunk_plan(n_frame, n_hw, q_chunk_tokens)
    for chunk in plan:
        assert chunk.q_start // n_hw == chunk.frame
        assert (chunk.q_stop - 1) // n_hw == chunk.frame
        assert chunk.kv_stop == (chunk.frame + 1) * n_hw


@pytest.mark.parametrize("n_frame,n_hw", [(1, 1), (3, 8), (5, 24), (7, 12)])
@pytest.mark.parametrize("q_chunk_tokens", [0, 1, 5, 24, 4096])
def test_prefix_slicing_selects_exactly_the_unmasked_mask_entries(n_frame, n_hw, q_chunk_tokens):
    """Key prefixes must equal the finite region of the production mask."""
    mask = _block_causal_mask(n_frame, n_hw)
    for chunk in block_causal_chunk_plan(n_frame, n_hw, q_chunk_tokens):
        rows = mask[chunk.q_start : chunk.q_stop]
        assert torch.all(rows[:, : chunk.kv_stop] == 0.0)
        assert torch.all(torch.isinf(rows[:, chunk.kv_stop :]))


@pytest.mark.parametrize("n_frame,n_hw", [(1, 4), (3, 8), (5, 24)])
def test_production_mask_matches_the_diffusers_reference(n_frame, n_hw):
    diffusers_module = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_hunyuanvideo15")
    expected = diffusers_module.HunyuanVideo15AttnBlock.prepare_causal_attention_mask(
        n_frame, n_hw, torch.float32, torch.device("cpu")
    )
    torch.testing.assert_close(_block_causal_mask(n_frame, n_hw), expected, rtol=0, atol=0)


def test_chunk_plan_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        block_causal_chunk_plan(0, 8)
    with pytest.raises(ValueError):
        block_causal_chunk_plan(4, 0)
    with pytest.raises(ValueError):
        block_causal_chunk_plan(4, 8, -1)


# ---------------------------------------------------------------------------
# Mathematical equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_frame,n_hw,channels", [(1, 16, 8), (4, 12, 16), (5, 24, 32), (7, 9, 64), (11, 6, 16)])
@pytest.mark.parametrize("q_chunk_tokens", [0, 1, 3, 8, 24, 1024])
@pytest.mark.parametrize("batch", [1, 2])
def test_chunked_attention_is_bit_comparable_to_masked_full_attention(n_frame, n_hw, channels, q_chunk_tokens, batch):
    """The chunked form is a rearrangement, not an approximation."""
    q, k, v = _qkv(batch, n_frame, n_hw, channels, seed=n_frame * 100 + n_hw)
    expected = full_attention_reference(q, k, v, n_frame, n_hw)
    actual = chunked_attention_reference(q, k, v, n_frame, n_hw, q_chunk_tokens)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n_frame,n_hw,channels", [(4, 12, 16), (5, 24, 32), (9, 7, 8)])
@pytest.mark.parametrize("q_chunk_tokens", [0, 5, 4096])
def test_chunked_attention_matches_torch_sdpa_with_the_diffusers_mask(n_frame, n_hw, channels, q_chunk_tokens):
    """Cross-check against the operator the CPU VAE actually runs."""
    q, k, v = _qkv(1, n_frame, n_hw, channels, seed=7, dtype=torch.float32)
    mask = _block_causal_mask(n_frame, n_hw)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    actual = chunked_attention_reference(q, k, v, n_frame, n_hw, q_chunk_tokens)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("q_chunk_tokens", [0, 1, 13, 512])
def test_block_size_only_changes_floating_point_summation_order(q_chunk_tokens):
    """Block size must not change the result beyond host BLAS rounding.

    Rows are independent, so differing block sizes only reorder the reductions
    inside each matmul.  In float64 that shows up at a few ULP.
    """
    q, k, v = _qkv(1, 6, 20, 24, seed=11)
    baseline = chunked_attention_reference(q, k, v, 6, 20, 0)
    actual = chunked_attention_reference(q, k, v, 6, 20, q_chunk_tokens)
    torch.testing.assert_close(actual, baseline, rtol=0, atol=8 * torch.finfo(torch.float64).eps)


def test_a_block_size_that_clamps_to_one_frame_is_bit_identical():
    """Any block at least a frame wide must produce the frame-granular plan."""
    q, k, v = _qkv(1, 6, 20, 24, seed=11)
    baseline = chunked_attention_reference(q, k, v, 6, 20, 0)
    assert block_causal_chunk_plan(6, 20, 512) == block_causal_chunk_plan(6, 20, 0)
    torch.testing.assert_close(chunked_attention_reference(q, k, v, 6, 20, 512), baseline, rtol=0, atol=0)


def test_first_frame_attends_only_to_itself_and_last_frame_sees_everything():
    """Guard the causal endpoints the prefix arithmetic depends on."""
    n_frame, n_hw, channels = 4, 10, 8
    q, k, v = _qkv(1, n_frame, n_hw, channels, seed=3)

    truncated_k = k.clone()
    truncated_v = v.clone()
    truncated_k[:, n_hw:, :] = 1e6  # anything after frame 0 must not reach frame 0
    truncated_v[:, n_hw:, :] = 1e6
    perturbed = chunked_attention_reference(q, truncated_k, truncated_v, n_frame, n_hw, 0)
    baseline = chunked_attention_reference(q, k, v, n_frame, n_hw, 0)

    torch.testing.assert_close(perturbed[:, :n_hw, :], baseline[:, :n_hw, :], rtol=0, atol=0)
    assert not torch.allclose(perturbed[:, -n_hw:, :], baseline[:, -n_hw:, :])


# ---------------------------------------------------------------------------
# Memory scaling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("resolution", sorted(PRODUCTION_LATENTS))
@pytest.mark.parametrize("q_chunk_tokens", [512, 1024, 2048])
def test_peak_score_memory_is_bounded_by_one_query_block(resolution, q_chunk_tokens):
    n_frame, height, width = PRODUCTION_LATENTS[resolution]
    n_hw = height * width
    seq = n_frame * n_hw

    monolithic = unchunked_score_elements(n_frame, n_hw) * BF16_BYTES
    plan = block_causal_chunk_plan(n_frame, n_hw, q_chunk_tokens)
    chunked = chunk_plan_peak_score_elements(plan) * BF16_BYTES

    # One block of at most q_chunk_tokens rows against the longest key prefix.
    assert chunked == tile_padded(min(q_chunk_tokens, n_hw)) * tile_padded(seq) * BF16_BYTES
    assert chunked < monolithic / 20
    assert chunked < 512 * 1024**2
    print(
        f"[vae attn memory] {resolution} seq={seq} monolithic={monolithic / 1024**3:.2f} GiB "
        f"chunk={q_chunk_tokens} peak={chunked / 1024**2:.1f} MiB "
        f"reduction={monolithic / chunked:.0f}x blocks={len(plan)}",
        flush=True,
    )


def test_720p_monolithic_score_matrix_reproduces_the_observed_allocation_failure():
    """The OOM request size is exactly the tile-padded bf16 ``seq x seq`` matrix."""
    n_frame, height, width = PRODUCTION_LATENTS["720p"]
    assert n_frame * height * width == 111_600
    assert tile_padded(111_600) == 111_616
    assert unchunked_score_elements(n_frame, height * width) * BF16_BYTES == 24_916_262_912
    # Blackhole spreads an interleaved DRAM buffer over eight banks.
    assert 24_916_262_912 // 8 == 3_114_532_864


def test_480p_cached_mask_explains_most_of_the_observed_post_decode_dram():
    """The mask survives the decode because ``_mask_cache`` keeps it resident."""
    n_frame, height, width = PRODUCTION_LATENTS["480p"]
    mask_bytes = unchunked_score_elements(n_frame, height * width) * BF16_BYTES
    observed_post_decode_bytes = 8.61 * 1024**3
    assert mask_bytes / observed_post_decode_bytes > 0.5
    print(
        f"[vae attn memory] 480p cached mask {mask_bytes / 1024**3:.2f} GiB of the "
        f"observed 8.61 GiB post-decode DRAM",
        flush=True,
    )


@pytest.mark.parametrize("tile_px", [128, 256, 512, 768])
def test_larger_tiles_are_attention_bound_only_without_chunking(tile_px):
    """Quantify the headroom chunking gives the tile path."""
    n_frame = 31
    n_hw = (tile_px // 16) ** 2
    monolithic = unchunked_score_elements(n_frame, n_hw) * BF16_BYTES
    chunked = chunk_plan_peak_score_elements(block_causal_chunk_plan(n_frame, n_hw, 1024)) * BF16_BYTES
    assert chunked <= monolithic
    print(
        f"[vae attn tile budget] {tile_px}px latent {n_hw} tokens/frame "
        f"monolithic={monolithic / 1024**2:.1f} MiB chunked={chunked / 1024**2:.1f} MiB",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment gate
# ---------------------------------------------------------------------------


def test_attention_chunking_is_off_by_default():
    assert attention_chunk_tokens_from_env({}) == 0
    assert attention_chunk_tokens_from_env({"HY_VAE_ATTN_CHUNK": "0"}) == 0


@pytest.mark.parametrize("raw,expected", [("1", 1), ("512", 512), ("4096", 4096)])
def test_attention_chunk_env_parses_token_counts(raw, expected):
    assert attention_chunk_tokens_from_env({"HY_VAE_ATTN_CHUNK": raw}) == expected


@pytest.mark.parametrize("raw", ["", "-1", "auto", "512.0", "1,024"])
def test_attention_chunk_env_fails_closed_on_bad_values(raw):
    with pytest.raises(ValueError):
        attention_chunk_tokens_from_env({"HY_VAE_ATTN_CHUNK": raw})


# ---------------------------------------------------------------------------
# Device cases (hardware required; not run during host validation)
# ---------------------------------------------------------------------------


def _attention_node(channels, generator):
    state = {"norm.gamma": torch.ones(channels)}
    for name in ("to_q", "to_k", "to_v", "proj_out"):
        weight = torch.randn(channels, channels, 1, 1, 1, generator=generator) / channels**0.5
        state[f"{name}.weight"] = weight
        state[f"{name}.bias"] = torch.randn(channels, generator=generator) * 0.01
    return SimpleNamespace(state_dict=lambda: state, in_channels=channels)


def _parallel_kwargs(mesh_device):
    shape = tuple(mesh_device.shape)
    config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(shape[0], 0),
        width_parallel=ParallelFactor(shape[1], 1),
    )
    return config, CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)


def _replicated_bthwc(mesh_device, host_bthwc):
    return ttnn.from_torch(
        host_bthwc,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _dram_allocated_bytes(mesh_device):
    ttnn.synchronize_device(mesh_device)
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return view.num_banks * view.total_bytes_allocated_per_bank


_FABRIC = [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}]


@pytest.mark.parametrize("q_chunk_tokens", [1, 7, 32, 512])
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_device_chunked_attention_matches_full_attention(mesh_device, q_chunk_tokens):
    """Same weights, same input, chunked versus monolithic on one chip."""
    generator = torch.Generator().manual_seed(5)
    channels = 64
    source = torch.randn(1, 6, 5, 7, channels, generator=generator)
    torch_attn = _attention_node(channels, generator)

    reference = AttnBlock(torch_attn, device=mesh_device, attention_chunk_tokens=0)
    reference_out = ttnn.to_torch(ttnn.get_device_tensors(reference(_replicated_bthwc(mesh_device, source)))[0]).float()

    chunked = AttnBlock(torch_attn, device=mesh_device, attention_chunk_tokens=q_chunk_tokens)
    chunked_out = ttnn.to_torch(ttnn.get_device_tensors(chunked(_replicated_bthwc(mesh_device, source)))[0]).float()

    assert chunked_out.shape == reference_out.shape
    ok, pcc = comp_pcc(reference_out, chunked_out, 0.9999)
    print(f"[vae chunked attn q={q_chunk_tokens}] PCC={pcc}", flush=True)
    assert ok, f"chunked attention PCC {pcc} < 0.9999"


@pytest.mark.parametrize("device_params", _FABRIC, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2), (2, 2)], indirect=True)
def test_device_chunked_attention_survives_hw_sharding(mesh_device):
    """Chunking must compose with the gather/repartition contract."""
    generator = torch.Generator().manual_seed(1)
    channels = 32
    source = torch.randn(1, 4, 5, 7, channels, generator=generator)  # uneven H and W
    torch_attn = _attention_node(channels, generator)

    reference = AttnBlock(torch_attn, device=mesh_device, attention_chunk_tokens=0)
    reference_out = ttnn.to_torch(ttnn.get_device_tensors(reference(_replicated_bthwc(mesh_device, source)))[0]).float()

    config, manager = _parallel_kwargs(mesh_device)
    plan = SpatialShardPlan(5, 7, config.height_parallel.factor, config.width_parallel.factor)
    sharded = AttnBlock(
        torch_attn,
        device=mesh_device,
        parallel_config=config,
        ccl_manager=manager,
        attention_chunk_tokens=8,
    )
    sharded_input = typed_tensor_2dshard(
        replicate_pad_to_plan(source, plan, h_dim=2, w_dim=3),
        mesh_device,
        shard_mapping={config.height_parallel.mesh_axis: 2, config.width_parallel.mesh_axis: 3},
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    dims = [None, None]
    dims[config.height_parallel.mesh_axis] = 2
    dims[config.width_parallel.mesh_axis] = 3
    sharded_out = fast_device_to_host(sharded(sharded_input, 5, 7), mesh_device, dims, ccl_manager=manager)[
        :, :, :5, :7, :
    ]

    ok, pcc = comp_pcc(reference_out, sharded_out, 0.999)
    print(f"[vae chunked attn h/w sharded] PCC={pcc}", flush=True)
    assert ok, f"chunked sharded attention PCC {pcc} < 0.999"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_device_chunked_attention_lowers_peak_dram(mesh_device):
    """Measure the allocation win at a shape whose monolithic form still fits."""
    generator = torch.Generator().manual_seed(9)
    channels = 64
    latent_t, latent_h, latent_w = 8, 16, 16
    source = torch.randn(1, latent_t, latent_h, latent_w, channels, generator=generator)
    torch_attn = _attention_node(channels, generator)

    measurements = {}
    for label, chunk_tokens in (("monolithic", 0), ("chunked", 256)):
        block = AttnBlock(torch_attn, device=mesh_device, attention_chunk_tokens=chunk_tokens)
        before = _dram_allocated_bytes(mesh_device)
        out = block(_replicated_bthwc(mesh_device, source))
        ttnn.synchronize_device(mesh_device)
        measurements[label] = _dram_allocated_bytes(mesh_device) - before
        ttnn.deallocate(out)

    print(f"[vae chunked attn dram] {measurements}", flush=True)
    assert measurements["chunked"] < measurements["monolithic"]


@pytest.mark.parametrize("device_params", _FABRIC, indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_device_720p_hw_sharded_attention_allocation_budget(mesh_device):
    """The gate that previously OOMed: full 720p latent extent, chunking on.

    Opt in with ``HY_VAE_ATTN_720P_GATE=1``; it allocates the real mid-block
    activations for ``(T, H, W) = (31, 45, 80)`` at 1024 channels.
    """
    if os.environ.get("HY_VAE_ATTN_720P_GATE", "0") != "1":
        pytest.skip("set HY_VAE_ATTN_720P_GATE=1 to run the 720p attention budget gate")

    generator = torch.Generator().manual_seed(17)
    channels = 1024
    latent_t, latent_h, latent_w = PRODUCTION_LATENTS["720p"]
    torch_attn = _attention_node(channels, generator)
    config, manager = _parallel_kwargs(mesh_device)
    plan = SpatialShardPlan(latent_h, latent_w, config.height_parallel.factor, config.width_parallel.factor)
    source = torch.randn(1, latent_t, latent_h, latent_w, channels, generator=generator)

    block = AttnBlock(
        torch_attn,
        device=mesh_device,
        parallel_config=config,
        ccl_manager=manager,
        attention_chunk_tokens=int(os.environ.get("HY_VAE_ATTN_CHUNK", "1024")),
    )
    sharded_input = typed_tensor_2dshard(
        replicate_pad_to_plan(source, plan, h_dim=2, w_dim=3),
        mesh_device,
        shard_mapping={config.height_parallel.mesh_axis: 2, config.width_parallel.mesh_axis: 3},
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    before = _dram_allocated_bytes(mesh_device)
    out = block(sharded_input, latent_h, latent_w)
    ttnn.synchronize_device(mesh_device)
    delta = _dram_allocated_bytes(mesh_device) - before

    print(f"[vae 720p chunked attn] allocated delta {delta / 1024**3:.2f} GiB", flush=True)
    assert out.shape[1] == latent_t
    assert delta < 24_916_262_912
