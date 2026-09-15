# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Transformer-level correctness for Wan2.2 TI2V-5B, including the per-token timestep path.

Separate from `test_transformer_wan.py` so the 14B tests are untouched: that file's model
constants are module-level and hardcoded to 14B, and its `_make_wan_transformer` does not
forward `model_type`.

Constants here are read off the checkpoint config rather than hardcoded, so the test cannot
drift from the weights it loads.

The headline test is `test_two_row_timestep_equals_scalar`, which needs no torch reference: it
feeds the production path a mask of all ones, where every token resolves to the same timestep,
and asserts it reproduces the proven scalar path. That isolates the whole per-token plumbing --
the two-row expansion, the sequence-parallel masks, the reshaped AdaLN tables, the feature-axis
chunking -- from any question about whether the model itself is right.
"""

import pytest
import torch
from diffusers import WanTransformer3DModel as TorchWanTransformer3DModel
from loguru import logger

import ttnn
from models.tt_dit.models.transformers.wan2_2.transformer_wan import WanTransformer3DModel
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.wan.pipeline_wan_ti2v_5b_i2v import WanTI2V5BI2VPipeline  # noqa: F401
from models.tt_dit.pipelines.wan.ti2v_5b_i2v_math import first_frame_mask, pad_timesteps_for_sequence_parallel
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.tensor import bf16_tensor, float32_tensor, from_torch, local_device_to_torch
from models.tt_dit.utils.test import ring_params_req_exact_devices, skip_if_unsupported_num_links

MODEL_NAME = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Real 1280x704 / 81-frame latent geometry: 704/16=44, 1280/16=80, (81-1)//4+1=21.
# Chosen so the token count matches the pipeline exactly (N=18480 -> padded 18688 -> 2336/dev)
# and the registered blockings are the ones actually exercised.
LATENT_T, LATENT_H, LATENT_W = 21, 44, 80
PROMPT_SEQ_LEN = 118
TIMESTEP = 731.0

MESH_PARAMS = [
    [(4, 8), 1, 0, 2, {**ring_params_req_exact_devices}, ttnn.Topology.Ring, False],
]


def _load_5b_config_and_state(num_layers: int):
    """Checkpoint config plus a state dict truncated to `num_layers` blocks."""
    torch_model = TorchWanTransformer3DModel.from_pretrained(
        MODEL_NAME, subfolder="transformer", torch_dtype=torch.float32, trust_remote_code=True
    )
    cfg = torch_model.config
    torch_model.blocks = torch.nn.ModuleList(list(torch_model.blocks[:num_layers]))
    torch_model.eval()
    return cfg, torch_model


def _make_tt_transformer(cfg, *, mesh_device, ccl_manager, parallel_config, num_layers):
    """Build the TT model from the checkpoint config, mirroring WanCheckpoint.build."""
    return WanTransformer3DModel(
        patch_size=cfg.patch_size,
        num_heads=cfg.num_attention_heads,
        dim=cfg.num_attention_heads * cfg.attention_head_dim,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        text_dim=cfg.text_dim,
        freq_dim=cfg.freq_dim,
        ffn_dim=cfg.ffn_dim,
        num_layers=num_layers,
        cross_attn_norm=cfg.cross_attn_norm,
        eps=cfg.eps,
        rope_max_seq_len=cfg.rope_max_seq_len,
        model_type="ti2v",  # gates in_channels==48; the default "t2v" asserts 16
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=False,
    )


def _parallel_config(mesh_device, sp_axis, tp_axis):
    return DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )


def _per_token_timestep_tensor(mask, t, *, sp, mesh_device, patch):
    """Build the SP-sharded (1, 1, N_padded, 1) per-token timestep the pipeline uploads."""
    _, ph, pw = patch
    tokens = (mask[0][0][:, ::ph, ::pw] * t).flatten()
    padded = pad_timesteps_for_sequence_parallel(tokens, sp.factor, fill=t)
    return from_torch(
        padded.reshape(1, 1, -1, 1).to(torch.float32),
        device=mesh_device,
        mesh_axes=[None, None, sp.mesh_axis, None],
        dtype=ttnn.float32,
    )


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    MESH_PARAMS,
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.skip(
    reason="The pipeline emits the 2-row timestep (see test_two_row_timestep_equals_scalar); the "
    "fully-per-token layout runs the timestep MLP at M=N/SP, which needs matmul blocking entries "
    "that were deliberately removed because their keys collided with T2V projection shapes in the "
    "process-global table. Kept for when that layout is wanted again."
)
def test_per_token_timestep_equals_scalar(mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp):
    """A per-token timestep that is uniform must reproduce the scalar path exactly.

    No torch reference is needed: this is TT-vs-TT on identical inputs, differing only in how
    the timestep reaches the model. With an all-ones mask every token sees the same `t`, so the
    two paths are mathematically the same computation and any gap is a plumbing or layout bug
    -- most likely the group-major assumption in the reshaped AdaLN tables.
    """
    if not ttnn.device.is_blackhole():
        pytest.skip("TI2V-5B targets BH Galaxy")
    skip_if_unsupported_num_links(mesh_device, num_links)

    parallel_config = _parallel_config(mesh_device, sp_axis, tp_axis)
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    cfg, torch_model = _load_5b_config_and_state(num_layers=1)
    state_dict = torch_model.state_dict()
    del torch_model

    tt_model = _make_tt_transformer(
        cfg, mesh_device=mesh_device, ccl_manager=ccl_manager, parallel_config=parallel_config, num_layers=1
    )
    tt_model.load_torch_state_dict(state_dict)

    torch.manual_seed(0)
    spatial = torch.randn((1, cfg.in_channels, LATENT_T, LATENT_H, LATENT_W), dtype=torch.float32)
    prompt = torch.randn((1, PROMPT_SEQ_LEN, cfg.text_dim), dtype=torch.float32)

    spatial_host, n = tt_model.preprocess_spatial_input_host(spatial)
    rope_cos, rope_sin, trans_mat = tt_model.prepare_rope_features(spatial)
    prompt_1BLP = tt_model.prepare_text_conditioning(bf16_tensor(prompt.unsqueeze(0), device=mesh_device))
    spatial_device = from_torch(
        spatial_host, device=mesh_device, mesh_axes=[None, None, parallel_config.sequence_parallel.mesh_axis, None]
    )
    logger.info(f"5B transformer: N={n} (padded {spatial_host.shape[2]}), per-device M={spatial_host.shape[2] // 8}")

    common = {
        "spatial_1BNI": spatial_device,
        "prompt_1BLP": prompt_1BLP,
        "rope_cos_1HND": rope_cos,
        "rope_sin_1HND": rope_sin,
        "trans_mat": trans_mat,
        "N": n,
    }

    # --- scalar timestep: the proven path -------------------------------------------------
    scalar_ts = float32_tensor(
        torch.full((1,), TIMESTEP, dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1),
        device=mesh_device,
    )
    out_scalar = tt_model.postprocess_spatial_output_host(
        local_device_to_torch(tt_model.inner_step(timestep=scalar_ts, **common)),
        LATENT_T,
        LATENT_H,
        LATENT_W,
        n,
    )

    # --- per-token timestep, all-ones mask: must match -------------------------------------
    ones = torch.ones(1, 1, LATENT_T, LATENT_H, LATENT_W, dtype=torch.float32)
    per_token_ts = _per_token_timestep_tensor(
        ones,
        TIMESTEP,
        sp=parallel_config.sequence_parallel,
        mesh_device=mesh_device,
        patch=cfg.patch_size,
    )
    out_per_token = tt_model.postprocess_spatial_output_host(
        local_device_to_torch(tt_model.inner_step(timestep=per_token_ts, **common)),
        LATENT_T,
        LATENT_H,
        LATENT_W,
        n,
    )

    del tt_model
    assert_quality(out_scalar, out_per_token, pcc=0.9999, relative_rmse=0.01)


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    MESH_PARAMS,
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.skip(reason="Same reason as test_per_token_timestep_equals_scalar: fully-per-token layout.")
def test_per_token_timestep_conditions_first_frame(
    mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp
):
    """A real first-frame mask must change the output, and change it only where expected.

    Complements the equivalence test: that one proves the per-token path can reproduce the
    scalar path, this one proves it is actually *doing* something -- the conditioned frame's
    tokens see timestep 0 and must diverge from the uniform-`t` result, while the mask itself
    is the only input that changed.
    """
    if not ttnn.device.is_blackhole():
        pytest.skip("TI2V-5B targets BH Galaxy")
    skip_if_unsupported_num_links(mesh_device, num_links)

    parallel_config = _parallel_config(mesh_device, sp_axis, tp_axis)
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    cfg, torch_model = _load_5b_config_and_state(num_layers=1)
    state_dict = torch_model.state_dict()
    del torch_model

    tt_model = _make_tt_transformer(
        cfg, mesh_device=mesh_device, ccl_manager=ccl_manager, parallel_config=parallel_config, num_layers=1
    )
    tt_model.load_torch_state_dict(state_dict)

    torch.manual_seed(0)
    spatial = torch.randn((1, cfg.in_channels, LATENT_T, LATENT_H, LATENT_W), dtype=torch.float32)
    prompt = torch.randn((1, PROMPT_SEQ_LEN, cfg.text_dim), dtype=torch.float32)

    spatial_host, n = tt_model.preprocess_spatial_input_host(spatial)
    rope_cos, rope_sin, trans_mat = tt_model.prepare_rope_features(spatial)
    prompt_1BLP = tt_model.prepare_text_conditioning(bf16_tensor(prompt.unsqueeze(0), device=mesh_device))
    spatial_device = from_torch(
        spatial_host, device=mesh_device, mesh_axes=[None, None, parallel_config.sequence_parallel.mesh_axis, None]
    )
    common = {
        "spatial_1BNI": spatial_device,
        "prompt_1BLP": prompt_1BLP,
        "rope_cos_1HND": rope_cos,
        "rope_sin_1HND": rope_sin,
        "trans_mat": trans_mat,
        "N": n,
    }

    def run(mask):
        ts = _per_token_timestep_tensor(
            mask,
            TIMESTEP,
            sp=parallel_config.sequence_parallel,
            mesh_device=mesh_device,
            patch=cfg.patch_size,
        )
        return tt_model.postprocess_spatial_output_host(
            local_device_to_torch(tt_model.inner_step(timestep=ts, **common)),
            LATENT_T,
            LATENT_H,
            LATENT_W,
            n,
        )

    uniform = run(torch.ones(1, 1, LATENT_T, LATENT_H, LATENT_W, dtype=torch.float32))
    conditioned = run(first_frame_mask(LATENT_T, LATENT_H, LATENT_W))
    del tt_model

    frame0_delta = (conditioned[:, :, 0] - uniform[:, :, 0]).abs().mean().item()
    tail_delta = (conditioned[:, :, 1:] - uniform[:, :, 1:]).abs().mean().item()
    scale = uniform.abs().mean().item()
    logger.info(f"frame0 |delta|={frame0_delta:.5f}  tail |delta|={tail_delta:.5f}  output scale={scale:.5f}")

    assert frame0_delta > 0.05 * scale, (
        f"conditioned frame 0 barely moved ({frame0_delta:.5f} vs scale {scale:.5f}); the "
        "per-token timestep is probably not reaching the AdaLN modulation"
    )
    # The tail is not expected to be bit-identical -- self-attention mixes frame 0's tokens
    # into every other token -- but it must move far less than frame 0 itself.
    assert tail_delta < frame0_delta, (
        f"tail moved as much as the conditioned frame ({tail_delta:.5f} vs {frame0_delta:.5f}); "
        "the mask is probably not being applied per-token"
    )


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    MESH_PARAMS,
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_two_row_timestep_equals_scalar(mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp):
    """The 2-row timestep optimisation must reproduce the scalar path exactly.

    This is the gate on the optimisation that replaced the per-token timestep MLP: instead of
    embedding N tokens, embed the two distinct values and expand through a mask. With an
    all-ones mask every token takes the second row, so the result must match the scalar path --
    any gap means the expansion (slice / repeat / lerp) or the mask layout is wrong.
    """
    if not ttnn.device.is_blackhole():
        pytest.skip("TI2V-5B targets BH Galaxy")
    skip_if_unsupported_num_links(mesh_device, num_links)

    parallel_config = _parallel_config(mesh_device, sp_axis, tp_axis)
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    cfg, torch_model = _load_5b_config_and_state(num_layers=1)
    state_dict = torch_model.state_dict()
    del torch_model

    tt_model = _make_tt_transformer(
        cfg, mesh_device=mesh_device, ccl_manager=ccl_manager, parallel_config=parallel_config, num_layers=1
    )
    tt_model.load_torch_state_dict(state_dict)

    torch.manual_seed(0)
    spatial = torch.randn((1, cfg.in_channels, LATENT_T, LATENT_H, LATENT_W), dtype=torch.float32)
    prompt = torch.randn((1, PROMPT_SEQ_LEN, cfg.text_dim), dtype=torch.float32)

    spatial_host, n = tt_model.preprocess_spatial_input_host(spatial)
    rope_cos, rope_sin, trans_mat = tt_model.prepare_rope_features(spatial)
    prompt_1BLP = tt_model.prepare_text_conditioning(bf16_tensor(prompt.unsqueeze(0), device=mesh_device))
    spatial_device = from_torch(
        spatial_host, device=mesh_device, mesh_axes=[None, None, parallel_config.sequence_parallel.mesh_axis, None]
    )
    common = {
        "spatial_1BNI": spatial_device,
        "prompt_1BLP": prompt_1BLP,
        "rope_cos_1HND": rope_cos,
        "rope_sin_1HND": rope_sin,
        "trans_mat": trans_mat,
        "N": n,
    }

    def unpatch(out):
        return tt_model.postprocess_spatial_output_host(local_device_to_torch(out), LATENT_T, LATENT_H, LATENT_W, n)

    # Scalar reference.
    scalar_ts = float32_tensor(
        torch.full((1,), TIMESTEP, dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1), device=mesh_device
    )
    out_scalar = unpatch(tt_model.inner_step(timestep=scalar_ts, **common))

    # Two-row + all-ones mask: every token selects row 1, which holds TIMESTEP.
    padded_n = spatial_host.shape[2]
    dim_tp = (cfg.num_attention_heads * cfg.attention_head_dim) // tuple(mesh_device.shape)[tp_axis]
    sp = parallel_config.sequence_parallel

    def ones_mask(width):
        m = torch.ones(1, 1, padded_n, width, dtype=torch.float32)
        return from_torch(m, device=mesh_device, mesh_axes=[None, None, sp.mesh_axis, None], dtype=ttnn.float32)

    tt_model.set_per_token_timestep_masks(ones_mask(dim_tp), ones_mask(6 * dim_tp))
    two_row_ts = float32_tensor(
        torch.tensor([0.0, TIMESTEP], dtype=torch.float32).reshape(1, 1, 2, 1), device=mesh_device
    )
    out_two_row = unpatch(tt_model.inner_step(timestep=two_row_ts, **common))

    del tt_model
    assert_quality(out_scalar, out_two_row, pcc=0.9999, relative_rmse=0.01)


# ---------------------------------------------------------------------------
# Absolute correctness against the torch reference
# ---------------------------------------------------------------------------
# Geometry is deliberately small so a CPU torch forward is quick, while keeping H and W even
# (patch_size=(1,2,2) floor-divides without asserting, so an odd dim would silently drop a row).
REF_T, REF_H, REF_W = 8, 22, 40  # -> N = 8 * 11 * 20 = 1760
REF_PCC = 0.992_000  # repo convention for a full-model / inner_step comparison
REF_RMSE = 0.15


def _reference_and_tt(mesh_device, sp_axis, tp_axis, num_links, topology, *, per_token):
    """Run the torch 5B reference and the TT model on identical inputs; return (torch, tt)."""
    parallel_config = _parallel_config(mesh_device, sp_axis, tp_axis)
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    cfg, torch_model = _load_5b_config_and_state(num_layers=1)
    state_dict = torch_model.state_dict()

    torch.manual_seed(0)
    spatial = torch.randn((1, cfg.in_channels, REF_T, REF_H, REF_W), dtype=torch.float32)
    prompt = torch.randn((1, PROMPT_SEQ_LEN, cfg.text_dim), dtype=torch.float32)

    _, ph, pw = cfg.patch_size
    n_tokens = REF_T * (REF_H // ph) * (REF_W // pw)
    tokens_per_frame = (REF_H // ph) * (REF_W // pw)

    if per_token:
        # 0 on the conditioned frame's tokens, t elsewhere -- the TI2V-5B schedule. The torch
        # reference takes its ndim==2 branch (transformer_wan.py:650), flattens, and passes
        # timestep_seq_len through to the condition embedder.
        ts_torch = torch.full((1, n_tokens), TIMESTEP, dtype=torch.float32)
        ts_torch[:, :tokens_per_frame] = 0.0
    else:
        ts_torch = torch.full((1,), TIMESTEP, dtype=torch.float32)

    logger.info(f"torch reference forward: spatial {tuple(spatial.shape)}, timestep {tuple(ts_torch.shape)}")
    with torch.no_grad():
        torch_out = torch_model(
            hidden_states=spatial, encoder_hidden_states=prompt, timestep=ts_torch, return_dict=False
        )[0]
    del torch_model

    tt_model = _make_tt_transformer(
        cfg, mesh_device=mesh_device, ccl_manager=ccl_manager, parallel_config=parallel_config, num_layers=1
    )
    tt_model.load_torch_state_dict(state_dict)

    spatial_host, n = tt_model.preprocess_spatial_input_host(spatial)
    assert n == n_tokens, f"token count mismatch: {n} vs {n_tokens}"
    rope_cos, rope_sin, trans_mat = tt_model.prepare_rope_features(spatial)
    prompt_1BLP = tt_model.prepare_text_conditioning(bf16_tensor(prompt.unsqueeze(0), device=mesh_device))
    sp = parallel_config.sequence_parallel
    spatial_device = from_torch(spatial_host, device=mesh_device, mesh_axes=[None, None, sp.mesh_axis, None])

    if per_token:
        padded_n = spatial_host.shape[2]
        dim_tp = (cfg.num_attention_heads * cfg.attention_head_dim) // tuple(mesh_device.shape)[tp_axis]
        mask = first_frame_mask(REF_T, REF_H, REF_W)
        tokens = mask[0, 0, :, ::ph, ::pw].reshape(-1)
        tokens = pad_timesteps_for_sequence_parallel(tokens, sp.factor, fill=1.0)

        def m(width):
            t = tokens.reshape(1, 1, -1, 1).expand(-1, -1, -1, width).contiguous()
            return from_torch(t, device=mesh_device, mesh_axes=[None, None, sp.mesh_axis, None], dtype=ttnn.float32)

        tt_model.set_per_token_timestep_masks(m(dim_tp), m(6 * dim_tp))
        ts_tt = float32_tensor(
            torch.tensor([0.0, TIMESTEP], dtype=torch.float32).reshape(1, 1, 2, 1), device=mesh_device
        )
    else:
        ts_tt = float32_tensor(ts_torch.unsqueeze(1).unsqueeze(1).unsqueeze(1), device=mesh_device)

    tt_raw = tt_model.inner_step(
        spatial_1BNI=spatial_device,
        prompt_1BLP=prompt_1BLP,
        rope_cos_1HND=rope_cos,
        rope_sin_1HND=rope_sin,
        trans_mat=trans_mat,
        N=n,
        timestep=ts_tt,
    )
    tt_out = tt_model.postprocess_spatial_output_host(local_device_to_torch(tt_raw), REF_T, REF_H, REF_W, n)
    del tt_model
    return torch_out, tt_out


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    MESH_PARAMS,
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_transformer_vs_torch_scalar_timestep(
    mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp
):
    """Absolute correctness of the 5B transformer port against the torch reference.

    This is the check that the other 5B tests cannot make: the equivalence tests are TT-vs-TT
    and the pipeline gates are self-consistency, so without this nothing pins the 5B port to
    torch in absolute terms. Scalar timestep, so it isolates the port from the conditioning.
    """
    if not ttnn.device.is_blackhole():
        pytest.skip("TI2V-5B targets BH Galaxy")
    skip_if_unsupported_num_links(mesh_device, num_links)

    torch_out, tt_out = _reference_and_tt(mesh_device, sp_axis, tp_axis, num_links, topology, per_token=False)
    assert_quality(torch_out, tt_out, pcc=REF_PCC, relative_rmse=REF_RMSE)


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    MESH_PARAMS,
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_transformer_vs_torch_per_token_timestep(
    mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp
):
    """Absolute correctness of the image-conditioning path against the torch reference.

    The torch model has explicit Wan2.2-TI2V branches for this: a 2-D timestep is flattened and
    `timestep_seq_len` threaded to the condition embedder, the blocks take the `temb.ndim == 4`
    path, and norm_out takes `temb.ndim == 3`. So the reference exercises the same per-token
    modulation the TT two-row expansion produces, and this pins the conditioning -- not just
    the port -- to torch.
    """
    if not ttnn.device.is_blackhole():
        pytest.skip("TI2V-5B targets BH Galaxy")
    skip_if_unsupported_num_links(mesh_device, num_links)

    torch_out, tt_out = _reference_and_tt(mesh_device, sp_axis, tp_axis, num_links, topology, per_token=True)
    assert_quality(torch_out, tt_out, pcc=REF_PCC, relative_rmse=REF_RMSE)
