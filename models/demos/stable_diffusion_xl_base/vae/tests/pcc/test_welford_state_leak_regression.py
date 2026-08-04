# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Fast regression test for tt-metal#39225 (welford GN state leak on
Blackhole).

This regression test is the *minimum* reproduction we found: just two
parametrize cases of test_vae_resnetblock2d, in order, both with welford
implicitly enabled (the production path on Blackhole). Reproducing requires
the trained VAE weights -- random-weight unit tests do not produce a
PCC drop large enough to differentiate fix-vs-no-fix above bf16 noise.

With the fix:    case_2 PCC ~0.9993        -- TEST PASSES.
Without fix:     case_2 PCC = 0.9129693567794291 (bit-identical) -- TEST FAILS.
"""

import gc

import pytest
import torch
from diffusers import AutoencoderKL

import ttnn
from models.common.utility_functions import is_blackhole, torch_random
from models.demos.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE
from models.demos.stable_diffusion_xl_base.vae.tt.model_configs import load_vae_model_optimisations
from models.demos.stable_diffusion_xl_base.vae.tt.tt_resnetblock2d import TtResnetBlock2D
from tests.ttnn.utils_for_testing import assert_with_pcc

# Case 1 of the failing parametrize sweep -- the trigger that primes the
# leaked TDMA SrcB state. This is `image_resolution1` from
# test_module_tt_resnetblock2d.py.
PRIME_CASE = dict(
    input_shape=(1, 512, 128, 128),
    block_id=0,
    resnet_id=0,
    conv_shortcut=False,
    block="up_blocks",
)

# Case 2 -- the failing case. This is `image_resolution2` from
# test_module_tt_resnetblock2d.py and reproduces the bit-identical 0.913
# PCC failure when run after PRIME_CASE without the welford_init fix.
PROBE_CASE = dict(
    input_shape=(1, 512, 256, 256),
    block_id=1,
    resnet_id=0,
    conv_shortcut=False,
    block="up_blocks",
)

# Production threshold from test_module_tt_resnetblock2d.py for this shape.
PROBE_PCC = 0.999


def _run_resnet_forward(
    device,
    is_ci_env,
    is_ci_v2_env,
    sdxl_base_vae_location,
    debug_mode,
    state_dict,
    case,
):
    """Mirrors test_vae_resnetblock2d's body. Returns (torch_output, tt_output)
    in (B, C, H, W) layout so the caller can do its own PCC assert."""
    image_resolution = (1024, 1024)
    block_id = case["block_id"]
    resnet_id = case["resnet_id"]
    conv_shortcut = case["conv_shortcut"]
    block = case["block"]
    input_shape = case["input_shape"]

    # Re-load the VAE module reference for this case. We do this fresh per
    # case so we get the right resnet block module from the (already loaded)
    # state_dict-equivalent diffusers model.
    vae = AutoencoderKL.from_pretrained(
        sdxl_base_vae_location,
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_v2_env or is_ci_env,
        subfolder=None if is_ci_v2_env else "vae",
    )
    vae.eval()

    is_encoder = False
    if block == "up_blocks":
        torch_resnet = vae.decoder.up_blocks[block_id].resnets[resnet_id]
        block_path = f"{block}.{block_id}"
    elif block == "down_blocks":
        torch_resnet = vae.encoder.down_blocks[block_id].resnets[resnet_id]
        block_path = f"{block}.{block_id}"
        is_encoder = True
    else:
        torch_resnet = vae.decoder.mid_block.resnets[resnet_id]
        block_path = block
    vae_block = "encoder" if is_encoder else "decoder"

    model_config = load_vae_model_optimisations(image_resolution)
    tt_resnet = TtResnetBlock2D(
        device,
        state_dict,
        f"{vae_block}.{block_path}.resnets.{resnet_id}",
        model_config,
        conv_shortcut,
        debug_mode=debug_mode,
    )

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch_resnet(torch_input_tensor, None)

    B, C, H, W = input_shape
    torch_input_perm = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    torch_input_perm = torch_input_perm.reshape(B, 1, H * W, C)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_perm,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output_tensor, output_shape = tt_resnet.forward(ttnn_input_tensor, [B, C, H, W])
    output_tensor = ttnn.to_torch(ttnn_output_tensor)
    output_tensor = output_tensor.reshape(input_shape[0], output_shape[1], output_shape[2], output_shape[0])
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    del vae, tt_resnet
    gc.collect()

    return torch_output_tensor, output_tensor


@pytest.mark.skipif(not is_blackhole(), reason="tt-metal#39225 only reproduces on Blackhole")
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_welford_state_leak_regression(
    device,
    is_ci_env,
    is_ci_v2_env,
    sdxl_base_vae_location,
    reset_seeds,
    debug_mode,
):
    """tt-metal#39225 regression. Runs the up_blocks-0-0 resnet block
    (PRIME) and then the up_blocks-1-0 resnet block (PROBE) in the same
    process. PROBE must reach PCC >= 0.999."""

    # Load the VAE state dict once and pass it to both cases. The
    # AutoencoderKL.from_pretrained call inside _run_resnet_forward
    # re-instantiates the diffusers model so we can grab the per-case
    # resnet submodule, but the heavy weight load happens just once
    # implicitly via diffusers' on-disk cache.
    vae = AutoencoderKL.from_pretrained(
        sdxl_base_vae_location,
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_v2_env or is_ci_env,
        subfolder=None if is_ci_v2_env else "vae",
    )
    state_dict = vae.state_dict()
    del vae
    gc.collect()

    # PRIME: run case 1 (up_blocks-0-0). We don't assert on its PCC; we
    # only need the side effect on the TDMA SrcB pipeline.
    _run_resnet_forward(
        device,
        is_ci_env,
        is_ci_v2_env,
        sdxl_base_vae_location,
        debug_mode,
        state_dict,
        PRIME_CASE,
    )

    # PROBE: run case 2 (up_blocks-1-0, the failing shape). With the
    # welford_init fix this passes; without it, PCC = 0.9129693567794291.
    torch_output, tt_output = _run_resnet_forward(
        device,
        is_ci_env,
        is_ci_v2_env,
        sdxl_base_vae_location,
        debug_mode,
        state_dict,
        PROBE_CASE,
    )

    assert_with_pcc(torch_output, tt_output, PROBE_PCC)
