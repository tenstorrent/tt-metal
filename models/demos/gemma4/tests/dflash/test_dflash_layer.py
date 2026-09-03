# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Step 3 validation: one isolated DFlash drafter layer (layer 0), real weights, real
context + noise inputs and RoPE tables taken directly from the torch reference
(dump_torch_layer0.py), PCC-compared against the real reference layer-0 output.

Run: pytest models/demos/gemma4/tests/dflash/test_dflash_layer.py -k 1x8 -s
"""

import torch

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.dflash.attention import build_attention_mask_additive
from models.demos.gemma4.tt.dflash.config import Gemma4DFlashDrafterConfig
from models.demos.gemma4.tt.dflash.layer import dflash_layer_forward
from models.demos.gemma4.tt.dflash.weights import load_gemma4_dflash_weights

TORCH_REF_PATH = (
    "/tmp/claude-1002/-home-user-proj-sdk/4b2381cd-910e-4293-86a0-feecee22c8e9/scratchpad/torch_layer0_ref.pt"
)


@parametrize_mesh_with_fabric([(1, 8)])
def test_dflash_layer0_t3k(mesh_device, device_params):
    from models.common.utility_functions import comp_pcc

    ref = torch.load(TORCH_REF_PATH, map_location="cpu")
    ctx_len = ref["ctx_len"]
    block_size = ref["block_size"]
    context_torch = ref["context"]  # [1, ctx_len, hidden]
    noise_torch = ref["noise_embedding"]  # [1, block_size, hidden]
    cos_torch = ref["cos"]  # [1, ctx_len+block_size, head_dim]
    sin_torch = ref["sin"]
    layer0_out_ref = ref["layer0_out"]  # [1, block_size, hidden]
    is_causal = bool(ref["layer0_is_causal"])
    sliding_window = ref["layer0_sliding_window"]
    sliding_window = int(sliding_window) if sliding_window is not None else None

    config = Gemma4DFlashDrafterConfig.from_pretrained()
    mesh_config = MeshConfig(tuple(mesh_device.shape), decode=ModeConfig(tp=mesh_device.shape[1]))
    ccl_manager = CCLManager(mesh_device)

    weights = load_gemma4_dflash_weights(mesh_device, config, mesh_config)

    replicate = ttnn.ReplicateTensorToMesh(mesh_device)

    def to_tt(x, dtype=ttnn.bfloat16):
        return ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype, mesh_mapper=replicate)

    context_tt = to_tt(context_torch.unsqueeze(0))  # [1,1,ctx_len,hidden]
    noise_tt = to_tt(noise_torch.unsqueeze(0))  # [1,1,block_size,hidden]
    cos_tt = to_tt(cos_torch.unsqueeze(0))  # [1,1,ctx_len+block_size,head_dim]
    sin_tt = to_tt(sin_torch.unsqueeze(0))

    mask_torch = build_attention_mask_additive(ctx_len, block_size, is_causal, sliding_window)
    mask_tt = to_tt(mask_torch)

    num_local_heads = config.num_attention_heads // mesh_config.tp
    num_local_kv_heads = config.num_key_value_heads // mesh_config.tp

    out_tt = dflash_layer_forward(
        context_tt,
        noise_tt,
        weights.layers[0],
        cos_tt,
        sin_tt,
        mask_tt,
        mesh_config,
        ccl_manager,
        num_local_heads,
        num_local_kv_heads,
        config.head_dim,
        config.rms_norm_eps,
    )

    out_back = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).float()
    out_back = out_back.reshape(1, block_size, -1)

    passing, pcc = comp_pcc(layer0_out_ref, out_back, pcc=0.97)
    print(f"layer 0 output PCC: {pcc}")
    assert passing, f"layer0 output PCC {pcc} below threshold"
    print("[PASSED] DFlash drafter layer 0 (context+noise K/V concat attention + MLP) matches torch reference")
