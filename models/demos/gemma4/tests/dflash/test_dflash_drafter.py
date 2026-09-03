# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Step 4 validation: the full 5-layer DFlash drafter chain (pre-final-norm), real
weights, real context + noise inputs and RoPE tables from the torch reference
(dump_torch_5layer.py), PCC-compared against the real reference output.

Run: pytest models/demos/gemma4/tests/dflash/test_dflash_drafter.py -k 1x8 -s
"""

import torch

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.dflash.config import Gemma4DFlashDrafterConfig
from models.demos.gemma4.tt.dflash.drafter import dflash_drafter_forward
from models.demos.gemma4.tt.dflash.weights import load_gemma4_dflash_weights

TORCH_REF_PATH = (
    "/tmp/claude-1002/-home-user-proj-sdk/4b2381cd-910e-4293-86a0-feecee22c8e9/scratchpad/torch_5layer_ref.pt"
)


@parametrize_mesh_with_fabric([(1, 8)])
def test_dflash_drafter_5layer_t3k(mesh_device, device_params):
    from models.common.utility_functions import comp_pcc

    ref = torch.load(TORCH_REF_PATH, map_location="cpu")
    ctx_len = ref["ctx_len"]
    block_size = ref["block_size"]
    context_torch = ref["context"]
    noise_torch = ref["noise_embedding"]
    cos_torch = ref["cos"]
    sin_torch = ref["sin"]
    layer_configs_raw = ref["layer_configs"]  # list of (is_causal, sliding_window)
    layer_configs = [(bool(c[0]), int(c[1]) if c[1] is not None else None) for c in layer_configs_raw]
    five_layer_out_ref = ref["five_layer_out"]

    config = Gemma4DFlashDrafterConfig.from_pretrained()
    assert len(layer_configs) == config.num_hidden_layers
    mesh_config = MeshConfig(tuple(mesh_device.shape), decode=ModeConfig(tp=mesh_device.shape[1]))
    ccl_manager = CCLManager(mesh_device)

    weights = load_gemma4_dflash_weights(mesh_device, config, mesh_config)

    replicate = ttnn.ReplicateTensorToMesh(mesh_device)

    def to_tt(x, dtype=ttnn.bfloat16):
        return ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype, mesh_mapper=replicate)

    context_tt = to_tt(context_torch.unsqueeze(0))
    noise_tt = to_tt(noise_torch.unsqueeze(0))
    cos_tt = to_tt(cos_torch.unsqueeze(0))
    sin_tt = to_tt(sin_torch.unsqueeze(0))

    num_local_heads = config.num_attention_heads // mesh_config.tp
    num_local_kv_heads = config.num_key_value_heads // mesh_config.tp

    out_tt = dflash_drafter_forward(
        context_tt,
        noise_tt,
        weights,
        cos_tt,
        sin_tt,
        mesh_device,
        mesh_config,
        ccl_manager,
        num_local_heads,
        num_local_kv_heads,
        config.head_dim,
        config.rms_norm_eps,
        layer_configs,
    )

    out_back = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).float()
    out_back = out_back.reshape(1, block_size, -1)

    passing, pcc = comp_pcc(five_layer_out_ref, out_back, pcc=0.97)
    print(f"5-layer drafter output PCC: {pcc}")
    assert passing, f"5-layer output PCC {pcc} below threshold"
    print("[PASSED] full 5-layer DFlash drafter chain matches torch reference")
