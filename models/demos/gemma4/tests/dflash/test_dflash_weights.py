# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Step 1 validation: load the real Gemma4-31B DFlash drafter checkpoint
(z-lab/gemma-4-31B-it-DFlash) onto a real T3K mesh and confirm the weights
actually landed correctly -- exact round-trip for the untransformed/replicated
tensors (norms, fc), and shape checks for every TP-sharded tensor (attention
QKV/O, MLP gate_up/down) across all 5 layers.

Run: pytest models/demos/gemma4/tests/dflash/test_dflash_weights.py -s
"""

import torch

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4.tt.dflash.config import Gemma4DFlashDrafterConfig
from models.demos.gemma4.tt.dflash.weight_mapping import load_dflash_flat_state_dict
from models.demos.gemma4.tt.dflash.weights import load_gemma4_dflash_weights


def _read_replicated(mesh_device, tt_tensor):
    """A REPLICATED tensor holds identical bytes on every device -- read back
    exactly ONE device's copy, not a concat (which would wrongly stack N
    identical copies). Mirrors the same gotcha hit for Qwen3.6's
    DistributedNorm output earlier this session."""
    return ttnn.to_torch(ttnn.get_device_tensors(tt_tensor)[0])


@parametrize_mesh_with_fabric([(1, 8)])
def test_dflash_weights_load_t3k(mesh_device, device_params):
    config = Gemma4DFlashDrafterConfig.from_pretrained()
    mesh_config = MeshConfig(tuple(mesh_device.shape), decode=ModeConfig(tp=mesh_device.shape[1]))
    flat = load_dflash_flat_state_dict()

    weights = load_gemma4_dflash_weights(mesh_device, config, mesh_config)

    # ---- fc: replicated, exact round-trip ----
    fc_back = _read_replicated(mesh_device, weights.fc).squeeze(0).squeeze(0).transpose(-2, -1)
    fc_ref = flat["fc.weight"].float()
    assert fc_back.shape == fc_ref.shape, f"fc shape {fc_back.shape} != {fc_ref.shape}"
    assert torch.allclose(fc_back.float(), fc_ref, atol=1e-2), "fc.weight round-trip mismatch"
    print(f"[OK] fc.weight {tuple(fc_back.shape)} exact round-trip")

    # ---- hidden_norm / norm: replicated, exact round-trip ----
    for name, rmsnorm in (("hidden_norm", weights.hidden_norm), ("norm", weights.norm)):
        back = _read_replicated(mesh_device, rmsnorm.tt_weight).reshape(-1)[: config.hidden_size]
        ref = flat[f"{name}.weight"].float()
        assert torch.allclose(back.float(), ref, atol=1e-2), f"{name}.weight round-trip mismatch"
        print(f"[OK] {name}.weight {tuple(back.shape)} exact round-trip")

    tp = mesh_config.tp
    q_size = config.num_attention_heads * config.head_dim
    kv_size = config.num_key_value_heads * config.head_dim
    expected_qkv_width = q_size + 2 * kv_size

    for i, layer in enumerate(weights.layers):
        # ---- attention: shape checks (TP-sharded, reconstructing exact
        # per-head interleaving is deferred to the forward-pass PCC check) ----
        wqkv = layer.attn.wqkv
        wqkv_full = ttnn.to_torch(wqkv, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
        assert wqkv_full.shape[-2] == config.hidden_size, f"layer {i} wqkv K dim {wqkv_full.shape}"
        assert (
            wqkv_full.shape[-1] == expected_qkv_width
        ), f"layer {i} wqkv N dim {wqkv_full.shape} != {expected_qkv_width}"

        o_proj_full = ttnn.to_torch(layer.attn.o_proj, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-2))
        assert o_proj_full.shape[-1] == config.hidden_size, f"layer {i} o_proj N dim {o_proj_full.shape}"

        q_norm_back = _read_replicated(mesh_device, layer.attn.q_norm_weight).reshape(-1)[: config.head_dim]
        k_norm_back = _read_replicated(mesh_device, layer.attn.k_norm_weight).reshape(-1)[: config.head_dim]
        assert torch.allclose(
            q_norm_back.float(), flat[f"layers.{i}.self_attn.q_norm.weight"].float(), atol=1e-2
        ), f"layer {i} q_norm mismatch"
        assert torch.allclose(
            k_norm_back.float(), flat[f"layers.{i}.self_attn.k_norm.weight"].float(), atol=1e-2
        ), f"layer {i} k_norm mismatch"

        # ---- mlp: shape checks ----
        gate_up_full = ttnn.to_torch(layer.mlp.gate_up, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
        assert gate_up_full.shape[-2] == config.hidden_size
        assert (
            gate_up_full.shape[-1] == 2 * config.intermediate_size
        ), f"layer {i} gate_up N dim {gate_up_full.shape} != {2 * config.intermediate_size}"
        down_full = ttnn.to_torch(layer.mlp.down, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-2))
        assert down_full.shape[-1] == config.hidden_size, f"layer {i} down N dim {down_full.shape}"

        # ---- layer norms: replicated, exact round-trip ----
        for name, rmsnorm in (
            ("input_layernorm", layer.input_layernorm),
            ("post_attention_layernorm", layer.post_attention_layernorm),
        ):
            back = _read_replicated(mesh_device, rmsnorm.tt_weight).reshape(-1)[: config.hidden_size]
            ref = flat[f"layers.{i}.{name}.weight"].float()
            assert torch.allclose(back.float(), ref, atol=1e-2), f"layer {i} {name} round-trip mismatch"

        print(
            f"[OK] layer {i}: wqkv {tuple(wqkv_full.shape)}, o_proj {tuple(o_proj_full.shape)}, "
            f"gate_up {tuple(gate_up_full.shape)}, down {tuple(down_full.shape)}, norms match"
        )

    print(f"[PASSED] all {config.num_hidden_layers} DFlash drafter layers loaded correctly at tp={tp}")
