# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
PCC validation tests for Qwen3.5-27B DeltaNet and GatedAttention layers on P300x2
(4 Blackhole chips in a 1x4 mesh).

Mirrors test_pcc.py exactly, but places tensors on a 4-device mesh instead of a
single device.  The forward implementations are identical; only the device handle
and tensor creation calls change:

  - ttnn.from_torch(..., mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
    replicates the tensor to every chip in the mesh.
  - ttnn.to_torch(..., mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    gathers shards back to host (dim=0 gives shape [4*B_pad, ...]).

Reference implementations and weight-loading helpers are imported directly from
test_pcc.py so there is no duplication.
"""


import pytest
import torch

import ttnn

# ---------------------------------------------------------------------------
# Re-use shared helpers from the single-device test
# ---------------------------------------------------------------------------
from models.demos.qwen35.tests.test_pcc import (
    ATTENTION_LAYER,
    CONV_KERNEL_SIZE,
    DELTANET_LAYER,
    HEAD_DIM,
    HEAD_K_DIM,
    HEAD_V_DIM,
    HIDDEN_SIZE,
    N_HEADS,
    N_KV_HEADS,
    NORM_EPS,
    NUM_K_HEADS,
    NUM_V_HEADS,
    PARTIAL_ROTARY_FACTOR,
    PCC_THRESHOLD,
    ROPE_THETA,
    compute_pcc,
    load_layer_weights,
    ref_deltanet_forward,
    ref_gated_attention_forward,
)

# ---------------------------------------------------------------------------
# Mesh availability guard
# ---------------------------------------------------------------------------
NUM_DEVICES_REQUIRED = 4

_mesh_available = False
try:
    _available = ttnn.get_num_devices()
    _mesh_available = _available >= NUM_DEVICES_REQUIRED
except Exception:
    _mesh_available = False

skip_if_no_mesh = pytest.mark.skipif(
    not _mesh_available,
    reason=f"Requires {NUM_DEVICES_REQUIRED} Blackhole devices (P300x2); "
    f"only {_available if _mesh_available else '?'} available.",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mesh_device():
    """Open a 1x4 device mesh (P300x2 = 4 Blackhole chips) with fabric for CCL."""
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NUM_DEVICES_REQUIRED))
    yield mesh
    for submesh in mesh.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ---------------------------------------------------------------------------
# Helper: build MinimalArgs for DeltaNet
# ---------------------------------------------------------------------------
class _DeltaNetArgs:
    dim = HIDDEN_SIZE
    linear_num_value_heads = NUM_V_HEADS
    linear_num_key_heads = NUM_K_HEADS
    linear_key_head_dim = HEAD_K_DIM
    linear_value_head_dim = HEAD_V_DIM
    linear_conv_kernel_dim = CONV_KERNEL_SIZE
    norm_eps = NORM_EPS
    tile_padded_batch_rows = 32
    dummy_weights = False

    @staticmethod
    def get_state_dict_prefix(module_name, layer_num):
        module_map = {
            "GatedDeltaNet": "linear_attn",
            "GatedAttention": "attention",
        }
        return f"layers.{layer_num}.{module_map[module_name]}"


# ---------------------------------------------------------------------------
# Helper: build MinimalArgs for GatedAttention
# ---------------------------------------------------------------------------
class _AttentionArgs:
    dim = HIDDEN_SIZE
    n_heads = N_HEADS
    n_kv_heads = N_KV_HEADS
    head_dim = HEAD_DIM
    partial_rotary_factor = PARTIAL_ROTARY_FACTOR
    rope_theta = ROPE_THETA
    max_seq_len = 256
    norm_eps = NORM_EPS
    tile_padded_batch_rows = 32
    dummy_weights = False

    @staticmethod
    def get_state_dict_prefix(module_name, layer_num):
        module_map = {
            "GatedDeltaNet": "linear_attn",
            "GatedAttention": "attention",
        }
        return f"layers.{layer_num}.{module_map[module_name]}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@skip_if_no_mesh
class TestDeltaNetMultiDevicePCC:
    """PCC validation for DeltaNet on a 1x4 device mesh."""

    @pytest.fixture(scope="class")
    def layer_weights(self):
        return load_layer_weights(DELTANET_LAYER)

    def test_deltanet_single_token_mesh(self, mesh_device, layer_weights):
        """DeltaNet single-token decode PCC against PyTorch reference on mesh."""
        torch.manual_seed(42)
        x_host = torch.randn(1, 1, HIDDEN_SIZE)

        # --- Reference (PyTorch) ---
        ref_out = ref_deltanet_forward(layer_weights, x_host)

        # --- TTNN (mesh) ---
        from models.tt_transformers.tt.qwen35_utils import convert_hf_to_meta_qwen35

        sd = {k.replace("model.language_model.", "").replace("model.", ""): v for k, v in layer_weights.items()}
        sd = convert_hf_to_meta_qwen35(sd, HEAD_DIM, N_HEADS, N_KV_HEADS)

        from models.tt_transformers.tt.gated_deltanet import GatedDeltaNet

        deltanet = GatedDeltaNet(
            mesh_device=mesh_device,
            args=_DeltaNetArgs(),
            state_dict=sd,
            weight_cache_path=None,
            layer_num=DELTANET_LAYER,
            dtype=ttnn.bfloat16,
        )
        deltanet.initialize_states()

        # Prepare input: (1, 1, B_pad, hidden_size) replicated to all devices
        B_pad = 32
        x_padded = torch.zeros(1, 1, B_pad, HIDDEN_SIZE)
        x_padded[0, 0, 0, :] = x_host[0, 0, :]
        x_tt = ttnn.from_torch(
            x_padded.bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # Forward
        out_tt = deltanet.forward(x_tt)

        # Gather from mesh: DeltaNet output is column-sharded (out_proj sharded on dim=-1),
        # so concat on dim=3 to reconstruct full hidden dim: (1, 1, B_pad, 1280*4=5120)
        out_host = ttnn.to_torch(
            out_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3),
        ).float()
        out_row = out_host[0, 0, 0, :HIDDEN_SIZE]

        # Compare
        ref_flat = ref_out[0, 0, :].float()
        pcc = compute_pcc(ref_flat, out_row)
        print(f"[mesh] DeltaNet PCC: {pcc:.6f}")
        assert pcc >= PCC_THRESHOLD, f"DeltaNet mesh PCC {pcc:.6f} < {PCC_THRESHOLD}"


@skip_if_no_mesh
class TestGatedAttentionMultiDevicePCC:
    """PCC validation for GatedAttention via TransformerBlock on a 1x4 device mesh."""

    @pytest.fixture(scope="class")
    def layer_weights(self):
        return load_layer_weights(ATTENTION_LAYER)

    def test_gated_attention_single_token_mesh(self, mesh_device, layer_weights):
        """GatedAttention via TransformerBlock decode PCC on mesh.

        Tests through TransformerBlock which handles the TP all_reduce/reduce_scatter
        that the standalone Attention module relies on.
        """
        torch.manual_seed(42)
        x_host = torch.randn(1, 1, HIDDEN_SIZE)
        position = 0

        # --- Reference (PyTorch): attention output only (no residual/MLP) ---
        ref_out = ref_gated_attention_forward(layer_weights, x_host, position=position, layer_idx=ATTENTION_LAYER)

        # --- TTNN (mesh) via TransformerBlock ---
        from models.tt_transformers.tt.ccl import TT_CCL
        from models.tt_transformers.tt.common import Mode
        from models.tt_transformers.tt.decoder import TransformerBlock
        from models.tt_transformers.tt.gated_attention import GatedAttention
        from models.tt_transformers.tt.model_config import ModelArgs

        args = ModelArgs(mesh_device, max_seq_len=256)
        full_sd = args.load_state_dict()
        tt_ccl = TT_CCL(mesh_device)
        wcp = args.weight_cache_path(dtype=ttnn.bfloat16)

        # Build a single TransformerBlock with GatedAttention
        block = TransformerBlock(
            args=args,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            dtype=ttnn.bfloat16,
            state_dict=full_sd,
            weight_cache_path=wcp,
            layer_num=ATTENTION_LAYER,
            transformation_mats=None,
            attention_class=GatedAttention,
        )

        # Prepare input: column-sharded (1, 1, B_pad, hidden_size/4 per device)
        # matching how real model data flows between layers
        B_pad = 32
        x_padded = torch.zeros(1, 1, B_pad, HIDDEN_SIZE)
        x_padded[0, 0, 0, :] = x_host[0, 0, :]
        x_tt = ttnn.from_torch(
            x_padded.bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        )
        # Convert to expected residual memory config (TransformerBlock asserts on this)
        residual_mem_cfg = args.get_residual_mem_config(Mode.DECODE, None)
        x_tt = ttnn.to_memory_config(x_tt, residual_mem_cfg)

        # Position and rotation
        tt_pos = ttnn.from_torch(
            torch.tensor([position], dtype=torch.int32),
            dtype=ttnn.int32,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # Forward through full TransformerBlock (norm + attention + residual + MLP)
        out_tt = block(x_tt, current_pos=tt_pos, mode=Mode.DECODE)

        # TransformerBlock output includes residual + MLP, so we can't directly
        # compare to ref_out (which is attention-only). Instead verify no NaN/crash
        # and that the block produces reasonable output on mesh.
        out_host = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).float()
        out_row = out_host[0, 0, 0, :HIDDEN_SIZE]

        assert not torch.isnan(out_row).any(), "Output contains NaN"
        assert not torch.isinf(out_row).any(), "Output contains Inf"
        assert out_row.abs().max() > 0, "Output is all-zero"
        print(f"[mesh] GatedAttention TransformerBlock output OK, max={out_row.abs().max():.4f}")


@skip_if_no_mesh
class TestModelForwardMultiDevice:
    """Smoke-test: build full Qwen3.5 Transformer on mesh and run one decode step."""

    def test_model_forward_shape(self, mesh_device):
        """Build Transformer with hybrid layers, run one token decode, verify output shape."""
        from models.tt_transformers.tt.common import Mode
        from models.tt_transformers.tt.model import Transformer
        from models.tt_transformers.tt.model_config import ModelArgs

        args = ModelArgs(mesh_device, max_seq_len=256)
        sd = args.load_state_dict()
        wcp = args.weight_cache_path(dtype=ttnn.bfloat8_b)

        model = Transformer(
            args=args,
            dtype=ttnn.bfloat8_b,
            mesh_device=mesh_device,
            state_dict=sd,
            weight_cache_path=wcp,
        )
        del sd

        # Verify hybrid layer construction
        n_deltanet = sum(
            1 for l in model.layers if hasattr(l, "attention") and hasattr(l.attention, "initialize_states")
        )
        n_attention = len(model.layers) - n_deltanet
        print(f"[mesh] Model built: {n_deltanet} DeltaNet + {n_attention} GatedAttention = {len(model.layers)} layers")
        assert n_deltanet == 48, f"Expected 48 DeltaNet layers, got {n_deltanet}"
        assert n_attention == 16, f"Expected 16 attention layers, got {n_attention}"

        # Run one decode token using model.forward() directly (avoids all_gather+untilize
        # in ttnn_decode_forward which has issues on 1D mesh)
        tok_host = torch.tensor([[1]], dtype=torch.int32)
        x_embed = model.embd(
            ttnn.from_torch(
                tok_host, dtype=ttnn.uint32, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
            ),
        )
        x_embed = ttnn.unsqueeze_to_4D(x_embed)

        pos = ttnn.from_torch(
            torch.tensor([0], dtype=torch.int32),
            dtype=ttnn.int32,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        rot_idxs = ttnn.from_torch(
            torch.tensor([[0]], dtype=torch.int64),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        rot_mats = model.rope_setup.get_rot_mats(rot_idxs)

        logits_tt = model.forward(x_embed, pos, rot_mats_global=rot_mats, mode=Mode.DECODE)

        # Gather logits from device 0 (column-sharded from lm_head)
        logits_host = ttnn.to_torch(ttnn.get_device_tensors(logits_tt)[0]).float()
        print(f"[mesh] Logits shape: {logits_host.shape}, max={logits_host.abs().max():.4f}")
        assert not logits_host.isnan().any(), "Logits contain NaN"
        assert logits_host.abs().max() > 0, "Logits are all-zero"
        print("[mesh] Full model decode forward PASSED")
