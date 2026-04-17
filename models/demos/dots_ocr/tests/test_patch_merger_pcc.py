# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

from models.demos.dots_ocr.reference.patch_merger import PatchMergerRef

try:
    import ttnn  # type: ignore

    _HAS_TTNN_RUNTIME = hasattr(ttnn, "DRAM_MEMORY_CONFIG") and hasattr(ttnn, "open_mesh_device")
except Exception:
    ttnn = None  # type: ignore
    _HAS_TTNN_RUNTIME = False

if not _HAS_TTNN_RUNTIME:
    pytest.skip("TTNN runtime not available (skipping TTNN PCC tests)", allow_module_level=True)

from models.demos.dots_ocr.tt.patch_merger import PatchMerger as PatchMergerTT


def _open_device():
    if os.environ.get("MESH_DEVICE") is None:
        return None
    from models.demos.dots_ocr.tt.mesh import open_mesh_device as _open

    return _open()


def _tt_to_torch(device, y_tt):
    """Gather a potentially-replicated tensor back to torch without concatenating replicas.

    On a 1x1 mesh we can safely use ``ConcatMeshToTensor(dim=0)``. On multi-chip
    meshes where the weights are replicated, we instead slice off one replica
    so shape stays ``[B, 1, S, D]``.
    """
    composer = ttnn.ConcatMeshToTensor(device, dim=0)
    out = ttnn.to_torch(y_tt, mesh_composer=composer).to(torch.float32)
    num_devices = device.get_num_devices()
    if num_devices > 1 and out.shape[0] % num_devices == 0:
        per = out.shape[0] // num_devices
        out = out[:per]
    return out


@pytest.mark.skipif(os.environ.get("MESH_DEVICE") is None, reason="Requires TT device (set MESH_DEVICE)")
def test_patch_merger_pcc_gt_0_99(tmp_path):
    torch.manual_seed(0)
    B, S_patch, hidden = 1, 256, 1536
    spatial_merge_size = 2
    out_hidden = 1536

    # Build reference + align weights
    ref = PatchMergerRef(hidden_size=hidden, out_hidden_size=out_hidden, spatial_merge_size=spatial_merge_size)
    # Create synthetic state_dict that matches TT module's expected keys
    state_dict_prefix = "patch_merger"
    state_dict = {
        f"{state_dict_prefix}.ln_q.weight": ref.norm.weight.detach().clone(),
        f"{state_dict_prefix}.feed_forward.0.weight": ref.fc1.weight.detach().clone().T,
        f"{state_dict_prefix}.feed_forward.2.weight": ref.fc2.weight.detach().clone().T,
    }

    device = _open_device()
    try:
        tt = PatchMergerTT(
            device,
            hidden_size=hidden,
            out_hidden_size=out_hidden,
            spatial_merge_size=spatial_merge_size,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_cache_path=tmp_path,
            dtype=ttnn.bfloat16,
        )

        x = torch.randn(B, 1, S_patch, hidden, dtype=torch.bfloat16)
        y_ref = ref(x.float()).to(torch.float32)

        x_tt = ttnn.from_torch(
            x,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        y_tt = tt(x_tt)
        y_tt_torch = _tt_to_torch(device, y_tt)

        # PCC
        ref_flat = y_ref.flatten()
        tt_flat = y_tt_torch.flatten()
        pcc = torch.corrcoef(torch.stack([ref_flat, tt_flat]))[0, 1].item()
        assert pcc > 0.99, f"PCC too low: {pcc}"
    finally:
        if device is not None:
            ttnn.close_mesh_device(device)
