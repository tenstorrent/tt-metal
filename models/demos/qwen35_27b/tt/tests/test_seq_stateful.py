"""Test if chunk_gated_delta_rule_seq gives consistent results across sequential calls."""
import os

os.environ["TT_METAL_HOME"] = os.getcwd()

import pytest
import torch

import ttnn
from models.demos.qwen35_27b.tt.gdn_chunk_ops import chunk_gated_delta_rule, create_chunk_masks
from models.demos.qwen35_27b.tt.gdn_chunk_ops_seq import chunk_gated_delta_rule_seq

_MESH_SHAPE = (1, 4)


def _pcc(a, b):
    af, bf = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([af, bf]))[0, 1].item()


def _to_torch(t, mesh):
    full = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
    return full[: full.shape[0] // 4]


@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "num_command_queues": 2, "trace_region_size": 100_000_000}],
    indirect=True,
)
def test_seq_stateful(mesh_device, reset_seeds, ensure_gc):
    """Run seq path 5 times with IDENTICAL random inputs and check consistency."""
    BH, T, K, V = 12, 512, 128, 128
    chunk_size = 128

    torch.manual_seed(42)
    q_t = torch.randn(BH, T, K, dtype=torch.float32)
    k_t = torch.nn.functional.normalize(torch.randn(BH, T, K, dtype=torch.float32), dim=-1)
    v_t = torch.randn(BH, T, V, dtype=torch.float32)
    beta_t = torch.sigmoid(torch.randn(BH, T, 1, dtype=torch.float32)) * 0.5
    g_t = -torch.rand(BH, T, dtype=torch.float32) * 2  # log-space decay (negative)

    def make_tt(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    masks = create_chunk_masks(chunk_size, mesh_device)

    # Run parallel scan once as reference
    q_tt = make_tt(q_t)
    k_tt = make_tt(k_t)
    v_tt = make_tt(v_t)
    beta_tt = make_tt(beta_t)
    g_tt = make_tt(g_t)
    out_par, _ = chunk_gated_delta_rule(
        q_tt,
        k_tt,
        v_tt,
        beta_tt,
        g_tt,
        chunk_size=chunk_size,
        scale=None,
        initial_state=None,
        mesh_device=mesh_device,
        cached_masks=masks,
    )
    o_par = _to_torch(out_par, mesh_device)
    ttnn.deallocate(out_par)
    print(f"\nParallel scan output norm: {o_par.norm():.4f}")

    # Run seq path 5 times with same inputs
    for run in range(5):
        q_tt2 = make_tt(q_t)
        k_tt2 = make_tt(k_t)
        v_tt2 = make_tt(v_t)
        beta_tt2 = make_tt(beta_t)
        g_tt2 = make_tt(g_t)
        out_seq, _ = chunk_gated_delta_rule_seq(
            q_tt2,
            k_tt2,
            v_tt2,
            beta_tt2,
            g_tt2,
            chunk_size=chunk_size,
            scale=None,
            initial_state=None,
            mesh_device=mesh_device,
            cached_masks=masks,
        )
        o_seq = _to_torch(out_seq, mesh_device)
        ttnn.deallocate(out_seq)
        p = _pcc(o_par, o_seq)
        print(f"  Run {run}: PCC vs par = {p:.5f}  seq_norm={o_seq.norm():.4f}")
    print("\nDone.")
