# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Diagnostic: compare chunk_gated_delta_rule vs chunk_gated_delta_rule_seq
on real model inputs (first 5 GDN layers, ISL=512 for speed).

Run:
  MESH_DEVICE=P150x4 pytest models/demos/qwen35_27b/tt/tests/test_seq_vs_par_pcc.py -v -s
"""
import os

import pytest
import torch

import ttnn
from models.demos.qwen35_27b.tt.gdn_chunk_ops import chunk_gated_delta_rule, create_chunk_masks
from models.demos.qwen35_27b.tt.gdn_chunk_ops_seq import chunk_gated_delta_rule_seq
from models.demos.qwen35_27b.tt.model import create_qwen35_model

HF_MODEL = os.environ.get(
    "HF_MODEL",
    "/home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654",
)
ISL = 512
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
def test_seq_vs_par_pcc(mesh_device, reset_seeds, ensure_gc):
    """Compare seq path vs parallel scan on real GDN layer inputs."""
    import models.demos.qwen35_27b.tt.gdn as gdn_mod

    captured = {}
    orig_fn = gdn_mod.chunk_gated_delta_rule
    call_count = [0]

    def capture_fn(q, k, v, beta, g, **kw):
        idx = call_count[0]
        if idx < 5:
            captured[idx] = tuple(ttnn.clone(t, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in (q, k, v, beta, g))
        call_count[0] += 1
        return orig_fn(q, k, v, beta, g, **kw)

    gdn_mod.chunk_gated_delta_rule = capture_fn

    print(f"\nLoading model for ISL={ISL}...")
    model = create_qwen35_model(
        mesh_device,
        model_path=HF_MODEL,
        max_batch_size=1,
        max_seq_len=ISL + 128,
        dtype=ttnn.bfloat8_b,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
    tokens = tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 60)[:ISL]
    tok_tensor = torch.tensor(tokens, dtype=torch.int32).reshape(1, 1, 1, len(tokens))
    tt_ids = ttnn.from_torch(
        tok_tensor,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    print("Running prefill to capture inputs...")
    model._reset_all_prefill_states(len(tokens))
    x = model._prefill_forward_device(tt_ids)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(x)
    ttnn.deallocate(tt_ids)

    gdn_mod.chunk_gated_delta_rule = orig_fn

    print(f"Captured {len(captured)} GDN layer inputs\n")
    assert len(captured) > 0, "No GDN layers captured — patch didn't work"

    masks = create_chunk_masks(128, mesh_device)

    min_pcc = 1.0
    for idx, (q, k, v, beta, g) in captured.items():

        def clone(t):
            return ttnn.clone(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        out_par, fs_par = chunk_gated_delta_rule(
            clone(q),
            clone(k),
            clone(v),
            clone(beta),
            clone(g),
            chunk_size=128,
            scale=None,
            initial_state=None,
            mesh_device=mesh_device,
            cached_masks=masks,
        )
        out_seq, fs_seq = chunk_gated_delta_rule_seq(
            clone(q),
            clone(k),
            clone(v),
            clone(beta),
            clone(g),
            chunk_size=128,
            scale=None,
            initial_state=None,
            mesh_device=mesh_device,
            cached_masks=masks,
        )

        o_par = _to_torch(out_par, mesh_device)
        o_seq = _to_torch(out_seq, mesh_device)
        s_par = _to_torch(fs_par, mesh_device)
        s_seq = _to_torch(fs_seq, mesh_device)

        p_out = _pcc(o_par, o_seq)
        p_state = _pcc(s_par, s_seq)
        max_err = (o_par - o_seq).abs().max().item()
        print(f"  Layer {idx}: output PCC={p_out:.5f}  state PCC={p_state:.5f}  max_err={max_err:.4f}")
        print(f"    par_norm={o_par.norm():.3f}  seq_norm={o_seq.norm():.3f}")

        min_pcc = min(min_pcc, p_out)

        ttnn.deallocate(out_par)
        ttnn.deallocate(out_seq)
        ttnn.deallocate(fs_par)
        ttnn.deallocate(fs_seq)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        ttnn.deallocate(beta)
        ttnn.deallocate(g)

    print(f"\nMin output PCC across {len(captured)} layers: {min_pcc:.5f}")
    assert min_pcc >= 0.99, f"seq path PCC {min_pcc:.5f} < 0.99 on real weights"
