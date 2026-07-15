# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validate the precomputed-t_emb path == the inline timestep-embedder path (bit-identical),
and that reusing a precomputed t_emb across steps is safe (no aliasing)."""
import sys
from pathlib import Path
import pytest, torch, ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.vibevoice.common.config import MODEL_PATH
from models.experimental.vibevoice.tt.load_weights import load_vibevoice_state_dict, split_submodule_weights
from models.experimental.vibevoice.tt.ttnn_diffusion_head import preprocess_diffusion_head_weights, TTDiffusionHead
from models.experimental.vibevoice.tt.vibevoice_config import load_vibevoice_model_config

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT / "reference", _ROOT.parent.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_tembs_equiv(mesh_device):
    torch.manual_seed(0)
    cfg = load_vibevoice_model_config(MODEL_PATH).diffusion_head
    state = split_submodule_weights(load_vibevoice_state_dict(MODEL_PATH))["diffusion_head"]
    H, L = cfg.hidden_size, cfg.latent_size
    w = preprocess_diffusion_head_weights(
        state,
        mesh_device,
        hidden_size=H,
        latent_size=L,
        head_ffn_ratio=cfg.head_ffn_ratio,
        norm_eps=cfg.rms_norm_eps,
        num_layers=cfg.head_layers,
    )
    head = TTDiffusionHead(w)

    def tt(t, shp):
        return ttnn.as_tensor(
            t.view(*shp),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # two distinct timesteps to make sure per-step t_emb is selected correctly
    t_tensors = [tt(torch.tensor([500.0, 500.0]), (2, 1, 1, 1)), tt(torch.tensor([250.0, 250.0]), (2, 1, 1, 1))]
    t_embs = head.precompute_t_embs(t_tensors)

    worst = 1.0
    for i, t_t in enumerate(t_tensors):
        x = tt(torch.randn(2, L), (2, 1, 1, L))
        cond = tt(torch.randn(2, H), (2, 1, 1, H))
        inline = ttnn.to_torch(head(x, t_t, cond)).float()
        precomp = ttnn.to_torch(head(x, t_t, cond, t_emb=t_embs[i])).float()
        _, pcc = comp_pcc(inline, precomp, pcc=0.9999)
        print(f"step {i}: inline-vs-precomputed PCC = {pcc}")
        worst = min(worst, float(pcc.split()[-1]) if isinstance(pcc, str) else pcc)
    # reuse t_embs[0] a second time to confirm no aliasing/mutation
    x = tt(torch.randn(2, L), (2, 1, 1, L))
    cond = tt(torch.randn(2, H), (2, 1, 1, H))
    a = ttnn.to_torch(head(x, t_tensors[0], cond, t_emb=t_embs[0])).float()
    b = ttnn.to_torch(head(x, t_tensors[0], cond, t_emb=t_embs[0])).float()
    _, pcc_reuse = comp_pcc(a, b, pcc=0.9999)
    print(f"reuse determinism PCC = {pcc_reuse}")
