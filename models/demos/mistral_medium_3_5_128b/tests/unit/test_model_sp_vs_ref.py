# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The whole model at target SP=8 x TP=4 vs the composed torch reference (cached whole-model golden):
sequence sharded over SP, residual SP x TP sharded through every layer, per-layer weights sliced from
one state dict, per-layer K/V landing in their own cache slots, final norm and last-tile logits.

REDUCED DEPTH (diagnostic): 4 of 88 layers at full width with random weights — the host cannot run a
100B random-weight model on both sides. The full-depth, real-weight result is the P1/P2 acceptance test.
"""

import pytest
import torch

from models.demos.mistral_medium_3_5_128b.reference.golden import LOGITS_TAIL, golden_forward
from models.demos.mistral_medium_3_5_128b.reference.model import random_state_dict
from models.demos.mistral_medium_3_5_128b.tt.kv_cache import allocate_kv_cache, naturalize, read_slot_kv
from models.demos.mistral_medium_3_5_128b.tt.model import Model
from models.demos.mistral_medium_3_5_128b.tt.rope import RopeSetup, hf_to_meta_perm
from models.demos.mistral_medium_3_5_128b.tt.weights import StateDictWeights

from .common import CFG, assert_pcc, full_to_torch, residual_to_torch, spec_dtypes


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("num_layers, seq_len", [(4, 10240)], ids=["reduced_depth_4L"])
def test_model_sp_vs_ref(galaxy_mesh, mesh_config, ccl_manager, num_layers, seq_len):
    cfg = CFG.reduced(num_hidden_layers=num_layers)
    _, ids, snaps, kv = golden_forward(cfg, seq_len, seed=0)

    model = Model(
        galaxy_mesh,
        mesh_config,
        ccl_manager,
        cfg,
        StateDictWeights(random_state_dict(cfg, seed=0)),
        dtypes=spec_dtypes(),
    )
    rope = RopeSetup(galaxy_mesh, mesh_config, cfg, max_seq_len=seq_len, chunk_size=seq_len)
    cache = allocate_kv_cache(
        galaxy_mesh,
        mesh_config,
        num_layers=num_layers,
        max_seq_len=seq_len,
        num_local_kv_heads=cfg.num_key_value_heads // mesh_config.tp,
        head_dim=cfg.head_dim,
    )
    x = model.embed(Model.token_tensor(ids[0], galaxy_mesh, mesh_config))
    assert torch.equal(residual_to_torch(x, galaxy_mesh, mesh_config), snaps[0][None].float())
    x = model.forward_layers(x, rope, kv_cache=cache)
    assert_pcc(
        "model_last_layer_residual[reduced 4L]", snaps[num_layers][None], residual_to_torch(x, galaxy_mesh, mesh_config)
    )
    normed, logits = model.head(x)
    assert_pcc("model_final_norm[reduced 4L]", snaps[-2][None], full_to_torch(normed, galaxy_mesh, mesh_config))
    got_logits = residual_to_torch(logits, galaxy_mesh, mesh_config)[:, :, -LOGITS_TAIL:]
    assert_pcc("model_logits_last_tile[reduced 4L]", snaps[-1][None], got_logits)

    k_blk, v_blk = read_slot_kv(galaxy_mesh, cache, 0)
    perm = hf_to_meta_perm(cfg.head_dim)
    for i in range(num_layers):
        ref_k, ref_v = kv[i][0][0], kv[i][1][0]
        assert_pcc(
            f"model_k[layer {i}, reduced 4L]",
            ref_k[..., perm],
            naturalize(k_blk[i], seq_len, mesh_config.sp, seq_len, seq_len),
        )
        assert_pcc(
            f"model_v[layer {i}, reduced 4L]", ref_v, naturalize(v_blk[i], seq_len, mesh_config.sp, seq_len, seq_len)
        )
