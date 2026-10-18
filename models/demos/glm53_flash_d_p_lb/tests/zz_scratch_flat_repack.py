# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: (re)build GLM-5.3-Flash's flat weight cache (generated/glm53_flash_d_p/tt_cache/flat/<mesh>) for every MoE
layer from the HF checkpoint, in the routed experts' plan of this grid (the token-split path's: 64 gate/up + 32 down;
the all-gather path's plan has the same weight layout). A layer whose files exist is skipped. GLM_REPACK_LAYERS
(comma list) limits the layers.

    TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 FLAT_CACHE_NO_ADOPT=1 BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml \\
    scripts/run_safe_pytest.sh --mesh --no-precompile models/demos/glm53_flash_d_p_lb/tests/zz_scratch_flat_repack.py -s
"""

import gc
import os
import time

from loguru import logger

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S_ = spec()
pytestmark = device_timeout(S_)


@mesh_parametrize
def test_flat_repack(mesh_device):
    import ttnn
    from models.demos.glm53_flash_d_p.bringup import hooks
    from models.demos.glm53_flash_d_p.tt.experts import LazyExpertWeights
    from models.demos.glm53_flash_d_p.tt.experts_ts import TtExpertsTs

    hooks.apply_device_settings(S_)
    loader, cfg = hooks._loader_cfg(S_)
    layers = (
        [int(v) for v in os.environ["GLM_REPACK_LAYERS"].split(",")]
        if os.environ.get("GLM_REPACK_LAYERS")
        else [i for i in range(cfg.num_hidden_layers) if cfg.is_moe(i)]
    )
    t_all = time.time()
    for layer in layers:
        t0 = time.time()
        ex = TtExpertsTs(
            mesh_device,
            layer,
            LazyExpertWeights(loader, layer, cfg.n_routed_experts),
            num_experts=cfg.n_routed_experts,
            emb_dim=cfg.hidden_size,
            hidden_dim=cfg.moe_intermediate_size,
            top_k=cfg.num_experts_per_tok,
            max_seq_len=int(S_.get("target.chunk")),
            weights_dtype=ttnn.bfloat4_b,
            cache=True,
        )
        for t in (ex.flat.w_gu, ex.flat.w_d, ex.flat.w_rd):
            if t is not None:
                ttnn.deallocate(t)
        del ex
        gc.collect()
        logger.info(f"REPACK layer {layer}: {time.time() - t0:.0f} s (total {time.time() - t_all:.0f} s)")
