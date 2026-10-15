# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real-path wall clock of forward_prefill for one video, with the main sub-stages wrapped
(vision backbone, ViT chunk forward, pooling, text prefill = remainder).

    MOLMO2_VERIF_DIR=<verification dir> pytest models/demos/molmo2/tests/profile_prefill_wall.py -s
"""

import time
from collections import defaultdict

from loguru import logger

from models.demos.molmo2.tests.profile_forward_pass import (  # noqa: F401
    _load_first_video_test,
    build_inputs,
    mesh_device,
    processor,
    tt_model,
)

acc = defaultdict(float)


def _wrap(obj, name, key):
    fn = getattr(obj, name)

    def w(*a, **k):
        t = time.perf_counter()
        r = fn(*a, **k)
        acc[key] += time.perf_counter() - t
        return r

    setattr(obj, name, w)


def test_prefill_wall(mesh_device, tt_model, processor):
    model, cfg = tt_model
    vp, prompt = _load_first_video_test()
    ids, pv, pool_idx, tti = build_inputs(processor, vp, prompt)
    _wrap(model, "run_vision_backbone", "vision_total")
    _wrap(model.vit_encoder, "forward", "vit_forward_calls (async submit+)")
    _wrap(model, "_run_chunked_ttnn_pooling", "pooling")
    _wrap(model.image_projector, "forward", "projector")
    for it in range(4):
        acc.clear()
        model.reset_kv_cache(user_id=0)
        t0 = time.perf_counter()
        model.forward_prefill(
            input_ids=ids, pixel_values=pv, pooled_patches_idx=pool_idx, token_type_ids=tti, user_id=0
        )
        tot = time.perf_counter() - t0
        rest = tot - acc["vision_total"]
        logger.info(
            f"iter {it}: total {tot*1e3:.0f} ms | vision {acc['vision_total']*1e3:.0f} "
            f"(vit-fwd {acc['vit_forward_calls (async submit+)']*1e3:.0f}, pooling {acc['pooling']*1e3:.0f}, "
            f"projector {acc['projector']*1e3:.0f}) | embed+inject+mask+decoder+lm_head {rest*1e3:.0f}"
        )
