# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Loop the 20-image (S~25k) prefill N times on one mesh to reproduce / measure a prefill hang.

    python models/demos/molmo2/tests/stress_prefill_hang.py [iterations] [sample_input.json]
"""
import sys
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.molmo2.demo.demo import build_inputs, load_inputs
from models.demos.molmo2.tt.hf_loading import load_molmo2_processor, load_molmo2_state_dict
from models.demos.molmo2.tt.model import TtMolmo2Model
from models.demos.molmo2.tt.model_config import Molmo2Config
from models.demos.molmo2.tt.vision_inputs import vision_inputs_from_processor
from models.tt_transformers.tt.ccl import TT_CCL

iters = int(sys.argv[1]) if len(sys.argv) > 1 else 10
sample = (
    sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).parents[1] / "demo/sample_inputs/multi_image_20_demo.json")
)

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
cfg = Molmo2Config(mesh_device=mesh)
cache = Path("/tmp/molmo2_weight_cache")
cache.mkdir(parents=True, exist_ok=True)
model = TtMolmo2Model(
    mesh_device=mesh,
    tt_ccl=TT_CCL(mesh),
    state_dict=load_molmo2_state_dict(dtype=torch.bfloat16),
    weight_cache_path=cache,
    dtype=ttnn.bfloat16,
    configuration=cfg,
)
proc = load_molmo2_processor()
out = build_inputs(proc, load_inputs(sample, 1))
pv, pool = vision_inputs_from_processor(out)
logger.info(f"S={out['input_ids'].shape[1]}; looping {iters} prefills")
for i in range(iters):
    t = time.time()
    logits = model.forward_prefill(
        input_ids=out["input_ids"],
        pixel_values=pv,
        pooled_patches_idx=pool,
        token_type_ids=out["token_type_ids"],
        user_id=0,
    )
    logger.info(f"ITER {i} ok {time.time()-t:.1f}s argmax={int(logits[0, -1].argmax())}")
ttnn.close_mesh_device(mesh)
logger.info("STRESS DONE")
