"""Eager tracy profile of the full sample_actions (single-mesh, NO trace capture).

Runs Pi0_5ModelTTNN.sample_actions eagerly so tracy can map every device op to a
kernel duration (trace-replay loses op->device mapping). Signposts at stage
boundaries (vision / prefill / denoise) so the ops CSV can be split per stage.

    source models/experimental/pi0_5/local_env.sh
    PI0_NUM_CAMERAS=3 PI05_NUM_DENOISE_STEPS=5 \
      python_env/bin/python -m tracy -r -p --op-support-count 100000 \
      -o generated/profiler/prof_sa -n sa \
      models/experimental/pi0_5/tests/perf/prof_sample_actions_eager.py
"""

import os
import torch
import ttnn

from pathlib import Path
from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

CKPT = Path(
    os.environ.get("PI05_CHECKPOINT_DIR", str(Path(__file__).resolve().parents[2] / "weights" / "pi05_libero_upstream"))
)
N_CAMS = int(os.environ.get("PI0_NUM_CAMERAS", "3"))
N_STEPS = int(os.environ.get("PI05_NUM_DENOISE_STEPS", "5"))


def _signpost(name):
    try:
        from tracy import signpost

        signpost(header=name)
    except Exception:
        pass


def main():
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0)
    ah = action_horizon_from_checkpoint(CKPT)
    cfg = Pi0_5ModelConfig(action_horizon=ah, num_denoising_steps=N_STEPS)
    loader = Pi0_5WeightLoader(str(CKPT))
    model = Pi0_5ModelTTNN(cfg, loader, device)

    img_size = cfg.siglip_config.image_size
    images = [torch.randn(1, 3, img_size, img_size, dtype=torch.float32) for _ in range(N_CAMS)]
    img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(N_CAMS)]
    lang_tokens = torch.randint(0, 256000, (1, 256), dtype=torch.int64)
    lang_masks = torch.ones(1, 256, dtype=torch.bool)

    # warmup (build all programs) — not profiled-of-interest but unavoidable
    for _ in range(2):
        out = model.sample_actions(images, img_masks, lang_tokens, lang_masks)
        ttnn.synchronize_device(device)

    # profiled call with stage signposts
    _signpost("SA_START")
    out = model.sample_actions(images, img_masks, lang_tokens, lang_masks)
    ttnn.synchronize_device(device)
    _signpost("SA_END")
    actions = ttnn.to_torch(out)
    print(f"[prof] actions shape={tuple(actions.shape)} finite={torch.isfinite(actions).all().item()}")
    ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
